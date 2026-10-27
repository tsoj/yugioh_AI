//! New guesses of the opponent's hidden cards on a reconstructed duel (the search's iterations
//! each get their own): the cards whose codes the reconstruction guessed get codes drawn anew
//! from the deck completion (`Completion::sample`), within what the observations require of
//! them (`Guess`: a known kind, ruled out codes) and what their place now shows (a card Set in
//! a monster zone is a monster, in a spell/trap zone a spell or trap). Cards the viewer can see
//! now keep their codes.

use std::collections::HashMap;

use ygo_core::constants::*;
use ygo_core::duel::Duel;

use crate::belief::completion::{Completion, VanillaCompletion, canonical_code, card, fits_need};
use crate::belief::tracker::{Guess, Need};
use ygo_arbiter::pyrandom::PyRandom;
use crate::rduel::recreate_card;
use ygo_arbiter::state::{LOCATIONS, viewer_view};

/// Stand-ins a hidden card of a known kind draws from when the drawn deck has no fitting card
const STAND_IN_CHOICES: usize = 4;

/// What a reconstruction guessed: the guessed cards (card id, `Guess`) and the opponent's deck
/// cards with known codes (main, extra).
#[derive(Clone, Default)]
pub struct Guesses {
    pub viewer: u32,
    pub cards: Vec<(u32, Guess)>,
    pub known: (Vec<u32>, Vec<u32>),
}

/// A guessed card to give a code: its id, section, what its code must fit and the ruled out codes.
struct Slot<'a> {
    cid: u32,
    needs: Vec<Need>,
    excluded: &'a [u32],
}

impl Slot<'_> {
    fn fits(&self, code: u32) -> bool {
        !self.excluded.contains(&code) && self.needs.iter().all(|&n| fits_need(card(code), n, true))
    }
}

/// Draws new codes for the opponent's guessed cards that are hidden from the viewer now.
pub fn resample(d: &mut Duel, g: &Guesses, completion: &dyn Completion, rng: &mut PyRandom) {
    let opp = 1 - g.viewer;
    let seen = viewer_view(d, g.viewer);
    // (the hidden cards of the opponent, by id: what their place requires; the codes in view)
    let mut hidden: HashMap<u32, Option<Need>> = HashMap::new();
    let mut present: Vec<u32> = Vec::new();
    for p in 0..2u32 {
        for (_, loc) in LOCATIONS {
            let ids = d.ext_get_location_cards(p as u8, loc);
            for (i, entry) in seen.players[p as usize].pile(loc).iter().enumerate() {
                let Some(e) = entry else { continue };
                if let Some(c) = e.code.filter(|&c| c != 0) {
                    present.push(c);
                    continue;
                }
                let cid = ids.get(i).map_or(0, |x| x.1);
                if p != opp || cid == 0 {
                    continue;
                }
                let need = match loc {
                    LOCATION_MZONE => Some(Need::Monster),
                    LOCATION_SZONE if i == 5 => Some(Need::Field),
                    LOCATION_SZONE => Some(Need::St),
                    _ => None,
                };
                hidden.insert(cid, need);
            }
        }
    }
    let (mut main_slots, mut extra_slots): (Vec<Slot>, Vec<Slot>) = (vec![], vec![]);
    for (cid, guess) in &g.cards {
        let Some(&place) = hidden.get(cid).filter(|_| guess.player == opp) else { continue };
        // (an Extra Deck card face-down on the field: only its own section's requirement)
        let needs = guess.need.into_iter().chain(place.filter(|_| !guess.extra)).collect();
        let slot = Slot { cid: *cid, needs, excluded: &guess.excluded };
        if guess.extra { extra_slots.push(slot) } else { main_slots.push(slot) }
    }
    if main_slots.is_empty() && extra_slots.is_empty() {
        return;
    }
    let (known_main, known_extra) = (&g.known.0, &g.known.1);
    let (n_main, n_extra) = (main_slots.len(), extra_slots.len());
    let Ok((mut pool_main, mut pool_extra)) = completion
        .sample(known_main, known_extra, n_main, n_extra, &present, rng)
        .or_else(|_| VanillaCompletion::get().complete(known_main, known_extra, n_main, n_extra, &present))
    else {
        return;
    };
    let mut copies: HashMap<u32, usize> = HashMap::new();
    for &c in known_main.iter().chain(&pool_main) {
        *copies.entry(canonical_code(c)).or_default() += 1;
    }
    let mut out: Vec<(u32, u32)> = Vec::new();
    // the cards of a known kind first: a fitting card of the drawn deck, else a stand-in
    rng.shuffle(&mut main_slots);
    let (constrained, plain): (Vec<Slot>, Vec<Slot>) = main_slots.into_iter().partition(|s| !s.needs.is_empty());
    for slot in &constrained {
        let fitting: Vec<usize> = (0..pool_main.len()).filter(|&k| slot.fits(pool_main[k])).collect();
        if !fitting.is_empty() {
            out.push((slot.cid, pool_main.remove(fitting[rng.randrange(fitting.len())])));
            continue;
        }
        let need = slot.needs[0];
        let others: Vec<u32> = known_main.iter().chain(&pool_main).copied().collect();
        let stand_ins: Vec<u32> = completion
            .stand_ins(need, &others)
            .into_iter()
            .filter(|&c| slot.fits(c) && copies.get(&canonical_code(c)).copied().unwrap_or(0) < 3)
            .take(STAND_IN_CHOICES)
            .collect();
        if stand_ins.is_empty() {
            continue; // (the card keeps its code)
        }
        let code = stand_ins[rng.randrange(stand_ins.len())];
        *copies.entry(canonical_code(code)).or_default() += 1;
        out.push((slot.cid, code));
        // (the drawn cards stay as many as the cards left to guess)
        if !pool_main.is_empty() {
            let k = rng.randrange(pool_main.len());
            pool_main.remove(k);
        }
    }
    // the others in a random order, a ruled out code swapped with a later one
    for (pool, slots) in [(&mut pool_main, plain), (&mut pool_extra, extra_slots)] {
        rng.shuffle(pool);
        for (i, slot) in slots.iter().enumerate() {
            if i >= pool.len() {
                break;
            }
            if !slot.fits(pool[i])
                && let Some(k) = (i + 1..pool.len()).find(|&k| slot.fits(pool[k]))
            {
                pool.swap(i, k);
            }
            out.push((slot.cid, pool[i]));
        }
    }
    for (cid, code) in out {
        if d.card_by_id(cid).is_some_and(|c| d.card(c).data.code != code) {
            recreate_card(d, cid, code);
        }
    }
}
