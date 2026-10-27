//! Re-synchronizing an approximate reconstruction with what its viewer observed (sync.py).
//!
//! When no consistent reconstruction exists, the lenient pass replays the best hypothesis while
//! re-aligning with the stream; after an early difference its duel drifts away from the real
//! one. At the viewer's Idle and Battle commands the stream holds what the viewer saw there;
//! `sync_to_view` moves the engine's cards to match it (silently), gives guessed cards the codes
//! that showed, and sets the Life Points. Only the viewer's observation is used, so nothing leaks.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use ygo_core::constants::*;

use crate::belief::completion::card;
use crate::rduel::RDuel;
use ygo_arbiter::state::{CardQuery, LOCATIONS, PRIVATE_STATUS, Snapshot};

const ZONES: u32 = LOCATION_MZONE | LOCATION_SZONE;
const EXTRA_TYPES: u32 = TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK;
const HIDDEN: u32 = LOCATION_DECK | LOCATION_HAND | LOCATION_EXTRA;
const STATUS_MASK: u32 = !PRIVATE_STATUS; // (the status bits a view shows)

type Place = (u32, u32, i64);

fn is_token(code: u32) -> bool {
    card(code).is_some_and(|c| c.type_ & TYPE_TOKEN != 0)
}

/// Changes the duel's state toward the observed view `obs` (`got`: the duel's current view of the
/// viewer). Returns how many changes it made.
pub fn sync_to_view(duel: &mut RDuel, viewer: u32, obs: &Snapshot, got: &Snapshot) -> usize {
    let mut lists: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::new();
    for e in duel.all_cards() {
        if e[3] == 0 {
            lists.entry((e[0], e[1])).or_default().push((e[4], e[5]));
        }
    }
    let mut place: IndexMap<u32, Place> = IndexMap::new();
    let mut code_of: HashMap<u32, u32> = HashMap::new();
    let mut view_of: HashMap<u32, CardQuery> = HashMap::new();
    for p in 0..2u32 {
        for (_, loc) in LOCATIONS {
            for (i, &(code, cid)) in lists.get(&(p, loc)).map_or(&[][..], |v| &v[..]).iter().enumerate() {
                if cid != 0 {
                    place.insert(cid, (p, loc, i as i64));
                    code_of.insert(cid, code);
                    view_of.insert(cid, got.players[p as usize].pile(loc).get(i).cloned().flatten().unwrap_or_default());
                }
            }
        }
    }
    let is_extra = |code_of: &HashMap<u32, u32>, cid: u32| card(code_of[&cid]).is_some_and(|c| c.type_ & EXTRA_TYPES != 0);
    let owner = |view_of: &HashMap<u32, CardQuery>, place: &IndexMap<u32, Place>, cid: u32| view_of[&cid].owner.unwrap_or(place[&cid].0);
    let mut slots: Vec<(u32, u32, usize, CardQuery)> = Vec::new();
    for p in 0..2u32 {
        for (_, loc) in LOCATIONS {
            for (i, e) in obs.players[p as usize].pile(loc).iter().enumerate() {
                if let Some(e) = e {
                    slots.push((p, loc, i, e.clone()));
                }
            }
        }
    }
    let mut target: IndexMap<(u32, u32, usize), u32> = IndexMap::new();
    let mut used: HashSet<u32> = HashSet::new();
    let engine_at = |p: u32, loc: u32, i: usize| -> Option<u32> { lists.get(&(p, loc)).and_then(|l| l.get(i)).map(|x| x.1).filter(|&c| c != 0) };
    // 1. cards already where they showed, as they showed
    for (p, loc, i, e) in &slots {
        if let Some(cid) = engine_at(*p, *loc, *i)
            && e.code.is_some()
            && view_of[&cid] == *e
        {
            target.insert((*p, *loc, *i), cid);
            used.insert(cid);
        }
    }
    // 2. the other visible cards: a copy the engine has elsewhere, else a hidden card of their
    // owner that becomes this code
    let mut recreate: IndexMap<u32, u32> = IndexMap::new();
    for (p, loc, i, e) in &slots {
        let Some(code) = e.code else { continue };
        if target.contains_key(&(*p, *loc, *i)) {
            continue;
        }
        let own = e.owner.unwrap_or(*p);
        let cands: Vec<u32> = place.keys().copied().filter(|c| !used.contains(c) && code_of[c] == code && owner(&view_of, &place, *c) == own).collect();
        if !cands.is_empty() {
            // (the one in the same pile first, then one in a place it did not show, then a hidden one)
            let best = *cands
                .iter()
                .min_by_key(|&&c| {
                    let (q, ql, _) = place[&c];
                    ((q, ql) != (*p, *loc), view_of[&c].code.is_some(), place[&c])
                })
                .unwrap();
            target.insert((*p, *loc, *i), best);
            used.insert(best);
            continue;
        }
        if is_token(code) {
            let cid = duel.create_card(own, code); // (a token the replay did not make)
            place.insert(cid, (own, 0, -1));
            code_of.insert(cid, code);
            view_of.insert(cid, CardQuery { owner: Some(own), ..CardQuery::default() });
            target.insert((*p, *loc, *i), cid);
            used.insert(cid);
            continue;
        }
        let Some(cd) = card(code) else { continue };
        if own == viewer {
            continue; // (the viewer's own cards are known: no guess to correct)
        }
        let extra = cd.type_ & EXTRA_TYPES != 0;
        let pool: Vec<u32> = place
            .keys()
            .copied()
            .filter(|&c| {
                !used.contains(&c)
                    && owner(&view_of, &place, c) == own
                    && view_of[&c].code.is_none()
                    && place[&c].1 & HIDDEN != 0
                    && is_extra(&code_of, c) == extra
                    && !is_token(code_of[&c])
            })
            .collect();
        if let Some(&cid) = pool.iter().min_by_key(|&&c| (place[&c].1 != if extra { LOCATION_EXTRA } else { LOCATION_DECK }, place[&c])) {
            // (from the deck first: a guessed card in the hand stays there)
            recreate.insert(cid, code);
            target.insert((*p, *loc, *i), cid);
            used.insert(cid);
        }
    }
    // 3. hidden cards: the one already there, else one of their owner's
    for (p, loc, i, e) in &slots {
        if e.code.is_some() || target.contains_key(&(*p, *loc, *i)) {
            continue;
        }
        if let Some(cid) = engine_at(*p, *loc, *i)
            && !used.contains(&cid)
            && view_of[&cid] == *e
        {
            target.insert((*p, *loc, *i), cid);
            used.insert(cid);
        }
    }
    for (p, loc, i, e) in &slots {
        if e.code.is_some() || target.contains_key(&(*p, *loc, *i)) {
            continue;
        }
        let own = e.owner.unwrap_or(*p);
        let want_extra = *loc == LOCATION_EXTRA;
        let pool: Vec<u32> = place
            .keys()
            .copied()
            .filter(|&c| !used.contains(&c) && owner(&view_of, &place, c) == own && !is_token(code_of[&c]) && (is_extra(&code_of, c) == want_extra || loc & HIDDEN == 0))
            .collect();
        if let Some(&cid) = pool.iter().min_by_key(|&&c| {
            let (q, ql, _) = place[&c];
            ((q, ql) != (*p, *loc), (q, ql) != (own, LOCATION_DECK), place[&c])
        }) {
            // (the same pile first, then a card the view did not place, then the deck)
            target.insert((*p, *loc, *i), cid);
            used.insert(cid);
        }
    }
    let obs_at = |p: u32, loc: u32, i: usize| -> Option<&CardQuery> { obs.players[p as usize].pile(loc).get(i).and_then(|x| x.as_ref()) };
    // what moves: assigned cards in another place, or (zones) in another position
    let mut moves: Vec<((u32, u32, usize), u32, CardQuery)> = Vec::new();
    for (&(p, loc, i), &cid) in &target {
        let e = obs_at(p, loc, i).cloned().unwrap_or_default();
        if place[&cid] != (p, loc, i as i64) || (loc & ZONES != 0 && e.position != view_of[&cid].position) || recreate.contains_key(&cid) {
            moves.push(((p, loc, i), cid, e));
        }
    }
    // cards no slot took that sit in a zone the view shows empty or another card in
    let taken: HashSet<(u32, u32, usize)> = target.keys().copied().filter(|t| t.1 & ZONES != 0).collect();
    let stray: Vec<u32> = place
        .keys()
        .copied()
        .filter(|c| {
            let (q, ql, qi) = place[c];
            !used.contains(c) && ql & ZONES != 0 && (taken.contains(&(q, ql, qi as usize)) || obs_at(q, ql, qi as usize).is_none())
        })
        .collect();
    let mut changes = 0;
    // leave the zones first (a card's zone can be another's target)
    for &((p, loc, i), cid, _) in &moves {
        if place[&cid].1 & ZONES != 0 && place[&cid] != (p, loc, i as i64) {
            duel.place_card(cid, owner(&view_of, &place, cid), LOCATION_DECK, 1, POS_FACEDOWN_DEFENSE);
        }
    }
    for &cid in &stray {
        // (a token the viewer did not see leaves play)
        let to = if is_token(code_of[&cid]) { 0 } else { LOCATION_DECK };
        duel.place_card(cid, owner(&view_of, &place, cid), to, 1, POS_FACEDOWN_DEFENSE);
        changes += 1;
    }
    for (&cid, &code) in &recreate {
        duel.recreate_card(cid, code);
    }
    let mut sorted_moves = moves.clone();
    sorted_moves.sort_by_key(|m| m.0);
    for ((p, loc, i), cid, e) in &sorted_moves {
        let pos = e.position.unwrap_or(POS_FACEDOWN_DEFENSE);
        let seq = if loc & ZONES != 0 { *i as u32 } else { 0 };
        if duel.place_card(*cid, *p, *loc, seq, pos) {
            changes += 1;
        }
    }
    // what the cards show of their history: why they are there, their status
    let moved: HashSet<u32> = moves.iter().map(|m| m.1).collect();
    for (&(p, loc, i), &cid) in &target {
        let e = obs_at(p, loc, i).cloned().unwrap_or_default();
        let v = &view_of[&cid];
        let reason = e.reason.filter(|&r| moved.contains(&cid) || Some(r) != v.reason);
        let status = e.status.is_some() && (moved.contains(&cid) || e.status != v.status);
        if reason.is_some() || status {
            duel.set_card_info(cid, reason, e.status.unwrap_or(0), if status { STATUS_MASK } else { 0 });
            changes += 1;
        }
    }
    // the order of the piles the viewer sees in order
    for p in 0..2u32 {
        for (_, loc) in LOCATIONS {
            if loc & (ZONES | LOCATION_DECK) != 0 {
                continue;
            }
            let n = obs.players[p as usize].pile(loc).len();
            let want: Vec<Option<u32>> = (0..n).map(|i| target.get(&(p, loc, i)).copied()).collect();
            if want.contains(&None) {
                continue;
            }
            let want: Vec<u32> = want.into_iter().flatten().collect();
            let now: Vec<u32> = duel.location_cards(p, loc).into_iter().map(|x| x.1).filter(|&c| c != 0).collect();
            let wset: HashSet<u32> = want.iter().copied().collect();
            let mut order = want.clone();
            order.extend(now.iter().copied().filter(|c| !wset.contains(c)));
            let (mut a, mut b) = (now.clone(), order.clone());
            a.sort();
            b.sort();
            if now != order && a == b {
                duel.set_location_order_by_id(p, loc, &order);
            }
        }
    }
    if obs.turn != got.turn {
        duel.set_turn(obs.turn);
        changes += 1;
    }
    for p in 0..2 {
        if obs.lp[p] != got.lp[p] {
            duel.set_lp(p as u32, obs.lp[p]);
            changes += 1;
        }
    }
    changes
}
