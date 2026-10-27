//! The heuristic player: a fixed order of priorities in the Main Phase and in battle, chains above
//! 2000 LP, yes to optional effects in its own turn, cards chosen by what the selection is for (the
//! core's select hint before it). Repeated activations of the same effect are capped per turn
//! (`ActivationCaps`). It is the search's rules (crate::search).

use std::cmp::Ordering;

use ygo_core::constants::*;
use ygo_core::types::CardData;

use crate::actions::{Command, Response, available_places, battle_options, idle_options};
use crate::agents::{ActivationCaps, random_response};
use crate::game::{Agent, AgentError, card_db};
use crate::messages::{Card, Msg};
use crate::pyrandom::PyRandom;
use crate::state::Snapshot;

pub const CHAIN_MIN_LP: i32 = 2000;
const UNKNOWN_VALUE: f64 = 1500.0;

// select hints (HINTMSG_* in the card scripts' constant.lua)
const RELEASE: u64 = 500;
const DISCARD: u64 = 501;
const DESTROY: u64 = 502;
const REMOVE: u64 = 503;
const TOGRAVE: u64 = 504;
const RTOHAND: u64 = 505;
const ATOHAND: u64 = 506;
const TODECK: u64 = 507;
const SUMMON: u64 = 508;
const SPSUMMON: u64 = 509;
const FMATERIAL: u64 = 511;
const SMATERIAL: u64 = 512;
const XMATERIAL: u64 = 513;
const FACEUP: u64 = 514;
const FACEDOWN: u64 = 515;
const EQUIP: u64 = 518;
const REMOVEXYZ: u64 = 519;
const CONTROL: u64 = 520;
const TOFIELD: u64 = 527;
const POSCHANGE: u64 = 528;
const TRIBUTE: u64 = 531;
const DEATTACHFROM: u64 = 532;
const LMATERIAL: u64 = 533;
const ATTACKTARGET: u64 = 549;
const TARGET: u64 = 551;
const NEGATE: u64 = 575;
/// own cards: the best first (else the least valuable first: costs, materials)
const OWN_BEST: [u64; 6] = [ATOHAND, SUMMON, SPSUMMON, FACEUP, EQUIP, TOFIELD];
/// the opponent's cards: the least valuable first (else the most valuable: removal)
const OPP_WORST: [u64; 1] = [EQUIP];
/// hints with a rule (the others, or none: as listed)
pub const RULED: [u64; 26] = [
    RELEASE, DISCARD, DESTROY, REMOVE, TOGRAVE, RTOHAND, ATOHAND, TODECK, SUMMON, SPSUMMON, FMATERIAL, SMATERIAL, XMATERIAL, FACEUP, FACEDOWN,
    EQUIP, REMOVEXYZ, CONTROL, TOFIELD, POSCHANGE, TRIBUTE, DEATTACHFROM, LMATERIAL, ATTACKTARGET, TARGET, NEGATE,
];

pub fn card(code: u32) -> Option<&'static CardData> {
    if code == 0 {
        return None;
    }
    card_db().cards.get(&code)
}

/// The card's ATK in the database (0 if unknown), for ordering summons and attacks.
pub fn db_attack(code: u32) -> i32 {
    card_db().cards.get(&code).map_or(0, |c| c.attack)
}

/// How much a card is worth (in ATK points): a monster its ATK (DEF if higher), a Spell/Trap a
/// fixed amount, a face-down or unknown card UNKNOWN_VALUE.
pub fn card_value(entry: &Card, c: Option<&CardData>) -> f64 {
    let Some(c) = c.filter(|_| entry.code != 0) else { return UNKNOWN_VALUE };
    if c.type_ & TYPE_MONSTER != 0 {
        return c.attack.max(c.defense).max(0) as f64;
    }
    if c.type_ & TYPE_SPELL != 0 { UNKNOWN_VALUE + 300.0 } else { UNKNOWN_VALUE }
}

/// A card to add to the hand: a monster by its ATK, less the more tributes its Normal Summon needs.
fn to_hand_value(c: Option<&CardData>) -> f64 {
    let Some(c) = c else { return UNKNOWN_VALUE };
    if c.type_ & TYPE_MONSTER != 0 {
        let tributes = if c.level <= 4 { 0 } else if c.level <= 6 { 1 } else { 2 };
        return (c.attack.max(0) - 1200 * tributes) as f64;
    }
    if c.type_ & TYPE_SPELL != 0 { UNKNOWN_VALUE + 300.0 } else { UNKNOWN_VALUE }
}

/// An attack target's stat to beat: ATK in Attack Position, DEF in Defense.
fn battle_value(entry: &Card, c: Option<&CardData>) -> f64 {
    let pos = entry.pos.unwrap_or(0);
    let Some(c) = c.filter(|_| entry.code != 0 && pos & POS_FACEDOWN == 0) else { return UNKNOWN_VALUE };
    (if pos & POS_DEFENSE != 0 { c.defense } else { c.attack }) as f64
}

#[derive(Clone)]
pub struct Heuristic {
    pub player: u32,
    pub rng: PyRandom,
    pub caps: ActivationCaps,
    /// the core's last select hint to this player
    pub hint: Option<u64>,
}

impl Heuristic {
    pub fn new(seed: i128, player: u32) -> Heuristic {
        Heuristic { player, rng: PyRandom::new(seed), caps: ActivationCaps::default(), hint: None }
    }

    /// The effect (or card) that `r` activates or Special Summons at `prompt` (an idle, battle or
    /// chain decision), whose uses this turn are capped.
    fn capped_use<'a>(prompt: &'a Msg, r: &Response) -> Option<&'a Card> {
        let i = usize::try_from(r.index?).ok()?;
        match (prompt, r.kind) {
            (Msg::SelectChain { chains, .. }, _) => chains.get(i),
            (_, Some(command @ (Command::Activate | Command::Spsummon))) => command.cards(prompt)?.get(i),
            _ => None,
        }
    }

    /// Whether `r` at `prompt` stays within the uses per turn (MAX_SAME_ACTIVATION_PER_TURN) of
    /// what it activates; another player's (a search's) choices are capped like the rules' own.
    pub fn within_cap(&self, prompt: &Msg, r: &Response) -> bool {
        Self::capped_use(prompt, r).is_none_or(|e| self.caps.allowed(e))
    }

    /// Counts `r` at `prompt` (chosen by someone else, e.g. the search) towards the cap.
    pub fn note_choice(&mut self, prompt: &Msg, r: &Response) {
        if let Some(e) = Self::capped_use(prompt, r) {
            self.caps.note(e);
        }
    }

    pub fn observe_msg(&mut self, m: &Msg) {
        self.caps.observe(m);
        match m {
            Msg::Hint { hint_type, player, data } if *hint_type == HINT_SELECTMSG && *player == self.player => self.hint = Some(*data),
            _ => {}
        }
    }

    pub fn respond_to(&mut self, prompt: &Msg, state: Option<&Snapshot>) -> Result<Response, AgentError> {
        let r = self.respond_inner(prompt, state);
        self.hint = None; // (a hint belongs to the selection right after it)
        r
    }

    fn respond_inner(&mut self, prompt: &Msg, state: Option<&Snapshot>) -> Result<Response, AgentError> {
        let me = self.player;
        let own_turn = state.is_some_and(|s| s.turn_player == me);
        match prompt {
            Msg::SelectIdlecmd { .. } => return Ok(self.main(prompt, state)),
            Msg::SelectBattlecmd { activatable, .. } => {
                let opts = battle_options(prompt);
                for o in &opts {
                    if o.kind == Some(Command::Activate) {
                        let e = &activatable[o.index.unwrap() as usize];
                        if self.caps.allowed(e) {
                            self.caps.note(e);
                            return Ok(o.clone());
                        }
                    }
                }
                for command in [Command::Attack, Command::ToM2, Command::ToEp] {
                    if let Some(o) = opts.iter().find(|o| o.kind == Some(command)) {
                        return Ok(o.clone());
                    }
                }
                return Ok(opts[0].clone());
            }
            Msg::SelectChain { forced, chains, .. } => {
                if *forced != 0 {
                    return Ok(Response::index(0));
                }
                if state.is_some_and(|s| s.lp[me as usize] > CHAIN_MIN_LP) {
                    for (i, c) in chains.iter().enumerate() {
                        if self.caps.allowed(c) {
                            self.caps.note(c);
                            return Ok(Response::index(i));
                        }
                    }
                }
                return Ok(Response::pass());
            }
            Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => return Ok(Response::yes(own_turn)),
            Msg::SelectPosition { code, positions, .. } => {
                let order = if card(*code).is_some_and(|c| c.attack <= 0) {
                    [POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK]
                } else {
                    [POS_FACEUP_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEDOWN_ATTACK]
                };
                for pos in order {
                    if positions & pos != 0 {
                        return Ok(Response { position: Some(pos as i64), ..Response::default() });
                    }
                }
            }
            _ => {}
        }
        self.first(prompt)
    }

    fn main(&mut self, prompt: &Msg, state: Option<&Snapshot>) -> Response {
        let Msg::SelectIdlecmd { summonable, spsummonable, repositionable, activatable, .. } = prompt else { unreachable!() };
        let me = self.player as usize;
        let opts = idle_options(prompt);
        let by_kind = |command: Command| -> Vec<&Response> { opts.iter().filter(|o| o.kind == Some(command)).collect() };
        if let Some(s) = state
            && s.phase == PHASE_MAIN2
            && let Some(o) = by_kind(Command::ToEp).first()
        {
            return (*o).clone(); // (after the battle: end the turn)
        }
        let own: Vec<_> = state.map(|s| s.players[me].mzone.iter().flatten().collect()).unwrap_or_default();
        let opp_best = state
            .map(|s| {
                s.players[1 - me]
                    .mzone
                    .iter()
                    .flatten()
                    .filter(|c| c.position.unwrap_or(0) & POS_FACEUP != 0)
                    .map(|c| c.attack.unwrap_or(0))
                    .max()
                    .unwrap_or(-1)
            })
            .unwrap_or(-1);
        // a. activate a card or effect
        for o in by_kind(Command::Activate) {
            let e = &activatable[o.index.unwrap() as usize];
            if self.caps.allowed(e) {
                self.caps.note(e);
                return o.clone();
            }
        }
        // b. Special Summon
        for o in by_kind(Command::Spsummon) {
            let e = &spsummonable[o.index.unwrap() as usize];
            if self.caps.allowed(e) {
                self.caps.note(e);
                return o.clone();
            }
        }
        // c. Normal Summon: on an empty field, or a monster stronger than the opponent's best
        for o in by_kind(Command::Summon) {
            if own.is_empty() || db_attack(summonable[o.index.unwrap() as usize].code) > opp_best {
                return o.clone();
            }
        }
        // d. set a monster when behind on an empty field
        let behind = state.is_some_and(|s| s.lp[me] < s.lp[1 - me]);
        if behind && own.is_empty() && let Some(o) = by_kind(Command::Mset).first() {
            return (*o).clone();
        }
        // e. change the position of a monster stuck in Defense Position
        for o in by_kind(Command::Repos) {
            let c = &repositionable[o.index.unwrap() as usize];
            let here = state.filter(|_| c.loc == LOCATION_MZONE).and_then(|s| s.players[me].mzone.get(c.seq as usize).cloned().flatten());
            if here.is_some_and(|h| h.position.unwrap_or(0) & POS_DEFENSE != 0) {
                return o.clone();
            }
        }
        // f. battle, with an attacker that has ATK
        let attackers = own.iter().any(|c| c.position.unwrap_or(0) & POS_FACEUP_ATTACK != 0 && c.attack.unwrap_or(0) > 0);
        if attackers && let Some(o) = by_kind(Command::ToBp).first() {
            return (*o).clone();
        }
        // g. set a Spell/Trap, h. Normal Summon anyway, i. end the turn
        for command in [Command::Sset, Command::Summon, Command::ToEp] {
            if let Some(o) = by_kind(command).first() {
                return (*o).clone();
            }
        }
        opts[0].clone()
    }

    /// The cards' indices, the preferred first, by the select hint (see the Python docstring).
    pub fn card_order(&self, cards: &[Card]) -> Vec<usize> {
        let Some(hint) = self.hint.filter(|h| RULED.contains(h)) else { return (0..cards.len()).collect() };
        let key = |i: usize| -> (i32, f64) {
            let c = &cards[i];
            let data = card(c.code);
            let own = c.con == self.player;
            if hint == ATTACKTARGET {
                // (a bare number in Python: compared with the others the same way)
                return (0, battle_value(c, data));
            }
            let v = if hint == ATOHAND { to_hand_value(data) } else { card_value(c, data) };
            if own {
                (1, if OWN_BEST.contains(&hint) { -v } else { v })
            } else {
                (0, if OPP_WORST.contains(&hint) { v } else { -v })
            }
        };
        let mut order: Vec<usize> = (0..cards.len()).collect();
        order.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap_or(Ordering::Equal));
        order
    }

    /// The preferred cards, else the first place or option listed; random where that does not
    /// make a valid answer (sums, counters).
    fn first(&mut self, prompt: &Msg) -> Result<Response, AgentError> {
        match prompt {
            Msg::SelectCard { cards, min, max, .. } if !cards.is_empty() && *max >= 1 => {
                let k = (*min.max(&1)).min(*max).min(cards.len() as u32) as usize;
                let mut idx: Vec<i64> = self.card_order(cards)[..k].iter().map(|&i| i as i64).collect();
                idx.sort();
                return Ok(Response { indices: Some(idx), ..Response::default() });
            }
            Msg::SelectUnselectCard { select_cards, finishable, cancelable, .. } => {
                if !select_cards.is_empty() {
                    return Ok(Response::index(self.card_order(select_cards)[0]));
                }
                if *finishable != 0 || *cancelable != 0 {
                    return Ok(Response { finish: Some(true), ..Response::default() });
                }
            }
            Msg::SelectTribute { cards, min, max, .. } if !cards.is_empty() => {
                // (the least valuable first, until enough is released)
                let mut order: Vec<usize> = (0..cards.len()).collect();
                let v = |i: usize| card_value(&cards[i], card(cards[i].code));
                order.sort_by(|&a, &b| v(a).partial_cmp(&v(b)).unwrap_or(Ordering::Equal));
                let (mut chosen, mut total) = (Vec::new(), 0u32);
                for i in order {
                    if total >= *min && !chosen.is_empty() {
                        break;
                    }
                    chosen.push(i as i64);
                    total += cards[i].release_param.unwrap_or(0);
                }
                if *min <= total && chosen.len() <= (*max).max(1) as usize {
                    chosen.sort();
                    return Ok(Response { indices: Some(chosen), ..Response::default() });
                }
            }
            Msg::SelectPlace { player, count, flag } | Msg::SelectDisfield { player, count, flag } => {
                let places = available_places(*player, *flag);
                let n = (*count as usize).min(places.len());
                return Ok(Response { places: Some(places[..n].to_vec()), ..Response::default() });
            }
            Msg::SelectOption { .. } | Msg::AnnounceNumber { .. } => return Ok(Response::index(0)),
            Msg::SortCard { cards, .. } | Msg::SortChain { cards, .. } => {
                return Ok(Response { order: Some((0..cards.len() as i64).collect()), ..Response::default() });
            }
            _ => {}
        }
        random_response(prompt, &mut self.rng, 0.0)
    }
}

/// The heuristic player as an agent.
pub struct HeuristicAgent(pub Heuristic);

impl HeuristicAgent {
    pub fn new(seed: i64) -> HeuristicAgent {
        HeuristicAgent(Heuristic::new(seed as i128, 0))
    }
}

impl Agent for HeuristicAgent {
    fn name(&self) -> String {
        "heuristic".into()
    }
    fn wants_state(&self) -> bool {
        true
    }
    fn start(&mut self, player: u32, _init: &crate::game::InitInfo) -> Result<(), AgentError> {
        self.0.player = player;
        Ok(())
    }
    fn observe(&mut self, msg: &Msg) {
        self.0.observe_msg(msg);
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
        self.0.respond_to(prompt, state)
    }
}
