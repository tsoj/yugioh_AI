//! The Rust engine's duel with the interface of ygoai.core.Duel (ygoai/core_rust.py): the
//! extension API (changes made while a card function waits are recorded and re-applied, see
//! Duel::outside_change) and the questions the C++ core answered with Lua chunks. Used by the
//! belief replay; the Python binding exposes the same calls.

use ygo_core::api::{NewCardInfo, QueryInfo};
use ygo_core::constants::*;
use ygo_core::duel::Duel;
use ygo_core::ext::{FORCED_COIN, FORCED_DICE, ForcedLoc};

use crate::belief::infer::PlanEntry;
use crate::game::{DuelConfig, new_duel};
use crate::messages::{Msg, ParseError, parse, split_messages};

/// A candidate's properties for `eligible_as`/`settable_as`: code, type, Level, attribute, race,
/// ATK, DEF (ATK/DEF not below 0).
pub type Props = (u32, u32, u32, u32, u64, i32, i32);

/// Runs `f` as the C++ core runs a Lua chunk given to `OCG_LoadScript`: one top-level call in
/// which no action is allowed, whose assumed card properties are restored at its end. `None` if a
/// script error happened; the errors are not kept in the logs.
pub fn as_query<T>(d: &mut Duel, f: impl FnOnce(&mut Duel) -> T) -> Option<T> {
    let before = d.logs.len();
    d.no_action += 1;
    d.call_depth += 1;
    let r = ygo_core::api::protected(d, None, |d| Some(f(d)));
    d.call_depth -= 1;
    if d.call_depth == 0 {
        d.restore_assumes();
    }
    d.no_action -= 1;
    let error = d.logs[before..].iter().any(|l| l.0 == 0);
    d.logs.truncate(before);
    if error { None } else { r }
}

fn assume_props(d: &mut Duel, c: ygo_core::types::CardId, p: &Props) {
    let card = ygo_core::api::card::assume_property;
    card(d, c, ASSUME_CODE, p.0 as u64);
    card(d, c, ASSUME_TYPE, p.1 as u64);
    card(d, c, ASSUME_LEVEL, p.2 as u64);
    card(d, c, ASSUME_ATTRIBUTE, p.3 as u64);
    card(d, c, ASSUME_RACE, p.4);
    card(d, c, ASSUME_ATTACK, p.5 as u64);
    card(d, c, ASSUME_DEFENSE, p.6 as u64);
}

/// `c:Recreate(code, ...)` for the card with this id, keeping its effects (as the replay's Lua for
/// the C++ core does: `replace_effect` was nil).
pub fn recreate_card(d: &mut Duel, card_id: u32, code: u32) {
    d.outside_change(move |d| {
        let Some(c) = d.card_by_id(card_id) else { return };
        as_query(d, |d| ygo_core::api::card::recreate(d, c, code, |_| {}, false));
    });
}

/// Whether the card at (con, loc, seq) is there by its own effect.
pub fn reason_effect_is_own(d: &mut Duel, con: u8, loc: u32, seq: u32) -> bool {
    as_query(d, |d| {
        let Some(c) = ygo_core::api::duel::get_field_card(d, con, loc, seq) else { return false };
        let Some(re) = ygo_core::api::card::get_reason_effect(d, c) else { return false };
        ygo_core::api::effect::get_handler(d, re) == Some(c)
    })
    .unwrap_or(false)
}

/// Whether the monster in zone `seq` may be Tributed by the other player for any Tribute Summon.
pub fn has_extra_release(d: &mut Duel, con: u8, seq: u32) -> bool {
    as_query(d, |d| {
        let Some(c) = ygo_core::api::duel::get_field_card(d, con, LOCATION_MZONE, seq) else { return false };
        !ygo_core::api::card::is_has_effect(d, c, EFFECT_EXTRA_RELEASE_SUM, None).is_empty()
            || !ygo_core::api::card::is_has_effect(d, c, EFFECT_EXTRA_RELEASE, None).is_empty()
    })
    .unwrap_or(false)
}

/// For each candidate: whether the hidden card at `li` would pass the check if it had the
/// candidate's properties: kind "tribute" (in the Tribute group of the card at `src`), "attack"
/// (an attack target of it) or "tribute_of" (the card at `src` in its Tribute group). Empty if a
/// card is missing or on errors.
pub fn eligible_as(d: &mut Duel, li: (u8, u32, u32), kind: &str, src: (u8, u32, u32), cands: &[Props]) -> Vec<(u32, bool)> {
    let kind = match kind {
        "tribute" => 0,
        "attack" => 1,
        _ => 2,
    };
    as_query(d, |d| {
        let c = ygo_core::api::duel::get_field_card(d, src.0, src.1, src.2);
        let t = if li.1 & (LOCATION_MZONE | LOCATION_SZONE) != 0 {
            ygo_core::api::duel::get_field_card(d, li.0, li.1, li.2)
        } else {
            let g = ygo_core::api::duel::get_field_group(d, li.0, li.1, 0);
            g.iter().find(|&x| d.card(x).current.sequence == li.2)
        };
        let mut out = Vec::new();
        let (Some(c), Some(t)) = (c, t) else { return out };
        for p in cands {
            assume_props(d, t, p);
            let ok = match kind {
                0 => ygo_core::api::duel::get_tribute_group(d, c).contains(t),
                1 => ygo_core::api::card::get_attackable_target(d, c).0.contains(t),
                _ => ygo_core::api::duel::get_tribute_group(d, t).contains(c),
            };
            out.push((p.0, ok));
            ygo_core::api::duel::assume_reset(d);
        }
        out
    })
    .unwrap_or_default()
}

/// For each candidate (None: the card as it is): whether its player may Set the hand card at
/// (con, seq) with the summon type `sumtype` if it had the candidate's properties, and with
/// tributes, whether it can Tribute exactly the monsters at `trib_locs`.
pub fn settable_as(d: &mut Duel, li: (u8, u32), sumtype: u32, trib_locs: &[(u8, u32)], cands: &[Option<Props>]) -> Vec<(Option<u32>, bool)> {
    as_query(d, |d| {
        let mut out = Vec::new();
        let Some(c) = ygo_core::api::duel::get_field_card(d, li.0, LOCATION_HAND, li.1) else { return out };
        let mut g = ygo_core::types::CardSet::new();
        for &(con, seq) in trib_locs {
            if let Some(t) = ygo_core::api::duel::get_field_card(d, con, LOCATION_MZONE, seq) {
                g.insert(t);
            }
        }
        let n = trib_locs.len() as u16;
        let check = |d: &mut Duel| {
            ygo_core::api::duel::can_player_set_monster(d, li.0, Some((sumtype, c)))
                && (n == 0
                    || (ygo_core::api::duel::check_tribute(d, c, n, Some(n), Some(&g), None, None)
                        && ygo_cards::lua::card::get_tribute_requirement(d, c).1 >= 1))
        };
        for p in cands {
            match p {
                None => out.push((None, check(d))),
                Some(p) => {
                    assume_props(d, c, p);
                    let ok = check(d);
                    out.push((Some(p.0), ok));
                    ygo_core::api::duel::assume_reset(d);
                }
            }
        }
        out
    })
    .unwrap_or_default()
}

/// The parsed messages of the last process call.
pub fn messages(d: &mut Duel) -> Result<Vec<Msg>, ParseError> {
    split_messages(d.ocg_get_message()).into_iter().map(parse).collect()
}

/// A duel of the belief replay: the engine's duel, stopping where the C++ core stops (fine
/// steps), plus what ygoai's Python Duel object carries (the shuffle plan the replay sets for
/// resolving references, whether its read log is on).
#[derive(Clone)]
pub struct RDuel {
    pub d: Duel,
    pub shuffle_plan: Vec<PlanEntry>,
    pub reads_on: bool,
}

impl RDuel {
    pub fn new(seed: [u64; 4], config: &DuelConfig) -> RDuel {
        let mut d = new_duel(seed, config);
        d.fine_steps = true;
        RDuel { d, shuffle_plan: Vec::new(), reads_on: false }
    }
    pub fn from_duel(mut d: Duel) -> RDuel {
        d.fine_steps = true;
        RDuel { d, shuffle_plan: Vec::new(), reads_on: false }
    }
    pub fn new_card(&mut self, owner: u8, code: u32, location: u32) {
        let info = NewCardInfo { team: owner, duelist: 0, code, con: owner, loc: location, seq: 0, pos: POS_FACEDOWN_DEFENSE };
        self.d.ocg_new_card(&info);
    }
    pub fn start(&mut self) {
        self.d.ocg_start_duel();
    }
    pub fn process(&mut self) -> i32 {
        self.d.ocg_process() as i32
    }
    pub fn get_messages(&mut self) -> Result<Vec<Msg>, ParseError> {
        messages(&mut self.d)
    }
    pub fn set_response(&mut self, data: &[u8]) {
        self.d.ocg_set_response(data);
    }
    pub fn location_cards(&self, player: u32, location: u32) -> Vec<(u32, u32)> {
        self.d.ext_get_location_cards(player as u8, location)
    }
    pub fn overlay_cards(&self, player: u32, location: u32, sequence: u32) -> Vec<(u32, u32)> {
        self.d.ext_get_overlay_cards(player as u8, location, sequence)
    }
    pub fn all_cards(&self) -> Vec<[u32; 6]> {
        self.d.ext_get_all_cards()
    }
    /// (turn, turn player, phase, [lp0, lp1])
    pub fn info(&self) -> (u32, u32, u32, [i32; 2]) {
        let [t, tp, ph, a, b] = self.d.ext_get_info();
        (t, tp, ph, [a as i32, b as i32])
    }
    pub fn query_location(&mut self, player: u32, location: u32, flags: u32) -> Vec<u8> {
        self.d.ocg_query_location(&QueryInfo { flags, con: player as u8, loc: location, seq: 0, overlay_seq: 0 })
    }
    pub fn set_location_order(&mut self, player: u32, location: u32, codes: &[u32]) -> bool {
        let codes = codes.to_vec();
        self.d.outside_change(move |d| d.ext_set_location_order(player as u8, location, &codes))
    }
    pub fn set_location_order_by_id(&mut self, player: u32, location: u32, ids: &[u32]) -> bool {
        let ids = ids.to_vec();
        self.d.outside_change(move |d| d.ext_set_location_order_by_id(player as u8, location, &ids))
    }
    pub fn push_forced_coins(&mut self, results: &[u32]) {
        let r: Vec<i32> = results.iter().map(|&x| x as i32).collect();
        self.d.outside_change(move |d| d.ext_push_forced_ints(FORCED_COIN, &r));
    }
    pub fn push_forced_dice(&mut self, results: &[u32]) {
        let r: Vec<i32> = results.iter().map(|&x| x as i32).collect();
        self.d.outside_change(move |d| d.ext_push_forced_ints(FORCED_DICE, &r));
    }
    pub fn push_forced_shuffle(&mut self, player: u32, location: u32, codes: &[u32]) {
        let codes = codes.to_vec();
        self.d.outside_change(move |d| d.ext_push_forced_shuffle(player as u8, location, &codes));
    }
    pub fn push_forced_shuffle_ids(&mut self, player: u32, location: u32, ids: &[u32], codes: &[u32]) {
        // (the C++ API takes one count for both arrays: the ids')
        let codes: Vec<u32> = (0..ids.len()).map(|i| codes.get(i).copied().unwrap_or(0)).collect();
        let ids = ids.to_vec();
        self.d.outside_change(move |d| d.ext_push_forced_shuffle_by_id(player as u8, location, &ids, &codes));
    }
    pub fn push_forced_random_select(&mut self, locs: &[(u32, u32, u32, u32)]) {
        let locs: Vec<ForcedLoc> = locs.iter().map(|&(con, loc, seq, ov)| ForcedLoc { con: con as u8, loc: loc as u8, seq, overlay_seq: ov }).collect();
        self.d.outside_change(move |d| d.ext_push_forced_random_select(&locs));
    }
    pub fn push_forced_set_card_shuffle(&mut self, codes: &[u32]) {
        let codes = codes.to_vec();
        self.d.outside_change(move |d| d.ext_push_forced_set_card_shuffle(&codes));
    }
    pub fn push_forced_set_card_shuffle_ids(&mut self, ids: &[u32], codes: &[u32]) {
        let codes: Vec<u32> = (0..ids.len()).map(|i| codes.get(i).copied().unwrap_or(0)).collect();
        let ids = ids.to_vec();
        self.d.outside_change(move |d| d.ext_push_forced_set_card_shuffle_by_id(&ids, &codes));
    }
    pub fn force_activatable(&mut self, player: u32, code: u32, desc: u64) {
        self.d.outside_change(move |d| d.ext_force_activatable(player as u8, code, desc));
    }
    pub fn clear_force_activatable(&mut self) {
        self.d.outside_change(|d| d.ext_clear_force_activatable());
    }
    pub fn select_chains(&self) -> Vec<(u32, u64)> {
        self.d.ext_select_chains()
    }
    pub fn set_forced_match(&mut self, ids: &[u32]) {
        let ids = ids.to_vec();
        self.d.outside_change(move |d| d.ext_set_forced_match(&ids));
    }
    pub fn set_forced_nomatch(&mut self, ids: &[u32]) {
        let ids = ids.to_vec();
        self.d.outside_change(move |d| d.ext_set_forced_nomatch(&ids));
    }
    pub fn take_forced_used(&mut self) -> Vec<(u32, u64)> {
        self.d.outside_change(|d| d.ext_take_forced_used())
    }
    pub fn swap_cards(&mut self, a: u32, b: u32) -> bool {
        self.d.outside_change(move |d| d.ext_swap_cards(a, b))
    }
    pub fn place_card(&mut self, card_id: u32, controller: u32, location: u32, sequence: u32, position: u32) -> bool {
        self.d.outside_change(move |d| d.ext_place_card(card_id, controller as u8, location, sequence, position))
    }
    pub fn set_lp(&mut self, player: u32, lp: i32) {
        self.d.outside_change(move |d| d.ext_set_lp(player as u8, lp));
    }
    pub fn set_card_info(&mut self, card_id: u32, reason: Option<u32>, status: u32, mask: u32) -> bool {
        let reason = reason.unwrap_or(0xFFFF_FFFF);
        self.d.outside_change(move |d| d.ext_set_card_info(card_id, reason, status, mask))
    }
    /// A new card outside the game (location 0); its card id.
    pub fn create_card(&mut self, owner: u32, code: u32) -> u32 {
        self.d.outside_change(move |d| d.ext_create_card(owner as u8, code))
    }
    pub fn set_turn(&mut self, turn: u32) {
        self.d.outside_change(move |d| d.ext_set_turn(turn));
    }
    pub fn restart_command(&mut self) -> bool {
        self.d.ext_restart_command()
    }
    pub fn forced_errors(&self) -> u32 {
        self.d.ext_get_errors()
    }
    /// The read log (card id -> properties read) since the last call; `enable` keeps logging.
    pub fn read_log(&mut self, enable: bool) -> Vec<(u32, u32)> {
        let mut v: Vec<(u32, u32)> = self.d.outside_change(move |d| d.ext_read_log(enable)).into_iter().collect();
        v.sort();
        v
    }
    pub fn recreate_card(&mut self, card_id: u32, code: u32) {
        recreate_card(&mut self.d, card_id, code)
    }
    pub fn reason_effect_is_own(&mut self, con: u32, loc: u32, seq: u32) -> bool {
        reason_effect_is_own(&mut self.d, con as u8, loc, seq)
    }
    pub fn has_extra_release(&mut self, con: u32, seq: u32) -> bool {
        has_extra_release(&mut self.d, con as u8, seq)
    }
    /// The script errors so far (log type 0).
    pub fn script_errors(&self) -> usize {
        self.d.logs.iter().filter(|l| l.0 == 0).count()
    }
}
