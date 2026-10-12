//! Python binding of the Rust duel engine (`rust/`, ygopro-core-rust): `RustDuel` has the methods of
//! `ygoai.core.Duel` (the ctypes wrapper of the patched C++ core) with the same arguments and
//! results, plus `clone()`. `ygoai/core_rust.py` wraps it into a drop-in `Duel`.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use ygo_core::api::{NewCardInfo, QueryInfo};
use ygo_core::carddb::CardDb;
use ygo_core::duel::{Duel, DuelOptions, PlayerOptions};
use ygo_core::ext::{ForcedLoc, FORCED_COIN, FORCED_DICE};
use ygo_core::script::Scripts;

static SHARED: OnceLock<(Arc<CardDb>, Arc<Scripts>)> = OnceLock::new();

/// Loads the card database (the BabelCDB directory) and the card scripts once per process.
#[pyfunction]
fn init(cdb_dir: &str) -> PyResult<()> {
    if SHARED.get().is_some() {
        return Ok(());
    }
    let db = CardDb::load(Path::new(cdb_dir)).map_err(|e| PyRuntimeError::new_err(format!("card database {cdb_dir}: {e}")))?;
    let _ = SHARED.set((Arc::new(db), Arc::new(ygo_cards::scripts())));
    Ok(())
}

fn shared() -> PyResult<&'static (Arc<CardDb>, Arc<Scripts>)> {
    SHARED.get().ok_or_else(|| PyRuntimeError::new_err("ygoai_rs.init(cdb_dir) was not called"))
}

/// The messages of an `ocg_get_message` buffer (each a u32 length, then the message).
fn split_messages(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 4 <= buf.len() {
        let n = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        out.push(buf[pos..pos + n].to_vec());
        pos += n;
    }
    out
}

#[pyclass(module = "ygoai_rs")]
struct RustDuel {
    d: Duel,
}

#[pymethods]
impl RustDuel {
    #[new]
    fn new(seed: [u64; 4], flags: u64, start_lp: u32, start_hand: u32, draw_per_turn: u32) -> PyResult<Self> {
        let (db, scripts) = shared()?;
        let mut seed = seed;
        if seed.iter().all(|&s| s == 0) {
            seed[0] = 1;
        }
        let p = PlayerOptions { starting_lp: start_lp, starting_draw_count: start_hand, draw_count_per_turn: draw_per_turn };
        let mut d = Duel::new(&DuelOptions { seed, flags, team1: p, team2: p }, db.clone(), scripts.clone());
        ygo_cards::load_global_scripts(&mut d);
        // (stop where the C++ core stops: the belief replay changes the duel between process calls)
        d.fine_steps = true;
        Ok(RustDuel { d })
    }

    /// A copy of the duel (cheap: cards and effects are shared until changed).
    fn clone(&self) -> RustDuel {
        RustDuel { d: self.d.clone() }
    }

    /// Whether `process` stops inside card functions where the C++ core does (default: on; off is
    /// faster, for callers that do not change the duel between `process` calls).
    fn set_fine_steps(&mut self, on: bool) {
        self.d.fine_steps = on;
    }

    /// The log messages since the last call: (type, text); type 0 is a script error.
    fn take_logs(&mut self) -> Vec<(u32, String)> {
        std::mem::take(&mut self.d.logs)
    }

    #[pyo3(signature = (owner, code, location, controller, sequence, position))]
    fn new_card(&mut self, owner: u8, code: u32, location: u32, controller: u8, sequence: u32, position: u32) {
        let info = NewCardInfo { team: owner, duelist: 0, code, con: controller, loc: location, seq: sequence, pos: position };
        self.d.ocg_new_card(&info);
    }
    fn start(&mut self) {
        self.d.ocg_start_duel();
    }
    fn process(&mut self) -> i32 {
        self.d.ocg_process() as i32
    }
    fn get_messages<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        split_messages(self.d.ocg_get_message()).iter().map(|m| PyBytes::new(py, m)).collect()
    }
    fn set_response(&mut self, data: &[u8]) {
        self.d.ocg_set_response(data);
    }

    // -- queries ---------------------------------------------------------------------------------
    fn query_count(&self, player: u8, location: u32) -> u32 {
        self.d.ocg_query_count(player, location)
    }
    fn query_location_raw<'py>(&mut self, py: Python<'py>, player: u8, location: u32, flags: u32) -> Bound<'py, PyBytes> {
        let info = QueryInfo { flags, con: player, loc: location, seq: 0, overlay_seq: 0 };
        PyBytes::new(py, &self.d.ocg_query_location(&info))
    }
    #[pyo3(signature = (player, location, sequence, flags, overlay_seq=0))]
    fn query_card_raw<'py>(&mut self, py: Python<'py>, player: u8, location: u32, sequence: u32, flags: u32, overlay_seq: u32) -> Bound<'py, PyBytes> {
        let info = QueryInfo { flags, con: player, loc: location, seq: sequence, overlay_seq };
        PyBytes::new(py, &self.d.ocg_query(&info))
    }
    fn query_field_raw<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.d.ocg_query_field())
    }

    // -- extension API ---------------------------------------------------------------------------
    fn location_cards(&self, player: u8, location: u32) -> Vec<(u32, u32)> {
        self.d.ext_get_location_cards(player, location)
    }
    fn overlay_cards(&self, player: u8, location: u32, sequence: u32) -> Vec<(u32, u32)> {
        self.d.ext_get_overlay_cards(player, location, sequence)
    }
    /// Every slot of both players: 6 little-endian u32 per entry (see parse_all_cards).
    fn all_cards_raw<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut buf = Vec::new();
        for e in self.d.ext_get_all_cards() {
            for v in e {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        PyBytes::new(py, &buf)
    }
    fn chain_related(&self) -> Vec<bool> {
        self.d.ext_chain_related()
    }
    #[pyo3(signature = (enable=true))]
    fn read_log<'py>(&mut self, py: Python<'py>, enable: bool) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (id, mask) in self.d.outside_change(move |d| d.ext_read_log(enable)) {
            out.set_item(id, mask)?;
        }
        Ok(out)
    }
    /// [turn, turn player, phase, lp0, lp1]
    fn info(&self) -> [u32; 5] {
        self.d.ext_get_info()
    }
    fn set_location_order(&mut self, player: u8, location: u32, codes: Vec<u32>) -> bool {
        self.d.outside_change(move |d| d.ext_set_location_order(player, location, &codes))
    }
    fn set_location_order_by_id(&mut self, player: u8, location: u32, ids: Vec<u32>) -> bool {
        self.d.outside_change(move |d| d.ext_set_location_order_by_id(player, location, &ids))
    }
    fn push_forced_coins(&mut self, results: Vec<i32>) {
        self.d.outside_change(move |d| d.ext_push_forced_ints(FORCED_COIN, &results));
    }
    fn push_forced_dice(&mut self, results: Vec<i32>) {
        self.d.outside_change(move |d| d.ext_push_forced_ints(FORCED_DICE, &results));
    }
    fn push_forced_shuffle(&mut self, player: u8, location: u32, codes: Vec<u32>) {
        self.d.outside_change(move |d| d.ext_push_forced_shuffle(player, location, &codes));
    }
    fn push_forced_shuffle_ids(&mut self, player: u8, location: u32, ids: Vec<u32>, codes: Vec<u32>) {
        // (the C++ API takes one count for both arrays: the ids')
        let codes: Vec<u32> = (0..ids.len()).map(|i| codes.get(i).copied().unwrap_or(0)).collect();
        self.d.outside_change(move |d| d.ext_push_forced_shuffle_by_id(player, location, &ids, &codes));
    }
    fn push_forced_random_select(&mut self, locs: Vec<(u32, u32, u32, u32)>) {
        let locs: Vec<ForcedLoc> = locs.iter().map(|&(con, loc, seq, ov)| ForcedLoc { con: con as u8, loc: loc as u8, seq, overlay_seq: ov }).collect();
        self.d.outside_change(move |d| d.ext_push_forced_random_select(&locs));
    }
    fn push_forced_set_card_shuffle(&mut self, codes: Vec<u32>) {
        self.d.outside_change(move |d| d.ext_push_forced_set_card_shuffle(&codes));
    }
    fn push_forced_set_card_shuffle_ids(&mut self, ids: Vec<u32>, codes: Vec<u32>) {
        let codes: Vec<u32> = (0..ids.len()).map(|i| codes.get(i).copied().unwrap_or(0)).collect();
        self.d.outside_change(move |d| d.ext_push_forced_set_card_shuffle_by_id(&ids, &codes));
    }
    fn force_activatable(&mut self, player: u8, code: u32, desc: u64) {
        self.d.outside_change(move |d| d.ext_force_activatable(player, code, desc));
    }
    fn clear_force_activatable(&mut self) {
        self.d.outside_change(|d| d.ext_clear_force_activatable());
    }
    fn select_chains(&self) -> Vec<(u32, u64)> {
        self.d.ext_select_chains()
    }
    fn pending_triggers(&self) -> Vec<[u64; 6]> {
        self.d.ext_pending_triggers()
    }
    fn set_forced_match(&mut self, ids: Vec<u32>) {
        self.d.outside_change(move |d| d.ext_set_forced_match(&ids));
    }
    fn set_forced_nomatch(&mut self, ids: Vec<u32>) {
        self.d.outside_change(move |d| d.ext_set_forced_nomatch(&ids));
    }
    fn take_forced_used(&mut self) -> Vec<(u32, u64)> {
        self.d.outside_change(|d| d.ext_take_forced_used())
    }
    fn swap_cards(&mut self, id_a: u32, id_b: u32) -> bool {
        self.d.outside_change(move |d| d.ext_swap_cards(id_a, id_b))
    }
    fn place_card(&mut self, card_id: u32, controller: u8, location: u32, sequence: u32, position: u32) -> bool {
        self.d.outside_change(move |d| d.ext_place_card(card_id, controller, location, sequence, position))
    }
    fn set_lp(&mut self, player: u8, lp: i32) {
        self.d.outside_change(move |d| d.ext_set_lp(player, lp));
    }
    fn create_card(&mut self, owner: u8, code: u32) -> u32 {
        self.d.outside_change(move |d| d.ext_create_card(owner, code))
    }
    fn set_card_info(&mut self, card_id: u32, reason: u32, status: u32, mask: u32) -> bool {
        self.d.outside_change(move |d| d.ext_set_card_info(card_id, reason, status, mask))
    }
    fn set_turn(&mut self, turn: u32) {
        self.d.outside_change(move |d| d.ext_set_turn(turn));
    }
    fn restart_command(&mut self) -> bool {
        self.d.ext_restart_command()
    }
    fn clear_forced(&mut self) {
        self.d.outside_change(|d| d.ext_clear_forced());
    }
    fn pending_forced(&self) -> u32 {
        self.d.ext_pending_forced()
    }
    fn forced_errors(&self) -> u32 {
        self.d.ext_get_errors()
    }
}

/// A candidate's properties for `eligible_as`/`settable_as`: code, type, Level, attribute, race,
/// ATK, DEF (ATK/DEF not below 0).
type Props = (u32, u32, u32, u32, u64, i32, i32);

/// Runs `f` as the C++ core runs a Lua chunk given to `OCG_LoadScript` (`Duel.lua_messages`): one
/// top-level call in which no action is allowed, whose assumed card properties are restored at
/// its end. `None` if a script error happened (in the C++ core its message would have been one of
/// the printed lines, and the result is discarded); the errors are not kept in the logs.
fn as_query<T>(d: &mut Duel, f: impl FnOnce(&mut Duel) -> T) -> Option<T> {
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
    use ygo_core::constants::*;
    let card = ygo_core::api::card::assume_property;
    card(d, c, ASSUME_CODE, p.0 as u64);
    card(d, c, ASSUME_TYPE, p.1 as u64);
    card(d, c, ASSUME_LEVEL, p.2 as u64);
    card(d, c, ASSUME_ATTRIBUTE, p.3 as u64);
    card(d, c, ASSUME_RACE, p.4);
    card(d, c, ASSUME_ATTACK, p.5 as u64);
    card(d, c, ASSUME_DEFENSE, p.6 as u64);
}

#[pymethods]
impl RustDuel {
    // -- the questions the belief replay asked the C++ core as Lua chunks ----------------------------

    /// `c:Recreate(code, nil, ..., true)` for the card with this id.
    fn recreate_card(&mut self, card_id: u32, code: u32) {
        self.d.outside_change(move |d| {
            let Some(c) = d.card_by_id(card_id) else { return };
            as_query(d, |d| ygo_core::api::card::recreate(d, c, code, |_| {}, true));
        });
    }

    /// Whether the card at (con, loc, seq) is there by its own effect (`c:GetReasonEffect()`'s
    /// handler is the card).
    fn reason_effect_is_own(&mut self, con: u8, loc: u32, seq: u32) -> bool {
        as_query(&mut self.d, |d| {
            let Some(c) = ygo_core::api::duel::get_field_card(d, con, loc, seq) else { return false };
            let Some(re) = ygo_core::api::card::get_reason_effect(d, c) else { return false };
            ygo_core::api::effect::get_handler(d, re) == Some(c)
        })
        .unwrap_or(false)
    }

    /// Whether the monster in zone `seq` may be Tributed by the other player for any Tribute
    /// Summon (EFFECT_EXTRA_RELEASE(_SUM)).
    fn has_extra_release(&mut self, con: u8, seq: u32) -> bool {
        use ygo_core::constants::*;
        as_query(&mut self.d, |d| {
            let Some(c) = ygo_core::api::duel::get_field_card(d, con, LOCATION_MZONE, seq) else { return false };
            !ygo_core::api::card::is_has_effect(d, c, EFFECT_EXTRA_RELEASE_SUM, None).is_empty()
                || !ygo_core::api::card::is_has_effect(d, c, EFFECT_EXTRA_RELEASE, None).is_empty()
        })
        .unwrap_or(false)
    }

    /// For each candidate: whether the hidden card `t` at `li` would pass the check if it had the
    /// candidate's properties: kind "tribute" (in the Tribute group of the card at `src`) or
    /// "attack" (an attack target of the card at `src`). Empty if a card is missing or on errors.
    fn eligible_as<'py>(&mut self, py: Python<'py>, li: (u8, u32, u32), kind: &str, src: (u8, u32, u32), cands: Vec<Props>) -> PyResult<Bound<'py, PyDict>> {
        use ygo_core::constants::*;
        // 0: t in the Tribute group of c, 1: t an attack target of c, 2: c in t's Tribute group
        let kind = match kind {
            "tribute" => 0,
            "attack" => 1,
            "tribute_of" => 2,
            _ => return Err(PyRuntimeError::new_err(format!("unknown check {kind}"))),
        };
        let res = as_query(&mut self.d, |d| {
            let c = ygo_core::api::duel::get_field_card(d, src.0, src.1, src.2);
            let t = if li.1 & (LOCATION_MZONE | LOCATION_SZONE) != 0 {
                ygo_core::api::duel::get_field_card(d, li.0, li.1, li.2)
            } else {
                let g = ygo_core::api::duel::get_field_group(d, li.0, li.1, 0);
                g.iter().find(|&x| d.card(x).current.sequence == li.2)
            };
            let mut out = Vec::new();
            let (Some(c), Some(t)) = (c, t) else { return out };
            for p in &cands {
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
        });
        let out = PyDict::new(py);
        for (code, ok) in res.unwrap_or_default() {
            out.set_item(code, ok)?;
        }
        Ok(out)
    }

    /// For each candidate (None: the card as it is): whether its player may Set the hand card at
    /// (con, seq) with the summon type `sumtype` if it had the candidate's properties, and with
    /// tributes, whether it can Tribute exactly the monsters at `trib_locs` (and takes Tributes at
    /// all). Empty if the card is missing or on errors.
    fn settable_as<'py>(&mut self, py: Python<'py>, li: (u8, u32), sumtype: u32, trib_locs: Vec<(u8, u32)>, cands: Vec<Option<Props>>) -> PyResult<Bound<'py, PyDict>> {
        use ygo_core::constants::*;
        let res = as_query(&mut self.d, |d| {
            let mut out = Vec::new();
            let Some(c) = ygo_core::api::duel::get_field_card(d, li.0, LOCATION_HAND, li.1) else { return out };
            let mut g = ygo_core::types::CardSet::new();
            for &(con, seq) in &trib_locs {
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
            for p in &cands {
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
        });
        let out = PyDict::new(py);
        for (code, ok) in res.unwrap_or_default() {
            out.set_item(code, ok)?;
        }
        Ok(out)
    }
}

#[pymodule]
fn ygoai_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_class::<RustDuel>()?;
    Ok(())
}
