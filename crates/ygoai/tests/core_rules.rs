//! Rules of the patched core that the engine keeps (formerly tests/test_core_rules.py against the C++
//! core): an effect whose condition holds again starts applying anew (a new timestamp), checked at
//! fixed points of the game, not by lazy evaluations.

use ygo_core::api::NewCardInfo;
use ygo_core::api::{card as cd, duel as du, effect as ef};
use ygo_core::constants::*;
use ygo_core::duel::Duel;
use ygo_core::types::{CardId, LuaFn, Value};
use ygoai::actions::{Command, Response, encode};
use ygoai::game::{DuelConfig, new_duel, run_until_prompt};
use ygoai::messages::Msg;

const MYSTICAL_ELF: u32 = 15025844; // (a Level 4 Normal Monster)
const COND: (u32, u64) = (0, 1); // (the test's condition flag, a script variable)

fn card(d: &mut Duel, team: u8, loc: u32, pos: u32) {
    d.ocg_new_card(&NewCardInfo { team, duelist: 0, code: MYSTICAL_ELF, con: team, loc, seq: 0, pos });
}

/// A duel with 40 Mystical Elves in each deck and one face-up in player 0's Monster Zone, at its
/// first prompt.
fn duel_with_monster() -> (Duel, Msg) {
    let mut d = new_duel([1, 2, 3, 4], &DuelConfig::default());
    for p in 0..2 {
        for _ in 0..40 {
            card(&mut d, p, LOCATION_DECK, POS_FACEDOWN_DEFENSE);
        }
    }
    card(&mut d, 0, LOCATION_MZONE, POS_FACEUP_ATTACK);
    d.ocg_start_duel();
    let prompt = run_until_prompt(&mut d, None).unwrap().prompt.unwrap();
    (d, prompt)
}

/// Answers `prompt` (None: nothing to answer) and returns the next prompt that is not a chain
/// window (chains are always declined).
fn answer(d: &mut Duel, mut prompt: Msg, mut resp: Option<Response>) -> Msg {
    loop {
        if let Some(r) = resp.take() {
            d.ocg_set_response(&encode(&prompt, &r).unwrap());
            let step = run_until_prompt(d, Some(&prompt)).unwrap();
            assert!(!step.ended);
            prompt = step.prompt.unwrap();
        }
        if !matches!(prompt, Msg::SelectChain { .. }) {
            return prompt;
        }
        resp = Some(Response::pass());
    }
}

fn monster(d: &Duel) -> CardId {
    du::get_field_card(d, 0, LOCATION_MZONE, 0).unwrap()
}

/// A single effect of the monster: `code` with `value`, under `cond` if any.
fn effect(d: &mut Duel, code: u32, value: i64, cond: Option<LuaFn>) {
    let c = monster(d);
    let e = ef::create_effect(d, c);
    ef::set_type(d, e, EFFECT_TYPE_SINGLE);
    ef::set_code(d, e, code);
    ef::set_value(d, e, value);
    if cond.is_some() {
        ef::set_condition(d, e, cond);
    }
    cd::register_effect(d, c, e, false);
}

fn level(d: &mut Duel) -> u32 {
    let c = monster(d);
    cd::get_level(d, c)
}

fn set_cond(d: &mut Duel, on: bool) {
    d.set_script_var(COND, Value::Bool(on));
}

#[test]
fn condition_renews_timestamp() {
    let (mut d, _) = duel_with_monster();
    set_cond(&mut d, true);
    let cond = LuaFn::new(|d: &mut Duel, _: &[Value]| d.script_var(COND));
    effect(&mut d, EFFECT_UPDATE_LEVEL, 2, Some(cond));
    du::adjust_instantly(&mut d, None);
    effect(&mut d, EFFECT_CHANGE_LEVEL, 2, None);
    du::adjust_instantly(&mut d, None);
    let mut out = vec![level(&mut d)]; // "+2", then "becomes 2"
    set_cond(&mut d, false);
    level(&mut d); // (a lazy evaluation while the condition fails)
    set_cond(&mut d, true);
    du::adjust_instantly(&mut d, None);
    out.push(level(&mut d)); // unchanged: no fixed point saw the condition fail
    set_cond(&mut d, false);
    du::adjust_instantly(&mut d, None);
    out.push(level(&mut d)); // only "becomes 2"
    set_cond(&mut d, true);
    out.push(level(&mut d)); // (lazy: the "+2" applies with its old timestamp)
    du::adjust_instantly(&mut d, None);
    out.push(level(&mut d)); // "becomes 2", then the renewed "+2"
    assert_eq!(out, vec![2, 2, 2, 2, 4]);
}

/// A monster with "+2 Level during the Main Phase" whose Level becomes 2 in the Battle Phase: in
/// Main Phase 2 the "+2" applies anew, after it.
#[test]
fn phase_condition() {
    let (mut d, prompt) = duel_with_monster();
    let mut prompt = answer(&mut d, prompt, None);
    assert!(matches!(prompt, Msg::SelectIdlecmd { .. }));
    let main_phase = LuaFn::new(|d: &mut Duel, _: &[Value]| Value::Bool(du::is_main_phase(d)));
    effect(&mut d, EFFECT_UPDATE_LEVEL, 2, Some(main_phase));
    assert_eq!(level(&mut d), 6);
    // turn 1: to the End Phase; turn 2 (player 1): to the Battle Phase
    let command = |c: Command| Some(Response::command(c, None));
    prompt = answer(&mut d, prompt, command(Command::ToEp));
    assert_eq!(d.infos.turn_id, 2);
    prompt = answer(&mut d, prompt, command(Command::ToBp));
    assert!(matches!(prompt, Msg::SelectBattlecmd { .. }));
    assert_eq!(d.infos.phase as u32, PHASE_BATTLE_STEP);
    effect(&mut d, EFFECT_CHANGE_LEVEL, 2, None);
    assert_eq!(level(&mut d), 2);
    prompt = answer(&mut d, prompt, command(Command::ToM2));
    assert!(matches!(prompt, Msg::SelectIdlecmd { .. }));
    assert_eq!(d.infos.phase as u32, PHASE_MAIN2);
    assert_eq!(level(&mut d), 4);
}
