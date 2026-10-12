//! A minimal reconstruction: no belief about the game's history, only what the viewer saw at its
//! latest Idle or Battle command, then the observations since.
//!
//! A fresh duel with the viewer's deck list and, for the opponent, the cards visible at that command
//! plus the deck completion's guesses for the rest is played (passing) to the viewer's turn and phase,
//! then forced into the view there (sync.rs: places, positions, codes, LP, turn, status bits) and the
//! command prompt computed again. From there the replay follows the stream leniently (re-aligning
//! where it differs) to the viewer's current decision. What the view does not show is lost: effects
//! already used, continuous effects' history, equip and target links, counters, materials, and the
//! opponent's hidden cards are guesses from the visible ones only.

use ygo_core::constants::*;

use crate::actions::{Response, encode};
use crate::belief::agent::Reconstruction;
use crate::belief::completion::{Completion, VanillaCompletion, card, is_extra_deck};
use crate::belief::replay::BeliefReplay;
use crate::game::{Deck, InitInfo, ObsItem, run_until_prompt, setup_duel};
use crate::messages::Msg;
use crate::rduel::RDuel;
use crate::state::{LOCATIONS, Snapshot, viewer_view};

/// prompts answered (passing) on the way to the viewer's turn and phase
const MAX_STEPS: usize = 2000;

/// The duel at the end of `items` by a minimal reconstruction from the viewer's latest command
/// (None: there is none, or the duel could not be brought there).
pub fn minimal_reconstruct(init: &InitInfo, items: &[ObsItem], completion: Option<&'static dyn Completion>, seed: u64) -> Option<(Reconstruction, RDuel)> {
    let r = minimal_inner(init, items, completion, seed);
    if let Err(why) = &r
        && std::env::var_os("YGOAI_MINIMAL_DEBUG").is_some()
    {
        eprintln!("minimal reconstruction failed: {why}");
    }
    r.ok()
}

fn minimal_inner(init: &InitInfo, items: &[ObsItem], completion: Option<&'static dyn Completion>, seed: u64) -> Result<(Reconstruction, RDuel), String> {
    let viewer = init.player;
    // the latest command of the viewer outside a chain, with its view
    let k = items
        .iter()
        .rposition(|it| matches!(it, ObsItem::Prompt(Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. }, Some(s)) if s.chain.is_empty()))
        .ok_or("no command yet")?;
    let ObsItem::Prompt(cmd, Some(state)) = &items[k] else { unreachable!() };
    let mut duel = fresh_duel(init, state, completion, seed).ok_or("no fresh duel")?;
    to_command(&mut duel, viewer, cmd, state).ok_or_else(|| format!("cannot reach the {} of turn {} phase {:#x}", cmd.name(), state.turn, state.phase))?;
    let got = viewer_view(&mut duel.d, viewer);
    crate::belief::sync::sync_to_view(&mut duel, viewer, state, &got);
    if !duel.restart_command() {
        return Err("no command to restart".into());
    }
    if std::env::var_os("YGOAI_MINIMAL_DEBUG").is_some() {
        let after = viewer_view(&mut duel.d, viewer);
        let (a, b) = (serde_json::to_value(state).unwrap(), serde_json::to_value(&after).unwrap());
        let mut paths = Vec::new();
        crate::belief_agents::diff_paths(&a, &b, String::new(), &mut paths);
        eprintln!("after the sync at item {k} of {}: {} differences: {}", items.len(), paths.len(), paths.iter().take(12).cloned().collect::<Vec<_>>().join(" "));
    }
    // (the stream from that command on, the replay positioned there)
    let mut br = BeliefReplay::new(viewer, init.clone(), items.to_vec(), completion);
    br.check_state = false;
    br.read_guided = false;
    br.log = Some(Vec::new());
    let duel = br.segment_run(k, duel).ok_or_else(|| format!("the stream from item {k} of {}: {}", items.len(), br.log.as_ref().and_then(|l| l.last().cloned()).unwrap_or_default()))?;
    Ok((Reconstruction { approximate: true, final_prompt: br.final_prompt.clone(), lenient_skips: br.lenient_skips, restarts: 0 }, duel))
}

/// The viewer's deck and the opponent's: the cards visible in `state` and guesses for the rest.
fn fresh_duel(init: &InitInfo, state: &Snapshot, completion: Option<&'static dyn Completion>, seed: u64) -> Option<RDuel> {
    let viewer = init.player;
    let opp = 1 - viewer;
    let visible = |p: u32| -> Vec<u32> {
        LOCATIONS
            .iter()
            .flat_map(|&(_, loc)| state.players[p as usize].pile(loc).iter().flatten().filter_map(|c| c.code.filter(|&x| x != 0)))
            .filter(|&c| card(c).is_some_and(|d| d.type_ & TYPE_TOKEN == 0))
            .collect()
    };
    let seen = visible(opp);
    let present: Vec<u32> = seen.iter().copied().chain(visible(viewer)).collect();
    let is_extra = |c: u32| card(c).is_some_and(is_extra_deck);
    let mut main: Vec<u32> = seen.iter().copied().filter(|&c| !is_extra(c)).collect();
    let mut extra: Vec<u32> = seen.iter().copied().filter(|&c| is_extra(c)).collect();
    main.truncate(init.opponent_deck_size);
    extra.truncate(init.opponent_extra_size);
    let (n_main, n_extra) = (init.opponent_deck_size - main.len(), init.opponent_extra_size - extra.len());
    let comp: &dyn Completion = completion.unwrap_or(VanillaCompletion::get());
    let (fm, fe) = comp.complete(&main, &extra, n_main, n_extra, &present).or_else(|_| VanillaCompletion::get().complete(&main, &extra, n_main, n_extra, &present)).ok()?;
    main.extend(fm);
    extra.extend(fe);
    let mut decks = vec![Deck::default(), Deck::default()];
    decks[viewer as usize] = init.deck.clone();
    decks[opp as usize] = Deck { name: String::new(), main, extra };
    let orders: Vec<Vec<u32>> = decks.iter().map(|d| d.main.clone()).collect();
    let s = [seed, seed ^ 0x9E37_79B9_7F4A_7C15, 0x1234, 0x5678];
    let d = setup_duel(&decks, s, &init.config, &orders, None, None, None).ok()?;
    let mut duel = RDuel::from_duel(d);
    duel.start();
    Some(duel)
}

/// Plays the fresh duel, passing, until the viewer has `cmd`'s kind of command in `state`'s turn
/// and phase (the turn's number is set by the sync).
fn to_command(duel: &mut RDuel, viewer: u32, cmd: &Msg, state: &Snapshot) -> Option<()> {
    let to_battle = matches!(cmd, Msg::SelectBattlecmd { .. }) || state.phase == PHASE_MAIN2;
    let mut prompt: Option<Msg> = None;
    for _ in 0..MAX_STEPS {
        let step = run_until_prompt(&mut duel.d, prompt.as_ref()).ok()?;
        if step.ended {
            return None;
        }
        let p = step.prompt?;
        let (_, turn_player, phase, _) = duel.info();
        if turn_player == viewer && p.player() == Some(viewer) && p.name() == cmd.name() && phase == state.phase {
            return Some(());
        }
        let resp = pass(&p, turn_player == viewer, to_battle, state.phase);
        duel.set_response(&encode(&p, &resp).ok()?);
        prompt = Some(p);
    }
    None
}

/// The answer that moves on: no activations, on to the next phase (to the Battle Phase in the
/// viewer's turn when the target is there or later), the first choice otherwise.
fn pass(p: &Msg, viewers_turn: bool, to_battle: bool, target: u32) -> Response {
    let kind = |k: &str| Response { kind: Some(k.into()), ..Response::default() };
    match p {
        Msg::SelectIdlecmd { to_bp, .. } => {
            if viewers_turn && to_battle && *to_bp != 0 { kind("to_bp") } else { kind("to_ep") }
        }
        Msg::SelectBattlecmd { to_m2, .. } => {
            if viewers_turn && target == PHASE_MAIN2 && *to_m2 != 0 { kind("to_m2") } else { kind("to_ep") }
        }
        Msg::SelectChain { forced, .. } if *forced == 0 => Response { index: Some(-1), ..Response::default() },
        _ => crate::belief::replay::decide::all_responses(p).into_iter().find(|r| encode(p, r).is_ok()).unwrap_or_else(|| first_valid(p)),
    }
}

fn first_valid(p: &Msg) -> Response {
    let mut rng = crate::pyrandom::PyRandom::new(1);
    crate::agents::random_response(p, &mut rng, 0.0).unwrap_or_default()
}
