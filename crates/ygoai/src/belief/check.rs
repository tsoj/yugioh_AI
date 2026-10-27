//! Consistency checks of the belief replay against its test games (scripts/check_belief.py's).

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};
use ygo_core::duel::Duel;

use crate::belief::agent::AgentBelief;
use crate::belief::completion::Completion;
use crate::belief::games::play_with_streams;
use crate::belief::replay::{BeliefReplay, RErr};
use crate::game::{Deck, GameError, GameRecord, InitInfo, ObsItem, init_info};
use crate::messages::Msg;
use crate::state::{Snapshot, viewer_view};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CheckResult {
    pub viewer: u32,
    pub cut: usize,
    pub length: usize,
    pub ok: bool,
    pub stage: String,
    pub error: String,
    pub seconds: f64,
    pub opponent_decisions: usize,
    pub probed: usize,
    pub probes: usize,
    pub unknown_atoms: usize,
    pub restarts: usize,
    pub fixes: BTreeMap<String, usize>,
    /// "reason -> fix" of every restart
    pub repairs: Vec<String>,
    /// the unfinished action the game ended in was not checked
    pub end_ignored: bool,
    /// the reconstruction ran out of time
    pub timeout: bool,
    /// the divergence's two sides, when it ended in one: {"expected", "got", "first_difference"}
    pub details: serde_json::Map<String, serde_json::Value>,
}

fn err_text(e: &RErr) -> String {
    match e {
        RErr::Div(d) => d.text(),
        RErr::Tracker(t) => format!("TrackerError: {}", t.0),
        e => e.to_string(),
    }
}

/// Reconstructs the duel from `items[..cut]` (the viewer's stream) and reports whether it is
/// consistent. `stopped`: the game was stopped (a limit) rather than decided.
pub fn check_prefix(
    init: &InitInfo,
    items: &[ObsItem],
    cut: usize,
    stopped: bool,
    completion: Option<&'static dyn Completion>,
    timeout: Option<Duration>,
) -> CheckResult {
    let viewer = init.player;
    let prefix = &items[..cut];
    let mut res = CheckResult { viewer, cut, length: items.len(), ..CheckResult::default() };
    let t0 = Instant::now();
    let deadline = timeout.map(|t| t0 + t);
    let make = |items: Vec<ObsItem>, stopped: bool| -> BeliefReplay {
        let mut br = BeliefReplay::new(viewer, init.clone(), items, completion);
        br.stopped = stopped;
        br.log = Some(Vec::new());
        br.deadline = deadline;
        br
    };
    let stopped = cut == items.len() && stopped;
    let mut br = make(prefix.to_vec(), stopped);
    let last_prompt = prefix.iter().rposition(|it| matches!(it, ObsItem::Prompt(..)));
    let ends_in_win = cut == items.len() && matches!(items.last(), Some(ObsItem::Msg(Msg::Win { .. })));
    if ends_in_win && let Some(lp) = last_prompt {
        br.tail_start = Some(lp + 2); // (see the retry below)
    }
    res.stage = "hypothesis".into();
    let outcome = match br.build_hypothesis() {
        Ok(h) => {
            res.unknown_atoms = h.unknown_atoms;
            res.stage = "replay".into();
            br.run().map(|_| ())
        }
        Err(e) => Err(e),
    };
    match outcome {
        Ok(()) => {
            res.ok = true;
            res.stage = "done".into();
            if let Some(f) = &br.read_guided_fix
                && std::env::var_os("YGOAI_READ_GUIDED_OK").is_none()
            {
                res.ok = false;
                res.error = format!("read-guided repair needed: {}", f.join("; "));
            }
        }
        Err(e) => {
            res.error = err_text(&e);
            if let RErr::Div(d) = &e {
                let (exp, got) = (d.expected.as_ref().and_then(|x| x.dict()), d.got.as_ref().and_then(|x| x.dict()));
                if let (Some(x), Some(y)) = (&exp, &got)
                    && let Some(diff) = crate::belief::replay::util::first_difference(x, y, "")
                {
                    res.details.insert("first_difference".into(), diff.into());
                }
                res.details.insert("expected".into(), exp.unwrap_or_default());
                res.details.insert("got".into(), got.unwrap_or_default());
            }
            res.timeout = matches!(&e, RErr::Other(s) if s == "timeout");
            if let RErr::Div(d) = &e
                && ends_in_win
                && let Some(lp) = last_prompt
                && d.pos > lp + 1
            {
                // the game ended in the middle of an action: check everything up to the viewer's
                // last decision instead
                let mut retry = make(items[..lp + 1].to_vec(), true);
                let ok = retry.build_hypothesis().is_ok() && retry.run().is_ok();
                let mut logs = br.log.take().unwrap_or_default();
                logs.extend(retry.log.take().unwrap_or_default());
                retry.log = Some(logs);
                if ok {
                    res.ok = true;
                    res.stage = "done".into();
                    res.end_ignored = true;
                    res.error.clear();
                    res.details.clear();
                    br = retry;
                } else {
                    br.log = retry.log.take();
                }
            }
        }
    }
    res.seconds = t0.elapsed().as_secs_f64();
    res.opponent_decisions = br.stats.opponent_decisions;
    res.probed = br.stats.probed_decisions;
    res.probes = br.stats.probes;
    res.restarts = br.stats.restarts;
    for (k, v) in [("forces", br.stats.forces), ("assumptions", br.stats.assumptions), ("swaps", br.stats.swaps), ("read_guided", br.stats.read_guided)] {
        if v != 0 {
            res.fixes.insert(k.into(), v);
        }
    }
    res.repairs = br.log.take().unwrap_or_default().into_iter().filter(|t| t.starts_with("restart")).collect();
    res
}

/// Facts that differ between two views (as JSON values; a missing key counts as null).
pub fn count_diffs(a: &Value, b: &Value) -> usize {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let keys: BTreeSet<&String> = x.keys().chain(y.keys()).collect();
            keys.into_iter().map(|k| count_diffs(x.get(k).unwrap_or(&Value::Null), y.get(k).unwrap_or(&Value::Null))).sum()
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => x.iter().zip(y).map(|(p, q)| count_diffs(p, q)).sum(),
        _ => (a != b) as usize,
    }
}

/// The public facts of the viewer's state `observed` that its reconstruction `duel` gets wrong.
pub fn state_diffs(observed: &Snapshot, duel: &mut Duel, viewer: u32) -> usize {
    let json = |s: &Snapshot| serde_json::to_value(s).expect("a snapshot serializes");
    count_diffs(&json(observed), &json(&viewer_view(duel, viewer)))
}

/// The "frac" of an agent sequence's result: the whole game, decision by decision.
pub const SEQUENCE_FRAC: f64 = 2.0;

/// How a game's cases are checked.
#[derive(Clone, Copy)]
pub enum Mode {
    /// a fresh reconstruction of each prefix (`check_prefix`)
    Prefix,
    /// as an agent reconstructs at its decision, the best effort allowed (`agent_case`)
    Agent,
    /// the viewer's belief at every n-th of its decisions, inherited as an agent's (`agent_sequence`)
    Sequence(usize),
}

/// The prefix length of a cut: with `frac` >= 0 right after the viewer's prompt nearest that share
/// of the stream (1: all of it); with -N right after the viewer's last prompt before turn N + 1
/// starts (None: the game has no turn N + 1, or the viewer no prompt before it).
pub fn cut_point(items: &[ObsItem], frac: f64) -> Option<usize> {
    let is_prompt = |it: &ObsItem| matches!(it, ObsItem::Prompt(..));
    if frac >= 0.0 {
        let target = frac * items.len() as f64;
        let distance = |i: usize| (i as f64 - target).abs();
        let nearest = (0..items.len()).filter(|&i| is_prompt(&items[i])).fold(None, |best: Option<usize>, i| match best {
            Some(b) if distance(b) <= distance(i) => Some(b),
            _ => Some(i),
        });
        return Some(match nearest {
            Some(i) if frac < 1.0 => i + 1,
            _ => items.len(),
        });
    }
    let turn = (-frac) as usize + 1;
    let new_turns = items.iter().enumerate().filter(|(_, it)| matches!(it, ObsItem::Msg(Msg::NewTurn { .. })));
    let (start, _) = new_turns.clone().nth(turn - 1)?;
    items[..start].iter().rposition(is_prompt).map(|j| j + 1)
}

/// A viewer's game for comparing runs (only results of the same games are compared): a CRC-32 of
/// its stream as JSON with sorted keys.
pub fn game_hash(items: &[ObsItem]) -> u32 {
    fn canonical(v: &Value, out: &mut String) {
        match v {
            Value::Object(m) => {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                out.push('{');
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::from(k.as_str()).to_string());
                    out.push(':');
                    canonical(&m[k], out);
                }
                out.push('}');
            }
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    canonical(x, out);
                }
                out.push(']');
            }
            v => out.push_str(&v.to_string()),
        }
    }
    let mut text = String::new();
    canonical(&serde_json::to_value(items).expect("a stream serializes"), &mut text);
    crc32fast::hash(text.as_bytes())
}

/// The fields every case's record starts with.
fn case_record(seed: i64, viewer: u32, frac: f64, cut: usize, len: usize) -> Map<String, Value> {
    let Value::Object(m) = json!({"seed": seed, "viewer": viewer, "frac": frac, "cut": cut, "len": len}) else { unreachable!() };
    m
}

fn timed_out(mut record: Map<String, Value>, seconds: f64) -> Map<String, Value> {
    record.extend([("ok".into(), false.into()), ("error".into(), "timeout".into()), ("seconds".into(), seconds.into())]);
    record
}

/// The check's outcome in a case's record (agent mode adds whether the reconstruction was the best
/// effort, and how far its state is from the real one).
fn add_result(record: &mut Map<String, Value>, r: &CheckResult, agent: Option<(bool, usize)>) {
    record.extend([
        ("ok".into(), r.ok.into()),
        ("error".into(), r.error.clone().into()),
        ("seconds".into(), r.seconds.into()),
        ("decisions".into(), r.opponent_decisions.into()),
        ("probed".into(), r.probed.into()),
        ("probes".into(), r.probes.into()),
        ("restarts".into(), r.restarts.into()),
        ("fixes".into(), json!(r.fixes)),
        ("repairs".into(), json!(r.repairs)),
        ("end_ignored".into(), r.end_ignored.into()),
    ]);
    if let Some((approximate, state_diffs)) = agent {
        record.extend([("approximate".into(), approximate.into()), ("state_diffs".into(), state_diffs.into())]);
    }
}

/// As an agent reconstructs: at the viewer's decision (the cut moved back to its latest prompt), the
/// best effort allowed when no consistent reconstruction is found (without the slow read-guided
/// repair); (the result, whether it was approximate, its state's differences from the real one).
fn agent_case(init: &InitInfo, items: &[ObsItem], cut: usize, completion: Option<&'static dyn Completion>, timeout: Duration) -> (CheckResult, bool, usize) {
    let viewer = init.player;
    let cut = items[..cut].iter().rposition(|it| matches!(it, ObsItem::Prompt(..))).map_or(0, |i| i + 1);
    let mut res = CheckResult { viewer, cut, length: items.len(), ..CheckResult::default() };
    if cut == 0 {
        res.error = "no decision of the viewer".into();
        return (res, false, 0);
    }
    let t0 = Instant::now();
    let (r, duel) = AgentBelief::new(viewer, init.clone(), completion).reconstruct(&items[..cut], Some(timeout));
    let mut diffs = 0;
    match (duel, &items[cut - 1]) {
        (Some(mut duel), ObsItem::Prompt(_, Some(observed))) => {
            diffs = state_diffs(observed, &mut duel.d, viewer);
            (res.ok, res.stage) = (!r.approximate, "done".into());
            if r.approximate {
                res.error = format!("approximate ({} re-aligned, {diffs} state differences)", r.lenient_skips);
            }
        }
        (Some(_), _) => res.error = "no state at the viewer's decision".into(),
        (None, _) => res.error = "no reconstruction".into(),
    }
    res.restarts = r.restarts;
    res.seconds = t0.elapsed().as_secs_f64();
    (res, r.approximate, diffs)
}

/// The viewer's belief at every `every`-th of its decisions, as an agent reconstructs it
/// (AgentBelief, each reconstruction inheriting the last); `timeout` per reconstruction (then a
/// fresh belief goes on).
fn agent_sequence(seed: i64, init: &InitInfo, items: &[ObsItem], every: usize, completion: Option<&'static dyn Completion>, timeout: Duration) -> Map<String, Value> {
    let viewer = init.player;
    let new_belief = || AgentBelief::new(viewer, init.clone(), completion);
    let mut belief = new_belief();
    let cuts: Vec<usize> = (0..items.len()).filter(|&i| matches!(items[i], ObsItem::Prompt(..))).map(|i| i + 1).step_by(every).collect();
    let (mut consistent, mut approximate, mut failed, mut diffs_total, mut max_diffs) = (0, 0, 0, 0, 0);
    let mut seconds = 0.0;
    let mut first_problem: Option<String> = None;
    for &cut in &cuts {
        let t0 = Instant::now();
        let (r, duel) = belief.reconstruct(&items[..cut], Some(timeout));
        let problem = if t0.elapsed() > timeout {
            failed += 1;
            let retries = belief.retries;
            belief = new_belief();
            belief.retries = retries;
            Some("timeout".to_string())
        } else {
            match (duel, &items[cut - 1]) {
                (None, _) => {
                    failed += 1;
                    Some("no reconstruction".into())
                }
                (Some(mut duel), ObsItem::Prompt(_, Some(observed))) if r.approximate => {
                    let d = state_diffs(observed, &mut duel.d, viewer);
                    (approximate, diffs_total, max_diffs) = (approximate + 1, diffs_total + d, max_diffs.max(d));
                    Some(format!("approximate ({d} state differences)"))
                }
                _ => {
                    consistent += 1;
                    None
                }
            }
        };
        seconds += t0.elapsed().as_secs_f64();
        if let Some(p) = problem
            && first_problem.is_none()
        {
            first_problem = Some(format!("at item {cut}: {p}"));
        }
    }
    let n = cuts.len();
    let mut record = case_record(seed, viewer, SEQUENCE_FRAC, items.len(), items.len());
    record.extend([
        ("reconstructions".into(), n.into()),
        ("consistent".into(), consistent.into()),
        ("approximate_n".into(), approximate.into()),
        ("failed".into(), failed.into()),
        ("diffs_total".into(), diffs_total.into()),
        ("approximate".into(), (approximate > 0).into()),
        ("state_diffs".into(), max_diffs.into()),
        ("retries".into(), belief.retries.into()),
        ("seconds".into(), seconds.into()),
        ("first_problem".into(), first_problem.clone().into()),
        ("extended".into(), belief.extended.into()),
        ("ok".into(), (consistent == n).into()),
        (
            "error".into(),
            if consistent == n { String::new() } else { format!("{} of {n} not consistent, first {}", n - consistent, first_problem.unwrap_or_default()) }.into(),
        ),
    ]);
    record
}

/// Whether a game was stopped (a limit) rather than decided.
pub fn stopped(record: &GameRecord) -> bool {
    record.result.get("reason").and_then(Value::as_str).is_some_and(|r| ["turn_limit", "decision_limit", "time_limit"].contains(&r))
}

/// The checks of test game `seed` (scripts/check_belief.py's records): the game played between the
/// agents named (`games::play_with_streams`), then each case (viewer, frac: a cut, `cut_point`)
/// checked, a prefix the same as another case's once; with `Mode::Sequence` one record per viewer.
/// `timeout`: per reconstruction.
pub fn check_game(
    seed: i64,
    decks: [Deck; 2],
    agents: [&str; 2],
    cases: &[(u32, f64)],
    mode: Mode,
    completion: Option<&'static dyn Completion>,
    timeout: Duration,
) -> Result<Vec<Value>, GameError> {
    let (record, streams) = play_with_streams(seed, decks, agents)?;
    let duel = record.setup(None)?;
    let inits = [0, 1].map(|p| init_info(&record, &duel, p));
    let stopped = stopped(&record);
    let games = streams.each_ref().map(|s| game_hash(s));
    let mut out = Vec::new();
    if let Mode::Sequence(every) = mode {
        for viewer in cases.iter().map(|&(v, _)| v).collect::<BTreeSet<u32>>() {
            let mut r = agent_sequence(seed, &inits[viewer as usize], &streams[viewer as usize], every, completion, timeout);
            r.insert("game".into(), games[viewer as usize].into());
            out.push(Value::Object(r));
        }
        return Ok(out);
    }
    let mut done = BTreeSet::new();
    for &(viewer, frac) in cases {
        let (init, items) = (&inits[viewer as usize], &streams[viewer as usize]);
        let Some(cut) = cut_point(items, frac) else { continue };
        if !done.insert((viewer, cut)) {
            continue;
        }
        let mut r = case_record(seed, viewer, frac, cut, items.len());
        let t0 = Instant::now();
        let (res, agent) = match mode {
            Mode::Agent => {
                let (res, approximate, diffs) = agent_case(init, items, cut, completion, timeout);
                (res, Some((approximate, diffs)))
            }
            _ => (check_prefix(init, items, cut, stopped, completion, Some(timeout)), None),
        };
        if res.timeout || t0.elapsed() > timeout {
            r = timed_out(r, t0.elapsed().as_secs_f64());
        } else {
            add_result(&mut r, &res, agent);
        }
        r.insert("game".into(), games[viewer as usize].into());
        out.push(Value::Object(r));
    }
    Ok(out)
}
