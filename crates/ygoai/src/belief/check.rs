//! Consistency checks of the belief replay against recorded games (check.py's check_prefix).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::belief::completion::Completion;
use crate::belief::replay::{BeliefReplay, RErr};
use crate::game::{InitInfo, ObsItem};
use crate::messages::Msg;

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
