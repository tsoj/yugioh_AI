//! An agent's belief state, reconstructed decision by decision (agent.py).
//!
//! At each decision the agent needs the duel as it believes it to be, from all its observations so
//! far (BeliefReplay). The stream of the previous decision is the start of the new one, so:
//!
//! 1. the previous consistent reconstruction's duel, kept waiting at that decision, goes on with
//!    the agent's answer and the observations since (BeliefReplay::extend), unless they reveal
//!    opponent cards not seen before (or more copies of one): the deck completion takes what was
//!    seen into account, so the guesses are made anew then;
//! 2. if that runs into a mismatch, a new reconstruction inherits what the previous one found
//!    (BeliefReplay::inherit) and mostly works on the new part;
//! 3. if that fails or ends only approximate, one from scratch.
//!
//! With no consistent reconstruction, the best effort (approximate, re-synchronized with the
//! agent's observations) is what the agent gets.

use std::collections::HashMap;

use crate::belief::completion::Completion;
use crate::belief::replay::BeliefReplay;
use crate::game::{InitInfo, ObsItem};
use crate::messages::Msg;
use crate::rduel::RDuel;

pub struct AgentBelief {
    pub viewer: u32,
    pub init: InitInfo,
    pub completion: Option<&'static dyn Completion>,
    pub incremental: bool,
    /// the latest consistent reconstruction
    prev: Option<BeliefReplay>,
    /// its duel, at the decision its stream ends with
    live: Option<RDuel>,
    /// decisions reached incrementally
    pub extended: usize,
    /// decisions with new reveals: reconstructed in full
    pub revealed: usize,
    /// opponent card codes the stream showed (see reveals)
    seen: HashMap<u32, usize>,
    seen_upto: usize,
    /// fresh attempts after an inherited one failed or was approximate
    pub retries: usize,
    /// when the current reconstruction gives up
    deadline: Option<std::time::Instant>,
}

/// What a reconstruction was (the parts of its BeliefReplay a caller reads).
#[derive(Debug, Clone, Default)]
pub struct Reconstruction {
    pub approximate: bool,
    pub final_prompt: Option<Msg>,
    pub lenient_skips: usize,
    pub restarts: usize,
}

fn summary(br: &BeliefReplay) -> Reconstruction {
    Reconstruction { approximate: br.approximate, final_prompt: br.final_prompt.clone(), lenient_skips: br.lenient_skips, restarts: br.stats.restarts }
}

impl AgentBelief {
    pub fn new(viewer: u32, init: InitInfo, completion: Option<&'static dyn Completion>) -> AgentBelief {
        AgentBelief {
            viewer,
            init,
            completion,
            incremental: std::env::var_os("YGOAI_NO_INCREMENTAL").is_none(),
            prev: None,
            live: None,
            extended: 0,
            revealed: 0,
            seen: HashMap::new(),
            seen_upto: 0,
            retries: 0,
            deadline: None,
        }
    }

    /// The duel at the end of `items` (a copy: the caller may change it), or None if there is
    /// none, not even an approximate one (or none within `timeout`).
    pub fn reconstruct(&mut self, items: &[ObsItem], timeout: Option<std::time::Duration>) -> (Reconstruction, Option<RDuel>) {
        self.deadline = timeout.map(|t| std::time::Instant::now() + t);
        let mut live = self.live.take();
        if self.reveals(items) && live.is_some() {
            self.revealed += 1;
            live = None;
        }
        if let (Some(mut duel), Some(prev)) = (live, self.prev.as_mut()) {
            prev.deadline = self.deadline;
            if prev.extend(&mut duel, items.to_vec()).is_ok() {
                self.extended += 1;
                let out = duel.clone();
                self.live = Some(duel);
                return (summary(prev), Some(out));
            }
        }
        let prev = self.prev.take();
        let (mut br, mut duel, inherited) = self.attempt(items, prev.as_ref());
        if inherited && (duel.is_none() || br.approximate) {
            self.retries += 1;
            (br, duel, _) = self.attempt(items, None);
        }
        let consistent = duel.is_some() && !br.approximate;
        let out = duel.clone();
        self.live = if consistent && self.incremental { duel } else { None };
        let s = summary(&br);
        self.prev = if consistent { Some(br) } else { None };
        (s, out)
    }

    /// Whether the stream since the last call shows opponent cards it did not show before: per
    /// code, as one card when a message shows it and as the most copies a state view showed at
    /// once.
    fn reveals(&mut self, items: &[ObsItem]) -> bool {
        let opp = 1 - self.viewer;
        let mut new: HashMap<u32, usize> = HashMap::new();
        let one = |new: &mut HashMap<u32, usize>, code: u32| {
            let e = new.entry(code).or_default();
            *e = (*e).max(1);
        };
        for it in &items[self.seen_upto.min(items.len())..] {
            match it {
                ObsItem::Prompt(_, Some(state)) => {
                    let mut here: HashMap<u32, usize> = HashMap::new();
                    for p in &state.players {
                        for (_, loc) in crate::state::LOCATIONS {
                            for c in p.pile(loc).iter().flatten() {
                                if let Some(code) = c.code.filter(|&x| x != 0)
                                    && c.owner == Some(opp)
                                {
                                    *here.entry(code).or_default() += 1;
                                }
                            }
                        }
                    }
                    for (code, n) in here {
                        let e = new.entry(code).or_default();
                        *e = (*e).max(n);
                    }
                }
                ObsItem::Msg(m) => match m {
                    Msg::Move { code, from, to, .. } if *code != 0 => {
                        let con = if to.loc != 0 { to.con } else { from.con };
                        if con == opp {
                            one(&mut new, *code);
                        }
                    }
                    Msg::Chaining { code, con, .. }
                    | Msg::PosChange { code, con, .. }
                    | Msg::Set { code, con, .. }
                    | Msg::Summoning { code, con, .. }
                    | Msg::Spsummoning { code, con, .. }
                        if *code != 0 && *con == opp =>
                    {
                        one(&mut new, *code)
                    }
                    Msg::ConfirmCards { cards, .. } => {
                        for c in cards {
                            if c.code != 0 && c.con == opp {
                                one(&mut new, c.code);
                            }
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        self.seen_upto = items.len();
        let more = new.iter().any(|(code, &n)| n > self.seen.get(code).copied().unwrap_or(0));
        for (code, n) in new {
            let e = self.seen.entry(code).or_default();
            *e = (*e).max(n);
        }
        more
    }

    fn attempt(&self, items: &[ObsItem], prev: Option<&BeliefReplay>) -> (BeliefReplay, Option<RDuel>, bool) {
        let mut br = BeliefReplay::new(self.viewer, self.init.clone(), items.to_vec(), self.completion);
        br.check_state = false;
        br.allow_approximate = true; // (an agent needs some state: the closest one, from observations only)
        br.read_guided = false; // (too slow per decision: the approximate pass comes right after the repairs)
        br.deadline = self.deadline;
        let inherited = prev.is_some_and(|p| br.inherit(p));
        let duel = br.run().ok();
        (br, duel, inherited)
    }
}
