//! Replays a recorded game and runs a belief agent's search at one player's last decisions, with
//! several seeds, to find a search that panics (the arbiter only says that an engine exited):
//!
//!   repro_search RECORDS.jsonl INDEX PLAYER LAST SEEDS [NAME]
//!
//! INDEX: the record's line; LAST: how many of PLAYER's last decisions to search (always also the
//! unanswered one after the record's end, if PLAYER's); SEEDS: the
//! agent's seeds 0..SEEDS. The search's options come from the environment as in ygoai-engine.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use ygoai::belief_agents::BeliefSearchAgent;
use ygoai::game::{Agent, AgentError, GameRecord, InitInfo, Observer, PlayOptions, play};
use ygoai::actions::Response;
use ygoai::messages::Msg;
use ygoai::state::Snapshot;

/// The recorded responses and the next one's index (hidden prompts' answers included).
type Cursor = Rc<Cell<usize>>;

fn next(all: &[Response], at: &Cursor) -> Result<Response, AgentError> {
    all.get(at.get()).cloned().ok_or_else(|| AgentError::Failed("no more responses".into()))
}

struct Count(Cursor);

impl Observer for Count {
    fn response(&mut self, _: u32, _: &Response) {
        self.0.set(self.0.get() + 1);
    }
}

struct Replay(Vec<Response>, Cursor);

impl Agent for Replay {
    fn name(&self) -> String {
        "replay".into()
    }
    fn respond(&mut self, _: &Msg, _: Option<&Snapshot>, _: Option<(f64, f64)>) -> Result<Response, AgentError> {
        next(&self.0, &self.1)
    }
}

/// The agent under test: its recorded responses, searched at the decisions from `from` on.
struct Probe {
    agent: BeliefSearchAgent,
    responses: Vec<Response>,
    at: Cursor,
    from: usize,
    seed: i64,
}

impl Agent for Probe {
    fn name(&self) -> String {
        "probe".into()
    }
    fn wants_state(&self) -> bool {
        true
    }
    fn start(&mut self, player: u32, init: &InitInfo) -> Result<(), AgentError> {
        self.agent.start(player, init)
    }
    fn observe(&mut self, msg: &Msg) {
        self.agent.observe(msg)
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, _: Option<(f64, f64)>) -> Result<Response, AgentError> {
        let i = self.at.get();
        // (past the record: the decision the engine did not answer, searched too)
        let recorded = self.responses.get(i).cloned();
        // (no search before `from`: a clock without time)
        let clock = if i >= self.from { None } else { Some((0.0, 0.0)) };
        let agent = &mut self.agent;
        match catch_unwind(AssertUnwindSafe(|| agent.respond(prompt, state, clock))) {
            Ok(r) => {
                if i >= self.from {
                    eprintln!("decision {i}: {} searches {} failures {} nodes {} -> {:?}", prompt.name(), self.agent.searches, self.agent.failures, self.agent.nodes, r);
                }
                r?;
            }
            Err(_) => {
                println!("PANIC seed {} decision {i} of {}: {prompt:?}", self.seed, self.responses.len());
                return Err(AgentError::Failed("panic".into()));
            }
        }
        let recorded = recorded.ok_or_else(|| AgentError::Failed("no more responses".into()))?;
        self.agent.set_last_response(recorded.clone());
        Ok(recorded)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let line = text.lines().nth(a[2].parse().unwrap()).unwrap();
    let player: u32 = a[3].parse().unwrap();
    let (last, seeds): (usize, i64) = (a[4].parse().unwrap(), a[5].parse().unwrap());
    let name = a.get(6).map_or("alphabeta", |s| s.as_str());
    ygoai::game::init(None).unwrap();
    let base: GameRecord = serde_json::from_str(line).unwrap();
    let all: Vec<Response> = base.responses.iter().map(|r| r.response.clone()).collect();
    // (the search starts at PLAYER's LAST-th last decision)
    let mine: Vec<usize> = (0..all.len()).filter(|&i| base.responses[i].player == player).collect();
    let from = if last == 0 { all.len() } else { mine[mine.len().saturating_sub(last)] };
    println!("{} decisions, {} of player {player}, searching from decision {from}; result {:?}", all.len(), mine.len(), base.result);
    let mut panics = 0;
    for seed in 0..seeds {
        let mut record = base.clone();
        record.responses.clear();
        let agent = BeliefSearchAgent::new(name, Some(seed)).unwrap();
        let at: Cursor = Rc::new(Cell::new(0));
        let mut probe = Probe { agent, responses: all.clone(), at: at.clone(), from, seed };
        let mut other = Replay(all.clone(), at.clone());
        let mut count = Count(at);
        let mut opts = PlayOptions::default();
        let agents: [&mut dyn Agent; 2] = if player == 0 { [&mut probe, &mut other] } else { [&mut other, &mut probe] };
        let mut agents = agents;
        let _ = play(&mut record, &mut agents, &mut [&mut count], &mut opts);
        if record.result.get("reason").and_then(|r| r.as_str()).is_some_and(|r| r.contains("panic")) {
            panics += 1;
        }
        eprintln!("seed {seed}: {:?}", record.result.get("reason"));
    }
    println!("{panics} of {seeds} seeds panicked");
}
