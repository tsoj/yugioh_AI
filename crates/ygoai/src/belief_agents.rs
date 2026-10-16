//! Search agents on the belief's reconstruction (agents/belief_mc.py's BeliefMCAgent and its
//! subclasses alphabeta.py, alphabeta_naive.py and mcts.py): at a decision with several candidates
//! (search::is_searched prompts), the agent reconstructs the duel from its observations
//! (belief::agent::AgentBelief) and searches it; other decisions, and searched ones when the
//! reconstruction fails or time is short, go to the heuristic player. Same seeds, same draws from
//! the same random generator as the Python agents: the same moves.

use std::collections::BTreeSet;
use std::time::Instant;

use serde_json::Value;
use ygo_core::constants::*;

use crate::actions::Response;
use crate::belief::agent::AgentBelief;
use crate::belief::completion::{Completion, CorpusCompletion, VanillaCompletion, card};
use crate::belief::neural::NeuralCompletion;
use crate::belief::tracker::Need;
use crate::game::{Agent, AgentError, InitInfo, ObsItem};
use crate::heuristic::Heuristic;
use crate::messages::Msg;
use crate::observe::to_core_response;
use crate::pyrandom::PyRandom;
use crate::rduel::RDuel;
use crate::search::{AlphaBeta, SearchParams, candidates_in, is_searched};
use crate::state::{LOCATIONS, Snapshot, viewer_view};

/// public facts an approximate reconstruction may get wrong to be searched on
pub const MAX_APPROX_DIFFS: usize = 4;
/// seconds left below which it plays by the rules (also 3 increments)
pub const MIN_SEARCH_TIME: f64 = 2.0;
pub const MAX_CANDIDATES: usize = 16;
/// alphabeta's budget per searched decision: remaining / MOVES_TO_GO + increment
pub const MOVES_TO_GO: f64 = 20.0;

/// What searches the reconstructed duel.
pub enum Search {
    AlphaBeta { ab: Box<AlphaBeta>, max_nodes: Option<u64> },
    /// alpha-beta on a copy whose opponent's hidden cards are guessed from the visible ones only
    AlphaBetaNaive { ab: Box<AlphaBeta>, max_nodes: Option<u64> },
    Mcts { max_sims: Option<u64>, c: Option<f64>, rollout_turns: Option<u32>, cutoff_eval: bool },
}

impl Search {
    fn alphabeta(kind: &str) -> Search {
        let max_nodes = env_num("YGOAI_SEARCH_NODES").filter(|&n| n > 0);
        let ab = Box::default();
        match kind {
            "naive" => Search::AlphaBetaNaive { ab, max_nodes },
            _ => Search::AlphaBeta { ab, max_nodes },
        }
    }
    fn mcts() -> Search {
        Search::Mcts {
            max_sims: env_num("YGOAI_MCTS_SIMS").filter(|&n| n > 0),
            c: std::env::var("YGOAI_MCTS_C").ok().and_then(|s| s.parse().ok()),
            rollout_turns: env_num("YGOAI_MCTS_ROLLOUT_TURNS").map(|n| n as u32),
            cutoff_eval: std::env::var("YGOAI_MCTS_EVAL").is_ok_and(|s| s == "1"),
        }
    }
}

fn env_num(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

/// The deck completion by name (vanilla, corpus, neural; neural without the models: corpus).
pub fn completion(name: &str) -> Result<Option<&'static dyn Completion>, String> {
    Ok(match name {
        "vanilla" => None,
        "corpus" => Some(CorpusCompletion::get()?),
        "neural" => match NeuralCompletion::get() {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("{e}: the corpus completion instead");
                Some(CorpusCompletion::get()?)
            }
        },
        _ => return Err(format!("unknown completion {name}")),
    })
}

pub struct BeliefSearchAgent {
    name: String,
    rng: PyRandom,
    fallback: Heuristic,
    search: Search,
    completion: Option<&'static dyn Completion>,
    player: u32,
    items: Vec<ObsItem>,
    belief: Option<AgentBelief>,
    /// (the last response's search: its root value and depth, `Agent::search_info`)
    info: Option<(f64, u32)>,
    // statistics
    pub searches: usize,
    pub failures: usize,
    pub nodes: u64,
}

impl BeliefSearchAgent {
    /// `name`: alphabeta, alphabeta_naive or mcts; `seed`: None for an unseeded generator.
    pub fn new(name: &str, seed: Option<i64>) -> Result<BeliefSearchAgent, String> {
        let search = match name {
            "alphabeta" => Search::alphabeta("full"),
            "alphabeta_naive" => Search::alphabeta("naive"),
            "mcts" => Search::mcts(),
            _ => return Err(format!("unknown belief agent {name}")),
        };
        let s = seed.map(|s| s as i128).unwrap_or_else(entropy);
        let completion = completion(&std::env::var("YGOAI_COMPLETION").unwrap_or_else(|_| "neural".into()))?;
        Ok(BeliefSearchAgent {
            name: name.into(),
            rng: PyRandom::new(s),
            // (the rules' player answers what is not searched: alphabeta.py's HeuristicAgent(seed))
            fallback: Heuristic::new(s, 0),
            search,
            completion,
            player: 0,
            items: Vec::new(),
            belief: None,
            info: None,
            searches: 0,
            failures: 0,
            nodes: 0,
        })
    }

    /// Replaces the response this agent just gave (a replay of a recorded game: the agent's
    /// history holds what was played).
    pub fn set_last_response(&mut self, r: Response) {
        if let Some(ObsItem::Response(last)) = self.items.last_mut() {
            *last = r;
        }
    }

    fn budget(&self, clock: Option<(f64, f64)>) -> Option<f64> {
        clock.map(|(remaining, inc)| remaining / MOVES_TO_GO + inc)
    }

    /// The index of the candidate to play, from a search of the reconstruction (None: no
    /// reconstruction fit to search).
    fn search(&mut self, prompt: &Msg, state: Option<&Snapshot>, cands: &[Response], clock: Option<(f64, f64)>) -> Option<usize> {
        let t0 = Instant::now();
        let budget = self.budget(clock);
        let (r, duel) = self.belief.as_mut()?.reconstruct(&self.items, None);
        let Some(mut duel) = duel else {
            self.failures += 1;
            return None;
        };
        if r.approximate {
            // (a best-effort state only if it matches what this agent observes now)
            let ok = state.is_some_and(|s| count_diffs(&jv(s), &jv(&viewer_view(&mut duel.d, self.player))) <= MAX_APPROX_DIFFS);
            if !ok {
                self.failures += 1;
                return None;
            }
        }
        self.searches += 1;
        let core_prompt = r.final_prompt.clone().unwrap_or_else(|| prompt.clone());
        let core_cands: Vec<Response> = cands.iter().map(|c| to_core_response(&core_prompt, self.player, c).ok()).collect::<Option<_>>()?;
        let hint = self.fallback.hint.take(); // (the select hint: the search answers this selection)
        let elapsed = t0.elapsed().as_secs_f64();
        let remaining = clock.map_or(0.0, |c| c.0);
        if matches!(self.search, Search::AlphaBetaNaive { .. }) {
            naive_hidden_cards(&mut duel, self.player, self.completion, &mut self.rng);
        }
        Some(match &mut self.search {
            Search::AlphaBeta { ab, max_nodes } | Search::AlphaBetaNaive { ab, max_nodes } => {
                let budget = if max_nodes.is_some() { None } else { budget };
                let seeds = [self.rng.getrandbits(32) as u32, self.rng.getrandbits(32) as u32];
                ab.prepare();
                let before = ab.nodes;
                let params = SearchParams { budget, elapsed, clock_remaining: remaining, hint, seeds, max_nodes: *max_nodes };
                let (best, depth) = ab.search(&duel.d, &core_prompt, &core_cands, &params);
                self.nodes += ab.nodes - before;
                self.info = ab.last_value.is_finite().then_some((ab.last_value, depth as u32));
                best
            }
            Search::Mcts { max_sims, c, rollout_turns, cutoff_eval } => {
                let seeds = [self.rng.getrandbits(32) as u32, self.rng.getrandbits(32) as u32, self.rng.getrandbits(32) as u32];
                let budget = if max_sims.is_some() { None } else { budget };
                let params = crate::mcts::MctsParams { budget, elapsed, max_sims: *max_sims, c: *c, rollout_turns: *rollout_turns, cutoff_eval: *cutoff_eval, hint, seeds };
                let (best, stats) = crate::mcts::search(&duel.d, &core_prompt, &core_cands, &params);
                self.nodes += stats.sims;
                best
            }
        })
    }
}

fn jv<T: serde::Serialize>(x: &T) -> Value {
    serde_json::to_value(x).unwrap_or(Value::Null)
}

/// Facts that differ between two views (check._count_diffs).
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

fn entropy() -> i128 {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    (t ^ ((std::process::id() as u128) << 64)) as i128 & ((1i128 << 64) - 1)
}

impl Agent for BeliefSearchAgent {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn wants_state(&self) -> bool {
        true
    }
    fn start(&mut self, player: u32, init: &InitInfo) -> Result<(), AgentError> {
        self.player = player;
        self.fallback.player = player;
        self.items.clear();
        self.belief = Some(AgentBelief::new(player, init.clone(), self.completion));
        Ok(())
    }
    fn observe(&mut self, msg: &Msg) {
        self.items.push(ObsItem::Msg(msg.clone()));
        self.fallback.observe_msg(msg);
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
        self.items.push(ObsItem::Prompt(prompt.clone(), state.cloned()));
        self.info = None;
        // (a card selection's candidates by the rules' order of its cards: the search keeps the first)
        let order = match prompt {
            Msg::SelectCard { cards, .. } | Msg::SelectUnselectCard { select_cards: cards, .. } => Some(self.fallback.card_order(cards)),
            _ => None,
        };
        let mut cands = if is_searched(prompt) { candidates_in(prompt, order.as_deref()) } else { vec![] };
        cands.truncate(MAX_CANDIDATES);
        let mut resp = None;
        if cands.len() > 1 && clock.is_none_or(|(remaining, inc)| remaining > MIN_SEARCH_TIME.max(3.0 * inc)) {
            resp = self.search(prompt, state, &cands, clock).map(|k| cands[k].clone());
        }
        let resp = match resp {
            Some(r) => r,
            None => self.fallback.respond_to(prompt, state)?,
        };
        self.items.push(ObsItem::Response(resp.clone()));
        Ok(resp)
    }
    fn search_info(&mut self) -> Option<(f64, u32)> {
        self.info.take()
    }
    fn end(&mut self, _result: &serde_json::Map<String, Value>) {
        if std::env::var_os("YGOAI_AGENT_STATS").is_some() {
            eprintln!("{} stats: searches {} failures {} nodes {}", self.name, self.searches, self.failures, self.nodes);
        }
    }
}

/// Gives the opponent's currently hidden cards new codes guessed from the cards visible now only
/// (agents/alphabeta_naive.py).
pub fn naive_hidden_cards(duel: &mut RDuel, me: u32, completion: Option<&'static dyn Completion>, rng: &mut PyRandom) {
    let comp: &dyn Completion = completion.unwrap_or(VanillaCompletion::get());
    let opp = 1 - me;
    let seen = viewer_view(&mut duel.d, me);
    let mut visible: Vec<u32> = Vec::new();
    let (mut hidden_field, mut hidden_main, mut hidden_extra): (Vec<(u32, Need)>, Vec<u32>, Vec<u32>) = (vec![], vec![], vec![]);
    for p in 0..2u32 {
        for (_, loc) in LOCATIONS {
            let ids = duel.location_cards(p, loc);
            for (i, entry) in seen.players[p as usize].pile(loc).iter().enumerate() {
                let Some(e) = entry else { continue };
                if let Some(c) = e.code.filter(|&c| c != 0) {
                    visible.push(c);
                    continue;
                }
                if p != opp {
                    continue;
                }
                let cid = ids.get(i).map_or(0, |x| x.1);
                if cid == 0 {
                    continue;
                }
                if loc & (LOCATION_MZONE | LOCATION_SZONE) != 0 {
                    let need = if loc == LOCATION_MZONE {
                        Need::Monster
                    } else if i == 5 {
                        Need::Field
                    } else {
                        Need::St
                    };
                    hidden_field.push((cid, need));
                } else if loc == LOCATION_EXTRA {
                    hidden_extra.push(cid);
                } else {
                    hidden_main.push(cid);
                }
            }
        }
    }
    let opp_visible: Vec<u32> = LOCATIONS
        .iter()
        .flat_map(|&(_, loc)| seen.players[opp as usize].pile(loc).iter().flatten().filter_map(|e| e.code.filter(|&c| c != 0)))
        .collect();
    let is_extra = |c: u32| card(c).is_some_and(crate::belief::completion::is_extra_deck);
    let known_main: Vec<u32> = opp_visible.iter().copied().filter(|&c| !is_extra(c)).collect();
    let known_extra: Vec<u32> = opp_visible.iter().copied().filter(|&c| is_extra(c)).collect();
    let (n_main, n_extra) = (hidden_field.len() + hidden_main.len(), hidden_extra.len());
    let (mut fill_main, fill_extra) = comp
        .complete(&known_main, &known_extra, n_main, n_extra, &visible)
        .or_else(|_| VanillaCompletion::get().complete(&known_main, &known_extra, n_main, n_extra, &visible))
        .unwrap_or_default();
    let fits = |code: u32, need: Need| {
        card(code).is_some_and(|c| match need {
            Need::Monster => c.type_ & TYPE_MONSTER != 0 && !crate::belief::completion::is_extra_deck(c),
            Need::Field => c.type_ & TYPE_FIELD != 0,
            _ => c.type_ & (TYPE_SPELL | TYPE_TRAP) != 0 && c.type_ & TYPE_FIELD == 0,
        })
    };
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (cid, need) in hidden_field {
        if let Some(k) = fill_main.iter().position(|&c| fits(c, need)) {
            out.push((cid, fill_main.remove(k)));
            continue;
        }
        if let Some(&s) = comp.stand_ins(need, &known_main).iter().find(|&&c| fits(c, need)) {
            out.push((cid, s));
            fill_main.pop(); // (the guesses stay as many as the hidden cards)
        }
    }
    rng.shuffle(&mut fill_main);
    out.extend(hidden_main.into_iter().zip(fill_main));
    out.extend(hidden_extra.into_iter().zip(fill_extra));
    for (cid, code) in out {
        duel.recreate_card(cid, code);
    }
}

/// Agents by name, the belief ones included (not Send: the belief's state is not); None: unknown.
pub fn make_agent(name: &str, seed: Option<i64>) -> Result<Box<dyn Agent>, String> {
    if let Some(a) = crate::agents::make_agent(name, seed.unwrap_or_else(|| entropy() as i64)) {
        return Ok(a);
    }
    Ok(Box::new(BeliefSearchAgent::new(name, seed)?))
}
