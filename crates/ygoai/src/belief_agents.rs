//! Search agents on the belief's reconstruction: at a decision with several candidates
//! (search::is_searched prompts), the agent reconstructs the duel from its observations
//! (belief::agent::AgentBelief) and searches it; other decisions, and searched ones when the
//! reconstruction fails or time is short, go to the heuristic player. And belief_mc, the
//! belief checks' test agent.

use std::time::Instant;

use serde_json::Value;
use ygo_core::constants::*;

use ygo_arbiter::actions::{Response, battle_options, encode, idle_options};
use crate::agents::GreedyAgent;
use crate::belief::agent::AgentBelief;
use crate::belief::check::state_diffs;
use crate::belief::completion::{Completion, CorpusCompletion, VanillaCompletion, by_name, card};
use crate::belief::tracker::Need;
use crate::belief::script_index::zone_sensitive;
use ygo_arbiter::game::{Agent, AgentError, InitInfo, ObsItem, run_until_prompt};
use crate::heuristic::Heuristic;
use ygo_arbiter::messages::Msg;
use ygo_arbiter::observe::to_core_response;
use ygo_arbiter::pyrandom::PyRandom;
use crate::rduel::RDuel;
use crate::search::{AlphaBeta, SearchParams, TIME_BUFFER, hard_limit, root_candidates};
use ygo_arbiter::state::{LOCATIONS, PlayerCards, Snapshot, capture, viewer_view};
use ygo_core::duel::Duel;

/// public facts an approximate reconstruction may get wrong to be searched on
pub const MAX_APPROX_DIFFS: usize = 4;
/// alphabeta's budget per searched decision: (remaining - the reserve) / MOVES_TO_GO + the clock's mean gain
/// between two searches (measured: the increments of the decisions in between, hidden ones too,
/// less what they and the searched decision's own overhead took; before any, CLOCK_GAIN_PRIOR
/// searches' worth of one increment each)
pub const MOVES_TO_GO: f64 = 20.0;
pub const CLOCK_GAIN_PRIOR: f64 = 2.0;
/// (the reconstruction before a search cannot be stopped, and a slow one can take several times a
/// short clock's increment: the reserve is this many of the game's slowest reconstruction. A search
/// only with more than TIME_BUFFER past the reserve on the clock, and the budget does not count it)
pub const RESERVE_RECONSTRUCTIONS: f64 = 2.0;

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

/// The deck completion by name (`completion::by_name`), with what the belief replay and the
/// completions load on first use loaded now, before the clock runs (the neural completion takes the
/// corpus completion's cards for what its model cannot supply).
pub fn completion(name: &str) -> Result<Option<&'static dyn Completion>, String> {
    crate::belief::script_index::warm_up();
    crate::belief::completion::monster_sset_codes();
    VanillaCompletion::get();
    if name != "vanilla" {
        CorpusCompletion::get()?;
    }
    by_name(name)
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
    /// (the last response's search, `Agent::search_info`)
    info: Option<ygo_arbiter::game::SearchInfo>,
    /// (the arbiter's node limit per decision, before $YGOAI_SEARCH_NODES)
    node_limit: Option<u64>,
    /// (the clock's gains between searches: their sum and number; the clock after the last
    /// search, as far as this agent measured it)
    clock_gains: (f64, f64),
    clock_after_search: Option<f64>,
    /// (the game's slowest reconstruction's seconds)
    slowest_reconstruction: f64,
    /// (whether this agent's deck has a card for which zones matter, `zone_sensitive`)
    own_zones: bool,
    /// (the last search's values by root move, see `AlphaBeta::last_root_values`)
    root_values: Vec<Option<f64>>,
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
            node_limit: None,
            clock_gains: (0.0, 0.0),
            clock_after_search: None,
            slowest_reconstruction: 0.0,
            own_zones: false,
            root_values: Vec::new(),
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

    /// Appends a root search to the file `path` as a JSON line (`YGOAI_ROOT_LOG`, a measurement):
    /// the decision type, its candidates, the chosen one (0: the rules' answer, null: no search),
    /// its value minus the rules' answer's (a lower bound if that one failed low), whether a card
    /// for which zones matter is in this agent's deck or visible, and the seconds.
    fn log_root(&self, path: &std::ffi::OsStr, prompt: &Msg, state: Option<&Snapshot>, n: usize, chosen: Option<usize>, secs: f64) {
        use std::io::Write;
        let visible = state.is_some_and(|s| {
            s.players.iter().any(|p| {
                [&p.hand, &p.mzone, &p.szone, &p.grave, &p.removed, &p.extra]
                    .into_iter()
                    .flatten()
                    .flatten()
                    .any(|c| c.code.is_some_and(|code| code != 0 && zone_sensitive(code)))
            })
        });
        let gap = chosen.and_then(|k| Some(self.root_values.get(k).copied()?? - self.root_values.first().copied()??));
        let line = serde_json::json!({
            "kind": prompt.name(), "n": n, "chosen": chosen, "gap": gap,
            "zones": self.own_zones || visible, "secs": secs, "player": self.player,
        });
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(format!("{line}\n").as_bytes()); // (one write: lines of parallel games stay whole)
        }
    }

    fn reserve(&self) -> f64 {
        RESERVE_RECONSTRUCTIONS * self.slowest_reconstruction
    }

    fn budget(&self, clock: Option<(f64, f64)>) -> Option<f64> {
        clock.map(|(remaining, inc)| {
            let gain = (self.clock_gains.0 + CLOCK_GAIN_PRIOR * inc) / (self.clock_gains.1 + CLOCK_GAIN_PRIOR);
            ((remaining - self.reserve()).max(0.0) / MOVES_TO_GO + gain).max(0.0)
        })
    }

    /// The index of the candidate to play, from a search of the reconstruction (None: no
    /// reconstruction fit to search).
    fn search(&mut self, prompt: &Msg, state: Option<&Snapshot>, cands: &[Response], clock: Option<(f64, f64)>) -> Option<usize> {
        let t0 = Instant::now();
        let budget = self.budget(clock);
        // (a reconstruction past the search's hard limit gives up: the rules' player answers)
        let timeout = clock.zip(budget).map(|((remaining, _), b)| std::time::Duration::from_secs_f64(hard_limit(b, remaining)));
        let (r, duel) = self.belief.as_mut()?.reconstruct(&self.items, timeout);
        let Some(mut duel) = duel else {
            self.failures += 1;
            return None;
        };
        if r.approximate {
            // (a best-effort state only if it matches what this agent observes now)
            let ok = state.is_some_and(|s| state_diffs(s, &mut duel.d, self.player) <= MAX_APPROX_DIFFS);
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
        self.slowest_reconstruction = self.slowest_reconstruction.max(elapsed);
        let remaining = clock.map_or(0.0, |c| c.0);
        if matches!(self.search, Search::AlphaBetaNaive { .. }) {
            naive_hidden_cards(&mut duel, self.player, self.completion, &mut self.rng);
        }
        Some(match &mut self.search {
            Search::AlphaBeta { ab, max_nodes } | Search::AlphaBetaNaive { ab, max_nodes } => {
                let max_nodes = self.node_limit.or(*max_nodes);
                let budget = if max_nodes.is_some() { None } else { budget };
                let seeds = [self.rng.getrandbits(32) as u32, self.rng.getrandbits(32) as u32];
                ab.prepare();
                let before = ab.nodes;
                let params = SearchParams { budget, elapsed, clock_remaining: remaining, hint, seeds, max_nodes, depth: None, exact_root: false };
                let t_search = Instant::now();
                let (best, depth) = ab.search(&duel.d, &core_prompt, &core_cands, &params);
                self.nodes += ab.nodes - before;
                let (nodes, secs) = (ab.nodes - before, t_search.elapsed().as_secs_f64());
                self.root_values = ab.last_root_values.clone();
                self.info = ab.last_value.is_finite().then_some(ygo_arbiter::game::SearchInfo { value: ab.last_value, depth: depth as u32, nodes, secs });
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
        (self.clock_gains, self.clock_after_search, self.slowest_reconstruction) = ((0.0, 0.0), None, 0.0);
        self.own_zones = init.deck.main.iter().chain(&init.deck.extra).any(|&c| zone_sensitive(c));
        Ok(())
    }
    fn observe(&mut self, msg: &Msg) {
        self.items.push(ObsItem::Msg(msg.clone()));
        self.fallback.observe_msg(msg);
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
        self.items.push(ObsItem::Prompt(prompt.clone(), state.cloned()));
        self.info = None;
        // (every decision: the rules' answer and alternatives, see `root_candidates`)
        let mut cands = root_candidates(prompt, &self.fallback, state);
        // (the same effect at most MAX_SAME_ACTIVATION_PER_TURN times per turn, as the rules' player:
        // an effect that changes nothing, e.g. negated, would otherwise be activated forever)
        if cands.iter().any(|c| self.fallback.within_cap(prompt, c)) {
            cands.retain(|c| self.fallback.within_cap(prompt, c));
        }
        let mut resp = None;
        // (with no more than the reserve and TIME_BUFFER on the clock, the rules' player answers)
        if cands.len() > 1 && clock.is_none_or(|(remaining, _)| remaining > TIME_BUFFER + self.reserve()) {
            let t0 = Instant::now();
            if let (Some((remaining, _)), Some(after)) = (clock, self.clock_after_search) {
                self.clock_gains.0 += remaining - after;
                self.clock_gains.1 += 1.0;
            }
            self.root_values.clear();
            let k = self.search(prompt, state, &cands, clock);
            if let Some(path) = std::env::var_os("YGOAI_ROOT_LOG") {
                self.log_root(&path, prompt, state, cands.len(), k, t0.elapsed().as_secs_f64());
            }
            resp = k.map(|k| cands[k].clone());
            if let Some(r) = &resp {
                self.fallback.note_choice(prompt, r);
            }
            self.clock_after_search = clock.map(|(remaining, _)| remaining - t0.elapsed().as_secs_f64());
        }
        let resp = match resp {
            Some(r) => r,
            None => self.fallback.respond_to(prompt, state)?,
        };
        self.items.push(ObsItem::Response(resp.clone()));
        Ok(resp)
    }
    fn search_info(&mut self) -> Option<ygo_arbiter::game::SearchInfo> {
        self.info.take()
    }
    fn set_node_limit(&mut self, nodes: Option<u64>) {
        self.node_limit = nodes;
    }
    fn end(&mut self, _result: &serde_json::Map<String, Value>) {
        if std::env::var_os("YGOAI_AGENT_STATS").is_some() {
            eprintln!("{} stats: searches {} failures {} nodes {}", self.name, self.searches, self.failures, self.nodes);
        }
    }
}

/// Gives the opponent's currently hidden cards new codes guessed from the cards visible now only.
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

/// belief_mc's rollouts per candidate, the decisions a rollout plays at most, and the candidates
/// it searches at most
const MC_ROLLOUTS: usize = 2;
const MC_HORIZON: usize = 80;
const MC_MAX_CANDIDATES: usize = 16;

/// Flat Monte Carlo on the belief's reconstruction, the belief checks' test agent (the choices of
/// the former Python agent with the same seed; it plays without a time control): at a decision with
/// several candidates (`mc_candidates`), each is played MC_ROLLOUTS times on copies of the
/// reconstructed duel, both sides continued by greedy players for at most MC_HORIZON decisions, and
/// valued by `material_value`; the best mean is played. The greedy player answers the other
/// decisions, and those without a reconstruction fit to search.
pub struct BeliefMcAgent {
    rng: PyRandom,
    fallback: GreedyAgent,
    completion: Option<&'static dyn Completion>,
    player: u32,
    items: Vec<ObsItem>,
    belief: Option<AgentBelief>,
}

impl BeliefMcAgent {
    pub fn new(seed: i64) -> Result<BeliefMcAgent, String> {
        Ok(BeliefMcAgent {
            rng: PyRandom::new(seed as i128),
            fallback: GreedyAgent::new(seed),
            completion: completion("neural")?,
            player: 0,
            items: Vec::new(),
            belief: None,
        })
    }

    /// The index of the candidate to play (None: no reconstruction fit to search).
    fn search(&mut self, prompt: &Msg, state: Option<&Snapshot>, cands: &[Response]) -> Result<Option<usize>, AgentError> {
        let (r, duel) = self.belief.as_mut().expect("started").reconstruct(&self.items, None);
        let Some(mut duel) = duel else { return Ok(None) };
        // (a best-effort state only if it matches what this agent observes now)
        if r.approximate && !state.is_some_and(|s| state_diffs(s, &mut duel.d, self.player) <= MAX_APPROX_DIFFS) {
            return Ok(None);
        }
        let core_prompt = r.final_prompt.unwrap_or_else(|| prompt.clone());
        let core_cands = cands
            .iter()
            .map(|c| to_core_response(&core_prompt, self.player, c))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| AgentError::Failed(format!("InvalidResponse: {}", e.0)))?;
        let mut best: Option<(usize, f64)> = None;
        for (k, cand) in core_cands.iter().enumerate() {
            let mut total = 0.0;
            for _ in 0..MC_ROLLOUTS {
                let seed = self.rng.getrandbits(32) as i64;
                total += rollout(duel.d.clone(), &core_prompt, cand, seed, self.player).unwrap_or(-1.0);
            }
            let mean = total / MC_ROLLOUTS as f64;
            if best.is_none_or(|(_, b)| mean > b) {
                best = Some((k, mean));
            }
        }
        Ok(best.map(|(k, _)| k))
    }
}

/// The candidates belief_mc searches: the commands of an idle or battle command, a chain window's
/// effects (and passing, unless forced), yes and no, the options, the cards of a one-card selection.
fn mc_candidates(prompt: &Msg) -> Vec<Response> {
    match prompt {
        Msg::SelectIdlecmd { .. } => idle_options(prompt),
        Msg::SelectBattlecmd { .. } => battle_options(prompt),
        Msg::SelectChain { chains, forced, .. } => {
            let pass = (*forced == 0).then(Response::pass);
            pass.into_iter().chain((0..chains.len()).map(Response::index)).collect()
        }
        Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => vec![Response::yes(true), Response::yes(false)],
        Msg::SelectOption { options, .. } => (0..options.len()).map(Response::index).collect(),
        Msg::SelectCard { min: 1, max: 1, cards, .. } => (0..cards.len()).map(|i| Response::cards(vec![i])).collect(),
        _ => Vec::new(),
    }
}

/// `cand` played on `duel` (a copy), then greedy players for both sides: 1 / -1 if `me` won / lost
/// within MC_HORIZON decisions, else `material_value` (None: a broken line).
fn rollout(mut duel: Duel, prompt: &Msg, cand: &Response, seed: i64, me: u32) -> Option<f64> {
    let mut players = [GreedyAgent::new(seed), GreedyAgent::new(seed + 1)];
    duel.ocg_set_response(&encode(prompt, cand).ok()?);
    let mut prompt = prompt.clone();
    for _ in 0..MC_HORIZON {
        let step = run_until_prompt(&mut duel, Some(&prompt)).ok()?;
        if let Some(winner) = step.messages.iter().find_map(|m| if let Msg::Win { player, .. } = m { Some(*player) } else { None }) {
            return Some(if winner == me { 1.0 } else { -1.0 });
        }
        if step.ended {
            break;
        }
        prompt = step.prompt?;
        let response = players[prompt.player()? as usize].respond(&prompt, None, None).ok()?;
        duel.ocg_set_response(&encode(&prompt, &response).ok()?);
    }
    Some(material_value(&mut duel, me))
}

/// belief_mc's value of the position for `me`, in [-1, 1]: the LP difference / 8000 plus a tenth
/// of the material difference (cards on the field, half a card per card in hand, the face-up
/// monsters' ATK / 3000).
fn material_value(duel: &mut Duel, me: u32) -> f64 {
    let snap = capture(duel);
    let material = |p: &PlayerCards| {
        let field = p.mzone.iter().chain(&p.szone).flatten().count() as f64;
        let atk: i64 = p.mzone.iter().flatten().filter(|c| c.position.unwrap_or(0) & POS_FACEUP != 0).map(|c| c.attack.unwrap_or(0) as i64).sum();
        field + 0.5 * p.hand.len() as f64 + atk as f64 / 3000.0
    };
    let (mine, theirs) = (&snap.players[me as usize], &snap.players[1 - me as usize]);
    let value = (snap.lp[me as usize] - snap.lp[1 - me as usize]) as f64 / 8000.0 + 0.1 * (material(mine) - material(theirs));
    value.clamp(-1.0, 1.0)
}

impl Agent for BeliefMcAgent {
    fn name(&self) -> String {
        "belief_mc".into()
    }
    fn wants_state(&self) -> bool {
        true
    }
    fn start(&mut self, player: u32, init: &InitInfo) -> Result<(), AgentError> {
        self.player = player;
        self.items.clear();
        self.belief = Some(AgentBelief::new(player, init.clone(), self.completion));
        Ok(())
    }
    fn observe(&mut self, msg: &Msg) {
        self.items.push(ObsItem::Msg(msg.clone()));
        self.fallback.observe(msg);
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
        self.items.push(ObsItem::Prompt(prompt.clone(), state.cloned()));
        let mut cands = mc_candidates(prompt);
        cands.truncate(MC_MAX_CANDIDATES);
        let searched = if cands.len() > 1 { self.search(prompt, state, &cands)? } else { None };
        let response = match searched {
            Some(k) => cands[k].clone(),
            None => self.fallback.respond(prompt, state, None)?,
        };
        self.items.push(ObsItem::Response(response.clone()));
        Ok(response)
    }
}

/// Agents by name, the belief ones included (not Send: the belief's state is not); None: unknown.
pub fn make_agent(name: &str, seed: Option<i64>) -> Result<Box<dyn Agent>, String> {
    let seed_or_entropy = seed.unwrap_or_else(|| entropy() as i64);
    if let Some(a) = crate::agents::make_agent(name, seed_or_entropy) {
        return Ok(a);
    }
    if name == "belief_mc" {
        return Ok(Box::new(BeliefMcAgent::new(seed_or_entropy)?));
    }
    Ok(Box::new(BeliefSearchAgent::new(name, seed)?))
}
