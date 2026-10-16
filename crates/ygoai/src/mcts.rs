//! Plain Monte Carlo tree search (UCT) with random rollouts: no value function, no policy.
//!
//! The tree's nodes are the same decisions alpha-beta's plies are (search::is_searched prompts with
//! more than one candidate). Every simulation starts from a copy of the root duel and walks down
//! by UCB1 (an untried move first, in the prompt's order); the prompts between two tree decisions
//! (targets, positions, ...) are answered by the rules' players with the same seeds in every
//! simulation, so a path of moves always leads to the same position. At the first move without a
//! node the position it leads to becomes a node, and a rollout plays the game on from there with
//! every answer uniformly random (the random agent's), until it ends: a win counts 1, a loss 0, a
//! game still going after `rollout_turns` more turns (ROLLOUT_TURNS) or ROLLOUT_DECISIONS answers
//! 1/2, or with `cutoff_eval` the handcrafted evaluation (search::evaluate) as a win probability,
//! sigmoid(VALUE_K * value). The result is backed up along the path, for each node's player. The
//! move: the root's most visited one.

use std::time::Instant;

use ygo_core::constants::HINT_SELECTMSG;
use ygo_core::duel::Duel;

use crate::actions::{Response, encode};
use crate::agents::random_response;
use crate::game::run_until_prompt;
use crate::heuristic::Heuristic;
use crate::messages::Msg;
use crate::observe::{auto_response, is_hidden_prompt};
use crate::pyrandom::PyRandom;
use crate::search::{MAX_AUTO_STEPS, candidates, is_searched};

/// UCB1's exploration constant by default (results in [0, 1]; MctsParams::c to tune it)
pub const C_UCT: f64 = std::f64::consts::SQRT_2;
pub const ROLLOUT_TURNS: u32 = 30;
/// the evaluation's LP -> win probability (logistic regression of the SPRT games' results, as
/// agents/puct.py's VALUE_K: 0.0216 per point of 100 LP)
pub const VALUE_K: f64 = 0.000216;
pub const ROLLOUT_DECISIONS: usize = 5000;
/// simulations per decision without a time control or a given number
pub const SIMS_NO_CLOCK: u64 = 200;

struct Node {
    player: u32,
    cands: Vec<Response>,
    visits: Vec<u32>,
    /// summed results for this node's player
    wins: Vec<f64>,
    children: Vec<Option<usize>>,
    total: u32,
    /// the game is over here: player 0's result
    terminal: Option<f64>,
}

impl Node {
    fn new(player: u32, cands: Vec<Response>) -> Node {
        let n = cands.len();
        Node { player, cands, visits: vec![0; n], wins: vec![0.0; n], children: vec![None; n], total: 0, terminal: None }
    }

    fn terminal(v0: f64) -> Node {
        Node { terminal: Some(v0), ..Node::new(0, vec![]) }
    }

    fn select(&self, c: f64) -> usize {
        if let Some(k) = self.visits.iter().position(|&n| n == 0) {
            return k;
        }
        let ln = (self.total.max(1) as f64).ln();
        let ucb = |k: usize| self.wins[k] / self.visits[k] as f64 + c * (ln / self.visits[k] as f64).sqrt();
        (0..self.cands.len()).max_by(|&a, &b| ucb(a).partial_cmp(&ucb(b)).unwrap_or(std::cmp::Ordering::Equal)).unwrap()
    }
}

/// What a search is given besides the duel: time (`budget` in seconds, what was already spent on
/// this decision), or a number of simulations; the exploration constant (None: C_UCT), the select
/// hint before a card selection and the seeds (the rules' players in the tree, the rollouts).
#[derive(Default)]
pub struct MctsParams {
    pub budget: Option<f64>,
    pub elapsed: f64,
    pub max_sims: Option<u64>,
    pub c: Option<f64>,
    /// new turns a rollout plays at most (None: ROLLOUT_TURNS)
    pub rollout_turns: Option<u32>,
    /// a rollout cut off at its length counts the evaluation, not 1/2
    pub cutoff_eval: bool,
    pub hint: Option<u64>,
    pub seeds: [u32; 3],
}

/// What a search did.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MctsStats {
    pub sims: u64,
    /// tree moves to each simulation's leaf: summed and the deepest
    pub leaf_depth_sum: u64,
    pub leaf_depth_max: usize,
    /// rollouts: run, their answers and new turns (summed), and how they ended (won by player 0,
    /// by player 1, or anything else: cut off at a cap, a draw, an engine error)
    pub rollouts: u64,
    pub rollout_decisions: u64,
    pub rollout_turns: u64,
    pub rollout_won0: u64,
    pub rollout_won1: u64,
    pub rollout_other: u64,
    /// simulations that ended in the tree (a game over found by the tree itself)
    pub terminal: u64,
    /// the tree's size, and the root's visits per candidate
    pub nodes: usize,
    pub root_visits: Vec<u32>,
}

/// (the best candidate, what the search did)
pub fn search(duel: &Duel, prompt: &Msg, cands: &[Response], p: &MctsParams) -> (usize, MctsStats) {
    let t0 = Instant::now() - std::time::Duration::from_secs_f64(p.elapsed);
    let deadline = p.budget.map(|b| t0 + std::time::Duration::from_secs_f64(b));
    let c_uct = p.c.unwrap_or(C_UCT);
    let roll = Rollout { turns: p.rollout_turns.unwrap_or(ROLLOUT_TURNS) as u64, eval: p.cutoff_eval };
    let max_sims = match (p.max_sims, p.budget) {
        (Some(n), _) => n,
        (None, Some(_)) => u64::MAX,
        (None, None) => SIMS_NO_CLOCK,
    };
    let root_player = prompt.player().unwrap_or(0);
    let mut nodes = vec![Node::new(root_player, cands.to_vec())];
    let mut rng = PyRandom::new(p.seeds[2] as i128);
    let mut st = MctsStats::default();
    let mut sims = 0u64;
    while sims < max_sims && deadline.is_none_or(|d| Instant::now() < d) {
        sims += 1;
        let mut d = duel.clone();
        let mut policies = [Heuristic::new(p.seeds[0] as i128, 0), Heuristic::new(p.seeds[1] as i128, 1)];
        policies[root_player as usize].hint = p.hint;
        let mut prompt = prompt.clone();
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut at = 0;
        let v0 = loop {
            let k = nodes[at].select(c_uct);
            path.push((at, k));
            let next = advance(&mut d, &prompt, &nodes[at].cands[k], &mut policies);
            match nodes[at].children[k] {
                Some(c) if nodes[c].terminal.is_some() => {
                    st.terminal += 1;
                    break nodes[c].terminal.unwrap();
                }
                Some(c) => match next {
                    Next::Ply(q) if candidates(&q).len() == nodes[c].cands.len() => {
                        prompt = q;
                        at = c;
                    }
                    // (another position than before, e.g. an engine error: a rollout from here)
                    Next::Ply(q) => break rollout(&mut d, q, &mut rng, &mut st, &roll),
                    Next::End(v) => {
                        st.terminal += 1;
                        break v;
                    }
                    Next::Broken => break broken(&nodes[at]),
                },
                None => {
                    let (node, v) = match next {
                        Next::Ply(q) => {
                            let n = Node::new(q.player().unwrap_or(0), candidates(&q));
                            let v = rollout(&mut d, q, &mut rng, &mut st, &roll);
                            (n, v)
                        }
                        Next::End(v) => {
                            st.terminal += 1;
                            (Node::terminal(v), v)
                        }
                        Next::Broken => {
                            let v = broken(&nodes[at]);
                            (Node::terminal(v), v)
                        }
                    };
                    nodes.push(node);
                    let c = nodes.len() - 1;
                    nodes[at].children[k] = Some(c);
                    break v;
                }
            }
        };
        st.leaf_depth_sum += path.len() as u64;
        st.leaf_depth_max = st.leaf_depth_max.max(path.len());
        for (n, k) in path {
            let node = &mut nodes[n];
            node.visits[k] += 1;
            node.total += 1;
            node.wins[k] += if node.player == 0 { v0 } else { 1.0 - v0 };
        }
    }
    let root = &nodes[0];
    let best = (0..root.cands.len())
        .max_by(|&a, &b| {
            let q = |k: usize| root.wins[k] / root.visits[k].max(1) as f64;
            (root.visits[a], q(a)).partial_cmp(&(root.visits[b], q(b))).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    st.sims = sims;
    st.nodes = nodes.len();
    st.root_visits = root.visits.clone();
    (best, st)
}

enum Next {
    /// the next tree decision
    Ply(Msg),
    /// the game ended: player 0's result
    End(f64),
    /// an engine error
    Broken,
}

/// (an engine error: the player who moved there avoids the line, as a loss)
fn broken(node: &Node) -> f64 {
    if node.player == 0 { 0.0 } else { 1.0 }
}

fn win_value(m: &Msg) -> Option<f64> {
    match m {
        Msg::Win { player, .. } if *player == 0 => Some(1.0),
        Msg::Win { player, .. } if *player == 1 => Some(0.0),
        Msg::Win { .. } => Some(0.5),
        _ => None,
    }
}

/// Plays `cand` and runs to the next tree decision, the rules' players answering in between.
fn advance(d: &mut Duel, prompt: &Msg, cand: &Response, policies: &mut [Heuristic; 2]) -> Next {
    let Ok(data) = encode(prompt, cand) else { return Next::Broken };
    d.ocg_set_response(&data);
    let mut prompt = prompt.clone();
    for _ in 0..MAX_AUTO_STEPS {
        let Ok(step) = run_until_prompt(d, Some(&prompt)) else { return Next::Broken };
        for m in &step.messages {
            if let Some(v) = win_value(m) {
                return Next::End(v);
            }
            if matches!(m, Msg::NewTurn { .. } | Msg::Hint { hint_type: HINT_SELECTMSG, .. }) {
                for p in policies.iter_mut() {
                    p.observe_msg(m);
                }
            }
        }
        if step.ended {
            return Next::End(0.5);
        }
        prompt = step.prompt.unwrap();
        let player = prompt.player().unwrap_or(0) as usize;
        let resp = if is_hidden_prompt(&prompt) {
            auto_response()
        } else if is_searched(&prompt) {
            let cands = candidates(&prompt);
            if cands.len() > 1 {
                return Next::Ply(prompt);
            }
            match cands.into_iter().next() {
                Some(c) => c,
                None => match policies[player].respond_to(&prompt, None) {
                    Ok(r) => r,
                    Err(_) => return Next::Broken,
                },
            }
        } else {
            match policies[player].respond_to(&prompt, None) {
                Ok(r) => r,
                Err(_) => return Next::Broken,
            }
        };
        let Ok(data) = encode(&prompt, &resp) else { return Next::Broken };
        d.ocg_set_response(&data);
    }
    Next::End(0.5)
}

/// Plays the game on from `prompt` with uniformly random answers: player 0's result.
/// How a rollout ends: its length in new turns, and whether one cut off counts the evaluation.
struct Rollout {
    turns: u64,
    eval: bool,
}

fn rollout(d: &mut Duel, prompt: Msg, rng: &mut PyRandom, st: &mut MctsStats, r: &Rollout) -> f64 {
    let (mut decisions, mut turns) = (0u64, 0u64);
    let v = rollout_inner(d, prompt, rng, &mut decisions, &mut turns, r.turns);
    st.rollouts += 1;
    st.rollout_decisions += decisions;
    st.rollout_turns += turns;
    match v {
        Some(1.0) => st.rollout_won0 += 1,
        Some(0.0) => st.rollout_won1 += 1,
        _ => st.rollout_other += 1,
    }
    match v {
        Some(x) => x,
        None if r.eval => 1.0 / (1.0 + (-(VALUE_K * crate::search::evaluate(d, 0)).clamp(-50.0, 50.0)).exp()),
        None => 0.5,
    }
}

/// (None: cut off at a cap, or an engine error)
fn rollout_inner(d: &mut Duel, mut prompt: Msg, rng: &mut PyRandom, decisions: &mut u64, turns: &mut u64, max_turns: u64) -> Option<f64> {
    for _ in 0..ROLLOUT_DECISIONS {
        *decisions += 1;
        let resp = if is_hidden_prompt(&prompt) {
            auto_response()
        } else {
            match random_response(&prompt, rng, 0.05) {
                Ok(r) => r,
                Err(_) => return None,
            }
        };
        let Ok(data) = encode(&prompt, &resp) else { return None };
        d.ocg_set_response(&data);
        let Ok(step) = run_until_prompt(d, Some(&prompt)) else { return None };
        for m in &step.messages {
            if let Some(v) = win_value(m) {
                return Some(v);
            }
            if matches!(m, Msg::NewTurn { .. }) {
                *turns += 1;
                if *turns > max_turns {
                    return None;
                }
            }
        }
        match step.prompt {
            Some(q) if !step.ended => prompt = q,
            _ => return None,
        }
    }
    None
}
