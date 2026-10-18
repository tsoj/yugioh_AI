//! Alpha-beta search with iterative deepening on a (reconstructed) duel, on copies of the duel,
//! in the shape of a chess engine's (Nalwald's search.nim, Stockfish's search.cpp): `search` is
//! a negamax node with a transposition table, reverse futility pruning, futility pruning, late
//! move reductions and PVS; `quiesce` goes on past the depth with attacks and chain answers;
//! `iterative_deepening` runs the root.
//!
//! What differs from chess is the move: a response to the core's prompt, after which the core
//! runs on and may ask the same player again, the other player, or nobody (the game ends). So a
//! child is reached by `search_move`, which makes the move (`do_move`: the response, then the
//! decisions the rules answer, up to the next searched decision) and searches the new position,
//! negating its value and window only if the other player decides there. Its value is always
//! the mover's, as `-search(child)` is in chess.

use std::collections::BTreeMap;
use std::time::Instant;

use ygo_core::constants::*;
use ygo_core::duel::Duel;

use super::evaluate::{LearnedEval, evaluate_at};
use super::history::{HistoryKey, HistoryTable, counter_key};
use super::movegen::{candidates, is_inner, is_searched};
use super::movepick::{listed, move_key, move_kind, ordered, prompt_label};
use super::params::*;
use super::tt::{NodeType, TranspositionTable, hash_key};
use crate::actions::{Response, encode};
use crate::game::run_until_prompt;
use crate::heuristic::Heuristic;
use crate::messages::Msg;
use crate::observe::{auto_response, is_hidden_prompt};

/// Why a line was not valued: out of time (the iteration is abandoned) or broken (the engine
/// failed on it: the player to move avoids it).
pub(super) enum Stop {
    Abort,
    Broken,
}

/// Statistics of the expanded nodes (nodes that got to their moves, not cut by the TT).
#[derive(Default)]
pub struct NodeStats {
    /// by (prompt, options listed, children after the caps): [nodes, children searched]
    pub widths: BTreeMap<(&'static str, u16, u16), [u64; 2]>,
    /// by (prompt, TT state, children, the move index of the cutoff or `NO_CUT`): nodes. The TT
    /// state: 0 no entry, 1 an entry without a move, else 2 + its node type (pv, cut, all).
    pub cuts: BTreeMap<(&'static str, u8, u8, u8), u64>,
    /// by (prompt, move index): [reduced searches by LMR, of them searched again]
    pub lmr: BTreeMap<(&'static str, u8), [u64; 2]>,
    /// by (prompt, depth): [nodes, nodes in their subtrees (themselves included)]
    pub subtree: BTreeMap<(&'static str, i8), [u64; 2]>,
    /// (futility margins) the moves after the first at depths 1..=4 with a finite alpha: (move
    /// kind, depth, alpha minus the node's static eval, whether the move raised alpha)
    pub gaps: Vec<(String, i8, f32, bool)>,
}

/// (`AlphaBeta::trace`) The edges a search went along, for comparing the trees of searches of
/// the same decision under different guesses of the hidden cards (ygoai-tree-compare).
#[derive(Default)]
pub struct Trace {
    /// the player whose view counts (the root's)
    pub viewer: u32,
    /// the cards in question (the opponent's hidden ones at the root), by card id
    pub cards: Vec<u32>,
    pub edges: Vec<TraceEdge>,
    /// (the path hashes of the edges being searched)
    path: Vec<u64>,
}

impl Trace {
    pub fn new(viewer: u32, cards: Vec<u32>) -> Trace {
        Trace { viewer, cards, edges: Vec::new(), path: Vec::new() }
    }
}

pub struct TraceEdge {
    /// the iteration's depth
    pub iteration: i32,
    /// a hash of the moves from the root (each move as `move_key` and the response)
    pub path: u64,
    pub height: i32,
    /// who made the move
    pub mover: u32,
    /// of `Trace::cards`, those not hidden any more after the move
    pub revealed: u8,
    /// whether the prompt the move answered listed one of `Trace::cards` still hidden
    pub listed: bool,
    /// a hash of what the viewer sees after the move (its prompt included if the next decision is
    /// its own)
    pub visible: u64,
    /// the move's value for the mover (None: aborted)
    pub value: Option<f64>,
}

/// Whether a card is hidden from the other player (not face-up, not in a graveyard or among
/// materials).
pub fn is_hidden(d: &Duel, cid: u32) -> bool {
    d.card_by_id(cid).is_some_and(|c| {
        let x = d.card(c);
        x.current.location as u32 & (LOCATION_GRAVE | LOCATION_OVERLAY) == 0 && x.overlay_target.is_none() && x.is_position(POS_FACEUP) == 0
    })
}

/// Whether a prompt lists a hidden card of `cards` (its card entries by place).
fn lists_hidden(d: &Duel, m: &Msg, cards: &[u32]) -> bool {
    fn walk(d: &Duel, v: &serde_json::Value, cards: &[u32]) -> bool {
        match v {
            serde_json::Value::Object(o) => {
                if let (Some(con), Some(loc), Some(seq)) = (o.get("con").and_then(|x| x.as_u64()), o.get("loc").and_then(|x| x.as_u64()), o.get("seq").and_then(|x| x.as_u64()))
                    && let Some(c) = d.get_field_card(con as u32, loc as u32 & 0xffff, seq as u32)
                    && cards.contains(&d.card(c).cardid)
                    && is_hidden(d, d.card(c).cardid)
                {
                    return true;
                }
                o.values().any(|x| walk(d, x, cards))
            }
            serde_json::Value::Array(a) => a.iter().any(|x| walk(d, x, cards)),
            _ => false,
        }
    }
    serde_json::to_value(m).is_ok_and(|v| walk(d, &v, cards))
}

/// (`AlphaBeta::killers`: no move)
const NO_KILLER: usize = usize::MAX;

/// (`NodeStats::cuts`: no cutoff, all the children searched)
pub const NO_CUT: u8 = 255;

/// The search's tables, kept from one decision to the next (an agent holds one).
pub struct AlphaBeta {
    hash_table: TranspositionTable,
    history_table: HistoryTable,
    /// by height: the history slots of the last two moves that caused a cutoff there (`killers`;
    /// NO_KILLER if none), cleared for each decision
    killers: Vec<[usize; 2]>,
    /// (`online`: the evaluation the search learns on, a copy of the trained one for each decision)
    online_eval: Option<LearnedEval>,
    /// (`online`) over the nodes it learned from: [count, sum of the learning evaluation's squared
    /// error before the step, the trained one's] (in points; 0 on a bound's side)
    pub online_stats: [f64; 3],
    /// (`nullmove`: the null move searches, and those that cut the node off)
    pub null_tries: u64,
    pub null_cuts: u64,
    /// (with $YGOAI_NULL_VERIFY the cut nodes are searched anyway: those below beta after all)
    pub null_wrong: u64,
    pub nodes: u64,
    /// transposition table lookups, and those that found the position
    pub tt_probes: u64,
    pub tt_hits: u64,
    /// (lines the core could not play: the search avoids them)
    pub broken: u64,
    /// (if set, statistics of the expanded nodes, see `NodeStats`)
    pub stats: Option<NodeStats>,
    /// (if set, the edges searched, see `Trace`)
    pub trace: Option<Trace>,
    /// the root value of the last search's last completed iteration (NaN if none completed)
    pub last_value: f64,
    /// by depth: [sum of ln(nodes of the iteration / nodes of the one before), weight], kept
    /// across searches (`predict`)
    growth: Vec<[f64; 2]>,
}

impl Default for AlphaBeta {
    fn default() -> Self {
        AlphaBeta {
            hash_table: TranspositionTable::default(),
            history_table: HistoryTable::default(),
            killers: vec![],
            online_eval: None,
            online_stats: [0.0; 3],
            null_tries: 0,
            null_cuts: 0,
            null_wrong: 0,
            nodes: 0,
            tt_probes: 0,
            tt_hits: 0,
            broken: 0,
            stats: None,
            trace: None,
            last_value: f64::NAN,
            growth: vec![[GROWTH_PRIOR.ln() * GROWTH_PRIOR_WEIGHT, GROWTH_PRIOR_WEIGHT]; MAX_DEPTH as usize + 2],
        }
    }
}

/// What a search is given besides the duel: time (`budget` in seconds for this decision, None
/// for a fixed depth; `elapsed` already spent on it; the clock's remaining seconds), the select
/// hint before a card selection and the seeds of the rules' two players. `max_nodes`: a fixed
/// number of nodes instead (deepening until it is used up; deterministic).
#[derive(Default)]
pub struct SearchParams {
    pub budget: Option<f64>,
    pub elapsed: f64,
    pub clock_remaining: f64,
    pub hint: Option<u64>,
    pub seeds: [u32; 2],
    pub max_nodes: Option<u64>,
}

impl AlphaBeta {
    /// The expected node growth of the iteration at `depth` over the one before.
    pub fn expected_growth(&self, depth: i32) -> f64 {
        let [s, w] = self.growth[(depth as usize).min(self.growth.len() - 1)];
        (s / w).exp()
    }

    fn observe_growth(&mut self, depth: i32, nodes: u64, prev_nodes: u64) {
        let last = self.growth.len() - 1;
        let g = &mut self.growth[(depth as usize).min(last)];
        g[0] += (nodes.max(1) as f64 / prev_nodes.max(1) as f64).ln();
        g[1] += 1.0;
        if g[1] > GROWTH_MAX_WEIGHT {
            let k = GROWTH_MAX_WEIGHT / g[1];
            (g[0], g[1]) = (g[0] * k, g[1] * k);
        }
    }

    /// The tables for a new decision.
    pub fn prepare(&mut self) {
        self.hash_table.new_generation();
        self.history_table.decay();
    }

    /// The best of `root_moves` (responses to the duel's pending prompt `root_prompt`) and the
    /// depth completed (after `prepare`). The duel is not changed.
    pub fn search(&mut self, duel: &Duel, root_prompt: &Msg, root_moves: &[Response], p: &SearchParams) -> (usize, i32) {
        self.search_resampling(duel, root_prompt, root_moves, p, None)
    }

    /// `search`, with `resample` changing the guessed hidden information of the root's copy for
    /// each iteration after the first (`resample`).
    pub fn search_resampling(&mut self, duel: &Duel, root_prompt: &Msg, root_moves: &[Response], p: &SearchParams, resample: Option<&mut dyn FnMut(&mut Duel)>) -> (usize, i32) {
        self.search_decks(std::slice::from_ref(duel), root_prompt, root_moves, p, resample)
    }

    /// `search` with the opponent choosing among `decks` (the root's duel under different guesses
    /// of its hidden cards) right after the root's move, as a pseudo-move (`deckpick`): a root
    /// move's value is its worst over the decks.
    pub fn search_decks(&mut self, decks: &[Duel], root_prompt: &Msg, root_moves: &[Response], p: &SearchParams, resample: Option<&mut dyn FnMut(&mut Duel)>) -> (usize, i32) {
        let duel = &decks[0];
        let start_time = Instant::now() - std::time::Duration::from_secs_f64(p.elapsed);
        let mut policies = [Heuristic::new(p.seeds[0] as i128, 0), Heuristic::new(p.seeds[1] as i128, 1)];
        let us = root_prompt.player().unwrap_or(0) as usize;
        policies[us].hint = p.hint;
        let turn_player = duel.ext_get_info()[1];
        let stop_time = p.budget.map(|b| start_time + std::time::Duration::from_secs_f64(hard_limit(b, p.clock_remaining)));
        let max_nodes = p.max_nodes.map(|n| self.nodes + n);
        let features = Features::from_env();
        self.killers.clear();
        self.online_eval = if features.online { LearnedEval::from_env().cloned() } else { None };
        // (`passrisk`: the opponent's cards hidden at the root)
        let unknown = if features.passrisk {
            duel.ext_get_all_cards().iter().filter(|e| e[0] != us as u32 && e[5] != 0 && is_hidden(duel, e[5])).map(|e| e[5]).collect()
        } else {
            vec![]
        };
        let mut state = SearchState { tables: self, policies, stop_time, max_nodes, features, root_depth: 0, deck_key: 0, root_player: us as u32, unknown, line_risk: false, in_null: false };
        let card_order = state.rules_card_order(root_prompt);
        let preference = ordered(root_prompt, turn_player, card_order.as_deref());
        let mut move_order: Vec<usize> = (0..root_moves.len()).collect();
        move_order.sort_by_key(|&k| preference.iter().position(|c| *c == root_moves[k]).unwrap_or(preference.len()));
        if matches!(root_prompt, Msg::SelectCard { min: 1, max: 1, .. }) {
            move_order.truncate(ROOT_CARD_WIDTH);
        } else if is_inner(root_prompt) || matches!(root_prompt, Msg::SelectCard { .. }) {
            // (a decision inside a move: the rules' first choices only, as below the root)
            let known = root_moves.iter().filter(|c| preference.contains(c)).count().max(1);
            move_order.truncate(known);
        }
        let fixed_depth = p.budget.is_none() && p.max_nodes.is_none();
        state.iterative_deepening(decks, root_prompt, root_moves, move_order, start_time, p.budget, fixed_depth, resample)
    }
}

/// One search: the tables, the rules' players, the deadline and the node limit.
struct SearchState<'a> {
    tables: &'a mut AlphaBeta,
    /// (the rules' players: the decisions the search leaves to the rules, and the move order)
    policies: [Heuristic; 2],
    stop_time: Option<Instant>,
    /// (in the count of `tables.nodes`)
    max_nodes: Option<u64>,
    features: Features,
    /// (the iteration's depth)
    root_depth: i32,
    /// (`deckpick`: the deck searched, in the TT key: the key leaves out the codes of hidden cards)
    deck_key: u64,
    /// (`passrisk`) the root's player, the opponent's cards hidden from it at the root, and
    /// whether the line searched has the opponent pass on its chain link while holding one of
    /// them in the hand or Set (in the TT key)
    root_player: u32,
    unknown: Vec<u32>,
    line_risk: bool,
    /// (`nullmove`: inside a null move's search, where no other is tried)
    in_null: bool,
}

/// Where `do_move` arrived.
enum NewPosition {
    /// the game ended with a winner
    Win(u32),
    /// the game ended without one
    End,
    /// a searched decision (with more than one move)
    Decision(Msg),
    /// MAX_AUTO_STEPS decisions of the rules in a row; the last one's prompt
    StepLimit(Msg),
}

impl SearchState<'_> {
    /// The static evaluation for `us`, less `passrisk_lp()` for the root's player on a line with
    /// the risk (`passrisk`).
    fn evaluate_at(&self, duel: &mut Duel, us: u32, prompt: Option<&Msg>) -> f64 {
        let v = match &self.tables.online_eval {
            Some(e) => e.value(duel, us, prompt),
            None => evaluate_at(duel, us, prompt),
        };
        self.with_risk(v, us)
    }

    fn evaluate(&self, duel: &mut Duel, us: u32) -> f64 {
        self.evaluate_at(duel, us, None)
    }

    fn with_risk(&self, v: f64, us: u32) -> f64 {
        if !self.line_risk {
            v
        } else if us == self.root_player {
            v - passrisk_lp()
        } else {
            v + passrisk_lp()
        }
    }

    /// (`passrisk`) Whether answering `prompt` with `response` is the opponent passing on the root
    /// player's chain link while holding a card the root's player does not know (hidden in the hand
    /// or Set): a line that may be optimistic about the opponent's hidden cards.
    fn passes_unknown(&self, duel: &Duel, prompt: &Msg, response: &Response) -> bool {
        if !self.features.passrisk || response.index != Some(-1) {
            return false;
        }
        let Msg::SelectChain { player, .. } = prompt else { return false };
        if *player == self.root_player || duel.core.current_chain.last().is_none_or(|l| l.triggering_controler as u32 != self.root_player) {
            return false;
        }
        self.unknown.iter().any(|&cid| {
            is_hidden(duel, cid) && duel.card_by_id(cid).is_some_and(|c| duel.card(c).current.location as u32 & (LOCATION_HAND | LOCATION_SZONE) != 0)
        })
    }

    fn should_stop(&self) -> bool {
        self.stop_time.is_some_and(|d| Instant::now() > d) || self.max_nodes.is_some_and(|n| self.tables.nodes >= n)
    }

    /// For a card selection: the rules' order of its cards (by the select hint the player got,
    /// which belongs to this selection only).
    fn rules_card_order(&mut self, m: &Msg) -> Option<Vec<usize>> {
        let (cards, player) = match m {
            Msg::SelectCard { cards, player, .. } => (cards, player),
            Msg::SelectUnselectCard { select_cards, player, .. } if is_inner(m) => (select_cards, player),
            _ => return None,
        };
        let policy = &mut self.policies[*player as usize];
        let order = policy.card_order(cards);
        policy.hint = None;
        Some(order)
    }

    #[allow(clippy::too_many_arguments)]
    fn iterative_deepening(
        &mut self,
        decks: &[Duel],
        prompt: &Msg,
        moves: &[Response],
        mut move_order: Vec<usize>,
        start_time: Instant,
        budget: Option<f64>,
        fixed_depth: bool,
        mut resample: Option<&mut dyn FnMut(&mut Duel)>,
    ) -> (usize, i32) {
        let (mut best_move, mut completed_depth) = (move_order[0], 0);
        self.tables.last_value = f64::NAN;
        // (the last completed iteration's nodes and seconds)
        let (mut last_nodes, mut last_secs) = (0u64, 0.0);
        let max_depth = if fixed_depth { DEPTH_NO_CLOCK } else { MAX_DEPTH };
        // (the root's moves are searched on copies of it: it stays as it is)
        let mut root_duels: Vec<Duel> = decks.to_vec();
        // (`deckpick`: per root move, the decks the worst first)
        let mut deck_orders: Vec<Vec<usize>> = vec![(0..decks.len()).collect(); moves.len()];
        for depth in 1..=max_depth {
            self.root_depth = depth;
            if let Some(b) = budget
                && depth > 1
            {
                let t = start_time.elapsed().as_secs_f64();
                let late = if self.features.predict { t + last_secs * self.tables.expected_growth(depth) > b } else { t > NEXT_ITERATION_SHARE * b };
                if late {
                    break;
                }
            }
            if depth > 1
                && let Some(f) = resample.as_mut()
            {
                root_duels[0] = decks[0].clone();
                f(&mut root_duels[0]);
            }
            let (iteration_nodes, iteration_start) = (self.tables.nodes, start_time.elapsed().as_secs_f64());
            let mut values: Vec<Option<f64>> = vec![None; moves.len()];
            let (mut alpha, mut iteration_best) = (f64::NEG_INFINITY, None);
            let mut aborted = false;
            for (move_counter, &m) in move_order.iter().enumerate() {
                let null_window = self.features.pvs && move_counter > 0 && alpha.is_finite();
                let mut result = if null_window {
                    // (PVS: does it beat the best so far? only then its value)
                    self.search_root_move(&mut root_duels, &mut deck_orders[m], prompt, &moves[m], alpha, alpha + NULL_WINDOW, depth - 1)
                } else {
                    self.search_root_move(&mut root_duels, &mut deck_orders[m], prompt, &moves[m], alpha, f64::INFINITY, depth - 1)
                };
                if null_window && matches!(result, Ok(v) if v > alpha) {
                    result = self.search_root_move(&mut root_duels, &mut deck_orders[m], prompt, &moves[m], alpha, f64::INFINITY, depth - 1);
                }
                match result {
                    Ok(value) => {
                        values[m] = Some(value);
                        if value > alpha {
                            (alpha, iteration_best) = (value, Some(m));
                        }
                    }
                    Err(_) => {
                        aborted = true;
                        break;
                    }
                }
            }
            if aborted {
                // (an unfinished iteration: its best counts if the previous best, searched first,
                // was searched completely and another one did better)
                if values[move_order[0]].is_some()
                    && let Some(b) = iteration_best
                {
                    best_move = b;
                }
                break;
            }
            let Some(b) = iteration_best else { break };
            let iteration_secs = start_time.elapsed().as_secs_f64() - iteration_start;
            if depth > 1 {
                self.tables.observe_growth(depth, self.tables.nodes - iteration_nodes, last_nodes);
            }
            if std::env::var_os("YGOAI_DEBUG_ITERS").is_some() {
                eprintln!(
                    "iter {depth} {} {iteration_secs:.6} {:.6} {} {:.3}",
                    self.tables.nodes - iteration_nodes,
                    start_time.elapsed().as_secs_f64(),
                    budget.unwrap_or(0.0),
                    self.tables.expected_growth(depth + 1)
                );
            }
            (last_nodes, last_secs) = (self.tables.nodes - iteration_nodes, iteration_secs);
            (best_move, completed_depth) = (b, depth);
            self.tables.last_value = values[b].unwrap();
            // (the next iteration: the best first; cut-off moves keep their bound's order)
            move_order.sort_by(|&a, &b| values[b].unwrap().partial_cmp(&values[a].unwrap()).unwrap_or(std::cmp::Ordering::Equal));
            if values[best_move].unwrap().abs() >= WIN / 2.0 {
                break; // (a forced win or loss: deeper changes nothing)
            }
        }
        (best_move, completed_depth)
    }

    /// A root move's value: its worst over the decks (one deck: its value), the deck that was worst
    /// for it searched first; a deck that takes it to `alpha` or below refutes it (the others
    /// are not searched). The decks after the first are searched `deckpick_reduction()` plies less
    /// deep, and again at the full depth if that finds them worse.
    #[allow(clippy::too_many_arguments)]
    fn search_root_move(&mut self, duels: &mut [Duel], order: &mut [usize], prompt: &Msg, mv: &Response, alpha: f64, beta: f64, depth: i32) -> Result<f64, Stop> {
        let mut value = f64::INFINITY;
        for i in 0..order.len() {
            let k = order[i];
            self.deck_key = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let reduced = if i > 0 { (depth - deckpick_reduction()).max(0) } else { depth };
            let mut v = self.search_move(&mut duels[k], prompt, mv, alpha, beta.min(value), reduced, 1, false);
            if reduced < depth && matches!(v, Ok(x) if x < value) {
                v = self.search_move(&mut duels[k], prompt, mv, alpha, beta.min(value), depth, 1, false);
            }
            self.deck_key = 0;
            let v = v?;
            if v < value {
                value = v;
                order[..=i].rotate_right(1);
            }
            if value <= alpha {
                break;
            }
        }
        Ok(value)
    }

    /// The value of the decision `prompt` for its player (negamax), searched `depth` plies deep;
    /// `height` plies below the root, reached by the move with history slot `previous`.
    #[allow(clippy::too_many_arguments)]
    fn search(&mut self, duel: &mut Duel, prompt: &Msg, mut alpha: f64, mut beta: f64, depth: i32, height: i32, previous: usize) -> Result<f64, Stop> {
        let us = prompt.player().unwrap_or(0);
        let f = self.features;
        if depth <= 0 {
            let quiet = match prompt {
                Msg::SelectChain { .. } => false,
                Msg::SelectBattlecmd { .. } => !candidates(prompt).iter().any(|c| c.kind.as_deref() == Some("attack")),
                _ => true,
            };
            if f.qsearch && depth > -MAX_QUIESCE_PLY && !quiet {
                return self.quiesce(duel, prompt, alpha, beta, depth, height);
            }
            return Ok(self.evaluate_at(duel, us, Some(prompt)));
        }

        // (moves are indices into the rules' order: the hash table keeps one, the history reorders them)
        let card_order = self.rules_card_order(prompt);
        let mut moves = ordered(prompt, duel.ext_get_info()[1], card_order.as_deref());
        if matches!(prompt, Msg::SelectCard { min: 1, max: 1, .. }) {
            moves.truncate(CARD_WIDTH);
        }

        // transposition table: update alpha, beta or return immediately
        let key = hash_key(duel, prompt) ^ self.deck_key ^ if self.line_risk { 0xD6E8_FEB8_6659_FD93 } else { 0 };
        let hash_result = self.tables.hash_table.get(key);
        let (mut tt_move, mut tt_state) = (None, 0u8);
        let nodes_before = self.tables.nodes;
        self.tables.tt_probes += 1;
        if let Some(h) = hash_result {
            self.tables.tt_hits += 1;
            tt_state = if h.best_move >= 0 { 2 + h.node_type as u8 } else { 1 };
            if h.depth as i32 >= depth {
                match h.node_type {
                    NodeType::PvNode => return Ok(h.value),
                    NodeType::CutNode => alpha = alpha.max(h.value),
                    NodeType::AllNode => beta = beta.min(h.value),
                }
                if alpha >= beta {
                    return Ok(h.value);
                }
            }
            if h.best_move >= 0 && (h.best_move as usize) < moves.len() {
                tt_move = Some(h.best_move as usize);
            }
        }

        // reverse futility pruning: so far ahead that the moves would not bring it below beta
        if f.rfp && beta - alpha <= NULL_WINDOW && depth <= RFP_MAX_DEPTH && beta.abs() < WIN / 2.0 {
            let static_eval = self.evaluate_at(duel, us, Some(prompt));
            if static_eval - RFP_MARGIN_MULTIPLIER * depth as f64 >= beta {
                return Ok(static_eval);
            }
        }
        // (`rfpall`: at any node below the root, by futility's margins)
        let (rfp_margin, rfp_per_ply, rfp_max_depth) = rfp_params();
        if f.rfpall && height > 0 && depth <= rfp_max_depth && beta.is_finite() && beta.abs() < WIN / 2.0 {
            let static_eval = self.evaluate_at(duel, us, Some(prompt));
            if static_eval - (rfp_margin + rfp_per_ply * depth as f64) >= beta {
                return Ok(static_eval);
            }
        }
        // null move pruning (`nullmove`): ending the phase (the turn) instead of acting, searched
        // NULL_R plies less deep with a null window at beta; still at least beta, acting would be
        // too (in a main or battle phase doing something is rarely worse than doing nothing)
        let (null_r, null_r_div, null_min_depth) = nullmove_params();
        let mut null_cut = false;
        if f.nullmove
            && !self.in_null
            && height > 0
            && depth >= null_min_depth
            && beta.is_finite()
            && beta.abs() < WIN / 2.0
            && matches!(prompt, Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. })
            && let Some(pass) = moves.iter().find(|c| c.kind.as_deref() == Some("to_ep")).cloned()
            && self.evaluate_at(duel, us, Some(prompt)) >= beta
        {
            let r = null_r + depth / null_r_div;
            self.in_null = true;
            let value = self.search_move(duel, prompt, &pass, beta - NULL_WINDOW, beta, (depth - 1 - r).max(0), height + 1, false);
            self.in_null = false;
            let value = value?;
            self.tables.null_tries += 1;
            if value >= beta {
                self.tables.null_cuts += 1;
                if !null_verify() {
                    return Ok(if value >= WIN / 2.0 { beta } else { value });
                }
                null_cut = true;
            }
        }

        let (futility_margin, futility_per_ply, futility_max_depth) = futility_params();
        let static_eval = if (f.futility && depth <= futility_max_depth) || (self.tables.stats.is_some() && depth <= 4) { Some(self.evaluate_at(duel, us, Some(prompt))) } else { None };
        // (futility pruning: the moves after the first whose static eval plus the margin cannot raise alpha)
        let futility_value = static_eval.filter(|_| f.futility && depth <= futility_max_depth).map(|s| s + futility_margin + futility_per_ply * depth as f64);

        // (`online`: the mover's features, to learn from the node's value)
        let (online_lr, online_lr_exp, online_min_depth, online_max_err) = online_params();
        let online_x = if f.online && depth >= online_min_depth { self.tables.online_eval.as_ref().and_then(|e| e.mover_x(duel, us, Some(prompt))) } else { None };

        // move ordering: the TT move first, then (`killers`) the killer moves, then by history
        // (the rules' order among equals)
        let history_keys: Vec<HistoryKey> = moves
            .iter()
            .map(|m| {
                let key = move_key(prompt, m);
                HistoryKey { key, counter_key: f.conthist.then(|| counter_key(previous, key)) }
            })
            .collect();
        let killers = if f.killers { self.tables.killers.get(height as usize).copied().unwrap_or([NO_KILLER; 2]) } else { [NO_KILLER; 2] };
        let killer_rank = |k: usize| killers.iter().position(|&x| x == history_keys[k].key).unwrap_or(2);
        let mut move_order: Vec<usize> = (0..moves.len()).collect();
        move_order.sort_by(|&a, &b| {
            let history = &self.tables.history_table;
            ((Some(a) != tt_move), killer_rank(a), -history.get(history_keys[a]))
                .partial_cmp(&((Some(b) != tt_move), killer_rank(b), -history.get(history_keys[b])))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        move_order.truncate(MAX_CHILDREN);

        let mut node_type = NodeType::AllNode;
        let (mut best_value, mut best_move) = (f64::NEG_INFINITY, move_order[0]);
        let (mut move_counter, mut cut_at) = (0, NO_CUT);
        for (i, &m) in move_order.iter().enumerate() {
            move_counter = i + 1;
            let mv = &moves[m];

            // futility pruning: a later move that cannot raise alpha by the static eval and the
            // margin; that bound counts as its value
            if let Some(v) = futility_value
                && i > 0
                && v <= alpha
                && alpha.abs() < WIN / 2.0
            {
                if v > best_value {
                    (best_value, best_move) = (v, m);
                }
                continue;
            }

            // (the node's last search may use its position: nothing needs it afterwards)
            let last = i == move_order.len() - 1;
            // extension (`narrowext`): the TT move of a node with only one or two moves, a ply
            // deeper (not past twice the iteration's depth, so that a line of such nodes ends)
            let extend = f.narrowext && i == 0 && tt_move == Some(m) && move_order.len() <= NARROW_EXT_MAX_CHILDREN && height < 2 * self.root_depth;
            let new_depth = depth - 1 + extend as i32;
            let mut value = None;

            // late move reduction: a late move, unlikely the best by the move ordering, one ply
            // less; the full depth after all if it raises alpha
            if f.lmr && depth >= LMR_MIN_DEPTH && i >= LMR_MIN_MOVE_COUNTER {
                let r = if f.lmrlog { lmr_reduction(depth, i) } else { 1 };
                // (`lmrzw`: with a null window, as it only has to show that the move does not raise alpha)
                let reduced_beta = if f.lmrzw && alpha.is_finite() { alpha + NULL_WINDOW } else { beta };
                let reduced = self.search_move(duel, prompt, mv, alpha, reduced_beta, (new_depth - r).max(0), height + 1, false)?;
                if reduced <= alpha {
                    value = Some(reduced);
                }
                if let Some(st) = &mut self.tables.stats {
                    let e = st.lmr.entry((prompt_label(prompt), i.min(15) as u8)).or_default();
                    e[0] += 1;
                    e[1] += (reduced > alpha) as u64;
                }
            }

            let value = match value {
                Some(v) => v,
                None if f.pvs && i > 0 && beta - alpha > NULL_WINDOW && alpha.is_finite() => {
                    // PVS: a later move first only has to show that it does not raise alpha;
                    // if it does, its value with the full window
                    let v = self.search_move(duel, prompt, mv, alpha, alpha + NULL_WINDOW, new_depth, height + 1, false)?;
                    if alpha < v && v < beta { self.search_move(duel, prompt, mv, alpha, beta, new_depth, height + 1, last)? } else { v }
                }
                None => self.search_move(duel, prompt, mv, alpha, beta, new_depth, height + 1, last)?,
            };

            if let Some(s) = static_eval
                && self.tables.stats.is_some()
                && i > 0
                && alpha.is_finite()
                && let Some(st) = &mut self.tables.stats
            {
                st.gaps.push((move_kind(prompt, mv), depth as i8, (alpha - s) as f32, value > alpha));
            }
            if value > best_value {
                (best_value, best_move) = (value, m);
            }
            if value >= beta {
                node_type = NodeType::CutNode;
                // (history: this move up, the ones tried before it down)
                let tried: Vec<HistoryKey> = move_order[..i].iter().map(|&k| history_keys[k]).collect();
                self.tables.history_table.update(history_keys[m], &tried, depth);
                if f.killers {
                    let h = height as usize;
                    if self.tables.killers.len() <= h {
                        self.tables.killers.resize(h + 1, [NO_KILLER; 2]);
                    }
                    let k = &mut self.tables.killers[h];
                    if k[0] != history_keys[m].key {
                        *k = [history_keys[m].key, k[0]];
                    }
                }
                cut_at = i as u8;
                break;
            }
            if value > alpha {
                node_type = NodeType::PvNode;
                alpha = value;
            }
        }

        if let Some(st) = &mut self.tables.stats {
            let label = prompt_label(prompt);
            let e = st.widths.entry((label, listed(prompt, moves.len()) as u16, move_order.len() as u16)).or_default();
            e[0] += 1;
            e[1] += move_counter as u64;
            *st.cuts.entry((label, tt_state, move_order.len() as u8, cut_at)).or_default() += 1;
            let e = st.subtree.entry((label, depth.clamp(-1, 60) as i8)).or_default();
            e[0] += 1;
            e[1] += self.tables.nodes - nodes_before + 1;
        }
        if null_cut && best_value < beta {
            self.tables.null_wrong += 1;
        }
        // (`online`: a step toward the node's value, a bound unless it is exact)
        if let Some(x) = online_x
            && best_value.abs() < WIN / 2.0
            && let Some(e) = &mut self.tables.online_eval
        {
            let bound = match node_type {
                NodeType::PvNode => 0,
                NodeType::CutNode => 1,
                NodeType::AllNode => -1,
            };
            let target = best_value / e.lp_factor();
            let err = e.learn(&x, target, bound, (online_lr * (depth as f64).powf(online_lr_exp)).min(1.0), online_max_err);
            let trained = LearnedEval::from_env().map_or(0.0, |t| {
                let e0 = t.mover_output(&x) - target;
                if (bound > 0 && e0 >= 0.0) || (bound < 0 && e0 <= 0.0) { 0.0 } else { e0 }
            });
            let st = &mut self.tables.online_stats;
            st[0] += 1.0;
            st[1] += err * err;
            st[2] += trained * trained;
        }
        if best_value.abs() < WIN / 2.0 {
            // (not a win or loss: those depend on the height)
            self.tables.hash_table.add(key, node_type, best_value, depth, best_move);
        }
        Ok(best_value)
    }

    /// Quiescence: past the nominal depth, battle and chain decisions go on (attacks and
    /// responses change the position the most, like captures in chess). In a Battle Phase the
    /// player may stand pat (the static value: it need not attack), else only its attacks are
    /// searched; in a chain window every answer, passing included.
    fn quiesce(&mut self, duel: &mut Duel, prompt: &Msg, mut alpha: f64, beta: f64, depth: i32, height: i32) -> Result<f64, Stop> {
        let mut moves = ordered(prompt, duel.ext_get_info()[1], None);
        let mut best_value = f64::NEG_INFINITY;
        if matches!(prompt, Msg::SelectBattlecmd { .. }) {
            let stand_pat = self.evaluate_at(duel, prompt.player().unwrap_or(0), Some(prompt));
            best_value = stand_pat;
            if stand_pat >= beta {
                return Ok(stand_pat);
            }
            alpha = alpha.max(stand_pat);
            moves.retain(|c| c.kind.as_deref() == Some("attack"));
        }
        moves.truncate(MAX_CHILDREN);
        for (i, mv) in moves.iter().enumerate() {
            let value = self.search_move(duel, prompt, mv, alpha, beta, depth - 1, height + 1, i == moves.len() - 1)?;
            best_value = best_value.max(value);
            if value >= beta {
                break;
            }
            alpha = alpha.max(value);
        }
        Ok(best_value)
    }

    /// The value of the move `mv` at the decision `prompt` for the player who makes it: the move
    /// made (`do_move`) and the new position searched `depth` plies deep at `height`, its value
    /// and window negated only if the other player decides there.
    ///
    /// With `in_place` on `duel` itself (the node's last search: nothing needs the position
    /// afterwards), else on a copy, after which the rules' players' activation caps and select
    /// hint are this node's again, and a broken line (one the core cannot play) is a loss for the
    /// mover, who avoids it (in place it is the caller's copy's loss).
    #[allow(clippy::too_many_arguments)]
    fn search_move(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, alpha: f64, beta: f64, depth: i32, height: i32, in_place: bool) -> Result<f64, Stop> {
        if self.should_stop() {
            return Err(Stop::Abort);
        }
        self.tables.nodes += 1;
        if in_place {
            return self.make_and_search(duel, prompt, mv, alpha, beta, depth, height);
        }
        let mut new_duel = duel.clone();
        let saved: Vec<_> = self.policies.iter().map(|p| (p.activations.clone(), p.hint)).collect();
        let result = self.make_and_search(&mut new_duel, prompt, mv, alpha, beta, depth, height);
        for (p, (activations, hint)) in self.policies.iter_mut().zip(saved) {
            (p.activations, p.hint) = (activations, hint);
        }
        match result {
            Err(Stop::Broken) if !self.should_stop() => {
                self.tables.broken += 1;
                if std::env::var_os("YGOAI_DEBUG_BROKEN").is_some() {
                    eprintln!("broken: {} {mv:?}", prompt.name());
                }
                Ok(-WIN)
            }
            Err(_) => Err(Stop::Abort),
            Ok(v) => Ok(v),
        }
    }

    /// (`search_move` on the given duel)
    #[allow(clippy::too_many_arguments)]
    fn make_and_search(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, alpha: f64, beta: f64, depth: i32, height: i32) -> Result<f64, Stop> {
        if self.tables.trace.is_none() {
            return self.make_and_search_untraced(duel, prompt, mv, alpha, beta, depth, height);
        }
        use std::hash::{Hash, Hasher};
        let (path, listed) = {
            let t = self.tables.trace.as_mut().unwrap();
            let mut h = rustc_hash::FxHasher::default();
            (t.path.last().copied().unwrap_or(0), move_key(prompt, mv), serde_json::to_string(mv).unwrap_or_default()).hash(&mut h);
            let path = h.finish();
            t.path.push(path);
            (path, lists_hidden(duel, prompt, &t.cards))
        };
        let result = self.make_and_search_untraced(duel, prompt, mv, alpha, beta, depth, height);
        let t = self.tables.trace.as_mut().unwrap();
        t.path.pop();
        let revealed = t.cards.iter().filter(|&&c| !is_hidden(duel, c)).count() as u8;
        let mut h = rustc_hash::FxHasher::default();
        serde_json::to_string(&crate::state::viewer_view(duel, t.viewer)).unwrap_or_default().hash(&mut h);
        let visible = h.finish();
        let edge = TraceEdge { iteration: self.root_depth, path, height, mover: prompt.player().unwrap_or(0), revealed, listed, visible, value: result.as_ref().ok().copied() };
        self.tables.trace.as_mut().unwrap().edges.push(edge);
        result
    }

    /// (`make_and_search` without the trace)
    #[allow(clippy::too_many_arguments)]
    fn make_and_search_untraced(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, alpha: f64, beta: f64, depth: i32, height: i32) -> Result<f64, Stop> {
        let saved_risk = self.line_risk;
        let result = self.make_and_search_risk(duel, prompt, mv, alpha, beta, depth, height);
        self.line_risk = saved_risk;
        result
    }

    /// (`make_and_search_untraced`, `line_risk` restored by the caller)
    #[allow(clippy::too_many_arguments)]
    fn make_and_search_risk(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, alpha: f64, beta: f64, depth: i32, height: i32) -> Result<f64, Stop> {
        if self.passes_unknown(duel, prompt, mv) {
            self.line_risk = true;
        }
        let us = prompt.player().unwrap_or(0);
        let previous = if self.features.conthist { move_key(prompt, mv) } else { 0 };
        Ok(match self.do_move(duel, prompt, mv, depth)? {
            NewPosition::Win(winner) => {
                if winner == us {
                    WIN - height as f64
                } else {
                    -WIN + height as f64
                }
            }
            NewPosition::End => self.evaluate(duel, us),
            NewPosition::StepLimit(last) => self.evaluate_at(duel, us, Some(&last)),
            NewPosition::Decision(next) => {
                if next.player().unwrap_or(0) == us {
                    self.search(duel, &next, alpha, beta, depth, height, previous)?
                } else {
                    -self.search(duel, &next, -beta, -alpha, depth, height, previous)?
                }
            }
        })
    }

    /// Makes the move: the response, then the core runs on, the rules answering the decisions
    /// the search does not make (hidden ones, the only move, and at the horizon the decisions
    /// inside a move, so that it completes first), up to the next searched decision.
    fn do_move(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, depth: i32) -> Result<NewPosition, Stop> {
        duel.ocg_set_response(&encode(prompt, mv).map_err(|_| Stop::Broken)?);
        let mut prompt = prompt.clone();
        for _ in 0..MAX_AUTO_STEPS {
            let step = run_core(duel, &prompt)?;
            for m in &step.messages {
                match m {
                    Msg::Win { player, .. } => return Ok(NewPosition::Win(*player)),
                    Msg::NewTurn { .. } | Msg::Hint { hint_type: HINT_SELECTMSG, .. } => {
                        // (the rules' players: their activation caps per turn, what a card selection is for)
                        for p in &mut self.policies {
                            p.observe_msg(m);
                        }
                    }
                    _ => {}
                }
            }
            if step.ended {
                return Ok(NewPosition::End);
            }
            prompt = step.prompt.unwrap();
            let player = prompt.player().unwrap_or(0);
            let response = if is_hidden_prompt(&prompt) {
                auto_response()
            } else if (matches!(prompt, Msg::SelectCard { .. }) || is_inner(&prompt)) && depth <= 0 {
                self.policies[player as usize].respond_to(&prompt, None).map_err(|_| Stop::Broken)?
            } else if is_searched(&prompt) {
                let moves = candidates(&prompt);
                if moves.len() > 1 {
                    return Ok(NewPosition::Decision(prompt));
                }
                match moves.into_iter().next() {
                    Some(m) => m,
                    None => self.policies[player as usize].respond_to(&prompt, None).map_err(|_| Stop::Broken)?,
                }
            } else {
                self.policies[player as usize].respond_to(&prompt, None).map_err(|_| Stop::Broken)?
            };
            if self.passes_unknown(duel, &prompt, &response) {
                self.line_risk = true;
            }
            duel.ocg_set_response(&encode(&prompt, &response).map_err(|_| Stop::Broken)?);
        }
        Ok(NewPosition::StepLimit(prompt))
    }
}

/// Runs the core to the next prompt. A line the core cannot play (an error, or a panic: rules
/// code that fails on a rare situation, as the C++ core would crash, e.g. Last Turn's forced
/// attack during an attack, which leaves the outer attack without an attacker) is broken: the
/// search avoids it, and the duel (a copy, or the last use of a position) is not used again.
fn run_core(duel: &mut Duel, prompt: &Msg) -> Result<crate::game::Step, Stop> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_until_prompt(duel, Some(prompt)))) {
        Ok(Ok(step)) => Ok(step),
        _ => Err(Stop::Broken),
    }
}

/// Prints a panic's message only the first time at its place (the search survives the core's
/// panics on lines it then avoids; one message each is enough).
pub fn panic_messages_once() {
    let seen = std::sync::Mutex::new(std::collections::HashSet::new());
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let at = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
        if seen.lock().map_or(true, |mut s| s.insert(at)) {
            hook(info);
        }
    }));
}
