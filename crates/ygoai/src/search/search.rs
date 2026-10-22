//! Alpha-beta search with iterative deepening on a (reconstructed) duel, on copies of the duel,
//! in the shape of a chess engine's (Nalwald's search.nim, Stockfish's search.cpp): `search` is
//! a negamax node with a transposition table, futility pruning and late move reductions (with a
//! null window); `iterative_deepening` runs the root.
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

use super::cardpolicy::{CardPolicy, applies};
use super::evaluate::evaluate_at;
use super::history::HistoryTable;
use super::movegen::{candidates, is_searched};
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

/// (`NodeStats::cuts`: no cutoff, all the children searched)
pub const NO_CUT: u8 = 255;

/// The search's tables, kept from one decision to the next (an agent holds one).
pub struct AlphaBeta {
    hash_table: TranspositionTable,
    history_table: HistoryTable,
    pub nodes: u64,
    /// transposition table lookups, and those that found the position
    pub tt_probes: u64,
    pub tt_hits: u64,
    /// (lines the core could not play: the search avoids them)
    pub broken: u64,
    /// (if set, statistics of the expanded nodes, see `NodeStats`)
    pub stats: Option<NodeStats>,
    /// the parameters ($YGOAI_TUNE, or set by a tuner)
    pub tune: Tune,
    /// the root value of the last search's last completed iteration (NaN if none completed)
    pub last_value: f64,
    /// by root move: its value in that iteration (exact only for the best, or with
    /// `SearchParams::exact_root`; None: not searched, as a card beyond the root's width)
    pub last_root_values: Vec<Option<f64>>,
    /// by depth: [sum of ln(nodes of the iteration / nodes of the one before), weight], kept
    /// across searches (when to start another iteration)
    growth: Vec<[f64; 2]>,
    /// the last search's greatest height (plies below the root), and the height its main line (the
    /// first move at every node) reached in the last completed iteration
    pub seldepth: i32,
    pub pv_height: i32,
    /// the main-line horizon extension's nodes: all, those with an exact value or a cutoff, and of
    /// these the ones whose best card is not the rules' first
    pub ext_nodes: [u64; 3],
    /// the last search's best root move's share of the nodes, and its time budget's scale
    pub best_fraction: f64,
    pub tm_scale: f64,
    pv_height_now: i32,
}

impl Default for AlphaBeta {
    fn default() -> Self {
        AlphaBeta {
            hash_table: TranspositionTable::default(),
            history_table: HistoryTable::default(),
            nodes: 0,
            tt_probes: 0,
            tt_hits: 0,
            broken: 0,
            stats: None,
            tune: Tune::from_env(),
            last_value: f64::NAN,
            last_root_values: Vec::new(),
            growth: vec![[GROWTH_PRIOR.ln() * GROWTH_PRIOR_WEIGHT, GROWTH_PRIOR_WEIGHT]; MAX_DEPTH as usize + 2],
            seldepth: 0,
            pv_height: 0,
            ext_nodes: [0; 3],
            best_fraction: 0.0,
            tm_scale: 1.0,
            pv_height_now: 0,
        }
    }
}

/// What a search is given besides the duel: time (`budget` in seconds for this decision, None
/// for a fixed depth; `elapsed` already spent on it; the clock's remaining seconds), the select
/// hint before a card selection and the seeds of the rules' two players. `max_nodes`: a fixed
/// number of nodes instead (deepening until it is used up; deterministic). `depth`: the last
/// iteration (at most; without a budget or node limit: DEPTH_NO_CLOCK). `exact_root`: every root
/// move searched with the full window (exact values for all of them, at a cost: references).
#[derive(Default)]
pub struct SearchParams {
    pub budget: Option<f64>,
    pub elapsed: f64,
    pub clock_remaining: f64,
    pub hint: Option<u64>,
    pub seeds: [u32; 2],
    pub max_nodes: Option<u64>,
    pub depth: Option<i32>,
    pub exact_root: bool,
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
        self.history_table.decay(self.tune.hist_decay);
    }

    /// The best of `root_moves` (responses to the duel's pending prompt `root_prompt`) and the
    /// depth completed (after `prepare`). The duel is not changed.
    pub fn search(&mut self, duel: &Duel, root_prompt: &Msg, root_moves: &[Response], p: &SearchParams) -> (usize, i32) {
        let start_time = Instant::now() - std::time::Duration::from_secs_f64(p.elapsed);
        let mut policies = [Heuristic::new(p.seeds[0] as i128, 0), Heuristic::new(p.seeds[1] as i128, 1)];
        let us = root_prompt.player().unwrap_or(0) as usize;
        policies[us].hint = p.hint;
        let turn_player = duel.ext_get_info()[1];
        let stop_time = p.budget.map(|b| start_time + std::time::Duration::from_secs_f64(hard_limit(b, p.clock_remaining)));
        let max_nodes = p.max_nodes.map(|n| self.nodes + n);
        (self.seldepth, self.pv_height) = (0, 0);
        let mut state = SearchState { tables: self, policies, stop_time, max_nodes };
        let card_order = state.card_order(duel, root_prompt);
        let preference = ordered(root_prompt, turn_player, card_order.as_deref());
        let mut move_order: Vec<usize> = (0..root_moves.len()).collect();
        move_order.sort_by_key(|&k| preference.iter().position(|c| *c == root_moves[k]).unwrap_or(preference.len()));
        if matches!(root_prompt, Msg::SelectCard { min: 1, max: 1, .. }) {
            move_order.truncate(state.tables.tune.root_card_width);
        }
        let max_depth = p.depth.unwrap_or(if p.budget.is_none() && p.max_nodes.is_none() { DEPTH_NO_CLOCK } else { MAX_DEPTH });
        state.iterative_deepening(duel, root_prompt, root_moves, move_order, start_time, p.budget, max_depth, p.exact_root)
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
    fn should_stop(&self) -> bool {
        self.stop_time.is_some_and(|d| Instant::now() > d) || self.max_nodes.is_some_and(|n| self.tables.nodes >= n)
    }

    /// For a card selection: the order of its cards, the rules' (by the select hint the player
    /// got, which belongs to this selection only), for a one-card selection the card policy's.
    fn card_order(&mut self, duel: &Duel, m: &Msg) -> Option<Vec<usize>> {
        let (cards, player) = match m {
            Msg::SelectCard { cards, player, .. } => (cards, player),
            _ => return None,
        };
        let policy = &mut self.policies[*player as usize];
        let (order, hint) = (policy.card_order(cards), policy.hint);
        policy.hint = None;
        Some(match CardPolicy::get() {
            Some(p) if applies(m) => p.order(duel, m, hint, &order),
            _ => order,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn iterative_deepening(
        &mut self,
        duel: &Duel,
        prompt: &Msg,
        moves: &[Response],
        mut move_order: Vec<usize>,
        start_time: Instant,
        budget: Option<f64>,
        max_depth: i32,
        exact_root: bool,
    ) -> (usize, i32) {
        let (mut best_move, mut completed_depth) = (move_order[0], 0);
        self.tables.last_value = f64::NAN;
        self.tables.last_root_values = vec![None; moves.len()];
        // (the last completed iteration's nodes and seconds)
        let (mut last_nodes, mut last_secs) = (0u64, 0.0);
        // (the root's moves are searched on a copy of it: it stays as it is)
        let mut root_duel = duel.clone();
        // (node-based time management, Reckless's: the nodes each root move took, over all
        // iterations; for how many iterations the best move has stayed; the budget's scale)
        let mut root_nodes = vec![0u64; moves.len()];
        let mut stable = 0u32;
        let mut scale = 1.0;
        for depth in 1..=max_depth {
            // (the next iteration only if it is expected to end within the budget, scaled)
            if let Some(b) = budget
                && depth > 1
                && start_time.elapsed().as_secs_f64() + last_secs * self.tables.expected_growth(depth) > b * scale
            {
                break;
            }
            let (iteration_nodes, iteration_start) = (self.tables.nodes, start_time.elapsed().as_secs_f64());
            self.tables.pv_height_now = 0;
            let mut values: Vec<Option<f64>> = vec![None; moves.len()];
            let (mut alpha, mut iteration_best) = (f64::NEG_INFINITY, None);
            let mut aborted = false;
            for &m in &move_order {
                let lower = if exact_root { f64::NEG_INFINITY } else { alpha };
                let pv = m == move_order[0];
                let nodes_before = self.tables.nodes;
                let result = self.search_move(&mut root_duel, prompt, &moves[m], lower, f64::INFINITY, (depth - 1) as f64, 1, false, pv);
                root_nodes[m] += self.tables.nodes - nodes_before;
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
            stable = if b == best_move && depth > 1 { stable + 1 } else { 0 };
            (best_move, completed_depth) = (b, depth);
            // (the budget's scale: less time when the best move took most of the nodes, or has
            // stayed the best for several iterations; more when it did not)
            let fraction = root_nodes[b] as f64 / root_nodes.iter().sum::<u64>().max(1) as f64;
            let by_nodes = (TM_NODES_BASE - TM_NODES_PER_FRACTION * fraction).max(TM_NODES_MIN);
            let by_stability = (TM_STABILITY_BASE - TM_STABILITY_PER_ITERATION * stable as f64).max(TM_STABILITY_MIN);
            scale = TM_SCALE_NORM * by_nodes * by_stability;
            (self.tables.best_fraction, self.tables.tm_scale) = (fraction, scale);
            self.tables.pv_height = self.tables.pv_height_now;
            self.tables.last_value = values[b].unwrap();
            self.tables.last_root_values = values.clone();
            // (the next iteration: the best first; cut-off moves keep their bound's order)
            move_order.sort_by(|&a, &b| values[b].unwrap().partial_cmp(&values[a].unwrap()).unwrap_or(std::cmp::Ordering::Equal));
            if values[best_move].unwrap().abs() >= WIN / 2.0 {
                break; // (a forced win or loss: deeper changes nothing)
            }
        }
        (best_move, completed_depth)
    }

    /// The value of the decision `prompt` for its player (negamax), searched `depth` plies deep;
    /// `height` plies below the root; `pv` on the main line (the first move at every node above).
    #[allow(clippy::too_many_arguments)]
    fn search(&mut self, duel: &mut Duel, prompt: &Msg, mut alpha: f64, mut beta: f64, depth: f64, height: i32, pv: bool) -> Result<f64, Stop> {
        let us = prompt.player().unwrap_or(0);
        if depth <= 0.0 {
            // (the main-line horizon extension: on the main line a card selection, a decision
            // inside a move, is searched one ply deep at the horizon too, so that the search
            // finishes the move, not the rules; SPRT at 10+0.1 H1, +25.4 +/- 16.4 at 890 games,
            // 2026-10-19)
            if !(extends(pv, height) && matches!(prompt, Msg::SelectCard { .. })) {
                return Ok(evaluate_at(duel, us, Some(prompt)));
            }
        }
        let extended = depth <= 0.0;
        let depth = if extended { 1.0 } else { depth };

        // (moves are indices into the rules' order: the hash table keeps one, the history reorders them)
        let card_order = self.card_order(duel, prompt);
        let mut moves = ordered(prompt, duel.ext_get_info()[1], card_order.as_deref());
        if moves.is_empty() {
            return Ok(evaluate_at(duel, us, Some(prompt)));
        }
        if matches!(prompt, Msg::SelectCard { min: 1, max: 1, .. }) {
            moves.truncate(self.tables.tune.card_width);
        }

        // transposition table: update alpha, beta or return immediately
        let key = hash_key(duel, prompt);
        let hash_result = self.tables.hash_table.get(key);
        let (mut tt_move, mut tt_state) = (None, 0u8);
        let nodes_before = self.tables.nodes;
        self.tables.tt_probes += 1;
        if let Some(h) = hash_result {
            self.tables.tt_hits += 1;
            tt_state = if h.best_move >= 0 { 2 + h.node_type as u8 } else { 1 };
            if h.depth as f64 >= depth {
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

        let t = self.tables.tune;
        let static_eval = if depth <= t.futility_max_depth || (self.tables.stats.is_some() && depth <= 4.0) { Some(evaluate_at(duel, us, Some(prompt))) } else { None };
        // (futility pruning: the moves after the first whose static eval plus the margin cannot raise alpha)
        let futility_value = static_eval.filter(|_| depth <= t.futility_max_depth).map(|s| s + t.futility_margin + t.futility_margin_per_ply * depth);

        // move ordering: the TT move first, then by history (the rules' order among equals)
        let history_keys: Vec<usize> = moves.iter().map(|m| move_key(prompt, m)).collect();
        let mut move_order: Vec<usize> = (0..moves.len()).collect();
        move_order.sort_by(|&a, &b| {
            let history = &self.tables.history_table;
            ((Some(a) != tt_move), -history.get(history_keys[a]))
                .partial_cmp(&((Some(b) != tt_move), -history.get(history_keys[b])))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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
            let new_depth = depth - 1.0;
            let mut value = None;

            // late move reduction: a late move, unlikely the best by the move ordering, searched
            // `lmr_reduction` plies less with a null window (it only has to show that it does not
            // raise alpha); the full depth and window after all if it raises alpha
            if depth >= t.lmr_min_depth && i >= t.lmr_min_move_counter {
                let reduced_beta = if alpha.is_finite() { alpha + NULL_WINDOW } else { beta };
                let reduced = self.search_move(duel, prompt, mv, alpha, reduced_beta, (new_depth - t.lmr_reduction).max(0.0), height + 1, false, false)?;
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
                None => self.search_move(duel, prompt, mv, alpha, beta, new_depth, height + 1, last, pv && i == 0)?,
            };

            if let Some(s) = static_eval
                && self.tables.stats.is_some()
                && i > 0
                && alpha.is_finite()
                && let Some(st) = &mut self.tables.stats
            {
                st.gaps.push((move_kind(prompt, mv), depth.round() as i8, (alpha - s) as f32, value > alpha));
            }
            if value > best_value {
                (best_value, best_move) = (value, m);
            }
            if value >= beta {
                node_type = NodeType::CutNode;
                // (history: this move up, the ones tried before it down)
                let tried: Vec<usize> = move_order[..i].iter().map(|&k| history_keys[k]).collect();
                self.tables.history_table.update(history_keys[m], &tried, depth, t.hist_exponent, t.hist_malus);
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
            let e = st.subtree.entry((label, depth.round().clamp(-1.0, 60.0) as i8)).or_default();
            e[0] += 1;
            e[1] += self.tables.nodes - nodes_before + 1;
        }
        if extended {
            let e = &mut self.tables.ext_nodes;
            e[0] += 1;
            if node_type != NodeType::AllNode {
                e[1] += 1;
                e[2] += (best_move != 0) as u64;
            }
        }
        if best_value.abs() < WIN / 2.0 {
            // (not a win or loss: those depend on the height)
            self.tables.hash_table.add(key, node_type, best_value, depth, best_move);
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
    /// mover, who avoids it (in place it is the caller's copy's loss). `pv`: the new position is on
    /// the main line.
    #[allow(clippy::too_many_arguments)]
    fn search_move(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, alpha: f64, beta: f64, depth: f64, height: i32, in_place: bool, pv: bool) -> Result<f64, Stop> {
        if self.should_stop() {
            return Err(Stop::Abort);
        }
        self.tables.nodes += 1;
        if in_place {
            return self.make_and_search(duel, prompt, mv, alpha, beta, depth, height, pv);
        }
        let mut new_duel = duel.clone();
        let saved: Vec<_> = self.policies.iter().map(|p| (p.activations.clone(), p.hint)).collect();
        let result = self.make_and_search(&mut new_duel, prompt, mv, alpha, beta, depth, height, pv);
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
    fn make_and_search(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, alpha: f64, beta: f64, depth: f64, height: i32, pv: bool) -> Result<f64, Stop> {
        let us = prompt.player().unwrap_or(0);
        self.tables.seldepth = self.tables.seldepth.max(height);
        if pv {
            self.tables.pv_height_now = self.tables.pv_height_now.max(height);
        }
        let position = self.do_move(duel, prompt, mv, depth, extends(pv, height))?;
        Ok(match position {
            NewPosition::Win(winner) => {
                if winner == us {
                    WIN - height as f64
                } else {
                    -WIN + height as f64
                }
            }
            NewPosition::End => evaluate_at(duel, us, None),
            NewPosition::StepLimit(last) => evaluate_at(duel, us, Some(&last)),
            NewPosition::Decision(next) => {
                if next.player().unwrap_or(0) == us {
                    self.search(duel, &next, alpha, beta, depth, height, pv)?
                } else {
                    -self.search(duel, &next, -beta, -alpha, depth, height, pv)?
                }
            }
        })
    }

    /// Makes the move: the response, then the core runs on, the rules answering the decisions
    /// the search does not make (hidden ones, the only move, and at the horizon the decisions
    /// inside a move, so that it completes first, unless `extend`: on the main line, see `search`),
    /// up to the next searched decision.
    fn do_move(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, depth: f64, extend: bool) -> Result<NewPosition, Stop> {
        let depth = if extend && depth <= 0.0 { 1.0 } else { depth };
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
            } else if applies(&prompt) && depth <= 0.0 && CardPolicy::get().is_some() {
                let first = self.card_order(duel, &prompt).unwrap()[0];
                Response { indices: Some(vec![first as i64]), ..Response::default() }
            } else if matches!(prompt, Msg::SelectCard { .. }) && depth <= 0.0 {
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
            duel.ocg_set_response(&encode(&prompt, &response).map_err(|_| Stop::Broken)?);
        }
        Ok(NewPosition::StepLimit(prompt))
    }
}

/// (the main-line horizon extension applies at `height` on the main line `pv`)
fn extends(pv: bool, height: i32) -> bool {
    pv && height < MAX_DEPTH
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
