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

use std::cmp::Ordering;
use std::time::{Duration, Instant};

use ygo_core::constants::*;
use ygo_core::duel::Duel;

use super::cardpolicy::{card_policy, card_policy_applies, selection_cards, set_policy, set_policy_applies};
use super::evaluate::evaluate_at;
use super::history::HistoryTable;
use super::movegen::{candidates, is_searched, root_candidates};
use super::movepick::{KeyContext, move_key, ordered};
use super::params::*;
use super::stats::{Extension, NodeStats, NodeSummary, TtEntry};
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

/// The node growth from one iteration to the next at a depth, learned across searches (when to
/// start another iteration): a weighted mean of its logarithm.
#[derive(Clone, Copy)]
struct Growth {
    log_sum: f64,
    weight: f64,
}

impl Growth {
    fn prior() -> Growth {
        Growth { log_sum: GROWTH_PRIOR.ln() * GROWTH_PRIOR_WEIGHT, weight: GROWTH_PRIOR_WEIGHT }
    }

    fn expected(&self) -> f64 {
        (self.log_sum / self.weight).exp()
    }

    /// An iteration of `nodes` after one of `previous_nodes`.
    fn observe(&mut self, nodes: u64, previous_nodes: u64) {
        self.log_sum += (nodes.max(1) as f64 / previous_nodes.max(1) as f64).ln();
        self.weight += 1.0;
        if self.weight > GROWTH_MAX_WEIGHT {
            let k = GROWTH_MAX_WEIGHT / self.weight;
            (self.log_sum, self.weight) = (self.log_sum * k, self.weight * k);
        }
    }
}

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
    /// (if set, statistics of the expanded nodes)
    pub stats: Option<NodeStats>,
    /// the parameters ($YGOAI_TUNE, or set by a tuner)
    pub tune: Tune,
    /// the root value of the last search's last completed iteration (NaN if none completed)
    pub last_value: f64,
    /// by root move: its value in that iteration (exact only for the best, or with
    /// `SearchParams::exact_root`; None: not searched, as a card beyond the root's width)
    pub last_root_values: Vec<Option<f64>>,
    /// by depth, kept across searches
    growth: Vec<Growth>,
    /// the last search's greatest height (plies below the root), and the height its main line (the
    /// first move at every node) reached in the last completed iteration
    pub seldepth: i32,
    pub pv_height: i32,
    pub extension: Extension,
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
            growth: vec![Growth::prior(); MAX_DEPTH as usize + 2],
            seldepth: 0,
            pv_height: 0,
            extension: Extension::default(),
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
    fn growth(&mut self, depth: i32) -> &mut Growth {
        let last = self.growth.len() - 1;
        &mut self.growth[(depth as usize).min(last)]
    }

    /// The tables for a new decision.
    pub fn prepare(&mut self) {
        self.hash_table.new_generation();
        self.history_table.decay(self.tune.hist_decay);
    }

    /// The best of `root_moves` (responses to the duel's pending prompt `root_prompt`) and the
    /// depth completed (after `prepare`). The duel is not changed.
    pub fn search(&mut self, duel: &Duel, root_prompt: &Msg, root_moves: &[Response], p: &SearchParams) -> (usize, i32) {
        let start_time = Instant::now() - Duration::from_secs_f64(p.elapsed);
        let mut policies = [Heuristic::new(p.seeds[0] as i128, 0), Heuristic::new(p.seeds[1] as i128, 1)];
        policies[root_prompt.player().unwrap_or(0) as usize].hint = p.hint;
        let stop_time = p.budget.map(|b| start_time + Duration::from_secs_f64(hard_limit(b, p.clock_remaining)));
        let max_nodes = p.max_nodes.map(|n| self.nodes + n);
        (self.seldepth, self.pv_height) = (0, 0);
        let mut state = SearchState { tables: self, policies, stop_time, max_nodes };

        // (the root's moves in the rules' order)
        let card_order = state.card_order(duel, root_prompt);
        let preference = ordered(root_prompt, duel.infos.turn_player as u32, card_order.as_ref().map(|c| c.order.as_slice()));
        let mut move_order: Vec<usize> = (0..root_moves.len()).collect();
        move_order.sort_by_key(|&k| preference.iter().position(|c| *c == root_moves[k]).unwrap_or(preference.len()));
        if matches!(root_prompt, Msg::SelectCard { min: 1, max: 1, .. }) {
            move_order.truncate(state.tables.tune.root_card_width);
        }

        let max_depth = p.depth.unwrap_or(if p.budget.is_none() && p.max_nodes.is_none() { DEPTH_NO_CLOCK } else { MAX_DEPTH });
        let root = Root { duel: duel.clone(), prompt: root_prompt, moves: root_moves, exact: p.exact_root };
        let time = TimeManager { start_time, budget: p.budget, root_nodes: vec![0; root_moves.len()], stable: 0, scale: 1.0 };
        state.iterative_deepening(root, move_order, time, max_depth)
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

/// The root of a search: a copy of the duel (its moves are searched on it), its prompt and moves;
/// with `exact` every move is searched with the full window.
struct Root<'a> {
    duel: Duel,
    prompt: &'a Msg,
    moves: &'a [Response],
    exact: bool,
}

/// One iteration at the root: each move's value (None: not searched), the best, and whether it
/// was aborted (out of time).
struct Iteration {
    values: Vec<Option<f64>>,
    best: Option<usize>,
    aborted: bool,
}

/// Node-based time management (Reckless's): an iteration starts only if it is expected to end
/// within the budget, which is scaled after each iteration by the best move's share of the root's
/// nodes and by for how many iterations it has stayed the best.
struct TimeManager {
    start_time: Instant,
    budget: Option<f64>,
    /// by root move: its nodes over all iterations
    root_nodes: Vec<u64>,
    /// the iterations the best move has stayed the best
    stable: u32,
    scale: f64,
}

impl TimeManager {
    fn elapsed(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Whether the iteration at `depth` is expected to end within the budget, by the seconds the
    /// one before took and the expected node growth.
    fn starts_iteration(&self, depth: i32, last_secs: f64, growth: f64) -> bool {
        match self.budget {
            Some(budget) if depth > 1 => self.elapsed() + last_secs * growth <= budget * self.scale,
            _ => true,
        }
    }

    /// After a completed iteration at `depth` whose best move is `best` (`previous_best` the one
    /// before): less time when it took most of the nodes, or has stayed the best for several
    /// iterations; more when it did not. Returns its share of the nodes.
    fn update(&mut self, depth: i32, best: usize, previous_best: usize) -> f64 {
        self.stable = if best == previous_best && depth > 1 { self.stable + 1 } else { 0 };
        let fraction = self.root_nodes[best] as f64 / self.root_nodes.iter().sum::<u64>().max(1) as f64;
        let by_nodes = (TM_NODES_BASE - TM_NODES_PER_FRACTION * fraction).max(TM_NODES_MIN);
        let by_stability = (TM_STABILITY_BASE - TM_STABILITY_PER_ITERATION * self.stable as f64).max(TM_STABILITY_MIN);
        self.scale = TM_SCALE_NORM * by_nodes * by_stability;
        fraction
    }
}

/// The order of a card selection's cards (indices, the preferred first) and the select hint.
struct CardOrder {
    order: Vec<usize>,
    hint: Option<u64>,
}

/// A node to search: its window, depth, height (plies below the root) and whether it is on the
/// main line (the first move at every node above).
#[derive(Clone, Copy)]
struct Node {
    alpha: f64,
    beta: f64,
    depth: f64,
    height: i32,
    pv: bool,
}

impl Node {
    /// The node for the other player (negamax): the window negated.
    fn negated(self) -> Node {
        Node { alpha: -self.beta, beta: -self.alpha, ..self }
    }

    /// Whether the main-line horizon extension applies (see `search`).
    fn extends(&self) -> bool {
        self.pv && self.height < MAX_DEPTH
    }
}

/// On which duel a move is made: a copy, or the node's own duel (its last search: nothing needs
/// the position afterwards).
#[derive(Clone, Copy, PartialEq)]
enum MakeOn {
    Copy,
    InPlace,
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

/// The futility margin at `depth`: a later move whose static eval plus it cannot raise alpha is
/// pruned.
fn futility_margin(t: &Tune, depth: f64) -> f64 {
    t.futility_margin + t.futility_margin_per_ply * depth
}

/// The depth of a late move's reduced search.
fn lmr_depth(t: &Tune, new_depth: f64) -> f64 {
    (new_depth - t.lmr_reduction).max(0.0)
}

impl SearchState<'_> {
    fn should_stop(&self) -> bool {
        self.stop_time.is_some_and(|d| Instant::now() > d) || self.max_nodes.is_some_and(|n| self.tables.nodes >= n)
    }

    /// For a card selection (or a sum): the order of its cards, the rules' (by the select hint the
    /// player got, which belongs to this selection only: it is used up), for a one-card selection
    /// the card policy's.
    fn card_order(&mut self, duel: &Duel, m: &Msg) -> Option<CardOrder> {
        let (Msg::SelectCard { cards, player, .. } | Msg::SelectSum { cards, player, .. }) = m else { return None };
        let rules = &mut self.policies[*player as usize];
        let (order, hint) = (rules.card_order(cards), rules.hint.take());
        let order = match card_policy() {
            Some(policy) if card_policy_applies(m) => policy.order(duel, m, hint, &order),
            _ => order,
        };
        Some(CardOrder { order, hint })
    }

    /// The set policy's answer to a selection of sets of cards (`set_policy_applies`) among the root's
    /// candidates (the rules' answer among them), None without either.
    fn set_answer(&mut self, duel: &Duel, m: &Msg) -> Option<Response> {
        let policy = set_policy()?;
        let (cards, player) = selection_cards(m)?;
        let rules = &mut self.policies[*player as usize];
        let moves = root_candidates(m, rules, None);
        let best = *policy.order_sets(duel, m, rules.hint, &rules.card_order(cards), &moves).first()?;
        rules.hint = None;
        Some(moves[best].clone())
    }

    fn iterative_deepening(&mut self, mut root: Root, mut move_order: Vec<usize>, mut time: TimeManager, max_depth: i32) -> (usize, i32) {
        let (mut best_move, mut completed_depth) = (move_order[0], 0);
        self.tables.last_value = f64::NAN;
        self.tables.last_root_values = vec![None; root.moves.len()];
        // (the last completed iteration's nodes and seconds)
        let (mut last_nodes, mut last_secs) = (0u64, 0.0);
        for depth in 1..=max_depth {
            if !time.starts_iteration(depth, last_secs, self.tables.growth(depth).expected()) {
                break;
            }
            let (nodes_before, start) = (self.tables.nodes, time.elapsed());
            self.tables.pv_height_now = 0;
            let iteration = self.search_root(&mut root, &move_order, depth, &mut time);
            if iteration.aborted {
                // (an unfinished iteration: its best counts if the previous best, searched first,
                // was searched completely and another one did better)
                if iteration.values[move_order[0]].is_some()
                    && let Some(best) = iteration.best
                {
                    best_move = best;
                }
                break;
            }
            let Some(best) = iteration.best else { break };
            let (nodes, secs) = (self.tables.nodes - nodes_before, time.elapsed() - start);
            if depth > 1 {
                self.tables.growth(depth).observe(nodes, last_nodes);
            }
            if std::env::var_os("YGOAI_DEBUG_ITERS").is_some() {
                eprintln!("iter {depth} {nodes} {secs:.6} {:.6} {} {:.3}", time.elapsed(), time.budget.unwrap_or(0.0), self.tables.growth(depth + 1).expected());
            }
            (last_nodes, last_secs) = (nodes, secs);
            self.tables.best_fraction = time.update(depth, best, best_move);
            self.tables.tm_scale = time.scale;
            (best_move, completed_depth) = (best, depth);
            self.tables.pv_height = self.tables.pv_height_now;
            let values = iteration.values;
            self.tables.last_value = values[best].unwrap();
            self.tables.last_root_values = values.clone();
            // (the next iteration: the best first; cut-off moves keep their bound's order)
            move_order.sort_by(|&a, &b| values[b].unwrap().partial_cmp(&values[a].unwrap()).unwrap_or(Ordering::Equal));
            if values[best].unwrap().abs() >= WIN / 2.0 {
                break; // (a forced win or loss: deeper changes nothing)
            }
        }
        (best_move, completed_depth)
    }

    /// The root's moves, in `move_order`, searched to `depth`.
    fn search_root(&mut self, root: &mut Root, move_order: &[usize], depth: i32, time: &mut TimeManager) -> Iteration {
        let mut iteration = Iteration { values: vec![None; root.moves.len()], best: None, aborted: false };
        let mut alpha = f64::NEG_INFINITY;
        for &m in move_order {
            let child = Node { alpha: if root.exact { f64::NEG_INFINITY } else { alpha }, beta: f64::INFINITY, depth: (depth - 1) as f64, height: 1, pv: m == move_order[0] };
            let nodes_before = self.tables.nodes;
            let result = self.search_move(&mut root.duel, root.prompt, &root.moves[m], child, MakeOn::Copy);
            time.root_nodes[m] += self.tables.nodes - nodes_before;
            let Ok(value) = result else {
                iteration.aborted = true;
                break;
            };
            iteration.values[m] = Some(value);
            if value > alpha {
                (alpha, iteration.best) = (value, Some(m));
            }
        }
        iteration
    }

    /// The value of the decision `prompt` for its player (negamax), searched `node.depth` plies deep.
    fn search(&mut self, duel: &mut Duel, prompt: &Msg, node: Node) -> Result<f64, Stop> {
        let Node { mut alpha, mut beta, depth, height, pv } = node;
        let us = prompt.player().unwrap_or(0);
        // (the main-line horizon extension: on the main line a card selection or sum, a decision
        // inside a move, is searched one ply deep at the horizon too, so that the search finishes
        // the move, not the rules; SPRT at 10+0.1 H1, +25.4 +/- 16.4 at 890 games, 2026-10-19)
        let extended = depth <= 0.0;
        if extended && !(node.extends() && inside_move(prompt)) {
            return Ok(evaluate_at(duel, us, Some(prompt)));
        }
        let depth = if extended { 1.0 } else { depth };

        // (moves are indices into the rules' order: the hash table keeps one, the history reorders them)
        let card_order = self.card_order(duel, prompt);
        let mut moves = ordered(prompt, duel.infos.turn_player as u32, card_order.as_ref().map(|c| c.order.as_slice()));
        if moves.is_empty() {
            return Ok(evaluate_at(duel, us, Some(prompt)));
        }
        if set_policy_applies(prompt)
            && let (Some(policy), Some(c)) = (set_policy(), &card_order)
        {
            // (sets of cards: the set policy's order)
            moves = policy.order_sets(duel, prompt, c.hint, &c.order, &moves).into_iter().map(|i| moves[i].clone()).collect();
        }
        if inside_move(prompt) {
            moves.truncate(self.tables.tune.card_width);
        }

        // transposition table: update alpha, beta or return immediately
        let key = hash_key(duel, prompt);
        let hash_result = self.tables.hash_table.get(key);
        let (mut tt_move, mut tt_entry) = (None, TtEntry::Missing);
        let nodes_before = self.tables.nodes;
        self.tables.tt_probes += 1;
        if let Some(h) = hash_result {
            self.tables.tt_hits += 1;
            tt_entry = if h.best_move >= 0 { TtEntry::WithMove(h.node_type) } else { TtEntry::WithoutMove };
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
        let collect_gaps = self.tables.stats.is_some() && depth <= 4.0;
        let static_eval = (depth <= t.futility_max_depth || collect_gaps).then(|| evaluate_at(duel, us, Some(prompt)));
        let futility_value = static_eval.filter(|_| depth <= t.futility_max_depth).map(|s| s + futility_margin(&t, depth));

        let ctx = key_context(duel, prompt, card_order.as_ref().and_then(|c| c.hint));
        let history_keys: Vec<usize> = moves.iter().map(|m| move_key(prompt, m, &ctx)).collect();
        if let Some(st) = &mut self.tables.stats {
            st.history_lookups(prompt, &moves, &history_keys, &ctx);
        }
        let move_order = self.move_order(tt_move, &history_keys);

        let mut node_type = NodeType::AllNode;
        let (mut best_value, mut best_move) = (f64::NEG_INFINITY, move_order[0]);
        let (mut searched, mut cut) = (0, None);
        for (i, &m) in move_order.iter().enumerate() {
            searched = i + 1;
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

            let make_on = if i == move_order.len() - 1 { MakeOn::InPlace } else { MakeOn::Copy };
            let child = Node { alpha, beta, depth: depth - 1.0, height: height + 1, pv: pv && i == 0 };
            let mut value = None;

            // late move reduction: a late move, unlikely the best by the move ordering, searched
            // `lmr_reduction` plies less with a null window (it only has to show that it does not
            // raise alpha); the full depth and window after all if it raises alpha
            if depth >= t.lmr_min_depth && i >= t.lmr_min_move_counter {
                let reduced_beta = if alpha.is_finite() { alpha + NULL_WINDOW } else { beta };
                let reduced = self.search_move(duel, prompt, mv, Node { beta: reduced_beta, depth: lmr_depth(&t, child.depth), pv: false, ..child }, MakeOn::Copy)?;
                if reduced <= alpha {
                    value = Some(reduced);
                }
                if let Some(st) = &mut self.tables.stats {
                    st.reduction(prompt, i, reduced > alpha);
                }
            }

            let value = match value {
                Some(v) => v,
                None => self.search_move(duel, prompt, mv, child, make_on)?,
            };

            if let Some(s) = static_eval
                && collect_gaps
                && i > 0
                && alpha.is_finite()
                && let Some(st) = &mut self.tables.stats
            {
                st.futility_gap(prompt, mv, depth, alpha - s, value > alpha);
            }
            if value > best_value {
                (best_value, best_move) = (value, m);
            }
            if value >= beta {
                node_type = NodeType::CutNode;
                // (history: this move up, the ones tried before it down)
                let tried: Vec<usize> = move_order[..i].iter().map(|&k| history_keys[k]).collect();
                self.tables.history_table.update(history_keys[m], &tried, depth, t.hist_exponent, t.hist_malus);
                cut = Some(i);
                break;
            }
            if value > alpha {
                node_type = NodeType::PvNode;
                alpha = value;
            }
        }

        if let Some(st) = &mut self.tables.stats {
            let subtree_nodes = self.tables.nodes - nodes_before + 1;
            st.node(prompt, NodeSummary { moves: move_order.len(), searched, tt: tt_entry, cut, depth, subtree_nodes });
        }
        if extended {
            let decided = node_type != NodeType::AllNode;
            self.tables.extension += Extension { nodes: 1, decided: decided as u64, not_rules_first: (decided && best_move != 0) as u64 };
        }
        if best_value.abs() < WIN / 2.0 {
            // (not a win or loss: those depend on the height)
            self.tables.hash_table.add(key, node_type, best_value, depth, best_move);
        }
        Ok(best_value)
    }

    /// Move ordering: the TT move first, then by history (the rules' order among equals).
    fn move_order(&self, tt_move: Option<usize>, history_keys: &[usize]) -> Vec<usize> {
        let history = |k: usize| self.tables.history_table.get(history_keys[k]);
        let mut order: Vec<usize> = (0..history_keys.len()).collect();
        order.sort_by(|&a, &b| (Some(a) != tt_move).cmp(&(Some(b) != tt_move)).then_with(|| history(b).partial_cmp(&history(a)).unwrap_or(Ordering::Equal)));
        order
    }

    /// The value of the move `mv` at the decision `prompt` for the player who makes it: the move
    /// made (`do_move`) and the new position searched as `child`, its value and window negated
    /// only if the other player decides there.
    ///
    /// On a copy of `duel` the rules' players' activation caps and select hint are this node's
    /// again afterwards, and a broken line (one the core cannot play) is a loss for the mover, who
    /// avoids it (in place it is the caller's copy's loss).
    fn search_move(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, child: Node, make_on: MakeOn) -> Result<f64, Stop> {
        if self.should_stop() {
            return Err(Stop::Abort);
        }
        self.tables.nodes += 1;
        if make_on == MakeOn::InPlace {
            return self.make_and_search(duel, prompt, mv, child);
        }
        let mut new_duel = duel.clone();
        let saved: Vec<_> = self.policies.iter().map(|p| (p.activations.clone(), p.hint)).collect();
        let result = self.make_and_search(&mut new_duel, prompt, mv, child);
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
    fn make_and_search(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, child: Node) -> Result<f64, Stop> {
        let us = prompt.player().unwrap_or(0);
        self.tables.seldepth = self.tables.seldepth.max(child.height);
        if child.pv {
            self.tables.pv_height_now = self.tables.pv_height_now.max(child.height);
        }
        Ok(match self.do_move(duel, prompt, mv, child)? {
            NewPosition::Win(winner) if winner == us => WIN - child.height as f64,
            NewPosition::Win(_) => -WIN + child.height as f64,
            NewPosition::End => evaluate_at(duel, us, None),
            NewPosition::StepLimit(last) => evaluate_at(duel, us, Some(&last)),
            NewPosition::Decision(next) if next.player().unwrap_or(0) == us => self.search(duel, &next, child)?,
            NewPosition::Decision(next) => -self.search(duel, &next, child.negated())?,
        })
    }

    /// Makes the move: the response, then the core runs on, the rules answering the decisions
    /// the search does not make (hidden ones, the only move, and at the horizon the decisions
    /// inside a move (`inside_move`), so that it completes first, unless the main-line extension
    /// applies, see `search`), up to the next searched decision. Sets of cards the set policy
    /// chooses, if there is one.
    fn do_move(&mut self, duel: &mut Duel, prompt: &Msg, mv: &Response, child: Node) -> Result<NewPosition, Stop> {
        let at_horizon = child.depth <= 0.0 && !child.extends();
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
            let rules = |state: &mut Self, prompt: &Msg| state.policies[prompt.player().unwrap_or(0) as usize].respond_to(prompt, None).map_err(|_| Stop::Broken);
            let response = if is_hidden_prompt(&prompt) {
                auto_response()
            } else if card_policy_applies(&prompt) && at_horizon && card_policy().is_some() {
                Response::cards(vec![self.card_order(duel, &prompt).unwrap().order[0]])
            } else if set_policy_applies(&prompt)
                && (at_horizon || !is_searched(&prompt))
                && let Some(answer) = self.set_answer(duel, &prompt)
            {
                answer
            } else if inside_move(&prompt) && at_horizon {
                rules(self, &prompt)?
            } else if is_searched(&prompt) {
                let mut moves = candidates(&prompt);
                if moves.len() > 1 {
                    return Ok(NewPosition::Decision(prompt));
                }
                match moves.pop() {
                    Some(only) => only,
                    None => rules(self, &prompt)?,
                }
            } else {
                rules(self, &prompt)?
            };
            duel.ocg_set_response(&encode(&prompt, &response).map_err(|_| Stop::Broken)?);
        }
        Ok(NewPosition::StepLimit(prompt))
    }
}

/// The history's context of a decision (`KeyContext`): the select hint, and for a chain response
/// the current chain's last link (its card's code and whether the mover controls it).
fn key_context(duel: &Duel, prompt: &Msg, hint: Option<u64>) -> KeyContext {
    let chain_top = match prompt {
        Msg::SelectChain { player, .. } => duel.core.current_chain.last().map_or(0, |l| {
            let code = l.triggering_effect.and_then(|e| duel.effect(e).handler).map_or(0, |c| duel.card(c).data.code);
            code as u128 * 2 + (l.triggering_controler as u32 == *player) as u128 + 1
        }),
        _ => 0,
    };
    KeyContext { hint, chain_top }
}

/// A decision inside a move (a card selection, a sum of cards): at the horizon the rules answer
/// it, so that the move completes before the eval.
fn inside_move(m: &Msg) -> bool {
    matches!(m, Msg::SelectCard { .. } | Msg::SelectSum { .. })
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
