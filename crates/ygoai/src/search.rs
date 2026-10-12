//! Alpha-beta search with iterative deepening on a (reconstructed) duel: agents/alphabeta.py's
//! search, on copies of the duel. See the Python module for the design (plies, move ordering,
//! transposition table, history, negamax, time, evaluation); this is the same search and makes
//! the same choices at a fixed depth (its transposition table hashes the same bytes the same
//! way, the rules' players draw the same random numbers).

use std::time::Instant;

use blake2::Blake2bVar;
use blake2::digest::{Update, VariableOutput};
use serde_json::Value;
use ygo_core::api::QueryInfo;
use ygo_core::constants::*;
use ygo_core::duel::Duel;

use crate::actions::{Response, battle_options, encode, idle_options};
use crate::game::run_until_prompt;
use crate::heuristic::{Heuristic, db_attack};
use crate::messages::Msg;
use crate::observe::{auto_response, is_hidden_prompt};
use crate::state::{LOCATIONS, QUERY_ALL, QUERY_BASIC, parse_location_query};

pub const WIN: f64 = 100000.0;
pub const MAX_DEPTH: i32 = 12;
/// (without a time control: a fixed depth)
pub const DEPTH_NO_CLOCK: i32 = 2;
/// (per node, after ordering)
pub const MAX_CHILDREN: usize = 12;
/// (a one-card selection: the rules' first choices only, below the root and at the root)
pub const CARD_WIDTH: usize = 3;
pub const ROOT_CARD_WIDTH: usize = 3;
pub const NEXT_ITERATION_SHARE: f64 = 0.5;
pub const HARD_FACTOR: f64 = 3.0;
pub const HARD_SHARE: f64 = 0.25;
/// (prompts answered by the rules between two plies)
pub const MAX_AUTO_STEPS: usize = 300;
pub const TT_BITS: u32 = 16;
pub const HIST_BITS: u32 = 14;
/// (per decision)
pub const HIST_DECAY: f64 = 0.5;

// evaluation weights (points; 1 point = 100 LP)
pub const W_LP: f64 = 1.0 / 100.0;
pub const W_MONSTER: f64 = 4.3;
/// (a face-up monster stronger than everything on the other side)
pub const W_DOMINANT: f64 = 17.6;
pub const W_HAND: f64 = 9.1;
pub const W_BACKROW: f64 = 5.1;
/// (two cards or fewer left to draw)
pub const W_DECK_OUT: f64 = 20.0;

const MZONE_QUERY: u32 = QUERY_POSITION | QUERY_ATTACK | QUERY_DEFENSE;
const EXACT: i8 = 0;
const LOWER: i8 = 1;
const UPPER: i8 = 2;

/// The prompts searched over (belief_mc.SEARCHED) when they have more than one candidate.
pub fn is_searched(m: &Msg) -> bool {
    matches!(
        m,
        Msg::SelectIdlecmd { .. }
            | Msg::SelectBattlecmd { .. }
            | Msg::SelectChain { .. }
            | Msg::SelectEffectyn { .. }
            | Msg::SelectYesno { .. }
            | Msg::SelectOption { .. }
            | Msg::SelectCard { .. }
    )
}

fn idx(i: usize) -> Response {
    Response { index: Some(i as i64), ..Response::default() }
}

/// The candidates of a searched prompt (belief_mc.candidates).
pub fn candidates(m: &Msg) -> Vec<Response> {
    match m {
        Msg::SelectIdlecmd { .. } => idle_options(m),
        Msg::SelectBattlecmd { .. } => battle_options(m),
        Msg::SelectChain { chains, forced, .. } => {
            let opts = (0..chains.len()).map(idx);
            if *forced != 0 { opts.collect() } else { std::iter::once(Response { index: Some(-1), ..Response::default() }).chain(opts).collect() }
        }
        Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => {
            [true, false].map(|y| Response { yes: Some(y), ..Response::default() }).to_vec()
        }
        Msg::SelectOption { options, .. } => (0..options.len()).map(idx).collect(),
        Msg::SelectCard { min: 1, max: 1, cards, .. } => {
            (0..cards.len()).map(|i| Response { indices: Some(vec![i as i64]), ..Response::default() }).collect()
        }
        _ => vec![],
    }
}

fn idle_order(kind: &str) -> i32 {
    match kind {
        "activate" => 0,
        "spsummon" => 1,
        "summon" => 2,
        "mset" => 3,
        "repos" => 4,
        "to_bp" => 5,
        "sset" => 6,
        "to_ep" => 7,
        "shuffle" => 8,
        _ => 9,
    }
}

fn battle_order(kind: &str) -> i32 {
    match kind {
        "activate" => 0,
        "attack" => 1,
        "to_m2" => 2,
        "to_ep" => 3,
        _ => 9,
    }
}

/// The candidates of a prompt, the heuristic player's preference first (for a card selection:
/// `card_order`, the rules' order by the select hint) (alphabeta.ordered).
pub fn ordered(m: &Msg, turn_player: u32, card_order: Option<&[usize]>) -> Vec<Response> {
    let mut cands = candidates(m);
    let k = |c: &Response| c.index.unwrap_or(0) as usize;
    match m {
        Msg::SelectCard { .. } if card_order.is_some() => {
            return card_order.unwrap().iter().filter(|&&i| i < cands.len()).map(|&i| cands[i].clone()).collect();
        }
        Msg::SelectIdlecmd { summonable, .. } => cands.sort_by_key(|c| {
            let kind = c.kind.as_deref().unwrap_or("");
            (idle_order(kind), if kind == "summon" { -db_attack(summonable[k(c)].code) } else { 0 })
        }),
        Msg::SelectBattlecmd { attackable, .. } => cands.sort_by_key(|c| {
            let kind = c.kind.as_deref().unwrap_or("");
            (battle_order(kind), if kind == "attack" { -db_attack(attackable[k(c)].code) } else { 0 })
        }),
        Msg::SelectChain { .. } => cands.sort_by_key(|c| c.index == Some(-1)),
        Msg::SelectEffectyn { player, .. } | Msg::SelectYesno { player, .. } => {
            cands.sort_by_key(|c| c.yes != Some(*player == turn_player))
        }
        _ => {}
    }
    cands
}

fn type_id(m: &Msg) -> u128 {
    match m {
        Msg::SelectIdlecmd { .. } => 1,
        Msg::SelectBattlecmd { .. } => 2,
        Msg::SelectChain { .. } => 3,
        Msg::SelectEffectyn { .. } => 4,
        Msg::SelectYesno { .. } => 5,
        Msg::SelectOption { .. } => 6,
        _ => 0,
    }
}

fn kind_id(kind: &str) -> u128 {
    match kind {
        "summon" => 1,
        "spsummon" => 2,
        "repos" => 3,
        "mset" => 4,
        "sset" => 5,
        "activate" => 6,
        "to_bp" => 7,
        "to_ep" => 8,
        "shuffle" => 9,
        "attack" => 10,
        "to_m2" => 11,
        _ => 0,
    }
}

/// The history slot of a move: its player, kind and card (and effect) (alphabeta._move_key).
pub fn move_key(m: &Msg, c: &Response) -> usize {
    let (mut kind, mut code, mut desc) = (0u128, 0u128, 0u128);
    let entry = |list: &[crate::messages::Card], i: i64| list.get(i as usize).map(|e| (e.code as u128, e.desc.unwrap_or(0) as u128));
    if let Some(k) = &c.kind {
        kind = kind_id(k);
        let list = match (m, k.as_str()) {
            (Msg::SelectIdlecmd { summonable, .. }, "summon") => Some(summonable),
            (Msg::SelectIdlecmd { spsummonable, .. }, "spsummon") => Some(spsummonable),
            (Msg::SelectIdlecmd { repositionable, .. }, "repos") => Some(repositionable),
            (Msg::SelectIdlecmd { msetable, .. }, "mset") => Some(msetable),
            (Msg::SelectIdlecmd { ssetable, .. }, "sset") => Some(ssetable),
            (Msg::SelectIdlecmd { activatable, .. }, "activate") => Some(activatable),
            (Msg::SelectBattlecmd { activatable, .. }, "activate") => Some(activatable),
            (Msg::SelectBattlecmd { attackable, .. }, "attack") => Some(attackable),
            _ => None,
        };
        if let Some((cd, ds)) = list.and_then(|l| entry(l, c.index.unwrap_or(0))) {
            (code, desc) = (cd, ds);
        }
    } else if let Msg::SelectChain { chains, .. } = m {
        if let Some(i) = c.index.filter(|&i| i >= 0)
            && let Some((cd, ds)) = entry(chains, i)
        {
            (kind, code, desc) = (1, cd, ds);
        }
    } else if let Some(yes) = c.yes {
        kind = 1 + yes as u128;
        match m {
            Msg::SelectEffectyn { code: cd, desc: ds, .. } => (code, desc) = (*cd as u128, *ds as u128),
            Msg::SelectYesno { desc: ds, .. } => desc = *ds as u128,
            _ => {}
        }
    } else if let Some(ind) = &c.indices {
        kind = 1;
        if let Msg::SelectCard { cards, .. } = m {
            code = cards.get(ind[0] as usize).map_or(0, |e| e.code as u128);
        }
    } else if let Some(i) = c.index {
        kind = (1 + i) as u128;
    }
    let player = m.player().unwrap_or(0) as u128;
    let mut h = (((type_id(m) * 31 + player) * 31 + kind) * 1000003 + code) * 1000003 + desc;
    h ^= h >> 17;
    let h = (h as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> (64 - HIST_BITS)) as usize
}

fn battle_stat(position: u32, attack: i32, defense: i32) -> i32 {
    if position & POS_DEFENSE != 0 { defense } else { attack }
}

fn query(d: &mut Duel, p: u32, loc: u32, flags: u32) -> Vec<u8> {
    d.ocg_query_location(&QueryInfo { flags, con: p as u8, loc, seq: 0, overlay_seq: 0 })
}

/// The position's value for `me` (in points of 100 LP) (alphabeta.evaluate).
pub fn evaluate(d: &mut Duel, me: u32) -> f64 {
    let info = d.ext_get_info();
    let lp = [info[3] as i32, info[4] as i32];
    let mons: Vec<Vec<(u32, i32, i32)>> = [me, 1 - me]
        .iter()
        .map(|&p| {
            parse_location_query(&query(d, p, LOCATION_MZONE, MZONE_QUERY))
                .into_iter()
                .flatten()
                .map(|c| (c.position.unwrap_or(0), c.attack.unwrap_or(0), c.defense.unwrap_or(0)))
                .collect()
        })
        .collect();
    let (me_i, opp_i) = (me as usize, 1 - me as usize);
    let mut value = W_LP * (lp[me_i] - lp[opp_i]) as f64 + W_MONSTER * (mons[0].len() as f64 - mons[1].len() as f64);
    for (s, p) in [(0usize, me), (1, 1 - me)] {
        let sign = if s == 0 { 1.0 } else { -1.0 };
        let best_other = mons[1 - s].iter().map(|&(pos, a, df)| battle_stat(pos, a, df)).max().unwrap_or(-1);
        for &(pos, a, _) in &mons[s] {
            if pos & POS_FACEUP_ATTACK != 0 && a > best_other {
                value += sign * W_DOMINANT;
            }
        }
        value += sign * W_HAND * d.ocg_query_count(p as u8, LOCATION_HAND) as f64;
        let backrow = parse_location_query(&query(d, p, LOCATION_SZONE, QUERY_POSITION))
            .into_iter()
            .flatten()
            .filter(|c| c.position.unwrap_or(0) & POS_FACEDOWN != 0)
            .count();
        value += sign * W_BACKROW * backrow as f64;
        if d.ocg_query_count(p as u8, LOCATION_DECK) <= 2 {
            value -= sign * W_DECK_OUT;
        }
    }
    value
}

/// Python's repr() of a JSON value as the dicts and lists of the Python module hold it.
fn py_repr(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("None"),
        Value::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => {
            out.push('\'');
            out.push_str(s);
            out.push('\'');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push('\'');
                out.push_str(k);
                out.push_str("': ");
                py_repr(x, out);
            }
            out.push('}');
        }
    }
}

/// repr(sorted(prompt.items())) of the prompt's dict (its fields in the parser's order, "type"
/// among them).
fn prompt_repr(m: &Msg) -> String {
    let Value::Object(o) = serde_json::to_value(m).expect("json") else { unreachable!() };
    let mut items: Vec<(&String, &Value)> = o.iter().collect();
    items.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::from("[");
    for (i, (k, v)) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str("('");
        out.push_str(k);
        out.push_str("', ");
        py_repr(v, &mut out);
        out.push(')');
    }
    out.push(']');
    out
}

/// The transposition table key of a position and its pending decision (alphabeta._tt_key).
pub fn tt_key(d: &mut Duel, m: &Msg) -> u64 {
    let mut h = Blake2bVar::new(8).unwrap();
    let info = d.ext_get_info();
    h.update(
        format!("{{'turn': {}, 'turn_player': {}, 'phase': {}, 'lp': [{}, {}]}}", info[0], info[1], info[2], info[3] as i32, info[4] as i32)
            .as_bytes(),
    );
    h.update(&d.ocg_query_field());
    for p in 0..2 {
        for (_, loc) in LOCATIONS {
            h.update(&query(d, p, loc, if loc == LOCATION_DECK { QUERY_BASIC } else { QUERY_ALL }));
        }
    }
    h.update(prompt_repr(m).as_bytes());
    let mut out = [0u8; 8];
    h.finalize_variable(&mut out).unwrap();
    let k = u64::from_le_bytes(out);
    if k == 0 { 1 } else { k }
}

#[derive(Clone, Copy, Default)]
struct TtEntry {
    key: u64,
    value: f64,
    depth: i16,
    flag: i8,
    mv: i8,
    gen_: u16,
}

/// Why a line was not valued: out of time (the iteration is abandoned) or broken (the engine
/// failed on it: the player to move avoids it).
enum Stop {
    Abort,
    Broken,
}

/// The search's tables, kept from one decision to the next (an agent holds one).
pub struct AlphaBeta {
    tt: Vec<TtEntry>,
    tt_gen: u16,
    hist: Vec<f64>,
    pub nodes: u64,
}

impl Default for AlphaBeta {
    fn default() -> Self {
        AlphaBeta { tt: vec![TtEntry::default(); 1 << TT_BITS], tt_gen: 0, hist: vec![0.0; 1 << HIST_BITS], nodes: 0 }
    }
}

/// One search: the rules' players, the deadline and the node limit.
struct Ctx<'a> {
    ab: &'a mut AlphaBeta,
    policies: [Heuristic; 2],
    deadline: Option<Instant>,
    /// (in the count of `ab.nodes`)
    node_limit: Option<u64>,
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
    /// The tables for a new decision.
    pub fn prepare(&mut self) {
        self.tt_gen = ((self.tt_gen as u32 + 1) % 65536) as u16;
        if self.tt_gen == 0 {
            self.tt_gen = 1;
        }
        for h in &mut self.hist {
            *h *= HIST_DECAY;
        }
    }

    /// The best of `core_cands` (responses to the duel's pending prompt `core_prompt`) and the
    /// depth completed (after `prepare`). The duel is not changed.
    pub fn search(&mut self, duel: &Duel, core_prompt: &Msg, core_cands: &[Response], p: &SearchParams) -> (usize, i32) {
        let t0 = Instant::now() - std::time::Duration::from_secs_f64(p.elapsed);
        let mut policies = [Heuristic::new(p.seeds[0] as i128, 0), Heuristic::new(p.seeds[1] as i128, 1)];
        let player = core_prompt.player().unwrap_or(0) as usize;
        policies[player].hint = p.hint;
        let turn_player = duel.ext_get_info()[1];
        let deadline = p.budget.map(|b| t0 + std::time::Duration::from_secs_f64(b.max((HARD_FACTOR * b).min(HARD_SHARE * p.clock_remaining))));
        let node_limit = p.max_nodes.map(|n| self.nodes + n);
        let mut ctx = Ctx { ab: self, policies, deadline, node_limit };
        let card_order = ctx.card_order(core_prompt);
        let pref = ordered(core_prompt, turn_player, card_order.as_deref());
        let mut order: Vec<usize> = (0..core_cands.len()).collect();
        order.sort_by_key(|&k| pref.iter().position(|c| *c == core_cands[k]).unwrap_or(pref.len()));
        if matches!(core_prompt, Msg::SelectCard { .. }) {
            order.truncate(ROOT_CARD_WIDTH);
        }
        let unbounded = p.budget.is_none() && p.max_nodes.is_none();
        ctx.deepen(duel, core_prompt, core_cands, order, t0, p.budget, unbounded)
    }
}

impl Ctx<'_> {
    fn timed_out(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() > d) || self.node_limit.is_some_and(|n| self.ab.nodes >= n)
    }

    /// For a card selection: the rules' order of its cards (by the select hint the player got,
    /// which belongs to this selection only).
    fn card_order(&mut self, m: &Msg) -> Option<Vec<usize>> {
        let Msg::SelectCard { cards, player, .. } = m else { return None };
        let policy = &mut self.policies[*player as usize];
        let order = policy.card_order(cards);
        policy.hint = None;
        Some(order)
    }

    #[allow(clippy::too_many_arguments)]
    fn deepen(&mut self, duel: &Duel, prompt: &Msg, cands: &[Response], mut order: Vec<usize>, t0: Instant, budget: Option<f64>, fixed_depth: bool) -> (usize, i32) {
        let (mut best, mut done) = (order[0], 0);
        let max_depth = if fixed_depth { DEPTH_NO_CLOCK } else { MAX_DEPTH };
        for depth in 1..=max_depth {
            if let Some(b) = budget
                && depth > 1
                && t0.elapsed().as_secs_f64() > NEXT_ITERATION_SHARE * b
            {
                break;
            }
            let mut values: Vec<Option<f64>> = vec![None; cands.len()];
            let (mut alpha, mut it_best) = (f64::NEG_INFINITY, None);
            let mut aborted = false;
            for &k in &order {
                match self.child(duel, prompt, &cands[k], depth - 1, alpha, f64::INFINITY, 1) {
                    Ok(v) => {
                        values[k] = Some(v);
                        if v > alpha {
                            (alpha, it_best) = (v, Some(k));
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
                if values[order[0]].is_some()
                    && let Some(b) = it_best
                {
                    best = b;
                }
                break;
            }
            let Some(b) = it_best else { break };
            (best, done) = (b, depth);
            // (the next iteration: the best first; cut-off moves keep their bound's order)
            order.sort_by(|&a, &b| values[b].unwrap().partial_cmp(&values[a].unwrap()).unwrap_or(std::cmp::Ordering::Equal));
            if values[best].unwrap().abs() >= WIN / 2.0 {
                break; // (a forced win or loss: deeper changes nothing)
            }
        }
        (best, done)
    }

    /// The value of `cand` (searched `depth` plies deeper) on a copy of the duel; the rules'
    /// players' caps and select hint are this node's again afterwards.
    #[allow(clippy::too_many_arguments)]
    fn child(&mut self, duel: &Duel, prompt: &Msg, cand: &Response, depth: i32, alpha: f64, beta: f64, ply: i32) -> Result<f64, Stop> {
        if self.timed_out() {
            return Err(Stop::Abort);
        }
        self.ab.nodes += 1;
        let mut copy = duel.clone();
        let saved: Vec<_> = self.policies.iter().map(|p| (p.activations.clone(), p.hint)).collect();
        let r = self.expand(&mut copy, prompt.clone(), cand, depth, alpha, beta, ply);
        for (p, (act, hint)) in self.policies.iter_mut().zip(saved) {
            (p.activations, p.hint) = (act, hint);
        }
        match r {
            Err(Stop::Broken) if !self.timed_out() => Ok(-WIN), // (the player to move avoids the line)
            Err(_) => Err(Stop::Abort),
            Ok(v) => Ok(v),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn node(&mut self, duel: &mut Duel, prompt: &Msg, depth: i32, mut alpha: f64, mut beta: f64, ply: i32) -> Result<f64, Stop> {
        // (moves are indices into the heuristic's order: the TT keeps one, the history reorders them)
        let card_order = self.card_order(prompt);
        let mut cands = ordered(prompt, duel.ext_get_info()[1], card_order.as_deref());
        if card_order.is_some() {
            cands.truncate(CARD_WIDTH);
        }
        let key = tt_key(duel, prompt);
        let slot = (key & ((1 << TT_BITS) - 1)) as usize;
        let e = self.ab.tt[slot];
        let mut tt_move = None;
        if e.key == key && e.gen_ == self.ab.tt_gen {
            let v = e.value;
            if e.depth as i32 >= depth {
                if e.flag == EXACT {
                    return Ok(v);
                }
                if e.flag == LOWER {
                    alpha = alpha.max(v);
                } else {
                    beta = beta.min(v);
                }
                if alpha >= beta {
                    return Ok(v);
                }
            }
            if e.mv >= 0 && (e.mv as usize) < cands.len() {
                tt_move = Some(e.mv as usize);
            }
        }
        let (alpha0, beta0) = (alpha, beta);
        let keys: Vec<usize> = cands.iter().map(|c| move_key(prompt, c)).collect();
        let mut idx: Vec<usize> = (0..cands.len()).collect();
        let hist = &self.ab.hist;
        // (the TT move first, then the moves with a better history; the heuristic's order among equals)
        idx.sort_by(|&a, &b| {
            ((Some(a) != tt_move), -hist[keys[a]])
                .partial_cmp(&((Some(b) != tt_move), -hist[keys[b]]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(MAX_CHILDREN);
        let (mut best, mut best_i) = (f64::NEG_INFINITY, idx[0]);
        for (i, &ci) in idx.iter().enumerate() {
            let c = &cands[ci];
            let v = if i == idx.len() - 1 {
                // (the last move: nothing needs this position afterwards, no copy)
                if self.timed_out() {
                    return Err(Stop::Abort);
                }
                self.ab.nodes += 1;
                self.expand(duel, prompt.clone(), c, depth - 1, alpha, beta, ply + 1)?
            } else {
                self.child(duel, prompt, c, depth - 1, alpha, beta, ply + 1)?
            };
            if v > best {
                (best, best_i) = (v, ci);
            }
            alpha = alpha.max(best);
            if alpha >= beta {
                // (a cutoff: this move up, the ones tried before it down)
                let bonus = (depth * depth) as f64;
                self.ab.hist[keys[ci]] += bonus;
                for &k in &idx[..i] {
                    self.ab.hist[keys[k]] -= bonus / i as f64;
                }
                break;
            }
        }
        if best.abs() < WIN / 2.0 {
            // (not a win or loss: those depend on the ply)
            let flag = if best <= alpha0 {
                UPPER
            } else if best >= beta0 {
                LOWER
            } else {
                EXACT
            };
            self.ab.tt[slot] = TtEntry { key, value: best, depth: depth as i16, flag, mv: best_i as i8, gen_: self.ab.tt_gen };
        }
        Ok(best)
    }

    /// Plays `cand`, runs to the next ply and values it, for the player who plays it (negamax:
    /// values and the window (alpha, beta) are the mover's; they change sign only where the next
    /// ply is the other player's, which in this game need not be the case).
    #[allow(clippy::too_many_arguments)]
    fn expand(&mut self, duel: &mut Duel, mut prompt: Msg, cand: &Response, depth: i32, alpha: f64, beta: f64, ply: i32) -> Result<f64, Stop> {
        let me = prompt.player().unwrap_or(0);
        duel.ocg_set_response(&encode(&prompt, cand).map_err(|_| Stop::Broken)?);
        for _ in 0..MAX_AUTO_STEPS {
            let step = run_until_prompt(duel, Some(&prompt)).map_err(|_| Stop::Broken)?;
            for m in &step.messages {
                match m {
                    Msg::Win { player, .. } => return Ok(if *player == me { WIN - ply as f64 } else { -WIN + ply as f64 }),
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
                return Ok(evaluate(duel, me));
            }
            prompt = step.prompt.unwrap();
            let player = prompt.player().unwrap_or(0);
            let resp = if is_hidden_prompt(&prompt) {
                auto_response()
            } else if matches!(prompt, Msg::SelectCard { .. }) && depth <= 0 {
                // (at the horizon a card selection is the rules': the effect resolves first)
                self.policies[player as usize].respond_to(&prompt, None).map_err(|_| Stop::Broken)?
            } else if is_searched(&prompt) {
                let cands = candidates(&prompt);
                if cands.len() > 1 {
                    if depth <= 0 {
                        return Ok(evaluate(duel, me));
                    }
                    if player == me {
                        return self.node(duel, &prompt, depth, alpha, beta, ply);
                    }
                    return Ok(-self.node(duel, &prompt, depth, -beta, -alpha, ply)?);
                }
                match cands.into_iter().next() {
                    Some(c) => c,
                    None => self.policies[player as usize].respond_to(&prompt, None).map_err(|_| Stop::Broken)?,
                }
            } else {
                self.policies[player as usize].respond_to(&prompt, None).map_err(|_| Stop::Broken)?
            };
            duel.ocg_set_response(&encode(&prompt, &resp).map_err(|_| Stop::Broken)?);
        }
        Ok(evaluate(duel, me))
    }
}
