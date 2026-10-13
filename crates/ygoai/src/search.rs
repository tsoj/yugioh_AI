//! Alpha-beta search with iterative deepening on a (reconstructed) duel: agents/alphabeta.py's
//! search, on copies of the duel. See the Python module for the design (plies, move ordering,
//! transposition table, history, negamax, time, evaluation); this is the same search and makes
//! the same choices at a fixed depth (its transposition table hashes the same bytes the same
//! way, the rules' players draw the same random numbers).

use std::collections::BTreeMap;
use std::time::Instant;

use ygo_core::api::QueryInfo;
use ygo_core::constants::*;
use ygo_core::duel::Duel;
use ygo_core::types::CardId;

use crate::actions::{Response, available_places, battle_options, check_sum, encode, idle_options};
use crate::game::run_until_prompt;
use crate::heuristic::{Heuristic, card, card_value, db_attack};
use crate::messages::Msg;
use crate::observe::{auto_response, is_hidden_prompt};
use crate::state::parse_location_query;

pub const WIN: f64 = 100000.0;
pub const MAX_DEPTH: i32 = 12;
/// (without a time control: a fixed depth)
pub const DEPTH_NO_CLOCK: i32 = 2;
/// (per node, after ordering)
pub const MAX_CHILDREN: usize = 12;
/// (a one-card selection: the rules' first choices only, below the root and at the root)
pub const CARD_WIDTH: usize = 3;
pub const ROOT_CARD_WIDTH: usize = 3;
/// (the searched decisions inside a move, each a feature: how many of the rules' first choices)
const UNSELECT_WIDTH: usize = 3;
const TRIBUTE_WIDTH: usize = 4;
const MULTI_CARDS: usize = 4;
const MULTI_WIDTH: usize = 6;
const SUM_WIDTH: usize = 4;
const SUM_STEPS: usize = 5000;
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
/// (PVS: the null window's width, 1 LP: the smallest step of a value)
const PVS_EPS: f64 = 0.01;
/// (RFP at null-window nodes up to this depth, by this margin per remaining ply)
const RFP_DEPTH: i32 = 2;
const RFP_MARGIN: f64 = 12.0;
/// (late move reductions from this depth on, for the moves after the first LMR_MOVES)
const LMR_DEPTH: i32 = 2;
const LMR_MOVES: usize = 2;
/// (quiescence plies past the nominal depth)
const MAX_QPLY: i32 = 6;
/// (continuation history entries: (previous move, move) pairs)
const CONT_BITS: u32 = 16;
/// (`budget`: the `depth` arguments are node budgets in 1/BUDGET_UNIT nodes; the root's
/// iterations, each a larger budget)
const BUDGET_UNIT: i32 = 16;
const BUDGET_ITERATIONS: i32 = 30;
/// (`budget`: the cutoff and re-search statistics, by prompt type, TT state and move index)
const HAZARD_TYPES: usize = 16;
const HAZARD_MOVES: usize = 12;

/// The parameters of the budget search (`budget`), from $YGOAI_BUDGET (a comma list of
/// name=value; the defaults otherwise): `growth`, the root budget's factor per iteration (about
/// the effective branching factor); `root`, the budget of a root move in the first iteration;
/// `k`, the divisor of the reduction (as in Stockfish's log(depth) * log(move number) / k, the
/// depth here log(budget) / log(growth)); the confidence of the move ordering by the node's TT
/// state (`none`: no entry, `nomove`: an entry without a move, `exact`, `lower`, `upper`), a
/// factor of the reduction: 0 is a uniform split among the moves.
#[derive(Clone, Copy, Debug)]
pub struct BudgetParams {
    pub growth: f64,
    pub root: f64,
    pub k: f64,
    pub conf: [f64; 5],
}

impl BudgetParams {
    pub fn from_env() -> BudgetParams {
        static P: std::sync::OnceLock<BudgetParams> = std::sync::OnceLock::new();
        *P.get_or_init(|| {
            let mut p = BudgetParams { growth: 1.6, root: 2.0, k: 2.0, conf: [0.5, 0.5, 0.5, 1.5, 1.0] };
            for kv in std::env::var("YGOAI_BUDGET").unwrap_or_default().split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (name, v) = kv.split_once('=').unwrap_or_else(|| panic!("YGOAI_BUDGET: {kv} is not name=value"));
                let v: f64 = v.parse().unwrap_or_else(|_| panic!("YGOAI_BUDGET: {kv}"));
                match name {
                    "growth" => p.growth = v,
                    "root" => p.root = v,
                    "k" => p.k = v,
                    "none" => p.conf[0] = v,
                    "nomove" => p.conf[1] = v,
                    "exact" => p.conf[2] = v,
                    "lower" => p.conf[3] = v,
                    "upper" => p.conf[4] = v,
                    _ => panic!("YGOAI_BUDGET: unknown parameter {name}"),
                }
            }
            p
        })
    }
}

/// Search features, each off unless named in $YGOAI_SEARCH (a comma list; unset: DEFAULT_FEATURES): `pvs`,
/// `rfp` (reverse futility pruning at null-window nodes; with pvs), `rfp_kind` (at depth 1, by
/// the least a move of each kind keeps), `futility` (depth 1, by the most a move of each kind
/// gains), `lmr`, `lmr_noresearch`, `conthist` (continuation history), `pv_ext` (the main
/// line sees the opponent's reply), `qsearch` (battle and chain decisions past the depth).
/// Decisions inside a move that the search can make instead of the rules' player (each the rules'
/// first few choices): `unselect` (card by card selections, e.g. most materials), `tribute`,
/// `multicard` (selections of several cards), `sum` (selections by a sum: Synchro and Ritual
/// materials), `position`, `place` (the zone).
/// `budget`: a node budget instead of the depth (see `Ctx::allocate`; replaces lmr, pvs, rfp,
/// futility, pv_ext and qsearch, which it ignores).
/// The features that passed their SPRTs (LMR: +19.1 +/- 14.9, LLR +3.01, 1204 games, 2026-10-13).
pub const DEFAULT_FEATURES: &str = "lmr";

#[derive(Clone, Copy, Default, Debug)]
pub struct Features {
    pub pvs: bool,
    pub rfp: bool,
    pub rfp_kind: bool,
    pub futility: bool,
    pub lmr: bool,
    pub lmr_noresearch: bool,
    pub conthist: bool,
    pub pv_ext: bool,
    pub qsearch: bool,
    pub unselect: bool,
    pub tribute: bool,
    pub multicard: bool,
    pub sum: bool,
    pub position: bool,
    pub place: bool,
    pub budget: bool,
}

impl Features {
    pub fn parse(list: &str) -> Features {
        let mut f = Features::default();
        for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            match name {
                "pvs" => f.pvs = true,
                "rfp" => f.rfp = true,
                "rfp_kind" => f.rfp_kind = true,
                "futility" => f.futility = true,
                "lmr" => f.lmr = true,
                "lmr_noresearch" => f.lmr_noresearch = true,
                "conthist" => f.conthist = true,
                "pv_ext" => f.pv_ext = true,
                "qsearch" => f.qsearch = true,
                "unselect" => f.unselect = true,
                "tribute" => f.tribute = true,
                "multicard" => f.multicard = true,
                "sum" => f.sum = true,
                "position" => f.position = true,
                "place" => f.place = true,
                "budget" => f.budget = true,
                _ => panic!("YGOAI_SEARCH: unknown feature {name}"),
            }
        }
        f
    }

    /// The features of $YGOAI_SEARCH, DEFAULT_FEATURES when it is unset (read once).
    pub fn from_env() -> Features {
        static F: std::sync::OnceLock<Features> = std::sync::OnceLock::new();
        *F.get_or_init(|| {
            Features::parse(&std::env::var("YGOAI_SEARCH").unwrap_or_else(|_| DEFAULT_FEATURES.to_string()))
        })
    }
}

/// A move's kind for the pruning margins: the prompt and the command kind, a chain's pass or
/// chain, a yes or no.
fn move_kind(m: &Msg, c: &Response) -> String {
    let t = m.name();
    if let Some(k) = &c.kind {
        return format!("SELECT_{}:{k}", t.trim_start_matches("SELECT_"));
    }
    if matches!(m, Msg::SelectChain { .. }) {
        return format!("{t}:{}", if c.index.unwrap_or(-1) < 0 { "pass" } else { "chain" });
    }
    if let Some(yes) = c.yes {
        return format!("{t}:{}", if yes { "yes" } else { "no" });
    }
    t.to_string()
}

/// (futility: the 99th percentile of a leaf move's value minus the static value before it, by
/// kind, over 378k leaves of alphabeta-vs-heuristic games; kinds not listed are never pruned)
fn futility_margin(kind: &str) -> Option<f64> {
    Some(match kind {
        "SELECT_IDLECMD:summon" => 48.0,
        "SELECT_IDLECMD:spsummon" => 30.9,
        "SELECT_IDLECMD:mset" => 30.4,
        "SELECT_IDLECMD:sset" => 0.0,
        "SELECT_IDLECMD:activate" => 43.8,
        "SELECT_IDLECMD:repos" => 17.6,
        "SELECT_IDLECMD:to_bp" | "SELECT_IDLECMD:to_ep" | "SELECT_IDLECMD:shuffle" => 0.0,
        "SELECT_BATTLECMD:attack" => 47.0,
        "SELECT_BATTLECMD:activate" => 16.8,
        "SELECT_BATTLECMD:to_m2" => 0.0,
        "SELECT_BATTLECMD:to_ep" => 26.1,
        "SELECT_CHAIN:pass" => 21.9,
        "SELECT_CHAIN:chain" => 41.5,
        "SELECT_EFFECTYN:yes" => 40.3,
        "SELECT_EFFECTYN:no" => 9.1,
        "SELECT_YESNO:yes" => 43.8,
        "SELECT_YESNO:no" => 9.1,
        "SELECT_OPTION" | "SELECT_CARD" => 39.5,
        _ => return None,
    })
}

/// (reverse futility by kind: the 0.1st percentile of a leaf move's value minus the static value
/// before it, same data)
fn rfp_low(kind: &str) -> Option<f64> {
    Some(match kind {
        "SELECT_IDLECMD:summon" | "SELECT_IDLECMD:mset" => -79.5,
        "SELECT_IDLECMD:spsummon" => -74.8,
        "SELECT_IDLECMD:sset" => -30.7,
        "SELECT_IDLECMD:activate" => -74.7,
        "SELECT_IDLECMD:repos" => -35.8,
        "SELECT_IDLECMD:to_bp" | "SELECT_IDLECMD:shuffle" => -9.1,
        "SELECT_IDLECMD:to_ep" => -57.2,
        "SELECT_BATTLECMD:attack" => -28.3,
        "SELECT_BATTLECMD:activate" | "SELECT_CHAIN:chain" => -93.1,
        "SELECT_BATTLECMD:to_m2" => -17.6,
        "SELECT_BATTLECMD:to_ep" => -62.3,
        "SELECT_CHAIN:pass" => -70.4,
        "SELECT_EFFECTYN:yes" | "SELECT_EFFECTYN:no" => -39.5,
        "SELECT_YESNO:yes" => -35.5,
        "SELECT_YESNO:no" => -26.5,
        "SELECT_OPTION" => -43.8,
        "SELECT_CARD" => -74.4,
        _ => return None,
    })
}

/// The continuation history slot of a move after the move `prev` (history slots).
fn cont_key(prev: usize, key: usize) -> usize {
    let mut h = ((prev as u32).wrapping_mul(0x9E3779B1)) ^ ((key as u32).wrapping_mul(0x85EBCA77));
    h ^= h >> 15;
    (h.wrapping_mul(0x2C1B3C6D) >> (32 - CONT_BITS)) as usize
}

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
    ) || is_inner(m)
}

/// A decision inside a move that a feature has the search make (see `Features`).
fn is_inner(m: &Msg) -> bool {
    let f = Features::from_env();
    match m {
        Msg::SelectUnselectCard { .. } => f.unselect,
        Msg::SelectTribute { .. } => f.tribute,
        Msg::SelectSum { .. } => f.sum,
        Msg::SelectPosition { .. } => f.position,
        Msg::SelectPlace { count: 1, .. } => f.place,
        _ => false,
    }
}

fn idx(i: usize) -> Response {
    Response { index: Some(i as i64), ..Response::default() }
}

/// The candidates of a searched prompt (belief_mc.candidates).
pub fn candidates(m: &Msg) -> Vec<Response> {
    candidates_in(m, None)
}

/// The candidates, for a card selection with the cards in `order` (the rules' order by the select
/// hint; None: as listed): a decision inside a move has the rules' first few choices, best first.
pub fn candidates_in(m: &Msg, order: Option<&[usize]>) -> Vec<Response> {
    let f = Features::from_env();
    let listed = |n: usize| -> Vec<usize> { order.map_or_else(|| (0..n).collect(), |o| o.iter().copied().filter(|&i| i < n).collect()) };
    let pick = |idx: Vec<usize>| {
        let mut idx: Vec<i64> = idx.into_iter().map(|i| i as i64).collect();
        idx.sort();
        Response { indices: Some(idx), ..Response::default() }
    };
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
        Msg::SelectCard { min, max, cards, .. } if f.multicard && *max >= 1 && !cards.is_empty() => {
            // (the rules' count of the first cards, then other counts and other cards among the first few)
            let top: Vec<usize> = listed(cards.len()).into_iter().take(MULTI_CARDS).collect();
            let lo = (*min).max(1).min(*max).min(cards.len() as u32) as usize;
            let hi = (*max as usize).min(top.len()).max(lo.min(top.len()));
            let mut out = Vec::new();
            for k in std::iter::once(lo).chain((lo + 1..=hi).rev()) {
                if k <= top.len() {
                    for c in combinations(top.len(), k) {
                        out.push(c.into_iter().map(|j| top[j]).collect::<Vec<_>>());
                    }
                }
            }
            if lo > top.len() {
                out.push(listed(cards.len())[..lo].to_vec());
            }
            out.into_iter().take(MULTI_WIDTH).map(pick).collect()
        }
        Msg::SelectUnselectCard { finishable, cancelable, select_cards, .. } if f.unselect => {
            let mut out: Vec<Response> = listed(select_cards.len()).into_iter().take(UNSELECT_WIDTH).map(idx).collect();
            if *finishable != 0 || (*cancelable != 0 && select_cards.is_empty()) {
                out.push(Response { finish: Some(true), ..Response::default() });
            }
            out
        }
        Msg::SelectTribute { min, max, cards, .. } if f.tribute && !cards.is_empty() => {
            // (the least valuable sets that release enough, without a card to spare)
            let n = cards.len().min(10);
            let rel = |i: usize| cards[i].release_param.unwrap_or(0);
            let mut sets: Vec<(f64, Vec<usize>)> = Vec::new();
            for mask in 1u32..(1 << n) {
                let set: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
                let total: u32 = set.iter().map(|&i| rel(i)).sum();
                if set.len() > (*max).max(1) as usize || total < *min || set.iter().any(|&i| total - rel(i) >= *min) {
                    continue;
                }
                sets.push((set.iter().map(|&i| card_value(&cards[i], card(cards[i].code))).sum(), set));
            }
            sets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            sets.into_iter().take(TRIBUTE_WIDTH).map(|(_, s)| pick(s)).collect()
        }
        Msg::SelectSum { cards, .. } if f.sum && !cards.is_empty() => {
            // (valid selections, the least valuable cards tried first)
            let mut by_value: Vec<usize> = (0..cards.len()).collect();
            let v = |i: usize| card_value(&cards[i], card(cards[i].code));
            by_value.sort_by(|&a, &b| v(a).partial_cmp(&v(b)).unwrap_or(std::cmp::Ordering::Equal));
            sum_selections(m, &by_value).into_iter().map(pick).collect()
        }
        Msg::SelectPosition { code, positions, .. } if f.position => {
            let order = if card(*code).is_some_and(|c| c.attack <= 0) {
                [POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK]
            } else {
                [POS_FACEUP_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEDOWN_ATTACK]
            };
            order.into_iter().filter(|p| positions & p != 0).map(|p| Response { position: Some(p as i64), ..Response::default() }).collect()
        }
        Msg::SelectPlace { player, count: 1, flag } if f.place => {
            available_places(*player, *flag).into_iter().map(|pl| Response { places: Some(vec![pl]), ..Response::default() }).collect()
        }
        _ => vec![],
    }
}

/// The k-element subsets of 0..n, in lexicographic order (the first elements first).
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut c: Vec<usize> = (0..k).collect();
    if k > n {
        return out;
    }
    loop {
        out.push(c.clone());
        let Some(i) = (0..k).rev().find(|&i| c[i] != i + n - k) else { return out };
        c[i] += 1;
        for j in i + 1..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

/// Up to SUM_WIDTH valid selections of a sum prompt, by a depth-first search over the cards in
/// `order` (the first cards first; a bounded number of steps).
fn sum_selections(m: &Msg, order: &[usize]) -> Vec<Vec<usize>> {
    let Msg::SelectSum { max, .. } = m else { return vec![] };
    fn dfs(m: &Msg, order: &[usize], start: usize, max: usize, chosen: &mut Vec<usize>, out: &mut Vec<Vec<usize>>, steps: &mut usize) {
        *steps += 1;
        if out.len() >= SUM_WIDTH || *steps > SUM_STEPS {
            return;
        }
        if !chosen.is_empty() && check_sum(m, chosen) {
            out.push(chosen.clone());
            return; // (a valid selection: not one with more cards on top)
        }
        if chosen.len() >= max {
            return;
        }
        for k in start..order.len() {
            chosen.push(order[k]);
            dfs(m, order, k + 1, max, chosen, out, steps);
            chosen.pop();
        }
    }
    let (mut out, mut steps) = (Vec::new(), 0);
    dfs(m, order, 0, (*max as usize).max(1), &mut Vec::new(), &mut out, &mut steps);
    out
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
        Msg::SelectCard { min: 1, max: 1, .. } if card_order.is_some() => {
            return card_order.unwrap().iter().filter(|&&i| i < cands.len()).map(|&i| cands[i].clone()).collect();
        }
        Msg::SelectCard { .. } | Msg::SelectUnselectCard { .. } => return candidates_in(m, card_order),
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
        Msg::SelectCard { .. } => 7,
        Msg::SelectUnselectCard { .. } => 8,
        Msg::SelectTribute { .. } => 9,
        Msg::SelectSum { .. } => 10,
        Msg::SelectPosition { .. } => 11,
        Msg::SelectPlace { .. } => 12,
        _ => 0,
    }
}

/// The prompt's name in the width statistics (`AlphaBeta::widths`).
pub fn prompt_label(m: &Msg) -> &'static str {
    match m {
        Msg::SelectIdlecmd { .. } => "idlecmd",
        Msg::SelectBattlecmd { .. } => "battlecmd",
        Msg::SelectChain { .. } => "chain",
        Msg::SelectEffectyn { .. } => "effectyn",
        Msg::SelectYesno { .. } => "yesno",
        Msg::SelectOption { .. } => "option",
        Msg::SelectCard { min: 1, max: 1, .. } => "card1",
        Msg::SelectCard { .. } => "cards",
        Msg::SelectUnselectCard { .. } => "unselect",
        Msg::SelectTribute { .. } => "tribute",
        Msg::SelectSum { .. } => "sum",
        Msg::SelectPosition { .. } => "position",
        Msg::SelectPlace { .. } => "place",
        _ => "other",
    }
}

/// The options a prompt lists before the search's caps: the cards of a selection, else the
/// candidates.
fn listed(m: &Msg, cands: usize) -> usize {
    match m {
        Msg::SelectCard { cards, .. } | Msg::SelectTribute { cards, .. } => cards.len(),
        Msg::SelectSum { cards, must_cards, .. } => cards.len() + must_cards.len(),
        Msg::SelectUnselectCard { select_cards, .. } => select_cards.len() + 1,
        _ => cands,
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
        let cards = match m {
            Msg::SelectCard { cards, .. } | Msg::SelectTribute { cards, .. } | Msg::SelectSum { cards, .. } => Some(cards),
            _ => None,
        };
        if let Some(cards) = cards {
            if ind.len() == 1 {
                code = cards.get(ind[0] as usize).map_or(0, |e| e.code as u128);
            } else {
                // (several cards: the codes in any order)
                code = ind.iter().filter_map(|&i| cards.get(i as usize)).fold(0u128, |a, e| a.wrapping_add((e.code as u128).wrapping_mul(0x9E37_79B9)));
                kind = 1 + ind.len() as u128;
            }
        }
    } else if let (Msg::SelectUnselectCard { select_cards, .. }, Some(i)) = (m, c.index) {
        (kind, code) = (1, select_cards.get(i as usize).map_or(0, |e| e.code as u128));
    } else if c.finish == Some(true) {
        kind = 2;
    } else if let Some(p) = c.position {
        kind = p as u128;
    } else if let Some(pl) = c.places.as_ref().and_then(|p| p.first()) {
        kind = (pl[1] * 16 + pl[2]) as u128;
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

/// The evaluation's terms for `me` (the differences, me minus the opponent): LP, monsters,
/// dominant monsters, hand, face-down backrow, and the deck-out flag (2 cards or fewer left to
/// draw; the opponent's minus mine). `evaluate` is their sum with the W_ weights (in its own
/// order of additions).
pub fn eval_features(d: &mut Duel, me: u32) -> [f64; 6] {
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
    let mut f = [(lp[me as usize] - lp[1 - me as usize]) as f64, mons[0].len() as f64 - mons[1].len() as f64, 0.0, 0.0, 0.0, 0.0];
    for (s, p) in [(0usize, me), (1, 1 - me)] {
        let sign = if s == 0 { 1.0 } else { -1.0 };
        let best_other = mons[1 - s].iter().map(|&(pos, a, df)| battle_stat(pos, a, df)).max().unwrap_or(-1);
        f[2] += sign * mons[s].iter().filter(|&&(pos, a, _)| pos & POS_FACEUP_ATTACK != 0 && a > best_other).count() as f64;
        f[3] += sign * d.ocg_query_count(p as u8, LOCATION_HAND) as f64;
        let backrow = parse_location_query(&query(d, p, LOCATION_SZONE, QUERY_POSITION))
            .into_iter()
            .flatten()
            .filter(|c| c.position.unwrap_or(0) & POS_FACEDOWN != 0)
            .count();
        f[4] += sign * backrow as f64;
        if d.ocg_query_count(p as u8, LOCATION_DECK) <= 2 {
            f[5] -= sign;
        }
    }
    f
}

/// The evaluation's weights, in the order of `eval_features`.
pub const EVAL_WEIGHTS: [f64; 6] = [W_LP, W_MONSTER, W_DOMINANT, W_HAND, W_BACKROW, W_DECK_OUT];

/// One player's side for the rich features (scripts/train_eval.py side_features): LP/1000, hand,
/// deck/10, grave/10, banished/10, extra/10, monsters, face-up ones, face-up attackers, face-down
/// ones, the attackers' ATK sum/1000, the best face-up battle stat/1000, face-down and face-up
/// spells/traps, a field spell, deck out (2 cards or fewer).
fn side_features(d: &mut Duel, p: u32) -> [f64; 16] {
    let mons: Vec<_> = parse_location_query(&query(d, p, LOCATION_MZONE, MZONE_QUERY)).into_iter().flatten().collect();
    let (mut up, mut atk_up, mut atk_sum, mut best) = (0, 0, 0i64, 0i32);
    for c in &mons {
        let pos = c.position.unwrap_or(0);
        if pos & POS_FACEUP != 0 {
            up += 1;
            let attacking = pos & POS_ATTACK != 0;
            best = best.max(if attacking { c.attack.unwrap_or(0) } else { c.defense.unwrap_or(0) });
            if attacking {
                atk_up += 1;
                atk_sum += c.attack.unwrap_or(0) as i64;
            }
        }
    }
    let sz = parse_location_query(&query(d, p, LOCATION_SZONE, QUERY_POSITION));
    let back = sz.iter().take(5).flatten();
    let down = back.clone().filter(|c| c.position.unwrap_or(0) & POS_FACEDOWN != 0).count();
    let n_back = back.count();
    let field = sz.get(5).is_some_and(|c| c.is_some());
    let lp = d.ext_get_info()[3 + p as usize] as i32;
    let n = |d: &mut Duel, loc: u32| d.ocg_query_count(p as u8, loc) as f64;
    let deck = n(d, LOCATION_DECK);
    [
        lp as f64 / 1000.0,
        n(d, LOCATION_HAND),
        deck / 10.0,
        n(d, LOCATION_GRAVE) / 10.0,
        n(d, LOCATION_REMOVED) / 10.0,
        n(d, LOCATION_EXTRA) / 10.0,
        mons.len() as f64,
        up as f64,
        atk_up as f64,
        (mons.len() - up) as f64,
        atk_sum as f64 / 1000.0,
        best as f64 / 1000.0,
        down as f64,
        (n_back - down) as f64,
        field as u8 as f64,
        (deck <= 2.0) as u8 as f64,
    ]
}

/// The rich features for `me` (scripts/train_eval.py rich_features): the sides' differences, mine
/// minus the opponent's, then the evaluation's dominance term.
pub fn rich_features(d: &mut Duel, me: u32) -> Vec<f64> {
    let (a, b) = (side_features(d, me), side_features(d, 1 - me));
    let mut f: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    f.push(eval_features(d, me)[2]);
    f
}

/// A trained evaluation (scripts/train_eval.py's export): linear on the raw features, or the
/// antisymmetric MLP v(z) = g(z) - g(-z), g(z) = w2 . relu(w1 z + b1), z the features divided by
/// `scale`. Its value is in eval points.
#[derive(serde::Deserialize)]
pub struct LearnedEval {
    features: String,
    model: String,
    scale: Vec<f64>,
    #[serde(default)]
    weights: Vec<f64>,
    #[serde(default)]
    w1: Vec<Vec<f64>>,
    #[serde(default)]
    b1: Vec<f64>,
    #[serde(default)]
    w2: Vec<f64>,
}

impl LearnedEval {
    pub fn value(&self, d: &mut Duel, me: u32) -> f64 {
        let x = if self.features == "rich" { rich_features(d, me) } else { eval_features(d, me).to_vec() };
        if self.model == "linear" {
            return x.iter().zip(&self.weights).map(|(a, w)| a * w).sum();
        }
        let z: Vec<f64> = x.iter().zip(&self.scale).map(|(a, s)| a / s).collect();
        let g = |sign: f64| -> f64 {
            self.w1
                .iter()
                .zip(&self.b1)
                .zip(&self.w2)
                .map(|((row, b), w2)| w2 * (row.iter().zip(&z).map(|(w, x)| w * sign * x).sum::<f64>() + b).max(0.0))
                .sum()
        };
        g(1.0) - g(-1.0)
    }

    /// The evaluation of $YGOAI_EVAL (a JSON file), if set (read once).
    pub fn from_env() -> Option<&'static LearnedEval> {
        static E: std::sync::OnceLock<Option<LearnedEval>> = std::sync::OnceLock::new();
        E.get_or_init(|| {
            let path = std::env::var("YGOAI_EVAL").ok().filter(|p| !p.is_empty())?;
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("YGOAI_EVAL {path}: {e}"));
            Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("YGOAI_EVAL {path}: {e}")))
        })
        .as_ref()
    }
}

/// The position's value for `me` (in points of 100 LP) (alphabeta.evaluate); a trained one
/// with $YGOAI_EVAL.
pub fn evaluate(d: &mut Duel, me: u32) -> f64 {
    if let Some(e) = LearnedEval::from_env() {
        return e.value(d, me);
    }
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

/// The transposition table key of a position and its pending decision: a fast hash over the
/// engine's state (where every card is, its position, status, counters, materials and equip target;
/// what the cards on the field are now, stats included; turn, phase, LP, chain) and the prompt.
/// The transposition table key of a position and its pending decision (alphabeta._tt_key).
pub fn tt_key(d: &mut Duel, m: &Msg) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    d.ext_get_info().hash(&mut h);
    for link in &d.core.current_chain {
        (link.chain_count, link.triggering_effect.map(|e| e.0), link.triggering_controler, link.triggering_location, link.triggering_sequence).hash(&mut h);
    }
    for p in 0..2 {
        let pl = &d.player[p];
        // (the piles: which card where; the zones also what the card is now, stats included)
        let piles: [&Vec<CardId>; 5] = [&pl.list_main, &pl.list_hand, &pl.list_extra, &pl.list_grave, &pl.list_remove];
        for pile in piles {
            pile.len().hash(&mut h);
            for &c in pile {
                card_state(d, c, &mut h);
            }
        }
        let zones: Vec<Option<CardId>> = pl.list_mzone.iter().chain(&pl.list_szone).copied().collect();
        for (i, z) in zones.into_iter().enumerate() {
            i.hash(&mut h);
            if let Some(c) = z {
                card_state(d, c, &mut h);
                (c.get_code(d), c.get_attack(d), c.get_defense(d), c.get_level(d), c.get_rank(d)).hash(&mut h);
            }
        }
    }
    // (the pending prompt, as its JSON text, streamed into the hash)
    struct W<'a>(&'a mut rustc_hash::FxHasher);
    impl std::io::Write for W<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.write(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let _ = serde_json::to_writer(W(&mut h), m);
    let k = h.finish();
    if k == 0 { 1 } else { k }
}

/// What the key takes of a card: which one, its position, status bits, counters, materials and
/// equip target.
fn card_state(d: &Duel, c: CardId, h: &mut rustc_hash::FxHasher) {
    use std::hash::Hash;
    let x = d.card(c);
    (x.cardid, x.current.position, x.status, &x.counters).hash(h);
    x.xyz_materials.iter().map(|m| d.card(*m).cardid).for_each(|id| id.hash(h));
    x.equiping_target.map(|t| d.card(t).cardid).hash(h);
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

/// Statistics of the expanded nodes (nodes that got to their moves, not cut by the TT).
#[derive(Default)]
pub struct NodeStats {
    /// by (prompt, options listed, children after the caps): [nodes, children searched]
    pub widths: BTreeMap<(&'static str, u16, u16), [u64; 2]>,
    /// by (prompt, TT state, children, the move index of the cutoff or `NO_CUT`): nodes. The TT
    /// state: 0 no entry, 1 an entry without a move, else 2 + its flag (exact, lower, upper).
    pub cuts: BTreeMap<(&'static str, u8, u8, u8), u64>,
    /// by (prompt, move index): [reduced searches by LMR, of them searched again]
    pub lmr: BTreeMap<(&'static str, u8), [u64; 2]>,
    /// by (prompt, depth): [nodes, nodes in their subtrees (themselves included)]
    pub subtree: BTreeMap<(&'static str, i8), [u64; 2]>,
    /// (`budget`) by (prompt, TT state, the budget's depth in plies): [nodes, their budgets'
    /// sum, the nodes their subtrees used]
    pub calib: BTreeMap<(&'static str, u8, i8), [f64; 3]>,
}

/// (`NodeStats::cuts`: no cutoff, all the children searched)
pub const NO_CUT: u8 = 255;

/// The budget search's prior cutoff rates (as 20 pseudo-nodes): at the first move by the TT
/// state (none, no move, exact, lower, upper; measured with `YGOAI_STATS`), 3% at later moves.
fn hazard_priors() -> Vec<[f64; 2]> {
    const FIRST: [f64; 5] = [0.35, 0.35, 0.15, 0.9, 0.02];
    let mut h = vec![[0.0; 2]; HAZARD_TYPES * 5 * HAZARD_MOVES];
    for (i, e) in h.iter_mut().enumerate() {
        let (tt, mv) = ((i / HAZARD_MOVES) % 5, i % HAZARD_MOVES);
        *e = [20.0, 20.0 * if mv == 0 { FIRST[tt] } else { 0.03 }];
    }
    h
}

fn hazard_slot(t: usize, tt: u8, i: usize) -> usize {
    (t.min(HAZARD_TYPES - 1) * 5 + tt as usize) * HAZARD_MOVES + i.min(HAZARD_MOVES - 1)
}

/// The search's tables, kept from one decision to the next (an agent holds one).
pub struct AlphaBeta {
    tt: Vec<TtEntry>,
    tt_gen: u16,
    hist: Vec<f64>,
    /// continuation history (with `conthist`)
    cont: Vec<f64>,
    pub nodes: u64,
    /// transposition table lookups, and those that found the position
    pub tt_probes: u64,
    pub tt_hits: u64,
    /// (lines the core could not play: the search avoids them)
    pub broken: u64,
    /// (if set, statistics of the expanded nodes, see `NodeStats`)
    pub stats: Option<NodeStats>,
    /// (`budget`, kept across searches: by (prompt type, TT state, move index) [nodes that
    /// searched the move, cut off by it]; by move index [reduced searches, searched again])
    hazard: Vec<[f64; 2]>,
    research: [[f64; 2]; HAZARD_MOVES],
    /// the root value of the last search's last completed iteration (NaN if none completed)
    pub last_value: f64,
}

impl Default for AlphaBeta {
    fn default() -> Self {
        AlphaBeta {
            tt: vec![TtEntry::default(); 1 << TT_BITS],
            tt_gen: 0,
            hist: vec![0.0; 1 << HIST_BITS],
            cont: vec![0.0; 1 << CONT_BITS],
            nodes: 0,
            tt_probes: 0,
            tt_hits: 0,
            broken: 0,
            stats: None,
            hazard: hazard_priors(),
            research: [[50.0, 1.0]; HAZARD_MOVES],
            last_value: f64::NAN,
        }
    }
}

/// One search: the rules' players, the deadline and the node limit.
struct Ctx<'a> {
    ab: &'a mut AlphaBeta,
    policies: [Heuristic; 2],
    deadline: Option<Instant>,
    /// (in the count of `ab.nodes`)
    node_limit: Option<u64>,
    f: Features,
    root_player: u32,
    /// the history slot of the move that led to the node being searched (`conthist`)
    prev_key: usize,
    /// on the line of first moves, and whether it was extended already (`pv_ext`)
    pv: bool,
    pv_extended: bool,
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
        for h in self.hist.iter_mut().chain(self.cont.iter_mut()) {
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
        let f = Features::from_env();
        let mut ctx = Ctx { ab: self, policies, deadline, node_limit, f, root_player: player as u32, prev_key: 0, pv: false, pv_extended: false };
        let card_order = ctx.card_order(core_prompt);
        let pref = ordered(core_prompt, turn_player, card_order.as_deref());
        let mut order: Vec<usize> = (0..core_cands.len()).collect();
        order.sort_by_key(|&k| pref.iter().position(|c| *c == core_cands[k]).unwrap_or(pref.len()));
        if matches!(core_prompt, Msg::SelectCard { min: 1, max: 1, .. }) {
            order.truncate(ROOT_CARD_WIDTH);
        } else if is_inner(core_prompt) || matches!(core_prompt, Msg::SelectCard { .. }) {
            // (a decision inside a move: the rules' first choices only, as below the root)
            let known = core_cands.iter().filter(|c| pref.contains(c)).count().max(1);
            order.truncate(known);
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
    fn deepen(&mut self, duel: &Duel, prompt: &Msg, cands: &[Response], mut order: Vec<usize>, t0: Instant, budget: Option<f64>, fixed_depth: bool) -> (usize, i32) {
        let (mut best, mut done) = (order[0], 0);
        self.ab.last_value = f64::NAN;
        let max_depth = if fixed_depth {
            DEPTH_NO_CLOCK
        } else if self.f.budget {
            BUDGET_ITERATIONS
        } else {
            MAX_DEPTH
        };
        for depth in 1..=max_depth {
            if let Some(b) = budget
                && depth > 1
                && t0.elapsed().as_secs_f64() > NEXT_ITERATION_SHARE * b
            {
                break;
            }
            let mut values: Vec<Option<f64>> = vec![None; cands.len()];
            let (mut alpha, mut it_best) = (f64::NEG_INFINITY, None);
            // (each root move's depth, or its budget)
            let sub = if self.f.budget {
                let bp = BudgetParams::from_env();
                (bp.root * bp.growth.powi(depth - 1) * BUDGET_UNIT as f64).min(1e9) as i32
            } else {
                depth - 1
            };
            let mut aborted = false;
            for (j, &k) in order.iter().enumerate() {
                (self.pv, self.pv_extended) = (j == 0, false);
                let mut r = if self.f.pvs && j > 0 && alpha.is_finite() {
                    // (PVS: does it beat the best so far? only then its value)
                    self.child(duel, prompt, &cands[k], sub, alpha, alpha + PVS_EPS, 1)
                } else {
                    self.child(duel, prompt, &cands[k], sub, alpha, f64::INFINITY, 1)
                };
                if self.f.pvs && j > 0 && alpha.is_finite() && matches!(r, Ok(v) if v > alpha) {
                    r = self.child(duel, prompt, &cands[k], sub, alpha, f64::INFINITY, 1);
                }
                match r {
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
            self.ab.last_value = values[b].unwrap();
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
            Err(Stop::Broken) if !self.timed_out() => {
                self.ab.broken += 1;
                if std::env::var_os("YGOAI_DEBUG_BROKEN").is_some() {
                    eprintln!("broken: {} {cand:?}", prompt.name());
                }
                Ok(-WIN) // (the player to move avoids the line)
            }
            Err(_) => Err(Stop::Abort),
            Ok(v) => Ok(v),
        }
    }

    /// Whether a decision with this depth (or budget) and `n` moves is past the horizon (with
    /// `budget`: the budget does not cover the node and the moves it expects to search, each at
    /// least one node).
    fn horizon(&self, depth: i32, prompt: &Msg, n: usize) -> bool {
        if !self.f.budget {
            return depth <= 0;
        }
        let t = type_id(prompt) as usize;
        let mut reach = 1.0;
        let mut need = BUDGET_UNIT as f64;
        for i in 0..n.min(MAX_CHILDREN) {
            need += reach * BUDGET_UNIT as f64;
            let h = self.ab.hazard[hazard_slot(t, 0, i)];
            reach *= 1.0 - h[1] / h[0];
        }
        (depth as f64) < need
    }

    /// The depth in plies (with `budget`: of a search with the budget, log(budget) / log(growth)).
    fn eq_depth(&self, depth: i32) -> f64 {
        if self.f.budget {
            (depth as f64 / BUDGET_UNIT as f64).max(1.0).ln() / BudgetParams::from_env().growth.ln()
        } else {
            depth as f64
        }
    }

    /// The depth as the TT keeps it (with `budget`: eighths of `eq_depth`).
    fn tt_depth(&self, depth: i32) -> i32 {
        if self.f.budget { (self.eq_depth(depth) * 8.0).round() as i32 } else { depth }
    }

    /// The budget search: the budgets of a node's `n` moves (in their order) when the node has
    /// `budget`. The node itself costs one node; the rest is what the moves are expected to use:
    /// move i gets a share s * w_i, w_i = (i + 1)^-a (Stockfish's log(depth) * log(move number)
    /// / k reduction as a factor on the budget, by the confidence in the ordering given the TT
    /// state), and s is such that the expected cost, sum over the moves of P(searched) * (s * w_i
    /// + P(a reduced move is searched again) * s * w_0), is the budget. P(searched) comes from
    /// the cutoff rates before it (online, by prompt type, TT state and move index).
    fn allocate(&self, prompt: &Msg, tt_state: u8, budget: i32, n: usize) -> Vec<i32> {
        let bp = BudgetParams::from_env();
        let avail = (budget - BUDGET_UNIT).max(0) as f64;
        let a = bp.growth.ln() * self.eq_depth(budget).max(1.0).ln() / bp.k * bp.conf[tt_state as usize];
        let t = type_id(prompt) as usize;
        let w: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).powf(-a)).collect();
        // (P(searched), P(searched again) of each move)
        let mut p = Vec::with_capacity(n);
        let mut reach = 1.0;
        for (i, &wi) in w.iter().enumerate() {
            let re = if i > 0 && wi < 0.75 {
                let r = self.ab.research[i.min(HAZARD_MOVES - 1)];
                r[1] / r[0]
            } else {
                0.0
            };
            p.push((reach, re));
            let h = self.ab.hazard[hazard_slot(t, tt_state, i)];
            reach *= 1.0 - h[1] / h[0];
        }
        // (a move costs one node at least: the expected cost with the factor s, increasing in s)
        let unit = BUDGET_UNIT as f64;
        let cost = |s: f64| -> f64 {
            w.iter().zip(&p).map(|(&wi, &(reach, re))| reach * ((s * wi).max(unit) + re * s.max(unit))).sum()
        };
        let (mut lo, mut hi) = (0.0, avail.max(unit));
        if cost(hi) < avail {
            hi = avail * 2.0;
            while cost(hi) < avail && hi < 1e12 {
                hi *= 2.0;
            }
        }
        for _ in 0..30 {
            let mid = (lo + hi) / 2.0;
            if cost(mid) < avail { lo = mid } else { hi = mid }
        }
        w.iter().map(|wi| (lo * wi).round().min(1e9) as i32).collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn node(&mut self, duel: &mut Duel, prompt: &Msg, depth: i32, mut alpha: f64, mut beta: f64, ply: i32) -> Result<f64, Stop> {
        // (moves are indices into the heuristic's order: the TT keeps one, the history reorders them)
        let card_order = self.card_order(prompt);
        let mut cands = ordered(prompt, duel.ext_get_info()[1], card_order.as_deref());
        if matches!(prompt, Msg::SelectCard { min: 1, max: 1, .. }) {
            cands.truncate(CARD_WIDTH);
        }
        let key = tt_key(duel, prompt);
        let slot = (key & ((1 << TT_BITS) - 1)) as usize;
        let e = self.ab.tt[slot];
        let mut tt_move = None;
        let mut tt_state = 0u8;
        let nodes0 = self.ab.nodes;
        self.ab.tt_probes += 1;
        if e.key == key && e.gen_ == self.ab.tt_gen {
            self.ab.tt_hits += 1;
            tt_state = if e.mv >= 0 { 2 + e.flag as u8 } else { 1 };
            let v = e.value;
            if e.depth as i32 >= self.tt_depth(depth) {
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
        let f = self.f;
        let mover = prompt.player().unwrap_or(0);
        if f.rfp && !f.budget && beta - alpha <= PVS_EPS && depth <= RFP_DEPTH && beta.abs() < WIN / 2.0 {
            // (reverse futility pruning: so far ahead that the moves would not bring it below beta)
            let s = evaluate(duel, mover);
            if s - RFP_MARGIN * depth as f64 >= beta {
                return Ok(s);
            }
        }
        let static_v = if (f.futility || f.rfp_kind) && !f.budget && depth == 1 { Some(evaluate(duel, mover)) } else { None };
        if f.rfp_kind
            && let Some(s) = static_v
            && beta.is_finite()
            && let Some(low) = cands.iter().filter_map(|c| rfp_low(&move_kind(prompt, c))).reduce(f64::max)
            && s + low >= beta
        {
            return Ok(s + low); // (reverse futility: a move keeps this much at least)
        }
        let (alpha0, beta0) = (alpha, beta);
        let keys: Vec<usize> = cands.iter().map(|c| move_key(prompt, c)).collect();
        let prev = self.prev_key;
        let ckeys: Vec<usize> = if f.conthist { keys.iter().map(|&k| cont_key(prev, k)).collect() } else { Vec::new() };
        let score = |ab: &AlphaBeta, k: usize| ab.hist[keys[k]] + if f.conthist { ab.cont[ckeys[k]] } else { 0.0 };
        let mut idx: Vec<usize> = (0..cands.len()).collect();
        // (the TT move first, then the moves with a better history; the heuristic's order among equals)
        idx.sort_by(|&a, &b| {
            ((Some(a) != tt_move), -score(self.ab, a))
                .partial_cmp(&((Some(b) != tt_move), -score(self.ab, b)))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(MAX_CHILDREN);
        let (node_pv, node_ext) = (self.pv, self.pv_extended);
        let (mut best, mut best_i) = (f64::NEG_INFINITY, idx[0]);
        let (mut visited, mut cut_at) = (0, NO_CUT);
        let alloc = if f.budget { self.allocate(prompt, tt_state, depth, idx.len()) } else { Vec::new() };
        for (i, &ci) in idx.iter().enumerate() {
            visited = i + 1;
            let c = &cands[ci];
            if let Some(s) = static_v
                && f.futility
                && let Some(m) = futility_margin(&move_kind(prompt, c))
                && s + m <= alpha
            {
                // (futile: it cannot raise alpha; its bound counts as its value)
                if s + m > best {
                    (best, best_i) = (s + m, ci);
                }
                continue;
            }
            (self.pv, self.pv_extended) = (node_pv && i == 0, node_ext);
            let last = i == idx.len() - 1;
            let mut v = None;
            if f.budget {
                let (b, full) = (alloc[i], alloc[0]);
                v = Some(if i > 0 && (b as i64) * 4 < (full as i64) * 3 {
                    // (a reduced share; the first move's share after all if it beats alpha)
                    let r = self.child(duel, prompt, c, b, alpha, beta, ply + 1)?;
                    let slot = i.min(HAZARD_MOVES - 1);
                    self.ab.research[slot][0] += 1.0;
                    self.ab.research[slot][1] += (r > alpha) as u8 as f64;
                    if let Some(st) = &mut self.ab.stats {
                        let e = st.lmr.entry((prompt_label(prompt), i.min(15) as u8)).or_default();
                        e[0] += 1;
                        e[1] += (r > alpha) as u64;
                    }
                    if r <= alpha { r } else { self.search_move(duel, prompt, c, full, alpha, beta, ply + 1, last)? }
                } else {
                    self.search_move(duel, prompt, c, b, alpha, beta, ply + 1, last)?
                });
            } else if (f.lmr || f.lmr_noresearch) && depth >= LMR_DEPTH && i >= LMR_MOVES {
                // (a late move, unlikely the best by the move ordering: one ply less; with lmr
                // the full depth after all if it beats alpha)
                if f.lmr_noresearch {
                    v = Some(self.search_move(duel, prompt, c, depth - 2, alpha, beta, ply + 1, last)?);
                } else {
                    let r = self.child(duel, prompt, c, depth - 2, alpha, beta, ply + 1)?;
                    if r <= alpha {
                        v = Some(r);
                    }
                    if let Some(st) = &mut self.ab.stats {
                        let e = st.lmr.entry((prompt_label(prompt), i.min(15) as u8)).or_default();
                        e[0] += 1;
                        e[1] += (r > alpha) as u64;
                    }
                }
            }
            let v = match v {
                Some(v) => v,
                None if f.pvs && i > 0 && beta - alpha > PVS_EPS && alpha.is_finite() => {
                    // (PVS: a later move only has to show it does not beat alpha; if it does,
                    // its value with the full window)
                    let r = self.child(duel, prompt, c, depth - 1, alpha, alpha + PVS_EPS, ply + 1)?;
                    if alpha < r && r < beta { self.search_move(duel, prompt, c, depth - 1, alpha, beta, ply + 1, last)? } else { r }
                }
                None => self.search_move(duel, prompt, c, depth - 1, alpha, beta, ply + 1, last)?,
            };
            if v > best {
                (best, best_i) = (v, ci);
            }
            alpha = alpha.max(best);
            if alpha >= beta {
                // (a cutoff: this move up, the ones tried before it down)
                let d = self.eq_depth(depth);
                let bonus = d * d;
                self.ab.hist[keys[ci]] += bonus;
                for &k in &idx[..i] {
                    self.ab.hist[keys[k]] -= bonus / i as f64;
                }
                if f.conthist {
                    self.ab.cont[ckeys[ci]] += bonus;
                    for &k in &idx[..i] {
                        self.ab.cont[ckeys[k]] -= bonus / i as f64;
                    }
                }
                cut_at = i as u8;
                break;
            }
        }
        if f.budget {
            let t = type_id(prompt) as usize;
            for j in 0..visited {
                let e = &mut self.ab.hazard[hazard_slot(t, tt_state, j)];
                e[0] += 1.0;
                e[1] += (cut_at as usize == j) as u8 as f64;
            }
        }
        let eq_d = if f.budget { self.eq_depth(depth).round() as i32 } else { depth };
        if let Some(st) = &mut self.ab.stats {
            let label = prompt_label(prompt);
            let e = st.widths.entry((label, listed(prompt, cands.len()) as u16, idx.len() as u16)).or_default();
            e[0] += 1;
            e[1] += visited as u64;
            *st.cuts.entry((label, tt_state, idx.len() as u8, cut_at)).or_default() += 1;
            let d = eq_d;
            let e = st.subtree.entry((label, d.clamp(-1, 60) as i8)).or_default();
            e[0] += 1;
            e[1] += self.ab.nodes - nodes0 + 1;
            if f.budget {
                let e = st.calib.entry((label, tt_state, d.clamp(-1, 60) as i8)).or_default();
                e[0] += 1.0;
                e[1] += depth as f64 / BUDGET_UNIT as f64;
                e[2] += (self.ab.nodes - nodes0 + 1) as f64;
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
            self.ab.tt[slot] = TtEntry { key, value: best, depth: self.tt_depth(depth) as i16, flag, mv: best_i as i8, gen_: self.ab.tt_gen };
        }
        Ok(best)
    }

    /// The value of `cand`: on a copy of the duel, or for the node's last search on the duel
    /// itself (nothing needs this position afterwards).
    #[allow(clippy::too_many_arguments)]
    fn search_move(&mut self, duel: &mut Duel, prompt: &Msg, cand: &Response, depth: i32, alpha: f64, beta: f64, ply: i32, last: bool) -> Result<f64, Stop> {
        if !last {
            return self.child(duel, prompt, cand, depth, alpha, beta, ply);
        }
        if self.timed_out() {
            return Err(Stop::Abort);
        }
        self.ab.nodes += 1;
        self.expand(duel, prompt.clone(), cand, depth, alpha, beta, ply)
    }

    /// Quiescence (`qsearch`): past the nominal depth, battle and chain decisions go on (attacks
    /// and responses change the position the most, like captures in chess). In a Battle Phase
    /// the player may stand pat (the static value: it need not attack), else only its attacks
    /// are searched; in a chain window every answer, passing included.
    fn qnode(&mut self, duel: &mut Duel, prompt: &Msg, depth: i32, mut alpha: f64, beta: f64, ply: i32) -> Result<f64, Stop> {
        let mut cands = ordered(prompt, duel.ext_get_info()[1], None);
        let mut best = f64::NEG_INFINITY;
        if matches!(prompt, Msg::SelectBattlecmd { .. }) {
            best = evaluate(duel, prompt.player().unwrap_or(0));
            if best >= beta {
                return Ok(best);
            }
            alpha = alpha.max(best);
            cands.retain(|c| c.kind.as_deref() == Some("attack"));
        }
        cands.truncate(MAX_CHILDREN);
        for (i, c) in cands.iter().enumerate() {
            let v = self.search_move(duel, prompt, c, depth - 1, alpha, beta, ply + 1, i == cands.len() - 1)?;
            best = best.max(v);
            alpha = alpha.max(best);
            if alpha >= beta {
                break;
            }
        }
        Ok(best)
    }

    /// Plays `cand`, runs to the next ply and values it, for the player who plays it (negamax:
    /// values and the window (alpha, beta) are the mover's; they change sign only where the next
    /// ply is the other player's, which in this game need not be the case).
    #[allow(clippy::too_many_arguments)]
    fn expand(&mut self, duel: &mut Duel, mut prompt: Msg, cand: &Response, mut depth: i32, alpha: f64, beta: f64, ply: i32) -> Result<f64, Stop> {
        let me = prompt.player().unwrap_or(0);
        if self.f.conthist {
            self.prev_key = move_key(&prompt, cand); // (for the continuation history of the next node)
        }
        duel.ocg_set_response(&encode(&prompt, cand).map_err(|_| Stop::Broken)?);
        for _ in 0..MAX_AUTO_STEPS {
            let step = run_core(duel, &prompt)?;
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
            } else if (matches!(prompt, Msg::SelectCard { .. }) || is_inner(&prompt)) && self.horizon(depth, &prompt, candidates(&prompt).len()) {
                // (at the horizon a decision inside a move is the rules': the move completes first)
                self.policies[player as usize].respond_to(&prompt, None).map_err(|_| Stop::Broken)?
            } else if is_searched(&prompt) {
                let cands = candidates(&prompt);
                if cands.len() > 1 {
                    if !self.f.budget && depth <= 0 && self.f.pv_ext && self.pv && !self.pv_extended && player != self.root_player {
                        (self.pv_extended, depth) = (true, 1); // (the PV extension: their reply to the main line)
                    }
                    if self.horizon(depth, &prompt, cands.len()) {
                        let quiet = match &prompt {
                            Msg::SelectChain { .. } => false,
                            Msg::SelectBattlecmd { .. } => !cands.iter().any(|c| c.kind.as_deref() == Some("attack")),
                            _ => true,
                        };
                        if self.f.qsearch && !self.f.budget && depth > -MAX_QPLY && !quiet {
                            if player == me {
                                return self.qnode(duel, &prompt, depth, alpha, beta, ply);
                            }
                            return Ok(-self.qnode(duel, &prompt, depth, -beta, -alpha, ply)?);
                        }
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
