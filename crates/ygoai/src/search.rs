//! Alpha-beta search with iterative deepening on a (reconstructed) duel: agents/alphabeta.py's
//! search, on copies of the duel. See the Python module for the design (plies, move ordering,
//! transposition table, history, negamax, time, evaluation); this is the same search and makes
//! the same choices at a fixed depth (its transposition table hashes the same bytes the same
//! way, the rules' players draw the same random numbers).

use std::time::Instant;

use ygo_core::api::QueryInfo;
use ygo_core::constants::*;
use ygo_core::duel::Duel;
use ygo_core::types::CardId;

use crate::actions::{Response, battle_options, encode, idle_options};
use crate::game::run_until_prompt;
use crate::heuristic::{Heuristic, db_attack};
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

/// Search features under test, each off unless named in $YGOAI_SEARCH (a comma list): `pvs`,
/// `rfp` (reverse futility pruning at null-window nodes; with pvs), `rfp_kind` (at depth 1, by
/// the least a move of each kind keeps), `futility` (depth 1, by the most a move of each kind
/// gains), `lmr`, `lmr_noresearch`, `conthist` (continuation history), `pv_ext` (the main
/// line sees the opponent's reply), `qsearch` (battle and chain decisions past the depth).
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
                _ => panic!("YGOAI_SEARCH: unknown feature {name}"),
            }
        }
        f
    }

    /// The features of $YGOAI_SEARCH (read once).
    pub fn from_env() -> Features {
        static F: std::sync::OnceLock<Features> = std::sync::OnceLock::new();
        *F.get_or_init(|| Features::parse(&std::env::var("YGOAI_SEARCH").unwrap_or_default()))
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
            for (j, &k) in order.iter().enumerate() {
                (self.pv, self.pv_extended) = (j == 0, false);
                let mut r = if self.f.pvs && j > 0 && alpha.is_finite() {
                    // (PVS: does it beat the best so far? only then its value)
                    self.child(duel, prompt, &cands[k], depth - 1, alpha, alpha + PVS_EPS, 1)
                } else {
                    self.child(duel, prompt, &cands[k], depth - 1, alpha, f64::INFINITY, 1)
                };
                if self.f.pvs && j > 0 && alpha.is_finite() && matches!(r, Ok(v) if v > alpha) {
                    r = self.child(duel, prompt, &cands[k], depth - 1, alpha, f64::INFINITY, 1);
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
        self.ab.tt_probes += 1;
        if e.key == key && e.gen_ == self.ab.tt_gen {
            self.ab.tt_hits += 1;
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
        let f = self.f;
        let mover = prompt.player().unwrap_or(0);
        if f.rfp && beta - alpha <= PVS_EPS && depth <= RFP_DEPTH && beta.abs() < WIN / 2.0 {
            // (reverse futility pruning: so far ahead that the moves would not bring it below beta)
            let s = evaluate(duel, mover);
            if s - RFP_MARGIN * depth as f64 >= beta {
                return Ok(s);
            }
        }
        let static_v = if (f.futility || f.rfp_kind) && depth == 1 { Some(evaluate(duel, mover)) } else { None };
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
        for (i, &ci) in idx.iter().enumerate() {
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
            if (f.lmr || f.lmr_noresearch) && depth >= LMR_DEPTH && i >= LMR_MOVES {
                // (a late move, unlikely the best by the move ordering: one ply less; with lmr
                // the full depth after all if it beats alpha)
                if f.lmr_noresearch {
                    v = Some(self.search_move(duel, prompt, c, depth - 2, alpha, beta, ply + 1, last)?);
                } else {
                    let r = self.child(duel, prompt, c, depth - 2, alpha, beta, ply + 1)?;
                    if r <= alpha {
                        v = Some(r);
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
                let bonus = (depth * depth) as f64;
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
                    if depth <= 0 && self.f.pv_ext && self.pv && !self.pv_extended && player != self.root_player {
                        (self.pv_extended, depth) = (true, 1); // (the PV extension: their reply to the main line)
                    }
                    if depth <= 0 {
                        let quiet = match &prompt {
                            Msg::SelectChain { .. } => false,
                            Msg::SelectBattlecmd { .. } => !cands.iter().any(|c| c.kind.as_deref() == Some("attack")),
                            _ => true,
                        };
                        if self.f.qsearch && depth > -MAX_QPLY && !quiet {
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
