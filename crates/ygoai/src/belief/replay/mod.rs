//! Belief replay: reconstruct a duel that looks identical from the viewer's side
//! (belief/replay.py; see there for the method).
//!
//! The port keeps the Python module's structure and names. Differences of form: exceptions are
//! [`RErr`] values, repairs are [`Fix`] values, probes run on copies of the replay and its duel
//! (instead of forked processes; a counter shared by the copies bounds them as the shared memory
//! did), and the diagnosis code reads JSON views of messages where the Python code reads dicts.

pub mod decide;
pub mod mismatch;
pub mod step;
pub mod util;

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use ygo_core::constants::*;

use crate::actions::Response;
use crate::belief::completion::{Completion, VanillaCompletion};
use crate::belief::infer::LocMap;
use crate::belief::tracker::{Hypothesis, PoolLoc, Tracker, TrackerError, VarId};
use crate::game::{Deck, DuelConfig, InitInfo, ObsItem, setup_duel};
use crate::messages::Msg;
use crate::rduel::RDuel;
use crate::state::Snapshot;

pub const STATUS_END: i32 = 0;
pub const STATUS_AWAITING: i32 = 1;
pub const STATUS_CONTINUE: i32 = 2;

/// The outcomes of a probe (the exit codes of the Python module's forked probes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    Ok,
    Failed,
    Timeout,
    Repairable,
    Progressed,
    /// repairable, but at the very first item after the decision
    RepairableAtOnce,
}

/// messages after which moves no longer result from the current selection
pub fn selection_end(t: &str) -> bool {
    matches!(
        t,
        "CHAINING" | "CHAINED" | "CHAIN_SOLVING" | "CHAIN_SOLVED" | "CHAIN_END" | "NEW_PHASE" | "NEW_TURN" | "WIN" | "SUMMONING" | "SPSUMMONING" | "FLIPSUMMONING"
    )
}

pub const HIDDEN_PILES: [u32; 3] = [LOCATION_DECK, LOCATION_HAND, LOCATION_EXTRA];

pub fn hidden_pile(loc: u32) -> bool {
    HIDDEN_PILES.contains(&loc)
}

/// where an action's consequences are over (a probe decides only after one)
pub fn boundary(t: &str) -> bool {
    matches!(t, "CHAIN_SOLVED" | "CHAIN_END" | "NEW_PHASE" | "NEW_TURN" | "SUMMONED" | "SPSUMMONED" | "FLIPSUMMONED" | "CHAINED")
}

/// messages by which the real game goes on after an effect did (nothing) more
pub fn nothing_more(t: &str) -> bool {
    matches!(t, "CHAIN_SOLVED" | "CHAIN_END" | "PLAYER_HINT" | "NEW_PHASE" | "NEW_TURN")
}

pub const SUMMON_SELECT_MSGS: [u64; 2] = [508, 509];
pub const BATTLE_START_EVENT: u64 = 28;
pub const CARD_PROMPTS: [&str; 4] = ["SELECT_CARD", "SELECT_UNSELECT_CARD", "SELECT_TRIBUTE", "SELECT_SUM"];

pub fn shuffle_loc(t: &str) -> Option<u32> {
    match t {
        "SHUFFLE_HAND" => Some(LOCATION_HAND),
        "SHUFFLE_DECK" => Some(LOCATION_DECK),
        "SHUFFLE_EXTRA" => Some(LOCATION_EXTRA),
        _ => None,
    }
}

/// status bits an operation in progress sets on the cards it was given
pub const OPERATION_STATUS: u32 = STATUS_DESTROY_CONFIRMED | STATUS_LEAVE_CONFIRMED | STATUS_BATTLE_DESTROYED;
pub const PROPERTY_TRIES: usize = 24;
pub const PROPERTY_BATCH: usize = 8;
pub const MAX_SELECTIONS: usize = 30;
pub const DECISION_WINDOW: usize = 10;
pub const TAIL_RESTARTS: usize = 4;
pub const PROBE_BUDGET: u64 = 48;
pub const SUSPECT_TESTS: usize = 150;
pub const SUSPECT_ROUNDS: usize = 3;
pub const PROBE_STEPS: usize = 2000;

fn decision_tries() -> usize {
    std::env::var("YGOAI_DECISION_TRIES").ok().and_then(|v| v.parse().ok()).unwrap_or(24)
}

fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| !v.is_empty())
}

/// A control point: (stream position, process calls since it was reached).
pub type Cp = (usize, usize);
/// A decision: (stream position, decisions at it before).
pub type Dkey = (usize, usize);
/// A forcing: (player, code, description).
pub type FKey = (u32, u32, u64);
/// An assumption: (position, con, loc, seq, code).
pub type Asm = (usize, u32, u32, u32, u32);
/// An exclusion: (position, con, loc, slots, code); slots [-1] for every slot of the pile.
pub type Excl = (usize, u32, u32, Vec<i64>, u32);
pub type Slot = (u32, u32, u32);

pub const ALL_SLOTS: i64 = -1;

/// What an identity says of the card in a slot.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Same {
    /// the physical card of that (earlier) var
    Var(VarId),
    /// one of the cards revealed before the shuffle
    Revealed,
    /// not the physical card of that var
    Not(VarId),
}

pub type Ident = (usize, Slot, Same);

/// A repair (the Python module's fix tuples).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Fix {
    Force(FKey),
    ForceLater(FKey),
    Unforce(FKey),
    Assume(Asm),
    AssumeMany(Vec<Asm>),
    Unassume(Asm),
    Exclude(Excl),
    Swap(Cp, u32, u32),
    IdentifyAll(Vec<Ident>),
    Identify(Ident),
    NomatchCard(usize, usize, Slot),
    Unflip(usize, usize),
    NomatchPile(usize, usize),
    MatchPile(usize, usize),
    NomatchLoc(usize, usize, u32),
    NomatchId(usize, usize, u32),
    Respond(Dkey, Vec<usize>),
    Decide(Dkey, usize),
    Match(Cp, u32),
    CodeTest(usize, usize, Slot, Vec<u32>),
    MatchSet((usize, usize), Vec<u32>, Vec<u32>),
    Snapshot(Cp),
}

impl Fix {
    pub fn kind(&self) -> &'static str {
        match self {
            Fix::Force(..) => "force",
            Fix::ForceLater(..) => "forcelater",
            Fix::Unforce(..) => "unforce",
            Fix::Assume(..) => "assume",
            Fix::AssumeMany(..) => "assume_many",
            Fix::Unassume(..) => "unassume",
            Fix::Exclude(..) => "exclude",
            Fix::Swap(..) => "swap",
            Fix::IdentifyAll(..) => "identify_all",
            Fix::Identify(..) => "identify",
            Fix::NomatchCard(..) => "nomatchcard",
            Fix::Unflip(..) => "unflip",
            Fix::NomatchPile(..) => "nomatchpile",
            Fix::MatchPile(..) => "matchpile",
            Fix::NomatchLoc(..) => "nomatchloc",
            Fix::NomatchId(..) => "nomatchid",
            Fix::Respond(..) => "respond",
            Fix::Decide(..) => "decide",
            Fix::Match(..) => "match",
            Fix::CodeTest(..) => "codetest",
            Fix::MatchSet(..) => "matchset",
            Fix::Snapshot(..) => "snapshot",
        }
    }
}

/// What a divergence's expected / got side is (the Python module passes items, messages, lists
/// of messages, prompts and state views).
#[derive(Debug, Clone)]
pub enum DV {
    Item(ObsItem),
    Msg(Msg),
    Msgs(Vec<Msg>),
    State(Snapshot),
}

impl DV {
    /// The dict a side shows (an item's message or prompt; None for a list).
    pub fn dict(&self) -> Option<serde_json::Value> {
        match self {
            DV::Item(ObsItem::Msg(m)) | DV::Item(ObsItem::Prompt(m, _)) | DV::Msg(m) => Some(util::jv(m)),
            DV::Item(ObsItem::Response(r)) => Some(util::jv(r)),
            DV::Msgs(_) => None,
            DV::State(s) => Some(util::jv(s)),
        }
    }
    /// Whether it is an item (a tuple in the Python module).
    pub fn is_item(&self) -> bool {
        matches!(self, DV::Item(_))
    }
}

#[derive(Debug, Clone)]
pub struct Divergence {
    pub pos: usize,
    pub reason: String,
    pub expected: Option<DV>,
    pub got: Option<DV>,
    pub fix: Option<Fix>,
    /// a repair for when no decision change helps
    pub fallback: Option<Fix>,
}

impl Divergence {
    pub fn new(pos: usize, reason: impl Into<String>) -> Divergence {
        Divergence { pos, reason: reason.into(), expected: None, got: None, fix: None, fallback: None }
    }
    pub fn with_fix(pos: usize, reason: impl Into<String>, fix: Option<Fix>) -> Divergence {
        Divergence { fix, ..Divergence::new(pos, reason) }
    }
    pub fn text(&self) -> String {
        format!("divergence at item {}: {}", self.pos, self.reason)
    }
}

/// The replay's exceptions.
#[derive(Debug, Clone)]
pub enum RErr {
    Div(Box<Divergence>),
    Tracker(TrackerError),
    /// a probe verified enough of the stream (_ProbeSuccess)
    ProbeSuccess,
    /// a probe ran out of steps or time (the forked probe's exit 3)
    ProbeTimeout,
    /// any other error (the Python module's KeyError, InvalidResponse, ...)
    Other(String),
}

impl From<Divergence> for RErr {
    fn from(d: Divergence) -> RErr {
        RErr::Div(Box::new(d))
    }
}

impl From<TrackerError> for RErr {
    fn from(e: TrackerError) -> RErr {
        RErr::Tracker(e)
    }
}

impl std::fmt::Display for RErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RErr::Div(d) => f.write_str(&d.text()),
            RErr::Tracker(e) => write!(f, "TrackerError: {}", e.0),
            RErr::ProbeSuccess => f.write_str("probe success"),
            RErr::ProbeTimeout => f.write_str("probe timeout"),
            RErr::Other(s) => f.write_str(s),
        }
    }
}

pub type R<T> = Result<T, RErr>;

pub fn div<T>(d: Divergence) -> R<T> {
    Err(RErr::Div(Box::new(d)))
}

/// replay._divergence_sig: what a divergence is about.
pub type DivSig = (usize, String, Option<(String, Option<i64>, String)>, Option<(String, Option<i64>, String)>);

pub fn divergence_sig(e: &Divergence) -> DivSig {
    use util::V;
    let kind = |x: &Option<DV>| -> Option<(String, Option<i64>, String)> {
        let d = x.as_ref()?.dict()?;
        if !d.is_object() {
            return None;
        }
        let f = d.g("from");
        let s = if f.truthy() { util::py_repr(f) } else { util::py_repr(d.g("seq")) };
        Some((d.t().to_string(), d.o("code"), s))
    };
    (e.pos, e.reason.clone(), kind(&e.expected), kind(&e.got))
}

/// replay._stat_mismatch: a message that shows stats differs only in them.
pub fn stat_mismatch(e: &Divergence) -> bool {
    use util::V;
    let (Some(exp), Some(got)) = (e.expected.as_ref().and_then(DV::dict), e.got.as_ref().and_then(DV::dict)) else { return false };
    if !exp.is_object() || !got.is_object() || exp.t() != got.t() {
        return false;
    }
    let mut keys: BTreeSet<&String> = exp.as_object().unwrap().keys().collect();
    keys.extend(got.as_object().unwrap().keys());
    let diff: Vec<&&String> = keys.iter().filter(|k| exp.g(k) != got.g(k)).collect();
    !diff.is_empty() && diff.iter().all(|k| k.ends_with("_atk") || k.ends_with("_def"))
}

#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    pub opponent_decisions: usize,
    pub probed_decisions: usize,
    pub probes: usize,
    pub restarts: usize,
    pub forces: usize,
    pub assumptions: usize,
    pub swaps: usize,
    pub read_guided: usize,
    pub suspect_tests: usize,
}

/// The lockstep tracker, brought up to date only when something reads it.
#[derive(Clone)]
pub struct LazyTracker {
    t: Tracker,
    pending: Vec<ObsItem>,
}

impl LazyTracker {
    pub fn new(t: Tracker) -> LazyTracker {
        LazyTracker { t, pending: Vec::new() }
    }
    pub fn process(&mut self, item: ObsItem) {
        self.pending.push(item);
    }
    /// The tracker, up to date (an error of processing a pending item: the rest is dropped, as in
    /// the Python module).
    pub fn get(&mut self) -> Result<&Tracker, TrackerError> {
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            for item in &pending {
                self.t.process(item)?;
            }
        }
        Ok(&self.t)
    }
}

/// A logging run's record per engine step: (stream position, the card properties read, where
/// each card was: id -> (player, location, index, code)).
pub type ReadStep = (usize, Vec<(u32, u32)>, indexmap::IndexMap<u32, (u32, u32, u32, u32)>);

#[derive(Clone, Debug)]
pub struct SuspectLog {
    pub pos: usize,
    pub from: usize,
    pub steps: Vec<ReadStep>,
    pub sig: DivSig,
}

/// A test of a suspect: the fix, what to do in the probe ("swap" a b / "code" card code), a text.
pub type SuspectTest = (Fix, (&'static str, u32, u32), String);

#[derive(Clone, Debug)]
pub struct SuspectTests {
    pub at: usize,
    pub div: usize,
    pub until: usize,
    pub tests: Vec<SuspectTest>,
    pub sig: DivSig,
}

/// Backtracking over the decisions before a mismatch (the Python module's _state_bt dict).
#[derive(Clone, Debug)]
pub struct StateBt {
    pub pos: usize,
    pub reason: String,
    pub sig: DivSig,
    pub list: Vec<(Dkey, usize, i64)>,
    pub key: Dkey,
    pub queue: Vec<(Dkey, usize, Option<(Dkey, Option<usize>)>)>,
    pub state: bool,
    pub tries: usize,
    pub fallback: Option<Fix>,
    pub parent: Option<(Dkey, Option<usize>)>,
}

/// (answer, (candidates, index) for the decision search, the stream it depended on, the card ids a
/// card prompt offered)
pub type MemoEntry = (Response, Option<(usize, i64)>, Option<usize>, Option<Vec<Option<u32>>>);

#[derive(Clone)]
pub struct BeliefReplay {
    pub viewer: u32,
    pub stopped: bool,
    pub opponent: u32,
    pub init: InitInfo,
    pub items: Vec<ObsItem>,
    pub completion: &'static dyn Completion,
    pub check_state: bool,
    pub max_probes: usize,
    pub probe_items: usize,
    pub max_depth: usize,
    pub max_restarts: usize,
    pub probe_timeout: u64,
    pub max_stall: usize,
    pub min_progress: usize,
    pub probe_state_window: usize,
    pub stats: ReplayStats,
    /// debugging texts of the restarts ("restart N: reason -> fix"), when on
    pub log: Option<Vec<String>>,
    pub read_guided: bool,
    pub tail_start: Option<usize>,
    pub tail_restarts: usize,
    pub read_guided_fix: Option<Vec<String>>,
    /// a logging run (read_guided): the reads per main-line step
    pub logged_steps: Option<Vec<ReadStep>>,
    pub forces: BTreeSet<FKey>,
    pub force_windows: Vec<(usize, usize, FKey)>,
    pub unforced: BTreeSet<FKey>,
    pub unforced_windows: BTreeSet<(usize, usize, FKey)>,
    pub nomatch_id_spans: BTreeSet<(usize, usize, u32)>,
    pub assumptions: Vec<Asm>,
    pub exclusions: Vec<Excl>,
    pub identities: Vec<Ident>,
    pub failed_identities: BTreeSet<Ident>,
    pub folded_swaps: BTreeSet<(Cp, u32, u32)>,
    pub swaps: BTreeMap<Cp, Vec<(u32, u32)>>,
    pub matches: BTreeMap<usize, Vec<u32>>,
    pub code_tests: Vec<((usize, Slot), (usize, Vec<u32>))>,
    pub code_tests_done: BTreeSet<(usize, Slot)>,
    pub suspects_on: bool,
    pub suspect_done: HashMap<usize, usize>,
    pub suspect_log: Option<SuspectLog>,
    pub searched: BTreeSet<usize>,
    pub allow_approximate: bool,
    pub lenient: bool,
    pub approximate: bool,
    pub lenient_skips: usize,
    pub lenient_syncs: usize,
    pub state_sync: bool,
    pub synced_at: Option<usize>,
    pub suspect_tests: Option<SuspectTests>,
    pub test_until: Option<usize>,
    pub match_piles: BTreeSet<(usize, usize)>,
    pub match_pile_from: HashMap<(usize, usize), Cp>,
    pub nomatch_loc_piles: BTreeSet<(usize, usize, u32)>,
    pub nomatch_cards: BTreeSet<(usize, usize, Slot)>,
    pub nomatch_piles: BTreeSet<(usize, usize)>,
    pub pile_made: HashMap<(usize, usize), usize>,
    pub pile_dropped: BTreeSet<(&'static str, (usize, usize))>,
    pub snapshot_cps: BTreeSet<Cp>,
    pub failed_assumptions: BTreeSet<Asm>,
    pub failed_exclusions: BTreeSet<Excl>,
    pub failed_swaps: BTreeSet<(Cp, u32, u32)>,
    pub state_bt: Option<StateBt>,
    pub last_swap: Option<(Fix, usize)>,
    pub overrides: BTreeMap<Dkey, usize>,
    pub forced_resp: HashMap<Dkey, Response>,
    pub tried_resp: BTreeSet<(Dkey, Vec<usize>)>,
    pub tried_silent: BTreeSet<(Dkey, usize)>,
    pub last_hyp_fix: Option<&'static str>,
    pub last_identify_count: usize,
    pub memo: HashMap<(usize, usize, String), MemoEntry>,
    pub memo_cards: HashMap<(usize, usize, String, Vec<Option<u32>>), Response>,
    pub use_memo: bool,
    pub hypothesis: Option<Rc<Hypothesis>>,
    /// the tracker the latest hypothesis was solved with (its claims assigned)
    pub solved: Option<Rc<Tracker>>,
    pub assumed_vars: HashMap<VarId, Asm>,
    pub assume_conflict: Option<Asm>,
    pub feasible_cache: HashMap<(Vec<Asm>, Vec<Excl>, Vec<Ident>, Asm), bool>,
    pub property_tried: HashMap<u32, BTreeSet<u32>>,
    pub property_values: HashMap<u32, BTreeSet<(bool, u32, u32)>>,
    pub property_placed: HashMap<u32, Vec<Asm>>,
    pub matchsets: BTreeSet<Fix>,
    pub race_cache: HashMap<u64, Vec<u32>>,
    pub seen_cache: Option<Vec<u32>>,
    pub inherit_limit: Option<usize>,
    pub final_prompt: Option<Msg>,
    pub shuffle_ids: Vec<Option<Vec<u32>>>,
    pub active_forces: BTreeSet<FKey>,
    pub lenient_rejected: HashSet<(String, String)>,
    // -- the state of one attempt (_reset) ----------------------------------------------------
    pub force_enabled: HashSet<usize>,
    pub forced_used_prev: HashSet<(u32, u64)>,
    pub pos: usize,
    pub plan: Vec<((usize, &'static str), Response)>,
    pub planned: Option<Response>,
    pub report_depth: i64,
    pub probe_decisions: Vec<((usize, &'static str), Response)>,
    pub cp_pos: i64,
    pub cp_n: usize,
    pub carry_swaps: Vec<(u32, u32)>,
    pub left_pile: HashMap<u32, (Cp, (u32, u32))>,
    pub last_listed: HashMap<u32, Cp>,
    pub prompt_cp: Option<Cp>,
    pub depth: usize,
    pub probe_start: usize,
    pub probe_state_at: Option<usize>,
    pub horizon: usize,
    pub stall: usize,
    pub last_shuffle_cp: HashMap<(u32, u32), Cp>,
    pub card_shuffled: HashMap<u32, Vec<Cp>>,
    pub shuffle_count: usize,
    pub jit_from: Option<usize>,
    pub jit_done: HashSet<usize>,
    pub event_hint: Option<(usize, u64)>,
    pub opp_select_hint: Option<u64>,
    pub decisions_at: BTreeMap<usize, usize>,
    pub recent: Vec<(Dkey, usize, i64)>,
    pub since_state: Vec<(Dkey, usize, i64)>,
    pub cancelled: HashSet<(usize, String)>,
    pub opp_prompt: Option<Msg>,
    pub prev_shuffle_cps: BTreeMap<Cp, Vec<usize>>,
    pub shuffle_cps: BTreeMap<Cp, Vec<usize>>,
    pub snapshots: HashMap<Cp, Rc<LocMap>>,
    pub roll_cp: Option<Cp>,
    pub card_decisions: HashMap<Dkey, HashMap<Option<u32>, usize>>,
    pub window_choices: Vec<(Dkey, usize)>,
    pub window_activations: Vec<(Dkey, usize)>,
    pub offered: HashMap<Dkey, (HashSet<Option<u32>>, HashSet<(u32, u32)>)>,
    pub hidden_picks: Vec<(usize, (usize, usize))>,
    pub sim_marks: HashMap<u64, Vec<u32>>,
    pub silent_choices: Vec<(Dkey, Vec<Option<u32>>, usize, usize, Option<Vec<Response>>)>,
    pub assumed_moves: Vec<(usize, Asm)>,
    pub multi_decisions: HashMap<Dkey, (Vec<Option<u32>>, Response)>,
    pub roll_dirty: bool,
    pub lockstep: LazyTracker,
    // -- transient state of decisions and probes -----------------------------------------------
    pub dkey: Option<Dkey>,
    pub choice_seen: Option<usize>,
    pub prompt_ids: Option<Vec<Option<u32>>>,
    pub inferred: Vec<Response>,
    pub pending_select_msg: Option<u64>,
    pub prompt_select_msg: Option<u64>,
    pub skip_shuffle_hand: Option<u32>,
    pub probe_steps: usize,
    pub probe_boundary: bool,
    pub probe_after_boundary: usize,
    pub probe_sig: Option<(usize, String)>,
    pub probe_wide: bool,
    pub probe_deadline: Option<std::time::Instant>,
    pub last_probe_reached: usize,
    pub last_probe_plan: Vec<((usize, &'static str), Response)>,
    pub last_probe_sig: Option<DivSig>,
    /// probes started since the main line's current probe began (shared by the copies)
    pub probe_counter: Rc<Cell<u64>>,
    /// when the whole reconstruction gives up (every engine step fails after it: "timeout")
    pub deadline: Option<std::time::Instant>,
}

impl BeliefReplay {
    pub fn new(viewer: u32, init: InitInfo, items: Vec<ObsItem>, completion: Option<&'static dyn Completion>) -> BeliefReplay {
        let lockstep = LazyTracker::new(Tracker::new(viewer, &init));
        let vanilla: &'static dyn Completion = VanillaCompletion::get();
        let mut br = BeliefReplay {
            viewer,
            stopped: false,
            opponent: 1 - viewer,
            init,
            items,
            completion: completion.unwrap_or(vanilla),
            check_state: true,
            max_probes: 24,
            probe_items: 60,
            max_depth: 4,
            max_restarts: std::env::var("YGOAI_MAX_RESTARTS").ok().and_then(|v| v.parse().ok()).unwrap_or(40),
            probe_timeout: 20,
            max_stall: 300,
            min_progress: 4,
            probe_state_window: std::env::var("YGOAI_PROBE_STATE_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(8),
            stats: ReplayStats::default(),
            log: None,
            read_guided: !env_flag("YGOAI_NO_READ_GUIDED"),
            tail_start: None,
            tail_restarts: 0,
            read_guided_fix: None,
            logged_steps: None,
            forces: BTreeSet::new(),
            force_windows: Vec::new(),
            unforced: BTreeSet::new(),
            unforced_windows: BTreeSet::new(),
            nomatch_id_spans: BTreeSet::new(),
            assumptions: Vec::new(),
            exclusions: Vec::new(),
            identities: Vec::new(),
            failed_identities: BTreeSet::new(),
            folded_swaps: BTreeSet::new(),
            swaps: BTreeMap::new(),
            matches: BTreeMap::new(),
            code_tests: Vec::new(),
            code_tests_done: BTreeSet::new(),
            suspects_on: !env_flag("YGOAI_NO_SUSPECTS"),
            suspect_done: HashMap::new(),
            suspect_log: None,
            searched: BTreeSet::new(),
            allow_approximate: false,
            lenient: false,
            approximate: false,
            lenient_skips: 0,
            lenient_syncs: 0,
            state_sync: !env_flag("YGOAI_NO_STATE_SYNC"),
            synced_at: None,
            suspect_tests: None,
            test_until: None,
            match_piles: BTreeSet::new(),
            match_pile_from: HashMap::new(),
            nomatch_loc_piles: BTreeSet::new(),
            nomatch_cards: BTreeSet::new(),
            nomatch_piles: BTreeSet::new(),
            pile_made: HashMap::new(),
            pile_dropped: BTreeSet::new(),
            snapshot_cps: BTreeSet::new(),
            failed_assumptions: BTreeSet::new(),
            failed_exclusions: BTreeSet::new(),
            failed_swaps: BTreeSet::new(),
            state_bt: None,
            last_swap: None,
            overrides: BTreeMap::new(),
            forced_resp: HashMap::new(),
            tried_resp: BTreeSet::new(),
            tried_silent: BTreeSet::new(),
            last_hyp_fix: None,
            last_identify_count: 0,
            memo: HashMap::new(),
            memo_cards: HashMap::new(),
            use_memo: true,
            hypothesis: None,
            solved: None,
            assumed_vars: HashMap::new(),
            assume_conflict: None,
            feasible_cache: HashMap::new(),
            property_tried: HashMap::new(),
            property_values: HashMap::new(),
            property_placed: HashMap::new(),
            matchsets: BTreeSet::new(),
            race_cache: HashMap::new(),
            seen_cache: None,
            inherit_limit: None,
            final_prompt: None,
            shuffle_ids: Vec::new(),
            active_forces: BTreeSet::new(),
            lenient_rejected: HashSet::new(),
            force_enabled: HashSet::new(),
            forced_used_prev: HashSet::new(),
            pos: 0,
            plan: Vec::new(),
            planned: None,
            report_depth: -1,
            probe_decisions: Vec::new(),
            cp_pos: -1,
            cp_n: 0,
            carry_swaps: Vec::new(),
            left_pile: HashMap::new(),
            last_listed: HashMap::new(),
            prompt_cp: None,
            depth: 0,
            probe_start: 0,
            probe_state_at: None,
            horizon: 0,
            stall: 0,
            last_shuffle_cp: HashMap::new(),
            card_shuffled: HashMap::new(),
            shuffle_count: 0,
            jit_from: None,
            jit_done: HashSet::new(),
            event_hint: None,
            opp_select_hint: None,
            decisions_at: BTreeMap::new(),
            recent: Vec::new(),
            since_state: Vec::new(),
            cancelled: HashSet::new(),
            opp_prompt: None,
            prev_shuffle_cps: BTreeMap::new(),
            shuffle_cps: BTreeMap::new(),
            snapshots: HashMap::new(),
            roll_cp: None,
            card_decisions: HashMap::new(),
            window_choices: Vec::new(),
            window_activations: Vec::new(),
            offered: HashMap::new(),
            hidden_picks: Vec::new(),
            sim_marks: HashMap::new(),
            silent_choices: Vec::new(),
            assumed_moves: Vec::new(),
            multi_decisions: HashMap::new(),
            roll_dirty: true,
            lockstep,
            dkey: None,
            choice_seen: None,
            prompt_ids: None,
            inferred: Vec::new(),
            pending_select_msg: None,
            prompt_select_msg: None,
            skip_shuffle_hand: None,
            probe_steps: 0,
            probe_boundary: false,
            probe_after_boundary: 0,
            probe_sig: None,
            probe_wide: false,
            deadline: None,
            probe_deadline: None,
            last_probe_reached: 0,
            last_probe_plan: Vec::new(),
            last_probe_sig: None,
            probe_counter: Rc::new(Cell::new(0)),
        };
        let (forces, windows) = br.stream_forces();
        br.forces = forces;
        br.force_windows = windows;
        br.reset();
        br
    }

    pub fn log(&mut self, text: String) {
        if let Some(l) = self.log.as_mut() {
            l.push(text);
        }
    }

    pub fn reset(&mut self) {
        self.force_enabled.clear();
        self.forced_used_prev.clear();
        self.pos = 0;
        self.plan.clear();
        self.planned = None;
        self.report_depth = -1;
        self.probe_decisions.clear();
        self.cp_pos = -1;
        self.cp_n = 0;
        self.carry_swaps.clear();
        self.left_pile.clear();
        self.last_listed.clear();
        self.prompt_cp = None;
        self.depth = 0;
        self.probe_start = 0;
        self.probe_state_at = None;
        self.horizon = self.items.len();
        self.stall = 0;
        self.last_shuffle_cp.clear();
        self.card_shuffled.clear();
        self.shuffle_count = 0;
        self.jit_from = None;
        self.jit_done.clear();
        self.event_hint = None;
        self.opp_select_hint = None;
        self.decisions_at.clear();
        self.recent.clear();
        self.since_state.clear();
        self.cancelled.clear();
        self.opp_prompt = None;
        // where the previous attempt shuffled (for folding swaps into shuffles)
        self.prev_shuffle_cps = std::mem::take(&mut self.shuffle_cps);
        self.snapshots.clear();
        self.roll_cp = None;
        self.card_decisions.clear();
        self.window_choices.clear();
        self.window_activations.clear();
        self.offered.clear();
        self.hidden_picks.clear();
        self.sim_marks.clear();
        self.silent_choices.clear();
        self.assumed_moves.clear();
        self.multi_decisions.clear();
        self.roll_dirty = true;
        self.lockstep = LazyTracker::new(Tracker::new(self.viewer, &self.init));
    }

    /// The lockstep tracker, up to date.
    pub fn lock(&mut self) -> Result<&Tracker, TrackerError> {
        self.lockstep.get()
    }

    /// The lockstep tracker's var at a slot (None also when the tracker fails).
    pub fn lock_var(&mut self, con: u32, loc: u32, seq: u32) -> Result<Option<VarId>, TrackerError> {
        self.lockstep.get()?.var_at(con, loc, seq, 0)
    }

    // -- setup -----------------------------------------------------------------------------

    pub fn build_hypothesis(&mut self) -> R<Rc<Hypothesis>> {
        let mut tracker = Tracker::new(self.viewer, &self.init);
        self.assumed_vars.clear();
        self.assume_conflict = None;
        let mut pending = self.assumptions.clone();
        pending.sort();
        let mut excl = self.exclusions.clone();
        excl.sort();
        let (mut k, mut j) = (0, 0);
        for i in 0..self.items.len() {
            while k < pending.len() && pending[k].0 == i {
                let (_, con, loc, seq, code) = pending[k];
                let av = tracker.var_at(con, loc, seq, 0)?;
                if let Err(e) = tracker.observe(av, code) {
                    // (the card another assumption named earlier: which one fails)
                    self.assume_conflict = av.and_then(|v| self.assumed_vars.get(&v).copied());
                    return Err(e.into());
                }
                if let Some(v) = av {
                    self.assumed_vars.insert(v, pending[k]);
                }
                k += 1;
            }
            while j < excl.len() && excl[j].0 == i {
                let (_, con, loc, ref seqs, code) = excl[j];
                let seqs: Vec<u32> = if seqs == &[ALL_SLOTS] {
                    (0..tracker.list(con, loc).map_or(0, |l| l.len()) as u32).collect()
                } else {
                    seqs.iter().map(|&s| s as u32).collect()
                };
                for seq in seqs {
                    let v = tracker.var_at(con, loc, seq, 0)?;
                    tracker.exclude(v, code)?;
                }
                j += 1;
            }
            let mut revealed = Vec::new();
            for (pos, (con, loc, seq), same) in self.identities.clone() {
                if pos != i {
                    continue;
                }
                match same {
                    Same::Revealed => revealed.push(tracker.var_at(con, loc, seq, 0)?),
                    Same::Not(other) => {
                        let v = tracker.var_at(con, loc, seq, 0)?;
                        if let Some(v) = v
                            && other < tracker.vars.len()
                        {
                            tracker.vars[v].not_same.push(other);
                            // (a constraint: not_same alone only orders members of the same kind)
                            let x = &tracker.vars[v];
                            if x.kind == crate::belief::tracker::Kind::Claim
                                && let Some(pool) = x.pool
                                && tracker.pools[pool].members.contains(&other)
                            {
                                let ids: BTreeSet<VarId> = tracker.pools[pool].members.iter().copied().filter(|&m| m != other).collect();
                                if !ids.is_empty() {
                                    let x = &mut tracker.vars[v];
                                    x.allowed = Some(match x.allowed.take() {
                                        None => ids,
                                        Some(a) => {
                                            let n: BTreeSet<VarId> = a.intersection(&ids).copied().collect();
                                            if n.is_empty() { a } else { n }
                                        }
                                    });
                                }
                            }
                        }
                    }
                    Same::Var(other) => {
                        if other < tracker.vars.len() {
                            let v = tracker.var_at(con, loc, seq, 0)?;
                            tracker.identify(v, Some(other));
                        }
                    }
                }
            }
            tracker.process(&self.items[i])?;
            for v in revealed.into_iter().flatten() {
                // the card moved from that slot is one of the cards revealed before the shuffle
                let x = &tracker.vars[v];
                if x.kind == crate::belief::tracker::Kind::Claim
                    && let Some(pool) = x.pool
                    && !x.prefer.is_empty()
                {
                    let prefer = x.prefer.clone();
                    let mut ids: BTreeSet<VarId> = tracker.pools[pool].members.iter().copied().filter(|m| prefer.contains(m)).collect();
                    if ids.is_empty() {
                        // (shuffled more than once since: the revealed cards themselves)
                        ids = prefer.iter().copied().collect();
                    }
                    if !ids.is_empty() {
                        let x = &mut tracker.vars[v];
                        x.allowed = Some(match x.allowed.take() {
                            None => ids,
                            Some(a) => {
                                let n: BTreeSet<VarId> = a.intersection(&ids).copied().collect();
                                if n.is_empty() { a } else { n }
                            }
                        });
                    }
                }
            }
        }
        let h = Rc::new(tracker.solve(Some(self.completion))?);
        self.hypothesis = Some(h.clone());
        self.solved = Some(Rc::new(tracker));
        Ok(h)
    }

    pub fn build_duel(&mut self) -> R<RDuel> {
        let hyp = match &self.hypothesis {
            Some(h) => h.clone(),
            None => self.build_hypothesis()?,
        };
        let decks: Vec<Deck> = (0..2).map(|p| Deck { name: String::new(), main: hyp.decks[p].0.clone(), extra: hyp.decks[p].1.clone() }).collect();
        let cfg: DuelConfig = self.init.config.clone();
        let duel = setup_duel(&decks, [0x1234, 0x5678, 0x9ABC, 0xDEF0], &cfg, &hyp.deck_orders, None, None, None).map_err(|e| RErr::Other(e.0))?;
        let mut duel = RDuel::from_duel(duel);
        // which physical copy is where: copies of a code are created in rank order
        let mut id_of: HashMap<(u32, u32, u32), u32> = HashMap::new();
        for p in 0..2u32 {
            for loc in [LOCATION_DECK, LOCATION_EXTRA] {
                let mut n: HashMap<u32, u32> = HashMap::new();
                let mut cards = duel.location_cards(p, loc);
                cards.sort_by_key(|x| x.1);
                for (code, cid) in cards {
                    let r = n.entry(code).or_insert(0);
                    id_of.insert((p, code, *r), cid);
                    *r += 1;
                }
            }
        }
        let ids = |player: u32, codes: &[u32], ranks: Option<&Vec<u32>>, owners: Option<&Vec<u32>>| -> Option<Vec<u32>> {
            let ranks = ranks?;
            let out: Vec<Option<u32>> =
                codes.iter().zip(ranks).enumerate().map(|(i, (&c, &r))| id_of.get(&(owners.map_or(player, |o| o[i]), c, r)).copied()).collect();
            out.into_iter().collect()
        };
        // a swap at the very start exchanges two copies' roles for the whole duel
        let roles = roles(self.swaps.get(&(0, 0)).map_or(&[][..], |v| &v[..]));
        let swap_roles = |lst: Option<Vec<u32>>| lst.map(|l| l.into_iter().map(|x| *roles.get(&x).unwrap_or(&x)).collect::<Vec<u32>>());
        for p in 0..2u32 {
            let pi = p as usize;
            let deck_ids = swap_roles(ids(p, &hyp.deck_orders[pi], hyp.deck_ranks.get(pi).and_then(|r| r.as_ref()), None));
            if let Some(d) = &deck_ids
                && !duel.set_location_order_by_id(p, LOCATION_DECK, d)
            {
                return div(Divergence::new(0, format!("cannot set deck order of player {p} by card id")));
            }
            if !hyp.extra_orders[pi].is_empty() {
                let extra_ids = swap_roles(ids(p, &hyp.extra_orders[pi], hyp.extra_ranks.get(pi).and_then(|r| r.as_ref()), None));
                let ok = match &extra_ids {
                    Some(e) => duel.set_location_order_by_id(p, LOCATION_EXTRA, e),
                    None => duel.set_location_order(p, LOCATION_EXTRA, &hyp.extra_orders[pi]),
                };
                if !ok {
                    return div(Divergence::new(0, format!("cannot set extra deck order of player {p}")));
                }
            }
        }
        let mut shuffle_ids: Vec<Option<Vec<u32>>> =
            hyp.shuffles.iter().map(|sh| swap_roles(ids(sh.player, &sh.codes, sh.ranks.as_ref(), sh.owners.as_ref()))).collect();
        // a copy swap scheduled for the step that shuffles the pile exchanges the two cards in that
        // shuffle's result
        self.folded_swaps = self.swaps.get(&(0, 0)).map_or(BTreeSet::new(), |v| v.iter().map(|&(a, b)| ((0, 0), a, b)).collect());
        let shuffle_cp: HashMap<usize, Cp> = self.prev_shuffle_cps.iter().flat_map(|(&c, ks)| ks.iter().map(move |&k| (k, c))).collect();
        // (only copies trade places)
        let code_of: HashMap<u32, u32> = id_of.iter().map(|(k, &cid)| (cid, k.1)).collect();
        let cps: Vec<Cp> = self.swaps.keys().copied().filter(|&c| c != (0, 0)).collect();
        for cp in cps {
            let pairs: Vec<(u32, u32)> = self.swaps[&cp].iter().copied().filter(|(a, b)| code_of.get(a) == code_of.get(b)).collect();
            let here: BTreeSet<usize> = self.prev_shuffle_cps.get(&cp).map_or(BTreeSet::new(), |v| v.iter().copied().collect());
            for &k in &here {
                let Some(Some(lst)) = shuffle_ids.get_mut(k) else { continue };
                for &(a, b) in &pairs {
                    if let (Some(i), Some(j)) = (lst.iter().position(|&x| x == a), lst.iter().position(|&x| x == b)) {
                        lst.swap(i, j);
                        self.folded_swaps.insert((cp, a, b));
                    }
                }
            }
            // from then on each of the two copies continues the other's path
            for (k, lst) in shuffle_ids.iter_mut().enumerate() {
                let Some(l) = lst else { continue };
                if here.contains(&k) || shuffle_cp.get(&k).is_some_and(|&c| c <= cp) {
                    continue;
                }
                for &(a, b) in &pairs {
                    *l = l.iter().map(|&x| if x == a { b } else if x == b { a } else { x }).collect();
                }
            }
        }
        for (sh, ids) in hyp.shuffles.iter().zip(&shuffle_ids) {
            let PoolLoc::Loc(loc) = sh.location else { continue };
            match ids {
                Some(ids) => duel.push_forced_shuffle_ids(sh.player, loc, ids, &sh.codes),
                None => duel.push_forced_shuffle(sh.player, loc, &sh.codes),
            }
        }
        self.shuffle_ids = shuffle_ids;
        for (k, codes) in hyp.set_card_shuffles.iter().enumerate() {
            let (ranks, owners) = hyp.set_card_shuffle_ranks.get(k).cloned().unwrap_or((None, None));
            let set_ids = if owners.as_ref().is_some_and(|o| !o.is_empty()) { swap_roles(ids(0, codes, ranks.as_ref(), owners.as_ref())) } else { None };
            match set_ids {
                Some(ids) => duel.push_forced_set_card_shuffle_ids(&ids, codes),
                None => duel.push_forced_set_card_shuffle(codes),
            }
        }
        push_random(&mut duel, &self.items);
        self.active_forces.clear();
        Ok(duel)
    }

    /// (an extended reconstruction) The viewer's own deck and hand in the part of the stream the
    /// hypothesis did not cover (see the Python docstring).
    pub fn jit_prepare(&mut self, duel: &mut RDuel) {
        let Some(jit_from) = self.jit_from else { return };
        let mut draws: Vec<u32> = Vec::new();
        let mut shuffle: Option<usize> = None;
        let end = self.items.len().min(self.pos + 40);
        let viewer = self.viewer;
        for j in self.pos.max(jit_from)..end {
            let it = &self.items[j];
            if matches!(it, ObsItem::Prompt(..)) {
                break;
            }
            let ObsItem::Msg(m) = it else { continue };
            if self.jit_done.contains(&j) {
                continue;
            }
            match m {
                Msg::ShuffleDeck { player } | Msg::ShuffleHand { player, .. } if *player == viewer && (!draws.is_empty() || shuffle.is_some()) => break,
                Msg::ShuffleDeck { player } if *player == viewer => {
                    let arrives = self.items[self.pos..j]
                        .iter()
                        .any(|x| matches!(x, ObsItem::Msg(Msg::Move { to, .. }) if to.con == viewer && to.loc == LOCATION_DECK));
                    if arrives {
                        return; // (cards still to come into the deck before it: not arranged)
                    }
                    shuffle = Some(j);
                    continue;
                }
                Msg::Draw { player, cards } if *player == viewer => {
                    draws.extend(cards.iter().map(|c| c.code));
                    self.jit_done.insert(j);
                }
                Msg::ConfirmDecktop { player, cards } if *player == viewer && draws.is_empty() => {
                    draws.extend(cards.iter().map(|c| c.code));
                    self.jit_done.insert(j);
                    break;
                }
                Msg::ShuffleHand { player, codes, prev, .. } if *player == viewer && codes.iter().all(|&c| c != 0) => {
                    let hand = duel.location_cards(viewer, LOCATION_HAND);
                    match prev {
                        Some(prev)
                            if !prev.is_empty()
                                && prev.len() == hand.len()
                                && hand.len() == codes.len()
                                && prev.iter().zip(codes).all(|(&k, &c)| hand.get(k as usize).is_some_and(|h| h.0 == c)) =>
                        {
                            let ids: Vec<u32> = prev.iter().map(|&k| hand[k as usize].1).collect();
                            duel.push_forced_shuffle_ids(viewer, LOCATION_HAND, &ids, codes);
                        }
                        _ => duel.push_forced_shuffle(viewer, LOCATION_HAND, codes),
                    }
                    self.jit_done.insert(j);
                    break;
                }
                _ => {}
            }
        }
        if !draws.is_empty() {
            let deck = duel.location_cards(viewer, LOCATION_DECK); // (bottom first)
            let (mut top, mut rest) = (Vec::new(), deck);
            for code in draws {
                let Some(k) = (0..rest.len()).rev().find(|&k| rest[k].0 == code) else { return };
                top.push(rest.remove(k));
            }
            top.reverse();
            let order: Vec<(u32, u32)> = rest.into_iter().chain(top).collect();
            if let Some(s) = shuffle {
                let ids: Vec<u32> = order.iter().map(|x| x.1).collect();
                let codes: Vec<u32> = order.iter().map(|x| x.0).collect();
                duel.push_forced_shuffle_ids(viewer, LOCATION_DECK, &ids, &codes);
                self.jit_done.insert(s);
            } else {
                let ids: Vec<u32> = order.iter().map(|x| x.1).collect();
                duel.set_location_order_by_id(viewer, LOCATION_DECK, &ids);
            }
        }
    }

    /// The forces of the opponent's activations the stream shows, and the windows they apply in.
    pub fn stream_forces(&self) -> (BTreeSet<FKey>, Vec<(usize, usize, FKey)>) {
        let mut forces = BTreeSet::new();
        for (j, it) in self.items.iter().enumerate() {
            if let ObsItem::Msg(Msg::Chaining { code, desc, .. }) = it
                && self.opp_activates(j)
            {
                forces.insert((self.opponent, *code, *desc));
            }
        }
        let mut windows = Vec::new();
        let mut start = 0;
        let mut chaining: Option<u32> = None;
        for (i, it) in self.items.iter().enumerate() {
            let ObsItem::Msg(m) = it else { continue };
            match m {
                Msg::NewPhase { .. } | Msg::NewTurn { .. } => start = i,
                Msg::Chaining { code, desc, .. } if self.opp_activates(i) => {
                    chaining = Some(*code);
                    windows.push((start, i + 1, (self.opponent, *code, *desc)));
                }
                Msg::Chained { .. } => chaining = None,
                Msg::Hint { hint_type, player, data } if *hint_type == HINT_OPSELECTED && *player == self.opponent && chaining.is_some_and(|c| data >> 20 == c as u64) => {
                    // a continuous trap activated together with one of its effects
                    let key = (self.opponent, chaining.unwrap(), *data);
                    forces.insert(key);
                    windows.push((start, i + 1, key));
                }
                _ => {}
            }
        }
        (forces, windows)
    }

    /// Whether the CHAINING at stream position j is the opponent's activation.
    pub fn opp_activates(&self, j: usize) -> bool {
        let ObsItem::Msg(Msg::Chaining { trigger_con, .. }) = &self.items[j] else { return false };
        if *trigger_con == self.opponent {
            return true;
        }
        for it in self.items.iter().skip(j + 1).take(11) {
            let ObsItem::Msg(x) = it else { continue };
            match x {
                Msg::Chained { .. } => break,
                Msg::PayLpcost { player, .. } if *player == self.opponent => return true,
                Msg::Hint { hint_type, player, .. } if *hint_type == HINT_OPSELECTED && *player == self.opponent => return true,
                _ => {}
            }
        }
        false
    }

    /// Continues this consistent reconstruction with a longer stream (see the Python docstring).
    pub fn extend(&mut self, duel: &mut RDuel, items: Vec<ObsItem>) -> R<()> {
        let n = self.items.len();
        let Some(prompt) = self.final_prompt.clone() else { return div(Divergence::new(n, "not a continuation of this reconstruction")) };
        if self.approximate || items.len() <= n || items[..n] != self.items[..] {
            return div(Divergence::new(n, "not a continuation of this reconstruction"));
        }
        self.inherit_limit = Some(self.inherit_limit.unwrap_or(n).min(n));
        self.items = items;
        self.final_prompt = None;
        let (forces, windows) = self.stream_forces();
        self.forces.extend(forces);
        for w in windows {
            if !self.force_windows.contains(&w) {
                self.force_windows.push(w);
            }
        }
        push_random(duel, &self.items[n..].to_vec());
        self.jit_from = Some(self.jit_from.unwrap_or(n));
        let ObsItem::Response(resp) = self.items[self.pos].clone() else { return div(Divergence::new(self.pos, "expected the viewer's response")) };
        self.advance()?;
        let core = crate::observe::to_core_response(&prompt, self.viewer, &resp).map_err(|e| RErr::Other(format!("InvalidResponse: {e}")))?;
        self.run_loop(duel, Some(prompt), Some(core), Some(resp))?;
        if self.final_prompt.is_none() {
            return div(Divergence::new(self.pos, "the replay did not reach the viewer's decision"));
        }
        self.inherit_limit = Some(self.items.len()); // (reached consistently)
        Ok(())
    }

    /// Replays the whole stream; the duel at the stream end. When the repairs give up, a last
    /// resort guided by the hidden cards the engine read (read_guided).
    pub fn run(&mut self) -> R<RDuel> {
        if self.allow_approximate && env_flag("YGOAI_LENIENT_ONLY") && self.depth == 0 {
            return self.lenient_run().ok_or_else(|| Divergence::new(0, "no approximate reconstruction").into());
        }
        match self.run_attempts() {
            Ok(d) => Ok(d),
            Err(RErr::Div(e)) => {
                let last_prompt = self.items.iter().rposition(|it| matches!(it, ObsItem::Prompt(..))).map_or(-1, |i| i as i64);
                let unfinished = matches!(self.items.last(), Some(ObsItem::Msg(Msg::Win { .. }))) && e.pos as i64 > last_prompt + 1;
                if self.depth > 0 || unfinished {
                    return Err(RErr::Div(e));
                }
                let mut duel = None;
                if self.read_guided {
                    duel = crate::belief::read_guided::read_guided_repair(self);
                }
                if duel.is_none() && self.allow_approximate {
                    duel = self.lenient_run();
                }
                duel.ok_or(RErr::Div(e))
            }
            Err(e) => Err(e),
        }
    }

    /// The repairs a reconstruction found, valid for any longer stream with the same start
    /// (inherit).
    pub fn inherit(&mut self, prev: &BeliefReplay) -> bool {
        let n = prev.items.len().min(prev.inherit_limit.unwrap_or(prev.items.len()));
        if prev.approximate || n > self.items.len() || prev.viewer != self.viewer || self.items[..n] != prev.items[..n] {
            return false;
        }
        fn merge_set<T: Ord + Clone>(mine: &BTreeSet<T>, theirs: &BTreeSet<T>) -> BTreeSet<T> {
            mine.union(theirs).cloned().collect()
        }
        fn merge_list<T: PartialEq + Clone>(mine: &[T], theirs: &[T]) -> Vec<T> {
            if mine.is_empty() {
                return theirs.to_vec();
            }
            let mut out = mine.to_vec();
            out.extend(theirs.iter().filter(|x| !mine.contains(x)).cloned());
            out
        }
        self.assumptions = merge_list(&self.assumptions, &prev.assumptions);
        self.exclusions = merge_list(&self.exclusions, &prev.exclusions);
        let mut swaps = self.swaps.clone();
        swaps.extend(prev.swaps.clone());
        self.swaps = if self.swaps.is_empty() { prev.swaps.clone() } else { swaps };
        self.identities = merge_list(&self.identities, &prev.identities);
        self.forces = merge_set(&self.forces, &prev.forces);
        self.force_windows = merge_list(&self.force_windows, &prev.force_windows);
        self.unforced = merge_set(&self.unforced, &prev.unforced);
        self.unforced_windows = merge_set(&self.unforced_windows, &prev.unforced_windows);
        let mut m = self.matches.clone();
        m.extend(prev.matches.clone());
        self.matches = if self.matches.is_empty() { prev.matches.clone() } else { m };
        self.match_piles = merge_set(&self.match_piles, &prev.match_piles);
        let mut mf = self.match_pile_from.clone();
        mf.extend(prev.match_pile_from.clone());
        self.match_pile_from = mf;
        self.nomatch_piles = merge_set(&self.nomatch_piles, &prev.nomatch_piles);
        self.nomatch_loc_piles = merge_set(&self.nomatch_loc_piles, &prev.nomatch_loc_piles);
        self.nomatch_cards = merge_set(&self.nomatch_cards, &prev.nomatch_cards);
        self.nomatch_id_spans = merge_set(&self.nomatch_id_spans, &prev.nomatch_id_spans);
        self.failed_assumptions = merge_set(&self.failed_assumptions, &prev.failed_assumptions);
        self.failed_exclusions = merge_set(&self.failed_exclusions, &prev.failed_exclusions);
        self.failed_identities = merge_set(&self.failed_identities, &prev.failed_identities);
        self.failed_swaps = merge_set(&self.failed_swaps, &prev.failed_swaps);
        self.code_tests_done = merge_set(&self.code_tests_done, &prev.code_tests_done);
        self.pile_dropped = merge_set(&self.pile_dropped, &prev.pile_dropped);
        let mut pm = self.pile_made.clone();
        pm.extend(prev.pile_made.clone());
        self.pile_made = pm;
        let cutoff = n as i64 - self.probe_items as i64;
        self.memo = prev.memo.iter().filter(|(k, v)| (k.0 as i64) < cutoff || v.2.is_some_and(|s| s <= n)).map(|(k, v)| (k.clone(), v.clone())).collect();
        self.memo_cards = self.memo.iter().filter_map(|(k, v)| v.3.clone().map(|ids| ((k.0, k.1, k.2.clone(), ids), v.0.clone()))).collect();
        true
    }

    /// The best effort when no consistent reconstruction exists (see the Python docstring).
    pub fn lenient_run(&mut self) -> Option<RDuel> {
        self.reset();
        if self.hypothesis.is_none() && self.build_hypothesis().is_err() {
            return None;
        }
        let mut duel = self.build_duel().ok()?;
        self.lenient = true;
        self.lenient_skips = 0;
        self.lenient_syncs = 0;
        self.synced_at = None;
        self.lenient_rejected.clear();
        duel.start();
        let r = self.run_loop(&mut duel, None, None, None);
        self.lenient = false;
        if let Err(e) = r {
            self.log(format!("no approximate reconstruction either: {e}"));
            return None;
        }
        self.final_prompt.as_ref()?;
        self.approximate = true;
        let s = self.lenient_skips;
        self.log(format!("approximate reconstruction ({s} items re-aligned)"));
        Some(duel)
    }

    /// The attempts with restarts (_run).
    pub fn run_attempts(&mut self) -> R<RDuel> {
        let mut last: Option<Box<Divergence>> = None;
        for _ in 0..=self.max_restarts {
            self.reset();
            if self.hypothesis.is_none() {
                match self.build_hypothesis() {
                    Ok(_) => {}
                    Err(RErr::Tracker(te)) => {
                        if !self.identities.is_empty() && matches!(self.last_hyp_fix, Some("identify") | Some("identify_all")) {
                            let n = if self.last_hyp_fix == Some("identify_all") { self.last_identify_count } else { 1 };
                            for _ in 0..n {
                                if let Some(i) = self.identities.pop() {
                                    self.failed_identities.insert(i);
                                }
                            }
                            self.last_hyp_fix = None;
                            let r = self.stats.restarts;
                            self.log(format!("restart {r}: hypothesis failed ({}) -> drop identity", te.0));
                            self.stats.restarts += 1;
                            continue;
                        }
                        if !self.exclusions.is_empty() && self.last_hyp_fix == Some("exclude") {
                            if let Some(x) = self.exclusions.pop() {
                                self.failed_exclusions.insert(x);
                            }
                            self.last_hyp_fix = None;
                            let r = self.stats.restarts;
                            self.log(format!("restart {r}: hypothesis failed ({}) -> drop exclusion", te.0));
                            self.stats.restarts += 1;
                            continue;
                        }
                        if self.assumptions.is_empty() {
                            return Err(RErr::Tracker(te));
                        }
                        let older = self.assume_conflict;
                        if let Some(o) = older
                            && self.assumptions[..self.assumptions.len() - 1].contains(&o)
                        {
                            // the latest assumption names another code for a card an earlier one
                            // named, at a later point: the earlier one fails
                            self.assumptions.retain(|a| *a != o);
                            self.failed_assumptions.insert(o);
                            let r = self.stats.restarts;
                            self.log(format!("restart {r}: hypothesis failed ({}) -> drop {o:?}", te.0));
                            self.stats.restarts += 1;
                            continue;
                        }
                        // the last assumption is impossible (e.g. too many copies)
                        let a = self.assumptions.pop().unwrap();
                        self.failed_assumptions.insert(a);
                        let r = self.stats.restarts;
                        self.log(format!("restart {r}: hypothesis failed ({}) -> drop assumption", te.0));
                        self.stats.restarts += 1;
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
            let mut duel = self.build_duel()?;
            duel.start();
            let r = self.run_loop(&mut duel, None, None, None);
            let e = match r {
                Ok(()) => return Ok(duel),
                Err(RErr::Div(e)) => e,
                Err(other) => return Err(other),
            };
            drop(duel);
            last = Some(e.clone());
            if let Some(ts) = self.tail_start
                && e.pos >= ts
            {
                // (in the unfinished action the game ended in: only a few repairs)
                self.tail_restarts += 1;
                if self.tail_restarts > TAIL_RESTARTS {
                    return Err(RErr::Div(e));
                }
            }
            match self.after_divergence(*e)? {
                true => continue,
                false => unreachable!(),
            }
        }
        match last {
            None => div(Divergence::new(self.pos, "too many restarts (no consistent hypothesis)")),
            Some(l) => div(Divergence {
                expected: l.expected.clone(),
                got: l.got.clone(),
                ..Divergence::new(self.pos, format!("too many restarts (last: {})", l.reason))
            }),
        }
    }

    /// What a divergence of an attempt leads to: a repair or decision change to restart with
    /// (Ok(true)), or the divergence itself (Err).
    fn after_divergence(&mut self, mut e: Divergence) -> R<bool> {
        let bt = self.state_bt.clone();
        if let Some(bt) = &bt
            && !bt.state
            && let Some(fix) = e.fix.clone()
            && (e.pos == bt.pos || e.pos < bt.pos && matches!(fix.kind(), "nomatchpile" | "matchpile" | "unflip"))
            && self.add_fix(&fix)
        {
            // (searching the decisions for a message mismatch: a repair at the same point applies
            // on top of the alternative being tried)
            let r = self.stats.restarts;
            let cand = self.overrides.get(&bt.key).copied();
            self.log(format!("restart {r}: {} -> {fix:?} (with decision {:?} candidate {cand:?})", e.reason, bt.key));
            self.stats.restarts += 1;
            return Ok(true);
        }
        if let Some(mut bt) = bt.clone()
            && (e.pos < bt.pos || e.pos == bt.pos && (!bt.state || e.reason.starts_with("state mismatch") || stat_mismatch(&e)))
        {
            // the decision changed for a mismatch made an earlier observation differ (or did not
            // get past it): its next alternative, else an earlier decision (within a budget)
            bt.tries += 1;
            if e.pos <= bt.key.0 + 1 && bt.parent.is_none() && !bt.state && (e.pos < bt.pos || e.reason.starts_with("no candidate for opponent") || e.reason != bt.reason) {
                // (the decisions it led to come next, with the alternative kept)
                let new: Vec<(Dkey, Vec<usize>)> = self
                    .recent
                    .iter()
                    .rev()
                    .filter(|&&(k, n, _)| k > bt.key && !self.overrides.contains_key(&k) && n > 1)
                    .map(|&(k, n, i)| (k, (0..n).filter(|&j| j as i64 != i).take(2).collect()))
                    .collect();
                let parent = (bt.key, self.overrides.get(&bt.key).copied());
                let maxd = new.iter().map(|(_, a)| a.len()).max().unwrap_or(0);
                let mut front = Vec::new();
                for d in 0..maxd {
                    for (k, a) in &new {
                        if d < a.len() {
                            front.push((*k, a[d], Some(parent)));
                        }
                    }
                }
                front.extend(bt.queue.drain(..));
                bt.queue = front;
            }
            if bt.tries > decision_tries() || bt.queue.is_empty() {
                for (kk, _, _) in &bt.list {
                    self.overrides.remove(kk);
                }
                self.overrides.remove(&bt.key);
                if let Some(p) = &bt.parent {
                    self.overrides.remove(&p.0);
                }
                self.state_bt = None;
                if let Some(fb) = bt.fallback.clone()
                    && self.add_fix(&fb)
                {
                    let r = self.stats.restarts;
                    self.log(format!("restart {r}: {} -> {fb:?} (no decision helped)", bt.reason));
                    self.stats.restarts += 1;
                    return Ok(true);
                }
                self.searched.insert(bt.pos);
                if self.suspect_round(bt.pos, bt.sig.clone()) {
                    self.stats.restarts += 1;
                    return Ok(true);
                }
                return div(Divergence::new(bt.pos, bt.reason.clone()));
            }
            self.overrides.remove(&bt.key);
            let entry = bt.queue.remove(0);
            let parent = entry.2;
            if let Some(p) = bt.parent
                && Some(p) != parent
            {
                self.overrides.remove(&p.0);
            }
            if let Some((pk, Some(pv))) = parent {
                self.overrides.insert(pk, pv);
            } else if let Some((pk, None)) = parent {
                self.overrides.remove(&pk);
            }
            bt.parent = parent;
            bt.key = entry.0;
            self.overrides.insert(bt.key, entry.1);
            let r = self.stats.restarts;
            self.log(format!("restart {r}: {} -> decision {:?} candidate {} (for the mismatch at {})", e.reason, bt.key, entry.1, bt.pos));
            self.stats.restarts += 1;
            self.state_bt = Some(bt);
            return Ok(true);
        }
        if bt.as_ref().is_some_and(|b| e.pos > b.pos) {
            self.state_bt = None;
        }
        if let Some((Fix::Swap(at, a, b), spos)) = self.last_swap.clone()
            && e.pos < spos
        {
            // the latest copy swap made things worse: undo it, so another point is tried
            self.last_swap = None;
            if let Some(lst) = self.swaps.get_mut(&at)
                && let Some(k) = lst.iter().rposition(|&p| p == (a, b))
            {
                lst.remove(k);
                let later: Vec<Cp> = self.swaps.keys().copied().filter(|&c| c > at && c != (0, 0)).collect();
                for c in later {
                    let v = self.swaps[&c].iter().map(|&(x, y)| (sw(x, a, b), sw(y, a, b))).collect();
                    self.swaps.insert(c, v);
                }
            }
            self.failed_swaps.insert((at, a, b));
            let r = self.stats.restarts;
            self.log(format!("restart {r}: {} -> undo swap {:?}", e.reason, (at, a, b)));
            self.stats.restarts += 1;
            return Ok(true);
        }
        if let Some(Fix::Swap(cp, a, b)) = e.fix
            && (self.failed_swaps.contains(&(cp, a, b)) || self.failed_swaps.contains(&(cp, b, a)))
        {
            e.fix = None;
        }
        self.last_swap = None;
        if let Some(f @ Fix::Swap(..)) = e.fix.clone()
            && self.add_fix(&f)
        {
            self.last_swap = Some((f.clone(), e.pos));
            let r = self.stats.restarts;
            self.log(format!("restart {r}: {} -> {f:?}", e.reason));
            self.stats.restarts += 1;
            return Ok(true);
        }
        let added = match e.fix.clone() {
            Some(f) => self.add_fix(&f),
            None => false,
        };
        if !added {
            // the decisions made since the last observed item are the ones whose consequence
            // diverged: try the latest one differently
            let state = e.reason.starts_with("state mismatch") || stat_mismatch(&e);
            let recent = if state { self.since_state.clone() } else { self.recent.clone() };
            let todo: Vec<(Dkey, Vec<usize>)> =
                recent.iter().rev().filter(|(k, _, _)| !self.overrides.contains_key(k)).map(|&(k, n, i)| (k, (0..n).filter(|&j| j as i64 != i).collect())).collect();
            let maxd = todo.iter().map(|(_, a)| a.len()).max().unwrap_or(0);
            let mut queue: Vec<(Dkey, usize, Option<(Dkey, Option<usize>)>)> = Vec::new();
            for d in 0..maxd {
                for (k, alts) in &todo {
                    if d < alts.len() {
                        queue.push((*k, alts[d], None));
                    }
                }
            }
            // (not again where a search already ran out)
            let alt = if !queue.is_empty() && !self.searched.contains(&e.pos) { Some(queue.remove(0)) } else { None };
            let fallback = e.fallback.clone();
            if alt.is_none()
                && let Some(fb) = fallback.clone()
                && self.add_fix(&fb)
            {
                let r = self.stats.restarts;
                self.log(format!("restart {r}: {} -> {fb:?} (no decision to try)", e.reason));
                self.stats.restarts += 1;
                return Ok(true);
            }
            if let Some((k, c, _)) = alt {
                self.overrides.insert(k, c);
                if self.state_bt.is_none() {
                    self.state_bt = Some(StateBt {
                        pos: e.pos,
                        reason: e.reason.clone(),
                        sig: divergence_sig(&e),
                        list: recent.clone(),
                        key: k,
                        queue,
                        state,
                        tries: 0,
                        fallback,
                        parent: None,
                    });
                }
                let r = self.stats.restarts;
                self.log(format!("restart {r}: {} -> decision {k:?} candidate {c}", e.reason));
                self.stats.restarts += 1;
                return Ok(true);
            }
            if !self.memo.is_empty() {
                // a remembered decision may be wrong under the repairs made since
                let r = self.stats.restarts;
                self.log(format!("restart {r}: {} -> forget remembered decisions", e.reason));
                self.memo.clear();
                self.memo_cards.clear();
                self.use_memo = false;
                self.stats.restarts += 1;
                return Ok(true);
            }
            let sig = divergence_sig(&e);
            if self.suspect_round(e.pos, sig) {
                self.stats.restarts += 1;
                return Ok(true);
            }
            return div(e);
        }
        let r = self.stats.restarts;
        let fix = e.fix.clone();
        self.log(format!("restart {r}: {} -> {fix:?}", e.reason));
        self.stats.restarts += 1;
        Ok(true)
    }

    pub fn add_fix(&mut self, fix: &Fix) -> bool {
        match fix.clone() {
            Fix::Force(key) => {
                if self.forces.contains(&key) || self.unforced.contains(&key) {
                    return false;
                }
                self.forces.insert(key);
                self.stats.forces += 1;
                true
            }
            Fix::ForceLater(key) => {
                // the forcing of that effect starts only after this position
                let mut moved = false;
                let pos = self.pos;
                for w in self.force_windows.iter_mut() {
                    if w.2 == key && w.0 <= pos && pos + 1 < w.1 {
                        *w = (pos + 1, w.1, key);
                        moved = true;
                    }
                }
                if moved {
                    self.stats.forces += 1;
                }
                moved
            }
            Fix::Unforce(key) => {
                let windows: BTreeSet<(usize, usize, FKey)> = self.force_windows.iter().copied().filter(|w| w.2 == key && w.0 <= self.pos && self.pos <= w.1).collect();
                if !self.forces.contains(&key) || windows.difference(&self.unforced_windows).next().is_none() {
                    return false;
                }
                self.unforced_windows.extend(windows);
                self.stats.forces += 1;
                true
            }
            Fix::Assume(key) => {
                if self.assumptions.contains(&key) {
                    return false;
                }
                // a later assumption for the same card replaces an earlier one
                let same = |a: &Asm| (a.0, a.1, a.2, a.3) == (key.0, key.1, key.2, key.3);
                for a in self.assumptions.iter().filter(|a| same(a)) {
                    self.failed_assumptions.insert(*a);
                }
                self.assumptions.retain(|a| !same(a));
                self.assumptions.push(key);
                self.hypothesis = None;
                self.stats.assumptions += 1;
                self.last_hyp_fix = Some("assume");
                true
            }
            Fix::AssumeMany(list) => {
                let new: Vec<Asm> = list
                    .into_iter()
                    .filter(|a| !self.assumptions.contains(a) && !self.assumptions.iter().any(|b| (b.0, b.1, b.2, b.3) == (a.0, a.1, a.2, a.3)))
                    .collect();
                if new.is_empty() {
                    return false;
                }
                self.stats.assumptions += new.len();
                self.assumptions.extend(new);
                self.hypothesis = None;
                self.last_hyp_fix = Some("assume");
                true
            }
            Fix::Unassume(a) => {
                if !self.assumptions.contains(&a) {
                    return false;
                }
                self.assumptions.retain(|x| *x != a);
                self.failed_assumptions.insert(a);
                self.hypothesis = None;
                self.stats.assumptions += 1;
                true
            }
            Fix::Exclude(mut key) => {
                let (pos, con, loc, ref seqs, code) = key;
                if hidden_pile(loc)
                    && seqs != &[ALL_SLOTS]
                    && self.exclusions.iter().chain(self.failed_exclusions.iter()).any(|x| (x.0, x.1, x.2) == (pos, con, loc) && x.4 == code && &x.3 != seqs)
                {
                    // (the code was excluded from another slot of this pile here already)
                    key = (pos, con, loc, vec![ALL_SLOTS], code);
                }
                if self.exclusions.contains(&key) || self.failed_exclusions.contains(&key) {
                    return false;
                }
                self.exclusions.push(key);
                self.hypothesis = None;
                self.stats.assumptions += 1;
                self.last_hyp_fix = Some("exclude");
                true
            }
            Fix::Swap(cp, a, b) => {
                if cp == (0, 0) && !self.role_swap_is_new(a, b) {
                    return false;
                }
                let lst = self.swaps.entry(cp).or_default();
                if cp != (0, 0) && (lst.contains(&(a, b)) || lst.contains(&(b, a))) {
                    return false;
                }
                lst.push((a, b));
                // swaps recorded later named the cards as they were without this one
                let later: Vec<Cp> = self.swaps.keys().copied().filter(|&c| c > cp && c != (0, 0)).collect();
                for c in later {
                    let v = self.swaps[&c].iter().map(|&(x, y)| (sw(x, a, b), sw(y, a, b))).collect();
                    self.swaps.insert(c, v);
                }
                self.stats.swaps += 1;
                true
            }
            Fix::IdentifyAll(entries) => {
                self.last_identify_count = entries.len();
                self.identities.extend(entries);
                self.hypothesis = None;
                self.last_hyp_fix = Some("identify_all");
                self.stats.assumptions += 1;
                true
            }
            Fix::Identify(entry) => {
                if self.identities.contains(&entry) {
                    return false;
                }
                self.identities.push(entry);
                self.hypothesis = None;
                self.last_hyp_fix = Some("identify");
                self.stats.assumptions += 1;
                true
            }
            Fix::NomatchCard(a, b, s) => {
                if !self.nomatch_cards.insert((a, b, s)) {
                    return false;
                }
                self.stats.assumptions += 1;
                true
            }
            Fix::Unflip(a, b) => {
                if !self.nomatch_piles.remove(&(a, b)) {
                    return false;
                }
                self.pile_dropped.insert(("nomatchpile", (a, b)));
                self.match_piles.insert((a, b));
                self.stats.assumptions += 1;
                true
            }
            Fix::NomatchPile(a, b) | Fix::MatchPile(a, b) => {
                let kind = fix.kind();
                let span = (a, b);
                let nomatch = kind == "nomatchpile";
                let (in_mine, in_other) = if nomatch {
                    (self.nomatch_piles.contains(&span), self.match_piles.contains(&span))
                } else {
                    (self.match_piles.contains(&span), self.nomatch_piles.contains(&span))
                };
                if in_mine || self.pile_dropped.contains(&(kind, span)) {
                    return false;
                }
                // the opposite repair of the same span, made for a divergence at this same point,
                // did not get past it: it is dropped
                if in_other && self.pile_made.get(&span) == Some(&self.pos) {
                    if nomatch {
                        self.match_piles.remove(&span);
                        self.pile_dropped.insert(("matchpile", span));
                    } else {
                        self.nomatch_piles.remove(&span);
                        self.pile_dropped.insert(("nomatchpile", span));
                    }
                }
                if nomatch {
                    self.nomatch_piles.insert(span);
                } else {
                    self.match_piles.insert(span);
                }
                self.pile_made.insert(span, self.pos);
                self.stats.assumptions += 1;
                true
            }
            Fix::NomatchLoc(a, b, loc) => {
                if !self.nomatch_loc_piles.insert((a, b, loc)) {
                    return false;
                }
                self.stats.assumptions += 1;
                true
            }
            Fix::NomatchId(a, b, cid) => {
                if !self.nomatch_id_spans.insert((a, b, cid)) {
                    return false;
                }
                self.stats.assumptions += 1;
                true
            }
            Fix::Respond(dkey, indices) => {
                self.tried_resp.insert((dkey, indices.clone()));
                self.forced_resp.insert(dkey, Response { indices: Some(indices.iter().map(|&i| i as i64).collect()), ..Response::default() });
                true
            }
            Fix::Decide(dkey, k) => {
                if self.overrides.get(&dkey) == Some(&k) {
                    return false;
                }
                self.overrides.insert(dkey, k);
                true
            }
            Fix::Match(cp, cid) => {
                let lst = self.matches.entry(cp.0).or_default();
                if lst.contains(&cid) {
                    return false;
                }
                lst.push(cid);
                self.stats.assumptions += 1;
                true
            }
            Fix::CodeTest(start, at, slot, codes) => {
                if self.code_tests.iter().any(|e| e.0 == (at, slot)) || self.code_tests_done.contains(&(at, slot)) {
                    return false;
                }
                self.code_tests.push(((at, slot), (start, codes)));
                true
            }
            Fix::MatchSet((a, b), taken, rest) => {
                if !self.matchsets.insert(fix.clone()) {
                    return false;
                }
                // (it refines a span over which all hidden cards matched, ending here)
                self.match_piles.retain(|sp| !(sp.1 == b && sp.0 <= a));
                for at in a..=b {
                    let lst = self.matches.entry(at).or_default();
                    for &c in &taken {
                        if !lst.contains(&c) {
                            lst.push(c);
                        }
                    }
                }
                self.nomatch_id_spans.extend(rest.iter().map(|&c| (a, b, c)));
                self.stats.assumptions += 1;
                true
            }
            Fix::Snapshot(cp) => self.snapshot_cps.insert(cp),
        }
    }

    pub fn done(&self) -> bool {
        self.pos >= self.horizon
    }

    pub fn advance(&mut self) -> R<()> {
        let it = self.items[self.pos].clone();
        let t = match &it {
            ObsItem::Msg(m) => Some(m.name()),
            _ => None,
        };
        if self.depth > 0 && (matches!(it, ObsItem::Prompt(..)) || t.is_some_and(boundary)) {
            self.probe_boundary = true;
            self.probe_after_boundary = 0;
        } else if self.depth > 0 && self.probe_boundary && t.is_some_and(|t| !matches!(t, "HINT" | "CARD_HINT" | "PLAYER_HINT")) {
            self.probe_after_boundary += 1;
        }
        if let ObsItem::Msg(Msg::Move { from, .. }) = &it
            && from.loc != 0
            && self.depth == 0
            && !self.assumed_vars.is_empty()
        {
            // (a card whose code an assumption named moves: its consequences follow)
            let lv = self.lockstep.get().ok().and_then(|t| t.var_at(from.con, from.loc, from.seq, 0).ok().flatten().map(|v| (v, t.vars[v].code)));
            if let Some((v, None)) = lv
                && let Some(a) = self.assumed_vars.get(&v).copied()
            {
                self.assumed_moves.push((self.pos, a));
            }
        }
        self.lockstep.process(it);
        self.pos += 1;
        self.stall = 0;
        let pos = self.pos;
        self.recent.retain(|d| d.0.0 + DECISION_WINDOW >= pos);
        Ok(())
    }
}

/// x with a and b exchanged.
pub fn sw(x: u32, a: u32, b: u32) -> u32 {
    if x == a {
        b
    } else if x == b {
        a
    } else {
        x
    }
}

/// Slot id -> id placed there after the start-of-duel swaps (_roles).
pub fn roles(pairs: &[(u32, u32)]) -> BTreeMap<u32, u32> {
    let mut roles: BTreeMap<u32, u32> = BTreeMap::new();
    for &(a, b) in pairs {
        roles.entry(a).or_insert(a);
        roles.entry(b).or_insert(b);
        for v in roles.values_mut() {
            *v = sw(*v, a, b);
        }
    }
    roles
}

/// The coin tosses, dice and random selections of the stream, for the engine to come out the same.
pub fn push_random(duel: &mut RDuel, items: &[ObsItem]) {
    for item in items {
        match item {
            ObsItem::Msg(Msg::TossCoin { results, .. }) => duel.push_forced_coins(results),
            ObsItem::Msg(Msg::TossDice { results, .. }) => duel.push_forced_dice(results),
            ObsItem::Msg(Msg::RandomSelected { cards, .. }) => {
                let locs: Vec<(u32, u32, u32, u32)> =
                    cards.iter().map(|c| (c.con, c.loc, c.seq, if c.loc & LOCATION_OVERLAY != 0 { c.pos } else { 0 })).collect();
                duel.push_forced_random_select(&locs);
            }
            _ => {}
        }
    }
}
