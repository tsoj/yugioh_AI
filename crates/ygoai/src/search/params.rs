//! The search's parameters and its optional features ($YGOAI_SEARCH).

pub const WIN: f64 = 100000.0;
/// (a safety bound only: the clock ends the iterations)
pub const MAX_DEPTH: i32 = 100;
/// (without a time control: a fixed depth)
pub const DEPTH_NO_CLOCK: i32 = 2;
/// (per node, after ordering)
pub const MAX_CHILDREN: usize = 12;
/// (a one-card selection: the rules' first choices only, below the root and at the root)
pub const CARD_WIDTH: usize = 3;
pub const ROOT_CARD_WIDTH: usize = 3;
/// (the searched decisions inside a move, each a feature: how many of the rules' first choices)
pub const UNSELECT_WIDTH: usize = 3;
pub const TRIBUTE_WIDTH: usize = 4;
pub const MULTI_CARDS: usize = 4;
pub const MULTI_WIDTH: usize = 6;
pub const SUM_WIDTH: usize = 4;
pub const SUM_STEPS: usize = 5000;

// time management
/// (without `predict`: a new iteration starts while less than this share of the budget is used)
pub const NEXT_ITERATION_SHARE: f64 = 0.5;
/// (`predict`: the prior of each depth's node growth from the previous iteration, its weight in
/// iterations, and the most weight the observed growths keep, so that the table follows the game)
pub const GROWTH_PRIOR: f64 = 1.6;
pub const GROWTH_PRIOR_WEIGHT: f64 = 4.0;
pub const GROWTH_MAX_WEIGHT: f64 = 200.0;
/// (the hard deadline: max(budget, min(HARD_FACTOR * budget, HARD_SHARE * the clock)))
pub const HARD_FACTOR: f64 = 3.0;
pub const HARD_SHARE: f64 = 0.25;

/// (prompts answered by the rules between two plies)
pub const MAX_AUTO_STEPS: usize = 300;
pub const TT_BITS: u32 = 16;
pub const HIST_BITS: u32 = 14;
/// (continuation history entries: (previous move, move) pairs)
pub const CONT_BITS: u32 = 16;
/// (the history tables' decay per decision)
pub const HIST_DECAY: f64 = 0.5;

/// (the null window's width, 1 LP: the smallest step of a value)
pub const NULL_WINDOW: f64 = 0.01;
/// (reverse futility pruning at null-window nodes up to this depth, by this margin per ply)
pub const RFP_MAX_DEPTH: i32 = 2;
pub const RFP_MARGIN_MULTIPLIER: f64 = 12.0;
/// (late move reductions from this depth on, for the moves after the first LMR_MIN_MOVE_COUNTER)
pub const LMR_MIN_DEPTH: i32 = 2;
pub const LMR_MIN_MOVE_COUNTER: usize = 2;
/// (futility pruning up to this depth, margin FUTILITY_MARGIN + FUTILITY_MARGIN_PER_PLY * depth: about
/// the 99th percentile of alpha minus the static eval over the later moves that raised alpha, at
/// depth 1 to 3 (49, 52, 55; 4M moves of 40 bench openings at 3000 nodes, mover2 eval, lmr))
pub const FUTILITY_MAX_DEPTH: i32 = 3;
pub const FUTILITY_MARGIN: f64 = 45.0;
pub const FUTILITY_MARGIN_PER_PLY: f64 = 4.0;
/// (quiescence plies past the nominal depth)
pub const MAX_QUIESCE_PLY: i32 = 6;

/// Search features, each off unless named in $YGOAI_SEARCH (a comma list; unset: DEFAULT_FEATURES):
/// `pvs`, `rfp` (reverse futility pruning at null-window nodes; with pvs), `futility` (futility
/// pruning up to FUTILITY_MAX_DEPTH), `lmr`, `conthist` (continuation history), `qsearch`
/// (battle and chain decisions past the depth).
/// Decisions inside a move that the search can make instead of the rules' player (each the rules'
/// first few choices): `unselect` (card by card selections, e.g. most materials), `tribute`,
/// `multicard` (selections of several cards), `sum` (selections by a sum: Synchro and Ritual
/// materials), `position`, `place` (the zone).
/// `predict`: with a clock, the next iteration starts only if it is expected to end within the
/// time budget (by this iteration's time and the depth's node growth, learned across searches)
/// instead of while less than half the budget is used.
/// The features that passed their SPRTs (LMR: +19.1 +/- 14.9, LLR +3.01, 1204 games, 2026-10-13),
/// and `predict`, taken without a pass (+3.8 +/- 11.7, LLR -0.35 at 1760 games, 2026-10-14) for
/// the room it leaves the time management. (The node-budget search, `lmr_noresearch`, `rfp_kind`
/// and `pv_ext` were removed after failing their tests: git history before the search/ split.)
pub const DEFAULT_FEATURES: &str = "lmr,predict";

#[derive(Clone, Copy, Default, Debug)]
pub struct Features {
    pub pvs: bool,
    pub rfp: bool,
    pub futility: bool,
    pub lmr: bool,
    pub conthist: bool,
    pub qsearch: bool,
    pub unselect: bool,
    pub tribute: bool,
    pub multicard: bool,
    pub sum: bool,
    pub position: bool,
    pub place: bool,
    pub predict: bool,
}

impl Features {
    pub fn parse(list: &str) -> Features {
        let mut f = Features::default();
        for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            match name {
                "pvs" => f.pvs = true,
                "rfp" => f.rfp = true,
                "futility" => f.futility = true,
                "lmr" => f.lmr = true,
                "conthist" => f.conthist = true,
                "qsearch" => f.qsearch = true,
                "unselect" => f.unselect = true,
                "tribute" => f.tribute = true,
                "multicard" => f.multicard = true,
                "sum" => f.sum = true,
                "position" => f.position = true,
                "place" => f.place = true,
                "predict" => f.predict = true,
                _ => panic!("YGOAI_SEARCH: unknown feature {name}"),
            }
        }
        f
    }

    /// The features of $YGOAI_SEARCH, DEFAULT_FEATURES when it is unset (read once).
    pub fn from_env() -> Features {
        static F: std::sync::OnceLock<Features> = std::sync::OnceLock::new();
        *F.get_or_init(|| Features::parse(&std::env::var("YGOAI_SEARCH").unwrap_or_else(|_| DEFAULT_FEATURES.to_string())))
    }
}
