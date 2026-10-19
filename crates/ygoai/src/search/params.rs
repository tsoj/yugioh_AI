//! The search's parameters.

/// (values are in LP: the learned eval's by its LP anchor)
pub const WIN: f64 = 10_000_000.0;
/// (a safety bound only: the clock ends the iterations)
pub const MAX_DEPTH: i32 = 100;
/// (without a time control: a fixed depth)
pub const DEPTH_NO_CLOCK: i32 = 2;
/// (a one-card selection: the rules' first choices only, below the root and at the root; the root's
/// width 3 -> 5 by SPSA run 1, see FUTILITY_MARGIN)
pub const CARD_WIDTH: usize = 3;
pub const ROOT_CARD_WIDTH: usize = 5;
/// (the root's other decisions, `root_candidates`: at most this many choices of several cards,
/// tributes, sums, places, orders or counters, the subsets among the rules' first
/// ROOT_COMBO_CARDS cards or places; a sum's search for valid selections stops after SUM_STEPS;
/// SPRT at 20+0.2 H1, +11.5 +/- 9.2 at 2836 games, 2026-10-18)
pub const ROOT_COMBO_WIDTH: usize = 5;
/// (the decisions inside a move searched in the tree too if > 0, `is_inner`: at most this many
/// choices each, with depth left; at the horizon the rules answer)
pub const INNER_WIDTH: usize = 0;
pub const ROOT_COMBO_CARDS: usize = 6;
pub const SUM_STEPS: usize = 20_000;

// time management
/// (the next iteration starts only if it is expected to end within the time budget: by this
/// iteration's time and the depth's node growth from the previous iteration, learned across
/// searches. The prior of that growth, its weight in iterations, and the most weight the observed
/// growths keep, so that the table follows the game)
pub const GROWTH_PRIOR: f64 = 1.6;
pub const GROWTH_PRIOR_WEIGHT: f64 = 4.0;
pub const GROWTH_MAX_WEIGHT: f64 = 200.0;
/// (the hard deadline: max(budget, min(HARD_FACTOR * budget, HARD_SHARE * the clock)), but at most
/// HARD_MAX_SHARE of the clock and at least TIME_BUFFER before it runs out; an iteration it
/// interrupts is dropped, and without a completed one the rules' first choice is played)
pub const HARD_FACTOR: f64 = 3.0;
pub const HARD_SHARE: f64 = 0.25;
pub const HARD_MAX_SHARE: f64 = 0.5;
pub const TIME_BUFFER: f64 = 0.05;

/// The hard limit's seconds for a decision with `budget` seconds and `remaining` on the clock.
pub fn hard_limit(budget: f64, remaining: f64) -> f64 {
    budget.max((HARD_FACTOR * budget).min(HARD_SHARE * remaining)).min(HARD_MAX_SHARE * remaining).min(remaining - TIME_BUFFER).max(0.0)
}

/// (prompts answered by the rules between two plies)
pub const MAX_AUTO_STEPS: usize = 300;
pub const TT_BITS: u32 = 16;
pub const HIST_BITS: u32 = 14;
/// (the history table's decay per decision; a cutoff's bonus is depth^HIST_EXPONENT, the moves
/// tried before it share HIST_MALUS times that as a penalty)
pub const HIST_DECAY: f64 = 0.5;
pub const HIST_EXPONENT: f64 = 2.0;
pub const HIST_MALUS: f64 = 1.0;

/// (the null window's width, 1 LP: the smallest step of a value)
pub const NULL_WINDOW: f64 = 1.0;
/// (late move reductions by LMR_REDUCTION plies from this depth on, for the moves after the first
/// LMR_MIN_MOVE_COUNTER, with a null window: SPRT H1 +19.1 +/- 14.9 at 1204 games, 2026-10-13; the
/// null window +11.1 +/- 8.8 at 3074 games at 1000 nodes, 2026-10-16. Depths are fractional: the
/// reduction grows by LMR_REDUCTION_PER_MOVE for each later move)
pub const LMR_MIN_DEPTH: f64 = 2.0;
pub const LMR_MIN_MOVE_COUNTER: usize = 2;
pub const LMR_REDUCTION: f64 = 1.0;
pub const LMR_REDUCTION_PER_MOVE: f64 = 0.0;
/// (futility pruning up to this depth, margin FUTILITY_MARGIN + FUTILITY_MARGIN_PER_PLY * depth:
/// 5 + 2 * depth points of the mover2 eval (1 point = 116.68 LP), chosen for +2 plies at 3000
/// nodes (mean depth 9.65 -> 12.37); SPRT vs lmr,predict H1, +19.4 +/- 13.5 at 1364 games. The
/// measured 99th percentile of alpha minus the static eval over the later moves that raised
/// alpha was 49, 52, 55, 72 points at depth 1-4: a margin about ten times larger (SPRT stopped
/// near zero). Then SPSA run 1 (ygoai-spsa, 1000 nodes vs a 2000-node reference, 3000
/// iterations; validation loss 538 -> ~500): margins 583.4 -> 546.6 and 233.4 -> 228.1 LP, root
/// card width 3 -> 5; SPRT at 20+0.2 H1, +17.8 +/- 12.8 at 1484 games, 2026-10-18)
pub const FUTILITY_MAX_DEPTH: f64 = 8.0;
pub const FUTILITY_MARGIN: f64 = 546.633177069564;
pub const FUTILITY_MARGIN_PER_PLY: f64 = 228.1061522305487;
/// (futility reductions instead of the pruning if > 0, Nalwald's with the static eval after the move
/// in place of static eval + SEE: from the move FUTILITY_REDUCTION_MIN_MOVE_COUNTER (0-based) on, a
/// move whose position is short of alpha by k times FUTILITY_REDUCTION_DIV LP is searched k plies
/// less, fractional)
pub const FUTILITY_REDUCTION_DIV: f64 = 0.0;
pub const FUTILITY_REDUCTION_MIN_MOVE_COUNTER: usize = 1;
/// (the decisions inside a move searched one ply deep at the horizon too, so that the search
/// finishes the move, not the rules: 1 on the main line, the first move at every node, SPRT at
/// 10+0.1 H1, +25.4 +/- 16.4 at 890 games, 2026-10-19; 2 at every node whose window is wider than
/// the null window)
pub const PV_EXTENSION: usize = 1;

// The search features that failed their SPRTs and their parameters were removed on 2026-10-18
// (pvs, rfp, rfpall, conthist, qsearch, narrowext, killers, nullmove, online, lmrlog, resample,
// deckpick, passrisk, and searching the decisions inside a move: unselect, tribute, multicard,
// sum, position, place): see TODO.md and the git history before that.

/// The search parameters that can be set at run time (for tuning, `ygoai-spsa`): by default the
/// constants above; $YGOAI_TUNE as "name=value,..." overrides some (read once).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tune {
    pub card_width: usize,
    pub root_card_width: usize,
    pub inner_width: usize,
    pub lmr_min_depth: f64,
    pub lmr_min_move_counter: usize,
    pub lmr_reduction: f64,
    pub lmr_reduction_per_move: f64,
    pub futility_max_depth: f64,
    pub futility_margin: f64,
    pub futility_margin_per_ply: f64,
    pub futility_reduction_div: f64,
    pub futility_reduction_min_move_counter: usize,
    pub pv_extension: usize,
    pub hist_decay: f64,
    pub hist_exponent: f64,
    pub hist_malus: f64,
}

impl Default for Tune {
    fn default() -> Self {
        Tune {
            card_width: CARD_WIDTH,
            root_card_width: ROOT_CARD_WIDTH,
            inner_width: INNER_WIDTH,
            lmr_min_depth: LMR_MIN_DEPTH,
            lmr_min_move_counter: LMR_MIN_MOVE_COUNTER,
            lmr_reduction: LMR_REDUCTION,
            lmr_reduction_per_move: LMR_REDUCTION_PER_MOVE,
            futility_max_depth: FUTILITY_MAX_DEPTH,
            futility_margin: FUTILITY_MARGIN,
            futility_margin_per_ply: FUTILITY_MARGIN_PER_PLY,
            futility_reduction_div: FUTILITY_REDUCTION_DIV,
            futility_reduction_min_move_counter: FUTILITY_REDUCTION_MIN_MOVE_COUNTER,
            pv_extension: PV_EXTENSION,
            hist_decay: HIST_DECAY,
            hist_exponent: HIST_EXPONENT,
            hist_malus: HIST_MALUS,
        }
    }
}

impl Tune {
    /// The parameters' names, in a fixed order.
    pub const NAMES: [&'static str; 16] = [
        "card_width",
        "root_card_width",
        "inner_width",
        "lmr_min_depth",
        "lmr_min_move_counter",
        "lmr_reduction",
        "lmr_reduction_per_move",
        "futility_max_depth",
        "futility_margin",
        "futility_margin_per_ply",
        "futility_reduction_div",
        "futility_reduction_min_move_counter",
        "pv_extension",
        "hist_decay",
        "hist_exponent",
        "hist_malus",
    ];

    /// The parameter `name`'s value (as a number).
    pub fn get(&self, name: &str) -> Option<f64> {
        Some(match name {
            "card_width" => self.card_width as f64,
            "root_card_width" => self.root_card_width as f64,
            "inner_width" => self.inner_width as f64,
            "lmr_min_depth" => self.lmr_min_depth,
            "lmr_min_move_counter" => self.lmr_min_move_counter as f64,
            "lmr_reduction" => self.lmr_reduction,
            "lmr_reduction_per_move" => self.lmr_reduction_per_move,
            "futility_max_depth" => self.futility_max_depth,
            "futility_margin" => self.futility_margin,
            "futility_margin_per_ply" => self.futility_margin_per_ply,
            "futility_reduction_div" => self.futility_reduction_div,
            "futility_reduction_min_move_counter" => self.futility_reduction_min_move_counter as f64,
            "pv_extension" => self.pv_extension as f64,
            "hist_decay" => self.hist_decay,
            "hist_exponent" => self.hist_exponent,
            "hist_malus" => self.hist_malus,
            _ => return None,
        })
    }

    /// Sets the parameter `name` (a count to the nearest integer; depths are fractional); false for
    /// an unknown name.
    pub fn set(&mut self, name: &str, v: f64) -> bool {
        let n = v.round().max(0.0);
        match name {
            "card_width" => self.card_width = (n as usize).max(1),
            "root_card_width" => self.root_card_width = (n as usize).max(1),
            "inner_width" => self.inner_width = n as usize,
            "lmr_min_depth" => self.lmr_min_depth = v,
            "lmr_min_move_counter" => self.lmr_min_move_counter = n as usize,
            "lmr_reduction" => self.lmr_reduction = v.max(0.0),
            "lmr_reduction_per_move" => self.lmr_reduction_per_move = v.max(0.0),
            "futility_max_depth" => self.futility_max_depth = v,
            "futility_margin" => self.futility_margin = v,
            "futility_margin_per_ply" => self.futility_margin_per_ply = v,
            "futility_reduction_div" => self.futility_reduction_div = v.max(0.0),
            "futility_reduction_min_move_counter" => self.futility_reduction_min_move_counter = (n as usize).max(1),
            "pv_extension" => self.pv_extension = n as usize,
            "hist_decay" => self.hist_decay = v.clamp(0.0, 1.0),
            "hist_exponent" => self.hist_exponent = v,
            "hist_malus" => self.hist_malus = v.max(0.0),
            _ => return false,
        }
        true
    }

    /// The defaults with "name=value,..." applied.
    pub fn parse(list: &str) -> Result<Tune, String> {
        let mut t = Tune::default();
        for item in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (name, value) = item.split_once('=').ok_or_else(|| format!("expected name=value: {item}"))?;
            let v: f64 = value.trim().parse().map_err(|_| format!("bad value: {item}"))?;
            if !t.set(name.trim(), v) {
                return Err(format!("unknown parameter {name}"));
            }
        }
        Ok(t)
    }

    /// "name=value,..." of all the parameters.
    pub fn to_list(&self) -> String {
        Tune::NAMES.iter().map(|n| format!("{n}={}", self.get(n).unwrap())).collect::<Vec<_>>().join(",")
    }

    /// The defaults with $YGOAI_TUNE applied (read once).
    pub fn from_env() -> Tune {
        static T: std::sync::OnceLock<Tune> = std::sync::OnceLock::new();
        *T.get_or_init(|| match std::env::var("YGOAI_TUNE") {
            Ok(s) => Tune::parse(&s).unwrap_or_else(|e| panic!("YGOAI_TUNE: {e}")),
            Err(_) => Tune::default(),
        })
    }
}
