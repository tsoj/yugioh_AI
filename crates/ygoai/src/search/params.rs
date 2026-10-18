//! The search's parameters.

/// (values are in LP: the learned eval's by its LP anchor)
pub const WIN: f64 = 10_000_000.0;
/// (a safety bound only: the clock ends the iterations)
pub const MAX_DEPTH: i32 = 100;
/// (without a time control: a fixed depth)
pub const DEPTH_NO_CLOCK: i32 = 2;
/// (per node, after ordering)
pub const MAX_CHILDREN: usize = 12;
/// (a one-card selection: the rules' first choices only, below the root and at the root)
pub const CARD_WIDTH: usize = 3;
pub const ROOT_CARD_WIDTH: usize = 3;

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
/// (the history table's decay per decision)
pub const HIST_DECAY: f64 = 0.5;

/// (the null window's width, 1 LP: the smallest step of a value)
pub const NULL_WINDOW: f64 = 1.0;
/// (late move reductions by one ply from this depth on, for the moves after the first
/// LMR_MIN_MOVE_COUNTER, with a null window: SPRT H1 +19.1 +/- 14.9 at 1204 games, 2026-10-13; the
/// null window +11.1 +/- 8.8 at 3074 games at 1000 nodes, 2026-10-16)
pub const LMR_MIN_DEPTH: i32 = 2;
pub const LMR_MIN_MOVE_COUNTER: usize = 2;
/// (futility pruning up to this depth, margin FUTILITY_MARGIN + FUTILITY_MARGIN_PER_PLY * depth:
/// 5 + 2 * depth points of the mover2 eval (1 point = 116.68 LP), chosen for +2 plies at 3000
/// nodes (mean depth 9.65 -> 12.37); SPRT vs lmr,predict H1, +19.4 +/- 13.5 at 1364 games. The
/// measured 99th percentile of alpha minus the static eval over the later moves that raised
/// alpha was 49, 52, 55, 72 points at depth 1-4: a margin about ten times larger (SPRT stopped
/// near zero))
pub const FUTILITY_MAX_DEPTH: i32 = 8;
pub const FUTILITY_MARGIN: f64 = 5.0 * 116.68036281706055;
pub const FUTILITY_MARGIN_PER_PLY: f64 = 2.0 * 116.68036281706055;

// The search features that failed their SPRTs and their parameters were removed on 2026-10-18
// (pvs, rfp, rfpall, conthist, qsearch, narrowext, killers, nullmove, online, lmrlog, resample,
// deckpick, passrisk, and searching the decisions inside a move: unselect, tribute, multicard,
// sum, position, place): see TODO.md and the git history before that.
