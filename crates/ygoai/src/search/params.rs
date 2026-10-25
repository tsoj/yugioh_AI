//! The search's parameters.

/// (values are in LP: the learned eval's by its LP anchor)
pub const WIN: f64 = 10_000_000.0;
/// (a safety bound only: the clock ends the iterations)
pub const MAX_DEPTH: i32 = 100;
/// (without a time control: a fixed depth)
pub const DEPTH_NO_CLOCK: i32 = 2;
/// (a card selection: the first choices only, below the root (of several cards: the first sets,
/// `card_sets`), and a one-card selection at the root; the root's width 3 -> 5 by SPSA run 1, see
/// FUTILITY_MARGIN)
pub const CARD_WIDTH: usize = 3;
pub const ROOT_CARD_WIDTH: usize = 5;
/// (the root's other decisions, `root_candidates`: at most this many choices of several cards,
/// tributes, sums, places, orders or counters, the subsets among the rules' first
/// ROOT_COMBO_CARDS cards or places; a sum's search for valid selections stops after SUM_STEPS;
/// SPRT at 20+0.2 H1, +11.5 +/- 9.2 at 2836 games, 2026-10-18)
pub const ROOT_COMBO_WIDTH: usize = 5;
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
/// (the budget's scale after each iteration, Reckless's: max(TM_NODES_MIN, TM_NODES_BASE -
/// TM_NODES_PER_FRACTION * the best move's share of the root's nodes) times max(TM_STABILITY_MIN,
/// TM_STABILITY_BASE - TM_STABILITY_PER_ITERATION * the iterations it has stayed the best))
pub const TM_NODES_BASE: f64 = 3.1838;
pub const TM_NODES_PER_FRACTION: f64 = 2.6554;
pub const TM_NODES_MIN: f64 = 0.5460;
pub const TM_STABILITY_BASE: f64 = 1.2881;
pub const TM_STABILITY_PER_ITERATION: f64 = 0.0440;
pub const TM_STABILITY_MIN: f64 = 0.7160;
/// (times this: the scale's mean on the bench's searches was 1.8 without it, 1 with it)
pub const TM_SCALE_NORM: f64 = 0.556;

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
/// null window +11.1 +/- 8.8 at 3074 games at 1000 nodes, 2026-10-16. Depths are fractional)
pub const LMR_MIN_DEPTH: f64 = 2.0;
pub const LMR_MIN_MOVE_COUNTER: usize = 2;
pub const LMR_REDUCTION: f64 = 1.0;
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

// The search features that failed their SPRTs and their parameters were removed on 2026-10-18
// (pvs, rfp, rfpall, conthist, qsearch, narrowext, killers, nullmove, online, lmrlog, resample,
// deckpick, passrisk, and searching the decisions inside a move: unselect, tribute, multicard,
// sum, position, place): see TODO.md and the git history before that. Since 2026-10-19 untested
// and failed features are branches (on there), not flags: futility-reductions, inner-decisions,
// pvext-open-window, pv-half-ply, pv-move-half-ply, lmr-linear.

/// Declares `Tune`: per parameter its type, default and how a value set at run time is bounded
/// (a count rounds to the nearest integer; depths are fractional).
macro_rules! tunable {
    ($($name:ident: $ty:ty = $default:expr, $bound:expr;)*) => {
        /// The search parameters that can be set at run time (for tuning, `ygoai-spsa`): by default
        /// the constants above; $YGOAI_TUNE as "name=value,..." overrides some (read once).
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct Tune {
            $(pub $name: $ty,)*
        }

        impl Default for Tune {
            fn default() -> Self {
                Tune { $($name: $default,)* }
            }
        }

        impl Tune {
            /// The parameters' names, in a fixed order.
            pub const NAMES: &[&str] = &[$(stringify!($name),)*];

            /// The parameter `name`'s value (as a number).
            pub fn get(&self, name: &str) -> Option<f64> {
                match name {
                    $(stringify!($name) => Some(self.$name as f64),)*
                    _ => None,
                }
            }

            /// Sets the parameter `name`; false for an unknown name.
            pub fn set(&mut self, name: &str, value: f64) -> bool {
                match name {
                    $(stringify!($name) => {
                        let bound: fn(f64) -> $ty = $bound;
                        self.$name = bound(value)
                    })*
                    _ => return false,
                }
                true
            }
        }
    };
}

/// (a count: the nearest non-negative integer)
fn count(value: f64) -> usize {
    value.round().max(0.0) as usize
}

tunable! {
    card_width: usize = CARD_WIDTH, |v| count(v).max(1);
    root_card_width: usize = ROOT_CARD_WIDTH, |v| count(v).max(1);
    lmr_min_depth: f64 = LMR_MIN_DEPTH, |v| v;
    lmr_min_move_counter: usize = LMR_MIN_MOVE_COUNTER, count;
    lmr_reduction: f64 = LMR_REDUCTION, |v| v.max(0.0);
    futility_max_depth: f64 = FUTILITY_MAX_DEPTH, |v| v;
    futility_margin: f64 = FUTILITY_MARGIN, |v| v;
    futility_margin_per_ply: f64 = FUTILITY_MARGIN_PER_PLY, |v| v;
    hist_decay: f64 = HIST_DECAY, |v| v.clamp(0.0, 1.0);
    hist_exponent: f64 = HIST_EXPONENT, |v| v;
    hist_malus: f64 = HIST_MALUS, |v| v.max(0.0);
}

impl Tune {
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
