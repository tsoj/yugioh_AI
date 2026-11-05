//! The search benchmark (bot-bench-search, `bot bench`): plays book openings with the heuristic
//! player and, at each searched decision of player 0, searches the true duel with fresh tables per
//! opening. With a node or depth budget it is deterministic: the checksum over the searches' moves,
//! depths and node counts shows that a change kept the bot's behavior, and the node count at a
//! fixed depth measures the move ordering (OpenBench's bench signature).

use std::path::Path;
use std::time::Instant;

use ygo_arbiter::actions::encode;
use ygo_arbiter::game::{DuelConfig, new_record, run_until_prompt};
use ygo_arbiter::matchrun::load_book;
use ygo_arbiter::messages::Msg;
use ygo_arbiter::observe::{auto_response, is_hidden_prompt};

use crate::heuristic::Heuristic;
use crate::search::stats::Extension;
use crate::search::{AlphaBeta, NodeStats, SearchParams, candidates, evaluate_at, is_searched};

/// (decisions per opening at most)
const MAX_DECISIONS: usize = 400;

/// The default of `bot bench`: the book books/bench.jsonl, its openings, the depth.
pub const BENCH_OPENINGS: usize = 6;
pub const BENCH_DEPTH: i32 = 8;

/// A search's limit: seconds, nodes ("Nn") or depth ("Nd").
#[derive(Clone, Copy, Debug)]
pub enum Budget {
    Seconds(f64),
    Nodes(u64),
    Depth(i32),
}

impl Budget {
    pub fn parse(s: &str) -> Result<Budget, String> {
        let parsed = match (s.strip_suffix('n'), s.strip_suffix('d')) {
            (Some(k), _) => k.parse().ok().map(Budget::Nodes),
            (_, Some(k)) => k.parse().ok().map(Budget::Depth),
            _ => s.parse().ok().map(Budget::Seconds),
        };
        parsed.ok_or_else(|| format!("bad budget: {s}"))
    }

    fn params(self) -> SearchParams {
        let (budget, max_nodes, depth) = match self {
            Budget::Seconds(s) => (Some(s), None, None),
            Budget::Nodes(n) => (None, Some(n), None),
            Budget::Depth(d) => (None, None, Some(d)),
        };
        SearchParams { budget, elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes, depth, exact_root: false }
    }
}

/// What the benchmark does besides the searches.
#[derive(Default)]
pub struct Options {
    /// openings of the book skipped first
    pub skip: usize,
    /// collect the expanded nodes' statistics
    pub stats: bool,
    /// time the evaluation (1000 calls at each searched position)
    pub time_eval: bool,
}

/// One search.
pub struct Search {
    pub opening: usize,
    /// (the search's index in its opening)
    pub index: usize,
    pub prompt: &'static str,
    pub best: usize,
    /// the completed depth, the greatest height, the main line's height
    pub depth: i32,
    pub seldepth: i32,
    pub pv_height: i32,
    pub nodes: u64,
    pub secs: f64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    pub broken: u64,
    /// the main-line extension's nodes in this search
    pub extension: Extension,
    /// the best move's share of the nodes, the time budget's scale
    pub best_fraction: f64,
    pub tm_scale: f64,
}

impl Search {
    /// Whether the main-line extension extended the last iteration's main line.
    pub fn extended(&self) -> bool {
        self.pv_height > self.depth
    }
}

/// The benchmark's result.
#[derive(Default)]
pub struct Report {
    pub searches: Vec<Search>,
    /// (with `Options::stats`)
    pub stats: NodeStats,
    /// the evaluation's seconds and calls (with `Options::time_eval`)
    pub eval: (f64, u64),
}

impl Report {
    pub fn nodes(&self) -> u64 {
        self.searches.iter().map(|s| s.nodes).sum()
    }

    pub fn secs(&self) -> f64 {
        self.searches.iter().map(|s| s.secs).sum()
    }

    pub fn checksum(&self) -> u64 {
        self.searches.iter().fold(0u64, |c, s| c.wrapping_mul(0x100000001b3) ^ (s.best as u64) ^ ((s.depth as u64) << 8) ^ (s.nodes << 16))
    }
}

/// Runs the benchmark on the first `openings` of `book` (ygo_arbiter::game::init done).
pub fn run(book: &Path, openings: usize, budget: Budget, opts: &Options) -> Result<Report, String> {
    let book = load_book(book, usize::MAX, openings)?;
    let params = budget.params();
    let mut report = Report::default();
    for (oi, opening) in book.iter().enumerate().take(openings).skip(opts.skip) {
        let record = new_record(&opening.decks, opening.seed, &DuelConfig::default());
        let mut duel = record.setup(None).map_err(|e| format!("{e:?}"))?;
        duel.ocg_start_duel();
        let mut players = [Heuristic::new(1, 0), Heuristic::new(2, 1)];
        let mut ab = AlphaBeta::default();
        if opts.stats {
            ab.stats = Some(Default::default());
        }
        let mut prompt: Option<Msg> = None;
        let mut index = 0;
        for _ in 0..MAX_DECISIONS {
            let step = run_until_prompt(&mut duel, prompt.as_ref()).map_err(|e| format!("{e:?}"))?;
            if step.ended || step.messages.iter().any(|m| matches!(m, Msg::Win { .. })) {
                break;
            }
            for m in &step.messages {
                for p in &mut players {
                    p.observe_msg(m);
                }
            }
            let p = step.prompt.ok_or("no prompt")?;
            let player = p.player().unwrap();
            let cands = candidates(&p);
            if player == 0 && is_searched(&p) && cands.len() > 1 {
                if opts.time_eval {
                    let t = Instant::now();
                    let mut acc = 0.0;
                    for _ in 0..1000 {
                        acc += evaluate_at(&mut duel, 0, Some(&p));
                    }
                    report.eval.0 += t.elapsed().as_secs_f64();
                    report.eval.1 += 1000;
                    std::hint::black_box(acc);
                }
                ab.prepare();
                let (nodes, probes, hits, broken, extension) = (ab.nodes, ab.tt_probes, ab.tt_hits, ab.broken, ab.extension);
                let t = Instant::now();
                let (best, done) = ab.search(&duel, &p, &cands, &params);
                report.searches.push(Search {
                    opening: oi,
                    index,
                    prompt: p.name(),
                    best,
                    depth: done,
                    seldepth: ab.seldepth,
                    pv_height: ab.pv_height,
                    nodes: ab.nodes - nodes,
                    secs: t.elapsed().as_secs_f64(),
                    tt_probes: ab.tt_probes - probes,
                    tt_hits: ab.tt_hits - hits,
                    broken: ab.broken - broken,
                    extension: ab.extension - extension,
                    best_fraction: ab.best_fraction,
                    tm_scale: ab.tm_scale,
                });
                index += 1;
            }
            let resp = if is_hidden_prompt(&p) { auto_response() } else { players[player as usize].respond_to(&p, None).map_err(|e| format!("{e:?}"))? };
            duel.ocg_set_response(&encode(&p, &resp).map_err(|e| format!("{e:?}"))?);
            prompt = Some(p);
        }
        if let Some(st) = ab.stats.take() {
            report.stats.merge(st);
        }
    }
    Ok(report)
}
