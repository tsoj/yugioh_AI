//! Speed of the alpha-beta search (crate::search): plays book openings with the heuristic player
//! and, at each searched decision of player 0, searches the true duel for `budget` seconds, or
//! `N` nodes with a budget of the form "Nn", or to depth `N` with "Nd" (both deterministic: the
//! nodes a fixed depth takes measure the move ordering; the checksum over the
//! searches' results and node counts shows that a change kept the bot's behavior).
//!
//!   bot-bench-search BOOK [OPENINGS] [BUDGET]
//!
//! With $BOT_SEARCH_STATS set, it also prints statistics of the expanded nodes (search::NodeStats):
//! widths by prompt (options listed, children after the caps, children searched before a cutoff),
//! cutoffs by prompt and TT state, LMR re-searches by move index, subtree sizes by depth; and
//! writes all the counts to the file $BOT_SEARCH_STATS as JSON lines.
//!
//! $BOT_BEST: write each search's chosen move to that file (a line "opening search move").
//! $BOT_SKIP: skip that many book openings first (to split a run).

use std::collections::BTreeMap;
use std::io::Write;

use bot::bench::{Budget, Options};
use bot::search::NodeStats;
use bot::search::stats::{Extension, Reductions, Subtree, TtEntry, Width};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(2).map_or(2, |s| s.parse().unwrap());
    let budget = Budget::parse(args.get(3).map_or("0.5", |s| s.as_str())).unwrap();
    let stats_path = std::env::var("BOT_SEARCH_STATS").ok();
    let opts = Options {
        skip: std::env::var("BOT_SKIP").map_or(0, |s| s.parse().unwrap()),
        stats: stats_path.is_some(),
        // ($BOT_TIME_EVAL: the evaluation's time per call, 1000 calls at each searched position)
        time_eval: std::env::var_os("BOT_TIME_EVAL").is_some(),
    };
    ygo_arbiter::game::init(None).unwrap();
    let report = bot::bench::run(args[1].as_ref(), n, budget, &opts).expect("bench");
    let searches = &report.searches;
    if let Ok(path) = std::env::var("BOT_BEST") {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("best file"));
        for s in searches {
            // (then the prompt, the extension's nodes in this search, and whether it extended the
            // last iteration's main line)
            writeln!(f, "{} {} {} {} {} {}", s.opening, s.index, s.best, s.prompt, s.extension.nodes, s.extended() as u8).unwrap();
        }
    }
    if let Some(path) = stats_path {
        print_stats(&report.stats, &path);
    }
    let heights: [(&str, fn(&bot::bench::Search) -> i32); 3] = [("iteration depth", |s| s.depth), ("greatest height", |s| s.seldepth), ("main-line height", |s| s.pv_height)];
    for (name, f) in heights {
        let mut v: Vec<i32> = searches.iter().map(f).collect();
        v.sort();
        let q = |f: f64| v.get(((v.len() as f64 - 1.0) * f).round() as usize).copied().unwrap_or(0);
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64;
        println!("{name:>16}: mean {mean:.2}, min {}, p10 {}, p25 {}, median {}, p75 {}, p90 {}, max {}", q(0.0), q(0.1), q(0.25), q(0.5), q(0.75), q(0.9), q(1.0));
    }
    let nodes = report.nodes();
    let mut extension = Extension::default();
    for s in searches {
        extension += s.extension;
    }
    println!(
        "main-line extension: {} nodes ({:.2}% of all), {} decided, {} of them ({:.1}%) another card than the rules'; it extended the last iteration's main line in {} of {} searches",
        extension.nodes,
        100.0 * extension.nodes as f64 / nodes.max(1) as f64,
        extension.decided,
        extension.not_rules_first,
        100.0 * extension.not_rules_first as f64 / extension.decided.max(1) as f64,
        searches.iter().filter(|s| s.extended()).count(),
        searches.len()
    );
    let tm: [(&str, fn(&bot::bench::Search) -> f64); 2] = [("best move's node share", |s| s.best_fraction), ("time budget scale", |s| s.tm_scale)];
    for (name, f) in tm {
        let mut v: Vec<f64> = searches.iter().map(f).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| v.get(((v.len() as f64 - 1.0) * f).round() as usize).copied().unwrap_or(0.0);
        let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
        println!("{name:>22}: mean {mean:.3}, p10 {:.3}, p25 {:.3}, median {:.3}, p75 {:.3}, p90 {:.3}", q(0.1), q(0.25), q(0.5), q(0.75), q(0.9));
    }
    if opts.time_eval {
        println!("evaluation: {:.2} us per call ({} calls)", report.eval.0 / report.eval.1 as f64 * 1e6, report.eval.1);
    }
    let (probes, hits, broken) = searches.iter().fold((0, 0, 0), |(p, h, b), s| (p + s.tt_probes, h + s.tt_hits, b + s.broken));
    println!(
        "{} searches, {nodes} nodes in {:.1} s: {:.0} nodes/s, mean depth {:.2}; TT hits {hits} of {probes} probes; {broken} broken lines; checksum {:016x}",
        searches.len(),
        report.secs(),
        nodes as f64 / report.secs(),
        searches.iter().map(|s| s.depth as f64).sum::<f64>() / searches.len().max(1) as f64,
        report.checksum()
    );
}

/// The tables, and the counts as JSON lines to `path`.
fn print_stats(st: &NodeStats, path: &str) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("stats file"));
    for (&(label, listed, children), w) in &st.widths {
        writeln!(f, r#"{{"t": "width", "prompt": "{label}", "listed": {listed}, "children": {children}, "nodes": {}, "visited": {}}}"#, w.nodes, w.searched).unwrap();
    }
    for (&(label, tt, children, cut), &n) in &st.cuts {
        let cut = cut.map_or("null".to_string(), |i| i.to_string());
        writeln!(f, r#"{{"t": "cut", "prompt": "{label}", "tt": "{}", "children": {children}, "cut": {cut}, "nodes": {n}}}"#, tt.label()).unwrap();
    }
    for (&(label, i), r) in &st.lmr {
        writeln!(f, r#"{{"t": "lmr", "prompt": "{label}", "move": {i}, "reduced": {}, "researched": {}}}"#, r.reduced, r.researched).unwrap();
    }
    for (&(label, depth), t) in &st.subtree {
        writeln!(f, r#"{{"t": "subtree", "prompt": "{label}", "depth": {depth}, "nodes": {}, "subtree": {}}}"#, t.nodes, t.subtree_nodes).unwrap();
    }
    print_widths(&st.widths);
    // cutoffs by prompt and TT entry, nodes with 2+ children
    println!("\n{:<10} {:<9} {:>9} {:>6} {:>8} {:>8} {:>8} {:>8}", "prompt", "TT", "nodes", "cut", "at 1st", "at 2nd", "at 3rd+", "children");
    #[derive(Default)]
    struct Cutoffs {
        nodes: u64,
        at: [u64; 3],
        children: u64,
    }
    let mut by: BTreeMap<(&str, TtEntry), Cutoffs> = BTreeMap::new();
    for (&(label, tt, children, cut), &n) in &st.cuts {
        let e = by.entry((label, tt)).or_default();
        e.nodes += n;
        e.children += n * children as u64;
        if let Some(i) = cut {
            e.at[(i as usize).min(2)] += n;
        }
    }
    for (&(label, tt), e) in &by {
        let pct = |x: u64| x as f64 / e.nodes as f64 * 100.0;
        println!(
            "{label:<10} {:<9} {:>9} {:>5.0}% {:>7.0}% {:>7.0}% {:>7.0}% {:>8.2}",
            tt.label(),
            e.nodes,
            pct(e.at.iter().sum()),
            pct(e.at[0]),
            pct(e.at[1]),
            pct(e.at[2]),
            e.children as f64 / e.nodes as f64
        );
    }
    println!("\nLMR: re-searched share of the reduced searches, by move index (all prompts)");
    let mut by_move: BTreeMap<u8, Reductions> = BTreeMap::new();
    for (&(_, i), &r) in &st.lmr {
        *by_move.entry(i).or_default() += r;
    }
    for (i, r) in by_move {
        println!("  move {:>2}: {:>9} reduced, {:>5.1}% re-searched", i + 1, r.reduced, r.researched as f64 / r.reduced as f64 * 100.0);
    }
    println!("\nmean subtree size (nodes) by depth (all prompts)");
    let mut by_depth: BTreeMap<i8, Subtree> = BTreeMap::new();
    for (&(_, d), &t) in &st.subtree {
        *by_depth.entry(d).or_default() += t;
    }
    for (d, t) in by_depth {
        println!("  depth {d:>2}: {:>9} nodes, mean subtree {:>9.1}", t.nodes, t.subtree_nodes as f64 / t.nodes as f64);
    }
}

/// The width table by prompt.
fn print_widths(widths: &BTreeMap<(&'static str, u16, u16), Width>) {
    struct Row {
        listed: u16,
        children: u16,
        width: Width,
    }
    let mut by: BTreeMap<&str, Vec<Row>> = BTreeMap::new();
    for (&(label, listed, children), &width) in widths {
        by.entry(label).or_default().push(Row { listed, children, width });
    }
    let total: u64 = widths.values().map(|w| w.nodes).sum();
    println!("{:<10} {:>9} {:>6} {:>7} {:>8} {:>8}  children: share of the nodes with 1, 2, 3, 4-6, 7-11, 12", "prompt", "nodes", "share", "listed", "children", "searched");
    for (label, rows) in &by {
        let n: u64 = rows.iter().map(|r| r.width.nodes).sum();
        let mean = |f: &dyn Fn(&Row) -> f64| rows.iter().map(|r| f(r) * r.width.nodes as f64).sum::<f64>() / n as f64;
        let share = |lo: u16, hi: u16| rows.iter().filter(|r| (lo..=hi).contains(&r.children)).map(|r| r.width.nodes).sum::<u64>() as f64 / n as f64 * 100.0;
        println!(
            "{label:<10} {n:>9} {:>5.1}% {:>7.2} {:>8.2} {:>8.2}  {:>4.0}% {:>4.0}% {:>4.0}% {:>4.0}% {:>4.0}% {:>4.0}%",
            n as f64 / total as f64 * 100.0,
            mean(&|r| r.listed as f64),
            mean(&|r| r.children as f64),
            rows.iter().map(|r| r.width.searched).sum::<u64>() as f64 / n as f64,
            share(1, 1), share(2, 2), share(3, 3), share(4, 6), share(7, 11), share(12, u16::MAX),
        );
    }
}
