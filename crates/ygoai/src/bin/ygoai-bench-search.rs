//! Speed of the alpha-beta search (crate::search): plays book openings with the heuristic player
//! and, at each searched decision of player 0, searches the true duel for `budget` seconds, or
//! `N` nodes with a budget of the form "Nn", or to depth `N` with "Nd" (both deterministic: the
//! nodes a fixed depth takes measure the move ordering; the checksum over the
//! searches' results and node counts shows that a change kept the engine's behavior).
//!
//!   ygoai-bench-search BOOK [OPENINGS] [BUDGET]
//!
//! With $YGOAI_STATS set, it also prints statistics of the expanded nodes (search::NodeStats):
//! widths by prompt (options listed, children after the caps, children searched before a cutoff),
//! cutoffs by prompt and TT state, LMR re-searches by move index, subtree sizes by depth; and
//! writes all the counts to the file $YGOAI_STATS as JSON lines.
//!
//! $YGOAI_BEST: write each search's chosen move to that file (a line "opening search move").
//! $YGOAI_SKIP: skip that many book openings first (to split a run).

use std::collections::BTreeMap;
use std::io::Write;
use std::ops::AddAssign;
use std::time::Instant;

use ygoai::game::{DuelConfig, init, new_record, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::matchrun::load_book_head;
use ygoai::observe::{auto_response, is_hidden_prompt};
use ygoai::search::stats::{Extension, Reductions, Subtree, TtEntry, Width};
use ygoai::search::{AlphaBeta, NodeStats, SearchParams, candidates, is_searched};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(2).map_or(2, |s| s.parse().unwrap());
    let book = load_book_head(args[1].as_ref(), 0, n).expect("book");
    let budget = args.get(3).map_or("0.5", |s| s.as_str());
    let (budget, max_nodes, depth) = match (budget.strip_suffix('n'), budget.strip_suffix('d')) {
        (Some(k), _) => (None, Some(k.parse::<u64>().unwrap()), None),
        (_, Some(k)) => (None, None, Some(k.parse::<i32>().unwrap())),
        _ => (Some(budget.parse::<f64>().unwrap()), None, None),
    };
    let mut checksum = 0u64;
    init(None).unwrap();
    let (mut nodes, mut searches, mut depth_sum, mut secs) = (0u64, 0u32, 0i64, 0.0);
    let mut tt = (0u64, 0u64);
    // (per search: the completed depth, the greatest height, the main line's height)
    let mut heights: [Vec<i32>; 3] = Default::default();
    // (per search: the best move's share of the nodes, the time budget's scale)
    let mut tm: [Vec<f64>; 2] = Default::default();
    // (the main-line extension's nodes, and the searches whose last iteration's main line it extended)
    let (mut extension, mut ext_searches) = (Extension::default(), 0u32);
    let mut broken = 0u64;
    let stats_path = std::env::var("YGOAI_STATS").ok();
    let mut stats = NodeStats::default();
    let skip: usize = std::env::var("YGOAI_SKIP").map_or(0, |s| s.parse().unwrap());
    // ($YGOAI_TIME_EVAL: the evaluation's time per call, 1000 calls at each searched position)
    let time_eval = std::env::var_os("YGOAI_TIME_EVAL").is_some();
    let (mut eval_secs, mut eval_calls) = (0.0, 0u64);
    let mut best_out = std::env::var("YGOAI_BEST").ok().map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("best file")));
    for (oi, opening) in book.iter().enumerate().take(n).skip(skip) {
        let mut si = 0;
        let record = new_record(&opening.decks, opening.seed, &DuelConfig::default());
        let mut duel = record.setup(None).unwrap();
        duel.ocg_start_duel();
        let mut players = [Heuristic::new(1, 0), Heuristic::new(2, 1)];
        let mut ab = AlphaBeta::default();
        if stats_path.is_some() {
            ab.stats = Some(Default::default());
        }
        let mut prompt = None;
        for _ in 0..400 {
            let step = run_until_prompt(&mut duel, prompt.as_ref()).unwrap();
            if step.messages.iter().any(|m| matches!(m, ygoai::messages::Msg::Win { .. })) || step.ended {
                break;
            }
            for m in &step.messages {
                for p in &mut players {
                    p.observe_msg(m);
                }
            }
            let p = step.prompt.unwrap();
            let player = p.player().unwrap();
            let cands = candidates(&p);
            if player == 0 && is_searched(&p) && cands.len() > 1 {
                if time_eval {
                    let t = Instant::now();
                    let mut acc = 0.0;
                    for _ in 0..1000 {
                        acc += ygoai::search::evaluate_at(&mut duel, 0, Some(&p));
                    }
                    eval_secs += t.elapsed().as_secs_f64();
                    eval_calls += 1000;
                    std::hint::black_box(acc);
                }
                ab.prepare();
                let before = ab.nodes;
                let (bp, bh, bb) = (ab.tt_probes, ab.tt_hits, ab.broken);
                let extension_before = ab.extension;
                let t = Instant::now();
                let params = SearchParams { budget, elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes, depth, exact_root: false };
                let (best, done) = ab.search(&duel, &p, &cands, &params);
                if let Some(f) = &mut best_out {
                    // (then the prompt, the extension's nodes in this search, and whether it extended
                    // the last iteration's main line)
                    writeln!(f, "{oi} {si} {best} {} {} {}", p.name(), ab.extension.nodes - extension_before.nodes, (ab.pv_height > done) as u8).unwrap();
                }
                si += 1;
                checksum = checksum.wrapping_mul(0x100000001b3) ^ (best as u64) ^ ((done as u64) << 8) ^ ((ab.nodes - before) << 16);
                secs += t.elapsed().as_secs_f64();
                nodes += ab.nodes - before;
                tt.0 += ab.tt_probes - bp;
                tt.1 += ab.tt_hits - bh;
                broken += ab.broken - bb;
                searches += 1;
                depth_sum += done as i64;
                extension += ab.extension - extension_before;
                ext_searches += (ab.pv_height > done) as u32;
                tm[0].push(ab.best_fraction);
                tm[1].push(ab.tm_scale);
                for (v, h) in heights.iter_mut().zip([done, ab.seldepth, ab.pv_height]) {
                    v.push(h);
                }
            }
            let resp = if is_hidden_prompt(&p) { auto_response() } else { players[player as usize].respond_to(&p, None).unwrap() };
            duel.ocg_set_response(&ygoai::actions::encode(&p, &resp).unwrap());
            prompt = Some(p);
        }
        if let Some(st) = ab.stats.take() {
            merge(&mut stats.widths, st.widths);
            merge(&mut stats.lmr, st.lmr);
            merge(&mut stats.subtree, st.subtree);
            merge(&mut stats.cuts, st.cuts);
        }
    }
    if let Some(path) = stats_path {
        print_stats(&stats, &path);
    }
    for (name, v) in ["iteration depth", "greatest height", "main-line height"].iter().zip(heights.iter_mut()) {
        v.sort();
        let q = |f: f64| v.get(((v.len() as f64 - 1.0) * f).round() as usize).copied().unwrap_or(0);
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / v.len().max(1) as f64;
        println!("{name:>16}: mean {mean:.2}, min {}, p10 {}, p25 {}, median {}, p75 {}, p90 {}, max {}", q(0.0), q(0.1), q(0.25), q(0.5), q(0.75), q(0.9), q(1.0));
    }
    println!(
        "main-line extension: {} nodes ({:.2}% of all), {} decided, {} of them ({:.1}%) another card than the rules'; it extended the last iteration's main line in {ext_searches} of {searches} searches",
        extension.nodes,
        100.0 * extension.nodes as f64 / nodes.max(1) as f64,
        extension.decided,
        extension.not_rules_first,
        100.0 * extension.not_rules_first as f64 / extension.decided.max(1) as f64
    );
    for (name, v) in ["best move's node share", "time budget scale"].iter().zip(tm.iter_mut()) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| v.get(((v.len() as f64 - 1.0) * f).round() as usize).copied().unwrap_or(0.0);
        let mean = v.iter().sum::<f64>() / v.len().max(1) as f64;
        println!("{name:>22}: mean {mean:.3}, p10 {:.3}, p25 {:.3}, median {:.3}, p75 {:.3}, p90 {:.3}", q(0.1), q(0.25), q(0.5), q(0.75), q(0.9));
    }
    if time_eval {
        println!("evaluation: {:.2} us per call ({eval_calls} calls)", eval_secs / eval_calls as f64 * 1e6);
    }
    println!(
        "{searches} searches, {nodes} nodes in {secs:.1} s: {:.0} nodes/s, mean depth {:.2}; TT hits {} of {} probes; {broken} broken lines; checksum {checksum:016x}",
        nodes as f64 / secs,
        depth_sum as f64 / searches.max(1) as f64,
        tt.1,
        tt.0
    );
}

fn merge<K: Ord, V: AddAssign + Default>(into: &mut BTreeMap<K, V>, from: BTreeMap<K, V>) {
    for (k, v) in from {
        *into.entry(k).or_default() += v;
    }
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
