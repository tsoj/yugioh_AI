//! Speed of the alpha-beta search (crate::search): plays book openings with the heuristic player
//! and, at each searched decision of player 0, searches the true duel for `budget` seconds, or
//! `N` nodes with a budget of the form "Nn" (then the run is deterministic: its checksum over the
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
use std::time::Instant;

use ygoai::game::{DuelConfig, init, new_record, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::matchrun::load_book_head;
use ygoai::observe::{auto_response, is_hidden_prompt};
use ygoai::search::{AlphaBeta, NO_CUT, NodeStats, SearchParams, candidates, is_searched};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(2).map_or(2, |s| s.parse().unwrap());
    let book = load_book_head(args[1].as_ref(), 0, n).expect("book");
    let budget = args.get(3).map_or("0.5", |s| s.as_str());
    let (budget, max_nodes) = match budget.strip_suffix('n') {
        Some(k) => (None, Some(k.parse::<u64>().unwrap())),
        None => (Some(budget.parse::<f64>().unwrap()), None),
    };
    let mut checksum = 0u64;
    init(None).unwrap();
    let (mut nodes, mut searches, mut depth_sum, mut secs) = (0u64, 0u32, 0i64, 0.0);
    let mut tt = (0u64, 0u64);
    let mut broken = 0u64;
    let mut online = [0.0f64; 3];
    let mut null = (0u64, 0u64, 0u64);
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
                let t = Instant::now();
                let params = SearchParams { budget, elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes };
                let (best, done) = ab.search(&duel, &p, &cands, &params);
                if let Some(f) = &mut best_out {
                    writeln!(f, "{oi} {si} {best}").unwrap();
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
            }
            let resp = if is_hidden_prompt(&p) { auto_response() } else { players[player as usize].respond_to(&p, None).unwrap() };
            duel.ocg_set_response(&ygoai::actions::encode(&p, &resp).unwrap());
            prompt = Some(p);
        }
        null = (null.0 + ab.null_tries, null.1 + ab.null_cuts, null.2 + ab.null_wrong);
        for (a, b) in online.iter_mut().zip(ab.online_stats) {
            *a += b;
        }
        if let Some(st) = ab.stats.take() {
            merge(&mut stats.widths, st.widths);
            merge(&mut stats.lmr, st.lmr);
            merge(&mut stats.subtree, st.subtree);
            stats.gaps.extend(st.gaps);
            for (k, n) in st.cuts {
                *stats.cuts.entry(k).or_default() += n;
            }
        }
    }
    if let Some(path) = stats_path {
        print_stats(&stats, &path);
    }
    if null.0 > 0 {
        println!("nullmove: {} null move searches, {} cutoffs ({} below beta in the full search)", null.0, null.1, null.2);
    }
    if online[0] > 0.0 {
        println!(
            "online: {} nodes learned from; RMS error before each step {:.3} points (the trained evaluation's {:.3}, ratio {:.4})",
            online[0],
            (online[1] / online[0]).sqrt(),
            (online[2] / online[0]).sqrt(),
            (online[1] / online[2]).sqrt()
        );
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

fn merge<K: Ord>(into: &mut BTreeMap<K, [u64; 2]>, from: BTreeMap<K, [u64; 2]>) {
    for (k, v) in from {
        let e = into.entry(k).or_default();
        e[0] += v[0];
        e[1] += v[1];
    }
}

/// The tables, and the counts as JSON lines to `path`.
fn print_stats(st: &NodeStats, path: &str) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("stats file"));
    for (&(label, listed, children), &[n, visited]) in &st.widths {
        writeln!(f, r#"{{"t": "width", "prompt": "{label}", "listed": {listed}, "children": {children}, "nodes": {n}, "visited": {visited}}}"#).unwrap();
    }
    for (&(label, tt, children, cut), &n) in &st.cuts {
        let cut = if cut == NO_CUT { "null".to_string() } else { cut.to_string() };
        writeln!(f, r#"{{"t": "cut", "prompt": "{label}", "tt": {tt}, "children": {children}, "cut": {cut}, "nodes": {n}}}"#).unwrap();
    }
    for (&(label, i), &[n, re]) in &st.lmr {
        writeln!(f, r#"{{"t": "lmr", "prompt": "{label}", "move": {i}, "reduced": {n}, "researched": {re}}}"#).unwrap();
    }
    for (&(label, depth), &[n, sub]) in &st.subtree {
        writeln!(f, r#"{{"t": "subtree", "prompt": "{label}", "depth": {depth}, "nodes": {n}, "subtree": {sub}}}"#).unwrap();
    }
    for (kind, depth, gap, raised) in &st.gaps {
        writeln!(f, r#"{{"t": "gap", "kind": "{kind}", "depth": {depth}, "gap": {gap}, "raised": {raised}}}"#).unwrap();
    }
    print_widths(&st.widths);
    // cutoffs by prompt and TT state, nodes with 2+ children
    const TT: [&str; 5] = ["no entry", "no move", "exact", "lower", "upper"];
    println!("\n{:<10} {:<9} {:>9} {:>6} {:>8} {:>8} {:>8} {:>8}", "prompt", "TT", "nodes", "cut", "at 1st", "at 2nd", "at 3rd+", "children");
    let mut by: BTreeMap<(&str, u8), [u64; 5]> = BTreeMap::new();
    for (&(label, tt, children, cut), &n) in &st.cuts {
        let e = by.entry((label, tt)).or_default();
        e[0] += n;
        e[4] += n * children as u64;
        match cut {
            NO_CUT => {}
            0 => e[1] += n,
            1 => e[2] += n,
            _ => e[3] += n,
        }
    }
    for (&(label, tt), e) in &by {
        let pct = |x: u64| x as f64 / e[0] as f64 * 100.0;
        println!(
            "{label:<10} {:<9} {:>9} {:>5.0}% {:>7.0}% {:>7.0}% {:>7.0}% {:>8.2}",
            TT[tt as usize], e[0], pct(e[1] + e[2] + e[3]), pct(e[1]), pct(e[2]), pct(e[3]), e[4] as f64 / e[0] as f64
        );
    }
    println!("\nLMR: re-searched share of the reduced searches, by move index (all prompts)");
    let mut by_i: BTreeMap<u8, [u64; 2]> = BTreeMap::new();
    for (&(_, i), v) in &st.lmr {
        let e = by_i.entry(i).or_default();
        e[0] += v[0];
        e[1] += v[1];
    }
    for (i, [n, re]) in by_i {
        println!("  move {:>2}: {:>9} reduced, {:>5.1}% re-searched", i + 1, n, re as f64 / n as f64 * 100.0);
    }
    println!("\nmean subtree size (nodes) by depth (all prompts)");
    let mut by_d: BTreeMap<i8, [u64; 2]> = BTreeMap::new();
    for (&(_, d), v) in &st.subtree {
        let e = by_d.entry(d).or_default();
        e[0] += v[0];
        e[1] += v[1];
    }
    for (d, [n, sub]) in by_d {
        println!("  depth {d:>2}: {n:>9} nodes, mean subtree {:>9.1}", sub as f64 / n as f64);
    }
}

/// The width table by prompt.
fn print_widths(widths: &BTreeMap<(&'static str, u16, u16), [u64; 2]>) {
    let mut by: BTreeMap<&str, Vec<(u16, u16, u64, u64)>> = BTreeMap::new();
    for (&(label, listed, children), &[n, visited]) in widths {
        by.entry(label).or_default().push((listed, children, n, visited));
    }
    let total: u64 = widths.values().map(|v| v[0]).sum();
    println!("{:<10} {:>9} {:>6} {:>7} {:>8} {:>8}  children: share of the nodes with 1, 2, 3, 4-6, 7-11, 12", "prompt", "nodes", "share", "listed", "children", "searched");
    for (label, rows) in &by {
        let n: u64 = rows.iter().map(|r| r.2).sum();
        let mean = |f: &dyn Fn(&(u16, u16, u64, u64)) -> f64| rows.iter().map(|r| f(r) * r.2 as f64).sum::<f64>() / n as f64;
        let share = |lo: u16, hi: u16| rows.iter().filter(|r| (lo..=hi).contains(&r.1)).map(|r| r.2).sum::<u64>() as f64 / n as f64 * 100.0;
        println!(
            "{label:<10} {n:>9} {:>5.1}% {:>7.2} {:>8.2} {:>8.2}  {:>4.0}% {:>4.0}% {:>4.0}% {:>4.0}% {:>4.0}% {:>4.0}%",
            n as f64 / total as f64 * 100.0,
            mean(&|r| r.0 as f64),
            mean(&|r| r.1 as f64),
            rows.iter().map(|r| r.3).sum::<u64>() as f64 / n as f64,
            share(1, 1), share(2, 2), share(3, 3), share(4, 6), share(7, 11), share(12, u16::MAX),
        );
    }
}
