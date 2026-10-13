//! Speed of the alpha-beta search (crate::search): plays book openings with the heuristic player
//! and, at each searched decision of player 0, searches the true duel for `budget` seconds, or
//! `N` nodes with a budget of the form "Nn" (then the run is deterministic: its checksum over the
//! searches' results and node counts shows that a change kept the engine's behavior).
//!
//!   ygoai-bench-search BOOK [OPENINGS] [BUDGET]
//!
//! With $YGOAI_WIDTHS set, it also prints the widths of the expanded nodes by prompt (options
//! listed, children after the caps, children searched before a cutoff), and writes every
//! (prompt, listed, children) count to the file $YGOAI_WIDTHS as JSON lines.

use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use ygoai::game::{DuelConfig, init, new_record, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::matchrun::load_book_head;
use ygoai::observe::{auto_response, is_hidden_prompt};
use ygoai::search::{AlphaBeta, SearchParams, candidates, is_searched};

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
    let widths_path = std::env::var("YGOAI_WIDTHS").ok();
    let mut widths: BTreeMap<(&'static str, u16, u16), [u64; 2]> = BTreeMap::new();
    for opening in book.iter().take(n) {
        let record = new_record(&opening.decks, opening.seed, &DuelConfig::default());
        let mut duel = record.setup(None).unwrap();
        duel.ocg_start_duel();
        let mut players = [Heuristic::new(1, 0), Heuristic::new(2, 1)];
        let mut ab = AlphaBeta::default();
        if widths_path.is_some() {
            ab.widths = Some(Default::default());
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
                ab.prepare();
                let before = ab.nodes;
                let (bp, bh, bb) = (ab.tt_probes, ab.tt_hits, ab.broken);
                let t = Instant::now();
                let params = SearchParams { budget, elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes };
                let (best, done) = ab.search(&duel, &p, &cands, &params);
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
        for (k, v) in ab.widths.take().unwrap_or_default() {
            let e = widths.entry(k).or_default();
            e[0] += v[0];
            e[1] += v[1];
        }
    }
    if let Some(path) = widths_path {
        print_widths(&widths, &path);
    }
    println!(
        "{searches} searches, {nodes} nodes in {secs:.1} s: {:.0} nodes/s, mean depth {:.2}; TT hits {} of {} probes; {broken} broken lines; checksum {checksum:016x}",
        nodes as f64 / secs,
        depth_sum as f64 / searches.max(1) as f64,
        tt.1,
        tt.0
    );
}

/// The width table by prompt, and the counts as JSON lines to `path`.
fn print_widths(widths: &BTreeMap<(&'static str, u16, u16), [u64; 2]>, path: &str) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("widths file"));
    let mut by: BTreeMap<&str, Vec<(u16, u16, u64, u64)>> = BTreeMap::new();
    for (&(label, listed, children), &[n, visited]) in widths {
        writeln!(f, r#"{{"prompt": "{label}", "listed": {listed}, "children": {children}, "nodes": {n}, "visited": {visited}}}"#).unwrap();
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
