//! Speed of the alpha-beta search (crate::search): plays book openings with the heuristic player
//! and, at each searched decision of player 0, searches the true duel for `budget` seconds.
//!
//!   ygoai-bench-search BOOK [OPENINGS] [BUDGET]

use std::time::Instant;

use ygoai::game::{DuelConfig, init, new_record, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::matchrun::load_book;
use ygoai::observe::{auto_response, is_hidden_prompt};
use ygoai::search::{AlphaBeta, SearchParams, candidates, is_searched};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let book = load_book(args[1].as_ref(), 0).expect("book");
    let n: usize = args.get(2).map_or(2, |s| s.parse().unwrap());
    let budget: f64 = args.get(3).map_or(0.5, |s| s.parse().unwrap());
    init(None).unwrap();
    let (mut nodes, mut searches, mut depth_sum, mut secs) = (0u64, 0u32, 0i64, 0.0);
    for opening in book.iter().take(n) {
        let record = new_record(&opening.decks, opening.seed, &DuelConfig::default());
        let mut duel = record.setup(None).unwrap();
        duel.ocg_start_duel();
        let mut players = [Heuristic::new(1, 0), Heuristic::new(2, 1)];
        let mut ab = AlphaBeta::default();
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
                let t = Instant::now();
                let params = SearchParams { budget: Some(budget), elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes: None };
                let (_, done) = ab.search(&duel, &p, &cands, &params);
                secs += t.elapsed().as_secs_f64();
                nodes += ab.nodes - before;
                searches += 1;
                depth_sum += done as i64;
            }
            let resp = if is_hidden_prompt(&p) { auto_response() } else { players[player as usize].respond_to(&p, None).unwrap() };
            duel.ocg_set_response(&ygoai::actions::encode(&p, &resp).unwrap());
            prompt = Some(p);
        }
    }
    println!(
        "{searches} searches, {nodes} nodes in {secs:.1} s: {:.0} nodes/s, mean depth {:.2}",
        nodes as f64 / secs,
        depth_sum as f64 / searches.max(1) as f64
    );
}
