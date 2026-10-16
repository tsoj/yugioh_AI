//! How much more search is worth when the opponent's hidden cards are guessed: plays book
//! openings with the heuristic player and, at player 0's searched decisions, takes a long search
//! with the true cards as the reference (ORACLE nodes); then searches the position with each of
//! NODES nodes, once with the true cards and once per guess of the cards hidden from player 0
//! (the likeliest deck as `naive_hidden_cards` guesses it, then GUESSES draws of
//! `belief::guesses::resample`), and counts how often each picks the reference's move. Each search
//! with fresh tables.
//!
//!   ygoai-guess-scaling BOOK [OPENINGS] [ORACLE] [NODES,...] [GUESSES] [EVERY]
//!
//! (EVERY: every that many searched decisions; $YGOAI_SKIP: openings to skip; $YGOAI_SCALING: a
//! file for the per-position results as JSON lines)

use std::io::Write;

use ygoai::belief::guesses::{Guesses, resample};
use ygoai::belief::tracker::Guess;
use ygoai::belief_agents::{completion, naive_hidden_cards};
use ygoai::game::{DuelConfig, init, new_record, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::matchrun::load_book_head;
use ygoai::observe::{auto_response, is_hidden_prompt};
use ygoai::pyrandom::PyRandom;
use ygoai::rduel::RDuel;
use ygoai::search::{AlphaBeta, SearchParams, candidates, is_hidden, is_searched};
use ygo_core::duel::Duel;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// (best move, completed depth)
fn search(duel: &Duel, prompt: &ygoai::messages::Msg, cands: &[ygoai::actions::Response], nodes: u64) -> (usize, i32) {
    let mut ab = AlphaBeta::default();
    ab.prepare();
    let params = SearchParams { budget: None, elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes: Some(nodes) };
    ab.search(duel, prompt, cands, &params)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(2).map_or(5, |s| s.parse().unwrap());
    let oracle_nodes: u64 = args.get(3).map_or(50000, |s| s.parse().unwrap());
    let budgets: Vec<u64> = args.get(4).map_or("300,1000,3000,10000", |s| s.as_str()).split(',').map(|x| x.parse().unwrap()).collect();
    let n_guesses: usize = args.get(5).map_or(3, |s| s.parse().unwrap());
    let every: usize = args.get(6).map_or(3, |s| s.parse().unwrap());
    let skip: usize = std::env::var("YGOAI_SKIP").map_or(0, |s| s.parse().unwrap());
    let book = load_book_head(args[1].as_ref(), 0, n).expect("book");
    init(None).unwrap();
    let comp = completion("neural").unwrap();
    let mut out = std::env::var("YGOAI_SCALING").ok().map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("output file")));
    let mut rng = PyRandom::new(7);
    // per budget: [truth, likeliest, drawn] x [searches, agreeing, depth sum]; the same for
    // positions with 3+ moves
    let mut all = vec![[[0.0f64; 3]; 3]; budgets.len()];
    let mut wide = vec![[[0.0f64; 3]; 3]; budgets.len()];
    let mut positions = 0;
    for (oi, opening) in book.iter().enumerate().take(n).skip(skip) {
        let record = new_record(&opening.decks, opening.seed, &DuelConfig::default());
        let mut duel = record.setup(None).unwrap();
        duel.ocg_start_duel();
        let mut players = [Heuristic::new(1, 0), Heuristic::new(2, 1)];
        let (mut prompt, mut si) = (None, 0usize);
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
                si += 1;
                if si % every == 0 {
                    positions += 1;
                    let all_cards = duel.ext_get_all_cards();
                    let hidden: Vec<u32> = all_cards.iter().filter(|e| e[0] == 1 && is_hidden(&duel, e[5])).map(|e| e[5]).collect();
                    let is_extra = |c: u32| ygoai::belief::completion::card(c).is_some_and(ygoai::belief::completion::is_extra_deck);
                    let known: Vec<u32> = all_cards.iter().filter(|e| e[0] == 1 && e[5] != 0 && !is_hidden(&duel, e[5])).map(|e| e[4]).collect();
                    let guesses = Guesses {
                        viewer: 0,
                        cards: hidden
                            .iter()
                            .map(|&cid| {
                                let extra = duel.card_by_id(cid).is_some_and(|c| is_extra(duel.card(c).data.code));
                                (cid, Guess { player: 1, extra, code: 0, rank: 0, need: None, excluded: vec![] })
                            })
                            .collect(),
                        known: (known.iter().copied().filter(|&c| !is_extra(c)).collect(), known.iter().copied().filter(|&c| is_extra(c)).collect()),
                    };
                    // the guessed duels: the likeliest deck, then draws
                    let mut likeliest = RDuel::from_duel(duel.clone());
                    naive_hidden_cards(&mut likeliest, 0, comp, &mut rng);
                    let mut guessed = vec![likeliest.d];
                    for _ in 0..n_guesses {
                        let mut d = duel.clone();
                        resample(&mut d, &guesses, comp.unwrap(), &mut rng);
                        guessed.push(d);
                    }
                    let (oracle, oracle_depth) = search(&duel, &p, &cands, oracle_nodes);
                    let mut row = Vec::new();
                    for (bi, &nodes) in budgets.iter().enumerate() {
                        // (kind 0 the truth, 1 the likeliest, 2 the draws)
                        let mut results: Vec<(usize, usize, i32)> = vec![];
                        let (b, dp) = search(&duel, &p, &cands, nodes);
                        results.push((0, b, dp));
                        for (k, g) in guessed.iter().enumerate() {
                            let (b, dp) = search(g, &p, &cands, nodes);
                            results.push(((k > 0) as usize + 1, b, dp));
                        }
                        for &(kind, b, dp) in &results {
                            let tables: &mut [&mut Vec<[[f64; 3]; 3]>] = if cands.len() >= 3 { &mut [&mut all, &mut wide] } else { &mut [&mut all] };
                            for table in tables.iter_mut() {
                                let e = &mut table[bi][kind];
                                e[0] += 1.0;
                                e[1] += (b == oracle) as u64 as f64;
                                e[2] += dp as f64;
                            }
                        }
                        row.push(results.iter().map(|r| (r.1, r.2)).collect::<Vec<_>>());
                    }
                    println!("opening {oi} search {si} {} ({} moves, {} hidden): oracle {oracle} (depth {oracle_depth}); per budget (best, depth): {row:?}", p.name(), cands.len(), hidden.len());
                    if let Some(f) = &mut out {
                        let j = serde_json::json!({"opening": oi, "search": si, "prompt": p.name(), "moves": cands.len(), "hidden": hidden.len(), "oracle": [oracle, oracle_depth], "budgets": budgets, "results": row});
                        writeln!(f, "{j}").unwrap();
                    }
                }
            }
            let resp = if is_hidden_prompt(&p) { auto_response() } else { players[player as usize].respond_to(&p, None).unwrap() };
            duel.ocg_set_response(&ygoai::actions::encode(&p, &resp).unwrap());
            prompt = Some(p);
        }
    }
    println!("\n{positions} positions, the reference: {oracle_nodes} nodes with the true cards");
    for (title, table) in [("all positions", &all), ("positions with 3+ moves", &wide)] {
        println!("{title}: the reference's move picked (mean depth)");
        println!("  {:>8}  {:>16}  {:>16}  {:>16}", "nodes", "true cards", "likeliest deck", "drawn decks");
        for (bi, &nodes) in budgets.iter().enumerate() {
            let cell = |e: [f64; 3]| format!("{:5.1}% ({:5.2})", e[1] / e[0].max(1.0) * 100.0, e[2] / e[0].max(1.0));
            println!("  {nodes:>8}  {:>16}  {:>16}  {:>16}", cell(table[bi][0]), cell(table[bi][1]), cell(table[bi][2]));
        }
    }
}
