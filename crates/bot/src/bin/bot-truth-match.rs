//! A match between two search bots in one process, for bots that search the true duel
//! (bot::belief_bots::Truth): how much strength the hidden cards cost. Plays each opening twice
//! with swapped seats, each decision's search with NODES nodes, and prints NEW's score as
//! ygo-arbiter does. The records go to OUT.jsonl.
//!
//!   bot-truth-match BOOK FIRST COUNT NODES OUT.jsonl [JOBS] [NEW] [BASE]
//!
//! NEW, BASE: names of bot::belief_bots::BeliefSearchBot (alphabeta_true, alphabeta). A bot plays
//! opening n's game in seat s with seed 2n + s.

use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use bot::belief_bots::BeliefSearchBot;
use ygo_arbiter::game::{Bot, DuelConfig, GameRecord, Observer, PlayOptions, init, new_record, play};
use ygo_arbiter::matchrun::{MatchStats, Opening, load_book};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The pairs between two printed results.
const REPORT_EVERY: u32 = 10;

fn usage() -> ! {
    eprintln!("usage: bot-truth-match BOOK FIRST COUNT NODES OUT.jsonl [JOBS] [NEW] [BASE]");
    std::process::exit(2)
}

/// The game of `opening` with `names` by seat; the bots' seeds are `seed` + the seat.
fn game(opening: &Opening, names: [&str; 2], nodes: u64, seed: i64) -> Result<GameRecord, String> {
    let mut bots = Vec::new();
    for (seat, name) in names.into_iter().enumerate() {
        let mut bot = BeliefSearchBot::new(name, Some(seed + seat as i64))?;
        bot.set_node_limit(Some(nodes));
        bots.push(bot);
    }
    let mut feeds: Vec<_> = bots.iter().filter_map(BeliefSearchBot::truth_feed).collect();
    let mut observers: Vec<&mut dyn Observer> = feeds.iter_mut().map(|f| f as &mut dyn Observer).collect();
    let mut record = new_record(&opening.decks, opening.seed, &DuelConfig::default());
    record.bots = names.map(String::from).to_vec();
    let [b0, b1] = &mut bots[..] else { unreachable!("two bots") };
    let mut opts = PlayOptions { illegal_is_loss: true, ..PlayOptions::default() };
    play(&mut record, &mut [b0, b1], &mut observers, &mut opts).map_err(|e| e.0)?;
    Ok(record)
}

fn main() {
    bot::search::panic_messages_once();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 {
        usage();
    }
    let number = |i: usize| -> usize { args[i].parse().unwrap_or_else(|_| usage()) };
    let (first, count, nodes) = (number(1), number(2), number(3) as u64);
    let jobs = if args.len() > 5 { number(5) } else { 8 };
    let new = args.get(6).map_or("alphabeta_true", String::as_str);
    let base = args.get(7).map_or("alphabeta", String::as_str);
    init(None).unwrap();
    let book = load_book(args[0].as_ref(), usize::MAX, first + count).unwrap_or_else(|e| panic!("{e}"));
    if book.len() < first + count {
        panic!("{}: {} openings, not {}", args[0], book.len(), first + count);
    }
    let out = Mutex::new(std::io::BufWriter::new(std::fs::File::create(&args[4]).expect("the records file")));
    let stats = Mutex::new(MatchStats::default());
    let next = AtomicUsize::new(first);
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| {
                loop {
                    let n = next.fetch_add(1, Ordering::Relaxed);
                    if n >= first + count {
                        break;
                    }
                    let mut scores = Vec::new();
                    for new_seat in 0..2usize {
                        let names = if new_seat == 0 { [new, base] } else { [base, new] };
                        let record = match game(&book[n], names, nodes, 2 * n as i64) {
                            Ok(r) => r,
                            Err(e) => {
                                eprintln!("opening {n} seat {new_seat}: {e}");
                                break;
                            }
                        };
                        scores.push(match record.result.get("winner").and_then(|w| w.as_u64()) {
                            None => 0.5,
                            Some(w) if w as usize == new_seat => 1.0,
                            Some(_) => 0.0,
                        });
                        let mut out = out.lock().unwrap();
                        writeln!(out, "{}", serde_json::to_string(&record).expect("a record")).expect("the records file");
                        out.flush().expect("the records file");
                    }
                    if scores.len() == 2 {
                        let mut stats = stats.lock().unwrap();
                        stats.add_pair(&scores);
                        if stats.games() % (2 * REPORT_EVERY) == 0 {
                            println!("{}", stats.summary(None));
                        }
                    }
                }
            });
        }
    });
    println!("{new} vs {base}, {nodes} nodes: {}", stats.lock().unwrap().summary(None));
}
