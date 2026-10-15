//! Training data for the evaluation: self-play of the alpha-beta search on the true duel (both
//! players, a fixed node budget per decision), one game per book opening. The data are the games
//! (`ygoai::datarow::GameData`: the record and, at each searched decision, the search's root
//! value and depth for the mover) in OUT.games.jsonl; OUT.jsonl gets their rows with the feature
//! sets FEATURES (`ygoai::datarow`; default mover2,static). Other features later come from the
//! games without search: ygoai-refeature.
//!
//!   ygoai-datagen BOOK FIRST COUNT NODES OUT.jsonl [JOBS] [--features SET,SET...]
//!
//! (openings FIRST..FIRST+COUNT of the book, every line; games in parallel on JOBS threads)

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use ygoai::datarow::{GameData, Label, row, set_results};
use ygoai::game::{DuelConfig, RecordedResponse, init, new_record, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::matchrun::load_book_head;
use ygoai::messages::Msg;
use ygoai::observe::{auto_response, is_hidden_prompt};
use ygoai::search::{AlphaBeta, LearnedEval, SearchParams, candidates, is_searched};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MAX_DECISIONS: usize = 3000;

fn play(book: &ygoai::matchrun::Opening, game: usize, nodes: u64, sets: &[String]) -> (Vec<Value>, GameData) {
    let mut record = new_record(&book.decks, book.seed, &DuelConfig::default());
    let eval = if LearnedEval::from_env().is_some() { std::env::var("YGOAI_EVAL").unwrap_or_else(|_| "built-in".into()) } else { "hand".into() };
    record.agents = vec![format!("alphabeta {nodes} nodes, eval {eval}"); 2];
    let mut duel = record.setup(None).unwrap();
    duel.ocg_start_duel();
    let mut players = [Heuristic::new(1, 0), Heuristic::new(2, 1)];
    let mut abs = [AlphaBeta::default(), AlphaBeta::default()];
    let (mut rows, mut labels, mut prompt, mut winner): (Vec<Value>, Vec<Label>, Option<Msg>, Option<u32>) = (Vec::new(), Vec::new(), None, None);
    for _ in 0..MAX_DECISIONS {
        let Ok(step) = run_until_prompt(&mut duel, prompt.as_ref()) else { break };
        if let Some(w) = step.messages.iter().find_map(|m| if let Msg::Win { player, .. } = m { Some(*player) } else { None }) {
            winner = Some(w);
            break;
        }
        if step.ended {
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
        let resp = if is_hidden_prompt(&p) {
            auto_response()
        } else if is_searched(&p) && cands.len() > 1 {
            let ab = &mut abs[player as usize];
            ab.prepare();
            let params = SearchParams { budget: None, elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes: Some(nodes) };
            let (best, done) = ab.search(&duel, &p, &cands, &params);
            if ab.last_value.is_finite() {
                let label = Label { decision: record.responses.len(), player, value: ab.last_value, depth: done as u32 };
                rows.push(row(&mut duel, game, &p, &label, sets).unwrap());
                labels.push(label);
            }
            cands[best].clone()
        } else {
            players[player as usize].respond_to(&p, None).unwrap()
        };
        let Ok(r) = ygoai::actions::encode(&p, &resp) else { break };
        duel.ocg_set_response(&r);
        record.responses.push(RecordedResponse { player, response: resp });
        prompt = Some(p);
    }
    record.result.insert("winner".into(), json!(winner));
    set_results(&mut rows, &record);
    (rows, GameData { game, record, labels })
}

fn main() {
    ygoai::search::panic_messages_once();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: ygoai-datagen BOOK FIRST COUNT NODES OUT.jsonl [JOBS] [--features SET,SET...]");
        std::process::exit(2);
    }
    let (first, count, nodes): (usize, usize, u64) = (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    let jobs: usize = a.get(6).filter(|s| !s.starts_with("--")).map_or(8, |s| s.parse().unwrap());
    let sets: Vec<String> = a.iter().position(|x| x == "--features").map_or("mover2,static", |i| a[i + 1].as_str()).split(',').map(String::from).collect();
    let sets = Arc::new(sets);
    let book = Arc::new(load_book_head(a[1].as_ref(), 0, first + count).expect("book").split_off(first));
    init(None).unwrap();
    let create = |path: String| Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(path).expect("output"))));
    let (out, games_out) = (create(a[5].clone()), create(format!("{}.games.jsonl", a[5].trim_end_matches(".jsonl"))));
    let (next, done) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let t0 = std::time::Instant::now();
    let threads: Vec<_> = (0..jobs)
        .map(|_| {
            let (book, out, games_out, next, done, sets) = (book.clone(), out.clone(), games_out.clone(), next.clone(), done.clone(), sets.clone());
            std::thread::spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= book.len() {
                        break;
                    }
                    let Ok((rows, data)) = std::panic::catch_unwind(|| play(&book[i], first + i, nodes, &sets)) else { continue };
                    let mut w = out.lock().unwrap();
                    for r in rows {
                        writeln!(w, "{r}").unwrap();
                    }
                    w.flush().unwrap();
                    drop(w);
                    let mut g = games_out.lock().unwrap();
                    writeln!(g, "{}", serde_json::to_string(&data).unwrap()).unwrap();
                    g.flush().unwrap();
                    let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if k % 50 == 0 {
                        eprintln!("[{:.0}s] {k}/{} games", t0.elapsed().as_secs_f64(), book.len());
                    }
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    eprintln!("[{:.0}s] done", t0.elapsed().as_secs_f64());
}
