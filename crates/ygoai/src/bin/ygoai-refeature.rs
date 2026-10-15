//! Features for datagen rows by replaying the games' recorded responses (no search): a line per
//! row of OUT.jsonl.games with the game, the decision and the features there.
//!
//!   ygoai-refeature BOOK DATA.jsonl.games OUT.jsonl [JOBS]
//!
//! (BOOK the datagen's book: a game's number is its line in the book)

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use ygoai::game::{DuelConfig, init, new_record, run_until_prompt};
use ygoai::matchrun::load_book_head;
use ygoai::messages::Msg;
use ygoai::search::mover_features2;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn replay(book: &ygoai::matchrun::Opening, log: &Value) -> Vec<Value> {
    let game = log["game"].as_u64().unwrap();
    let wanted: HashSet<u64> = log["rows"].as_array().unwrap().iter().filter_map(|d| d.as_u64()).collect();
    let responses: Vec<Vec<u8>> = log["responses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| {
            let h = h.as_str().unwrap();
            (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap()).collect()
        })
        .collect();
    let record = new_record(&book.decks, book.seed, &DuelConfig::default());
    let mut duel = record.setup(None).unwrap();
    duel.ocg_start_duel();
    let (mut out, mut prompt): (Vec<Value>, Option<Msg>) = (Vec::new(), None);
    for (i, r) in responses.iter().enumerate() {
        let Ok(step) = run_until_prompt(&mut duel, prompt.as_ref()) else { break };
        let Some(p) = step.prompt else { break };
        if wanted.contains(&(i as u64)) {
            let player = p.player().unwrap();
            out.push(json!({"game": game, "decision": i, "player": player, "mover_x": mover_features2(&mut duel, player, Some(&p))}));
        }
        duel.ocg_set_response(r);
        prompt = Some(p);
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: ygoai-refeature BOOK DATA.jsonl.games OUT.jsonl [JOBS]");
        std::process::exit(2);
    }
    let jobs: usize = a.get(4).map_or(8, |s| s.parse().unwrap());
    let logs: Vec<Value> = std::io::BufReader::new(std::fs::File::open(&a[2]).expect("games"))
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    let last = logs.iter().map(|l| l["game"].as_u64().unwrap() as usize).max().unwrap_or(0);
    let book = Arc::new(load_book_head(a[1].as_ref(), 0, last + 1).expect("book"));
    init(None).unwrap();
    let logs = Arc::new(logs);
    let out = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&a[3]).expect("output"))));
    let (next, done) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let t0 = std::time::Instant::now();
    let threads: Vec<_> = (0..jobs)
        .map(|_| {
            let (book, logs, out, next, done) = (book.clone(), logs.clone(), out.clone(), next.clone(), done.clone());
            std::thread::spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(log) = logs.get(i) else { break };
                    let g = log["game"].as_u64().unwrap() as usize;
                    let rows = std::panic::catch_unwind(|| replay(&book[g], log)).unwrap_or_default();
                    let mut w = out.lock().unwrap();
                    for r in rows {
                        writeln!(w, "{r}").unwrap();
                    }
                    w.flush().unwrap();
                    let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if k % 500 == 0 {
                        eprintln!("[{:.0}s] {k}/{} games", t0.elapsed().as_secs_f64(), logs.len());
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
