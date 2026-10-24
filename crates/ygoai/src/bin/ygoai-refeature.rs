//! Training rows from recorded games, by replaying them (no search): ygoai-datagen's
//! OUT.games.jsonl, or ygoai-match's records (--records; the engines' searches are their labels,
//! the game number its line), in; a row per labeled decision with the feature sets FEATURES out
//! (`ygoai::datarow`).
//!
//!   ygoai-refeature GAMES.jsonl OUT.jsonl --features SET,SET... [JOBS]

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ygoai::datarow::{FEATURE_SETS, GameData, replay_rows};
use ygoai::game::{GameRecord, init};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let usage = || -> ! {
        eprintln!("usage: ygoai-refeature GAMES.jsonl OUT.jsonl --features SET,SET... [JOBS]  (sets: {})", FEATURE_SETS.join(", "));
        std::process::exit(2)
    };
    if a.len() < 5 || a[3] != "--features" {
        usage();
    }
    let sets: Arc<Vec<String>> = Arc::new(a[4].split(',').map(String::from).collect());
    if let Some(s) = sets.iter().find(|s| !FEATURE_SETS.contains(&s.as_str())) {
        eprintln!("unknown feature set {s}");
        usage();
    }
    let jobs: usize = a.get(5).map_or(8, |s| s.parse().unwrap_or_else(|_| usage()));
    init(None).unwrap();
    // (the games are read as the workers go, not all at once: a large file would not fit in memory)
    let games = Arc::new(Mutex::new(std::io::BufReader::new(std::fs::File::open(&a[1]).expect("games")).lines().enumerate()));
    let out = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&a[2]).expect("output"))));
    let (done, failed) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let t0 = std::time::Instant::now();
    let threads: Vec<_> = (0..jobs)
        .map(|_| {
            let (games, out, done, failed, sets) = (games.clone(), out.clone(), done.clone(), failed.clone(), sets.clone());
            std::thread::spawn(move || {
                loop {
                    let Some((i, l)) = games.lock().unwrap().next() else { break };
                    let l = l.unwrap();
                    let g = serde_json::from_str::<GameData>(&l).unwrap_or_else(|_| {
                        let record: GameRecord = serde_json::from_str(&l).expect("a GameData or GameRecord line");
                        GameData { game: i, labels: record.labels.clone(), record, viewer_responses: true }
                    });
                    let g = &g;
                    match std::panic::catch_unwind(|| replay_rows(g, &sets)) {
                        Ok(Ok(rows)) => {
                            let mut w = out.lock().unwrap();
                            for r in rows {
                                writeln!(w, "{r}").unwrap();
                            }
                        }
                        Ok(Err(e)) => {
                            eprintln!("game {}: {e}", g.game);
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            eprintln!("game {}: panicked", g.game);
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if k % 1000 == 0 {
                        eprintln!("[{:.0}s] {k} games", t0.elapsed().as_secs_f64());
                    }
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    out.lock().unwrap().flush().unwrap();
    eprintln!("[{:.0}s] done, {} games failed", t0.elapsed().as_secs_f64(), failed.load(Ordering::Relaxed));
}
