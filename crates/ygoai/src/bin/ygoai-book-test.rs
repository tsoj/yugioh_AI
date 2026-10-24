//! The opening book's test games (scripts/make_match_book.py's, with the policy agents in Rust:
//! ygoai::rl_policy). Each candidate (a row with "seed" and "decks", e.g. from
//! scripts/onesided_candidates.py score) is played up to GAMES times by policy agents (each game a
//! random one of POLICIES for both seats, sampling at temperature 1; the decks keep their seats),
//! stopping once each seat won STOP_MIN. Each tested candidate is appended to OUT as it finishes:
//! its row with the games each seat won ("wins") and the games played ("games"). Candidates
//! already in OUT are skipped, so a stopped run goes on where it was.
//!
//!   ygoai-book-test CANDIDATES.jsonl OUT.jsonl --policies A.npz,B.npz [--games 20] [--stop-min 2] [-j 30] [-n LIMIT]
//!                   [--records GAMES.jsonl]
//!
//! (--records: every game's record with its candidate's "seed" and "game" index, for comparing with
//! the Python agents)
//! (the games and their seeds are make_match_book.py's; a candidate whose games panic is left out)

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{Value, json};
use ygoai::game::{Deck, DuelConfig, PlayOptions, init, new_record, play};
use ygoai::rl_policy::{RlPolicyAgent, RlPolicyNet};
use ygoai::pyrandom::PyRandom;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MAX_DECISIONS: usize = 3000;
const GAME_SECONDS: f64 = 120.0;

/// (wins per seat, games played) of one candidate.
fn test(seed: i64, decks: &[Deck], nets: &[Arc<RlPolicyNet>], games: usize, stop_min: u32, records: Option<&Mutex<std::fs::File>>) -> ([u32; 2], usize) {
    let mut wins = [0u32; 2];
    for g in 0..games {
        if wins[0].min(wins[1]) >= stop_min {
            return (wins, g);
        }
        let s = seed as i128 * 1000;
        let net = PyRandom::new(s + g as i128).choice(nets).clone(); // (both seats)
        let mut agents = [0, 1].map(|p| RlPolicyAgent::new(s + 2 * g as i128 + p, net.clone(), 1.0));
        let mut record = new_record(decks, seed, &DuelConfig::default());
        let [a, b] = &mut agents;
        let mut opts = PlayOptions { max_decisions: MAX_DECISIONS, max_turns: 100, time_limit: Some(GAME_SECONDS), ..PlayOptions::default() };
        play(&mut record, &mut [a, b], &mut [], &mut opts).expect("a game");
        if let Some(f) = records {
            let mut r = serde_json::to_value(&record).unwrap();
            r["candidate"] = json!(seed);
            r["game"] = json!(g);
            writeln!(f.lock().unwrap(), "{r}").unwrap();
        }
        if let Some(w) = record.result.get("winner").and_then(Value::as_u64).filter(|&w| w < 2) {
            wins[w as usize] += 1;
        }
    }
    (wins, games)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut pos, mut policies, mut games, mut stop_min, mut jobs, mut limit) = (Vec::new(), None, 20usize, 2u32, 30usize, usize::MAX);
    let mut records = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().expect("a value").clone();
        match a.as_str() {
            "--policies" => policies = Some(val()),
            "--games" => games = val().parse().unwrap(),
            "--stop-min" => stop_min = val().parse().unwrap(),
            "-j" => jobs = val().parse().unwrap(),
            "-n" => limit = val().parse().unwrap(),
            "--records" => records = Some(Mutex::new(std::fs::File::create(val()).expect("records"))),
            _ => pos.push(a.clone()),
        }
    }
    let [cands_path, out_path] = &pos[..] else {
        eprintln!("usage: ygoai-book-test CANDIDATES.jsonl OUT.jsonl --policies A.npz,B.npz [--games 20] [--stop-min 2] [-j 30] [-n LIMIT]");
        std::process::exit(2);
    };
    init(None).unwrap();
    let nets: Vec<Arc<RlPolicyNet>> = policies
        .expect("--policies")
        .split(',')
        .map(|p| Arc::new(RlPolicyNet::load(Path::new(p)).unwrap_or_else(|e| panic!("{e}"))))
        .collect();
    let done_seeds: HashSet<i64> = std::fs::File::open(out_path)
        .map(|f| std::io::BufReader::new(f).lines().map_while(Result::ok).filter_map(|l| serde_json::from_str::<Value>(&l).ok()?["seed"].as_i64()).collect())
        .unwrap_or_default();
    let rows: Vec<Value> = std::io::BufReader::new(std::fs::File::open(cands_path).expect("candidates"))
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .take(limit)
        .map(|l| serde_json::from_str(&l).expect("a candidate row"))
        .filter(|r: &Value| !done_seeds.contains(&r["seed"].as_i64().expect("a seed")))
        .collect();
    eprintln!("{} candidates to test ({} already in {out_path})", rows.len(), done_seeds.len());
    let out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(out_path).expect("output"));
    let (next, done, played, failed) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let t0 = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(row) = rows.get(i) else { break };
                    let seed = row["seed"].as_i64().unwrap();
                    let decks: Vec<Deck> = serde_json::from_value(row["decks"].clone()).expect("decks");
                    // (a panic in a card script loses this candidate, not the run)
                    match std::panic::catch_unwind(|| test(seed, &decks, &nets, games, stop_min, records.as_ref())) {
                        Ok((wins, n)) => {
                            let mut r = row.clone();
                            let o = r.as_object_mut().unwrap();
                            o.insert("wins".into(), json!(wins));
                            o.insert("games".into(), json!(n));
                            let mut f = out.lock().unwrap();
                            writeln!(f, "{r}").unwrap();
                            f.flush().unwrap();
                            played.fetch_add(n, Ordering::Relaxed);
                        }
                        Err(_) => {
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 500 == 0 {
                        let secs = t0.elapsed().as_secs_f64();
                        eprintln!("{d}/{} candidates, {} games, {:.0}s ({:.1} candidates/s)", rows.len(), played.load(Ordering::Relaxed), secs, d as f64 / secs);
                    }
                }
            });
        }
    });
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "{} candidates ({} failed), {} games in {secs:.0}s -> {out_path}",
        rows.len(),
        failed.load(Ordering::Relaxed),
        played.load(Ordering::Relaxed)
    );
}
