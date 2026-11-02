//! Policy training data from recorded games (ygo-arbiter's records): each game is replayed, and
//! every decision with more than one option becomes a row of what its player sees (the view,
//! the filtered prompt) and the option it chose. Nothing hidden from the player is written.
//!
//!   bot-policy-data OUT_DIR RECORDS.jsonl [RECORDS.jsonl ...] [--jobs N] [--max-games N]
//!
//! OUT_DIR gets, per worker w, three little-endian arrays (see scripts/policy/data.py):
//! - `w.hdr`: a row of HDR i64 per decision (the `H_*` columns)
//! - `w.tok`: a row of TOK i16 per card token of a decision (the `T_*` columns); a decision's
//!   tokens are rows `H_TOK_START..H_TOK_START + H_N_TOK` of its worker's file
//! - `w.opt`: a row of OPT i32 per option (the `O_*` columns), the same way
//!
//! and `games.jsonl`: per game its number, file and line, both decks and `codes`, the card codes
//! a game's code indices (`T_CODE`, ...: index + 1, 0 for an unknown card) stand for.
//!
//! The rows: bot::policy_features.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;
use ygo_core::duel::Duel;

use bot::policy_features::*;
use ygo_arbiter::actions::encode;
use ygo_arbiter::game::{GameRecord, init, run_until_prompt};
use ygo_arbiter::messages::Msg;
use ygo_arbiter::observe::{filter_prompt, to_core_response};
use ygo_arbiter::state::viewer_view;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

struct Game {
    number: usize,
    record: GameRecord,
}

/// Replays `g`, writing its decisions to `shard`; returns the game's codes.
fn game(g: &Game, shard: &mut Shard) -> Result<Vec<u32>, String> {
    let rec = &g.record;
    let labels: HashMap<usize, (u32, f64)> = rec.labels.iter().map(|l| (l.decision, (l.depth, l.value))).collect();
    let winner = rec.result.get("winner").and_then(|w| w.as_u64());
    let mut codes = Codes::default();
    // (the decks' codes first: every game's index table starts with its decks)
    for d in &rec.decks {
        for &c in d.main.iter().chain(&d.extra) {
            codes.of(c);
        }
    }
    let mut duel: Duel = rec.setup(None).map_err(|e| e.0)?;
    duel.ocg_start_duel();
    let mut prompt: Option<Msg> = None;
    let mut seen = [Seen::new(0), Seen::new(1)];
    let mut rows: Vec<Row> = Vec::new();
    for (i, entry) in rec.responses.iter().enumerate() {
        let step = run_until_prompt(&mut duel, prompt.as_ref()).map_err(|e| e.0)?;
        let p = step.prompt.ok_or_else(|| format!("the duel ended before response {i}"))?;
        let me = entry.player;
        for msg in &step.messages {
            for s in &mut seen {
                s.message(msg);
            }
        }
        let fp = filter_prompt(&p, me);
        let view = viewer_view(&mut duel, me);
        let mut chosen = None;
        if let Some(d) = decision(&mut codes, &seen[me as usize], &view, &fp, &entry.response, &rec.decks[me as usize].main) {
            let picked: Vec<usize> = (0..d.opt.len()).filter(|&k| d.opt[k].0[O_CHOSEN] != 0).collect();
            chosen = picked.first().map(|&k| d.chosen_key(k));
            if d.opt.len() >= 2 && !picked.is_empty() {
                let mut h = d.hdr;
                h[H_GAME] = g.number as i64;
                h[H_DECISION] = i as i64;
                (h[H_DEPTH], h[H_VALUE]) = labels.get(&i).map_or((-1, 0), |&(d, v)| (d as i64, v.round() as i64));
                h[H_RESULT] = match winner {
                    Some(w) if w == me as u64 => 1,
                    Some(w) if w < 2 => -1,
                    _ => 0,
                };
                rows.push(Row { hdr: h, tok: d.tok, opt: d.opt.iter().map(|(o, _)| *o).collect() });
            }
        }
        seen[me as usize].decided(chosen);
        let core = to_core_response(&p, me, &entry.response).map_err(|e| format!("response {i}: {e:?}"))?;
        let r = encode(&p, &core).map_err(|e| format!("response {i}: {e:?}"))?;
        duel.ocg_set_response(&r);
        prompt = Some(p);
    }
    shard.write(&rows)?;
    Ok(codes.codes)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let usage = || -> ! {
        eprintln!("usage: bot-policy-data OUT_DIR RECORDS.jsonl [RECORDS.jsonl ...] [--jobs N] [--max-games N]");
        std::process::exit(2)
    };
    let (mut files, mut jobs, mut max_games) = (Vec::new(), 8usize, usize::MAX);
    let mut it = a.iter().skip(2);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--jobs" => jobs = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--max-games" => max_games = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            _ => files.push(x.clone()),
        }
    }
    if a.len() < 3 || files.is_empty() {
        usage();
    }
    let out_dir = std::path::PathBuf::from(&a[1]);
    std::fs::create_dir_all(&out_dir).expect("output directory");
    init(None).unwrap();
    // (the games are read as the workers go: (file, line) in order, numbered as read)
    let lines = files.clone().into_iter().enumerate().flat_map(|(f, path)| {
        std::io::BufReader::new(std::fs::File::open(path).expect("records")).lines().enumerate().map(move |(l, s)| (f, l, s))
    });
    let source = Arc::new(Mutex::new((lines.take(max_games), 0usize)));
    let games_out = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(out_dir.join("games.jsonl")).expect("games.jsonl"))));
    let (done, failed) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let t0 = std::time::Instant::now();
    let threads: Vec<_> = (0..jobs)
        .map(|w| {
            let (source, games_out, done, failed, out_dir) = (source.clone(), games_out.clone(), done.clone(), failed.clone(), out_dir.clone());
            std::thread::spawn(move || {
                let mut shard = Shard::create(&out_dir, w).expect("shard");
                loop {
                    let (number, file, line, text) = {
                        let mut s = source.lock().unwrap();
                        let Some((f, l, text)) = s.0.next() else { break };
                        s.1 += 1;
                        (s.1 - 1, f, l, text)
                    };
                    let Ok(text) = text else { continue };
                    let record: GameRecord = match serde_json::from_str(&text) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("game {number} (file {file} line {line}): {e}");
                            failed.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };
                    let g = Game { number, record };
                    match game(&g, &mut shard) {
                        Ok(codes) => {
                            let row = json!({"game": number, "file": file, "line": line, "worker": w, "decks": g.record.decks.iter().map(|d| json!({"main": d.main, "extra": d.extra})).collect::<Vec<_>>(), "codes": codes});
                            writeln!(games_out.lock().unwrap(), "{row}").expect("games.jsonl");
                        }
                        Err(e) => {
                            eprintln!("game {number} (file {file} line {line}): {e}");
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 1000 == 0 {
                        eprintln!("{n} games, {:.0} s", t0.elapsed().as_secs_f64());
                    }
                }
                shard.finish().expect("shard");
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    games_out.lock().unwrap().flush().unwrap();
    write_meta(&out_dir, json!({"files": files})).expect("meta.json");
    eprintln!("{} games, {} failed, {:.0} s", done.load(Ordering::Relaxed), failed.load(Ordering::Relaxed), t0.elapsed().as_secs_f64());
}
