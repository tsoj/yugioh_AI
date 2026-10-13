//! Training data for the evaluation: self-play of the alpha-beta search on the true duel (both
//! players, a fixed node budget per decision), one game per book opening. At each searched
//! decision a line of JSON: the game, the mover, turn and phase, the search's root value and
//! depth (for the mover), the static evaluation and its terms (`eval_features`, for the mover), a fuller state
//! (`state`, the mover first) and the game's result for the mover (1 win, 0 draw or unfinished,
//! -1 loss).
//!
//!   ygoai-datagen BOOK FIRST COUNT NODES OUT.jsonl [JOBS]
//!
//! (openings FIRST..FIRST+COUNT of the book, every line; games in parallel on JOBS threads)

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use ygo_core::api::QueryInfo;
use ygo_core::constants::*;
use ygo_core::duel::Duel;
use ygoai::game::{DuelConfig, init, new_record, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::matchrun::load_book_head;
use ygoai::messages::Msg;
use ygoai::observe::{auto_response, is_hidden_prompt};
use ygoai::search::{AlphaBeta, EVAL_WEIGHTS, LearnedEval, SearchParams, candidates, eval_features, evaluate, is_searched, rich_features};
use ygoai::state::parse_location_query;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const MAX_DECISIONS: usize = 3000;

/// One player's side: pile sizes, the monster zones (code, face-up, attack position, ATK, DEF,
/// level, rank, type) and the spell/trap zones (code, face-down, type); empty zones are null.
fn side(d: &mut Duel, p: u32) -> Value {
    let q = |d: &mut Duel, loc: u32, flags: u32| parse_location_query(&d.ocg_query_location(&QueryInfo { flags, con: p as u8, loc, seq: 0, overlay_seq: 0 }));
    let mz: Vec<Value> = q(d, LOCATION_MZONE, QUERY_CODE | QUERY_POSITION | QUERY_TYPE | QUERY_LEVEL | QUERY_RANK | QUERY_ATTACK | QUERY_DEFENSE)
        .into_iter()
        .map(|c| match c {
            None => Value::Null,
            Some(c) => {
                let pos = c.position.unwrap_or(0);
                json!([c.code, (pos & POS_FACEUP != 0) as u8, (pos & POS_ATTACK != 0) as u8, c.attack, c.defense, c.level, c.rank, c.type_])
            }
        })
        .collect();
    let sz: Vec<Value> = q(d, LOCATION_SZONE, QUERY_CODE | QUERY_POSITION | QUERY_TYPE)
        .into_iter()
        .map(|c| match c {
            None => Value::Null,
            Some(c) => json!([c.code, (c.position.unwrap_or(0) & POS_FACEDOWN != 0) as u8, c.type_]),
        })
        .collect();
    let n = |d: &mut Duel, loc: u32| d.ocg_query_count(p as u8, loc);
    json!({
        "lp": d.ext_get_info()[3 + p as usize] as i32,
        "hand": n(d, LOCATION_HAND), "deck": n(d, LOCATION_DECK), "grave": n(d, LOCATION_GRAVE),
        "removed": n(d, LOCATION_REMOVED), "extra": n(d, LOCATION_EXTRA),
        "mzone": mz, "szone": sz,
    })
}

fn play(book: &ygoai::matchrun::Opening, game: usize, nodes: u64) -> Vec<Value> {
    let record = new_record(&book.decks, book.seed, &DuelConfig::default());
    let mut duel = record.setup(None).unwrap();
    duel.ocg_start_duel();
    let mut players = [Heuristic::new(1, 0), Heuristic::new(2, 1)];
    let mut abs = [AlphaBeta::default(), AlphaBeta::default()];
    let (mut rows, mut prompt, mut winner): (Vec<Value>, Option<Msg>, Option<u32>) = (Vec::new(), None, None);
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
                let info = duel.ext_get_info();
                let (f, stat) = (eval_features(&mut duel, player), evaluate(&mut duel, player));
                let dot: f64 = f.iter().zip(EVAL_WEIGHTS).map(|(x, w)| x * w).sum();
                if LearnedEval::from_env().is_none() {
                    assert!((dot - stat).abs() < 1e-6, "eval_features disagrees with evaluate: {dot} vs {stat}");
                }
                rows.push(json!({
                    "game": game, "player": player, "turn": info[0], "turn_player": info[1], "phase": info[2],
                    "prompt": p.name(), "value": ab.last_value, "depth": done,
                    "static": stat, "eval": f.to_vec(), "rich": rich_features(&mut duel, player),
                    "state": [side(&mut duel, player), side(&mut duel, 1 - player)],
                }));
            }
            cands[best].clone()
        } else {
            players[player as usize].respond_to(&p, None).unwrap()
        };
        let Ok(r) = ygoai::actions::encode(&p, &resp) else { break };
        duel.ocg_set_response(&r);
        prompt = Some(p);
    }
    for r in &mut rows {
        let me = r["player"].as_u64().unwrap() as u32;
        r["result"] = json!(match winner {
            Some(w) if w == me => 1,
            Some(w) if w < 2 => -1,
            _ => 0,
        });
    }
    rows
}

fn main() {
    ygoai::search::panic_messages_once();
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: ygoai-datagen BOOK FIRST COUNT NODES OUT.jsonl [JOBS]");
        std::process::exit(2);
    }
    let (first, count, nodes): (usize, usize, u64) = (a[2].parse().unwrap(), a[3].parse().unwrap(), a[4].parse().unwrap());
    let jobs: usize = a.get(6).map_or(8, |s| s.parse().unwrap());
    let book = Arc::new(load_book_head(a[1].as_ref(), 0, first + count).expect("book").split_off(first));
    init(None).unwrap();
    let out = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(&a[5]).expect("output"))));
    let (next, done) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let t0 = std::time::Instant::now();
    let threads: Vec<_> = (0..jobs)
        .map(|_| {
            let (book, out, next, done) = (book.clone(), out.clone(), next.clone(), done.clone());
            std::thread::spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= book.len() {
                        break;
                    }
                    let rows = std::panic::catch_unwind(|| play(&book[i], first + i, nodes)).unwrap_or_default();
                    let mut w = out.lock().unwrap();
                    for r in rows {
                        writeln!(w, "{r}").unwrap();
                    }
                    w.flush().unwrap();
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
