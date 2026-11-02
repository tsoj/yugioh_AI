//! Games of the imitation policy for reinforcement learning (scripts/policy/rl.py): the learner
//! (`--net`, sampling at `--temp`) plays book openings against itself or `--opponent`, in both
//! seats, `--repeat` times each; its decisions as it saw them (bot::policy_features' rows, the
//! chosen options marked) go to OUT_DIR in bot-policy-data's format, H_RESULT the game's result
//! for the decider. A decision's H_GAME is its game's player: game * 2 + player.
//!
//!   bot-rl-games OUT_DIR --net NET.bin --tables T.bin --book BOOK --first F --count N
//!       [--opponent NET.bin] [--repeat G] [--temp T] [--jobs J] [--seed S]
//!   bot-rl-games --test --net NET.bin --opponent NET.bin --tables T.bin --book BOOK --first F --count N [--jobs J]
//!
//! --remote (instead of --net and --opponent): the caller scores, over stdin and stdout
//! (bot::policy_remote; scripts/policy/rl.py on the GPU): network 0 the learner's, 1 the
//! opponent's with --versus (else self-play); the summary is the end's. Then --jobs games at a
//! time (each its thread), their decisions batched.
//!
//! OUT_DIR/games.jsonl: per game player {"game" (game * 2 + player), "opening", "seat" (the
//! learner's), "repeat", "player", "learner" (its rows are the learner's), "names" (its
//! pseudo-name seed), "codes", "result"}. --test: both play their best options (no rows); prints
//! the net's score as JSON.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;

use std::time::Duration;

use bot::imitation::{ImitationBot, Net, Scorer};
use bot::policy_features::*;
use bot::policy_remote::Remote;
use policynet::Tables;
use ygo_arbiter::game::{DuelConfig, PlayOptions, init, new_record, play};
use ygo_arbiter::matchrun::{Opening, load_book, score_to_elo};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

struct Args {
    out: Option<PathBuf>,
    net: PathBuf,
    opponent: Option<PathBuf>,
    remote: bool,
    versus: bool,
    tables: PathBuf,
    book: PathBuf,
    first: usize,
    count: usize,
    repeat: usize,
    temp: f32,
    jobs: usize,
    seed: i64,
    test: bool,
}

fn args() -> Args {
    let usage = || -> ! {
        eprintln!("usage: bot-rl-games OUT_DIR --net NET.bin --tables T.bin --book BOOK --first F --count N [--opponent NET.bin] [--repeat G] [--temp T] [--jobs J] [--seed S]");
        eprintln!("       bot-rl-games --test --net NET.bin --opponent NET.bin --tables T.bin --book BOOK --first F --count N [--jobs J]");
        eprintln!("       (--remote [--versus] instead of --net and --opponent: the caller scores over stdin/stdout)");
        std::process::exit(2)
    };
    let mut a = Args {
        out: None,
        net: PathBuf::new(),
        opponent: None,
        remote: false,
        versus: false,
        tables: PathBuf::new(),
        book: PathBuf::new(),
        first: 0,
        count: 0,
        repeat: 1,
        temp: 1.0,
        jobs: 8,
        seed: 1,
        test: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage());
        match x.as_str() {
            "--net" => a.net = value().into(),
            "--opponent" => a.opponent = Some(value().into()),
            "--tables" => a.tables = value().into(),
            "--book" => a.book = value().into(),
            "--first" => a.first = value().parse().unwrap_or_else(|_| usage()),
            "--count" => a.count = value().parse().unwrap_or_else(|_| usage()),
            "--repeat" => a.repeat = value().parse().unwrap_or_else(|_| usage()),
            "--temp" => a.temp = value().parse().unwrap_or_else(|_| usage()),
            "--jobs" => a.jobs = value().parse().unwrap_or_else(|_| usage()),
            "--seed" => a.seed = value().parse().unwrap_or_else(|_| usage()),
            "--test" => a.test = true,
            "--remote" => a.remote = true,
            "--versus" => a.versus = true,
            _ if !x.starts_with("--") && a.out.is_none() => a.out = Some(x.into()),
            _ => usage(),
        }
    }
    let nets = match a.remote {
        true => a.net.as_os_str().is_empty() && a.opponent.is_none(),
        false => !a.net.as_os_str().is_empty() && !a.versus,
    };
    let versus = a.versus || a.opponent.is_some();
    if a.count == 0 || !nets || a.tables.as_os_str().is_empty() || (a.test != a.out.is_none()) || (a.test && !versus) {
        usage();
    }
    a
}

/// A game: the opening, the learner's seat, the repetition.
struct Task {
    opening: usize,
    seat: u32,
    repeat: usize,
}

/// The game's result for `player`: 1 won, -1 lost, 0 a draw or unfinished.
fn result_for(result: &serde_json::Map<String, serde_json::Value>, player: u32) -> i64 {
    match result.get("winner").and_then(|w| w.as_u64()) {
        Some(w) if w == player as u64 => 1,
        Some(w) if w < 2 => -1,
        _ => 0,
    }
}

fn main() {
    let a = args();
    init(None).unwrap();
    let book: Vec<Opening> = load_book(&a.book, usize::MAX, a.first + a.count).unwrap_or_else(|e| panic!("{e}"));
    if book.len() < a.first + a.count {
        panic!("{}: {} openings, not {}", a.book.display(), book.len(), a.first + a.count);
    }
    let tables = Arc::new(Tables::load(&a.tables).unwrap_or_else(|e| panic!("{e}")));
    let remote = a.remote.then(|| Arc::new(Remote::new(a.jobs)));
    let scorer = |path: &PathBuf, net: u32| match &remote {
        Some(remote) => Scorer::Remote { remote: remote.clone(), tables: tables.clone(), net },
        None => Scorer::Net(Net::load(path, tables.clone()).unwrap_or_else(|e| panic!("{e}"))),
    };
    let net = scorer(&a.net, 0);
    let opponent = (a.versus || a.opponent.is_some()).then(|| scorer(a.opponent.as_ref().unwrap_or(&a.net), 1));
    let self_play = opponent.is_none();
    let tasks: Vec<Task> = (a.first..a.first + a.count)
        .flat_map(|opening| (0..2).flat_map(move |seat| (0..a.repeat).map(move |repeat| Task { opening, seat, repeat })))
        .collect();
    if let Some(out) = &a.out {
        std::fs::create_dir_all(out).expect("output directory");
    }
    let games_out = a.out.as_ref().map(|o| Mutex::new(std::io::BufWriter::new(std::fs::File::create(o.join("games.jsonl")).expect("games.jsonl"))));
    let (next, done) = (AtomicUsize::new(0), AtomicUsize::new(0));
    // (the learner's points per game, for --test)
    let points = Mutex::new((0.0f64, 0usize));
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for w in 0..a.jobs {
            let (a, book, tasks, net, opponent, games_out, next, done, points) = (&a, &book, &tasks, &net, &opponent, &games_out, &next, &done, &points);
            let remote = remote.clone();
            s.spawn(move || {
                // (the server's count of playing threads, even if this one panics)
                struct Done(Option<Arc<Remote>>);
                impl Drop for Done {
                    fn drop(&mut self) {
                        if let Some(r) = &self.0 {
                            r.worker_done();
                        }
                    }
                }
                let _done = Done(remote);
                let mut shard = a.out.as_ref().map(|o| Shard::create(o, w).expect("shard"));
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(t) = tasks.get(i) else { break };
                    let opening = &book[t.opening];
                    // (the bots' seeds: per task, the same for any number of jobs)
                    let seed = |p: u32| a.seed.wrapping_mul(1_000_003).wrapping_add(i as i64 * 2 + p as i64);
                    let temp = if a.test { 0.0 } else { a.temp };
                    let learner_log = shard.is_some();
                    let make = |p: u32| -> ImitationBot {
                        if p == t.seat || self_play {
                            ImitationBot::with(net.clone(), seed(p), temp, learner_log)
                        } else {
                            ImitationBot::with(opponent.clone().unwrap(), seed(p), temp, false)
                        }
                    };
                    let (mut b0, mut b1) = (make(0), make(1));
                    let mut record = new_record(&opening.decks, opening.seed, &DuelConfig::default());
                    if let Err(e) = play(&mut record, &mut [&mut b0, &mut b1], &mut [], &mut PlayOptions::default()) {
                        eprintln!("opening {} seat {} repeat {}: {}", t.opening, t.seat, t.repeat, e.0);
                        continue;
                    }
                    let mine = result_for(&record.result, t.seat);
                    {
                        let mut p = points.lock().unwrap();
                        p.0 += (mine as f64 + 1.0) / 2.0;
                        p.1 += 1;
                    }
                    if let (Some(shard), Some(games_out)) = (&mut shard, games_out) {
                        for (p, bot) in [(0u32, &mut b0), (1, &mut b1)] {
                            let Some(log) = bot.log.take() else { continue };
                            let (game, result) = (i as i64 * 2 + p as i64, result_for(&record.result, p));
                            let rows: Vec<Row> = log
                                .into_iter()
                                .map(|mut r| {
                                    r.hdr[H_GAME] = game;
                                    r.hdr[H_RESULT] = result;
                                    (r.hdr[H_DEPTH], r.hdr[H_VALUE]) = (-1, 0);
                                    r
                                })
                                .collect();
                            shard.write(&rows).expect("shard");
                            let row = json!({"game": game, "opening": t.opening, "seat": t.seat, "repeat": t.repeat, "player": p,
                                             "learner": p == t.seat || self_play, "names": bot.names, "codes": bot.codes(), "result": result,
                                             "decks": record.decks.iter().map(|d| json!({"main": d.main, "extra": d.extra})).collect::<Vec<_>>()});
                            writeln!(games_out.lock().unwrap(), "{row}").expect("games.jsonl");
                        }
                    }
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 100 == 0 {
                        eprintln!("{n}/{} games, {:.0} s", tasks.len(), t0.elapsed().as_secs_f64());
                    }
                }
                if let Some(shard) = shard {
                    shard.finish().expect("shard");
                }
            });
        }
        if let Some(remote) = &remote {
            let summary = || summary(&points, t0);
            remote
                .serve(&mut std::io::stdin().lock(), &mut std::io::BufWriter::new(std::io::stdout().lock()), Duration::from_millis(20), summary)
                .expect("remote scores");
        }
    });
    if let Some(out) = &a.out {
        games_out.unwrap().into_inner().unwrap().flush().unwrap();
        write_meta(out, json!({"net": a.net, "opponent": a.opponent, "book": a.book, "first": a.first, "count": a.count, "repeat": a.repeat, "temp": a.temp, "seed": a.seed}))
            .expect("meta.json");
    }
    if remote.is_none() {
        println!("{}", summary(&points, t0));
    }
}

/// The learner's score as JSON.
fn summary(points: &Mutex<(f64, usize)>, t0: std::time::Instant) -> String {
    let (sum, n) = *points.lock().unwrap();
    let score = sum / n.max(1) as f64;
    json!({"games": n, "score": score, "elo": score_to_elo(score.clamp(1e-3, 1.0 - 1e-3)), "seconds": t0.elapsed().as_secs_f64().round()}).to_string()
}
