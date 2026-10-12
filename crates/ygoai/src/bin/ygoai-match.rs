//! fastchess-style match between two Yu-Gi-Oh engines (scripts/match.py in Rust; the same
//! options, output and records).
//!
//!   ygoai-match --engine new="python -m ygoai.engine greedy" --engine base=builtin:random \
//!       --book runs/books/match1.jsonl --sprt 0 10 --tc 60+0.5 -n 5000 -c 24
//!
//! Openings come from a book (scripts/make_match_book.py); every opening is played twice, the
//! engines swapping seats. Engines are commands speaking the engine protocol (ygoai/protocol.py)
//! or built-in Rust agents (builtin:random).

use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;

use ygoai::matchrun::{EngineSpec, MatchStats, Opening, load_book, play_opening, sprt_bounds};

struct Args {
    engines: Vec<EngineSpec>,
    openings: usize,
    tc: Option<(f64, f64)>,
    max_turns: u32,
    sprt: Option<(f64, f64)>,
    alpha: f64,
    beta: f64,
    concurrency: usize,
    records: Option<PathBuf>,
    book: Option<PathBuf>,
    book_min: i64,
}

fn usage(msg: &str) -> ! {
    eprintln!("ygoai-match: {msg}");
    eprintln!(
        "usage: ygoai-match --engine NAME=COMMAND --engine NAME=COMMAND --book BOOK [-n N] [--tc BASE+INC] \
         [--sprt ELO0 ELO1] [--alpha A] [--beta B] [-c J] [--records PATH] [--book-min N] [--max-turns N]"
    );
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args {
        engines: Vec::new(),
        openings: 100,
        tc: None,
        max_turns: 100,
        sprt: None,
        alpha: 0.05,
        beta: 0.05,
        concurrency: std::thread::available_parallelism().map_or(1, |n| n.get()),
        records: None,
        book: None,
        book_min: 2,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let next = |i: &mut usize, what: &str| -> String {
        *i += 1;
        argv.get(*i).cloned().unwrap_or_else(|| usage(&format!("{what} needs a value")))
    };
    let num = |s: String, what: &str| -> f64 { s.parse().unwrap_or_else(|_| usage(&format!("bad {what}: {s}"))) };
    while i < argv.len() {
        let arg = argv[i].clone();
        match arg.as_str() {
            "--engine" => {
                let v = next(&mut i, "--engine");
                a.engines.push(EngineSpec::parse(&v).unwrap_or_else(|| usage("expected NAME=COMMAND")));
            }
            "-n" | "--openings" => a.openings = num(next(&mut i, &arg), "-n") as usize,
            "--tc" => {
                let v = next(&mut i, "--tc");
                let (base, inc) = v.split_once('+').unwrap_or((&v, "0"));
                a.tc = Some((num(base.into(), "--tc"), num(inc.into(), "--tc")));
            }
            "--max-turns" => a.max_turns = num(next(&mut i, &arg), &arg) as u32,
            "--sprt" => {
                let e0 = num(next(&mut i, &arg), &arg);
                a.sprt = Some((e0, num(next(&mut i, &arg), &arg)));
            }
            "--alpha" => a.alpha = num(next(&mut i, &arg), &arg),
            "--beta" => a.beta = num(next(&mut i, &arg), &arg),
            "-c" | "--concurrency" => a.concurrency = num(next(&mut i, &arg), &arg) as usize,
            "--records" => a.records = Some(next(&mut i, &arg).into()),
            "--book" => a.book = Some(next(&mut i, &arg).into()),
            "--book-min" => a.book_min = num(next(&mut i, &arg), &arg) as i64,
            // (accepted for scripts/match.py's command lines: only random openings use them)
            "--seed" | "--mirror-prob" => {
                next(&mut i, &arg);
            }
            _ => usage(&format!("unknown argument {arg}")),
        }
        i += 1;
    }
    if a.engines.len() != 2 {
        usage("exactly two engines are required");
    }
    if a.book.is_none() {
        usage("--book is required (random openings from the deck corpus: scripts/match.py)");
    }
    a
}

fn main() {
    let args = parse_args();
    if let Err(e) = ygoai::game::init(None) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    let mut openings: Vec<Opening> = load_book(args.book.as_ref().unwrap(), args.book_min).unwrap_or_else(|e| usage(&format!("book: {e}")));
    openings.truncate(args.openings);
    let (a, b) = (args.engines[0].clone(), args.engines[1].clone());
    println!("{} vs {}: {} openings x 2 games", a.name, b.name, openings.len());
    let mut out = args.records.as_ref().map(|p| std::fs::File::create(p).unwrap_or_else(|e| usage(&format!("records: {e}"))));
    let (lo, hi) = args.sprt.map(|_| sprt_bounds(args.alpha, args.beta)).unwrap_or((0.0, 0.0));
    let openings = Arc::new(openings);
    let next_task = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let n = openings.len();
    for _ in 0..args.concurrency.min(n).max(1) {
        let (openings, next_task, stop, tx, a, b) = (openings.clone(), next_task.clone(), stop.clone(), tx.clone(), a.clone(), b.clone());
        let (tc, max_turns) = (args.tc, args.max_turns);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let i = next_task.fetch_add(1, Ordering::Relaxed);
                let Some(opening) = openings.get(i) else { break };
                let res = catch_unwind(AssertUnwindSafe(|| play_opening(opening, &a, &b, tc, max_turns)))
                    .unwrap_or_else(|p| Err(format!("crash: {}", p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())));
                if tx.send((opening.seed, res)).is_err() {
                    break;
                }
            }
        });
    }
    drop(tx);
    let mut stats = MatchStats::default();
    let mut errors = 0;
    let t0 = Instant::now();
    for (i, (seed, res)) in rx.iter().enumerate() {
        let i = i + 1;
        let games = match res {
            Ok(g) => g,
            Err(e) => {
                errors += 1;
                println!("error: opening {seed}: {e}");
                continue;
            }
        };
        stats.add_pair(&games.iter().map(|g| g.score).collect::<Vec<_>>());
        if let Some(f) = out.as_mut() {
            for g in &games {
                let _ = writeln!(f, "{}", serde_json::to_string(&g.record).unwrap());
            }
            let _ = f.flush();
        }
        if i % 10 == 0 || i == n {
            println!("[{:6.0}s] {}", t0.elapsed().as_secs_f64(), stats.summary(args.sprt));
        }
        if let Some((e0, e1)) = args.sprt {
            let llr = stats.llr(e0, e1);
            if llr <= lo || llr >= hi {
                println!(
                    "SPRT finished: {} (LLR {llr:.2}, bounds [{lo:.2}, {hi:.2}])",
                    if llr >= hi { "H1 accepted" } else { "H0 accepted" }
                );
                stop.store(true, Ordering::Relaxed);
                break;
            }
        }
        let _ = std::io::stdout().flush();
    }
    println!("final: {} errors: {errors}", stats.summary(args.sprt));
    let _ = std::io::stdout().flush();
    // (games still running are abandoned, as the Python arbiter does; their engines end with
    // this process)
    std::process::exit(0);
}
