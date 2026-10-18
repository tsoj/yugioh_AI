//! SPSA tuning of the search parameters (`search::Tune`) on positions instead of games: the
//! update rule of fishtest and OpenBench (c_k = c / k^gamma, a_k = a / (A + k)^alpha, set by each
//! parameter's c_end and r_end = a_end / c_end^2; integer parameters rounded probabilistically, c
//! at least 0.5), but each iteration compares theta + c*flip with theta - c*flip on the same batch
//! of positions instead of a game pair.
//!
//! The positions: from each game of the files one searched decision at random (true state: the
//! tuned values are numbers, nothing of the hidden cards reaches play), the games shuffled. Each
//! position has a reference: every root move searched with the full window (an exact value each)
//! to the depth that a search of REF nodes completes. A search under test has NODES nodes; its loss is the regret of the move it
//! chose by the reference's values, plus VALUE_WEIGHT times its root value's distance to the
//! reference's best (both clamped to CLAMP LP). Searches start from fresh tables (so the history's
//! decay and the time management are not tuned here).
//!
//!   ygoai-spsa OUT_DIR --games FILE [--games FILE ...] [--positions N] [--validation N]
//!       [--nodes N] [--ref-nodes N] [--batch N] [--iterations N] [--jobs N] [--seed N]
//!       [--params name,name,...] [--spec FILE] [--value-weight X] [--clamp LP]
//!       [--alpha X] [--gamma X] [--a-ratio X] [--every N] [--compare NAME=VALUE,...]
//!
//! FILE: ygoai-datagen's OUT.games.jsonl or ygoai-match's records. --spec: lines "name, int|float,
//! min, max, c_end, r_end" (the start is the current default). OUT_DIR gets refs.jsonl (the
//! positions and their references, reused by a later run with the same files, seed, reference
//! nodes and default parameters; --compare: only the defaults vs the given parameters on all the
//! positions, with the paired standard error), spsa.jsonl (each iteration) and the final values as tune.txt ($YGOAI_TUNE).

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::mpsc;

use serde_json::{Value, json};
use ygo_core::duel::Duel;
use ygoai::actions::{Response, encode};
use ygoai::datarow::GameData;
use ygoai::game::{GameRecord, init, run_until_prompt};
use ygoai::messages::Msg;
use ygoai::observe::{is_hidden_prompt, to_core_response};
use ygoai::pyrandom::PyRandom;
use ygoai::search::{AlphaBeta, SearchParams, Tune, candidates, is_searched};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// (the reference searches every root move, also cards beyond the root's width)
const REF_ROOT_WIDTH: usize = 64;
/// (the reference's exact search stops at this many times its nodes, at the depth it completed)
const REF_EXACT_NODES_FACTOR: u64 = 50;
/// (positions whose reference values all lie within this many LP: no move choice to learn from)
const MIN_SPREAD: f64 = 1.0;

struct Spec {
    name: String,
    is_int: bool,
    min: f64,
    max: f64,
    c_end: f64,
    r_end: f64,
}

fn default_specs() -> Vec<Spec> {
    let s = |name: &str, is_int: bool, min: f64, max: f64, c_end: f64| Spec { name: name.into(), is_int, min, max, c_end, r_end: 0.002 };
    vec![
        // (the widths max_children, card_width, root_card_width are not tuned: a spec line adds them)
        s("lmr_min_depth", false, 1.0, 8.0, 0.5),
        s("lmr_min_move_counter", true, 0.0, 10.0, 1.0),
        s("lmr_reduction", false, 0.0, 4.0, 0.25),
        s("lmr_reduction_per_move", false, 0.0, 2.0, 0.05),
        s("futility_max_depth", false, 0.0, 20.0, 1.0),
        s("futility_margin", false, 0.0, 4000.0, 120.0),
        s("futility_margin_per_ply", false, 0.0, 2000.0, 60.0),
        s("hist_exponent", false, 0.0, 6.0, 0.25),
        s("hist_malus", false, 0.0, 4.0, 0.15),
    ]
}

struct Args {
    out: PathBuf,
    games: Vec<PathBuf>,
    positions: usize,
    validation: usize,
    nodes: u64,
    ref_nodes: u64,
    batch: usize,
    iterations: usize,
    jobs: usize,
    seed: u64,
    params: Option<Vec<String>>,
    spec: Option<PathBuf>,
    value_weight: f64,
    clamp: f64,
    alpha: f64,
    gamma: f64,
    a_ratio: f64,
    every: usize,
    compare: Option<String>,
}

fn usage(msg: &str) -> ! {
    eprintln!("ygoai-spsa: {msg}");
    eprintln!("usage: ygoai-spsa OUT_DIR --games FILE [...] (see the source's doc comment for the options)");
    std::process::exit(2)
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = argv.first().filter(|a| !a.starts_with("--")) else { usage("OUT_DIR first") };
    let mut a = Args {
        out: out.into(),
        games: vec![],
        positions: 4000,
        validation: 500,
        nodes: 1000,
        ref_nodes: 2000,
        batch: 64,
        iterations: 3000,
        jobs: std::thread::available_parallelism().map_or(4, |n| n.get()),
        seed: 1,
        params: None,
        spec: None,
        value_weight: 0.1,
        clamp: 8000.0,
        alpha: 0.602,
        gamma: 0.101,
        a_ratio: 0.1,
        every: 100,
        compare: None,
    };
    let mut i = 1;
    while i < argv.len() {
        let k = argv[i].as_str();
        let v = argv.get(i + 1).cloned().unwrap_or_else(|| usage(&format!("{k} needs a value")));
        let num = |s: &str| -> f64 { s.parse().unwrap_or_else(|_| usage(&format!("bad {k}: {s}"))) };
        match k {
            "--games" => a.games.push(v.into()),
            "--positions" => a.positions = num(&v) as usize,
            "--validation" => a.validation = num(&v) as usize,
            "--nodes" => a.nodes = num(&v) as u64,
            "--ref-nodes" => a.ref_nodes = num(&v) as u64,
            "--batch" => a.batch = num(&v) as usize,
            "--iterations" => a.iterations = num(&v) as usize,
            "--jobs" => a.jobs = num(&v) as usize,
            "--seed" => a.seed = num(&v) as u64,
            "--params" => a.params = Some(v.split(',').map(|s| s.trim().to_string()).collect()),
            "--spec" => a.spec = Some(v.into()),
            "--value-weight" => a.value_weight = num(&v),
            "--clamp" => a.clamp = num(&v),
            "--alpha" => a.alpha = num(&v),
            "--gamma" => a.gamma = num(&v),
            "--a-ratio" => a.a_ratio = num(&v),
            "--every" => a.every = num(&v) as usize,
            "--compare" => a.compare = Some(v),
            _ => usage(&format!("unknown option {k}")),
        }
        i += 2;
    }
    if a.games.is_empty() {
        usage("--games is required");
    }
    a
}

fn load_specs(a: &Args) -> Vec<Spec> {
    let mut specs = default_specs();
    if let Some(path) = &a.spec {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| usage(&format!("{}: {e}", path.display())));
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let f: Vec<&str> = line.split(',').map(str::trim).collect();
            if f.len() != 6 || Tune::default().get(f[0]).is_none() {
                usage(&format!("bad spec line: {line}"));
            }
            let p = |s: &str| -> f64 { s.parse().unwrap_or_else(|_| usage(&format!("bad spec line: {line}"))) };
            let spec = Spec { name: f[0].into(), is_int: f[1] == "int", min: p(f[2]), max: p(f[3]), c_end: p(f[4]), r_end: p(f[5]) };
            specs.retain(|s| s.name != spec.name);
            specs.push(spec);
        }
    }
    if let Some(names) = &a.params {
        specs.retain(|s| names.contains(&s.name));
        for n in names {
            if !specs.iter().any(|s| &s.name == n) {
                usage(&format!("no spec for {n}"));
            }
        }
    }
    specs
}

/// A game of a file: its record, and whether its responses are the players' views (records).
fn parse_game(line: &str) -> Option<(GameRecord, bool)> {
    if let Ok(g) = serde_json::from_str::<GameData>(line) {
        return Some((g.record, g.viewer_responses));
    }
    serde_json::from_str::<GameRecord>(line).ok().map(|r| (r, true))
}

/// Replays the game up to response `upto` (all of it: None), calling `at` at each prompt with
/// its index; returns the duel at response `upto` and its prompt.
fn replay(record: &GameRecord, viewer: bool, upto: Option<usize>, mut at: impl FnMut(usize, &Msg)) -> Option<(Duel, Msg)> {
    let mut duel = record.setup(None).ok()?;
    duel.ocg_start_duel();
    let mut prompt: Option<Msg> = None;
    for (i, entry) in record.responses.iter().enumerate() {
        let step = run_until_prompt(&mut duel, prompt.as_ref()).ok()?;
        let p = step.prompt?;
        if upto == Some(i) {
            return Some((duel, p));
        }
        at(i, &p);
        let core = if viewer { to_core_response(&p, entry.player, &entry.response).ok()? } else { entry.response.clone() };
        duel.ocg_set_response(&encode(&p, &core).ok()?);
        prompt = Some(p);
    }
    None
}

/// A decision the search makes: searched, not hidden, with more than one move.
fn searchable(p: &Msg) -> Option<Vec<Response>> {
    if is_hidden_prompt(p) || !is_searched(p) {
        return None;
    }
    let c = candidates(p);
    (c.len() > 1).then_some(c)
}

struct Position {
    duel: Duel,
    prompt: Msg,
    cands: Vec<Response>,
    /// the reference's value of each move (LP, for the mover)
    values: Vec<f64>,
}

impl Position {
    fn best(&self) -> f64 {
        self.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}

/// A search of the position with `tune` (fresh tables), `nodes` and at most `depth` plies: (chosen
/// move, root value, the root moves' values, the depth completed).
fn search(p: &Position, tune: Tune, nodes: u64, depth: Option<i32>, exact_root: bool) -> (usize, f64, Vec<Option<f64>>, i32) {
    let mut ab = AlphaBeta::default();
    ab.tune = tune;
    ab.prepare();
    let params = SearchParams { budget: None, elapsed: 0.0, clock_remaining: 1000.0, hint: None, seeds: [1, 2], max_nodes: Some(nodes), depth, exact_root };
    let (best, done) = ab.search(&p.duel, &p.prompt, &p.cands, &params);
    (best, ab.last_value, std::mem::take(&mut ab.last_root_values), done)
}

/// A position's loss under `tune`: [loss, regret, value error].
fn loss(p: &Position, tune: Tune, a: &LossCfg) -> [f64; 3] {
    let (chosen, value, _, _) = search(p, tune, a.nodes, None, false);
    let best = p.best();
    let regret = (best - p.values[chosen]).min(a.clamp);
    let verr = if value.is_finite() { (value - best).abs().min(a.clamp) } else { a.clamp };
    [regret + a.value_weight * verr, regret, verr]
}

#[derive(Clone, Copy)]
struct LossCfg {
    nodes: u64,
    value_weight: f64,
    clamp: f64,
}

/// The reference of a game's position: (line key, decision index, values), or None.
fn reference(record: &GameRecord, viewer: bool, game_seed: u64, ref_nodes: u64) -> Option<(usize, Position)> {
    let mut eligible = Vec::new();
    replay(record, viewer, None, |i, p| {
        if searchable(p).is_some() {
            eligible.push(i);
        }
    });
    if eligible.is_empty() {
        return None;
    }
    let mut rng = PyRandom::new(game_seed as i128);
    let k = eligible[rng.randrange(eligible.len())];
    let pos = position_at(record, viewer, k, vec![])?;
    // (the depth a search of ref_nodes completes, then every root move with the full window to it)
    let tune = ref_tune();
    let (_, _, _, depth) = search(&pos, tune, ref_nodes, None, false);
    let (_, _, values, _) = search(&pos, tune, REF_EXACT_NODES_FACTOR * ref_nodes, Some(depth.max(1)), true);
    let values: Vec<f64> = values.into_iter().collect::<Option<_>>()?;
    let spread = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - values.iter().cloned().fold(f64::INFINITY, f64::min);
    (spread >= MIN_SPREAD).then_some((k, Position { values, ..pos }))
}

/// The reference's parameters: the defaults with every root card.
fn ref_tune() -> Tune {
    Tune { root_card_width: REF_ROOT_WIDTH, ..Tune::default() }
}

fn position_at(record: &GameRecord, viewer: bool, k: usize, values: Vec<f64>) -> Option<Position> {
    let (duel, prompt) = replay(record, viewer, Some(k), |_, _| {})?;
    let cands = searchable(&prompt)?;
    Some(Position { duel, prompt, cands, values })
}

/// A worker's job: evaluate its positions (local indices) under a parameter set (tag).
enum Job {
    Eval(Vec<(usize, u8)>, Vec<Tune>),
    Stop,
}

fn main() {
    ygoai::search::panic_messages_once();
    let a = parse_args();
    let specs = load_specs(&a);
    init(None).unwrap();
    std::fs::create_dir_all(&a.out).unwrap();
    let refs_path = a.out.join("refs.jsonl");

    // the games, shuffled (one position each)
    let mut lines: Vec<(String, usize)> = Vec::new();
    for f in &a.games {
        let file = std::fs::File::open(f).unwrap_or_else(|e| usage(&format!("{}: {e}", f.display())));
        for (i, _) in std::io::BufReader::new(file).lines().enumerate() {
            lines.push((f.display().to_string(), i));
        }
    }
    let mut rng = PyRandom::new(a.seed as i128);
    rng.shuffle(&mut lines);
    let wanted = a.positions + a.validation;
    // (references of an earlier run: by file and line)
    let mut cached: BTreeMap<(String, usize), (usize, Vec<f64>)> = BTreeMap::new();
    if let Ok(text) = std::fs::read_to_string(&refs_path) {
        for l in text.lines() {
            let v: Value = serde_json::from_str(l).unwrap();
            if v["ref_nodes"].as_u64() == Some(a.ref_nodes) && v["ref_tune"].as_str() == Some(&ref_tune().to_list()) {
                let values = v["values"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                cached.insert((v["file"].as_str().unwrap().into(), v["line"].as_u64().unwrap() as usize), (v["decision"].as_u64().unwrap() as usize, values));
            }
        }
        println!("{} cached references", cached.len());
    }
    // the lines' texts, by file (read once)
    let mut texts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in &a.games {
        let t = std::fs::read_to_string(f).unwrap();
        texts.insert(f.display().to_string(), t.lines().map(String::from).collect());
    }

    // workers: each builds and keeps its share of the positions (a duel stays on its thread)
    let t0 = std::time::Instant::now();
    let cfg = LossCfg { nodes: a.nodes, value_weight: a.value_weight, clamp: a.clamp };
    let (res_tx, res_rx) = mpsc::channel::<(usize, Vec<(usize, u8, [f64; 3])>)>();
    let (built_tx, built_rx) = mpsc::channel::<(usize, Option<(String, usize, usize, Vec<f64>)>)>();
    let mut job_txs = Vec::new();
    let mut handles = Vec::new();
    let next = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let found = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let lines = std::sync::Arc::new(lines);
    let texts = std::sync::Arc::new(texts);
    let cached = std::sync::Arc::new(cached);
    for w in 0..a.jobs {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        job_txs.push(job_tx);
        let (res_tx, built_tx, next, found, lines, texts, cached) = (res_tx.clone(), built_tx.clone(), next.clone(), found.clone(), lines.clone(), texts.clone(), cached.clone());
        let (seed, ref_nodes) = (a.seed, a.ref_nodes);
        handles.push(std::thread::spawn(move || {
            let mut mine: Vec<Position> = Vec::new();
            loop {
                if found.load(std::sync::atomic::Ordering::Relaxed) >= wanted {
                    break;
                }
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some((file, line)) = lines.get(i) else { break };
                let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let (record, viewer) = parse_game(&texts[file][*line])?;
                    match cached.get(&(file.clone(), *line)) {
                        Some((k, values)) => position_at(&record, viewer, *k, values.clone()).map(|p| (*k, p)),
                        None => reference(&record, viewer, seed.wrapping_mul(1_000_003).wrapping_add(i as u64), ref_nodes),
                    }
                }))
                .ok()
                .flatten();
                match built {
                    Some((k, p)) => {
                        found.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let _ = built_tx.send((w, Some((file.clone(), *line, k, p.values.clone()))));
                        mine.push(p);
                    }
                    None => {
                        let _ = built_tx.send((w, None));
                    }
                }
            }
            drop(built_tx);
            while let Ok(job) = job_rx.recv() {
                match job {
                    Job::Stop => break,
                    Job::Eval(items, tunes) => {
                        let out = items
                            .into_iter()
                            .map(|(local, tag)| {
                                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loss(&mine[local], tunes[tag as usize], &cfg))).unwrap_or([cfg.clamp; 3]);
                                (local, tag, r)
                            })
                            .collect();
                        if res_tx.send((w, out)).is_err() {
                            break;
                        }
                    }
                }
            }
        }));
    }
    drop(built_tx);
    // (the positions as (worker, local index), in the order they were found)
    let mut pool: Vec<(usize, usize)> = Vec::new();
    let mut counts = vec![0usize; a.jobs];
    let mut refs_out = std::io::BufWriter::new(std::fs::File::create(a.out.join("refs.new.jsonl")).unwrap());
    let mut tried = 0;
    for (w, b) in built_rx.iter() {
        tried += 1;
        if let Some((file, line, k, values)) = b {
            pool.push((w, counts[w]));
            counts[w] += 1;
            writeln!(refs_out, "{}", json!({"file": file, "line": line, "decision": k, "ref_nodes": a.ref_nodes, "ref_tune": ref_tune().to_list(), "values": values})).unwrap();
        }
        if tried % 200 == 0 {
            println!("[{:6.0}s] {} positions of {tried} games", t0.elapsed().as_secs_f64(), pool.len());
        }
    }
    drop(refs_out);
    std::fs::rename(a.out.join("refs.new.jsonl"), &refs_path).unwrap();
    pool.truncate(wanted);
    println!("[{:6.0}s] {} positions ({} for validation)", t0.elapsed().as_secs_f64(), pool.len(), a.validation.min(pool.len()));
    if pool.len() <= a.validation + a.batch {
        usage("too few positions");
    }
    let (validation, train) = pool.split_at(a.validation);
    let validation = validation.to_vec();
    // (the training positions by worker: a batch takes the same number from each, so that the
    // workers finish an iteration together)
    let mut train: Vec<Vec<(usize, usize)>> = (0..a.jobs).map(|w| train.iter().cloned().filter(|p| p.0 == w).collect()).filter(|v: &Vec<_>| !v.is_empty()).collect();

    // evaluates (position, tag) pairs on the workers: each pair's [loss, regret, value error]
    let evaluate = |items: &[((usize, usize), u8)], tunes: &[Tune]| -> BTreeMap<((usize, usize), u8), [f64; 3]> {
        let mut by_worker: Vec<Vec<(usize, u8)>> = vec![vec![]; job_txs.len()];
        for &((w, local), tag) in items {
            by_worker[w].push((local, tag));
        }
        let mut waiting = 0;
        for (w, its) in by_worker.into_iter().enumerate() {
            if !its.is_empty() {
                job_txs[w].send(Job::Eval(its, tunes.to_vec())).unwrap();
                waiting += 1;
            }
        }
        let mut out = BTreeMap::new();
        for _ in 0..waiting {
            let (w, results) = res_rx.recv().unwrap();
            for (local, tag, r) in results {
                out.insert(((w, local), tag), r);
            }
        }
        out
    };
    // the mean [loss, regret, value error] per tag
    let means = |results: &BTreeMap<((usize, usize), u8), [f64; 3]>, tags: usize| -> Vec<[f64; 3]> {
        let mut sums = vec![[0.0f64; 4]; tags];
        for ((_, tag), r) in results {
            let s = &mut sums[*tag as usize];
            (s[0], s[1], s[2], s[3]) = (s[0] + r[0], s[1] + r[1], s[2] + r[2], s[3] + 1.0);
        }
        sums.iter().map(|s| [s[0] / s[3].max(1.0), s[1] / s[3].max(1.0), s[2] / s[3].max(1.0)]).collect()
    };
    // tag 1's loss minus tag 0's over the same positions: (mean, standard error)
    let paired = |results: &BTreeMap<((usize, usize), u8), [f64; 3]>| -> (f64, f64) {
        let d: Vec<f64> = results.iter().filter(|((_, tag), _)| *tag == 1).filter_map(|((p, _), r)| results.get(&(*p, 0)).map(|r0| r[0] - r0[0])).collect();
        let n = d.len().max(2) as f64;
        let mean = d.iter().sum::<f64>() / n;
        let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, (var / n).sqrt())
    };
    // (--compare: the defaults vs the given parameters on every position, then exit)
    if let Some(list) = &a.compare {
        let other = Tune::parse(list).unwrap_or_else(|e| usage(&e));
        let items: Vec<((usize, usize), u8)> = pool.iter().flat_map(|&p| [(p, 0u8), (p, 1u8)]).collect();
        let results = evaluate(&items, &[Tune::default(), other]);
        let r = means(&results, 2);
        let (d, se) = paired(&results);
        println!(
            "[{:6.0}s] {} positions: loss default {:.1} (regret {:.1}, value error {:.1}), given {:.1} (regret {:.1}, value error {:.1}); given - default {d:+.1} +/- {se:.1}",
            t0.elapsed().as_secs_f64(), pool.len(), r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2]
        );
        for tx in &job_txs {
            let _ = tx.send(Job::Stop);
        }
        return;
    }

    // SPSA (fishtest's and OpenBench's constants)
    let n_iter = a.iterations as f64;
    let big_a = a.a_ratio * n_iter;
    let mut theta: Vec<f64> = specs.iter().map(|s| Tune::default().get(&s.name).unwrap()).collect();
    let c0: Vec<f64> = specs.iter().map(|s| s.c_end * n_iter.powf(a.gamma)).collect();
    let a0: Vec<f64> = specs.iter().map(|s| s.r_end * s.c_end * s.c_end * (big_a + n_iter).powf(a.alpha)).collect();
    let tune_of = |values: &[f64]| -> Tune {
        let mut t = Tune::default();
        for (s, v) in specs.iter().zip(values) {
            t.set(&s.name, *v);
        }
        t
    };
    let validate = |theta: &[f64], k: usize| {
        let rounded: Vec<f64> = specs.iter().zip(theta).map(|(s, v)| if s.is_int { v.round() } else { *v }).collect();
        let items: Vec<((usize, usize), u8)> = validation.iter().flat_map(|&p| [(p, 0u8), (p, 1u8)]).collect();
        let results = evaluate(&items, &[Tune::default(), tune_of(&rounded)]);
        let r = means(&results, 2);
        let (d, se) = paired(&results);
        println!(
            "[{:6.0}s] iter {k}: validation loss default {:.1} (regret {:.1}, value error {:.1}), theta {:.1} (regret {:.1}, value error {:.1}), theta - default {d:+.1} +/- {se:.1}; {}",
            t0.elapsed().as_secs_f64(),
            r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2],
            specs.iter().zip(theta).map(|(s, v)| format!("{}={v:.2}", s.name)).collect::<Vec<_>>().join(" ")
        );
        let _ = std::io::stdout().flush();
    };
    let mut log = std::io::BufWriter::new(std::fs::File::create(a.out.join("spsa.jsonl")).unwrap());
    validate(&theta, 0);
    let (mut sq_sum, mut sq_n) = (0.0f64, 0.0f64);
    let mut cursors: Vec<usize> = train.iter().map(|v| v.len()).collect();
    for k in 1..=a.iterations {
        let kf = k as f64;
        let mut flips = Vec::new();
        let (mut plus, mut minus, mut cs, mut rs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, s) in specs.iter().enumerate() {
            let c = (c0[i] / kf.powf(a.gamma)).max(if s.is_int { 0.5 } else { 0.0 });
            let r = a0[i] / (big_a + kf).powf(a.alpha) / (c * c);
            let flip = if rng.random() < 0.5 { -1.0 } else { 1.0 };
            let (mut dev, mut base) = (theta[i] + flip * c, theta[i] - flip * c);
            if s.is_int {
                let u = rng.random();
                (dev, base) = ((dev + u).floor(), (base + u).floor());
            }
            plus.push(dev.clamp(s.min, s.max));
            minus.push(base.clamp(s.min, s.max));
            flips.push(flip);
            cs.push(c);
            rs.push(r);
        }
        // the batch: the next positions of each worker's shuffled training positions, in turn
        let mut batch = Vec::with_capacity(a.batch);
        while batch.len() < a.batch {
            let w = (k * a.batch + batch.len()) % train.len();
            if cursors[w] >= train[w].len() {
                rng.shuffle(&mut train[w]);
                cursors[w] = 0;
            }
            batch.push(train[w][cursors[w]]);
            cursors[w] += 1;
        }
        let items: Vec<((usize, usize), u8)> = batch.iter().flat_map(|&p| [(p, 0u8), (p, 1u8)]).collect();
        let r = means(&evaluate(&items, &[tune_of(&plus), tune_of(&minus)]), 2);
        // (the result in units of its typical size, like a game pair's: minus' loss over plus')
        let raw = r[1][0] - r[0][0];
        sq_sum += raw * raw;
        sq_n += 1.0;
        let result = (raw / (sq_sum / sq_n).sqrt().max(1e-9)).clamp(-3.0, 3.0);
        for (i, s) in specs.iter().enumerate() {
            theta[i] = (theta[i] + rs[i] * cs[i] * result * flips[i]).clamp(s.min, s.max);
        }
        writeln!(log, "{}", json!({"iter": k, "plus": r[0], "minus": r[1], "result": result, "theta": theta, "c": cs})).unwrap();
        if k % a.every == 0 || k == a.iterations {
            let _ = log.flush();
            validate(&theta, k);
        }
    }
    for tx in &job_txs {
        let _ = tx.send(Job::Stop);
    }
    let final_tune = tune_of(&specs.iter().zip(&theta).map(|(s, v)| if s.is_int { v.round() } else { *v }).collect::<Vec<_>>());
    let list = final_tune.to_list();
    std::fs::write(a.out.join("tune.txt"), format!("{list}\n")).unwrap();
    println!("final: YGOAI_TUNE={list}");
}
