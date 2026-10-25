//! Training data for the card policy (`search::cardpolicy`): from games' records, one-card
//! selections (`cardpolicy::card_policy_applies`) of the search's player, each with a reference value of
//! every card: the depth a search of REF nodes completes, then every card searched with the full
//! window to it (as ygoai-spsa's references), on the true duel.
//!
//!   ygoai-card-refs OUT.jsonl --games FILE [--games FILE ...] [--independent] [--positions N]
//!       [--per-game N] [--ref-nodes N] [--jobs N] [--seed N]
//!
//! FILE: ygoai-datagen's OUT.games.jsonl or ygoai-match's records; the games shuffled, up to
//! --per-game selections of each at random. A row: {"file", "line", "k": the decision's index,
//! "hint", "player", "values": each card's (LP, for the selecting player), "rules": the rules'
//! order, "depth", "features": each card's `cardpolicy::card_features`}.
//!
//!   ygoai-card-refs OUT.jsonl --games FILE [...] --decisions [--independent] [options as above]
//!
//! The decisions the search leaves to the rules below the root (not `is_searched`), or a set
//! policy orders (a card selection of several cards, a sum), with more than one of `root_candidates`, each with every
//! candidate's reference value as above and the rules' answer: {"file", "line", "k", "type",
//! "count": the game's decisions of that type, "player", "values", "rules": the rules' answer's
//! index (added as a candidate if missing), "depth"}; --per-game of each type. How often and by
//! how much searching them would change the rules' answers. A card selection's rows (several
//! cards, one at a time, a sum, tributes) also have "hint", "order": the rules' order of its
//! cards, "features": each card's `cardpolicy::card_features`, and "sets": each candidate's cards
//! (finishing: none), the data of a policy for these. A zone selection's rows (every zone the
//! candidates if one is chosen) have "hint", "features": each zone's
//! `placepolicy::place_features` (`available_places`' order), and "sets": each candidate's zones.
//! --types T,T: only these types (as "type"). --independent: each candidate's value in a search of
//! its own (`exact_values`).

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use ygo_core::constants::HINT_SELECTMSG;
use ygo_core::duel::Duel;
use ygoai::actions::encode;
use ygoai::datarow::GameData;
use ygoai::game::{GameRecord, init, run_until_prompt};
use ygoai::heuristic::Heuristic;
use ygoai::messages::Msg;
use ygoai::observe::to_core_response;
use ygoai::pyrandom::PyRandom;
use ygoai::search::cardpolicy::{card_features, card_policy_applies, selection_cards};
use ygoai::actions::Response;
use ygoai::actions::available_places;
use ygoai::search::movegen::{is_searched, root_candidates};
use ygoai::search::placepolicy::place_features;
use ygoai::search::{AlphaBeta, SearchParams, Tune, candidates};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// (every card searched, as ygoai-spsa's references)
const REF_ROOT_WIDTH: usize = 64;
const REF_EXACT_NODES_FACTOR: u64 = 50;

struct Args {
    out: PathBuf,
    games: Vec<PathBuf>,
    positions: usize,
    per_game: usize,
    ref_nodes: u64,
    jobs: usize,
    seed: u64,
    decisions: bool,
    types: Option<Vec<String>>,
    independent: bool,
}

fn usage(msg: &str) -> ! {
    eprintln!("ygoai-card-refs: {msg}");
    eprintln!("usage: ygoai-card-refs OUT.jsonl --games FILE [...] (see the source's doc comment for the options)");
    std::process::exit(2)
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(out) = argv.first().filter(|a| !a.starts_with("--")) else { usage("OUT.jsonl first") };
    let mut a = Args {
        out: out.into(),
        games: vec![],
        positions: 20000,
        per_game: 2,
        ref_nodes: 2000,
        jobs: std::thread::available_parallelism().map_or(4, |n| n.get()),
        seed: 1,
        decisions: false,
        types: None,
        independent: false,
    };
    let mut i = 1;
    while i < argv.len() {
        let k = argv[i].as_str();
        if k == "--decisions" || k == "--independent" {
            if k == "--decisions" {
                a.decisions = true;
            } else {
                a.independent = true;
            }
            i += 1;
            continue;
        }
        let v = argv.get(i + 1).cloned().unwrap_or_else(|| usage(&format!("{k} needs a value")));
        let num = |s: &str| -> f64 { s.parse().unwrap_or_else(|_| usage(&format!("bad {k}: {s}"))) };
        match k {
            "--games" => a.games.push(v.into()),
            "--positions" => a.positions = num(&v) as usize,
            "--per-game" => a.per_game = num(&v) as usize,
            "--ref-nodes" => a.ref_nodes = num(&v) as u64,
            "--jobs" => a.jobs = num(&v) as usize,
            "--seed" => a.seed = num(&v) as u64,
            "--types" => a.types = Some(v.split(',').map(str::to_string).collect()),
            _ => usage(&format!("unknown option {k}")),
        }
        i += 2;
    }
    if a.games.is_empty() {
        usage("--games is required");
    }
    a
}

/// A game of a file: its record, and whether its responses are the players' views (records).
fn parse_game(line: &str) -> Option<(GameRecord, bool)> {
    if let Ok(g) = serde_json::from_str::<GameData>(line) {
        return Some((g.record, g.viewer_responses));
    }
    serde_json::from_str::<GameRecord>(line).ok().map(|r| (r, true))
}

/// Replays the game, calling `at` at each prompt with its index, the prompt, the select hint its
/// player got before it, and the duel; `at` returns whether to go on.
fn replay(record: &GameRecord, viewer: bool, mut at: impl FnMut(usize, &Msg, Option<u64>, &Duel) -> bool) -> Option<()> {
    let mut duel = record.setup(None).ok()?;
    duel.ocg_start_duel();
    let mut prompt: Option<Msg> = None;
    let mut hints: [Option<u64>; 2] = [None, None];
    for (i, entry) in record.responses.iter().enumerate() {
        let step = run_until_prompt(&mut duel, prompt.as_ref()).ok()?;
        for m in &step.messages {
            if let Msg::Hint { hint_type, player, data } = m
                && *hint_type == HINT_SELECTMSG
                && *player < 2
            {
                hints[*player as usize] = Some(*data);
            }
        }
        let p = step.prompt?;
        let hint = p.player().and_then(|q| hints.get_mut(q as usize)).and_then(|h| h.take());
        if !at(i, &p, hint, &duel) {
            return Some(());
        }
        let core = if viewer { to_core_response(&p, entry.player, &entry.response).ok()? } else { entry.response.clone() };
        duel.ocg_set_response(&encode(&p, &core).ok()?);
        prompt = Some(p);
    }
    Some(())
}

fn search(duel: &Duel, prompt: &Msg, moves: &[Response], hint: Option<u64>, nodes: u64, depth: Option<i32>, exact_root: bool) -> (Vec<Option<f64>>, i32) {
    let mut ab = AlphaBeta::default();
    ab.tune = Tune { root_card_width: REF_ROOT_WIDTH, ..Tune::default() };
    ab.prepare();
    let params = SearchParams { budget: None, elapsed: 0.0, clock_remaining: 1000.0, hint, seeds: [1, 2], max_nodes: Some(nodes), depth, exact_root };
    let (_, done) = ab.search(duel, prompt, moves, &params);
    (std::mem::take(&mut ab.last_root_values), done)
}

/// Every move's value searched with the full window to `depth`: in one search, or with
/// `independent` each in a search of its own (fresh tables: a move searched first in a shared
/// search ends up with another value than the same move later, where the moves' choice does not
/// matter, by the tables' state).
fn exact_values(duel: &Duel, prompt: &Msg, moves: &[Response], hint: Option<u64>, ref_nodes: u64, depth: i32, independent: bool) -> Option<Vec<f64>> {
    let exact = |ms: &[Response]| search(duel, prompt, ms, hint, REF_EXACT_NODES_FACTOR * ref_nodes, Some(depth.max(1)), true).0;
    if independent {
        moves.iter().map(|m| exact(std::slice::from_ref(m)).into_iter().next().flatten()).collect()
    } else {
        exact(moves).into_iter().collect()
    }
}

/// Whether the rules answer `m` below the root (`--decisions`), or a set policy orders it (several
/// cards, a sum), and its moves at the root with the rules' answer's index.
fn rules_decision(m: &Msg, hint: Option<u64>) -> Option<(Vec<Response>, usize)> {
    if is_searched(m) && !matches!(m, Msg::SelectCard { min, max, .. } if (*min, *max) != (1, 1)) && !matches!(m, Msg::SelectSum { .. }) {
        return None;
    }
    let player = m.player()?;
    let mut rules = Heuristic::new(1, player);
    rules.hint = hint;
    let mut moves = match m {
        // (one zone: every zone, for the place policy)
        Msg::SelectPlace { player, count: 1, flag } | Msg::SelectDisfield { player, count: 1, flag } => {
            available_places(*player, *flag).into_iter().map(|z| Response { places: Some(vec![z]), ..Response::default() }).collect()
        }
        _ => root_candidates(m, &rules, None),
    };
    let answer = rules.respond_to(m, None).ok()?;
    let at = match moves.iter().position(|r| *r == answer) {
        Some(i) => i,
        None => {
            moves.insert(0, answer);
            0
        }
    };
    (moves.len() > 1).then_some((moves, at))
}

/// The rows of a game (`--decisions`): up to `per_game` of each type of the decisions the rules
/// answer below the root, at random ("count": the game's decisions of that type).
fn decision_rows(file: &str, line: usize, text: &str, a: &Args) -> Vec<String> {
    let Some((record, viewer)) = parse_game(text) else { return Vec::new() };
    let mut by_type: std::collections::BTreeMap<&'static str, Vec<usize>> = Default::default();
    let wanted = |p: &Msg| a.types.as_ref().is_none_or(|t| t.iter().any(|t| t == p.name()));
    replay(&record, viewer, |i, p, hint, _| {
        if wanted(p) && rules_decision(p, hint).is_some() {
            by_type.entry(p.name()).or_default().push(i);
        }
        true
    });
    let mut rng = PyRandom::new((a.seed as i128) << 32 | line as i128);
    let mut eligible = Vec::new();
    for ks in by_type.values_mut() {
        let count = ks.len();
        rng.shuffle(ks);
        eligible.extend(ks.iter().take(a.per_game).map(|&k| (k, count)));
    }
    let mut rows = Vec::new();
    for (k, count) in eligible {
        let mut found = None;
        replay(&record, viewer, |i, p, hint, duel| {
            if i == k {
                found = Some((duel.clone(), p.clone(), hint));
            }
            i < k
        });
        let Some((duel, prompt, hint)) = found else { continue };
        let Some((moves, at)) = rules_decision(&prompt, hint) else { continue };
        let (_, depth) = search(&duel, &prompt, &moves, hint, a.ref_nodes, None, false);
        let Some(values) = exact_values(&duel, &prompt, &moves, hint, a.ref_nodes, depth, a.independent) else { continue };
        let player = prompt.player();
        let mut row = json!({"file": file, "line": line, "k": k, "type": prompt.name(), "count": count, "player": player, "values": values, "rules": at, "depth": depth});
        if let Some((cards, &p)) = selection_cards(&prompt) {
            let mut rules_player = Heuristic::new(1, p);
            rules_player.hint = hint;
            let order = rules_player.card_order(cards);
            let features: Vec<Vec<f32>> = card_features(&duel, &prompt, hint, &order).iter().map(|x| x.to_vec()).collect();
            let sets: Vec<Vec<i64>> = moves.iter().map(|m| m.indices.clone().or(m.index.map(|i| vec![i])).unwrap_or_default()).collect();
            row["hint"] = json!(hint);
            row["order"] = json!(order);
            row["features"] = json!(features);
            row["sets"] = json!(sets);
        }
        if let Msg::SelectPlace { player, flag, .. } | Msg::SelectDisfield { player, flag, .. } = &prompt {
            let places = available_places(*player, *flag);
            let features: Vec<Vec<f32>> = place_features(&duel, &prompt, hint).iter().map(|x| x.to_vec()).collect();
            let sets: Vec<Vec<usize>> = moves.iter().map(|m| m.places.iter().flatten().filter_map(|z| places.iter().position(|p| p == z)).collect()).collect();
            row["hint"] = json!(hint);
            row["features"] = json!(features);
            row["sets"] = json!(sets);
        }
        rows.push(row.to_string());
    }
    rows
}

/// The rows of a game: up to `per_game` of its selections at random.
fn game_rows(file: &str, line: usize, text: &str, a: &Args) -> Vec<String> {
    if a.decisions {
        return decision_rows(file, line, text, a);
    }
    let Some((record, viewer)) = parse_game(text) else { return Vec::new() };
    let mut eligible = Vec::new();
    replay(&record, viewer, |i, p, _, _| {
        if card_policy_applies(p) {
            eligible.push(i);
        }
        true
    });
    let mut rng = PyRandom::new((a.seed as i128) << 32 | line as i128);
    rng.shuffle(&mut eligible);
    eligible.truncate(a.per_game);
    let mut rows = Vec::new();
    for k in eligible {
        let mut found = None;
        replay(&record, viewer, |i, p, hint, duel| {
            if i == k {
                found = Some((duel.clone(), p.clone(), hint));
            }
            i < k
        });
        let Some((duel, prompt, hint)) = found else { continue };
        let Msg::SelectCard { cards, player, .. } = &prompt else { continue };
        let mut rules_player = Heuristic::new(1, *player);
        rules_player.hint = hint;
        let rules = rules_player.card_order(cards);
        let moves = candidates(&prompt);
        let (_, depth) = search(&duel, &prompt, &moves, hint, a.ref_nodes, None, false);
        let Some(values) = exact_values(&duel, &prompt, &moves, hint, a.ref_nodes, depth, a.independent) else { continue };
        let features: Vec<Vec<f32>> = card_features(&duel, &prompt, hint, &rules).iter().map(|x| x.to_vec()).collect();
        rows.push(
            json!({"file": file, "line": line, "k": k, "hint": hint, "player": player, "values": values, "rules": rules, "depth": depth, "features": features})
                .to_string(),
        );
    }
    rows
}

fn main() {
    ygoai::search::panic_messages_once();
    let a = parse_args();
    init(None).unwrap();
    let mut games: Vec<(String, usize, String)> = Vec::new();
    for f in &a.games {
        let file = std::fs::File::open(f).unwrap_or_else(|e| usage(&format!("{}: {e}", f.display())));
        for (i, l) in std::io::BufReader::new(file).lines().enumerate() {
            games.push((f.display().to_string(), i, l.unwrap()));
        }
    }
    let mut rng = PyRandom::new(a.seed as i128);
    rng.shuffle(&mut games);
    let out = Mutex::new(std::io::BufWriter::new(std::fs::File::create(&a.out).unwrap_or_else(|e| usage(&format!("{}: {e}", a.out.display())))));
    let (next, written) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let start = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..a.jobs {
            s.spawn(|| {
                while written.load(Ordering::Relaxed) < a.positions {
                    let g = next.fetch_add(1, Ordering::Relaxed);
                    let Some((file, line, text)) = games.get(g) else { break };
                    let rows = game_rows(file, *line, text, &a);
                    let mut o = out.lock().unwrap();
                    for r in &rows {
                        writeln!(o, "{r}").unwrap();
                    }
                    o.flush().unwrap();
                    let n = written.fetch_add(rows.len(), Ordering::Relaxed) + rows.len();
                    if n / 100 > (n - rows.len()) / 100 {
                        eprintln!("{n} positions, {:.0} s", start.elapsed().as_secs_f64());
                    }
                }
            });
        }
    });
    eprintln!("ygoai-card-refs: {} positions in {:.0} s", written.load(Ordering::Relaxed), start.elapsed().as_secs_f64());
}
