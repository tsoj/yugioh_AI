//! A recorded game's decision as the alpha-beta search sees it (on the true duel, not a belief's
//! reconstruction): replays RECORDS' line LINE to its decision K (the record's response index) and
//! prints every root candidate (the bot's: `root_candidates` of the prompt as the player sees it)
//! with its value, the one played marked.
//!
//!   bot-analyze RECORDS.jsonl LINE K [--nodes 5000] [--exact] [--line N OUT.jsonl]
//!
//! --exact: every candidate's value with the full window. --line: from K on, N more decisions
//! with the search choosing for both players (searching each with more than one candidate); the
//! game's record up to then to OUT (for a viewer).
//!
//!   bot-analyze RECORDS.jsonl LINE K --bot [--nodes 5000]
//!
//! --bot: the bot itself (alphabeta: its belief's reconstruction, the player's view) at decision
//! K, after the game's earlier decisions as played (searched with few nodes, then replaced by the
//! recorded responses); prints its response, its search and its reconstruction failures.
//! $BOT_ANALYZE_SEED: the bot's seed (default 1; its reconstructions' guesses).
//!
//!   bot-analyze RECORDS.jsonl LINE K --playout N [--force RESPONSE] [--nodes 5000]
//!
//! --playout: N games from decision K on, the bot (alphabeta) on both sides, after the game's
//! earlier decisions as played; decision K the bot's own or RESPONSE (JSON). Game i: the bots'
//! seeds 2i+1, 2i+2. Prints each game's winner. $BOT_ANALYZE_WINDBOT=DIR (WindBot's build) and
//! $BOT_ANALYZE_SEATS (default "ab,wind"): which player WindBot is (runs/windbot/ygo-windbot), its
//! earlier decisions asked and replaced as well. $BOT_ANALYZE_RECORDS=OUT: the games' records
//! appended to OUT.

use std::io::BufRead;

use ygo_arbiter::actions::{Response, encode};
use ygo_arbiter::game::{Bot, GameRecord, ObsItem, ObservationLog, RecordedResponse, init, init_info, replay, run_until_prompt};
use ygo_arbiter::messages::Msg;
use ygo_arbiter::observe::{filter_prompt, to_core_response};
use ygo_core::constants::HINT_SELECTMSG;
use ygo_core::duel::Duel;
use bot::belief_bots::BeliefSearchBot;
use bot::heuristic::Heuristic;
use bot::search::movegen::root_candidates;
use bot::search::{AlphaBeta, SearchParams};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn usage(msg: &str) -> ! {
    eprintln!("bot-analyze: {msg}");
    eprintln!("usage: bot-analyze RECORDS.jsonl LINE K [--nodes 5000] [--exact] [--line N OUT.jsonl]");
    std::process::exit(2)
}

/// The prompt's candidates as the player sees it, and as the core's responses.
fn candidates(prompt: &Msg, hint: Option<u64>) -> (Vec<Response>, Vec<Response>) {
    let player = prompt.player().unwrap_or(0);
    let mut rules = Heuristic::new(1, player);
    rules.hint = hint;
    let seen = filter_prompt(prompt, player);
    let cands = root_candidates(&seen, &rules, None);
    let core = cands.iter().filter_map(|c| to_core_response(prompt, player, c).ok()).collect();
    (cands, core)
}

/// The search's best of `moves`, the root values and the depth completed.
fn search(duel: &Duel, prompt: &Msg, moves: &[Response], hint: Option<u64>, nodes: u64, exact: bool) -> (usize, Vec<Option<f64>>, i32) {
    let mut ab = AlphaBeta::default();
    ab.prepare();
    let params = SearchParams { budget: None, elapsed: 0.0, clock_remaining: 1000.0, hint, seeds: [1, 2], max_nodes: Some(nodes), depth: None, exact_root: exact };
    let (best, depth) = ab.search(duel, prompt, moves, &params);
    (best, std::mem::take(&mut ab.last_root_values), depth)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.len() < 3 {
        usage("RECORDS.jsonl LINE K");
    }
    let num = |s: &str| -> u64 { s.parse().unwrap_or_else(|_| usage(&format!("not a number: {s}"))) };
    let (path, line, k) = (&argv[0], num(&argv[1]) as usize, num(&argv[2]) as usize);
    let (mut nodes, mut exact, mut cont, mut as_bot): (u64, bool, Option<(usize, String)>, bool) = (5000, false, None, false);
    let mut as_eval = false;
    let (mut playout, mut force): (Option<usize>, Option<Response>) = (None, None);
    let mut i = 3;
    while i < argv.len() {
        match argv[i].as_str() {
            "--nodes" => {
                nodes = num(argv.get(i + 1).unwrap_or_else(|| usage("--nodes N")));
                i += 2;
            }
            "--exact" => {
                exact = true;
                i += 1;
            }
            "--bot" => {
                as_bot = true;
                i += 1;
            }
            "--playout" => {
                playout = Some(num(argv.get(i + 1).unwrap_or_else(|| usage("--playout N"))) as usize);
                i += 2;
            }
            "--force" => {
                let r = argv.get(i + 1).unwrap_or_else(|| usage("--force RESPONSE"));
                force = Some(serde_json::from_str(r).unwrap_or_else(|e| usage(&format!("not a response: {e}"))));
                i += 2;
            }
            "--eval" => {
                as_eval = true;
                i += 1;
            }
            "--line" => {
                let (n, out) = (argv.get(i + 1), argv.get(i + 2));
                let (Some(n), Some(out)) = (n, out) else { usage("--line N OUT.jsonl") };
                cont = Some((num(n) as usize, out.clone()));
                i += 3;
            }
            a => usage(&format!("unknown option {a}")),
        }
    }
    init(None).unwrap();
    let file = std::fs::File::open(path).unwrap_or_else(|e| usage(&format!("{path}: {e}")));
    let text = std::io::BufReader::new(file).lines().nth(line).unwrap_or_else(|| usage("no such line")).unwrap();
    let mut record: GameRecord = serde_json::from_str(&text).unwrap_or_else(|e| usage(&format!("not a record: {e}")));

    if as_bot {
        run_bot(&record, k, nodes);
        return;
    }
    if let Some(n) = playout {
        record.responses.truncate(k);
        if let Some(r) = force {
            let player = prompt_player(&record, k);
            record.responses.push(RecordedResponse { player, response: r });
        }
        run_playouts(&record, n, nodes);
        return;
    }
    let mut duel = record.setup(None).unwrap();
    duel.ocg_start_duel();
    let mut prompt: Option<Msg> = None;
    let mut hints: [Option<u64>; 2] = [None, None];
    let last = k + cont.as_ref().map_or(0, |c| c.0);
    let mut responses: Vec<RecordedResponse> = Vec::new();
    for i in 0..=last {
        let Ok(step) = run_until_prompt(&mut duel, prompt.as_ref()) else { break };
        for m in &step.messages {
            if let Msg::Hint { hint_type, player, data } = m
                && *hint_type == HINT_SELECTMSG
                && *player < 2
            {
                hints[*player as usize] = Some(*data);
            }
        }
        let Some(p) = step.prompt else { break };
        let player = p.player().unwrap_or(0);
        let hint = hints[player as usize].take();
        if as_eval && i == k {
            // (--eval: the static evaluation at decision K for its player, with and without the prompt)
            let with = bot::search::evaluate::evaluate_at(&mut duel, player, Some(&p));
            let without = bot::search::evaluate::evaluate_at(&mut duel, player, None);
            println!("decision {k}: {} of P{player}: eval {with:+.0} (without the prompt {without:+.0})", p.name());
            return;
        }
        let response = if i < k {
            record.responses[i].response.clone()
        } else {
            let (cands, core) = candidates(&p, hint);
            if cands.len() <= 1 || cands.len() != core.len() {
                let mut rules = Heuristic::new(1, player);
                rules.hint = hint;
                let r = if i < record.responses.len() && i == k { record.responses[i].response.clone() } else { rules.respond_to(&filter_prompt(&p, player), None).unwrap() };
                if i == k {
                    println!("decision {k}: {} of P{player}, {} candidate(s): not searched", p.name(), cands.len());
                }
                r
            } else {
                let (best, values, depth) = search(&duel, &p, &core, hint, nodes, exact);
                if i == k {
                    println!("decision {k}: {} of P{player} ({}), depth {depth}, {nodes} nodes{}", p.name(), record.bots.get(player as usize).map_or("", |s| s), if exact { ", exact" } else { "" });
                    let played = record.responses.get(k).map(|r| &r.response);
                    for (j, (c, v)) in cands.iter().zip(&values).enumerate() {
                        let mark = format!("{}{}", if j == best { "*" } else { " " }, if Some(c) == played { "P" } else { " " });
                        println!("  {mark} {:>9} {}", v.map_or("-".into(), |v| format!("{v:+.0}")), serde_json::to_string(c).unwrap());
                    }
                    if played.is_some_and(|r| !cands.contains(r)) {
                        println!("  (played, not a candidate: {})", serde_json::to_string(played.unwrap()).unwrap());
                    }
                } else {
                    eprintln!("line {i}: {} of P{player}: {} candidates, depth {depth}", p.name(), cands.len());
                }
                cands[best].clone()
            }
        };
        if i >= k && cont.is_none() {
            break;
        }
        let core = to_core_response(&p, player, &response).unwrap();
        duel.ocg_set_response(&encode(&p, &core).unwrap());
        responses.push(RecordedResponse { player, response });
        prompt = Some(p);
    }
    if let Some((_, out)) = cont {
        record.responses = responses;
        record.result = Default::default();
        std::fs::write(&out, serde_json::to_string(&record).unwrap() + "\n").unwrap();
        eprintln!("the line: {out}");
    }
}

/// The bot at decision `k` (`--bot`).
fn run_bot(record: &GameRecord, k: usize, nodes: u64) {
    let player = record.responses.get(k).unwrap_or_else(|| usage("no such decision")).player;
    let mut log = ObservationLog::new(player, true);
    replay(record, &mut [&mut log], Some(k + 1), None).unwrap();
    let seed = std::env::var("BOT_ANALYZE_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut bot = BeliefSearchBot::new("alphabeta", Some(seed)).unwrap();
    bot.start(player, &init_info(record, &record.setup(None).unwrap(), player)).unwrap();
    let last = log.items.iter().rposition(|x| matches!(x, ObsItem::Prompt(..))).unwrap();
    for (j, item) in log.items[..=last].iter().enumerate() {
        match item {
            ObsItem::Msg(m) => bot.observe(m),
            ObsItem::Prompt(p, state) if j < last => {
                // (an earlier decision: as played)
                bot.set_node_limit(Some(100));
                let _ = bot.respond(p, state.as_ref(), None);
                if let Some(ObsItem::Response(r)) = log.items.get(j + 1) {
                    bot.set_last_response(r.clone());
                }
            }
            ObsItem::Prompt(p, state) => {
                let failures = bot.failures;
                bot.set_node_limit(Some(nodes));
                let r = bot.respond(p, state.as_ref(), None);
                let info = bot.search_info();
                println!("bot at decision {k} ({} of P{player}): {}", p.name(), r.map_or_else(|e| format!("{e:?}"), |r| serde_json::to_string(&r).unwrap()));
                println!("  played: {}", serde_json::to_string(&record.responses[k].response).unwrap());
                println!("  search: {info:?}; reconstruction failed: {}", bot.failures > failures);
            }
            ObsItem::Response(_) => {}
        }
    }
}

/// The player of decision `k`'s prompt.
fn prompt_player(record: &GameRecord, k: usize) -> u32 {
    let mut duel = record.setup(None).unwrap();
    duel.ocg_start_duel();
    let mut prompt: Option<Msg> = None;
    for i in 0..=k {
        let p = run_until_prompt(&mut duel, prompt.as_ref()).unwrap().prompt.unwrap_or_else(|| usage("no such decision"));
        if i == k {
            return p.player().unwrap_or(0);
        }
        let core = to_core_response(&p, p.player().unwrap_or(0), &record.responses[i].response).unwrap();
        duel.ocg_set_response(&encode(&p, &core).unwrap());
        prompt = Some(p);
    }
    unreachable!()
}

/// A bot that first answers with the given responses (its earlier decisions as played).
struct Prefixed<B: Bot> {
    bot: B,
    played: std::collections::VecDeque<Response>,
    nodes: u64,
    /// (the bot's history: the response it gave replaced by the one played)
    replace: fn(&mut B, Response),
}

impl<B: Bot> Bot for Prefixed<B> {
    fn name(&self) -> String {
        self.bot.name()
    }
    fn wants_state(&self) -> bool {
        true
    }
    fn start(&mut self, player: u32, init: &ygo_arbiter::game::InitInfo) -> Result<(), ygo_arbiter::game::BotError> {
        self.bot.start(player, init)
    }
    fn observe(&mut self, msg: &Msg) {
        self.bot.observe(msg)
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&ygo_arbiter::state::Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, ygo_arbiter::game::BotError> {
        let Some(r) = self.played.pop_front() else {
            self.bot.set_node_limit(Some(self.nodes));
            return self.bot.respond(prompt, state, None);
        };
        self.bot.set_node_limit(Some(100));
        let _ = self.bot.respond(prompt, state, None);
        (self.replace)(&mut self.bot, r.clone());
        Ok(r)
    }
}

/// `--playout`.
fn run_playouts(record: &GameRecord, n: usize, nodes: u64) {
    let played: Vec<Vec<Response>> = (0..2)
        .map(|p| {
            let mut log = ObservationLog::new(p, false);
            replay(record, &mut [&mut log], None, None).unwrap();
            log.items.iter().filter_map(|x| if let ObsItem::Response(r) = x { Some(r.clone()) } else { None }).collect()
        })
        .collect();
    let threads = std::env::var("BOT_ANALYZE_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(8usize);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let windbot = std::env::var("BOT_ANALYZE_WINDBOT").ok();
                let seats = std::env::var("BOT_ANALYZE_SEATS").unwrap_or_else(|_| "ab,wind".into());
                let mut bots: Vec<Box<dyn Bot>> = seats
                    .split(',')
                    .enumerate()
                    .map(|(p, name)| -> Box<dyn Bot> {
                        let played = played[p].iter().cloned().collect();
                        match (name, &windbot) {
                            ("wind", Some(dir)) => {
                                let w = ygo_arbiter::protocol::BotProcess::start(&["runs/windbot/ygo-windbot".into()], "wind", &[("WindBot".into(), dir.clone())]).unwrap();
                                Box::new(Prefixed { bot: w, played, nodes, replace: |_, _| {} })
                            }
                            _ => Box::new(Prefixed { bot: BeliefSearchBot::new("alphabeta", Some(2 * i as i64 + 1 + p as i64)).unwrap(), played, nodes, replace: BeliefSearchBot::set_last_response }),
                        }
                    })
                    .collect();
                let mut game = record.clone();
                game.responses.clear();
                game.labels.clear();
                let mut opts = ygo_arbiter::game::PlayOptions::default();
                let (a, b) = bots.split_at_mut(1);
                ygo_arbiter::game::play(&mut game, &mut [&mut *a[0], &mut *b[0]], &mut [], &mut opts).unwrap();
                if let Ok(out) = std::env::var("BOT_ANALYZE_RECORDS") {
                    use std::io::Write;
                    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(out).unwrap();
                    writeln!(f, "{}", serde_json::to_string(&game).unwrap()).unwrap();
                }
                let diverged = game.responses.iter().zip(&record.responses).position(|(x, y)| x.response != y.response);
                println!("game {i}: winner {} ({}), turns {}, decisions {}{}", game.result.get("winner").map_or("-".into(), |w| w.to_string()), game.result.get("reason").map_or("-".into(), |w| w.to_string()), game.result.get("turns").map_or("-".into(), |w| w.to_string()), game.responses.len(), diverged.map_or(String::new(), |d| format!(", prefix differs at {d}")));
            });
        }
    });
}
