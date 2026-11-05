//! The bot, for the arbiter ygo-arbiter (ygopro-core-rust/ygo-arbiter/PROTOCOL.md):
//!
//!   bot NAME [SEED]
//!
//! NAME: random, greedy, heuristic, or a search on the belief's reconstruction: alphabeta,
//! alphabeta_naive, mcts (options by environment: BOT_SEARCH_NODES, BOT_MCTS_*, BOT_COMPLETION).
//! Game n of the process plays with seed SEED + n (without SEED: unseeded).
//!
//!   bot bench [DEPTH]
//!
//! The search benchmark (bot::bench) on the first BENCH_OPENINGS of books/bench.jsonl to depth
//! DEPTH (BENCH_DEPTH): the nodes (OpenBench's bench signature) and the nodes per second, in the
//! form OpenBench reads.

use ygo_arbiter::protocol::Identity;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    bot::search::panic_messages_once();
    let args: Vec<String> = std::env::args().collect();
    let name = args.get(1).cloned().unwrap_or_else(|| "random".into());
    if let Err(e) = ygo_arbiter::game::init(None) {
        eprintln!("bot: {e:?}");
        std::process::exit(1);
    }
    if name == "bench" {
        bench(args.get(2).map(|s| s.parse().expect("DEPTH: an integer")));
        return;
    }
    let seed: Option<i64> = args.get(2).map(|s| s.parse().expect("SEED: an integer"));
    let mut n = 0;
    let factory = |_: &serde_json::Map<String, serde_json::Value>| {
        let s = seed.map(|s| s + n);
        n += 1;
        bot::belief_bots::make_bot(&name, s)
    };
    let id = Identity { name: format!("bot {name}"), more: serde_json::Map::new() };
    if let Err(e) = ygo_arbiter::bot::serve(id, Vec::new(), factory) {
        eprintln!("bot {name}: {e}");
        std::process::exit(1);
    }
}

fn bench(depth: Option<i32>) {
    use bot::bench::{BENCH_DEPTH, BENCH_OPENINGS, Budget, Options};
    let book = bot::paths::root().join("books").join("bench.jsonl");
    let report = bot::bench::run(&book, BENCH_OPENINGS, Budget::Depth(depth.unwrap_or(BENCH_DEPTH)), &Options::default()).unwrap_or_else(|e| {
        eprintln!("bot bench: {e}");
        std::process::exit(1);
    });
    println!("Nodes searched  : {}", report.nodes());
    println!("Nodes/second    : {:.0}", report.nodes() as f64 / report.secs());
}
