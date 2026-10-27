//! The bot, for the arbiter ygo-arbiter (ygopro-core-rust/ygo-arbiter/PROTOCOL.md):
//!
//!   bot NAME [SEED]
//!
//! NAME: random, greedy, heuristic, or a search on the belief's reconstruction: alphabeta,
//! alphabeta_naive, mcts (options by environment: BOT_SEARCH_NODES, BOT_MCTS_*, BOT_COMPLETION).
//! Game n of the process plays with seed SEED + n (without SEED: unseeded).

use ygo_arbiter::protocol::Identity;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    bot::search::panic_messages_once();
    let args: Vec<String> = std::env::args().collect();
    let name = args.get(1).cloned().unwrap_or_else(|| "random".into());
    let seed: Option<i64> = args.get(2).map(|s| s.parse().expect("SEED: an integer"));
    if let Err(e) = ygo_arbiter::game::init(None) {
        eprintln!("bot: {e:?}");
        std::process::exit(1);
    }
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
