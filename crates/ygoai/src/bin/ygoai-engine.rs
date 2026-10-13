//! An agent as an engine process for an arbiter (match.py, ygoai-match):
//!
//!   ygoai-engine NAME [SEED]
//!
//! NAME: random, greedy, heuristic, or a search on the belief's reconstruction: alphabeta,
//! alphabeta_naive, mcts (options by environment: YGOAI_SEARCH_NODES, YGOAI_MCTS_*, YGOAI_COMPLETION).
//! Game n of the process plays with seed SEED + n (without SEED: unseeded).

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    ygoai::search::panic_messages_once();
    let args: Vec<String> = std::env::args().collect();
    let name = args.get(1).cloned().unwrap_or_else(|| "random".into());
    let seed: Option<i64> = args.get(2).map(|s| s.parse().expect("SEED: an integer"));
    if let Err(e) = ygoai::game::init(None) {
        eprintln!("ygoai-engine: {e:?}");
        std::process::exit(1);
    }
    let mut n = 0;
    let factory = || {
        let s = seed.map(|s| s + n);
        n += 1;
        ygoai::belief_agents::make_agent(&name, s)
    };
    if let Err(e) = ygoai::engine::serve(factory) {
        eprintln!("ygoai-engine {name}: {e}");
        std::process::exit(1);
    }
}
