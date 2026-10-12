//! The ygoai package in Rust: the parts of the Python package `ygoai` that the arbiter and the
//! engines need, ported module by module and checked against the Python modules
//! (scripts/rs_diff.py).
//!
//! * [`messages`]: parsing of the core's binary messages (ygoai/messages.py)
//! * [`observe`]: what each player may see of them, prompts in canonical order (ygoai/observe.py)
//! * [`actions`]: responses to prompts and their binary form (ygoai/actions.py)
//! * [`state`]: god view snapshots of a duel and each player's view (ygoai/state.py)
//! * [`game`]: setting up, playing and replaying games, game records (ygoai/game.py)
//! * [`agents`], [`heuristic`]: built-in agents (ygoai/agents)
//! * [`search`]: alpha-beta search on a duel (agents/alphabeta.py's search)
//! * [`pyrandom`]: Python's random.Random, for the same games and choices as the Python package
//!
//! Every type serializes (serde) to the dicts of the Python modules, under the same keys.

pub mod actions;
pub mod belief;
pub mod agents;
pub mod game;
pub mod heuristic;
pub mod matchrun;
pub mod messages;
pub mod observe;
pub mod paths;
pub mod protocol;
pub mod pyrandom;
pub mod search;
pub mod state;
