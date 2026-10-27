//! The AI: agents and their search, the belief (what a player can deduce of the hidden cards),
//! the learned evaluation's features. The duels, what each player sees of them, the engine
//! protocol and matches are the arbiter's (ygopro-core-rust/ygo-arbiter); `ygoai-engine` runs an
//! agent as an engine for it.
//!
//! * [`agents`], [`heuristic`], [`belief_agents`]: agents
//! * [`search`], [`mcts`]: searches on a duel
//! * [`belief`]: the belief replay and deck completion
//! * [`datarow`], [`rl_policy`]: training data, the learned policy

pub mod belief;
pub mod belief_agents;
pub mod datarow;
pub mod agents;
pub mod heuristic;
pub mod mcts;
pub mod rl_policy;
pub mod paths;
pub mod rduel;
pub mod search;
