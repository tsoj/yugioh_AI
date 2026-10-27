//! The bot: its strategies and their search, the belief (what a player can deduce of the hidden
//! cards), the learned evaluation's features. The duels, what each player sees of them, the bot
//! protocol and matches are the arbiter's (ygopro-core-rust/ygo-arbiter); the binary `bot` plays
//! under it.
//!
//! * [`bots`], [`heuristic`], [`belief_bots`]: the strategies
//! * [`search`], [`mcts`]: searches on a duel
//! * [`belief`]: the belief replay and deck completion
//! * [`datarow`], [`rl_policy`]: training data, the learned policy

pub mod belief;
pub mod belief_bots;
pub mod datarow;
pub mod bots;
pub mod heuristic;
pub mod mcts;
pub mod rl_policy;
pub mod paths;
pub mod rduel;
pub mod search;
