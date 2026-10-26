//! The alpha-beta search (laid out like a chess engine's: Stockfish, Nalwald):
//!
//! - `search`: iterative deepening, the negamax node, and making a move
//! - `tt`: the transposition table and the position's hash key
//! - `history`: the history heuristic
//! - `movegen`: the searched decisions and their moves
//! - `movepick`: move ordering
//! - `cardpolicy`: the order of the cards of a card selection (a trained network)
//! - `evaluate`: the static evaluation (hand-written and learned) and its features
//! - `params`: constants
//! - `stats`: statistics of the search for its diagnostics

pub mod cardpolicy;
pub mod evaluate;
pub mod history;
pub mod movegen;
pub mod movepick;
pub mod params;
pub mod placepolicy;
#[allow(clippy::module_inception)]
pub mod search;
pub mod stats;
pub mod tt;

pub use evaluate::*;
pub use movegen::{candidates, is_searched, root_candidates};
pub use movepick::{KeyContext, history_slot, move_id, ordered, prompt_label};
pub use params::*;
pub use search::{AlphaBeta, SearchParams, panic_messages_once};
pub use stats::NodeStats;
pub use tt::hash_key;
