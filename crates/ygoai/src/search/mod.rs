//! The alpha-beta search (laid out like a chess engine's: Stockfish, Nalwald):
//!
//! - `search`: iterative deepening, the negamax node, quiescence, and making a move
//! - `tt`: the transposition table and the position's hash key
//! - `history`: the history heuristic
//! - `movegen`: the searched decisions and their moves
//! - `movepick`: move ordering
//! - `evaluate`: the static evaluation (hand-written and learned) and its features
//! - `params`: constants and the optional features ($YGOAI_SEARCH)

pub mod evaluate;
pub mod history;
pub mod movegen;
pub mod movepick;
pub mod params;
#[allow(clippy::module_inception)]
pub mod search;
pub mod tt;

pub use evaluate::*;
pub use movegen::{candidates, candidates_in, is_searched};
pub use movepick::{move_key, ordered, prompt_label};
pub use params::*;
pub use search::{AlphaBeta, NO_CUT, NodeStats, SearchParams, Trace, TraceEdge, is_hidden, panic_messages_once};
pub use tt::hash_key;
