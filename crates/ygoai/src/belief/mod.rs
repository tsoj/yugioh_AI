//! The belief replay (ygoai/belief): a player's reconstruction of the hidden information from
//! its observation stream, ported module by module.
//!
//! * [`script_index`]: what card scripts are about, read statically
//! * [`tracker`]: card identities through the stream, as vars and pools

pub mod agent;
pub mod check;
pub mod completion;
pub mod flow;
pub mod games;
pub mod guesses;
pub mod infer;
pub mod neural;
pub mod read_guided;
pub mod replay;
pub mod script_index;
pub mod sync;
pub mod tracker;
