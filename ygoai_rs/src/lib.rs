//! Python binding of the ygoai crate for the belief checks (scripts/check_belief.py and its
//! tools): their test games, played and checked in Rust, and replays of recorded games.

use pyo3::prelude::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod belief;
mod convert;
mod games;

#[pymodule]
fn ygoai_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    belief::register(m)?;
    games::register(m)?;
    Ok(())
}
