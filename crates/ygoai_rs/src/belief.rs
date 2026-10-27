//! The belief checks (crates/ygoai/src/belief) for Python: their test games, the checks of a
//! game, and one check of a stream prefix with all its details.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ygoai::belief::check::{self, Mode};
use ygoai::belief::completion::{Completion, by_name};
use ygo_arbiter::game::{Deck, GameError, GameRecord};

use crate::convert::{de, items_of, py, py_items};

fn err(e: GameError) -> PyErr {
    PyValueError::new_err(e.0)
}

fn completion(name: &str) -> PyResult<Option<&'static dyn Completion>> {
    by_name(name).map_err(PyValueError::new_err)
}

/// The belief checks' test game `seed` of the decks and agents (belief::games::play_with_streams):
/// (its record, both players' observation streams).
#[pyfunction]
fn play_with_streams<'py>(p: Python<'py>, seed: i64, decks: &Bound<'py, PyAny>, agents: [String; 2]) -> PyResult<(Bound<'py, PyAny>, Vec<Vec<Bound<'py, PyAny>>>)> {
    let decks: [Deck; 2] = de(decks)?;
    let (record, streams) = p.detach(|| ygoai::belief::games::play_with_streams(seed, decks, [&agents[0], &agents[1]])).map_err(err)?;
    let streams = streams.iter().map(|items| py_items(p, items)).collect::<PyResult<_>>()?;
    Ok((py(p, &record)?, streams))
}

/// The checks of test game `seed` of the decks and agents (belief::check::check_game): its
/// records. `cases`: (viewer, frac); `every` > 0: an agent sequence of every `every`-th decision;
/// `agent_mode`: as an agent reconstructs; `timeout`: seconds per reconstruction.
#[pyfunction]
#[pyo3(signature = (seed, decks, agents, cases, completion_name, timeout, every=0, agent_mode=false))]
#[allow(clippy::too_many_arguments)]
fn check_game<'py>(
    p: Python<'py>,
    seed: i64,
    decks: &Bound<'py, PyAny>,
    agents: [String; 2],
    cases: Vec<(u32, f64)>,
    completion_name: &str,
    timeout: f64,
    every: usize,
    agent_mode: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let decks: [Deck; 2] = de(decks)?;
    let completion = completion(completion_name)?;
    let mode = if every > 0 { Mode::Sequence(every) } else if agent_mode { Mode::Agent } else { Mode::Prefix };
    let timeout = std::time::Duration::from_secs_f64(timeout);
    let records = p.detach(|| check::check_game(seed, decks, [&agents[0], &agents[1]], &cases, mode, completion, timeout)).map_err(err)?;
    py(p, &records)
}

/// The prefix length of a cut of a stream (belief::check::cut_point; None: no such cut).
#[pyfunction]
fn cut_point(items: &Bound<'_, PyAny>, frac: f64) -> PyResult<Option<usize>> {
    Ok(check::cut_point(&items_of(items)?, frac))
}

/// The belief replay's check of the prefix `items[..cut]` of `viewer`'s stream of the game
/// `record` (belief::check::check_prefix): the CheckResult's fields as a dict. `timeout` in seconds.
#[pyfunction]
#[pyo3(signature = (record, viewer, items, cut, completion_name, timeout=None))]
fn check_prefix<'py>(
    p: Python<'py>,
    record: &Bound<'py, PyAny>,
    viewer: u32,
    items: &Bound<'py, PyAny>,
    cut: usize,
    completion_name: &str,
    timeout: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let record: GameRecord = de(record)?;
    let items = items_of(items)?;
    if cut > items.len() {
        return Err(PyValueError::new_err("cut past the stream"));
    }
    let init = ygo_arbiter::game::init_info(&record, &record.setup(None).map_err(err)?, viewer);
    let completion = completion(completion_name)?;
    let timeout = timeout.map(std::time::Duration::from_secs_f64);
    let res = p.detach(|| check::check_prefix(&init, &items, cut, check::stopped(&record), completion, timeout));
    py(p, &res)
}

/// A viewer's game for comparing runs (belief::check::game_hash).
#[pyfunction]
fn game_hash(items: &Bound<'_, PyAny>) -> PyResult<u32> {
    Ok(check::game_hash(&items_of(items)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(play_with_streams, m)?)?;
    m.add_function(wrap_pyfunction!(check_game, m)?)?;
    m.add_function(wrap_pyfunction!(cut_point, m)?)?;
    m.add_function(wrap_pyfunction!(check_prefix, m)?)?;
    m.add_function(wrap_pyfunction!(game_hash, m)?)?;
    Ok(())
}
