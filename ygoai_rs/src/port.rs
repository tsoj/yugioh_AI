//! The Rust port of the ygoai package (crates/ygoai) for Python: each function takes the core's
//! raw message and returns what the Python function returns, as Python objects (scripts/rs_diff.py
//! compares them).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;

use ygoai::actions::{self, Response};
use ygoai::messages::{self, Msg};
use ygoai::{observe, state};

use crate::RustDuel;

fn to_py<'py, T: Serialize>(py: Python<'py>, v: &T) -> PyResult<Bound<'py, PyAny>> {
    pythonize::pythonize(py, v).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse(msg: &[u8]) -> PyResult<Msg> {
    messages::parse(msg).map_err(|e| PyValueError::new_err(e.0))
}

fn response(resp: &Bound<'_, PyAny>) -> PyResult<Response> {
    pythonize::depythonize(resp).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// messages.parse
#[pyfunction]
pub fn parse_message<'py>(py: Python<'py>, msg: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &parse(msg)?)
}

/// observe.filter_message(parse(msg), viewer)
#[pyfunction]
pub fn filter_message<'py>(py: Python<'py>, msg: &[u8], viewer: u32) -> PyResult<Option<Bound<'py, PyAny>>> {
    observe::filter_message(&parse(msg)?, viewer).map(|m| to_py(py, &m)).transpose()
}

/// observe.filter_prompt(parse(msg), viewer)
#[pyfunction]
pub fn filter_prompt<'py>(py: Python<'py>, msg: &[u8], viewer: u32) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &observe::filter_prompt(&parse(msg)?, viewer))
}

/// observe.option_order(parse(msg), viewer)
#[pyfunction]
pub fn option_order<'py>(py: Python<'py>, msg: &[u8], viewer: u32) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &observe::option_order(&parse(msg)?, viewer))
}

/// observe.is_hidden_prompt(parse(msg))
#[pyfunction]
pub fn is_hidden_prompt(msg: &[u8]) -> PyResult<bool> {
    Ok(observe::is_hidden_prompt(&parse(msg)?))
}

/// observe.to_core_response(parse(msg), viewer, resp)
#[pyfunction]
pub fn to_core_response<'py>(py: Python<'py>, msg: &[u8], viewer: u32, resp: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let r = observe::to_core_response(&parse(msg)?, viewer, &response(resp)?).map_err(|e| PyValueError::new_err(e.0))?;
    to_py(py, &r)
}

/// actions.encode(parse(msg), resp)
#[pyfunction]
pub fn encode_response<'py>(py: Python<'py>, msg: &[u8], resp: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let raw = actions::encode(&parse(msg)?, &response(resp)?).map_err(|e| PyValueError::new_err(e.0))?;
    Ok(PyBytes::new(py, &raw))
}

/// actions.idle_options / battle_options (by the prompt's type)
#[pyfunction]
pub fn command_options<'py>(py: Python<'py>, msg: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    let m = parse(msg)?;
    let opts = if matches!(m, Msg::SelectIdlecmd { .. }) { actions::idle_options(&m) } else { actions::battle_options(&m) };
    to_py(py, &opts)
}

/// actions.check_sum(parse(msg), indices)
#[pyfunction]
pub fn check_sum(msg: &[u8], indices: Vec<usize>) -> PyResult<bool> {
    Ok(actions::check_sum(&parse(msg)?, &indices))
}

/// actions.is_declarable(card, opcodes) for card = (code, alias, type, race, attribute, setcodes)
#[pyfunction]
pub fn is_declarable(card: (u32, u32, u32, u64, u32, Vec<u16>), opcodes: Vec<u64>) -> bool {
    let (code, alias, type_, race, attribute, setcodes) = card;
    actions::is_declarable(&actions::DeclCard { code, alias, type_, race, attribute, setcodes }, &opcodes)
}

/// Plays the game of a record (game.new_record's JSON) between two built-in Rust agents, each
/// (name, seed); returns the record's JSON with responses and result (game.play).
#[pyfunction]
#[pyo3(signature = (record, agents, max_turns=100, max_decisions=20000))]
pub fn play_game(py: Python<'_>, record: &str, agents: Vec<(String, i64)>, max_turns: u32, max_decisions: usize) -> PyResult<String> {
    use ygoai::agents::RandomAgent;
    use ygoai::game::{self, GameRecord, PlayOptions};
    let mut record: GameRecord = serde_json::from_str(record).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut made: Vec<RandomAgent> = Vec::new();
    for (name, seed) in &agents {
        match name.as_str() {
            "random" => made.push(RandomAgent::new(*seed)),
            _ => return Err(PyValueError::new_err(format!("no built-in Rust agent {name}"))),
        }
    }
    let [a, b] = &mut made[..] else { return Err(PyValueError::new_err("two agents")) };
    let mut opts = PlayOptions { max_turns, max_decisions, ..PlayOptions::default() };
    py.detach(|| game::play(&mut record, &mut [a, b], &mut [], &mut opts)).map_err(|e| PyValueError::new_err(e.0))?;
    serde_json::to_string(&record).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The observation streams of both players of a recorded game (game.ObservationLog over
/// game.replay), as JSON: per player a list of ["msg", msg] / ["prompt", prompt, state] /
/// ["response", response].
#[pyfunction]
#[pyo3(signature = (record, with_state=true))]
pub fn observation_streams(py: Python<'_>, record: &str, with_state: bool) -> PyResult<String> {
    use ygoai::game::{self, GameRecord, ObsItem, ObservationLog};
    let record: GameRecord = serde_json::from_str(record).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (mut l0, mut l1) = (ObservationLog::new(0, with_state), ObservationLog::new(1, with_state));
    py.detach(|| game::replay(&record, &mut [&mut l0, &mut l1], None, None)).map_err(|e| PyValueError::new_err(e.0))?;
    let stream = |l: &ObservationLog| -> Vec<serde_json::Value> {
        l.items
            .iter()
            .map(|it| match it {
                ObsItem::Msg(m) => serde_json::json!(["msg", m]),
                ObsItem::Prompt(m, s) => serde_json::json!(["prompt", m, s]),
                ObsItem::Response(r) => serde_json::json!(["response", r]),
            })
            .collect()
    };
    serde_json::to_string(&[stream(&l0), stream(&l1)]).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// alphabeta's search (agents/alphabeta.py) on a Rust duel, with its tables kept from one
/// decision to the next.
#[pyclass(module = "ygoai_rs", unsendable)]
pub struct AlphaBetaSearch {
    ab: ygoai::search::AlphaBeta,
}

#[pymethods]
impl AlphaBetaSearch {
    #[new]
    fn new() -> Self {
        AlphaBetaSearch { ab: ygoai::search::AlphaBeta::default() }
    }

    /// The tables for a new decision.
    fn prepare(&mut self) {
        self.ab.prepare();
    }

    /// (best candidate index, depth completed, nodes) for the duel's pending prompt `prompt`
    /// (the core's, a dict as messages.parse makes it) and its candidates `cands` (core responses).
    #[pyo3(signature = (duel, prompt, cands, seeds, hint, budget, elapsed, clock_remaining))]
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        py: Python<'_>,
        duel: &RustDuel,
        prompt: &Bound<'_, PyAny>,
        cands: &Bound<'_, PyAny>,
        seeds: [u32; 2],
        hint: Option<u64>,
        budget: Option<f64>,
        elapsed: f64,
        clock_remaining: f64,
    ) -> PyResult<(usize, i32, u64)> {
        let prompt: Msg = pythonize::depythonize(prompt).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let cands: Vec<Response> = pythonize::depythonize(cands).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if cands.is_empty() {
            return Err(PyValueError::new_err("no candidates"));
        }
        let params = ygoai::search::SearchParams { budget, elapsed, clock_remaining, hint, seeds };
        let before = self.ab.nodes;
        let d = &duel.d;
        let ab = &mut self.ab;
        let (best, done) = py
            .detach(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ab.search(d, &prompt, &cands, &params))))
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("the search panicked"))?;
        Ok((best, done, self.ab.nodes - before))
    }
}

#[pymethods]
impl RustDuel {
    /// state.capture(duel)
    fn capture<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &state::capture(&mut self.d))
    }

    /// state.viewer_view(duel, viewer)
    fn viewer_view<'py>(&mut self, py: Python<'py>, viewer: u32) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &state::viewer_view(&mut self.d, viewer))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_message, m)?)?;
    m.add_function(wrap_pyfunction!(filter_message, m)?)?;
    m.add_function(wrap_pyfunction!(filter_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(option_order, m)?)?;
    m.add_function(wrap_pyfunction!(is_hidden_prompt, m)?)?;
    m.add_function(wrap_pyfunction!(to_core_response, m)?)?;
    m.add_function(wrap_pyfunction!(encode_response, m)?)?;
    m.add_function(wrap_pyfunction!(command_options, m)?)?;
    m.add_function(wrap_pyfunction!(check_sum, m)?)?;
    m.add_function(wrap_pyfunction!(is_declarable, m)?)?;
    m.add_function(wrap_pyfunction!(play_game, m)?)?;
    m.add_class::<AlphaBetaSearch>()?;
    m.add_function(wrap_pyfunction!(observation_streams, m)?)?;
    Ok(())
}
