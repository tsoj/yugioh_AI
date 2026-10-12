//! The alpha-beta search (crates/ygoai/src/search.rs) for Python's AlphaBetaAgent.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ygoai::actions::Response;
use ygoai::messages::Msg;

use crate::RustDuel;

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
    #[pyo3(signature = (duel, prompt, cands, seeds, hint, budget, elapsed, clock_remaining, max_nodes=None))]
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
        max_nodes: Option<u64>,
    ) -> PyResult<(usize, i32, u64)> {
        let prompt: Msg = pythonize::depythonize(prompt).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let cands: Vec<Response> = pythonize::depythonize(cands).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if cands.is_empty() {
            return Err(PyValueError::new_err("no candidates"));
        }
        let params = ygoai::search::SearchParams { budget, elapsed, clock_remaining, hint, seeds, max_nodes };
        let before = self.ab.nodes;
        let d = &duel.d;
        let ab = &mut self.ab;
        let (best, done) = py
            .detach(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ab.search(d, &prompt, &cands, &params))))
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("the search panicked"))?;
        Ok((best, done, self.ab.nodes - before))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AlphaBetaSearch>()?;
    Ok(())
}
