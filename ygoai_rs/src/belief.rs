//! The belief replay (crates/ygoai/src/belief) for Python: the consistency check of a stream
//! prefix (belief/check.py) and an agent's belief (belief/agent.py).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::json;

use ygoai::belief::completion::Completion;
use ygoai::belief::tracker::Need;
use ygoai::game::{InitInfo, ObsItem};

fn de<T: serde::de::DeserializeOwned>(v: &Bound<'_, PyAny>) -> PyResult<T> {
    pythonize::depythonize(v).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// An observation stream item from Python: ("msg", msg) / ("prompt", prompt, state) /
/// ("response", response).
pub fn obs_item(item: &Bound<'_, PyAny>) -> PyResult<ObsItem> {
    let kind: String = item.get_item(0)?.extract()?;
    Ok(match kind.as_str() {
        "msg" => ObsItem::Msg(de(&item.get_item(1)?)?),
        "prompt" => {
            let s = item.get_item(2)?;
            ObsItem::Prompt(de(&item.get_item(1)?)?, if s.is_none() { None } else { Some(de(&s)?) })
        }
        "response" => ObsItem::Response(de(&item.get_item(1)?)?),
        _ => return Err(PyValueError::new_err(format!("unknown item {kind}"))),
    })
}

/// A deck completion of Python's (e.g. the neural one, or its model service) for the Rust replay:
/// its `complete` and `stand_ins`, called with the GIL.
struct PyCompletion(Py<PyAny>);

impl Completion for PyCompletion {
    fn complete(&self, known_main: &[u32], known_extra: &[u32], n_main: usize, n_extra: usize, present: &[u32]) -> Result<(Vec<u32>, Vec<u32>), String> {
        Python::attach(|py| {
            let r = self.0.call_method1(py, "complete", (known_main.to_vec(), known_extra.to_vec(), n_main, n_extra, present.to_vec())).map_err(|e| e.to_string())?;
            r.extract::<(Vec<u32>, Vec<u32>)>(py).map_err(|e| e.to_string())
        })
    }

    fn stand_ins(&self, need: Need, known: &[u32]) -> Vec<u32> {
        Python::attach(|py| {
            let need: Py<PyAny> = match need {
                Need::Mset(k) => ("mset", k).into_pyobject(py).map(|t| t.into_any().unbind()),
                n => (n.tag(),).into_pyobject(py).map(|t| t.into_any().unbind()),
            }
            .expect("a tuple");
            self.0.call_method1(py, "stand_ins", (need, known.to_vec())).and_then(|r| r.extract::<Vec<u32>>(py)).unwrap_or_default()
        })
    }
}

/// The completion for the Rust replay: "vanilla" / "corpus" by name (the Rust ones), else a
/// Python completion object (one wrapper per object, kept for the process's life).
fn completion_of(c: &Bound<'_, PyAny>) -> PyResult<Option<&'static dyn Completion>> {
    use std::sync::Mutex;
    if let Ok(name) = c.extract::<String>() {
        return match name.as_str() {
            "vanilla" => Ok(None),
            name => Ok(Some(named_completion(name)?)),
        };
    }
    static WRAPPED: Mutex<Vec<(usize, &'static PyCompletion)>> = Mutex::new(Vec::new());
    let key = c.as_ptr() as usize;
    let mut w = WRAPPED.lock().unwrap();
    if let Some(&(_, pc)) = w.iter().find(|(k, _)| *k == key) {
        return Ok(Some(pc));
    }
    let pc: &'static PyCompletion = Box::leak(Box::new(PyCompletion(c.clone().unbind())));
    w.push((key, pc));
    Ok(Some(pc))
}

fn items_of(items: &Bound<'_, PyAny>) -> PyResult<Vec<ObsItem>> {
    items.try_iter()?.map(|it| obs_item(&it?)).collect()
}

/// The Rust belief replay's check of a stream prefix (check.check_prefix): the CheckResult's
/// fields as a dict. `completion`: "vanilla", "corpus" or a Python completion; `timeout` in seconds.
#[pyfunction]
#[pyo3(signature = (init, items, cut, stopped, completion, timeout=None))]
fn check_prefix<'py>(
    py: Python<'py>,
    init: &Bound<'py, PyAny>,
    items: &Bound<'py, PyAny>,
    cut: usize,
    stopped: bool,
    completion: &Bound<'py, PyAny>,
    timeout: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let init: InitInfo = de(init)?;
    let items = items_of(items)?;
    if cut > items.len() {
        return Err(PyValueError::new_err("cut past the stream"));
    }
    let comp = completion_of(completion)?;
    let timeout = timeout.map(std::time::Duration::from_secs_f64);
    let res = py.detach(|| ygoai::belief::check::check_prefix(&init, &items, cut, stopped, comp, timeout));
    pythonize::pythonize(py, &res).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// An agent's belief state in Rust (belief/agent.py's AgentBelief): `reconstruct(items)` gives
/// (what the reconstruction was, a copy of its duel or None).
#[pyclass(module = "ygoai_rs", unsendable)]
struct AgentBelief {
    b: ygoai::belief::agent::AgentBelief,
}

#[pymethods]
impl AgentBelief {
    #[new]
    fn new(viewer: u32, init: &Bound<'_, PyAny>, completion: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(AgentBelief { b: ygoai::belief::agent::AgentBelief::new(viewer, de(init)?, completion_of(completion)?) })
    }

    #[pyo3(signature = (items, timeout=None))]
    fn reconstruct<'py>(&mut self, py: Python<'py>, items: &Bound<'py, PyAny>, timeout: Option<f64>) -> PyResult<(Bound<'py, PyAny>, Option<crate::RustDuel>)> {
        let items = items_of(items)?;
        // (with the GIL: the replay's state is not Send; an agent's process does one thing at a time)
        let (r, duel) = self.b.reconstruct(&items, timeout.map(std::time::Duration::from_secs_f64));
        let info = json!({
            "approximate": r.approximate, "final_prompt": r.final_prompt, "lenient_skips": r.lenient_skips,
            "restarts": r.restarts,
        });
        let info = pythonize::pythonize(py, &info).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((info, duel.map(|d| crate::RustDuel { d: d.d })))
    }

    #[getter]
    fn retries(&self) -> usize {
        self.b.retries
    }
    #[getter]
    fn extended(&self) -> usize {
        self.b.extended
    }
    #[getter]
    fn revealed(&self) -> usize {
        self.b.revealed
    }
}

fn named_completion(name: &str) -> PyResult<&'static dyn Completion> {
    use ygoai::belief::completion::{CorpusCompletion, VanillaCompletion};
    Ok(match name {
        "vanilla" => VanillaCompletion::get(),
        "corpus" => CorpusCompletion::get().map_err(PyValueError::new_err)?,
        "neural" => ygoai::belief::neural::NeuralCompletion::get().map_err(PyValueError::new_err)?,
        _ => return Err(PyValueError::new_err(format!("unknown completion {name}"))),
    })
}

/// A completion's fills (main, extra) for a deck's unknown cards (Completion::complete), by name.
#[pyfunction]
#[pyo3(signature = (name, known_main, known_extra, n_main, n_extra, present=Vec::new()))]
fn complete(name: &str, known_main: Vec<u32>, known_extra: Vec<u32>, n_main: usize, n_extra: usize, present: Vec<u32>) -> PyResult<(Vec<u32>, Vec<u32>)> {
    named_completion(name)?.complete(&known_main, &known_extra, n_main, n_extra, &present).map_err(PyValueError::new_err)
}

/// A completion's stand-ins for a hidden card of a kind, by name; `need`: ("st",) / ("field",) /
/// ("mset", k) / ("monster",).
#[pyfunction]
#[pyo3(signature = (name, need, known=Vec::new()))]
fn stand_ins(name: &str, need: &Bound<'_, PyAny>, known: Vec<u32>) -> PyResult<Vec<u32>> {
    let tag: String = need.get_item(0)?.extract()?;
    let need = match tag.as_str() {
        "st" => Need::St,
        "field" => Need::Field,
        "monster" => Need::Monster,
        "mset" => Need::Mset(need.get_item(1)?.extract()?),
        t => return Err(PyValueError::new_err(format!("unknown need {t}"))),
    };
    Ok(named_completion(name)?.stand_ins(need, &known))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(complete, m)?)?;
    m.add_function(wrap_pyfunction!(stand_ins, m)?)?;
    m.add_function(wrap_pyfunction!(check_prefix, m)?)?;
    m.add_class::<AgentBelief>()?;
    Ok(())
}
