//! The Rust belief modules (crates/ygoai/src/belief) for Python, for comparing them with the
//! Python ones (scripts/rs_diff.py).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Value, json};

use ygoai::belief::completion::Completion;
use ygoai::belief::tracker::{Kind, Mark, Need, PoolLoc, Tracker};
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

pub fn need_name(n: &Option<Need>) -> Value {
    match n {
        None => Value::Null,
        Some(Need::Mset(k)) => json!(format!("mset{k}")),
        Some(n) => json!(n.tag()),
    }
}

pub fn mark_name(m: &Mark) -> String {
    match m {
        Mark::Hint(t, v) => format!("h{t}:{v}"),
        Mark::Public(c) => format!("p{c}"),
    }
}

pub fn pool_loc(l: PoolLoc) -> Value {
    match l {
        PoolLoc::Loc(l) => json!(l),
        PoolLoc::Set => json!("set"),
    }
}

/// The tracker's state as JSON, in the form scripts/rs_diff.py makes of the Python tracker's.
pub fn dump(t: &Tracker) -> Value {
    let vars: Vec<Value> = t
        .vars
        .iter()
        .map(|v| {
            let mut marks: Vec<String> = v.marks.iter().map(mark_name).collect();
            marks.sort();
            json!({
                "owner": v.owner, "kind": v.kind.name(), "fixed": v.fixed_code, "code": v.code, "pool": v.pool,
                "faceup": v.faceup, "overlays": v.overlays, "excluded": v.excluded.iter().collect::<Vec<_>>(),
                "need": need_name(&v.need), "marks": marks,
                "allowed": v.allowed.as_ref().map(|a| a.iter().collect::<Vec<_>>()),
                "same_as": v.same_as, "not_same": v.not_same, "prefer": v.prefer, "seen_owner": v.seen_owner,
            })
        })
        .collect();
    let pools: Vec<Value> = t
        .pools
        .iter()
        .map(|p| json!({"player": p.player, "location": pool_loc(p.location), "members": p.members, "claims": p.claims, "zones": p.zones}))
        .collect();
    let _ = Kind::Main;
    json!({
        "vars": vars, "pools": pools, "lists": t.lists, "mzones": t.mzones, "szones": t.szones,
        "observations": t.observations, "searched": t.searched, "exclusions": t.exclusions, "needs": t.needs,
        "contains": t.contains.iter().collect::<Vec<_>>(), "shuffle_events": t.shuffle_events,
        "set_shuffle_events": t.set_shuffle_events, "turn_player": t.turn_player,
    })
}

/// belief/tracker.py's Tracker in Rust.
#[pyclass(module = "ygoai_rs", name = "Tracker", unsendable)]
pub struct PyTracker {
    pub t: Tracker,
}

#[pymethods]
impl PyTracker {
    #[new]
    fn new(viewer: u32, init: &Bound<'_, PyAny>) -> PyResult<Self> {
        let init: InitInfo = de(init)?;
        Ok(PyTracker { t: Tracker::new(viewer, &init) })
    }

    /// Processes a stream item; None, or the TrackerError's text.
    fn process(&mut self, item: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
        let item = obs_item(item)?;
        Ok(self.t.process(&item).err().map(|e| e.0))
    }

    /// The state as a JSON text.
    fn dump(&self) -> String {
        dump(&self.t).to_string()
    }

    /// Tracker.solve with the completion "vanilla" or "corpus": the hypothesis as a JSON text,
    /// or the TrackerError's text (prefixed "error: ").
    fn solve(&mut self, py: Python<'_>, completion: &str) -> PyResult<String> {
        use ygoai::belief::completion::{Completion, CorpusCompletion, VanillaCompletion};
        let comp: &dyn Completion = match completion {
            "vanilla" => VanillaCompletion::get(),
            "corpus" => CorpusCompletion::get().map_err(PyValueError::new_err)?,
            _ => return Err(PyValueError::new_err(format!("unknown completion {completion}"))),
        };
        let t = &mut self.t;
        Ok(match py.detach(|| t.solve(Some(comp))) {
            Err(e) => format!("error: {}", e.0),
            Ok(h) => hypothesis_json(&h).to_string(),
        })
    }
}

pub fn hypothesis_json(h: &ygoai::belief::tracker::Hypothesis) -> Value {
    let mut var_codes: Vec<(usize, Option<u32>)> = h.var_codes.iter().map(|(&k, &v)| (k, v)).collect();
    var_codes.sort();
    let mut observed: Vec<usize> = h.observed.iter().copied().collect();
    observed.sort();
    json!({
        "decks": h.decks.iter().map(|(m, e)| json!({"main": m, "extra": e})).collect::<Vec<_>>(),
        "deck_orders": h.deck_orders, "extra_orders": h.extra_orders,
        "shuffles": h.shuffles.iter().map(|s| json!({"player": s.player, "location": pool_loc(s.location), "codes": s.codes,
                                                   "ranks": s.ranks, "owners": s.owners})).collect::<Vec<_>>(),
        "set_card_shuffles": h.set_card_shuffles,
        "set_card_shuffle_ranks": h.set_card_shuffle_ranks.iter().map(|(r, o)| json!([r, o])).collect::<Vec<_>>(),
        "unknown_atoms": h.unknown_atoms, "var_codes": var_codes, "observed": observed,
        "deck_ranks": h.deck_ranks, "extra_ranks": h.extra_ranks,
    })
}

/// One of belief/infer.py's functions in Rust: `name` without "infer_", the prompt, the upcoming
/// messages and the other arguments by name (up_ext, descs, player, ids, select_msg); for the ones
/// that read the duel, a snapshot of its cards (all_cards_raw) and its shuffle plan. Returns
/// (candidates, certain) with certain None for "likely".
#[pyfunction]
#[pyo3(signature = (name, prompt, up, kwargs, all_cards=None, plan=None))]
fn infer<'py>(
    py: Python<'py>,
    name: &str,
    prompt: &Bound<'py, PyAny>,
    up: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, pyo3::types::PyDict>,
    all_cards: Option<&[u8]>,
    plan: Option<Vec<ygoai::belief::infer::PlanEntry>>,
) -> PyResult<(Bound<'py, PyAny>, Option<bool>)> {
    use ygoai::belief::infer::{self as inf, Certain, LocMap};
    use ygoai::messages::Msg;
    let prompt: Msg = de(prompt)?;
    let up: Vec<Msg> = de(up)?;
    let get = |k: &str| kwargs.get_item(k).ok().flatten().filter(|v| !v.is_none());
    let up_ext: Option<Vec<Msg>> = get("up_ext").map(|v| de(&v)).transpose()?;
    let lm = || {
        let raw = all_cards.unwrap_or(&[]);
        let all: Vec<[u32; 6]> = raw
            .as_chunks::<24>()
            .0
            .iter()
            .map(|c| std::array::from_fn(|i| u32::from_le_bytes(c[4 * i..4 * i + 4].try_into().unwrap())))
            .collect();
        LocMap::new(&all, plan.as_deref().unwrap_or(&[]))
    };
    let ids: Option<Vec<Option<u32>>> = get("ids").map(|v| v.extract()).transpose()?;
    let (cands, certain) = match name {
        "idle" => inf::infer_idle(&prompt, &up, up_ext.as_deref()),
        "battle" => inf::infer_battle(&prompt, &up, up_ext.as_deref()),
        "chain" => inf::infer_chain(&prompt, &up),
        "effectyn" => {
            let descs: Option<Vec<u64>> = get("descs").map(|v| v.extract()).transpose()?;
            inf::infer_effectyn(&prompt, &up, descs.as_deref())
        }
        "option" => inf::infer_option(&prompt, &up, get("player").map(|v| v.extract()).transpose()?.unwrap_or(0)),
        "place" => inf::infer_place(&prompt, &up, up_ext.as_deref()),
        "position" => inf::infer_position(&prompt, &up),
        "announce" => inf::infer_announce(&prompt, &up),
        "cards" => inf::infer_cards(&prompt, &up, &lm(), ids, get("select_msg").map(|v| v.extract()).transpose()?),
        "unselect" => inf::infer_unselect(&prompt, &up, &lm(), ids),
        "counter" => inf::infer_counter(&prompt, &up),
        "rps" => inf::infer_rps(&prompt, &up),
        "sort" => inf::infer_sort(&prompt, &up),
        _ => return Err(PyValueError::new_err(format!("no infer_{name}"))),
    };
    let certain = match certain {
        Certain::Yes => Some(true),
        Certain::No => Some(false),
        Certain::Likely => None,
    };
    Ok((pythonize::pythonize(py, &cands).map_err(|e| PyValueError::new_err(e.to_string()))?, certain))
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
    use ygoai::belief::completion::CorpusCompletion;
    if let Ok(name) = c.extract::<String>() {
        return match name.as_str() {
            "vanilla" => Ok(None),
            "corpus" => Ok(Some(CorpusCompletion::get().map_err(PyValueError::new_err)?)),
            _ => Err(PyValueError::new_err(format!("unknown completion {name}"))),
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(infer, m)?)?;
    m.add_function(wrap_pyfunction!(check_prefix, m)?)?;
    m.add_class::<AgentBelief>()?;
    m.add_class::<PyTracker>()?;
    Ok(())
}
