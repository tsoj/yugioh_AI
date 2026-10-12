//! The Rust belief modules (crates/ygoai/src/belief) for Python, for comparing them with the
//! Python ones (scripts/rs_diff.py).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Value, json};

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
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTracker>()?;
    Ok(())
}
