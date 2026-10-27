//! Values between Python and Rust: by serde (dicts and lists as the JSON forms), and the
//! observation streams' items as tuples ("msg", msg) / ("prompt", prompt, state) / ("response",
//! response).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ygoai::game::ObsItem;

pub fn de<T: serde::de::DeserializeOwned>(v: &Bound<'_, PyAny>) -> PyResult<T> {
    pythonize::depythonize(v).map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn py<'py, T: serde::Serialize>(p: Python<'py>, x: &T) -> PyResult<Bound<'py, PyAny>> {
    pythonize::pythonize(p, x).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// An observation stream item from Python.
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

pub fn items_of(items: &Bound<'_, PyAny>) -> PyResult<Vec<ObsItem>> {
    items.try_iter()?.map(|it| obs_item(&it?)).collect()
}

/// An observation stream item for Python.
pub fn py_item<'py>(p: Python<'py>, item: &ObsItem) -> PyResult<Bound<'py, PyAny>> {
    Ok(match item {
        ObsItem::Msg(m) => ("msg", py(p, m)?).into_pyobject(p)?.into_any(),
        ObsItem::Prompt(m, state) => ("prompt", py(p, m)?, py(p, state)?).into_pyobject(p)?.into_any(),
        ObsItem::Response(r) => ("response", py(p, r)?).into_pyobject(p)?.into_any(),
    })
}

pub fn py_items<'py>(p: Python<'py>, items: &[ObsItem]) -> PyResult<Vec<Bound<'py, PyAny>>> {
    items.iter().map(|it| py_item(p, it)).collect()
}
