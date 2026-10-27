//! Helpers for the replay's diagnosis code, which reads messages, prompts and state views the way
//! the Python module reads its dicts: JSON views (`jv`) with dict-like accessors.

use serde::Serialize;
use serde_json::Value;

use ygo_arbiter::messages::{Loc, Msg};

pub static NULL: Value = Value::Null;

/// The JSON view of a message, prompt, state view, ... (the Python module's dict).
pub fn jv<T: Serialize>(x: &T) -> Value {
    serde_json::to_value(x).unwrap_or(Value::Null)
}

/// Dict-like access to a JSON view.
pub trait V {
    /// d[k] / d.get(k): Null when missing
    fn g(&self, k: &str) -> &Value;
    /// d.get(k, 0) as an int (0 for missing or null)
    fn i(&self, k: &str) -> i64;
    /// d.get(k) as an int (None for missing or null)
    fn o(&self, k: &str) -> Option<i64>;
    /// d["type"]
    fn t(&self) -> &str;
    fn has(&self, k: &str) -> bool;
    /// the value as a list (empty for anything else)
    fn l(&self) -> &[Value];
    /// truthiness as Python's
    fn truthy(&self) -> bool;
}

impl V for Value {
    fn g(&self, k: &str) -> &Value {
        self.get(k).unwrap_or(&NULL)
    }
    fn i(&self, k: &str) -> i64 {
        self.o(k).unwrap_or(0)
    }
    fn o(&self, k: &str) -> Option<i64> {
        match self.get(k) {
            Some(Value::Number(n)) => n.as_i64().or_else(|| n.as_u64().map(|u| u as i64)),
            Some(Value::Bool(b)) => Some(*b as i64),
            _ => None,
        }
    }
    fn t(&self) -> &str {
        self.get("type").and_then(Value::as_str).unwrap_or("")
    }
    fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    fn l(&self) -> &[Value] {
        self.as_array().map_or(&[], |v| &v[..])
    }
    fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        }
    }
}

/// A place from a JSON view (a dict with con/loc/seq and maybe pos).
pub fn li_of(v: &Value) -> crate::belief::infer::Li {
    crate::belief::infer::Li { con: v.i("con") as u32, loc: v.i("loc") as u32, seq: v.i("seq") as u32, pos: v.i("pos") as u32 }
}

pub fn li_loc(l: &Loc) -> crate::belief::infer::Li {
    l.into()
}

/// (con, loc, seq) of a JSON place.
pub fn key3(v: &Value) -> (u32, u32, u32) {
    (v.i("con") as u32, v.i("loc") as u32, v.i("seq") as u32)
}

/// Python's repr() of a JSON value (for texts and signatures).
pub fn py_repr(v: &Value) -> String {
    let mut out = String::new();
    rec(v, &mut out);
    fn rec(v: &Value, out: &mut String) {
        match v {
            Value::Null => out.push_str("None"),
            Value::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
            Value::Number(n) => out.push_str(&n.to_string()),
            Value::String(s) => {
                out.push('\'');
                out.push_str(s);
                out.push('\'');
            }
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    rec(x, out);
                }
                out.push(']');
            }
            Value::Object(o) => {
                out.push('{');
                for (i, (k, x)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push('\'');
                    out.push_str(k);
                    out.push_str("': ");
                    rec(x, out);
                }
                out.push('}');
            }
        }
    }
    out
}

/// replay.first_difference: where two views first differ, as text.
pub fn first_difference(a: &Value, b: &Value, path: &str) -> Option<String> {
    let same_type = std::mem::discriminant(a) == std::mem::discriminant(b);
    if !same_type {
        return Some(format!("{path}: {} != {}", py_repr(a), py_repr(b)));
    }
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                match (x.get(k), y.get(k)) {
                    (Some(p), Some(q)) => {
                        if let Some(d) = first_difference(p, q, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                    (p, q) => {
                        return Some(format!(
                            "{path}.{k}: missing on one side ({} vs {})",
                            py_repr(p.unwrap_or(&NULL)),
                            py_repr(q.unwrap_or(&NULL))
                        ));
                    }
                }
            }
            None
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{path}: length {} != {}", x.len(), y.len()));
            }
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                if let Some(d) = first_difference(p, q, &format!("{path}[{i}]")) {
                    return Some(d);
                }
            }
            None
        }
        _ => (a != b).then(|| format!("{path}: {} != {}", py_repr(a), py_repr(b))),
    }
}

/// A message's type name and its view, for the scans over the stream.
pub fn mtype(m: &Msg) -> &'static str {
    m.name()
}
