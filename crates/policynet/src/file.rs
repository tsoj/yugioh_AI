//! The model file (its format: scripts/policy/export_policy.py).

use std::collections::HashMap;

use burn::prelude::*;
use serde::Deserialize;

use crate::Sizes;
use crate::cols::{OPT, TOK};

const MAGIC: &[u8] = b"YGOPOL01";

#[derive(Deserialize)]
pub(crate) struct Header {
    args: Args,
    consts: Consts,
    tensors: Vec<Entry>,
    pub checks: Vec<Check>,
}

#[derive(Deserialize)]
struct Args {
    dim: usize,
    layers: usize,
    heads: usize,
    strings: String,
}

#[derive(Deserialize)]
struct Consts {
    n_prompts: usize,
    sys_buckets: usize,
    max_tokens: usize,
    max_options: usize,
    locs: usize,
    kinds: usize,
    emb_dim: usize,
}

impl Header {
    pub fn sizes(&self) -> Sizes {
        let (a, c) = (&self.args, &self.consts);
        Sizes {
            dim: a.dim,
            layers: a.layers,
            heads: a.heads,
            emb_dim: c.emb_dim,
            n_prompts: c.n_prompts,
            sys_buckets: c.sys_buckets,
            max_tokens: c.max_tokens,
            max_options: c.max_options,
            locs: c.locs,
            kinds: c.kinds,
            strings: !a.strings.is_empty(),
        }
    }
}

#[derive(Deserialize)]
struct Entry {
    name: String,
    shape: Vec<usize>,
}

/// A decision of the training data and PyTorch's scores of its options.
#[derive(Deserialize)]
pub struct Check {
    hdr: Vec<i64>,
    tok: Vec<Vec<i64>>,
    opt: Vec<Vec<i64>>,
    texts: HashMap<String, Vec<Option<Vec<f32>>>>,
    pub scores: Vec<f32>,
}

impl Check {
    pub fn inputs(&self) -> crate::Inputs<'_> {
        let text = |k: &str| -> Vec<Option<&[f32]>> { self.texts.get(k).map_or(Vec::new(), |v| v.iter().map(|t| t.as_deref()).collect()) };
        let (glob, glob_str) = (text("glob_rows"), text("glob_str_rows"));
        let n_opt = self.opt.len();
        crate::Inputs {
            hdr: self.hdr.clone().try_into().expect("a header row"),
            tok: self.tok.iter().map(|r| <[i64; TOK]>::try_from(r.clone()).expect("a token row")).collect(),
            opt: self.opt.iter().map(|r| <[i64; OPT]>::try_from(r.clone()).expect("an option row")).collect(),
            tok_text: text("tok_row"),
            ref_text: text("ref_row"),
            opt_desc_text: text("opt_desc_row"),
            opt_str_text: if self.texts.contains_key("opt_str_row") { text("opt_str_row") } else { vec![None; n_opt] },
            glob_text: [glob[0], glob[1], glob[2], glob[3]],
            glob_str_text: if glob_str.len() == 2 { [glob_str[0], glob_str[1]] } else { [None, None] },
        }
    }
}

/// The file's tensors by name; each is taken once.
pub(crate) struct Tensors {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

/// The header and the tensors of a model file.
pub(crate) fn decode(bytes: &[u8]) -> Result<(Header, Tensors), String> {
    let raw = zstd::decode_all(bytes).map_err(|e| format!("policy net: {e}"))?;
    let mut rest: &[u8] = &raw;
    if take(&mut rest, MAGIC.len())? != MAGIC {
        return Err("policy net: not an export_policy.py file".into());
    }
    let len = u32::from_le_bytes(take(&mut rest, 4)?.try_into().unwrap()) as usize;
    let header: Header = serde_json::from_slice(take(&mut rest, len)?).map_err(|e| format!("policy net: {e}"))?;
    let mut tensors = HashMap::new();
    for e in &header.tensors {
        let n: usize = e.shape.iter().product();
        let values = take(&mut rest, 2 * n)?.chunks(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
        tensors.insert(e.name.clone(), (e.shape.clone(), values));
    }
    if !rest.is_empty() {
        return Err(format!("policy net: {} bytes after the tensors", rest.len()));
    }
    Ok((header, Tensors { tensors }))
}

fn take<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
    if rest.len() < n {
        return Err("policy net: the file ends early".into());
    }
    let (head, tail) = rest.split_at(n);
    *rest = tail;
    Ok(head)
}

impl Tensors {
    pub fn floats(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        self.tensors.remove(name).ok_or_else(|| format!("policy net: no tensor {name}"))
    }

    pub fn matrix(&mut self, name: &str, device: &Device) -> Result<Tensor<2>, String> {
        match self.floats(name)? {
            (shape, v) if shape.len() == 2 => Ok(Tensor::from_data(TensorData::new(v, [shape[0], shape[1]]), device)),
            (shape, _) => Err(format!("policy net: {name} has the shape {shape:?}, not a matrix's")),
        }
    }

    /// A vector as a row (1, n).
    pub fn row(&mut self, name: &str, device: &Device) -> Result<Tensor<2>, String> {
        match self.floats(name)? {
            (shape, v) if shape.len() == 1 => Ok(Tensor::from_data(TensorData::new(v, [1, shape[0]]), device)),
            (shape, _) => Err(format!("policy net: {name} has the shape {shape:?}, not a vector's")),
        }
    }

    /// Fails if a tensor was not taken: the file is for another model.
    pub fn finish(self) -> Result<(), String> {
        let mut left: Vec<&String> = self.tensors.keys().collect();
        left.sort();
        match left.is_empty() {
            true => Ok(()),
            false => Err(format!("policy net: unused tensors {left:?}")),
        }
    }
}
