//! The model file, deck_models.bin (its format: human_decks/export_deck_models.py).

use std::collections::HashMap;

use burn::prelude::*;
use serde::Deserialize;

const MAGIC: &[u8] = b"YGODECK1";

#[derive(Deserialize)]
pub(crate) struct Header {
    pub avg_copies: f64,
    pub existence: ExistenceConfig,
    pub counts: CountsConfig,
    tensors: Vec<Entry>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub checks: Checks,
}

#[derive(Deserialize)]
pub(crate) struct ExistenceConfig {
    pub d_model: usize,
    pub n_trunk_layers: usize,
    pub slots_scale: f32,
    /// The card table's buckets: (cards, their own dimension; 0: none)
    pub buckets: Vec<(usize, usize)>,
    pub features: bool,
}

#[derive(Deserialize)]
pub(crate) struct CountsConfig {
    pub n_trunk_layers: usize,
    pub count_bias: bool,
}

#[derive(Deserialize)]
struct Entry {
    name: String,
    shape: Vec<usize>,
    dtype: String,
}

/// Test cases: inputs and the outputs the export computed.
#[derive(Deserialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct Checks {
    pub existence: Vec<ExistenceCheck>,
    pub counts: Vec<CountsCheck>,
}

#[derive(Deserialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct ExistenceCheck {
    pub cards: Vec<(usize, i64)>,
    pub slots: f32,
    /// The mean logit
    pub mean: f32,
    /// The highest logits: (index, logit)
    pub top: Vec<(usize, f32)>,
}

#[derive(Deserialize)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct CountsCheck {
    pub set: Vec<usize>,
    pub logits: Vec<f32>,
}

enum Values {
    Float(Vec<f32>),
    Int(Vec<u32>),
}

/// The file's tensors by name; each is taken once.
pub(crate) struct Tensors {
    tensors: HashMap<String, (Vec<usize>, Values)>,
}

/// The header and the tensors of a deck_models.bin.
pub(crate) fn decode(bytes: &[u8]) -> Result<(Header, Tensors), String> {
    let raw = zstd::decode_all(bytes).map_err(|e| format!("deck models: {e}"))?;
    let mut rest: &[u8] = &raw;
    if take(&mut rest, MAGIC.len())? != MAGIC {
        return Err("deck models: not a deck_models.bin".into());
    }
    let len = u32::from_le_bytes(take(&mut rest, 4)?.try_into().unwrap()) as usize;
    let header: Header = serde_json::from_slice(take(&mut rest, len)?).map_err(|e| format!("deck models: {e}"))?;
    let mut tensors = HashMap::new();
    for e in &header.tensors {
        let n: usize = e.shape.iter().product();
        let values = match e.dtype.as_str() {
            "f16" => Values::Float(take(&mut rest, 2 * n)?.chunks(2).map(f16).collect()),
            "u8" => Values::Int(take(&mut rest, n)?.iter().map(|&b| b as u32).collect()),
            "u16" => Values::Int(take(&mut rest, 2 * n)?.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as u32).collect()),
            "u32" => Values::Int(take(&mut rest, 4 * n)?.chunks(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect()),
            q if q.starts_with('q') && e.shape.len() == 2 => {
                let ints = take(&mut rest, n)?;
                let scales: Vec<f32> = take(&mut rest, 2 * e.shape[0])?.chunks(2).map(f16).collect();
                Values::Float(ints.iter().enumerate().map(|(i, &q)| q as i8 as f32 * scales[i / e.shape[1]]).collect())
            }
            d => return Err(format!("deck models: {}: unknown dtype {d}", e.name)),
        };
        tensors.insert(e.name.clone(), (e.shape.clone(), values));
    }
    if !rest.is_empty() {
        return Err(format!("deck models: {} bytes after the tensors", rest.len()));
    }
    Ok((header, Tensors { tensors }))
}

fn take<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
    if rest.len() < n {
        return Err("deck models: the file ends early".into());
    }
    let (head, tail) = rest.split_at(n);
    *rest = tail;
    Ok(head)
}

fn f16(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

impl Tensors {
    fn take(&mut self, name: &str) -> Result<(Vec<usize>, Values), String> {
        self.tensors.remove(name).ok_or_else(|| format!("deck models: no tensor {name}"))
    }

    pub fn floats(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        match self.take(name)? {
            (shape, Values::Float(v)) => Ok((shape, v)),
            _ => Err(format!("deck models: {name} is not a float tensor")),
        }
    }

    pub fn ints(&mut self, name: &str) -> Result<(Vec<usize>, Vec<u32>), String> {
        match self.take(name)? {
            (shape, Values::Int(v)) => Ok((shape, v)),
            _ => Err(format!("deck models: {name} is not an integer tensor")),
        }
    }

    pub fn matrix(&mut self, name: &str, device: &Device) -> Result<Tensor<2>, String> {
        match self.floats(name)? {
            (shape, v) if shape.len() == 2 => Ok(Tensor::from_data(TensorData::new(v, [shape[0], shape[1]]), device)),
            (shape, _) => Err(format!("deck models: {name} has the shape {shape:?}, not a matrix's")),
        }
    }

    /// A vector as a row (1, n).
    pub fn row(&mut self, name: &str, device: &Device) -> Result<Tensor<2>, String> {
        match self.floats(name)? {
            (shape, v) if shape.len() == 1 => Ok(Tensor::from_data(TensorData::new(v, [1, shape[0]]), device)),
            (shape, _) => Err(format!("deck models: {name} has the shape {shape:?}, not a vector's")),
        }
    }

    /// Fails if a tensor was not taken: the file is for other models.
    pub fn finish(self) -> Result<(), String> {
        let mut left: Vec<&String> = self.tensors.keys().collect();
        left.sort();
        match left.is_empty() {
            true => Ok(()),
            false => Err(format!("deck models: unused tensors {left:?}")),
        }
    }
}
