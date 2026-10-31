//! The text embedding tables (scripts/policy/embed_tables.py): every card and card string under
//! each of several pseudo-name seeds.

use std::collections::HashMap;

use serde::Deserialize;

const MAGIC: &[u8] = b"YGOPTAB1";

#[derive(Deserialize)]
struct Header {
    dim: usize,
    seeds: Vec<u64>,
    codes: Vec<u32>,
    strings: Vec<(u32, u32)>,
}

pub struct Tables {
    pub dim: usize,
    seeds: usize,
    card_row: HashMap<u32, usize>,
    string_row: HashMap<(u32, u32), usize>,
    /// per seed: the cards' rows, then the strings' (float16)
    values: Vec<half::f16>,
}

impl Tables {
    pub fn load(path: &std::path::Path) -> Result<Tables, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() < 12 || &raw[..8] != MAGIC {
            return Err(format!("{}: not an embed_tables.py file", path.display()));
        }
        let len = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let h: Header = serde_json::from_slice(&raw[12..12 + len]).map_err(|e| format!("{}: {e}", path.display()))?;
        let values: Vec<half::f16> = raw[12 + len..].chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]])).collect();
        if values.len() != h.seeds.len() * (h.codes.len() + h.strings.len()) * h.dim {
            return Err(format!("{}: {} values, not seeds x rows x dim", path.display(), values.len()));
        }
        Ok(Tables {
            dim: h.dim,
            seeds: h.seeds.len(),
            card_row: h.codes.iter().enumerate().map(|(i, &c)| (c, i)).collect(),
            string_row: h.strings.iter().enumerate().map(|(i, &k)| (k, h.codes.len() + i)).collect(),
            values,
        })
    }

    pub fn seeds(&self) -> usize {
        self.seeds
    }

    fn row(&self, seed: usize, row: usize) -> Vec<f32> {
        let per_seed = self.card_row.len() + self.string_row.len();
        let at = (seed * per_seed + row) * self.dim;
        self.values[at..at + self.dim].iter().map(|x| x.to_f32()).collect()
    }

    /// A card's text vector under the names of `seed` (None: not in the tables).
    pub fn card(&self, seed: usize, code: u32) -> Option<Vec<f32>> {
        self.card_row.get(&code).map(|&r| self.row(seed, r))
    }

    /// A card string's (its number, 0-15).
    pub fn string(&self, seed: usize, code: u32, string: u32) -> Option<Vec<f32>> {
        self.string_row.get(&(code, string)).map(|&r| self.row(seed, r))
    }
}
