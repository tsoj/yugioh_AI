//! The history heuristic: a score per move slot (`movepick::move_key`) that cutoffs raise and the
//! moves tried before a cutoff lower; with `conthist` also per (previous move, move) pair.

use super::params::{CONT_BITS, HIST_BITS, HIST_DECAY};

pub struct HistoryTable {
    table: Vec<f64>,
    /// (continuation history, `conthist`)
    counter_table: Vec<f64>,
}

impl Default for HistoryTable {
    fn default() -> Self {
        HistoryTable { table: vec![0.0; 1 << HIST_BITS], counter_table: vec![0.0; 1 << CONT_BITS] }
    }
}

/// The continuation history slot of a move after the move `previous` (history slots).
pub fn counter_key(previous: usize, key: usize) -> usize {
    let mut h = ((previous as u32).wrapping_mul(0x9E3779B1)) ^ ((key as u32).wrapping_mul(0x85EBCA77));
    h ^= h >> 15;
    (h.wrapping_mul(0x2C1B3C6D) >> (32 - CONT_BITS)) as usize
}

/// A move's slots in the tables: its history slot and, with `conthist`, its continuation slot.
#[derive(Clone, Copy)]
pub struct HistoryKey {
    pub key: usize,
    pub counter_key: Option<usize>,
}

impl HistoryTable {
    /// The move's score (its history plus, with `conthist`, its continuation history).
    pub fn get(&self, k: HistoryKey) -> f64 {
        self.table[k.key] + k.counter_key.map_or(0.0, |c| self.counter_table[c])
    }

    /// A cutoff at depth `depth` by the move `best`: it gains depth^2, the moves tried before it
    /// (`tried`) lose depth^2 / their count.
    pub fn update(&mut self, best: HistoryKey, tried: &[HistoryKey], depth: i32) {
        let bonus = (depth * depth) as f64;
        self.table[best.key] += bonus;
        for k in tried {
            self.table[k.key] -= bonus / tried.len() as f64;
        }
        if let Some(c) = best.counter_key {
            self.counter_table[c] += bonus;
            for k in tried {
                self.counter_table[k.counter_key.unwrap()] -= bonus / tried.len() as f64;
            }
        }
    }

    /// (per decision: older searches count less)
    pub fn decay(&mut self) {
        for h in self.table.iter_mut().chain(self.counter_table.iter_mut()) {
            *h *= HIST_DECAY;
        }
    }
}
