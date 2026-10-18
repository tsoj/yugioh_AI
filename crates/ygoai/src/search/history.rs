//! The history heuristic: a score per move slot (`movepick::move_key`) that cutoffs raise and the
//! moves tried before a cutoff lower.

use super::params::{HIST_BITS, HIST_DECAY};

pub struct HistoryTable {
    table: Vec<f64>,
}

impl Default for HistoryTable {
    fn default() -> Self {
        HistoryTable { table: vec![0.0; 1 << HIST_BITS] }
    }
}

impl HistoryTable {
    /// The score of the move with history slot `key`.
    pub fn get(&self, key: usize) -> f64 {
        self.table[key]
    }

    /// A cutoff at depth `depth` by the move `best`: it gains depth^2, the moves tried before it
    /// (`tried`) lose depth^2 / their count.
    pub fn update(&mut self, best: usize, tried: &[usize], depth: i32) {
        let bonus = (depth * depth) as f64;
        self.table[best] += bonus;
        for &k in tried {
            self.table[k] -= bonus / tried.len() as f64;
        }
    }

    /// (per decision: older searches count less)
    pub fn decay(&mut self) {
        for h in self.table.iter_mut() {
            *h *= HIST_DECAY;
        }
    }
}
