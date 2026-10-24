//! The history heuristic: a score per move slot (`movepick::move_key`) that cutoffs raise and the
//! moves tried before a cutoff lower. The search keeps three, which add up in the move order: by
//! the move, by the move in its turn and phase, and by the move after the one that led to the node.

use super::params::HIST_BITS;

pub struct HistoryTable {
    table: Vec<f64>,
}

impl Default for HistoryTable {
    fn default() -> Self {
        HistoryTable::new(HIST_BITS)
    }
}

impl HistoryTable {
    /// A table of 2^`bits` slots.
    pub fn new(bits: u32) -> Self {
        HistoryTable { table: vec![0.0; 1 << bits] }
    }

    /// The score of the move with history slot `key`.
    pub fn get(&self, key: usize) -> f64 {
        self.table[key]
    }

    /// A cutoff at depth `depth` by the move `best`: it gains depth^`exponent`, the moves tried
    /// before it (`tried`) lose `malus` times that / their count.
    pub fn update(&mut self, best: usize, tried: &[usize], depth: f64, exponent: f64, malus: f64) {
        let bonus = depth.max(0.0).powf(exponent);
        self.table[best] += bonus;
        for &k in tried {
            self.table[k] -= malus * bonus / tried.len() as f64;
        }
    }

    /// (per decision: older searches count less, by the factor `decay`)
    pub fn decay(&mut self, decay: f64) {
        for h in self.table.iter_mut() {
            *h *= decay;
        }
    }
}
