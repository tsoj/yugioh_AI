//! The transposition table: a search result per position (`hash_key`), kept within a game.

use ygo_core::duel::Duel;
use ygo_core::types::CardId;

use super::params::{EFFORT_DECAY, EFFORT_MAX_SLOPE, EFFORT_MIN_DEPTH_VARIANCE, EFFORT_PRIOR_SLOPE, TT_BITS};
use ygo_arbiter::messages::Msg;

/// What a node's value is: exact (a PV node: a move raised alpha without reaching beta), a lower
/// bound (a cut node: a move reached beta) or an upper bound (an all node: no move raised alpha).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
pub enum NodeType {
    #[default]
    PvNode,
    CutNode,
    AllNode,
}

/// A stored search result: its value, depth, node type and best move (an index into the node's
/// ordered moves).
#[derive(Clone, Copy, Default)]
pub struct HashResult {
    key: u64,
    pub value: f64,
    pub depth: f32,
    pub node_type: NodeType,
    pub best_move: i8,
    generation: u16,
}

pub struct TranspositionTable {
    entries: Vec<HashResult>,
    /// (the decision the new entries are of: for the count of the hits on earlier decisions' entries)
    generation: u16,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        TranspositionTable { entries: vec![HashResult::default(); 1 << TT_BITS], generation: 0 }
    }
}

impl TranspositionTable {
    fn slot(key: u64) -> usize {
        (key & ((1 << TT_BITS) - 1)) as usize
    }

    /// The entry of the position, if it has one from this game's searches (a key is never 0).
    pub fn get(&self, key: u64) -> Option<HashResult> {
        let e = self.entries[Self::slot(key)];
        (e.key == key).then_some(e)
    }

    /// Whether the entry is of an earlier decision's search.
    pub fn is_earlier(&self, e: &HashResult) -> bool {
        e.generation != self.generation
    }

    /// Stores a result (always replacing the slot's entry).
    pub fn add(&mut self, key: u64, node_type: NodeType, value: f64, depth: f64, best_move: usize) {
        self.entries[Self::slot(key)] = HashResult { key, value, depth: depth as f32, node_type, best_move: best_move as i8, generation: self.generation };
    }

    /// A new decision: the entries stay.
    pub fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// A new game: no entries.
    pub fn clear(&mut self) {
        self.entries.fill(HashResult::default());
    }
}

/// How much a move's searches took: decayed sums of log2(nodes + 1) and of the depth searched.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Effort {
    weight: f64,
    log_nodes: f64,
    depth: f64,
}

impl Effort {
    /// One more search of the move: the older ones count EFFORT_DECAY as much.
    pub fn add(&mut self, nodes: u64, depth: f64) {
        self.weight = self.weight * EFFORT_DECAY + 1.0;
        self.log_nodes = self.log_nodes * EFFORT_DECAY + ((nodes + 1) as f64).log2();
        self.depth = self.depth * EFFORT_DECAY + depth;
    }

    /// log2(nodes + 1) less what `slope` expects for the depth searched (None: never searched).
    pub fn residual(&self, slope: f64) -> Option<f64> {
        (self.weight > 0.0).then(|| (self.log_nodes - slope * self.depth) / self.weight)
    }
}

/// The fit of log2(nodes + 1) against the depth a move was searched to, within each node (the
/// moves of one search of a position: their sums less the node's mean), over a decision.
#[derive(Default)]
pub struct EffortFit {
    dd: f64,
    dy: f64,
}

impl EffortFit {
    /// The moves of one search of a position: (log2(nodes + 1), depth) each.
    pub fn add(&mut self, samples: &[(f64, f64)]) {
        if samples.len() < 2 {
            return;
        }
        let n = samples.len() as f64;
        let (my, md) = (samples.iter().map(|s| s.0).sum::<f64>() / n, samples.iter().map(|s| s.1).sum::<f64>() / n);
        for &(y, d) in samples {
            self.dd += (d - md) * (d - md);
            self.dy += (d - md) * (y - my);
        }
    }

    /// log2 nodes per ply (the prior while the depths have hardly differed).
    pub fn slope(&self) -> f64 {
        if self.dd < EFFORT_MIN_DEPTH_VARIANCE { EFFORT_PRIOR_SLOPE } else { (self.dy / self.dd).clamp(0.0, EFFORT_MAX_SLOPE) }
    }
}

/// Each move's effort in a position's searches (by the move's index into the node's ordered
/// moves), kept within a decision as the TT. At most as many positions as the TT has entries
/// (later ones are not stored).
#[derive(Default)]
pub struct MoveEffort {
    positions: rustc_hash::FxHashMap<u64, Vec<Effort>>,
}

impl MoveEffort {
    pub fn get(&self, key: u64) -> Option<&[Effort]> {
        self.positions.get(&key).map(Vec::as_slice)
    }

    pub fn set(&mut self, key: u64, efforts: Vec<Effort>) {
        if self.positions.len() < 1 << TT_BITS || self.positions.contains_key(&key) {
            self.positions.insert(key, efforts);
        }
    }

    /// A new decision: the positions so far are removed.
    pub fn clear(&mut self) {
        self.positions.clear();
    }
}

/// `order` by its moves' effort residuals (`Effort::residual`: how much longer they took to refute
/// than expected for the depths they were searched to) and their ranks in it, both standardized,
/// with equal weight. A move never searched gets the mean residual. Equal scores keep their order.
pub fn effort_order(order: &mut [usize], efforts: &[Effort], slope: f64) {
    let res: Vec<Option<f64>> = order.iter().map(|&m| efforts[m].residual(slope)).collect();
    let known: Vec<f64> = res.iter().flatten().copied().collect();
    if order.len() < 2 || known.is_empty() {
        return;
    }
    let mean = known.iter().sum::<f64>() / known.len() as f64;
    let res = standardized(&res.iter().map(|r| r.unwrap_or(mean)).collect::<Vec<_>>());
    let rank = standardized(&(0..order.len()).map(|i| -(i as f64)).collect::<Vec<_>>());
    let mut scored: Vec<(f64, usize)> = order.iter().enumerate().map(|(i, &m)| (res[i] + rank[i], m)).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for (o, (_, m)) in order.iter_mut().zip(scored) {
        *o = m;
    }
}

/// Zero mean and unit standard deviation (all 0 if the values are equal).
fn standardized(v: &[f64]) -> Vec<f64> {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
    v.iter().map(|x| if sd > 1e-12 { (x - mean) / sd } else { 0.0 }).collect()
}

/// The key of a position and its pending decision (the chess engines' Zobrist key): a fast hash
/// over the engine's state (where every card is, its position, status, counters, materials and
/// equip target; what the cards on the field are now, stats included; turn, phase, LP, chain)
/// and the prompt.
pub fn hash_key(d: &mut Duel, m: &Msg) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    d.ext_get_info().hash(&mut h);
    for link in &d.core.current_chain {
        (link.chain_count, link.triggering_effect.map(|e| e.0), link.triggering_controler, link.triggering_location, link.triggering_sequence).hash(&mut h);
    }
    for p in 0..2 {
        let pl = &d.player[p];
        // (the piles: which card where; the zones also what the card is now, stats included)
        let piles: [&Vec<CardId>; 5] = [&pl.list_main, &pl.list_hand, &pl.list_extra, &pl.list_grave, &pl.list_remove];
        for pile in piles {
            pile.len().hash(&mut h);
            for &c in pile {
                card_state(d, c, &mut h);
            }
        }
        let zones: Vec<Option<CardId>> = pl.list_mzone.iter().chain(&pl.list_szone).copied().collect();
        for (i, z) in zones.into_iter().enumerate() {
            i.hash(&mut h);
            if let Some(c) = z {
                card_state(d, c, &mut h);
                (c.get_code(d), c.get_attack(d), c.get_defense(d), c.get_level(d), c.get_rank(d)).hash(&mut h);
            }
        }
    }
    // (the pending prompt, as its JSON text, streamed into the hash)
    struct W<'a>(&'a mut rustc_hash::FxHasher);
    impl std::io::Write for W<'_> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.write(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let _ = serde_json::to_writer(W(&mut h), m);
    let k = h.finish();
    if k == 0 { 1 } else { k }
}

/// What the key takes of a card: which one, its code (a hidden card's guess can change from one
/// decision to the next), its position, status bits, counters, materials and
/// equip target.
fn card_state(d: &Duel, c: CardId, h: &mut rustc_hash::FxHasher) {
    use std::hash::Hash;
    let x = d.card(c);
    (x.cardid, x.data.code, x.current.position, x.status, &x.counters).hash(h);
    x.xyz_materials.iter().map(|m| d.card(*m).cardid).for_each(|id| id.hash(h));
    x.equiping_target().map(|t| d.card(t).cardid).hash(h);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_order_takes_out_the_depth() {
        // (moves 0..4 in order a, c, d, b: a searched deep, d and b reduced, c never searched)
        let (a, b, c, d) = (0, 1, 2, 3);
        let mut efforts = [Effort::default(); 4];
        efforts[a].add(4095, 10.0);
        efforts[b].add(255, 1.0);
        efforts[d].add(1023, 3.0);
        // (by nodes alone, a took longest)
        let mut order = vec![a, c, d, b];
        effort_order(&mut order, &efforts, 0.0);
        assert_eq!(order[0], a);
        // (for its depth, a took the least)
        let mut order = vec![a, c, d, b];
        effort_order(&mut order, &efforts, 1.0);
        assert_eq!(order, vec![c, d, a, b]);
    }

    #[test]
    fn effort_fit_within_nodes() {
        let mut fit = EffortFit::default();
        assert_eq!(fit.slope(), EFFORT_PRIOR_SLOPE);
        // (two nodes of different sizes, each 1.5 log2 nodes more per ply)
        fit.add(&[(10.0, 4.0), (7.0, 2.0)]);
        fit.add(&[(20.0, 4.0), (17.0, 2.0)]);
        assert!((fit.slope() - 1.5).abs() < 1e-12);
    }
}
