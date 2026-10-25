//! The transposition table: a search result per position (`hash_key`), kept within a decision.

use ygo_core::duel::Duel;
use ygo_core::types::CardId;

use super::params::TT_BITS;
use crate::messages::Msg;

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
    /// (the decision the entries are for: entries of earlier ones do not count)
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

    /// The entry of the position, if it has one from this decision's searches.
    pub fn get(&self, key: u64) -> Option<HashResult> {
        let e = self.entries[Self::slot(key)];
        (e.key == key && e.generation == self.generation).then_some(e)
    }

    /// Stores a result (always replacing the slot's entry).
    pub fn add(&mut self, key: u64, node_type: NodeType, value: f64, depth: f64, best_move: usize) {
        self.entries[Self::slot(key)] = HashResult { key, value, depth: depth as f32, node_type, best_move: best_move as i8, generation: self.generation };
    }

    /// A new decision: the entries so far stop counting.
    pub fn new_generation(&mut self) {
        self.generation = ((self.generation as u32 + 1) % 65536) as u16;
        if self.generation == 0 {
            self.generation = 1;
        }
    }
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

/// What the key takes of a card: which one, its position, status bits, counters, materials and
/// equip target.
fn card_state(d: &Duel, c: CardId, h: &mut rustc_hash::FxHasher) {
    use std::hash::Hash;
    let x = d.card(c);
    (x.cardid, x.current.position, x.status, &x.counters).hash(h);
    x.xyz_materials.iter().map(|m| d.card(*m).cardid).for_each(|id| id.hash(h));
    x.equiping_target.map(|t| d.card(t).cardid).hash(h);
}
