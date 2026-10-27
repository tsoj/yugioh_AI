//! Statistics of the search for its diagnostics (bot-bench-search with $BOT_SEARCH_STATS): only
//! collected when `AlphaBeta::stats` is set.

use std::collections::BTreeMap;
use std::ops::{AddAssign, Sub};

use super::movepick::prompt_label;
use super::tt::NodeType;
use ygo_arbiter::messages::Msg;

/// Declares a struct of named counters that add up and subtract (for totals and differences).
macro_rules! counters {
    ($(#[$doc:meta])* $name:ident { $($(#[$field_doc:meta])* $field:ident),* $(,)? }) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Default, Debug, PartialEq)]
        pub struct $name {
            $($(#[$field_doc])* pub $field: u64,)*
        }

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                $(self.$field += other.$field;)*
            }
        }

        impl Sub for $name {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                $name { $($field: self.$field - other.$field,)* }
            }
        }
    };
}

counters! {
    /// The nodes of a width, and the children searched in them (all, or up to the cutoff).
    Width { nodes, searched }
}

counters! {
    /// Late move reductions: the reduced searches, and those searched again (they raised alpha).
    Reductions { reduced, researched }
}

counters! {
    /// Nodes, and the nodes in their subtrees (themselves included).
    Subtree { nodes, subtree_nodes }
}

counters! {
    /// The main-line horizon extension's nodes: all, those that decided (an exact value or a
    /// cutoff), and of these the ones whose best card is not the rules' first.
    Extension { nodes, decided, not_rules_first }
}

/// What the transposition table had for a node.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum TtEntry {
    Missing,
    WithoutMove,
    WithMove(NodeType),
}

impl TtEntry {
    pub fn label(self) -> &'static str {
        match self {
            TtEntry::Missing => "no entry",
            TtEntry::WithoutMove => "no move",
            TtEntry::WithMove(NodeType::PvNode) => "exact",
            TtEntry::WithMove(NodeType::CutNode) => "lower",
            TtEntry::WithMove(NodeType::AllNode) => "upper",
        }
    }
}

/// Statistics of the expanded nodes (nodes that got to their moves, not cut by the TT).
#[derive(Default)]
pub struct NodeStats {
    /// by (prompt, options listed, children after the caps)
    pub widths: BTreeMap<(&'static str, u16, u16), Width>,
    /// nodes by (prompt, TT entry, children, the move index of the cutoff: None without one)
    pub cuts: BTreeMap<(&'static str, TtEntry, u8, Option<u8>), u64>,
    /// by (prompt, move index)
    pub lmr: BTreeMap<(&'static str, u8), Reductions>,
    /// by (prompt, depth)
    pub subtree: BTreeMap<(&'static str, i8), Subtree>,
}

/// A node's moves and what came of them (`NodeStats::node`).
pub struct NodeSummary {
    pub moves: usize,
    pub searched: usize,
    pub tt: TtEntry,
    pub cut: Option<usize>,
    pub depth: f64,
    pub subtree_nodes: u64,
}

impl NodeStats {
    /// A reduced search of the move at index `i`, searched again if it raised alpha.
    pub fn reduction(&mut self, prompt: &Msg, i: usize, researched: bool) {
        let e = self.lmr.entry((prompt_label(prompt), i.min(15) as u8)).or_default();
        *e += Reductions { reduced: 1, researched: researched as u64 };
    }

    /// An expanded node of `prompt`.
    pub fn node(&mut self, prompt: &Msg, s: NodeSummary) {
        let label = prompt_label(prompt);
        *self.widths.entry((label, listed(prompt, s.moves) as u16, s.moves as u16)).or_default() += Width { nodes: 1, searched: s.searched as u64 };
        *self.cuts.entry((label, s.tt, s.moves as u8, s.cut.map(|i| i as u8))).or_default() += 1;
        let depth = s.depth.round().clamp(-1.0, 60.0) as i8;
        *self.subtree.entry((label, depth)).or_default() += Subtree { nodes: 1, subtree_nodes: s.subtree_nodes };
    }
}

/// The options a prompt lists before the search's caps: the cards of a selection, else the
/// candidates.
fn listed(m: &Msg, candidates: usize) -> usize {
    match m {
        Msg::SelectCard { cards, .. } => cards.len(),
        _ => candidates,
    }
}

