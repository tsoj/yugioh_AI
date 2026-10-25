//! Statistics of the search for its diagnostics (ygoai-bench-search with $YGOAI_STATS): only
//! collected when `AlphaBeta::stats` is set.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::{AddAssign, Sub};

use super::movepick::{KeyContext, move_identity, move_kind, prompt_label};
use super::tt::NodeType;
use crate::actions::Response;
use crate::messages::Msg;

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
    /// History collisions: move identities and of them those that share their slot with another
    /// identity; their lookups and of them those in a shared slot.
    Collisions { identities, shared_identities, lookups, shared_lookups }
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

/// (for the futility margins) A move after the first at depths 1 to 4 with a finite alpha: alpha
/// minus the node's static eval, and whether the move raised alpha.
pub struct FutilityGap {
    pub kind: String,
    pub depth: i8,
    pub gap: f32,
    pub raised: bool,
}

/// A history lookup's key in the collision counts: slot, move identity, prompt.
type Lookup = (usize, u128, &'static str);

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
    pub gaps: Vec<FutilityGap>,
    /// the history lookups of the current search and game, folded into `hist_search` after each
    /// search and into `hist_game` after each game (`fold_history`)
    hist_now: HashMap<Lookup, u64>,
    hist_game_now: HashMap<Lookup, u64>,
    /// by prompt, within each search and within each game
    pub hist_search: BTreeMap<&'static str, Collisions>,
    pub hist_game: BTreeMap<&'static str, Collisions>,
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
    /// The history lookups of a node's moves (their slots `keys`).
    pub fn history_lookups(&mut self, prompt: &Msg, moves: &[Response], keys: &[usize], ctx: &KeyContext) {
        for (mv, &k) in moves.iter().zip(keys) {
            *self.hist_now.entry((k, move_identity(prompt, mv, ctx), prompt_label(prompt))).or_default() += 1;
        }
    }

    /// A reduced search of the move at index `i`, searched again if it raised alpha.
    pub fn reduction(&mut self, prompt: &Msg, i: usize, researched: bool) {
        let e = self.lmr.entry((prompt_label(prompt), i.min(15) as u8)).or_default();
        *e += Reductions { reduced: 1, researched: researched as u64 };
    }

    pub fn futility_gap(&mut self, prompt: &Msg, mv: &Response, depth: f64, gap: f64, raised: bool) {
        self.gaps.push(FutilityGap { kind: move_kind(prompt, mv), depth: depth.round() as i8, gap: gap as f32, raised });
    }

    /// An expanded node of `prompt`.
    pub fn node(&mut self, prompt: &Msg, s: NodeSummary) {
        let label = prompt_label(prompt);
        *self.widths.entry((label, listed(prompt, s.moves) as u16, s.moves as u16)).or_default() += Width { nodes: 1, searched: s.searched as u64 };
        *self.cuts.entry((label, s.tt, s.moves as u8, s.cut.map(|i| i as u8))).or_default() += 1;
        let depth = s.depth.round().clamp(-1.0, 60.0) as i8;
        *self.subtree.entry((label, depth)).or_default() += Subtree { nodes: 1, subtree_nodes: s.subtree_nodes };
    }

    /// (history collisions) The end of a search, and with `game_ended` of the game.
    pub fn fold_history(&mut self, game_ended: bool) {
        fold_collisions(&self.hist_now, &mut self.hist_search);
        for (k, n) in self.hist_now.drain() {
            *self.hist_game_now.entry(k).or_default() += n;
        }
        if game_ended {
            fold_collisions(&self.hist_game_now, &mut self.hist_game);
            self.hist_game_now.clear();
        }
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

/// History lookups by (slot, identity, prompt) into collisions by prompt.
fn fold_collisions(lookups: &HashMap<Lookup, u64>, out: &mut BTreeMap<&'static str, Collisions>) {
    let mut identities: HashMap<usize, HashSet<u128>> = HashMap::new();
    for &(slot, identity, _) in lookups.keys() {
        identities.entry(slot).or_default().insert(identity);
    }
    for (&(slot, _, label), &n) in lookups {
        let shared = identities[&slot].len() > 1;
        *out.entry(label).or_default() += Collisions { identities: 1, shared_identities: shared as u64, lookups: n, shared_lookups: if shared { n } else { 0 } };
    }
}
