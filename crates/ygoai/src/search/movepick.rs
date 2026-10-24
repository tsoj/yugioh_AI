//! Move ordering: the rules' preference among a decision's moves, and the history slot of a move.

use super::movegen::{candidates, card_sets};
use super::params::HIST_BITS;
use crate::actions::Response;
use crate::heuristic::db_attack;
use crate::messages::Msg;

/// A move's kind for the pruning margins: the prompt and the command kind, a chain's pass or
/// chain, a yes or no.
pub(super) fn move_kind(m: &Msg, c: &Response) -> String {
    let t = m.name();
    if let Some(k) = &c.kind {
        return format!("SELECT_{}:{k}", t.trim_start_matches("SELECT_"));
    }
    if matches!(m, Msg::SelectChain { .. }) {
        return format!("{t}:{}", if c.index.unwrap_or(-1) < 0 { "pass" } else { "chain" });
    }
    if let Some(yes) = c.yes {
        return format!("{t}:{}", if yes { "yes" } else { "no" });
    }
    t.to_string()
}

fn idle_order(kind: &str) -> i32 {
    match kind {
        "activate" => 0,
        "spsummon" => 1,
        "summon" => 2,
        "mset" => 3,
        "repos" => 4,
        "to_bp" => 5,
        "sset" => 6,
        "to_ep" => 7,
        "shuffle" => 8,
        _ => 9,
    }
}

fn battle_order(kind: &str) -> i32 {
    match kind {
        "activate" => 0,
        "attack" => 1,
        "to_m2" => 2,
        "to_ep" => 3,
        _ => 9,
    }
}

/// The candidates of a prompt, the heuristic player's preference first (for a card selection:
/// `card_order`, the rules' order by the select hint) (alphabeta.ordered).
pub fn ordered(m: &Msg, turn_player: u32, card_order: Option<&[usize]>) -> Vec<Response> {
    let mut cands = candidates(m);
    let k = |c: &Response| c.index.unwrap_or(0) as usize;
    match m {
        Msg::SelectCard { min: 1, max: 1, .. } if card_order.is_some() => {
            return card_order.unwrap().iter().filter(|&&i| i < cands.len()).map(|&i| cands[i].clone()).collect();
        }
        Msg::SelectCard { .. } if card_order.is_some() => return card_sets(m, card_order.unwrap()),
        Msg::SelectIdlecmd { summonable, .. } => cands.sort_by_key(|c| {
            let kind = c.kind.as_deref().unwrap_or("");
            (idle_order(kind), if kind == "summon" { -db_attack(summonable[k(c)].code) } else { 0 })
        }),
        Msg::SelectBattlecmd { attackable, .. } => cands.sort_by_key(|c| {
            let kind = c.kind.as_deref().unwrap_or("");
            (battle_order(kind), if kind == "attack" { -db_attack(attackable[k(c)].code) } else { 0 })
        }),
        Msg::SelectChain { .. } => cands.sort_by_key(|c| c.index == Some(-1)),
        Msg::SelectEffectyn { player, .. } | Msg::SelectYesno { player, .. } => {
            cands.sort_by_key(|c| c.yes != Some(*player == turn_player))
        }
        _ => {}
    }
    cands
}

fn type_id(m: &Msg) -> u128 {
    match m {
        Msg::SelectIdlecmd { .. } => 1,
        Msg::SelectBattlecmd { .. } => 2,
        Msg::SelectChain { .. } => 3,
        Msg::SelectEffectyn { .. } => 4,
        Msg::SelectYesno { .. } => 5,
        Msg::SelectOption { .. } => 6,
        Msg::SelectCard { .. } => 7,
        Msg::SelectSum { .. } => 14,
        _ => 0,
    }
}

/// The prompt's name in the width statistics (`AlphaBeta::widths`).
pub fn prompt_label(m: &Msg) -> &'static str {
    match m {
        Msg::SelectIdlecmd { .. } => "idlecmd",
        Msg::SelectBattlecmd { .. } => "battlecmd",
        Msg::SelectChain { .. } => "chain",
        Msg::SelectEffectyn { .. } => "effectyn",
        Msg::SelectYesno { .. } => "yesno",
        Msg::SelectOption { .. } => "option",
        Msg::SelectCard { min: 1, max: 1, .. } => "card1",
        Msg::SelectCard { .. } => "cards",
        Msg::SelectSum { .. } => "sum",
        _ => "other",
    }
}

/// The options a prompt lists before the search's caps: the cards of a selection, else the
/// candidates.
pub(super) fn listed(m: &Msg, cands: usize) -> usize {
    match m {
        Msg::SelectCard { cards, .. } => cards.len(),
        _ => cands,
    }
}

fn kind_id(kind: &str) -> u128 {
    match kind {
        "summon" => 1,
        "spsummon" => 2,
        "repos" => 3,
        "mset" => 4,
        "sset" => 5,
        "activate" => 6,
        "to_bp" => 7,
        "to_ep" => 8,
        "shuffle" => 9,
        "attack" => 10,
        "to_m2" => 11,
        _ => 0,
    }
}

/// The history slot of a move: its player, kind and card (and effect) (alphabeta._move_key).
pub fn move_key(m: &Msg, c: &Response) -> usize {
    slot(move_identity(m, c))
}

/// The history slot of a move identity.
fn slot(identity: u128) -> usize {
    let mut h = identity;
    h ^= h >> 17;
    let h = (h as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> (64 - HIST_BITS)) as usize
}

/// What the history tells apart about a move, before it is hashed to a slot (`move_key`): moves
/// with the same identity share their history by design, others only by a collision.
pub fn move_identity(m: &Msg, c: &Response) -> u128 {
    let (mut kind, mut code, mut desc) = (0u128, 0u128, 0u128);
    let entry = |list: &[crate::messages::Card], i: i64| list.get(i as usize).map(|e| (e.code as u128, e.desc.unwrap_or(0) as u128));
    if let Some(k) = &c.kind {
        kind = kind_id(k);
        let list = match (m, k.as_str()) {
            (Msg::SelectIdlecmd { summonable, .. }, "summon") => Some(summonable),
            (Msg::SelectIdlecmd { spsummonable, .. }, "spsummon") => Some(spsummonable),
            (Msg::SelectIdlecmd { repositionable, .. }, "repos") => Some(repositionable),
            (Msg::SelectIdlecmd { msetable, .. }, "mset") => Some(msetable),
            (Msg::SelectIdlecmd { ssetable, .. }, "sset") => Some(ssetable),
            (Msg::SelectIdlecmd { activatable, .. }, "activate") => Some(activatable),
            (Msg::SelectBattlecmd { activatable, .. }, "activate") => Some(activatable),
            (Msg::SelectBattlecmd { attackable, .. }, "attack") => Some(attackable),
            _ => None,
        };
        if let Some((cd, ds)) = list.and_then(|l| entry(l, c.index.unwrap_or(0))) {
            (code, desc) = (cd, ds);
        }
    } else if let Msg::SelectChain { chains, .. } = m {
        if let Some(i) = c.index.filter(|&i| i >= 0)
            && let Some((cd, ds)) = entry(chains, i)
        {
            (kind, code, desc) = (1, cd, ds);
        }
    } else if let Some(yes) = c.yes {
        kind = 1 + yes as u128;
        match m {
            Msg::SelectEffectyn { code: cd, desc: ds, .. } => (code, desc) = (*cd as u128, *ds as u128),
            Msg::SelectYesno { desc: ds, .. } => desc = *ds as u128,
            _ => {}
        }
    } else if let Some(ind) = &c.indices {
        kind = 1;
        if let Msg::SelectCard { cards, .. } | Msg::SelectSum { cards, .. } = m {
            // (the cards' codes)
            code = ind.iter().map(|&i| cards.get(i as usize).map_or(0, |e| e.code as u128)).fold(0, |h: u128, c| h.wrapping_mul(1000003).wrapping_add(c));
        }
    } else if let Some(i) = c.index {
        kind = (1 + i) as u128;
    }
    let player = m.player().unwrap_or(0) as u128;
    (((type_id(m) * 31 + player) * 31 + kind) * 1000003 + code) * 1000003 + desc
}
