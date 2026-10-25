//! Move ordering: the rules' preference among a decision's moves, and the history slot of a move.

use super::movegen::{candidates, card_sets};
use super::params::HIST_BITS;
use crate::actions::{Command, Response};
use crate::heuristic::db_attack;
use crate::messages::Msg;

/// A move's kind for the pruning margins: the prompt and the command kind, a chain's pass or
/// chain, a yes or no.
pub(super) fn move_kind(m: &Msg, c: &Response) -> String {
    let t = m.name();
    if let Some(k) = c.kind {
        return format!("{t}:{}", k.name());
    }
    if matches!(m, Msg::SelectChain { .. }) {
        return format!("{t}:{}", if c.index.unwrap_or(-1) < 0 { "pass" } else { "chain" });
    }
    if let Some(yes) = c.yes {
        return format!("{t}:{}", if yes { "yes" } else { "no" });
    }
    t.to_string()
}

/// The rules' preference among an idle command's commands (the lowest first).
fn idle_rank(command: Command) -> i32 {
    match command {
        Command::Activate => 0,
        Command::Spsummon => 1,
        Command::Summon => 2,
        Command::Mset => 3,
        Command::Repos => 4,
        Command::ToBp => 5,
        Command::Sset => 6,
        Command::ToEp => 7,
        Command::Shuffle => 8,
        Command::Attack | Command::ToM2 => 9,
    }
}

/// The same among a battle command's.
fn battle_rank(command: Command) -> i32 {
    match command {
        Command::Activate => 0,
        Command::Attack => 1,
        Command::ToM2 => 2,
        Command::ToEp => 3,
        _ => 9,
    }
}

/// The candidates of a prompt, the heuristic player's preference first (for a card selection:
/// `card_order`, the rules' order by the select hint) (alphabeta.ordered).
pub fn ordered(m: &Msg, turn_player: u32, card_order: Option<&[usize]>) -> Vec<Response> {
    let mut cands = candidates(m);
    let index = |c: &Response| c.index.unwrap_or(0) as usize;
    match m {
        Msg::SelectCard { min: 1, max: 1, .. } if card_order.is_some() => {
            return card_order.unwrap().iter().filter(|&&i| i < cands.len()).map(|&i| cands[i].clone()).collect();
        }
        Msg::SelectCard { .. } if card_order.is_some() => return card_sets(m, card_order.unwrap()),
        // (the summons and attacks by the card's ATK, the highest first)
        Msg::SelectIdlecmd { summonable, .. } => cands.sort_by_key(|c| {
            let command = c.kind.unwrap();
            (idle_rank(command), if command == Command::Summon { -db_attack(summonable[index(c)].code) } else { 0 })
        }),
        Msg::SelectBattlecmd { attackable, .. } => cands.sort_by_key(|c| {
            let command = c.kind.unwrap();
            (battle_rank(command), if command == Command::Attack { -db_attack(attackable[index(c)].code) } else { 0 })
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

/// (`move_identity`)
fn command_id(command: Command) -> u128 {
    match command {
        Command::Summon => 1,
        Command::Spsummon => 2,
        Command::Repos => 3,
        Command::Mset => 4,
        Command::Sset => 5,
        Command::Activate => 6,
        Command::ToBp => 7,
        Command::ToEp => 8,
        Command::Shuffle => 9,
        Command::Attack => 10,
        Command::ToM2 => 11,
    }
}

/// What the history keys a move by besides the move itself: the select hint before a card
/// selection (what it is for), and the chain a response answers (its last link's card and whether
/// the mover controls it; 0: no chain).
#[derive(Clone, Copy, Default)]
pub struct KeyContext {
    pub hint: Option<u64>,
    pub chain_top: u128,
}

/// The history slot of a move: its player, kind and card (and effect) (alphabeta._move_key).
pub fn move_key(m: &Msg, c: &Response, ctx: &KeyContext) -> usize {
    slot(move_identity(m, c, ctx))
}

/// The history slot of a move identity (all its 128 bits mixed: splitmix64's finalizer).
fn slot(identity: u128) -> usize {
    let mut h = (identity as u64) ^ ((identity >> 64) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h >> (64 - HIST_BITS)) as usize
}

/// What the history tells apart about a move, before it is hashed to a slot (`move_key`): moves
/// with the same identity share their history by design, others only by a collision. A command's
/// or chain link's card, effect and location; a pass by the chain it lets resolve; a yes or no by
/// its effect; an option by its effect; a card selection by its hint and the cards chosen (code,
/// whose, where).
pub fn move_identity(m: &Msg, c: &Response, ctx: &KeyContext) -> u128 {
    let (mut kind, mut code, mut desc) = (0u128, 0u128, 0u128);
    let me = m.player().unwrap_or(0);
    let entry = |list: &[crate::messages::Card], i: i64| {
        list.get(i as usize).map(|e| (e.code as u128, (e.desc.unwrap_or(0) as u128) * 64 + e.loc as u128))
    };
    if let Some(command) = c.kind {
        kind = command_id(command);
        if let Some((cd, ds)) = command.cards(m).and_then(|l| entry(l, c.index.unwrap_or(0))) {
            (code, desc) = (cd, ds);
        }
    } else if let Msg::SelectChain { chains, .. } = m {
        match c.index.filter(|&i| i >= 0).and_then(|i| entry(chains, i)) {
            Some((cd, ds)) => (kind, code, desc) = (1, cd, ds),
            // (a pass: by what it lets resolve)
            None => (kind, code) = (2, ctx.chain_top),
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
            // (the cards: code, whose and where; what the selection is for)
            code = ind
                .iter()
                .map(|&i| cards.get(i as usize).map_or(0, |e| (e.code as u128 * 2 + (e.con == me) as u128) * 64 + e.loc as u128))
                .fold(0, |h: u128, c| h.wrapping_mul(1000003).wrapping_add(c));
            desc = ctx.hint.map_or(0, |h| h as u128 + 1);
        }
    } else if let Some(i) = c.index {
        kind = 1;
        // (an option: its effect)
        match m {
            Msg::SelectOption { options, .. } => desc = options.get(i as usize).map_or(i as u128, |&d| d as u128),
            _ => kind = (1 + i) as u128,
        }
    }
    let player = me as u128;
    ((((type_id(m) * 31 + player) * 31 + kind) * 1000003).wrapping_add(code).wrapping_mul(1000003)).wrapping_add(desc)
}
