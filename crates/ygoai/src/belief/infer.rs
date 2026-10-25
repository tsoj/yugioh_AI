//! Inference of the opponent's hidden decisions from the upcoming observations
//! (belief/infer.py).
//!
//! During a belief replay the opponent's prompts are not part of the viewer's stream; their
//! answers have to be deduced from what the viewer observed next. Each function returns
//! (candidates, certain): candidate responses ranked by plausibility, and whether the first one
//! is known to be right. (`certain` is a tri-state where the Python function returns None.)

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use ygo_core::constants::*;

use crate::actions::{Command, Response, available_places, battle_options, check_sum, idle_options};
use crate::agents::{random_response, random_sum_selection};
use crate::messages::{Card, Loc, Msg};
use crate::pyrandom::PyRandom;

pub const LIST_LOCS: [u32; 5] = [LOCATION_DECK, LOCATION_HAND, LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA];

/// A place: con, loc, seq, pos (pos 0 where a message has none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Li {
    pub con: u32,
    pub loc: u32,
    pub seq: u32,
    pub pos: u32,
}

impl From<&Loc> for Li {
    fn from(l: &Loc) -> Li {
        Li { con: l.con, loc: l.loc, seq: l.seq, pos: l.pos }
    }
}

impl From<&Card> for Li {
    fn from(c: &Card) -> Li {
        Li { con: c.con, loc: c.loc, seq: c.seq, pos: c.pos.unwrap_or(0) }
    }
}

/// The place of a message that names one with top-level con/loc/seq (and pos).
pub fn msg_li(m: &Msg) -> Option<Li> {
    use Msg::*;
    Some(match m {
        Set { con, loc, seq, pos, .. }
        | Summoning { con, loc, seq, pos, .. }
        | Spsummoning { con, loc, seq, pos, .. }
        | Flipsummoning { con, loc, seq, pos, .. }
        | Chaining { con, loc, seq, pos, .. }
        | PosChange { con, loc, seq, pos, .. }
        | MissedEffect { con, loc, seq, pos, .. }
        | CardHint { con, loc, seq, pos, .. }
        | SelectEffectyn { con, loc, seq, pos, .. } => Li { con: *con, loc: *loc, seq: *seq, pos: *pos },
        AddCounter { con, loc, seq, .. } | RemoveCounter { con, loc, seq, .. } => Li { con: *con, loc: *loc, seq: *seq, pos: 0 },
        _ => return None,
    })
}

pub fn same_loc(a: &Li, b: &Li) -> bool {
    a.con == b.con && a.loc == b.loc && a.seq == b.seq
}

fn hint(m: &Msg) -> Option<(u32, u32, u64)> {
    if let Msg::Hint { hint_type, player, data } = m { Some((*hint_type, *player, *data)) } else { None }
}

fn is_hint(m: &Msg, ty: u32) -> bool {
    hint(m).is_some_and(|h| h.0 == ty)
}

fn is_activation_prelude(m: &Msg) -> bool {
    matches!(m, Msg::Move { .. } | Msg::PosChange { .. } | Msg::Hint { .. } | Msg::CardHint { .. } | Msg::PayLpcost { .. })
}

/// What `certain` is: known right, not known, or (where Python returns None) "likely".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Certain {
    Yes,
    No,
    Likely,
}

impl From<bool> for Certain {
    fn from(b: bool) -> Certain {
        if b { Certain::Yes } else { Certain::No }
    }
}

pub type Inferred = (Vec<Response>, Certain);

fn indices(v: Vec<usize>) -> Response {
    Response { indices: Some(v.into_iter().map(|i| i as i64).collect()), ..Response::default() }
}

/// Whether the messages before CHAINING `ch` belong to that activation.
pub fn prelude_fits(before: &[Msg], ch: &Msg) -> bool {
    let Msg::Chaining { trigger_con, .. } = ch else { return false };
    let chl = msg_li(ch).unwrap();
    let mut replaced = false; // (a field spell left for the activated one)
    for m in before {
        match m {
            Msg::PayLpcost { player, .. } if player == trigger_con => continue,
            Msg::PosChange { prev_pos, pos, .. } if replaced && prev_pos & POS_FACEUP != 0 && pos & POS_FACEUP != 0 => continue,
            _ => {}
        }
        if is_hint(m, HINT_EVENT) {
            return false; // a timing event starts a new chain window: a later one
        }
        if matches!(m, Msg::Hint { .. } | Msg::CardHint { .. }) {
            continue;
        }
        if let Msg::PosChange { prev_pos, pos, .. } = m
            && same_loc(&msg_li(m).unwrap(), &chl)
            && prev_pos & POS_FACEDOWN != 0
            && pos & POS_FACEUP != 0
        {
            continue;
        }
        if let Msg::Move { from, to, .. } = m
            && ((same_loc(&to.into(), &chl) && to.loc == LOCATION_SZONE) || (from.loc == LOCATION_SZONE && from.seq == 5))
        {
            replaced = replaced || (from.loc == LOCATION_SZONE && from.seq == 5);
            continue;
        }
        return false;
    }
    true
}

/// Location the card at `dest` came from, undoing shifts caused by the earlier moves in `before`.
pub fn source_of(dest: &Li, before: &[Msg]) -> Option<Li> {
    let idx = (0..before.len()).rev().find(|&i| matches!(&before[i], Msg::Move { to, .. } if same_loc(&to.into(), dest)))?;
    let Msg::Move { from, .. } = &before[idx] else { unreachable!() };
    let from_li: Li = from.into();
    if before[..idx].iter().any(|m| matches!(m, Msg::Move { to, .. } if same_loc(&to.into(), &from_li))) {
        // the card got there in the same span: where it was before that
        if let Some(e) = source_of(&from_li, &before[..idx]) {
            return Some(e);
        }
    }
    let mut src = from_li;
    if LIST_LOCS.contains(&src.loc) {
        // undo the shifts of earlier moves out of / into the same pile
        for m in before[..idx].iter().rev() {
            let Msg::Move { from: f, to: t, .. } = m else { continue };
            if t.con == src.con && t.loc == src.loc && t.seq <= src.seq {
                src.seq = src.seq.wrapping_sub(1);
            }
            if f.con == src.con && f.loc == src.loc && f.seq <= src.seq {
                src.seq = src.seq.wrapping_add(1);
            }
        }
    }
    Some(src)
}

/// Indices of activation options that produce CHAINING `ch`.
pub fn match_chaining(options: &[Card], ch: &Msg, before: &[Msg]) -> Vec<usize> {
    let Msg::Chaining { code, desc, .. } = ch else { return vec![] };
    let cands: Vec<usize> = (0..options.len()).filter(|&i| options[i].code == *code && options[i].desc.unwrap_or(0) == *desc).collect();
    if cands.len() <= 1 {
        return cands;
    }
    let chl = msg_li(ch).unwrap();
    if let Some(src) = source_of(&chl, before) {
        let from_src: Vec<usize> = cands.iter().copied().filter(|&i| same_loc(&(&options[i]).into(), &src)).collect();
        if !from_src.is_empty() {
            return [from_src.clone(), cands.iter().copied().filter(|i| !from_src.contains(i)).collect()].concat();
        }
    }
    let exact: Vec<usize> = cands.iter().copied().filter(|&i| same_loc(&(&options[i]).into(), &chl)).collect();
    if exact.len() == 1 {
        return exact;
    }
    if !exact.is_empty() {
        return [exact.clone(), cands.iter().copied().filter(|i| !exact.contains(i)).collect()].concat();
    }
    cands
}

/// timing events of the window the non-turn player gets when the turn player leaves the Main
/// Phase (23) or the Battle Step (29)
pub const PHASE_LEAVE_EVENTS: [u64; 2] = [23, 29];

/// Candidates if `m` is the window opened by a phase-change command.
fn phase_leave(prompt: &Msg, m: &Msg, up_ext: &[Msg], moves: &[(u32, Command)]) -> Option<Inferred> {
    let (ht, _, data) = hint(m)?;
    if ht != HINT_EVENT || !PHASE_LEAVE_EVENTS.contains(&data) {
        return None;
    }
    let mut commands: Vec<Command> = moves.iter().map(|m| m.1).filter(|c| c.offered(prompt)).collect();
    if commands.is_empty() {
        return None;
    }
    // the phase entered afterwards
    let target = up_ext.iter().find_map(|x| if let Msg::NewPhase { phase } = x { Some(moves.iter().find(|m| m.0 == *phase).map(|m| m.1)) } else { None }).flatten();
    if let Some(t) = target
        && let Some(p) = commands.iter().position(|&c| c == t)
    {
        commands.remove(p);
        commands.insert(0, t);
    }
    Some((commands.into_iter().map(|c| Response::command(c, None)).collect(), Certain::Yes))
}

/// The cards of the prompt that `command` takes one of (none if the prompt has no such list).
fn cards_of(prompt: &Msg, command: Command) -> &[Card] {
    command.cards(prompt).unwrap_or(&[])
}

pub fn infer_idle(prompt: &Msg, up: &[Msg], up_ext: Option<&[Msg]>) -> Inferred {
    let opts = idle_options(prompt);
    let ext = up_ext.unwrap_or(&[]);
    for (i, m) in up.iter().enumerate() {
        let before = &up[..i];
        if let Some(l) = phase_leave(prompt, m, ext, &[(PHASE_BATTLE_START, Command::ToBp), (PHASE_END, Command::ToEp)]) {
            return l;
        }
        match m {
            Msg::Chaining { .. } => {
                let ix = match_chaining(cards_of(prompt, Command::Activate), m, before);
                if !ix.is_empty() {
                    let c = ix.len() == 1;
                    return (ix.into_iter().map(|j| Response::command(Command::Activate, Some(j))).collect(), c.into());
                }
                break;
            }
            Msg::Summoning { code, loc, .. } | Msg::Spsummoning { code, loc, .. } | Msg::Set { code, loc, .. } | Msg::Flipsummoning { code, loc, .. } => {
                let command = match m {
                    Msg::Summoning { .. } => Command::Summon,
                    Msg::Spsummoning { .. } => Command::Spsummon,
                    Msg::Flipsummoning { .. } => Command::Repos,
                    _ if *loc == LOCATION_MZONE => Command::Mset,
                    _ => Command::Sset,
                };
                let ml = msg_li(m).unwrap();
                let mut src = if matches!(m, Msg::Flipsummoning { .. }) { Some(ml) } else { source_of(&ml, before) };
                if let Some(s) = src
                    && before.iter().any(|x| match x {
                        Msg::ShuffleHand { player, .. } | Msg::ShuffleDeck { player } | Msg::ShuffleExtra { player, .. } => *player == s.con,
                        _ => false,
                    })
                {
                    src = None; // (the pile was shuffled since the prompt: its positions are others)
                }
                let entries = cards_of(prompt, command);
                let mut cands: Vec<usize> = vec![];
                if let Some(s) = src {
                    cands = (0..entries.len()).filter(|&j| same_loc(&(&entries[j]).into(), &s)).collect();
                }
                if cands.is_empty() && *code != 0 {
                    cands = (0..entries.len()).filter(|&j| entries[j].code == *code).collect();
                }
                if !cands.is_empty() {
                    let c = cands.len() == 1;
                    return (cands.into_iter().map(|j| Response::command(command, Some(j))).collect(), c.into());
                }
                break;
            }
            _ => {}
        }
        if let Msg::CardSelected { cards } = m
            && before.iter().all(|x| matches!(x, Msg::Hint { .. } | Msg::CardHint { .. }))
        {
            // a Pendulum Summon announces the Pendulum Scales first
            let sp = cards_of(prompt, Command::Spsummon);
            let cands: Vec<usize> = (0..sp.len()).filter(|&j| sp[j].loc == LOCATION_SZONE && cards.iter().any(|c| same_loc(&(&sp[j]).into(), &c.into()))).collect();
            if !cands.is_empty() {
                let c = cands.len() == 1;
                return (cands.into_iter().map(|j| Response::command(Command::Spsummon, Some(j))).collect(), c.into());
            }
        }
        if let Msg::PosChange { prev_pos, pos, .. } = m
            && prev_pos & POS_FACEUP != 0
            && pos & POS_FACEUP != 0
            && before.iter().any(|x| matches!(x, Msg::Move { from, .. } if from.loc == LOCATION_SZONE && from.seq == 5))
        {
            continue; // (forced once a field spell left)
        }
        if let Msg::PosChange { loc: LOCATION_MZONE, .. } = m {
            let rep = cards_of(prompt, Command::Repos);
            let ml = msg_li(m).unwrap();
            let cands: Vec<usize> = (0..rep.len()).filter(|&j| same_loc(&(&rep[j]).into(), &ml)).collect();
            if !cands.is_empty() {
                let c = cands.len() == 1;
                return (cands.into_iter().map(|j| Response::command(Command::Repos, Some(j))).collect(), c.into());
            }
            break;
        }
        if let Msg::NewPhase { phase } = m {
            if *phase == PHASE_BATTLE_START && Command::ToBp.offered(prompt) {
                return (vec![Response::command(Command::ToBp, None)], Certain::Yes);
            }
            if Command::ToEp.offered(prompt) {
                return (vec![Response::command(Command::ToEp, None)], Certain::Yes);
            }
            break;
        }
        if let Msg::ShuffleHand { player, .. } = m
            && Command::Shuffle.offered(prompt)
            && Some(*player) == prompt.player()
            && i == 0
        {
            // the shuffle command shuffles right away
            let mut c = vec![Response::command(Command::Shuffle, None)];
            c.extend(opts.iter().filter(|o| o.kind != Some(Command::Shuffle)).cloned());
            return (c, Certain::Yes);
        }
        if matches!(
            m,
            Msg::Move { .. }
                | Msg::Hint { .. }
                | Msg::CardHint { .. }
                | Msg::PosChange { .. }
                | Msg::PayLpcost { .. }
                | Msg::ConfirmCards { .. }
                | Msg::ShuffleHand { .. }
                | Msg::ShuffleDeck { .. }
                | Msg::Lpupdate { .. }
                | Msg::Damage { .. }
                | Msg::Recover { .. }
                | Msg::BecomeTarget { .. }
                | Msg::CardSelected { .. }
                | Msg::RandomSelected { .. }
                | Msg::RemoveCounter { .. }
                | Msg::AddCounter { .. }
                | Msg::FieldDisabled { .. }
                | Msg::TossCoin { .. }
                | Msg::TossDice { .. }
                | Msg::PlayerHint { .. }
                | Msg::ChainSolving { .. }
                | Msg::Draw { .. }
        ) {
            continue;
        }
        break;
    }
    let phase = next_phase(ext);
    if phase == Some(PHASE_BATTLE_START) && Command::ToBp.offered(prompt) {
        return (vec![Response::command(Command::ToBp, None)], Certain::Yes);
    }
    if matches!(phase, Some(PHASE_END) | Some(PHASE_MAIN2)) && Command::ToEp.offered(prompt) {
        return (vec![Response::command(Command::ToEp, None)], Certain::Yes);
    }
    (opts, Certain::No)
}

/// The next phase change, looking past the viewer's chain-window prompts.
pub fn next_phase(up_ext: &[Msg]) -> Option<u32> {
    for m in up_ext {
        if let Msg::NewPhase { phase } = m {
            return Some(*phase);
        }
        if !matches!(m, Msg::Hint { .. } | Msg::CardHint { .. } | Msg::PlayerHint { .. }) {
            return None;
        }
    }
    None
}

pub fn infer_battle(prompt: &Msg, up: &[Msg], up_ext: Option<&[Msg]>) -> Inferred {
    let opts = battle_options(prompt);
    let ext = up_ext.unwrap_or(&[]);
    for (i, m) in up.iter().enumerate() {
        if let Some(l) = phase_leave(prompt, m, ext, &[(PHASE_MAIN2, Command::ToM2), (PHASE_END, Command::ToEp)]) {
            return l;
        }
        match m {
            Msg::Chaining { .. } => {
                let ix = match_chaining(cards_of(prompt, Command::Activate), m, &up[..i]);
                if !ix.is_empty() {
                    let c = ix.len() == 1;
                    return (ix.into_iter().map(|j| Response::command(Command::Activate, Some(j))).collect(), c.into());
                }
                break;
            }
            Msg::Attack { attacker, .. } => {
                let att = cards_of(prompt, Command::Attack);
                let cands: Vec<usize> = (0..att.len()).filter(|&j| same_loc(&(&att[j]).into(), &attacker.into())).collect();
                if !cands.is_empty() {
                    let c = cands.len() == 1;
                    return (cands.into_iter().map(|j| Response::command(Command::Attack, Some(j))).collect(), c.into());
                }
                break;
            }
            Msg::NewPhase { phase } => {
                if *phase == PHASE_MAIN2 && Command::ToM2.offered(prompt) {
                    return (vec![Response::command(Command::ToM2, None)], Certain::Yes);
                }
                if Command::ToEp.offered(prompt) {
                    return (vec![Response::command(Command::ToEp, None)], Certain::Yes);
                }
                break;
            }
            _ if is_activation_prelude(m) => continue,
            _ => break,
        }
    }
    let phase = next_phase(ext);
    if phase == Some(PHASE_MAIN2) && Command::ToM2.offered(prompt) {
        return (vec![Response::command(Command::ToM2, None)], Certain::Yes);
    }
    if phase == Some(PHASE_END) && Command::ToEp.offered(prompt) {
        return (vec![Response::command(Command::ToEp, None)], Certain::Yes);
    }
    (opts, Certain::No)
}

pub fn infer_chain(prompt: &Msg, up: &[Msg]) -> Inferred {
    let Msg::SelectChain { chains, forced, .. } = prompt else { return (vec![], Certain::No) };
    let n = chains.len();
    let pass = Response::pass();
    if let Some(first @ Msg::CardHint { hint_type, .. }) = up.first()
        && *hint_type != CHINT_DESC_REMOVE
    {
        // an option that is no chain link acts right away, e.g. updating a turn counter on its card
        let fl = msg_li(first).unwrap();
        let ch = up.iter().find(|m| matches!(m, Msg::Chaining { .. }));
        let own: Vec<usize> = (0..n).filter(|&j| same_loc(&(&chains[j]).into(), &fl)).collect();
        if !own.is_empty() && ch.is_none_or(|c| !same_loc(&msg_li(c).unwrap(), &fl)) {
            if *forced == 0 {
                let mut c: Vec<Response> = own.iter().map(|&j| Response::index(j)).collect();
                c.push(pass.clone());
                return (c, Certain::No);
            }
            let c = own.len() == 1;
            return (own.into_iter().map(Response::index).collect(), c.into());
        }
    }
    if *forced != 0 {
        // An option that is no chain link (e.g. Imperial Order's maintenance cost) acts before the
        // effect that chains next
        let ch = up.iter().find(|m| matches!(m, Msg::Chaining { .. }));
        let head: Vec<&Msg> = up.iter().take_while(|m| !matches!(m, Msg::Chaining { .. })).collect();
        let named: HashSet<u64> = head
            .iter()
            .take_while(|m| matches!(m, Msg::Hint { .. } | Msg::CardHint { .. }))
            .filter_map(|m| hint(m).filter(|h| h.0 == HINT_CARD).map(|h| h.2))
            .collect();
        let others: Vec<usize> = (0..n).filter(|&j| ch.is_none_or(|c| !same_loc(&(&chains[j]).into(), &msg_li(c).unwrap()))).collect();
        let hinted: Vec<usize> = others.iter().copied().filter(|&j| named.contains(&(chains[j].code as u64))).collect();
        if !hinted.is_empty() && ch.is_some() {
            let c = hinted.len() == 1;
            return (hinted.into_iter().map(Response::index).collect(), c.into());
        }
        let first = head.iter().find(|m| !matches!(m, Msg::Hint { .. } | Msg::CardHint { .. }));
        if !others.is_empty() && ch.is_some() && first.is_some_and(|f| matches!(f, Msg::PayLpcost { .. })) {
            let rest: Vec<usize> = (0..n).filter(|j| !others.contains(j)).collect();
            return ([others, rest].concat().into_iter().map(Response::index).collect(), Certain::No);
        }
    }
    for (i, m) in up.iter().enumerate() {
        if let Msg::Chaining { .. } = m {
            if !prelude_fits(&up[..i], m) {
                break;
            }
            let ix = match_chaining(chains, m, &up[..i]);
            if !ix.is_empty() {
                let c = ix.len() == 1;
                let mut out: Vec<Response> = ix.into_iter().map(Response::index).collect();
                if *forced == 0 {
                    out.push(pass.clone());
                }
                return (out, c.into());
            }
            break;
        }
        if is_activation_prelude(m) {
            continue;
        }
        break;
    }
    if *forced != 0 {
        return ((0..n).map(Response::index).collect(), (n == 1).into());
    }
    let mut out = vec![pass];
    out.extend((0..n).map(Response::index));
    (out, Certain::Yes)
}

/// `descs`: descriptions of the effects the prompt may be about (if known).
pub fn infer_effectyn(prompt: &Msg, up: &[Msg], descs: Option<&[u64]>) -> Inferred {
    let pcode = match prompt {
        Msg::SelectEffectyn { code, .. } => *code,
        _ => 0,
    };
    for (i, m) in up.iter().enumerate() {
        if let Msg::Chaining { code, desc, .. } = m {
            if *code == pcode && prelude_fits(&up[..i], m) {
                if let Some(d) = descs.filter(|d| !d.is_empty())
                    && !d.contains(desc)
                {
                    // another effect of the same card is activated next
                    return (vec![Response::yes(false), Response::yes(true)], Certain::Yes);
                }
                return (vec![Response::yes(true), Response::yes(false)], (*desc != 0).into());
            }
            break;
        }
        if is_activation_prelude(m) {
            continue;
        }
        break;
    }
    (vec![Response::yes(false), Response::yes(true)], Certain::No)
}

pub fn infer_option(prompt: &Msg, up: &[Msg], _player: u32) -> Inferred {
    let Msg::SelectOption { options: opts, .. } = prompt else { return (vec![], Certain::No) };
    let head = &up[..up.len().min(12)];
    for m in head {
        if let Some((HINT_OPSELECTED, _, data)) = hint(m)
            && let Some(i) = opts.iter().position(|&o| o == data)
        {
            let mut c = vec![Response::index(i)];
            c.extend((0..opts.len()).filter(|&j| j != i).map(Response::index));
            return (c, Certain::Yes);
        }
    }
    for m in head {
        if let Some((HINT_CARD, _, data)) = hint(m)
            && data != 0
        {
            // the card whose effect is applied names the option that comes from it
            let named: Vec<usize> = (0..opts.len()).filter(|&j| opts[j] >> 20 == data).collect();
            if named.len() == 1 {
                let i = named[0];
                let mut c = vec![Response::index(i)];
                c.extend((0..opts.len()).filter(|&j| j != i).map(Response::index));
                return (c, Certain::No);
            }
        }
    }
    ((0..opts.len()).map(Response::index).collect(), (opts.len() == 1).into())
}

pub fn infer_hint_value(up: &[Msg], hint_type: u32) -> Option<u64> {
    up.iter().take(12).find_map(|m| hint(m).filter(|h| h.0 == hint_type).map(|h| h.2))
}

fn places(v: Vec<[u32; 3]>) -> Response {
    Response { places: Some(v), ..Response::default() }
}

fn zone_places(player: u32, count: u32, zone: u32) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for bit in 0..32 {
        if zone & (1 << bit) == 0 {
            continue;
        }
        let p = if bit < 16 { player } else { 1 - player };
        let b = bit % 16;
        let (loc, seq) = if b < 8 { (LOCATION_MZONE, b) } else { (LOCATION_SZONE, b - 8) };
        out.push([p, loc, seq]);
    }
    out.truncate(count as usize);
    out
}

fn field_move_follows(msgs: &[Msg]) -> bool {
    for m in msgs {
        if matches!(m, Msg::ChainSolved { .. } | Msg::ChainEnd {} | Msg::NewPhase { .. } | Msg::NewTurn { .. }) || is_hint(m, HINT_EVENT) {
            return false;
        }
        if matches!(m, Msg::Move { to, .. } if to.loc == LOCATION_MZONE || to.loc == LOCATION_SZONE) || matches!(m, Msg::Swap { .. }) {
            return true;
        }
    }
    true
}

/// itertools.combinations(items, k), lexicographic.
pub fn combinations<T: Clone>(items: &[T], k: usize) -> impl Iterator<Item = Vec<T>> + '_ {
    let n = items.len();
    let mut idx: Vec<usize> = (0..k).collect();
    let mut done = k > n;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        let out: Vec<T> = idx.iter().map(|&i| items[i].clone()).collect();
        // next combination
        let mut i = k;
        loop {
            if i == 0 {
                done = true;
                break;
            }
            i -= 1;
            if idx[i] != i + n - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
        }
        Some(out)
    })
}

pub fn infer_place(prompt: &Msg, up: &[Msg], up_ext: Option<&[Msg]>) -> Inferred {
    let (player, count, pflag, is_place) = match prompt {
        Msg::SelectPlace { player, count, flag } => (*player, *count, *flag, true),
        Msg::SelectDisfield { player, count, flag } => (*player, *count, *flag, false),
        _ => return (vec![], Certain::No),
    };
    let avail_list = available_places(player, pflag);
    let avail: BTreeSet<[u32; 3]> = avail_list.iter().copied().collect();
    let mut chosen_places: Vec<[u32; 3]> = Vec::new();
    if is_place && !field_move_follows(up_ext.unwrap_or(up)) {
        // a card is put there before the action is over: no such move, no such prompt
        return (vec![], Certain::No);
    }
    for (i, m) in up.iter().enumerate() {
        if !is_place
            && chosen_places.is_empty()
            && i < 12
            && let Some((HINT_ZONE, hp, data)) = hint(m)
        {
            // the script announces the selected zones
            let mut zone = data as u32;
            if hp != player {
                zone = ((zone & 0xffff) << 16) | ((zone >> 16) & 0xffff);
            }
            return (vec![places(zone_places(player, count, zone))], Certain::Yes);
        }
        if !is_place
            && chosen_places.is_empty()
            && let Msg::FieldDisabled { flag: f } = m
        {
            // the zones disabled now (absolute), among those the prompt offered
            let mut zone = *f;
            if player == 1 {
                zone = ((zone & 0xffff) << 16) | ((zone >> 16) & 0xffff);
            }
            let chosen: Vec<[u32; 3]> = zone_places(player, 32, zone).into_iter().filter(|p| avail.contains(p)).collect();
            if chosen.len() == count as usize {
                return (vec![places(chosen)], Certain::Yes);
            }
            if chosen.len() > count as usize {
                return (combinations(&chosen, count as usize).take(20).map(places).collect(), Certain::No);
            }
        }
        if let Msg::Swap { loc1, loc2, .. } = m {
            // (a control swap: each card takes the other's zone)
            for l in [loc1, loc2] {
                let key = [l.con, l.loc, l.seq];
                if avail.contains(&key) && !chosen_places.contains(&key) {
                    chosen_places.push(key);
                    if chosen_places.len() == count as usize {
                        return (vec![places(chosen_places)], Certain::Yes);
                    }
                }
            }
        }
        if let Msg::Move { to, .. } = m {
            let key = [to.con, to.loc, to.seq];
            if (to.loc == LOCATION_MZONE || to.loc == LOCATION_SZONE) && avail.contains(&key) && !chosen_places.contains(&key) {
                chosen_places.push(key);
                if chosen_places.len() == count as usize {
                    return (vec![places(chosen_places)], Certain::Yes);
                }
            }
        }
    }
    // fallback: every single place (only sensible for count == 1)
    (avail.iter().take(40).map(|&p| places(vec![p])).collect(), Certain::No)
}

pub fn infer_position(prompt: &Msg, up: &[Msg]) -> Inferred {
    let Msg::SelectPosition { code: pcode, positions, .. } = prompt else { return (vec![], Certain::No) };
    let allowed: Vec<u32> = [POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE].into_iter().filter(|p| positions & p != 0).collect();
    let pos_r = |p: u32| Response { position: Some(p as i64), ..Response::default() };
    for m in up {
        let mut pos = None;
        if let Msg::Move { to, code, from, .. } = m
            && to.loc == LOCATION_MZONE
            && *code != 0
            && *pcode != 0
            && code != pcode
            && from.loc != LOCATION_MZONE
        {
            // another card is put onto the field next: this card is not being placed
            return (vec![], Certain::No);
        }
        match m {
            Msg::Move { to, code, .. } if to.loc == LOCATION_MZONE && (*code == 0 || *pcode == 0 || code == pcode) => pos = Some(to.pos),
            Msg::PosChange { loc: LOCATION_MZONE, code, pos: p, .. } if *code == 0 || *pcode == 0 || code == pcode => pos = Some(*p),
            _ => {}
        }
        if let Some(p) = pos {
            if allowed.contains(&p) {
                let mut c = vec![pos_r(p)];
                c.extend(allowed.iter().filter(|&&q| q != p).map(|&q| pos_r(q)));
                return (c, Certain::Yes);
            }
            break;
        }
    }
    let c = allowed.len() == 1;
    (allowed.into_iter().map(pos_r).collect(), c.into())
}

pub fn infer_announce(prompt: &Msg, up: &[Msg]) -> Inferred {
    match prompt {
        Msg::AnnounceCard { .. } => match infer_hint_value(up, HINT_CODE) {
            Some(c) => (vec![Response { code: Some(c as i64), ..Response::default() }], Certain::Yes),
            None => (vec![], Certain::No),
        },
        Msg::AnnounceNumber { values, .. } => {
            if let Some(v) = infer_hint_value(up, HINT_NUMBER)
                && let Some(i) = values.iter().position(|&x| x == v)
            {
                return (vec![Response::index(i)], Certain::Yes);
            }
            ((0..values.len()).map(Response::index).collect(), Certain::No)
        }
        Msg::AnnounceRace { .. } | Msg::AnnounceAttrib { .. } => {
            let ht = if matches!(prompt, Msg::AnnounceRace { .. }) { HINT_RACE } else { HINT_ATTRIB };
            match infer_hint_value(up, ht) {
                Some(v) => (vec![Response { mask: Some(v), ..Response::default() }], Certain::Yes),
                None => (vec![], Certain::No),
            }
        }
        _ => (vec![], Certain::No),
    }
}

/// A forced shuffle result the replay knows: (player, location, card ids bottom first).
pub type PlanEntry = (u32, u32, Option<Vec<Option<u32>>>);

/// Follows sim card ids through upcoming messages to resolve references (infer.LocMap), from a
/// snapshot of where every card is (ext_get_all_cards: player, location, sequence, material
/// index + 1, code, card id).
#[derive(Debug, Clone, Default)]
pub struct LocMap {
    pub lists: HashMap<(u32, u32), Vec<Option<u32>>>,
    /// zones: card ids, 0 for an empty zone
    pub zones: HashMap<(u32, u32), Vec<u32>>,
    pub overlays: HashMap<u32, Vec<Option<u32>>>,
    pub unknown: HashSet<(u32, u32)>,
    pub codes: HashMap<u32, u32>,
    pub plan: VecDeque<PlanEntry>,
    /// cards moved with REASON_RELEASE by the last `references`, in order
    pub released: Vec<u32>,
}

/// A card id as Python sees it: truthy if present and not 0.
fn truthy(c: Option<u32>) -> bool {
    c.is_some_and(|c| c != 0)
}

impl LocMap {
    pub fn new(all_cards: &[[u32; 6]], plan: &[PlanEntry]) -> LocMap {
        let mut lm = LocMap { plan: plan.iter().cloned().collect(), ..LocMap::default() };
        let mut mats: HashMap<(u32, u32, u32), Vec<(u32, u32)>> = HashMap::new();
        for &[p, loc, seq, mat, code, cid] in all_cards {
            if mat != 0 {
                mats.entry((p, loc, seq)).or_default().push((code, cid));
            } else if LIST_LOCS.contains(&loc) {
                lm.lists.entry((p, loc)).or_default().push(Some(cid));
                if cid != 0 {
                    lm.codes.insert(cid, code);
                }
            } else if loc == LOCATION_MZONE || loc == LOCATION_SZONE {
                lm.zones.entry((p, loc)).or_default().push(cid);
                if cid != 0 {
                    lm.codes.insert(cid, code);
                }
            }
        }
        for p in 0..2 {
            for loc in LIST_LOCS {
                lm.lists.entry((p, loc)).or_default();
            }
        }
        for (&(p, loc), z) in &lm.zones {
            for (seq, &cid) in z.iter().enumerate() {
                if cid != 0 {
                    let ov = mats.get(&(p, loc, seq as u32)).cloned().unwrap_or_default();
                    lm.overlays.insert(cid, ov.iter().map(|&(_, o)| Some(o)).collect());
                    for (code, o) in ov {
                        lm.codes.insert(o, code);
                    }
                }
            }
        }
        lm
    }

    /// The card id at a place (Some(0) for an empty zone, as the Python map's 0).
    pub fn at(&self, li: &Li) -> Option<u32> {
        if li.loc == 0 {
            return None;
        }
        if li.loc & LOCATION_OVERLAY != 0 {
            let host = self.at(&Li { loc: li.loc & !LOCATION_OVERLAY, pos: 0, ..*li });
            let ov = host.filter(|&h| h != 0).and_then(|h| self.overlays.get(&h));
            return ov.and_then(|ov| ov.get(li.pos as usize).copied()).flatten();
        }
        let key = (li.con, li.loc);
        if self.unknown.contains(&key) {
            return None;
        }
        if let Some(z) = self.zones.get(&key) {
            return z.get(li.seq as usize).copied();
        }
        self.lists.get(&key).and_then(|l| l.get(li.seq as usize).copied()).flatten()
    }

    pub fn move_card(&mut self, from: &Li, to: &Li) -> Option<u32> {
        let cid = self.at(from);
        if from.loc & LOCATION_OVERLAY != 0 {
            let host = self.at(&Li { loc: from.loc & !LOCATION_OVERLAY, ..*from });
            if let Some(h) = host
                && let Some(ov) = self.overlays.get_mut(&h)
                && let Some(p) = ov.iter().position(|&x| x == cid)
            {
                ov.remove(p);
            }
        } else if let Some(z) = self.zones.get_mut(&(from.con, from.loc)) {
            if let Some(s) = z.get_mut(from.seq as usize) {
                *s = 0;
            }
        } else if from.loc != 0
            && !self.unknown.contains(&(from.con, from.loc))
            && let Some(l) = self.lists.get_mut(&(from.con, from.loc))
            && (from.seq as usize) < l.len()
        {
            l.remove(from.seq as usize);
        }
        if to.loc & LOCATION_OVERLAY != 0 {
            if let Some(h) = self.at(&Li { loc: to.loc & !LOCATION_OVERLAY, ..*to }) {
                let ov = self.overlays.entry(h).or_default();
                let p = (to.pos as usize).min(ov.len());
                ov.insert(p, cid);
            }
        } else if let Some(z) = self.zones.get_mut(&(to.con, to.loc)) {
            if let Some(s) = z.get_mut(to.seq as usize) {
                *s = cid.unwrap_or(0);
            }
        } else if !self.unknown.contains(&(to.con, to.loc))
            && let Some(l) = self.lists.get_mut(&(to.con, to.loc))
        {
            let p = (to.seq as usize).min(l.len());
            l.insert(p, cid);
        }
        cid
    }

    /// Card ids moved in place of other cards (REASON_REPLACE) by the upcoming messages.
    pub fn replaced(&mut self, up: &[Msg]) -> Vec<u32> {
        let mut out = Vec::new();
        for m in up {
            if let Msg::Move { from, to, reason, .. } = m {
                let cid = self.at(&from.into());
                if truthy(cid) && reason & REASON_REPLACE != 0 {
                    out.push(cid.unwrap());
                }
                self.move_card(&from.into(), &to.into());
            }
        }
        out
    }

    /// Card ids referenced by the upcoming messages, in order.
    pub fn references(&mut self, up: &[Msg]) -> Vec<u32> {
        let mut refs: Vec<u32> = Vec::new();
        self.released.clear();
        let add = |refs: &mut Vec<u32>, cid: Option<u32>| {
            if let Some(c) = cid.filter(|&c| c != 0)
                && !refs.contains(&c)
            {
                refs.push(c);
            }
        };
        for m in up {
            match m {
                Msg::Move { from, to, reason, .. } => {
                    let cid = self.at(&from.into());
                    if reason & REASON_LOST_TARGET == 0 {
                        add(&mut refs, cid);
                    }
                    if truthy(cid) && reason & REASON_RELEASE != 0 && !self.released.contains(&cid.unwrap()) {
                        self.released.push(cid.unwrap());
                    }
                    self.move_card(&from.into(), &to.into());
                }
                Msg::BecomeTarget { cards } | Msg::CardSelected { cards } | Msg::RandomSelected { cards, .. } => {
                    for li in cards {
                        add(&mut refs, self.at(&li.into()));
                    }
                }
                Msg::ConfirmCards { cards, .. } | Msg::ConfirmDecktop { cards, .. } | Msg::ConfirmExtratop { cards, .. } => {
                    for c in cards {
                        add(&mut refs, self.at(&c.into()));
                    }
                }
                Msg::PosChange { .. }
                | Msg::Set { .. }
                | Msg::Spsummoning { .. }
                | Msg::Summoning { .. }
                | Msg::Flipsummoning { .. }
                | Msg::Chaining { .. }
                | Msg::AddCounter { .. }
                | Msg::RemoveCounter { .. } => {
                    let li = msg_li(m).unwrap();
                    add(&mut refs, self.at(&li));
                }
                Msg::Equip { card, target } | Msg::CardTarget { card, target } | Msg::CancelTarget { card, target } => {
                    add(&mut refs, self.at(&card.into()));
                    add(&mut refs, self.at(&target.into()));
                }
                Msg::Attack { target, .. } => add(&mut refs, self.at(&target.into())),
                Msg::Draw { player, cards } => {
                    for _ in cards {
                        let card = self.lists.get_mut(&(*player, LOCATION_DECK)).and_then(|d| d.pop());
                        if let Some(c) = card {
                            self.lists.entry((*player, LOCATION_HAND)).or_default().push(c);
                        }
                    }
                }
                Msg::ShuffleHand { player, .. } | Msg::ShuffleDeck { player } | Msg::ShuffleExtra { player, .. } => {
                    let loc = match m {
                        Msg::ShuffleHand { .. } => LOCATION_HAND,
                        Msg::ShuffleDeck { .. } => LOCATION_DECK,
                        _ => LOCATION_EXTRA,
                    };
                    let key = (*player, loc);
                    let forced = self.plan.pop_front();
                    if let Some((fp, fl, Some(ids))) = &forced
                        && (*fp, *fl) == key
                        && !self.unknown.contains(&key)
                    {
                        let cur = self.lists.get(&key).cloned().unwrap_or_default();
                        if !cur.contains(&None) && !ids.contains(&None) {
                            let mut a: Vec<u32> = cur.iter().map(|c| c.unwrap()).collect();
                            let mut b: Vec<u32> = ids.iter().map(|c| c.unwrap()).collect();
                            a.sort();
                            b.sort();
                            if a == b {
                                self.lists.insert(key, ids.clone()); // the replay decides the result
                                continue;
                            }
                        }
                    }
                    self.unknown.insert(key);
                }
                Msg::Swap { loc1, loc2, .. } => {
                    let (a, b) = (self.at(&loc1.into()), self.at(&loc2.into()));
                    if let Some(z) = self.zones.get_mut(&(loc1.con, loc1.loc)) {
                        z[loc1.seq as usize] = b.unwrap_or(0);
                    }
                    if let Some(z) = self.zones.get_mut(&(loc2.con, loc2.loc)) {
                        z[loc2.seq as usize] = a.unwrap_or(0);
                    }
                }
                _ => {}
            }
        }
        refs
    }
}

/// Card ids of all cards a prompt lists.
pub fn prompt_card_ids(lm: &LocMap, prompt: &Msg) -> Vec<u32> {
    let mut out = Vec::new();
    for (_, list) in crate::belief::tracker::prompt_card_lists(prompt) {
        for c in list {
            if c.loc != 0
                && let Some(cid) = lm.at(&c.into()).filter(|&c| c != 0)
            {
                out.push(cid);
            }
        }
    }
    if let Some(li) = msg_li(prompt).filter(|_| matches!(prompt, Msg::SelectEffectyn { .. }))
        && let Some(cid) = lm.at(&li).filter(|&c| c != 0)
    {
        out.push(cid);
    }
    out
}

/// Card ids a response selects (the card objects the core will act on).
pub fn response_card_ids(lm: &LocMap, prompt: &Msg, resp: Option<&Response>) -> Vec<u32> {
    let Some(resp) = resp else { return vec![] };
    let mut cards: Vec<Li> = Vec::new();
    match prompt {
        Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. } => {
            if let (Some(list), Some(i)) = (resp.kind.and_then(|k| k.cards(prompt)), resp.index)
                && let Some(c) = list.get(i as usize)
            {
                cards.push(c.into());
            }
        }
        Msg::SelectChain { chains, .. } => {
            if let Some(i) = resp.index.filter(|&i| i >= 0)
                && let Some(c) = chains.get(i as usize)
            {
                cards.push(c.into());
            }
        }
        Msg::SelectCard { cards: cs, .. } | Msg::SelectTribute { cards: cs, .. } | Msg::SelectSum { cards: cs, .. } => {
            for &i in resp.indices.as_deref().unwrap_or(&[]) {
                if let Some(c) = cs.get(i as usize) {
                    cards.push(c.into());
                }
            }
            if let Msg::SelectSum { must_cards, .. } = prompt {
                cards.extend(must_cards.iter().map(Li::from));
            }
        }
        Msg::SelectUnselectCard { select_cards, unselect_cards, .. } => {
            if let Some(i) = resp.index {
                let all: Vec<&Card> = select_cards.iter().chain(unselect_cards).collect();
                if let Some(c) = all.get(i as usize) {
                    cards.push((*c).into());
                }
            }
        }
        Msg::SelectEffectyn { .. } => cards.push(msg_li(prompt).unwrap()),
        _ => {}
    }
    cards.iter().filter(|c| c.loc != 0).filter_map(|c| lm.at(c).filter(|&x| x != 0)).collect()
}

/// Card ids of cards (infer.card_ids).
pub fn card_ids(lm: &LocMap, cards: &[Card]) -> Vec<Option<u32>> {
    cards.iter().map(|c| lm.at(&c.into())).collect()
}

/// enumerated alternatives per decision (only a few are probed anyway)
const MAX_CANDIDATES: usize = 64;

/// messages that end the action a selection belongs to
fn window_end(m: &Msg) -> bool {
    matches!(
        m,
        Msg::Chaining { .. }
            | Msg::Chained { .. }
            | Msg::ChainSolved { .. }
            | Msg::ChainEnd {}
            | Msg::NewPhase { .. }
            | Msg::NewTurn { .. }
            | Msg::Win { .. }
            | Msg::Summoning { .. }
            | Msg::Spsummoning { .. }
            | Msg::Flipsummoning { .. }
    )
}

fn sel_cards(m: &Msg) -> &[Loc] {
    match m {
        Msg::BecomeTarget { cards } | Msg::CardSelected { cards } => cards,
        _ => &[],
    }
}

/// Whether `x` reveals only cards of the selection `sel` announces.
fn reveals_part_of(x: &Msg, sel: &Msg, after: &[Msg]) -> bool {
    let Msg::ConfirmCards { cards, .. } = x else { return false };
    if cards.is_empty() {
        return false;
    }
    let mut used: HashSet<(u32, u32, u32)> = sel_cards(sel).iter().map(|c| (c.con, c.loc, c.seq)).collect();
    for m in after {
        if window_end(m) || matches!(m, Msg::ChainSolved { .. } | Msg::ChainEnd {}) {
            break;
        }
        if let Msg::Move { from, .. } = m {
            used.insert((from.con, from.loc, from.seq));
        }
    }
    cards.iter().all(|c| used.contains(&(c.con, c.loc, c.seq)))
}

pub fn selection_window(up: &[Msg]) -> &[Msg] {
    for (i, m) in up.iter().enumerate() {
        if window_end(m) {
            return &up[..i];
        }
        if matches!(m, Msg::BecomeTarget { .. } | Msg::CardSelected { .. })
            && up[..i].iter().any(|x| !matches!(x, Msg::Hint { .. } | Msg::CardHint { .. }) && !reveals_part_of(x, m, &up[i + 1..]))
        {
            return &up[..i];
        }
    }
    up
}

pub const HINTMSG_DESTROY: u64 = 502;
pub const HINTMSG_SUMMON_MSGS: [u64; 2] = [508, 509];
pub const HINTMSG_DESREPLACE: u64 = 521;

fn ids_index(ids: &[Option<u32>], cid: u32) -> Option<usize> {
    ids.iter().position(|&x| x == Some(cid))
}

/// SELECT_CARD / SELECT_TRIBUTE / SELECT_SUM. `select_msg`: the selection's hint text.
pub fn infer_cards(prompt: &Msg, up: &[Msg], lm0: &LocMap, ids: Option<Vec<Option<u32>>>, select_msg: Option<u64>) -> Inferred {
    let (cards, lo, hi, cancelable) = match prompt {
        Msg::SelectCard { cards, min, max, cancelable, .. } | Msg::SelectTribute { cards, min, max, cancelable, .. } => (cards, *min as usize, *max as usize, *cancelable != 0),
        Msg::SelectSum { cards, min, max, .. } => (cards, *min as usize, *max as usize, false),
        _ => return (vec![], Certain::No),
    };
    let is_card = matches!(prompt, Msg::SelectCard { .. });
    let ids = ids.unwrap_or_else(|| card_ids(lm0, cards));
    let mut lm = lm0.clone();
    let refs = lm.references(up);
    let substitutes: HashSet<u32> = if select_msg == Some(HINTMSG_DESTROY) { lm0.clone().replaced(selection_window(up)).into_iter().collect() } else { HashSet::new() };
    let chosen: Vec<usize> = refs.iter().filter(|c| !substitutes.contains(c)).filter_map(|&c| ids_index(&ids, c)).collect();
    let n = cards.len();
    let mut cands: Vec<Response> = Vec::new();
    let mut certain = false;
    let push_new = |cands: &mut Vec<Response>, c: Response| {
        if !cands.contains(&c) {
            cands.push(c);
        }
    };
    let first = up.iter().find(|m| !matches!(m, Msg::Hint { .. } | Msg::CardHint { .. }));
    if is_card
        && let Some(Msg::CardSelected { cards: fc }) = first
        && !fc.iter().all(|x| lm0.at(&x.into()).is_some_and(|c| substitutes.contains(&c)))
    {
        // the core announces exactly the selected cards
        let sel: Vec<usize> = (0..n).filter(|&i| fc.iter().any(|x| same_loc(&(&cards[i]).into(), &x.into()))).collect();
        if sel.len() == fc.len() && lo <= sel.len() && sel.len() <= hi {
            cands.push(indices(sel));
        }
    }
    match prompt {
        Msg::SelectTribute { .. } => {
            let valid = |sel: &[usize]| !sel.is_empty() && sel.len() <= hi && sel.iter().map(|&i| cards[i].release_param.unwrap_or(0) as usize).sum::<usize>() >= lo;
            let released: Vec<usize> = lm.released.iter().filter_map(|&c| ids_index(&ids, c)).collect();
            if !released.is_empty() && valid(&released) {
                let mut r = released.clone();
                r.sort();
                cands.push(indices(r));
                certain = true;
            }
            for k in 1..=chosen.len() {
                let mut s = chosen[..k].to_vec();
                s.sort();
                let c = indices(s);
                if valid(&chosen[..k]) && !cands.contains(&c) {
                    cands.push(c);
                    break;
                }
            }
            let all: Vec<usize> = (0..n).collect();
            for k in 1..=hi.min(n) {
                for sub in combinations(&all, k).take(MAX_CANDIDATES) {
                    let c = indices(sub.clone());
                    if valid(&sub) && !cands.contains(&c) {
                        cands.push(c);
                    }
                }
            }
        }
        Msg::SelectSum { .. } => {
            let sorted = |v: &[usize]| {
                let mut v = v.to_vec();
                v.sort();
                v
            };
            let near: Vec<usize> = lm0.clone().references(selection_window(up)).iter().filter_map(|&c| ids_index(&ids, c)).collect();
            if near.len() > 1 && check_sum(prompt, &sorted(&near)) {
                cands.push(indices(sorted(&near)));
            }
            for k in 1..=chosen.len() {
                if check_sum(prompt, &sorted(&chosen[..k])) {
                    push_new(&mut cands, indices(sorted(&chosen[..k])));
                    certain = k == chosen.len() && cands.len() == 1;
                    break;
                }
            }
            if cands.is_empty() {
                for k in 1..=chosen.len().min(5) {
                    for sub in combinations(&chosen, k).take(MAX_CANDIDATES) {
                        if check_sum(prompt, &sorted(&sub)) {
                            cands.push(indices(sorted(&sub)));
                        }
                    }
                }
            }
            let mut rng = PyRandom::new(0);
            for _ in 0..8 {
                if let Some(sel) = random_sum_selection(prompt, &mut rng, 20000)
                    && !sel.is_empty()
                {
                    push_new(&mut cands, indices(sorted(&sel)));
                }
            }
        }
        _ if !chosen.is_empty() => {
            // the cards referenced before the action ends are the selection
            let near: Vec<usize> = lm0.clone().references(selection_window(up)).iter().filter(|c| !substitutes.contains(c)).filter_map(|&c| ids_index(&ids, c)).collect();
            let kn = near.len().min(hi);
            if !near.is_empty() && kn >= lo {
                let mut s = near[..kn].to_vec();
                s.sort();
                cands.push(indices(s));
                certain = kn == near.len() && (kn == hi || chosen.len() == near.len());
            }
            let k = chosen.len().min(hi);
            for j in lo.max(1)..=k {
                let mut s = chosen[..j].to_vec();
                s.sort();
                push_new(&mut cands, indices(s));
            }
            if k < chosen.len() {
                for sub in combinations(&chosen, k).take(MAX_CANDIDATES) {
                    let mut s = sub;
                    s.sort();
                    push_new(&mut cands, indices(s));
                }
            }
        }
        _ => {}
    }
    if is_card && !chosen.is_empty() && chosen.len() < lo && lo <= n {
        // (fewer cards are used later than must be selected: the used ones and any others)
        let others: Vec<usize> = (0..n).filter(|i| !chosen.contains(i)).collect();
        for sub in combinations(&others, lo - chosen.len()).take(MAX_CANDIDATES) {
            let mut s = [chosen.clone(), sub].concat();
            s.sort();
            push_new(&mut cands, indices(s));
        }
    }
    if is_card && select_msg.is_some_and(|m| HINTMSG_SUMMON_MSGS.contains(&m)) && hi == 1 && !cands.is_empty() {
        // the monster to summon is the one summoned, even where other cards are used first
        let mut lm2 = lm0.clone();
        let mut summoned = None;
        for m in up {
            if matches!(m, Msg::ChainSolved { .. } | Msg::ChainEnd {}) {
                break;
            }
            if let Msg::Move { from, to, .. } = m {
                let cid = lm2.at(&from.into());
                if to.loc == LOCATION_MZONE
                    && let Some(i) = cid.and_then(|c| ids_index(&ids, c))
                {
                    summoned = Some(i);
                    break;
                }
                lm2.move_card(&from.into(), &to.into());
            }
        }
        if let Some(s) = summoned
            && cands[0] != indices(vec![s])
        {
            let sc = indices(vec![s]);
            cands = std::iter::once(sc.clone()).chain(cands.into_iter().filter(|c| *c != sc)).collect();
            certain = false;
        }
    }
    if is_card && select_msg == Some(HINTMSG_DESREPLACE) && hi == 1 {
        // the card a replacement protects, among cards being destroyed, is the one that stays
        let mut lm2 = lm0.clone();
        let (mut moved, mut destroyed): (HashSet<Option<u32>>, HashSet<Option<u32>>) = Default::default();
        for m in up {
            if matches!(m, Msg::ChainSolved { .. } | Msg::ChainEnd {} | Msg::NewPhase { .. } | Msg::NewTurn { .. }) {
                break;
            }
            if let Msg::Move { from, to, reason, .. } = m {
                let cid = lm2.at(&from.into());
                moved.insert(cid);
                if reason & REASON_DESTROY != 0 {
                    destroyed.insert(cid);
                }
                lm2.move_card(&from.into(), &to.into());
            }
        }
        let stays: Vec<Response> = (0..n).filter(|&i| !moved.contains(&ids[i])).map(|i| indices(vec![i])).collect();
        if !stays.is_empty() && ids.iter().any(|i| destroyed.contains(i)) {
            cands = [stays.clone(), cands.into_iter().filter(|c| !stays.contains(c)).collect()].concat();
            certain = false;
        }
    }
    if is_card && (hi == 1 || lo == 1) {
        // (substitutes last)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| ids[i].is_some_and(|c| substitutes.contains(&c)));
        for i in order {
            push_new(&mut cands, indices(vec![i]));
        }
    }
    if cands.is_empty() && lo <= n {
        cands.push(indices((0..lo).collect()));
    }
    if cancelable {
        cands.push(Response { cancel: Some(true), ..Response::default() });
    }
    if select_msg.is_some_and(|m| m >= 1 << 20) {
        // (a script's own select message: the candidates are probed)
        certain = false;
    }
    (cands, certain.into())
}

pub fn infer_unselect(prompt: &Msg, up: &[Msg], lm0: &LocMap, ids: Option<Vec<Option<u32>>>) -> Inferred {
    let Msg::SelectUnselectCard { select_cards: sel, unselect_cards: unsel, finishable, cancelable, .. } = prompt else { return (vec![], Certain::No) };
    let all: Vec<Card> = sel.iter().chain(unsel).cloned().collect();
    let ids = ids.unwrap_or_else(|| card_ids(lm0, &all));
    let unsel_ids: HashSet<Option<u32>> = ids[sel.len()..].iter().copied().collect();
    let near = lm0.clone().references(selection_window(up));
    let refs = lm0.clone().references(up);
    // (cards the selection then moves as one group move in the engine's order of the group)
    let mut lm = lm0.clone();
    let mut moved: Vec<(Option<u32>, u32, u32)> = Vec::new();
    let mut unknown: HashSet<Option<u32>> = HashSet::new();
    for m in selection_window(up) {
        if let Msg::Move { from, to, reason, code } = m {
            moved.push((lm.at(&from.into()), *reason, to.loc));
            if *code == 0 {
                unknown.insert(moved.last().unwrap().0);
            }
            lm.move_card(&from.into(), &to.into());
        }
    }
    let sel_ids: HashSet<Option<u32>> = ids[..sel.len()].iter().copied().filter(|i| !unsel_ids.contains(i)).collect();
    let group: Vec<(u32, u32)> = moved.iter().filter(|(c, _, _)| sel_ids.contains(c)).map(|&(_, r, l)| (r, l)).collect();
    let grouped = group.len() > 1 && group.iter().collect::<HashSet<_>>().len() == 1;
    let whole = grouped
        && group[0].1 & LOCATION_OVERLAY == 0
        && moved.iter().filter(|&&(_, r, l)| (r, l) == group[0]).all(|(c, _, _)| ids.contains(c) && !unknown.contains(c));
    let fin = *finishable != 0 || *cancelable != 0;
    let finish = Response { finish: Some(true), ..Response::default() };
    for (window, certain) in [(&near, if whole || !grouped { Certain::Yes } else { Certain::Likely }), (&refs, Certain::No)] {
        let picks: Vec<usize> = window.iter().filter(|&&c| ids[..sel.len()].contains(&Some(c)) && !unsel_ids.contains(&Some(c))).filter_map(|&c| ids_index(&ids, c)).collect();
        if !picks.is_empty() {
            let mut first: Vec<Response> = Vec::new();
            for j in picks {
                let r = Response::index(j);
                if !first.contains(&r) {
                    first.push(r);
                }
            }
            let rest: Vec<Response> = (0..sel.len()).map(Response::index).filter(|r| !first.contains(r)).collect();
            let f: Vec<Response> = if fin { vec![finish.clone()] } else { vec![] };
            if unsel.is_empty() {
                return ([first, f, rest].concat(), certain);
            }
            return ([first[..1].to_vec(), f, first[1..].to_vec(), rest].concat(), certain);
        }
    }
    let mut cands = Vec::new();
    if fin {
        cands.push(finish);
    }
    cands.extend((0..sel.len()).map(Response::index));
    let c = !cands.is_empty() && fin && !near.is_empty();
    (cands, c.into())
}

pub fn infer_counter(prompt: &Msg, up: &[Msg]) -> Inferred {
    let Msg::SelectCounter { cards, counter_type, count, .. } = prompt else { return (vec![], Certain::No) };
    let mut counts = vec![0i64; cards.len()];
    for m in up {
        if matches!(m, Msg::Hint { .. } | Msg::CardHint { .. }) {
            continue;
        }
        // the removal follows right away; the messages carry the counter's flags
        let Msg::RemoveCounter { counter_type: ct, con, loc, seq, count: c } = m else { break };
        if (ct & 0xfff) != (counter_type & 0xfff) {
            break;
        }
        for (i, card) in cards.iter().enumerate() {
            if card.con == *con && card.loc == *loc && card.seq == *seq {
                counts[i] += *c as i64;
            }
        }
    }
    if counts.iter().sum::<i64>() == *count as i64 {
        return (vec![Response { counts: Some(counts), ..Response::default() }], Certain::Yes);
    }
    match random_response(prompt, &mut PyRandom::new(0), 0.05) {
        Ok(r) => (vec![r], Certain::No),
        Err(_) => (vec![], Certain::No),
    }
}

pub fn infer_rps(prompt: &Msg, up: &[Msg]) -> Inferred {
    let player = prompt.player().unwrap_or(0);
    for m in up {
        if let Msg::HandRes { result: r } = m {
            let hand = if player == 0 { r & 3 } else { (r >> 2) & 3 };
            if (1..=3).contains(&hand) {
                return (vec![Response { hand: Some(hand as i64), ..Response::default() }], Certain::Yes);
            }
        }
    }
    ((1..=3).map(|h| Response { hand: Some(h), ..Response::default() }).collect(), Certain::No)
}

/// Sorting the top/bottom cards of a deck (Duel.SortDecktop/SortDeckbottom).
fn infer_deck_sort(cards: &[Card], up: &[Msg]) -> Option<Vec<i64>> {
    let count = cards.len();
    if count == 0 || cards.iter().any(|c| c.loc != LOCATION_DECK) {
        return None;
    }
    let con = cards[0].con;
    let seqs: Vec<u32> = cards.iter().map(|c| c.seq).collect();
    let bottom = seqs[0] == 0;
    let mut observed = Vec::new();
    for m in up {
        match m {
            Msg::Move { from, to, .. } if from.loc == LOCATION_DECK && to.loc == LOCATION_DECK && from.con == con => {
                observed.push((from.seq, to.seq));
                if observed.len() == count {
                    break;
                }
            }
            Msg::Hint { .. } | Msg::CardHint { .. } => {}
            _ => break,
        }
    }
    if observed.len() != count {
        return None;
    }
    let max = *seqs.iter().max().unwrap();
    let mut deck: Vec<u32> = (0..max + 1 + if bottom { 64 } else { 0 }).collect();
    let mut tc: Vec<Option<u32>> = vec![None; count];
    for (i, &(frm, to)) in observed.iter().enumerate() {
        if frm as usize >= deck.len() {
            return None;
        }
        let card = deck.remove(frm as usize);
        if bottom {
            deck.insert(0, card);
        } else {
            deck.push(card);
        }
        if deck.iter().position(|&x| x == card) != Some(to as usize) || !seqs.contains(&card) {
            return None;
        }
        tc[if bottom { i } else { count - i - 1 }] = Some(card);
    }
    if tc.contains(&None) || tc.iter().collect::<HashSet<_>>().len() != count {
        return None;
    }
    Some((0..count).map(|i| tc.iter().position(|&x| x == Some(seqs[i])).unwrap() as i64).collect())
}

/// An event kind, ordered as Python's str() of it: ('MOVE', loc) or the message name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Event {
    Move(u32),
    Other(&'static str),
}

impl Event {
    fn py_str(&self) -> String {
        match self {
            Event::Move(l) => format!("('MOVE', {l})"),
            Event::Other(s) => s.to_string(),
        }
    }
}

/// itertools.product(kinds, repeat=r) in order.
fn product(n: usize, r: usize) -> impl Iterator<Item = Vec<usize>> {
    let mut cur = vec![0usize; r];
    let mut done = n == 0 && r > 0;
    std::iter::from_fn(move || {
        if done {
            return None;
        }
        let out = cur.clone();
        let mut i = r;
        loop {
            if i == 0 {
                done = true;
                break;
            }
            i -= 1;
            cur[i] += 1;
            if cur[i] < n {
                break;
            }
            cur[i] = 0;
        }
        Some(out)
    })
}

/// Effects that are no chain links applied in the chosen order: each shows as one event.
fn sort_by_events(cards: &[Card], up: &[Msg]) -> Vec<Vec<i64>> {
    let mut events: Vec<Event> = Vec::new();
    for m in up {
        if matches!(m, Msg::Hint { .. } | Msg::CardHint { .. } | Msg::PlayerHint { .. }) {
            continue;
        }
        if events.len() == cards.len() {
            break;
        }
        events.push(match m {
            Msg::Move { from, .. } => Event::Move(from.loc),
            _ => Event::Other(m.name()),
        });
    }
    if events.len() != cards.len() {
        return vec![];
    }
    let codes: Vec<u32> = cards.iter().map(|c| c.code).collect::<BTreeSet<_>>().into_iter().collect();
    let mut kinds: Vec<Event> = Vec::new();
    for e in &events {
        if !kinds.contains(e) {
            kinds.push(e.clone());
        }
    }
    kinds.sort_by_key(|k| k.py_str());
    let count = |c: u32| cards.iter().filter(|x| x.code == c).count();
    let mut fits: Vec<Vec<usize>> = Vec::new();
    for assign in product(kinds.len(), codes.len()).take(4096) {
        if fits.len() > 4 {
            break;
        }
        let ok = (0..kinds.len()).all(|t| {
            let have: usize = (0..codes.len()).filter(|&ci| assign[ci] == t).map(|ci| count(codes[ci])).sum();
            have == events.iter().filter(|e| **e == kinds[t]).count()
        });
        if ok {
            fits.push(assign);
        }
    }
    let mut orders = Vec::new();
    for f in fits.iter().take(4) {
        let mut order: Vec<Option<i64>> = vec![None; cards.len()];
        for (k, e) in events.iter().enumerate() {
            let t = kinds.iter().position(|x| x == e).unwrap();
            let i = (0..cards.len()).find(|&i| order[i].is_none() && f[codes.iter().position(|&c| c == cards[i].code).unwrap()] == t).unwrap();
            order[i] = Some(k as i64);
        }
        orders.push(order.into_iter().map(|o| o.unwrap()).collect());
    }
    orders
}

/// itertools.permutations(range(n)) in order.
fn permutations(n: usize) -> Vec<Vec<i64>> {
    fn rec(cur: &mut Vec<i64>, used: &mut Vec<bool>, n: usize, out: &mut Vec<Vec<i64>>) {
        if cur.len() == n {
            out.push(cur.clone());
            return;
        }
        for i in 0..n {
            if !used[i] {
                used[i] = true;
                cur.push(i as i64);
                rec(cur, used, n, out);
                cur.pop();
                used[i] = false;
            }
        }
    }
    let mut out = Vec::new();
    rec(&mut Vec::new(), &mut vec![false; n], n, &mut out);
    out
}

pub fn infer_sort(prompt: &Msg, up: &[Msg]) -> Inferred {
    let (cards, chain) = match prompt {
        Msg::SortChain { cards, .. } => (cards, true),
        Msg::SortCard { cards, .. } => (cards, false),
        _ => return (vec![], Certain::No),
    };
    let n = cards.len();
    let order_r = |o: Option<Vec<i64>>| Response { order: o, ..Response::default() };
    let mut cands = Vec::new();
    if chain {
        // the triggers are chained in the chosen order (order[i] = rank of card i)
        let mut ranks: Vec<Option<i64>> = vec![None; n];
        let mut k = 0;
        for m in up {
            let Msg::Chaining { code, trigger_con, trigger_loc, trigger_seq, .. } = m else { continue };
            for (i, c) in cards.iter().enumerate() {
                if ranks[i].is_none() && c.code == *code && c.con == *trigger_con && c.loc == *trigger_loc && c.seq == *trigger_seq {
                    ranks[i] = Some(k);
                    k += 1;
                    break;
                }
            }
            if k as usize == n {
                break;
            }
        }
        if k > 0 {
            let mut rest = k..n as i64;
            let order: Vec<i64> = ranks.iter().map(|r| r.unwrap_or_else(|| rest.next().unwrap())).collect();
            cands.push(order_r(Some(order)));
            if k as usize == n {
                return (cands, Certain::Yes);
            }
        } else {
            cands.extend(sort_by_events(cards, up).into_iter().map(|o| order_r(Some(o))));
        }
    } else if let Some(order) = infer_deck_sort(cards, up) {
        return (vec![order_r(Some(order))], Certain::Yes);
    }
    // (the order None: Python's {"order": None}, encoded as no order)
    cands.push(Response { order: None, ..Response::default() });
    if n <= 5 {
        cands.extend(permutations(n).into_iter().map(|p| order_r(Some(p))));
    }
    (cands, Certain::No)
}
