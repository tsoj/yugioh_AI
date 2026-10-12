//! Last-resort repair of a belief replay: the hidden card the engine read (read_guided.py).
//!
//! When the replay's own repairs give up, the card whose identity is wrong is usually one the
//! engine read just before the divergence: a hidden card of the opponent (another code) or a copy
//! of a card that another copy should be. The core's read log names the hidden cards the engine
//! looked at and the properties it read; the candidates are swaps with same-name copies in the
//! same place and other codes that differ in those properties. Each candidate is tried as a whole
//! replay with the usual repairs; one that gets further is followed for a few rounds.

use indexmap::IndexMap;
use ygo_core::constants::*;

use crate::belief::completion::card;
use crate::belief::replay::util::{V, jv};
use crate::belief::replay::{Asm, BeliefReplay, Cp, RErr, ReadStep, Swaps, Where};
use crate::game::ObsItem;
use crate::rduel::RDuel;

/// code, type, Level, Rank, attribute, race, ATK, DEF, archetypes, Link (see OCG_ExtReadLog)
pub const PROPERTIES: u32 = 0x3ff;
/// items before the divergence whose reads count
pub const WINDOW: usize = 5;
/// replays per round
pub const CANDIDATES: usize = 20;
pub const ROUNDS: usize = 3;
pub const CODES_PER_CARD: usize = 8;

/// A candidate: (a swap: control point and pairs, assumptions, text).
pub type Candidate = (Option<(Cp, Vec<(u32, u32)>)>, Vec<Asm>, String);

fn sub_replay(br: &BeliefReplay, swaps: &Swaps, assumptions: &[Asm], restarts: usize) -> BeliefReplay {
    let mut sub = BeliefReplay::new(br.viewer, br.init.clone(), br.items.clone(), Some(br.completion));
    sub.check_state = br.check_state;
    sub.max_probes = br.max_probes;
    sub.probe_items = br.probe_items;
    sub.max_depth = br.max_depth;
    sub.max_restarts = restarts;
    sub.probe_timeout = br.probe_timeout;
    sub.max_stall = br.max_stall;
    sub.min_progress = br.min_progress;
    sub.stopped = br.stopped;
    sub.read_guided = false;
    sub.deadline = br.deadline;
    sub.swaps = swaps.clone();
    sub.assumptions = assumptions.to_vec();
    sub
}

/// A replay without repairs, logging per engine step: (divergence position: None if consistent,
/// -1 for a tracker error), the steps, the replay.
fn logged_run(br: &BeliefReplay, swaps: &Swaps, assumptions: &[Asm]) -> (Option<i64>, Vec<ReadStep>, BeliefReplay) {
    let mut sub = sub_replay(br, swaps, assumptions, 0);
    sub.logged_steps = Some(Vec::new());
    let r = sub.run_attempts();
    let steps = sub.logged_steps.take().unwrap_or_default();
    match r {
        Ok(_) => (None, steps, sub),
        Err(RErr::Div(e)) => (Some(e.pos as i64), steps, sub),
        Err(_) => (Some(-1), steps, sub),
    }
}

/// The candidate repairs for the hidden cards read shortly before the divergence at `pos`.
pub fn candidates(br: &BeliefReplay, steps: &[ReadStep], pos: usize, codes: &[u32]) -> Vec<Candidate> {
    let opp = br.opponent;
    // (card id -> (properties read, at which position, where it was))
    type Read = (u32, usize, (u32, u32, u32, u32));
    let mut read: IndexMap<u32, Read> = IndexMap::new();
    let mut before: Option<&Where> = None;
    for (p, reads, wh) in steps {
        if pos.saturating_sub(WINDOW) <= *p && *p <= pos && *p + WINDOW >= pos {
            for &(cid, mask) in reads {
                // (where it was when read: the step may move it)
                let loc = before.and_then(|b| b.get(&cid)).or_else(|| wh.get(&cid)).copied();
                if let Some(loc) = loc
                    && (matches!(loc.1, LOCATION_HAND | LOCATION_DECK | LOCATION_EXTRA) || (loc.0 == opp && matches!(loc.1, LOCATION_MZONE | LOCATION_SZONE)))
                {
                    let m = read.get(&cid).map_or(0, |r| r.0) | mask;
                    read.insert(cid, (m, *p, loc));
                }
            }
        }
        before = Some(wh);
    }
    let signature = |code: u32, mask: u32| -> Option<Vec<i64>> {
        let c = card(code)?;
        let mut parts: Vec<i64> = Vec::new();
        for (bit, v) in [(2u32, c.type_ as i64), (4, c.level as i64), (16, c.attribute as i64), (32, c.race as i64), (64, c.attack as i64), (128, c.defense as i64)] {
            if mask & bit != 0 {
                parts.push(v);
            }
        }
        if mask & 256 != 0 {
            // (the archetypes as one part)
            parts.push(-1);
            parts.extend(c.setcodes.iter().map(|s| s as i64));
            parts.push(-1);
        }
        if parts.is_empty() {
            parts = vec![-2, c.type_ as i64, c.level as i64, c.attribute as i64, c.race as i64];
        }
        Some(parts)
    };
    let mut order: Vec<u32> = read.keys().copied().collect();
    // (the cards whose properties were read first, the latest first)
    order.sort_by_key(|c| (read[c].0 & PROPERTIES == 0, std::cmp::Reverse(read[c].1)));
    let mut per_card: Vec<Vec<Candidate>> = Vec::new();
    for cid in order {
        let (mask, p, (con, loc, i, code)) = read[&cid];
        let mut mine: Vec<Candidate> = Vec::new();
        let empty = IndexMap::new();
        let wh = steps.iter().rev().find(|s| s.0 < p).map_or(&empty, |s| &s.2);
        let arrived = steps.iter().find(|s| s.0 <= p && s.2.get(&cid).is_some_and(|w| (w.0, w.1) == (con, loc))).map_or(p, |s| s.0);
        for (&other, &(c2, l2, _, code2)) in wh {
            if other != cid && (c2, l2, code2) == (con, loc, code) {
                let mut ats = vec![p];
                if arrived != p {
                    ats.push(arrived);
                }
                for at in ats {
                    mine.push((Some(((at, 0), vec![(cid, other)])), vec![], format!("copies of {code} at P{con} loc {loc} swapped at item {at}")));
                }
            }
        }
        if con == opp && matches!(loc, LOCATION_HAND | LOCATION_DECK | LOCATION_MZONE | LOCATION_SZONE) {
            let mut seen = vec![signature(code, mask)];
            for &new in codes {
                let sig = signature(new, mask);
                if new != code && sig.is_some() && !seen.contains(&sig) {
                    seen.push(sig);
                    mine.push((None, vec![(p, con, loc, i, new)], format!("P{con} loc {loc} seq {i} at item {p}: {code} -> {new}")));
                }
                if mine.len() >= CODES_PER_CARD {
                    break;
                }
            }
        }
        per_card.push(mine);
    }
    // (breadth first: each card's first candidate, then the second ones, ...)
    let longest = per_card.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = Vec::new();
    for k in 0..longest {
        for lst in &per_card {
            if k < lst.len() {
                out.push(lst[k].clone());
            }
        }
    }
    out
}

/// A duel reproducing the stream after a read-guided repair, or None.
pub fn read_guided_repair(br: &mut BeliefReplay) -> Option<RDuel> {
    let mut swaps = br.swaps.clone();
    let mut assumptions = br.assumptions.clone();
    // (codes the opponent's cards showed first, then the hypothesis's deck)
    let mut seen: Vec<u32> = Vec::new();
    for it in &br.items {
        let ObsItem::Msg(m) = it else { continue };
        let v = jv(m);
        let code = v.i("code");
        if code == 0 {
            continue;
        }
        let con = v.o("con").or_else(|| v.o("player")).or_else(|| v.g("from").o("con"));
        if con == Some(br.opponent as i64) && !seen.contains(&(code as u32)) {
            seen.push(code as u32);
        }
    }
    let mut path: Vec<String> = Vec::new();
    for _ in 0..ROUNDS {
        let (pos, steps, first) = logged_run(br, &swaps, &assumptions);
        let pos = match pos {
            Some(p) if p >= 0 => p as usize,
            _ => return None, // (consistent without repairs, or no hypothesis: nothing to add here)
        };
        let mut codes = seen.clone();
        if let Some(h) = &first.hypothesis {
            let deck: std::collections::BTreeSet<u32> = h.decks[br.opponent as usize].0.iter().copied().collect();
            codes.extend(deck);
        }
        let mut uniq: Vec<u32> = Vec::new();
        for c in codes {
            if !uniq.contains(&c) {
                uniq.push(c);
            }
        }
        let mut best: Option<(usize, Swaps, Vec<Asm>, String)> = None;
        for (sw, asm, what) in candidates(br, &steps, pos, &uniq).into_iter().take(CANDIDATES) {
            let mut new_swaps = swaps.clone();
            if let Some((k, v)) = sw {
                new_swaps.entry(k).or_default().extend(v);
            }
            let mut new_asm = assumptions.clone();
            new_asm.extend(asm);
            let mut sub = sub_replay(br, &new_swaps, &new_asm, br.max_restarts);
            br.stats.read_guided += 1;
            match sub.run_attempts() {
                Ok(duel) => {
                    br.stats.restarts += sub.stats.restarts;
                    br.hypothesis = sub.hypothesis.clone();
                    br.swaps = new_swaps;
                    br.assumptions = new_asm;
                    let mut p = path.clone();
                    p.push(what);
                    br.read_guided_fix = Some(p);
                    return Some(duel);
                }
                Err(RErr::Div(e)) => {
                    if e.pos > pos && best.as_ref().is_none_or(|b| e.pos > b.0) {
                        best = Some((e.pos, new_swaps, new_asm, what));
                    }
                }
                Err(_) => continue,
            }
        }
        let (_, s, a, what) = best?;
        swaps = s;
        assumptions = a;
        path.push(what);
    }
    None
}
