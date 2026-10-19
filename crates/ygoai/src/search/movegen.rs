//! The moves of a decision ("move generation"): which prompts the search decides, and their
//! candidates.

use ygo_core::constants::*;

use super::params::{ROOT_COMBO_CARDS, ROOT_COMBO_WIDTH, SUM_STEPS};
use crate::actions::{Response, available_places, battle_options, check_sum, idle_options};
use crate::heuristic::{Heuristic, card, card_value};
use crate::messages::Msg;
use crate::state::Snapshot;

/// The prompts searched over (belief_mc.SEARCHED) when they have more than one candidate (a
/// selection of several cards has none: the rules make it).
pub fn is_searched(m: &Msg) -> bool {
    matches!(
        m,
        Msg::SelectIdlecmd { .. }
            | Msg::SelectBattlecmd { .. }
            | Msg::SelectChain { .. }
            | Msg::SelectEffectyn { .. }
            | Msg::SelectYesno { .. }
            | Msg::SelectOption { .. }
            | Msg::SelectCard { .. }
    )
}

fn idx(i: usize) -> Response {
    Response { index: Some(i as i64), ..Response::default() }
}

/// The candidates of a searched prompt (belief_mc.candidates).
pub fn candidates(m: &Msg) -> Vec<Response> {
    match m {
        Msg::SelectIdlecmd { .. } => idle_options(m),
        Msg::SelectBattlecmd { .. } => battle_options(m),
        Msg::SelectChain { chains, forced, .. } => {
            let opts = (0..chains.len()).map(idx);
            if *forced != 0 { opts.collect() } else { std::iter::once(Response { index: Some(-1), ..Response::default() }).chain(opts).collect() }
        }
        Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => {
            [true, false].map(|y| Response { yes: Some(y), ..Response::default() }).to_vec()
        }
        Msg::SelectOption { options, .. } => (0..options.len()).map(idx).collect(),
        Msg::SelectCard { min: 1, max: 1, cards, .. } => {
            (0..cards.len()).map(|i| Response { indices: Some(vec![i as i64]), ..Response::default() }).collect()
        }
        _ => vec![],
    }
}

/// The moves compared at the root, best first by the rules: a searched decision's candidates
/// (above), and for every other decision the rules' answer and alternatives to it (several cards,
/// sums, tributes, places, orders and counters: at most ROOT_COMBO_WIDTH; positions, numbers,
/// types and attributes: each). None of these is searched below the root (the rules answer
/// them there). Declaring a card name and rock-paper-scissors: none (the rules answer).
pub fn root_candidates(m: &Msg, rules: &Heuristic, state: Option<&Snapshot>) -> Vec<Response> {
    if is_searched(m) && !matches!(m, Msg::SelectCard { .. }) {
        return candidates(m);
    }
    let rules_answer = || rules.clone().respond_to(m, state).ok();
    let indices = |mut idx: Vec<usize>| {
        idx.sort();
        Response { indices: Some(idx.into_iter().map(|i| i as i64).collect()), ..Response::default() }
    };
    let mut out: Vec<Response> = match m {
        Msg::SelectCard { min: 1, max: 1, .. } => return candidates(m),
        Msg::SelectCard { min, max, cards, .. } if *max >= 1 && !cards.is_empty() => {
            // (the rules' count of their first cards, then subsets of the first few cards by how far
            // they are from the best of their size)
            let order = rules.card_order(cards);
            let n = cards.len();
            let lo = (*min).max(1).min(*max).min(n as u32) as usize;
            let hi = (*max as usize).min(n);
            let top = &order[..n.min(ROOT_COMBO_CARDS)];
            let mut sets: Vec<(usize, usize, Vec<usize>)> = Vec::new();
            for k in lo..=hi.min(top.len()) {
                for c in combinations(top.len(), k) {
                    let rank_sum: usize = c.iter().sum();
                    sets.push((rank_sum - k * (k - 1) / 2, k - lo, c.into_iter().map(|j| top[j]).collect()));
                }
            }
            sets.sort_by_key(|(far, more, _)| (*far, *more));
            let mut out: Vec<Response> = sets.into_iter().map(|(_, _, s)| indices(s)).collect();
            if lo > top.len() {
                out.push(indices(order[..lo].to_vec()));
            }
            out
        }
        Msg::SelectUnselectCard { finishable, cancelable, select_cards, .. } => {
            let mut out: Vec<Response> = rules.card_order(select_cards).into_iter().take(ROOT_COMBO_WIDTH).map(idx).collect();
            if *finishable != 0 || (*cancelable != 0 && select_cards.is_empty()) {
                out.push(Response { finish: Some(true), ..Response::default() });
            }
            out
        }
        Msg::SelectTribute { min, max, cards, .. } if !cards.is_empty() => {
            // (the rules' answer, then the least valuable sets that release enough, none to spare)
            let n = cards.len().min(10);
            let rel = |i: usize| cards[i].release_param.unwrap_or(0);
            let mut sets: Vec<(f64, Vec<usize>)> = Vec::new();
            for mask in 1u32..(1 << n) {
                let set: Vec<usize> = (0..n).filter(|&i| mask & (1 << i) != 0).collect();
                let total: u32 = set.iter().map(|&i| rel(i)).sum();
                if set.len() > (*max).max(1) as usize || total < *min || set.iter().any(|&i| total - rel(i) >= *min) {
                    continue;
                }
                sets.push((set.iter().map(|&i| card_value(&cards[i], card(cards[i].code))).sum(), set));
            }
            sets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            rules_answer().into_iter().chain(sets.into_iter().map(|(_, s)| indices(s))).collect()
        }
        Msg::SelectSum { cards, .. } if !cards.is_empty() => {
            // (valid selections, the least valuable cards tried first; the rules' answer is random)
            let mut by_value: Vec<usize> = (0..cards.len()).collect();
            let v = |i: usize| card_value(&cards[i], card(cards[i].code));
            by_value.sort_by(|&a, &b| v(a).partial_cmp(&v(b)).unwrap_or(std::cmp::Ordering::Equal));
            sum_selections(m, &by_value).into_iter().map(indices).collect()
        }
        Msg::SelectPosition { code, positions, .. } => {
            let order = if card(*code).is_some_and(|c| c.attack <= 0) {
                [POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK]
            } else {
                [POS_FACEUP_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEDOWN_ATTACK]
            };
            order.into_iter().filter(|p| positions & p != 0).map(|p| Response { position: Some(p as i64), ..Response::default() }).collect()
        }
        Msg::SelectPlace { player, count, flag } | Msg::SelectDisfield { player, count, flag } => {
            // (the places as listed, the rules taking the first)
            let places = available_places(*player, *flag);
            let k = (*count as usize).min(places.len());
            let top = &places[..places.len().min(ROOT_COMBO_CARDS)];
            let mut sets: Vec<(usize, Vec<usize>)> = combinations(top.len(), k).into_iter().map(|c| (c.iter().sum::<usize>(), c)).collect();
            sets.sort_by_key(|(rank_sum, _)| *rank_sum);
            sets.into_iter().map(|(_, c)| Response { places: Some(c.into_iter().map(|j| top[j]).collect()), ..Response::default() }).collect()
        }
        Msg::SelectCounter { count, cards, .. } => {
            // (the rules' answer, then as many as fit on each card in turn, the rest on the next)
            let cap = |i: usize| cards[i].counters.unwrap_or(0) as i64;
            let mut out: Vec<Response> = rules_answer().into_iter().collect();
            for first in 0..cards.len() {
                let mut counts = vec![0i64; cards.len()];
                let mut left = *count as i64;
                for i in (first..cards.len()).chain(0..first) {
                    let put = left.min(cap(i));
                    counts[i] = put;
                    left -= put;
                }
                if left == 0 {
                    out.push(Response { counts: Some(counts), ..Response::default() });
                }
            }
            out
        }
        Msg::SortCard { cards, .. } | Msg::SortChain { cards, .. } => {
            // (the orders in lexicographic order: as listed first, the rules' answer)
            let mut order: Vec<i64> = (0..cards.len() as i64).collect();
            let mut out = vec![Response { order: Some(order.clone()), ..Response::default() }];
            while out.len() < ROOT_COMBO_WIDTH && next_permutation(&mut order) {
                out.push(Response { order: Some(order.clone()), ..Response::default() });
            }
            out
        }
        Msg::AnnounceNumber { values, .. } => (0..values.len()).map(idx).collect(),
        Msg::AnnounceRace { count: 1, available, .. } => (0..64).map(|i| 1u64 << i).filter(|b| available & b != 0).map(|b| Response { mask: Some(b), ..Response::default() }).collect(),
        Msg::AnnounceAttrib { count: 1, available, .. } => {
            (0..32).map(|i| 1u64 << i).filter(|b| *available as u64 & b != 0).map(|b| Response { mask: Some(b), ..Response::default() }).collect()
        }
        _ => vec![],
    };
    // (no duplicates; combinations at most ROOT_COMBO_WIDTH)
    let mut seen: Vec<Response> = Vec::new();
    out.retain(|r| {
        let new = !seen.contains(r);
        if new {
            seen.push(r.clone());
        }
        new
    });
    if !matches!(m, Msg::SelectPosition { .. } | Msg::AnnounceNumber { .. } | Msg::AnnounceRace { .. } | Msg::AnnounceAttrib { .. }) {
        out.truncate(ROOT_COMBO_WIDTH);
    }
    out
}

/// The k-element subsets of 0..n, in lexicographic order (the first elements first).
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k > n {
        return out;
    }
    let mut c: Vec<usize> = (0..k).collect();
    loop {
        out.push(c.clone());
        let Some(i) = (0..k).rev().find(|&i| c[i] != i + n - k) else { return out };
        c[i] += 1;
        for j in i + 1..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

/// The next order in lexicographic order (false after the last).
fn next_permutation(v: &mut [i64]) -> bool {
    let Some(i) = (1..v.len()).rev().find(|&i| v[i - 1] < v[i]) else { return false };
    let j = (i..v.len()).rev().find(|&j| v[j] > v[i - 1]).unwrap();
    v.swap(i - 1, j);
    v[i..].reverse();
    true
}

/// Up to ROOT_COMBO_WIDTH valid selections of a sum prompt, by a depth-first search over the cards
/// in `order` (the first cards first; a bounded number of steps).
fn sum_selections(m: &Msg, order: &[usize]) -> Vec<Vec<usize>> {
    let Msg::SelectSum { max, .. } = m else { return vec![] };
    #[allow(clippy::too_many_arguments)]
    fn dfs(m: &Msg, order: &[usize], start: usize, max: usize, chosen: &mut Vec<usize>, out: &mut Vec<Vec<usize>>, steps: &mut usize) {
        *steps += 1;
        if out.len() >= ROOT_COMBO_WIDTH || *steps > SUM_STEPS {
            return;
        }
        if !chosen.is_empty() && check_sum(m, chosen) {
            out.push(chosen.clone());
            return; // (a valid selection: not one with more cards on top)
        }
        if chosen.len() >= max {
            return;
        }
        for k in start..order.len() {
            chosen.push(order[k]);
            dfs(m, order, k + 1, max, chosen, out, steps);
            chosen.pop();
        }
    }
    let (mut out, mut steps) = (Vec::new(), 0);
    dfs(m, order, 0, (*max as usize).max(1), &mut Vec::new(), &mut out, &mut steps);
    out
}
