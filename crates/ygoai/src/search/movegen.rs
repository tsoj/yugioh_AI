//! The moves of a decision ("move generation"): which prompts the search decides, and their
//! candidates (for the decisions inside a move, the rules' first few choices).

use ygo_core::constants::*;

use super::params::*;
use crate::actions::{Response, available_places, battle_options, check_sum, idle_options};
use crate::heuristic::{card, card_value};
use crate::messages::Msg;

/// The prompts searched over (belief_mc.SEARCHED) when they have more than one candidate.
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
    ) || is_inner(m)
}

/// A decision inside a move that a feature has the search make (see `Features`).
pub(super) fn is_inner(m: &Msg) -> bool {
    let f = Features::from_env();
    match m {
        Msg::SelectUnselectCard { .. } => f.unselect,
        Msg::SelectTribute { .. } => f.tribute,
        Msg::SelectSum { .. } => f.sum,
        Msg::SelectPosition { .. } => f.position,
        Msg::SelectPlace { count: 1, .. } => f.place,
        _ => false,
    }
}

fn idx(i: usize) -> Response {
    Response { index: Some(i as i64), ..Response::default() }
}

/// The candidates of a searched prompt (belief_mc.candidates).
pub fn candidates(m: &Msg) -> Vec<Response> {
    candidates_in(m, None)
}

/// The candidates, for a card selection with the cards in `order` (the rules' order by the select
/// hint; None: as listed): a decision inside a move has the rules' first few choices, best first.
pub fn candidates_in(m: &Msg, order: Option<&[usize]>) -> Vec<Response> {
    let f = Features::from_env();
    let listed = |n: usize| -> Vec<usize> { order.map_or_else(|| (0..n).collect(), |o| o.iter().copied().filter(|&i| i < n).collect()) };
    let pick = |idx: Vec<usize>| {
        let mut idx: Vec<i64> = idx.into_iter().map(|i| i as i64).collect();
        idx.sort();
        Response { indices: Some(idx), ..Response::default() }
    };
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
        Msg::SelectCard { min, max, cards, .. } if f.multicard && *max >= 1 && !cards.is_empty() => {
            // (the rules' count of the first cards, then other counts and other cards among the first few)
            let top: Vec<usize> = listed(cards.len()).into_iter().take(MULTI_CARDS).collect();
            let lo = (*min).max(1).min(*max).min(cards.len() as u32) as usize;
            let hi = (*max as usize).min(top.len()).max(lo.min(top.len()));
            let mut out = Vec::new();
            for k in std::iter::once(lo).chain((lo + 1..=hi).rev()) {
                if k <= top.len() {
                    for c in combinations(top.len(), k) {
                        out.push(c.into_iter().map(|j| top[j]).collect::<Vec<_>>());
                    }
                }
            }
            if lo > top.len() {
                out.push(listed(cards.len())[..lo].to_vec());
            }
            out.into_iter().take(MULTI_WIDTH).map(pick).collect()
        }
        Msg::SelectUnselectCard { finishable, cancelable, select_cards, .. } if f.unselect => {
            let mut out: Vec<Response> = listed(select_cards.len()).into_iter().take(UNSELECT_WIDTH).map(idx).collect();
            if *finishable != 0 || (*cancelable != 0 && select_cards.is_empty()) {
                out.push(Response { finish: Some(true), ..Response::default() });
            }
            out
        }
        Msg::SelectTribute { min, max, cards, .. } if f.tribute && !cards.is_empty() => {
            // (the least valuable sets that release enough, without a card to spare)
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
            sets.into_iter().take(TRIBUTE_WIDTH).map(|(_, s)| pick(s)).collect()
        }
        Msg::SelectSum { cards, .. } if f.sum && !cards.is_empty() => {
            // (valid selections, the least valuable cards tried first)
            let mut by_value: Vec<usize> = (0..cards.len()).collect();
            let v = |i: usize| card_value(&cards[i], card(cards[i].code));
            by_value.sort_by(|&a, &b| v(a).partial_cmp(&v(b)).unwrap_or(std::cmp::Ordering::Equal));
            sum_selections(m, &by_value).into_iter().map(pick).collect()
        }
        Msg::SelectPosition { code, positions, .. } if f.position => {
            let order = if card(*code).is_some_and(|c| c.attack <= 0) {
                [POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK]
            } else {
                [POS_FACEUP_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE, POS_FACEDOWN_ATTACK]
            };
            order.into_iter().filter(|p| positions & p != 0).map(|p| Response { position: Some(p as i64), ..Response::default() }).collect()
        }
        Msg::SelectPlace { player, count: 1, flag } if f.place => {
            available_places(*player, *flag).into_iter().map(|pl| Response { places: Some(vec![pl]), ..Response::default() }).collect()
        }
        _ => vec![],
    }
}

/// The k-element subsets of 0..n, in lexicographic order (the first elements first).
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut c: Vec<usize> = (0..k).collect();
    if k > n {
        return out;
    }
    loop {
        out.push(c.clone());
        let Some(i) = (0..k).rev().find(|&i| c[i] != i + n - k) else { return out };
        c[i] += 1;
        for j in i + 1..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

/// Up to SUM_WIDTH valid selections of a sum prompt, by a depth-first search over the cards in
/// `order` (the first cards first; a bounded number of steps).
fn sum_selections(m: &Msg, order: &[usize]) -> Vec<Vec<usize>> {
    let Msg::SelectSum { max, .. } = m else { return vec![] };
    fn dfs(m: &Msg, order: &[usize], start: usize, max: usize, chosen: &mut Vec<usize>, out: &mut Vec<Vec<usize>>, steps: &mut usize) {
        *steps += 1;
        if out.len() >= SUM_WIDTH || *steps > SUM_STEPS {
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
