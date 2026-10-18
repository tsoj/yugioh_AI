//! The moves of a decision ("move generation"): which prompts the search decides, and their
//! candidates.

use crate::actions::{Response, battle_options, idle_options};
use crate::messages::Msg;

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
