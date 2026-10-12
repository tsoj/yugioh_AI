//! Diagnosis of a divergence: which repair could explain it (BeliefReplay._mismatch and the
//! *_fix helpers). The helpers read the stream's messages and the engine's as JSON views, as the
//! Python module reads its dicts.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::Value;
use ygo_core::constants::*;

use super::step::{card_codes, is_activation_prelude_t, next_activation};
use super::util::{V, jv, key3, li_of};
use super::*;
use crate::belief::completion::{card, is_extra_deck};
use crate::belief::infer::{self, Li};
use crate::belief::tracker::Kind;
use crate::messages::Msg;
use crate::state::{LOCATIONS, parse_location_query};

fn li(con: u32, loc: u32, seq: u32) -> Li {
    Li { con, loc, seq, pos: 0 }
}

pub fn cd_type(code: u32) -> u32 {
    card(code).map_or(0, |c| c.type_)
}

impl BeliefReplay {
    /// The JSON view of item j's message (Null for prompts and responses).
    pub fn iv(&self, j: usize) -> Value {
        match self.items.get(j) {
            Some(ObsItem::Msg(m)) => jv(m),
            _ => Value::Null,
        }
    }

    /// The message of item j.
    pub fn imsg(&self, j: usize) -> Option<&Msg> {
        match self.items.get(j) {
            Some(ObsItem::Msg(m)) => Some(m),
            _ => None,
        }
    }

    fn itype(&self, j: usize) -> &str {
        self.imsg(j).map_or("", |m| m.name())
    }

    /// The solved tracker's claim for the lockstep var at a slot: (lockstep var, its member in the
    /// solution).
    fn solved_member(&mut self, con: u32, loc: u32, seq: u32) -> Option<(VarId, VarId)> {
        let solved = self.solved.clone()?;
        let v = self.lock_var(con, loc, seq).ok().flatten()?;
        let sv = solved.vars.get(v)?;
        if sv.kind != Kind::Claim {
            return None;
        }
        Some((v, sv.assigned?))
    }

    fn ident_fix(&self, entry: Ident) -> Option<Fix> {
        if self.identities.contains(&entry) || self.failed_identities.contains(&entry) {
            return None;
        }
        Some(Fix::Identify(entry))
    }

    pub fn force_fix(&self, expected: &Value) -> Option<Fix> {
        if expected.t() == "CHAINING" && expected.i("trigger_con") as u32 == self.opponent {
            let key = (self.opponent, expected.i("code") as u32, expected.i("desc") as u64);
            if !self.forces.contains(&key) && !self.unforced.contains(&key) {
                return Some(Fix::Force(key));
            }
        }
        None
    }

    /// A recent card selection of the opponent must have included the card at each of `locs`.
    fn include_fix(&self, locs: &[Value], duel: &RDuel) -> Option<Fix> {
        let live = self.loc_map(duel);
        for l in locs {
            let Some(cid) = live.at(&li_of(l)).filter(|&c| c != 0) else { continue };
            for &(dkey, _, _) in self.recent.iter().rev() {
                let Some((ids, resp)) = self.multi_decisions.get(&dkey) else { continue };
                let Some(j) = ids.iter().position(|&x| x == Some(cid)) else { continue };
                let picks: Vec<usize> = resp.indices.clone().unwrap_or_default().iter().map(|&i| i as usize).collect();
                if picks.contains(&j) {
                    continue;
                }
                for p in (0..picks.len()).rev() {
                    let mut new: Vec<usize> = picks[..p].to_vec();
                    new.push(j);
                    new.extend_from_slice(&picks[p + 1..]);
                    new.sort();
                    if !self.tried_resp.contains(&(dkey, new.clone())) {
                        return Some(Fix::Respond(dkey, new));
                    }
                }
            }
        }
        None
    }

    /// The opponent's activation is forced and still being activated: let the hidden cards qualify.
    pub fn forced_cost_fix(&mut self) -> Option<Fix> {
        let start = self.chain_start()?;
        if self.match_piles.contains(&(start, self.pos)) {
            return None;
        }
        let ch = self.iv(start);
        if !self.active_forces.contains(&(self.opponent, ch.i("code") as u32, ch.i("desc") as u64)) {
            return None;
        }
        if (start..self.pos).any(|j| self.itype(j) == "CHAINED") {
            return None;
        }
        if let Some(pc) = self.prompt_cp
            && pc.0 == start
        {
            self.match_pile_from.insert((start, self.pos), pc);
        }
        Some(Fix::MatchPile(start, self.pos))
    }

    /// The replay activated a forced effect that the real game did not.
    pub fn unforce_fix(&self, got: &Value) -> Option<Fix> {
        if got.t() == "CHAINING" && got.i("trigger_con") as u32 == self.opponent {
            let key = (self.opponent, got.i("code") as u32, got.i("desc") as u64);
            if self.active_forces.contains(&key) {
                return Some(Fix::Unforce(key));
            }
        }
        let lo = self.pos as i64 - 12;
        let mut j = self.pos as i64 - 1;
        while j > lo.max(-1) {
            let ju = j as usize;
            let Some(m) = self.imsg(ju) else { break };
            if matches!(m, Msg::Chained { .. } | Msg::ChainSolving { .. } | Msg::ChainEnd {}) {
                break;
            }
            if let Msg::Chaining { code, desc, trigger_con, .. } = m {
                let key = (self.opponent, *code, *desc);
                if *trigger_con == self.opponent && self.active_forces.contains(&key) && self.force_enabled.contains(&ju) {
                    return Some(Fix::Unforce(key));
                }
                break;
            }
            j -= 1;
        }
        None
    }

    pub fn mismatch(&mut self, cp: Cp, msgs: &[Msg], idx: usize, item: &ObsItem, got_m: &Msg, duel: &mut RDuel) -> Divergence {
        let got = jv(got_m);
        let mut fix: Option<Fix> = None;
        let mut faceup_match: Option<Fix> = None;
        let gt = got.t().to_string();
        if let ObsItem::Msg(em) = item {
            let exp = jv(em);
            let et = exp.t().to_string();
            fix = self.force_fix(&exp);
            if fix.is_none()
                && et == "MOVE"
                && gt == "MOVE"
                && exp.g("code").truthy()
                && exp.g("code") == got.g("code")
                && exp.g("from") == got.g("from")
                && exp.g("to").i("loc") == got.g("to").i("loc")
                && exp.g("to").i("con") != got.g("to").i("con")
                && hidden_pile(exp.g("from").i("loc") as u32)
            {
                // (the same card goes to another player's pile: the other copy)
                let live = self.loc_map(duel);
                let a = live.at(&li_of(got.g("to"))).filter(|&c| c != 0);
                let codes = card_codes(duel);
                let mixed: BTreeSet<Cp> = a.and_then(|a| self.card_shuffled.get(&a)).map_or(BTreeSet::new(), |v| v.iter().copied().collect());
                let shuffled: Vec<(u32, Vec<Cp>)> = self.card_shuffled.iter().map(|(&k, v)| (k, v.clone())).collect();
                for (b, at) in shuffled {
                    if Some(b) != a && codes.get(&b).copied() == Some(exp.i("code") as u32) && at.iter().any(|c| mixed.contains(c)) {
                        fix = a.and_then(|a| self.history_swap(a, b, cp));
                        if fix.is_some() {
                            break;
                        }
                    }
                }
            }
            if fix.is_none() && et == "HINT" && gt == "HINT" && exp.i("hint_type") as u32 == HINT_EVENT && got.i("hint_type") as u32 == HINT_EVENT && exp.g("data") != got.g("data") {
                // the real game is in another step than the replay
                for &(dkey, passing) in self.window_activations.iter().rev() {
                    if self.overrides.get(&dkey) != Some(&passing) {
                        fix = Some(Fix::Decide(dkey, passing));
                        break;
                    }
                }
            }
            if fix.is_none() && et == "SHUFFLE_HAND" && gt == "CHAIN_SOLVED" {
                fix = self.activated_copy_fix(&exp, duel, cp);
            }
            if fix.is_none() && gt == "SHUFFLE_HAND" && et != "SHUFFLE_HAND" && got.i("player") as u32 == self.opponent {
                fix = self.moved_copy_fix(duel, cp);
            }
            if fix.is_none() && et == "MOVE" && matches!(gt.as_str(), "CHAIN_SOLVED" | "CHAIN_SOLVING" | "CHAIN_END") && hidden_pile(exp.g("from").i("loc") as u32) {
                fix = self.unmoved_copy_fix(&exp, duel, cp);
            }
            if fix.is_none()
                && et == "CHAINING"
                && gt == "CHAINING"
                && exp.g("code") == got.g("code")
                && exp.g("desc") == got.g("desc")
                && ["con", "loc", "seq", "trigger_con", "trigger_loc"].iter().all(|k| exp.g(k) == got.g(k))
                && exp.i("trigger_loc") as u32 == LOCATION_HAND
                && exp.g("trigger_seq") != got.g("trigger_seq")
            {
                fix = self.stale_trigger_fix(&exp);
            }
            if fix.is_none() && et == "CHAINING" && exp.i("trigger_con") as u32 == self.opponent && exp.i("trigger_loc") as u32 == LOCATION_HAND && gt != "CHAINING" {
                fix = self.arrival_fix(exp.i("trigger_con") as u32, exp.i("trigger_loc") as u32, exp.i("trigger_seq") as u32, exp.i("code") as u32);
            }
            if fix.is_none()
                && et == "CHAINING"
                && exp.i("trigger_con") as u32 == self.opponent
                && matches!(gt.as_str(), "CHAIN_SOLVING" | "CHAIN_SOLVED" | "CHAIN_END" | "ATTACK_DISABLED" | "DAMAGE_STEP_START" | "BATTLE")
                && self.active_forces.contains(&(self.opponent, exp.i("code") as u32, exp.i("desc") as u64))
            {
                fix = self.unavailable_fix();
            }
            if fix.is_none()
                && et == gt
                && (matches!(et.as_str(), "FLIPSUMMONING" | "POS_CHANGE") || et == "SET" && exp.i("loc") as u32 == LOCATION_HAND)
                && exp.i("con") as u32 == self.opponent
                && got.i("con") as u32 == self.opponent
                && exp.i("loc") == got.i("loc")
                && exp.i("seq") != got.i("seq")
            {
                fix = self.slot_identity_fix(key3(&exp), key3(&got), true);
            }
            if fix.is_none()
                && et == gt
                && matches!(et.as_str(), "FLIPSUMMONING" | "POS_CHANGE")
                && exp.g("code").truthy()
                && exp.g("code") == got.g("code")
                && exp.i("con") == got.i("con")
                && exp.i("loc") == got.i("loc")
                && exp.i("seq") != got.i("seq")
                && self.last_shuffle_cp.contains_key(&(exp.i("con") as u32, exp.i("loc") as u32))
            {
                // another face-down copy, the copies shuffled in their zones since
                let live = self.loc_map(duel);
                let (a, b) = (live.at(&li_of(&got)).unwrap_or(0), live.at(&li_of(&exp)).unwrap_or(0));
                if let Some(at) = self.prompt_cp
                    && a != 0
                    && b != 0
                    && a != b
                    && !self.failed_swaps.contains(&(at, a, b))
                    && !self.failed_swaps.contains(&(at, b, a))
                    && !self.swaps.get(&at).is_some_and(|l| l.contains(&(a, b)) || l.contains(&(b, a)))
                {
                    fix = Some(Fix::Swap(at, a, b));
                }
            }
            if fix.is_none() && et == "MOVE" {
                // (a card assumed for this very action whose consequences differ: first another candidate)
                let recent: Vec<Asm> =
                    self.assumptions.iter().copied().filter(|a| self.pos.saturating_sub(2) <= a.0 && a.0 <= self.pos && a.2 != LOCATION_DECK && a.2 != LOCATION_EXTRA).collect();
                if let Some(&last) = recent.last() {
                    fix = Some(Fix::Unassume(last));
                }
            }
            if fix.is_none() && et == "MOVE" && gt == "MOVE" {
                fix = self.hand_slot_fix(&exp, &got);
            }
            if fix.is_none() && et == "MOVE" {
                fix = self.hidden_monster_fix(&exp, &got);
            }
            if fix.is_none() && et == "MOVE" && exp.g("from").i("con") as u32 == self.opponent {
                fix = self.tribute_summoner_fix(&exp, &got, duel);
            }
            if fix.is_none()
                && et == "MOVE"
                && exp.i("reason") as u32 & REASON_RELEASE != 0
                && exp.g("from").i("con") as u32 == self.opponent
                && gt == "MOVE"
                && got.g("from").i("con") as u32 == self.opponent
                && got.g("from").i("loc") as u32 == LOCATION_HAND
                && got.g("to").i("loc") as u32 == LOCATION_MZONE
                && !got.g("code").truthy()
            {
                // (the hidden card is summoned or Set with fewer Tributes than the real one takes)
                let live = self.loc_map(duel);
                let cid = live.at(&li_of(got.g("to"))).filter(|&c| c != 0);
                let code = cid.and_then(|c| live.codes.get(&c).copied());
                if let Some(code) = code.filter(|&c| c != 0) {
                    let key: Excl = (self.pos, got.g("from").i("con") as u32, LOCATION_HAND, vec![got.g("from").i("seq")], code);
                    if !self.exclusions.contains(&key) && !self.failed_exclusions.contains(&key) {
                        fix = Some(Fix::Exclude(key));
                    }
                }
            }
            if fix.is_none() && gt != "MOVE" {
                fix = self.granted_set_fix(&exp, duel);
            }
            if fix.is_none() && et == "MOVE" && gt == "MOVE" {
                fix = self.summon_copy_fix(&exp, &got);
            }
            if fix.is_none() && et == "MOVE" && gt == "MOVE" && matches!(exp.g("from").i("loc") as u32, LOCATION_DECK | LOCATION_EXTRA) && exp != got {
                // (a group taken out of hidden piles by one operation: before the per-card repairs)
                fix = self.group_match_fix(&exp, &got, cp, msgs, idx);
            }
            if fix.is_none() && is_activation_prelude_t(&et) {
                // the start of an activation: the replay did not activate it
                let up = self.upcoming();
                if let Some(ch) = next_activation(&up) {
                    let chv = jv(&ch);
                    fix = self.force_fix(&chv);
                    if fix.is_none() && chv.i("trigger_con") as u32 == self.opponent && self.active_forces.contains(&(self.opponent, chv.i("code") as u32, chv.i("desc") as u64)) {
                        fix = self.unavailable_fix();
                    }
                }
            }
            if fix.is_none() && !(et == "CHAINING" && exp.g("code") == got.g("code") && exp.g("desc") == got.g("desc")) {
                fix = self.unforce_fix(&got);
            }
            if fix.is_none() && gt == "CARD_HINT" && et == "CARD_HINT" {
                fix = self.hint_swap_fix(cp, &got, duel);
            }
            if fix.is_none()
                && et == gt
                && matches!(et.as_str(), "SHUFFLE_HAND" | "SHUFFLE_EXTRA")
                && exp.g("public") != got.g("public")
                && exp.g("codes").l().len() == got.g("codes").l().len()
            {
                let (ec, gc) = (exp.g("codes").l(), got.g("codes").l());
                let (ep, gp) = (exp.g("public").l(), got.g("public").l());
                let same = ec == gc || ec.iter().zip(gc).zip(ep.iter().zip(gp)).filter(|(_, (x, y))| x.truthy() && y.truthy()).all(|((a, b), _)| a == b);
                if same {
                    // another copy of a revealed card ended up in its place: swap the two copies
                    let loc = if et == "SHUFFLE_HAND" { LOCATION_HAND } else { LOCATION_EXTRA };
                    let cards = duel.location_cards(exp.i("player") as u32, loc);
                    let i = (0..ep.len()).find(|&i| ep[i].truthy() && !gp.get(i).is_some_and(|x| x.truthy()));
                    let k = i.and_then(|i| {
                        (0..gp.len()).find(|&k| {
                            gp[k].truthy() && !ep.get(k).is_some_and(|x| x.truthy()) && (if gc[k].truthy() { &gc[k] } else { &ec[k] }) == &ec[i]
                        })
                    });
                    if let (Some(i), Some(k)) = (i, k)
                        && k < cards.len()
                        && i < cards.len()
                    {
                        let (a, b) = (cards[k].1, cards[i].1);
                        if !self.failed_swaps.contains(&(cp, a, b))
                            && !self.failed_swaps.contains(&(cp, b, a))
                            && !self.swaps.get(&cp).is_some_and(|l| l.contains(&(a, b)) || l.contains(&(b, a)))
                        {
                            fix = Some(Fix::Swap(cp, a, b));
                        }
                    }
                }
            }
            let mut hinted: Option<(&Value, bool)> = if et == "CARD_HINT" && gt != "CARD_HINT" {
                Some((&exp, true))
            } else if gt == "CARD_HINT" && et != "CARD_HINT" {
                Some((&got, false))
            } else {
                None
            };
            if hinted.is_none()
                && fix.is_none()
                && et == "CARD_HINT"
                && gt == "CARD_HINT"
                && !(self.pos..(self.pos + 12).min(self.items.len())).any(|j| self.itype(j) == "CARD_HINT" && infer::same_loc(&li_of(&self.iv(j)), &li_of(&got)))
            {
                // (the replay shows a hint on a card where the real game shows none)
                hinted = Some((&got, false));
            }
            if fix.is_none()
                && let Some((h, is_exp)) = hinted
            {
                let live = self.loc_map(duel);
                if let Some(a) = live.at(&li_of(h)).filter(|&c| c != 0) {
                    let ac = live.codes.get(&a).copied();
                    let others: Vec<(usize, u32)> =
                        duel.location_cards(h.i("con") as u32, LOCATION_HAND).into_iter().enumerate().filter(|&(_, (code, cid))| cid != a && Some(code) == ac).map(|(i, (_, c))| (i, c)).collect();
                    if is_exp {
                        // (as an identity: the card that just moved there is the physical card the
                        // other copy is in the replay)
                        for &(i, _) in &others {
                            fix = self.moved_identify_fix(key3(h), (h.i("con") as u32, LOCATION_HAND, i as u32));
                            if fix.is_some() {
                                break;
                            }
                        }
                    }
                    for &(_, b) in &others {
                        if fix.is_some() {
                            break;
                        }
                        fix = self.history_swap(a, b, cp);
                    }
                }
                if fix.is_none() && !is_exp && got.i("hint_type") as u32 == CHINT_DESC_REMOVE {
                    fix = self.unhinted_fix(&got);
                }
                if fix.is_none() && is_exp && exp.i("hint_type") as u32 == CHINT_DESC_REMOVE {
                    fix = self.mark_fix(&exp, duel);
                }
            }
            if fix.is_none() && et == gt && exp.g("cards").is_array() && got.g("cards").is_array() && exp.g("cards").l().len() == got.g("cards").l().len() {
                // the same card in another place of a pile: another copy was chosen
                for (e, g) in exp.g("cards").l().iter().zip(got.g("cards").l()) {
                    if e == g {
                        continue;
                    }
                    if e.g("code") == g.g("code")
                        && e.i("con") == g.i("con")
                        && e.i("loc") == g.i("loc")
                        && e.i("seq") != g.i("seq")
                        && (hidden_pile(e.i("loc") as u32) || matches!(e.i("loc") as u32, LOCATION_GRAVE | LOCATION_REMOVED))
                    {
                        fix = self.swap_fix(cp, msgs, idx, e, g, duel);
                    }
                    break;
                }
            }
            let (ef, gf) = (exp.g("from"), got.g("from"));
            if fix.is_none()
                && et == "MOVE"
                && gt == "MOVE"
                && (!exp.g("code").truthy() || exp.g("code") != got.g("code"))
                && ef.i("con") == gf.i("con")
                && ef.i("loc") == gf.i("loc")
                && ef.i("seq") != gf.i("seq")
                && hidden_pile(ef.i("loc") as u32)
                && exp.g("to") == got.g("to")
            {
                fix = self.revealed_fix(key3(ef));
            }
            if fix.is_none()
                && et == "MOVE"
                && gt == "MOVE"
                && exp.g("code").truthy()
                && exp.g("code") == got.g("code")
                && ef.i("con") as u32 == self.opponent
                && gf.i("con") as u32 == self.opponent
                && ef.i("loc") as u32 == LOCATION_HAND
                && gf.i("loc") as u32 == LOCATION_HAND
                && ef.i("seq") != gf.i("seq")
                && exp.g("to") == got.g("to")
            {
                fix = self.slot_identity_fix(key3(ef), key3(gf), true);
            }
            if fix.is_none() && et == gt && exp.g("code") == got.g("code") {
                let (le, lg) = (loc_of(&exp), loc_of(&got));
                if let (Some(le), Some(lg)) = (le, lg)
                    && le.i("con") == lg.i("con")
                    && le.i("loc") == lg.i("loc")
                    && (hidden_pile(le.i("loc") as u32) || matches!(le.i("loc") as u32, LOCATION_GRAVE | LOCATION_REMOVED))
                {
                    let mcode = jv(&msgs[idx]).o("code");
                    if le.i("seq") != lg.i("seq") {
                        if le.i("con") as u32 == self.viewer && le.i("loc") as u32 == LOCATION_DECK && lg.i("seq") < le.i("seq") && !self.viewer_prompted() {
                            // (the viewer's own deck, no selection of the viewer: none further down)
                            fix = self.exclude_fix(le, mcode);
                        }
                        if fix.is_none() {
                            fix = self.swap_fix(cp, msgs, idx, le, lg, duel);
                        }
                        if (fix.is_none() || fix.as_ref().is_some_and(|f| self.has_swap(f))) && lg.i("seq") < le.i("seq") {
                            // the replay's scan hit this code lower down than the real one
                            fix = self.exclude_fix(le, mcode);
                        }
                        if (fix.is_none() || fix.as_ref().is_some_and(|f| self.has_swap(f)))
                            && le.i("con") as u32 == self.opponent
                            && le.i("loc") as u32 == LOCATION_DECK
                            && exp.g("code").truthy()
                        {
                            fix = self.all_copies_fix(&exp, le, duel, Some(&got)).or(fix);
                        }
                    } else if exp.g("code").truthy() {
                        // right card, wrong copy (the copies differ in hidden history)
                        fix = self.other_copy_fix(cp, msgs, idx, le);
                    }
                }
            }
            if fix.is_none()
                && et == "MOVE"
                && gt == "MOVE"
                && gf.i("con") as u32 == self.opponent
                && ef.i("con") == gf.i("con")
                && ef.i("loc") == gf.i("loc")
                && ef.i("seq") != gf.i("seq")
                && hidden_pile(gf.i("loc") as u32)
                && exp.g("code").truthy()
                && got.g("code").truthy()
                && exp.g("code") != got.g("code")
            {
                fix = self.nomatch_card_fix(key3(gf), Some(duel), Some(got.g("to")));
            }
            if fix.is_none() && gt == "CONFIRM_CARDS" && et != "CONFIRM_CARDS" {
                fix = self.win_reveal_fix(&got);
            }
            if fix.is_none()
                && gt == "MOVE"
                && gf.i("con") as u32 == self.opponent
                && got.g("code").truthy()
                && (hidden_pile(gf.i("loc") as u32) || gf.i("loc") as u32 & LOCATION_ONFIELD != 0 && gf.i("pos") as u32 & POS_FACEDOWN != 0)
                && !(et == "MOVE" && ef == gf)
            {
                fix = self.self_move_fix(&got, duel);
            }
            if fix.is_none() && gt == "MOVE" && gf.i("con") as u32 == self.opponent && hidden_pile(gf.i("loc") as u32) && !(et == "MOVE" && ef.i("con") == gf.i("con") && ef.i("loc") == gf.i("loc")) {
                // the replay took a hidden card of the opponent where the real game took none
                let start = self.chain_start();
                let mut span = self.nomatch_span(start);
                if span.is_none() {
                    let s = (self.action_start(), self.pos);
                    if !self.nomatch_piles.contains(&s) && !self.match_piles.contains(&s) {
                        span = Some(s);
                    }
                }
                fix = match span {
                    Some(s) => Some(Fix::NomatchPile(s.0, s.1)),
                    None => self.nomatch_card_fix(key3(gf), Some(duel), Some(got.g("to"))),
                };
            }
            if fix.is_none()
                && et == "MOVE"
                && gt == "MOVE"
                && ef.i("con") == gf.i("con")
                && ef.i("loc") == gf.i("loc")
                && ef.i("seq") != gf.i("seq")
                && ef.i("loc") as u32 == LOCATION_HAND
                && exp.g("to") == got.g("to")
            {
                // the replay's script moved a specific physical card from where the hypothesis
                // shuffled it
                fix = if !exp.g("code").truthy() { self.revealed_fix(key3(ef)) } else { None };
                if fix.is_none() {
                    fix = self.identify_fix(key3(ef), key3(gf));
                }
            }
            if fix.is_none()
                && et == "MOVE"
                && exp.g("code").truthy()
                && hidden_pile(ef.i("loc") as u32)
                && let Some(op) = self.opp_prompt.clone()
            {
                // the opponent chose among cards offered to it, one of them a copy of the card
                let opv = jv(&op);
                let cards: Vec<Value> = if opv.g("cards").truthy() { opv.g("cards").l().to_vec() } else { opv.g("select_cards").l().to_vec() };
                for c in &cards {
                    if c.g("code") == exp.g("code") && c.i("con") == ef.i("con") && c.i("loc") == ef.i("loc") && c.i("seq") != ef.i("seq") {
                        fix = self.swap_fix(cp, msgs, idx, ef, c, duel);
                        if fix.is_some() {
                            break;
                        }
                    }
                }
            }
            if fix.is_none() && et == "MOVE" && gt == "MOVE" && got.g("code").truthy() && exp.g("code") != got.g("code") && ef.i("loc") == gf.i("loc") && hidden_pile(gf.i("loc") as u32) {
                fix = self.later_copy_fix(cp, msgs, idx, &got, duel);
            }
            if fix.is_none() && et == "RANDOM_SELECTED" && gt == "RANDOM_SELECTED" {
                let missing: Vec<Value> = exp.g("cards").l().iter().filter(|c| !got.g("cards").l().contains(c)).cloned().collect();
                fix = self.include_fix(&missing, duel);
            }
            if fix.is_none() && et == "HINT" && exp.i("hint_type") as u32 == HINT_OPSELECTED && exp.i("player") as u32 == self.opponent && exp != got {
                // the opponent chose an option the replay did not offer or take
                let start = self.chain_start();
                if let Some(s) = start {
                    fix = [(s + 1).min(self.pos), s].into_iter().find(|&a| !self.match_piles.contains(&(a, self.pos))).map(|a| Fix::MatchPile(a, self.pos));
                }
                if fix.is_none()
                    && let Some(s) = start
                {
                    let code = self.iv(s).i("code") as u32;
                    fix = self.property_fix(code, duel);
                }
            }
            let revealed: Option<Value> = if gt == "POS_CHANGE" && got.i("prev_pos") as u32 & POS_FACEDOWN != 0 && got.i("pos") as u32 & POS_FACEUP != 0 {
                Some(got.clone())
            } else if gt == "MOVE" && matches!(gf.i("loc") as u32, LOCATION_MZONE | LOCATION_SZONE) && gf.i("pos") as u32 & POS_FACEDOWN != 0 {
                Some(gf.clone())
            } else {
                None
            };
            if fix.is_none()
                && let Some(r) = &revealed
                && et != gt
                && r.i("con") as u32 == self.opponent
                && got.g("code").truthy()
            {
                // the replay showed the opponent's face-down card where the real game did not
                let key: Excl = (self.pos, r.i("con") as u32, r.i("loc") as u32, vec![r.i("seq")], got.i("code") as u32);
                if !self.exclusions.contains(&key) && !self.failed_exclusions.contains(&key) {
                    fix = Some(Fix::Exclude(key));
                }
            }
            if fix.is_none() && et == "CONFIRM_CARDS" && exp.g("cards").truthy() && exp.g("cards").l()[0].i("con") as u32 == self.opponent && {
                let ek: BTreeSet<(u32, u32, u32)> = exp.g("cards").l().iter().map(key3).collect();
                let gk: BTreeSet<(u32, u32, u32)> = got.g("cards").l().iter().map(key3).collect();
                gt != "CONFIRM_CARDS" || ek.difference(&gk).next().is_some()
            } {
                // the opponent revealed a card that the replay's filter rejects
                let shown: Vec<(u32, u32, u32)> = if gt == "CONFIRM_CARDS" { got.g("cards").l().iter().map(key3).collect() } else { vec![] };
                fix = self.forced_cost_fix();
                if fix.is_none() {
                    for c in exp.g("cards").l() {
                        if !shown.contains(&key3(c)) {
                            fix = self.match_fix(cp, msgs, idx, c, None);
                            if fix.is_some() {
                                break;
                            }
                        }
                    }
                }
                if fix.is_none()
                    && let Some(s) = self.chain_start()
                    && !self.match_piles.contains(&(s, self.pos))
                {
                    fix = Some(Fix::MatchPile(s, self.pos));
                }
            }
            if fix.is_none() && et == "BECOME_TARGET" && gt == "BECOME_TARGET" && exp.g("cards").l().len() == 1 && got.g("cards").l().len() == 1 {
                let (e, g) = (&exp.g("cards").l()[0], &got.g("cards").l()[0]);
                if e.i("con") == g.i("con") && e.i("loc") == g.i("loc") && e.i("seq") != g.i("seq") && (matches!(e.i("loc") as u32, LOCATION_GRAVE | LOCATION_REMOVED) || hidden_pile(e.i("loc") as u32)) {
                    fix = self.swap_fix(cp, msgs, idx, e, g, duel);
                }
            }
            if fix.is_none() && nothing_more(&et) && !nothing_more(&gt) && gt != et && gt != "DECK_TOP" {
                // the replay did something more where the real game went on
                let start = self.chain_start();
                if let Some(s) = self.nomatch_span(start) {
                    fix = Some(Fix::NomatchPile(s.0, s.1));
                }
            }
            if fix.is_none() && gt == "CHAINING" && got.i("trigger_con") as u32 == self.opponent && !(et == "CHAINING" && exp.g("code") == got.g("code")) && self.unforce_fix(&got).is_none() {
                // the replay's opponent activated an effect whose check passes thanks to the hidden cards
                let span = (self.action_start(), self.pos);
                if !self.nomatch_piles.contains(&span) && !self.match_piles.contains(&span) {
                    fix = Some(Fix::NomatchPile(span.0, span.1));
                }
            }
            if fix.is_none() && et == "WIN" && gt != "WIN" {
                // the real game ended where the replay went on
                if let Some(s) = self.chain_start() {
                    'outer: for a in [s.max(self.pos.saturating_sub(1)), s] {
                        for kind in ["matchpile", "nomatchpile"] {
                            if !self.match_piles.contains(&(a, self.pos)) && !self.nomatch_piles.contains(&(a, self.pos)) {
                                fix = Some(if kind == "matchpile" { Fix::MatchPile(a, self.pos) } else { Fix::NomatchPile(a, self.pos) });
                                break 'outer;
                            }
                        }
                    }
                }
            }
            if fix.is_none() && gt == "DECK_TOP" && et != "DECK_TOP" {
                fix = self.not_top_fix(got.i("player") as u32);
            }
            if fix.is_none()
                && gt == "HINT"
                && matches!(got.i("hint_type") as u32, HINT_SELECTMSG | HINT_CARD)
                && (got.i("hint_type") as u32 == HINT_CARD || got.i("player") as u32 == self.opponent)
                && et != "HINT"
                && !(self.pos..(self.pos + 6).min(self.items.len())).any(|j| self.imsg(j).is_some_and(|m| m == got_m))
            {
                // the replay's opponent goes on to select where the real game does not
                let phase = (self.pos.saturating_sub(12)..self.pos).rev().find(|&j| matches!(self.itype(j), "NEW_PHASE" | "NEW_TURN"));
                let phase = if self.pos >= 12 { phase } else { (0..self.pos).rev().find(|&j| matches!(self.itype(j), "NEW_PHASE" | "NEW_TURN")) };
                let cs = self.chain_start();
                let mut spans: Vec<Option<(usize, usize)>> = vec![self.nomatch_span(cs), Some((self.action_start(), self.pos))];
                if let Some(ph) = phase {
                    let end = (self.pos..self.items.len().min(self.pos + 60)).find(|&j| matches!(self.itype(j), "NEW_PHASE" | "NEW_TURN")).unwrap_or(self.pos);
                    let at = if cs.is_none() { 0 } else { 2 };
                    spans.insert(at.min(spans.len()), Some((ph, end.max(self.pos))));
                }
                for span in spans.into_iter().flatten() {
                    if fix.is_none() && !self.nomatch_piles.contains(&span) && !self.match_piles.contains(&span) {
                        fix = Some(Fix::NomatchPile(span.0, span.1));
                    }
                }
            }
            if fix.is_none() && gt == "CARD_SELECTED" && et != "CARD_SELECTED" {
                let cs = self.chain_start();
                if let Some(s) = self.nomatch_span(cs) {
                    fix = Some(Fix::NomatchPile(s.0, s.1));
                }
            }
            if fix.is_none()
                && et == "HINT"
                && gt == "HINT"
                && exp.i("hint_type") == got.i("hint_type")
                && got.i("hint_type") as u32 == HINT_SELECTMSG
                && exp.i("player") == got.i("player")
                && got.i("player") as u32 == self.viewer
                && exp.g("data") != got.g("data")
            {
                // the viewer's own selection hint names the card it handles
                fix = self.hint_code_fix(exp.i("data") as u32, got.i("data") as u32, duel);
                if fix.is_none() {
                    for loc in [LOCATION_HAND, LOCATION_DECK, LOCATION_GRAVE] {
                        let ids: Vec<u32> = duel.location_cards(self.viewer, loc).into_iter().filter(|&(c, _)| c as i64 == got.i("data")).map(|x| x.1).collect();
                        for i in 0..ids.len() {
                            for &b in &ids[i + 1..] {
                                if fix.is_none() {
                                    fix = self.history_swap(ids[i], b, cp);
                                }
                            }
                        }
                        if fix.is_some() {
                            break;
                        }
                    }
                }
            }
            if fix.is_none() && gt == "HINT" && got.i("hint_type") as u32 == HINT_OPSELECTED && got.i("player") as u32 == self.opponent && exp != got && !(et == "HINT" && exp.i("hint_type") as u32 == HINT_OPSELECTED) {
                // the replay offered the opponent a choice of options that the real game did not
                let start = self.chain_start();
                if let Some(span) = self.nomatch_span(start) {
                    fix = [LOCATION_HAND, LOCATION_DECK].into_iter().find(|&l| !self.nomatch_loc_piles.contains(&(span.0, span.1, l))).map(|l| Fix::NomatchLoc(span.0, span.1, l));
                    if fix.is_none() {
                        fix = Some(Fix::NomatchPile(span.0, span.1));
                    }
                }
            }
            if fix.is_none() && et == "BECOME_TARGET" && exp.g("cards").truthy() && (gt != "BECOME_TARGET" || got.g("cards") != exp.g("cards")) {
                // the opponent targeted a card that the replay's filter rejects: let it match
                let c0 = exp.g("cards").l()[0].clone();
                fix = self.match_fix(cp, msgs, idx, &c0, Some(duel));
                let mut start = self.pos;
                while start > 0 && self.itype(start - 1) == "BECOME_TARGET" {
                    start -= 1;
                }
                if let Some(Fix::Match(_, cid)) = fix
                    && start < self.pos
                {
                    // (the targets of one selection are announced one by one)
                    fix = if self.matches.get(&start).is_some_and(|l| l.contains(&cid)) { None } else { Some(Fix::Match((start, 0), cid)) };
                }
            }
            if fix.is_none() && et == "MOVE" && ef.i("con") as u32 == self.opponent && hidden_pile(ef.i("loc") as u32) && exp != got {
                fix = self.kind_fix(cp, msgs, idx, &exp);
                if fix.is_none() {
                    fix = self.match_fix(cp, msgs, idx, ef, None);
                }
                if fix.is_none() && !exp.g("code").truthy() {
                    // several hidden cards must qualify together: let all of them qualify
                    let start = self.chain_start().or_else(|| self.resolving_link());
                    if let Some(s) = start
                        && !self.match_piles.contains(&(s, self.pos))
                    {
                        fix = Some(Fix::MatchPile(s, self.pos));
                    }
                }
            } else if fix.is_none() && et == "MOVE" && ef.i("con") as u32 == self.opponent && ef.i("loc") != 0 && ef.i("loc") as u32 & LOCATION_OVERLAY == 0 && exp != got {
                // a public card of the opponent the replay did not take
                if ef.i("loc") as u32 & (LOCATION_MZONE | LOCATION_SZONE) != 0 && ef.i("pos") as u32 & POS_FACEUP != 0 && self.was_offered(duel, ef) != Some(false) {
                    faceup_match = self.match_fix(cp, msgs, idx, ef, None);
                } else {
                    fix = self.match_fix(cp, msgs, idx, ef, None);
                }
            }
            if fix.is_none() {
                fix = self.forced_cost_fix();
            }
            let nm_or_chained = |t: &str| nothing_more(t) || t == "CHAINED";
            if fix.is_none() && nm_or_chained(&gt) && !nm_or_chained(&et) {
                // (last resort) the real game did more before going on
                let mut start = self.chain_start();
                if start.is_none() && et == "CHAINING" && exp.i("trigger_con") as u32 == self.opponent {
                    start = self.prev_chain_start();
                }
                if let Some(s) = start
                    && !self.match_piles.contains(&(s, self.pos))
                    && !self.nomatch_piles.contains(&(s, self.pos))
                {
                    fix = Some(Fix::MatchPile(s, self.pos));
                }
            }
        }
        let item_prompt = match item {
            ObsItem::Prompt(p, _) => Some(jv(p)),
            _ => None,
        };
        if fix.is_none() && item_prompt.as_ref().is_some_and(|p| matches!(p.t(), "SELECT_BATTLECMD" | "SELECT_IDLECMD")) && gt == "NEW_PHASE" {
            // the replay skipped a phase the real game did not
            for &(dkey, act) in self.window_choices.iter().rev() {
                if self.overrides.get(&dkey) != Some(&act) {
                    fix = Some(Fix::Decide(dkey, act));
                    break;
                }
            }
        }
        if fix.is_none() && item_prompt.is_some() {
            fix = self.unforce_fix(&got);
        }
        if fix.is_none() {
            // a card whose code an assumption named moved just before, and the consequences differ
            for &(at, a) in self.assumed_moves.iter().rev() {
                if at + 12 < self.pos {
                    break;
                }
                if self.assumptions.contains(&a) {
                    fix = Some(Fix::Unassume(a));
                    break;
                }
            }
        }
        if fix.is_none() && matches!(item, ObsItem::Msg(_)) {
            // (inside the opponent's activation, whose script checks values of several hidden cards)
            let j = self.link_start();
            if j < self.pos
                && let Some(Msg::Chaining { trigger_con, code, .. }) = self.imsg(j)
                && *trigger_con == self.opponent
            {
                let code = *code;
                fix = self.property_fix(code, duel);
            }
        }
        if fix.is_none() && item_prompt.as_ref().is_some_and(|p| p.t() == "SELECT_CHAIN" && p.g("chains").truthy()) && gt != "SELECT_CHAIN" {
            for &(dkey, passing) in self.window_activations.iter().rev() {
                if dkey.0 + 12 >= self.pos && self.overrides.get(&dkey) != Some(&passing) {
                    fix = Some(Fix::Decide(dkey, passing));
                    break;
                }
            }
        }
        if fix.is_none() && item_prompt.is_some() && self.link_start() < self.pos {
            // the viewer is asked something while a chain link resolves, the replay's link went on
            let span = (self.link_start(), self.pos);
            if !self.match_piles.contains(&span) && !self.nomatch_piles.contains(&span) && !self.pile_dropped.contains(&("matchpile", span)) {
                fix = Some(Fix::MatchPile(span.0, span.1));
            }
        }
        if fix.is_none() && item_prompt.as_ref().is_some_and(|p| p.t() == "SELECT_CHAIN" && p.g("chains").truthy()) && gt != "SELECT_CHAIN" {
            // the viewer may chain in the real game, in the replay there was no chain window
            let span = (self.action_start(), self.pos);
            if !self.match_piles.contains(&span) && !self.nomatch_piles.contains(&span) && !self.pile_dropped.contains(&("matchpile", span)) {
                fix = Some(Fix::MatchPile(span.0, span.1));
            }
        }
        let solving = (self.pos.saturating_sub(29)..self.pos).rev().find(|&j| self.itype(j) == "CHAIN_SOLVING");
        let solving = if self.pos >= 30 { solving } else { (0..self.pos).rev().find(|&j| self.itype(j) == "CHAIN_SOLVING") };
        if fix.is_none()
            && matches!(item, ObsItem::Msg(m) if matches!(m, Msg::ChainSolved { .. } | Msg::ChainSolving { .. } | Msg::ChainEnd {}))
            && let Some(s) = solving
            && self.hidden_picks.last().is_some_and(|h| h.0 >= s)
        {
            // the real link ended where the replay's went on with a card the opponent picked among
            // its hidden cards
            let span = self.hidden_picks.last().unwrap().1;
            if !self.nomatch_piles.contains(&span) && !self.match_piles.contains(&span) {
                fix = Some(Fix::NomatchPile(span.0, span.1));
            }
        }
        if fix.is_none() {
            fix = self.facedown_pick_fix();
        }
        if fix.is_none() {
            fix = self.hidden_set_fix(duel);
        }
        let mut fallback = None;
        if fix.is_none() {
            let (fb, at) = self.hidden_pick_fallback();
            fallback = fb;
            if fallback.is_some() && !self.recent.iter().any(|&(k, n, _)| k.0 as i64 > at && n > 1) {
                // (no later decision could explain the consequences instead)
                fix = fallback.take();
            }
        }
        if fix.is_none() && fallback.is_none() && gt == "MOVE" && self.link_start() == self.pos {
            // (last resort) a card moved that an earlier choice of the opponent may have singled out
            let cid = if got.g("to").i("loc") != 0 { self.loc_map(duel).at(&li_of(got.g("to"))) } else { None };
            fallback = self.silent_choice_fix(cid);
        }
        if fix.is_none() && fallback.is_none() {
            fallback = faceup_match;
        }
        let what = match item {
            ObsItem::Msg(m) => m.name().to_string(),
            ObsItem::Prompt(..) => "prompt".to_string(),
            ObsItem::Response(_) => "response".to_string(),
        };
        Divergence {
            pos: self.pos,
            reason: format!("message mismatch [{what} vs {gt}]"),
            expected: Some(DV::Item(item.clone())),
            got: Some(DV::Msg(got_m.clone())),
            fix,
            fallback,
        }
    }

    pub fn note_silent(&mut self, duel: &RDuel, up: &[Msg], dkey: Dkey, ids: &[Option<u32>], valid: &[Response], i: usize) {
        if valid.len() > 1 {
            let refs = self.loc_map(duel).references(infer::selection_window(up));
            if !ids.iter().any(|c| c.is_some_and(|c| refs.contains(&c))) {
                // (nothing shows which card was chosen before the action ends)
                self.silent_choices.push((dkey, ids.to_vec(), valid.len(), i, Some(valid.to_vec())));
            }
        }
    }

    /// The opponent's card checks values of several of its hidden cards: put cards its script
    /// refers to in hidden slots of the piles it names (see the Python docstring).
    pub fn property_fix(&mut self, code: u32, duel: &RDuel) -> Option<Fix> {
        use crate::belief::script_index as si;
        if code == 0 || !si::script_info(code).value_checks {
            return None;
        }
        if self.property_tried.entry(code).or_default().len() >= PROPERTY_TRIES {
            return None;
        }
        let (deck_pile, extra_pile) = si::piles(code);
        let seen: BTreeSet<u32> = self.opp_seen_codes().into_iter().collect();
        let placed = self.property_placed.entry(code).or_default().clone();
        let reusable: BTreeSet<(u32, u32)> = placed.iter().filter(|a| self.assumptions.contains(a)).map(|a| (a.2, a.3)).collect();
        let value_of = |c: u32| -> (bool, u32, u32) {
            let d = card(c).unwrap();
            (is_extra_deck(d), d.lscale, d.level)
        };
        let race = si::races(code);
        let db = crate::game::card_db();
        let mut refs = si::candidates(code);
        if race != 0 {
            let mut extra: Vec<u32> = db.cards.values().filter(|c| c.race & race != 0 && c.alias == 0 && db.ots.get(&c.code).copied().unwrap_or(0) & 0x3 != 0).map(|c| c.code).collect();
            extra.sort();
            refs.extend(extra);
        }
        let tried = self.property_tried.get(&code).cloned().unwrap_or_default();
        let tried_values = self.property_values.get(&code).cloned().unwrap_or_default();
        let mut pool: Vec<u32> = Vec::new();
        for c in refs {
            if pool.contains(&c) || tried.contains(&c) {
                continue;
            }
            let Some(d) = card(c) else { continue };
            if d.type_ & TYPE_MONSTER == 0 || d.type_ & TYPE_TOKEN != 0 {
                continue;
            }
            let pile_ok = if is_extra_deck(d) { extra_pile } else { deck_pile };
            if pile_ok {
                pool.push(c);
            }
        }
        // (values not tried yet first, then the opponent's seen cards)
        pool.sort_by_key(|&c| (tried_values.contains(&value_of(c)), !seen.contains(&c)));
        let mut batch: Vec<Asm> = Vec::new();
        let mut values: BTreeSet<(bool, u32, u32)> = BTreeSet::new();
        let mut taken: BTreeSet<(u32, u32)> = BTreeSet::new();
        for c in pool {
            let value = value_of(c);
            if values.contains(&value) {
                continue;
            }
            let loc = if value.0 { LOCATION_EXTRA } else { LOCATION_DECK };
            let n = duel.location_cards(self.opponent, loc).len();
            for seq in 0..n as u32 {
                if taken.contains(&(loc, seq)) {
                    continue;
                }
                let Ok(Some(var)) = self.lock_var(self.opponent, loc, seq) else { continue };
                let (faceup, vid) = {
                    let t = self.lock().unwrap();
                    (t.vars[var].faceup, t.vars[var].id)
                };
                if faceup {
                    continue;
                }
                let observed = self.hypothesis.as_ref().is_some_and(|h| h.observed.contains(&vid));
                if !reusable.contains(&(loc, seq)) && (observed || self.assumptions.iter().any(|a| (a.1, a.2, a.3) == (self.opponent, loc, seq))) {
                    continue;
                }
                batch.push((self.pos, self.opponent, loc, seq, c));
                taken.insert((loc, seq));
                values.insert(value);
                self.property_tried.entry(code).or_default().insert(c);
                self.property_values.entry(code).or_default().insert(value);
                break;
            }
            if batch.len() >= PROPERTY_BATCH {
                break;
            }
        }
        if batch.is_empty() {
            return None;
        }
        // (the previous batch for this card made way)
        self.assumptions.retain(|a| !placed.contains(a));
        self.property_placed.insert(code, batch.clone());
        Some(Fix::AssumeMany(batch))
    }

    /// Only the stats of a card on the field differ.
    pub fn stat_fix(&mut self, exp: &Value, got: &Value, duel: &RDuel) -> Option<Fix> {
        let prev = (0..self.pos).rev().find_map(|j| if let ObsItem::Prompt(_, Some(s)) = &self.items[j] { Some(jv(s)) } else { None });
        for p in 0..2 {
            for name in ["mzone", "szone"] {
                let (el, gl) = (exp.g("players").l()[p].g(name).l(), got.g("players").l()[p].g(name).l());
                for (i, (e, g)) in el.iter().zip(gl).enumerate() {
                    if e == g || !e.truthy() || !g.truthy() {
                        continue;
                    }
                    let before = prev.as_ref().map(|pv| pv.g("players").l()[p].g(name).l().get(i).cloned().unwrap_or(Value::Null));
                    if before.as_ref().is_some_and(|b| b.truthy() && b.g("code") == e.g("code")) {
                        return self.silent_stat_fix(exp, got, duel).or_else(|| self.stat_copy_fix(exp, got));
                    }
                    return self.stat_copy_fix(exp, got).or_else(|| self.silent_stat_fix(exp, got, duel));
                }
            }
        }
        None
    }

    /// Only the stats of a card on the field differ: a silent answer about it long before.
    fn silent_stat_fix(&mut self, exp: &Value, got: &Value, duel: &RDuel) -> Option<Fix> {
        let stats = ["level", "rank", "attack", "defense"];
        for p in 0..2u32 {
            for (name, loc) in [("mzone", LOCATION_MZONE), ("szone", LOCATION_SZONE)] {
                let (el, gl) = (exp.g("players").l()[p as usize].g(name).l(), got.g("players").l()[p as usize].g(name).l());
                for (i, (e, g)) in el.iter().zip(gl).enumerate() {
                    if e == g || !e.truthy() || !g.truthy() || e.g("code") != g.g("code") {
                        continue;
                    }
                    if !diff_keys(e, g).iter().all(|k| stats.contains(&k.as_str())) {
                        return None;
                    }
                    let cid = self.loc_map(duel).at(&li(p, loc, i as u32));
                    return self.silent_choice_fix(cid);
                }
            }
        }
        None
    }

    /// The other answer of a silent choice of the opponent's made since the viewer's latest state
    /// view, the latest first.
    pub fn silent_since_view_fix(&mut self) -> Option<Fix> {
        let view = (0..self.pos).filter(|&j| matches!(self.items[j], ObsItem::Prompt(_, Some(_)))).max().map_or(-1, |j| j as i64);
        let choices = self.silent_choices.clone();
        for (dkey, _, n, i, _) in choices.iter().rev() {
            if dkey.0 as i64 <= view {
                break;
            }
            if self.shows_consequence(dkey.0) {
                continue; // (its answer showed: not silent)
            }
            for k in 0..*n {
                if k != *i && !self.tried_silent.contains(&(*dkey, k)) {
                    self.tried_silent.insert((*dkey, k));
                    return Some(Fix::Decide(*dkey, k));
                }
            }
        }
        None
    }

    /// Whether something visible happens between the decision at `at` and the end of its chain link.
    pub fn shows_consequence(&self, at: usize) -> bool {
        for j in at..(at + 40).min(self.items.len()) {
            let Some(m) = self.imsg(j) else { continue };
            let t = m.name();
            if matches!(t, "CHAIN_SOLVED" | "CHAIN_END" | "NEW_PHASE" | "NEW_TURN") {
                return false;
            }
            if !matches!(t, "HINT" | "CARD_HINT" | "PLAYER_HINT" | "CHAIN_SOLVED" | "CHAIN_END" | "SHUFFLE_HAND" | "SHUFFLE_DECK") {
                return true;
            }
        }
        false
    }

    pub fn silent_choice_fix(&mut self, cid: Option<u32>) -> Option<Fix> {
        let cid = cid.filter(|&c| c != 0)?;
        // (older than the decision search's window: nearer ones are its business)
        let choices: Vec<_> = self.silent_choices.iter().rev().filter(|c| c.1.contains(&Some(cid)) && c.0.0 + DECISION_WINDOW < self.pos).cloned().collect();
        for (dkey, ids, _, i, valid) in &choices {
            let k = valid.as_ref().and_then(|v| {
                let pick = Response { indices: Some(vec![ids.iter().position(|&x| x == Some(cid)).unwrap() as i64]), ..Response::default() };
                v.iter().position(|r| *r == pick)
            });
            if let Some(k) = k
                && k != *i
                && self.overrides.get(dkey) != Some(&k)
                && !self.tried_silent.contains(&(*dkey, k))
            {
                self.tried_silent.insert((*dkey, k));
                return Some(Fix::Decide(*dkey, k));
            }
        }
        for (dkey, _, n, i, _) in &choices {
            for k in 0..*n {
                if k != *i && !self.tried_silent.contains(&(*dkey, k)) {
                    self.tried_silent.insert((*dkey, k));
                    return Some(Fix::Decide(*dkey, k));
                }
            }
        }
        None
    }

    /// (Last resort) The opponent recently chose among its hidden cards: the repair and where the
    /// choice was.
    pub fn hidden_pick_fallback(&self) -> (Option<Fix>, i64) {
        for &(at, span) in self.hidden_picks.iter().rev() {
            if at + 40 < self.pos {
                break;
            }
            if !self.nomatch_piles.contains(&span) && !self.match_piles.contains(&span) {
                return (Some(Fix::NomatchPile(span.0, span.1)), at as i64);
            }
        }
        (None, -1)
    }

    /// A hidden card set into the spell/trap zone must be a spell or trap.
    fn kind_fix(&mut self, cp: Cp, msgs: &[Msg], idx: usize, exp: &Value) -> Option<Fix> {
        let to = exp.g("to");
        if to.i("loc") as u32 != LOCATION_SZONE || to.i("pos") as u32 & POS_FACEDOWN == 0 || to.i("seq") > 5 {
            return None;
        }
        let Some(snap) = self.snapshots.get(&cp) else { return if self.snapshot_cps.contains(&cp) { None } else { Some(Fix::Snapshot(cp)) } };
        let mut lm = (**snap).clone();
        lm.references(&msgs[..idx]);
        let cid = lm.at(&li_of(exp.g("from")));
        let field_zone = to.i("seq") == 5;
        let fits = |code: u32| card(code).is_some_and(|c| c.type_ & (TYPE_SPELL | TYPE_TRAP) != 0 && (c.type_ & TYPE_FIELD != 0) == field_zone);
        if let Some(c) = cid.filter(|&c| c != 0)
            && fits(lm.codes.get(&c).copied().unwrap_or(0))
        {
            return None;
        }
        let codes: Vec<u32> = self.filler().settable_spells.iter().copied().filter(|&c| fits(c)).collect();
        self.assume_fix(key3(exp.g("from")), &codes)
    }

    /// Whether the card at `loc` was among the cards the opponent's latest card selections offered
    /// (None: there was no such selection).
    fn was_offered(&self, duel: &RDuel, loc: &Value) -> Option<bool> {
        let cid = self.loc_map(duel).at(&li_of(loc));
        let place = (loc.i("con") as u32, loc.i("loc") as u32);
        let recent: Vec<&std::collections::HashSet<Option<u32>>> = self
            .recent
            .iter()
            .filter(|(k, _, _)| k.0 + 3 >= self.pos && self.offered.contains_key(k))
            .map(|(k, _, _)| &self.offered[k])
            .filter(|(_, places)| places.contains(&place))
            .map(|(ids, _)| ids)
            .collect();
        if cid.is_none() || recent.is_empty() {
            return None;
        }
        Some(recent.iter().any(|ids| ids.contains(&cid)))
    }

    /// The opponent took a hidden card for an effect, but no card at that position satisfies the
    /// filter: let that card match every filter during the step.
    pub fn match_fix(&mut self, cp: Cp, msgs: &[Msg], idx: usize, loc: &Value, duel: Option<&RDuel>) -> Option<Fix> {
        if let Some(d) = duel
            && matches!(loc.i("loc") as u32, LOCATION_MZONE | LOCATION_SZONE)
            && !msgs[..idx].iter().any(|m| matches!(m, Msg::Move { .. }))
        {
            let cid = self.loc_map(d).at(&li_of(loc)).filter(|&c| c != 0)?;
            if self.matches.get(&cp.0).is_some_and(|l| l.contains(&cid)) {
                return None;
            }
            return Some(Fix::Match(cp, cid));
        }
        let Some(snap) = self.snapshots.get(&cp) else { return if self.snapshot_cps.contains(&cp) { None } else { Some(Fix::Snapshot(cp)) } };
        let mut lm = (**snap).clone();
        lm.references(&msgs[..idx]);
        let cid = lm.at(&li_of(loc)).filter(|&c| c != 0)?;
        if self.matches.get(&cp.0).is_some_and(|l| l.contains(&cid)) {
            return None;
        }
        Some(Fix::Match(cp, cid))
    }

    /// A group of cards taken out of hidden piles by one operation: from the operation's chain
    /// link on, the cards at the stream's positions match its filter, the others do not.
    fn group_match_fix(&mut self, exp: &Value, got: &Value, cp: Cp, msgs: &[Msg], idx: usize) -> Option<Fix> {
        let key = |m: &Value| (m.g("from").i("loc"), m.g("to").i("loc"), m.i("reason"));
        if key(exp) != key(got) || exp.g("from").i("con") != got.g("from").i("con") || exp.g("from").i("loc") == exp.g("to").i("loc") {
            return None;
        }
        if exp.g("code").truthy() && exp.g("code") == got.g("code") {
            return None; // (two copies of one card: a copy swap)
        }
        let ke = key(exp);
        let is_run = |s: &Self, j: usize| s.itype(j) == "MOVE" && key(&s.iv(j)) == ke;
        let mut g0 = self.pos;
        while g0 > 0 && is_run(self, g0 - 1) {
            g0 -= 1;
        }
        let mut g1 = self.pos;
        while g1 + 1 < self.items.len() && is_run(self, g1 + 1) {
            g1 += 1;
        }
        if g1 < g0 + 1 {
            return None; // (a single card: the per-card repairs)
        }
        let Some(snap) = self.snapshots.get(&cp) else { return if self.snapshot_cps.contains(&cp) { None } else { Some(Fix::Snapshot(cp)) } };
        let mut lm = (**snap).clone();
        lm.references(&msgs[..idx]);
        let mut taken = Vec::new();
        for j in g0..self.pos {
            let m = self.iv(j);
            let cid = lm.at(&li_of(m.g("to"))).filter(|&c| c != 0)?;
            if m.g("from").i("con") as u32 == self.opponent {
                taken.push(cid);
            }
        }
        let loc = exp.g("from").i("loc") as u32;
        let players: BTreeSet<u32> = (g0..=g1).map(|j| self.iv(j).g("from").i("con") as u32).filter(|&p| p == self.opponent).collect();
        let mut pile: BTreeSet<u32> = BTreeSet::new();
        for p in players {
            pile.extend(lm.lists.get(&(p, loc)).map_or(vec![], |l| l.iter().flatten().copied().collect::<Vec<_>>()));
        }
        for j in self.pos..=g1 {
            let ObsItem::Msg(Msg::Move { from, to, .. }) = self.items[j].clone() else { return None };
            let cid = lm.at(&(&from).into()).filter(|&c| c != 0)?;
            if from.con == self.opponent {
                taken.push(cid);
            }
            lm.move_card(&(&from).into(), &(&to).into());
        }
        let tset: BTreeSet<u32> = taken.iter().copied().collect();
        let rest: Vec<u32> = pile.difference(&tset).copied().collect();
        let a = self.link_start();
        let fix = Fix::MatchSet((a.min(g0), self.pos), taken, rest);
        if self.matchsets.contains(&fix) { None } else { Some(fix) }
    }

    /// The opponent's hidden card the replay handles has the code of the real hint.
    fn hint_code_fix(&mut self, code: u32, replay_code: u32, duel: &mut RDuel) -> Option<Fix> {
        card(code)?;
        card(replay_code)?;
        let mut found: Vec<(u32, u32, u32)> = Vec::new();
        for loc in [LOCATION_DECK, LOCATION_HAND, LOCATION_EXTRA, LOCATION_MZONE, LOCATION_SZONE] {
            let cards = duel.location_cards(self.opponent, loc);
            let pos = if matches!(loc, LOCATION_MZONE | LOCATION_SZONE) { Some(parse_location_query(&duel.query_location(self.opponent, loc, QUERY_POSITION))) } else { None };
            for (seq, &(c, cid)) in cards.iter().enumerate() {
                let hidden = match &pos {
                    None => true,
                    Some(p) => p.get(seq).cloned().flatten().is_some_and(|x| x.position.unwrap_or(0) & POS_FACEUP == 0),
                };
                if cid != 0 && c == replay_code && hidden {
                    found.push((self.opponent, loc, seq as u32));
                }
            }
        }
        let mut field: Vec<(u32, u32, u32)> = found.iter().copied().filter(|f| matches!(f.1, LOCATION_MZONE | LOCATION_SZONE)).collect();
        if field.len() > 1 {
            // (several: the one the chain link being resolved targeted)
            let start = (self.pos.saturating_sub(29)..self.pos).rev().find(|&j| self.itype(j) == "CHAINING").unwrap_or(self.pos);
            let targets: Vec<(u32, u32, u32)> = (start..self.pos)
                .filter_map(|j| if let Some(Msg::BecomeTarget { cards }) = self.imsg(j) { Some(cards.iter().map(|c| (c.con, c.loc, c.seq)).collect::<Vec<_>>()) } else { None })
                .flatten()
                .collect();
            field.retain(|f| targets.contains(f));
        }
        if field.len() == 1 {
            found = field;
        }
        if found.len() != 1 {
            return None;
        }
        let k = found[0];
        let var = self.lock_var(k.0, k.1, k.2).ok()?;
        let tried: BTreeSet<u32> =
            self.assumptions.iter().chain(self.failed_assumptions.iter()).filter(|a| (a.0, a.1, a.2, a.3) == (self.pos, k.0, k.1, k.2)).map(|a| a.4).collect();
        if var.is_none() || tried.contains(&code) {
            return None;
        }
        Some(Fix::Assume((self.pos, k.0, k.1, k.2, code)))
    }

    /// In the chain link being resolved, the opponent took a card out of a hidden pile and put it
    /// face-down: "not that card" for the slot it came from.
    fn facedown_pick_fix(&mut self) -> Option<Fix> {
        self.solved.as_ref()?;
        let start = self.link_start();
        let lo = (start as i64 - 1).max(self.pos as i64 - 30).max(-1);
        let mut j = self.pos as i64 - 1;
        while j > lo {
            let ju = j as usize;
            j -= 1;
            let Some(Msg::Move { code, from: f, to: t, .. }) = self.imsg(ju).cloned() else { continue };
            if code != 0 || f.con != self.opponent || !hidden_pile(f.loc) || !matches!(t.loc, LOCATION_MZONE | LOCATION_SZONE) || t.pos & POS_FACEDOWN == 0 {
                continue;
            }
            let Some((_, member)) = self.solved_member(t.con, t.loc, t.seq) else { continue };
            let entry = (ju, (f.con, f.loc, f.seq), Same::Not(member));
            if !self.identities.contains(&entry) && !self.failed_identities.contains(&entry) {
                return Some(Fix::Identify(entry));
            }
        }
        None
    }

    /// The replay's card lost a hint as it moved from a hidden pile to a public place, the real one
    /// did not: the real card is another copy.
    fn unhinted_fix(&mut self, got: &Value) -> Option<Fix> {
        let j = (self.pos.saturating_sub(5)..self.pos).rev().find(|&j| {
            let m = self.iv(j);
            m.t() == "MOVE" && key3(m.g("to")) == key3(got)
        });
        let j = if self.pos >= 6 { j } else { (0..self.pos).rev().find(|&j| self.iv(j).t() == "MOVE" && key3(self.iv(j).g("to")) == key3(got)) }?;
        let m = self.iv(j);
        if !hidden_pile(m.g("from").i("loc") as u32) {
            return None;
        }
        let (gc, gl, gs) = key3(got);
        let (_, member) = self.solved_member(gc, gl, gs)?;
        let f = key3(m.g("from"));
        self.ident_fix((j, f, Same::Not(member)))
    }

    /// The card in a hidden slot is another copy than the one the hypothesis put there.
    pub fn not_this_copy_fix(&mut self, slot: Slot) -> Option<Fix> {
        let (_, member) = self.solved_member(slot.0, slot.1, slot.2)?;
        self.ident_fix((self.pos, slot, Same::Not(member)))
    }

    /// The card in the hidden slot `e` is the physical card that the replay has in slot `g`.
    pub fn slot_identity_fix(&mut self, e: Slot, g: Slot, copies: bool) -> Option<Fix> {
        let solved = self.solved.clone()?;
        let ve = self.lock_var(e.0, e.1, e.2).ok().flatten()?;
        let vg = self.lock_var(g.0, g.1, g.2).ok().flatten()?;
        let (kind, pool) = {
            let t = self.lock().ok()?;
            (t.vars[ve].kind, t.vars[ve].pool)
        };
        if kind != Kind::Claim {
            return None;
        }
        let pool = pool?;
        let sv = solved.vars.get(vg)?;
        let member = if sv.kind == Kind::Claim { sv.assigned } else { None }?;
        if !self.lock().ok()?.pools[pool].members.contains(&member) {
            return None;
        }
        let code = |mut v: Option<VarId>| -> Option<u32> {
            while let Some(x) = v {
                if solved.vars[x].kind != Kind::Claim {
                    return solved.vars[x].resolved;
                }
                v = solved.vars[x].assigned;
            }
            None
        };
        let se = solved.vars.get(ve);
        if se.is_none() || (!copies && code(Some(ve)) == code(Some(member))) {
            return None; // (copies of one card: a swap of the two)
        }
        self.ident_fix((self.pos, e, Same::Var(member)))
    }

    /// The real game moves a hidden hand card from another place of the hand than the replay.
    fn hand_slot_fix(&mut self, exp: &Value, got: &Value) -> Option<Fix> {
        let (e, g) = (exp.g("from"), got.g("from"));
        if e.i("con") as u32 != self.opponent
            || e.i("loc") as u32 != LOCATION_HAND
            || g.i("loc") as u32 != LOCATION_HAND
            || g.i("con") != e.i("con")
            || e.i("seq") == g.i("seq")
            || exp.g("code").truthy()
            || got.g("code").truthy()
            || exp.g("to") != got.g("to")
        {
            return None;
        }
        self.slot_identity_fix(key3(e), key3(g), false)
    }

    /// The replay's Tribute Summon from the hand takes another number of Tributes than the real one.
    fn summon_copy_fix(&mut self, exp: &Value, got: &Value) -> Option<Fix> {
        let tribute = REASON_RELEASE | REASON_SUMMON;
        let opp = self.opponent;
        let summon = |m: &Value| m.t() == "MOVE" && m.g("from").i("con") as u32 == opp && m.g("from").i("loc") as u32 == LOCATION_HAND && m.g("to").i("loc") as u32 == LOCATION_MZONE;
        if !(exp.i("reason") as u32 & tribute == tribute && summon(got) || summon(exp) && got.i("reason") as u32 & tribute == tribute) {
            return None;
        }
        let j = (self.pos..self.items.len().min(self.pos + 12)).find(|&j| summon(&self.iv(j)))?;
        let m = self.iv(j);
        if !m.g("code").truthy() {
            return None;
        }
        let f = key3(m.g("from"));
        let (_, member) = self.solved_member(f.0, f.1, f.2)?;
        self.ident_fix((j, f, Same::Not(member)))
    }

    /// The real game removes a card hint from a card that just came out of a hidden pile.
    fn mark_fix(&mut self, hint: &Value, duel: &RDuel) -> Option<Fix> {
        let k = (self.pos.saturating_sub(5)..self.pos).rev().find(|&j| self.iv(j).t() == "MOVE" && infer::same_loc(&li_of(self.iv(j).g("to")), &li_of(hint)));
        let k = if self.pos >= 6 { k } else { (0..self.pos).rev().find(|&j| self.iv(j).t() == "MOVE" && infer::same_loc(&li_of(self.iv(j).g("to")), &li_of(hint))) };
        let solved = self.solved.clone();
        let k = k?;
        solved.as_ref()?;
        let solved = solved.unwrap();
        let km = self.iv(k);
        if !hidden_pile(km.g("from").i("loc") as u32) {
            return None;
        }
        let f = key3(km.g("from"));
        let where_: HashMap<u32, (u32, u32, u32)> = duel.all_cards().into_iter().filter(|e| e[3] == 0 && e[5] != 0).map(|e| (e[5], (e[0], e[1], e[2]))).collect();
        let here = self.loc_map(duel).at(&li_of(hint));
        let marks = self.sim_marks.get(&(hint.i("value") as u64)).cloned().unwrap_or_default();
        for cid in marks.into_iter().rev() {
            if Some(cid) == here || !where_.contains_key(&cid) {
                continue;
            }
            let w = where_[&cid];
            let Ok(Some(v)) = self.lock_var(w.0, w.1, w.2) else { continue };
            let Some(sv) = solved.vars.get(v) else { continue };
            let same = match sv.assigned {
                Some(a) if sv.kind == Kind::Claim => a,
                _ => sv.id,
            };
            let entry = (k, f, Same::Var(same));
            if !self.identities.contains(&entry) && !self.failed_identities.contains(&entry) {
                return Some(Fix::Identify(entry));
            }
        }
        None
    }

    /// The replay's hidden hand of the opponent completes a win condition the real game's does not.
    fn win_reveal_fix(&self, got: &Value) -> Option<Fix> {
        let wins = crate::belief::script_index::win_condition_codes();
        let mut pieces: Vec<&Value> = got
            .g("cards")
            .l()
            .iter()
            .filter(|c| c.i("con") as u32 == self.opponent && c.i("loc") as u32 == LOCATION_HAND && c.o("code").is_some_and(|x| wins.contains(&(x as u32))))
            .collect();
        if pieces.len() < 2 {
            return None;
        }
        pieces.sort_by_key(|c| -c.i("seq"));
        for c in pieces {
            let key: Excl = (self.pos, c.i("con") as u32, c.i("loc") as u32, vec![c.i("seq")], c.i("code") as u32);
            if !self.exclusions.contains(&key) && !self.failed_exclusions.contains(&key) {
                return Some(Fix::Exclude(key));
            }
        }
        None
    }

    /// The replay's hidden card of the opponent moved by its own effect where the real game moved
    /// nothing from there.
    fn self_move_fix(&self, got: &Value, duel: &mut RDuel) -> Option<Fix> {
        let (f, t) = (got.g("from"), got.g("to"));
        if !duel.reason_effect_is_own(t.i("con") as u32, t.i("loc") as u32, t.i("seq") as u32) {
            return None;
        }
        let key: Excl = (self.pos, f.i("con") as u32, f.i("loc") as u32, vec![f.i("seq")], got.i("code") as u32);
        if self.exclusions.contains(&key) || self.failed_exclusions.contains(&key) { None } else { Some(Fix::Exclude(key)) }
    }

    /// The opponent summons a card from the hand shortly that the replay does not have in the hand.
    fn hidden_set_fix(&mut self, duel: &RDuel) -> Option<Fix> {
        let opp = self.opponent;
        let j = (self.pos..self.items.len().min(self.pos + 15)).find(|&j| {
            matches!(self.imsg(j), Some(Msg::Move { code, from, to, .. }) if *code != 0 && from.con == opp && from.loc == LOCATION_HAND && to.loc == LOCATION_MZONE)
        })?;
        let Some(Msg::Move { code, .. }) = self.imsg(j) else { return None };
        let code = *code;
        if duel.location_cards(opp, LOCATION_HAND).iter().any(|&(c, _)| c == code) {
            return None;
        }
        let solved = self.solved.clone()?;
        let lo = (self.pos as i64 - 30).max(-1);
        let mut k = self.pos as i64 - 1;
        while k > lo {
            let ku = k as usize;
            k -= 1;
            let Some(Msg::Move { code: mc, from: f, to: t, .. }) = self.imsg(ku).cloned() else { continue };
            if mc != 0 || f.con != opp || f.loc != LOCATION_HAND {
                continue;
            }
            let Ok(Some(v)) = self.lock_var(t.con, t.loc, t.seq) else { continue };
            let Some(sv) = solved.vars.get(v) else { continue };
            let member = if sv.kind == Kind::Claim { sv.assigned } else { None };
            let mut a = member;
            while let Some(x) = a {
                if solved.vars[x].kind != Kind::Claim {
                    break;
                }
                a = solved.vars[x].assigned;
            }
            let (Some(member), Some(a)) = (member, a) else { continue };
            if solved.vars[a].resolved != Some(code) {
                continue;
            }
            let entry = (ku, (f.con, f.loc, f.seq), Same::Not(member));
            if !self.identities.contains(&entry) && !self.failed_identities.contains(&entry) {
                return Some(Fix::Identify(entry));
            }
        }
        None
    }

    /// A card on the field has other stats than the real one and came there from a hidden pile.
    fn stat_copy_fix(&mut self, exp: &Value, got: &Value) -> Option<Fix> {
        let stats = ["level", "rank", "attack", "defense", "base_attack", "base_defense"];
        self.solved.as_ref()?;
        for p in 0..2u32 {
            for (name, loc) in [("mzone", LOCATION_MZONE), ("szone", LOCATION_SZONE)] {
                let (el, gl) = (exp.g("players").l()[p as usize].g(name).l().to_vec(), got.g("players").l()[p as usize].g(name).l().to_vec());
                for (i, (e, g)) in el.iter().zip(&gl).enumerate() {
                    if e == g || !e.truthy() || !g.truthy() || e.g("code") != g.g("code") {
                        continue;
                    }
                    if !diff_keys(e, g).iter().all(|k| stats.contains(&k.as_str())) {
                        return None;
                    }
                    // (the move that brought it there, from a hidden pile)
                    let j = (0..self.pos).rev().find(|&j| self.iv(j).t() == "MOVE" && key3(self.iv(j).g("to")) == (p, loc, i as u32))?;
                    let m = self.iv(j);
                    if !hidden_pile(m.g("from").i("loc") as u32) {
                        return None;
                    }
                    let (_, member) = self.solved_member(p, loc, i as u32)?;
                    let f = key3(m.g("from"));
                    return self.ident_fix((j, f, Same::Not(member)));
                }
            }
        }
        None
    }

    pub fn nomatch_card_fix(&mut self, slot: Slot, duel: Option<&RDuel>, dest: Option<&Value>) -> Option<Fix> {
        let v = self.lock_var(slot.0, slot.1, slot.2).ok().flatten()?;
        {
            let t = self.lock().ok()?;
            if t.vars[v].code.is_some() || t.vars[v].fixed_code.is_some() {
                return None;
            }
        }
        let start = self.chain_start().unwrap_or_else(|| {
            // (the link being resolved may be the viewer's)
            (self.pos.saturating_sub(29)..self.pos).rev().find(|&j| matches!(self.itype(j), "CHAINING" | "CHAIN_SOLVING")).unwrap_or(self.pos)
        });
        let cid = match (duel, dest) {
            (Some(d), Some(dv)) => self.loc_map(d).at(&li_of(dv)).filter(|&c| c != 0),
            _ => None,
        };
        if let Some(cid) = cid {
            // (by card id: the slots shift while the operation moves the cards one by one)
            let entry = (start, self.pos, cid);
            return if self.nomatch_id_spans.contains(&entry) { None } else { Some(Fix::NomatchId(start, self.pos, cid)) };
        }
        let entry = (start, self.pos, slot);
        if self.nomatch_cards.contains(&entry) { None } else { Some(Fix::NomatchCard(start, self.pos, slot)) }
    }

    /// Every card this operation moves from the pile is one of the cards revealed before it was shuffled.
    fn revealed_fix(&mut self, le: Slot) -> Option<Fix> {
        let v = self.lock_var(le.0, le.1, le.2).ok().flatten()?;
        let t = self.lock().ok()?;
        if t.vars[v].kind != Kind::Claim {
            return None;
        }
        let (refs, piles) = t.referenced_state();
        if refs == 0 || !piles.contains(&(le.0, le.1)) {
            return None;
        }
        let first_to = self.iv(self.pos).g("to").clone();
        let mut entries = Vec::new();
        for k in 0..refs {
            let j = self.pos + k;
            if j >= self.items.len() {
                break;
            }
            let m = self.iv(j);
            if m.t() != "MOVE" || *m.g("to") != first_to || (m.g("from").i("con") as u32, m.g("from").i("loc") as u32) != (le.0, le.1) {
                break;
            }
            entries.push((j, (le.0, le.1, m.g("from").i("seq") as u32), Same::Revealed));
        }
        entries.retain(|e| !self.identities.contains(e));
        if entries.is_empty() || self.failed_identities.contains(&entries[0]) {
            return None;
        }
        Some(Fix::IdentifyAll(entries))
    }

    /// Swaps the viewer's card at `li` with another copy of it.
    pub fn viewer_copy_fix(&mut self, l: &Value, duel: &RDuel, cp: Cp) -> Option<Fix> {
        let live = self.loc_map(duel);
        let a = live.at(&li_of(l)).filter(|&c| c != 0)?;
        let code = live.codes.get(&a).copied().filter(|&c| c != 0)?;
        for loc in [LOCATION_HAND, LOCATION_DECK, LOCATION_GRAVE, LOCATION_REMOVED] {
            for (c, b) in duel.location_cards(self.viewer, loc) {
                if c == code && b != a
                    && let Some(f) = self.history_swap(a, b, cp)
                {
                    return Some(f);
                }
            }
        }
        None
    }

    fn not_top_fix(&mut self, player: u32) -> Option<Fix> {
        let solved = self.solved.clone()?;
        let (top, n) = {
            let t = self.lock().ok()?;
            let deck = t.list(player, LOCATION_DECK)?;
            let &top = deck.last()?;
            if t.vars[top].kind != Kind::Claim {
                return None;
            }
            (top, deck.len())
        };
        let sv = solved.vars.get(top)?;
        let member = if sv.kind == Kind::Claim { sv.assigned } else { None }?;
        self.ident_fix((self.pos, (player, LOCATION_DECK, (n - 1) as u32), Same::Not(member)))
    }

    /// A hidden hand card of the opponent shows revealed in the real view but not in the replay's.
    pub fn public_arrival_fix(&mut self, exp: &Value, got: &Value) -> Option<Fix> {
        let p = self.opponent as usize;
        let (eh, gh) = (exp.g("players").l()[p].g("hand").l().to_vec(), got.g("players").l()[p].g("hand").l().to_vec());
        for (i, (e, g)) in eh.iter().zip(&gh).enumerate() {
            if e.truthy() && e.g("is_public").truthy() && e.g("code").truthy() && !(g.truthy() && g.g("is_public").truthy()) {
                let fix = self.arrival_fix(p as u32, LOCATION_HAND, i as u32, e.i("code") as u32);
                if fix.is_some() {
                    return fix;
                }
            }
        }
        None
    }

    /// A hand card shows revealed in the replay's view but hidden in the real one.
    pub fn public_hidden_fix(&mut self, exp: &Value, got: &Value) -> Option<Fix> {
        for p in [self.opponent, self.viewer] {
            let (eh, gh) = (exp.g("players").l()[p as usize].g("hand").l().to_vec(), got.g("players").l()[p as usize].g("hand").l().to_vec());
            for (i, (e, g)) in eh.iter().zip(&gh).enumerate() {
                if g.truthy() && g.g("is_public").truthy() && e.truthy() && !e.g("is_public").truthy() && !e.g("code").truthy() {
                    let fix = self.not_this_copy_fix((p, LOCATION_HAND, i as u32));
                    if fix.is_some() {
                        return fix;
                    }
                }
            }
        }
        None
    }

    /// Identifies the activating card's slot with one of the latest cards of the pile it was
    /// shuffled from (the latest first), each tried once.
    pub fn arrival_fix(&mut self, con: u32, loc: u32, seq: u32, code: u32) -> Option<Fix> {
        let slot = (con, loc, seq);
        let v = self.lock_var(con, loc, seq).ok().flatten()?;
        let members = {
            let t = self.lock().ok()?;
            if t.vars[v].kind != Kind::Claim {
                return None;
            }
            let pool = t.vars[v].pool?;
            t.pools[pool].members.iter().map(|&m| (m, t.vars[m].known_code())).collect::<Vec<_>>()
        };
        let mut arrived = 0;
        let lo = (self.pos as i64 - 60).max(-1);
        let mut j = self.pos as i64 - 1;
        while j > lo {
            let Some(m) = self.imsg(j as usize) else {
                j -= 1;
                continue;
            };
            match m {
                Msg::NewPhase { .. } | Msg::NewTurn { .. } => break,
                Msg::Draw { player, cards } if *player == con && loc == LOCATION_HAND => arrived += cards.len(),
                Msg::Move { from, to, .. } if (to.con, to.loc) == (con, loc) && (from.con, from.loc) != (con, loc) => arrived += 1,
                _ => {}
            }
            j -= 1;
        }
        for &(m, known) in members.iter().rev().take(arrived.min(4)) {
            if known.is_some_and(|k| k != code) {
                continue;
            }
            let entry = (self.pos, slot, Same::Var(m));
            if !self.identities.contains(&entry) && !self.failed_identities.contains(&entry) {
                return Some(Fix::Identify(entry));
            }
        }
        None
    }

    fn identify_fix(&mut self, le: Slot, lg: Slot) -> Option<Fix> {
        let solved = self.solved.clone()?;
        let vg = self.lock_var(lg.0, lg.1, lg.2).ok().flatten()?;
        let ve = self.lock_var(le.0, le.1, le.2).ok().flatten()?;
        {
            let t = self.lock().ok()?;
            if t.vars[ve].kind != Kind::Claim || t.vars[ve].same_as.is_some() {
                return None;
            }
        }
        let sv = solved.vars.get(vg)?;
        let member = if sv.kind == Kind::Claim { sv.assigned } else { None }?;
        self.ident_fix((self.pos, le, Same::Var(member)))
    }

    /// The real game shuffles the hand as a chain link activated from it ends: another copy was the
    /// activated one.
    fn activated_copy_fix(&mut self, exp: &Value, duel: &RDuel, cp: Cp) -> Option<Fix> {
        let n = (self.pos.saturating_sub(39)..self.pos).rev().find_map(|j| if let Some(Msg::ChainSolving { chain_count }) = self.imsg(j) { Some(*chain_count) } else { None });
        let n = if self.pos >= 40 { n } else { (0..self.pos).rev().find_map(|j| if let Some(Msg::ChainSolving { chain_count }) = self.imsg(j) { Some(*chain_count) } else { None }) }?;
        let lo = self.pos.saturating_sub(79);
        let range: Vec<usize> = if self.pos >= 80 { (lo..self.pos).rev().collect() } else { (0..self.pos).rev().collect() };
        let ch = range.into_iter().find_map(|j| if let Some(m @ Msg::Chaining { chain_count, .. }) = self.imsg(j) { (*chain_count == n).then(|| jv(m)) } else { None })?;
        if ch.i("trigger_loc") as u32 != LOCATION_HAND || ch.i("trigger_con") != exp.i("player") {
            return None;
        }
        let code = ch.i("code") as u32;
        let player = exp.i("player") as u32;
        let all = duel.all_cards();
        let in_hand: Vec<u32> = all.iter().filter(|e| e[3] == 0 && e[0] == player && e[1] == LOCATION_HAND && e[4] == code).map(|e| e[5]).collect();
        let moved: Vec<u32> = all.iter().filter(|e| e[3] == 0 && e[1] != LOCATION_HAND && e[5] != 0 && e[4] == code).map(|e| e[5]).collect();
        for &a in &moved {
            for &b in &in_hand {
                if let Some(f @ Fix::Swap(at, _, _)) = self.history_swap(a, b, cp)
                    && at != (0, 0)
                {
                    return Some(f);
                }
            }
        }
        None
    }

    /// The real game moves a hidden card that the replay has right there but does not move.
    fn unmoved_copy_fix(&mut self, exp: &Value, duel: &RDuel, cp: Cp) -> Option<Fix> {
        let f = exp.g("from");
        let all = duel.all_cards();
        let lst: Vec<(u32, u32)> = all.iter().filter(|e| e[3] == 0 && e[0] == f.i("con") as u32 && e[1] == f.i("loc") as u32).map(|e| (e[4], e[5])).collect();
        let s = f.i("seq") as usize;
        if s >= lst.len() || exp.g("code").truthy() && lst[s].0 as i64 != exp.i("code") {
            return None;
        }
        let (code_a, a) = lst[s];
        // (the lists in location order of all_cards)
        for e in all.iter().filter(|e| e[3] == 0) {
            let (code, b) = (e[4], e[5]);
            if b != 0 && b != a && code == code_a
                && let Some(fx @ Fix::Swap(at, _, _)) = self.history_swap(a, b, cp)
                && at != (0, 0)
            {
                return Some(fx);
            }
        }
        None
    }

    fn stale_trigger_fix(&mut self, ch: &Value) -> Option<Fix> {
        let v = self.lock_var(ch.i("con") as u32, ch.i("loc") as u32, ch.i("seq") as u32).ok().flatten()?;
        let member = {
            let t = self.lock().ok()?;
            let x = &t.vars[v];
            if x.kind != Kind::Claim || x.same_as.is_some() {
                return None;
            }
            let pool = &t.pools[x.pool?];
            let ts = ch.i("trigger_seq");
            if pool.location != PoolLoc::Loc(LOCATION_HAND) || ts < 0 || ts as usize >= pool.members.len() {
                return None;
            }
            pool.members[ts as usize]
        };
        self.ident_fix((self.pos, key3(ch), Same::Var(member)))
    }

    /// The card the latest move brought to `l` is the physical card that the one at `other`
    /// descends from in the replay.
    fn moved_identify_fix(&mut self, l: Slot, other: Slot) -> Option<Fix> {
        let mv = (self.pos.saturating_sub(5)..self.pos).rev().find(|&j| self.iv(j).t() == "MOVE" && key3(self.iv(j).g("to")) == l);
        let mv = if self.pos >= 6 { mv } else { (0..self.pos).rev().find(|&j| self.iv(j).t() == "MOVE" && key3(self.iv(j).g("to")) == l) };
        let solved = self.solved.clone();
        let mv = mv?;
        let solved = solved?;
        let frm = key3(self.iv(mv).g("from"));
        let ve = self.lock_var(l.0, l.1, l.2).ok().flatten()?;
        let vg = self.lock_var(other.0, other.1, other.2).ok().flatten()?;
        let pool = {
            let t = self.lock().ok()?;
            let x = &t.vars[ve];
            if x.kind != Kind::Claim || x.same_as.is_some() {
                return None;
            }
            x.pool?
        };
        let sv = solved.vars.get(vg)?;
        let member = if sv.kind == Kind::Claim { sv.assigned } else { None }?;
        if !self.lock().ok()?.pools[pool].members.contains(&member) {
            return None;
        }
        self.ident_fix((mv, frm, Same::Var(member)))
    }

    fn later_copy_fix(&mut self, cp: Cp, msgs: &[Msg], idx: usize, got: &Value, duel: &RDuel) -> Option<Fix> {
        let mut earlier: Vec<Value> = Vec::new();
        for j in self.pos..(self.pos + 12).min(self.items.len()) {
            let m = self.iv(j);
            if !matches!(m.t(), "MOVE" | "CARD_HINT" | "HINT") {
                break;
            }
            if m.t() != "MOVE" {
                continue;
            }
            if m.g("code") == got.g("code") && m.g("from").i("con") == got.g("from").i("con") && m.g("from").i("loc") == got.g("from").i("loc") {
                // where that card was before the earlier moves of the operation
                let mut src = m.g("from").clone();
                for e in earlier.iter().rev() {
                    let f = e.g("from");
                    if f.i("con") == src.i("con") && f.i("loc") == src.i("loc") && f.i("seq") <= src.i("seq") {
                        let s = src.i("seq") + 1;
                        src["seq"] = Value::from(s);
                    }
                }
                return self.swap_fix(cp, msgs, idx, &src, got.g("from"), duel);
            }
            earlier.push(m);
        }
        None
    }

    /// The replay shuffles the opponent's hand where the real game does not: a card that just left
    /// the hand was the other copy of it.
    fn moved_copy_fix(&mut self, duel: &RDuel, cp: Cp) -> Option<Fix> {
        let live = self.loc_map(duel);
        let hand = duel.location_cards(self.opponent, LOCATION_HAND);
        let lo = (self.pos as i64 - 15).max(-1);
        let mut j = self.pos as i64 - 1;
        while j > lo {
            let m = self.iv(j as usize);
            j -= 1;
            if m.t() != "MOVE" || m.g("from").i("con") as u32 != self.opponent || m.g("from").i("loc") as u32 != LOCATION_HAND {
                continue;
            }
            let a = live.at(&li_of(m.g("to")));
            for &(code, b) in &hand {
                if let Some(a) = a.filter(|&x| x != 0)
                    && Some(code) == live.codes.get(&a).copied()
                    && b != a
                    && let Some(f) = self.history_swap(a, b, cp)
                {
                    return Some(f);
                }
            }
        }
        None
    }

    /// Exchanges two copies whose positions or histories are visible.
    pub fn history_swap(&self, a: u32, b: u32, cp: Cp) -> Option<Fix> {
        let sa: BTreeSet<Cp> = self.card_shuffled.get(&a).map_or(BTreeSet::new(), |v| v.iter().copied().collect());
        let sb: BTreeSet<Cp> = self.card_shuffled.get(&b).map_or(BTreeSet::new(), |v| v.iter().copied().collect());
        let mut shared: Vec<Cp> = sa.intersection(&sb).copied().collect();
        shared.sort_by(|x, y| y.cmp(x));
        for at in shared {
            if at <= cp
                && !self.swaps.get(&at).is_some_and(|l| l.contains(&(a, b)) || l.contains(&(b, a)))
                && !self.failed_swaps.contains(&(at, a, b))
                && !self.failed_swaps.contains(&(at, b, a))
            {
                return Some(Fix::Swap(at, a, b));
            }
        }
        if self.role_swap_is_new(a, b) {
            return Some(Fix::Swap((0, 0), a, b));
        }
        None
    }

    /// Two copies on the field are the wrong way round: swap them just before the first of them
    /// left the pile they shared.
    fn field_copy_fix(&self, a: u32, b: u32) -> Option<Fix> {
        let (&(cpa, pa), &(cpb, pb)) = (self.left_pile.get(&a)?, self.left_pile.get(&b)?);
        if pa != pb {
            if pa.0 != pb.0 || pa.1 == LOCATION_EXTRA || pb.1 == LOCATION_EXTRA {
                return None;
            }
            // (e.g. one was drawn first) they were together in the deck before
            return self.history_swap(a, b, cpa.max(cpb));
        }
        Some(Fix::Swap(cpa.min(cpb), a, b))
    }

    pub fn role_swap_is_new(&self, a: u32, b: u32) -> bool {
        let pairs: Vec<(u32, u32)> = self.swaps.get(&(0, 0)).cloned().unwrap_or_default();
        let set = |r: BTreeMap<u32, u32>| -> BTreeSet<(u32, u32)> { r.into_iter().filter(|(k, v)| k != v).collect() };
        let tried: BTreeSet<BTreeSet<(u32, u32)>> = (0..=pairs.len()).map(|i| set(roles(&pairs[..i]))).collect();
        let mut with = pairs.clone();
        with.push((a, b));
        !tried.contains(&set(roles(&with)))
    }

    /// The viewer is offered an effect the replay does not offer: assume one copy at the bottom of
    /// the opponent's deck, once.
    pub fn enabler_fix(&mut self, exp: &Value, got: &Value, duel: &RDuel) -> Option<Fix> {
        let key = if exp.has("chains") { "chains" } else { "activatable" };
        if !exp.has(key) || (exp.t() != got.t() && exp.t() != "SELECT_CHAIN") {
            return None;
        }
        let mut have: HashMap<(i64, i64), i64> = HashMap::new();
        if exp.t() == got.t() {
            for o in got.g(key).l() {
                *have.entry((o.i("code"), o.i("desc"))).or_default() += 1;
            }
        }
        let n = duel.location_cards(self.opponent, LOCATION_DECK).len();
        for o in exp.g(key).l().to_vec() {
            if have.get(&(o.i("code"), o.i("desc"))).copied().unwrap_or(0) > 0 || o.i("con") as u32 != self.viewer || n == 0 {
                continue;
            }
            let code = o.i("code") as u32;
            if self.assumptions.iter().chain(self.failed_assumptions.iter()).any(|a| a.4 == code && (a.1, a.2) == (self.opponent, LOCATION_DECK)) {
                continue;
            }
            return self.assume_fix((self.opponent, LOCATION_DECK, 0), &[code]);
        }
        None
    }

    /// The viewer's prompt offers an effect granted by Xyz materials a different number of times.
    pub fn granted_copy_fix(&mut self, exp: &Value, got: &Value, duel: &RDuel) -> Option<Fix> {
        if exp.t() != got.t() || !exp.has("activatable") {
            return None;
        }
        let mut ce: BTreeMap<(i64, i64), i64> = BTreeMap::new();
        for o in exp.g("activatable").l() {
            *ce.entry((o.i("code"), o.i("desc"))).or_default() += 1;
        }
        let mut cg: HashMap<(i64, i64), i64> = HashMap::new();
        for o in got.g("activatable").l() {
            *cg.entry((o.i("code"), o.i("desc"))).or_default() += 1;
        }
        for (&(code, desc), &n) in &ce {
            if cg.get(&(code, desc)).copied().unwrap_or(0) == n || desc >> 20 == 0 || desc >> 20 == code {
                continue;
            }
            let src = (desc >> 20) as u32;
            let o = exp.g("activatable").l().iter().find(|o| (o.i("code"), o.i("desc")) == (code, desc)).unwrap();
            if o.i("loc") as u32 & LOCATION_MZONE == 0 {
                continue;
            }
            let attached: Vec<u32> = duel.overlay_cards(o.i("con") as u32, LOCATION_MZONE, o.i("seq") as u32).into_iter().filter(|&(c, _)| c == src).map(|x| x.1).collect();
            let mut others = Vec::new();
            for p in 0..2 {
                for loc in [LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_HAND, LOCATION_DECK, LOCATION_EXTRA] {
                    others.extend(duel.location_cards(p, loc).into_iter().filter(|&(c, _)| c == src).map(|x| x.1));
                }
            }
            for &a in &attached {
                for &b in &others {
                    if self.role_swap_is_new(a, b) && !self.failed_swaps.contains(&((0, 0), a, b)) && !self.failed_swaps.contains(&((0, 0), b, a)) {
                        return Some(Fix::Swap((0, 0), a, b));
                    }
                }
            }
        }
        None
    }

    /// The viewer's prompt lists a different copy of a card: swap the copies.
    pub fn prompt_copy_fix(&mut self, exp: &Value, got: &Value, duel: &RDuel, cp: Cp) -> Option<Fix> {
        let live = self.loc_map(duel);
        let mut pairs: Vec<(Value, Value)> = Vec::new();
        if ["con", "loc", "seq"].iter().all(|k| exp.has(k) && got.has(k)) {
            pairs.push((exp.clone(), got.clone()));
        }
        for key in ["summonable", "spsummonable", "repositionable", "msetable", "ssetable", "activatable", "attackable", "chains", "cards", "select_cards", "unselect_cards"] {
            let (el, gl) = (exp.g(key).l().to_vec(), got.g(key).l().to_vec());
            if el.len() == gl.len() {
                let de: Vec<Value> = el.iter().filter(|e| !gl.contains(e)).cloned().collect();
                let dg: Vec<Value> = gl.iter().filter(|g| !el.contains(g)).cloned().collect();
                let mut ce: Vec<i64> = de.iter().map(|e| e.i("code")).collect();
                let mut cg: Vec<i64> = dg.iter().map(|g| g.i("code")).collect();
                ce.sort();
                cg.sort();
                if de.len() == dg.len() && ce == cg {
                    let ecodes: Vec<Value> = de.iter().map(|e| e.g("code").clone()).collect();
                    let mut dg = dg;
                    dg.sort_by_key(|g| ecodes.iter().position(|c| c == g.g("code")).unwrap_or(0));
                    pairs.extend(de.into_iter().zip(dg));
                    continue;
                }
            }
            pairs.extend(el.into_iter().zip(gl));
        }
        for (e, g) in pairs {
            if e == g {
                continue;
            }
            if e.g("code") != g.g("code") || e.i("con") != g.i("con") || e.i("loc") != g.i("loc") || e.i("seq") == g.i("seq") {
                return None;
            }
            let (a, b) = (live.at(&li_of(&g)).unwrap_or(0), live.at(&li_of(&e)).unwrap_or(0));
            if a == 0 || b == 0 || live.codes.get(&a) != live.codes.get(&b) {
                return None;
            }
            let (gcon, gloc) = (g.i("con") as u32, g.i("loc") as u32);
            if matches!(gloc, LOCATION_MZONE | LOCATION_SZONE) {
                if self.last_shuffle_cp.contains_key(&(gcon, gloc))
                    && !self.failed_swaps.contains(&(cp, a, b))
                    && !self.failed_swaps.contains(&(cp, b, a))
                    && !self.swaps.get(&cp).is_some_and(|l| l.contains(&(a, b)) || l.contains(&(b, a)))
                {
                    return Some(Fix::Swap(cp, a, b));
                }
                return self.field_copy_fix(a, b);
            }
            if matches!(gloc, LOCATION_GRAVE | LOCATION_REMOVED) && self.role_swap_is_new(a, b) {
                return Some(Fix::Swap((0, 0), a, b));
            }
            let mut at = self.last_shuffle_cp.get(&(gcon, gloc)).copied().unwrap_or(cp);
            if self.swaps.get(&at).is_some_and(|l| l.contains(&(a, b)) || l.contains(&(b, a))) {
                at = cp;
            }
            return Some(Fix::Swap(at, a, b));
        }
        None
    }

    /// A card in the state view differs from the real one, but another copy of it in the replay
    /// looks exactly like it: swap the two copies.
    pub fn state_copy_fix(&mut self, exp: &Value, got: &Value, duel: &RDuel, cp: Cp) -> Option<Fix> {
        let live = self.loc_map(duel);
        for (le, lg) in exp.g("chain").l().iter().zip(got.g("chain").l()) {
            let (he, hg) = (le.g("handler"), lg.g("handler"));
            if le.g("code") != lg.g("code") || !he.truthy() || !hg.truthy() || infer::same_loc(&li_of(he), &li_of(hg)) {
                continue;
            }
            let a = live.at(&li_of(hg)).filter(|&c| c != 0);
            let mut cands: Vec<Option<u32>> = vec![live.at(&li_of(he))];
            cands.extend(duel.location_cards(he.i("con") as u32, he.i("loc") as u32 & 0xff).into_iter().filter(|&(c, _)| c as i64 == le.i("code")).map(|x| Some(x.1)));
            let mut seen = Vec::new();
            for b in cands.into_iter().flatten() {
                if b == 0 || Some(b) == a || seen.contains(&b) {
                    continue;
                }
                seen.push(b);
                if live.codes.get(&b).copied().unwrap_or(le.i("code") as u32) as i64 != le.i("code") {
                    continue;
                }
                if let Some(a) = a
                    && let Some(f) = self.history_swap(a, b, cp)
                {
                    return Some(f);
                }
            }
        }
        for p in 0..2u32 {
            for (name, loc) in LOCATIONS {
                let (ec, gc) = (exp.g("players").l()[p as usize].g(name).l().to_vec(), got.g("players").l()[p as usize].g(name).l().to_vec());
                if ec.len() != gc.len() {
                    return None;
                }
                for (i, (e, g)) in ec.iter().zip(&gc).enumerate() {
                    if e == g {
                        continue;
                    }
                    if !e.truthy() || !g.truthy() || !e.g("code").truthy() || e.g("code") != g.g("code") {
                        return None;
                    }
                    if matches!(loc, LOCATION_MZONE | LOCATION_SZONE) && diff_keys(e, g) == vec!["status".to_string()] && (e.i("status") ^ g.i("status")) as u32 & OPERATION_STATUS == 0 {
                        // only the card's history differs: another copy from a hidden pile
                        let a = live.at(&li(p, loc, i as u32)).filter(|&c| c != 0);
                        if let Some(a) = a
                            && self.last_shuffle_cp.contains_key(&(p, loc))
                        {
                            for (code, b) in duel.location_cards(p, loc) {
                                if code as i64 == e.i("code")
                                    && b != a
                                    && !self.failed_swaps.contains(&(cp, a, b))
                                    && !self.failed_swaps.contains(&(cp, b, a))
                                    && !self.swaps.get(&cp).is_some_and(|l| l.contains(&(a, b)) || l.contains(&(b, a)))
                                {
                                    return Some(Fix::Swap(cp, a, b));
                                }
                            }
                        }
                        for hloc in [LOCATION_DECK, LOCATION_HAND, LOCATION_EXTRA] {
                            for (code, b) in duel.location_cards(p, hloc) {
                                if code as i64 == e.i("code")
                                    && let Some(a) = a
                                    && b != a
                                    && let Some(f) = self.history_swap(a, b, cp)
                                {
                                    return Some(f);
                                }
                            }
                        }
                    }
                    for (j, g2) in gc.iter().enumerate() {
                        if j == i || g2 != e || &ec[j] == g2 {
                            continue;
                        }
                        let a = live.at(&li(p, loc, i as u32)).filter(|&c| c != 0);
                        let b = live.at(&li(p, loc, j as u32)).filter(|&c| c != 0);
                        let (Some(a), Some(b)) = (a, b) else { return None };
                        if matches!(loc, LOCATION_MZONE | LOCATION_SZONE) {
                            return self.field_copy_fix(a, b);
                        }
                        if matches!(loc, LOCATION_GRAVE | LOCATION_REMOVED) {
                            return self.history_swap(a, b, cp);
                        }
                        let mut at = self.copy_swap_cp((p, loc), cp);
                        if self.swaps.get(&at).is_some_and(|l| l.contains(&(a, b)) || l.contains(&(b, a))) {
                            at = cp;
                        }
                        return Some(Fix::Swap(at, a, b));
                    }
                    if matches!(loc, LOCATION_GRAVE | LOCATION_REMOVED) {
                        // no copy looks exactly like it: try the copies of the same pile the other way round
                        let a = live.at(&li(p, loc, i as u32)).filter(|&c| c != 0);
                        for (j, g2) in gc.iter().enumerate() {
                            if j != i && g2.truthy() && g2.g("code") == e.g("code") {
                                let b = live.at(&li(p, loc, j as u32)).filter(|&c| c != 0);
                                if let (Some(a), Some(b)) = (a, b)
                                    && let Some(f) = self.history_swap(a, b, cp)
                                {
                                    return Some(f);
                                }
                            }
                        }
                    }
                    return None;
                }
            }
        }
        None
    }

    /// The viewer selects among the opponent's hidden cards: the hypothesis must put eligible cards
    /// exactly where the prompt shows them.
    pub fn hidden_candidates_fix(&mut self, exp: &Value, got: &Value, duel: Option<&mut RDuel>) -> Option<Fix> {
        let key = if exp.has("cards") { "cards" } else { "select_cards" };
        if exp.t() != got.t() || !exp.has(key) || !got.has(key) {
            return None;
        }
        let opp = self.opponent;
        let hidden = |c: &Value| {
            c.i("con") as u32 == opp
                && !c.g("code").truthy()
                && (hidden_pile(c.i("loc") as u32) || (matches!(c.i("loc") as u32, LOCATION_MZONE | LOCATION_SZONE) && c.o("pos").unwrap_or(POS_FACEDOWN as i64) as u32 & POS_FACEDOWN != 0))
        };
        let want: BTreeSet<Slot> = exp.g(key).l().iter().filter(|c| hidden(c)).map(key3).collect();
        let have: BTreeSet<Slot> = got.g(key).l().iter().filter(|c| hidden(c)).map(key3).collect();
        if want == have {
            return None;
        }
        let hyp = self.hypothesis.clone()?;
        for l in have.difference(&want) {
            if matches!(l.1, LOCATION_MZONE | LOCATION_SZONE) {
                // a face-down card that is not eligible: not the replay's card
                let var = self.lock_var(l.0, l.1, l.2).ok()?;
                let code = var.and_then(|v| hyp.var_codes.get(&v).copied().flatten());
                if let Some(code) = code.filter(|&c| c != 0) {
                    let entry: Excl = (self.pos, l.0, l.1, vec![l.2 as i64], code);
                    if !self.exclusions.contains(&entry) && !self.failed_exclusions.contains(&entry) {
                        return Some(Fix::Exclude(entry));
                    }
                }
            }
        }
        let field_missing: Vec<Slot> = want.difference(&have).copied().filter(|x| matches!(x.1, LOCATION_MZONE | LOCATION_SZONE)).collect();
        let want: BTreeSet<Slot> = want.into_iter().filter(|x| hidden_pile(x.1)).collect();
        let have: BTreeSet<Slot> = have.into_iter().filter(|x| hidden_pile(x.1)).collect();
        let duel_ptr: Option<&mut RDuel> = duel;
        let mut duel_ref = duel_ptr;
        if have.difference(&want).next().is_some()
            && let Some(d) = duel_ref.as_deref()
        {
            // slots the real game leaves out: a recent hidden selection of the opponent took that card
            let live = self.loc_map(d);
            for l in have.difference(&want) {
                let cid = live.at(&li(l.0, l.1, l.2));
                for &(dkey, _, i) in self.recent.iter().rev() {
                    if let Some(&k) = self.card_decisions.get(&dkey).and_then(|m| m.get(&cid))
                        && k as i64 != i
                    {
                        return Some(Fix::Decide(dkey, k));
                    }
                }
            }
        }
        // an eligible code, taken from one of the replay's eligible cards
        let mut codes: Vec<u32> = Vec::new();
        let both: BTreeSet<Slot> = {
            let a: BTreeSet<Slot> = exp.g(key).l().iter().filter(|c| hidden(c)).map(key3).collect();
            let b: BTreeSet<Slot> = got.g(key).l().iter().filter(|c| hidden(c)).map(key3).collect();
            a.intersection(&b).copied().collect()
        };
        let mut order: Vec<Slot> = have.difference(&want).copied().collect();
        order.extend(both.iter().copied());
        for c in order {
            let var = self.lock_var(c.0, c.1, c.2).ok().flatten();
            if let Some(code) = var.and_then(|v| hyp.var_codes.get(&v).copied().flatten()).filter(|&c| c != 0) {
                codes.push(code);
            }
        }
        // (which codes make the face-down card eligible here, the engine tells)
        let mut check: Option<(&'static str, (u8, u32, u32))> = None;
        if !field_missing.is_empty() && duel_ref.is_some() && exp.t() == "SELECT_TRIBUTE" {
            if let Some(s) = self.summoning_card() {
                check = Some(("tribute", (s.i("con") as u8, s.i("loc") as u32, s.i("seq") as u32)));
            }
        } else if !field_missing.is_empty() && duel_ref.is_some() && exp.t() == "SELECT_CARD" && self.select_hint() == Some(549)
            && let Some(a) = self.attacking_card()
        {
            check = Some(("attack", (a.i("con") as u8, a.i("loc") as u32, a.i("seq") as u32)));
        }
        if let Some(chk) = check
            && let Some(d) = duel_ref.as_mut()
        {
            codes = self.eligible_codes(d, field_missing[0], &codes, chk, &|_| true);
        }
        let targets: Vec<Slot> = want.difference(&have).copied().chain(field_missing.iter().copied()).collect();
        for l in targets {
            if let Some(f) = self.assume_fix(l, &codes) {
                return Some(f);
            }
        }
        let filler: Vec<u32> = self.filler().main_pool.clone();
        for l in have.difference(&want).copied().collect::<Vec<_>>() {
            if let Some(f) = self.assume_fix(l, &filler) {
                return Some(f);
            }
        }
        None
    }

    /// Monster codes for the hidden card at `l` that pass the engine check, of `codes` first, then of
    /// all main deck monsters, until three are found.
    pub fn eligible_codes(&mut self, duel: &mut RDuel, l: Slot, codes: &[u32], check: (&str, (u8, u32, u32)), fits: &dyn Fn(u32) -> bool) -> Vec<u32> {
        let mut seen_races: HashMap<u64, i64> = HashMap::new();
        for c in self.opp_seen_codes() {
            if let Some(d) = card(c).filter(|d| d.type_ & TYPE_MONSTER != 0) {
                *seen_races.entry(d.race).or_default() += 1;
            }
        }
        let db = crate::game::card_db();
        let mut pool: Vec<u32> = db
            .cards
            .values()
            .filter(|c| c.type_ & TYPE_MONSTER != 0 && c.alias == 0 && db.ots.get(&c.code).copied().unwrap_or(0) & 0x3 != 0 && !is_extra_deck(c) && c.type_ & TYPE_TOKEN == 0 && fits(c.code))
            .map(|c| c.code)
            .collect();
        pool.sort_by_key(|&c| {
            let d = card(c).unwrap();
            (-seen_races.get(&d.race).copied().unwrap_or(0), d.type_ & TYPE_NORMAL == 0, c)
        });
        let mut found: Vec<u32> = Vec::new();
        for &c in codes {
            if fits(c) && self.eligible_as(duel, l, &[c], check).get(&c).copied().unwrap_or(false) {
                found.push(c);
            }
        }
        let mut k = 0;
        while k < pool.len() {
            if found.len() >= 3 {
                break;
            }
            let chunk = &pool[k..(k + 400).min(pool.len())];
            let ok = self.eligible_as(duel, l, chunk, check);
            for &c in chunk {
                if ok.get(&c).copied().unwrap_or(false) && !found.contains(&c) {
                    found.push(c);
                }
            }
            k += 400;
        }
        found
    }

    /// The opponent Tributes a public monster for the summon or Set of a hidden hand card, and the
    /// replay does not: which codes it can be Tributed for, the engine tells.
    fn tribute_summoner_fix(&mut self, exp: &Value, got: &Value, duel: &mut RDuel) -> Option<Fix> {
        let r = exp.i("reason") as u32;
        if !(r & REASON_RELEASE != 0 && r & REASON_SUMMON != 0) || !exp.g("code").truthy() || exp.g("from").i("loc") as u32 != LOCATION_MZONE || (got.t() == "MOVE" && got.g("from") == exp.g("from")) {
            return None;
        }
        let (mut tributes, mut summoned): (u32, Option<Value>) = (0, None);
        for j in self.pos..self.items.len().min(self.pos + 10) {
            let m = self.iv(j);
            if m.t() == "MOVE" && m.i("reason") as u32 & REASON_RELEASE != 0 && m.i("reason") as u32 & REASON_SUMMON != 0 {
                tributes += 1;
            } else if m.t() == "MOVE" && m.g("from").i("con") as u32 == self.opponent && m.g("from").i("loc") as u32 == LOCATION_HAND && m.g("to").i("loc") as u32 == LOCATION_MZONE {
                summoned = Some(m.g("from").clone());
                break;
            }
        }
        for j in (self.pos.saturating_sub(3)..self.pos).rev() {
            let m = self.iv(j);
            if m.t() == "MOVE" && m.i("reason") as u32 & REASON_RELEASE != 0 && m.i("reason") as u32 & REASON_SUMMON != 0 {
                tributes += 1;
            }
        }
        let summoned = summoned?;
        if summoned.g("code").truthy() {
            return None;
        }
        let levels = crate::belief::completion::tribute_levels_any(tributes);
        let l = (summoned.i("con") as u32, LOCATION_HAND, summoned.i("seq") as u32);
        let t = exp.g("from");
        let check = ("tribute_of", (t.i("con") as u8, t.i("loc") as u32, t.i("seq") as u32));
        let seen = self.opp_seen_codes();
        let codes = self.eligible_codes(duel, l, &seen, check, &|code| card(code).is_some_and(|c| levels.contains(&c.level)));
        self.assume_fix(l, &codes)
    }

    /// The viewer's card being Normal Summoned or Set: from its latest answer.
    fn summoning_card(&self) -> Option<Value> {
        let lo = self.pos.saturating_sub(19);
        for j in (lo..self.pos).rev() {
            let ObsItem::Response(r) = &self.items[j] else { continue };
            if matches!(r.kind.as_deref(), Some("summon") | Some("mset")) {
                let prompt = (j.saturating_sub(4)..j).rev().find_map(|k| if let ObsItem::Prompt(p, _) = &self.items[k] { Some(jv(p)) } else { None });
                let key = if r.kind.as_deref() == Some("summon") { "summonable" } else { "msetable" };
                let lst = prompt.as_ref().map_or(vec![], |p| p.g(key).l().to_vec());
                let i = r.index.unwrap_or(-1);
                return if i >= 0 && (i as usize) < lst.len() { Some(lst[i as usize].clone()) } else { None };
            }
        }
        None
    }

    /// The select message (HINT_SELECTMSG) of the viewer's current prompt.
    pub fn select_hint(&self) -> Option<u64> {
        let mut j = self.pos as i64 - 1;
        while j >= 0 {
            let Some(Msg::Hint { hint_type, player, data }) = self.imsg(j as usize) else { break };
            if *hint_type == HINT_SELECTMSG && *player == self.viewer {
                return Some(*data);
            }
            j -= 1;
        }
        None
    }

    /// The viewer's attacking monster: from its latest attack answer.
    fn attacking_card(&self) -> Option<Value> {
        let lo = self.pos.saturating_sub(11);
        for j in (lo..self.pos).rev() {
            let ObsItem::Response(r) = &self.items[j] else { continue };
            if r.kind.as_deref() == Some("attack") {
                let prompt = (j.saturating_sub(4)..j).rev().find_map(|k| if let ObsItem::Prompt(p, _) = &self.items[k] { Some(jv(p)) } else { None });
                let lst = prompt.as_ref().map_or(vec![], |p| p.g("attackable").l().to_vec());
                let i = r.index.unwrap_or(-1);
                return if i >= 0 && (i as usize) < lst.len() { Some(lst[i as usize].clone()) } else { None };
            }
        }
        None
    }

    /// A card's properties for eligible_as / settable_as (None: not in the database).
    pub fn props(code: u32) -> Option<crate::rduel::Props> {
        let c = card(code)?;
        Some((code, c.type_, c.level, c.attribute, c.race, c.attack.max(0), c.defense.max(0)))
    }

    pub fn eligible_as(&self, duel: &mut RDuel, l: Slot, codes: &[u32], check: (&str, (u8, u32, u32))) -> HashMap<u32, bool> {
        let mut uniq: Vec<u32> = Vec::new();
        for &c in codes {
            if !uniq.contains(&c) {
                uniq.push(c);
            }
        }
        let cands: Vec<crate::rduel::Props> = uniq.into_iter().filter_map(Self::props).collect();
        crate::rduel::eligible_as(&mut duel.d, (l.0 as u8, l.1, l.2), check.0, check.1, &cands).into_iter().collect()
    }

    /// Swap the card at `loc` with another copy of it from the same pile.
    fn other_copy_fix(&mut self, cp: Cp, msgs: &[Msg], idx: usize, loc: &Value) -> Option<Fix> {
        let Some(snap) = self.snapshots.get(&cp).cloned() else { return if self.snapshot_cps.contains(&cp) { None } else { Some(Fix::Snapshot(cp)) } };
        let mut lm = (*snap).clone();
        lm.references(&msgs[..idx]);
        let a = lm.at(&li_of(loc));
        let pile: Vec<Option<u32>> = lm.lists.get(&(loc.i("con") as u32, loc.i("loc") as u32)).cloned().unwrap_or_default();
        let at = self.copy_swap_cp((loc.i("con") as u32, loc.i("loc") as u32), cp);
        let tried: BTreeSet<BTreeSet<u32>> = self.swaps.get(&at).map_or(BTreeSet::new(), |l| l.iter().map(|&(x, y)| [x, y].into_iter().collect()).collect());
        let a = a?;
        for b in pile.into_iter().flatten() {
            if b != 0 && b != a && snap.codes.get(&b) == snap.codes.get(&a) && !tried.contains(&[a, b].into_iter().collect()) {
                return Some(Fix::Swap(at, a, b));
            }
        }
        None
    }

    /// Copies are best exchanged right before their pile was last shuffled; otherwise now.
    pub fn copy_swap_cp(&self, pile: (u32, u32), cp: Cp) -> Cp {
        match self.last_shuffle_cp.get(&pile) {
            Some(&at) if at <= cp => at,
            _ => cp,
        }
    }

    /// A card hint sits on the wrong copy of a card: swap it with the copy the real game hinted.
    fn hint_swap_fix(&self, cp: Cp, got: &Value, duel: &RDuel) -> Option<Fix> {
        let live = self.loc_map(duel);
        let a = live.at(&li_of(got)).filter(|&c| c != 0)?;
        let mut j = self.pos;
        while j < self.items.len() && self.itype(j) == "CARD_HINT" {
            let e = self.iv(j);
            if (e.i("hint_type"), e.i("value")) == (got.i("hint_type"), got.i("value")) && !infer::same_loc(&li_of(&e), &li_of(got))
                && let Some(b) = live.at(&li_of(&e)).filter(|&c| c != 0)
                && b != a
                && live.codes.get(&b) == live.codes.get(&a)
            {
                return Some(Fix::Swap(cp, a, b));
            }
            j += 1;
        }
        None
    }

    pub fn has_swap(&self, fix: &Fix) -> bool {
        let Fix::Swap(at, a, b) = fix else { return false };
        self.swaps.get(at).is_some_and(|l| l.contains(&(*a, *b)) || l.contains(&(*b, *a)))
    }

    /// Whether the viewer answered a prompt since the chain link (or action) being processed began.
    fn viewer_prompted(&self) -> bool {
        let ls = self.link_start();
        (ls..self.pos).rev().any(|j| matches!(self.items[j], ObsItem::Prompt(..)))
    }

    /// No card below `loc` can be this card (scans run from the bottom of the pile).
    fn exclude_fix(&self, l: &Value, code: Option<i64>) -> Option<Fix> {
        let code = code.filter(|&c| c != 0)? as u32;
        if !hidden_pile(l.i("loc") as u32) || l.i("seq") == 0 {
            return None;
        }
        let key: Excl = (self.pos, l.i("con") as u32, l.i("loc") as u32, (0..l.i("seq")).collect(), code);
        if self.failed_exclusions.contains(&key) || self.exclusions.contains(&key) { None } else { Some(Fix::Exclude(key)) }
    }

    /// Two copies of the same card are processed in the wrong order.
    pub fn swap_fix(&mut self, cp: Cp, msgs: &[Msg], idx: usize, loc_exp: &Value, loc_got: &Value, duel: &RDuel) -> Option<Fix> {
        let Some(snap) = self.snapshots.get(&cp).cloned() else { return if self.snapshot_cps.contains(&cp) { None } else { Some(Fix::Snapshot(cp)) } };
        let mut lm = (*snap).clone();
        lm.references(&msgs[..idx]);
        let (mut a, mut b) = (lm.at(&li_of(loc_got)), lm.at(&li_of(loc_exp)));
        if a.is_none() || b.is_none() {
            let now = self.loc_map(duel);
            a = now.at(&li_of(loc_got));
            b = now.at(&li_of(loc_exp));
        }
        let (a, b) = (a.filter(|&c| c != 0)?, b.filter(|&c| c != 0)?);
        if a == b || snap.codes.get(&a) != snap.codes.get(&b) {
            return None; // only copies of the same card may trade places
        }
        if matches!(loc_exp.i("loc") as u32, LOCATION_GRAVE | LOCATION_REMOVED)
            && let Some(f) = self.history_swap(a, b, cp)
        {
            return Some(f);
        }
        let mut at = self.copy_swap_cp((loc_exp.i("con") as u32, loc_exp.i("loc") as u32), cp);
        if self.swaps.get(&at).is_some_and(|l| l.contains(&(a, b)) || l.contains(&(b, a))) {
            at = cp;
        }
        Some(Fix::Swap(at, a, b))
    }

    /// The real game moves every copy of a card out of the opponent's Deck, then shuffles it: none
    /// of the guessed cards of that Deck is one.
    fn all_copies_fix(&mut self, exp: &Value, le: &Value, duel: &RDuel, got: Option<&Value>) -> Option<Fix> {
        let code = exp.g("code").clone();
        let mut moved: BTreeSet<i64> = BTreeSet::new();
        let mut ended = false;
        for j in self.pos..(self.pos + 20).min(self.items.len()) {
            let m = self.iv(j);
            if m.is_null() || matches!(m.t(), "HINT" | "CARD_HINT") {
                continue;
            }
            if m.t() == "MOVE" && *m.g("code") == code && m.i("reason") == exp.i("reason") && (m.g("from").i("con"), m.g("from").i("loc")) == (le.i("con"), le.i("loc")) {
                moved.insert(m.g("from").i("seq"));
                continue;
            }
            if m.t() != "SHUFFLE_DECK" || m.i("player") != le.i("con") {
                return None;
            }
            ended = true;
            break;
        }
        if !ended {
            return None;
        }
        let (con, loc) = (le.i("con") as u32, le.i("loc") as u32);
        let mut n = duel.location_cards(con, loc).len();
        if let Some(g) = got
            && g.t() == "MOVE"
            && (g.g("from").i("con") as u32, g.g("from").i("loc") as u32) == (con, loc)
            && (g.g("to").i("con") as u32, g.g("to").i("loc") as u32) != (con, loc)
        {
            n += 1;
        }
        let mut guessed: Vec<i64> = Vec::new();
        for seq in 0..n {
            if moved.contains(&(seq as i64)) {
                continue;
            }
            let Ok(Some(v)) = self.lock_var(con, loc, seq as u32) else { continue };
            let t = self.lock().unwrap();
            if t.vars[v].code.is_none() && t.vars[v].fixed_code.is_none() {
                guessed.push(seq as i64);
            }
        }
        let key: Excl = (self.pos, con, loc, guessed.clone(), code.as_i64().unwrap_or(0) as u32);
        if !guessed.is_empty() && !self.exclusions.contains(&key) && !self.failed_exclusions.contains(&key) {
            return Some(Fix::Exclude(key));
        }
        None
    }

    /// The real game Tributes the opponent's monsters for a Summon or Set the replay does not make.
    pub fn granted_set_fix(&mut self, exp: &Value, duel: &mut RDuel) -> Option<Fix> {
        if exp.t() != "MOVE" || exp.g("from").i("con") as u32 != self.opponent || exp.i("reason") as u32 & (REASON_RELEASE | REASON_SUMMON) != REASON_RELEASE | REASON_SUMMON {
            return None;
        }
        let up: Vec<Msg> = (self.pos..(self.pos + 16).min(self.items.len())).filter_map(|j| self.imsg(j).cloned()).collect();
        match self.check_idle_assumption(None, &up, Some(duel)) {
            Err(RErr::Div(e)) => e.fix,
            _ => None,
        }
    }

    /// Whether a card with `code` can be put face-down where `to` says.
    pub fn fits_destination(code: u32, to: &crate::messages::Loc) -> bool {
        let Some(c) = card(code) else { return true };
        if to.loc != LOCATION_SZONE || to.pos & POS_FACEDOWN == 0 {
            return true;
        }
        if to.seq == 5 { c.type_ & TYPE_FIELD != 0 } else { c.type_ & TYPE_FIELD == 0 }
    }

    /// Raises an "assume" fix for the first untried code, if the slot is free.
    pub fn assume(&mut self, l: Slot, codes: &[u32], reason: &str) -> R<()> {
        if let Some(fix) = self.assume_fix(l, codes) {
            return div(Divergence::with_fix(self.pos, reason, Some(fix)));
        }
        Ok(())
    }

    /// An "assume" fix for the first untried code, if the slot is free.
    pub fn assume_fix(&mut self, l: Slot, codes: &[u32]) -> Option<Fix> {
        let var = self.lock_var(l.0, l.1, l.2).ok()?;
        let var = var?;
        let observed = self.hypothesis.as_ref().is_some_and(|h| h.observed.contains(&var));
        let here = |a: &Asm| (a.0, a.1, a.2, a.3) == (self.pos, l.0, l.1, l.2);
        if observed && !self.assumptions.iter().any(here) {
            return None; // (observed, other than by an assumption at this slot and point)
        }
        let tried: BTreeSet<u32> = self.assumptions.iter().chain(self.failed_assumptions.iter()).filter(|a| here(a)).map(|a| a.4).collect();
        let mut checked = 0;
        for &code in codes {
            if code != 0 && !tried.contains(&code) {
                // (only codes the flow model admits there)
                if checked < 40 {
                    checked += 1;
                    if !self.feasible((self.pos, l.0, l.1, l.2, code)) {
                        continue;
                    }
                }
                return Some(Fix::Assume((self.pos, l.0, l.1, l.2, code)));
            }
        }
        None
    }

    /// Whether the hypothesis can be built with the assumption `entry` added.
    pub fn feasible(&mut self, entry: Asm) -> bool {
        if self.depth > 0 {
            return true; // (a probe only reports that a repair exists)
        }
        let mut a = self.assumptions.clone();
        a.sort();
        let mut e = self.exclusions.clone();
        e.sort();
        let key = (a, e, self.identities.clone(), entry);
        if let Some(&ok) = self.feasible_cache.get(&key) {
            return ok;
        }
        let saved = (self.assumptions.clone(), self.hypothesis.clone(), self.solved.clone(), self.assumed_vars.clone(), self.assume_conflict);
        self.assumptions.retain(|x| (x.0, x.1, x.2, x.3) != (entry.0, entry.1, entry.2, entry.3));
        self.assumptions.push(entry);
        let ok = self.build_hypothesis().is_ok();
        (self.assumptions, self.hypothesis, self.solved, self.assumed_vars, self.assume_conflict) = saved;
        if self.feasible_cache.len() > 2000 {
            self.feasible_cache.clear();
        }
        self.feasible_cache.insert(key, ok);
        ok
    }

    /// The codes of the player's cards that the stream has shown.
    pub fn known_codes(&mut self, player: u32) -> BTreeSet<u32> {
        let Ok(t) = self.lock() else { return BTreeSet::new() };
        t.vars.iter().filter(|v| v.owner == player).filter_map(|v| v.known_code()).filter(|&c| c != 0).collect()
    }

    /// The opponent puts a hidden hand card on the field as a monster and the replay does not.
    fn hidden_monster_fix(&mut self, exp: &Value, got: &Value) -> Option<Fix> {
        let frm = exp.g("from");
        if frm.i("con") as u32 != self.opponent || frm.i("loc") as u32 != LOCATION_HAND || exp.g("code").truthy() || exp.g("to").i("loc") as u32 != LOCATION_MZONE {
            return None;
        }
        if got.t() == "MOVE" && infer::same_loc(&li_of(got.g("from")), &li_of(frm)) {
            return None;
        }
        // (not a Tribute Set)
        for j in (self.pos.saturating_sub(11)..self.pos).rev() {
            let Some(m) = self.imsg(j) else { break };
            if let Msg::Move { reason, .. } = m
                && reason & REASON_RELEASE != 0
                && reason & REASON_SUMMON != 0
            {
                return None;
            }
            if matches!(m, Msg::ChainEnd {} | Msg::ChainSolved { .. } | Msg::NewPhase { .. } | Msg::Set { .. } | Msg::Summoned {} | Msg::Spsummoned {}) {
                break;
            }
        }
        let monster = |code: u32| card(code).is_some_and(|c| c.type_ & TYPE_MONSTER != 0 && !is_extra_deck(c) && c.type_ & TYPE_TOKEN == 0);
        let fits = |code: u32| monster(code) && card(code).is_some_and(|c| (1..=4).contains(&c.level));
        let seen = self.opp_seen_codes();
        let link = self.resolving_link();
        let rf = link.map(|l| self.iv(l).i("code") as u32).filter(|&c| c != 0);
        let mut listed: Vec<u32> = Vec::new();
        if let Some(rf) = rf {
            let names: BTreeSet<u32> = crate::belief::script_index::candidates(rf).into_iter().collect();
            listed = seen.iter().copied().filter(|c| names.contains(c) && monster(*c)).collect();
            for &c in &names {
                if !listed.contains(&c) && monster(c) {
                    listed.push(c);
                }
            }
            let slot = key3(frm);
            if !listed.is_empty() && !self.code_tests.iter().any(|e| e.0 == (self.pos, slot)) && !self.code_tests_done.contains(&(self.pos, slot)) {
                // (which of them the effect takes, the engine tells)
                return Some(Fix::CodeTest(link.unwrap(), self.pos, slot, listed.iter().copied().take(30).collect()));
            }
            listed.truncate(8);
        }
        let mut codes = listed.clone();
        codes.extend(seen.iter().copied().filter(|&c| fits(c) && !listed.contains(&c)));
        let filler = self.filler();
        for &c in filler.vanillas.iter().chain(&filler.settable_monsters_all) {
            if fits(c) && !codes.contains(&c) {
                codes.push(c);
            }
        }
        codes.truncate(listed.len() + 12);
        self.assume_fix(key3(frm), &codes)
    }
}

/// The keys whose values differ between two dicts.
pub fn diff_keys(e: &Value, g: &Value) -> Vec<String> {
    let mut keys: BTreeSet<String> = BTreeSet::new();
    if let Some(o) = e.as_object() {
        keys.extend(o.keys().cloned());
    }
    if let Some(o) = g.as_object() {
        keys.extend(o.keys().cloned());
    }
    keys.into_iter().filter(|k| e.g(k) != g.g(k)).collect()
}

/// The location a message is primarily about (replay._loc_of).
pub fn loc_of(m: &Value) -> Option<&Value> {
    if m.t() == "MOVE" {
        return Some(m.g("from"));
    }
    if m.has("loc") && m.has("con") && m.has("seq") {
        return Some(m);
    }
    None
}
