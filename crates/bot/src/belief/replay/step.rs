//! The replay's main loop (BeliefReplay._loop) and its small helpers.

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use serde_json::Value;
use ygo_core::constants::*;

use super::util::{V, jv};
use super::*;
use ygo_arbiter::actions::{Response, encode};
use crate::belief::infer::{self, LocMap};
use ygo_arbiter::messages::Msg;
use ygo_arbiter::observe::{auto_response, filter_message, filter_prompt, is_hidden_prompt, to_core_response};
use ygo_arbiter::state::{parse_location_query, viewer_view};

/// messages that leave every card where it was (the rolling snapshot stays valid)
fn no_moves(t: &str) -> bool {
    matches!(
        t,
        "HINT" | "CARD_HINT"
            | "PLAYER_HINT"
            | "CHAINING"
            | "CHAINED"
            | "CHAIN_SOLVING"
            | "CHAIN_SOLVED"
            | "CHAIN_END"
            | "CHAIN_NEGATED"
            | "CHAIN_DISABLED"
            | "NEW_PHASE"
            | "NEW_TURN"
            | "POS_CHANGE"
            | "SUMMONING"
            | "SUMMONED"
            | "SPSUMMONING"
            | "SPSUMMONED"
            | "FLIPSUMMONING"
            | "FLIPSUMMONED"
            | "BECOME_TARGET"
            | "CONFIRM_CARDS"
            | "DAMAGE"
            | "RECOVER"
            | "PAY_LPCOST"
            | "LPUPDATE"
            | "ATTACK"
            | "BATTLE"
            | "ATTACK_DISABLED"
            | "DAMAGE_STEP_START"
            | "DAMAGE_STEP_END"
            | "ADD_COUNTER"
            | "REMOVE_COUNTER"
            | "FIELD_DISABLED"
            | "CARD_SELECTED"
            | "RANDOM_SELECTED"
            | "SET"
    )
}

/// A stable text of a prompt (the Python module's repr(sorted(prompt.items()))).
pub fn prompt_sig(m: &Msg) -> String {
    serde_json::to_string(m).unwrap_or_default()
}

pub fn resp_sig(r: &Response) -> String {
    serde_json::to_string(r).unwrap_or_default()
}

pub fn other<E: std::fmt::Display>(e: E) -> RErr {
    RErr::Other(e.to_string())
}

/// Where every card is: id -> (player, location, index, code) (read_guided._where).
pub fn where_cards(duel: &RDuel) -> Where {
    duel.all_cards().into_iter().filter(|e| e[3] == 0 && e[5] != 0).map(|e| (e[5], (e[0], e[1], e[2], e[4]))).collect()
}

/// The card ids of the cards in the list slots and zones (all_cards()[0]: no materials) with their codes.
pub fn card_codes(duel: &RDuel) -> HashMap<u32, u32> {
    duel.all_cards().into_iter().filter(|e| e[3] == 0 && e[5] != 0).map(|e| (e[5], e[4])).collect()
}

impl BeliefReplay {
    pub fn loc_map(&self, duel: &RDuel) -> LocMap {
        LocMap::new(&duel.all_cards(), &duel.shuffle_plan)
    }

    pub fn apply_forces(&mut self, duel: &mut RDuel) {
        let active: BTreeSet<FKey> = self
            .force_windows
            .iter()
            .filter(|&&(start, end, k)| start <= self.pos && self.pos <= end && self.forces.contains(&k) && !self.unforced_windows.contains(&(start, end, k)))
            .map(|w| w.2)
            .collect();
        if active != self.active_forces {
            duel.clear_force_activatable();
            for &(p, c, d) in &active {
                duel.force_activatable(p, c, d);
            }
            self.active_forces = active;
        }
    }

    /// Runs of hint messages come out of hash sets in the core: accept any order.
    pub fn match_unordered_hint(&mut self, o: &Msg) -> bool {
        if !matches!(o, Msg::PlayerHint { .. } | Msg::CardHint { .. }) {
            return false;
        }
        let mut j = self.pos;
        while j < self.items.len() {
            let ObsItem::Msg(m) = &self.items[j] else { break };
            if !matches!(m, Msg::PlayerHint { .. } | Msg::CardHint { .. }) {
                break;
            }
            if m == o {
                self.items.swap(self.pos, j);
                return true;
            }
            j += 1;
        }
        false
    }

    /// Card ids of the opponent's cards the viewer cannot see (hidden piles and face-down cards
    /// on the field).
    pub fn hidden_ids(&self, duel: &mut RDuel) -> Vec<u32> {
        let mut out: Vec<u32> = HIDDEN_PILES.iter().flat_map(|&loc| duel.location_cards(self.opponent, loc).into_iter().map(|x| x.1)).collect();
        for loc in [LOCATION_MZONE, LOCATION_SZONE] {
            let cards = parse_location_query(&duel.query_location(self.opponent, loc, QUERY_POSITION));
            for ((_, cid), card) in duel.location_cards(self.opponent, loc).into_iter().zip(cards) {
                if cid != 0
                    && let Some(c) = card
                    && c.position.unwrap_or(0) & POS_FACEUP == 0
                {
                    out.push(cid);
                }
            }
        }
        out
    }

    /// Remembers when cards went from a pile to the field (for copy swaps).
    pub fn note_field_arrivals(&mut self, cp: Cp, msgs: &[Msg], duel: &RDuel) {
        for m in msgs {
            let Msg::Move { from: f, to: t, .. } = m else { continue };
            if (hidden_pile(f.loc) || f.loc == LOCATION_GRAVE || f.loc == LOCATION_REMOVED) && (t.loc == LOCATION_MZONE || t.loc == LOCATION_SZONE) {
                let cards = duel.location_cards(t.con, t.loc);
                if let Some(&(_, cid)) = cards.get(t.seq as usize)
                    && cid != 0
                {
                    let at = self.last_shuffle_cp.get(&(f.con, f.loc)).copied().unwrap_or_else(|| self.last_listed.get(&cid).copied().unwrap_or(cp));
                    self.left_pile.insert(cid, (at, (f.con, f.loc)));
                }
            }
        }
    }

    fn on_process(&mut self, duel: &mut RDuel) {
        if self.depth > 0 {
            if self.suspect_log.is_some() || self.logged_steps.is_some() {
                // (the probes' steps are not recorded; their reads still clear)
            }
            return;
        }
        let pos = self.pos;
        if let Some(steps) = self.logged_steps.as_mut() {
            let reads = duel.read_log(true);
            steps.push((pos, reads, where_cards(duel)));
        }
        if let Some(log) = self.suspect_log.as_mut() {
            let reads = duel.read_log(true);
            if pos >= log.from {
                log.steps.push((pos, reads, where_cards(duel)));
            }
        }
    }

    fn on_process_active(&self) -> bool {
        self.logged_steps.is_some() || self.suspect_log.is_some()
    }

    /// (lenient) At the viewer's Idle or Battle command, with no chain: corrects the duel's public
    /// state to what the viewer saw there (see sync.rs), then lets the core compute the prompt again.
    pub fn lenient_sync(&mut self, duel: &mut RDuel, prompt: &Msg) -> bool {
        if !self.state_sync || !matches!(prompt, Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. }) || self.pos >= self.items.len() || self.synced_at == Some(self.pos) {
            return false;
        }
        let ObsItem::Prompt(p, Some(state)) = self.items[self.pos].clone() else { return false };
        if !state.chain.is_empty() || p.name() != prompt.name() {
            return false;
        }
        self.synced_at = Some(self.pos);
        let got = viewer_view(&mut duel.d, self.viewer);
        if got == state || !got.chain.is_empty() || (got.turn_player, got.phase) != (state.turn_player, state.phase) {
            return false;
        }
        let n = crate::belief::sync::sync_to_view(duel, self.viewer, &state, &got);
        self.roll_dirty = true;
        if n == 0 || !duel.restart_command() {
            return false;
        }
        self.lenient_syncs += n;
        true
    }

    /// (lenient) Moves to the stream's next copy of the replay's message, if it comes before the
    /// viewer's next prompt.
    pub fn realign(&mut self, o: &Msg) -> bool {
        for j in self.pos..self.items.len().min(self.pos + 40) {
            let ObsItem::Msg(m) = &self.items[j] else { return false };
            if m == o {
                self.lenient_skips += j - self.pos;
                self.pos = j;
                return true;
            }
        }
        false
    }

    /// (lenient) `resp`, unless the core rejected it for this prompt before: then another legal
    /// answer (seeded: the same every run).
    pub fn lenient_pick(&self, prompt: &Msg, mut resp: Response) -> Response {
        let sig = prompt_sig(prompt);
        let mut k = 0;
        while self.lenient_rejected.contains(&(sig.clone(), resp_sig(&resp))) && k < 50 {
            let mut rng = ygo_arbiter::pyrandom::PyRandom::new((self.pos * 131 + k) as i128);
            if let Ok(r) = ygo_arbiter::random::random_response(prompt, &mut rng, 0.05) {
                resp = r;
            }
            k += 1;
        }
        resp
    }

    /// (lenient) Some legal answer: the first of the usual candidates, else a random legal one.
    pub fn first_valid(&self, prompt: &Msg) -> R<Response> {
        for c in super::decide::all_responses(prompt) {
            if encode(prompt, &c).is_ok() {
                return Ok(c);
            }
        }
        for k in 0..20 {
            let mut rng = ygo_arbiter::pyrandom::PyRandom::new((self.pos * 31 + k) as i128);
            if let Ok(c) = ygo_arbiter::random::random_response(prompt, &mut rng, 0.05)
                && encode(prompt, &c).is_ok()
            {
                return Ok(c);
            }
        }
        div(Divergence::new(self.pos, "no valid response"))
    }

    pub fn run_loop(&mut self, duel: &mut RDuel, prompt_in: Option<Msg>, response: Option<Response>, pending_resp_in: Option<Response>) -> R<()> {
        let mut prompt: Option<Msg> = prompt_in;
        let mut pending: Option<Msg> = None; // the prompt whose response the next step processes
        let mut pending_resp: Option<Response> = pending_resp_in;
        if let Some(r) = &response {
            let p = prompt.clone().expect("a prompt with a response");
            duel.set_response(&encode(&p, r).map_err(other)?);
            pending = Some(p);
            if pending_resp.is_none() {
                pending_resp = Some(r.clone());
            }
        }
        loop {
            if self.pos as i64 != self.cp_pos {
                self.cp_pos = self.pos as i64;
                self.cp_n = 0;
            } else {
                self.cp_n += 1;
            }
            let cp: Cp = (self.cp_pos as usize, self.cp_n);
            let mut todo: Vec<(u32, u32)> = self.carry_swaps.clone();
            if let Some(l) = self.swaps.get(&cp) {
                todo.extend(l.iter().copied().filter(|&(a, b)| !self.folded_swaps.contains(&(cp, a, b))));
            }
            if !todo.is_empty() {
                self.roll_dirty = true;
                // the pending response selected card objects: never move those
                let busy: Vec<u32> = match &pending {
                    Some(p) => infer::response_card_ids(&self.loc_map(duel), p, pending_resp.as_ref()),
                    None => vec![],
                };
                // (only copies trade places)
                let codes = card_codes(duel);
                let todo: Vec<(u32, u32)> = todo.into_iter().filter(|(a, b)| codes.get(a) == codes.get(b)).collect();
                let mut carry = Vec::new();
                for (a, b) in todo {
                    if busy.contains(&a) || busy.contains(&b) || !duel.swap_cards(a, b) {
                        carry.push((a, b)); // not possible yet: try again next step
                    }
                }
                self.carry_swaps = carry;
            }
            self.apply_forces(duel);
            if !self.snapshots.contains_key(&cp) && (self.depth == 0 || self.snapshot_cps.contains(&cp)) {
                // (the main line keeps the latest step's positions)
                if self.depth == 0 {
                    let prev = self.roll_cp.and_then(|r| self.snapshots.get(&r).cloned());
                    if let Some(r) = self.roll_cp
                        && !self.snapshot_cps.contains(&r)
                    {
                        self.snapshots.remove(&r);
                    }
                    self.roll_cp = Some(cp);
                    match prev {
                        Some(p) if !self.roll_dirty => {
                            self.snapshots.insert(cp, p); // (no card moved since)
                        }
                        _ => {
                            let lm = Rc::new(self.loc_map(duel));
                            self.snapshots.insert(cp, lm);
                        }
                    }
                    self.roll_dirty = false;
                } else {
                    let lm = Rc::new(self.loc_map(duel));
                    self.snapshots.insert(cp, lm);
                }
            }
            // active for every step at this stream position
            let mut forced_match: Vec<u32> = self.matches.get(&cp.0).cloned().unwrap_or_default();
            if self.match_piles.iter().any(|&(a, b)| a <= cp.0 && cp.0 <= b && cp >= self.match_pile_from.get(&(a, b)).copied().unwrap_or(cp)) {
                let h = self.hidden_ids(duel);
                forced_match.extend(h);
            }
            if !forced_match.is_empty() {
                duel.set_forced_match(&forced_match);
            }
            let mut nomatch: Vec<u32> = Vec::new();
            let spans: Vec<(usize, usize)> = self.nomatch_piles.iter().copied().filter(|&(a, b)| a <= cp.0 && cp.0 <= b).collect();
            if !spans.is_empty() {
                // (except the hidden cards the stream shows being used before the span ends)
                let end = spans.iter().map(|s| s.1).max().unwrap();
                let msgs: Vec<Msg> =
                    self.items[self.pos.min(self.items.len())..(end + 1).min(self.items.len())].iter().filter_map(|it| if let ObsItem::Msg(m) = it { Some(m.clone()) } else { None }).collect();
                let used = self.loc_map(duel).references(&msgs);
                let h = self.hidden_ids(duel);
                nomatch = h.into_iter().filter(|c| !used.contains(c)).collect();
            }
            nomatch.extend(self.nomatch_id_spans.iter().filter(|&&(a, b, _)| a <= cp.0 && cp.0 <= b).map(|&(_, _, c)| c));
            for &(a, b, loc) in &self.nomatch_loc_piles {
                if a <= cp.0 && cp.0 <= b {
                    nomatch.extend(duel.location_cards(self.opponent, loc).into_iter().map(|x| x.1).filter(|&c| c != 0));
                }
            }
            if !self.nomatch_cards.is_empty() {
                let lm = self.loc_map(duel);
                for &(a, b, slot) in &self.nomatch_cards {
                    if a <= cp.0 && cp.0 <= b
                        && let Some(cid) = lm.at(&infer::Li { con: slot.0, loc: slot.1, seq: slot.2, pos: 0 }).filter(|&c| c != 0)
                    {
                        nomatch.push(cid);
                    }
                }
            }
            if !nomatch.is_empty() {
                duel.set_forced_nomatch(&nomatch);
            }
            let (answered, answered_resp) = (pending.clone(), pending_resp.clone()); // (the response this step processes)
            if self.on_process_active() && !duel.reads_on {
                duel.read_log(true);
                duel.reads_on = true;
            }
            if self.deadline.is_some_and(|d| std::time::Instant::now() > d) {
                return Err(RErr::Other("timeout".into()));
            }
            if self.depth > 0 {
                // (a probe's work is bounded by engine steps)
                self.probe_steps += 1;
                if self.probe_steps > PROBE_STEPS {
                    return Err(RErr::ProbeTimeout);
                }
                if self.probe_deadline.is_some_and(|d| std::time::Instant::now() > d) {
                    return Err(RErr::ProbeTimeout);
                }
            }
            if self.jit_from.is_some() {
                self.jit_prepare(duel);
            }
            let status = duel.process();
            if self.on_process_active() {
                self.on_process(duel);
            }
            if !nomatch.is_empty() {
                duel.set_forced_nomatch(&[]);
            }
            if !forced_match.is_empty() {
                duel.set_forced_match(&[]);
            }
            let mut msgs = duel.get_messages().map_err(|e| RErr::Other(format!("ParseError: {e}")))?;
            if !self.active_forces.is_empty() {
                // (mandatory triggers only the forcing made activatable)
                let used: std::collections::HashSet<(u32, u64)> = duel.take_forced_used().into_iter().collect();
                for (k, m) in msgs.iter().enumerate() {
                    if let Msg::Chaining { code, desc, .. } = m
                        && (used.contains(&(*code, *desc)) || self.forced_used_prev.contains(&(*code, *desc)))
                    {
                        let shown = msgs[..k].iter().filter(|x| filter_message(x, self.viewer).is_some()).count();
                        self.force_enabled.insert(self.pos + shown);
                    }
                }
                self.forced_used_prev = used;
            }
            if !self.roll_dirty && msgs.iter().any(|m| !no_moves(m.name())) {
                self.roll_dirty = true;
            }
            // (the hint text the replay's next prompt comes with)
            for m in &msgs {
                if let Msg::Hint { hint_type: HINT_SELECTMSG, data, .. } = m {
                    self.pending_select_msg = Some(*data);
                }
            }
            if status == STATUS_AWAITING {
                self.prompt_select_msg = self.pending_select_msg.take();
            }
            self.note_field_arrivals(cp, &msgs, duel);
            if status == STATUS_AWAITING {
                self.prompt_cp = Some(cp);
                if let Some(last) = msgs.last()
                    && !matches!(last, Msg::Retry {})
                {
                    // a copy swap must happen before the prompt that chooses the card
                    for cid in infer::prompt_card_ids(&self.loc_map(duel), last) {
                        self.last_listed.insert(cid, cp);
                    }
                }
            }
            // reported after the step's messages: an earlier mismatch in them is the cause
            let forced_error = duel.forced_errors();
            if status == STATUS_AWAITING && msgs.last().is_some_and(|m| !matches!(m, Msg::Retry {})) {
                prompt = msgs.pop();
            }
            let check_forced = |s: &Self, msgs: &Vec<Msg>| -> R<()> {
                if forced_error != 0 && !s.lenient {
                    return div(Divergence {
                        got: Some(DV::Msgs(msgs.clone())),
                        ..Divergence::new(s.pos, "a forced shuffle/random outcome could not be applied (the replay's cards differ from the hypothesis)")
                    });
                }
                Ok(())
            };
            for m in &msgs {
                if let Msg::Hint { hint_type, player, data } = m
                    && *player == self.opponent
                {
                    if *hint_type == HINT_EVENT {
                        self.event_hint = Some((self.pos, *data));
                    }
                    if *hint_type == HINT_SELECTMSG {
                        self.opp_select_hint = Some(*data);
                    }
                }
            }
            pending = if status == STATUS_AWAITING { prompt.clone() } else { None };
            pending_resp = None;
            for idx in 0..msgs.len() {
                let m = &msgs[idx];
                match m {
                    Msg::ShuffleSetCard { old, location, .. } if !old.is_empty() => {
                        // (face-down cards shuffled in their zones: copies trade places there)
                        self.last_shuffle_cp.insert((old[0].con, *location), cp);
                    }
                    _ => {}
                }
                if let Some(loc) = shuffle_loc(m.name()) {
                    let player = m.player().unwrap_or(0);
                    self.last_shuffle_cp.insert((player, loc), cp);
                    if matches!(m, Msg::ShuffleDeck { .. } | Msg::ShuffleHand { .. }) {
                        for (_, cid) in duel.location_cards(player, loc) {
                            self.card_shuffled.entry(cid).or_default().push(cp);
                        }
                    }
                    self.shuffle_cps.entry(cp).or_default().push(self.shuffle_count);
                    self.shuffle_count += 1;
                }
                if matches!(m, Msg::Retry {})
                    && answered.as_ref().is_some_and(|a| a.player() == Some(self.viewer))
                    && self.pos < self.items.len()
                    && matches!(self.items[self.pos], ObsItem::Prompt(..))
                {
                    // (the viewer's answer was invalid in the real game too)
                    continue;
                }
                if matches!(m, Msg::Retry {})
                    && self.lenient
                    && self.depth == 0
                    && let Some(a) = answered.as_ref()
                {
                    // (lenient: the core asks again; another answer next time)
                    let r = answered_resp.as_ref().map(resp_sig).unwrap_or_default();
                    self.lenient_rejected.insert((prompt_sig(a), r));
                    self.lenient_skips += 1;
                    if self.lenient_rejected.len() > 200 {
                        return div(Divergence::new(self.pos, "core rejected too many responses"));
                    }
                    continue;
                }
                if matches!(m, Msg::Retry {}) {
                    if let (Some(Msg::AnnounceCard { player, .. }), Some(r)) = (&answered, &answered_resp)
                        && *player == self.opponent
                    {
                        // the announcement must name a card in the deck: the deck holds a copy somewhere
                        let n = duel.location_cards(self.opponent, LOCATION_DECK).len();
                        let code = r.code.unwrap_or(0) as u32;
                        for seq in 0..n {
                            self.assume((self.opponent, LOCATION_DECK, seq as u32), &[code], "the opponent announced a card of its deck")?;
                        }
                    }
                    return div(Divergence::new(self.pos, "core rejected a response"));
                }
                if let Msg::CardHint { hint_type, loc, value, con, seq, pos } = m
                    && *hint_type == CHINT_DESC_ADD
                    && *loc != 0
                    && let Some(cid) = self.loc_map(duel).at(&infer::Li { con: *con, loc: *loc, seq: *seq, pos: *pos }).filter(|&c| c != 0)
                {
                    self.sim_marks.entry(*value).or_default().push(cid);
                }
                let Some(o) = filter_message(m, self.viewer) else { continue };
                if self.pos >= self.items.len() {
                    if self.depth > 0 {
                        check_forced(self, &msgs)?;
                        return Ok(());
                    }
                    if self.stopped {
                        return Ok(()); // (the arbiter stopped the game somewhere after what was observed)
                    }
                    if self.lenient {
                        self.lenient_skips += 1;
                        continue;
                    }
                    return div(Divergence { got: Some(DV::Msg(o)), ..Divergence::new(self.pos, "replay produced more messages than observed") });
                }
                if let Msg::ShuffleHand { player, .. } = &o
                    && Some(*player) == self.skip_shuffle_hand
                    && self.depth > 0
                {
                    self.skip_shuffle_hand = None;
                    if !matches!(&self.items[self.pos], ObsItem::Msg(x) if *x == o) {
                        continue; // (a code test's own card arrival)
                    }
                }
                let matches_item = matches!(&self.items[self.pos], ObsItem::Msg(x) if *x == o);
                if !matches_item && !self.match_unordered_hint(&o) {
                    if self.lenient && self.depth == 0 {
                        if !self.realign(&o) {
                            self.lenient_skips += 1;
                            continue; // (a message the stream does not show: passed over)
                        }
                    } else {
                        let item = self.items[self.pos].clone();
                        return Err(self.mismatch(cp, &msgs, idx, &item, &o, duel).into());
                    }
                }
                self.advance()?;
                if let Msg::Win { .. } = o {
                    check_forced(self, &msgs)?;
                    if self.pos < self.items.len() && self.depth == 0 {
                        return div(Divergence::new(self.pos, "stream continues after the game ended"));
                    }
                    return Ok(());
                }
                if self.depth > 0 && self.done() {
                    check_forced(self, &msgs)?;
                    return Ok(());
                }
            }
            check_forced(self, &msgs)?;
            if status == STATUS_END {
                if self.pos < self.items.len() {
                    let it = self.items[self.pos].clone();
                    return div(Divergence { expected: Some(DV::Item(it)), ..Divergence::new(self.pos, "replay ended early") });
                }
                return Ok(());
            }
            if status == STATUS_CONTINUE {
                continue;
            }
            let Some(p) = prompt.clone() else { return div(Divergence::new(self.pos, "core awaits a response without a prompt")) };
            if self.stopped && self.depth == 0 && self.pos >= self.items.len() {
                return Ok(()); // (the game was stopped at this prompt, whoever answers it)
            }
            let player = p.player().unwrap_or(0);
            if player == self.viewer && is_hidden_prompt(&p) {
                // answered by the arbiter, not part of the viewer's stream
                let r = auto_response();
                duel.set_response(&encode(&p, &r).map_err(other)?);
                pending = Some(p);
                pending_resp = Some(r);
                continue;
            }
            if player == self.viewer {
                if self.lenient && self.depth == 0 {
                    // (the stream's next prompt of the viewer is this one)
                    let mut j = self.pos;
                    while j < self.items.len() && !matches!(self.items[j], ObsItem::Prompt(..)) {
                        j += 1;
                    }
                    self.lenient_skips += j - self.pos;
                    self.pos = j;
                    if self.lenient_sync(duel, &p) {
                        pending = None;
                        pending_resp = None;
                        continue; // (the command prompt again, computed for the corrected state)
                    }
                }
                if self.pos >= self.items.len() {
                    if self.depth > 0 {
                        return Ok(());
                    }
                    if self.lenient {
                        self.final_prompt = Some(p);
                        return Ok(());
                    }
                    return div(Divergence::new(self.pos, "stream ended before the viewer's prompt"));
                }
                let item = self.items[self.pos].clone();
                let fp = filter_prompt(&p, self.viewer);
                let same = matches!(&item, ObsItem::Prompt(x, _) if *x == fp);
                if !same && !(self.lenient && self.depth == 0) {
                    let fix = self.viewer_prompt_fix(&item, &fp, duel, cp);
                    let what = match &item {
                        ObsItem::Prompt(x, _) | ObsItem::Msg(x) => x.name().to_string(),
                        ObsItem::Response(_) => "response".to_string(),
                    };
                    return div(Divergence {
                        expected: Some(DV::Item(item.clone())),
                        got: Some(DV::Msg(fp.clone())),
                        fix,
                        ..Divergence::new(self.pos, format!("prompt mismatch [{what} vs {}]", fp.name()))
                    });
                }
                if let ObsItem::Prompt(_, Some(state)) = &item
                    && (self.check_state || Some(self.pos) == self.probe_state_at)
                    && !self.lenient
                {
                    let got = viewer_view(&mut duel.d, self.viewer);
                    if got != *state {
                        let (ev, gv) = (jv(state), jv(&got));
                        let mut fix = self.state_copy_fix(&ev, &gv, duel, cp);
                        if fix.is_none() {
                            fix = self.public_arrival_fix(&ev, &gv);
                        }
                        if fix.is_none() {
                            fix = self.public_hidden_fix(&ev, &gv);
                        }
                        if fix.is_none() {
                            fix = self.stat_fix(&ev, &gv, duel);
                        }
                        if fix.is_none() && state.phase != got.phase {
                            // (the same messages in another phase or step: an activation came in
                            // another chain window)
                            let mut all: Vec<(Dkey, usize)> = self.window_activations.iter().chain(&self.window_choices).copied().collect();
                            all.sort_by(|a, b| b.cmp(a));
                            for (dkey, ans) in all {
                                if self.overrides.get(&dkey) != Some(&ans) {
                                    fix = Some(Fix::Decide(dkey, ans));
                                    break;
                                }
                            }
                        }
                        let d = super::util::first_difference(&ev, &gv, "").unwrap_or_else(|| "None".into());
                        return div(Divergence {
                            expected: Some(DV::State(state.clone())),
                            got: Some(DV::State(got)),
                            fix,
                            ..Divergence::new(self.pos, format!("state mismatch: {d}"))
                        });
                    }
                    if self.depth == 0 {
                        self.since_state.clear();
                    }
                }
                self.advance()?;
                if self.pos >= self.items.len() {
                    self.final_prompt = Some(p); // (the core's version: see to_core_response)
                    return Ok(()); // positioned at the viewer's decision
                }
                let ObsItem::Response(resp) = self.items[self.pos].clone() else {
                    return div(Divergence::new(self.pos, "expected the viewer's response"));
                };
                self.advance()?;
                let attempt: R<()> = (|| {
                    if self.lenient && self.lenient_rejected.contains(&(prompt_sig(&p), resp_sig(&resp))) {
                        return Err(RErr::Other("InvalidResponse: rejected before".into()));
                    }
                    let core = to_core_response(&p, self.viewer, &resp).map_err(|e| RErr::Other(format!("InvalidResponse: {e}")))?;
                    duel.set_response(&encode(&p, &core).map_err(|e| RErr::Other(format!("InvalidResponse: {e}")))?);
                    Ok(())
                })();
                match attempt {
                    Ok(()) => pending_resp = Some(resp),
                    Err(e) => {
                        if !self.lenient {
                            return Err(e);
                        }
                        // (the recorded answer does not fit the replay's prompt)
                        let r = self.first_valid(&p)?;
                        let r = self.lenient_pick(&p, r);
                        duel.set_response(&encode(&p, &r).map_err(other)?);
                        pending_resp = Some(r);
                        self.lenient_skips += 1;
                    }
                }
                if self.depth > 0 && self.done() {
                    return Ok(());
                }
            } else {
                self.opp_prompt = Some(p.clone());
                let resp = match self.decide(duel, &p) {
                    Ok(r) => r,
                    Err(RErr::Div(d)) => {
                        if !(self.lenient && self.depth == 0) {
                            return Err(RErr::Div(d));
                        }
                        self.lenient_skips += 1;
                        self.first_valid(&p)?
                    }
                    Err(e) => return Err(e),
                };
                let resp = if self.lenient && self.depth == 0 { self.lenient_pick(&p, resp) } else { resp };
                duel.set_response(&encode(&p, &resp).map_err(other)?);
                pending_resp = Some(resp);
            }
        }
    }

    /// The repair for a viewer's prompt that differs from the stream's item (the diagnosis in
    /// _loop's viewer branch).
    fn viewer_prompt_fix(&mut self, item: &ObsItem, fp: &Msg, duel: &mut RDuel, cp: Cp) -> Option<Fix> {
        let fpv = jv(fp);
        let (is_msg, iv): (bool, Value) = match item {
            ObsItem::Msg(m) => (true, jv(m)),
            ObsItem::Prompt(m, _) => (false, jv(m)),
            ObsItem::Response(r) => (false, jv(r)),
        };
        let is_prompt = matches!(item, ObsItem::Prompt(..));
        let mut fix = if is_msg { self.force_fix(&iv) } else { None };
        let mut ch: Option<Value> = if is_msg && iv.t() == "CHAINING" { Some(iv.clone()) } else { None };
        if fix.is_none() && is_msg && is_activation_prelude_t(iv.t()) {
            // (the activation's prelude: the activation itself was not offered)
            let up = self.upcoming();
            ch = next_activation(&up).map(|m| jv(&m));
            fix = ch.as_ref().and_then(|c| self.force_fix(c));
        }
        if fix.is_none()
            && let Some(c) = &ch
            && c.i("trigger_con") as u32 == self.opponent
        {
            // the opponent's trigger did not come in the replay: its event can depend on hidden cards
            let start = self.prev_chain_start().unwrap_or_else(|| self.action_start());
            if !self.match_piles.contains(&(start, self.pos)) && !self.nomatch_piles.contains(&(start, self.pos)) {
                fix = Some(Fix::MatchPile(start, self.pos));
            }
        }
        if fix.is_none() && is_msg && fpv.t() == "SELECT_EFFECTYN" {
            // an optional effect of the viewer's card the real game did not offer: another copy
            fix = self.viewer_copy_fix(&fpv, duel, cp);
        }
        if fix.is_none() && is_msg && iv.t() == "NEW_PHASE" && matches!(fpv.t(), "SELECT_BATTLECMD" | "SELECT_IDLECMD") {
            for &(dkey, passing) in self.window_activations.iter().rev() {
                if self.overrides.get(&dkey) != Some(&passing) {
                    fix = Some(Fix::Decide(dkey, passing));
                    break;
                }
            }
        }
        if fix.is_none() && is_msg {
            fix = self.granted_set_fix(&iv, duel);
        }
        if fix.is_none() && is_prompt && !iv.g("chains").is_null() && !fpv.g("chains").is_null() {
            // (an effect the viewer's card has in one game only: the other answers of that choice)
            let live = self.loc_map(duel);
            let mut exp_n: std::collections::BTreeMap<(u32, u32, u32), i64> = Default::default();
            for c in iv.g("chains").l() {
                *exp_n.entry(super::util::key3(c)).or_default() += 1;
            }
            let mut got_n: std::collections::BTreeMap<(u32, u32, u32), i64> = Default::default();
            for c in fpv.g("chains").l() {
                *got_n.entry(super::util::key3(c)).or_default() += 1;
            }
            // (Counter differences in insertion order: the cards the replay lacks first)
            let order = |a: &Value| -> Vec<(u32, u32, u32)> {
                let mut v = Vec::new();
                for c in a.l() {
                    let k = super::util::key3(c);
                    if !v.contains(&k) {
                        v.push(k);
                    }
                }
                v
            };
            let mut keys: Vec<(u32, u32, u32)> = order(iv.g("chains")).into_iter().filter(|k| exp_n[k] > got_n.get(k).copied().unwrap_or(0)).collect();
            keys.extend(order(fpv.g("chains")).into_iter().filter(|k| got_n[k] > exp_n.get(k).copied().unwrap_or(0)));
            for k in keys {
                fix = self.silent_choice_fix(live.at(&infer::Li { con: k.0, loc: k.1, seq: k.2, pos: 0 }));
                if fix.is_some() {
                    break;
                }
            }
        }
        if fix.is_none() && is_prompt {
            fix = self.prompt_copy_fix(&iv, &fpv, duel, cp);
            if fix.is_none() {
                fix = self.hidden_candidates_fix(&iv, &fpv, Some(duel));
            }
            if fix.is_none() {
                fix = self.granted_copy_fix(&iv, &fpv, duel);
            }
            if fix.is_none() {
                fix = self.enabler_fix(&iv, &fpv, duel);
            }
        }
        if fix.is_none() && is_prompt && iv.t() == fpv.t() && public_missing(&iv, &fpv) {
            // the real prompt offers a public card the replay's does not: the opponent's hidden
            // cards satisfy a condition that prevents it
            let span = (self.link_start().min(self.pos.saturating_sub(2)), self.pos);
            if !self.nomatch_piles.contains(&span) && !self.match_piles.contains(&span) {
                fix = Some(Fix::NomatchPile(span.0, span.1));
            }
        }
        fix
    }
}

pub fn is_activation_prelude_t(t: &str) -> bool {
    matches!(t, "MOVE" | "POS_CHANGE" | "HINT" | "CARD_HINT" | "PAY_LPCOST")
}

/// The next activation among the upcoming messages, if only its prelude comes before it.
pub fn next_activation(up: &[Msg]) -> Option<Msg> {
    for (i, m) in up.iter().enumerate() {
        if let Msg::Chaining { .. } = m {
            return if infer::prelude_fits(&up[..i], m) { Some(m.clone()) } else { None };
        }
        if !is_activation_prelude_t(m.name()) {
            return None;
        }
    }
    None
}

/// Whether the real prompt offers a public card the replay's does not.
pub fn public_missing(exp: &Value, got: &Value) -> bool {
    for key in ["cards", "select_cards", "chains", "activatable", "summonable", "attackable"] {
        let gl: BTreeSet<(u32, u32, u32)> = got.g(key).l().iter().map(super::util::key3).collect();
        if exp.g(key).l().iter().any(|c| c.g("code").truthy() && !gl.contains(&super::util::key3(c))) {
            return true;
        }
    }
    false
}
