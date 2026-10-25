//! The opponent's decisions: the checks before one (assumptions the upcoming messages call for),
//! the candidates inference names, and probes of them (BeliefReplay._decide and what it uses).
//! A probe runs on copies of the replay and the duel where the Python module forks.

use std::collections::{BTreeSet, HashMap};

use ygo_core::constants::*;

use super::util::jv;
use super::*;
use crate::actions::{Command, Response, battle_options, encode, idle_options};
use crate::belief::completion::{card, fits_need, is_extra_deck, monster_sset_codes};
use crate::belief::infer::{self, Certain, Li};
use crate::belief::script_index as si;
use crate::belief::tracker::{Kind, Need, leaves_field};
use crate::messages::{Card, Msg};
use crate::state::capture;

const SUMMON_TYPE_NORMAL: u32 = 0x10000000;
const SUMMON_TYPE_TRIBUTE: u32 = 0x11000000;

/// What a probe does to its copy of the duel before it starts.
#[derive(Clone, Copy, Debug)]
pub enum ProbeSetup {
    None,
    /// a new card of `code` takes the slot (con, loc, seq)
    NewCard(u32, u32, u32, u32),
    Swap(u32, u32),
    Recreate(u32, u32),
}

/// The legal answers to an opponent prompt that inference may have left out (replay._all_responses).
pub fn all_responses(prompt: &Msg) -> Vec<Response> {
    let idx = |j: usize| Response { index: Some(j as i64), ..Response::default() };
    let yes = |y: bool| Response { yes: Some(y), ..Response::default() };
    match prompt {
        Msg::SelectIdlecmd { .. } => idle_options(prompt),
        Msg::SelectBattlecmd { .. } => battle_options(prompt),
        Msg::SelectChain { forced, chains, .. } => {
            let mut out = if *forced != 0 { vec![] } else { vec![Response { index: Some(-1), ..Response::default() }] };
            out.extend((0..chains.len()).map(idx));
            out
        }
        Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => vec![yes(true), yes(false)],
        Msg::SelectOption { options, .. } => (0..options.len()).map(idx).collect(),
        Msg::SelectPosition { positions, .. } => [POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE]
            .into_iter()
            .filter(|p| positions & p != 0)
            .map(|p| Response { position: Some(p as i64), ..Response::default() })
            .collect(),
        Msg::SelectTribute { min, max, cards, .. } if *min <= 1 && 1 <= *max => {
            (0..cards.len()).map(|j| Response { indices: Some(vec![j as i64]), ..Response::default() }).collect()
        }
        Msg::SelectCard { min, max, cards, .. } => {
            let n = cards.len();
            let sel = |v: Vec<usize>| Response { indices: Some(v.into_iter().map(|j| j as i64).collect()), ..Response::default() };
            let mut out: Vec<Response> = Vec::new();
            let singles: Vec<Response> = if *min <= 1 { (0..n).map(|j| sel(vec![j])).collect() } else { vec![] };
            let all: Vec<usize> = (0..n).collect();
            for k in (*min as usize).max(2)..=(*max as usize).min(n) {
                let room = MAX_SELECTIONS - out.len();
                out.extend(infer::combinations(&all, k).take(room).map(sel));
                if out.len() >= MAX_SELECTIONS {
                    break;
                }
            }
            let mut r = singles;
            r.extend(out);
            r
        }
        Msg::SelectUnselectCard { select_cards, finishable, cancelable, .. } => {
            let mut out: Vec<Response> = (0..select_cards.len()).map(idx).collect();
            if *finishable != 0 || *cancelable != 0 {
                out.push(Response { finish: Some(true), ..Response::default() });
            }
            out
        }
        _ => vec![],
    }
}

/// The cards a card prompt offers (prompt["cards"], else select_cards [+ unselect_cards]).
fn prompt_cards(p: &Msg, with_unselect: bool) -> Vec<Card> {
    match p {
        Msg::SelectCard { cards, .. } | Msg::SelectTribute { cards, .. } | Msg::SelectSum { cards, .. } => cards.clone(),
        Msg::SelectUnselectCard { select_cards, unselect_cards, .. } => {
            let mut v = select_cards.clone();
            if with_unselect {
                v.extend(unselect_cards.iter().cloned());
            }
            v
        }
        _ => vec![],
    }
}

/// prompt.get("cards") or prompt.get("select_cards") or []
fn offered_cards(p: &Msg) -> Vec<Card> {
    match p {
        Msg::SelectCard { cards, .. }
        | Msg::SelectTribute { cards, .. }
        | Msg::SelectSum { cards, .. }
        | Msg::SortCard { cards, .. }
        | Msg::SortChain { cards, .. }
        | Msg::SelectCounter { cards, .. } => cards.clone(),
        Msg::SelectUnselectCard { select_cards, .. } => select_cards.clone(),
        _ => vec![],
    }
}

/// The memo's and the cycle check's text of a prompt (repr(prompt)).
pub fn prompt_key(p: &Msg) -> String {
    serde_json::to_string(p).unwrap_or_default()
}

fn most_common<K: PartialEq + Clone>(items: impl IntoIterator<Item = K>, n: usize) -> Vec<K> {
    let mut counts: Vec<(K, usize)> = Vec::new();
    for k in items {
        match counts.iter_mut().find(|(x, _)| *x == k) {
            Some(e) => e.1 += 1,
            None => counts.push((k, 1)),
        }
    }
    // (stable: ties in order of first appearance, as Counter.most_common)
    counts.sort_by_key(|c| std::cmp::Reverse(c.1));
    counts.into_iter().take(n).map(|x| x.0).collect()
}

fn li(con: u32, loc: u32, seq: u32) -> Li {
    Li { con, loc, seq, pos: 0 }
}

impl BeliefReplay {
    pub fn upcoming(&self) -> Vec<Msg> {
        let mut out = Vec::new();
        for item in &self.items[self.pos.min(self.items.len())..] {
            let ObsItem::Msg(m) = item else { break };
            out.push(m.clone());
        }
        out
    }

    /// Upcoming messages, skipping over the viewer's prompts and responses.
    pub fn upcoming_ext(&self, limit: usize) -> Vec<Msg> {
        let end = (self.pos + limit).min(self.items.len());
        self.items[self.pos.min(end)..end].iter().filter_map(|it| if let ObsItem::Msg(m) = it { Some(m.clone()) } else { None }).collect()
    }

    /// The messages that show what the opponent selected (and whether they reach past the viewer's
    /// own decisions).
    fn selection_window(&self, up: Vec<Msg>) -> (Vec<Msg>, bool) {
        if up.iter().any(|m| !matches!(m, Msg::Hint { .. } | Msg::CardHint { .. })) {
            return (up, false);
        }
        let i = self.pos + up.len();
        if i >= self.items.len() || !matches!(self.items[i], ObsItem::Prompt(..)) {
            return (up, false);
        }
        (self.upcoming_ext(40), true)
    }

    /// Raises a force fix if the opponent's next activation is not available.
    fn check_force(&mut self, prompt: &Msg, up: &[Msg], duel: &RDuel) -> R<()> {
        let t = prompt.name();
        if !matches!(t, "SELECT_IDLECMD" | "SELECT_BATTLECMD" | "SELECT_CHAIN" | "SELECT_EFFECTYN") {
            return Ok(());
        }
        // (the next activation, by its index in `up`)
        let mut ch_i = None;
        for (i, m) in up.iter().enumerate() {
            if let Msg::Chaining { .. } = m {
                if infer::prelude_fits(&up[..i], m) {
                    ch_i = Some(i);
                }
                break;
            }
            if !super::step::is_activation_prelude_t(m.name()) {
                break;
            }
        }
        let ch = ch_i.map(|i| &up[i]);
        if let Msg::SelectChain { forced, chains, .. } = prompt
            && *forced != 0
        {
            let matched = matches!(ch, Some(Msg::Chaining { code, desc, .. }) if chains.iter().any(|o| o.code == *code && o.desc == Some(*desc)));
            if !matched {
                for o in chains {
                    let key = (self.opponent, o.code, o.desc.unwrap_or(0));
                    if self.active_forces.contains(&key) {
                        let later = self.force_windows.iter().any(|w| w.2 == key && w.0 <= self.pos && self.pos + 1 < w.1);
                        return div(Divergence::with_fix(self.pos, "a forced effect became mandatory", Some(if later { Fix::ForceLater(key) } else { Fix::Unforce(key) })));
                    }
                }
            }
        }
        let (Some(ch_i), Some(Msg::Chaining { code, desc, trigger_con, trigger_loc, trigger_seq, .. })) = (ch_i, ch) else { return Ok(()) };
        let (code, desc, trigger_con, trigger_loc, trigger_seq) = (*code, *desc, *trigger_con, *trigger_loc, *trigger_seq);
        let j = self.pos + ch_i;
        if trigger_con != self.opponent && !self.opp_activates(j) {
            return Ok(());
        }
        let options: Vec<(u32, Option<u64>)> = match prompt {
            Msg::SelectEffectyn { code, desc, .. } => vec![(*code, Some(*desc))],
            Msg::SelectChain { chains, .. } => chains.iter().map(|o| (o.code, o.desc)).collect(),
            _ => Command::Activate.cards(prompt).unwrap_or(&[]).iter().map(|o| (o.code, o.desc)).collect(),
        };
        if options.iter().any(|&(c, d)| c == code && d.unwrap_or(desc) == desc) {
            return Ok(());
        }
        let mut fix = None;
        let command = matches!(t, "SELECT_IDLECMD" | "SELECT_BATTLECMD");
        let start = if command { self.prev_chain_start() } else { None };
        if let Some(s) = start
            && trigger_con == self.opponent
            && si::has_trigger(code)
            && !self.match_piles.contains(&(s, self.pos))
            && !self.nomatch_piles.contains(&(s, self.pos))
        {
            fix = Some(Fix::MatchPile(s, self.pos));
        }
        let chv = jv(&up[ch_i]);
        if fix.is_none() {
            fix = self.force_fix(&chv);
        }
        let key = (self.opponent, code, desc);
        if fix.is_none() && command && self.active_forces.contains(&key) && !options.iter().any(|o| o.0 == code) {
            let tl = li(trigger_con, trigger_loc, trigger_seq);
            let placed = up.iter().find_map(|m| match m {
                Msg::Move { from, to, .. } if infer::same_loc(&to.into(), &tl) && from.loc == LOCATION_HAND => Some((from.con, from.loc, from.seq)),
                _ => None,
            });
            let slot = placed.or(if trigger_loc == LOCATION_HAND { Some((trigger_con, LOCATION_HAND, trigger_seq)) } else { None });
            if let Some(slot) = slot {
                let live = self.loc_map(duel);
                let a = live.at(&li(slot.0, slot.1, slot.2)).filter(|&c| c != 0);
                if let Some(a) = a
                    && live.codes.get(&a) == Some(&code)
                {
                    for (c, b) in duel.location_cards(slot.0, LOCATION_HAND) {
                        if b != a && c == code {
                            fix = self.history_swap(a, b, (self.pos, 0));
                            if fix.is_some() {
                                break;
                            }
                        }
                    }
                    if fix.is_none() {
                        fix = self.not_this_copy_fix(slot);
                    }
                }
            }
            if fix.is_none() && trigger_loc == LOCATION_HAND {
                fix = self.arrival_fix(trigger_con, trigger_loc, trigger_seq, code);
            }
            if fix.is_none() {
                fix = self.unavailable_fix();
            }
        }
        if fix.is_some() {
            return div(Divergence::with_fix(self.pos, "opponent activation is not available in the replay", fix));
        }
        Ok(())
    }

    /// The opponent's forced activation is still not available: let the hidden cards qualify
    /// until it is chained (then the other way round, then neither).
    pub fn unavailable_fix(&self) -> Option<Fix> {
        let end = (self.pos..self.items.len()).find(|&j| matches!(self.items[j], ObsItem::Msg(Msg::Chained { .. })))?;
        let span = (self.pos, end);
        if !self.match_piles.contains(&span) && !self.pile_dropped.contains(&("matchpile", span)) {
            return Some(Fix::MatchPile(span.0, span.1));
        }
        if !self.nomatch_piles.contains(&span) && !self.pile_dropped.contains(&("nomatchpile", span)) {
            return Some(Fix::NomatchPile(span.0, span.1));
        }
        if self.nomatch_piles.contains(&span) && self.pile_dropped.contains(&("matchpile", span)) {
            return Some(Fix::Unflip(span.0, span.1));
        }
        None
    }

    /// The opponent took a hidden card that is no valid choice in the replay.
    fn check_assumption(&mut self, duel: &RDuel, prompt: &Msg, up: &[Msg], ids: &[Option<u32>]) -> R<()> {
        let cards = prompt_cards(prompt, false);
        let mut lm = LocMap::new(&duel.all_cards(), &[]); // (past a shuffle, the moves belong to other selections)
        let mut earlier: Vec<(crate::messages::Loc, crate::messages::Loc)> = Vec::new();
        let opp = self.opponent;
        for (mi, m) in up.iter().enumerate() {
            if selection_end(m.name()) {
                return Ok(());
            }
            if let Msg::CardSelected { cards: sel } = m {
                for (xi, l) in sel.iter().enumerate() {
                    let cid = lm.at(&l.into());
                    if cid.is_none() || ids.contains(&cid) || l.con != opp || !hidden_pile(l.loc) {
                        continue;
                    }
                    let others: BTreeSet<Option<u32>> =
                        sel.iter().enumerate().filter(|&(k, _)| k != xi).map(|(_, x)| lm.at(&x.into()).and_then(|c| lm.codes.get(&c).copied())).collect();
                    let mut order: Vec<&Card> = cards.iter().collect();
                    order.sort_by_key(|c| (others.contains(&Some(c.code)), c.loc != l.loc));
                    let mut codes: Vec<u32> = Vec::new();
                    for c in order {
                        if c.code != 0 && !codes.contains(&c.code) {
                            codes.push(c.code);
                        }
                    }
                    self.assume((l.con, l.loc, l.seq), &codes, "opponent selected a card the hypothesis does not provide")?;
                }
                return Ok(());
            }
            let Msg::Move { from: f, to, .. } = m else {
                lm.references(std::slice::from_ref(m));
                continue;
            };
            let cid = lm.at(&f.into());
            lm.move_card(&f.into(), &to.into());
            earlier.push((*f, *to));
            if (f.con, f.loc) == (to.con, to.loc) {
                continue;
            }
            if cid.is_none() || ids.contains(&cid) {
                continue;
            }
            if f.con != opp || !hidden_pile(f.loc) {
                continue;
            }
            let mut codes: Vec<u32> = Vec::new();
            for x in up.iter().skip(mi + 1).take(7) {
                if let Msg::ConfirmCards { cards: cc, .. } = x {
                    for c in cc {
                        if c.code != 0 && (c.con, c.loc) == (to.con, to.loc) && !codes.contains(&c.code) {
                            codes.push(c.code);
                        }
                    }
                }
            }
            let mut offered: Vec<u32> = Vec::new();
            for c in &cards {
                if c.con == f.con && c.loc == f.loc && c.code != 0 && !codes.contains(&c.code) && Self::fits_destination(c.code, to) {
                    offered.push(c.code);
                }
            }
            let mut common: Option<u32> = None;
            for &code in &offered {
                if let Some(d) = card(code) {
                    common = Some(common.map_or(d.type_, |x| x & d.type_));
                }
            }
            let seen: Vec<u32> = match common.filter(|&c| c != 0) {
                Some(common) => self
                    .opp_seen_codes()
                    .into_iter()
                    .filter(|c| !codes.contains(c) && !offered.contains(c) && card(*c).is_some_and(|d| d.type_ & common == common && !is_extra_deck(d)) && Self::fits_destination(*c, to))
                    .take(4)
                    .collect(),
                None => vec![],
            };
            codes.extend(seen);
            codes.extend(offered);
            let mut src = (f.con, f.loc, f.seq);
            for (ef, _) in earlier[..earlier.len() - 1].iter().rev() {
                if ef.con == src.0 && ef.loc == src.1 && ef.seq <= src.2 {
                    src.2 += 1;
                }
            }
            self.assume(src, &codes, "opponent selected a card the hypothesis does not provide")?;
            return Ok(());
        }
        Ok(())
    }

    /// Runs the pending engine tests of candidate codes for a hidden hand card.
    fn run_code_tests(&mut self, duel: &RDuel, prompt: &Msg, resp: &Response) -> R<()> {
        let tests = self.code_tests.clone();
        for ((at, slot), (start, codes)) in tests {
            if !(start <= self.pos && self.pos < at) {
                continue;
            }
            let chained = (start..at).find(|&j| matches!(self.items[j], ObsItem::Msg(Msg::Chained { .. })));
            let Some(chained) = chained else { continue };
            if self.pos <= chained || self.items[self.pos..at].iter().any(|it| matches!(it, ObsItem::Msg(Msg::Chaining { .. }))) {
                continue;
            }
            self.code_tests.retain(|e| e.0 != (at, slot));
            self.code_tests_done.insert((at, slot));
            let (con, loc, seq) = slot;
            for code in codes {
                self.probe(duel, prompt, resp, ProbeSetup::NewCard(con, loc, seq, code), None);
                if self.last_probe_reached > at {
                    let a = (at, con, loc, seq, code);
                    if !self.failed_assumptions.contains(&a) && !self.assumptions.contains(&a) {
                        return div(Divergence::with_fix(self.pos, "the engine takes this card for the hidden one", Some(Fix::Assume(a))));
                    }
                }
            }
            return Ok(());
        }
        Ok(())
    }

    /// A divergence no repair explains: record the reads before it, then test alternatives for
    /// the hidden cards read (see _run_suspect_tests). Returns whether to restart for that.
    pub fn suspect_round(&mut self, pos: usize, sig: DivSig) -> bool {
        use crate::belief::read_guided as rg;
        if self.depth > 0 || !self.suspects_on {
            return false;
        }
        let Some(log) = self.suspect_log.take() else {
            if self.suspect_done.get(&pos).copied().unwrap_or(0) >= SUSPECT_ROUNDS || self.suspect_tests.is_some() {
                self.suspect_tests = None;
                return false;
            }
            self.suspect_log = Some(SuspectLog { pos, from: pos.saturating_sub(rg::WINDOW + 1), steps: vec![], sig });
            let r = self.stats.restarts;
            self.log(format!("restart {r}: divergence at {pos} -> record the reads before it"));
            return true;
        };
        *self.suspect_done.entry(log.pos).or_default() += 1;
        if pos != log.pos || log.steps.is_empty() {
            return false;
        }
        let mut seen: Vec<u32> = Vec::new();
        for it in &self.items {
            let ObsItem::Msg(m) = it else { continue };
            let v = jv(m);
            use super::util::V;
            let code = v.i("code");
            let con = v.o("con").or_else(|| v.g("from").o("con"));
            if code != 0 && con == Some(self.opponent as i64) && !seen.contains(&(code as u32)) {
                seen.push(code as u32);
            }
        }
        let mut codes = seen;
        if let Some(h) = &self.hypothesis {
            let deck: BTreeSet<u32> = h.decks[self.opponent as usize].0.iter().copied().collect();
            for c in deck {
                if !codes.contains(&c) {
                    codes.push(c);
                }
            }
        }
        let mut cands = rg::candidates(self, &log.steps, pos, &codes);
        cands.truncate(SUSPECT_TESTS);
        if cands.is_empty() {
            return false;
        }
        let Some(first) = log.steps.iter().filter(|s| !s.1.is_empty()).map(|s| s.0).min() else { return false };
        let Some(at) = self.decisions_at.keys().copied().filter(|&q| q <= first).max() else { return false };
        let mut tests: Vec<SuspectTest> = Vec::new();
        for (sw, asm, what) in cands {
            if let Some((cp, pairs)) = sw {
                let (a, b) = pairs[0];
                tests.push((Fix::Swap(cp, a, b), ("swap", a, b), what));
            } else if let Some(&(p, con, loc, i, code)) = asm.first() {
                let w = log.steps.iter().rev().find(|s| s.0 <= p).map(|s| &s.2);
                let cid = w.and_then(|w| w.iter().find(|(_, w)| (w.0, w.1, w.2) == (con, loc, i)).map(|(c, _)| *c));
                if let Some(cid) = cid {
                    tests.push((Fix::Assume((p, con, loc, i, code)), ("code", cid, code), what));
                }
            }
        }
        if tests.is_empty() {
            return false;
        }
        let n = tests.len();
        self.suspect_tests = Some(SuspectTests { at, div: pos, until: self.items.len().min(pos + 2 * self.min_progress + 10), tests, sig: log.sig });
        let r = self.stats.restarts;
        self.log(format!("restart {r}: divergence at {pos} -> test {n} suspects at {at}"));
        true
    }

    /// Tests the alternatives for the suspects of a divergence, each in a probe of this decision.
    fn run_suspect_tests(&mut self, duel: &RDuel, prompt: &Msg, resp: &Response) -> R<()> {
        let Some(batch) = self.suspect_tests.take() else { return Ok(()) };
        let mut best: Option<((u8, usize), Fix)> = None;
        for (fix, (kind, a, b), what) in &batch.tests {
            match fix {
                Fix::Assume(x) if self.failed_assumptions.contains(x) || self.assumptions.contains(x) => continue,
                Fix::Swap(cp, x, y) if self.failed_swaps.contains(&(*cp, *x, *y)) || self.has_swap(fix) => continue,
                _ => {}
            }
            let setup = if *kind == "swap" { ProbeSetup::Swap(*a, *b) } else { ProbeSetup::Recreate(*a, *b) };
            self.stats.suspect_tests += 1;
            let outcome = self.probe(duel, prompt, resp, setup, Some(batch.until));
            let reached = self.last_probe_reached;
            self.log(format!("suspect test {what}: reached {reached} exit {} (divergence at {})", outcome.code(), batch.div));
            let key = if reached > batch.div || outcome == Probe::Ok {
                Some((1, reached))
            } else if reached >= batch.div && self.last_probe_sig.as_ref().is_some_and(|s| *s != batch.sig) {
                Some((0, reached))
            } else {
                None
            };
            if let Some(k) = key
                && best.as_ref().is_none_or(|b| k > b.0)
            {
                best = Some((k, fix.clone()));
            }
            if outcome == Probe::Ok {
                break;
            }
        }
        if let Some((k, fix)) = best {
            let reason = if k.0 == 1 { "a suspect's alternative reproduces the stream further" } else { "a suspect's alternative changes the divergence" };
            return div(Divergence::with_fix(self.pos, reason, Some(fix)));
        }
        Ok(())
    }

    /// Stream position of the CHAINING of the link being resolved (None outside a resolution).
    pub fn resolving_link(&self) -> Option<usize> {
        let mut n = None;
        let lo = self.pos as i64 - 200;
        let mut j = self.pos as i64 - 1;
        while j > lo.max(-1) {
            let ju = j as usize;
            j -= 1;
            let ObsItem::Msg(m) = &self.items[ju] else { continue };
            match m {
                Msg::ChainEnd {} | Msg::NewPhase { .. } | Msg::NewTurn { .. } => return None,
                Msg::ChainSolved { .. } if n.is_none() => return None,
                Msg::ChainSolving { chain_count } if n.is_none() => n = Some(*chain_count),
                Msg::Chaining { chain_count, .. } if n.is_some() && Some(*chain_count) == n => return Some(ju),
                _ => {}
            }
        }
        None
    }

    /// Main deck monsters of a race that fit (the lowest codes), when no filler has it.
    fn race_stand_ins(&mut self, race: u64, fits: &dyn Fn(u32) -> bool) -> Vec<u32> {
        let list = self.race_cache.entry(race).or_insert_with(|| {
            let db = crate::game::card_db();
            let mut v: Vec<u32> = db
                .cards
                .values()
                .filter(|c| c.race == race && c.type_ & TYPE_MONSTER != 0 && !is_extra_deck(c) && c.alias == 0 && c.type_ & (TYPE_RITUAL | TYPE_TOKEN) == 0)
                .map(|c| c.code)
                .collect();
            v.sort();
            v
        });
        list.iter().copied().filter(|&c| fits(c)).take(2).collect()
    }

    /// Codes of the opponent's cards the stream showed, in order of appearance.
    pub fn opp_seen_codes(&mut self) -> Vec<u32> {
        if self.seen_cache.is_none() {
            let mut out: Vec<u32> = Vec::new();
            let mut set: std::collections::HashSet<u32> = Default::default();
            for it in &self.items {
                let ObsItem::Msg(m) = it else { continue };
                let (code, con) = match m {
                    Msg::Move { code, from, .. } => (*code, Some(from.con)),
                    _ => {
                        let v = jv(m);
                        use super::util::V;
                        (v.i("code") as u32, v.o("con").map(|c| c as u32))
                    }
                };
                if code != 0 && con == Some(self.opponent) && set.insert(code) {
                    out.push(code);
                }
            }
            self.seen_cache = Some(out.into_iter().filter(|&c| card(c).is_some()).collect());
        }
        self.seen_cache.clone().unwrap()
    }

    pub fn filler(&self) -> &'static VanillaCompletion {
        VanillaCompletion::get()
    }

    /// The opponent Special Summons a hand card by its own procedure that the replay's hand does
    /// not have: one of the hidden hand cards is that card.
    fn check_procedure_assumption(&mut self, prompt: &Msg, up: &[Msg], duel: &RDuel) -> R<()> {
        let mut code = None;
        for (i, m) in up.iter().enumerate() {
            if matches!(m, Msg::Chaining { .. } | Msg::NewPhase { .. } | Msg::NewTurn { .. } | Msg::Summoning { .. } | Msg::Flipsummoning { .. } | Msg::Set { .. }) {
                return Ok(());
            }
            if let Msg::Spsummoning { .. } = m {
                let mv = up[..i].iter().rev().find(|x| matches!(x, Msg::Move { .. }));
                match mv {
                    Some(Msg::Move { code: c, from, to, .. }) if *c != 0 && from.con == self.opponent && from.loc == LOCATION_HAND && to.loc == LOCATION_MZONE => code = Some(*c),
                    _ => return Ok(()),
                }
                break;
            }
        }
        let Some(code) = code else { return Ok(()) };
        if Command::Spsummon.cards(prompt).unwrap_or(&[]).iter().any(|e| e.code == code && e.loc == LOCATION_HAND) {
            return Ok(());
        }
        let hand = duel.location_cards(self.opponent, LOCATION_HAND);
        if hand.iter().any(|&(c, _)| c == code) {
            return Ok(());
        }
        for seq in 0..hand.len() as u32 {
            if let Some(fix) = self.assume_fix((self.opponent, LOCATION_HAND, seq), &[code]) {
                return div(Divergence::with_fix(self.pos, "the opponent summons a hand card by its procedure that the hypothesis does not have", Some(fix)));
            }
        }
        Ok(())
    }

    /// The opponent set a hidden hand card that has the wrong kind in the hypothesis.
    pub fn check_idle_assumption(&mut self, prompt: Option<&Msg>, up: &[Msg], mut duel: Option<&mut RDuel>) -> R<()> {
        for (i, m) in up.iter().enumerate() {
            let t = m.name();
            if super::step::is_activation_prelude_t(t)
                || matches!(t, "PAY_LPCOST" | "FIELD_DISABLED" | "DAMAGE" | "RECOVER" | "LPUPDATE" | "ADD_COUNTER" | "REMOVE_COUNTER" | "MISSED_EFFECT")
            {
                continue;
            }
            if let Msg::ShuffleHand { player, .. } = m
                && *player != self.opponent
            {
                continue;
            }
            if let Msg::Draw { .. } = m {
                continue;
            }
            let Msg::Set { con, loc, seq, pos, .. } = m else { return Ok(()) };
            let before = &up[..i];
            let (src, failed_monster) = if *loc == LOCATION_HAND {
                let fm = before.iter().any(|x| matches!(x, Msg::Move { reason, .. } if reason & REASON_RELEASE != 0)) || *pos == POS_FACEDOWN_DEFENSE;
                (Some(li(*con, *loc, *seq)), fm)
            } else {
                (infer::source_of(&li(*con, *loc, *seq), before), false)
            };
            let Some(src) = src else { return Ok(()) };
            if src.con != self.opponent || !hidden_pile(src.loc) {
                return Ok(());
            }
            let mset = *loc == LOCATION_MZONE || failed_monster;
            let command = if mset { Command::Mset } else { Command::Sset };
            let filler = self.filler();
            let mut tributes: u32 = 0;
            let mut trib_locs: Vec<(u32, u32)> = Vec::new();
            if mset {
                tributes = before
                    .iter()
                    .filter(|x| match x {
                        Msg::Move { from, reason, .. } => from.loc != LOCATION_HAND && reason & REASON_DESTROY == 0 && (reason & (REASON_RELEASE | REASON_COST) != 0 || leaves_field(x)),
                        _ => false,
                    })
                    .count() as u32;
                trib_locs = before
                    .iter()
                    .filter_map(|x| match x {
                        Msg::Move { from, reason, .. } if reason & REASON_RELEASE != 0 && from.loc == LOCATION_MZONE && from.con == src.con => Some((from.con, from.seq)),
                        _ => None,
                    })
                    .collect();
            }
            let levels = match tributes {
                0 => 1..5,
                1 => 5..13,
                _ => 7..13,
            };
            let field_zone = *seq == 5;
            let entries: Vec<Card> = match prompt {
                Some(p) => command.cards(p).unwrap_or(&[]).iter().filter(|e| infer::same_loc(&(*e).into(), &src)).cloned().collect(),
                None => vec![],
            };
            let foreign_moves: Vec<(u32, u32)> = if mset {
                before
                    .iter()
                    .filter_map(|x| match x {
                        Msg::Move { from, reason, .. } if reason & REASON_RELEASE != 0 && from.con != src.con => Some((from.con, from.seq)),
                        _ => None,
                    })
                    .collect()
            } else {
                vec![]
            };
            let foreign = !foreign_moves.is_empty()
                && !match duel.as_deref_mut() {
                    Some(d) => foreign_moves.iter().all(|&(c, s)| d.has_extra_release(c, s)),
                    None => false,
                };
            let earlier = self.lock_var(src.con, src.loc, src.seq).ok().flatten();
            let earlier_need: Option<Need> = match earlier {
                Some(v) => self.lock().ok().and_then(|t| t.vars[v].need),
                None => None,
            };
            let discards = mset
                && before.iter().any(|x| matches!(x, Msg::Move { from, reason, .. } if from.loc == LOCATION_HAND && from.con == src.con && reason & REASON_DISCARD != 0 && reason & REASON_COST != 0));
            let proc_mset = si::proc_cards("mset");
            let set_proc_trib = if mset && tributes >= 3 { si::set_proc_tribute_cards(tributes) } else { Default::default() };
            let discard_procs = si::discard_set_proc_cards();
            let fits = |code: u32| -> bool {
                let Some(c) = card(code) else { return false };
                let base = if mset { levels.contains(&c.level) && c.type_ & TYPE_RITUAL == 0 && !si::unsummonable(code) } else { (c.type_ & TYPE_FIELD != 0) == field_zone };
                base && (!foreign || proc_mset.contains(&code))
                    && (!(mset && tributes >= 3) || set_proc_trib.contains(&code))
                    && earlier_need.is_none_or(|n| fits_need(Some(c), n, false))
                    && (!discards || discard_procs.contains(&code))
            };
            let usual_levels = match tributes {
                0 => 1..5,
                1 => 5..7,
                _ => 7..13,
            };
            let seen_codes = self.opp_seen_codes();
            let filler_codes: BTreeSet<u32> = filler.vanillas.iter().chain(&filler.settable_monsters_all).copied().filter(|c| !seen_codes.contains(c)).collect();
            let usual_fit = |code: u32| -> bool {
                let c = card(code);
                !mset || c.is_none() || usual_levels.contains(&c.unwrap().level) || !filler_codes.contains(&code) || tributes == 0
            };
            let trib_locs: Vec<(u32, u32)> = if mset && !foreign && src.loc == LOCATION_HAND { trib_locs } else { vec![] };
            let tribute_ok = trib_locs.is_empty()
                || match duel.as_deref_mut() {
                    None => true,
                    Some(d) => self.settable_as(d, src, &[None], tributes, &trib_locs).get(&None).copied().unwrap_or(true),
                };
            if !entries.is_empty() && entries.iter().all(|e| fits(e.code) && usual_fit(e.code)) && tribute_ok {
                return Ok(());
            }
            let hand_now = |d: &mut RDuel| -> Option<crate::state::CardQuery> {
                let snap = capture(&mut d.d);
                snap.players[src.con as usize].hand.get(src.seq as usize).cloned().flatten()
            };
            if !entries.is_empty() && mset && src.loc == LOCATION_HAND && !foreign && !discards && tributes < 3 && tribute_ok
                && let Some(d) = duel.as_deref_mut()
                && let Some(now) = hand_now(d)
                && let Some(lv) = now.level
                && levels.contains(&lv)
                && (usual_levels.contains(&lv) || now.code.and_then(card).is_none_or(|c| c.level != lv) || !filler_codes.contains(&now.code.unwrap_or(0)) || tributes == 0)
            {
                return Ok(()); // (lowered by an effect to fit, or a usual Level)
            }
            let want = if mset { TYPE_MONSTER } else { TYPE_SPELL | TYPE_TRAP };
            let mut codes: Vec<u32> = match prompt {
                Some(p) => command.cards(p).unwrap_or(&[]).iter().map(|e| e.code).filter(|&c| fits(c)).collect(),
                None => vec![],
            };
            if mset && earlier_need == Some(Need::St) {
                let mut extra: Vec<u32> = monster_sset_codes().iter().copied().filter(|&c| card(c).is_some() && fits(c)).collect();
                extra.sort();
                codes.extend(extra);
            }
            let sorted_procs = |procs: &dyn Fn(u32) -> bool, all: Vec<u32>, seen_first: bool| -> Vec<u32> {
                let mut out: Vec<u32> = seen_codes.iter().copied().filter(|&c| procs(c) && fits(c)).collect();
                let mut rest: Vec<u32> = all.into_iter().filter(|c| (!seen_first || !seen_codes.contains(c)) && card(*c).is_some() && fits(*c)).collect();
                rest.sort();
                out.extend(rest);
                out
            };
            if mset && tributes >= 3 && !discards {
                codes = sorted_procs(&|c| set_proc_trib.contains(&c), set_proc_trib.iter().copied().collect(), false);
            }
            if discards {
                codes = sorted_procs(&|c| discard_procs.contains(&c), discard_procs.iter().copied().collect(), true);
            }
            if foreign {
                codes = sorted_procs(&|c| proc_mset.contains(&c), proc_mset.iter().copied().collect(), true);
            }
            codes.extend(seen_codes.iter().copied().filter(|&c| {
                fits(c) && card(c).is_some_and(|d| d.type_ & want != 0 && d.type_ & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK) == 0)
            }));
            if mset {
                let races: Vec<u64> = most_common(seen_codes.iter().filter_map(|&c| card(c)).filter(|d| d.type_ & TYPE_MONSTER != 0).map(|d| d.race), 3);
                let view = self.items[..(self.pos + 1).min(self.items.len())].iter().rev().find_map(|it| if let ObsItem::Prompt(_, Some(s)) = it { Some(s.clone()) } else { None });
                let now_races: Vec<u64> = most_common(
                    view.iter().flat_map(|v| v.players.iter()).flat_map(|pl| pl.mzone.iter()).filter_map(|c| c.as_ref().and_then(|c| c.race)).filter(|&r| r != 0),
                    2,
                );
                for race in now_races {
                    let mut found: Vec<u32> = filler.vanillas.iter().copied().filter(|&c| card(c).unwrap().race == race && fits(c)).take(2).collect();
                    if found.is_empty() {
                        found = filler.settable_monsters_all.iter().copied().filter(|&c| card(c).unwrap().race == race && fits(c)).take(2).collect();
                    }
                    if found.is_empty() {
                        found = self.race_stand_ins(race, &fits);
                    }
                    codes.extend(found);
                }
                for race in races {
                    codes.extend(filler.vanillas.iter().copied().filter(|&c| card(c).unwrap().race == race && fits(c)).take(2));
                }
            }
            codes.extend((if mset { &filler.settable_monsters_all } else { &filler.settable_spells }).iter().copied().filter(|&c| fits(c)));
            if mset {
                let mut delta: i64 = 0;
                if src.loc == LOCATION_HAND
                    && let Some(d) = duel.as_deref_mut()
                    && let Some(now) = hand_now(d)
                    && let Some(base) = now.code.filter(|&c| c != 0).and_then(card)
                    && let Some(lv) = now.level
                    && base.level != 0
                {
                    delta = lv as i64 - base.level as i64;
                }
                codes.sort_by_key(|&c| !usual_levels.contains(&((card(c).unwrap().level as i64 + delta) as u32)));
            }
            if src.loc == LOCATION_HAND {
                let known = self.known_codes(src.con);
                let unknown = codes.iter().copied().find(|c| !known.contains(c));
                if let Some(u) = unknown
                    && !self.feasible((self.pos, src.con, src.loc, src.seq, u))
                {
                    codes = known.iter().copied().filter(|&c| fits(c) && card(c).is_some_and(|d| d.type_ & want != 0 && !is_extra_deck(d))).collect();
                }
            }
            if mset
                && src.loc == LOCATION_HAND
                && let Some(d) = duel.as_deref_mut()
            {
                let first: Vec<Option<u32>> = codes.iter().take(60).map(|&c| Some(c)).collect();
                let mut ok = self.settable_as(d, src, &first, tributes, &trib_locs);
                if !ok.is_empty() && !ok.values().any(|&x| x) {
                    let mut seen_races: HashMap<u64, i64> = HashMap::new();
                    for c in &seen_codes {
                        if let Some(x) = card(*c).filter(|x| x.type_ & TYPE_MONSTER != 0) {
                            *seen_races.entry(x.race).or_default() += 1;
                        }
                    }
                    let db = crate::game::card_db();
                    let mut pool: Vec<u32> = db
                        .cards
                        .values()
                        .filter(|c| {
                            c.type_ & TYPE_MONSTER != 0
                                && c.alias == 0
                                && db.ots.get(&c.code).copied().unwrap_or(0) & 0x3 != 0
                                && !is_extra_deck(c)
                                && c.type_ & TYPE_TOKEN == 0
                                && !ok.contains_key(&Some(c.code))
                                && fits(c.code)
                        })
                        .map(|c| c.code)
                        .collect();
                    pool.sort_by_key(|&c| {
                        let x = card(c).unwrap();
                        (x.type_ & TYPE_NORMAL == 0, -seen_races.get(&x.race).copied().unwrap_or(0), c)
                    });
                    let mut found: Vec<u32> = Vec::new();
                    let mut k = 0;
                    while k < pool.len().min(1600) {
                        let chunk: Vec<Option<u32>> = pool[k..(k + 200).min(pool.len())].iter().map(|&c| Some(c)).collect();
                        let more = self.settable_as(d, src, &chunk, tributes, &trib_locs);
                        found.extend(chunk.iter().flatten().copied().filter(|c| more.get(&Some(*c)).copied().unwrap_or(false)));
                        if found.len() >= 3 {
                            break;
                        }
                        k += 200;
                    }
                    let mut new: Vec<u32> = found.iter().copied().take(6).collect();
                    new.extend(codes);
                    codes = new;
                    for &c in found.iter().take(6) {
                        ok.insert(Some(c), true);
                    }
                }
                codes.sort_by_key(|&c| !ok.get(&Some(c)).copied().unwrap_or(false));
            }
            let known: Vec<u32> = self.slot_codes((src.con, src.loc, src.seq)).into_iter().filter(|&c| fits(c)).collect();
            let mut all = known.clone();
            all.extend(codes.into_iter().filter(|c| !known.contains(c)));
            self.assume((src.con, src.loc, src.seq), &all, "opponent set a card the hypothesis cannot set")?;
            return Ok(());
        }
        Ok(())
    }

    /// Whether the player may Set the hand card at `l` if it had each candidate's properties.
    fn settable_as(&self, duel: &mut RDuel, l: Li, codes: &[Option<u32>], tributes: u32, trib_locs: &[(u32, u32)]) -> HashMap<Option<u32>, bool> {
        let sumtype = if tributes > 0 { SUMMON_TYPE_TRIBUTE } else { SUMMON_TYPE_NORMAL };
        let mut uniq: Vec<Option<u32>> = Vec::new();
        for &c in codes {
            if !uniq.contains(&c) {
                uniq.push(c);
            }
        }
        let cands: Vec<Option<crate::rduel::Props>> = uniq
            .into_iter()
            .filter_map(|c| match c {
                None => Some(None),
                Some(code) => Self::props(code).map(Some),
            })
            .collect();
        let tl: Vec<(u8, u32)> = trib_locs.iter().map(|&(c, s)| (c as u8, s)).collect();
        crate::rduel::settable_as(&mut duel.d, (l.con as u8, l.seq), sumtype, &tl, &cands).into_iter().collect()
    }

    /// The codes a hidden slot is known to be limited to.
    fn slot_codes(&mut self, l: Slot) -> Vec<u32> {
        let Ok(Some(mut v)) = self.lock_var(l.0, l.1, l.2) else { return vec![] };
        let Ok(t) = self.lock() else { return vec![] };
        let code_of = |x: VarId| t.vars[x].fixed_code.or(t.vars[x].code);
        let done = |codes: Vec<Option<u32>>| -> Vec<u32> {
            if !codes.is_empty() && codes.iter().all(|c| c.is_some_and(|c| c != 0)) {
                let s: BTreeSet<u32> = codes.into_iter().flatten().collect();
                s.into_iter().collect()
            } else {
                vec![]
            }
        };
        for _ in 0..16 {
            let x = &t.vars[v];
            if x.kind != Kind::Claim {
                return vec![];
            }
            if let Some(allowed) = &x.allowed
                && !allowed.is_empty()
            {
                let xs: Vec<VarId> = allowed.iter().copied().filter(|&a| a < t.vars.len()).collect();
                let codes: Vec<Option<u32>> = xs.iter().map(|&a| code_of(a)).collect();
                return if xs.len() == allowed.len() { done(codes) } else { vec![] };
            }
            let Some(pool) = x.pool else { return vec![] };
            let members = &t.pools[pool].members;
            if members.len() != 1 {
                return done(members.iter().map(|&m| code_of(m)).collect());
            }
            v = members[0];
        }
        vec![]
    }

    /// Cards selected one by one and then moved together: if another of them cannot be chosen
    /// any more, an earlier pick went wrong.
    fn check_selection_dead_end(&self, prompt: &Msg, up: &[Msg], duel: &RDuel, ids: Option<&[Option<u32>]>) -> R<()> {
        let Msg::SelectUnselectCard { select_cards: sel, unselect_cards, .. } = prompt else { return Ok(()) };
        let Some(ids) = ids else { return Ok(()) };
        let unsel: BTreeSet<Option<u32>> = ids.iter().skip(sel.len()).copied().collect();
        if unsel.is_empty() {
            return Ok(());
        }
        let Some(k) = up.iter().position(|m| !matches!(m, Msg::Hint { .. } | Msg::CardHint { .. })) else { return Ok(()) };
        let Msg::Move { reason: r0, .. } = &up[k] else { return Ok(()) };
        let places: BTreeSet<(u32, u32)> = sel.iter().chain(unselect_cards).map(|c| (c.con, c.loc)).collect();
        let mut group = Vec::new();
        for m in &up[k..] {
            match m {
                Msg::Move { reason, from, .. } if reason == r0 && places.contains(&(from.con, from.loc)) => group.push(m.clone()),
                _ => break,
            }
        }
        let used = self.loc_map(duel).references(&group);
        let selected = &ids[..sel.len().min(ids.len())];
        if used.iter().any(|c| selected.contains(&Some(*c))) || !unsel.iter().all(|u| u.is_some_and(|u| used.contains(&u))) {
            return Ok(());
        }
        if used.iter().any(|c| !unsel.contains(&Some(*c))) {
            return div(Divergence::new(self.pos, "a card used with the selected ones cannot be selected any more"));
        }
        Ok(())
    }

    /// The opponent special summoned a hand card by its own procedure, which the replay does not
    /// offer although it has that card there: let the hidden cards qualify until the summon.
    fn check_summon_condition(&mut self, duel: &RDuel, prompt: &Msg, up: &[Msg]) -> R<()> {
        for (i, m) in up.iter().enumerate() {
            if matches!(m.name(), "MOVE" | "HINT" | "CARD_HINT" | "CARD_SELECTED" | "CONFIRM_CARDS" | "PAY_LPCOST" | "SHUFFLE_HAND" | "SHUFFLE_DECK" | "SHUFFLE_EXTRA" | "LPUPDATE") {
                continue;
            }
            let Msg::Spsummoning { code, con, loc, seq, .. } = m else { return Ok(()) };
            let Some(src) = infer::source_of(&li(*con, *loc, *seq), &up[..i]) else { return Ok(()) };
            if src.con != self.opponent || !matches!(src.loc, LOCATION_HAND | LOCATION_EXTRA) {
                return Ok(());
            }
            let sps = Command::Spsummon.cards(prompt).unwrap_or(&[]);
            if src.loc == LOCATION_EXTRA {
                if src.pos & POS_FACEUP != 0 || sps.iter().any(|e| e.loc == LOCATION_SZONE || (e.loc == LOCATION_EXTRA && e.code == *code)) {
                    return Ok(());
                }
                if !duel.location_cards(self.opponent, LOCATION_EXTRA).iter().any(|&(c, _)| c == *code) {
                    return Ok(());
                }
            } else {
                if sps.iter().any(|e| infer::same_loc(&e.into(), &src) || e.loc == LOCATION_SZONE) {
                    return Ok(());
                }
                let lm = self.loc_map(duel);
                let cid = lm.at(&src).filter(|&c| c != 0);
                if cid.is_none_or(|c| lm.codes.get(&c) != Some(code)) {
                    return Ok(());
                }
            }
            if let Some(fix) = self.silent_since_view_fix() {
                return div(Divergence::with_fix(self.pos, "opponent summoned a card whose condition the replay does not meet (an earlier silent choice)", Some(fix)));
            }
            let span = (self.prompt_cp.map_or(self.pos, |c| c.0), self.pos + i);
            if !self.match_piles.contains(&span) {
                return div(Divergence::with_fix(self.pos, "opponent summoned a card whose condition the replay does not meet", Some(Fix::MatchPile(span.0, span.1))));
            }
            return Ok(());
        }
        Ok(())
    }

    fn candidates(&mut self, duel: &RDuel, prompt: &Msg, up: Vec<Msg>, ids: Option<&[Option<u32>]>) -> R<infer::Inferred> {
        let t = prompt.name();
        let idx_of = |r: &Response| r.index.unwrap_or(-1);
        let pass = || Response { index: Some(-1), ..Response::default() };
        Ok(match t {
            "SELECT_IDLECMD" => infer::infer_idle(prompt, &up, Some(&self.upcoming_ext(40))),
            "SELECT_BATTLECMD" => infer::infer_battle(prompt, &up, Some(&self.upcoming_ext(40))),
            "SELECT_CHAIN" => {
                let (mut cands, mut certain) = infer::infer_chain(prompt, &up);
                let forced = matches!(prompt, Msg::SelectChain { forced, .. } if *forced != 0);
                let act_first = cands.first().is_some_and(|c| idx_of(c) >= 0) && !forced;
                if act_first && self.leaving_phase() {
                    if !self.restarts_after_chain() {
                        let mut v = vec![pass()];
                        v.extend(cands.into_iter().filter(|c| idx_of(c) >= 0));
                        cands = v;
                    }
                } else if act_first && duel.info().2 == PHASE_BATTLE_START && self.next_state_phase() == Some(PHASE_BATTLE_STEP) {
                    let mut v = vec![pass()];
                    v.extend(cands.into_iter().filter(|c| idx_of(c) >= 0));
                    cands = v;
                    certain = Certain::Yes;
                }
                (cands, certain)
            }
            "SELECT_EFFECTYN" => {
                let Msg::SelectEffectyn { code, .. } = prompt else { unreachable!() };
                let descs: Vec<u64> = duel.select_chains().into_iter().filter(|&(c, _)| c == *code).map(|x| x.1).collect();
                infer::infer_effectyn(prompt, &up, Some(&descs))
            }
            "SELECT_YESNO" => (vec![Response { yes: Some(true), ..Response::default() }, Response { yes: Some(false), ..Response::default() }], Certain::No),
            "SELECT_OPTION" => infer::infer_option(prompt, &up, self.opponent),
            "SELECT_CARD" | "SELECT_TRIBUTE" | "SELECT_SUM" => {
                let (window, extended) = self.selection_window(up);
                let lm = self.loc_map(duel);
                let (c, certain) = infer::infer_cards(prompt, &window, &lm, ids.map(|x| x.to_vec()), self.prompt_select_msg);
                (c, if extended && certain == Certain::Yes { Certain::No } else { certain })
            }
            "SELECT_UNSELECT_CARD" => {
                let (window, extended) = self.selection_window(up);
                self.check_selection_dead_end(prompt, &window, duel, ids)?;
                let lm = self.loc_map(duel);
                let (c, certain) = infer::infer_unselect(prompt, &window, &lm, ids.map(|x| x.to_vec()));
                (c, if extended && certain == Certain::Yes { Certain::No } else { certain })
            }
            "SELECT_PLACE" | "SELECT_DISFIELD" => infer::infer_place(prompt, &up, Some(&self.upcoming_ext(40))),
            "SELECT_POSITION" => infer::infer_position(prompt, &up),
            "ANNOUNCE_CARD" | "ANNOUNCE_NUMBER" | "ANNOUNCE_RACE" | "ANNOUNCE_ATTRIB" => infer::infer_announce(prompt, &up),
            "SELECT_COUNTER" => infer::infer_counter(prompt, &up),
            "ROCK_PAPER_SCISSORS" => infer::infer_rps(prompt, &up),
            "SORT_CARD" | "SORT_CHAIN" => infer::infer_sort(prompt, &up),
            _ => return div(Divergence::new(self.pos, format!("no inference for opponent prompt {t}"))),
        })
    }

    /// Stream position of the opponent's latest CHAINING (shortly before).
    pub fn chain_start(&self) -> Option<usize> {
        let lo = (self.pos as i64 - 30).max(-1);
        let mut j = self.pos as i64 - 1;
        while j > lo {
            let ju = j as usize;
            j -= 1;
            match &self.items[ju] {
                ObsItem::Msg(Msg::ChainEnd {} | Msg::NewPhase { .. } | Msg::NewTurn { .. }) => return None,
                ObsItem::Msg(Msg::Chaining { .. }) => return if self.opp_activates(ju) { Some(ju) } else { None },
                _ => {}
            }
        }
        None
    }

    /// Stream position after the latest chain end or phase start.
    pub fn action_start(&self) -> usize {
        let lo = (self.pos as i64 - 40).max(-1);
        let mut j = self.pos as i64 - 1;
        while j > lo {
            let ju = j as usize;
            if let ObsItem::Msg(Msg::ChainEnd {} | Msg::ChainSolved { .. } | Msg::NewPhase { .. } | Msg::NewTurn { .. }) = &self.items[ju] {
                return ju + 1;
            }
            j -= 1;
        }
        self.pos.saturating_sub(40)
    }

    /// The first untried span over which hidden cards may match nothing.
    pub fn nomatch_span(&self, start: Option<usize>) -> Option<(usize, usize)> {
        let link = self.link_start();
        let mut spans = Vec::new();
        if link < self.pos && matches!(self.items[link], ObsItem::Msg(Msg::ChainSolving { .. })) && start.is_none_or(|s| link > s) {
            spans.push((link, self.pos));
        }
        spans.push((link.min(start.unwrap_or(self.pos)), self.pos));
        spans.into_iter().find(|sp| !self.nomatch_piles.contains(sp) && !self.match_piles.contains(sp))
    }

    /// Stream position of the first CHAINING of the chain that ended just before.
    pub fn prev_chain_start(&self) -> Option<usize> {
        let mut j = self.pos as i64 - 1;
        while j >= 0 && matches!(self.items[j as usize], ObsItem::Msg(Msg::Hint { .. } | Msg::PlayerHint { .. } | Msg::CardHint { .. })) {
            j -= 1;
        }
        if j < 0 || !matches!(self.items[j as usize], ObsItem::Msg(Msg::ChainEnd {})) {
            return None;
        }
        let mut first = None;
        let lo = (j - 80).max(-1);
        let mut k = j - 1;
        while k > lo {
            match &self.items[k as usize] {
                ObsItem::Msg(Msg::ChainEnd {} | Msg::NewPhase { .. } | Msg::NewTurn { .. }) => break,
                ObsItem::Msg(Msg::Chaining { .. }) => first = Some(k as usize),
                _ => {}
            }
            k -= 1;
        }
        first
    }

    /// Stream position where the chain link being built or resolved began.
    pub fn link_start(&self) -> usize {
        let lo = (self.pos as i64 - 30).max(-1);
        let mut j = self.pos as i64 - 1;
        while j > lo {
            let ju = j as usize;
            match &self.items[ju] {
                ObsItem::Msg(Msg::ChainEnd {} | Msg::NewPhase { .. } | Msg::NewTurn { .. } | Msg::ChainSolved { .. }) => return self.pos,
                ObsItem::Msg(Msg::Chaining { .. } | Msg::ChainSolving { .. }) => return ju,
                _ => {}
            }
            j -= 1;
        }
        self.pos
    }

    /// The phase in the viewer's next state view while the upcoming chain is built or resolved.
    fn next_state_phase(&self) -> Option<u32> {
        let mut ended = false;
        let end = (self.pos + 400).min(self.items.len());
        for item in &self.items[self.pos.min(end)..end] {
            match item {
                ObsItem::Msg(m) => match m {
                    Msg::NewPhase { .. } | Msg::NewTurn { .. } => return None,
                    Msg::ChainEnd {} => ended = true,
                    Msg::Hint { .. } | Msg::CardHint { .. } | Msg::PlayerHint { .. } => {}
                    _ if ended => return None,
                    _ => {}
                },
                ObsItem::Prompt(p, Some(s)) => {
                    if ended && !(matches!(p, Msg::SelectEffectyn { .. }) || matches!(p, Msg::SelectChain { forced, .. } if *forced != 0)) {
                        return None;
                    }
                    return Some(s.phase);
                }
                _ => {}
            }
        }
        None
    }

    fn battle_start_window(&self) -> bool {
        self.event_hint.is_some_and(|h| h.0 == self.pos && h.1 == BATTLE_START_EVENT)
    }

    fn after_chain_end(&self) -> bool {
        let mut j = self.pos as i64 - 1;
        while j >= 0 && matches!(self.items[j as usize], ObsItem::Msg(Msg::Hint { .. } | Msg::CardHint { .. } | Msg::PlayerHint { .. })) {
            j -= 1;
        }
        j >= 0 && matches!(self.items[j as usize], ObsItem::Msg(Msg::ChainEnd {}))
    }

    fn leaving_phase(&self) -> bool {
        self.event_hint.is_some_and(|h| h.0 == self.pos && infer::PHASE_LEAVE_EVENTS.contains(&h.1))
    }

    /// The phase the viewer's phase command right before this window asked for.
    fn command_target(&self) -> Option<u32> {
        let mut j = self.pos as i64 - 1;
        while j >= 1 && matches!(self.items[j as usize], ObsItem::Msg(Msg::Hint { .. })) {
            j -= 1;
        }
        if j < 1 {
            return None;
        }
        let (ObsItem::Response(r), ObsItem::Prompt(before, _)) = (&self.items[j as usize], &self.items[j as usize - 1]) else { return None };
        if !matches!(before, Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. }) {
            return None;
        }
        match r.kind {
            Some(Command::ToEp) => Some(PHASE_END),
            Some(Command::ToM2) => Some(PHASE_MAIN2),
            Some(Command::ToBp) => Some(PHASE_BATTLE_START),
            _ => None,
        }
    }

    /// Whether the stream shows the turn player's command again after the upcoming chain.
    fn restarts_after_chain(&self) -> bool {
        let (mut depth, mut started) = (0, false);
        let end = (self.pos + 600).min(self.items.len());
        let window = &self.items[self.pos.min(end)..end];
        for (n, item) in window.iter().enumerate() {
            match item {
                ObsItem::Msg(m) => match m {
                    Msg::Chaining { .. } => started = true,
                    Msg::ChainEnd {} if started => depth = 1,
                    Msg::NewPhase { phase } if depth != 0 => {
                        let target = self.command_target();
                        let mut reached = *phase;
                        for x in &window[n + 1..] {
                            match x {
                                ObsItem::Msg(Msg::NewPhase { phase }) => reached = *phase,
                                ObsItem::Msg(Msg::Hint { .. }) => {}
                                _ => break,
                            }
                        }
                        return target.is_some_and(|t| reached != t);
                    }
                    Msg::Win { .. } => return true,
                    _ => {}
                },
                ObsItem::Prompt(p, _) if depth != 0 && matches!(p, Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. }) => return true,
                _ => {}
            }
        }
        true
    }

    pub fn decide(&mut self, duel: &mut RDuel, prompt: &Msg) -> R<Response> {
        self.stats.opponent_decisions += 1;
        self.stall += 1;
        if self.stall > self.max_stall {
            return div(Divergence::new(self.pos, "opponent decisions make no progress"));
        }
        let up = self.upcoming();
        // where the upcoming shuffles put the cards (for resolving references)
        if let Some(h) = &self.hypothesis {
            let sc = self.shuffle_count;
            duel.shuffle_plan = h
                .shuffles
                .iter()
                .skip(sc)
                .zip(self.shuffle_ids.iter().skip(sc))
                .map(|(sh, ids)| {
                    let loc = match sh.location {
                        PoolLoc::Loc(l) => l,
                        PoolLoc::Set => LOCATION_SZONE,
                    };
                    (sh.player, loc, ids.as_ref().map(|v| v.iter().map(|&c| Some(c)).collect()))
                })
                .collect();
        }
        self.check_force(prompt, &up, duel)?;
        if let Msg::SelectIdlecmd { .. } = prompt {
            self.check_idle_assumption(Some(prompt), &up, Some(duel))?;
            self.check_procedure_assumption(prompt, &up, duel)?;
            self.check_summon_condition(duel, prompt, &up)?;
        }
        if self.depth == 0 {
            let n = self.decisions_at.get(&self.pos).copied().unwrap_or(0);
            self.decisions_at.insert(self.pos, n + 1);
            self.dkey = Some((self.pos, n));
        } else {
            self.dkey = None;
        }
        let at = (self.pos, prompt.name());
        self.planned = None;
        if !self.plan.is_empty() {
            if self.plan[0].0 == at {
                self.planned = Some(self.plan.remove(0).1);
            } else {
                self.plan.clear();
            }
        }
        let resp = self.decide_memo(duel, prompt, up)?;
        if self.depth == 0 && !self.code_tests.is_empty() {
            self.run_code_tests(duel, prompt, &resp)?;
        }
        if self.depth == 0 && self.suspect_tests.as_ref().is_some_and(|s| self.pos >= s.at) {
            self.run_suspect_tests(duel, prompt, &resp)?;
        }
        if self.planned.as_ref().is_some_and(|p| *p != resp) {
            self.plan.clear();
        }
        if self.depth > 0 && self.depth as i64 == self.report_depth {
            self.probe_decisions.push((at, resp.clone()));
        }
        Ok(resp)
    }

    fn decide_memo(&mut self, duel: &mut RDuel, prompt: &Msg, up: Vec<Msg>) -> R<Response> {
        let card_prompt = CARD_PROMPTS.contains(&prompt.name());
        let mut key = None;
        if self.depth == 0 && self.use_memo && !self.dkey.is_some_and(|d| self.overrides.contains_key(&d)) {
            let k = (self.pos, self.stall, prompt_key(prompt));
            if let Some(hit) = self.memo.get(&k).cloned()
                && !card_prompt
            {
                let (resp, rec, _, _) = hit;
                if let (Some((n, i)), Some(d)) = (rec, self.dkey) {
                    self.recent.push((d, n, i));
                    self.since_state.push((d, n, i));
                }
                return Ok(resp);
            }
            key = Some(k);
        }
        let n = self.recent.len();
        self.choice_seen = None;
        self.prompt_ids = None;
        if let Some(k) = &key
            && card_prompt
        {
            let cards = prompt_cards(prompt, true);
            let ids = infer::card_ids(&self.loc_map(duel), &cards);
            if self.planned.is_none() && !self.memo_cards.is_empty() {
                self.planned = self.memo_cards.get(&(k.0, k.1, k.2.clone(), ids.clone())).cloned();
            }
            self.prompt_ids = Some(ids);
        }
        let resp = self.decide_fresh(duel, prompt, up)?;
        if let Some(k) = key {
            let rec = match self.recent.last() {
                Some(&(d, nv, i)) if self.recent.len() > n && Some(d) == self.dkey && i >= 0 => Some((nv, i)),
                _ => None,
            };
            self.memo.insert(k, (resp.clone(), rec, self.choice_seen, self.prompt_ids.clone()));
        }
        Ok(resp)
    }

    fn decide_fresh(&mut self, duel: &mut RDuel, prompt: &Msg, up: Vec<Msg>) -> R<Response> {
        let t = prompt.name();
        let card_prompt = CARD_PROMPTS.contains(&t);
        let mut ids: Option<Vec<Option<u32>>> = None;
        if card_prompt {
            let v = match &self.prompt_ids {
                Some(p) => p.clone(),
                None => infer::card_ids(&self.loc_map(duel), &prompt_cards(prompt, true)),
            };
            self.check_assumption(duel, prompt, &up, &v)?;
            ids = Some(v);
        }
        if let Msg::SelectCard { min, cards, .. } = prompt
            && *min >= 1
            && self.depth == 0
            && !cards.is_empty()
            && self.prompt_select_msg.is_some_and(|s| SUMMON_SELECT_MSGS.contains(&s))
            && cards.iter().all(|c| c.con == self.opponent && (c.loc & (LOCATION_DECK | LOCATION_HAND | LOCATION_EXTRA) != 0 || c.pos.unwrap_or(POS_FACEUP) & POS_FACEUP == 0))
        {
            let first = up.iter().find(|m| !matches!(m, Msg::Hint { .. } | Msg::CardHint { .. }));
            let mut later = Vec::new();
            for m in up.iter().take(15) {
                if matches!(m, Msg::NewPhase { .. } | Msg::NewTurn { .. } | Msg::Win { .. }) {
                    break;
                }
                later.push(m.clone());
            }
            let mut lm = self.loc_map(duel);
            let ids_offered: BTreeSet<Option<u32>> = infer::card_ids(&lm, cards).into_iter().collect();
            let summoned = later.iter().any(|m| matches!(m, Msg::Move { to, .. } if to.loc == LOCATION_MZONE && to.con == self.opponent));
            let refs = lm.references(&later);
            if matches!(first, Some(Msg::ChainSolved { .. } | Msg::ChainEnd {})) && !summoned && !refs.iter().any(|c| ids_offered.contains(&Some(*c))) {
                let cs = self.chain_start();
                if let Some(span) = self.nomatch_span(cs) {
                    return div(Divergence::with_fix(self.pos, "opponent selects among hidden cards without consequence", Some(Fix::NomatchPile(span.0, span.1))));
                }
            }
        }
        let (cands, certain) = self.candidates(duel, prompt, up.clone(), ids.as_deref())?;
        self.inferred = cands.clone();
        let mut cands = cands;
        if self.depth == 0 {
            for c in all_responses(prompt) {
                if !cands.contains(&c) {
                    cands.push(c);
                }
            }
        }
        let sig = (self.pos, prompt_key(prompt));
        if self.depth > 0 && self.probe_sig.as_ref() == Some(&sig) && matches!(t, "SELECT_IDLECMD" | "SELECT_BATTLECMD" | "SELECT_CHAIN") {
            return div(Divergence::new(self.pos, "probe returned to its prompt"));
        }
        let mut valid: Vec<Response> = Vec::new();
        for c in cands {
            if encode(prompt, &c).is_err() {
                continue;
            }
            if (c.cancel == Some(true) || c.finish == Some(true)) && self.cancelled.contains(&sig) {
                continue;
            }
            if !valid.contains(&c) {
                valid.push(c);
            }
        }
        if valid.is_empty() {
            let mut fix = None;
            if matches!(t, "SELECT_PLACE" | "SELECT_POSITION") {
                let mut start = self.chain_start();
                if start.is_none() {
                    let lo = (self.pos as i64 - 12).max(-1);
                    let mut j = self.pos as i64 - 1;
                    while j > lo {
                        if let ObsItem::Msg(Msg::Chaining { trigger_con, .. }) = &self.items[j as usize]
                            && *trigger_con == self.opponent
                        {
                            start = Some(j as usize);
                            break;
                        }
                        j -= 1;
                    }
                }
                if let (Some(s), Some(code)) = (start, self.opp_select_hint.filter(|&c| c != 0))
                    && card(code as u32).is_some()
                {
                    for loc in [LOCATION_DECK, LOCATION_HAND] {
                        if duel.location_cards(self.opponent, loc).iter().any(|&(c, _)| c as u64 == code) {
                            let key: Excl = (s, self.opponent, loc, vec![ALL_SLOTS], code as u32);
                            if !self.exclusions.contains(&key) && !self.failed_exclusions.contains(&key) {
                                fix = Some(Fix::Exclude(key));
                                break;
                            }
                        }
                    }
                }
                let span = self.nomatch_span(start);
                if fix.is_none()
                    && let Some(sp) = span
                {
                    fix = Some(Fix::NomatchPile(sp.0, sp.1));
                }
            }
            return div(Divergence::with_fix(self.pos, format!("no candidate response for opponent {t}"), fix));
        }
        let dkey = if self.depth == 0 { self.dkey } else { None };
        if let (Some(d), Some(ids)) = (dkey, &ids) {
            let mut m: HashMap<Option<u32>, usize> = HashMap::new();
            for (k, c) in valid.iter().enumerate() {
                if let Some(ix) = &c.indices
                    && ix.len() == 1
                    && (ix[0] as usize) < ids.len()
                {
                    m.insert(ids[ix[0] as usize], k);
                }
            }
            self.card_decisions.insert(d, m);
        }
        if let Some(d) = dkey
            && let Some(r) = self.forced_resp.get(&d).cloned()
            && encode(prompt, &r).is_ok()
        {
            if let Some(ids) = &ids {
                self.multi_decisions.insert(d, (ids.clone(), r.clone()));
            }
            self.recent.push((d, valid.len(), -1));
            self.trace(prompt, &r, false, valid.len());
            return Ok(r);
        }
        if let Some(d) = dkey
            && let Some(&i) = self.overrides.get(&d)
            && i < valid.len()
        {
            if let Some(ids) = &ids
                && valid[i].indices.as_ref().is_some_and(|x| !x.is_empty())
            {
                let ids = ids.clone();
                self.note_silent(duel, &up, d, &ids, &valid, i);
            }
            self.recent.push((d, valid.len(), i as i64));
            self.since_state.push((d, valid.len(), i as i64));
            self.trace(prompt, &valid[i], false, valid.len());
            return Ok(valid[i].clone());
        }
        let resp = match self.planned.clone() {
            Some(p) if valid.contains(&p) => {
                self.trace(prompt, &p, false, valid.len());
                p
            }
            _ => self.choose(duel, prompt, &valid, certain)?,
        };
        if resp.cancel == Some(true) || resp.finish == Some(true) {
            self.cancelled.insert(sig);
        }
        let ri = valid.iter().position(|c| *c == resp).unwrap_or(0);
        let is_chain = t == "SELECT_CHAIN";
        if let Some(d) = dkey
            && is_chain
            && resp.index.unwrap_or(-1) == -1
            && self.leaving_phase()
        {
            if let Some(act) = valid.iter().position(|c| c.index.unwrap_or(-1) >= 0) {
                self.window_choices.push((d, act));
            }
        } else if let Some(d) = dkey
            && is_chain
            && resp.index.unwrap_or(-1) >= 0
            && (self.leaving_phase() || self.battle_start_window() || self.after_chain_end())
            && let Some(passing) = valid.iter().position(|c| c.index.unwrap_or(-1) == -1)
        {
            self.window_activations.push((d, passing));
        }
        let offered = offered_cards(prompt);
        if let (Some(d), Some(ids)) = (dkey, &ids) {
            self.offered.insert(d, (ids.iter().copied().collect(), offered.iter().map(|c| (c.con, c.loc)).collect()));
        }
        if let (Some(d), Some(ids)) = (dkey, &ids)
            && resp.indices.as_ref().is_some_and(|x| !x.is_empty())
        {
            let ids = ids.clone();
            self.multi_decisions.insert(d, (ids.clone(), resp.clone()));
            self.note_silent(duel, &up, d, &ids, &valid, ri);
        }
        if let Some(d) = dkey
            && matches!(t, "SELECT_YESNO" | "SELECT_OPTION")
            && valid.len() > 1
        {
            let chain = capture(&mut duel.d).chain;
            if let Some(h) = chain.last().and_then(|l| l.handler.as_ref())
                && let Some(cid) = self.loc_map(duel).at(&h.into()).filter(|&c| c != 0)
            {
                self.silent_choices.push((d, vec![Some(cid)], valid.len(), ri, None));
            }
        }
        if dkey.is_some() && card_prompt && offered.iter().any(|c| c.con == self.opponent && c.loc & (LOCATION_DECK | LOCATION_HAND | LOCATION_EXTRA) != 0) {
            self.hidden_picks.push((self.pos, (self.link_start(), self.pos)));
        }
        if let Some(d) = dkey {
            self.recent.push((d, valid.len(), ri as i64));
            self.since_state.push((d, valid.len(), ri as i64));
        }
        Ok(resp)
    }

    fn choose(&mut self, duel: &RDuel, prompt: &Msg, valid: &[Response], certain: Certain) -> R<Response> {
        let n_inferred = valid.iter().filter(|c| self.inferred.contains(c)).count();
        if certain == Certain::Yes || valid.len() == 1 || n_inferred == 1 {
            self.trace(prompt, &valid[0], true, valid.len());
            return Ok(valid[0].clone());
        }
        if self.depth > 0
            && self.pos - self.probe_start >= self.min_progress
            && self.probe_boundary
            && self.test_until.is_none_or(|t| self.pos >= t)
            && self.probe_after_boundary > 0
            && self.probe_state_at.is_none_or(|s| self.pos > s)
        {
            return Err(RErr::ProbeSuccess);
        }
        if self.depth >= self.max_depth || (self.depth > 0 && self.probe_counter.get() >= PROBE_BUDGET) {
            return Ok(valid[0].clone());
        }
        self.stats.probed_decisions += 1;
        let cards = offered_cards(prompt);
        let t = prompt.name();
        self.probe_wide = matches!(t, "SELECT_YESNO" | "SELECT_OPTION")
            || (CARD_PROMPTS.contains(&t) && !cards.is_empty() && cards.iter().all(|c| matches!(c.loc, LOCATION_MZONE | LOCATION_SZONE) && c.pos.unwrap_or(POS_FACEUP) & POS_FACEUP != 0));
        let mut best: Option<(usize, bool, Response)> = None;
        let mut unrefuted: Option<Response> = None;
        let mut failures: Vec<usize> = Vec::new();
        for (k, c) in valid.iter().take(self.max_probes).enumerate() {
            if k >= n_inferred && (best.is_some() || unrefuted.is_some()) {
                break;
            }
            if self.depth == 0 {
                self.probe_counter.set(0);
            }
            self.stats.probes += 1;
            let outcome = self.probe(duel, prompt, c, ProbeSetup::None, None);
            let reached = self.last_probe_reached;
            if outcome == Probe::Ok {
                self.trace(prompt, c, false, valid.len());
                self.plan = std::mem::take(&mut self.last_probe_plan);
                if k == 0 {
                    let window = if self.probe_wide { self.probe_items } else { self.probe_state_window };
                    self.choice_seen = Some(reached.max(self.pos + window.max(40)) + 1);
                }
                return Ok(c.clone());
            }
            if matches!(outcome, Probe::Repairable | Probe::RepairableAtOnce | Probe::Progressed) {
                let key = (reached, outcome != Probe::Progressed);
                if best.as_ref().is_none_or(|b| key > (b.0, b.1)) {
                    best = Some((key.0, key.1, c.clone()));
                }
            }
            if outcome == Probe::Timeout && unrefuted.is_none() {
                unrefuted = Some(c.clone());
            }
            if outcome == Probe::Failed {
                failures.push(reached);
            }
        }
        let mut choice = best.map(|b| b.2).or(unrefuted);
        if choice.is_none() && certain == Certain::Likely {
            choice = Some(valid[0].clone());
        }
        if choice.is_none() && self.depth == 0 && failures.len() == valid.len() && valid.len() > 1 && failures.iter().all(|&f| f == failures[0]) && failures[0] >= self.pos {
            choice = Some(valid[0].clone());
        }
        if let Some(c) = choice {
            self.trace(prompt, &c, false, valid.len());
            return Ok(c);
        }
        let mut fix = None;
        if t == "SELECT_OPTION"
            && !self.upcoming().iter().take(12).any(|m| matches!(m, Msg::Hint { hint_type, player, .. } if *hint_type == HINT_OPSELECTED && *player == self.opponent))
        {
            let start = self.chain_start();
            if let Some(span) = self.nomatch_span(start) {
                fix = Some(Fix::NomatchPile(span.0, span.1));
            }
        }
        if fix.is_none()
            && CARD_PROMPTS.contains(&t)
            && !cards.is_empty()
            && cards.iter().all(|c| c.con == self.opponent && (c.loc & (LOCATION_DECK | LOCATION_HAND | LOCATION_EXTRA) != 0 || c.pos.unwrap_or(POS_FACEUP) & POS_FACEUP == 0))
        {
            let start = self.chain_start();
            if let Some(span) = self.nomatch_span(start) {
                fix = Some(Fix::NomatchPile(span.0, span.1));
            }
        }
        let mut d = Divergence { got: Some(DV::Msg(prompt.clone())), fix, ..Divergence::new(self.pos, format!("no candidate for opponent {t} survived probing")) };
        if d.fix.is_none() {
            d.fallback = self.hidden_pick_fallback().0;
        }
        div(d)
    }

    fn trace(&mut self, _prompt: &Msg, _response: &Response, _certain: bool, _n: usize) {}

    /// Checks a candidate response on copies of the replay and the duel (the Python module forks).
    pub fn probe(&mut self, duel: &RDuel, prompt: &Msg, response: &Response, setup: ProbeSetup, horizon: Option<usize>) -> Probe {
        if self.depth > 0 {
            self.probe_counter.set(self.probe_counter.get() + 1);
        }
        let mut c = self.clone();
        let mut d = duel.clone();
        // (nested probes get less time, so they cannot eat their parent's time)
        let secs = 10u64.max((self.probe_timeout * 6) >> self.depth);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
        c.probe_deadline = Some(match self.probe_deadline {
            Some(p) => p.min(deadline),
            None => deadline,
        });
        c.depth += 1;
        c.probe_steps = 0;
        c.report_depth = c.depth as i64;
        c.probe_decisions.clear();
        c.plan.clear();
        c.check_state = false;
        match setup {
            ProbeSetup::None => {}
            ProbeSetup::NewCard(con, loc, seq, code) => {
                c.skip_shuffle_hand = Some(con);
                let before: BTreeSet<u32> = d.location_cards(con, loc).into_iter().map(|x| x.1).collect();
                d.new_card(con as u8, code, loc);
                let hand = d.location_cards(con, loc);
                let new: Vec<u32> = hand.iter().map(|x| x.1).filter(|cid| !before.contains(cid)).collect();
                if !new.is_empty() && (seq as usize) < hand.len() {
                    d.swap_cards(new[0], hand[seq as usize].1);
                }
            }
            ProbeSetup::Swap(a, b) => {
                d.swap_cards(a, b);
            }
            ProbeSetup::Recreate(a, b) => d.recreate_card(a, b),
        }
        c.max_stall = c.max_stall.min(50);
        c.probe_start = c.pos;
        c.probe_boundary = false;
        c.probe_after_boundary = 0;
        c.probe_sig = Some((c.pos, prompt_key(prompt)));
        c.horizon = c.items.len().min(c.pos + c.probe_items);
        if let Some(h) = horizon {
            c.horizon = c.items.len().min(h);
            c.test_until = Some(c.horizon);
        }
        let window = if c.probe_wide { c.probe_items } else { c.probe_state_window };
        let hi = c.horizon.min(c.pos + window + 1);
        c.probe_state_at = (c.pos..hi).find(|&j| matches!(c.items[j], ObsItem::Prompt(_, Some(_))));
        let result = c.run_loop(&mut d, Some(prompt.clone()), Some(response.clone()), None);
        let (code, reached, sig) = match result {
            Ok(()) | Err(RErr::ProbeSuccess) => (Probe::Ok, c.pos, None),
            Err(RErr::Div(e)) => {
                let code = if e.fix.is_some() {
                    if e.pos > c.probe_start { Probe::Repairable } else { Probe::RepairableAtOnce }
                } else if e.pos - c.probe_start.min(e.pos) >= c.min_progress && e.pos >= c.probe_start {
                    Probe::Progressed
                } else {
                    Probe::Failed
                };
                if std::env::var_os("YGOAI_PROBE_DEBUG").is_some() {
                    eprintln!("   probe {} @{}: {} fix={:?}", serde_json::to_string(response).unwrap_or_default(), c.probe_start, e.text(), e.fix);
                }
                (code, e.pos, Some(divergence_sig(&e)))
            }
            Err(RErr::ProbeTimeout) => (Probe::Timeout, self.pos, None),
            Err(e) => {
                if std::env::var_os("YGOAI_PROBE_DEBUG").is_some() {
                    eprintln!("   probe {} @{}: {e}", serde_json::to_string(response).unwrap_or_default(), c.probe_start);
                }
                (Probe::Failed, self.pos, None)
            }
        };
        self.last_probe_reached = reached;
        self.last_probe_plan = Vec::new();
        self.last_probe_sig = None;
        if code == Probe::Ok {
            // (the answers along the way, with enough verified after them; up to a yes/no or option
            // past the state view the probe checked)
            let mut plan = Vec::new();
            for dd in &c.probe_decisions {
                if matches!(dd.0.1, "SELECT_YESNO" | "SELECT_OPTION") && c.probe_state_at.is_none_or(|s| dd.0.0 >= s) {
                    break;
                }
                if dd.0.0 + c.min_progress <= reached {
                    plan.push(dd.clone());
                }
            }
            plan.truncate(200);
            self.last_probe_plan = plan;
        } else if horizon.is_some() {
            self.last_probe_sig = sig;
        }
        if self.depth == 0 {
            let s = format!("probe @{} {} {} -> exit {}", self.pos, prompt.name(), serde_json::to_string(response).unwrap_or_default(), code.code());
            self.log(s);
        }
        code
    }
}

impl Probe {
    /// The forked probe's exit code in the Python module (for logs).
    pub fn code(self) -> i32 {
        match self {
            Probe::Ok => 0,
            Probe::Failed => 1,
            Probe::Timeout => 3,
            Probe::Repairable => 4,
            Probe::Progressed => 5,
            Probe::RepairableAtOnce => 6,
        }
    }
}
