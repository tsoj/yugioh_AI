//! The RL policy: the policy/value network of the self-play (reinforcement learning) runs
//! (runs/nn/rl*, boot1) and the agent that plays whole games with it, without a search (the opening
//! book's test games, ygoai-book-test). Not the alpha-beta search's move-ordering policies
//! (search::cardpolicy). A port of the Python originals (ygoai/nn/model.py PolicyValueNet, its
//! features ygoai/nn/features.py, scripts/train_policy_value.py, agents/policy_agent.py
//! PolicyAgent, scripts/make_match_book.py; removed with the Python play stack, see the git
//! history before it): it made the Python agent's choices with the same seeds, up to float
//! rounding (numpy sums in another order: a sampled choice can differ right at a boundary).
//!
//! The network: each card [static features | dynamic features] -> MLP -> embedding, summed per
//! group (decks by count) -> [groups x dim | scalars] -> trunk MLP -> h; a candidate's score is
//! q(h) . (the embeddings of its parts + a linear map of its card's features). The static parts
//! of the first layers are computed once per card of the model's table.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use ygo_core::constants::*;
use ygo_core::types::CardData;

use crate::actions::{Response, battle_options, idle_options};
use crate::belief::completion::npy_array;
use crate::game::{Agent, AgentError, InitInfo};
use crate::heuristic::{Heuristic, card};
use crate::messages::{Card, Msg};
use crate::pyrandom::PyRandom;
use crate::state::{CardQuery, Snapshot};

// -- cards: what they are (features.static_features) --------------------------------------------

const TYPE_BITS: usize = 27;
const ATTRIBUTE_BITS: usize = 7;
const RACE_BITS: usize = 26;
const LEVELS: usize = 14;
const ATK_BUCKETS: [i32; 6] = [0, 1000, 1500, 2000, 2500, 3000];
pub const STATIC_DIM: usize = TYPE_BITS + ATTRIBUTE_BITS + RACE_BITS + LEVELS + 2 * (ATK_BUCKETS.len() + 1) + 2 + 1;

fn bucket(out: &mut Vec<f32>, value: Option<i32>) {
    let mut b = [0.0f32; ATK_BUCKETS.len() + 1];
    match value {
        Some(v) if v >= 0 => b[ATK_BUCKETS.iter().rposition(|&x| v >= x).unwrap()] = 1.0,
        _ => b[ATK_BUCKETS.len()] = 1.0,
    }
    out.extend(b);
}

/// What a card is (zeros but the last for an unknown card).
pub fn static_features(c: Option<&CardData>) -> Vec<f32> {
    let Some(c) = c else {
        let mut v = vec![0.0; STATIC_DIM];
        v[STATIC_DIM - 1] = 1.0;
        return v;
    };
    let bits = |out: &mut Vec<f32>, value: u64, n: usize| out.extend((0..n).map(|i| (value >> i & 1) as f32));
    let monster = c.type_ & TYPE_MONSTER != 0;
    let link = c.type_ & TYPE_LINK != 0;
    let level = if link { c.link_marker.count_ones() } else { c.level } as usize;
    let mut v = Vec::with_capacity(STATIC_DIM);
    bits(&mut v, c.type_ as u64, TYPE_BITS);
    bits(&mut v, c.attribute as u64, ATTRIBUTE_BITS);
    bits(&mut v, c.race, RACE_BITS);
    let mut lv = [0.0f32; LEVELS];
    if monster {
        lv[level.min(LEVELS - 1)] = 1.0;
    }
    v.extend(lv);
    let atk = monster.then_some(c.attack);
    let def = (monster && !link).then_some(c.defense);
    bucket(&mut v, atk);
    bucket(&mut v, def);
    v.push(atk.unwrap_or(0).max(0) as f32 / 5000.0);
    v.push(def.unwrap_or(0).max(0) as f32 / 5000.0);
    v.push(0.0);
    v
}

// -- cards: how and where they are ---------------------------------------------------------------

const LOCATIONS: [u32; 8] =
    [LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA, LOCATION_OVERLAY];
const POSITIONS: [u32; 4] = [POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE];
pub const DYN_DIM: usize = POSITIONS.len() + 6;
const CARD_DIM: usize = STATIC_DIM + DYN_DIM;
/// (position index + 1 (0: none), ATK, DEF, overlays, counters, flags (1 disabled, 2 public))
type Raw = [i64; 6];

/// The piles of a player in the groups' order ("own_hand", ..., "own_deck", "opp_hand", ...,
/// "chain"); the Extra Deck and the deck are one row per distinct card, with its count.
const PILES: [u32; 7] = [LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA, LOCATION_DECK];
const GROUPS: usize = 2 * PILES.len() + 1;
const CHAIN_GROUP: usize = 2 * PILES.len();

fn pos_index(pos: u32) -> i64 {
    POSITIONS.iter().position(|&p| p == pos & 0xF).map_or(0, |i| i as i64 + 1)
}

/// How a card of a capture is now (features.dynamic_raw).
fn raw_of_query(c: &CardQuery) -> Raw {
    let counters = c.counters.as_deref().unwrap_or(&[]);
    [
        pos_index(c.position.unwrap_or(0)),
        c.attack.unwrap_or(0).max(0) as i64,
        c.defense.unwrap_or(0).max(0) as i64,
        c.overlays.as_ref().map_or(0, |o| o.len()) as i64,
        counters.iter().skip(1).step_by(2).map(|&n| n as i64).sum(),
        (c.status.unwrap_or(0) & STATUS_DISABLED != 0) as i64 | 2 * (c.is_public.unwrap_or(0) != 0) as i64,
    ]
}

/// A prompt's entry: only its position is known.
fn raw_of_entry(pos: Option<u32>) -> Raw {
    [pos_index(pos.unwrap_or(0)), 0, 0, 0, 0, 0]
}

fn expand(raw: &Raw, out: &mut [f32]) {
    out[..POSITIONS.len()].fill(0.0);
    if raw[0] > 0 {
        out[raw[0] as usize - 1] = 1.0;
    }
    let n = POSITIONS.len();
    out[n] = raw[1] as f32 / 5000.0;
    out[n + 1] = raw[2] as f32 / 5000.0;
    out[n + 2] = raw[3] as f32 / 5.0;
    out[n + 3] = raw[4] as f32 / 5.0;
    out[n + 4] = (raw[5] & 1) as f32;
    out[n + 5] = (raw[5] >> 1 & 1) as f32;
}

// -- positions ---------------------------------------------------------------------------------

const PHASES: [u32; 10] = [
    PHASE_DRAW,
    PHASE_STANDBY,
    PHASE_MAIN1,
    PHASE_BATTLE_START,
    PHASE_BATTLE_STEP,
    PHASE_DAMAGE,
    PHASE_DAMAGE_CAL,
    PHASE_BATTLE,
    PHASE_MAIN2,
    PHASE_END,
];
const SCALAR_DIM: usize = 2 + PHASES.len() + 2 + 2 * 4 + 1;

/// The cards (code, group, count, dynamic raw) and the scalars of a view, for `me`
/// (features.position).
fn position(snap: &Snapshot, me: u32) -> (Vec<(u32, usize, f32, Raw)>, Vec<f32>) {
    let mut cards = Vec::new();
    for (side, p) in [me, 1 - me].into_iter().enumerate() {
        let pl = &snap.players[p as usize];
        for (k, &loc) in PILES.iter().enumerate() {
            let g = side * PILES.len() + k;
            let pile = pl.pile(loc);
            if loc == LOCATION_EXTRA || loc == LOCATION_DECK {
                let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
                for c in pile.iter().flatten() {
                    *counts.entry(c.code.unwrap_or(0)).or_default() += 1;
                }
                cards.extend(counts.into_iter().map(|(code, n)| (code, g, n as f32, [0; 6])));
            } else {
                cards.extend(pile.iter().flatten().map(|c| (c.code.unwrap_or(0), g, 1.0, raw_of_query(c))));
            }
        }
    }
    cards.extend(snap.chain.iter().map(|l| (l.code, CHAIN_GROUP, 1.0, [0; 6])));
    let lp = snap.lp;
    let mut s = vec![lp[me as usize] as f32 / 8000.0, lp[1 - me as usize] as f32 / 8000.0];
    s.extend(PHASES.iter().map(|&ph| (snap.phase == ph) as u32 as f32));
    s.push((snap.turn_player == me) as u32 as f32);
    s.push(snap.turn.min(40) as f32 / 20.0);
    for p in [me, 1 - me] {
        let pl = &snap.players[p as usize];
        s.push(pl.deck.len() as f32 / 40.0);
        s.push(pl.hand.len() as f32 / 10.0);
        s.push(pl.extra.len() as f32 / 15.0);
        s.push(pl.mzone.iter().flatten().count() as f32 / 5.0);
    }
    s.push(snap.chain.len() as f32 / 5.0);
    (cards, s)
}

// -- candidates ----------------------------------------------------------------------------------

const PROMPTS: [&str; 7] = ["SELECT_IDLECMD", "SELECT_BATTLECMD", "SELECT_CHAIN", "SELECT_EFFECTYN", "SELECT_YESNO", "SELECT_OPTION", "SELECT_CARD"];
const KINDS: [&str; 17] = [
    "summon", "spsummon", "repos", "mset", "sset", "activate", "to_bp", "to_ep", "shuffle", "attack", "to_m2", "chain", "pass", "yes", "no", "option",
    "select",
];
const HINTS: usize = 81; // (select hints 500..579 and "none")
const EFFECTS: usize = 18; // (effect index 0..15 of a card, a system string, none)

/// The candidates the policy chooses from (belief_mc.candidates): none for other prompts.
pub fn candidates(p: &Msg) -> Vec<Response> {
    match p {
        Msg::SelectIdlecmd { .. } => idle_options(p),
        Msg::SelectBattlecmd { .. } => battle_options(p),
        Msg::SelectChain { chains, forced, .. } => {
            let chain = (0..chains.len()).map(Response::index);
            if *forced != 0 { chain.collect() } else { std::iter::once(Response::pass()).chain(chain).collect() }
        }
        Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => vec![Response::yes(true), Response::yes(false)],
        Msg::SelectOption { options, .. } => (0..options.len()).map(Response::index).collect(),
        Msg::SelectCard { min: 1, max: 1, cards, .. } => (0..cards.len()).map(|i| Response::cards(vec![i])).collect(),
        _ => vec![],
    }
}

/// A candidate's card: where it is (code, controller, location, sequence, position).
struct Entry {
    code: u32,
    con: u32,
    loc: u32,
    seq: u32,
    pos: Option<u32>,
}

impl Entry {
    fn of(c: &Card) -> Entry {
        Entry { code: c.code, con: c.con, loc: c.loc, seq: c.seq, pos: c.pos }
    }
}

/// A candidate's features (features.actions): decision type, kind, select hint, effect index,
/// card code (0: none), card location index (-1: none), card is own, dynamic raw.
struct Act {
    cat: [usize; 4],
    code: u32,
    loc: i64,
    own: bool,
    raw: Raw,
}

fn effect(desc: Option<u64>) -> usize {
    match desc {
        None => EFFECTS - 1,
        Some(d) if d >> 20 != 0 => (d & 0xF) as usize,
        Some(_) => EFFECTS - 2,
    }
}

fn actions(prompt: &Msg, cands: &[Response], snap: &Snapshot, me: u32, hint: Option<u64>) -> Vec<Act> {
    let pt = PROMPTS.iter().position(|&t| t == prompt.name()).expect("a policy prompt");
    let h = match hint {
        Some(x) if (500..500 + HINTS as u64 - 1).contains(&x) => (x - 500) as usize,
        _ => HINTS - 1,
    };
    cands
        .iter()
        .map(|c| {
            let (kind, entry, desc): (&str, Option<Entry>, Option<u64>) = match prompt {
                Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. } => {
                    let command = c.kind.unwrap();
                    let e = command.cards(prompt).map(|l| &l[c.index.unwrap() as usize]);
                    (command.name(), e.map(Entry::of), e.and_then(|e| e.desc))
                }
                Msg::SelectChain { chains, .. } => match c.index.unwrap() {
                    i if i < 0 => ("pass", None, None),
                    i => {
                        let e = &chains[i as usize];
                        ("chain", Some(Entry::of(e)), e.desc)
                    }
                },
                Msg::SelectEffectyn { code, con, loc, seq, pos, desc, .. } => {
                    let kind = if c.yes == Some(true) { "yes" } else { "no" };
                    (kind, Some(Entry { code: *code, con: *con, loc: *loc, seq: *seq, pos: Some(*pos) }), Some(*desc))
                }
                Msg::SelectYesno { desc, .. } => (if c.yes == Some(true) { "yes" } else { "no" }, None, Some(*desc)),
                Msg::SelectOption { options, .. } => ("option", None, Some(options[c.index.unwrap() as usize])),
                Msg::SelectCard { cards, .. } => ("select", Some(Entry::of(&cards[c.indices.as_ref().unwrap()[0] as usize])), None),
                _ => unreachable!("not a policy prompt"),
            };
            let code = match &entry {
                Some(e) => e.code,
                None => desc.map_or(0, |d| (d >> 20) as u32),
            };
            let (loc, own, raw) = match &entry {
                None => (-1, false, [0; 6]),
                Some(e) => {
                    let loc = e.loc & !LOCATION_OVERLAY;
                    // (the view's card there, else what the prompt says)
                    let here = crate::state::LOCATIONS
                        .iter()
                        .any(|&(_, l)| l == loc)
                        .then(|| snap.players[e.con as usize].pile(loc).get(e.seq as usize).and_then(|c| c.as_ref()))
                        .flatten();
                    let raw = here.map_or_else(|| raw_of_entry(e.pos), raw_of_query);
                    (LOCATIONS.iter().position(|&l| l == loc).map_or(-1, |i| i as i64), e.con == me, raw)
                }
            };
            let kind = KINDS.iter().position(|&k| k == kind).unwrap();
            let hint = if pt == 6 { h } else { HINTS - 1 };
            Act { cat: [pt, kind, hint, effect(desc)], code, loc, own, raw }
        })
        .collect()
}

// -- the network ---------------------------------------------------------------------------------

/// A linear layer's weight (out x in, row-major, as torch stores it).
struct Mat {
    rows: usize,
    cols: usize,
    w: Vec<f32>,
}

impl Mat {
    fn row(&self, i: usize) -> &[f32] {
        &self.w[i * self.cols..(i + 1) * self.cols]
    }
    /// out[i] += w[i, cols] . x (the columns from `from`)
    fn mul_add(&self, x: &[f32], from: usize, out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate().take(self.rows) {
            *o += dot(&self.row(i)[from..from + x.len()], x);
        }
    }
}

/// (in 8 lanes, which the compiler vectorizes: a sequential float sum cannot be reordered)
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let tail: f32 = a.chunks_exact(8).remainder().iter().zip(b.chunks_exact(8).remainder()).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

fn relu(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = x.max(0.0));
}

/// What the network keeps per card of its table (or per card it met outside it).
struct CardRow {
    /// the first card layer's static part plus its bias
    h1: Vec<f32>,
    /// the embedding with no dynamic features (deck and Extra Deck rows, chain links)
    still: Vec<f32>,
    /// the candidates' card map of the static features
    act: Vec<f32>,
}

pub struct RlPolicyNet {
    index: HashMap<u32, usize>,
    rows: Vec<CardRow>,
    card0: Mat,
    card0_b: Vec<f32>,
    card2: Mat,
    card2_b: Vec<f32>,
    trunk0: Mat,
    trunk0_b: Vec<f32>,
    trunk2: Mat,
    trunk2_b: Vec<f32>,
    query: Mat,
    query_b: Vec<f32>,
    act_cat: [Mat; 4],
    act_loc: Mat,
    act_own: Mat,
    act_card: Mat,
}

fn shape(header: &str) -> Vec<usize> {
    let s = &header[header.find("'shape': (").expect("an npy shape") + 10..];
    s[..s.find(')').unwrap()].split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

impl RlPolicyNet {
    /// A network exported by scripts/train_policy_value.py (an .npz: weights by name, the codes of
    /// the static table).
    pub fn load(path: &Path) -> Result<RlPolicyNet, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut z = zip::ZipArchive::new(f).map_err(|e| e.to_string())?;
        let mut mat = |name: &str| -> Result<Mat, String> {
            let (header, data) = npy_array(&mut z, &format!("{name}.npy"))?;
            if !header.contains("'<f4'") {
                return Err(format!("{name}: not float32"));
            }
            let s = shape(&header);
            let (rows, cols) = match s[..] {
                [n] => (1, n),
                [r, c] => (r, c),
                _ => return Err(format!("{name}: shape {s:?}")),
            };
            let w = data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            Ok(Mat { rows, cols, w })
        };
        let table = mat("static_table")?;
        let (card0, card0_b, card2, card2_b) = (mat("card.0.weight")?, mat("card.0.bias")?.w, mat("card.2.weight")?, mat("card.2.bias")?.w);
        let (trunk0, trunk0_b, trunk2, trunk2_b) = (mat("trunk.0.weight")?, mat("trunk.0.bias")?.w, mat("trunk.2.weight")?, mat("trunk.2.bias")?.w);
        let (query, query_b) = (mat("query.weight")?, mat("query.bias")?.w);
        let act_cat = [mat("act_cat.0.weight")?, mat("act_cat.1.weight")?, mat("act_cat.2.weight")?, mat("act_cat.3.weight")?];
        let (act_loc, act_own, act_card) = (mat("act_loc.weight")?, mat("act_own.weight")?, mat("act_card.weight")?);
        let (header, data) = npy_array(&mut z, "codes.npy")?;
        if !header.contains("'<i8'") {
            return Err("codes: not int64".into());
        }
        let codes: Vec<u32> = data.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as u32).collect();
        if table.cols != STATIC_DIM || card0.cols != CARD_DIM || table.rows != codes.len() || trunk0.cols != GROUPS * card2.rows + SCALAR_DIM {
            return Err(format!("{}: unexpected shapes", path.display()));
        }
        let mut net = RlPolicyNet {
            index: codes.iter().enumerate().map(|(i, &c)| (c, i)).collect(),
            rows: Vec::new(),
            card0,
            card0_b,
            card2,
            card2_b,
            trunk0,
            trunk0_b,
            trunk2,
            trunk2_b,
            query,
            query_b,
            act_cat,
            act_loc,
            act_own,
            act_card,
        };
        net.rows = (0..table.rows).map(|i| net.card_row(table.row(i))).collect();
        Ok(net)
    }

    fn card_row(&self, stat: &[f32]) -> CardRow {
        let mut h1 = self.card0_b.clone();
        self.card0.mul_add(stat, 0, &mut h1);
        let mut zero = [0.0f32; DYN_DIM];
        expand(&[0; 6], &mut zero);
        let still = self.embed(&h1, &zero);
        let mut act = vec![0.0; self.act_card.rows];
        self.act_card.mul_add(stat, 0, &mut act);
        CardRow { h1, still, act }
    }

    /// A card's row: from the table, else from the card database (python: PolicyValueNet._static).
    fn row(&self, code: u32) -> std::borrow::Cow<'_, CardRow> {
        match self.index.get(&code) {
            Some(&i) => std::borrow::Cow::Borrowed(&self.rows[i]),
            None => std::borrow::Cow::Owned(self.card_row(&static_features(card(code)))),
        }
    }

    /// A card's embedding from its row's h1 and its dynamic features.
    fn embed(&self, h1: &[f32], dynamic: &[f32]) -> Vec<f32> {
        let mut h = h1.to_vec();
        self.card0.mul_add(dynamic, STATIC_DIM, &mut h);
        relu(&mut h);
        let mut e = self.card2_b.clone();
        self.card2.mul_add(&h, 0, &mut e);
        e
    }

    /// The trunk's output h for a position (cards and scalars as `position` gives them).
    fn trunk(&self, cards: &[(u32, usize, f32, Raw)], scalars: &[f32]) -> Vec<f32> {
        let dim = self.card2_b.len();
        let mut x = vec![0.0f32; GROUPS * dim + scalars.len()];
        let mut dynamic = [0.0f32; DYN_DIM];
        for (code, g, n, raw) in cards {
            let row = self.row(*code);
            let e = if *raw == [0; 6] {
                std::borrow::Cow::Borrowed(&row.still)
            } else {
                expand(raw, &mut dynamic);
                std::borrow::Cow::Owned(self.embed(&row.h1, &dynamic))
            };
            for (o, v) in x[g * dim..(g + 1) * dim].iter_mut().zip(e.iter()) {
                *o += v * n;
            }
        }
        x[GROUPS * dim..].copy_from_slice(scalars);
        let mut h = self.trunk0_b.clone();
        self.trunk0.mul_add(&x, 0, &mut h);
        relu(&mut h);
        let mut out = self.trunk2_b.clone();
        self.trunk2.mul_add(&h, 0, &mut out);
        relu(&mut out);
        out
    }

    /// The candidates' scores.
    fn scores(&self, h: &[f32], acts: &[Act]) -> Vec<f32> {
        let mut q = self.query_b.clone();
        self.query.mul_add(h, 0, &mut q);
        let mut dynamic = [0.0f32; DYN_DIM];
        acts.iter()
            .map(|a| {
                let mut e: Vec<f32> = (0..q.len()).map(|j| (0..4).map(|i| self.act_cat[i].row(a.cat[i])[j]).sum::<f32>()).collect();
                let (loc, own) = (self.act_loc.row((a.loc + 1) as usize), self.act_own.row(a.own as usize));
                for (j, v) in e.iter_mut().enumerate() {
                    *v += loc[j] + own[j];
                }
                if a.code != 0 {
                    let row = self.row(a.code);
                    expand(&a.raw, &mut dynamic);
                    let mut c = row.act.clone();
                    self.act_card.mul_add(&dynamic, STATIC_DIM, &mut c);
                    for (v, x) in e.iter_mut().zip(c) {
                        *v += x;
                    }
                }
                dot(&e, &q)
            })
            .collect()
    }
}

impl Clone for CardRow {
    fn clone(&self) -> Self {
        CardRow { h1: self.h1.clone(), still: self.still.clone(), act: self.act.clone() }
    }
}

// -- the agent ----------------------------------------------------------------------------------

/// The RL policy's player (agents/policy_agent.py PolicyAgent): the network's choice at the decisions it
/// has candidates for (sampled at `temperature`, the best one at 0), the heuristic player's
/// elsewhere; the heuristic player's cap on repeated activations holds for both.
pub struct RlPolicyAgent {
    net: Arc<RlPolicyNet>,
    rng: PyRandom,
    temperature: f64,
    fallback: Heuristic,
    player: u32,
    hint: Option<u64>,
}

impl RlPolicyAgent {
    pub fn new(seed: i128, net: Arc<RlPolicyNet>, temperature: f64) -> RlPolicyAgent {
        RlPolicyAgent { net, rng: PyRandom::new(seed), temperature, fallback: Heuristic::new(seed, 0), player: 0, hint: None }
    }

    fn choose(&mut self, prompt: &Msg, cands: &[Response], state: &Snapshot, hint: Option<u64>) -> usize {
        let (cards, scalars) = position(state, self.player);
        let acts = actions(prompt, cands, state, self.player, hint);
        let h = self.net.trunk(&cards, &scalars);
        let s: Vec<f64> = self.net.scores(&h, &acts).into_iter().map(f64::from).collect();
        if self.temperature <= 0.0 {
            // (the first best one)
            return (0..s.len()).fold(0, |k, i| if s[i] > s[k] { i } else { k });
        }
        let top = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // (random.choices(range(n), weights): the cumulative weights, bisect_right)
        let mut cum = Vec::with_capacity(s.len());
        let mut total = 0.0;
        for x in &s {
            total += ((x - top) / self.temperature).exp();
            cum.push(total);
        }
        let r = self.rng.random() * total;
        cum[..cum.len() - 1].partition_point(|&c| c <= r)
    }
}

impl Agent for RlPolicyAgent {
    fn name(&self) -> String {
        "policy".into()
    }
    fn wants_state(&self) -> bool {
        true
    }
    fn start(&mut self, player: u32, _init: &InitInfo) -> Result<(), AgentError> {
        self.player = player;
        self.fallback.player = player;
        Ok(())
    }
    fn observe(&mut self, msg: &Msg) {
        self.fallback.observe_msg(msg);
        if let Msg::Hint { hint_type, player, data } = msg
            && *hint_type == HINT_SELECTMSG
            && *player == self.player
        {
            self.hint = Some(*data);
        }
    }
    fn respond(&mut self, prompt: &Msg, state: Option<&Snapshot>, _clock: Option<(f64, f64)>) -> Result<Response, AgentError> {
        let hint = self.hint.take();
        let cands: Vec<Response> = candidates(prompt).into_iter().filter(|c| self.fallback.within_cap(prompt, c)).collect();
        let choice = match (state, cands.len()) {
            (_, 0) | (None, 2..) => return self.fallback.respond_to(prompt, state),
            (_, 1) => cands[0].clone(),
            (Some(s), _) => cands[self.choose(prompt, &cands, s, hint)].clone(),
        };
        self.fallback.hint = None; // (this selection is answered here)
        self.fallback.note_choice(prompt, &choice);
        Ok(choice)
    }
}
