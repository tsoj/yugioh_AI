//! The evaluation: the hand-written one (`YGOAI_EVAL=hand`), the trained ones (`LearnedEval`, the
//! built-in default) and the features they and the training data use.

use ygo_core::api::QueryInfo;
use ygo_core::constants::*;
use ygo_core::duel::Duel;
use ygo_core::types::CardId;

use crate::messages::Msg;
use crate::state::parse_location_query;

// evaluation weights (points; 1 point = 100 LP)
pub const W_LP: f64 = 1.0 / 100.0;
pub const W_MONSTER: f64 = 4.3;
/// (a face-up monster stronger than everything on the other side)
pub const W_DOMINANT: f64 = 17.6;
pub const W_HAND: f64 = 9.1;
pub const W_BACKROW: f64 = 5.1;
/// (two cards or fewer left to draw)
pub const W_DECK_OUT: f64 = 20.0;

const MZONE_QUERY: u32 = QUERY_POSITION | QUERY_ATTACK | QUERY_DEFENSE;

fn battle_stat(position: u32, attack: i32, defense: i32) -> i32 {
    if position & POS_DEFENSE != 0 { defense } else { attack }
}

fn query(d: &mut Duel, p: u32, loc: u32, flags: u32) -> Vec<u8> {
    d.ocg_query_location(&QueryInfo { flags, con: p as u8, loc, seq: 0, overlay_seq: 0 })
}

/// The evaluation's terms for `me` (the differences, me minus the opponent): LP, monsters,
/// dominant monsters, hand, face-down backrow, and the deck-out flag (2 cards or fewer left to
/// draw; the opponent's minus mine). `evaluate` is their sum with the W_ weights (in its own
/// order of additions).
pub fn eval_features(d: &mut Duel, me: u32) -> [f64; 6] {
    let info = d.ext_get_info();
    let lp = [info[3] as i32, info[4] as i32];
    let mons: Vec<Vec<(u32, i32, i32)>> = [me, 1 - me]
        .iter()
        .map(|&p| {
            parse_location_query(&query(d, p, LOCATION_MZONE, MZONE_QUERY))
                .into_iter()
                .flatten()
                .map(|c| (c.position.unwrap_or(0), c.attack.unwrap_or(0), c.defense.unwrap_or(0)))
                .collect()
        })
        .collect();
    let mut f = [(lp[me as usize] - lp[1 - me as usize]) as f64, mons[0].len() as f64 - mons[1].len() as f64, 0.0, 0.0, 0.0, 0.0];
    for (s, p) in [(0usize, me), (1, 1 - me)] {
        let sign = if s == 0 { 1.0 } else { -1.0 };
        let best_other = mons[1 - s].iter().map(|&(pos, a, df)| battle_stat(pos, a, df)).max().unwrap_or(-1);
        f[2] += sign * mons[s].iter().filter(|&&(pos, a, _)| pos & POS_FACEUP_ATTACK != 0 && a > best_other).count() as f64;
        f[3] += sign * d.ocg_query_count(p as u8, LOCATION_HAND) as f64;
        let backrow = parse_location_query(&query(d, p, LOCATION_SZONE, QUERY_POSITION))
            .into_iter()
            .flatten()
            .filter(|c| c.position.unwrap_or(0) & POS_FACEDOWN != 0)
            .count();
        f[4] += sign * backrow as f64;
        if d.ocg_query_count(p as u8, LOCATION_DECK) <= 2 {
            f[5] -= sign;
        }
    }
    f
}

/// The evaluation's weights, in the order of `eval_features`.
pub const EVAL_WEIGHTS: [f64; 6] = [W_LP, W_MONSTER, W_DOMINANT, W_HAND, W_BACKROW, W_DECK_OUT];

/// One player's side for the rich features (scripts/train_eval.py side_features): LP/1000, hand,
/// deck/10, grave/10, banished/10, extra/10, monsters, face-up ones, face-up attackers, face-down
/// ones, the attackers' ATK sum/1000, the best face-up battle stat/1000, face-down and face-up
/// spells/traps, a field spell, deck out (2 cards or fewer).
fn side_features(d: &mut Duel, p: u32) -> [f64; 16] {
    let mons: Vec<_> = parse_location_query(&query(d, p, LOCATION_MZONE, MZONE_QUERY)).into_iter().flatten().collect();
    let (mut up, mut atk_up, mut atk_sum, mut best) = (0, 0, 0i64, 0i32);
    for c in &mons {
        let pos = c.position.unwrap_or(0);
        if pos & POS_FACEUP != 0 {
            up += 1;
            let attacking = pos & POS_ATTACK != 0;
            best = best.max(if attacking { c.attack.unwrap_or(0) } else { c.defense.unwrap_or(0) });
            if attacking {
                atk_up += 1;
                atk_sum += c.attack.unwrap_or(0) as i64;
            }
        }
    }
    let sz = parse_location_query(&query(d, p, LOCATION_SZONE, QUERY_POSITION));
    let back = sz.iter().take(5).flatten();
    let down = back.clone().filter(|c| c.position.unwrap_or(0) & POS_FACEDOWN != 0).count();
    let n_back = back.count();
    let field = sz.get(5).is_some_and(|c| c.is_some());
    let lp = d.ext_get_info()[3 + p as usize] as i32;
    let n = |d: &mut Duel, loc: u32| d.ocg_query_count(p as u8, loc) as f64;
    let deck = n(d, LOCATION_DECK);
    [
        lp as f64 / 1000.0,
        n(d, LOCATION_HAND),
        deck / 10.0,
        n(d, LOCATION_GRAVE) / 10.0,
        n(d, LOCATION_REMOVED) / 10.0,
        n(d, LOCATION_EXTRA) / 10.0,
        mons.len() as f64,
        up as f64,
        atk_up as f64,
        (mons.len() - up) as f64,
        atk_sum as f64 / 1000.0,
        best as f64 / 1000.0,
        down as f64,
        (n_back - down) as f64,
        field as u8 as f64,
        (deck <= 2.0) as u8 as f64,
    ]
}

/// The rich features for `me` (scripts/train_eval.py rich_features): the sides' differences, mine
/// minus the opponent's, then the evaluation's dominance term.
pub fn rich_features(d: &mut Duel, me: u32) -> Vec<f64> {
    let (a, b) = (side_features(d, me), side_features(d, 1 - me));
    let mut f: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    f.push(eval_features(d, me)[2]);
    f
}

/// (`mover_features`: the phases by group, and the prompts by type)
const MOVER_PHASES: [&[u32]; 6] = [&[PHASE_DRAW], &[PHASE_STANDBY], &[PHASE_MAIN1], &[PHASE_BATTLE_START, PHASE_BATTLE_STEP, PHASE_DAMAGE, PHASE_DAMAGE_CAL, PHASE_BATTLE], &[PHASE_MAIN2], &[PHASE_END]];
const MOVER_PROMPTS: [&str; 7] = ["SELECT_IDLECMD", "SELECT_CHAIN", "SELECT_CARD", "SELECT_BATTLECMD", "SELECT_EFFECTYN", "SELECT_YESNO", "SELECT_OPTION"];

/// The face-up Attack Position monsters of `mine` with more ATK than every monster of `theirs`
/// can stand against (each (position, ATK, DEF)).
fn dominant(mine: &[(u32, i32, i32)], theirs: &[(u32, i32, i32)]) -> usize {
    let best = theirs.iter().map(|&(pos, a, df)| battle_stat(pos, a, df)).max().unwrap_or(-1);
    mine.iter().filter(|&&(pos, a, _)| pos & POS_FACEUP_ATTACK != 0 && a > best).count()
}

/// One side read straight from the duel, with the values the location queries give
/// (`side_features`, then `side_features2` below), and its monsters as (position, ATK, DEF).
struct SideScan {
    base: [f64; 16],
    more: [f64; 14],
    mons: Vec<(u32, i32, i32)>,
}

/// `SideScan` of player `p`: the queries' values without building and parsing their buffers (the
/// duel is marked as querying meanwhile, as a query does).
fn side_scan(d: &mut Duel, p: u32) -> SideScan {
    const EXTRA_TYPES: u32 = TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK;
    let was = d.querying;
    d.querying = true;
    let pl = p as usize;
    let mut mons = Vec::with_capacity(7);
    let (mut up, mut atk_up, mut atk_sum, mut best) = (0, 0, 0i64, 0i32);
    let (mut lv_sum, mut lv_max, mut high, mut extra, mut overlays, mut counters, mut emz) = (0u32, 0u32, 0, 0, 0, 0u32, 0);
    let count = |cs: &[(u16, [u16; 2])]| cs.iter().map(|(_, v)| v[0] as u32 + v[1] as u32).sum::<u32>();
    for i in 0..d.player[pl].list_mzone.len() {
        let Some(c) = d.player[pl].list_mzone[i] else { continue };
        let pos = c.get_info_location(d).position;
        let (atk, def) = (c.get_attack(d), c.get_defense(d));
        mons.push((pos, atk, def));
        if i >= 5 {
            emz += 1;
        }
        overlays += d.card(c).xyz_materials.len();
        counters += count(&d.card(c).counters);
        if pos & POS_FACEUP != 0 {
            up += 1;
            let attacking = pos & POS_ATTACK != 0;
            best = best.max(if attacking { atk } else { def });
            if attacking {
                atk_up += 1;
                atk_sum += atk as i64;
            }
            let level = c.get_level(d);
            let stars = if level > 0 { level } else { c.get_rank(d) };
            (lv_sum, lv_max) = (lv_sum + stars, lv_max.max(stars));
            high += (stars >= 5) as u32;
            extra += (c.get_type(d, None, 0, 2) & EXTRA_TYPES != 0) as u32;
        }
    }
    let (mut down, mut n_back, mut field, mut equips) = (0, 0, false, 0);
    for i in 0..d.player[pl].list_szone.len() {
        let Some(c) = d.player[pl].list_szone[i] else { continue };
        if i < 5 {
            n_back += 1;
            down += (c.get_info_location(d).position & POS_FACEDOWN != 0) as u32;
        }
        field |= i == 5;
        equips += d.card(c).equiping_target.is_some() as u32;
        counters += count(&d.card(c).counters);
    }
    // (monsters, spells and traps of a pile, by their current types)
    let kinds = |d: &mut Duel, pile: fn(&ygo_core::duel::PlayerInfo) -> &Vec<CardId>| -> [f64; 3] {
        let mut k = [0.0; 3];
        for i in 0..pile(&d.player[pl]).len() {
            let t = pile(&d.player[pl])[i].get_type(d, None, 0, 2);
            for (j, flag) in [TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP].into_iter().enumerate() {
                k[j] += (t & flag != 0) as u8 as f64;
            }
        }
        k
    };
    let hand = kinds(d, |p| &p.list_hand);
    let grave = kinds(d, |p| &p.list_grave);
    let removed = kinds(d, |p| &p.list_remove);
    d.querying = was;
    let lp = d.ext_get_info()[3 + pl] as i32;
    let pl_ = &d.player[pl];
    let deck = pl_.list_main.len() as f64;
    let base = [
        lp as f64 / 1000.0,
        pl_.list_hand.len() as f64,
        deck / 10.0,
        pl_.list_grave.len() as f64 / 10.0,
        pl_.list_remove.len() as f64 / 10.0,
        pl_.list_extra.len() as f64 / 10.0,
        mons.len() as f64,
        up as f64,
        atk_up as f64,
        (mons.len() - up) as f64,
        atk_sum as f64 / 1000.0,
        best as f64 / 1000.0,
        down as f64,
        (n_back - down) as f64,
        field as u8 as f64,
        (deck <= 2.0) as u8 as f64,
    ];
    let more = [
        lv_sum as f64 / 10.0,
        lv_max as f64 / 10.0,
        high as f64,
        extra as f64,
        hand[0],
        hand[1],
        hand[2],
        grave[0] / 10.0,
        (grave[1] + grave[2]) / 10.0,
        removed[0] / 10.0,
        overlays as f64,
        equips as f64,
        counters as f64 / 10.0,
        emz as f64,
    ];
    SideScan { base, more, mons }
}

/// The features of the mover's view (the player whose decision is pending; scripts/train_eval.py
/// mover features): the mover's side (`side_features`, then its dominant monsters), the
/// opponent's the same way, whether the mover is the turn player, the phase group, the prompt
/// type (none without a prompt) and the turn / 10 (at most 3). With `more` (`mover2`) then: the
/// mover's and the opponent's further terms (their face-up monsters' Levels (Ranks for Xyz)
/// summed / 10 and the most / 10, those of Level 5 or more, their Extra Deck monsters on the
/// field; monsters, spells and traps in hand; monsters and spells/traps in the GY / 10; banished
/// monsters / 10; Xyz materials, equipped cards and counters on the field; the Extra Monster
/// Zones held), the pending prompt's options (Normal Summons, Special Summons, repositions,
/// monster Sets, spell/trap Sets and activations of an idle command; attacks and activations of a
/// battle command; the effects of a chain window) and the chain's length.
fn mover_view(d: &mut Duel, mover: u32, prompt: Option<&Msg>, more: bool) -> Vec<f64> {
    let (a, b) = (side_scan(d, mover), side_scan(d, 1 - mover));
    let mut f = Vec::with_capacity(87);
    f.extend(a.base);
    f.push(dominant(&a.mons, &b.mons) as f64);
    f.extend(b.base);
    f.push(dominant(&b.mons, &a.mons) as f64);
    let info = d.ext_get_info();
    f.push((info[1] == mover) as u8 as f64);
    f.extend(MOVER_PHASES.iter().map(|g| g.contains(&info[2]) as u8 as f64));
    let name = prompt.map(|m| m.name());
    f.extend(MOVER_PROMPTS.iter().map(|p| (name == Some(*p)) as u8 as f64));
    f.push(info[0].min(30) as f64 / 10.0);
    if more {
        f.extend(a.more);
        // (the opponent's hand by kind stays out, its count only: 0 here, the feature kept so the
        // datagen rows' layout holds; training drops it with --drop 67,68,69)
        let mut opp = b.more;
        opp[4..7].fill(0.0);
        f.extend(opp);
        let mut m = [0usize; 9];
        match prompt {
            Some(Msg::SelectIdlecmd { summonable, spsummonable, repositionable, msetable, ssetable, activatable, .. }) => {
                m[..6].copy_from_slice(&[summonable.len(), spsummonable.len(), repositionable.len(), msetable.len(), ssetable.len(), activatable.len()]);
            }
            Some(Msg::SelectBattlecmd { attackable, activatable, .. }) => (m[6], m[7]) = (attackable.len(), activatable.len()),
            Some(Msg::SelectChain { chains, .. }) => m[8] = chains.len(),
            _ => {}
        }
        f.extend(m.iter().map(|&x| x as f64));
        f.push(d.core.current_chain.len() as f64);
    }
    f
}

/// The 49 features of the `mover` evaluation (see `mover_view`).
pub fn mover_features(d: &mut Duel, mover: u32, prompt: Option<&Msg>) -> Vec<f64> {
    mover_view(d, mover, prompt, false)
}

/// The 87 features of the `mover2` evaluation (see `mover_view`).
pub fn mover_features2(d: &mut Duel, mover: u32, prompt: Option<&Msg>) -> Vec<f64> {
    mover_view(d, mover, prompt, true)
}

/// A trained evaluation (scripts/train_eval.py's export): linear on the raw features, or the
/// antisymmetric MLP v(z) = g(z) - g(-z), g(z) = w2 . relu(w1 z + b1), z the features divided by
/// `scale`. Its value is in eval points.
#[derive(serde::Deserialize)]
pub struct LearnedEval {
    features: String,
    model: String,
    scale: Vec<f64>,
    #[serde(default)]
    weights: Vec<f64>,
    #[serde(default)]
    w1: Vec<Vec<f64>>,
    #[serde(default)]
    b1: Vec<f64>,
    #[serde(default)]
    w2: Vec<f64>,
    /// (`mover`: the features' mean, subtracted before `scale` divides; the output's bias)
    #[serde(default)]
    mu: Vec<f64>,
    #[serde(default)]
    b2: f64,
    /// (`mover`: hidden layers after the first, each relu(w h + b); `w2` reads the last)
    #[serde(default)]
    layers: Vec<Layer>,
}

#[derive(serde::Deserialize)]
struct Layer {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

/// relu(w x + b)
fn dense_relu(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter().zip(b).map(|(row, b)| (row.iter().zip(x).map(|(w, x)| w * x).sum::<f64>() + b).max(0.0)).collect()
}

impl LearnedEval {
    pub fn value(&self, d: &mut Duel, me: u32, prompt: Option<&Msg>) -> f64 {
        if self.features == "mover" || self.features == "mover2" {
            // (the mover's value, an unconstrained MLP: plus for me, minus for the opponent)
            let mover = prompt.and_then(|m| m.player()).unwrap_or(me);
            let x = if self.features == "mover" { mover_features(d, mover, prompt) } else { mover_features2(d, mover, prompt) };
            let z: Vec<f64> = x.iter().zip(&self.mu).zip(&self.scale).map(|((a, m), s)| (a - m) / s).collect();
            let mut h = dense_relu(&self.w1, &self.b1, &z);
            for l in &self.layers {
                h = dense_relu(&l.w, &l.b, &h);
            }
            let v = self.b2 + self.w2.iter().zip(&h).map(|(w, x)| w * x).sum::<f64>();
            return if mover == me { v } else { -v };
        }
        let x = if self.features == "rich" { rich_features(d, me) } else { eval_features(d, me).to_vec() };
        if self.model == "linear" {
            return x.iter().zip(&self.weights).map(|(a, w)| a * w).sum();
        }
        let z: Vec<f64> = x.iter().zip(&self.scale).map(|(a, s)| a / s).collect();
        let g = |sign: f64| -> f64 {
            self.w1
                .iter()
                .zip(&self.b1)
                .zip(&self.w2)
                .map(|((row, b), w2)| w2 * (row.iter().zip(&z).map(|(w, x)| w * sign * x).sum::<f64>() + b).max(0.0))
                .sum()
        };
        g(1.0) - g(-1.0)
    }

    /// The trained evaluation in use (read once): $YGOAI_EVAL's JSON file if set, the built-in
    /// models/eval_mover2_mlp16.json otherwise; none with YGOAI_EVAL=hand (the hand-written eval).
    pub fn from_env() -> Option<&'static LearnedEval> {
        static E: std::sync::OnceLock<Option<LearnedEval>> = std::sync::OnceLock::new();
        E.get_or_init(|| match std::env::var("YGOAI_EVAL").ok().filter(|p| !p.is_empty()) {
            Some(p) if p == "hand" => None,
            Some(path) => {
                let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("YGOAI_EVAL {path}: {e}"));
                Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("YGOAI_EVAL {path}: {e}")))
            }
            None => Some(serde_json::from_str(DEFAULT_EVAL).expect("models/eval_mover2_mlp16.json")),
        })
        .as_ref()
    }
}

/// The default evaluation (scripts/train_eval_mover.py: MLP16 on the 87 mover2 features, trained on
/// runs/data/pairs500x20b_2000n.jsonl; SPRT +31.9 +/- 18.7 vs the 49-feature MLP16 (on
/// runs/data/pairs500x20_2000n.jsonl), which had +28.5 +/- 17.6 vs the hand-written eval).
const DEFAULT_EVAL: &str = include_str!("../../models/eval_mover2_mlp16.json");

/// The position's value for `me` (in points of 100 LP): the trained evaluation (see
/// `LearnedEval::from_env`), or the hand-written one (alphabeta.evaluate) with YGOAI_EVAL=hand.
pub fn evaluate(d: &mut Duel, me: u32) -> f64 {
    evaluate_at(d, me, None)
}

/// `evaluate` with the pending prompt, if any (a trained evaluation of the mover's view values
/// the position for the player who decides it).
pub fn evaluate_at(d: &mut Duel, me: u32, prompt: Option<&Msg>) -> f64 {
    if let Some(e) = LearnedEval::from_env() {
        return e.value(d, me, prompt);
    }
    let info = d.ext_get_info();
    let lp = [info[3] as i32, info[4] as i32];
    let mons: Vec<Vec<(u32, i32, i32)>> = [me, 1 - me]
        .iter()
        .map(|&p| {
            parse_location_query(&query(d, p, LOCATION_MZONE, MZONE_QUERY))
                .into_iter()
                .flatten()
                .map(|c| (c.position.unwrap_or(0), c.attack.unwrap_or(0), c.defense.unwrap_or(0)))
                .collect()
        })
        .collect();
    let (me_i, opp_i) = (me as usize, 1 - me as usize);
    let mut value = W_LP * (lp[me_i] - lp[opp_i]) as f64 + W_MONSTER * (mons[0].len() as f64 - mons[1].len() as f64);
    for (s, p) in [(0usize, me), (1, 1 - me)] {
        let sign = if s == 0 { 1.0 } else { -1.0 };
        let best_other = mons[1 - s].iter().map(|&(pos, a, df)| battle_stat(pos, a, df)).max().unwrap_or(-1);
        for &(pos, a, _) in &mons[s] {
            if pos & POS_FACEUP_ATTACK != 0 && a > best_other {
                value += sign * W_DOMINANT;
            }
        }
        value += sign * W_HAND * d.ocg_query_count(p as u8, LOCATION_HAND) as f64;
        let backrow = parse_location_query(&query(d, p, LOCATION_SZONE, QUERY_POSITION))
            .into_iter()
            .flatten()
            .filter(|c| c.position.unwrap_or(0) & POS_FACEDOWN != 0)
            .count();
        value += sign * W_BACKROW * backrow as f64;
        if d.ocg_query_count(p as u8, LOCATION_DECK) <= 2 {
            value -= sign * W_DECK_OUT;
        }
    }
    value
}
