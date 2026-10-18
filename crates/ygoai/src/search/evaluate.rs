//! The evaluation: the hand-written one (`YGOAI_EVAL=hand`), the trained ones (`LearnedEval`, the
//! built-in default) and the features they and the training data use.

use ygo_core::api::QueryInfo;
use ygo_core::constants::*;
use ygo_core::duel::Duel;
use ygo_core::types::CardId;

use crate::messages::Msg;
use crate::state::parse_location_query;


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
    /// (`mover3`'s further terms)
    turn: [f64; 7],
    /// (the attack of the monsters that can still attack this turn, for `mover3`'s lethal terms)
    ready_atk: i64,
    /// (`mover4`'s further terms)
    fx: [f64; 8],
    mons: Vec<(u32, i32, i32)>,
}

/// (`side_scan`: the effects one can choose to activate, by their own timing)
const CHOSEN_EFFECTS: u32 = EFFECT_TYPE_IGNITION | EFFECT_TYPE_QUICK_O | EFFECT_TYPE_TRIGGER_O;

/// Whether card `c` has an effect of the types `types` that can be activated from `location`
/// with uses left this turn for player `p` (not whether it could be activated now).
fn has_effect(d: &mut Duel, c: CardId, location: u32, types: u32, p: u32) -> bool {
    for i in 0..d.card(c).field_effect.items.len() {
        let e = d.card(c).field_effect.items[i].1;
        let fx = d.effect(e);
        if !fx.deleted && fx.type_ as u32 & types != 0 && fx.range as u32 & location != 0 && e.check_count_limit(d, p as u8) {
            return true;
        }
    }
    false
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
    let (mut ready, mut ready_atk, mut free) = (0, 0i64, 5);
    let (mut fx_field, mut fx_set) = (0u32, 0u32);
    for i in 0..d.player[pl].list_mzone.len() {
        let Some(c) = d.player[pl].list_mzone[i] else { continue };
        let pos = c.get_info_location(d).position;
        let (atk, def) = (c.get_attack(d), c.get_defense(d));
        mons.push((pos, atk, def));
        if i < 5 {
            free -= 1;
        }
        if pos & POS_FACEUP_ATTACK != 0 && d.card(c).announce_count == 0 {
            ready += 1;
            ready_atk += atk as i64;
        }
        if i >= 5 {
            emz += 1;
        }
        overlays += d.card(c).xyz_materials.len();
        counters += count(&d.card(c).counters);
        if pos & POS_FACEUP != 0 {
            up += 1;
            fx_field += has_effect(d, c, LOCATION_MZONE, EFFECT_TYPE_IGNITION | EFFECT_TYPE_QUICK_O, p) as u32;
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
        let facedown = c.get_info_location(d).position & POS_FACEDOWN != 0;
        if i < 5 {
            n_back += 1;
            down += facedown as u32;
        }
        if facedown {
            // (Traps and Quick-Play Spells, ready for the opponent's turn)
            let t = c.get_type(d, None, 0, 2);
            fx_set += (t & TYPE_TRAP != 0 || t & TYPE_QUICKPLAY != 0) as u32;
        } else {
            fx_field += has_effect(d, c, LOCATION_SZONE, EFFECT_TYPE_IGNITION | EFFECT_TYPE_QUICK_O, p) as u32;
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
    let mut hand_low = 0;
    for i in 0..d.player[pl].list_hand.len() {
        let c = d.player[pl].list_hand[i];
        if c.get_type(d, None, 0, 2) & TYPE_MONSTER != 0 && (1..=4).contains(&c.get_level(d)) {
            hand_low += 1;
        }
    }
    let grave = kinds(d, |p| &p.list_grave);
    let mut fx_hand = 0;
    for i in 0..d.player[pl].list_hand.len() {
        let c = d.player[pl].list_hand[i];
        fx_hand += has_effect(d, c, LOCATION_HAND, CHOSEN_EFFECTS, p) as u32;
    }
    let mut fx_grave = 0;
    for i in 0..d.player[pl].list_grave.len() {
        let c = d.player[pl].list_grave[i];
        fx_grave += has_effect(d, c, LOCATION_GRAVE, CHOSEN_EFFECTS, p) as u32;
    }
    let mut extra_kinds = [0u32; 4];
    for i in 0..d.player[pl].list_extra.len() {
        let t = d.player[pl].list_extra[i].get_type(d, None, 0, 2);
        for (k, flag) in [TYPE_FUSION, TYPE_SYNCHRO, TYPE_XYZ, TYPE_LINK].into_iter().enumerate() {
            extra_kinds[k] += (t & flag != 0) as u32;
        }
    }
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
    let turn = [
        d.core.summon_count[pl] as f64,
        d.core.spsummon_state_count[pl] as f64 / 5.0,
        d.core.attack_state_count[pl] as f64,
        ready as f64,
        ready_atk as f64 / 1000.0,
        hand_low as f64,
        free as f64,
    ];
    // (the face-up monsters' battle stats: the second best, and the weakest attacker's ATK)
    let mut stats: Vec<i32> = mons.iter().filter(|m| m.0 & POS_FACEUP != 0).map(|&(pos, a, df)| battle_stat(pos, a, df)).collect();
    stats.sort_unstable_by(|a, b| b.cmp(a));
    let weakest = mons.iter().filter(|m| m.0 & POS_FACEUP_ATTACK != 0).map(|m| m.1).min().unwrap_or(0);
    let fx = [
        fx_field as f64,
        fx_grave as f64,
        fx_hand as f64,
        fx_set as f64,
        (extra_kinds[0] + extra_kinds[1]) as f64,
        (extra_kinds[2] + extra_kinds[3]) as f64,
        stats.get(1).copied().unwrap_or(0) as f64 / 1000.0,
        weakest as f64 / 1000.0,
    ];
    SideScan { base, more, turn, ready_atk, fx, mons }
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
/// battle command; the effects of a chain window) and the chain's length. With `turn` (`mover3`)
/// then: for the mover and the opponent their Normal Summons, Special Summons / 5 and attacks
/// this turn, their face-up attack position monsters that have not attacked yet and their ATK /
/// 1000, the Level 1-4 monsters in hand (the opponent's: 0, hidden) and the free Main Monster
/// Zones; whether each could attack directly for the other's LP (no monsters there, those ready
/// enough), and whether the chain's last link is the mover's. With level 4 (`mover4`) then: for the
/// mover and the opponent their face-up cards with an Ignition or Quick Effect with uses left this
/// turn, their GY's cards with an effect to choose to activate there (Ignition, Quick, optional
/// Trigger) with uses left, the same in hand (the opponent's: 0), their set Traps and Quick-Play
/// Spells (0), their Extra Deck's Fusion and Synchro, and Xyz and Link Monsters (0), their face-up
/// monsters' second best battle stat / 1000 and their weakest attacker's ATK / 1000; each side's
/// face-up monsters that the other's strongest attacker beats.
fn mover_view(d: &mut Duel, mover: u32, prompt: Option<&Msg>, level: u8) -> Vec<f64> {
    let (a, b) = (side_scan(d, mover), side_scan(d, 1 - mover));
    let mut f = Vec::with_capacity(122);
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
    if level >= 2 {
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
    if level >= 3 {
        f.extend(a.turn);
        let mut opp = b.turn;
        opp[5] = 0.0;
        f.extend(opp);
        let lp = |p: u32| d.ext_get_info()[3 + p as usize] as i64;
        f.push((b.mons.is_empty() && a.ready_atk > 0 && a.ready_atk >= lp(1 - mover)) as u8 as f64);
        f.push((a.mons.is_empty() && b.ready_atk > 0 && b.ready_atk >= lp(mover)) as u8 as f64);
        f.push(d.core.current_chain.last().is_some_and(|l| l.triggering_controler as u32 == mover) as u8 as f64);
    }
    if level >= 4 {
        f.extend(a.fx);
        // (the opponent's hand, set cards and Extra Deck stay out: 0)
        let mut opp = b.fx;
        opp[2..6].fill(0.0);
        f.extend(opp);
        // (each side's face-up monsters that the other's strongest ready attacker beats)
        let beaten = |mine: &[(u32, i32, i32)], theirs: &[(u32, i32, i32)]| {
            let top = theirs.iter().filter(|m| m.0 & POS_FACEUP_ATTACK != 0).map(|m| m.1).max().unwrap_or(-1);
            mine.iter().filter(|&&(pos, a, df)| pos & POS_FACEUP != 0 && battle_stat(pos, a, df) < top).count() as f64
        };
        f.push(beaten(&a.mons, &b.mons));
        f.push(beaten(&b.mons, &a.mons));
    }
    f
}

/// The 49 features of the `mover` evaluation (see `mover_view`).
pub fn mover_features(d: &mut Duel, mover: u32, prompt: Option<&Msg>) -> Vec<f64> {
    mover_view(d, mover, prompt, 1)
}

/// The 87 features of the `mover2` evaluation (see `mover_view`).
pub fn mover_features2(d: &mut Duel, mover: u32, prompt: Option<&Msg>) -> Vec<f64> {
    mover_view(d, mover, prompt, 2)
}

/// The 104 features of the `mover3` evaluation (see `mover_view`).
pub fn mover_features3(d: &mut Duel, mover: u32, prompt: Option<&Msg>) -> Vec<f64> {
    mover_view(d, mover, prompt, 3)
}

/// The 122 features of the `mover4` evaluation (see `mover_view`).
pub fn mover_features4(d: &mut Duel, mover: u32, prompt: Option<&Msg>) -> Vec<f64> {
    mover_view(d, mover, prompt, 4)
}

/// The trained evaluation (scripts/train_eval_mover.py's export): an MLP on the 87 `mover2`
/// features of the view of the player to move, z = (x - mu) / scale, hidden layers relu(w h + b),
/// output w2 . h + b2 for the mover (negated for the other player). Its value is in LP: the
/// network's output times its `lp_anchor`'s factor (see `LpAnchor`; without one, in the network's
/// own points, with a warning). Nets on other feature sets are not loaded (the other evaluations
/// were removed on 2026-10-18: the hand-written one, linear and antisymmetric models, the `mover`,
/// `mover3`, `mover4` and `rich` feature sets; datagen still writes those features).
#[derive(serde::Deserialize, Clone)]
pub struct LearnedEval {
    features: String,
    /// (`model` and `weights`, of the removed model types: only in `weights_hash`, so that the
    /// existing anchors stay valid)
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
    /// (the features' mean, subtracted before `scale` divides; the output's bias)
    #[serde(default)]
    mu: Vec<f64>,
    #[serde(default)]
    b2: f64,
    /// (hidden layers after the first, each relu(w h + b); `w2` reads the last)
    #[serde(default)]
    layers: Vec<Layer>,
    #[serde(default)]
    pub lp_anchor: Option<LpAnchor>,
    /// (the factor in use: the anchor's if it belongs to these weights, else 1)
    #[serde(skip)]
    lp_factor: f64,
}

/// The network's output in LP (ygoai-anchor): `factor` = 1 / `slope`, the mean change of the
/// output per LP of the LP difference (the mover's LP up and the opponent's down by 250 each,
/// against the reverse) over `positions` positions of `data` where both players have at least
/// LP_ANCHOR_MIN_LP. `weights_hash`: `LearnedEval::weights_hash` of the weights it was measured
/// on (another hash: the network changed since, and the engine warns).
#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct LpAnchor {
    pub factor: f64,
    pub slope: f64,
    pub positions: u64,
    pub data: String,
    pub weights_hash: String,
}

/// (`LpAnchor`: the positions measured have both players at this many LP or more)
pub const LP_ANCHOR_MIN_LP: f64 = 2000.0;
/// (`LpAnchor`: each player's LP moved by this much, both ways)
pub const LP_ANCHOR_STEP: f64 = 250.0;

#[derive(serde::Deserialize, Clone)]
struct Layer {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

/// relu(w x + b)
fn dense_relu(w: &[Vec<f64>], b: &[f64], x: &[f64]) -> Vec<f64> {
    w.iter().zip(b).map(|(row, b)| (row.iter().zip(x).map(|(w, x)| w * x).sum::<f64>() + b).max(0.0)).collect()
}

impl LearnedEval {
    /// The evaluation of a model file's text; `name` for the warnings.
    pub fn from_json(text: &str, name: &str) -> LearnedEval {
        let mut e: LearnedEval = serde_json::from_str(text).unwrap_or_else(|err| panic!("eval {name}: {err}"));
        assert!(e.features == "mover2" && e.mu.len() == e.scale.len(), "eval {name}: a {} net; only mover2 MLPs are evaluated", e.features);
        e.lp_factor = 1.0;
        match &e.lp_anchor {
            None => eprintln!("warning: eval {name} has no LP anchor (run ygoai-anchor): its values are in the network's own points, the search's margins are in LP"),
            Some(a) if a.weights_hash != e.weights_hash() => eprintln!(
                "warning: eval {name}: the LP anchor was measured on other weights (hash {} vs {}): run ygoai-anchor again; values in the network's own points",
                a.weights_hash,
                e.weights_hash()
            ),
            Some(a) => e.lp_factor = a.factor,
        }
        e
    }

    /// A hash of the network (its features, model type and every parameter's bits; FNV-1a).
    pub fn weights_hash(&self) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |bytes: &[u8]| {
            for b in bytes {
                h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
            }
        };
        eat(self.features.as_bytes());
        eat(self.model.as_bytes());
        let mut all: Vec<f64> = Vec::new();
        for v in [&self.mu, &self.scale, &self.weights, &self.b1, &self.w2] {
            all.extend(v.iter().copied());
            all.push(f64::NAN); // (a separator)
        }
        all.extend(self.w1.iter().flatten().copied());
        for l in &self.layers {
            all.extend(l.w.iter().flatten().chain(&l.b).copied());
        }
        all.push(self.b2);
        for x in all {
            eat(&x.to_bits().to_le_bytes());
        }
        format!("{h:016x}")
    }

    /// The indices of the mover's and the opponent's LP / 1000 in the features.
    pub fn lp_features(&self) -> (usize, usize) {
        (0, 17)
    }

    /// The feature set's name (`mover2`, ...).
    pub fn feature_set(&self) -> &str {
        &self.features
    }

    /// The number of features the network reads.
    pub fn n_features(&self) -> usize {
        self.scale.len()
    }

    /// The network's output (its own points) for the mover's features `x`.
    pub fn mover_output(&self, x: &[f64]) -> f64 {
        let z: Vec<f64> = x.iter().zip(&self.mu).zip(&self.scale).map(|((a, m), s)| (a - m) / s).collect();
        let mut h = dense_relu(&self.w1, &self.b1, &z);
        for l in &self.layers {
            h = dense_relu(&l.w, &l.b, &h);
        }
        self.b2 + self.w2.iter().zip(&h).map(|(w, x)| w * x).sum::<f64>()
    }

    /// The factor from the network's points to LP.
    pub fn lp_factor(&self) -> f64 {
        self.lp_factor
    }

    /// The position's value for `me` (in LP): the mover's value, plus for me and minus for the
    /// opponent.
    pub fn value(&self, d: &mut Duel, me: u32, prompt: Option<&Msg>) -> f64 {
        let mover = prompt.and_then(|m| m.player()).unwrap_or(me);
        let v = self.mover_output(&mover_features2(d, mover, prompt));
        self.lp_factor * if mover == me { v } else { -v }
    }

    /// The trained evaluation in use (read once): $YGOAI_EVAL's JSON file if set, the built-in
    /// models/eval_mover2_r23os_mlp16.json otherwise.
    pub fn from_env() -> &'static LearnedEval {
        static E: std::sync::OnceLock<LearnedEval> = std::sync::OnceLock::new();
        E.get_or_init(|| match std::env::var("YGOAI_EVAL").ok().filter(|p| !p.is_empty()) {
            Some(path) => {
                let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("YGOAI_EVAL {path}: {e}"));
                LearnedEval::from_json(&text, &path)
            }
            None => LearnedEval::from_json(DEFAULT_EVAL, "models/eval_mover2_r23os_mlp16.json"),
        })
    }
}

/// The default evaluation (scripts/train_eval_mover.py --drop-onesided: MLP16 on the 87 mover2
/// features, trained on rounds 2 and 3, runs/data/pairs500x20b_2000n.jsonl and
/// runs/data/pairs500x20c_2000n_all.games.jsonl, without the deck pairs where one deck won every
/// game; SPRT H1 +21.0 +/- 14.4 at 1242 games vs the same on all pairs,
/// models/eval_mover2_r23_mlp16.json, which vs the round-2 net, models/eval_mover2_mlp16.json,
/// stopped at +6.7 +/- 10.0 after 2540 games.
/// That one had SPRT +31.9 +/- 18.7 vs the 49-feature MLP16 (on runs/data/pairs500x20_2000n.jsonl),
/// which had +28.5 +/- 17.6 vs the hand-written eval).
const DEFAULT_EVAL: &str = include_str!("../../models/eval_mover2_r23os_mlp16.json");

/// The position's value for `me` (in LP): the trained evaluation (`LearnedEval::from_env`).
pub fn evaluate(d: &mut Duel, me: u32) -> f64 {
    evaluate_at(d, me, None)
}

/// `evaluate` with the pending prompt, if any (the evaluation of the mover's view values the
/// position for the player who decides it).
pub fn evaluate_at(d: &mut Duel, me: u32, prompt: Option<&Msg>) -> f64 {
    LearnedEval::from_env().value(d, me, prompt)
}
