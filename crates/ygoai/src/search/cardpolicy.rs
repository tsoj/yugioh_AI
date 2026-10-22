//! The card policy: a small network scoring the cards of a one-card selection (`SelectCard` with
//! min = max = 1) for the search, in place of the rules' order by the select hint. The order
//! decides which cards the search tries (`Tune::card_width` of them) and which one the rules take
//! at the horizon. Its features (`card_features`): the card's data and effects (types, ranges and
//! categories, no card identity), where it is, its copies elsewhere, the rules' rank, the hint and
//! a little of the position. Trained by scripts/train_card_policy.py on ygoai-card-refs' exact
//! values of every card.

use ygo_core::constants::*;
use ygo_core::duel::Duel;

use crate::heuristic::RULED;
use crate::messages::Msg;

const TYPES: [u32; 18] = [
    TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_NORMAL, TYPE_EFFECT, TYPE_FUSION, TYPE_RITUAL, TYPE_SYNCHRO, TYPE_XYZ, TYPE_LINK, TYPE_PENDULUM, TYPE_TUNER,
    TYPE_FLIP, TYPE_QUICKPLAY, TYPE_CONTINUOUS, TYPE_EQUIP, TYPE_FIELD, TYPE_COUNTER,
];
const LOCATIONS: [u32; 8] = [LOCATION_DECK, LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA, LOCATION_OVERLAY];
/// (an effect's range, for ignition and quick effects)
const RANGES: [u32; 5] = [LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE | LOCATION_FZONE | LOCATION_PZONE, LOCATION_GRAVE, LOCATION_REMOVED];
const PHASES: [u32; 5] = [PHASE_DRAW | PHASE_STANDBY, PHASE_MAIN1, PHASE_BATTLE_START | PHASE_BATTLE_STEP | PHASE_DAMAGE | PHASE_DAMAGE_CAL | PHASE_BATTLE, PHASE_MAIN2, PHASE_END];
/// (the effects one activates, whose categories count)
const ACTION_TYPES: u32 = EFFECT_TYPE_ACTIVATE | EFFECT_TYPE_IGNITION | EFFECT_TYPE_QUICK_O | EFFECT_TYPE_QUICK_F | EFFECT_TYPE_TRIGGER_O | EFFECT_TYPE_TRIGGER_F | EFFECT_TYPE_FLIP;
const CATEGORIES: usize = 33;

/// The number of features of a card.
pub const CARD_FEATURES: usize = TYPES.len() + 7 + 5 + LOCATIONS.len() + 2 + 1 + 2 * RANGES.len() + 7 + CATEGORIES + 4 + 2 + RULED.len() + 2 + 1 + PHASES.len() + 6;

/// Whether the policy orders this selection.
pub fn applies(prompt: &Msg) -> bool {
    matches!(prompt, Msg::SelectCard { min: 1, max: 1, cards, .. } if cards.len() >= 2)
}

/// The features of each card of the selection `prompt` (`applies`), with the select hint `hint`
/// and the rules' order `rules` (the cards' indices, the preferred first).
pub fn card_features(d: &Duel, prompt: &Msg, hint: Option<u64>, rules: &[usize]) -> Vec<[f32; CARD_FEATURES]> {
    let Msg::SelectCard { cards, player, .. } = prompt else { return Vec::new() };
    let me = *player as usize;
    let info = d.ext_get_info();
    // (the selection's and the position's features, the same for every card)
    let mut shared = Vec::with_capacity(RULED.len() + 16);
    match hint.and_then(|h| RULED.iter().position(|r| *r == h)) {
        Some(i) => shared.extend((0..RULED.len()).map(|j| (i == j) as u32 as f32)),
        None => shared.extend(std::iter::repeat_n(0.0, RULED.len())),
    }
    shared.push(hint.is_some_and(|h| !RULED.contains(&h)) as u32 as f32);
    shared.push(hint.is_none() as u32 as f32);
    shared.push((info[1] as usize == me) as u32 as f32);
    shared.extend(PHASES.iter().map(|p| (info[2] & p != 0) as u32 as f32));
    let side = |p: usize| (&d.player[p], d.player[p].list_mzone.iter().take(5).filter(|c| c.is_some()).count() as f32);
    let ((mine, my_mons), (_, their_mons)) = (side(me), side(1 - me));
    shared.extend([
        cards.len() as f32 / 10.0,
        info[3 + me] as f32 / 8000.0,
        info[4 - me] as f32 / 8000.0,
        mine.list_hand.len() as f32 / 10.0,
        my_mons / 5.0,
        their_mons / 5.0,
    ]);

    let code = |c: ygo_core::types::CardId| d.card(c).data.code;
    let copies = |list: &mut dyn Iterator<Item = ygo_core::types::CardId>, k: u32| list.filter(|&c| code(c) == k).count().min(3) as f32 / 3.0;
    let n = cards.len();
    let mut out = Vec::with_capacity(n);
    for (i, c) in cards.iter().enumerate() {
        let mut x = [0f32; CARD_FEATURES];
        let mut f = 0;
        let mut put = |v: f32| {
            x[f] = v;
            f += 1;
        };
        let id = (c.loc & LOCATION_OVERLAY == 0).then(|| d.get_field_card(c.con, c.loc, c.seq)).flatten();
        let data = id.map(|id| &d.card(id).data).or_else(|| crate::heuristic::card(c.code));
        // the card's data
        let ty = data.map_or(0, |c| c.type_);
        TYPES.iter().for_each(|t| put((ty & t != 0) as u32 as f32));
        let attribute = data.map_or(0, |c| c.attribute);
        (0..7).for_each(|b| put((attribute & (1 << b) != 0) as u32 as f32));
        let monster = ty & TYPE_MONSTER != 0;
        let level = data.filter(|_| monster).map_or(0, |c| c.level & 0xff);
        let stat = |v: i32| v.max(0) as f32 / 3000.0;
        put(level as f32 / 12.0);
        put(data.filter(|_| monster).map_or(0.0, |c| stat(c.attack)));
        put(data.filter(|_| monster && ty & TYPE_LINK == 0).map_or(0.0, |c| stat(c.defense)));
        let main_deck = monster && ty & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK) == 0;
        put((main_deck && (5..=6).contains(&level)) as u32 as f32);
        put((main_deck && level >= 7) as u32 as f32);
        // where it is
        let location = c.loc & !LOCATION_OVERLAY;
        LOCATIONS.iter().for_each(|l| put(if *l == LOCATION_OVERLAY { (c.loc & LOCATION_OVERLAY != 0) as u32 as f32 } else { (location == *l) as u32 as f32 }));
        put((c.con as usize == me) as u32 as f32);
        put((c.pos.unwrap_or(0) & POS_FACEUP != 0) as u32 as f32);
        // its effects (none known of an overlay card)
        let (mut activate, mut ignition, mut quick) = (0f32, [0f32; RANGES.len()], [0f32; RANGES.len()]);
        let (mut trigger_o, mut trigger_f, mut trigger_grave, mut flip, mut passive, mut limited, mut categories) = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32, 0u64);
        if let Some(id) = id {
            for &(e, _) in &d.card(id).indexer {
                let fx = d.effect(e);
                let (t, range) = (fx.type_ as u32, fx.range as u32);
                if fx.deleted {
                    continue;
                }
                if t & ACTION_TYPES != 0 {
                    categories |= fx.category;
                    limited = limited.max((fx.count_limit_max > 0 || fx.count_code != 0) as u32 as f32);
                }
                if t & EFFECT_TYPE_ACTIVATE != 0 {
                    activate = 1.0;
                } else if t & EFFECT_TYPE_IGNITION != 0 {
                    RANGES.iter().zip(&mut ignition).for_each(|(r, v)| *v = v.max((range & r != 0) as u32 as f32));
                } else if t & (EFFECT_TYPE_QUICK_O | EFFECT_TYPE_QUICK_F) != 0 {
                    RANGES.iter().zip(&mut quick).for_each(|(r, v)| *v = v.max((range & r != 0) as u32 as f32));
                } else if t & EFFECT_TYPE_TRIGGER_O != 0 {
                    trigger_o = 1.0;
                    trigger_grave = trigger_grave.max((range & LOCATION_GRAVE != 0) as u32 as f32);
                } else if t & EFFECT_TYPE_TRIGGER_F != 0 {
                    trigger_f = 1.0;
                    trigger_grave = trigger_grave.max((range & LOCATION_GRAVE != 0) as u32 as f32);
                } else if t & EFFECT_TYPE_FLIP != 0 {
                    flip = 1.0;
                } else {
                    passive = (passive + 0.25).min(1.0);
                }
            }
        }
        put(activate);
        ignition.into_iter().for_each(&mut put);
        quick.into_iter().for_each(&mut put);
        [trigger_o, trigger_f, trigger_grave, flip, passive, limited].into_iter().for_each(&mut put);
        put(id.is_none() as u32 as f32);
        (0..CATEGORIES).for_each(|b| put((categories >> b & 1) as f32));
        // its copies with the selecting player (not itself)
        let k = data.map_or(0, |c| c.code);
        let field = mine.list_mzone.iter().chain(&mine.list_szone).flatten().copied();
        let mut copies_in = |list: &mut dyn Iterator<Item = ygo_core::types::CardId>| copies(&mut list.filter(|&o| Some(o) != id), k);
        put(copies_in(&mut mine.list_hand.iter().copied()));
        put(copies_in(&mut field.into_iter()));
        put(copies_in(&mut mine.list_grave.iter().copied()));
        put(copies_in(&mut mine.list_main.iter().copied()));
        // the rules' rank
        let rank = rules.iter().position(|&r| r == i).unwrap_or(n - 1);
        put(rank as f32 / (n - 1).max(1) as f32);
        put((rank == 0) as u32 as f32);
        for &v in &shared {
            put(v);
        }
        debug_assert_eq!(f, CARD_FEATURES);
        out.push(x);
    }
    out
}

/// The network: score = w2 . relu(W1 x + b1) + b2 (the features' standardization folded into W1, b1).
#[derive(serde::Deserialize)]
pub struct CardPolicy {
    features: usize,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

impl CardPolicy {
    /// The policy in use (read once): $YGOAI_CARD_POLICY's JSON file if set, else none (the rules' order).
    pub fn get() -> Option<&'static CardPolicy> {
        static P: std::sync::OnceLock<Option<CardPolicy>> = std::sync::OnceLock::new();
        P.get_or_init(|| {
            let path = std::env::var("YGOAI_CARD_POLICY").ok().filter(|p| !p.is_empty())?;
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("YGOAI_CARD_POLICY {path}: {e}"));
            let p: CardPolicy = serde_json::from_str(&text).unwrap_or_else(|e| panic!("YGOAI_CARD_POLICY {path}: {e}"));
            assert_eq!(p.features, CARD_FEATURES, "YGOAI_CARD_POLICY {path}: features");
            Some(p)
        })
        .as_ref()
    }

    pub fn score(&self, x: &[f32; CARD_FEATURES]) -> f32 {
        let mut s = self.b2;
        for ((w, b), v) in self.w1.iter().zip(&self.b1).zip(&self.w2) {
            let h = b + w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>();
            if h > 0.0 {
                s += v * h;
            }
        }
        s
    }

    /// The cards' indices by score, the best first (ties: the rules' order `rules`).
    pub fn order(&self, d: &Duel, prompt: &Msg, hint: Option<u64>, rules: &[usize]) -> Vec<usize> {
        let scores: Vec<f32> = card_features(d, prompt, hint, rules).iter().map(|x| self.score(x)).collect();
        let mut order = rules.to_vec();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        order
    }
}
