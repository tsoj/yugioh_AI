//! The place policy's features: of each zone of a zone selection (`SelectPlace`,
//! `SelectDisfield`; the zones `available_places` lists, the rules taking the first), the zone
//! (whose, which row, the column), what is in its column and next to it, the link arrows that
//! point to it, the card placed (the select hint's code: its types and arrows) and its arrows from
//! there, the rules' rank and a little of the position. A set of zones scores as the set policy's
//! (`cardpolicy::CardPolicy::order_sets`). Data: ygoai-card-refs --decisions.

use ygo_core::constants::*;
use ygo_core::duel::Duel;
use ygo_core::types::CardId;

use super::cardpolicy::PHASES;
use crate::actions::available_places;
use crate::messages::Msg;

const MARKERS: [u32; 8] = [
    LINK_MARKER_BOTTOM_LEFT, LINK_MARKER_BOTTOM, LINK_MARKER_BOTTOM_RIGHT, LINK_MARKER_LEFT, LINK_MARKER_RIGHT, LINK_MARKER_TOP_LEFT, LINK_MARKER_TOP, LINK_MARKER_TOP_RIGHT,
];

/// The number of features of a zone.
/// (the zone's 39; the placed card's 1 + 8 + 2 + its arrows, the selection's 4, the position's)
pub const PLACE_FEATURES: usize = 39 + 1 + 8 + 2 + MARKERS.len() + 4 + PHASES.len() + 3;

/// The zones a link card's arrows point to from (`loc`, `seq`) of its controller, as
/// `card::get_linked_zone` (with the extra monster zones): bits 0-6 its monster zones, 8-12 its
/// spell and trap zones, 16- the opponent's.
pub fn linked_zones(loc: u32, seq: u32, marker: u32) -> u32 {
    let m = |dir: u32| marker & dir != 0;
    let s = seq as i32;
    let mut zones = 0u32;
    let mut set = |b: i32| zones |= 1u32 << b;
    if loc == LOCATION_MZONE {
        if (1..=4).contains(&s) && m(LINK_MARKER_LEFT) {
            set(s - 1);
        }
        if s <= 3 && m(LINK_MARKER_RIGHT) {
            set(s + 1);
        }
        if s < 5 {
            if s > 0 && m(LINK_MARKER_BOTTOM_LEFT) {
                set(s + 7);
            }
            if s < 4 && m(LINK_MARKER_BOTTOM_RIGHT) {
                set(s + 9);
            }
            if m(LINK_MARKER_BOTTOM) {
                set(s + 8);
            }
        }
        if (s == 0 && m(LINK_MARKER_TOP_RIGHT)) || (s == 1 && m(LINK_MARKER_TOP)) || (s == 2 && m(LINK_MARKER_TOP_LEFT)) {
            set(5);
            set(16 + 6);
        }
        if (s == 2 && m(LINK_MARKER_TOP_RIGHT)) || (s == 3 && m(LINK_MARKER_TOP)) || (s == 4 && m(LINK_MARKER_TOP_LEFT)) {
            set(6);
            set(16 + 5);
        }
        // (an extra monster zone: down to the main zones of its columns' sides, up to the
        // opponent's, as seen from it)
        let first = match s {
            5 => Some(0),
            6 => Some(2),
            _ => None,
        };
        if let Some(b) = first {
            for (k, dir) in [LINK_MARKER_BOTTOM_LEFT, LINK_MARKER_BOTTOM, LINK_MARKER_BOTTOM_RIGHT].into_iter().enumerate() {
                if m(dir) {
                    set(b + k as i32);
                }
            }
            for (k, dir) in [LINK_MARKER_TOP_LEFT, LINK_MARKER_TOP, LINK_MARKER_TOP_RIGHT].into_iter().enumerate() {
                if m(dir) {
                    set(16 + 4 - b - k as i32);
                }
            }
        }
    } else if loc == LOCATION_SZONE && s <= 4 {
        if s > 0 && m(LINK_MARKER_LEFT) {
            set(s + 7);
        }
        if s > 0 && m(LINK_MARKER_TOP_LEFT) {
            set(s - 1);
        }
        if s < 4 && m(LINK_MARKER_RIGHT) {
            set(s + 9);
        }
        if s < 4 && m(LINK_MARKER_TOP_RIGHT) {
            set(s + 1);
        }
        if m(LINK_MARKER_TOP) {
            set(s);
        }
    }
    zones
}

/// A zone's bit (as `linked_zones`) for player `p`.
fn zone_bit(p: u32, [owner, loc, seq]: [u32; 3]) -> u32 {
    1u32 << ((if owner == p { 0 } else { 16 }) + if loc == LOCATION_MZONE { seq } else { seq + 8 })
}

/// The column (0-4, from player `me`'s side, left to right) of a zone, if it has one.
fn column(me: u32, [owner, loc, seq]: [u32; 3]) -> Option<u32> {
    let own = owner == me;
    match (loc, seq) {
        (_, 0..=4) => Some(if own { seq } else { 4 - seq }),
        (LOCATION_MZONE, 5) => Some(if own { 1 } else { 3 }),
        (LOCATION_MZONE, 6) => Some(if own { 3 } else { 1 }),
        _ => None,
    }
}

/// The features of each zone of the zone selection `prompt` (`place_applies`; `available_places`'
/// order, the rules' answer first) with the select hint `hint` (the card placed: its code).
pub fn place_features(d: &Duel, prompt: &Msg, hint: Option<u64>) -> Vec<[f32; PLACE_FEATURES]> {
    let (Msg::SelectPlace { player, count, flag } | Msg::SelectDisfield { player, count, flag }) = prompt else { return Vec::new() };
    let me = *player;
    let places = available_places(me, *flag);
    let at = |[owner, loc, seq]: [u32; 3]| -> Option<CardId> {
        let p = &d.player[owner as usize];
        let list = if loc == LOCATION_MZONE { &p.list_mzone } else { &p.list_szone };
        list.get(seq as usize).copied().flatten()
    };
    let faceup = |c: CardId| d.card(c).current.position & POS_FACEUP != 0;
    // (the face-up link cards' arrows: which zones each player's point to, as `zone_bit` for them)
    let mut arrows: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    for p in 0..2u32 {
        for (loc, n) in [(LOCATION_MZONE, 7), (LOCATION_SZONE, 5)] {
            for seq in 0..n {
                if let Some(c) = at([p, loc, seq]).filter(|&c| faceup(c) && d.card(c).data.type_ & TYPE_LINK != 0) {
                    arrows[p as usize].push(linked_zones(loc, seq, d.card(c).data.link_marker));
                }
            }
        }
    }
    let pointed = |p: u32, z: [u32; 3]| arrows[p as usize].iter().filter(|&&a| a & zone_bit(p, z) != 0).count() as f32;
    // the card placed
    let placed = hint.filter(|&h| h > 10000).and_then(|h| crate::heuristic::card(h as u32));
    let ty = placed.map_or(0, |c| c.type_);
    let marker = placed.filter(|_| ty & TYPE_LINK != 0).map_or(0, |c| c.link_marker);
    let mut shared = Vec::with_capacity(32);
    shared.push(placed.is_some() as u32 as f32);
    for t in [TYPE_MONSTER, TYPE_SPELL, TYPE_TRAP, TYPE_LINK, TYPE_PENDULUM, TYPE_FIELD, TYPE_CONTINUOUS, TYPE_EQUIP] {
        shared.push((ty & t != 0) as u32 as f32);
    }
    shared.push((ty & (TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK) != 0) as u32 as f32);
    shared.push(placed.filter(|_| ty & TYPE_LINK != 0).map_or(0.0, |c| c.level as f32 / 6.0));
    shared.extend(MARKERS.iter().map(|&b| (marker & b != 0) as u32 as f32));
    let mons = |p: u32| d.player[p as usize].list_mzone.iter().flatten().count() as f32 / 5.0;
    shared.extend([*count as f32 / 3.0, matches!(prompt, Msg::SelectDisfield { .. }) as u32 as f32, places.len() as f32 / 10.0, (d.infos.turn_player as u32 == me) as u32 as f32]);
    shared.extend(PHASES.iter().map(|p| (d.infos.phase as u32 & p != 0) as u32 as f32));
    shared.extend([mons(me), mons(1 - me), d.player[me as usize].list_hand.len() as f32 / 10.0]);

    let n = places.len();
    let mut out = Vec::with_capacity(n);
    for (i, &z) in places.iter().enumerate() {
        let [owner, loc, seq] = z;
        let own = owner == me;
        let mut x = [0f32; PLACE_FEATURES];
        let mut f = 0;
        let mut put = |v: f32| {
            x[f] = v;
            f += 1;
        };
        let b = |v: bool| v as u32 as f32;
        // the zone
        put(b(own));
        put(b(!own));
        put(b(loc == LOCATION_MZONE && seq < 5));
        put(b(loc == LOCATION_MZONE && seq >= 5));
        put(b(loc == LOCATION_SZONE && seq < 5));
        put(b(loc == LOCATION_SZONE && seq >= 5));
        let col = column(me, z);
        (0..5).for_each(|c| put(b(col == Some(c))));
        // what is in it (a zone to disable)
        let here = at(z);
        put(b(here.is_some()));
        put(b(here.is_some_and(faceup)));
        // its column (not the zone itself): each side's monster and spell or trap, face-up, and the extra monster zone's
        let occupant = |z2: [u32; 3]| if z2 == z { None } else { at(z2) };
        let mut column_cards = [0f32; 4];
        let mut column_up = [0f32; 4];
        if let Some(c) = col {
            for (k, (p, l)) in [(me, LOCATION_MZONE), (me, LOCATION_SZONE), (1 - me, LOCATION_MZONE), (1 - me, LOCATION_SZONE)].into_iter().enumerate() {
                let s = if p == me { c } else { 4 - c };
                if let Some(o) = occupant([p, l, s]) {
                    column_cards[k] = 1.0;
                    column_up[k] = b(faceup(o));
                }
            }
        }
        column_cards.into_iter().for_each(&mut put);
        column_up.into_iter().for_each(&mut put);
        let emz = |p: u32| match col {
            Some(1) => occupant([p, LOCATION_MZONE, if p == me { 5 } else { 6 }]),
            Some(3) => occupant([p, LOCATION_MZONE, if p == me { 6 } else { 5 }]),
            _ => None,
        };
        put(b(emz(me).is_some()));
        put(b(emz(1 - me).is_some()));
        // its neighbours in the row
        let side = |s: i64| (0..5).contains(&s).then(|| at([owner, loc, s as u32])).flatten();
        let (left, right) = if seq < 5 { (side(seq as i64 - 1), side(seq as i64 + 1)) } else { (None, None) };
        put(b(left.is_some()) + b(right.is_some()));
        put(b(left.is_some_and(|c| d.card(c).data.type_ & TYPE_LINK != 0)) + b(right.is_some_and(|c| d.card(c).data.type_ & TYPE_LINK != 0)));
        // the link arrows pointing to it
        put(pointed(me, z) / 2.0);
        put(pointed(1 - me, z) / 2.0);
        put(b(pointed(owner, z) > 0.0));
        // the placed link card's arrows from it (relative to the zone's owner): to its monsters,
        // to its link monsters that point back, to its free main monster zones, to its spell and
        // trap zones, to the other player's zones, off the field
        let mut placed_arrows = [0f32; 9];
        if marker != 0 && loc == LOCATION_MZONE {
            // (an arrow up into an extra monster zone: its bits for both players, counted once)
            let mut to = linked_zones(loc, seq, marker);
            to &= !(((to >> 5) & 1) << 22 | ((to >> 6) & 1) << 21);
            let from_here = zone_bit(owner, z);
            for t in 0..32u32 {
                if to & (1 << t) == 0 {
                    continue;
                }
                let target = [if t < 16 { owner } else { 1 - owner }, if t % 16 < 8 { LOCATION_MZONE } else { LOCATION_SZONE }, t % 8];
                let card = at(target);
                let mutual = card.filter(|&c| faceup(c) && d.card(c).data.type_ & TYPE_LINK != 0).is_some_and(|c| {
                    linked_zones(target[1], target[2], d.card(c).data.link_marker) & if target[0] == owner { from_here } else { from_here.rotate_left(16) } != 0
                });
                if t < 8 {
                    if card.is_some() {
                        placed_arrows[0] += 1.0;
                        placed_arrows[1] += b(mutual);
                    } else if t < 5 {
                        placed_arrows[2] += 1.0;
                    } else {
                        placed_arrows[3] += 1.0;
                    }
                } else if t < 16 {
                    placed_arrows[4] += 1.0;
                } else {
                    placed_arrows[5] += 1.0;
                    placed_arrows[6] += b(card.is_some());
                }
            }
            let markers = marker.count_ones() as f32;
            placed_arrows[7] = (markers - to.count_ones() as f32).max(0.0);
            placed_arrows[8] = 1.0;
        }
        placed_arrows.iter().take(8).for_each(|&v| put(v / 3.0));
        put(placed_arrows[8]);
        // the rules' rank (the first is their answer)
        put(i as f32 / (n - 1).max(1) as f32);
        put(b(i == 0));
        for &v in &shared {
            put(v);
        }
        debug_assert_eq!(f, PLACE_FEATURES);
        out.push(x);
    }
    out
}
