//! Policy training data from recorded games (ygo-arbiter's records): each game is replayed, and
//! every decision with more than one option becomes a row of what its player sees (the view,
//! the filtered prompt) and the option it chose. Nothing hidden from the player is written.
//!
//!   bot-policy-data OUT_DIR RECORDS.jsonl [RECORDS.jsonl ...] [--jobs N] [--max-games N]
//!
//! OUT_DIR gets, per worker w, three little-endian arrays (see scripts/policy/data.py):
//! - `w.hdr`: a row of HDR i64 per decision (the `H_*` columns)
//! - `w.tok`: a row of TOK i16 per card token of a decision (the `T_*` columns); a decision's
//!   tokens are rows `H_TOK_START..H_TOK_START + H_N_TOK` of its worker's file
//! - `w.opt`: a row of OPT i32 per option (the `O_*` columns), the same way
//!
//! and `games.jsonl`: per game its number, file and line, both decks and `codes`, the card codes
//! a game's code indices (`T_CODE`, ...: index + 1, 0 for an unknown card) stand for.
//!
//! Tokens: the cards of both players' hands, Monster and Spell/Trap Zones, GYs and banishment,
//! the player's own Extra Deck and the opponent's face-up one, Xyz materials, the chain's links
//! the player's own Main Deck as it knows it (its deck list less the cards it has seen
//! elsewhere: a token per code with the number of copies) and this turn's latest public actions
//! (activations and summons). Cards of a prompt that are none of these (cards in a deck) get a
//! token of their own.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;
use ygo_core::constants::*;
use ygo_core::duel::Duel;

use ygo_arbiter::actions::{Command, Response, available_places, battle_options, encode, idle_options};
use ygo_arbiter::game::{GameRecord, init, run_until_prompt};
use ygo_arbiter::messages::{Card, Msg};
use ygo_arbiter::observe::{filter_message, filter_prompt, to_core_response};
use ygo_arbiter::state::{CardQuery, Snapshot, viewer_view};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// -- header columns (i64) ---------------------------------------------------------------------
const H_GAME: usize = 0;
const H_DECISION: usize = 1;
const H_PLAYER: usize = 2;
const H_TURN: usize = 3;
/// 0: the player's own turn
const H_TURN_PLAYER: usize = 4;
const H_PHASE: usize = 5;
const H_LP_ME: usize = 6;
const H_LP_OPP: usize = 7;
const H_PROMPT: usize = 8;
const H_MIN: usize = 9;
const H_MAX: usize = 10;
/// bit 0 forced (a chain), bit 1 cancelable, bit 2 finishable
const H_FLAGS: usize = 11;
/// the last selection hint (HINT_SELECTMSG) to the player: a card's string (code index, string
/// number) or a system string (0, its number)
const H_HINT_CODE: usize = 12;
const H_HINT_STR: usize = 13;
/// a yes/no prompt's question, the same way
const H_DESC_CODE: usize = 14;
const H_DESC_STR: usize = 15;
/// the search's depth and value (-1 and 0: not searched, the rules answered)
const H_DEPTH: usize = 16;
const H_VALUE: usize = 17;
/// for the player: 1 win, 0 draw or unfinished, -1 loss
const H_RESULT: usize = 18;
const H_TOK_START: usize = 19;
const H_N_TOK: usize = 20;
const H_OPT_START: usize = 21;
const H_N_OPT: usize = 22;
/// the player's previous chosen option (its kind, -1 none, and its card's code index)
const H_PREV_KIND: usize = 23;
const H_PREV_CODE: usize = 24;
const H_DECK_ME: usize = 25;
const H_DECK_OPP: usize = 26;
const H_EXTRA_OPP: usize = 27;
/// SELECT_POSITION's card (code index)
const H_CODE: usize = 28;
const HDR: usize = 29;

// -- token columns (i16) ----------------------------------------------------------------------
const T_CODE: usize = 0;
/// 0 the player, 1 the opponent
const T_CON: usize = 1;
/// 0 the player, 1 the opponent, 2 unknown
const T_OWNER: usize = 2;
const T_LOC: usize = 3;
const T_SEQ: usize = 4;
/// bit 0 face-up, bit 1 attack position, bit 2 public (revealed)
const T_POS: usize = 5;
/// -1: unknown or '?'
const T_ATK: usize = 6;
const T_DEF: usize = 7;
const T_BASE_ATK: usize = 8;
const T_BASE_DEF: usize = 9;
/// level, rank or link rating
const T_LEVEL: usize = 10;
const T_SCALE: usize = 11;
const T_COUNTERS: usize = 12;
const T_OVERLAYS: usize = 13;
/// another token of the decision: an Xyz material's monster, an Equip Card's monster, a chain
/// link's card (-1 none)
const T_REF: usize = 14;
/// `STATUS_BITS`' bits
const T_STATUS: usize = 15;
/// a deck summary's copies; a chain link's number (1 first)
const T_COUNT: usize = 16;
const TOK: usize = 17;

// T_LOC values
const L_HAND: i16 = 1;
const L_MZONE: i16 = 2;
const L_SZONE: i16 = 3;
const L_GRAVE: i16 = 4;
const L_REMOVED: i16 = 5;
const L_EXTRA: i16 = 6;
const L_OVERLAY: i16 = 7;
const L_DECK: i16 = 8;
const L_CHAIN: i16 = 9;
/// a public action of this turn (T_SEQ its kind, `E_*`; T_COUNT 1 for the latest)
const L_EVENT: i16 = 10;

const E_ACTIVATE: i16 = 1;
const E_SUMMON: i16 = 2;
const E_SPSUMMON: i16 = 3;
const E_FLIPSUMMON: i16 = 4;
/// The events kept (the latest).
const MAX_EVENTS: usize = 16;

/// A text reference to a card itself (the string number of `Codes::text`).
const S_CARD: i64 = 16;

/// The status bits a token keeps (bit i of T_STATUS is STATUS_BITS[i]).
const STATUS_BITS: [u32; 8] = [
    STATUS_DISABLED,
    STATUS_SUMMON_TURN,
    STATUS_SET_TURN,
    STATUS_SPSUMMON_TURN,
    STATUS_FLIP_SUMMON_TURN,
    STATUS_FORBIDDEN,
    STATUS_CHAINING,
    STATUS_BATTLE_DESTROYED,
];

// -- option columns (i32) ---------------------------------------------------------------------
const O_KIND: usize = 0;
/// the option's card (a token, -1 none)
const O_TOKEN: usize = 1;
/// a zone (player 0/1 * 16 + 8 for the Spell/Trap Zones + sequence), a position, a number, an
/// attribute or type bit, a sum's value
const O_AUX: usize = 2;
/// the option's text (an activation's effect, an option): code index and string number, or 0
/// and a system string
const O_DESC_CODE: usize = 3;
const O_DESC_STR: usize = 4;
/// 1: in the player's response
const O_CHOSEN: usize = 5;
const OPT: usize = 6;

/// Option kinds (O_KIND); the idle commands first, in `Command::IDLE`'s order.
const K_IDLE: i32 = 0; // + Command::IDLE's position: 0..9
const K_BATTLE: i32 = 9; // + Command::BATTLE's position: 9..13
const K_CHAIN_PASS: i32 = 13;
const K_CHAIN: i32 = 14;
const K_YES: i32 = 15;
const K_NO: i32 = 16;
const K_OPTION: i32 = 17;
const K_CARD: i32 = 18;
const K_SELECT: i32 = 19;
const K_UNSELECT: i32 = 20;
const K_FINISH: i32 = 21;
const K_PLACE: i32 = 22;
const K_POSITION: i32 = 23;
const K_TRIBUTE: i32 = 24;
const K_SUM: i32 = 25;
const K_NUMBER: i32 = 26;
const K_ATTRIBUTE: i32 = 27;
const K_RACE: i32 = 28;

/// Prompt types (H_PROMPT), by `Msg::name`.
const PROMPTS: [&str; 17] = [
    "SELECT_IDLECMD",
    "SELECT_BATTLECMD",
    "SELECT_CHAIN",
    "SELECT_EFFECTYN",
    "SELECT_YESNO",
    "SELECT_OPTION",
    "SELECT_CARD",
    "SELECT_UNSELECT_CARD",
    "SELECT_PLACE",
    "SELECT_DISFIELD",
    "SELECT_POSITION",
    "SELECT_TRIBUTE",
    "SELECT_SUM",
    "ANNOUNCE_NUMBER",
    "ANNOUNCE_ATTRIB",
    "ANNOUNCE_RACE",
    "SELECT_COUNTER",
];

fn clamp16(v: i64) -> i16 {
    v.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

/// A game's code indices (index + 1; 0 is an unknown card).
#[derive(Default)]
struct Codes {
    index: HashMap<u32, i64>,
    codes: Vec<u32>,
}

impl Codes {
    fn of(&mut self, code: u32) -> i64 {
        if code == 0 {
            return 0;
        }
        *self.index.entry(code).or_insert_with(|| {
            self.codes.push(code);
            self.codes.len() as i64
        })
    }

    /// A text reference (a desc or a hint): a card's string (code index, string number), a card
    /// itself (code index, S_CARD) or a system string (0, its number).
    fn text(&mut self, desc: u64) -> (i64, i64) {
        if desc >= 1 << 32 {
            (self.of((desc >> 20) as u32), (desc & 0xfffff) as i64)
        } else if desc >= 1 << 20 {
            (self.of(desc as u32), S_CARD)
        } else {
            (0, desc as i64)
        }
    }
}

/// A decision's tokens, and where its cards are among them.
struct Tokens {
    rows: Vec<[i16; TOK]>,
    /// (absolute player, location, sequence) -> token
    at: HashMap<(u32, u32, u32), usize>,
    /// (absolute player, monster zone sequence, material index) -> token
    materials: HashMap<(u32, u32, u32), usize>,
}

fn loc_kind(loc: u32) -> i16 {
    match loc {
        LOCATION_HAND => L_HAND,
        LOCATION_MZONE => L_MZONE,
        LOCATION_SZONE => L_SZONE,
        LOCATION_GRAVE => L_GRAVE,
        LOCATION_REMOVED => L_REMOVED,
        LOCATION_EXTRA => L_EXTRA,
        LOCATION_DECK => L_DECK,
        _ => 0,
    }
}

fn card_token(codes: &mut Codes, c: &CardQuery, con: i16, loc: i16, seq: u32, me: u32) -> [i16; TOK] {
    let mut t = [0i16; TOK];
    t[T_CODE] = clamp16(codes.of(c.code.unwrap_or(0)));
    t[T_CON] = con;
    t[T_OWNER] = match c.owner {
        Some(o) => (o != me) as i16,
        None => 2,
    };
    t[T_LOC] = loc;
    t[T_SEQ] = seq as i16;
    let pos = c.position.unwrap_or(0);
    t[T_POS] = (pos & POS_FACEUP != 0) as i16 | ((pos & POS_ATTACK != 0) as i16) << 1 | ((c.is_public.unwrap_or(0) != 0) as i16) << 2;
    let stat = |v: Option<i32>| v.map_or(-1, |v| clamp16(v.max(-1) as i64));
    let known = c.code.is_some_and(|x| x != 0);
    t[T_ATK] = if known { stat(c.attack) } else { -1 };
    t[T_DEF] = if known { stat(c.defense) } else { -1 };
    t[T_BASE_ATK] = if known { stat(c.base_attack) } else { -1 };
    t[T_BASE_DEF] = if known { stat(c.base_defense) } else { -1 };
    let link = c.link.unwrap_or(0);
    let rank = c.rank.unwrap_or(0);
    t[T_LEVEL] = clamp16(if link != 0 { link } else if rank != 0 { rank } else { c.level.unwrap_or(0) } as i64);
    t[T_SCALE] = clamp16(c.lscale.unwrap_or(0) as i64);
    t[T_COUNTERS] = clamp16(c.counters.as_ref().map_or(0, |v| v.iter().map(|&x| (x >> 16) as i64).sum()));
    t[T_OVERLAYS] = c.overlays.as_ref().map_or(0, |o| o.len() as i16);
    t[T_REF] = -1;
    let status = c.status.unwrap_or(0);
    t[T_STATUS] = STATUS_BITS.iter().enumerate().fold(0, |a, (i, &b)| a | (((status & b != 0) as i16) << i));
    t
}

/// The tokens of `view` (the player `me`'s), with its own deck list `deck`.
fn tokens(codes: &mut Codes, view: &Snapshot, me: u32, deck: &[u32], events: &[(u32, i16, u32)]) -> Tokens {
    let mut tk = Tokens { rows: Vec::new(), at: HashMap::new(), materials: HashMap::new() };
    let mut equips: Vec<(usize, ygo_arbiter::messages::Loc)> = Vec::new();
    // the copies of its deck list the player has seen elsewhere
    let mut seen: BTreeMap<u32, i64> = BTreeMap::new();
    for rel in 0..2u32 {
        let p = if rel == 0 { me } else { 1 - me };
        let pl = &view.players[p as usize];
        for loc in [LOCATION_HAND, LOCATION_MZONE, LOCATION_SZONE, LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA] {
            for (seq, c) in pl.pile(loc).iter().enumerate() {
                let Some(c) = c else { continue };
                let code = c.code.unwrap_or(0);
                // (the opponent's Extra Deck: only its face-up cards; its count is in the header)
                if loc == LOCATION_EXTRA && rel == 1 && code == 0 {
                    continue;
                }
                if rel == 0 && code != 0 && c.owner == Some(me) && !(loc == LOCATION_EXTRA && c.position.unwrap_or(0) & POS_FACEUP == 0) {
                    *seen.entry(code).or_default() += 1;
                }
                let i = tk.rows.len();
                tk.rows.push(card_token(codes, c, rel as i16, loc_kind(loc), seq as u32, me));
                tk.at.insert((p, loc, seq as u32), i);
                if let Some(Some(e)) = c.equip_card {
                    equips.push((i, e));
                }
                for (k, &m) in c.overlays.iter().flatten().enumerate() {
                    if rel == 0 && m != 0 {
                        *seen.entry(m).or_default() += 1;
                    }
                    let mut t = [0i16; TOK];
                    t[T_CODE] = clamp16(codes.of(m));
                    t[T_CON] = rel as i16;
                    t[T_OWNER] = 2;
                    t[T_LOC] = L_OVERLAY;
                    t[T_SEQ] = k as i16;
                    t[T_POS] = 1 | 4;
                    t[T_ATK] = -1;
                    t[T_DEF] = -1;
                    t[T_BASE_ATK] = -1;
                    t[T_BASE_DEF] = -1;
                    t[T_REF] = i as i16;
                    tk.materials.insert((p, seq as u32, k as u32), tk.rows.len());
                    tk.rows.push(t);
                }
            }
        }
    }
    for (i, e) in equips {
        if let Some(&j) = tk.at.get(&(e.con, e.loc, e.seq)) {
            tk.rows[i][T_REF] = j as i16;
        }
    }
    for (n, link) in view.chain.iter().enumerate() {
        let mut t = [0i16; TOK];
        t[T_CODE] = clamp16(codes.of(link.code));
        t[T_CON] = (link.trigger_con != me) as i16;
        t[T_OWNER] = 2;
        t[T_LOC] = L_CHAIN;
        t[T_SEQ] = loc_kind(link.trigger_loc);
        t[T_POS] = 1 | 4;
        t[T_ATK] = -1;
        t[T_DEF] = -1;
        t[T_BASE_ATK] = -1;
        t[T_BASE_DEF] = -1;
        t[T_REF] = link.handler.and_then(|h| tk.at.get(&(h.con, h.loc, h.seq))).map_or(-1, |&j| j as i16);
        t[T_COUNT] = n as i16 + 1;
        tk.rows.push(t);
    }
    let mut left: BTreeMap<u32, i64> = BTreeMap::new();
    for &c in deck {
        *left.entry(c).or_default() += 1;
    }
    for (code, n) in left {
        let n = n - seen.get(&code).copied().unwrap_or(0);
        if n > 0 {
            let mut t = [0i16; TOK];
            t[T_CODE] = clamp16(codes.of(code));
            t[T_LOC] = L_DECK;
            t[T_POS] = 0;
            t[T_ATK] = -1;
            t[T_DEF] = -1;
            t[T_BASE_ATK] = -1;
            t[T_BASE_DEF] = -1;
            t[T_REF] = -1;
            t[T_COUNT] = n as i16;
            tk.rows.push(t);
        }
    }
    for (n, &(actor, kind, code)) in events.iter().rev().enumerate() {
        let mut t = [0i16; TOK];
        t[T_CODE] = clamp16(codes.of(code));
        t[T_CON] = (actor != me) as i16;
        t[T_OWNER] = 2;
        t[T_LOC] = L_EVENT;
        t[T_SEQ] = kind;
        t[T_POS] = 1 | 4;
        t[T_ATK] = -1;
        t[T_DEF] = -1;
        t[T_BASE_ATK] = -1;
        t[T_BASE_DEF] = -1;
        t[T_REF] = -1;
        t[T_COUNT] = n as i16 + 1;
        tk.rows.push(t);
    }
    tk
}

/// The token of a prompt's card (a new one for a card that has none, e.g. one in a deck).
fn card_ref(tk: &mut Tokens, codes: &mut Codes, c: &Card, me: u32) -> i32 {
    let found = if c.loc & LOCATION_OVERLAY != 0 {
        tk.materials.get(&(c.con, c.seq, c.pos.unwrap_or(0)))
    } else {
        tk.at.get(&(c.con, c.loc, c.seq))
    };
    if let Some(&i) = found {
        return i as i32;
    }
    let mut t = [0i16; TOK];
    t[T_CODE] = clamp16(codes.of(c.code));
    t[T_CON] = (c.con != me) as i16;
    t[T_OWNER] = 2;
    t[T_LOC] = loc_kind(c.loc & !LOCATION_OVERLAY);
    t[T_SEQ] = c.seq as i16;
    t[T_POS] = 4;
    t[T_ATK] = -1;
    t[T_DEF] = -1;
    t[T_BASE_ATK] = -1;
    t[T_BASE_DEF] = -1;
    t[T_REF] = -1;
    tk.rows.push(t);
    (tk.rows.len() - 1) as i32
}

fn option(kind: i32, token: i32, aux: i64, desc: (i64, i64)) -> [i32; OPT] {
    let mut o = [0i32; OPT];
    o[O_KIND] = kind;
    o[O_TOKEN] = token;
    o[O_AUX] = aux as i32;
    o[O_DESC_CODE] = desc.0 as i32;
    o[O_DESC_STR] = desc.1 as i32;
    o
}

/// The options of the (filtered) prompt `m`, the ones in `resp` chosen.
fn options(m: &Msg, resp: &Response, tk: &mut Tokens, codes: &mut Codes, me: u32) -> Vec<[i32; OPT]> {
    let mut out = Vec::new();
    let cards_chosen = |n: usize| -> Vec<bool> {
        let mut v = vec![false; n];
        for &i in resp.indices.iter().flatten() {
            if (i as usize) < n {
                v[i as usize] = true;
            }
        }
        v
    };
    match m {
        Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. } => {
            let (opts, base, cmds): (Vec<Response>, i32, &[Command]) = if matches!(m, Msg::SelectIdlecmd { .. }) {
                (idle_options(m), K_IDLE, &Command::IDLE)
            } else {
                (battle_options(m), K_BATTLE, &Command::BATTLE)
            };
            for r in opts {
                let cmd = r.kind.unwrap();
                let kind = base + cmds.iter().position(|&c| c == cmd).unwrap() as i32;
                let (tok, desc) = match (cmd.cards(m), r.index) {
                    (Some(list), Some(i)) => {
                        let c = &list[i as usize];
                        (card_ref(tk, codes, c, me), c.desc.map_or((0, 0), |d| codes.text(d)))
                    }
                    _ => (-1, (0, 0)),
                };
                let mut o = option(kind, tok, 0, desc);
                o[O_CHOSEN] = (resp.kind == r.kind && resp.index == r.index) as i32;
                out.push(o);
            }
        }
        Msg::SelectChain { chains, forced, .. } => {
            if *forced == 0 {
                let mut o = option(K_CHAIN_PASS, -1, 0, (0, 0));
                o[O_CHOSEN] = (resp.index == Some(-1)) as i32;
                out.push(o);
            }
            for (i, c) in chains.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(K_CHAIN, tok, 0, c.desc.map_or((0, 0), |d| codes.text(d)));
                o[O_CHOSEN] = (resp.index == Some(i as i64)) as i32;
                out.push(o);
            }
        }
        Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => {
            for (kind, yes) in [(K_YES, true), (K_NO, false)] {
                let mut o = option(kind, -1, 0, (0, 0));
                o[O_CHOSEN] = (resp.yes == Some(yes)) as i32;
                out.push(o);
            }
        }
        Msg::SelectOption { options: descs, .. } => {
            for (i, &d) in descs.iter().enumerate() {
                let mut o = option(K_OPTION, -1, i as i64, codes.text(d));
                o[O_CHOSEN] = (resp.index == Some(i as i64)) as i32;
                out.push(o);
            }
        }
        Msg::SelectCard { cards, .. } | Msg::SelectTribute { cards, .. } => {
            let kind = if matches!(m, Msg::SelectCard { .. }) { K_CARD } else { K_TRIBUTE };
            let chosen = cards_chosen(cards.len());
            for (i, c) in cards.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(kind, tok, c.release_param.unwrap_or(0) as i64, (0, 0));
                o[O_CHOSEN] = chosen[i] as i32;
                out.push(o);
            }
        }
        Msg::SelectSum { cards, .. } => {
            let chosen = cards_chosen(cards.len());
            for (i, c) in cards.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(K_SUM, tok, c.param.unwrap_or(0) as i64, (0, 0));
                o[O_CHOSEN] = chosen[i] as i32;
                out.push(o);
            }
        }
        Msg::SelectUnselectCard { finishable, cancelable, select_cards, unselect_cards, .. } => {
            for (i, c) in select_cards.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(K_SELECT, tok, 0, (0, 0));
                o[O_CHOSEN] = (resp.finish != Some(true) && resp.index == Some(i as i64)) as i32;
                out.push(o);
            }
            for (i, c) in unselect_cards.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(K_UNSELECT, tok, 0, (0, 0));
                o[O_CHOSEN] = (resp.finish != Some(true) && resp.index == Some((select_cards.len() + i) as i64)) as i32;
                out.push(o);
            }
            if *finishable != 0 || *cancelable != 0 {
                let mut o = option(K_FINISH, -1, 0, (0, 0));
                o[O_CHOSEN] = (resp.finish == Some(true)) as i32;
                out.push(o);
            }
        }
        Msg::SelectPlace { player, flag, .. } | Msg::SelectDisfield { player, flag, .. } => {
            for [p, loc, seq] in available_places(*player, *flag) {
                let zone = ((p != me) as i64) * 16 + if loc == LOCATION_SZONE { 8 } else { 0 } + seq as i64;
                let tok = tk.at.get(&(p, loc, seq)).map_or(-1, |&i| i as i32);
                let mut o = option(K_PLACE, tok, zone, (0, 0));
                o[O_CHOSEN] = resp.places.iter().flatten().any(|&x| x == [p, loc, seq]) as i32;
                out.push(o);
            }
        }
        Msg::SelectPosition { positions, .. } => {
            for b in [POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE] {
                if positions & b != 0 {
                    let mut o = option(K_POSITION, -1, b as i64, (0, 0));
                    o[O_CHOSEN] = (resp.position == Some(b as i64)) as i32;
                    out.push(o);
                }
            }
        }
        Msg::AnnounceNumber { values, .. } => {
            for (i, &v) in values.iter().enumerate() {
                let mut o = option(K_NUMBER, -1, v as i64, (0, 0));
                o[O_CHOSEN] = (resp.index == Some(i as i64)) as i32;
                out.push(o);
            }
        }
        Msg::AnnounceAttrib { count: 1, available, .. } => {
            for i in 0..32 {
                if available & (1 << i) != 0 {
                    let mut o = option(K_ATTRIBUTE, -1, i, (0, 0));
                    o[O_CHOSEN] = (resp.mask == Some(1 << i)) as i32;
                    out.push(o);
                }
            }
        }
        Msg::AnnounceRace { count: 1, available, .. } => {
            for i in 0..64 {
                if available & (1 << i) != 0 {
                    let mut o = option(K_RACE, -1, i, (0, 0));
                    o[O_CHOSEN] = (resp.mask == Some(1 << i)) as i32;
                    out.push(o);
                }
            }
        }
        _ => {}
    }
    out
}

/// A worker's output files.
struct Shard {
    hdr: std::io::BufWriter<std::fs::File>,
    tok: std::io::BufWriter<std::fs::File>,
    opt: std::io::BufWriter<std::fs::File>,
    n_tok: i64,
    n_opt: i64,
}

struct Game {
    number: usize,
    record: GameRecord,
}

/// Replays `g`, writing its decisions to `shard`; returns the game's codes.
fn game(g: &Game, shard: &mut Shard) -> Result<Vec<u32>, String> {
    let rec = &g.record;
    let labels: HashMap<usize, (u32, f64)> = rec.labels.iter().map(|l| (l.decision, (l.depth, l.value))).collect();
    let winner = rec.result.get("winner").and_then(|w| w.as_u64());
    let mut codes = Codes::default();
    // (the decks' codes first: every game's index table starts with its decks)
    for d in &rec.decks {
        for &c in d.main.iter().chain(&d.extra) {
            codes.of(c);
        }
    }
    let mut duel: Duel = rec.setup(None).map_err(|e| e.0)?;
    duel.ocg_start_duel();
    let mut prompt: Option<Msg> = None;
    let mut hint: [Option<u64>; 2] = [None, None];
    // (per viewer: this turn's public actions as it saw them, (actor, E_*, code))
    let mut events: [Vec<(u32, i16, u32)>; 2] = [Vec::new(), Vec::new()];
    let mut prev: [(i64, i64); 2] = [(-1, 0), (-1, 0)];
    let mut hdrs: Vec<[i64; HDR]> = Vec::new();
    let mut toks: Vec<[i16; TOK]> = Vec::new();
    let mut opts: Vec<[i32; OPT]> = Vec::new();
    for (i, entry) in rec.responses.iter().enumerate() {
        let step = run_until_prompt(&mut duel, prompt.as_ref()).map_err(|e| e.0)?;
        let p = step.prompt.ok_or_else(|| format!("the duel ended before response {i}"))?;
        let me = entry.player;
        for msg in &step.messages {
            if let Msg::Hint { hint_type, player, data } = msg {
                if *hint_type == HINT_SELECTMSG && *player < 2 {
                    hint[*player as usize] = Some(*data);
                }
            }
            if matches!(msg, Msg::NewPhase { .. } | Msg::NewTurn { .. }) {
                hint = [None, None];
            }
            if matches!(msg, Msg::NewTurn { .. }) {
                events = [Vec::new(), Vec::new()];
            }
            for (v, ev) in events.iter_mut().enumerate() {
                let e = match filter_message(msg, v as u32) {
                    Some(Msg::Chaining { code, trigger_con, .. }) => (trigger_con, E_ACTIVATE, code),
                    Some(Msg::Summoning { code, con, .. }) => (con, E_SUMMON, code),
                    Some(Msg::Spsummoning { code, con, .. }) => (con, E_SPSUMMON, code),
                    Some(Msg::Flipsummoning { code, con, .. }) => (con, E_FLIPSUMMON, code),
                    _ => continue,
                };
                ev.push(e);
                if ev.len() > MAX_EVENTS {
                    ev.remove(0);
                }
            }
        }
        let fp = filter_prompt(&p, me);
        let view = viewer_view(&mut duel, me);
        let mut tk = tokens(&mut codes, &view, me, &rec.decks[me as usize].main, &events[me as usize]);
        let o = options(&fp, &entry.response, &mut tk, &mut codes, me);
        let prompt_id = PROMPTS.iter().position(|&n| n == fp.name());
        let chosen: Vec<&[i32; OPT]> = o.iter().filter(|x| x[O_CHOSEN] != 0).collect();
        if let (Some(pid), true, false) = (prompt_id, o.len() >= 2, chosen.is_empty()) {
            let mut h = [0i64; HDR];
            h[H_GAME] = g.number as i64;
            h[H_DECISION] = i as i64;
            h[H_PLAYER] = me as i64;
            h[H_TURN] = view.turn as i64;
            h[H_TURN_PLAYER] = (view.turn_player != me) as i64;
            h[H_PHASE] = view.phase as i64;
            h[H_LP_ME] = view.lp[me as usize] as i64;
            h[H_LP_OPP] = view.lp[1 - me as usize] as i64;
            h[H_PROMPT] = pid as i64;
            let (min, max, flags, desc, code) = match &fp {
                Msg::SelectCard { min, max, cancelable, .. } | Msg::SelectTribute { min, max, cancelable, .. } => (*min, *max, 2 * (*cancelable != 0) as i64, None, 0),
                Msg::SelectSum { min, max, .. } => (*min, *max, 0, None, 0),
                Msg::SelectUnselectCard { min, max, cancelable, finishable, .. } => {
                    (*min, *max, 2 * (*cancelable != 0) as i64 + 4 * (*finishable != 0) as i64, None, 0)
                }
                Msg::SelectChain { forced, .. } => (0, 0, (*forced != 0) as i64, None, 0),
                Msg::SelectEffectyn { desc, code, .. } => (0, 0, 0, Some(*desc), *code),
                Msg::SelectYesno { desc, .. } => (0, 0, 0, Some(*desc), 0),
                Msg::SelectPlace { count, .. } | Msg::SelectDisfield { count, .. } => (*count, *count, 0, None, 0),
                Msg::SelectPosition { code, .. } => (0, 0, 0, None, *code),
                _ => (0, 0, 0, None, 0),
            };
            h[H_MIN] = min as i64;
            h[H_MAX] = max as i64;
            h[H_FLAGS] = flags;
            (h[H_HINT_CODE], h[H_HINT_STR]) = hint[me as usize].map_or((0, 0), |d| codes.text(d));
            (h[H_DESC_CODE], h[H_DESC_STR]) = desc.map_or((0, 0), |d| codes.text(d));
            h[H_CODE] = codes.of(code);
            (h[H_DEPTH], h[H_VALUE]) = labels.get(&i).map_or((-1, 0), |&(d, v)| (d as i64, v.round() as i64));
            h[H_RESULT] = match winner {
                Some(w) if w == me as u64 => 1,
                Some(w) if w < 2 => -1,
                _ => 0,
            };
            h[H_TOK_START] = shard.n_tok + toks.len() as i64;
            h[H_N_TOK] = tk.rows.len() as i64;
            h[H_OPT_START] = shard.n_opt + opts.len() as i64;
            h[H_N_OPT] = o.len() as i64;
            (h[H_PREV_KIND], h[H_PREV_CODE]) = prev[me as usize];
            let count = |p: u32, loc: u32| view.players[p as usize].pile(loc).iter().flatten().count() as i64;
            h[H_DECK_ME] = count(me, LOCATION_DECK);
            h[H_DECK_OPP] = count(1 - me, LOCATION_DECK);
            h[H_EXTRA_OPP] = view.players[1 - me as usize].extra.iter().flatten().filter(|c| c.code.unwrap_or(0) == 0).count() as i64;
            hdrs.push(h);
            toks.extend(&tk.rows);
            opts.extend(&o);
        }
        if let Some(c) = chosen.first() {
            let code = if c[O_TOKEN] >= 0 { tk.rows[c[O_TOKEN] as usize][T_CODE] as i64 } else { 0 };
            prev[me as usize] = (c[O_KIND] as i64, code);
        }
        hint[me as usize] = None;
        let core = to_core_response(&p, me, &entry.response).map_err(|e| format!("response {i}: {e:?}"))?;
        let r = encode(&p, &core).map_err(|e| format!("response {i}: {e:?}"))?;
        duel.ocg_set_response(&r);
        prompt = Some(p);
    }
    let w = |f: &mut std::io::BufWriter<std::fs::File>, bytes: &[u8]| f.write_all(bytes).map_err(|e| e.to_string());
    for h in &hdrs {
        for v in h {
            w(&mut shard.hdr, &v.to_le_bytes())?;
        }
    }
    for t in &toks {
        for v in t {
            w(&mut shard.tok, &v.to_le_bytes())?;
        }
    }
    for o in &opts {
        for v in o {
            w(&mut shard.opt, &v.to_le_bytes())?;
        }
    }
    shard.n_tok += toks.len() as i64;
    shard.n_opt += opts.len() as i64;
    Ok(codes.codes)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let usage = || -> ! {
        eprintln!("usage: bot-policy-data OUT_DIR RECORDS.jsonl [RECORDS.jsonl ...] [--jobs N] [--max-games N]");
        std::process::exit(2)
    };
    let (mut files, mut jobs, mut max_games) = (Vec::new(), 8usize, usize::MAX);
    let mut it = a.iter().skip(2);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--jobs" => jobs = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--max-games" => max_games = it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            _ => files.push(x.clone()),
        }
    }
    if a.len() < 3 || files.is_empty() {
        usage();
    }
    let out_dir = std::path::PathBuf::from(&a[1]);
    std::fs::create_dir_all(&out_dir).expect("output directory");
    init(None).unwrap();
    // (the games are read as the workers go: (file, line) in order, numbered as read)
    let lines = files.clone().into_iter().enumerate().flat_map(|(f, path)| {
        std::io::BufReader::new(std::fs::File::open(path).expect("records")).lines().enumerate().map(move |(l, s)| (f, l, s))
    });
    let source = Arc::new(Mutex::new((lines.take(max_games), 0usize)));
    let games_out = Arc::new(Mutex::new(std::io::BufWriter::new(std::fs::File::create(out_dir.join("games.jsonl")).expect("games.jsonl"))));
    let (done, failed) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let t0 = std::time::Instant::now();
    let threads: Vec<_> = (0..jobs)
        .map(|w| {
            let (source, games_out, done, failed, out_dir) = (source.clone(), games_out.clone(), done.clone(), failed.clone(), out_dir.clone());
            std::thread::spawn(move || {
                let create = |ext: &str| std::io::BufWriter::new(std::fs::File::create(out_dir.join(format!("{w}.{ext}"))).expect("shard"));
                let mut shard = Shard { hdr: create("hdr"), tok: create("tok"), opt: create("opt"), n_tok: 0, n_opt: 0 };
                loop {
                    let (number, file, line, text) = {
                        let mut s = source.lock().unwrap();
                        let Some((f, l, text)) = s.0.next() else { break };
                        s.1 += 1;
                        (s.1 - 1, f, l, text)
                    };
                    let Ok(text) = text else { continue };
                    let record: GameRecord = match serde_json::from_str(&text) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("game {number} (file {file} line {line}): {e}");
                            failed.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                    };
                    let g = Game { number, record };
                    match game(&g, &mut shard) {
                        Ok(codes) => {
                            let row = json!({"game": number, "file": file, "line": line, "worker": w, "decks": g.record.decks.iter().map(|d| json!({"main": d.main, "extra": d.extra})).collect::<Vec<_>>(), "codes": codes});
                            writeln!(games_out.lock().unwrap(), "{row}").expect("games.jsonl");
                        }
                        Err(e) => {
                            eprintln!("game {number} (file {file} line {line}): {e}");
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 1000 == 0 {
                        eprintln!("{n} games, {:.0} s", t0.elapsed().as_secs_f64());
                    }
                }
                shard.hdr.flush().unwrap();
                shard.tok.flush().unwrap();
                shard.opt.flush().unwrap();
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    games_out.lock().unwrap().flush().unwrap();
    let files_json = json!({"files": files, "hdr": HDR, "tok": TOK, "opt": OPT, "prompts": PROMPTS});
    std::fs::write(out_dir.join("meta.json"), files_json.to_string()).expect("meta.json");
    eprintln!("{} games, {} failed, {:.0} s", done.load(Ordering::Relaxed), failed.load(Ordering::Relaxed), t0.elapsed().as_secs_f64());
}
