//! The imitation policy's inputs (scripts/policy/train.py): a decision as its player sees it, as
//! rows of numbers. bot-policy-data writes them for recorded games, `imitation::ImitationBot`
//! computes them while it plays; nothing hidden from the player goes in.
//!
//! A decision: a header (the `H_*` columns), a token per card (`T_*`): the cards of both players'
//! hands, Monster and Spell/Trap Zones, GYs and banishment, the player's own Extra Deck and the
//! opponent's face-up one, Xyz materials, the chain's links, the player's own Main Deck as it
//! knows it (its deck list less the cards it has seen elsewhere: a token per code with the number
//! of copies) and this turn's latest public actions (activations and summons); cards of a prompt
//! that are none of these (cards in a deck) get a token of their own. And a row per option
//! (`O_*`). Card codes are a game's code indices (`Codes`).

use std::collections::{BTreeMap, HashMap};

use ygo_core::constants::*;

use ygo_arbiter::actions::{Command, Response, available_places, battle_options, idle_options};
use ygo_arbiter::messages::{Card, Msg};
use ygo_arbiter::observe::filter_message;
use ygo_arbiter::state::{CardQuery, Snapshot};

pub use policynet::cols::*;

// T_LOC values
pub const L_HAND: i16 = 1;
pub const L_MZONE: i16 = 2;
pub const L_SZONE: i16 = 3;
pub const L_GRAVE: i16 = 4;
pub const L_REMOVED: i16 = 5;
pub const L_EXTRA: i16 = 6;
pub const L_OVERLAY: i16 = 7;
pub const L_DECK: i16 = 8;
pub const L_CHAIN: i16 = 9;
/// a public action of this turn (T_SEQ its kind, `E_*`; T_COUNT 1 for the latest)
pub const L_EVENT: i16 = 10;

pub const E_ACTIVATE: i16 = 1;
pub const E_SUMMON: i16 = 2;
pub const E_SPSUMMON: i16 = 3;
pub const E_FLIPSUMMON: i16 = 4;
/// The events kept (the latest).
pub const MAX_EVENTS: usize = 16;

/// A text reference to a card itself (the string number of `Codes::text`).
pub const S_CARD: i64 = 16;

/// The status bits a token keeps (bit i of T_STATUS is STATUS_BITS[i]).
pub const STATUS_BITS: [u32; 8] = [
    STATUS_DISABLED,
    STATUS_SUMMON_TURN,
    STATUS_SET_TURN,
    STATUS_SPSUMMON_TURN,
    STATUS_FLIP_SUMMON_TURN,
    STATUS_FORBIDDEN,
    STATUS_CHAINING,
    STATUS_BATTLE_DESTROYED,
];

/// Option kinds (O_KIND); the idle commands first, in `Command::IDLE`'s order.
pub const K_IDLE: i32 = 0; // + Command::IDLE's position: 0..9
pub const K_BATTLE: i32 = 9; // + Command::BATTLE's position: 9..13
pub const K_CHAIN_PASS: i32 = 13;
pub const K_CHAIN: i32 = 14;
pub const K_YES: i32 = 15;
pub const K_NO: i32 = 16;
pub const K_OPTION: i32 = 17;
pub const K_CARD: i32 = 18;
pub const K_SELECT: i32 = 19;
pub const K_UNSELECT: i32 = 20;
pub const K_FINISH: i32 = 21;
pub const K_PLACE: i32 = 22;
pub const K_POSITION: i32 = 23;
pub const K_TRIBUTE: i32 = 24;
pub const K_SUM: i32 = 25;
pub const K_NUMBER: i32 = 26;
pub const K_ATTRIBUTE: i32 = 27;
pub const K_RACE: i32 = 28;

/// Prompt types (H_PROMPT), by `Msg::name`.
pub const PROMPTS: [&str; 17] = [
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

pub fn clamp16(v: i64) -> i16 {
    v.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

/// A game's code indices (index + 1; 0 is an unknown card).
#[derive(Default)]
pub struct Codes {
    index: HashMap<u32, i64>,
    pub codes: Vec<u32>,
}

impl Codes {
    pub fn of(&mut self, code: u32) -> i64 {
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
    pub fn text(&mut self, desc: u64) -> (i64, i64) {
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
pub struct Tokens {
    pub rows: Vec<[i16; TOK]>,
    /// (absolute player, location, sequence) -> token
    pub at: HashMap<(u32, u32, u32), usize>,
    /// (absolute player, monster zone sequence, material index) -> token
    pub materials: HashMap<(u32, u32, u32), usize>,
}

pub fn loc_kind(loc: u32) -> i16 {
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

pub fn card_token(codes: &mut Codes, c: &CardQuery, con: i16, loc: i16, seq: u32, me: u32) -> [i16; TOK] {
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
pub fn tokens(codes: &mut Codes, view: &Snapshot, me: u32, deck: &[u32], events: &[(u32, i16, u32)]) -> Tokens {
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
pub fn card_ref(tk: &mut Tokens, codes: &mut Codes, c: &Card, me: u32) -> i32 {
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

pub fn option(kind: i32, token: i32, aux: i64, desc: (i64, i64)) -> [i32; OPT] {
    let mut o = [0i32; OPT];
    o[O_KIND] = kind;
    o[O_TOKEN] = token;
    o[O_AUX] = aux as i32;
    o[O_DESC_CODE] = desc.0 as i32;
    o[O_DESC_STR] = desc.1 as i32;
    o
}

/// What an option does: a whole response, or a card (by its index in the prompt's list) or a
/// zone of a selection of several.
#[derive(Clone, Debug)]
pub enum Act {
    Respond(Response),
    Card(usize),
    Place([u32; 3]),
}

/// The options of the (filtered) prompt `m`, the ones in `resp` chosen (none for
/// `Response::default()`), and what each does.
pub fn options(m: &Msg, resp: &Response, tk: &mut Tokens, codes: &mut Codes, me: u32) -> Vec<([i32; OPT], Act)> {
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
                out.push((o, Act::Respond(r)));
            }
        }
        Msg::SelectChain { chains, forced, .. } => {
            if *forced == 0 {
                let mut o = option(K_CHAIN_PASS, -1, 0, (0, 0));
                o[O_CHOSEN] = (resp.index == Some(-1)) as i32;
                out.push((o, Act::Respond(Response::pass())));
            }
            for (i, c) in chains.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(K_CHAIN, tok, 0, c.desc.map_or((0, 0), |d| codes.text(d)));
                o[O_CHOSEN] = (resp.index == Some(i as i64)) as i32;
                out.push((o, Act::Respond(Response::index(i))));
            }
        }
        Msg::SelectEffectyn { .. } | Msg::SelectYesno { .. } => {
            for (kind, yes) in [(K_YES, true), (K_NO, false)] {
                let mut o = option(kind, -1, 0, (0, 0));
                o[O_CHOSEN] = (resp.yes == Some(yes)) as i32;
                out.push((o, Act::Respond(Response::yes(yes))));
            }
        }
        Msg::SelectOption { options: descs, .. } => {
            for (i, &d) in descs.iter().enumerate() {
                let mut o = option(K_OPTION, -1, i as i64, codes.text(d));
                o[O_CHOSEN] = (resp.index == Some(i as i64)) as i32;
                out.push((o, Act::Respond(Response::index(i))));
            }
        }
        Msg::SelectCard { cards, .. } | Msg::SelectTribute { cards, .. } => {
            let kind = if matches!(m, Msg::SelectCard { .. }) { K_CARD } else { K_TRIBUTE };
            let chosen = cards_chosen(cards.len());
            for (i, c) in cards.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(kind, tok, c.release_param.unwrap_or(0) as i64, (0, 0));
                o[O_CHOSEN] = chosen[i] as i32;
                out.push((o, Act::Card(i)));
            }
        }
        Msg::SelectSum { cards, .. } => {
            let chosen = cards_chosen(cards.len());
            for (i, c) in cards.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(K_SUM, tok, c.param.unwrap_or(0) as i64, (0, 0));
                o[O_CHOSEN] = chosen[i] as i32;
                out.push((o, Act::Card(i)));
            }
        }
        Msg::SelectUnselectCard { finishable, cancelable, select_cards, unselect_cards, .. } => {
            for (i, c) in select_cards.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(K_SELECT, tok, 0, (0, 0));
                o[O_CHOSEN] = (resp.finish != Some(true) && resp.index == Some(i as i64)) as i32;
                out.push((o, Act::Respond(Response::index(i))));
            }
            for (i, c) in unselect_cards.iter().enumerate() {
                let tok = card_ref(tk, codes, c, me);
                let mut o = option(K_UNSELECT, tok, 0, (0, 0));
                o[O_CHOSEN] = (resp.finish != Some(true) && resp.index == Some((select_cards.len() + i) as i64)) as i32;
                out.push((o, Act::Respond(Response::index(select_cards.len() + i))));
            }
            if *finishable != 0 || *cancelable != 0 {
                let mut o = option(K_FINISH, -1, 0, (0, 0));
                o[O_CHOSEN] = (resp.finish == Some(true)) as i32;
                out.push((o, Act::Respond(Response { finish: Some(true), ..Response::default() })));
            }
        }
        Msg::SelectPlace { player, flag, .. } | Msg::SelectDisfield { player, flag, .. } => {
            for [p, loc, seq] in available_places(*player, *flag) {
                let zone = ((p != me) as i64) * 16 + if loc == LOCATION_SZONE { 8 } else { 0 } + seq as i64;
                let tok = tk.at.get(&(p, loc, seq)).map_or(-1, |&i| i as i32);
                let mut o = option(K_PLACE, tok, zone, (0, 0));
                o[O_CHOSEN] = resp.places.iter().flatten().any(|&x| x == [p, loc, seq]) as i32;
                out.push((o, Act::Place([p, loc, seq])));
            }
        }
        Msg::SelectPosition { positions, .. } => {
            for b in [POS_FACEUP_ATTACK, POS_FACEDOWN_ATTACK, POS_FACEUP_DEFENSE, POS_FACEDOWN_DEFENSE] {
                if positions & b != 0 {
                    let mut o = option(K_POSITION, -1, b as i64, (0, 0));
                    o[O_CHOSEN] = (resp.position == Some(b as i64)) as i32;
                    out.push((o, Act::Respond(Response { position: Some(b as i64), ..Response::default() })));
                }
            }
        }
        Msg::AnnounceNumber { values, .. } => {
            for (i, &v) in values.iter().enumerate() {
                let mut o = option(K_NUMBER, -1, v as i64, (0, 0));
                o[O_CHOSEN] = (resp.index == Some(i as i64)) as i32;
                out.push((o, Act::Respond(Response::index(i))));
            }
        }
        Msg::AnnounceAttrib { count: 1, available, .. } => {
            for i in 0..32 {
                if available & (1 << i) != 0 {
                    let mut o = option(K_ATTRIBUTE, -1, i, (0, 0));
                    o[O_CHOSEN] = (resp.mask == Some(1 << i)) as i32;
                    out.push((o, Act::Respond(Response { mask: Some(1 << i), ..Response::default() })));
                }
            }
        }
        Msg::AnnounceRace { count: 1, available, .. } => {
            for i in 0..64 {
                if available & (1 << i) != 0 {
                    let mut o = option(K_RACE, -1, i, (0, 0));
                    o[O_CHOSEN] = (resp.mask == Some(1 << i)) as i32;
                    out.push((o, Act::Respond(Response { mask: Some(1 << i), ..Response::default() })));
                }
            }
        }
        _ => {}
    }
    out
}

/// What a player has seen that a decision's header keeps: the last selection hint, this turn's
/// public actions and its own previous choice.
#[derive(Clone)]
pub struct Seen {
    pub me: u32,
    hint: Option<u64>,
    /// (actor, E_*, code), the latest last
    events: Vec<(u32, i16, u32)>,
    /// the previous decision's chosen option: kind and code index (-1, 0: none)
    prev: (i64, i64),
}

impl Seen {
    pub fn new(me: u32) -> Seen {
        Seen { me, hint: None, events: Vec::new(), prev: (-1, 0) }
    }

    /// A message of the duel (unfiltered; the player's own view of it is taken here).
    pub fn message(&mut self, msg: &Msg) {
        if let Msg::Hint { hint_type, player, data } = msg
            && *hint_type == HINT_SELECTMSG
            && *player == self.me
        {
            self.hint = Some(*data);
        }
        if matches!(msg, Msg::NewPhase { .. } | Msg::NewTurn { .. }) {
            self.hint = None;
        }
        if matches!(msg, Msg::NewTurn { .. }) {
            self.events.clear();
        }
        let e = match filter_message(msg, self.me) {
            Some(Msg::Chaining { code, trigger_con, .. }) => (trigger_con, E_ACTIVATE, code),
            Some(Msg::Summoning { code, con, .. }) => (con, E_SUMMON, code),
            Some(Msg::Spsummoning { code, con, .. }) => (con, E_SPSUMMON, code),
            Some(Msg::Flipsummoning { code, con, .. }) => (con, E_FLIPSUMMON, code),
            _ => return,
        };
        self.events.push(e);
        if self.events.len() > MAX_EVENTS {
            self.events.remove(0);
        }
    }

    /// After the player's decision: its chosen option (if any).
    pub fn decided(&mut self, chosen: Option<(i64, i64)>) {
        if let Some(c) = chosen {
            self.prev = c;
        }
        self.hint = None;
    }
}

/// A decision's rows: the header (without the data's own columns: game, decision, labels,
/// offsets), the tokens and the options with what each does.
pub struct Decision {
    pub hdr: [i64; HDR],
    pub tok: Vec<[i16; TOK]>,
    pub opt: Vec<([i32; OPT], Act)>,
}

impl Decision {
    /// The option's (kind, code index) for `Seen::decided`.
    pub fn chosen_key(&self, i: usize) -> (i64, i64) {
        let o = &self.opt[i].0;
        let code = if o[O_TOKEN] >= 0 { self.tok[o[O_TOKEN] as usize][T_CODE] as i64 } else { 0 };
        (o[O_KIND] as i64, code)
    }
}

/// The decision of the player `seen.me` at the filtered prompt `fp`, with its view and deck list;
/// `resp` marks the chosen options (`Response::default()`: none). None for a prompt of none of
/// the PROMPTS.
pub fn decision(codes: &mut Codes, seen: &Seen, view: &Snapshot, fp: &Msg, resp: &Response, deck: &[u32]) -> Option<Decision> {
    let me = seen.me;
    let pid = PROMPTS.iter().position(|&n| n == fp.name())?;
    let mut tk = tokens(codes, view, me, deck, &seen.events);
    let o = options(fp, resp, &mut tk, codes, me);
    let mut h = [0i64; HDR];
    h[H_PLAYER] = me as i64;
    h[H_TURN] = view.turn as i64;
    h[H_TURN_PLAYER] = (view.turn_player != me) as i64;
    h[H_PHASE] = view.phase as i64;
    h[H_LP_ME] = view.lp[me as usize] as i64;
    h[H_LP_OPP] = view.lp[1 - me as usize] as i64;
    h[H_PROMPT] = pid as i64;
    let (min, max, flags, desc, code) = match fp {
        Msg::SelectCard { min, max, cancelable, .. } | Msg::SelectTribute { min, max, cancelable, .. } => (*min, *max, 2 * (*cancelable != 0) as i64, None, 0),
        Msg::SelectSum { min, max, .. } => (*min, *max, 0, None, 0),
        Msg::SelectUnselectCard { min, max, cancelable, finishable, .. } => (*min, *max, 2 * (*cancelable != 0) as i64 + 4 * (*finishable != 0) as i64, None, 0),
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
    (h[H_HINT_CODE], h[H_HINT_STR]) = seen.hint.map_or((0, 0), |d| codes.text(d));
    (h[H_DESC_CODE], h[H_DESC_STR]) = desc.map_or((0, 0), |d| codes.text(d));
    h[H_CODE] = codes.of(code);
    h[H_N_TOK] = tk.rows.len() as i64;
    h[H_N_OPT] = o.len() as i64;
    (h[H_PREV_KIND], h[H_PREV_CODE]) = seen.prev;
    let count = |p: u32, loc: u32| view.players[p as usize].pile(loc).iter().flatten().count() as i64;
    h[H_DECK_ME] = count(me, LOCATION_DECK);
    h[H_DECK_OPP] = count(1 - me, LOCATION_DECK);
    h[H_EXTRA_OPP] = view.players[1 - me as usize].extra.iter().flatten().filter(|c| c.code.unwrap_or(0) == 0).count() as i64;
    Some(Decision { hdr: h, tok: tk.rows, opt: o })
}
