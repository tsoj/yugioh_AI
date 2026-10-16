//! Deduction of card identities from one player's observation stream (belief/tracker.py).
//!
//! The tracker follows every card slot of both players through the observation stream, exactly
//! mirroring the core's list semantics (removal from a pile shifts the cards above it, draws take
//! the top of the deck, ...). The content of a slot is a [`Var`]: the identity of one physical
//! card between two shuffles.
//!
//! * Atoms are the physical cards the players start with (the main and extra deck). The viewer's
//!   own atoms have known codes; the opponent's are free.
//! * A shuffle turns the vars of a location into the members of a [`Pool`] and puts fresh vars
//!   ("claims") into the slots. Each claim is equal to exactly one member, but which one is
//!   unknown.
//! * Observations ("the card at hand slot 3 is X") pin the code of a var.
//!
//! The flow solver (belief::flow) then assigns every claim to a member and every free atom to a
//! code such that all observations hold. Vars and pools live in arenas (`vars`, `pools`) and refer
//! to each other by index; a var's id is its index.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use ygo_core::constants::*;

use crate::actions::Response;
use crate::game::{InitInfo, ObsItem};
use crate::messages::{Card, Msg};
use crate::observe::loc_visible;
use crate::state::{LOCATIONS, Snapshot};

pub type VarId = usize;
pub type PoolId = usize;

pub const LIST_LOCS: [u32; 5] = [LOCATION_DECK, LOCATION_HAND, LOCATION_GRAVE, LOCATION_REMOVED, LOCATION_EXTRA];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackerError(pub String);

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TrackerError {}

pub type Result<T> = std::result::Result<T, TrackerError>;

fn err<T>(s: impl Into<String>) -> Result<T> {
    Err(TrackerError(s.into()))
}

/// The index of a list location in `Tracker::lists[p]`.
pub fn list_index(loc: u32) -> Option<usize> {
    LIST_LOCS.iter().position(|&l| l == loc)
}

/// The number of zones of a zone location (None for other locations).
pub fn zone_count(loc: u32) -> Option<usize> {
    match loc {
        LOCATION_MZONE => Some(7),
        LOCATION_SZONE => Some(8),
        _ => None,
    }
}

/// A card sent from the field in place of a Tribute (e.g. with Floowandereeze and the
/// Unexplored Winds). Not destroyed cards or cards moved by the rules: those can be consequences
/// of the previous action.
pub fn leaves_field(m: &Msg) -> bool {
    let Msg::Move { from, to, reason, .. } = m else { return false };
    let field = LOCATION_MZONE | LOCATION_SZONE;
    from.loc & field != 0 && from.loc & LOCATION_OVERLAY == 0 && to.loc & field == 0 && reason & (REASON_DESTROY | REASON_RULE) == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// a main deck atom
    Main,
    /// an extra deck atom
    Extra,
    Claim,
    Created,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Main => "main",
            Kind::Extra => "extra",
            Kind::Claim => "claim",
            Kind::Created => "created",
        }
    }
    pub fn is_atom(self) -> bool {
        matches!(self, Kind::Main | Kind::Extra)
    }
}

/// The kind a hidden card must have (completion.fits_need): Set in a spell/trap zone, in the
/// field zone, in a monster zone with that many Tributes, summoned face-down.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Need {
    St,
    Field,
    Mset(u32),
    Monster,
}

impl Need {
    pub fn tag(self) -> &'static str {
        match self {
            Need::St => "st",
            Need::Field => "field",
            Need::Mset(_) => "mset",
            Need::Monster => "monster",
        }
    }
}

/// A card hint on a physical card: (hint type, value), or ("public", code) for a card kept
/// revealed in the hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mark {
    Hint(u32, u64),
    Public(u32),
}

#[derive(Debug, Clone)]
pub struct Var {
    pub id: VarId,
    pub owner: u32,
    pub kind: Kind,
    /// known code of an atom (viewer's own cards)
    pub fixed_code: Option<u32>,
    /// observed code
    pub code: Option<u32>,
    /// pool this var is a claim of
    pub pool: Option<PoolId>,
    /// member this claim resolved to
    pub assigned: Option<VarId>,
    pub faceup: bool,
    pub overlays: Vec<VarId>,
    /// final code after solving
    pub resolved: Option<u32>,
    /// codes this card is known not to have
    pub excluded: BTreeSet<u32>,
    /// the kind it must have
    pub need: Option<Need>,
    /// card hints on this physical card
    pub marks: BTreeSet<Mark>,
    /// claim: ids of the members it can be
    pub allowed: Option<BTreeSet<VarId>>,
    /// claim: an earlier slot that held the same physical card
    pub same_as: Option<VarId>,
    /// claim: earlier slots whose physical card this is not
    pub not_same: Vec<VarId>,
    /// claim: earlier slots whose physical card this preferably is
    pub prefer: Vec<VarId>,
    /// the owner a state view showed for this slot
    pub seen_owner: Option<u32>,
}

impl Var {
    /// The code if known (fixed or observed).
    pub fn known_code(&self) -> Option<u32> {
        self.fixed_code.or(self.code)
    }
    pub fn describe(&self) -> String {
        let o = |v: Option<u32>| v.map_or("None".to_string(), |c| c.to_string());
        format!("Var({},{},code={},fixed={})", self.id, self.kind.name(), o(self.code), o(self.fixed_code))
    }
}

/// Where a pool's cards were shuffled: a location, or "set" for shuffled set cards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolLoc {
    Loc(u32),
    Set,
}

#[derive(Debug, Clone)]
pub struct Pool {
    pub index: PoolId,
    pub player: u32,
    pub location: PoolLoc,
    pub members: Vec<VarId>,
    /// in slot order at creation time
    pub claims: Vec<VarId>,
    /// for set card shuffles: zone sequence per claim
    pub zones: Vec<u32>,
    /// ids of members assigned to a claim
    pub used: HashSet<VarId>,
}

/// A shuffle of a hypothesis: the codes in slot order, and which copies (ranks) of whose deck.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShuffleOutcome {
    pub player: u32,
    pub location: PoolLoc,
    pub codes: Vec<u32>,
    pub ranks: Option<Vec<u32>>,
    pub owners: Option<Vec<u32>>,
}

impl Default for PoolLoc {
    fn default() -> Self {
        PoolLoc::Loc(0)
    }
}

/// The ranks of a shuffle's cards (None: not ranked).
pub type Ranks = Option<Vec<u32>>;
/// The key of leaves_memo.
type LeavesKey = (PoolId, u32, usize, u64);

/// A deck card whose code the completion guessed (a wildcard atom): which physical copy (its
/// owner's copies of the code in creation order, as `Hypothesis::deck_ranks`) and what the
/// observations require of it.
#[derive(Debug, Clone)]
pub struct Guess {
    pub player: u32,
    pub extra: bool,
    pub code: u32,
    pub rank: u32,
    /// (a hidden card of a known kind)
    pub need: Option<Need>,
    /// codes the observations ruled out
    pub excluded: Vec<u32>,
}

/// A complete, consistent guess of the hidden information.
#[derive(Debug, Clone, Default)]
pub struct Hypothesis {
    /// per player (main, extra)
    pub decks: Vec<(Vec<u32>, Vec<u32>)>,
    /// initial main deck order, bottom first
    pub deck_orders: Vec<Vec<u32>>,
    /// initial extra deck order
    pub extra_orders: Vec<Vec<u32>>,
    /// in stream order
    pub shuffles: Vec<ShuffleOutcome>,
    /// codes per ascending zone, in stream order
    pub set_card_shuffles: Vec<Vec<u32>>,
    /// opponent atoms whose code was guessed
    pub unknown_atoms: usize,
    /// those atoms (`Guess`), and per player the deck cards whose codes are known (main, extra)
    pub guesses: Vec<Guess>,
    pub known_decks: Vec<(Vec<u32>, Vec<u32>)>,
    /// atom -> its rank (see `deck_ranks`)
    pub atom_ranks: HashMap<VarId, u32>,
    /// tracker var id -> code
    pub var_codes: HashMap<VarId, Option<u32>>,
    /// var ids fixed by observations
    pub pinned: HashSet<VarId>,
    /// var ids observed directly
    pub observed: HashSet<VarId>,
    /// which copy: rank of the card among its owner's copies of that code in the same section
    /// (= creation order), per initial main/extra deck slot
    pub deck_ranks: Vec<Option<Vec<u32>>>,
    pub extra_ranks: Vec<Option<Vec<u32>>>,
    /// (ranks, owners) per set card shuffle
    pub set_card_shuffle_ranks: Vec<(Ranks, Ranks)>,
}

type Sig = (&'static str, Option<u32>);

#[derive(Debug, Clone)]
pub struct Tracker {
    pub viewer: u32,
    pub vars: Vec<Var>,
    /// per player, per LIST_LOCS
    pub lists: [[Vec<VarId>; 5]; 2],
    /// per player: monster zones (7), spell/trap zones (8)
    pub mzones: [Vec<Option<VarId>>; 2],
    pub szones: [Vec<Option<VarId>>; 2],
    pub pools: Vec<Pool>,
    pub atoms: [Vec<VarId>; 2],
    pub observations: Vec<(VarId, u32)>,
    /// observations and kind requirements (code None) in stream order: the solver satisfies them
    /// in this order
    pub searched: Vec<(VarId, Option<u32>)>,
    pub exclusions: Vec<VarId>,
    /// hidden cards whose kind shows (e.g. Set as a spell/trap)
    pub needs: Vec<VarId>,
    /// (player, code) named by hints
    pub contains: BTreeSet<(u32, u32)>,
    /// the initial pools (decks), as indices
    pub initial_pools: Vec<PoolId>,
    /// shuffles in stream order
    pub shuffle_events: Vec<PoolId>,
    pub set_shuffle_events: Vec<PoolId>,
    pub turn_player: u32,
    resolving: bool,
    last_prompt: Option<Msg>,
    /// (var, con, code): the viewer's command's hand card
    chosen: Option<(Option<VarId>, u32, u32)>,
    resent_hints: bool,
    owners_seen: u64,
    owned_at: HashMap<VarId, u64>,
    owners_memo: HashMap<VarId, (u64, BTreeSet<u32>)>,
    leaves_memo: HashMap<LeavesKey, (Vec<VarId>, Vec<VarId>)>,
    pool_owners_memo: HashMap<(PoolId, u64), Vec<BTreeSet<u32>>>,
    confirmed: Vec<VarId>,
    referenced: Vec<VarId>,
    own_picks: Vec<(VarId, u32)>,
    /// (con, loc) of the confirmed cards, or [None] for a deck top / extra top confirmation
    confirmed_locs: Vec<Option<(u32, u32)>>,
    confirmed_shuffled: bool,
    confirmed_moves: usize,
    referenced_piles: BTreeSet<(u32, u32)>,
    picked: Vec<VarId>,
    referenced_in_chain: bool,
    referenced_link: usize,
    chain: Vec<Option<VarId>>,
    chain_from_hand: Vec<bool>,
    left_hand: Vec<VarId>,
    tributed_to_hand: Vec<VarId>,
    solved: Option<(VarId, u32, Vec<VarId>)>,
    last_sig: Option<Sig>,
    releases: u32,
    faceup_in_deck: Vec<VarId>,
    deck_arrivals: Vec<VarId>,
    faceup_left: Vec<(VarId, u32)>,
    prev_sig: Option<Sig>,
    last_move_to: Option<(u32, u32, u32)>,
    top_check: Option<u32>,
    top_check_move: Option<u32>,
    n_messages: u64,
    last_state: Option<(Snapshot, u64)>,
}

const COMMAND_LISTS: [(&str, &str); 4] = [("summon", "summonable"), ("spsummon", "spsummonable"), ("mset", "msetable"), ("sset", "ssetable")];

/// The card lists of a prompt in the order the tracker reads them (with their keys).
pub fn prompt_card_lists(p: &Msg) -> Vec<(&'static str, &[Card])> {
    use Msg::*;
    let mut out: Vec<(&'static str, &[Card])> = Vec::new();
    match p {
        SelectIdlecmd { summonable, spsummonable, repositionable, msetable, ssetable, activatable, .. } => {
            out.extend([
                ("summonable", &summonable[..]),
                ("spsummonable", &spsummonable[..]),
                ("repositionable", &repositionable[..]),
                ("msetable", &msetable[..]),
                ("ssetable", &ssetable[..]),
                ("activatable", &activatable[..]),
            ]);
        }
        SelectBattlecmd { activatable, attackable, .. } => out.extend([("activatable", &activatable[..]), ("attackable", &attackable[..])]),
        SelectChain { chains, .. } => out.push(("chains", chains)),
        SelectCard { cards, .. } | SelectTribute { cards, .. } | SortChain { cards, .. } | SortCard { cards, .. } | SelectCounter { cards, .. } => {
            out.push(("cards", cards))
        }
        SelectUnselectCard { select_cards, unselect_cards, .. } => out.extend([("select_cards", &select_cards[..]), ("unselect_cards", &unselect_cards[..])]),
        SelectSum { must_cards, cards, .. } => out.extend([("cards", &cards[..]), ("must_cards", &must_cards[..])]),
        _ => {}
    }
    out
}

/// A command prompt's list by key.
pub fn command_list<'a>(p: &'a Msg, key: &str) -> &'a [Card] {
    prompt_card_lists(p).into_iter().find(|(k, _)| *k == key).map_or(&[], |(_, l)| l)
}

fn ids_set(it: impl IntoIterator<Item = VarId>) -> BTreeSet<VarId> {
    it.into_iter().collect()
}

impl Tracker {
    pub fn new(viewer: u32, init: &InitInfo) -> Tracker {
        let mut t = Tracker {
            viewer,
            vars: Vec::new(),
            lists: Default::default(),
            mzones: [vec![None; 7], vec![None; 7]],
            szones: [vec![None; 8], vec![None; 8]],
            pools: Vec::new(),
            atoms: [Vec::new(), Vec::new()],
            observations: Vec::new(),
            searched: Vec::new(),
            exclusions: Vec::new(),
            needs: Vec::new(),
            contains: BTreeSet::new(),
            initial_pools: Vec::new(),
            shuffle_events: Vec::new(),
            set_shuffle_events: Vec::new(),
            turn_player: 0,
            resolving: false,
            last_prompt: None,
            chosen: None,
            resent_hints: false,
            owners_seen: 0,
            owned_at: HashMap::new(),
            owners_memo: HashMap::new(),
            leaves_memo: HashMap::new(),
            pool_owners_memo: HashMap::new(),
            confirmed: Vec::new(),
            referenced: Vec::new(),
            own_picks: Vec::new(),
            confirmed_locs: Vec::new(),
            confirmed_shuffled: false,
            confirmed_moves: 0,
            referenced_piles: BTreeSet::new(),
            picked: Vec::new(),
            referenced_in_chain: false,
            referenced_link: 0,
            chain: Vec::new(),
            chain_from_hand: Vec::new(),
            left_hand: Vec::new(),
            tributed_to_hand: Vec::new(),
            solved: None,
            last_sig: None,
            releases: 0,
            faceup_in_deck: Vec::new(),
            deck_arrivals: Vec::new(),
            faceup_left: Vec::new(),
            prev_sig: None,
            last_move_to: None,
            top_check: None,
            top_check_move: None,
            n_messages: 0,
            last_state: None,
        };
        let deck = list_index(LOCATION_DECK).unwrap();
        let extra = list_index(LOCATION_EXTRA).unwrap();
        for p in 0..2u32 {
            let (main_atoms, extra_atoms): (Vec<VarId>, Vec<VarId>) = if p == viewer {
                let mut main = init.deck.main.clone();
                main.sort();
                let extra_codes = if init.extra_order.is_empty() {
                    let mut e = init.deck.extra.clone();
                    e.sort();
                    e
                } else {
                    init.extra_order.clone()
                };
                (
                    main.into_iter().map(|c| t.new_var(p, Kind::Main, Some(c))).collect(),
                    extra_codes.into_iter().map(|c| t.new_var(p, Kind::Extra, Some(c))).collect(),
                )
            } else {
                (
                    (0..init.opponent_deck_size).map(|_| t.new_var(p, Kind::Main, None)).collect(),
                    (0..init.opponent_extra_size).map(|_| t.new_var(p, Kind::Extra, None)).collect(),
                )
            };
            t.atoms[p as usize] = main_atoms.iter().chain(&extra_atoms).copied().collect();
            let pool = t.make_pool(p, PoolLoc::Loc(LOCATION_DECK), main_atoms);
            t.lists[p as usize][deck] = t.pools[pool].claims.clone();
            if p == viewer {
                // the viewer knows the order of their own extra deck
                t.lists[p as usize][extra] = extra_atoms;
            } else {
                let pool = t.make_pool(p, PoolLoc::Loc(LOCATION_EXTRA), extra_atoms);
                t.lists[p as usize][extra] = t.pools[pool].claims.clone();
            }
        }
        t.initial_pools = (0..t.pools.len()).collect();
        t
    }

    // -- helpers -------------------------------------------------------------------------------

    fn new_var(&mut self, owner: u32, kind: Kind, fixed: Option<u32>) -> VarId {
        let id = self.vars.len();
        self.vars.push(Var {
            id,
            owner,
            kind,
            fixed_code: fixed,
            code: None,
            pool: None,
            assigned: None,
            faceup: false,
            overlays: Vec::new(),
            resolved: None,
            excluded: BTreeSet::new(),
            need: None,
            marks: BTreeSet::new(),
            allowed: None,
            same_as: None,
            not_same: Vec::new(),
            prefer: Vec::new(),
            seen_owner: None,
        });
        id
    }

    fn make_pool(&mut self, player: u32, location: PoolLoc, members: Vec<VarId>) -> PoolId {
        let index = self.pools.len();
        let mut claims = Vec::new();
        for &m in &members {
            let c = self.new_var(self.vars[m].owner, Kind::Claim, None);
            self.vars[c].faceup = self.vars[m].faceup;
            self.vars[c].pool = Some(index);
            claims.push(c);
        }
        self.pools.push(Pool { index, player, location, members, claims, zones: Vec::new(), used: HashSet::new() });
        index
    }

    /// (the number of cards revealed before the latest shuffle, the piles they were revealed in)
    pub fn referenced_state(&self) -> (usize, &BTreeSet<(u32, u32)>) {
        (self.referenced.len(), &self.referenced_piles)
    }

    pub fn var(&self, v: VarId) -> &Var {
        &self.vars[v]
    }

    fn code_of(&self, x: VarId) -> Option<u32> {
        self.vars[x].known_code()
    }

    /// The list of a location of a player (None for zones and unknown locations).
    pub fn list(&self, con: u32, loc: u32) -> Option<&Vec<VarId>> {
        list_index(loc).map(|i| &self.lists[con as usize][i])
    }

    fn list_mut(&mut self, con: u32, loc: u32) -> Option<&mut Vec<VarId>> {
        list_index(loc).map(|i| &mut self.lists[con as usize][i])
    }

    pub fn zones(&self, con: u32, loc: u32) -> &Vec<Option<VarId>> {
        if loc == LOCATION_MZONE { &self.mzones[con as usize] } else { &self.szones[con as usize] }
    }

    fn zones_mut(&mut self, con: u32, loc: u32) -> &mut Vec<Option<VarId>> {
        if loc == LOCATION_MZONE { &mut self.mzones[con as usize] } else { &mut self.szones[con as usize] }
    }

    pub fn observe(&mut self, var: Option<VarId>, code: u32) -> Result<()> {
        let Some(v) = var else { return Ok(()) };
        if code == 0 {
            return Ok(());
        }
        let x = &self.vars[v];
        if x.code.is_some_and(|c| c != code) {
            return err(format!("{} observed as {code} but was {}", x.describe(), x.code.unwrap()));
        }
        if x.fixed_code.is_some_and(|c| c != code) {
            return err(format!("{} observed as {code} but is {}", x.describe(), x.fixed_code.unwrap()));
        }
        if x.code.is_none() {
            self.vars[v].code = Some(code);
            self.observations.push((v, code));
            self.searched.push((v, Some(code)));
        }
        Ok(())
    }

    /// `var` is known not to be `code` (e.g. a scan for the first matching card passed over it).
    pub fn exclude(&mut self, var: Option<VarId>, code: u32) -> Result<()> {
        let Some(v) = var else { return Ok(()) };
        if code == 0 {
            return Ok(());
        }
        let x = &self.vars[v];
        if x.code == Some(code) || x.fixed_code == Some(code) {
            return err(format!("{} cannot be excluded from {code}", x.describe()));
        }
        self.vars[v].excluded.insert(code);
        self.exclusions.push(v);
        Ok(())
    }

    pub fn plain_var(&self, con: u32, loc: u32, seq: u32) -> Option<VarId> {
        if zone_count(loc).is_some() {
            return self.zones(con, loc).get(seq as usize).copied().flatten();
        }
        self.list(con, loc).and_then(|l| l.get(seq as usize).copied())
    }

    fn overlay_host(&self, con: u32, loc: u32, seq: u32) -> Result<VarId> {
        match self.plain_var(con, loc & !LOCATION_OVERLAY, seq) {
            Some(v) => Ok(v),
            None => err(format!("no xyz host at {con}/{loc:#x}/{seq}")),
        }
    }

    pub fn var_at(&self, con: u32, loc: u32, seq: u32, pos: u32) -> Result<Option<VarId>> {
        if loc & LOCATION_OVERLAY != 0 {
            let xyz = self.overlay_host(con, loc, seq)?;
            return match self.vars[xyz].overlays.get(pos as usize) {
                Some(&v) => Ok(Some(v)),
                None => err(format!("no overlay {pos} at {con}/{loc:#x}/{seq}")),
            };
        }
        if zone_count(loc).is_some() {
            return Ok(self.zones(con, loc).get(seq as usize).copied().flatten());
        }
        let Some(lst) = self.list(con, loc) else { return err(format!("bad location {loc:#x}")) };
        match lst.get(seq as usize) {
            Some(&v) => Ok(Some(v)),
            None => err(format!("no card at {con}/{loc:#x}/{seq} (size {})", lst.len())),
        }
    }

    fn take(&mut self, con: u32, loc: u32, seq: u32, pos: u32) -> Result<VarId> {
        if loc & LOCATION_OVERLAY != 0 {
            let xyz = self.overlay_host(con, loc, seq)?;
            if pos as usize >= self.vars[xyz].overlays.len() {
                return err(format!("no overlay at {con}/{loc:#x}/{seq}/{pos}"));
            }
            return Ok(self.vars[xyz].overlays.remove(pos as usize));
        }
        if zone_count(loc).is_some() {
            let z = self.zones_mut(con, loc);
            let Some(v) = z.get_mut(seq as usize).and_then(Option::take) else { return err(format!("empty zone at {con}/{loc:#x}/{seq}")) };
            return Ok(v);
        }
        let Some(lst) = self.list_mut(con, loc) else { return err(format!("bad location {loc:#x}")) };
        if seq as usize >= lst.len() {
            let n = lst.len();
            return err(format!("no card to take at {con}/{loc:#x}/{seq} (size {n})"));
        }
        Ok(lst.remove(seq as usize))
    }

    fn put(&mut self, v: VarId, con: u32, loc: u32, seq: u32, pos: u32) -> Result<()> {
        self.vars[v].faceup = if loc & LOCATION_OVERLAY != 0 { true } else { pos & POS_FACEUP != 0 };
        if loc & LOCATION_OVERLAY != 0 {
            let host = self.overlay_host(con, loc, seq)?;
            let ov = &mut self.vars[host].overlays;
            if pos as usize > ov.len() {
                ov.push(v);
            } else {
                ov.insert(pos as usize, v);
            }
            return Ok(());
        }
        if zone_count(loc).is_some() {
            let z = self.zones_mut(con, loc);
            match z.get_mut(seq as usize) {
                Some(slot @ None) => *slot = Some(v),
                Some(Some(_)) => return err(format!("zone occupied at {con}/{loc:#x}/{seq}")),
                None => return err(format!("bad zone {con}/{loc:#x}/{seq}")),
            }
            return Ok(());
        }
        let Some(lst) = self.list_mut(con, loc) else { return err(format!("bad location {loc:#x}")) };
        if seq as usize > lst.len() {
            let n = lst.len();
            return err(format!("bad insert position {con}/{loc:#x}/{seq} (size {n})"));
        }
        lst.insert(seq as usize, v);
        Ok(())
    }

    /// Narrows a claim's allowed members (to `ids` if that leaves any, else keeps them).
    fn narrow(&mut self, v: VarId, ids: BTreeSet<VarId>) {
        let x = &mut self.vars[v];
        x.allowed = Some(match x.allowed.take() {
            None => ids,
            Some(a) => {
                let i: BTreeSet<VarId> = a.intersection(&ids).copied().collect();
                if i.is_empty() { a } else { i }
            }
        });
    }

    /// Narrows a claim's allowed members to `ids` (possibly to none).
    fn narrow_strict(&mut self, v: VarId, ids: BTreeSet<VarId>) {
        let x = &mut self.vars[v];
        x.allowed = Some(match x.allowed.take() {
            None => ids,
            Some(a) => a.intersection(&ids).copied().collect(),
        });
    }

    fn is_claim_with_pool(&self, v: VarId) -> Option<PoolId> {
        let x = &self.vars[v];
        if x.kind == Kind::Claim { x.pool } else { None }
    }

    fn add_not_same(&mut self, v: VarId, x: VarId) {
        if !self.vars[v].not_same.contains(&x) {
            self.vars[v].not_same.push(x);
        }
    }

    // -- stream processing ---------------------------------------------------------------------

    pub fn process(&mut self, item: &ObsItem) -> Result<()> {
        match item {
            ObsItem::Msg(m) => self.message(m),
            ObsItem::Prompt(p, state) => {
                self.prompt(p)?;
                if let Some(s) = state {
                    self.state(s)?;
                }
                Ok(())
            }
            ObsItem::Response(r) => {
                self.response(r)?;
                Ok(())
            }
        }
    }

    fn require(&mut self, con: u32, loc: u32, seq: u32, pos: u32, need: Need) -> Result<()> {
        let v = self.var_at(con, loc, seq, pos)?;
        if let Some(v) = v {
            let x = &self.vars[v];
            if x.code.is_none() && x.fixed_code.is_none() && x.need.is_none() {
                self.vars[v].need = Some(need);
                self.needs.push(v);
                self.searched.push((v, None));
            }
        }
        Ok(())
    }

    /// Cards kept revealed in a hand (EFFECT_PUBLIC) stay revealed through a shuffle: if a
    /// shuffled hand shows the same public codes as were public before, each is one of the earlier
    /// public cards of its code.
    fn public_hand(&mut self, lst: &[VarId], cards: &[(u32, bool)]) {
        let public: Vec<(VarId, u32)> = lst.iter().zip(cards).filter(|(_, c)| c.1).map(|(&v, c)| (v, c.0)).collect();
        let pools: BTreeSet<PoolId> = public.iter().filter_map(|&(v, _)| self.is_claim_with_pool(v)).collect();
        if pools.len() == 1 {
            let pool = *pools.iter().next().unwrap();
            if public.iter().all(|&(v, _)| self.vars[v].pool == Some(pool)) {
                let carriers: Vec<(VarId, u32)> = self.pools[pool]
                    .members
                    .iter()
                    .flat_map(|&m| {
                        self.vars[m].marks.iter().filter_map(move |mk| if let Mark::Public(c) = mk { Some((m, *c)) } else { None })
                    })
                    .collect();
                let mut a: Vec<u32> = carriers.iter().map(|c| c.1).collect();
                let mut b: Vec<u32> = public.iter().map(|c| c.1).collect();
                a.sort();
                b.sort();
                if !carriers.is_empty() && a == b {
                    for &(v, code) in &public {
                        let same = ids_set(carriers.iter().filter(|c| c.1 == code).map(|c| c.0));
                        self.narrow_strict(v, same);
                    }
                }
            }
        }
        for (v, code) in public {
            self.vars[v].marks.insert(Mark::Public(code));
        }
    }

    /// Who can own the card in slot `v` (through the shuffles back to the decks).
    fn possible_owners(&mut self, v: VarId) -> BTreeSet<u32> {
        if let Some(hit) = self.owners_memo.get(&v)
            && (hit.0 == self.owners_seen || hit.1.len() == 1)
        {
            // (owners only narrow down as owners are seen: a single one stays)
            return hit.1.clone();
        }
        let mut out = BTreeSet::new();
        let (mut todo, mut seen) = (vec![v], HashSet::new());
        while let Some(x) = todo.pop() {
            if out.len() >= 2 {
                break;
            }
            let known = if x != v { self.owners_memo.get(&x) } else { None };
            let xv = &self.vars[x];
            if let Some(o) = xv.seen_owner {
                out.insert(o); // (shown by a state view: no need to look further)
            } else if let Some(k) = known.filter(|k| k.0 == self.owners_seen || k.1.len() == 1) {
                out.extend(k.1.iter().copied()); // (an earlier slot's answer, still valid)
            } else if let (Kind::Claim, Some(p)) = (xv.kind, xv.pool) {
                if seen.insert(p) {
                    todo.extend(self.pools[p].members.iter().copied());
                }
            } else {
                out.insert(xv.owner);
            }
        }
        self.owners_memo.insert(v, (self.owners_seen, out.clone()));
        out
    }

    /// The cards (slots of a single possible owner) the shuffles before slot `v` mixed: those of
    /// `owner` and those of the other player.
    fn owner_leaves(&mut self, v: VarId, owner: u32, limit: usize) -> (Vec<VarId>, Vec<VarId>) {
        // (the same for every slot of v's shuffle: the search starts at its members)
        let key = self.is_claim_with_pool(v).map(|p| (p, owner, limit, self.owners_seen));
        if let Some(k) = key
            && let Some(hit) = self.leaves_memo.get(&k)
        {
            return hit.clone();
        }
        let out = self.owner_leaves_search(v, owner, limit);
        if let Some(k) = key {
            self.leaves_memo.insert(k, out.clone());
        }
        out
    }

    fn owner_leaves_search(&mut self, v: VarId, owner: u32, limit: usize) -> (Vec<VarId>, Vec<VarId>) {
        let (mut mine, mut other) = (Vec::new(), Vec::new());
        let (mut todo, mut seen) = (vec![v], HashSet::new());
        while let Some(x) = todo.pop() {
            if !seen.insert(x) {
                continue;
            }
            if seen.len() > limit {
                return (Vec::new(), Vec::new());
            }
            let os = if x != v { self.possible_owners(x) } else { [0, 1].into_iter().collect() };
            if os.len() == 1 {
                if os.contains(&owner) { mine.push(x) } else { other.push(x) }
            } else if let Some(p) = self.is_claim_with_pool(x) {
                todo.extend(self.pools[p].members.iter().copied());
            } else {
                return (Vec::new(), Vec::new());
            }
        }
        (mine, other)
    }

    /// A card's owner shows even when it is hidden (a card given to the other player's hand keeps
    /// its sleeve): after a shuffle, the slot holds one of the shuffled cards that owner could own.
    fn owned(&mut self, v: Option<VarId>, owner: u32) {
        let Some(v) = v else { return };
        if self.vars[v].seen_owner != Some(owner) {
            // (news only if the card could have had another owner)
            let informative = self.possible_owners(v) != BTreeSet::from([owner]);
            self.vars[v].seen_owner = Some(owner);
            if informative {
                self.owners_seen += 1;
            }
        }
        let Some(pool) = self.is_claim_with_pool(v) else { return };
        if self.owned_at.get(&v) == Some(&self.owners_seen) {
            return;
        }
        self.owned_at.insert(v, self.owners_seen);
        let members = self.pools[pool].members.clone();
        // (the same for every slot of the shuffle)
        let key = (pool, self.owners_seen);
        let owners = match self.pool_owners_memo.get(&key) {
            Some(o) => o.clone(),
            None => {
                let o: Vec<BTreeSet<u32>> = members.iter().map(|&m| self.possible_owners(m)).collect();
                self.pool_owners_memo.insert(key, o.clone());
                o
            }
        };
        if owners.iter().all(|os| *os == owners[0]) {
            // (tells nothing directly) through earlier shuffles: not the only card of the other
            // owner among the cards the shuffles mixed
            let (_, other) = self.owner_leaves(v, owner, 200);
            if other.len() == 1 && !self.vars[v].not_same.contains(&other[0]) {
                self.vars[v].not_same.push(other[0]);
            }
            return;
        }
        let ids = ids_set(members.iter().zip(&owners).filter(|(_, os)| os.contains(&owner)).map(|(&m, _)| m));
        if !ids.is_empty() {
            self.narrow(v, ids);
        }
    }

    /// Card hints stick to a physical card; after a shuffle the core shows them again at the
    /// card's new position, which tells which copy went where.
    fn card_hint(&mut self, con: u32, loc: u32, seq: u32, pos: u32, hint_type: u32, value: u64) -> Result<()> {
        let Some(v) = self.var_at(con, loc, seq, pos)? else { return Ok(()) };
        let mark = Mark::Hint(hint_type, value);
        if hint_type == CHINT_DESC_REMOVE {
            self.vars[v].marks.remove(&Mark::Hint(CHINT_DESC_ADD, value));
            return Ok(());
        }
        if self.vars[v].marks.contains(&mark) {
            return Ok(());
        }
        if self.resent_hints
            && let Some(pool) = self.is_claim_with_pool(v)
            && pool == self.pools.len() - 1
        {
            let carriers = ids_set(self.pools[pool].members.iter().copied().filter(|&x| self.vars[x].marks.contains(&mark)));
            // (more slots showing it than cards that had it: not hints shown again after the
            // shuffle, but new ones)
            let shown: Vec<VarId> = self.pools[pool].claims.iter().copied().filter(|&x| self.vars[x].marks.contains(&mark) && x != v).collect();
            if !carriers.is_empty() && shown.len() < carriers.len() {
                self.narrow_strict(v, carriers);
            } else if !carriers.is_empty() {
                for x in shown {
                    if self.vars[x].allowed.as_ref() == Some(&carriers) {
                        self.vars[x].allowed = None;
                    }
                }
            }
        }
        self.vars[v].marks.insert(mark);
        Ok(())
    }

    pub fn message(&mut self, m: &Msg) -> Result<()> {
        self.n_messages += 1;
        let t = m.name();
        let mplayer = m.player();
        if let Some(p) = self.top_check.take()
            && !(t == "DECK_TOP" && mplayer == Some(p))
            && !self.faceup_in_deck.is_empty()
        {
            // a shuffled deck whose top card is face-up is announced (DECK_TOP): this one was
            // not, so its top card is none of the face-up ones
            let deck = &self.lists[p as usize][0];
            if let Some(&top) = deck.last()
                && self.vars[top].kind == Kind::Claim
            {
                for x in self.faceup_in_deck.clone() {
                    if x != top {
                        self.add_not_same(top, x);
                    }
                }
            }
        }
        if let Some(p) = self.top_check_move.take() {
            let from_deck = matches!(m, Msg::Move { from, .. } if from.loc == LOCATION_DECK && from.con == p);
            if !(t == "DECK_TOP" && mplayer == Some(p)) && !from_deck {
                // (the top card left the deck and the new top was not announced: it is none of
                // the face-up ones; not between the cards of one operation)
                self.not_faceup_top(p);
            }
        }
        if let Msg::ShuffleDeck { player } = m {
            self.top_check = Some(*player);
        }
        if !matches!(t, "HINT" | "CARD_HINT" | "CHAIN_SOLVED") {
            self.prev_sig = self.last_sig;
            self.last_sig = Some((t, mplayer));
        }
        if t != "CHAIN_SOLVED"
            && let Some((var, player, left)) = self.solved.take()
            && let Msg::ShuffleHand { player: sp, .. } = m
            && *sp == player
        {
            // the core shuffles the hand after a chain link activated from it only if the card is
            // still there face-down: the cards that left the hand were other cards
            for v in left {
                if v != var && self.vars[v].kind == Kind::Claim {
                    self.vars[v].not_same.push(var);
                }
            }
        }
        if matches!(t, "SUMMONING" | "SPSUMMONING" | "NEW_PHASE" | "CHAINING") {
            self.tributed_to_hand.clear();
        }
        if let Msg::NewTurn { player } = m {
            self.turn_player = *player;
        }
        let reset_refs = match m {
            Msg::ChainEnd {} | Msg::NewPhase { .. } | Msg::NewTurn { .. } => true,
            Msg::ChainSolved { chain_count } => (*chain_count as usize) <= self.referenced_link.max(1),
            Msg::Chaining { chain_count, .. } => *chain_count == 1,
            _ => false,
        };
        if reset_refs {
            // (cards revealed e.g. as a cost stay relevant until their chain link resolved; a new
            // chain starts afresh)
            self.referenced.clear();
            self.referenced_piles.clear();
            self.picked.clear();
            self.referenced_in_chain = false;
        }
        if let Msg::Move { to, .. } = m {
            self.last_move_to = Some((to.con, to.loc, to.seq));
        }
        if let Msg::Move { from, to, code, .. } = m
            && from.loc != 0
            && from.loc & LOCATION_OVERLAY == 0
        {
            let v = self.plain_var(from.con, from.loc, from.seq);
            if from.loc == LOCATION_HAND
                && let Some(v) = v
            {
                self.left_hand.push(v);
                if let Some((cv, ccon, ccode)) = self.chosen
                    && ccon == from.con
                    && ccode == *code
                    && (to.loc == LOCATION_MZONE || to.loc == LOCATION_SZONE)
                {
                    self.identify(Some(v), cv);
                    self.chosen = None;
                }
            }
        }
        let confirmed_move = if let Msg::Move { from, code, .. } = m {
            !self.confirmed.is_empty()
                && self.confirmed_shuffled
                && from.loc == LOCATION_DECK
                && self.confirmed_locs.iter().all(|x| *x == Some((from.con, LOCATION_DECK)))
                && (*code == 0 || self.confirmed.iter().any(|&x| self.code_of(x) == Some(*code)))
        } else {
            false
        };
        if confirmed_move {
            // revealed, shuffled back and then taken out of that pile, card after card: each card
            // taken is one of the revealed ones
            let Msg::Move { from, .. } = m else { unreachable!() };
            if let Some(v) = self.plain_var(from.con, from.loc, from.seq)
                && let Some(pool) = self.is_claim_with_pool(v)
            {
                let ids = ids_set(self.pools[pool].members.iter().copied().filter(|x| self.confirmed.contains(x)));
                if !ids.is_empty() {
                    self.narrow(v, ids);
                }
            }
            self.confirmed_moves += 1;
            if self.confirmed_moves >= self.confirmed.len() {
                self.confirmed.clear();
            }
        } else if !matches!(
            t,
            "CONFIRM_CARDS"
                | "CONFIRM_DECKTOP"
                | "CONFIRM_EXTRATOP"
                | "SHUFFLE_DECK"
                | "SHUFFLE_HAND"
                | "SHUFFLE_EXTRA"
                | "HINT"
                | "CARD_HINT"
                | "RANDOM_SELECTED"
                | "DECK_TOP"
                | "SPSUMMONING"
                | "SPSUMMONED"
                | "SUMMONING"
                | "SUMMONED"
        ) {
            self.confirmed.clear(); // only "reveal, shuffle back, pick one at random" counts
        }
        if t == "SHUFFLE_DECK" && !self.confirmed.is_empty() {
            self.confirmed_shuffled = true;
        }
        match m {
            Msg::CardHint { con, hint_type, value, .. } | Msg::PlayerHint { player: con, hint_type, value }
                if value >> 20 != 0 && (*hint_type == CHINT_DESC_ADD || *hint_type == CHINT_DESC_REMOVE) =>
            {
                // a hint text names the card whose script it comes from: that player's deck
                // contains the card
                self.contains.insert((*con, (value >> 20) as u32));
            }
            _ => {}
        }
        if let Msg::CardHint { con, loc, seq, pos, hint_type, value } = m {
            self.card_hint(*con, *loc, *seq, *pos, *hint_type, *value)?;
        } else {
            // field::shuffle shows the client hints of every hand card again
            self.resent_hints = t == "SHUFFLE_HAND";
        }
        let moved_release = match m {
            Msg::Move { reason, .. } => reason & (REASON_RELEASE | REASON_COST) != 0 || leaves_field(m),
            _ => false,
        };
        if moved_release {
            // (a card can also be sent instead of Tributed: it counts for the Level of a Set
            // monster)
            self.releases += 1;
        } else if !matches!(
            t,
            "MOVE" | "HINT" | "CARD_HINT" | "PLAYER_HINT" | "MISSED_EFFECT" | "SET" | "SPSUMMONING" | "ADD_COUNTER" | "REMOVE_COUNTER" | "FIELD_DISABLED"
        ) {
            self.releases = 0;
        }
        match m {
            Msg::Set { code: 0, con, loc, seq, pos } => {
                // a hidden card was Set: its kind follows from the zone (and tributes)
                if *loc == LOCATION_SZONE {
                    self.require(*con, *loc, *seq, *pos, if *seq == 5 { Need::Field } else { Need::St })?;
                } else if *loc == LOCATION_MZONE {
                    self.require(*con, *loc, *seq, *pos, Need::Mset(self.releases))?;
                }
                self.releases = 0;
            }
            Msg::Spsummoning { code: 0, con, loc, seq, pos } => self.require(*con, *loc, *seq, *pos, Need::Monster)?,
            _ => {}
        }
        self.message_effects(m)
    }

    fn message_effects(&mut self, m: &Msg) -> Result<()> {
        match m {
            Msg::Move { code, from, to, reason } => self.on_move(*code, from, to, *reason)?,
            Msg::Draw { player, cards } => {
                let p = *player;
                for c in cards {
                    let Some(v) = self.lists[p as usize][0].pop() else { return err("draw from empty deck") };
                    self.left_deck(Some(v));
                    self.vars[v].faceup = c.pos & POS_FACEUP != 0;
                    self.lists[p as usize][1].push(v);
                    self.observe(Some(v), c.code)?;
                }
                if self.prev_sig != Some(("DECK_TOP", Some(p))) {
                    // a draw that leaves a face-up card on top announces it: this one did not, so
                    // the new top card is none of the face-up ones
                    self.not_faceup_top(p);
                }
            }
            Msg::ShuffleDeck { player } => {
                let deck = self.lists[*player as usize][0].clone();
                self.deck_arrivals.retain(|x| !deck.contains(x));
                self.shuffle(*player, LOCATION_DECK);
            }
            Msg::ShuffleHand { player, codes, public, prev } => {
                let pool = self.shuffle(*player, LOCATION_HAND);
                let claims = self.pools[pool].claims.clone();
                let members = self.pools[pool].members.clone();
                if codes.len() != claims.len() {
                    return err("hand size mismatch on shuffle");
                }
                if let Some(prev) = prev.as_ref().filter(|p| !p.is_empty() && p.len() == claims.len()) {
                    // (its own hand: the owner knows where each card went)
                    for (&v, &i) in claims.iter().zip(prev) {
                        if (i as usize) < members.len() {
                            self.identify(Some(v), Some(members[i as usize]));
                        }
                    }
                }
                for (k, (&v, &code)) in claims.iter().zip(codes).enumerate() {
                    let pubk = public.get(k).copied().unwrap_or(false);
                    self.vars[v].faceup = pubk;
                    self.observe(Some(v), code)?;
                    if pubk {
                        // (a revealed card stays the same physical card: the one that was face-up
                        // in the hand before, if only one with its code was)
                        let before: Vec<VarId> = members
                            .iter()
                            .copied()
                            .filter(|&x| self.vars[x].faceup && (self.vars[x].code.filter(|&c| c != 0).or(self.vars[x].fixed_code)) == Some(code))
                            .collect();
                        if before.len() == 1 {
                            self.identify(Some(v), Some(before[0]));
                        }
                    }
                }
            }
            Msg::ShuffleExtra { player, codes, .. } => {
                let pool = self.shuffle(*player, LOCATION_EXTRA);
                for (&v, &code) in self.pools[pool].claims.clone().iter().zip(codes) {
                    self.observe(Some(v), code)?;
                }
            }
            Msg::ShuffleSetCard { old, new, .. } => {
                let mut vars_ = Vec::new();
                for li in old {
                    vars_.push(self.var_at(li.con, li.loc, li.seq, 0)?);
                }
                let members: Vec<VarId> = vars_.iter().map(|v| v.expect("a set card")).collect();
                let pool = self.make_pool(old[0].con, PoolLoc::Set, members.clone());
                let mut order: Vec<usize> = (0..old.len()).collect();
                order.sort_by_key(|&i| old[i].seq);
                self.pools[pool].zones = order.iter().map(|&i| old[i].seq).collect();
                let claims: Vec<VarId> = order.iter().map(|&i| self.pools[pool].claims[i]).collect();
                self.pools[pool].claims = claims.clone();
                for (&claim, &i) in claims.iter().zip(&order) {
                    self.vars[claim].overlays = self.vars[members[i]].overlays.clone();
                    self.vars[claim].faceup = false;
                }
                let zones = self.pools[pool].zones.clone();
                for (&claim, &zone) in claims.iter().zip(&zones) {
                    self.zones_mut(old[0].con, old[0].loc)[zone as usize] = Some(claim);
                    if let Some(new) = new.as_ref().filter(|n| !n.is_empty())
                        && let Some(k) = new.iter().position(|&z| z == zone)
                    {
                        // (its own cards: the controller knows which went where)
                        self.identify(Some(claim), Some(members[k]));
                    }
                }
                self.set_shuffle_events.push(pool);
            }
            Msg::Summoning { code, con, loc, seq, pos }
            | Msg::Spsummoning { code, con, loc, seq, pos }
            | Msg::Flipsummoning { code, con, loc, seq, pos }
            | Msg::Set { code, con, loc, seq, pos } => {
                let v = self.var_at(*con, *loc, *seq, *pos)?;
                self.observe(v, *code)?;
            }
            Msg::Chaining { code, con, loc, seq, pos, .. } => {
                let v = self.var_at(*con, *loc, *seq, *pos)?;
                self.observe(v, *code)?;
                self.chain.push(v);
                self.chain_from_hand.push(*loc == LOCATION_HAND);
            }
            Msg::ChainSolving { .. } => {
                self.chosen = None;
                self.left_hand.clear();
                self.resolving = true;
            }
            Msg::ChainSolved { .. } => {
                self.resolving = false;
                if let Some(var) = self.chain.pop() {
                    let from_hand = self.chain_from_hand.pop().unwrap_or(false);
                    if from_hand && let Some(var) = var {
                        let owner = self.vars[var].owner;
                        self.solved = Some((var, owner, self.left_hand.clone()));
                        if self.last_sig == Some(("SHUFFLE_HAND", Some(owner))) && !self.left_hand.is_empty() {
                            // proc_workaround.lua shuffles the hand on EVENT_CHAIN_SOLVED if the
                            // activating card moved while the link resolved: it is one of the
                            // cards that left the hand
                            let vc = self.vars[var].code;
                            let cands: Vec<VarId> = self
                                .left_hand
                                .iter()
                                .copied()
                                .filter(|&v| self.vars[v].code.is_none() || vc.is_none() || self.vars[v].code == vc)
                                .collect();
                            if cands.len() == 1 {
                                self.identify(Some(cands[0]), Some(var));
                            }
                        }
                    }
                }
                self.left_hand.clear();
            }
            Msg::ChainEnd {} => {
                self.chain.clear();
                self.chain_from_hand.clear();
            }
            Msg::PosChange { code, con, loc, seq, pos, .. } => {
                let v = self.var_at(*con, *loc, *seq, 0)?;
                if let Some(v) = v {
                    self.vars[v].faceup = pos & POS_FACEUP != 0;
                }
                self.observe(v, *code)?;
            }
            Msg::ConfirmCards { cards, .. } if self.reconfirmed(cards) => {
                // (one of the revealed and shuffled-back cards revealed again, e.g. the one picked
                // at random: the revealed ones stay the ones taken out next)
                for c in cards {
                    let v = self.var_at(c.con, c.loc, c.seq, 0)?;
                    if c.code != 0 {
                        self.observe(v, c.code)?;
                    }
                }
            }
            Msg::ConfirmCards { cards, .. } | Msg::ConfirmDecktop { cards, .. } | Msg::ConfirmExtratop { cards, .. } => {
                self.confirmed.clear();
                self.confirmed_locs =
                    if matches!(m, Msg::ConfirmCards { .. }) { cards.iter().map(|c| Some((c.con, c.loc))).collect() } else { vec![None] };
                self.confirmed_shuffled = false;
                self.confirmed_moves = 0;
                for c in cards {
                    let v = self.var_at(c.con, c.loc, c.seq, 0)?;
                    if let Some(v) = v {
                        self.confirmed.push(v);
                        self.referenced.push(v);
                        self.referenced_piles.insert((c.con, c.loc));
                        self.referenced_in_chain = self.referenced_in_chain || !self.chain.is_empty();
                        self.referenced_link = self.chain.len();
                    }
                    if c.code != 0 {
                        self.observe(v, c.code)?;
                    }
                }
            }
            Msg::RandomSelected { cards, .. } => {
                // after revealing cards and shuffling them back into their pile, a random pick
                // from that pile is a pick from the revealed cards; only a pick by the chain link
                // the cards were revealed for
                for c in cards {
                    if self.referenced_link != self.chain.len() {
                        break;
                    }
                    if c.loc & LOCATION_OVERLAY != 0 {
                        continue;
                    }
                    let Some(v) = self.plain_var(c.con, c.loc, c.seq) else { continue };
                    let in_piles = self.referenced_piles.contains(&(c.con, c.loc));
                    if in_piles {
                        self.picked.push(v);
                    }
                    if in_piles && let Some(pool) = self.is_claim_with_pool(v) {
                        let ids = ids_set(self.pools[pool].members.iter().copied().filter(|x| self.referenced.contains(x)));
                        if !ids.is_empty() {
                            self.narrow(v, ids);
                        }
                    }
                }
            }
            Msg::ReverseDeck {} => {
                // both decks are turned over (field::reverse_deck)
                for p in 0..2 {
                    self.lists[p][0].reverse();
                }
            }
            Msg::DeckTop { player, seq, code, pos } => self.deck_top(*player, *seq, *code, *pos)?,
            Msg::Swap { code1, loc1, code2, loc2 } => {
                let v1 = self.var_at(loc1.con, loc1.loc, loc1.seq, 0)?;
                let v2 = self.var_at(loc2.con, loc2.loc, loc2.seq, 0)?;
                self.observe(v1, *code1)?;
                self.observe(v2, *code2)?;
                self.zones_mut(loc1.con, loc1.loc)[loc1.seq as usize] = v2;
                self.zones_mut(loc2.con, loc2.loc)[loc2.seq as usize] = v1;
            }
            Msg::MissedEffect { con, loc, seq, pos, code } => {
                let v = self.var_at(*con, *loc, *seq, *pos)?;
                self.observe(v, *code)?;
            }
            Msg::SwapGraveDeck { player, extra_mask, .. } => self.swap_grave_deck(*player, extra_mask),
            Msg::RemoveCards { cards } => {
                for li in cards {
                    self.take(li.con, li.loc, li.seq, li.pos)?;
                }
            }
            Msg::TagSwap { .. } | Msg::ReloadField { .. } => return err(format!("unsupported message {}", m.name())),
            _ => {}
        }
        Ok(())
    }

    /// CONFIRM_CARDS of revealed and shuffled-back cards again.
    fn reconfirmed(&self, cards: &[Card]) -> bool {
        if self.confirmed.is_empty() || !self.confirmed_shuffled {
            return false;
        }
        if !cards.iter().all(|c| self.confirmed_locs.contains(&Some((c.con, c.loc))) || Some((c.con, c.loc, c.seq)) == self.last_move_to) {
            return false;
        }
        // Counter(shown codes) <= Counter(the confirmed cards' codes)
        let mut have: BTreeMap<Option<u32>, i64> = BTreeMap::new();
        for &x in &self.confirmed {
            *have.entry(self.code_of(x)).or_default() += 1;
        }
        let mut want: BTreeMap<Option<u32>, i64> = BTreeMap::new();
        for c in cards {
            *want.entry(Some(c.code)).or_default() += 1;
        }
        want.iter().all(|(k, n)| have.get(k).copied().unwrap_or(0) >= *n)
    }

    fn on_move(&mut self, code: u32, from: &crate::messages::Loc, to: &crate::messages::Loc, reason: u32) -> Result<()> {
        if from.loc == LOCATION_DECK && from.pos & POS_FACEUP != 0 {
            self.faceup_leaves(from.con, from.loc, from.seq, code);
        }
        let v: Option<VarId>;
        if from.loc == 0 {
            v = Some(self.new_var(to.con, Kind::Created, None));
        } else {
            let top = from.loc == LOCATION_DECK && from.seq as usize + 1 == self.lists[from.con as usize][0].len();
            let x = self.take(from.con, from.loc, from.seq, from.pos)?;
            v = Some(x);
            if from.loc == LOCATION_DECK {
                self.left_deck(Some(x));
                if top {
                    self.top_check_move = Some(from.con);
                }
            }
            if from.loc == LOCATION_HAND
                && from.pos & POS_FACEUP != 0
                && code == 0
                && let Some(pool) = self.is_claim_with_pool(x)
                && self.vars[x].code.is_none()
            {
                // a card leaves the hand face-up though its code is hidden now: one of the cards
                // that were revealed in the hand before it was shuffled
                let ids = ids_set(self.pools[pool].members.iter().copied().filter(|&m| self.vars[m].faceup));
                if !ids.is_empty() {
                    self.narrow(x, ids);
                }
            }
            if !self.own_picks.is_empty()
                && let Some(pool) = self.is_claim_with_pool(x)
                && from.con == self.viewer
                && from.loc == LOCATION_DECK
                && self.pools[pool].location == PoolLoc::Loc(LOCATION_DECK)
            {
                let members = &self.pools[pool].members;
                let picked: Vec<VarId> = self.own_picks.iter().filter(|&&(y, c)| c == code && members.contains(&y)).map(|&(y, _)| y).collect();
                if !picked.is_empty() {
                    self.narrow(x, ids_set(picked.iter().copied()));
                    let first = picked[0];
                    self.own_picks.retain(|&(y, _)| y != first);
                }
            }
            if let Some(pool) = self.is_claim_with_pool(x)
                && !self.referenced.is_empty()
                && self.vars[x].prefer.is_empty()
                && self.referenced_piles.contains(&(from.con, from.loc))
            {
                // (the first move out of the shuffled pile)
                if !self.referenced_in_chain && to.loc == LOCATION_MZONE {
                    // cards revealed outside a chain are the cost of a summon procedure: the
                    // summoned card is most likely another one, but can be among them
                    if code != 0 && crate::belief::script_index::reveals_itself(code) {
                        let others: Vec<VarId> = self.pools[pool].members.iter().copied().filter(|y| !self.referenced.contains(y)).collect();
                        if !others.is_empty() {
                            self.vars[x].prefer = others;
                        }
                    } else {
                        let refs: Vec<VarId> = self.referenced.iter().copied().filter(|&y| y != x).collect();
                        self.vars[x].not_same.extend(refs);
                    }
                    // (the reveal was that procedure's cost: it says nothing more)
                    self.referenced.clear();
                    self.referenced_piles.clear();
                    self.picked.clear();
                } else {
                    let pool_vars = if self.picked.iter().any(|&y| y != x) { self.picked.clone() } else { self.referenced.clone() };
                    // (only the revealed cards that can be this one: another name shows another card)
                    self.vars[x].prefer = pool_vars
                        .into_iter()
                        .filter(|&y| y != x && (code == 0 || self.code_of(y).is_none() || self.code_of(y) == Some(code)))
                        .collect();
                }
            }
        }
        let x = v.unwrap();
        if self.is_claim_with_pool(x).is_some()
            && self.vars[x].prefer.is_empty()
            && self.vars[x].same_as.is_none()
            && from.loc == LOCATION_HAND
            && self.resolving
            && !self.chain.is_empty()
            && *self.chain_from_hand.last().unwrap()
            && let Some(last) = *self.chain.last().unwrap()
            && code != 0
            && self.vars[last].code == Some(code)
            && last != x
        {
            // (a card of that name leaving the hand while a link activated from the hand resolves
            // is most likely that link's card)
            self.vars[x].prefer = vec![last];
        }
        if to.loc == LOCATION_MZONE && from.loc == LOCATION_HAND && !self.tributed_to_hand.is_empty() && self.vars[x].kind == Kind::Claim {
            let t: Vec<VarId> = self.tributed_to_hand.iter().copied().filter(|&y| y != x).collect();
            self.vars[x].not_same.extend(t);
        }
        if to.loc == LOCATION_HAND && reason & REASON_RELEASE != 0 {
            self.tributed_to_hand.push(x);
        }
        if to.loc != 0 {
            self.put(x, to.con, to.loc, to.seq, to.pos)?;
        }
        self.observe(Some(x), code)?;
        if to.loc == LOCATION_DECK {
            self.deck_arrivals.push(x);
            if to.pos & POS_FACEUP != 0 {
                self.faceup_in_deck.push(x); // (its place in the deck stays public)
            }
        }
        Ok(())
    }

    fn deck_top(&mut self, player: u32, seq: u32, code: u32, pos: u32) -> Result<()> {
        let deck = &self.lists[player as usize][0];
        let idx = deck.len() as i64 - 1 - seq as i64;
        if !(0 <= idx && (idx as usize) < deck.len() && code != 0) {
            return Ok(());
        }
        let v = deck[idx as usize];
        self.observe(Some(v), code)?;
        if pos & POS_FACEUP != 0 && !self.faceup_in_deck.contains(&v) {
            // a card just put there (and turned face-up), or turned face-up where it is: its place
            // stays public. Otherwise a face-up card seen again on top: the same card
            let same: Vec<VarId> = self
                .faceup_deck_cards(player)
                .into_iter()
                .filter(|&x| self.code_of(x) == Some(code) && self.descends(v, x, player))
                .collect();
            if self.deck_arrivals.contains(&v) || same.is_empty() || (self.prev_sig == Some(("CONFIRM_DECKTOP", Some(player))) && !self.vars[v].faceup) {
                self.faceup_in_deck.push(v);
            } else if same.len() == 1 {
                self.identify(Some(v), Some(same[0]));
            }
        } else if pos & POS_FACEUP == 0 && self.vars[v].kind == Kind::Claim {
            // (shown face-down, e.g. all decks turned over): none of the face-up cards in that deck
            for x in self.faceup_deck_cards(player) {
                if self.code_of(x) == Some(code) && x != v {
                    self.add_not_same(v, x);
                }
            }
        }
        self.vars[v].faceup = pos & POS_FACEUP != 0;
        Ok(())
    }

    fn shuffle(&mut self, player: u32, location: u32) -> PoolId {
        let li = list_index(location).unwrap();
        let lst = self.lists[player as usize][li].clone();
        let (members, rest) = if location == LOCATION_EXTRA {
            let n = lst.iter().filter(|&&v| !self.vars[v].faceup).count();
            (lst[..n].to_vec(), lst[n..].to_vec())
        } else {
            (lst, Vec::new())
        };
        let pool = self.make_pool(player, PoolLoc::Loc(location), members);
        let mut new = self.pools[pool].claims.clone();
        new.extend(rest);
        self.lists[player as usize][li] = new;
        self.shuffle_events.push(pool);
        pool
    }

    fn swap_grave_deck(&mut self, p: u32, extra_mask: &str) {
        let mask: Vec<u8> = (0..extra_mask.len() / 2).filter_map(|i| u8::from_str_radix(&extra_mask[2 * i..2 * i + 2], 16).ok()).collect();
        let (gi, di, ei) = (list_index(LOCATION_GRAVE).unwrap(), 0, list_index(LOCATION_EXTRA).unwrap());
        let grave = self.lists[p as usize][gi].clone();
        let deck = self.lists[p as usize][di].clone();
        let (mut new_deck, mut to_extra) = (Vec::new(), Vec::new());
        for (i, &v) in grave.iter().enumerate() {
            if i / 8 < mask.len() && mask[i / 8] & (1 << (i % 8)) != 0 {
                to_extra.push(v);
            } else {
                new_deck.push(v);
            }
        }
        for &v in &deck {
            self.vars[v].faceup = true;
        }
        for &v in new_deck.iter().chain(&to_extra) {
            self.vars[v].faceup = false;
        }
        self.lists[p as usize][gi] = deck;
        self.lists[p as usize][di] = new_deck;
        let extra = self.lists[p as usize][ei].clone();
        let n_fd = extra.iter().filter(|&&v| !self.vars[v].faceup).count();
        let mut e = extra[..n_fd].to_vec();
        e.extend(to_extra);
        e.extend_from_slice(&extra[n_fd..]);
        self.lists[p as usize][ei] = e;
    }

    /// The viewer's own command names a physical card: the hand card it summons, sets or
    /// activates is the one that moves to the field next.
    pub fn response(&mut self, r: &Response) -> Result<()> {
        let Some(p) = self.last_prompt.clone() else { return Ok(()) };
        if let Msg::SelectCard { cards, .. } = &p
            && r.indices.as_ref().is_some_and(|i| !i.is_empty())
        {
            // the viewer picked cards of its own deck: a card of that code that leaves the deck
            // after it was shuffled is one of the picked ones
            for &i in r.indices.as_ref().unwrap() {
                let Some(c) = usize::try_from(i).ok().and_then(|i| cards.get(i)) else { continue };
                if c.con == self.viewer && c.loc == LOCATION_DECK && c.code != 0
                    && let Some(v) = self.plain_var(c.con, c.loc, c.seq)
                {
                    self.own_picks.push((v, c.code));
                }
            }
            return Ok(());
        }
        let Msg::SelectIdlecmd { .. } = &p else { return Ok(()) };
        let Some(key) = r.kind.as_deref().and_then(|k| COMMAND_LISTS.iter().find(|(c, _)| *c == k).map(|(_, l)| *l)) else { return Ok(()) };
        let entries = command_list(&p, key);
        let i = r.index.unwrap_or(-1);
        if i >= 0 && (i as usize) < entries.len() {
            let e = &entries[i as usize];
            if e.loc == LOCATION_HAND && e.code != 0 {
                self.chosen = Some((self.var_at(e.con, e.loc, e.seq, 0)?, e.con, e.code));
            }
        }
        Ok(())
    }

    pub fn prompt(&mut self, p: &Msg) -> Result<()> {
        self.last_prompt = Some(p.clone());
        if let Msg::SelectCard { cards, .. } = p
            && !self.confirmed.is_empty()
            && cards.len() == self.confirmed.len()
            && cards.iter().all(|c| c.code == 0 && c.loc & LOCATION_OVERLAY == 0)
        {
            // right after revealing cards and shuffling them back into their pile, the viewer is
            // offered exactly that many hidden cards of the pile: the revealed ones
            for c in cards {
                if let Some(v) = self.plain_var(c.con, c.loc, c.seq)
                    && let Some(pool) = self.is_claim_with_pool(v)
                {
                    let ids = ids_set(self.pools[pool].members.iter().copied().filter(|x| self.confirmed.contains(x)));
                    if !ids.is_empty() {
                        self.narrow(v, ids);
                    }
                }
            }
        }
        if matches!(p, Msg::SelectIdlecmd { .. } | Msg::SelectBattlecmd { .. }) {
            self.chosen = None;
            self.tributed_to_hand.clear();
            self.own_picks.clear(); // (the action that picked them is over)
        }
        self.releases = 0;
        // (option lists are shown in a canonical order: their order carries no information)
        let lists: Vec<Card> = prompt_card_lists(p).into_iter().flat_map(|(_, l)| l.iter().cloned()).collect();
        for c in lists {
            if c.code == 0 || c.loc == 0 {
                continue;
            }
            let v = self.var_at(c.con, c.loc, c.seq, c.pos.unwrap_or(0))?;
            self.observe(v, c.code)?;
        }
        if let Msg::SelectEffectyn { code, con, loc, seq, pos, .. } = p
            && *code != 0
        {
            let v = self.var_at(*con, *loc, *seq, *pos)?;
            self.observe(v, *code)?;
        }
        Ok(())
    }

    /// Whether `v` can be `x` or a later claim on it through the shuffles of `player`'s deck since.
    pub fn descends(&self, v: VarId, x: VarId, player: u32) -> bool {
        let want = self.code_of(x);
        let (mut frontier, mut seen) = (vec![v], HashSet::new());
        while let Some(y) = frontier.pop() {
            if y == x {
                return true;
            }
            if want.is_some() && self.code_of(y).is_some() && self.code_of(y) != want {
                continue;
            }
            let yv = &self.vars[y];
            let Some(pool) = yv.pool.filter(|_| yv.kind == Kind::Claim) else { continue };
            let pl = &self.pools[pool];
            if seen.contains(&pool) || (pl.player, pl.location) != (player, PoolLoc::Loc(LOCATION_DECK)) {
                continue;
            }
            seen.insert(pool);
            frontier.extend(pl.members.iter().copied());
        }
        false
    }

    /// `v` left the deck: if it is one of the face-up cards there (or was identified with one),
    /// that one is gone.
    fn left_deck(&mut self, v: Option<VarId>) {
        if let Some(v) = v
            && !self.faceup_in_deck.is_empty()
        {
            let same = self.vars[v].same_as;
            self.faceup_in_deck.retain(|&x| x != v && same != Some(x));
        }
    }

    /// A face-up card leaves the deck: that one of the face-up cards there.
    fn faceup_leaves(&mut self, con: u32, loc: u32, seq: u32, code: u32) {
        let v = self.plain_var(con, loc, seq);
        let code = if code != 0 { Some(code) } else { v.and_then(|v| self.code_of(v)) }; // (e.g. Set: seen in the deck)
        match code.filter(|&c| c != 0) {
            None => {
                // (told later, e.g. when the card is revealed)
                if let Some(v) = v {
                    self.faceup_left.push((v, con));
                }
            }
            Some(c) => self.faceup_gone(v, con, c),
        }
    }

    fn faceup_gone(&mut self, v: Option<VarId>, player: u32, code: u32) {
        let cands: Vec<VarId> = self.faceup_deck_cards(player).into_iter().filter(|&x| self.code_of(x) == Some(code)).collect();
        // (of several face-up copies any: they are alike)
        let x = cands.iter().copied().find(|&x| Some(x) == v || v.is_some_and(|v| self.vars[v].same_as == Some(x))).or(cands.first().copied());
        let Some(x) = x else { return };
        if let Some(v) = v
            && v != x
        {
            self.identify(Some(v), Some(x));
        }
        self.faceup_in_deck.retain(|&y| y != x);
    }

    /// The top card of a player's deck is none of the face-up cards in it.
    fn not_faceup_top(&mut self, p: u32) {
        let Some(&top) = self.lists[p as usize][0].last() else { return };
        if self.vars[top].kind == Kind::Claim {
            for x in self.faceup_deck_cards(p) {
                if x != top {
                    self.add_not_same(top, x);
                }
            }
        }
    }

    /// The face-up cards in a player's deck: those put there face-up (or turned face-up there)
    /// that are still there.
    fn faceup_deck_cards(&mut self, player: u32) -> Vec<VarId> {
        let pending: Vec<(VarId, u32)> = self.faceup_left.iter().copied().filter(|&(v, _)| self.code_of(v).is_some_and(|c| c != 0)).collect();
        if !pending.is_empty() {
            self.faceup_left.retain(|&(v, _)| !self.vars[v].known_code().is_some_and(|c| c != 0));
            for (v, p) in pending {
                let c = self.code_of(v).unwrap();
                self.faceup_gone(Some(v), p, c);
            }
        }
        let deck = &self.lists[player as usize][0];
        self.faceup_in_deck.iter().copied().filter(|&x| deck.iter().any(|&v| self.descends(v, x, player))).collect()
    }

    /// `v` is known to be the physical card that was `same` before the last shuffle of its pile.
    pub fn identify(&mut self, v: Option<VarId>, same: Option<VarId>) {
        let (Some(v), Some(same)) = (v, same) else { return };
        if v == same || self.vars[v].kind != Kind::Claim || self.vars[v].same_as.is_some() {
            return;
        }
        // `same` must be an ancestor: a member of v's pool, or of the pools before
        let (mut frontier, mut seen) = (vec![v], HashSet::new());
        while let Some(x) = frontier.pop() {
            let Some(pool) = self.vars[x].pool else { continue };
            if !seen.insert(pool) {
                continue;
            }
            let members = &self.pools[pool].members;
            if members.contains(&same) {
                self.vars[v].same_as = Some(same);
                return;
            }
            frontier.extend(members.iter().copied().filter(|&m| self.vars[m].kind == Kind::Claim));
        }
    }

    pub fn state(&mut self, s: &Snapshot) -> Result<()> {
        // (the same view as the previous one, e.g. in a row of chain windows: nothing new)
        if let Some((last, n)) = &self.last_state
            && last == s
            && *n == self.n_messages
        {
            return Ok(());
        }
        self.last_state = Some((s.clone(), self.n_messages));
        // the chain shows where each link's card is now (e.g. after its pile was shuffled)
        for (link, &var) in s.chain.iter().zip(self.chain.clone().iter()) {
            let Some(h) = link.handler else { continue };
            if h.loc != 0 && h.loc & (LOCATION_OVERLAY | LOCATION_DECK) == 0 && loc_visible(h.con, h.loc, h.pos, self.viewer) {
                let pv = self.plain_var(h.con, h.loc, h.seq);
                self.identify(pv, var);
            } else if h.loc == LOCATION_DECK
                && let Some(var) = var
            {
                // (where in the deck is masked) the link's card is in the deck: no card outside of
                // it is that card
                for p in 0..2usize {
                    let mut outside: Vec<VarId> = self.mzones[p].iter().chain(&self.szones[p]).flatten().copied().collect();
                    for (i, lst) in self.lists[p].iter().enumerate() {
                        if LIST_LOCS[i] != LOCATION_DECK {
                            outside.extend(lst.iter().copied());
                        }
                    }
                    for v in outside {
                        if self.vars[v].kind == Kind::Claim && v != var {
                            self.add_not_same(v, var);
                        }
                    }
                }
            }
        }
        for (pl_idx, pl) in s.players.iter().enumerate() {
            let pi = pl_idx as u32;
            for (name, loc) in LOCATIONS {
                let cards = pl.pile(loc);
                if zone_count(loc).is_some() {
                    for (seq, card) in cards.iter().enumerate() {
                        let v = self.zones(pi, loc).get(seq).copied().flatten();
                        if card.is_none() != v.is_none() {
                            return err(format!("zone mismatch P{pl_idx} {name}{seq}"));
                        }
                        let (Some(card), Some(v)) = (card, v) else { continue };
                        self.observe(Some(v), card.code.unwrap_or(0))?;
                        if let Some(o) = card.owner
                            && card.code.unwrap_or(0) == 0
                        {
                            // (a face-down card's owner shows too)
                            self.owned(Some(v), o);
                        }
                        let overlays = card.overlays.clone().unwrap_or_default();
                        if overlays.len() != self.vars[v].overlays.len() {
                            return err(format!("overlay count mismatch P{pl_idx} {name}{seq}"));
                        }
                        for (ov, code) in self.vars[v].overlays.clone().into_iter().zip(overlays) {
                            self.observe(Some(ov), code)?;
                        }
                    }
                } else {
                    let li = list_index(loc).unwrap();
                    let lst = self.lists[pl_idx][li].clone();
                    if cards.len() != lst.len() {
                        return err(format!("size mismatch P{pl_idx} {name}: {} vs {}", cards.len(), lst.len()));
                    }
                    let code = |c: &Option<crate::state::CardQuery>| c.as_ref().and_then(|c| c.code).unwrap_or(0);
                    let position = |c: &Option<crate::state::CardQuery>| c.as_ref().and_then(|c| c.position).unwrap_or(0);
                    // (the owners of deck cards show too: they tell apart only where a card of the
                    // other player is in the deck)
                    let foreign = loc == LOCATION_DECK && cards.iter().any(|c| c.as_ref().and_then(|c| c.owner).unwrap_or(pi) != pi);
                    if loc != LOCATION_HAND && !foreign && !cards.iter().any(|c| code(c) != 0 || position(c) & POS_FACEUP != 0) {
                        // (nothing shows: e.g. a hidden deck, the usual case)
                        for v in lst {
                            self.vars[v].faceup = false;
                        }
                        continue;
                    }
                    let mut used: HashSet<VarId> = HashSet::new();
                    let shown: Option<BTreeMap<u32, usize>> = (loc == LOCATION_DECK).then(|| {
                        let mut m = BTreeMap::new();
                        for c in cards.iter().filter(|c| code(c) != 0 && position(c) & POS_FACEUP != 0) {
                            *m.entry(code(c)).or_default() += 1;
                        }
                        m
                    });
                    let faceup_now = if shown.as_ref().is_some_and(|m| !m.is_empty()) { self.faceup_deck_cards(pi) } else { Vec::new() };
                    for (&v, card) in lst.iter().zip(cards) {
                        self.observe(Some(v), code(card))?;
                        if loc == LOCATION_DECK && code(card) != 0 && position(card) & POS_FACEUP != 0 {
                            // a face-up card in the deck: one put there face-up (of several such
                            // copies any, each once: they are alike)
                            let known: Vec<VarId> = faceup_now.iter().copied().filter(|&x| self.code_of(x) == Some(code(card))).collect();
                            let same: Vec<VarId> = known.iter().copied().filter(|x| !used.contains(x)).collect();
                            // (only if the shown copies are exactly the known ones)
                            if !same.is_empty() && Some(known.len()) == shown.as_ref().and_then(|m| m.get(&code(card)).copied()) {
                                used.insert(same[0]);
                                self.identify(Some(v), Some(same[0]));
                            }
                        }
                        self.vars[v].faceup = position(card) & POS_FACEUP != 0;
                        if let Some(o) = card.as_ref().and_then(|c| c.owner)
                            && (loc == LOCATION_HAND || foreign)
                        {
                            self.owned(Some(v), o);
                        }
                    }
                    if loc == LOCATION_HAND {
                        // (a card being activated from the hand is public too, but only while it
                        // is on the chain: not kept revealed)
                        let on_chain: BTreeSet<u32> = s
                            .chain
                            .iter()
                            .filter_map(|l| l.handler)
                            .filter(|h| h.con == pi && h.loc == LOCATION_HAND)
                            .map(|h| h.seq)
                            .collect();
                        let public: Vec<(u32, bool)> = cards
                            .iter()
                            .enumerate()
                            .map(|(i, c)| (code(c), !on_chain.contains(&(i as u32)) && c.as_ref().and_then(|c| c.is_public).unwrap_or(0) != 0))
                            .collect();
                        self.public_hand(&lst, &public);
                    }
                }
            }
        }
        Ok(())
    }
}
