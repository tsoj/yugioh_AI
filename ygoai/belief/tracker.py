"""Deduction of card identities from one player's observation stream.

The tracker follows every card slot of both players through the observation
stream, exactly mirroring the core's list semantics (removal from a pile
shifts the cards above it, draws take the top of the deck, ...). The content
of a slot is a `Var`: the identity of one physical card between two shuffles.

* Atoms are the physical cards the players start with (the main and extra
  deck). The viewer's own atoms have known codes; the opponent's are free.
* A shuffle turns the vars of a location into the members of a `Pool` and
  puts fresh vars ("claims") into the slots. Each claim is equal to exactly
  one member, but which one is unknown.
* Observations ("the card at hand slot 3 is X") pin the code of a var.

`solve()` then assigns every claim to a member and every free atom to a code
such that all observations hold, preferring explanations that reuse cards
already known over inventing new copies. The result (`Hypothesis`) contains
everything the belief replay needs: the initial deck orders, the opponent's
deck list and the outcome of every shuffle.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .. import constants as C

LIST_LOCS = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA)
ZONE_LOCS = {C.LOCATION_MZONE: 7, C.LOCATION_SZONE: 8}


class TrackerError(RuntimeError):
    pass


def _leaves_field(m: dict) -> bool:
    """A card sent from the field in place of a Tribute (e.g. with Floowandereeze
    and the Unexplored Winds). Not destroyed cards or cards moved by the rules:
    those can be consequences of the previous action."""
    field = C.LOCATION_MZONE | C.LOCATION_SZONE
    return bool(m["from"]["loc"] & field) and not m["from"]["loc"] & C.LOCATION_OVERLAY \
        and not m["to"]["loc"] & field and not m.get("reason", 0) & (C.REASON_DESTROY | C.REASON_RULE)


class Var:
    __slots__ = ("id", "owner", "code", "pool", "assigned", "kind", "faceup", "overlays",
                 "fixed_code", "resolved", "excluded", "need", "marks", "allowed", "same_as",
                 "not_same", "prefer", "seen_owner")

    def __init__(self, vid: int, owner: int, kind: str, fixed_code: int | None = None):
        self.id = vid
        self.owner = owner
        self.kind = kind            # "main"/"extra" atom, "claim", "created"
        self.fixed_code = fixed_code  # known code of an atom (viewer's own cards)
        self.code: int | None = None  # observed code
        self.pool: Pool | None = None  # pool this var is a claim of
        self.assigned: Var | None = None  # member this claim resolved to
        self.faceup = False
        self.overlays: list[Var] = []
        self.resolved: int | None = None  # final code after solve()
        self.excluded: frozenset = frozenset()  # codes this card is known not to have
        self.need: tuple | None = None  # the kind it must have (see completion.fits_need)
        self.marks: frozenset = frozenset()  # card hints on this physical card
        self.allowed: frozenset | None = None  # claim: ids of the members it can be
        self.same_as: Var | None = None  # claim: an earlier slot that held the same physical card
        self.not_same: list = []  # claim: earlier slots whose physical card this is not
        # claim: earlier slots whose physical card this preferably is (copies that
        # nothing tells apart are interchangeable; see Tracker._referenced)
        self.prefer: list = []
        self.seen_owner: int | None = None  # the owner a state view showed for this slot

    def __repr__(self) -> str:
        return f"Var({self.id},{self.kind},code={self.code},fixed={self.fixed_code})"


@dataclass
class Pool:
    index: int
    player: int
    location: int          # LOCATION_* or "set" for shuffled set cards
    members: list[Var]
    claims: list[Var]      # in slot order at creation time
    zones: list[int] = field(default_factory=list)  # for set card shuffles: zone sequence per claim
    used: set = field(default_factory=set)  # ids of members assigned to a claim


@dataclass
class Hypothesis:
    """A complete, consistent guess of the hidden information."""
    decks: list[dict]                   # per player {"main": [...], "extra": [...]}
    deck_orders: list[list[int]]        # initial main deck order, bottom first
    extra_orders: list[list[int]]       # initial extra deck order
    shuffles: list[dict]                # {"player", "location", "codes"} in stream order
    set_card_shuffles: list[list[int]]  # codes per ascending zone, in stream order
    unknown_atoms: int = 0              # opponent atoms whose code was guessed
    var_codes: dict = field(default_factory=dict)  # tracker var id -> code
    pinned: set = field(default_factory=set)       # var ids fixed by observations
    observed: set = field(default_factory=set)     # var ids observed directly
    # which copy: rank of the card among its owner's copies of that code in the
    # same section (= creation order), per initial main/extra deck slot
    deck_ranks: list = field(default_factory=list)
    extra_ranks: list = field(default_factory=list)


class Tracker:
    def __init__(self, viewer: int, init: dict):
        self.viewer = viewer
        self.next_id = 0
        self.all_vars: list[Var] = []
        self.lists: list[dict[int, list[Var]]] = [{loc: [] for loc in LIST_LOCS} for _ in range(2)]
        self.zones: list[dict[int, list[Var | None]]] = [
            {loc: [None] * n for loc, n in ZONE_LOCS.items()} for _ in range(2)]
        self.pools: list[Pool] = []
        self.atoms: list[list[Var]] = [[], []]
        self.observations: list[tuple[Var, int]] = []
        # observations and kind requirements (code None) in stream order: the
        # solver satisfies them in this order
        self.searched: list[tuple[Var, int | None]] = []
        self.exclusions: list[Var] = []
        self.needs: list[Var] = []  # hidden cards whose kind shows (e.g. Set as a spell/trap)
        self.contains: set[tuple[int, int]] = set()  # (player, code) named by hints
        # (a, b): a is an earlier created copy than b (only meaningful for copies of
        # the same card; the core lists and processes such copies in creation order)
        self._resolving = False  # between CHAIN_SOLVING and CHAIN_SOLVED
        self._last_prompt: dict | None = None
        self._chosen: tuple | None = None  # (var, con, code): the viewer's command's hand card
        self._resent_hints = False  # right after a shuffle (see _card_hint)
        # (_owned's conclusions depend only on the owners seen so far: a slot seen
        # again with no new owner anywhere tells nothing new)
        self._owners_seen = 0
        self._owned_at: dict[int, int] = {}
        self._owners_memo: dict[int, tuple[int, set[int]]] = {}
        self._confirmed: list[Var] = []  # the cards of the latest CONFIRM_* message
        # cards revealed in the current resolution: a script that reveals cards,
        # shuffles them back and then moves cards of the same names moves the same
        # physical cards as far as anything shows (it keeps them in a group)
        self._referenced: list[Var] = []
        self._own_picks: list[tuple[Var, int]] = []  # (card, code) the viewer picked from its deck
        self._confirmed_locs: list[tuple[int, int]] = []
        self._confirmed_shuffled = False
        self._confirmed_moves = 0
        self._referenced_piles: set[tuple[int, int]] = set()  # (con, loc) they were revealed in
        self._picked: list[Var] = []  # of those, the ones picked at random
        self._referenced_in_chain = False  # revealed while a chain was built or resolved
        self._referenced_link = 0  # the chain link they were revealed for (0: none)
        self._chain: list[Var | None] = []  # the card of each chain link, by the time it chained
        self._chain_from_hand: list[bool] = []
        self._left_hand: list[Var] = []  # cards that left a hand while the current link resolves
        # cards Tributed into the hand (e.g. by Fantastic Striborg's summon): not the
        # monster whose summon Tributed them
        self._tributed_to_hand: list[Var] = []
        self._solved: tuple | None = None  # (card, player, left_hand) of a link just solved
        self._last_sig: tuple | None = None  # the latest message that is not a hint
        self.turn_player = 0
        self._releases = 0  # cards released for the current action (tribute count)
        self._faceup_in_deck: list[Var] = []  # cards put into a deck face-up (or turned so)
        self._deck_arrivals: list[Var] = []  # cards moved into a deck since its last shuffle
        self._faceup_left: list[tuple[Var, int]] = []  # face-up cards that left a deck, code unknown yet
        self._prev_sig: tuple | None = None  # the message before the latest one (not hints)
        self._last_move_to: tuple | None = None  # (con, loc, seq) the latest MOVE put a card at
        self._top_check: int | None = None  # a deck just shuffled (whose top may be announced)
        own = init["deck"]
        opp_main, opp_extra = init["opponent_deck_size"], init["opponent_extra_size"]
        for p in (0, 1):
            if p == viewer:
                main_atoms = [self._new(p, "main", code) for code in sorted(own["main"])]
                extra_atoms = [self._new(p, "extra", code)
                               for code in init.get("extra_order") or sorted(own["extra"])]
            else:
                main_atoms = [self._new(p, "main") for _ in range(opp_main)]
                extra_atoms = [self._new(p, "extra") for _ in range(opp_extra)]
            self.atoms[p] = main_atoms + extra_atoms
            self.lists[p][C.LOCATION_DECK] = list(self._make_pool(p, C.LOCATION_DECK, main_atoms).claims)
            if p == viewer:
                # the viewer knows the order of their own extra deck
                self.lists[p][C.LOCATION_EXTRA] = list(extra_atoms)
            else:
                self.lists[p][C.LOCATION_EXTRA] = list(self._make_pool(p, C.LOCATION_EXTRA, extra_atoms).claims)
        self.initial_pools = list(self.pools)
        # shuffles in stream order: (pool, player, location)
        self.shuffle_events: list[Pool] = []
        self.set_shuffle_events: list[Pool] = []

    # -- helpers --------------------------------------------------------------
    def _new(self, owner: int, kind: str, fixed: int | None = None) -> Var:
        v = Var(self.next_id, owner, kind, fixed)
        self.next_id += 1
        self.all_vars.append(v)
        return v

    def _make_pool(self, player: int, location, members: list[Var]) -> Pool:
        claims = []
        for m in members:
            c = self._new(m.owner, "claim")
            c.faceup = m.faceup
            claims.append(c)
        pool = Pool(len(self.pools), player, location, list(members), list(claims))
        for c in claims:
            c.pool = pool
        self.pools.append(pool)
        return pool

    def observe(self, var: Var | None, code: int) -> None:
        if var is None or not code:
            return
        if var.code is not None and var.code != code:
            raise TrackerError(f"{var} observed as {code} but was {var.code}")
        if var.fixed_code is not None and var.fixed_code != code:
            raise TrackerError(f"{var} observed as {code} but is {var.fixed_code}")
        if var.code is None:
            var.code = code
            self.observations.append((var, code))
            self.searched.append((var, code))

    def exclude(self, var: Var | None, code: int) -> None:
        """`var` is known not to be `code` (e.g. a scan for the first matching
        card passed over it)."""
        if var is None or not code:
            return
        if var.code == code or var.fixed_code == code:
            raise TrackerError(f"{var} cannot be excluded from {code}")
        var.excluded = var.excluded | {code}
        self.exclusions.append(var)

    def _plain_var(self, con: int, loc: int, seq: int) -> Var | None:
        if loc in ZONE_LOCS:
            return self.zones[con][loc][seq]
        lst = self.lists[con].get(loc)
        if lst is None or seq >= len(lst):
            return None
        return lst[seq]

    def _overlay_host(self, li: dict) -> Var:
        xyz = self._plain_var(li["con"], li["loc"] & ~C.LOCATION_OVERLAY, li["seq"])
        if xyz is None:
            raise TrackerError(f"no xyz host at {li}")
        return xyz

    def var_at(self, con: int, loc: int, seq: int, pos: int = 0) -> Var | None:
        if loc & C.LOCATION_OVERLAY:
            xyz = self._overlay_host({"con": con, "loc": loc, "seq": seq})
            if pos >= len(xyz.overlays):
                raise TrackerError(f"no overlay {pos} at {con}/{loc:#x}/{seq}")
            return xyz.overlays[pos]
        if loc in ZONE_LOCS:
            return self.zones[con][loc][seq]
        lst = self.lists[con].get(loc)
        if lst is None:
            raise TrackerError(f"bad location {loc:#x}")
        if seq >= len(lst):
            raise TrackerError(f"no card at {con}/{loc:#x}/{seq} (size {len(lst)})")
        return lst[seq]

    def _take(self, li: dict) -> Var:
        con, loc, seq = li["con"], li["loc"], li["seq"]
        if loc & C.LOCATION_OVERLAY:
            xyz = self._overlay_host(li)
            if li["pos"] >= len(xyz.overlays):
                raise TrackerError(f"no overlay at {li}")
            return xyz.overlays.pop(li["pos"])
        if loc in ZONE_LOCS:
            v = self.zones[con][loc][seq]
            if v is None:
                raise TrackerError(f"empty zone at {li}")
            self.zones[con][loc][seq] = None
            return v
        lst = self.lists[con][loc]
        if seq >= len(lst):
            raise TrackerError(f"no card to take at {li} (size {len(lst)})")
        return lst.pop(seq)

    def _put(self, v: Var, li: dict) -> None:
        con, loc, seq = li["con"], li["loc"], li["seq"]
        v.faceup = bool(li["pos"] & C.POS_FACEUP) if not loc & C.LOCATION_OVERLAY else True
        if loc & C.LOCATION_OVERLAY:
            self._overlay_host(li).overlays.insert(li["pos"], v)
            return
        if loc in ZONE_LOCS:
            if self.zones[con][loc][seq] is not None:
                raise TrackerError(f"zone occupied at {li}")
            self.zones[con][loc][seq] = v
            return
        lst = self.lists[con][loc]
        if seq > len(lst):
            raise TrackerError(f"bad insert position {li} (size {len(lst)})")
        lst.insert(seq, v)

    # -- stream processing ------------------------------------------------------
    def process(self, item: tuple) -> None:
        kind = item[0]
        if kind == "msg":
            self.message(item[1])
        elif kind == "prompt":
            self.prompt(item[1])
            if item[2] is not None:
                self.state(item[2])
        elif kind == "response":
            self.response(item[1])

    def _require(self, m: dict, need: tuple) -> None:
        v = self.var_at(m["con"], m["loc"], m["seq"], m.get("pos", 0))
        if v is not None and v.code is None and v.fixed_code is None and v.need is None:
            v.need = need
            self.needs.append(v)
            self.searched.append((v, None))

    def _public_hand(self, lst: list[Var], cards: list[dict]) -> None:
        """Cards kept revealed in a hand (EFFECT_PUBLIC) stay revealed through a
        shuffle: if a shuffled hand shows the same public codes as were public
        before, each is one of the earlier public cards of its code. (Public
        status can also end and start between two views; then the codes differ
        and nothing is concluded.)"""
        public = [(v, c.get("code", 0)) for v, c in zip(lst, cards) if c.get("is_public")]
        pools = {v.pool.index: v.pool for v, _ in public if v.kind == "claim" and v.pool is not None}
        if len(pools) == 1 and all(v.pool is next(iter(pools.values())) for v, _ in public):
            pool = next(iter(pools.values()))
            carriers = [(m.id, mk[1]) for m in pool.members for mk in m.marks if mk[0] == "public"]
            if carriers and sorted(c for _, c in carriers) == sorted(c for _, c in public):
                for v, code in public:
                    same = frozenset(i for i, c in carriers if c == code)
                    v.allowed = same if v.allowed is None else v.allowed & same
        for v, code in public:
            v.marks = v.marks | {("public", code)}

    def _possible_owners(self, v: Var) -> set[int]:
        """Who can own the card in slot `v` (through the shuffles back to the decks)."""
        hit = self._owners_memo.get(v.id)
        if hit is not None and (hit[0] == self._owners_seen or len(hit[1]) == 1):
            # (owners only narrow down as owners are seen: a single one stays)
            return hit[1]
        out: set[int] = set()
        todo, seen = [v], set()
        while todo and len(out) < 2:
            x = todo.pop()
            if x.seen_owner is not None:
                out.add(x.seen_owner)  # (shown by a state view: no need to look further)
            elif x.kind != "claim" or x.pool is None:
                out.add(x.owner)
            elif x.pool.index not in seen:
                seen.add(x.pool.index)
                todo.extend(x.pool.members)
        self._owners_memo[v.id] = (self._owners_seen, out)
        return out

    def _owner_leaves(self, v: Var, owner: int, limit: int = 200) -> tuple[list[Var], list[Var]]:
        """The cards (slots of a single possible owner) the shuffles before slot `v`
        mixed: those of `owner` and those of the other player."""
        mine: list[Var] = []
        other: list[Var] = []
        todo, seen = [v], set()
        while todo:
            x = todo.pop()
            if x.id in seen:
                continue
            seen.add(x.id)
            if len(seen) > limit:
                return [], []
            os = self._possible_owners(x) if x is not v else {0, 1}
            if len(os) == 1:
                (mine if owner in os else other).append(x)
            elif x.kind == "claim" and x.pool is not None:
                todo.extend(x.pool.members)
            else:
                return [], []
        return mine, other

    def _owned(self, v: Var | None, owner: int) -> None:
        """A card's owner shows even when it is hidden (a card given to the other
        player's hand keeps its sleeve): after a shuffle, the slot holds one of
        the shuffled cards that owner could own."""
        if v is None:
            return
        if v.seen_owner != owner:
            v.seen_owner = owner
            self._owners_seen += 1
        if v.kind != "claim" or v.pool is None or self._owned_at.get(v.id) == self._owners_seen:
            return
        self._owned_at[v.id] = self._owners_seen
        members = v.pool.members
        owners = [self._possible_owners(m) for m in members]
        if all(os == owners[0] for os in owners):
            # (tells nothing directly) through earlier shuffles: not the only card
            # of the other owner among the cards the shuffles mixed (e.g. one card
            # taken from the other player's hand)
            _, other = self._owner_leaves(v, owner)
            if len(other) == 1 and not any(x is other[0] for x in v.not_same):
                v.not_same.append(other[0])
            return
        ids = frozenset(m.id for m, os in zip(members, owners) if owner in os)
        if ids:
            v.allowed = ids if v.allowed is None else (v.allowed & ids) or v.allowed

    def _card_hint(self, m: dict) -> None:
        """Card hints stick to a physical card; after a shuffle the core shows them
        again at the card's new position, which tells which copy went where."""
        v = self.var_at(m["con"], m["loc"], m["seq"], m.get("pos", 0))
        if v is None:
            return
        mark = (m["hint_type"], m["value"])
        if m["hint_type"] == C.CHINT_DESC_REMOVE:
            v.marks = v.marks - {(C.CHINT_DESC_ADD, m["value"])}
            return
        if mark in v.marks:
            return
        if self._resent_hints and v.kind == "claim" and v.pool is not None and v.pool is self.pools[-1]:
            carriers = frozenset(x.id for x in v.pool.members if mark in x.marks)
            # (more slots showing it than cards that had it: not hints shown again
            # after the shuffle, but new ones)
            shown = [x for x in v.pool.claims if mark in x.marks and x is not v]
            if carriers and len(shown) < len(carriers):
                v.allowed = carriers if v.allowed is None else v.allowed & carriers
            elif carriers:
                for x in shown:
                    if x.allowed == carriers:
                        x.allowed = None
        v.marks = v.marks | {mark}

    def message(self, m: dict) -> None:
        t = m["type"]
        if self._top_check is not None:
            p, self._top_check = self._top_check, None
            if not (t == "DECK_TOP" and m["player"] == p) and self._faceup_in_deck:
                # a shuffled deck whose top card is face-up is announced (DECK_TOP):
                # this one was not, so its top card is none of the face-up ones
                deck = self.lists[p][C.LOCATION_DECK]
                top = deck[-1] if deck else None
                if top is not None and top.kind == "claim":
                    for x in self._faceup_in_deck:
                        if x is not top and not any(y is x for y in top.not_same):
                            top.not_same.append(x)
        if t == "SHUFFLE_DECK":
            self._top_check = m["player"]
        if t not in ("HINT", "CARD_HINT", "CHAIN_SOLVED"):
            self._prev_sig, self._last_sig = self._last_sig, (t, m.get("player"))
        if self._solved is not None and t != "CHAIN_SOLVED":
            var, player, left = self._solved
            self._solved = None
            if t == "SHUFFLE_HAND" and m["player"] == player:
                # the core shuffles the hand after a chain link activated from it
                # only if the card is still there face-down (SolveChain, right after
                # MSG_CHAIN_SOLVED): the cards that left the hand were other cards
                for v in left:
                    if v is not var and v.kind == "claim":
                        v.not_same.append(var)
        if t in ("SUMMONING", "SPSUMMONING", "NEW_PHASE", "CHAINING"):
            self._tributed_to_hand = []
        if t == "NEW_TURN":
            self.turn_player = m["player"]
        if t in ("CHAIN_SOLVED", "CHAIN_END", "NEW_PHASE", "NEW_TURN") \
                or (t == "CHAINING" and m["chain_count"] == 1):
            # (cards revealed e.g. as a cost stay relevant until their chain link
            # resolved; a new chain starts afresh)
            self._referenced = []
            self._referenced_piles = set()
            self._picked = []
            self._referenced_in_chain = False
        if t == "MOVE":
            self._last_move_to = (m["to"]["con"], m["to"]["loc"], m["to"]["seq"])
        if t == "MOVE" and m["from"]["loc"] and not m["from"]["loc"] & C.LOCATION_OVERLAY:
            frm, to = m["from"], m["to"]
            v = self._plain_var(frm["con"], frm["loc"], frm["seq"])
            if frm["loc"] == C.LOCATION_HAND and v is not None:
                self._left_hand.append(v)
                if self._chosen is not None and self._chosen[1] == frm["con"] and self._chosen[2] == m["code"] \
                        and to["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE):
                    self._identify(v, self._chosen[0])
                    self._chosen = None
        confirmed_move = t == "MOVE" and self._confirmed and self._confirmed_shuffled \
            and m["from"]["loc"] == C.LOCATION_DECK \
            and all(x == (m["from"]["con"], C.LOCATION_DECK) for x in self._confirmed_locs)
        if confirmed_move:
            # revealed, shuffled back and then taken out of that pile, card after
            # card (e.g. "reveal 3, shuffle, your opponent picks 1 to add, the rest
            # go to the GY"): each card taken is one of the revealed ones
            v = self._plain_var(m["from"]["con"], m["from"]["loc"], m["from"]["seq"])
            if v is not None and v.kind == "claim" and v.pool is not None:
                ids = frozenset(x.id for x in v.pool.members if any(x is y for y in self._confirmed))
                if ids:
                    v.allowed = ids if v.allowed is None else (v.allowed & ids) or v.allowed
            self._confirmed_moves += 1
            if self._confirmed_moves >= len(self._confirmed):
                self._confirmed = []
        elif t not in ("CONFIRM_CARDS", "CONFIRM_DECKTOP", "CONFIRM_EXTRATOP", "SHUFFLE_DECK", "SHUFFLE_HAND",
                       "SHUFFLE_EXTRA", "HINT", "CARD_HINT", "RANDOM_SELECTED",
                       # (the top card shown after the shuffle, e.g. decks turned over)
                       "DECK_TOP",
                       # (the picked card being summoned: the others are taken out next)
                       "SPSUMMONING", "SPSUMMONED", "SUMMONING", "SUMMONED"):
            self._confirmed = []  # only "reveal, shuffle back, pick one at random" counts
        if t == "SHUFFLE_DECK" and self._confirmed:
            self._confirmed_shuffled = True
        if t in ("CARD_HINT", "PLAYER_HINT") and m.get("value", 0) >> 20 \
                and m["hint_type"] in (C.CHINT_DESC_ADD, C.CHINT_DESC_REMOVE):  # (= PHINT_DESC_*)
            # a hint text names the card whose script it comes from (e.g. a global
            # check of Marincess Battle Ocean, which exists only if a copy is in the
            # deck): that player's deck contains the card (see flow: "contains")
            self.contains.add((m["con"] if t == "CARD_HINT" else m["player"], m["value"] >> 20))
        if t == "CARD_HINT":
            self._card_hint(m)
        elif t == "SHUFFLE_HAND":
            # field::shuffle shows the client hints of every hand card again
            self._resent_hints = True
        else:
            self._resent_hints = False
        if t == "MOVE" and (m.get("reason", 0) & (C.REASON_RELEASE | C.REASON_COST) or _leaves_field(m)):
            # (a card can also be sent instead of Tributed, e.g. with Floowandereeze
            # and the Unexplored Winds, or banished from a graveyard with Tribute
            # Burial: it counts for the Level of a Set monster)
            self._releases += 1
        elif t not in ("MOVE", "HINT", "CARD_HINT", "PLAYER_HINT", "MISSED_EFFECT", "SET", "SPSUMMONING",
                       "ADD_COUNTER", "REMOVE_COUNTER", "FIELD_DISABLED"):
            # (a counter placed because of the Tribute is a consequence, not a new
            # action; neither is a notice, e.g. a trigger missed because of a Tribute)
            self._releases = 0
        if t == "SET" and not m["code"]:
            # a hidden card was Set: its kind follows from the zone (and tributes)
            if m["loc"] == C.LOCATION_SZONE:
                self._require(m, ("field",) if m["seq"] == 5 else ("st",))
            elif m["loc"] == C.LOCATION_MZONE:
                self._require(m, ("mset", self._releases))
            self._releases = 0
        elif t == "SPSUMMONING" and not m["code"]:
            self._require(m, ("monster",))
        if t == "MOVE":
            frm, to = m["from"], m["to"]
            if frm["loc"] == C.LOCATION_DECK and frm.get("pos", 0) & C.POS_FACEUP:
                self._faceup_leaves(frm, m["code"])
            if frm["loc"] == 0:
                v = self._new(to["con"], "created")
            else:
                v = self._take(frm)
                if frm["loc"] == C.LOCATION_DECK:
                    self._left_deck(v)
                if frm["loc"] == C.LOCATION_HAND and frm.get("pos", 0) & C.POS_FACEUP and not m["code"] \
                        and v.kind == "claim" and v.pool is not None and v.code is None:
                    # (not if it was revealed since the shuffle: then it shows itself)
                    # a card leaves the hand face-up (revealed, e.g. kept public by its
                    # own effect) though its code is hidden now: one of the cards that
                    # were revealed in the hand before it was shuffled
                    ids = frozenset(x.id for x in v.pool.members if x.faceup)
                    if ids:
                        v.allowed = ids if v.allowed is None else (v.allowed & ids) or v.allowed
                if self._own_picks and v.kind == "claim" and v.pool is not None \
                        and frm["con"] == self.viewer and frm["loc"] == C.LOCATION_DECK \
                        and v.pool.location == C.LOCATION_DECK:  # (a card of the shuffled deck, not
                    # one put into it since, e.g. from the hand)
                    picked = [x for x, code in self._own_picks if code == m["code"]
                              and any(x is y for y in v.pool.members)]
                    if picked:
                        ids = frozenset(x.id for x in picked)
                        v.allowed = ids if v.allowed is None else (v.allowed & ids) or v.allowed
                        self._own_picks = [(x, c) for x, c in self._own_picks if x is not picked[0]]
                if v.kind == "claim" and v.pool is not None and self._referenced and not v.prefer \
                        and (frm["con"], frm["loc"]) in self._referenced_piles:
                    # (the first move out of the shuffled pile)
                    if not self._referenced_in_chain and to["loc"] == C.LOCATION_MZONE:
                        # cards revealed outside a chain are the cost of a summon
                        # procedure: the summoned card is most likely another one
                        # (e.g. Crimson Nova the Dark Cubic Lord reveals 3 other Cubic
                        # cards), but can be among them (e.g. Evil Dragon Cards reveal
                        # 3 LIGHT monsters including the card itself)
                        from .script_index import reveals_itself
                        if m.get("code") and reveals_itself(m["code"]):
                            others = [x for x in v.pool.members if not any(x is y for y in self._referenced)]
                            if others:
                                v.prefer = others
                        else:
                            v.not_same.extend(x for x in self._referenced if x is not v)
                        # (the reveal was that procedure's cost: it says nothing more)
                        self._referenced, self._referenced_piles, self._picked = [], set(), []
                    else:
                        pool = self._picked if any(x is not v for x in self._picked) else self._referenced
                        v.prefer = [x for x in pool if x is not v]
            if to["loc"] == C.LOCATION_MZONE and frm["loc"] == C.LOCATION_HAND and self._tributed_to_hand \
                    and v is not None and v.kind == "claim":
                v.not_same.extend(x for x in self._tributed_to_hand if x is not v)
            if v is not None and to["loc"] == C.LOCATION_HAND and m.get("reason", 0) & C.REASON_RELEASE:
                self._tributed_to_hand.append(v)
            if to["loc"]:
                self._put(v, to)
            self.observe(v, m["code"])
            if to["loc"] == C.LOCATION_DECK:
                self._deck_arrivals.append(v)
                if to.get("pos", 0) & C.POS_FACEUP:
                    self._faceup_in_deck.append(v)  # (its place in the deck stays public)
        elif t == "DRAW":
            p = m["player"]
            deck = self.lists[p][C.LOCATION_DECK]
            for c in m["cards"]:
                if not deck:
                    raise TrackerError("draw from empty deck")
                v = deck.pop()
                self._left_deck(v)
                v.faceup = bool(c["pos"] & C.POS_FACEUP)
                self.lists[p][C.LOCATION_HAND].append(v)
                self.observe(v, c["code"])
            if deck and self._prev_sig != ("DECK_TOP", p) and deck[-1].kind == "claim":
                # a draw that leaves a face-up card on top announces it (DECK_TOP
                # before the draw): this one did not, so the new top card is none of
                # the face-up ones
                top = deck[-1]
                for x in self._faceup_deck_cards(p):
                    if x is not top and not any(y is x for y in top.not_same):
                        top.not_same.append(x)
        elif t == "SHUFFLE_DECK":
            deck = self.lists[m["player"]][C.LOCATION_DECK]
            self._deck_arrivals = [x for x in self._deck_arrivals if not any(x is y for y in deck)]
            self._shuffle(m["player"], C.LOCATION_DECK)
        elif t == "SHUFFLE_HAND":
            p = m["player"]
            pool = self._shuffle(p, C.LOCATION_HAND)
            if len(m["codes"]) != len(pool.claims):
                raise TrackerError("hand size mismatch on shuffle")
            for v, code, pub in zip(pool.claims, m["codes"], m.get("public") or [False] * len(m["codes"])):
                v.faceup = pub
                self.observe(v, code)
                if pub:
                    # (a revealed card stays the same physical card: the one that
                    # was face-up in the hand before, if only one with its code was)
                    before = [x for x in pool.members if x.faceup and (x.code or x.fixed_code) == code]
                    if len(before) == 1:
                        self._identify(v, before[0])
        elif t == "SHUFFLE_EXTRA":
            p = m["player"]
            pool = self._shuffle(p, C.LOCATION_EXTRA)
            for v, code in zip(pool.claims, m["codes"]):
                self.observe(v, code)
        elif t == "SHUFFLE_SET_CARD":
            locs = m["old"]
            vars_ = [self.var_at(li["con"], li["loc"], li["seq"]) for li in locs]
            pool = self._make_pool(locs[0]["con"], "set", vars_)
            order = sorted(range(len(locs)), key=lambda i: locs[i]["seq"])
            pool.zones = [locs[i]["seq"] for i in order]
            pool.claims = [pool.claims[i] for i in order]
            for claim, i in zip(pool.claims, order):
                li = locs[i]
                claim.overlays = vars_[i].overlays
                claim.faceup = False
            for claim, zone in zip(pool.claims, pool.zones):
                self.zones[locs[0]["con"]][locs[0]["loc"]][zone] = claim
            self.set_shuffle_events.append(pool)
        elif t in ("SUMMONING", "SPSUMMONING", "FLIPSUMMONING", "SET"):
            v = self.var_at(m["con"], m["loc"], m["seq"], m["pos"])
            self.observe(v, m["code"])
        elif t == "CHAINING":
            v = self.var_at(m["con"], m["loc"], m["seq"], m["pos"])
            self.observe(v, m["code"])
            self._chain.append(v)
            self._chain_from_hand.append(m["loc"] == C.LOCATION_HAND)
        elif t == "CHAIN_SOLVING":
            self._chosen = None
            self._left_hand = []
            self._resolving = True
        elif t == "CHAIN_SOLVED":
            self._resolving = False
            if self._chain:
                var, from_hand = self._chain.pop(), self._chain_from_hand.pop()
                if from_hand and var is not None:
                    self._solved = (var, var.owner, self._left_hand)
                    if self._last_sig == ("SHUFFLE_HAND", var.owner) and self._left_hand:
                        # proc_workaround.lua shuffles the hand on EVENT_CHAIN_SOLVED
                        # (before the message) if the activating card moved while
                        # the link resolved: it is one of the cards that left the hand
                        cands = [v for v in self._left_hand
                                 if v.code is None or var.code is None or v.code == var.code]
                        if len(cands) == 1:
                            self._identify(cands[0], var)
            self._left_hand = []
        elif t == "CHAIN_END":
            self._chain = []
            self._chain_from_hand = []
        elif t == "POS_CHANGE":
            v = self.var_at(m["con"], m["loc"], m["seq"])
            if v is not None:
                v.faceup = bool(m["pos"] & C.POS_FACEUP)
            self.observe(v, m["code"])
        elif t in ("CONFIRM_CARDS", "CONFIRM_DECKTOP", "CONFIRM_EXTRATOP") and t == "CONFIRM_CARDS" \
                and self._confirmed and self._confirmed_shuffled \
                and all((c["con"], c["loc"]) in self._confirmed_locs
                        or (c["con"], c["loc"], c["seq"]) == self._last_move_to for c in m["cards"]) \
                and Counter(c["code"] for c in m["cards"]) <= Counter(
                    x.fixed_code if x.fixed_code is not None else x.code for x in self._confirmed):
            # (one of the revealed and shuffled-back cards revealed again, e.g. the one
            # picked at random: the revealed ones stay the ones taken out next)
            for c in m["cards"]:
                v = self.var_at(c["con"], c["loc"], c["seq"])
                if c["code"]:
                    self.observe(v, c["code"])
        elif t in ("CONFIRM_CARDS", "CONFIRM_DECKTOP", "CONFIRM_EXTRATOP"):
            self._confirmed = []
            self._confirmed_locs = [(c["con"], c["loc"]) for c in m["cards"]] if t == "CONFIRM_CARDS" else [None]
            self._confirmed_shuffled = False
            self._confirmed_moves = 0
            for c in m["cards"]:
                v = self.var_at(c["con"], c["loc"], c["seq"])
                if v is not None:
                    self._confirmed.append(v)
                    self._referenced.append(v)
                    self._referenced_piles.add((c["con"], c["loc"]))
                    self._referenced_in_chain = self._referenced_in_chain or bool(self._chain)
                    self._referenced_link = len(self._chain)
                if c["code"]:
                    self.observe(v, c["code"])
        elif t == "RANDOM_SELECTED":
            # after revealing cards and shuffling them back into their pile, a
            # random pick from that pile is a pick from the revealed cards (also
            # the later picks of the same resolution, e.g. Draw of Fate); and a
            # card taken from the pile later on is preferably the picked one
            # (e.g. picked before the shuffle, taken after it). Only a pick by the
            # chain link the cards were revealed for: another link picks from the
            # whole pile (e.g. a cost revealed for one link, a random banish from the
            # hand by the opponent's link resolving first)
            for c in m["cards"]:
                if self._referenced_link != len(self._chain):
                    break
                if c["loc"] & C.LOCATION_OVERLAY:
                    continue
                v = self._plain_var(c["con"], c["loc"], c["seq"])
                if v is None:
                    continue
                if (c["con"], c["loc"]) in self._referenced_piles:
                    self._picked.append(v)
                if v.kind == "claim" and v.pool is not None and (c["con"], c["loc"]) in self._referenced_piles:
                    ids = frozenset(x.id for x in v.pool.members if any(x is y for y in self._referenced))
                    if ids:
                        v.allowed = ids if v.allowed is None else (v.allowed & ids) or v.allowed
        elif t == "REVERSE_DECK":
            # both decks are turned over (field::reverse_deck)
            for p in (0, 1):
                self.lists[p][C.LOCATION_DECK].reverse()
        elif t == "DECK_TOP":
            deck = self.lists[m["player"]][C.LOCATION_DECK]
            idx = len(deck) - 1 - m["seq"]
            if 0 <= idx < len(deck) and m["code"]:  # (shown whatever the position says)
                v = deck[idx]
                self.observe(v, m["code"])
                if m.get("pos", 0) & C.POS_FACEUP and not any(x is v for x in self._faceup_in_deck):
                    # a card just put there (and turned face-up), or turned face-up
                    # where it is (e.g. Grave Lure: revealed, then turned): its place
                    # stays public. Otherwise a face-up card seen again on top (after a
                    # shuffle, a draw, a card taken from the top): the same card
                    same = [x for x in self._faceup_deck_cards(m["player"])
                            if self._code(x) == m["code"] and self._descends(v, x, m["player"])]
                    if any(x is v for x in self._deck_arrivals) or not same \
                            or (self._prev_sig == ("CONFIRM_DECKTOP", m["player"]) and not v.faceup):
                        self._faceup_in_deck.append(v)
                    elif len(same) == 1:
                        self._identify(v, same[0])
                elif not m.get("pos", 0) & C.POS_FACEUP and v.kind == "claim":
                    # (shown face-down, e.g. all decks turned over): none of the
                    # face-up cards in that deck
                    for x in self._faceup_deck_cards(m["player"]):
                        if self._code(x) == m["code"] and x is not v and not any(x is y for y in v.not_same):
                            v.not_same.append(x)
                v.faceup = bool(m.get("pos", 0) & C.POS_FACEUP)
        elif t == "SWAP":
            l1, l2 = m["loc1"], m["loc2"]
            v1 = self.var_at(l1["con"], l1["loc"], l1["seq"])
            v2 = self.var_at(l2["con"], l2["loc"], l2["seq"])
            self.observe(v1, m["code1"])
            self.observe(v2, m["code2"])
            self.zones[l1["con"]][l1["loc"]][l1["seq"]] = v2
            self.zones[l2["con"]][l2["loc"]][l2["seq"]] = v1
        elif t == "MISSED_EFFECT":
            self.observe(self.var_at(m["con"], m["loc"], m["seq"], m["pos"]), m["code"])
        elif t == "SWAP_GRAVE_DECK":
            self._swap_grave_deck(m)
        elif t == "REMOVE_CARDS":
            for li in m["cards"]:
                self._take(li)
        elif t in ("TAG_SWAP", "RELOAD_FIELD", "REVERSE_DECK"):
            raise TrackerError(f"unsupported message {t}")

    def _shuffle(self, player: int, location: int) -> Pool:
        lst = self.lists[player][location]
        if location == C.LOCATION_EXTRA:
            n = sum(1 for v in lst if not v.faceup)
            members, rest = lst[:n], lst[n:]
        else:
            members, rest = lst, []
        pool = self._make_pool(player, location, members)
        self.lists[player][location] = list(pool.claims) + rest
        self.shuffle_events.append(pool)
        return pool

    def _swap_grave_deck(self, m: dict) -> None:
        p = m["player"]
        grave = self.lists[p][C.LOCATION_GRAVE]
        deck = self.lists[p][C.LOCATION_DECK]
        mask = bytes.fromhex(m["extra_mask"])
        new_deck, to_extra = [], []
        for i, v in enumerate(grave):
            if i // 8 < len(mask) and mask[i // 8] & (1 << (i % 8)):
                to_extra.append(v)
            else:
                new_deck.append(v)
        for v in deck:
            v.faceup = True
        for v in new_deck + to_extra:
            v.faceup = False
        self.lists[p][C.LOCATION_GRAVE] = deck
        self.lists[p][C.LOCATION_DECK] = new_deck
        extra = self.lists[p][C.LOCATION_EXTRA]
        n_fd = sum(1 for v in extra if not v.faceup)
        self.lists[p][C.LOCATION_EXTRA] = extra[:n_fd] + to_extra + extra[n_fd:]

    # (an activation moves its card before the CHAINING, so no shuffle can
    # come in between; and the card that moves later may be another one)
    _COMMAND_LISTS = {"summon": "summonable", "spsummon": "spsummonable", "mset": "msetable",
                      "sset": "ssetable"}

    def response(self, r: dict) -> None:
        """The viewer's own command names a physical card: the hand card it
        summons, sets or activates is the one that moves to the field next
        (even if the hand is shuffled in between, e.g. after a reveal)."""
        p = self._last_prompt
        if p is not None and p["type"] == "SELECT_CARD" and r.get("indices"):
            # the viewer picked cards of its own deck: a card of that code that
            # leaves the deck after it was shuffled is one of the picked ones (e.g.
            # "shuffle the deck, then place it on top")
            for i in r["indices"]:
                c = p["cards"][i] if i < len(p["cards"]) else None
                if c and c["con"] == self.viewer and c["loc"] == C.LOCATION_DECK and c.get("code"):
                    v = self._plain_var(c["con"], c["loc"], c["seq"])
                    if v is not None:
                        self._own_picks.append((v, c["code"]))
            return
        if p is None or p["type"] != "SELECT_IDLECMD" or r.get("kind") not in self._COMMAND_LISTS:
            return
        entries = p.get(self._COMMAND_LISTS[r["kind"]], ())
        i = r.get("index", -1)
        if 0 <= i < len(entries) and entries[i].get("loc") == C.LOCATION_HAND and entries[i].get("code"):
            e = entries[i]
            self._chosen = (self.var_at(e["con"], e["loc"], e["seq"]), e["con"], e["code"])

    def prompt(self, p: dict) -> None:
        self._last_prompt = p
        if p["type"] == "SELECT_CARD" and self._confirmed and len(p["cards"]) == len(self._confirmed) \
                and all(not c.get("code") and not c["loc"] & C.LOCATION_OVERLAY for c in p["cards"]):
            # right after revealing cards and shuffling them back into their pile,
            # the viewer is offered exactly that many hidden cards of the pile: the
            # revealed ones (e.g. "your opponent picks 1 of them")
            for c in p["cards"]:
                v = self._plain_var(c["con"], c["loc"], c["seq"])
                if v is not None and v.kind == "claim" and v.pool is not None:
                    ids = frozenset(x.id for x in v.pool.members if any(x is y for y in self._confirmed))
                    if ids:
                        v.allowed = ids if v.allowed is None else (v.allowed & ids) or v.allowed
        if p["type"] in ("SELECT_IDLECMD", "SELECT_BATTLECMD"):
            self._chosen = None
            self._tributed_to_hand = []
            self._own_picks = []  # (the action that picked them is over)
        self._releases = 0
        # (option lists are shown in a canonical order: their order carries no
        # information, see observe.option_order)
        for key in ("summonable", "spsummonable", "repositionable", "msetable", "ssetable",
                    "activatable", "attackable", "chains", "cards", "select_cards",
                    "unselect_cards", "must_cards"):
            for c in p.get(key, ()):
                if not c.get("code") or "loc" not in c:
                    continue
                loc = c["loc"]
                if loc == 0:
                    continue
                pos = c.get("pos", 0)
                self.observe(self.var_at(c["con"], loc, c["seq"], pos), c["code"])
        if p["type"] == "SELECT_EFFECTYN" and p.get("code"):
            self.observe(self.var_at(p["con"], p["loc"], p["seq"], p["pos"]), p["code"])

    @staticmethod
    def _descends(v: Var, x: Var, player: int) -> bool:
        """Whether `v` can be `x` or a later claim on it through the shuffles of
        `player`'s deck since. Only through those (a card that left the deck and
        came back, e.g. through the hand whose cards came from the deck, is
        another arrival), and not through slots known to hold another card."""
        def code(y: Var) -> int | None:
            return y.fixed_code if y.fixed_code is not None else y.code
        want = code(x)
        frontier, seen = [v], set()
        while frontier:
            y = frontier.pop()
            if y is x:
                return True
            if want is not None and code(y) is not None and code(y) != want:
                continue
            if y.kind != "claim" or y.pool is None or y.pool.index in seen \
                    or (y.pool.player, y.pool.location) != (player, C.LOCATION_DECK):
                continue
            seen.add(y.pool.index)
            frontier.extend(y.pool.members)
        return False

    @staticmethod
    def _code(x: Var) -> int | None:
        return x.fixed_code if x.fixed_code is not None else x.code

    def _left_deck(self, v: Var | None) -> None:
        """`v` left the deck: if it is one of the face-up cards there (or was
        identified with one), that one is gone."""
        if v is not None and self._faceup_in_deck:
            self._faceup_in_deck = [x for x in self._faceup_in_deck if x is not v and v.same_as is not x]

    def _faceup_leaves(self, frm: dict, code: int) -> None:
        """A face-up card leaves the deck: that one of the face-up cards there."""
        v = self._plain_var(frm["con"], frm["loc"], frm["seq"])
        code = code or (self._code(v) if v is not None else None)  # (e.g. Set: seen in the deck)
        if not code:
            # (e.g. Set from the deck since the last view of it: told later, e.g.
            # when the card is revealed)
            if v is not None:
                self._faceup_left.append((v, frm["con"]))
            return
        self._faceup_gone(v, frm["con"], code)

    def _faceup_gone(self, v: Var | None, player: int, code: int) -> None:
        cands = [x for x in self._faceup_deck_cards(player) if self._code(x) == code]
        # (of several face-up copies any: they are alike)
        x = next((x for x in cands if x is v or (v is not None and v.same_as is x)), None) \
            or (cands[0] if cands else None)
        if x is None:
            return
        if v is not None and v is not x:
            self._identify(v, x)
        self._faceup_in_deck = [y for y in self._faceup_in_deck if y is not x]

    def _faceup_deck_cards(self, player: int) -> list[Var]:
        """The face-up cards in a player's deck: those put there face-up (or turned
        face-up there) that are still there."""
        pending = [(v, p) for v, p in self._faceup_left if self._code(v)]
        if pending:
            self._faceup_left = [(v, p) for v, p in self._faceup_left if not self._code(v)]
            for v, p in pending:
                self._faceup_gone(v, p, self._code(v))
        deck = self.lists[player][C.LOCATION_DECK]
        return [x for x in self._faceup_in_deck if any(self._descends(v, x, player) for v in deck)]

    def _identify(self, v: Var | None, same: Var | None) -> None:
        """`v` is known to be the physical card that was `same` before the last
        shuffle of its pile."""
        if v is None or same is None or v is same or v.kind != "claim" or v.same_as is not None:
            return
        # `same` must be an ancestor: a member of v's pool, or of the pools before
        frontier, seen = [v], set()
        while frontier:
            x = frontier.pop()
            if x.pool is None or x.pool.index in seen:
                continue
            seen.add(x.pool.index)
            if any(m is same for m in x.pool.members):
                v.same_as = same
                return
            frontier.extend(m for m in x.pool.members if m.kind == "claim")

    def state(self, s: dict) -> None:
        from ..state import LOCATIONS
        # the chain shows where each link's card is now (e.g. after its pile was shuffled)
        from ..observe import loc_visible
        for link, var in zip(s.get("chain", ()), self._chain):
            h = link.get("handler")
            # (positions in the deck and in hidden piles are masked in the view)
            if h and h.get("loc") and not h["loc"] & (C.LOCATION_OVERLAY | C.LOCATION_DECK) \
                    and loc_visible(h, self.viewer):
                self._identify(self._plain_var(h["con"], h["loc"], h["seq"]), var)
            elif h and h.get("loc") == C.LOCATION_DECK and var is not None:
                # (where in the deck is masked) the link's card is in the deck: no
                # card outside of it is that card (e.g. it was shuffled back and
                # other cards excavated)
                for p in (0, 1):
                    outside = [v for zone in (C.LOCATION_MZONE, C.LOCATION_SZONE)
                               for v in self.zones[p][zone] if v is not None]
                    outside += [v for loc, lst in self.lists[p].items() if loc != C.LOCATION_DECK for v in lst]
                    for v in outside:
                        if v.kind == "claim" and v is not var and not any(x is var for x in v.not_same):
                            v.not_same.append(var)
        for pl_idx, pl in enumerate(s["players"]):
            for name, loc in LOCATIONS:
                cards = pl[name]
                if loc in ZONE_LOCS:
                    for seq, card in enumerate(cards):
                        v = self.zones[pl_idx][loc][seq]
                        if (card is None) != (v is None):
                            raise TrackerError(f"zone mismatch P{pl_idx} {name}{seq}")
                        if card is None:
                            continue
                        self.observe(v, card.get("code", 0))
                        if "owner" in card and not card.get("code"):
                            # (a face-down card's owner shows too)
                            self._owned(v, card["owner"])
                        overlays = card.get("overlays", [])
                        if len(overlays) != len(v.overlays):
                            raise TrackerError(f"overlay count mismatch P{pl_idx} {name}{seq}")
                        for ov, code in zip(v.overlays, overlays):
                            self.observe(ov, code)
                else:
                    lst = self.lists[pl_idx][loc]
                    if len(cards) != len(lst):
                        raise TrackerError(f"size mismatch P{pl_idx} {name}: {len(cards)} vs {len(lst)}")
                    used: set[int] = set()
                    shown = Counter(c.get("code") for c in cards if loc == C.LOCATION_DECK and c.get("code")
                                    and c.get("position", 0) & C.POS_FACEUP)
                    faceup_now = self._faceup_deck_cards(pl_idx) if shown else []
                    # (the owners of deck cards show too: they tell apart only where a
                    # card of the other player is in the deck, e.g. one shuffled in)
                    foreign = loc == C.LOCATION_DECK and any(c.get("owner", pl_idx) != pl_idx for c in cards)
                    for v, card in zip(lst, cards):
                        self.observe(v, card.get("code", 0))
                        if loc == C.LOCATION_DECK and card.get("code") and card.get("position", 0) & C.POS_FACEUP:
                            # a face-up card in the deck: one put there face-up (of several
                            # such copies any, each once: they are alike)
                            known = [x for x in faceup_now
                                     if (x.fixed_code if x.fixed_code is not None else x.code) == card["code"]]
                            same = [x for x in known if id(x) not in used]
                            # (only if the shown copies are exactly the known ones: another
                            # copy can have been turned face-up unseen)
                            if same and len(known) == shown[card["code"]]:
                                used.add(id(same[0]))
                                self._identify(v, same[0])
                        v.faceup = bool(card.get("position", 0) & C.POS_FACEUP)
                        if "owner" in card and (loc == C.LOCATION_HAND or foreign):
                            self._owned(v, card["owner"])
                    if loc == C.LOCATION_HAND:
                        # (a card being activated from the hand is public too, but
                        # only while it is on the chain: not kept revealed)
                        on_chain = {h["seq"] for h in (link.get("handler") for link in s.get("chain", ()))
                                    if h and h.get("con") == pl_idx and h.get("loc") == C.LOCATION_HAND}
                        self._public_hand(lst, [{**c, "is_public": 0} if i in on_chain else c
                                                for i, c in enumerate(cards)])

    # -- solving -------------------------------------------------------------
    def solve(self, completion=None, card_db=None) -> Hypothesis:
        from .flow import FlowSolver
        return FlowSolver(self, completion, card_db).solve()


class Solver:
    """Assigns claims to pool members and codes to free atoms."""

    def __init__(self, tracker: Tracker, completion, card_db):
        self.t = tracker
        self.completion = completion
        self.db = card_db
        self.copies: list[Counter] = [Counter(), Counter()]  # codes assigned to free atoms
        self.free_atoms: list[list[Var]] = [[], []]
        self._memo: dict = {}
        self.rng = None
        self.pinned: set[int] = set()  # vars whose code follows from an observation
        for p in (0, 1):
            for a in tracker.atoms[p]:
                if a.fixed_code is None:
                    self.free_atoms[p].append(a)

    # value of a var if already determined
    def _known(self, v: Var) -> int | None:
        while True:
            if v.fixed_code is not None:
                return v.fixed_code
            if v.resolved is not None:
                return v.resolved
            if v.assigned is None:
                return v.code
            v = v.assigned

    def _base(self, code: int) -> int:
        if self.db is None:
            return code
        card = self.db.get(code)
        if card is not None and card.alias and abs(card.alias - card.code) < 10:
            return card.alias
        return code

    def _atom_can_take(self, a: Var, code: int) -> bool:
        if a.fixed_code is not None:
            return a.fixed_code == code
        if a.resolved is not None:
            return a.resolved == code
        if self.db is not None:
            card = self.db.get(code)
            if card is None:
                return False
            if (a.kind == "extra") != card.is_extra_deck:
                return False
        return self.copies[a.owner][self._base(code)] < 3

    def _can(self, v: Var, code: int) -> bool:
        """Whether var v can be made to have `code` (no side effects).

        Memoized per top-level assignment: an assignment only changes pools
        that are not queried again while it descends to earlier pools.
        """
        key = (v.id, code)
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        if (v.code is not None and v.code != code) or code in v.excluded:
            res = False
        elif v.kind in ("main", "extra"):
            res = self._atom_can_take(v, code)
        elif v.kind == "created":
            res = v.code is None or v.code == code
        elif v.assigned is not None:
            res = self._can(v.assigned, code)
        elif v.resolved is not None:
            res = v.resolved == code
        else:
            pool = v.pool
            res = any(m.id not in pool.used and (v.allowed is None or m.id in v.allowed)
                      and self._can(m, code) for m in pool.members)
        self._memo[key] = res
        return res

    def _member_rank(self, m: Var, code: int) -> int:
        k = self._known(m)
        if k == code:
            return 0  # a card we already know is this code
        if m.kind in ("main", "extra"):
            return 2  # a brand new copy
        return 1

    def _assignments(self, v: Var, code: int):
        """Generator over the ways to make var v have `code`.

        Each yield leaves one consistent assignment in place; resuming the
        generator undoes it and moves on to the next alternative. Members
        are tried in preference order (cards already known to have this code
        first, brand new copies last).
        """
        if code in v.excluded:
            return
        if v.kind in ("main", "extra"):
            if v.fixed_code is not None or v.resolved is not None:
                if self._known(v) == code:
                    yield
                return
            if not self._atom_can_take(v, code):
                return
            v.resolved = code
            self.copies[v.owner][self._base(code)] += 1
            try:
                yield
            finally:  # (also when the search abandons this generator)
                self.copies[v.owner][self._base(code)] -= 1
                v.resolved = None
            return
        if v.kind == "created":
            if v.resolved is not None:
                if v.resolved == code:
                    yield
                return
            if v.code is not None and v.code != code:
                return
            v.resolved = code
            try:
                yield
            finally:
                v.resolved = None
            return
        if v.code is not None and v.code != code:
            return
        if v.assigned is not None:
            yield from self._assignments(v.assigned, code)
            return
        pool = v.pool
        self._memo = {}
        cands = [m for m in pool.members if m.id not in pool.used and self._can(m, code)
                 and (v.allowed is None or m.id in v.allowed)]
        if self.rng is not None:
            self.rng.shuffle(cands)  # random tie-breaking (the sort below is stable)
        cands.sort(key=lambda m: self._member_rank(m, code))
        for m in cands:
            if m.id in pool.used:
                continue
            pool.used.add(m.id)
            v.assigned = m
            try:
                yield from self._assignments(m, code)
            finally:
                v.assigned = None
                pool.used.discard(m.id)

    @staticmethod
    def _abandon(stack: list) -> None:
        """Undoes a partial search (latest choice first)."""
        while stack:
            stack.pop().close()

    def _reset_assignments(self) -> None:
        for v in self.t.all_vars:
            v.assigned = None
            if v.kind in ("main", "extra", "created"):
                v.resolved = None
        for pool in self.t.pools:
            pool.used = set()
        self.copies = [Counter(), Counter()]
        self.pinned = set()

    def _resolve_observations_restarting(self, tries: int = 30, budget: int = 20_000) -> None:
        """Depth first search; on failure retried with random tie-breaking.

        Chronological backtracking can thrash when a conflict only shows up
        much later in the stream, while consistent assignments are usually
        plentiful, so a few randomized restarts are far cheaper.
        """
        import random
        error = None
        for attempt in range(tries):
            self.rng = None if attempt == 0 else random.Random(attempt)
            try:
                self._resolve_observations(budget)
                return
            except TrackerError as e:
                error = e
                self._reset_assignments()
        raise error

    def _need_assignments(self, v: Var):
        """Ways to give a hidden card of a known kind a fitting code: a card the
        owner is known to have first, else a stand-in from the completion."""
        from .completion import DeckCompletion, fits_need
        if v.code is not None or self._known(v) is not None:
            yield  # revealed later: the observation decides
            return
        known = []
        for a in self.t.atoms[v.owner]:
            k = self._known(a)
            if k and k not in known and self.db is not None and fits_need(self.db.get(k), v.need):
                known.append(k)
        for code in known:
            self._memo = {}
            yield from self._assignments(v, code)
        # stand-ins are interchangeable: backtracking over them is pointless, so
        # only the first one that fits is tried
        comp = self.completion if self.completion is not None else DeckCompletion()
        stand_ins = comp.stand_ins(v.need) if self.db is not None else []
        for code in stand_ins:
            if code in known:
                continue
            self._memo = {}
            gen = self._assignments(v, code)
            try:
                next(gen)
            except StopIteration:
                continue
            yield
            gen.close()
            return

    def _resolve_observations(self, budget: int = 200_000) -> None:
        """Depth first search over the assignment choices of all observations
        and kind requirements, in stream order."""
        obs = self.t.searched
        stack = []
        i = 0
        steps = 0
        while i < len(obs):
            if len(stack) == i:
                v, code = obs[i]
                stack.append(self._assignments(v, code) if code is not None else self._need_assignments(v))
            steps += 1
            if steps > budget:
                self._abandon(stack)
                raise TrackerError("hypothesis search budget exceeded")
            try:
                next(stack[i])
                i += 1
            except StopIteration:
                stack.pop()
                i -= 1
                if i < 0:
                    v, code = obs[0]
                    raise TrackerError(f"observations are inconsistent (first: {code} for {v})")
        # the suspended generators hold the assignment: keep them alive
        self._stack = stack
        # vars on the resolution chain of an observation
        for v, _ in self.t.observations:
            while v is not None and v.id not in self.pinned:
                self.pinned.add(v.id)
                v = v.assigned

    def solve(self) -> Hypothesis:
        t = self.t
        # 1. every observation (depth first search with backtracking)
        self._resolve_observations_restarting()

        # 2. fill the remaining free atoms with guessed codes
        unknown = 0
        for p in (0, 1):
            free = [a for a in self.free_atoms[p] if a.resolved is None]
            if not free:
                continue
            unknown += len(free)
            known_main = [self._known(a) for a in t.atoms[p] if a.kind == "main" and self._known(a)]
            known_extra = [self._known(a) for a in t.atoms[p] if a.kind == "extra" and self._known(a)]
            n_main = sum(1 for a in free if a.kind == "main")
            n_extra = sum(1 for a in free if a.kind == "extra")
            if self.completion is None:
                raise TrackerError("unknown opponent cards but no deck completion available")
            fill_main, fill_extra = self.completion.complete(known_main, known_extra, n_main, n_extra)
            it_main, it_extra = iter(fill_main), iter(fill_extra)
            for a in free:
                a.resolved = next(it_main) if a.kind == "main" else next(it_extra)
        # 3. assign the remaining claims pool by pool (earliest first); members
        #    belong to earlier pools, so their codes are final by now
        for pool in t.pools:
            free_members = [m for m in pool.members if m.id not in pool.used]
            open_claims = [c for c in pool.claims if c.assigned is None]
            for c, m in zip(open_claims, self._match_claims(open_claims, free_members)):
                c.assigned = m
                pool.used.add(m.id)
        return self.finish(unknown)

    def finish(self, unknown: int) -> Hypothesis:
        """Orders, shuffles and copy ranks from a complete assignment (every
        claim assigned, every atom resolved)."""
        t = self.t

        def final(v: Var) -> int:
            k = self._known(v)
            if k is None:
                raise TrackerError(f"unresolved {v}")
            return k

        def atom(v: Var) -> Var | None:
            while v is not None and v.kind not in ("main", "extra", "created"):
                v = v.assigned
            if v is None or v.kind == "created":
                return None
            return v

        ranks: dict[int, int] = {}
        for p in (0, 1):
            seen: dict[tuple, int] = {}
            for a in t.atoms[p]:
                k = (a.kind, final(a))
                ranks[a.id] = seen.get(k, 0)
                seen[k] = ranks[a.id] + 1

        def rank_list(vs: list[Var]) -> list[int] | None:
            out = []
            for v in vs:
                a = atom(v)
                if a is None:
                    return None
                out.append(ranks[a.id])
            return out

        decks, deck_orders, extra_orders = [], [], []
        deck_ranks, extra_ranks = [], []
        for p in (0, 1):
            di = self._initial_index(p, C.LOCATION_DECK)
            deck_ranks.append(rank_list(t.initial_pools[di].claims))
            ei = self._initial_index(p, C.LOCATION_EXTRA)
            extra_ranks.append(rank_list([a for a in t.atoms[p] if a.kind == "extra"]) if ei is None
                               else rank_list(t.initial_pools[ei].claims))
        for p in (0, 1):
            main = [final(a) for a in t.atoms[p] if a.kind == "main"]
            extra = [final(a) for a in t.atoms[p] if a.kind == "extra"]
            decks.append({"main": main, "extra": extra})
            deck_orders.append([final(c) for c in t.initial_pools[self._initial_index(p, C.LOCATION_DECK)].claims])
            ei = self._initial_index(p, C.LOCATION_EXTRA)
            if ei is None:
                extra_orders.append(extra)  # (the viewer's own extra deck, in its order)
            else:
                extra_orders.append([final(c) for c in t.initial_pools[ei].claims])
        def owner_list(vs: list[Var]) -> list[int] | None:
            # (whose deck list each card is from: e.g. a card taken from the other
            # player's deck into a hand)
            out = []
            for v in vs:
                a = atom(v)
                if a is None:
                    return None
                out.append(a.owner)
            return out

        shuffles = [{"player": pool.player, "location": pool.location,
                     "codes": [final(c) for c in pool.claims], "ranks": rank_list(pool.claims),
                     "owners": owner_list(pool.claims)}
                    for pool in t.shuffle_events]
        set_shuffles = [[final(c) for c in pool.claims] for pool in t.set_shuffle_events]
        for v in t.exclusions:
            if final(v) in v.excluded:
                raise TrackerError(f"{v} is {final(v)}, which was excluded")
        var_codes = {v.id: self._known(v) for v in t.all_vars}
        return Hypothesis(decks=decks, deck_orders=deck_orders, extra_orders=extra_orders,
                          shuffles=shuffles, set_card_shuffles=set_shuffles, unknown_atoms=unknown,
                          var_codes=var_codes, pinned=set(self.pinned),
                          observed={v.id for v, _ in t.observations},
                          deck_ranks=deck_ranks, extra_ranks=extra_ranks)

    def _match_claims(self, claims: list[Var], members: list[Var]) -> list[Var]:
        """Members for the claims in order, respecting excluded codes and allowed
        members (a bipartite matching; without constraints simply in order)."""
        if not any(c.excluded or c.allowed is not None for c in claims):
            return members[:len(claims)]
        codes = [self._known(m) for m in members]
        owner: dict[int, int] = {}  # member index -> claim index

        def augment(ci: int, seen: set) -> bool:
            c = claims[ci]
            for mi, code in enumerate(codes):
                if mi in seen or code in c.excluded or (c.allowed is not None and members[mi].id not in c.allowed):
                    continue
                seen.add(mi)
                if mi not in owner or augment(owner[mi], seen):
                    owner[mi] = ci
                    return True
            return False

        # constrained claims first, so the rest keep the plain order where possible
        for ci in sorted(range(len(claims)), key=lambda i: not (claims[i].excluded or claims[i].allowed)):
            if not augment(ci, set()):
                raise TrackerError(f"no card left for {claims[ci]} outside {set(claims[ci].excluded)}")
        by_claim = {ci: mi for mi, ci in owner.items()}
        return [members[by_claim[ci]] for ci in range(len(claims))]

    def _initial_index(self, player: int, location: int) -> int | None:
        for i, pool in enumerate(self.t.initial_pools):
            if pool.player == player and pool.location == location:
                return i
        return None
