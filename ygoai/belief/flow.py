"""Exact hypothesis construction as an integer flow problem.

The tracker describes the hidden information as a network: every card slot
(a `Var`) is produced either by the start of the duel (an "atom", a physical
card of a deck list), by a shuffle (a claim of a `Pool`) or by an effect that
creates a card, and it is consumed by at most one later shuffle (as a member of
that pool). Each slot carries exactly one card code, and a shuffle only
permutes cards, so every pool conserves the multiset of codes.

That makes the deduction a multi-commodity flow over a small DAG whose nodes
are the pools. Slots between the same two nodes that are not constrained
individually are aggregated into one edge with a capacity (a count), so the
model has a few thousand integer variables, which HiGHS
(`scipy.optimize.milp`) solves exactly and deterministically:

* observed slots have their code;
* slots whose kind is known (a Set spell/trap, a Set monster of some level)
  take a known code of that kind or a typed wildcard ("some spell/trap");
* excluded codes are forbidden for their slot;
* the opponent has at most 3 copies of a card, main and extra deck cards stay
  in their section;
* the objective prefers wildcards to assuming additional copies of known cards.

The solution is turned into concrete assignments: codes per slot, a matching of
claims to members per pool (respecting which physical copy a card hint showed),
wildcards filled by the deck completion, and copy ranks (which physical card
each atom is).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .. import constants as C
from .tracker import Hypothesis, Pool, Solver, Tracker, TrackerError, Var

# typed wildcards: an unknown card of some kind, filled by the deck completion
WILD_MAIN, WILD_EXTRA = "main", "extra"
_NEED_WILDS = {("st",): "st", ("field",): "field", ("monster",): "monster"}


# identity sets without known codes are modelled only while small (every member
# needs variables of its own)
_MAX_UNTAGGED_ALLOWED = 8


def _need_wilds(need: tuple) -> list[str]:
    if need[0] == "mset":
        return [f"mset{min(need[1], 2)}"]
    if need[0] == "monster":
        return ["monster", "mset0", "mset1", "mset2"]
    return [_NEED_WILDS[need]]


def _wild_need(wild: str) -> tuple | None:
    if wild.startswith("mset"):
        return ("mset", int(wild[4:]))
    return {"st": ("st",), "field": ("field",), "monster": ("monster",)}.get(wild)


class FlowSolver:
    def __init__(self, tracker: Tracker, completion, card_db):
        self.t = tracker
        self.completion = completion
        self.db = card_db
        self._legacy = Solver(tracker, completion, card_db)  # helpers (_known, finish)

    # -- helpers ------------------------------------------------------------------
    def _base(self, code: int) -> int:
        return self._legacy._base(code)

    def _is_extra(self, code: int) -> bool:
        card = self.db.get(code) if self.db is not None else None
        return bool(card is not None and card.is_extra_deck)

    def _fits(self, code: int, need: tuple, usual: bool = True) -> bool:
        from .completion import fits_need
        return self.db is not None and fits_need(self.db.get(code), need, usual)

    def _joint_need_codes(self) -> set[int]:
        """Cards for slots whose requirement differs in kind from those of all the
        cards they can be (e.g. a monster Set with a Tribute from among cards that
        were Set as Spells/Traps): no wildcard has both kinds, only a card that fits
        both can (a monster that can be Set as a Spell, e.g. Sunlit Sentinel)."""
        if self.db is None:
            return set()
        from .completion import fits_need, monster_sset_codes
        out: set[int] = set()
        for v in self.t.all_vars:
            if v.need is None or v.kind != "claim" or v.pool is None:
                continue
            needs = {m.need for m in v.pool.members}
            if None in needs or all(n[0] == v.need[0] for n in needs):
                continue
            for code in monster_sset_codes():
                card = self.db.get(code)
                if card is not None and fits_need(card, v.need, False) and all(fits_need(card, n, False) for n in needs):
                    out.add(code)
        return out

    # -- model ----------------------------------------------------------------------
    def solve(self) -> Hypothesis:
        t = self.t
        viewer = t.viewer
        if self.db is not None:
            from .completion import fits_need
            byid_ = {v.id: v for v in t.all_vars}
            for v in t.all_vars:
                # (a guess which revealed card a slot is, see Tracker "confirmed_move",
                # that none of them can be: e.g. a card Set in a spell/trap zone after
                # monsters were revealed)
                if v.allowed is not None and v.need is not None:
                    codes = [byid_[a].fixed_code or byid_[a].code for a in v.allowed if a in byid_]
                    if codes and all(c and not fits_need(self.db.get(c), v.need, False) for c in codes):
                        v.allowed = None
        consumer: dict[int, int] = {}
        for pool in t.pools:
            for m in pool.members:
                consumer[m.id] = pool.index

        # commodities: (None, code) for known codes (whoever owns the card: a
        # hand can hold cards of both players), (owner, wild) for wildcards
        known: set[int] = set()
        for v in t.all_vars:
            c = v.fixed_code if v.fixed_code is not None else v.code
            if c:
                known.add(c)
        known |= self._joint_need_codes()
        # cards named by hints that no card of that player is known to be: its
        # deck contains a copy (not for cards of the viewer's own deck list: the
        # hint can come from the viewer's effect)
        own_list = {a.fixed_code for a in t.atoms[viewer]}
        owned = {(v.owner, v.fixed_code if v.fixed_code is not None else v.code) for v in t.all_vars}
        contains = []
        for p, code in sorted(t.contains):
            card = self.db.get(code) if self.db is not None else None
            if p == viewer or card is None or card.type & C.TYPE_TOKEN or code in own_list \
                    or (p, code) in owned:
                continue
            contains.append((p, code))
            known.add(code)
        commodities: list[tuple] = []
        cindex: dict[tuple, int] = {}

        def com(owner, key) -> int:
            k = (owner, key)
            if k not in cindex:
                cindex[k] = len(commodities)
                commodities.append(k)
            return cindex[k]

        wilds = [WILD_MAIN, WILD_EXTRA, "st", "field", "monster", "mset0", "mset1", "mset2"]
        # known codes: one commodity per owner (a card keeps its owner wherever it
        # goes, and state views show the owner of hidden cards in the hand)
        main_codes: dict[int, list[int]] = {0: [], 1: []}
        extra_codes: dict[int, list[int]] = {0: [], 1: []}
        token_codes: dict[int, list[int]] = {0: [], 1: []}
        for code in sorted(known):
            card = self.db.get(code) if self.db is not None else None
            for p in (0, 1):
                if card is not None and card.type & C.TYPE_TOKEN:
                    token_codes[p].append(com(p, code))
                else:
                    (extra_codes if self._is_extra(code) else main_codes)[p].append(com(p, code))
        opp = 1 - viewer
        main_wilds = [com(opp, w) for w in wilds if w != WILD_EXTRA]
        extra_wilds = [com(opp, WILD_EXTRA)]

        # tags: physical cards the stream singles out (identity constraints refer
        # to them). A tag is a commodity of its own with a known code; a slot
        # referred to later may also be a card tagged earlier.
        byid = {v.id: v for v in t.all_vars}
        refs: dict[int, Var] = {}
        for v in t.all_vars:
            if v.same_as is not None:
                refs[v.same_as.id] = v.same_as
            for x in v.not_same:
                refs[x.id] = x
            if v.allowed is not None:
                for a in v.allowed:
                    if a in byid:
                        refs[a] = byid[a]

        def known_code(v: Var) -> int | None:
            c = v.fixed_code if v.fixed_code is not None else v.code
            return c or None
        # (an earlier slot is the same card as a later one whose code shows, e.g. a
        # drawn card that activates after the hand was shuffled: it has that code)
        same_code: dict[int, int] = {}
        for v in t.all_vars:
            if v.same_as is not None and known_code(v):
                same_code.setdefault(v.same_as.id, known_code(v))
        tag_of: dict[int, int] = {}  # ref var id -> its tag commodity
        tags_by_code: dict[int, list[int]] = defaultdict(list)
        for rid in sorted(refs):
            code = known_code(refs[rid]) or same_code.get(rid)
            if code is None:
                continue
            k = com(None, ("tag", rid, code))
            tag_of[rid] = k
            tags_by_code[code].append(k)
        # the tags a ref can carry: its own or one of an earlier ref of that code
        ref_tags: dict[int, list[int]] = {}
        for rid, k in tag_of.items():
            code = commodities[k][1][2]
            ref_tags[rid] = [q for q in tags_by_code[code] if commodities[q][1][1] <= rid]

        def code_of(k: int):
            key = commodities[k][1]
            return key[2] if isinstance(key, tuple) else key

        # who may own the card in a slot: wildcards stand for the opponent's cards
        owners: dict[int, frozenset] = {}
        for v in t.all_vars:
            if v.kind in ("main", "extra", "created"):
                owners[v.id] = frozenset((v.owner,))
        for pool in t.pools:
            ow = frozenset().union(*(owners.get(m.id, frozenset()) for m in pool.members))
            for c in pool.claims:
                owners[c.id] = ow

        def tag_owner(k: int) -> int | None:
            # (the owner of the physical card a tag stands for, if it is known)
            ref = refs[commodities[k][1][1]]
            ow = owners.get(ref.id, frozenset((ref.owner,)))
            if ref.seen_owner is not None:
                return ref.seen_owner
            return next(iter(ow)) if len(ow) == 1 else None

        def domain(v: Var) -> list[int]:
            if v.id in ref_tags:
                return list(ref_tags[v.id])
            dom = plain_domain(v)
            ows = slot_owners(v)
            extra = list(dict.fromkeys(q for k in dom if not isinstance(commodities[k][1], (str, tuple))
                                       for q in tags_by_code.get(commodities[k][1], ())
                                       if tag_owner(q) is None or tag_owner(q) in ows))
            return dom + extra

        def slot_owners(v: Var) -> list[int]:
            ow = owners.get(v.id, frozenset((v.owner,)))
            if v.seen_owner is not None:
                # (shown by a state view)
                ow = (ow & {v.seen_owner}) or frozenset((v.seen_owner,))
            return sorted(ow)

        def plain_domain(v: Var) -> list[int]:
            ows = slot_owners(v)
            code = v.fixed_code if v.fixed_code is not None else v.code
            if code:
                return [com(p, code) for p in ows]
            opp_ok = opp in ows
            mains = [k for p in ows for k in main_codes[p]]
            extras = [k for p in ows for k in extra_codes[p]]
            tokens = [k for p in ows for k in token_codes[p]]
            if v.kind == "main":
                dom = mains + main_wilds
            elif v.kind == "extra":
                dom = extras + extra_wilds
            elif v.pool is not None and v.pool.location in (C.LOCATION_DECK, C.LOCATION_HAND):
                dom = mains + (main_wilds if opp_ok else [])
            elif v.pool is not None and v.pool.location == C.LOCATION_EXTRA:
                dom = extras + (extra_wilds if opp_ok else [])
            else:
                dom = mains + extras + tokens + (main_wilds + extra_wilds if opp_ok else [])
            if v.need is not None:
                ok_wilds = set(_need_wilds(v.need))
                dom = [k for k in dom if isinstance(commodities[k][1], str) and commodities[k][1] in ok_wilds] + \
                      [k for k in dom if not isinstance(commodities[k][1], str)
                       and self._fits(commodities[k][1], v.need, usual=False)]
            if v.excluded:
                dom = [k for k in dom if commodities[k][1] not in v.excluded]
            return dom

        # edges: (producer, consumer) groups of slots; constrained slots alone
        def producer(v: Var):
            if v.kind in ("main", "extra"):
                return ("atoms", v.owner, v.kind)
            if v.kind == "created":
                return ("created", v.owner)
            return ("pool", v.pool.index)

        # slots in identity constraints need their own variables
        linked: set[int] = set(tag_of)
        for v in t.all_vars:
            if v.allowed is not None or v.same_as is not None or v.not_same:
                linked.add(v.id)
            if v.allowed is not None and len(v.allowed) <= _MAX_UNTAGGED_ALLOWED:
                linked.update(a for a in v.allowed if a in byid)
        edges: dict[tuple, list[Var]] = defaultdict(list)
        for v in t.all_vars:
            if v.kind == "claim" and v.pool is None:
                continue
            cons = ("pool", consumer[v.id]) if v.id in consumer else ("sink",)
            constrained = (v.code is not None or v.need is not None or v.excluded
                           or v.fixed_code is not None or v.kind == "created" or v.id in linked)
            # (slots grouped only with slots of the same seen owner: their domains differ)
            key = (producer(v), cons, v.id if constrained else None, v.owner,
                   None if constrained else (v.kind, v.pool.location if v.pool else None, v.seen_owner))
            edges[key].append(v)
        edge_keys = list(edges)

        # variables x[e, k]
        fixed_counts: dict[int, Counter] = {}  # viewer's atom edges: cards per code
        cols: list[tuple[int, int]] = []
        lo: list[float] = []
        hi: list[float] = []
        cost: list[float] = []
        for ei, key in enumerate(edge_keys):
            vs = edges[key]
            dom = domain(vs[0])
            if not dom:
                raise TrackerError(f"no possible card for {vs[0]}")
            n = len(vs)
            if key[0][0] == "atoms" and key[0][1] == viewer:
                codes = Counter(v.fixed_code for v in vs)
                fixed_counts[ei] = codes
                dom = [k for k in dom if code_of(k) in codes]
            v0 = vs[0]
            holder = v0.pool.player if v0.kind == "claim" and v0.pool is not None else v0.owner
            for k in dom:
                cols.append((ei, k))
                lo.append(0)
                hi.append(n)
                opp_atom = key[0][0] == "atoms" and key[0][1] != viewer
                what = commodities[k][1]
                # prefer unknown cards to extra copies of known ones, and plain
                # unknown cards to ones of a specific kind
                cost.append((0.0 if not opp_atom else 1.0 if not isinstance(what, str)
                             else 0.0 if what in (WILD_MAIN, WILD_EXTRA) else 0.01)
                            # (a known card whose Level needs fewer Tributes than usual,
                            # e.g. through an effect: only if nothing else fits)
                            + (2.0 if vs[0].need is not None and not isinstance(what, str)
                               and not self._fits(what, vs[0].need) else 0.0)
                            # (a card a player holds is their own unless something shows
                            # otherwise: the other player's copy only if needed)
                            + (0.05 if commodities[k][0] is not None
                               and commodities[k][0] != holder else 0.0))
        colmap: dict[tuple[int, int], int] = {c: j for j, c in enumerate(cols)}
        ncol = len(cols)
        rows, colsi, vals, rlo, rhi = [], [], [], [], []
        r = 0
        by_edge: dict[int, list[int]] = defaultdict(list)
        for j, (ei, k) in enumerate(cols):
            by_edge[ei].append(j)
        # capacity of every edge
        for ei, key in enumerate(edge_keys):
            for j in by_edge[ei]:
                rows.append(r); colsi.append(j); vals.append(1.0)
            n = len(edges[key])
            rlo.append(n); rhi.append(n)
            r += 1
        # conservation per pool and commodity
        flow: dict[tuple[int, int], list[tuple[int, float]]] = defaultdict(list)
        for j, (ei, k) in enumerate(cols):
            key = edge_keys[ei]
            if key[1][0] == "pool":
                flow[(key[1][1], k)].append((j, 1.0))
            if key[0][0] == "pool":
                flow[(key[0][1], k)].append((j, -1.0))
        for (_, k), terms in flow.items():
            for j, s in terms:
                rows.append(r); colsi.append(j); vals.append(s)
            rlo.append(0); rhi.append(0)
            r += 1
        def add_row(terms, low, high):
            nonlocal r
            for j, v in terms:
                rows.append(r); colsi.append(j); vals.append(v)
            rlo.append(low); rhi.append(high)
            r += 1

        # the viewer's own deck list: exactly its cards (tagged or not)
        for ei, codes in fixed_counts.items():
            per: dict[int, list[int]] = defaultdict(list)
            for j in by_edge[ei]:
                per[code_of(cols[j][1])].append(j)
            for code, n in codes.items():
                add_row([(j, 1.0) for j in per.get(code, [])], n, n)
        # the deck of the player a hint named a card for contains it
        for p, code in contains:
            add_row([(j, 1.0) for j, (ei, k) in enumerate(cols)
                     if edge_keys[ei][0][:2] == ("atoms", p) and code_of(k) == code], 1, 3)
        # at most 3 copies of a card per opponent deck (by base code, tags included)
        limit: dict[int, list[int]] = defaultdict(list)
        for j, (ei, k) in enumerate(cols):
            key = edge_keys[ei]
            code = code_of(k)
            if key[0][0] == "atoms" and key[0][1] != viewer and not isinstance(code, str):
                limit[self._base(code)].append(j)
        for js in limit.values():
            add_row([(j, 1.0) for j in js], 0, 3)
        # a tag exists (is supplied by one card of a deck list) iff its ref carries it
        edge_of: dict[int, int] = {}
        for ei, key in enumerate(edge_keys):
            if key[2] is not None:
                edge_of[key[2]] = ei
        for rid, k in tag_of.items():
            supply = [(j, 1.0) for j, (ei, kk) in enumerate(cols)
                      if kk == k and edge_keys[ei][0][0] == "atoms"]
            jr = colmap.get((edge_of[rid], k))
            if jr is None:
                continue
            add_row(supply + [(jr, -1.0)], 0, 0)

        def var_cols(v: Var) -> dict[int, int]:
            ei = edge_of.get(v.id)
            return {} if ei is None else {cols[j][1]: j for j in by_edge[ei]}
        for v in t.all_vars:
            if v.id not in linked:
                continue
            mine = var_cols(v)
            if v.same_as is not None and v.same_as.id in ref_tags:
                # the same card: the same tag
                other = var_cols(v.same_as)
                ts = ref_tags[v.same_as.id]
                for k in ts:
                    terms = [(other[k], -1.0)] if k in other else []
                    if k in mine:
                        terms.append((mine[k], 1.0))
                    add_row(terms, 0, 0)
            for x in v.not_same:
                if x.id not in ref_tags:
                    continue
                other = var_cols(x)
                for k in ref_tags[x.id]:
                    if k in mine and k in other:
                        add_row([(mine[k], 1.0), (other[k], 1.0)], -np.inf, 1)
            if v.allowed is not None and not all(a in ref_tags for a in v.allowed) \
                    and len(v.allowed) <= _MAX_UNTAGGED_ALLOWED:
                # (members without a known code) the slot's card is one of them:
                # whatever it carries, one of them carries too
                ms = [byid[a] for a in v.allowed if a in byid]
                for k, j in mine.items():
                    terms = [(var_cols(m)[k], -1.0) for m in ms if k in var_cols(m)]
                    add_row([(j, 1.0)] + terms, -np.inf, 0)
            if v.allowed is not None:
                ms = [byid[a] for a in v.allowed if a in ref_tags]
                if ms and len(ms) == len(v.allowed):
                    ts = sorted({k for m in ms for k in ref_tags[m.id]})
                    for k in ts:
                        terms = [(var_cols(m)[k], -1.0) for m in ms if k in var_cols(m)]
                        if k in mine:
                            terms.append((mine[k], 1.0))
                        add_row(terms, -np.inf, 0)
                    add_row([(mine[k], 1.0) for k in ts if k in mine], 1, 1)
        a = coo_matrix((vals, (rows, colsi)), shape=(r, ncol)).tocsr()
        res = milp(c=np.array(cost), constraints=LinearConstraint(a, np.array(rlo), np.array(rhi)),
                   integrality=np.ones(ncol), bounds=Bounds(np.array(lo), np.array(hi)),
                   # (a time limit makes the result depend on the machine's load: only one
                   # far beyond any check's own, for a pathological model)
                   options={"time_limit": 600.0})
        if res.status != 0 or res.x is None:
            raise TrackerError(f"observations are inconsistent (flow model: {res.message})")
        x = np.rint(res.x).astype(int)

        # commodity of every slot
        vcom: dict[int, int] = {}
        for ei, key in enumerate(edge_keys):
            vs = edges[key]
            it = iter(vs)
            for j in by_edge[ei]:
                for _ in range(x[j]):
                    vcom[next(it).id] = cols[j][1]
        return self._realize(vcom, commodities)

    @staticmethod
    def _carried(pool: Pool, x: Var, depth: int = 0) -> bool:
        """Whether the card of slot `x` is certainly one of the pool's members
        (directly, or through earlier pools all of whose cards went into it)."""
        if any(m is x for m in pool.members):
            return True
        if depth > 8:
            return False
        ids = {m.id for m in pool.members}
        for m in pool.members:
            q = m.pool if m.kind == "claim" else None
            if q is not None and all(c.id in ids for c in q.claims) and FlowSolver._carried(q, x, depth + 1):
                return True
        return False

    @staticmethod
    def _descends(v: Var, ancestor: Var) -> bool:
        while v is not None:
            if v is ancestor:
                return True
            v = v.assigned if v.kind == "claim" else None
        return False

    # -- from commodities to a hypothesis ----------------------------------------------
    def _realize(self, vcom: dict[int, int], commodities: list[tuple]) -> Hypothesis:
        t = self.t
        # 1. every pool: claims get members of the same commodity (hint identity first)
        byid = {v.id: v for v in t.all_vars}

        def allows(c: Var, m: Var) -> bool:
            # (allowed may name earlier cards: then the member that is one of them)
            return c.allowed is None or m.id in c.allowed or any(
                a in byid and self._descends(m, byid[a]) for a in c.allowed)
        for pool in t.pools:
            free: dict[int, list[Var]] = defaultdict(list)
            for m in pool.members:
                free[vcom[m.id]].append(m)
            claims = sorted(pool.claims, key=lambda c: (c.same_as is None, c.allowed is None, not c.not_same,
                                                        not c.prefer))
            for c in claims:
                available = free[vcom[c.id]]
                cands = available
                if c.prefer:
                    # prefer members that are one of these physical cards
                    ok = [m for m in available if any(self._descends(m, x) for x in c.prefer)]
                    cands = ok + [m for m in available if m not in ok]
                if c.not_same:
                    # prefer members that are none of these physical cards
                    ok = [m for m in available if not any(self._descends(m, x) for x in c.not_same)]
                    cands = ok + [m for m in available if m not in ok]
                if c.same_as is not None:
                    # the member whose own history contains that slot
                    pick = next((m for m in cands if self._descends(m, c.same_as)), None)
                    if pick is None:
                        pick = next((m for m in cands if allows(c, m)), None)
                else:
                    pick = next((m for m in cands if allows(c, m)), None)
                if pick is None:
                    raise TrackerError(f"no member for {c} in pool {pool.index}")
                available.remove(pick)
                c.assigned = pick

        # 2. codes of atoms: known ones directly, wildcards from the completion
        from .completion import DeckCompletion
        comp = self.completion if self.completion is not None else DeckCompletion()
        unknown = 0
        # codes excluded for a slot are excluded for the card it realizes to
        excluded: dict[int, set] = defaultdict(set)
        for v in t.exclusions:
            a = v
            while a is not None and a.kind == "claim":
                a = a.assigned
            if a is not None:
                excluded[a.id] |= v.excluded
        for p in (0, 1):
            for a in t.atoms[p]:
                key = commodities[vcom[a.id]][1]
                if isinstance(key, tuple):
                    key = key[2]  # a tagged card
                a.resolved = a.fixed_code if a.fixed_code is not None else (None if isinstance(key, str) else key)
        present = [a.resolved for p in (0, 1) for a in t.atoms[p] if a.resolved is not None]
        for p in (0, 1):
            atoms = t.atoms[p]
            wild = [a for a in atoms if a.resolved is None]
            if not wild:
                continue
            unknown += len(wild)
            counts = Counter(self._base(a.resolved) for a in atoms if a.resolved is not None)
            typed = [a for a in wild if commodities[vcom[a.id]][1] not in (WILD_MAIN, WILD_EXTRA)]
            for a in typed:
                need = _wild_need(commodities[vcom[a.id]][1])
                cands = [c for c in comp.stand_ins(need, list(counts.elements())) if counts[self._base(c)] < 3
                         and c not in excluded.get(a.id, ())]
                if self.db is not None and need[0] == "mset" \
                        and excluded.get(a.id, set()) & set(comp.stand_ins(need)):
                    # (once a stand-in was ruled out, e.g. it could be attacked where the
                    # real card could not: the races of the player's known monsters first)
                    races = Counter(self.db.get(r).race for r in counts if self.db.get(r)
                                    and self.db.get(r).type & C.TYPE_MONSTER)
                    cands.sort(key=lambda c: -races[self.db.get(c).race] if self.db.get(c) else 0)
                if not cands:
                    raise TrackerError(f"no stand-in for a hidden card of kind {need}")
                # (different stand-ins first: checks can count different names, e.g.
                # Tierra, Source of Destruction needs 10 differently named cards)
                cands.sort(key=lambda c: counts[self._base(c)] > 0)
                a.resolved = cands[0]
                counts[self._base(cands[0])] += 1
            plain = [a for a in wild if a.resolved is None]
            if plain:
                known_main = [a.resolved for a in atoms if a.kind == "main" and a.resolved is not None]
                known_extra = [a.resolved for a in atoms if a.kind == "extra" and a.resolved is not None]
                n_main = sum(1 for a in plain if a.kind == "main")
                n_extra = sum(1 for a in plain if a.kind == "extra")
                if self.completion is None:
                    raise TrackerError("unknown opponent cards but no deck completion available")
                # (plus a spare code per slot with ruled out codes: e.g. a second copy
                # of a searched card that no slot may hold; the completion's order is
                # kept, so its first codes are the same)
                spare = {kind: sum(1 for a in plain if a.kind == kind and excluded.get(a.id))
                         for kind in ("main", "extra")}
                try:
                    fill_main, fill_extra = comp.complete(known_main, known_extra, n_main + spare["main"],
                                                          n_extra + spare["extra"], present)
                except RuntimeError:
                    fill_main, fill_extra = comp.complete(known_main, known_extra, n_main, n_extra, present)
                fills = {"main": list(fill_main), "extra": list(fill_extra)}
                # (in order, except where the next code was ruled out for this slot,
                # e.g. by a scan of the deck: a realistic completion can guess it; the
                # slots with ruled out codes first, so that a code the completion adds
                # again goes where it may be)
                for a in sorted(plain, key=lambda a: not excluded.get(a.id)):
                    lst, ex = fills[a.kind], excluded.get(a.id, ())
                    k = 0 if lst[0] not in ex else next((i for i, c in enumerate(lst) if c not in ex), 0)
                    a.resolved = lst.pop(k)
        for v in t.all_vars:
            if v.kind == "created" and v.resolved is None:
                key = commodities[vcom[v.id]][1] if v.id in vcom else None
                if isinstance(key, tuple):
                    key = key[2]
                v.resolved = v.code or (key if not isinstance(key, str) else None)

        # 3. the rest as in the search based solver (orders, shuffles, copy ranks)
        return self._legacy.finish(unknown)
