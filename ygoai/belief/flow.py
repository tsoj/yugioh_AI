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
wildcards filled by the deck completion, and copy ranks from the creation order
constraints (see Solver._relabel).
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
        self._legacy = Solver(tracker, completion, card_db)  # helpers (_relabel, _known)

    # -- helpers ------------------------------------------------------------------
    def _base(self, code: int) -> int:
        return self._legacy._base(code)

    def _is_extra(self, code: int) -> bool:
        card = self.db.get(code) if self.db is not None else None
        return bool(card is not None and card.is_extra_deck)

    def _fits(self, code: int, need: tuple) -> bool:
        from .completion import fits_need
        return self.db is not None and fits_need(self.db.get(code), need)

    # -- model ----------------------------------------------------------------------
    def solve(self) -> Hypothesis:
        t = self.t
        viewer = t.viewer
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
        commodities: list[tuple] = []
        cindex: dict[tuple, int] = {}

        def com(owner, key) -> int:
            k = (owner, key)
            if k not in cindex:
                cindex[k] = len(commodities)
                commodities.append(k)
            return cindex[k]

        wilds = [WILD_MAIN, WILD_EXTRA, "st", "field", "monster", "mset0", "mset1", "mset2"]
        main_codes, extra_codes, token_codes = [], [], []
        for code in sorted(known):
            card = self.db.get(code) if self.db is not None else None
            if card is not None and card.type & C.TYPE_TOKEN:
                token_codes.append(com(None, code))
            else:
                (extra_codes if self._is_extra(code) else main_codes).append(com(None, code))
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
        tag_of: dict[int, int] = {}  # ref var id -> its tag commodity
        tags_by_code: dict[int, list[int]] = defaultdict(list)
        for rid in sorted(refs):
            code = known_code(refs[rid])
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

        def domain(v: Var) -> list[int]:
            if v.id in ref_tags:
                return list(ref_tags[v.id])
            dom = plain_domain(v)
            extra = [q for k in dom if not isinstance(commodities[k][1], (str, tuple))
                     for q in tags_by_code.get(commodities[k][1], ())]
            return dom + extra

        def plain_domain(v: Var) -> list[int]:
            code = v.fixed_code if v.fixed_code is not None else v.code
            if code:
                return [com(None, code)]
            opp_ok = opp in owners.get(v.id, frozenset((v.owner,)))
            if v.kind == "main":
                dom = main_codes + main_wilds
            elif v.kind == "extra":
                dom = extra_codes + extra_wilds
            elif v.pool is not None and v.pool.location in (C.LOCATION_DECK, C.LOCATION_HAND):
                dom = main_codes + (main_wilds if opp_ok else [])
            elif v.pool is not None and v.pool.location == C.LOCATION_EXTRA:
                dom = extra_codes + (extra_wilds if opp_ok else [])
            else:
                dom = main_codes + extra_codes + token_codes + (main_wilds + extra_wilds if opp_ok else [])
            if v.need is not None:
                ok_wilds = set(_need_wilds(v.need))
                dom = [k for k in dom if isinstance(commodities[k][1], str) and commodities[k][1] in ok_wilds] + \
                      [k for k in dom if not isinstance(commodities[k][1], str)
                       and self._fits(commodities[k][1], v.need)]
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
        # creation order constraints between cards of one player: cards are
        # created sorted by code (game.setup_duel), so a card created earlier
        # has a code that sorts no later (the order among copies is left to
        # Solver._relabel)
        ordered: list[tuple[Var, Var]] = []
        for va, vb in t.rank_constraints:
            if va.kind == "created" or vb.kind == "created" or va is vb:
                continue
            if known_code(va) is not None and known_code(vb) is not None:
                continue  # nothing to decide
            oa, ob = owners.get(va.id), owners.get(vb.id)
            if oa is None or len(oa) != 1 or oa != ob:
                continue
            if (va.kind == "claim" and va.pool is None) or (vb.kind == "claim" and vb.pool is None):
                continue
            ordered.append((va, vb))
            linked.update((va.id, vb.id))
        edges: dict[tuple, list[Var]] = defaultdict(list)
        for v in t.all_vars:
            if v.kind == "claim" and v.pool is None:
                continue
            cons = ("pool", consumer[v.id]) if v.id in consumer else ("sink",)
            constrained = (v.code is not None or v.need is not None or v.excluded
                           or v.fixed_code is not None or v.kind == "created" or v.id in linked)
            key = (producer(v), cons, v.id if constrained else None, v.owner,
                   None if constrained else (v.kind, v.pool.location if v.pool else None))
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
            for k in dom:
                cols.append((ei, k))
                lo.append(0)
                hi.append(n)
                opp_atom = key[0][0] == "atoms" and key[0][1] != viewer
                what = commodities[k][1]
                # prefer unknown cards to extra copies of known ones, and plain
                # unknown cards to ones of a specific kind
                cost.append(0.0 if not opp_atom else 1.0 if not isinstance(what, str)
                            else 0.0 if what in (WILD_MAIN, WILD_EXTRA) else 0.01)
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
        def sort_key(k: int):
            code = code_of(k)
            return None if isinstance(code, str) else (self._is_extra(code), code)
        for va, vb in ordered:
            ca, cb = var_cols(va), var_cols(vb)
            kb = sorted(((sort_key(k), j) for k, j in cb.items() if sort_key(k) is not None))
            for k, j in ca.items():
                key = sort_key(k)
                if key is None:
                    continue
                below = [(jj, 1.0) for kk, jj in kb if kk < key]
                if below:
                    add_row([(j, 1.0)] + below, -np.inf, 1)
        a = coo_matrix((vals, (rows, colsi)), shape=(r, ncol)).tocsr()
        res = milp(c=np.array(cost), constraints=LinearConstraint(a, np.array(rlo), np.array(rhi)),
                   integrality=np.ones(ncol), bounds=Bounds(np.array(lo), np.array(hi)),
                   options={"time_limit": 20.0})
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

    def _rank_atoms(self) -> list[tuple[Var, Var]]:
        """Hard creation order constraints as pairs of atoms (after claim matching)."""
        def atom(v: Var) -> Var | None:
            while v is not None and v.kind not in ("main", "extra", "created"):
                v = v.assigned
            return v if v is not None and v.kind != "created" else None
        out = []
        for va, vb in self.t.rank_constraints:
            a, b = atom(va), atom(vb)
            if a is not None and b is not None and a is not b and a.owner == b.owner and a.kind == b.kind:
                out.append((a, b))
        return out

    def _order_bounds(self, typed: list[Var]) -> tuple[dict, dict]:
        """atom id -> atoms that must sort no later / no earlier."""
        lower: dict[int, list[Var]] = defaultdict(list)
        upper: dict[int, list[Var]] = defaultdict(list)
        for a, b in self._rank_atoms():
            lower[b.id].append(a)
            upper[a.id].append(b)
        return lower, upper

    def _order_first(self, typed: list[Var]) -> list[Var]:
        """`typed` with every card after the ones that must sort before it."""
        ids = {a.id for a in typed}
        before: dict[int, set[int]] = defaultdict(set)
        for a, b in self._rank_atoms():
            if a.id in ids and b.id in ids:
                before[b.id].add(a.id)
        out, done = [], set()
        pending = list(typed)
        while pending:
            rest = []
            for a in pending:
                if before[a.id] <= done:
                    out.append(a)
                    done.add(a.id)
                else:
                    rest.append(a)
            if len(rest) == len(pending):  # (a cycle: keep the order)
                out += rest
                break
            pending = rest
        return out

    # -- from commodities to a hypothesis ----------------------------------------------
    def _realize(self, vcom: dict[int, int], commodities: list[tuple]) -> Hypothesis:
        t = self.t
        # 1. every pool: claims get members of the same commodity (hint identity first)
        for pool in t.pools:
            free: dict[int, list[Var]] = defaultdict(list)
            for m in pool.members:
                free[vcom[m.id]].append(m)
            claims = sorted(pool.claims, key=lambda c: (c.same_as is None, c.allowed is None, not c.not_same))
            for c in claims:
                available = free[vcom[c.id]]
                cands = available
                if c.not_same:
                    # prefer members that are none of these physical cards
                    ok = [m for m in available if not any(self._descends(m, x) for x in c.not_same)]
                    cands = ok + [m for m in available if m not in ok]
                if c.same_as is not None:
                    # the member whose own history contains that slot
                    pick = next((m for m in cands if self._descends(m, c.same_as)), None)
                    if pick is None:
                        pick = next((m for m in cands if c.allowed is None or m.id in c.allowed), None)
                else:
                    pick = next((m for m in cands if c.allowed is None or m.id in c.allowed), None)
                if pick is None:
                    raise TrackerError(f"no member for {c} in pool {pool.index}")
                available.remove(pick)
                c.assigned = pick

        # 2. codes of atoms: known ones directly, wildcards from the completion
        from .completion import DeckCompletion
        comp = self.completion if self.completion is not None else DeckCompletion()
        unknown = 0
        for p in (0, 1):
            atoms = t.atoms[p]
            for a in atoms:
                key = commodities[vcom[a.id]][1]
                if isinstance(key, tuple):
                    key = key[2]  # a tagged card
                a.resolved = a.fixed_code if a.fixed_code is not None else (None if isinstance(key, str) else key)
            wild = [a for a in atoms if a.resolved is None]
            if not wild:
                continue
            unknown += len(wild)
            counts = Counter(self._base(a.resolved) for a in atoms if a.resolved is not None)
            typed = [a for a in wild if commodities[vcom[a.id]][1] not in (WILD_MAIN, WILD_EXTRA)]
            lower, upper = self._order_bounds(typed)
            for a in self._order_first(typed):
                need = _wild_need(commodities[vcom[a.id]][1])
                cands = [c for c in comp.stand_ins(need) if counts[self._base(c)] < 3]
                if not cands:
                    raise TrackerError(f"no stand-in for a hidden card of kind {need}")
                # respect the creation order constraints where possible (cards are
                # created sorted by code): the smallest code above the earlier ones
                lo = max((b.resolved for b in lower.get(a.id, ()) if b.resolved is not None), default=0)
                hi = min((b.resolved for b in upper.get(a.id, ()) if b.resolved is not None), default=1 << 40)
                fitting = [c for c in cands if lo <= c <= hi]
                code = min(fitting) if fitting else cands[0]
                a.resolved = code
                counts[self._base(code)] += 1
            plain = [a for a in wild if a.resolved is None]
            if plain:
                known_main = [a.resolved for a in atoms if a.kind == "main" and a.resolved is not None]
                known_extra = [a.resolved for a in atoms if a.kind == "extra" and a.resolved is not None]
                n_main = sum(1 for a in plain if a.kind == "main")
                n_extra = sum(1 for a in plain if a.kind == "extra")
                if self.completion is None:
                    raise TrackerError("unknown opponent cards but no deck completion available")
                fill_main, fill_extra = comp.complete(known_main, known_extra, n_main, n_extra)
                it_main, it_extra = iter(fill_main), iter(fill_extra)
                for a in plain:
                    a.resolved = next(it_main) if a.kind == "main" else next(it_extra)
                # the filler codes are interchangeable: hand them out along the
                # creation order constraints (cards are created sorted by code)
                ranked = {x.id for pair in self._rank_atoms() for x in pair}
                for kind in ("main", "extra"):
                    group = [a for a in plain if a.kind == kind and a.id in ranked]
                    codes = sorted(a.resolved for a in group)
                    for a, code in zip(self._order_first(group), codes):
                        a.resolved = code
        for v in t.all_vars:
            if v.kind == "created" and v.resolved is None:
                key = commodities[vcom[v.id]][1] if v.id in vcom else None
                if isinstance(key, tuple):
                    key = key[2]
                v.resolved = v.code or (key if not isinstance(key, str) else None)

        # 3. the rest as in the search based solver (orders, shuffles, copy ranks)
        return self._legacy.finish(unknown)
