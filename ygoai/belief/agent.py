"""An agent's belief state, reconstructed decision by decision.

At each decision the agent needs the duel as it believes it to be, from all its
observations so far (BeliefReplay). The stream of the previous decision is the
start of the new one, so:

1. the previous consistent reconstruction's duel, kept waiting at that decision,
   goes on with the agent's answer and the observations since (BeliefReplay.extend,
   incremental: only the new part is replayed), unless they reveal opponent cards
   not seen before (or more copies of one): the guesses for the opponent's hidden
   cards (the deck completion) take what was seen into account, so they are made
   anew then;
2. if that runs into a mismatch (which needs repairs, i.e. a run from the start),
   a new reconstruction inherits what the previous one found (its repairs and
   decisions, see BeliefReplay.inherit) and mostly works on the new part;
3. if that fails or ends only approximate, one from scratch (an inherited repair
   may not fit the longer stream).

With no consistent reconstruction, the best effort (approximate, re-synchronized
with the agent's observations, see sync.py) is what the agent gets.

The duels belong to this object: the caller must not close them or change them
(search in forks of the process, as belief_mc does).
"""

from __future__ import annotations

import os
from collections import Counter

from ..actions import InvalidResponse
from .replay import BeliefReplay, Divergence
from .tracker import TrackerError

_INCREMENTAL = not os.environ.get("YGOAI_NO_INCREMENTAL")


class _RustAgentBelief:
    """AgentBelief on the Rust belief replay (the default; YGOAI_REPLAY=python for this module's):
    the duels it hands out are copies, the caller's to keep."""

    def __init__(self, viewer: int, init: dict, completion=None):
        import ygoai_rs

        from ..core import DuelConfig
        from .check import rust_completion
        self.viewer, self.init = viewer, init
        self._b = ygoai_rs.AgentBelief(viewer, init, rust_completion(completion))
        self._config = DuelConfig(**init["config"]) if init.get("config") else None

    def reconstruct(self, items: list[tuple], timeout: float | None = None):
        """(`timeout`: seconds, after which it gives up: no duel; signals cannot stop Rust code)"""
        from types import SimpleNamespace

        from ..core_rust import Duel
        info, d = self._b.reconstruct(items, timeout)
        return SimpleNamespace(**info), (Duel.from_rust(d, self._config) if d is not None else None)

    @property
    def retries(self) -> int:
        return self._b.retries

    @property
    def extended(self) -> int:
        return self._b.extended

    @property
    def revealed(self) -> int:
        return self._b.revealed

    def close(self) -> None:
        pass


class AgentBelief:
    def __new__(cls, viewer: int, init: dict, completion=None, incremental: bool = _INCREMENTAL):
        from .check import use_rust
        if cls is AgentBelief and use_rust() and incremental == _INCREMENTAL:
            return _RustAgentBelief(viewer, init, completion)
        return super().__new__(cls)

    def __init__(self, viewer: int, init: dict, completion=None, incremental: bool = _INCREMENTAL):
        self.viewer, self.init, self.completion = viewer, init, completion
        self.incremental = incremental
        self._prev: BeliefReplay | None = None  # (the latest consistent reconstruction)
        self._live = None  # (its duel, at the decision its stream ends with)
        self._last = None  # (the duel handed out last, closed at the next call)
        self.extended = 0  # (decisions reached incrementally)
        self.revealed = 0  # (decisions with new reveals: reconstructed in full)
        self._seen = Counter()  # opponent card codes the stream showed (see _reveals)
        self._seen_upto = 0
        self.retries = 0  # (fresh attempts after an inherited one failed or was approximate)

    def reconstruct(self, items: list[tuple], timeout: float | None = None):
        """The duel at the end of `items` (or None if there is none, not even an
        approximate one) and its BeliefReplay (see its `approximate`). (`timeout`: for the
        Rust replay; this one is stopped from outside.)"""
        live, self._live = self._live, None
        if self._last is not None and self._last is not live:
            self._last.close()
        self._last = None
        if self._reveals(items) and live is not None:
            self.revealed += 1
            live.close()
            live = None
        if live is not None:
            try:
                duel = self._prev.extend(live, items)
                self.extended += 1
                self._live = self._last = duel
                return self._prev, duel
            except (Divergence, TrackerError, InvalidResponse, KeyError, ValueError, IndexError, TypeError):
                live.close()
        br, duel, inherited = self._attempt(items, self._prev)
        if inherited and (duel is None or br.approximate):
            if duel is not None:
                duel.close()
            self.retries += 1
            br, duel, _ = self._attempt(items, None)
        consistent = duel is not None and not br.approximate
        self._prev = br if consistent else None
        self._live = duel if consistent and self.incremental else None
        self._last = duel
        return br, duel

    def _reveals(self, items) -> bool:
        """Whether the stream since the last call shows opponent cards it did not
        show before: per code, as one card when a message shows it (moved, summoned,
        flipped, activated, confirmed) and as the most copies a state view showed
        at once."""
        opp = 1 - self.viewer
        new = Counter()
        for it in items[self._seen_upto:]:
            if it[0] == "prompt" and len(it) > 2 and it[2] is not None:
                here = Counter(c["code"] for p in it[2]["players"] for pile in p.values() for c in pile
                               if c and c.get("code") and c.get("owner") == opp)
                new |= here
            elif it[0] == "msg":
                m = it[1]
                if m["type"] == "MOVE" and m.get("code"):
                    con = m["to"]["con"] if m["to"]["loc"] else m["from"]["con"]
                    if con == opp:
                        new[m["code"]] = max(new[m["code"]], 1)
                elif m["type"] in ("CHAINING", "POS_CHANGE", "SET", "SUMMONING", "SPSUMMONING") \
                        and m.get("code") and m.get("con") == opp:
                    new[m["code"]] = max(new[m["code"]], 1)
                elif m["type"] == "CONFIRM_CARDS":
                    for c in m["cards"]:
                        if c.get("code") and c.get("con") == opp:
                            new[c["code"]] = max(new[c["code"]], 1)
        self._seen_upto = len(items)
        more = any(n > self._seen[code] for code, n in new.items())
        self._seen |= new
        return more

    def close(self) -> None:
        for d in {id(d): d for d in (self._live, self._last) if d is not None}.values():
            d.close()
        self._live = self._last = None

    def _attempt(self, items, prev):
        br = BeliefReplay(self.viewer, self.init, list(items), completion=self.completion, check_state=False)
        br.allow_approximate = True  # (an agent needs some state: the closest one, from observations only)
        br.read_guided = False  # (too slow per decision: the approximate pass comes right after the repairs)
        inherited = prev is not None and br.inherit(prev)
        try:
            return br, br.run(), inherited
        except (Divergence, TrackerError):
            return br, None, inherited
