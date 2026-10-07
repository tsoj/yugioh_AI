"""An agent's belief state, reconstructed decision by decision.

At each decision the agent needs the duel as it believes it to be, from all its
observations so far (BeliefReplay). The stream of the previous decision is the
start of the new one, so:

1. the previous consistent reconstruction's duel, kept waiting at that decision,
   goes on with the agent's answer and the observations since (BeliefReplay.extend,
   incremental: only the new part is replayed);
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

from ..actions import InvalidResponse
from .replay import BeliefReplay, Divergence
from .tracker import TrackerError

_INCREMENTAL = not os.environ.get("YGOAI_NO_INCREMENTAL")


class AgentBelief:
    def __init__(self, viewer: int, init: dict, completion=None, incremental: bool = _INCREMENTAL):
        self.viewer, self.init, self.completion = viewer, init, completion
        self.incremental = incremental
        self._prev: BeliefReplay | None = None  # (the latest consistent reconstruction)
        self._live = None  # (its duel, at the decision its stream ends with)
        self._last = None  # (the duel handed out last, closed at the next call)
        self.extended = 0  # (decisions reached incrementally)
        self.retries = 0  # (fresh attempts after an inherited one failed or was approximate)

    def reconstruct(self, items: list[tuple]):
        """The duel at the end of `items` (or None if there is none, not even an
        approximate one) and its BeliefReplay (see its `approximate`)."""
        live, self._live = self._live, None
        if self._last is not None and self._last is not live:
            self._last.close()
        self._last = None
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
