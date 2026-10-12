"""An agent's belief state, reconstructed decision by decision (the belief replay in Rust:
crates/ygoai/src/belief/agent.rs, through the ygoai_rs binding).

At each decision the agent needs the duel as it believes it to be, from all its observations so far.
The stream of the previous decision is the start of the new one, so:

1. the previous consistent reconstruction's duel, kept waiting at that decision, goes on with the
   agent's answer and the observations since (incremental: only the new part is replayed), unless
   they reveal opponent cards not seen before (or more copies of one): the guesses for the
   opponent's hidden cards (the deck completion) take what was seen into account, so they are made
   anew then;
2. if that runs into a mismatch (which needs repairs, i.e. a run from the start), a new
   reconstruction inherits what the previous one found (its repairs and decisions) and mostly works
   on the new part;
3. if that fails or ends only approximate, one from scratch (an inherited repair may not fit the
   longer stream).

With no consistent reconstruction, the best effort (approximate, re-synchronized with the agent's
observations) is what the agent gets. The duels it hands out are copies, the caller's to keep.
YGOAI_NO_INCREMENTAL=1 turns step 1 off.
"""

from __future__ import annotations

from types import SimpleNamespace


class AgentBelief:
    def __init__(self, viewer: int, init: dict, completion=None):
        """`completion`: a deck completion object (vanilla and corpus run in Rust, any other, e.g.
        the neural one, is called from Rust)."""
        import ygoai_rs

        from ..core import DuelConfig
        from .check import rust_completion
        self.viewer, self.init = viewer, init
        self._b = ygoai_rs.AgentBelief(viewer, init, rust_completion(completion))
        self._config = DuelConfig(**init["config"]) if init.get("config") else None

    def reconstruct(self, items: list[tuple], timeout: float | None = None):
        """What the reconstruction at the end of `items` was (`approximate`, `final_prompt`,
        `lenient_skips`, `restarts`) and its duel (None if there is none, not even an approximate
        one, or none within `timeout` seconds: signals cannot stop the Rust code)."""
        from ..core import Duel
        info, d = self._b.reconstruct(items, timeout)
        return SimpleNamespace(**info), (Duel.from_rust(d, self._config) if d is not None else None)

    @property
    def retries(self) -> int:
        """Fresh attempts after an inherited one failed or was approximate."""
        return self._b.retries

    @property
    def extended(self) -> int:
        """Decisions reached incrementally."""
        return self._b.extended

    @property
    def revealed(self) -> int:
        """Decisions with new reveals: reconstructed in full."""
        return self._b.revealed

    def close(self) -> None:
        pass
