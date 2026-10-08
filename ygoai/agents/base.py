"""Agent interface used by the arbiter."""

from __future__ import annotations


class Agent:
    """An AI playing one side of a duel.

    The arbiter calls `start` once, then `observe` for every message the
    player is allowed to see (already filtered), and `respond` whenever the
    player has to make a decision. `respond` receives the (filtered) prompt
    and, if `wants_state` is true, the player's view of the current state.
    Under a time control the arbiter sets `clock` before each `respond`:
    (seconds left on the player's clock, increment per decision); the time the
    decision takes is subtracted, then the increment added; below zero the
    player loses on time. Without a time control `clock` is None.
    """

    name = "agent"
    wants_state = False
    clock: tuple[float, float] | None = None

    def warm_up(self) -> None:
        """Loads what the agent needs, before any clock runs (an engine does it when
        the arbiter asks whether it is ready). Default: nothing."""

    def start(self, player: int, init: dict) -> None:
        self.player = player
        self.init = init

    def observe(self, msg: dict) -> None:
        pass

    def respond(self, prompt: dict, state: dict | None) -> dict:
        raise NotImplementedError

    def end(self, result: dict) -> None:
        pass

    def close(self) -> None:
        pass
