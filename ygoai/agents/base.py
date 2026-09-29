"""Agent interface used by the arbiter."""

from __future__ import annotations


class Agent:
    """An AI playing one side of a duel.

    The arbiter calls `start` once, then `observe` for every message the
    player is allowed to see (already filtered), and `respond` whenever the
    player has to make a decision. `respond` receives the (filtered) prompt
    and, if `wants_state` is true, the player's view of the current state.
    """

    name = "agent"
    wants_state = False

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
