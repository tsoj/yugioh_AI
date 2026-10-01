"""More simple agents with different styles, to test the belief replay on more
kinds of games (e.g. many face-down cards, or erratic play)."""

from __future__ import annotations

import random

from .. import constants as C
from ..actions import battle_options, idle_options, random_response
from ..carddb import default_db
from .base import Agent
from .greedy_agent import GreedyAgent

MAX_SAME_ACTIVATION_PER_TURN = 3


class SetterAgent(Agent):
    """Plays defensively: sets monsters and spells/traps face-down, activates
    only sometimes in its own turn, responds in chain windows (e.g. with its set
    traps), attacks now and then."""

    name = "setter"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.activations: dict[tuple, int] = {}

    def observe(self, msg: dict) -> None:
        if msg["type"] == "NEW_TURN":
            self.activations.clear()

    def _allowed(self, entry: dict) -> bool:
        key = (entry["code"], entry["loc"], entry["seq"], entry.get("desc"))
        return self.activations.get(key, 0) < MAX_SAME_ACTIVATION_PER_TURN

    def _note(self, entry: dict) -> None:
        key = (entry["code"], entry["loc"], entry["seq"], entry.get("desc"))
        self.activations[key] = self.activations.get(key, 0) + 1

    def respond(self, prompt: dict, state: dict | None) -> dict:
        t = prompt["type"]
        rng = self.rng
        if t == "SELECT_IDLECMD":
            opts = idle_options(prompt)
            by_kind: dict[str, list[dict]] = {}
            for o in opts:
                by_kind.setdefault(o["kind"], []).append(o)
            acts = [o for o in by_kind.get("activate", ()) if self._allowed(prompt["activatable"][o["index"]])]
            if acts and rng.random() < 0.3:
                choice = rng.choice(acts)
                self._note(prompt["activatable"][choice["index"]])
                return choice
            for kind in ("mset", "sset", "summon"):
                if by_kind.get(kind) and rng.random() < 0.8:
                    return rng.choice(by_kind[kind])
            if by_kind.get("to_bp") and rng.random() < 0.5:
                return by_kind["to_bp"][0]
            if by_kind.get("to_ep"):
                return by_kind["to_ep"][0]
            return rng.choice(opts)
        if t == "SELECT_BATTLECMD":
            opts = battle_options(prompt)
            attacks = [o for o in opts if o["kind"] == "attack"]
            if attacks and rng.random() < 0.5:
                return rng.choice(attacks)
            for kind in ("to_m2", "to_ep"):
                for o in opts:
                    if o["kind"] == kind:
                        return o
            return rng.choice(opts)
        if t == "SELECT_CHAIN":
            cands = [i for i, c in enumerate(prompt["chains"]) if self._allowed(c)]
            if cands and rng.random() < 0.7:
                i = rng.choice(cands)
                self._note(prompt["chains"][i])
                return {"index": i}
            return {"index": 0} if prompt["forced"] else {"index": -1}
        if t == "SELECT_POSITION":
            for pos in (C.POS_FACEDOWN_DEFENSE, C.POS_FACEUP_DEFENSE):
                if prompt["positions"] & pos:
                    return {"position": pos}
        return random_response(prompt, rng, default_db(), cancel_prob=0.0)


class MixedAgent(GreedyAgent):
    """Greedy, but a random response instead 30% of the time."""

    name = "mixed"

    def respond(self, prompt: dict, state: dict | None) -> dict:
        if self.rng.random() < 0.3:
            return random_response(prompt, self.rng, default_db())
        return super().respond(prompt, state)
