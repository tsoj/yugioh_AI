"""Simple greedy agent: activates everything it can, summons, sets, always attacks.

Everything the agent does not have an opinion about is chosen randomly (from
a seeded RNG). Repeated activations of the same effect are capped per turn so
that unlimited effects cannot loop forever.
"""

from __future__ import annotations

import random

from .. import constants as C
from ..actions import battle_options, idle_options, random_response
from ..carddb import default_db
from .base import Agent

MAX_SAME_ACTIVATION_PER_TURN = 3


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.turn = -1
        self.activations: dict[tuple, int] = {}

    def observe(self, msg: dict) -> None:
        if msg["type"] == "NEW_TURN":
            self.turn += 1
            self.activations.clear()

    def _allowed(self, entry: dict) -> bool:
        key = (entry["code"], entry["loc"], entry["seq"], entry.get("desc"))
        return self.activations.get(key, 0) < MAX_SAME_ACTIVATION_PER_TURN

    def _note(self, entry: dict) -> None:
        key = (entry["code"], entry["loc"], entry["seq"], entry.get("desc"))
        self.activations[key] = self.activations.get(key, 0) + 1

    def respond(self, prompt: dict, state: dict | None) -> dict:
        t = prompt["type"]
        if t == "SELECT_IDLECMD":
            opts = idle_options(prompt)
            for kind in ("activate", "spsummon", "summon", "sset", "mset", "to_bp", "to_ep"):
                cands = [o for o in opts if o["kind"] == kind]
                if kind in ("activate", "spsummon"):
                    lst = "activatable" if kind == "activate" else "spsummonable"
                    cands = [o for o in cands if self._allowed(prompt[lst][o["index"]])]
                if cands:
                    choice = self.rng.choice(cands)
                    if kind in ("activate", "spsummon"):
                        lst = "activatable" if kind == "activate" else "spsummonable"
                        self._note(prompt[lst][choice["index"]])
                    return choice
            return self.rng.choice(opts)
        if t == "SELECT_BATTLECMD":
            opts = battle_options(prompt)
            acts = [o for o in opts if o["kind"] == "activate"
                    and self._allowed(prompt["activatable"][o["index"]])]
            if acts:
                choice = self.rng.choice(acts)
                self._note(prompt["activatable"][choice["index"]])
                return choice
            attacks = [o for o in opts if o["kind"] == "attack"]
            if attacks:
                return self.rng.choice(attacks)
            for kind in ("to_m2", "to_ep"):
                for o in opts:
                    if o["kind"] == kind:
                        return o
            return self.rng.choice(opts)
        if t == "SELECT_CHAIN":
            cands = [i for i, c in enumerate(prompt["chains"]) if self._allowed(c)]
            if cands:
                i = self.rng.choice(cands)
                self._note(prompt["chains"][i])
                return {"index": i}
            if prompt["forced"]:
                return {"index": 0}
            return {"index": -1}
        if t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
            return {"yes": True}
        if t == "SELECT_POSITION":
            if prompt["positions"] & C.POS_FACEUP_ATTACK:
                return {"position": C.POS_FACEUP_ATTACK}
        if t == "SELECT_CARD" and prompt["cancelable"]:
            # never cancel: pick the maximum amount of cards
            n = len(prompt["cards"])
            k = min(prompt["max"], n)
            if k >= max(prompt["min"], 1):
                return {"indices": sorted(self.rng.sample(range(n), k))}
        if t == "SELECT_UNSELECT_CARD" and prompt["select_cards"]:
            return {"index": self.rng.randrange(len(prompt["select_cards"]))}
        return random_response(prompt, self.rng, default_db(), cancel_prob=0.0)
