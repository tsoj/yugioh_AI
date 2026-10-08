"""A player that follows the policy network alone (no search).

At the decisions the search covers (belief_mc.SEARCHED with more than one
candidate) it scores the candidates with the policy (ygoai/nn/model.py) on
what it sees: its view of the state and the prompt as it got it (the network is
trained on the same, build_policy_data.py --view); everything else, and every
decision when there is no model, goes to the heuristic player. It plays the best
scored candidate, or with `temperature` > 0 samples by softmax(score / T).

The model: $YGOAI_POLICY (a train_policy_value.py export), else none.
"""

from __future__ import annotations

import math
import os
import random

from .. import constants as C
from .base import Agent
from .belief_mc import SEARCHED, candidates
from .heuristic_agents import HeuristicAgent

_MODELS: dict[str, object] = {}


def load_policy(path: str | None):
    if not path:
        return None
    model = _MODELS.get(path)
    if model is None:
        from ..nn.model import PolicyValueNet
        model = _MODELS[path] = PolicyValueNet(path)
    return model


class PolicyAgent(Agent):
    name = "policy"
    wants_state = True

    def __init__(self, seed: int | None = None, model=None, temperature: float = 0.0):
        self.rng = random.Random(seed)
        self.model = model if model is not None else load_policy(os.environ.get("YGOAI_POLICY"))
        self.temperature = temperature
        self.fallback = HeuristicAgent(seed)
        self.hint = None

    def start(self, player: int, init: dict) -> None:
        super().start(player, init)
        self.fallback.start(player, init)

    def observe(self, msg: dict) -> None:
        self.fallback.observe(msg)
        if msg["type"] == "HINT" and msg["hint_type"] == C.HINT_SELECTMSG and msg["player"] == self.player:
            self.hint = msg["data"]

    def respond(self, prompt: dict, state: dict | None) -> dict:
        hint, self.hint = self.hint, None
        cands = candidates(prompt) if prompt["type"] in SEARCHED else []
        if self.model is None or state is None or len(cands) < 2:
            return self.fallback.respond(prompt, state)
        k = self.choose(prompt, cands, state, hint)
        self.fallback.hint = None  # (this selection is answered here)
        return cands[k]

    def scores(self, prompt: dict, cands: list[dict], state: dict, hint: int | None) -> list[float]:
        from ..nn import features as F
        cards, scalars = F.position(state, self.player, lambda c: c or 0)
        acts = F.actions(prompt, cands, state, self.player, hint, lambda c: c or 0)
        h = self.model.trunk(cards, scalars)
        return [float(s) for s in self.model.scores(h, acts)]

    def choose(self, prompt: dict, cands: list[dict], state: dict, hint: int | None) -> int:
        s = self.scores(prompt, cands, state, hint)
        if self.temperature <= 0:
            return max(range(len(s)), key=s.__getitem__)
        top = max(s)
        weights = [math.exp((x - top) / self.temperature) for x in s]
        return self.rng.choices(range(len(s)), weights)[0]
