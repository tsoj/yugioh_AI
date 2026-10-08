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
from ..actions import BATTLE_LISTS, IDLE_LISTS
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

    def __init__(self, seed: int | None = None, model=None, temperature: float = 0.0, record: bool = False):
        self.rng = random.Random(seed)
        self.model = model if model is not None else load_policy(os.environ.get("YGOAI_POLICY"))
        self.temperature = temperature
        self.fallback = HeuristicAgent(seed)
        self.hint = None
        # (with `record`: each decision of the policy as (cards, scalars, actions, chosen, log-probability), for training)
        self.decisions: list[tuple] | None = [] if record else None

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
        # (the heuristic player's cap on repeating an activation or Special Summon in a
        # turn: an effect without a limit could otherwise loop forever)
        cands = [c for c in cands if self._capped(prompt, c) is None or self.fallback._allowed(self._capped(prompt, c))]
        if self.model is None or state is None or len(cands) < 2:
            if len(cands) == 1 and self.model is not None:
                choice = cands[0]
            else:
                return self.fallback.respond(prompt, state)
        else:
            choice = cands[self.choose(prompt, cands, state, hint)]
        self.fallback.hint = None  # (this selection is answered here)
        entry = self._capped(prompt, choice)
        if entry is not None:
            self.fallback._note(entry)
        return choice

    @staticmethod
    def _capped(prompt: dict, cand: dict) -> dict | None:
        """The entry of a candidate that counts toward the cap (activations, Special
        Summons, chaining), else None."""
        t = prompt["type"]
        if t in ("SELECT_IDLECMD", "SELECT_BATTLECMD") and cand.get("kind") in ("activate", "spsummon"):
            return prompt[(IDLE_LISTS if t == "SELECT_IDLECMD" else BATTLE_LISTS)[cand["kind"]]][cand["index"]]
        if t == "SELECT_CHAIN" and cand.get("index", -1) >= 0:
            return prompt["chains"][cand["index"]]
        return None

    def choose(self, prompt: dict, cands: list[dict], state: dict, hint: int | None) -> int:
        from ..nn import features as F
        cards, scalars = F.position(state, self.player, lambda c: c or 0)
        acts = F.actions(prompt, cands, state, self.player, hint, lambda c: c or 0)
        s = [float(x) for x in self.model.scores(self.model.trunk(cards, scalars), acts)]
        if self.temperature <= 0:
            k, logp = max(range(len(s)), key=s.__getitem__), 0.0
        else:
            top = max(s)
            weights = [math.exp((x - top) / self.temperature) for x in s]
            k = self.rng.choices(range(len(s)), weights)[0]
            logp = math.log(weights[k] / sum(weights))
        if self.decisions is not None:
            self.decisions.append((cards, scalars, acts, k, logp))  # (logp: of the move played, as sampled)
        return k
