"""pUCT search (AlphaZero-style tree search) on the belief's reconstruction.

The tree's nodes are the same plies as alphabeta's (decisions at command
prompts, chain windows, yes/no questions, options and one-card selections with
more than one candidate; everything in between by the heuristic player's rules).
Each node keeps, per candidate, the policy network's prior (nn/policy.npz, on what
the player to decide sees), visits and the summed value for the node's player:
values are win probabilities of that player, so the players need not alternate.
The priors: softened (PRIOR_TEMPERATURE) and mixed with a uniform one (PRIOR_MIX).

A simulation walks down by pUCT (Q + C_PUCT * P * sqrt(visits) / (1 + n), Q
normalized to [0, 1] by the least and greatest values in the tree as in MuZero:
the win probabilities of an evaluation stay close to one half; an unvisited
candidate's Q is its node's value minus FPU_REDUCTION), and at the first
candidate without a node it forks the search process, replays the path from the
root (the core is deterministic for identical call histories) and reports the new
node: its player, candidates, priors and value (the handmade evaluation as a win
probability: sigmoid(VALUE_K * evaluate), VALUE_K fitted on the SPRT games'
results), or the result if the game ended. The value is backed up along the path,
for each node's player. The move: the root's most visited candidate.

It runs in alphabeta's search server (search_server.py), whose mirror it searches.
"""

from __future__ import annotations

import gc
import math
import os
import pickle
import time

from ..actions import encode
from ..game import run_until_prompt
from ..observe import AUTO_RESPONSE, filter_prompt, is_hidden_prompt, to_core_response
from .. import constants as C
from .alphabeta import MAX_AUTO_STEPS, AlphaBetaAgent, evaluate, ordered
from .belief_mc import SEARCHED, candidates
from .heuristic_agents import HeuristicAgent

C_PUCT = 1.25
FPU_REDUCTION = 0.1
PRIOR_TEMPERATURE = 2.0  # (the policy is far too sure for a prior: it was not trained on search visits)
PRIOR_MIX = 0.2  # (share of a uniform prior mixed in)
VALUE_K = 0.0216  # (per point of the evaluation: logistic regression of the result, SPRT games)
SIMULATIONS_NO_CLOCK = 64
MAX_SIMULATIONS = 10000
FINAL = "visits"  # (the move: "visits", the most visited; "q", the best value among those visited at
FINAL_SHARE = 0.25  # least FINAL_SHARE as often as the most visited)
# ($YGOAI_PUCT, e.g. "C_PUCT=0.5,FINAL=q": other values of the parameters above, for tuning)
for _item in filter(None, os.environ.get("YGOAI_PUCT", "").split(",")):
    _name, _value = _item.split("=")
    globals()[_name] = type(globals()[_name])(_value)


class _Node:
    __slots__ = ("player", "cands", "prior", "visits", "value_sum", "children", "value0", "terminal")

    def __init__(self, player: int, cands: list[dict], prior: list[float], value0: float,
                 terminal: bool = False):
        self.player = player
        self.cands = cands
        self.prior = prior
        self.visits = [0] * len(cands)
        self.value_sum = [0.0] * len(cands)
        self.children: dict[int, _Node] = {}
        self.value0 = value0  # (player 0's win probability here, by the evaluation or the result)
        self.terminal = terminal

    @property
    def value(self) -> float:
        return self.value0 if self.player == 0 else 1.0 - self.value0

    def select(self, bounds: list[float]) -> int:
        total = sum(self.visits)
        root = math.sqrt(total + 1)
        low, span = bounds[0], bounds[1] - bounds[0]

        def norm(q):
            return (q - low) / span if span > 0 else 0.5
        fpu = norm(self.value) - FPU_REDUCTION

        def score(k):
            n = self.visits[k]
            q = norm(self.value_sum[k] / n) if n else fpu
            return q + C_PUCT * self.prior[k] * root / (1 + n)
        return max(range(len(self.cands)), key=score)


def _same(code):
    return code or 0


_policy = None


def _policy_net():
    global _policy
    if _policy is None:
        from ..carddb import default_db
        from .policy_agent import DEFAULT_MODEL, load_policy
        _policy = load_policy(str(DEFAULT_MODEL))
        # (every card's static features now, before the forks: each would compute them
        # again from the card database)
        _policy._static(sorted(default_db().cards))
    return _policy


class PuctAgent(AlphaBetaAgent):
    name = "puct"
    server_agent = ("ygoai.agents.puct", "PuctAgent")

    @staticmethod
    def warm_server() -> None:
        _policy_net()

    def warm_up(self) -> None:
        super().warm_up()
        if self._server is None:
            _policy_net()

    def search(self, duel, core_prompt: dict, core_cands: list[dict], t0: float,
               budget: float | None, hint: int | None = None) -> tuple[int, int]:
        policies = [HeuristicAgent(self.rng.getrandbits(32)), HeuristicAgent(self.rng.getrandbits(32))]
        for p, a in enumerate(policies):
            a.player = p
        self._policies = policies
        policies[core_prompt["player"]].hint = hint
        gc.freeze()  # (the forks never collect the inherited objects: no copies of their pages)
        try:
            info = self._simulate(duel, core_prompt, [], core_cands)
            if info is None or info[0] != "leaf":
                return 0, 0
            _, player, cands, prior, value0 = info
            root = _Node(player, core_cands, prior, value0)
            deadline = None if budget is None else t0 + budget
            sims = 0 if budget is not None else MAX_SIMULATIONS - SIMULATIONS_NO_CLOCK
            depth = 0
            bounds = [math.inf, -math.inf]  # (the least and greatest Q in the tree, for the nodes' players)
            while sims < MAX_SIMULATIONS:
                if deadline is not None and time.monotonic() > deadline:
                    break
                sims += 1
                node, path = root, []
                while True:
                    k = node.select(bounds)
                    path.append((node, k))
                    child = node.children.get(k)
                    if child is None:
                        child = self._expand(duel, core_prompt, path)
                        node.children[k] = child
                        break
                    if child.terminal:
                        break
                    node = child
                depth = max(depth, len(path))
                value0 = child.value0
                for n, k in path:
                    n.visits[k] += 1
                    n.value_sum[k] += value0 if n.player == 0 else 1.0 - value0
                    q = n.value_sum[k] / n.visits[k]
                    bounds[0], bounds[1] = min(bounds[0], q), max(bounds[1], q)
            self.nodes += sims
            if FINAL == "q" and max(root.visits) > 0:
                most = max(root.visits)
                best = max((k for k in range(len(root.cands)) if root.visits[k] >= FINAL_SHARE * most),
                           key=lambda k: root.value_sum[k] / root.visits[k])
            else:
                best = max(range(len(root.cands)),
                           key=lambda k: (root.visits[k], root.value_sum[k] / max(root.visits[k], 1)))
            return best, depth
        finally:
            gc.unfreeze()

    def _expand(self, duel, root_prompt: dict, path: list[tuple]) -> _Node:
        responses = [n.cands[k] for n, k in path]
        info = self._simulate(duel, root_prompt, responses, None)
        mover = path[-1][0].player
        if info is None:  # (an engine error: the player who moved there avoids the line)
            return _Node(mover, [], [], 0.0 if mover == 0 else 1.0, terminal=True)
        if info[0] == "end":
            return _Node(mover, [], [], info[1], terminal=True)
        _, player, cands, prior, value0 = info
        return _Node(player, cands, prior, value0)

    def _simulate(self, duel, prompt: dict, responses: list[dict], cands: list[dict] | None):
        """(in a fork) Plays `responses` from the root and reports where they lead:
        ("end", player 0's result) or ("leaf", player, candidates, priors, player 0's
        win probability); None on an engine error."""
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            gc.disable()  # (short-lived)
            data = b""
            try:
                data = pickle.dumps(self._play(duel, prompt, responses, cands))
            except BaseException:  # noqa: BLE001 - (a broken line: no report)
                data = b""
            finally:
                with os.fdopen(w, "wb") as f:
                    f.write(data)
                os._exit(0)
        os.close(w)
        with os.fdopen(r, "rb") as f:
            data = f.read()
        os.waitpid(pid, 0)
        return pickle.loads(data) if data else None

    def _play(self, duel, prompt: dict, responses: list[dict], cands: list[dict] | None):
        for resp in responses:
            duel.set_response(encode(prompt, resp))
            prompt, result = self._advance(duel, prompt)
            if prompt is None:
                return "end", result
        player = prompt["player"]
        hint = self._policies[player].hint  # (before the card order: it uses the hint up)
        if cands is None:
            cands = ordered(prompt, duel.info()["turn_player"], self._card_order(prompt))
        prior = self._prior(duel, prompt, cands, hint)
        return "leaf", player, cands, prior, self._value0(duel)

    def _advance(self, duel, prompt: dict):
        """Runs to the next ply: (its prompt, None), or (None, player 0's result or value)."""
        for _ in range(MAX_AUTO_STEPS):
            step = run_until_prompt(duel, prompt)
            for m in step.messages:
                if m["type"] == "WIN":
                    return None, 1.0 if m["player"] == 0 else 0.0
                if m["type"] == "NEW_TURN" or m["type"] == "HINT" and m["hint_type"] == C.HINT_SELECTMSG:
                    for policy in self._policies:
                        policy.observe(m)
            if step.ended:
                return None, self._value0(duel)
            prompt = step.prompt
            if is_hidden_prompt(prompt):
                resp = AUTO_RESPONSE
            elif prompt["type"] in SEARCHED:
                cs = candidates(prompt)
                if len(cs) > 1:
                    return prompt, None
                resp = cs[0] if cs else self._policies[prompt["player"]].respond(prompt, None)
            else:
                resp = self._policies[prompt["player"]].respond(prompt, None)
            duel.set_response(encode(prompt, resp))
        return None, self._value0(duel)

    @staticmethod
    def _value0(duel) -> float:
        x = max(-50.0, min(50.0, VALUE_K * evaluate(duel, 0)))
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _prior(duel, prompt: dict, cands: list[dict], hint: int | None) -> list[float]:
        """The policy's probabilities of `cands` (core responses), on what the player to
        decide sees; uniform if that fails."""
        from ..nn import features as NNF
        from ..state import capture, view
        uniform = [1.0 / len(cands)] * len(cands)
        if len(cands) < 2:
            return uniform
        p = prompt["player"]
        try:
            net = _policy_net()
            shown = filter_prompt(prompt, p)
            seen = view(capture(duel), p)
            vcands = candidates(shown)
            cards, scalars = NNF.position(seen, p, _same)
            scores = net.scores(net.trunk(cards, scalars), NNF.actions(shown, vcands, seen, p, hint, _same))
            score = {repr(to_core_response(prompt, p, c)): float(x) / PRIOR_TEMPERATURE
                     for c, x in zip(vcands, scores)}
        except Exception:  # noqa: BLE001
            return uniform
        x = [score.get(repr(c)) for c in cands]
        known = [v for v in x if v is not None]
        if not known:
            return uniform
        low = min(known)
        x = [low if v is None else v for v in x]
        top = max(x)
        e = [math.exp(v - top) for v in x]
        total = sum(e)
        return [(1 - PRIOR_MIX) * v / total + PRIOR_MIX / len(e) for v in e]
