"""Flat Monte Carlo search on a reconstructed belief state.

At every important decision the agent reconstructs a complete duel from its
own observations (belief replay), then evaluates each candidate response by
forking the process, applying the response and playing the game forward with
a randomized greedy policy for both players. The response with the best
average evaluation is played. This is deliberately simple; it demonstrates
the pipeline observation stream -> belief duel -> fork() based search.
"""

from __future__ import annotations

import os
import random
import struct
import time

from ..actions import battle_options, encode, idle_options
from ..game import run_until_prompt
from ..state import capture
from .base import Agent
from .greedy_agent import GreedyAgent

MAX_APPROX_DIFFS = 4  # public facts an approximate reconstruction may get wrong to be searched on
# Time management under a time control (Agent.clock): each searched decision may use
# remaining / MOVES_TO_GO + INC_SHARE * increment (reconstruction included), rollouts
# round-robin over the candidates until it is spent; with little time left, no search.
MOVES_TO_GO = 40
INC_SHARE = 0.8
MIN_SEARCH_TIME = 2.0  # (seconds left below which it plays greedily; also 3 increments)
MAX_ROLLOUTS = 64  # (per candidate)
SEARCHED = ("SELECT_IDLECMD", "SELECT_BATTLECMD", "SELECT_CHAIN", "SELECT_EFFECTYN", "SELECT_YESNO",
            "SELECT_OPTION", "SELECT_CARD")


_NEURAL = None


def _neural_completion():
    """The learned deck models, in their own process (this one stays small: its
    probes fork it), else the corpus completion (e.g. without PyTorch)."""
    global _NEURAL
    if _NEURAL is None:
        try:
            from ..belief.neural_completion import ensure_service, neural_completion
            ensure_service()
            _NEURAL = neural_completion()
        except (ImportError, RuntimeError):
            from ..belief.corpus import CorpusCompletion
            _NEURAL = CorpusCompletion()
    return _NEURAL


def candidates(prompt: dict) -> list[dict]:
    t = prompt["type"]
    if t == "SELECT_IDLECMD":
        return idle_options(prompt)
    if t == "SELECT_BATTLECMD":
        return battle_options(prompt)
    if t == "SELECT_CHAIN":
        opts = [{"index": i} for i in range(len(prompt["chains"]))]
        return opts if prompt["forced"] else [{"index": -1}] + opts
    if t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
        return [{"yes": True}, {"yes": False}]
    if t == "SELECT_OPTION":
        return [{"index": i} for i in range(len(prompt["options"]))]
    if t == "SELECT_CARD" and prompt["min"] == 1 and prompt["max"] == 1:
        return [{"indices": [i]} for i in range(len(prompt["cards"]))]  # (one card: each a candidate)
    return []


def evaluate(duel, player: int) -> float:
    """Heuristic value of the position for `player` (roughly in [-1, 1])."""
    snap = capture(duel)
    lp = snap["lp"]
    me, opp = snap["players"][player], snap["players"][1 - player]

    def material(pl):
        field = sum(1 for c in pl["mzone"] + pl["szone"] if c is not None)
        atk = sum(c.get("attack", 0) for c in pl["mzone"] if c is not None and c.get("position", 0) & 0x5)
        return field + 0.5 * len(pl["hand"]) + atk / 3000.0

    value = (lp[player] - lp[1 - player]) / 8000.0 + 0.1 * (material(me) - material(opp))
    return max(-1.0, min(1.0, value))


class BeliefMCAgent(Agent):
    name = "belief_mc"
    wants_state = True

    def __init__(self, seed: int | None = None, rollouts: int = 2, horizon: int = 80,
                 max_candidates: int = 16, completion=None):
        self.rng = random.Random(seed)
        self.rollouts = rollouts
        self.horizon = horizon
        self.max_candidates = max_candidates
        self.completion = completion if completion is not None else _neural_completion()
        self.fallback = GreedyAgent(seed)
        self.searches = 0
        self.failures = 0
        self.rollouts_done = 0
        self.approximate = 0  # (searches on a best-effort reconstruction)

    def warm_up(self) -> None:
        from ..belief import agent, flow  # noqa: F401 - (the replay's modules: scipy, ...)
        from ..carddb import default_db
        from ..core import Duel
        Duel((0, 0, 0, 0), None, default_db()).close()  # (compiles the utility scripts)

    def start(self, player: int, init: dict) -> None:
        super().start(player, init)
        self.items: list[tuple] = []
        from ..belief.agent import AgentBelief
        self._belief = AgentBelief(player, init, self.completion)
        self.fallback.start(player, init)

    def observe(self, msg: dict) -> None:
        self.items.append(("msg", msg))
        self.fallback.observe(msg)

    def respond(self, prompt: dict, state: dict | None) -> dict:
        self.items.append(("prompt", prompt, state))
        resp = None
        cands = candidates(prompt)[:self.max_candidates] if prompt["type"] in SEARCHED else []
        if len(cands) > 1 and (self.clock is None or self.clock[0] > max(MIN_SEARCH_TIME, 3 * self.clock[1])):
            resp = self._search(prompt, cands)
        if resp is None:
            resp = self.fallback.respond(prompt, state)
        self.items.append(("response", resp))
        return resp

    def _search(self, prompt: dict, cands: list[dict]) -> dict | None:
        t0 = time.monotonic()
        budget = self._budget()
        # (from the previous decision's reconstruction: only the new part needs work;
        # the duel belongs to self._belief)
        br, duel = self._belief.reconstruct(self.items)
        if duel is None:
            self.failures += 1
            return None
        if br.approximate:
            # (a best-effort state only if it matches what this agent observes now: an
            # early divergence can leave the rest of the game far off)
            from ..belief.check import _count_diffs
            from ..state import viewer_view
            observed = self.items[-1][2] if len(self.items[-1]) > 2 else None
            if observed is None or _count_diffs(observed, viewer_view(duel, self.player)) > MAX_APPROX_DIFFS:
                self.failures += 1
                return None
        self.searches += 1
        self.approximate += br.approximate
        from ..observe import to_core_response
        core_prompt = getattr(br, "final_prompt", None) or prompt
        # (the duel stays as it is, the search runs in forks: the next decision's
        # reconstruction goes on from it, see AgentBelief)
        core_cands = [to_core_response(core_prompt, self.player, c) for c in cands]
        k = self._select(duel, core_prompt, core_cands, t0, budget)
        return None if k is None else cands[k]

    def _budget(self) -> float | None:
        """Seconds this searched decision may use (None: no time control)."""
        return None if self.clock is None else self.clock[0] / MOVES_TO_GO + INC_SHARE * self.clock[1]

    def _select(self, duel, core_prompt: dict, core_cands: list[dict], t0: float, budget: float | None) -> int | None:
        """Which of the candidates (core responses to the duel's prompt) to play:
        flat Monte Carlo with greedy rollouts."""
        totals, counts = [0.0] * len(core_cands), [0] * len(core_cands)
        if budget is None:  # (no time control: a fixed number of rollouts)
            for k, cc in enumerate(core_cands):
                for _ in range(self.rollouts):
                    totals[k] += self._rollout(duel, core_prompt, cc, self.rng.getrandbits(32))
                    counts[k] += 1
        else:
            # (round-robin until the budget is spent; the first round completes while
            # the clock allows twice the budget)
            for r in range(MAX_ROLLOUTS):
                for k, cc in enumerate(core_cands):
                    spent = time.monotonic() - t0
                    if spent > (2 * budget if r == 0 else budget):
                        break
                    totals[k] += self._rollout(duel, core_prompt, cc, self.rng.getrandbits(32))
                    counts[k] += 1
                else:
                    continue
                break
            self.rollouts_done += sum(counts)
        tried = [k for k in range(len(core_cands)) if counts[k]]
        if not tried:
            return None
        return max(tried, key=lambda k: totals[k] / counts[k])

    def _rollout(self, duel, prompt: dict, cand: dict, seed: int) -> float:
        """Forks, plays `cand` and continues greedily; returns the evaluation."""
        rfd, wfd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(rfd)
            value = -1.0
            try:
                value = self._play_out(duel, prompt, cand, seed)
            except BaseException:
                value = -1.0
            finally:
                os.write(wfd, struct.pack("<d", value))
                os._exit(0)
        os.close(wfd)
        data = os.read(rfd, 8)
        os.close(rfd)
        os.waitpid(pid, 0)
        return struct.unpack("<d", data)[0] if len(data) == 8 else -1.0

    def _play_out(self, duel, prompt: dict, cand: dict, seed: int) -> float:
        me = self.player
        policies = [GreedyAgent(seed), GreedyAgent(seed + 1)]
        duel.set_response(encode(prompt, cand))
        for _ in range(self.horizon):
            step = run_until_prompt(duel, prompt)
            for m in step.messages:
                if m["type"] == "WIN":
                    return 1.0 if m["player"] == me else -1.0
            if step.ended:
                break
            prompt = step.prompt
            resp = policies[prompt["player"]].respond(prompt, None)
            duel.set_response(encode(prompt, resp))
        return evaluate(duel, me)
