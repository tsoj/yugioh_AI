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

from ..actions import battle_options, encode, idle_options
from ..game import run_until_prompt
from ..state import capture
from .base import Agent
from .greedy_agent import GreedyAgent

MAX_APPROX_DIFFS = 4  # public facts an approximate reconstruction may get wrong to be searched on
SEARCHED = ("SELECT_IDLECMD", "SELECT_BATTLECMD", "SELECT_CHAIN", "SELECT_EFFECTYN", "SELECT_YESNO",
            "SELECT_OPTION")


_NEURAL = None


def _neural_completion():
    """The learned deck models (one per process), else the corpus completion
    (e.g. without PyTorch)."""
    global _NEURAL
    if _NEURAL is None:
        try:
            from ..belief.neural_completion import neural_completion
            _NEURAL = neural_completion()
        except ImportError:
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
        self.approximate = 0  # (searches on a best-effort reconstruction)

    def start(self, player: int, init: dict) -> None:
        super().start(player, init)
        self.items: list[tuple] = []
        self.fallback.start(player, init)

    def observe(self, msg: dict) -> None:
        self.items.append(("msg", msg))
        self.fallback.observe(msg)

    def respond(self, prompt: dict, state: dict | None) -> dict:
        self.items.append(("prompt", prompt, state))
        resp = None
        cands = candidates(prompt)[:self.max_candidates] if prompt["type"] in SEARCHED else []
        if len(cands) > 1:
            resp = self._search(prompt, cands)
        if resp is None:
            resp = self.fallback.respond(prompt, state)
        self.items.append(("response", resp))
        return resp

    def _search(self, prompt: dict, cands: list[dict]) -> dict | None:
        from ..belief.replay import BeliefReplay, Divergence
        from ..belief.tracker import TrackerError
        br = BeliefReplay(self.player, self.init, self.items, completion=self.completion,
                          check_state=False)
        br.allow_approximate = True  # (an agent needs some state: the closest one, from observations only)
        br.read_guided = False  # (too slow per decision: the approximate pass comes right after the repairs)
        try:
            duel = br.run()
        except (Divergence, TrackerError):
            self.failures += 1
            return None
        if br.approximate:
            # (a best-effort state only if it matches what this agent observes now: an
            # early divergence can leave the rest of the game far off)
            from ..belief.check import _count_diffs
            from ..state import viewer_view
            observed = self.items[-1][2] if len(self.items[-1]) > 2 else None
            if observed is None or _count_diffs(observed, viewer_view(duel, self.player)) > MAX_APPROX_DIFFS:
                duel.close()
                self.failures += 1
                return None
        self.searches += 1
        self.approximate += br.approximate
        from ..observe import to_core_response
        core_prompt = getattr(br, "final_prompt", None) or prompt
        try:
            best, best_value = None, -2.0
            for cand in cands:
                core_cand = to_core_response(core_prompt, self.player, cand)
                values = [self._rollout(duel, core_prompt, core_cand, self.rng.getrandbits(32))
                          for _ in range(self.rollouts)]
                value = sum(values) / len(values)
                if value > best_value:
                    best, best_value = cand, value
            return best
        finally:
            duel.close()

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
