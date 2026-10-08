"""Alpha-beta search with iterative deepening on the reconstructed belief duel.

The duel comes from the belief replay as for belief_mc (same reconstruction, time
budget and fallbacks); only the choice differs. The engine cannot copy a duel, so
every node of the search is a forked process: a child applies its move, runs the
duel to the next decision that counts as a ply, and either evaluates it (depth
exhausted, game over) or searches on, reporting its value through a pipe. At most
depth + 1 processes exist at a time (depth first).

- Plies are the decisions of either player at command prompts, chain windows,
  yes/no questions and options (belief_mc.SEARCHED) with more than one candidate;
  everything in between (cards, places, positions, ...) is answered by the
  heuristic player's rules, the opponent's hidden cards are the belief's guesses.
- Move ordering: the heuristic player's priorities (activate, Special Summon,
  Normal Summon the stronger first, set a monster, change positions, battle, set
  Spells/Traps, end the turn; in battle: activate, attack the stronger attacker
  first; chain before passing; yes first in one's own turn). At the root, the
  previous iteration's values order the moves.
- Time: iterative deepening within belief_mc's budget per decision; an iteration
  that runs out of time is discarded (the deeper one only counts complete).
- Value (`evaluate`): win/loss, then LP, monsters, monsters that beat everything
  on the other side, cards in hand, set Spells/Traps, a nearly empty deck.
"""

from __future__ import annotations

import gc
import math
import os
import struct
import time

from .. import constants as C
from ..actions import encode
from ..carddb import default_db
from ..game import run_until_prompt
from ..observe import AUTO_RESPONSE, is_hidden_prompt
from ..state import capture
from .belief_mc import SEARCHED, BeliefMCAgent, candidates
from .heuristic_agents import HeuristicAgent

WIN = 100000.0
MAX_DEPTH = 12
DEPTH_NO_CLOCK = 2  # (without a time control: a fixed depth)
MAX_CHILDREN = 12  # (per node, after ordering)
# time: a soft bound per searched decision (remaining / MOVES_TO_GO + increment): a new
# iteration starts only within NEXT_ITERATION_SHARE of it; a hard bound (HARD_FACTOR x the
# soft one, at most HARD_SHARE of the clock) aborts the iteration in progress
MOVES_TO_GO = 20
NEXT_ITERATION_SHARE = 0.5
HARD_FACTOR = 3.0
HARD_SHARE = 0.25
MAX_AUTO_STEPS = 300  # (prompts answered by the rules between two plies)

# evaluation weights (points; 1 point = 100 LP)
W_LP = 1 / 100
W_MONSTER = 8.0
W_DOMINANT = 4.0  # (a face-up monster stronger than everything on the other side)
W_HAND = 3.0
W_BACKROW = 2.0
W_DECK_OUT = 20.0  # (two cards or fewer left to draw)

_IDLE_ORDER = {"activate": 0, "spsummon": 1, "summon": 2, "mset": 3, "repos": 4, "to_bp": 5, "sset": 6,
               "to_ep": 7, "shuffle": 8}
_BATTLE_ORDER = {"activate": 0, "attack": 1, "to_m2": 2, "to_ep": 3}


class _Abort(Exception):
    pass


def _battle_stat(c: dict) -> int:
    return c.get("defense", 0) if c.get("position", 0) & C.POS_DEFENSE else c.get("attack", 0)


def evaluate(duel, me: int) -> float:
    """The position's value for `me` (in points of 100 LP)."""
    snap = capture(duel)
    lp = snap["lp"]
    sides = [snap["players"][me], snap["players"][1 - me]]
    mons = [[c for c in side["mzone"] if c is not None] for side in sides]
    value = W_LP * (lp[me] - lp[1 - me]) + W_MONSTER * (len(mons[0]) - len(mons[1]))
    for s in (0, 1):
        sign = 1 if s == 0 else -1
        best_other = max((_battle_stat(c) for c in mons[1 - s]), default=-1)
        for c in mons[s]:
            if c.get("position", 0) & C.POS_FACEUP_ATTACK and c.get("attack", 0) > best_other:
                value += sign * W_DOMINANT
        side = sides[s]
        value += sign * W_HAND * len(side["hand"])
        value += sign * W_BACKROW * sum(1 for c in side["szone"]
                                        if c is not None and c.get("position", 0) & C.POS_FACEDOWN)
        if len(side["deck"]) <= 2:
            value -= sign * W_DECK_OUT
    return value


def ordered(prompt: dict, turn_player: int) -> list[dict]:
    """The candidates of a prompt, the heuristic player's preference first."""
    cands = candidates(prompt)
    t = prompt["type"]
    db = default_db()

    def atk(entry):
        card = db.get(entry["code"])
        return card.attack if card is not None else 0
    if t == "SELECT_IDLECMD":
        def key(c):
            k = c["kind"]
            strength = -atk(prompt["summonable"][c["index"]]) if k == "summon" else 0
            return (_IDLE_ORDER.get(k, 9), strength)
        return sorted(cands, key=key)
    if t == "SELECT_BATTLECMD":
        def key(c):
            k = c["kind"]
            return (_BATTLE_ORDER.get(k, 9), -atk(prompt["attackable"][c["index"]]) if k == "attack" else 0)
        return sorted(cands, key=key)
    if t == "SELECT_CHAIN":
        return sorted(cands, key=lambda c: c["index"] == -1)
    if t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
        return sorted(cands, key=lambda c: c["yes"] != (prompt["player"] == turn_player))
    return cands


class AlphaBetaAgent(BeliefMCAgent):
    name = "alphabeta"

    def __init__(self, seed: int | None = None, **kw):
        super().__init__(seed=seed, **kw)
        # (the decisions it does not search, e.g. attack targets, cards, positions,
        # by the same rules the search assumes for them)
        self.fallback = HeuristicAgent(seed)
        self.depths: list[int] = []  # (depth completed per search)
        self.nodes = 0

    def _budget(self) -> float | None:
        return None if self.clock is None else self.clock[0] / MOVES_TO_GO + self.clock[1]

    def _select(self, duel, core_prompt: dict, core_cands: list[dict], t0: float, budget: float | None) -> int | None:
        me = self.player
        turn_player = duel.info()["turn_player"]
        pref = ordered(core_prompt, turn_player)
        order = sorted(range(len(core_cands)), key=lambda k: pref.index(core_cands[k])
                       if core_cands[k] in pref else len(pref))
        policies = [HeuristicAgent(self.rng.getrandbits(32)), HeuristicAgent(self.rng.getrandbits(32))]
        for p, a in enumerate(policies):
            a.player = p
        self._policies = policies
        deadline = None if budget is None else \
            t0 + max(budget, min(HARD_FACTOR * budget, HARD_SHARE * self.clock[0]))
        best, done = order[0], 0
        # (the forked nodes never collect the inherited objects: no copies of their pages)
        gc.freeze()
        try:
            best, done = self._deepen(duel, core_prompt, core_cands, order, t0, budget, deadline)
        finally:
            gc.unfreeze()
        self.depths.append(done)
        return best

    def _deepen(self, duel, core_prompt, core_cands, order, t0, budget, deadline) -> tuple[int, int]:
        best, done = order[0], 0
        for depth in range(1, (DEPTH_NO_CLOCK if budget is None else MAX_DEPTH) + 1):
            if budget is not None and depth > 1 and time.monotonic() - t0 > NEXT_ITERATION_SHARE * budget:
                break
            values = {}
            alpha, it_best = -math.inf, None
            try:
                for k in order:
                    v = self._child(duel, core_prompt, core_cands[k], depth - 1, alpha, math.inf, deadline, 1)
                    values[k] = v
                    if v > alpha:
                        alpha, it_best = v, k
            except _Abort:
                # (an unfinished iteration: its best counts if the previous best, searched
                # first, was searched completely and another one did better)
                if order[0] in values and it_best is not None:
                    best = it_best
                break
            best, done = it_best, depth
            # (the next iteration: the best first; cut-off moves keep their bound's order)
            order = sorted(order, key=lambda k: -values[k])
            if abs(values[best]) >= WIN / 2:
                break  # (a forced win or loss: deeper changes nothing)
        return best, done

    def _node(self, duel, prompt: dict, depth: int, alpha: float, beta: float, deadline, ply: int) -> float:
        cands = ordered(prompt, duel.info()["turn_player"])[:MAX_CHILDREN]
        maximizing = prompt["player"] == self.player
        best = -math.inf if maximizing else math.inf
        for i, c in enumerate(cands):
            if i == len(cands) - 1:
                # (the last move: nothing needs this position afterwards, no fork)
                if deadline is not None and time.monotonic() > deadline:
                    raise _Abort()
                self.nodes += 1
                v = self._expand(duel, prompt, c, depth - 1, alpha, beta, deadline, ply + 1)
            else:
                v = self._child(duel, prompt, c, depth - 1, alpha, beta, deadline, ply + 1)
            if maximizing:
                best = max(best, v)
                alpha = max(alpha, best)
            else:
                best = min(best, v)
                beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    def _child(self, duel, prompt: dict, cand: dict, depth: int, alpha: float, beta: float, deadline,
               ply: int) -> float:
        """The value of `cand` (searched `depth` plies deeper) in a forked process."""
        if deadline is not None and time.monotonic() > deadline:
            raise _Abort()
        self.nodes += 1
        rfd, wfd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(rfd)
            gc.disable()  # (short-lived)
            value = math.nan
            try:
                value = self._expand(duel, prompt, cand, depth, alpha, beta, deadline, ply)
            except BaseException:  # noqa: BLE001 - (_Abort, or a broken line: no value)
                value = math.nan
            finally:
                os.write(wfd, struct.pack("<d", value))
                os._exit(0)
        os.close(wfd)
        data = os.read(rfd, 8)
        os.close(rfd)
        os.waitpid(pid, 0)
        value = struct.unpack("<d", data)[0] if len(data) == 8 else math.nan
        if math.isnan(value):
            if deadline is not None and time.monotonic() > deadline:
                raise _Abort()
            value = -WIN if prompt["player"] == self.player else WIN  # (an engine error: avoid the line)
        return value

    def _expand(self, duel, prompt: dict, cand: dict, depth: int, alpha: float, beta: float, deadline,
                ply: int) -> float:
        """(in the child) Plays `cand`, runs to the next ply and values it."""
        me = self.player
        duel.set_response(encode(prompt, cand))
        for _ in range(MAX_AUTO_STEPS):
            step = run_until_prompt(duel, prompt)
            for m in step.messages:
                if m["type"] == "WIN":
                    return WIN - ply if m["player"] == me else -WIN + ply
            if step.ended:
                return evaluate(duel, me)
            prompt = step.prompt
            if is_hidden_prompt(prompt):
                resp = AUTO_RESPONSE
            elif prompt["type"] in SEARCHED and len(candidates(prompt)) > 1:
                if depth <= 0:
                    return evaluate(duel, me)
                return self._node(duel, prompt, depth, alpha, beta, deadline, ply)
            elif prompt["type"] in SEARCHED:
                resp = candidates(prompt)[0] if candidates(prompt) else \
                    self._policies[prompt["player"]].respond(prompt, None)
            else:
                resp = self._policies[prompt["player"]].respond(prompt, None)
            duel.set_response(encode(prompt, resp))
        return evaluate(duel, me)
