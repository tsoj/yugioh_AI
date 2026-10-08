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
  heuristic player's rules (which see the core's select hints and new turns), the
  opponent's hidden cards are the belief's guesses.
- Move ordering: the heuristic player's priorities (activate, Special Summon,
  Normal Summon the stronger first, set a monster, change positions, battle, set
  Spells/Traps, end the turn; in battle: activate, attack the stronger attacker
  first; chain before passing; yes first in one's own turn). At the root, the
  previous iteration's values order the moves.
- Speed: the search runs in a small server process (search_server.py) that
  rebuilds the duel from its journal: every node forks a process, and a fork
  costs by the size of the process (this one holds the belief, the deck models'
  clients, scipy); its memory in huge pages before a search (cheaper forks).
- Transposition table (`_tt_key`, `_node`): positions reached by other move
  orders (e.g. two independent activations), keyed by a hash of the engine's
  state queries and the pending decision; in shared memory (the nodes are forked
  processes): value bounds for cutoffs and the best move to try first (also
  across the iterations of one decision).
- History (`_move_key`, `_node`): moves that caused a cutoff (by player, kind and
  card) go first among the moves of later nodes, in this search and (halved per
  decision) the next ones; in memory shared with the forked nodes.
- Negamax: a node's value is for the player to move there; it changes sign
  (and the window becomes (-beta, -alpha)) only from one player's decision to
  the other's: the same player often decides several times in a row.
- Time: iterative deepening within belief_mc's budget per decision; an iteration
  that runs out of time is discarded (the deeper one only counts complete).
- Value (`evaluate`): win/loss, then LP, monsters, monsters that beat everything
  on the other side, cards in hand, set Spells/Traps, a nearly empty deck.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import math
import mmap
import os
import struct
import time

import numpy as np

from .. import constants as C
from ..actions import BATTLE_LISTS, IDLE_LISTS, encode
from ..carddb import default_db
from ..game import run_until_prompt
from ..observe import AUTO_RESPONSE, is_hidden_prompt
from ..state import LOCATIONS, QUERY_ALL, QUERY_BASIC, _parse_location_cached
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
TT_BITS = 16  # (transposition table entries: 2^16, in memory shared with the forked nodes)
_TT_DTYPE = np.dtype([("key", "<u8"), ("value", "<f8"), ("depth", "<i2"), ("flag", "i1"), ("move", "i1"),
                      ("gen", "<u2")])
_EXACT, _LOWER, _UPPER = 0, 1, 2
HIST_BITS = 14  # (history entries: 2^14, in memory shared with the forked nodes)
HIST_DECAY = 0.5  # (per decision)

# evaluation weights (points; 1 point = 100 LP)
W_LP = 1 / 100
W_MONSTER = 8.0
W_DOMINANT = 4.0  # (a face-up monster stronger than everything on the other side)
W_HAND = 3.0
W_BACKROW = 2.0
W_DECK_OUT = 20.0  # (two cards or fewer left to draw)

_MZONE_QUERY = C.QUERY_POSITION | C.QUERY_ATTACK | C.QUERY_DEFENSE

_IDLE_ORDER = {"activate": 0, "spsummon": 1, "summon": 2, "mset": 3, "repos": 4, "to_bp": 5, "sset": 6,
               "to_ep": 7, "shuffle": 8}
_BATTLE_ORDER = {"activate": 0, "attack": 1, "to_m2": 2, "to_ep": 3}


class _Abort(Exception):
    pass


_MADV_HUGEPAGE, _MADV_COLLAPSE = 14, 25
_HUGE = 1 << 21


def _huge_pages() -> None:
    """Backs this process's anonymous memory with huge pages where it can (Linux,
    best effort): a fork copies the page tables, one entry per 2 MB instead of per
    4 kB (the search forks at every node; a child's write splits only its copy)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        madvise = libc.madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        with open("/proc/self/maps") as f:
            maps = f.read().splitlines()
    except (OSError, AttributeError):
        return
    for line in maps:
        parts = line.split()
        if not parts[1].startswith("rw") or (len(parts) > 5 and parts[5] != "[heap]"):
            continue
        start, end = (int(x, 16) for x in parts[0].split("-"))
        start, end = (start + _HUGE - 1) & ~(_HUGE - 1), end & ~(_HUGE - 1)
        if end > start:
            madvise(start, end - start, _MADV_HUGEPAGE)
            madvise(start, end - start, _MADV_COLLAPSE)


_TYPE_ID = {"SELECT_IDLECMD": 1, "SELECT_BATTLECMD": 2, "SELECT_CHAIN": 3, "SELECT_EFFECTYN": 4,
            "SELECT_YESNO": 5, "SELECT_OPTION": 6}
_KIND_ID = {k: i for i, k in enumerate(("summon", "spsummon", "repos", "mset", "sset", "activate", "to_bp",
                                        "to_ep", "shuffle", "attack", "to_m2"), 1)}


def _move_key(prompt: dict, cand: dict) -> int:
    """The history slot of a move: its player, kind and card (and effect)."""
    t = prompt["type"]
    kind, code, desc = 0, 0, 0
    if "kind" in cand:
        kind = _KIND_ID.get(cand["kind"], 0)
        lst = (IDLE_LISTS if t == "SELECT_IDLECMD" else BATTLE_LISTS).get(cand["kind"])
        if lst is not None:
            entry = prompt[lst][cand["index"]]
            code, desc = entry["code"], entry.get("desc", 0) or 0
    elif t == "SELECT_CHAIN":
        if cand["index"] >= 0:
            entry = prompt["chains"][cand["index"]]
            kind, code, desc = 1, entry["code"], entry.get("desc", 0) or 0
    elif "yes" in cand:
        kind, code, desc = 1 + cand["yes"], prompt.get("code", 0) or 0, prompt.get("desc", 0) or 0
    elif "index" in cand:
        kind = 1 + cand["index"]
    h = (((_TYPE_ID.get(t, 0) * 31 + prompt["player"]) * 31 + kind) * 1000003 + code) * 1000003 + desc
    h ^= h >> 17
    h = (h * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    return h >> (64 - HIST_BITS)


def _battle_stat(c: dict) -> int:
    return c.get("defense", 0) if c.get("position", 0) & C.POS_DEFENSE else c.get("attack", 0)


def evaluate(duel, me: int) -> float:
    """The position's value for `me` (in points of 100 LP). (Only the queries it
    needs: a leaf's evaluation is a good part of its cost.)"""
    lp = duel.info()["lp"]
    mons = [[c for c in _parse_location_cached(duel.query_location_raw(p, C.LOCATION_MZONE, _MZONE_QUERY))
             if c is not None] for p in (me, 1 - me)]
    value = W_LP * (lp[me] - lp[1 - me]) + W_MONSTER * (len(mons[0]) - len(mons[1]))
    for s, p in ((0, me), (1, 1 - me)):
        sign = 1 if s == 0 else -1
        best_other = max((_battle_stat(c) for c in mons[1 - s]), default=-1)
        for c in mons[s]:
            if c.get("position", 0) & C.POS_FACEUP_ATTACK and c.get("attack", 0) > best_other:
                value += sign * W_DOMINANT
        value += sign * W_HAND * duel.query_count(p, C.LOCATION_HAND)
        value += sign * W_BACKROW * sum(
            1 for c in _parse_location_cached(duel.query_location_raw(p, C.LOCATION_SZONE, C.QUERY_POSITION))
            if c is not None and c.get("position", 0) & C.POS_FACEDOWN)
        if duel.query_count(p, C.LOCATION_DECK) <= 2:
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
        self._tt = None
        self._tt_gen = 0
        self._hist = None
        self._server = None  # (the search server, see warm_up)
        self.collapse_pages = True  # (huge pages before each search, see _huge_pages)
        self.server_failures = 0

    def _budget(self) -> float | None:
        return None if self.clock is None else self.clock[0] / MOVES_TO_GO + self.clock[1]

    def warm_up(self) -> None:
        super().warm_up()
        if os.environ.get("YGOAI_SEARCH_SERVER", "1") != "0":
            from ..core import journal_duels
            from .search_server import SearchClient
            journal_duels(True)  # (the server rebuilds the duels from their journals)
            self._server = SearchClient()

    def _select(self, duel, core_prompt: dict, core_cands: list[dict], t0: float, budget: float | None) -> int | None:
        if self._server is not None and duel.journal is not None:
            result = self._server.search(duel, self.player, self.rng.getrandbits(32), self.clock, core_prompt,
                                         core_cands, t0, budget)
            if result is not None:
                best, done, nodes = result
                self.nodes += nodes
                self.depths.append(done)
                return best
            self.server_failures += 1  # (then here, as without the server)
        self.prepare()
        best, done = self.search(duel, core_prompt, core_cands, t0, budget)
        self.depths.append(done)
        return best

    def prepare(self) -> None:
        """The tables for a new decision (in the process that keeps them)."""
        if self._tt is None:
            buf = mmap.mmap(-1, _TT_DTYPE.itemsize << TT_BITS)  # (anonymous: shared with the forks)
            self._tt = np.frombuffer(buf, dtype=_TT_DTYPE)
        self._tt_gen = (self._tt_gen + 1) % 65536 or 1  # (a new decision: older entries are stale)
        if self._hist is None:
            self._hist = np.frombuffer(mmap.mmap(-1, 8 << HIST_BITS), dtype=np.float64)  # (shared with the forks)
        self._hist *= HIST_DECAY

    def search(self, duel, core_prompt: dict, core_cands: list[dict], t0: float,
               budget: float | None) -> tuple[int, int]:
        """The best of `core_cands` and the depth completed (after `prepare`)."""
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
        # (the forked nodes never collect the inherited objects: no copies of their pages)
        gc.freeze()
        if self.collapse_pages:
            _huge_pages()
        try:
            return self._deepen(duel, core_prompt, core_cands, order, t0, budget, deadline)
        finally:
            gc.unfreeze()

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

    def _tt_key(self, duel, prompt: dict) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(repr(duel.info()).encode())
        h.update(duel.query_field_raw())
        for p in (0, 1):
            for _, loc in LOCATIONS:
                h.update(duel.query_location_raw(p, loc, QUERY_BASIC if loc == C.LOCATION_DECK else QUERY_ALL))
        h.update(repr(sorted(prompt.items())).encode())
        return int.from_bytes(h.digest(), "little") or 1

    def _node(self, duel, prompt: dict, depth: int, alpha: float, beta: float, deadline, ply: int) -> float:
        # (moves are indices into the heuristic's order: the TT keeps one, the history reorders them)
        cands = ordered(prompt, duel.info()["turn_player"])
        key = self._tt_key(duel, prompt)
        e = self._tt[key & ((1 << TT_BITS) - 1)]
        tt_move = None
        if e["key"] == key and e["gen"] == self._tt_gen:
            v = float(e["value"])
            if e["depth"] >= depth:
                if e["flag"] == _EXACT:
                    return v
                if e["flag"] == _LOWER:
                    alpha = max(alpha, v)
                else:
                    beta = min(beta, v)
                if alpha >= beta:
                    return v
            if 0 <= e["move"] < len(cands):
                tt_move = int(e["move"])
        alpha0, beta0 = alpha, beta
        hist = self._hist
        keys = [_move_key(prompt, c) for c in cands]
        # (the TT move first, then the moves with a better history; the heuristic's order among equals)
        idx = sorted(range(len(cands)), key=lambda k: (k != tt_move, -hist[keys[k]]))[:MAX_CHILDREN]
        best = -math.inf
        best_i = idx[0]
        for i, ci in enumerate(idx):
            c = cands[ci]
            if i == len(idx) - 1:
                # (the last move: nothing needs this position afterwards, no fork)
                if deadline is not None and time.monotonic() > deadline:
                    raise _Abort()
                self.nodes += 1
                v = self._expand(duel, prompt, c, depth - 1, alpha, beta, deadline, ply + 1)
            else:
                v = self._child(duel, prompt, c, depth - 1, alpha, beta, deadline, ply + 1)
            if v > best:
                best, best_i = v, ci
            alpha = max(alpha, best)
            if alpha >= beta:
                # (a cutoff: this move up, the ones tried before it down)
                bonus = depth * depth
                hist[keys[ci]] += bonus
                for k in idx[:i]:
                    hist[keys[k]] -= bonus / i
                break
        if abs(best) < WIN / 2:  # (not a win or loss: those depend on the ply)
            e = self._tt[key & ((1 << TT_BITS) - 1)]
            e["key"], e["value"], e["depth"], e["move"], e["gen"] = key, best, depth, best_i, self._tt_gen
            e["flag"] = _UPPER if best <= alpha0 else _LOWER if best >= beta0 else _EXACT
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
            value = -WIN  # (an engine error: the player to move avoids the line)
        return value

    def _expand(self, duel, prompt: dict, cand: dict, depth: int, alpha: float, beta: float, deadline,
                ply: int) -> float:
        """(in the child) Plays `cand`, runs to the next ply and values it, for the
        player who plays it (negamax: values and the window (alpha, beta) are the
        mover's; they change sign only where the next ply is the other player's,
        which in this game need not be the case)."""
        me = prompt["player"]
        duel.set_response(encode(prompt, cand))
        for _ in range(MAX_AUTO_STEPS):
            step = run_until_prompt(duel, prompt)
            for m in step.messages:
                if m["type"] == "WIN":
                    return WIN - ply if m["player"] == me else -WIN + ply
                if m["type"] == "NEW_TURN" or m["type"] == "HINT" and m["hint_type"] == C.HINT_SELECTMSG:
                    # (the rules' players: their activation caps per turn, what a card selection is for)
                    for policy in self._policies:
                        policy.observe(m)
            if step.ended:
                return evaluate(duel, me)
            prompt = step.prompt
            if is_hidden_prompt(prompt):
                resp = AUTO_RESPONSE
            elif prompt["type"] in SEARCHED and len(candidates(prompt)) > 1:
                if depth <= 0:
                    return evaluate(duel, me)
                if prompt["player"] == me:
                    return self._node(duel, prompt, depth, alpha, beta, deadline, ply)
                return -self._node(duel, prompt, depth, -beta, -alpha, deadline, ply)
            elif prompt["type"] in SEARCHED:
                resp = candidates(prompt)[0] if candidates(prompt) else \
                    self._policies[prompt["player"]].respond(prompt, None)
            else:
                resp = self._policies[prompt["player"]].respond(prompt, None)
            duel.set_response(encode(prompt, resp))
        return evaluate(duel, me)
