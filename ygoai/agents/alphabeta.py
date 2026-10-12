"""Alpha-beta search with iterative deepening on the reconstructed belief duel.

The duel comes from the belief replay as for belief_mc (same reconstruction, time budget and
fallbacks); only the choice differs. The search itself runs in Rust (crates/ygoai/src/search.rs,
ygoai_rs.AlphaBetaSearch), on copies of the duel:

- Plies are the decisions of either player at command prompts, chain windows,
  yes/no questions and options (belief_mc.SEARCHED) with more than one candidate;
  everything in between (cards, places, positions, ...) is answered by the
  heuristic player's rules (which see the core's select hints and new turns), the
  opponent's hidden cards are the belief's guesses.
- Card selections of one card are plies too (the rules' first choices).
- Move ordering: the heuristic player's priorities (`ordered` here), a transposition table
  (positions reached by other move orders; the best move first) and a history of moves that
  caused cutoffs (halved per decision).
- Negamax: a node's value is for the player to move there; it changes sign only from one
  player's decision to the other's: the same player often decides several times in a row.
- Time: iterative deepening within belief_mc's budget per decision; an iteration that runs out of
  time is discarded.
- Value (`evaluate`, the same in Rust): win/loss, then LP, monsters, monsters that beat everything
  on the other side, cards in hand, set Spells/Traps, a nearly empty deck.

`evaluate`, `ordered` and the rules' card order are also what PuctAgent's search uses.
"""

from __future__ import annotations

import os
import time

from .. import constants as C
from ..carddb import default_db
from ..state import _parse_location_cached
from .belief_mc import BeliefMCAgent, candidates
from .heuristic_agents import HeuristicAgent

# time: a soft bound per searched decision (remaining / MOVES_TO_GO + increment); the Rust search
# applies the same bounds as before (crates/ygoai/src/search.rs)
MOVES_TO_GO = 20
MAX_AUTO_STEPS = 300  # (prompts answered by the rules between two plies)

# evaluation weights (points; 1 point = 100 LP)
# (the weights: logistic regression of the result on these terms (Texel tuning), on the
# positions of 11k alphabeta games (the SPRT records), with the LP weight fixed)
W_LP = 1 / 100
W_MONSTER = 4.3
W_DOMINANT = 17.6  # (a face-up monster stronger than everything on the other side)
W_HAND = 9.1
W_BACKROW = 5.1
W_DECK_OUT = 20.0  # (two cards or fewer left to draw)

_MZONE_QUERY = C.QUERY_POSITION | C.QUERY_ATTACK | C.QUERY_DEFENSE

_IDLE_ORDER = {"activate": 0, "spsummon": 1, "summon": 2, "mset": 3, "repos": 4, "to_bp": 5, "sset": 6,
               "to_ep": 7, "shuffle": 8}
_BATTLE_ORDER = {"activate": 0, "attack": 1, "to_m2": 2, "to_ep": 3}


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


def ordered(prompt: dict, turn_player: int, card_order: list[int] | None = None) -> list[dict]:
    """The candidates of a prompt, the heuristic player's preference first (for a
    card selection: `card_order`, the rules' order by the select hint)."""
    cands = candidates(prompt)
    t = prompt["type"]
    if t == "SELECT_CARD" and card_order is not None:
        return [cands[i] for i in card_order if i < len(cands)]
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
        self._rs = None  # (the Rust search: its tables live across decisions)
        # (a fixed number of nodes per search instead of the clock's budget: YGOAI_SEARCH_NODES)
        self.max_nodes = int(os.environ.get("YGOAI_SEARCH_NODES") or 0) or None

    def _budget(self) -> float | None:
        return None if self.clock is None else self.clock[0] / MOVES_TO_GO + self.clock[1]

    def _select(self, duel, core_prompt: dict, core_cands: list[dict], t0: float, budget: float | None) -> int | None:
        hint = self.fallback.hint  # (the select hint: the search answers this selection, not the fallback)
        self.fallback.hint = None
        if self._rs is None:
            import ygoai_rs
            self._rs = ygoai_rs.AlphaBetaSearch()
        self._rs.prepare()
        seeds = [self.rng.getrandbits(32), self.rng.getrandbits(32)]  # (the rules' players)
        if self.max_nodes is not None:
            budget = None
        best, done, nodes = self._rs.search(duel._d, core_prompt, core_cands, seeds, hint, budget,
                                            time.monotonic() - t0, self.clock[0] if self.clock else 0.0,
                                            self.max_nodes)
        self.nodes += nodes
        self.depths.append(done)
        return best

    def _card_order(self, prompt: dict) -> list[int] | None:
        """For a card selection: the rules' order of its cards (by the select hint the
        player got, which belongs to this selection only)."""
        if prompt["type"] != "SELECT_CARD":
            return None
        policy = self._policies[prompt["player"]]
        order = policy._card_order(prompt["cards"])
        policy.hint = None
        return order
