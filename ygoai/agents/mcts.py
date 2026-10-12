"""Plain Monte Carlo tree search (UCT) with random rollouts on the belief's reconstruction: no
value function, no policy (the search: crates/ygoai/src/mcts.rs).

The tree's decisions are alpha-beta's plies; the prompts in between are answered by the rules'
players (the same in every simulation, so a path of moves always leads to the same position); a
rollout plays on with uniformly random answers until the game ends (a win 1, a loss 0, a game
still going after 30 more turns 1/2). The move: the root's most visited one.

YGOAI_MCTS_SIMS: simulations per decision (else the clock's budget as alphabeta's, without a clock
200); YGOAI_MCTS_C: UCB1's exploration constant (default sqrt(2)); YGOAI_MCTS_ROLLOUT_TURNS: new turns
a rollout plays at most (default 30); YGOAI_MCTS_EVAL=1: a rollout cut off there counts the
handcrafted evaluation as a win probability instead of 1/2.
"""

from __future__ import annotations

import os
import time

from .alphabeta import AlphaBetaAgent


class MctsAgent(AlphaBetaAgent):
    name = "mcts"

    def __init__(self, seed: int | None = None, **kw):
        super().__init__(seed=seed, **kw)
        self.max_sims = int(os.environ.get("YGOAI_MCTS_SIMS") or 0) or None
        self.c = float(os.environ["YGOAI_MCTS_C"]) if os.environ.get("YGOAI_MCTS_C") else None
        self.rollout_turns = int(os.environ["YGOAI_MCTS_ROLLOUT_TURNS"]) if os.environ.get("YGOAI_MCTS_ROLLOUT_TURNS") else None
        self.cutoff_eval = os.environ.get("YGOAI_MCTS_EVAL", "0") == "1"
        self.stats: list[dict] = []  # (per search: crates/ygoai/src/mcts.rs MctsStats)

    def _select(self, duel, core_prompt: dict, core_cands: list[dict], t0: float, budget: float | None) -> int | None:
        import ygoai_rs
        hint = self.fallback.hint  # (the select hint: the search answers this selection, not the fallback)
        self.fallback.hint = None
        seeds = [self.rng.getrandbits(32) for _ in range(3)]
        if self.max_sims is not None:
            budget = None
        best, stats = ygoai_rs.mcts_search(duel._d, core_prompt, core_cands, seeds, hint, budget,
                                           time.monotonic() - t0, self.max_sims, self.c, self.rollout_turns,
                                           self.cutoff_eval)
        self.nodes += stats["sims"]
        self.depths.append(stats["leaf_depth_max"])
        self.stats.append(stats)
        return best
