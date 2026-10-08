#!/usr/bin/env python3
"""Self-play games for training the policy and value networks.

  scripts/gen_selfplay.py -n 20000 -o runs/data/boot1.jsonl [--agent heuristic] [--temperature 1] [--floor 0.05]

Random pairs of corpus decks (one game each: the seats do not need to be
swapped for training data), both players the same agent. At the decisions the
search covers (belief_mc.SEARCHED, more than one candidate), the agent's choice
is the policy target ("targets": {response index: response}); the move played
is drawn from a probabilistic version of the agent: the candidates ranked (the
agent's choice first, then the rules' priorities, alphabeta.ordered), rank r
with probability ~ exp(-r / temperature) (1: of 5 candidates the choice ~64%,
the next ~23%), mixed with a uniform `floor` (every candidate at least floor/n):
more varied positions for the value, mostly plausible moves. Each line is a game
record (GameRecord.to_json) with "targets" added; the result is the value's
label.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.agents import make_agent  # noqa: E402
from ygoai.agents.alphabeta import ordered  # noqa: E402
from ygoai.agents.base import Agent  # noqa: E402
from ygoai.agents.belief_mc import SEARCHED, candidates  # noqa: E402
from ygoai.game import new_record, play  # noqa: E402
from ygoai.match import random_openings  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402


class Exploring(Agent):
    """`inner`'s choices, recorded as targets; the move played drawn by rank."""

    def __init__(self, inner: Agent, rng: random.Random, temperature: float, floor: float, record, targets: dict):
        self.inner, self.rng, self.temperature, self.floor = inner, rng, temperature, floor
        self.record, self.targets = record, targets
        self.name = inner.name
        self.wants_state = inner.wants_state

    def start(self, player: int, init: dict) -> None:
        super().start(player, init)
        self.inner.start(player, init)

    def observe(self, msg: dict) -> None:
        self.inner.observe(msg)

    def respond(self, prompt: dict, state: dict | None) -> dict:
        cands = candidates(prompt) if prompt["type"] in SEARCHED else []
        card_order = None
        if len(cands) > 1 and prompt["type"] == "SELECT_CARD":
            card_order = self.inner._card_order(prompt["cards"])  # (before respond: it uses the hint)
        target = self.inner.respond(prompt, state)
        if len(cands) < 2:
            return target
        self.targets[len(self.record.responses)] = target
        turn_player = state["turn_player"] if state is not None else prompt["player"]
        ranked = [target] + [c for c in ordered(prompt, turn_player, card_order) if c != target]
        ranked += [c for c in cands if c not in ranked]
        weights = [math.exp(-r / self.temperature) for r in range(len(ranked))]
        total = sum(weights)
        weights = [(1 - self.floor) * w / total + self.floor / len(ranked) for w in weights]
        return self.rng.choices(ranked, weights)[0]

    def end(self, result: dict) -> None:
        self.inner.end(result)


def game(task) -> dict:
    opening, agent, temperature, floor, max_turns = task
    record = new_record(opening.decks, opening.seed)
    rng = random.Random(opening.seed)
    targets: dict[int, dict] = {}
    agents = [Exploring(make_agent(agent, opening.seed * 2 + p), rng, temperature, floor, record, targets)
              for p in (0, 1)]
    play(record, agents, max_turns=max_turns)
    out = record.to_json()
    out["targets"] = {str(k): v for k, v in targets.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--games", type=int, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--agent", default="heuristic")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("-j", "--jobs", type=int, default=24)
    args = ap.parse_args()
    openings = random_openings(args.games, args.seed, 0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0, done, failed, decided = time.time(), 0, 0, 0
    with args.output.open("w") as f:
        tasks = [(op, args.agent, args.temperature, args.floor, args.max_turns) for op in openings]
        for _, res in run_parallel(game, tasks, args.jobs):
            if isinstance(res, Crash):
                failed += 1
                continue
            f.write(json.dumps(res, separators=(",", ":")) + "\n")
            done += 1
            decided += res["result"].get("winner") in (0, 1)
            if done % 500 == 0:
                print(f"[{time.time() - t0:7.0f}s] {done} games ({decided} decided, {failed} failed)", flush=True)
    print(f"{done} games ({decided} decided, {failed} failed) in {time.time() - t0:.0f}s -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
