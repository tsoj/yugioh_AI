#!/usr/bin/env python3
"""An opening book: random openings (deck pair, seed: the shuffles and the core's
seed) whose result is not settled by the decks alone.

  scripts/make_match_book.py -n 3000 -o runs/books/match1.jsonl [--games 20] [--seed 77]

Each candidate is played --games times by the policy agent (nn/policy.npz) on both
sides, sampling at temperature 1 with a different seed per game: the decks keep
their seats, only the players' choices vary. The book lists every candidate
with the number of games each seat won ("wins"); match.py --book uses those where
the less winning seat won at least --book-min games (in a match each opening is
played twice with the engines' seats swapped: if one deck always wins, the pair
says nothing about the engines).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.match import random_openings  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402

MAX_DECISIONS = 3000
GAME_SECONDS = 120


def test_game(task):
    """(candidate index, opening, game) -> (candidate index, winning seat or None)."""
    i, opening, g = task
    import signal
    from ygoai.agents.policy_agent import PolicyAgent
    # (a game that hangs in the core cannot be interrupted from Python: the alarm ends the
    # worker, and the game counts as failed)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(GAME_SECONDS)
    from ygoai.game import new_record, play
    record = new_record(opening.decks, opening.seed)
    agents = [PolicyAgent(opening.seed * 1000 + 2 * g + p, temperature=1.0) for p in (0, 1)]
    play(record, agents, max_turns=100, max_decisions=MAX_DECISIONS)
    signal.alarm(0)
    w = record.result.get("winner")
    return i, w if w in (0, 1) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--candidates", type=int, default=3000)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--games", type=int, default=20, help="test games per candidate")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("-j", "--jobs", type=int, default=30)
    args = ap.parse_args()
    openings = random_openings(args.candidates, args.seed, 0.0)
    tasks = [(i, op, g) for i, op in enumerate(openings) for g in range(args.games)]
    wins = [[0, 0] for _ in openings]
    failed = 0
    for done, (_, res) in enumerate(run_parallel(test_game, tasks, args.jobs), 1):
        if isinstance(res, Crash):
            failed += 1
        else:
            i, w = res
            if w is not None:
                wins[i][w] += 1
        if done % 5000 == 0:
            print(f"{done}/{len(tasks)} games", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for op, w in zip(openings, wins):
            f.write(json.dumps({"seed": op.seed, "decks": [d.to_json() for d in op.decks], "wins": w}) + "\n")
    minority = [min(w) for w in wins]
    print(f"{len(openings)} candidates, {args.games} games each ({failed} failed) -> {args.output}")
    print("  openings by the less winning seat's wins:",
          {k: minority.count(k) for k in range(args.games // 2 + 1) if minority.count(k)})
    for k in (1, 2, 3, 5):
        kept = sum(m >= k for m in minority)
        print(f"  --book-min {k}: {kept} kept, {len(openings) - kept} skipped ({1 - kept / len(openings):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
