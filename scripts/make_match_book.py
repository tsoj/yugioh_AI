#!/usr/bin/env python3
"""An opening book: random openings (deck pair, seed: the shuffles and the core's
seed) whose result is not settled by the decks alone.

  scripts/make_match_book.py -n 3000 -o runs/books/match1.jsonl [--games 20] [--seed 77]

Each candidate is played up to --games times by policy agents (--policies: each
game a random one of them for both seats, default nn/policy.npz), sampling at
temperature 1 with a different seed per game: the
decks keep their seats, only the players' choices vary. Its testing stops early
once each seat won --stop-min games (enough for --book-min up to that). The book
lists every candidate with the number of games each seat won ("wins", over
"games" played); match.py --book uses those where
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


def test_opening(task):
    """(candidate index, opening, games, stop_min, policies) -> (candidate index, wins per seat, games played)."""
    i, opening, games, stop_min, policies = task
    import random
    import signal
    from ygoai.agents.policy_agent import PolicyAgent, load_policy
    from ygoai.game import new_record, play
    wins = [0, 0]
    for g in range(games):
        if min(wins) >= stop_min:
            return i, wins, g
        # (a game that hangs in the core cannot be interrupted from Python: the alarm ends
        # the worker, and the candidate counts as failed)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(GAME_SECONDS)
        record = new_record(opening.decks, opening.seed)
        model = load_policy(random.Random(opening.seed * 1000 + g).choice(policies))  # (both seats)
        agents = [PolicyAgent(opening.seed * 1000 + 2 * g + p, model=model, temperature=1.0) for p in (0, 1)]
        play(record, agents, max_turns=100, max_decisions=MAX_DECISIONS)
        signal.alarm(0)
        w = record.result.get("winner")
        if w in (0, 1):
            wins[w] += 1
    return i, wins, games


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--candidates", type=int, default=3000)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--games", type=int, default=20, help="test games per candidate (at most)")
    ap.add_argument("--stop-min", type=int, default=2, help="stop once each seat won this many")
    ap.add_argument("--policies", default=None, help="comma-separated policy exports (default: nn/policy.npz)")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("-j", "--jobs", type=int, default=30)
    args = ap.parse_args()
    openings = random_openings(args.candidates, args.seed, 0.0)
    from ygoai.agents.policy_agent import DEFAULT_MODEL
    policies = [str(Path(p).resolve()) for p in args.policies.split(",")] if args.policies else [str(DEFAULT_MODEL)]
    tasks = [(i, op, args.games, args.stop_min, policies) for i, op in enumerate(openings)]
    wins: list = [None] * len(openings)
    played = [0] * len(openings)
    failed = 0
    for done, (_, res) in enumerate(run_parallel(test_opening, tasks, args.jobs), 1):
        if isinstance(res, Crash):
            failed += 1
        else:
            i, wins[i], played[i] = res
        if done % 500 == 0:
            print(f"{done}/{len(tasks)} candidates, {sum(played)} games", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for op, w, n in zip(openings, wins, played):
            if w is not None:
                f.write(json.dumps({"seed": op.seed, "decks": [d.to_json() for d in op.decks], "wins": w,
                                    "games": n}) + "\n")
    tested = [w for w in wins if w is not None]
    minority = [min(w) for w in tested]
    print(f"{len(openings)} candidates ({failed} failed), {sum(played)} games (at most {args.games} each, "
          f"stopped once each seat won {args.stop_min}) -> {args.output}")
    print("  openings by the less winning seat's wins:",
          {k: minority.count(k) for k in range(args.games // 2 + 1) if minority.count(k)})
    for k in range(1, args.stop_min + 1):
        kept = sum(m >= k for m in minority)
        print(f"  --book-min {k}: {kept} kept, {len(tested) - kept} skipped ({1 - kept / max(len(tested), 1):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
