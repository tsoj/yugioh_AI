#!/usr/bin/env python3
"""Which cards the test games actually exercise: plays the games of a seed range
(as check_belief.py would) and counts the cards that appeared in any message of
either player's observations, and the ones that were activated (CHAINING).

    python3 scripts/card_coverage.py -n 381 --start 7000 --book build/book.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.belief.check import load_book, make_game, observation_stream  # noqa: E402
from ygoai.carddb import default_db  # noqa: E402
from ygoai.parallel import run_parallel  # noqa: E402


def _codes(x, out: set) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "code" and isinstance(v, int) and v:
                out.add(v)
            else:
                _codes(v, out)
    elif isinstance(x, list):
        for v in x:
            _codes(v, out)


def play(task: tuple) -> tuple[set, set, set]:
    seed, pairs, decks, book = task
    rec = make_game(seed, pairs=[tuple(p) for p in pairs] if pairs else None, decks=decks, book=book)
    in_decks = {c for d in rec.decks for c in d.main + d.extra}
    seen: set[int] = set()
    activated: set[int] = set()
    for viewer in (0, 1):
        for it in observation_stream(rec, viewer):
            if it[0] == "msg":
                _codes(it[1], seen)
                if it[1].get("type") == "CHAINING":
                    activated.add(it[1]["code"])
    return in_decks, seen & in_decks, activated & in_decks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--pairs")
    ap.add_argument("--decks", choices=["random", "diverse"], default="random")
    ap.add_argument("--book")
    args = ap.parse_args()
    db = default_db()
    pairs = tuple(tuple(p.split("-")) for p in args.pairs.split(",")) if args.pairs else None
    tasks = [(s, pairs, args.decks, args.book) for s in range(args.start, args.start + args.n)]
    base = lambda codes: {db.get(c).alias or c for c in codes if db.get(c)}  # noqa: E731
    in_decks, seen, activated = set(), set(), set()
    for _, res in run_parallel(play, tasks, os.cpu_count()):
        if isinstance(res, tuple):
            in_decks |= base(res[0])
            seen |= base(res[1])
            activated |= base(res[2])
    total = in_decks
    if args.book:
        total = base(c for pair in load_book(args.book) for d in pair for c in d.main + d.extra)
    print(f"{args.n} games: {len(in_decks)} different cards in the decks played"
          + (f" (book: {len(total)})" if args.book else ""))
    print(f"  appeared in a game: {len(seen)} ({100 * len(seen) / max(1, len(total)):.1f}%)")
    print(f"  activated:          {len(activated)} ({100 * len(activated) / max(1, len(total)):.1f}%)")


if __name__ == "__main__":
    main()
