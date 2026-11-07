"""Candidate openings of one era each: random deck pairs of the corpus whose eras (a deck's newest
card's first release, card_dates.py) are at most --window days apart, for the policy games
(bot-book-test). The first deck's era year is drawn with weight sqrt(the year's decks), then a
deck of that year; the second deck uniformly among the decks of its window. Every deck (by its
cards: copies under other names too) is in one part, "sprt" (--sprt-share of the decks) or
"data", and a pair's decks are of the same part: the SPRT book's decks are never in the training
data's games.

    .venv/bin/python scripts/book/era_candidates.py OUT.jsonl --part sprt|data -n N [--seed S]
        [--window DAYS] [--sprt-share F]

A row: {"seed", "decks", "eras": the two decks' eras}.
"""

import argparse
import bisect
import collections
import datetime
import json
import math
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # (scripts/: common)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from card_dates import dates  # noqa: E402
from common.carddb import default_db  # noqa: E402
from common.decks import DeckPool  # noqa: E402


def deck_part(deck, sprt_share: float) -> str:
    key = ",".join(map(str, sorted(deck.main))) + ";" + ",".join(map(str, sorted(deck.extra)))
    return "sprt" if zlib.crc32(key.encode()) % 1000 < sprt_share * 1000 else "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--part", choices=["sprt", "data"], required=True)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0, help="(default: 92 for sprt, 93 for data)")
    ap.add_argument("--window", type=int, default=365)
    ap.add_argument("--sprt-share", type=float, default=0.16)
    a = ap.parse_args()
    seed = a.seed or {"sprt": 92, "data": 93}[a.part]

    db = default_db()
    released = dates()
    pool = DeckPool(db)
    decks, seen, undated = [], set(), 0
    for p in pool.paths:
        deck = pool.load(p)
        if deck is None or deck_part(deck, a.sprt_share) != a.part:
            continue
        key = (tuple(sorted(deck.main)), tuple(sorted(deck.extra)))
        if key in seen:
            continue
        seen.add(key)
        era = [released.get(c) or released.get(db.original(c)) for c in deck.main + deck.extra]
        if None in era:
            undated += 1
            continue
        decks.append((max(era), deck))
    decks.sort(key=lambda x: x[0])
    days = [datetime.date.fromisoformat(e).toordinal() for e, _ in decks]
    by_year = collections.defaultdict(list)
    for i, (e, _) in enumerate(decks):
        by_year[e[:4]].append(i)
    years = sorted(by_year)
    weights = [math.sqrt(len(by_year[y])) for y in years]
    print(f"part {a.part}: {len(decks)} distinct dated decks ({undated} with an undated card); years "
          + " ".join(f"{y}:{len(by_year[y])}" for y in years), flush=True)

    rng = random.Random(seed)
    with open(a.out, "w") as f:
        for i in range(a.n):
            first = rng.choice(by_year[rng.choices(years, weights)[0]])
            lo = bisect.bisect_left(days, days[first] - a.window)
            hi = bisect.bisect_right(days, days[first] + a.window)
            second = rng.randrange(lo, hi)
            pair = [decks[first], decks[second]]
            if rng.random() < 0.5:
                pair.reverse()
            f.write(json.dumps({"seed": seed * 1_000_003 + i, "decks": [d.to_json() for _, d in pair], "eras": [e for e, _ in pair]}) + "\n")
    print(f"{a.n} candidates -> {a.out}")


if __name__ == "__main__":
    main()
