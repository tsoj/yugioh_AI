#!/usr/bin/env python3
"""Builds an opening book of deck pairs that together contain every card of the
deck corpus (a greedy set cover: the first round of the diverse deck order, see
common.decks.diverse_order). The decks are stored inline, so a book stays the
same when the corpus changes.

    python3 scripts/make_book.py -o build/book.jsonl
    python3 scripts/check_belief.py -n 1000 --start 7000 --book build/book.jsonl -o runs/b.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common.carddb import default_db
from common.decks import DeckPool, diverse_order
from common.paths import ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=ROOT / "build" / "book.jsonl")
    ap.add_argument("--seed", type=int, default=0, help="pairing of the decks")
    args = ap.parse_args()
    db = default_db()
    pool = DeckPool(db)
    by_name = {p.name: p for p in pool.paths}
    covered: set[int] = set()
    cover = []
    for name in diverse_order(pool, ROOT / "build" / "diverse_decks.json"):
        deck = pool.load(by_name[name])
        cards = {db.original(c) for c in deck.main + deck.extra}
        if not cards - covered:
            break  # (the first round of the order is the cover)
        covered |= cards
        cover.append(deck)
    rng = random.Random(args.seed)
    rng.shuffle(cover)
    if len(cover) % 2:
        cover.append(cover[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for i in range(0, len(cover), 2):
            out.write(json.dumps({"id": i // 2, "decks": [cover[i].to_json(), cover[i + 1].to_json()]}) + "\n")
    print(f"{len(cover) // 2} openings, {len(cover)} decks, {len(covered)} different cards -> {args.output}")


if __name__ == "__main__":
    main()
