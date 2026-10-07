#!/usr/bin/env python3
"""Count how many decks each card appears in (ignoring multiple copies in the same deck).

Outputs:
  card_deck_frequency.tsv  — card_id, name, deck_count, sorted by deck_count desc
  card_deck_frequency.png  — histogram of deck counts
"""

import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from deck_common import parse_ydk_deck

HERE = Path(__file__).resolve().parent
DECKS_DIR = HERE / "decks"
CDB_PATH = HERE.parent / "external" / "BabelCDB" / "cards.cdb"
OUT_TSV = HERE / "card_deck_frequency.tsv"
OUT_PNG = HERE / "card_deck_frequency.png"


def main():
    paths = sorted(DECKS_DIR.glob("*.ydk"))
    print(f"scanning {len(paths)} decks ...")

    deck_count = Counter()
    for p in paths:
        cards = parse_ydk_deck(p)
        for cid in set(cards):
            deck_count[cid] += 1

    print(f"found {len(deck_count)} unique cards")

    db = sqlite3.connect(f"file:{CDB_PATH}?mode=ro", uri=True)
    name_of = {}
    for cid in deck_count:
        row = db.execute("SELECT name FROM texts WHERE id = ?", (cid,)).fetchone()
        name_of[cid] = row[0] if row else ""
    db.close()

    rows = sorted(deck_count.items(), key=lambda x: -x[1])
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("card_id\tname\tdeck_count\n")
        for cid, cnt in rows:
            f.write(f"{cid}\t{name_of[cid]}\t{cnt}\n")
    print(f"wrote {OUT_TSV}")

    counts = list(deck_count.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Card deck-appearance frequency")

    axes[0].hist(counts, bins=60, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Number of decks")
    axes[0].set_ylabel("Number of cards")
    axes[0].set_title("Linear scale")

    axes[1].hist(counts, bins=60, color="steelblue", edgecolor="none", log=True)
    axes[1].set_xlabel("Number of decks")
    axes[1].set_ylabel("Number of cards (log)")
    axes[1].set_title("Log-Y scale")

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")

    n = len(counts)
    total_decks = len(paths)
    print(f"\ntop 10 cards by deck count:")
    for cid, cnt in rows[:10]:
        print(f"  {cnt:5d}/{total_decks}  {name_of[cid]}")
    print(f"\nmedian deck count: {sorted(counts)[n // 2]}")
    print(f"mean deck count:   {sum(counts) / n:.1f}")


if __name__ == "__main__":
    main()
