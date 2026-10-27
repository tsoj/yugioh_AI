"""Builds the corpus completion's index (crates/ygoai/src/belief/completion.rs: CorpusCompletion,
the neural completion's source for what its models cannot supply): the human deck lists
(human_decks/decks/*.ydk) as numpy arrays in build/deck_index.npz, one row per (deck, card).

Only the "train" split is indexed, so that games played with "test" split decks cannot look up the
exact opponent deck list.

    python3 scripts/build_deck_index.py [--limit N]
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from common.carddb import default_db
from common.decks import canonical_code, deck_split, parse_ydk
from common.paths import DECKS_DIR, ROOT

INDEX_PATH = ROOT / "build" / "deck_index.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="index only the first N decks")
    args = ap.parse_args()
    db = default_db()
    paths = sorted(p for p in DECKS_DIR.glob("*.ydk") if deck_split(p) == "train")[:args.limit]
    deck_ids, codes, counts, extra = [], [], [], []
    n = 0
    for path in paths:
        deck = parse_ydk(path)
        count = Counter(canonical_code(db, c) for c in deck.main + deck.extra)
        if not count or any(c not in db for c in count):
            continue
        for code, k in count.items():
            deck_ids.append(n)
            codes.append(code)
            counts.append(min(k, 3))
            extra.append(db.get(code).is_extra_deck)
        n += 1
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDEX_PATH, deck=np.array(deck_ids, dtype=np.int32), code=np.array(codes, dtype=np.int64),
                        count=np.array(counts, dtype=np.int8), extra=np.array(extra, dtype=bool), n=np.array([n]))
    print(f"{INDEX_PATH}: {n} decks, {len(codes)} rows")


if __name__ == "__main__":
    main()
