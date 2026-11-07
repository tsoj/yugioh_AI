"""Selects the SPRT book from tested candidates (bot-book-test's OUT): the openings in which each
seat won at least --min-wins test games, in their order, as book lines without the test results
(the book format: ygopro-core-rust/ygo-arbiter/README.md). Openings that one seat wins too often
measure the decks, not the bots. Each deck pair is in the book once: a pair (in either order, decks
compared by their cards) that is already in the book is not written again. With more than one
TESTED file, the book has the openings of the first file, then those of the next, and so on.

    python3 scripts/book/select_openings.py TESTED.jsonl [TESTED.jsonl ...] BOOK.jsonl [--min-wins 2]
"""

import argparse
import json


def pair_key(decks: list) -> tuple:
    # (the two decks by their cards, in either order)
    return tuple(sorted((tuple(sorted(d["main"])), tuple(sorted(d["extra"]))) for d in decks))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tested", nargs="+")
    ap.add_argument("book")
    ap.add_argument("--min-wins", type=int, default=2)
    args = ap.parse_args()
    n = kept = repeated = 0
    pairs = set()
    with open(args.book, "w") as out:
        for path in args.tested:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    n += 1
                    if min(row["wins"]) < args.min_wins:
                        continue
                    key = pair_key(row["decks"])
                    if key in pairs:
                        repeated += 1
                        continue
                    pairs.add(key)
                    out.write(json.dumps({"seed": row["seed"], "decks": row["decks"]}) + "\n")
                    kept += 1
    print(f"{kept} of {n} openings ({repeated} deck pairs already in the book)")


if __name__ == "__main__":
    main()
