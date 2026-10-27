"""Selects the SPRT book from tested candidates (bot-book-test's OUT): the openings in which each
seat won at least --min-wins test games, in their order, as book lines without the test results
(the book format: ygopro-core-rust/ygo-arbiter/README.md). Openings that one seat wins too often
measure the decks, not the bots.

    python3 scripts/book/select_openings.py TESTED.jsonl BOOK.jsonl [--min-wins 2]
"""

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tested")
    ap.add_argument("book")
    ap.add_argument("--min-wins", type=int, default=2)
    args = ap.parse_args()
    n = kept = 0
    with open(args.tested) as f, open(args.book, "w") as out:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            if min(row["wins"]) >= args.min_wins:
                out.write(json.dumps({"seed": row["seed"], "decks": row["decks"]}) + "\n")
                kept += 1
    print(f"{kept} of {n} openings")


if __name__ == "__main__":
    main()
