"""Openings of the SPRT records with how often the seat decided them: in a match each opening is
played twice with the bots' seats swapped, and if the same seat won both games, the decks (and
the shuffles) decided it, not the bots. Per opening (its seed): both decks, each main deck in
draw order ("order", the top card first; "top": its first TOP cards, the opening hands and first
draws), the pairs played over all records, how many the seat decided, and how many seat 0 won
both.

    python3 scripts/book/build_onesided_data.py OUT.jsonl [RECORDS.jsonl ...]   (default: runs/sprt/*.jsonl)
"""

import glob
import json
import re
import sys
from collections import defaultdict

TOP = 8
SKIP = re.compile(r"results|filtertest|ladder")


def main():
    out = sys.argv[1]
    paths = sys.argv[2:] or sorted(p for p in glob.glob("runs/sprt/*.jsonl") if not SKIP.search(p))
    openings = {}
    counts = defaultdict(lambda: [0, 0, 0])
    for path in paths:
        games = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            games[r["seed"]].append(r)
        for seed, rs in games.items():
            if len(rs) != 2:
                continue
            w = [r.get("result", {}).get("winner") for r in rs]
            c = counts[seed]
            c[0] += 1
            if w[0] == w[1] and w[0] in (0, 1):
                c[1] += 1
                c[2] += w[0] == 0
            if seed not in openings:
                r = rs[0]
                # (deck_orders: bottom card first; the top cards are drawn first)
                openings[seed] = {
                    "seed": seed,
                    "decks": [{"main": d["main"], "extra": d["extra"]} for d in r["decks"]],
                    "top": [list(reversed(o[-TOP:])) for o in r["deck_orders"]],
                    "order": [list(reversed(o)) for o in r["deck_orders"]],
                }
    with open(out, "w") as f:
        for seed, o in openings.items():
            n, seat, seat0 = counts[seed]
            f.write(json.dumps({**o, "pairs": n, "seat_decided": seat, "seat0_both": seat0}) + "\n")
    print(f"{len(openings)} openings, {sum(c[0] for c in counts.values())} pairs from {len(paths)} files -> {out}")


if __name__ == "__main__":
    main()
