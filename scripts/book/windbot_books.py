"""Opening books of WindBot's decks (github.com/ProjectIgnis/windbot, cloned to WINDBOT): each
deck under its executor's name (the name ygo-windbot passes to WindBot), its cards from the
deck's .ydk. Decks with cards the engine does not load (external/BabelCDB's databases as
ygo-core reads them) are left out. Mirror: each deck against itself, --mirror openings each; pairs: --pairs openings of two
different decks drawn at random. Either seat order is a separate opening (the arbiter plays
both seats of each).

  .venv/bin/python scripts/book/windbot_books.py WINDBOT OUT_DIR [--mirror 20] [--pairs 2000] [--seed 1]

OUT_DIR/windbot_mirror.jsonl, OUT_DIR/windbot_pairs.jsonl.
"""

import argparse
import glob
import json
import os
import random
import re
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# (executors that are tests, not decks)
SKIP = {"Lucky", "Test"}


def executors(windbot):
    """{executor name: deck file name} of WindBot's decks."""
    out = {}
    for f in glob.glob(os.path.join(windbot, "Game/AI/Decks/*.cs")):
        for name, deck in re.findall(r'\[Deck\("([^"]*)",\s*"([^"]*)"', open(f, encoding="utf-8-sig").read()):
            if name not in SKIP:
                out[name] = deck
    return out


def ydk(path):
    main, extra, part = [], [], None
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if line.startswith("#main"):
            part = main
        elif line.startswith("#extra"):
            part = extra
        elif line.startswith("!side"):
            part = None
        elif line.isdigit() and part is not None:
            part.append(int(line))
    return main, extra


def known_codes():
    """The codes of the databases the engine loads (ygo-core's carddb.rs: DEFAULT_CDBS without
    EXCLUDED_CDB_PARTS: no Rush Duel, skill or GOAT cards)."""
    codes = set()
    for pattern in ["cards-unofficial.cdb", "prerelease-*.cdb", "release-*.cdb", "cards.cdb"]:
        for f in glob.glob(os.path.join(ROOT, "external/BabelCDB", pattern)):
            if not any(x in os.path.basename(f) for x in ("rush", "skills", "goat")):
                codes.update(r[0] for r in sqlite3.connect(f).execute("select id from datas"))
    return codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windbot")
    ap.add_argument("out")
    ap.add_argument("--mirror", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    known = known_codes()
    decks = {}
    for name, file in sorted(executors(a.windbot).items()):
        path = os.path.join(a.windbot, "Decks", file + ".ydk")
        if not os.path.exists(path):
            print(f"{name}: no {file}.ydk")
            continue
        main, extra = ydk(path)
        missing = sorted({c for c in main + extra if c not in known})
        if missing:
            print(f"{name}: left out, cards not in the database: {missing}")
            continue
        decks[name] = {"name": name, "main": main, "extra": extra}
    print(f"{len(decks)} decks: {', '.join(decks)}")
    rng = random.Random(a.seed)
    seed = lambda: rng.randrange(1 << 31)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "windbot_mirror.jsonl"), "w") as f:
        for d in decks.values():
            for _ in range(a.mirror):
                f.write(json.dumps({"seed": seed(), "decks": [d, d]}) + "\n")
    names = list(decks)
    with open(os.path.join(a.out, "windbot_pairs.jsonl"), "w") as f:
        for _ in range(a.pairs):
            x, y = rng.sample(names, 2)
            f.write(json.dumps({"seed": seed(), "decks": [decks[x], decks[y]]}) + "\n")


if __name__ == "__main__":
    main()
