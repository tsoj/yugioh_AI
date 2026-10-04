#!/usr/bin/env python3
"""The belief regression suite: every range of games ever checked, with the
exact settings it was played with (other settings play other games).

Each range is checked against its latest baseline (runs/suite/baselines.json):

    python3 scripts/suite.py TAG                  # all ranges
    python3 scripts/suite.py TAG reg1000 new50000 # some of them
    python3 scripts/suite.py --list

A range's run becomes its new baseline when nothing that passed there before
broke (or with --accept). A run that did not finish leaves the baseline alone.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE_DIR = ROOT / "runs" / "suite"
BASELINES = SUITE_DIR / "baselines.json"
P = "setter-greedy,mixed-setter,greedy-mixed,setter-random,mixed-mixed,random-setter"
MC = "belief_mc-greedy,setter-belief_mc,belief_mc-mixed,random-belief_mc,belief_mc-belief_mc"

# name: (games, first seed, other flags)
RANGES: dict[str, tuple[int, int, list[str]]] = {
    "reg1000": (1000, 1000, []),
    "reg2000": (1000, 2000, []),
    "reg3000": (1000, 3000, []),
    "reg4000": (1000, 4000, ["--pairs", P]),
    "reg5000": (1000, 5000, ["--decks", "diverse"]),
    "reg6000": (1000, 6000, ["--decks", "diverse", "--pairs", P]),
    "reg7000": (1000, 7000, ["--decks", "diverse", "--pairs", P]),
    "book9000": (1000, 9000, ["--book", "build/book.jsonl"]),
    "fresh10000": (1000, 10000, []),
    "fresh11000": (1000, 11000, ["--pairs", P]),
    "fresh12000": (1000, 12000, ["--decks", "diverse"]),
    "fresh13000": (1000, 13000, ["--decks", "diverse", "--pairs", P]),
    # (games of the belief_mc agent take long to play: in parts)
    "mc14000": (100, 14000, ["--decks", "diverse", "--pairs", MC]),
    "mc14100": (100, 14100, ["--decks", "diverse", "--pairs", MC]),
    "mc14200": (100, 14200, ["--decks", "diverse", "--pairs", MC]),
    "fresh15000": (1000, 15000, ["--decks", "diverse", "--pairs", P]),
    "fresh16000": (1000, 16000, ["--decks", "cards", "--pairs", P]),
    "fresh17000": (1000, 17000, ["--decks", "diverse", "--pairs", P]),
    "fresh18000": (1000, 18000, ["--decks", "cards", "--pairs", P]),
    "book20000": (762, 20000, ["--book", "build/book.jsonl"]),
    "book22000": (1000, 22000, ["--book", "build/book.jsonl"]),
    "cards30000": (1000, 30000, ["--decks", "cards"]),
    "cards31000": (1000, 31000, ["--decks", "cards", "--pairs", P]),
    "cards40000": (1000, 40000, ["--decks", "cards"]),
    "cards41000": (1000, 41000, ["--decks", "cards", "--pairs", P]),
    "new50000": (1000, 50000, ["--decks", "new"]),
    "new51000": (1000, 51000, ["--decks", "new", "--pairs", P]),
    "new52000": (1000, 52000, ["--decks", "new"]),
    "new53000": (1000, 53000, ["--decks", "new", "--pairs", P]),
    "new54000": (500, 54000, ["--decks", "new"]),  # (in two parts: slow)
    "new54500": (500, 54500, ["--decks", "new"]),
    "nc55000": (1000, 55000, ["--decks", "new-cards", "--pairs", P]),
    "nm56000": (1000, 56000, ["--decks", "new-mixed"]),
    "new62000": (400, 62000, ["--decks", "new"]),
    "new63000": (400, 63000, ["--decks", "new", "--pairs", P]),
    "mixed64000": (400, 64000, ["--decks", "new-mixed", "--pairs", P]),
    "nc65000": (400, 65000, ["--decks", "new-cards"]),
    # with the realistic (effect card) deck completion
    "corpus1000": (200, 1000, ["--completion", "corpus"]),
    "corpus52000": (300, 52000, ["--decks", "new", "--completion", "corpus"]),
    "corp58000": (400, 58000, ["--decks", "new", "--completion", "corpus"]),
    "corp59000": (400, 59000, ["--decks", "new-mixed", "--completion", "corpus"]),
    "corp60000": (400, 60000, ["--decks", "new", "--pairs", P, "--completion", "corpus"]),
    "corp61000": (400, 61000, ["--decks", "new-cards", "--pairs", P, "--completion", "corpus"]),
    "corp66000": (400, 66000, ["--decks", "diverse", "--pairs", P, "--completion", "corpus"]),
}


def load_baselines() -> dict[str, str]:
    try:
        return json.loads(BASELINES.read_text())
    except (OSError, ValueError):
        return {}


def compare(old: Path, new: Path) -> tuple[str, int]:
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), str(old), str(new)],
                         capture_output=True, text=True).stdout
    m = re.search(r"\((\d+) fixed, (\d+) broke, (\d+) still failing\)", out)
    broke = int(m.group(2)) if m else -1
    return out.strip(), broke


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", help="suffix of this run's outputs")
    ap.add_argument("ranges", nargs="*", help="only these ranges (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--accept", action="store_true", help="make the runs baselines even if something broke")
    args = ap.parse_args()
    base = load_baselines()
    if args.list or not args.tag:
        for name, (n, start, flags) in RANGES.items():
            print(f"{name:12s} -n {n} --start {start} {' '.join(flags):70s} {base.get(name, '(no baseline)')}")
        return 0
    unknown = [r for r in args.ranges if r not in RANGES]
    if unknown:
        sys.exit(f"unknown ranges: {unknown}")
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in args.ranges or list(RANGES):
        n, start, flags = RANGES[name]
        out = SUITE_DIR / f"{name}.{args.tag}.jsonl"
        cmd = [sys.executable, str(ROOT / "scripts" / "check_belief.py"), "-n", str(n), "--start", str(start),
               *flags, "-o", str(out)]
        old = base.get(name)
        if old:
            cmd += ["--previous", old]
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        summary = next((l for l in r.stdout.splitlines() if "reconstructions consistent" in l), "?")
        if r.returncode not in (0, 1) or summary == "?":  # (1: some reconstructions failed)
            print(f"{name}: run failed (exit {r.returncode})\n{r.stderr[-2000:]}", flush=True)
            failed = True
            continue
        line = f"{name}: {summary.strip()} [{time.time() - t:.0f}s]"
        if old:
            text, broke = compare(Path(old), out)
            line += "\n    " + text.replace("\n", "\n    ")
            if broke != 0:
                failed = True
            if broke == 0 or args.accept:
                base[name] = str(out.relative_to(ROOT))
        else:
            base[name] = str(out.relative_to(ROOT))
        BASELINES.write_text(json.dumps(base, indent=1) + "\n")
        print(line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
