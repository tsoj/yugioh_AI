#!/usr/bin/env python3
"""Compare two belief check runs (JSON lines of scripts/check_belief.py).

Lists the reconstructions that were fixed, that broke, and that still fail,
and counts failures per error category, e.g. after a change:

    python3 scripts/check_belief.py --cases runs/old.jsonl --only-failed -o runs/new.jsonl
    python3 scripts/compare_runs.py runs/old.jsonl runs/new.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


def category(error: str) -> str:
    """The kind of failure, without positions, codes and values."""
    e = error.split("\n", 1)[0]
    e = re.sub(r"^divergence at item \d+: ", "", e)
    e = re.sub(r"too many restarts \(last: (.*)\)", r"\1", e)
    e = re.sub(r"(state mismatch: [^:]*):.*", r"\1", e)
    e = re.sub(r"\[\d+\]", "[N]", e)
    e = re.sub(r"\d+", "N", e)
    return e[:90]


def load(path: Path) -> dict[tuple, dict]:
    rows = {}
    for line in path.open():
        r = json.loads(line)
        if r["ok"] and (r.get("fixes") or {}).get("read_guided"):
            # (older runs counted reconstructions the read-guided repair saved as consistent)
            r["ok"], r["error"] = False, "read-guided repair needed (older run)"
        rows[(r["seed"], r.get("viewer"), r.get("frac"))] = r
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("-v", "--verbose", action="store_true", help="list every case")
    args = ap.parse_args()
    old, new = load(args.old), load(args.new)
    shared = set(old) & set(new)
    # (only the same games: e.g. agents that use the belief play other games when it changes)
    common = sorted((k for k in shared if old[k].get("len") == new[k].get("len")), key=str)
    if len(common) < len(shared):
        print(f"{len(shared) - len(common)} reconstructions are of games that were played differently "
              f"(not compared)")
    fixed = [k for k in common if not old[k]["ok"] and new[k]["ok"]]
    broke = [k for k in common if old[k]["ok"] and not new[k]["ok"]]
    still = [k for k in common if not old[k]["ok"] and not new[k]["ok"]]
    ok_old = sum(old[k]["ok"] for k in common)
    ok_new = sum(new[k]["ok"] for k in common)
    print(f"{len(common)} common reconstructions: {ok_old} -> {ok_new} consistent "
          f"({len(fixed)} fixed, {len(broke)} broke, {len(still)} still failing)")
    zero_old = sum(1 for k in common if old[k]["ok"] and not old[k].get("restarts"))
    zero_new = sum(1 for k in common if new[k]["ok"] and not new[k].get("restarts"))
    print(f"without any restart: {zero_old} -> {zero_new}")

    def show(title, keys, run):
        if not keys:
            return
        cats = collections.Counter(category(run[k].get("error", "")) for k in keys)
        print(f"\n{title}:")
        for c, n in cats.most_common():
            print(f"  {n:5d}  {c}")
        if args.verbose:
            for k in keys:
                print(f"    seed={k[0]} viewer={k[1]} frac={k[2]}: {run[k].get('error', '')[:120]}")

    show("broke (new errors)", broke, new)
    show("still failing (new errors)", still, new)
    if args.verbose:
        show("fixed (old errors)", fixed, old)
    return 0


if __name__ == "__main__":
    sys.exit(main())
