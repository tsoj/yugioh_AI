#!/usr/bin/env python3
"""The belief regression suite: every range of games ever checked, with the
exact settings it was played with (other settings play other games).

Each range is checked against its latest baseline (runs/suite/baselines.json):

    python3 scripts/suite.py TAG                  # all ranges, in one pool of workers
    python3 scripts/suite.py TAG reg1000 new50000 # some of them
    python3 scripts/suite.py --list

A range's run becomes its new baseline when nothing that passed there before
broke (or with --accept). A run that did not finish leaves the baseline alone.
The baselines are kept in git, slimmed down (tests/baselines/RANGE.jsonl.gz: per
reconstruction the game, whether it was consistent, the error, the restarts and
a coarse cost); the full results of each run stay in runs/suite/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE_DIR = ROOT / "runs" / "suite"
BASE_DIR = ROOT / "tests" / "baselines"
P = "setter-greedy,mixed-setter,greedy-mixed,setter-random,mixed-mixed,random-setter"
MC = "belief_mc-greedy,setter-belief_mc,belief_mc-mixed,random-belief_mc,belief_mc-belief_mc"

# name: (games, first seed, other flags); the unknown opponent cards are filled from
# real decks (--completion corpus, cards with effects) except where vanilla is given
RANGES: dict[str, tuple[int, int, list[str]]] = {
    "reg1000": (1000, 1000, ["--completion", "vanilla"]),
    "reg2000": (1000, 2000, ["--completion", "vanilla"]),
    "reg3000": (1000, 3000, ["--completion", "vanilla"]),
    "reg4000": (1000, 4000, ["--pairs", P, "--completion", "vanilla"]),
    "reg5000": (1000, 5000, ["--decks", "diverse", "--completion", "vanilla"]),
    "reg6000": (1000, 6000, ["--decks", "diverse", "--pairs", P, "--completion", "vanilla"]),
    "reg7000": (1000, 7000, ["--decks", "diverse", "--pairs", P, "--completion", "vanilla"]),
    "book9000": (1000, 9000, ["--book", "build/book.jsonl", "--completion", "vanilla"]),
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
    "cards40000": (1000, 40000, ["--decks", "cards", "--completion", "vanilla"]),
    "cards41000": (1000, 41000, ["--decks", "cards", "--pairs", P, "--completion", "vanilla"]),
    "new50000": (1000, 50000, ["--decks", "new"]),
    "neural50000": (200, 50000, ["--decks", "new", "--completion", "neural"]),  # (the learned deck models)
    # (as belief_mc reconstructs: every 3rd decision, inheriting; one result per game and viewer)
    "agentseq50000": (100, 50000, ["--decks", "new", "--completion", "neural", "--agent-sequence", "3"]),
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
    "corp58000": (400, 58000, ["--decks", "new", "--completion", "corpus"]),
    "corp59000": (400, 59000, ["--decks", "new-mixed", "--completion", "corpus"]),
    "corp60000": (400, 60000, ["--decks", "new", "--pairs", P, "--completion", "corpus"]),
    "corp61000": (400, 61000, ["--decks", "new-cards", "--pairs", P, "--completion", "corpus"]),
    "corp66000": (400, 66000, ["--decks", "diverse", "--pairs", P, "--completion", "corpus"]),
}


EXPLORE_SEEDS = ROOT / "tests" / "explore_seeds.json"


def explored() -> dict[str, tuple[list[int], list[str]]]:
    """The regression set from scripts/explore.py: games of fresh seeds worth
    checking again, per settings: range name -> (seeds, flags)."""
    try:
        groups = json.loads(EXPLORE_SEEDS.read_text())
    except (OSError, ValueError):
        return {}
    return {f"x{zlib.crc32(flags.encode()) & 0xffffff:06x}": (seeds, flags.split()) for flags, seeds in groups.items()}


def all_ranges() -> dict[str, list[str]]:
    """Range name -> its check_belief arguments (the fixed ranges and the explored games)."""
    out = {name: ["-n", str(n), "--start", str(start), *flags] for name, (n, start, flags) in RANGES.items()}
    for name, (seeds, flags) in explored().items():
        out[name] = ["--seeds", ",".join(map(str, seeds)), *flags]
    return out


def load_baselines() -> dict[str, str]:
    """Range name -> its baseline file (relative to the repository root)."""
    return {name: str((BASE_DIR / f"{name}.jsonl.gz").relative_to(ROOT)) for name in all_ranges()
            if (BASE_DIR / f"{name}.jsonl.gz").exists()}


def read_rows(path: Path):
    """The result rows of a run file (.jsonl, or a gzipped baseline)."""
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        for line in f:
            yield json.loads(line)


def slim(r: dict) -> dict:
    """What a baseline keeps of a result row: enough to compare runs (and to pick
    the quick tier and the order of the games), without the run's noise."""
    out = {k: r[k] for k in ("seed", "viewer", "frac", "cut", "len", "game", "ok") if k in r}
    if not r["ok"]:
        out["error"] = (r.get("error") or "")[:300]
    if r.get("restarts"):
        out["restarts"] = r["restarts"]
    if r.get("end_ignored"):
        out["end_ignored"] = True
    seconds = r.get("seconds") or 0.0
    out["cost"] = max(-3, math.ceil(math.log2(max(seconds, 0.1))))  # (seconds, as a power of 2)
    return out


def store_baseline(name: str, run: Path) -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted((slim(r) for r in read_rows(run)), key=lambda r: (r["seed"], r.get("viewer") or 0, -(r.get("frac") or 0)))
    data = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode()
    tmp = BASE_DIR / f"{name}.jsonl.gz.tmp"
    with open(tmp, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as gz:
        gz.write(data)  # (no time stamp: the same results give the same file)
    tmp.replace(BASE_DIR / f"{name}.jsonl.gz")


def _seconds(r: dict) -> float:
    return r["seconds"] if "seconds" in r else 2.0 ** r.get("cost", 0)


def compare(old: Path, new: Path) -> tuple[str, int]:
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), str(old), str(new)],
                         capture_output=True, text=True).stdout
    m = re.search(r"\((\d+) fixed, (\d+) broke, (\d+) still failing\)", out)
    broke = int(m.group(2)) if m else -1
    return out.strip(), broke


def _cost(baseline: str | None) -> dict[int, float]:
    """Seconds each game's reconstructions took in the baseline (a timeout as its limit)."""
    out: dict[int, float] = {}
    if baseline:
        for r in read_rows(ROOT / baseline):
            out[r["seed"]] = out.get(r["seed"], 0.0) + _seconds(r)
    return out


def _quick_cases(baseline: str | None, failed_only: bool = False, sample: bool = True) -> set[tuple] | None:
    """The quick tier's reconstructions of a range: those that failed, needed
    restarts, took long or ended in an unfinished action in the baseline (where
    regressions show), and a fixed tenth of the others."""
    if not baseline:
        return None
    keep = set()
    for r in read_rows(ROOT / baseline):
        if "frac" not in r:
            continue
        key = (r["seed"], r["viewer"], r["frac"])
        if failed_only:
            if not r["ok"]:
                keep.add(key)
            continue
        if not r["ok"] or (r.get("restarts") or 0) > 0 or _seconds(r) > 2 or r.get("end_ignored") \
                or sample and zlib.crc32(f"{key[0]} {key[1]} {key[2]}".encode()) % 10 == 0:
            keep.add(key)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", nargs="?", help="suffix of this run's outputs")
    ap.add_argument("ranges", nargs="*", help="only these ranges (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--accept", action="store_true", help="make the runs baselines even if something broke")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true",
                    help="only the reconstructions where regressions show (see _quick_cases); its runs never "
                         "become baselines")
    ap.add_argument("--regression", action="store_true",
                    help="the regression set: as --quick, without the sample of the others (never made baselines)")
    ap.add_argument("--failed", action="store_true", help="only the reconstructions that failed in the baselines "
                                                          "(never made baselines)")
    args = ap.parse_args()
    base = load_baselines()
    ranges = all_ranges()
    if args.list or not args.tag:
        for name, argv in ranges.items():
            print(f"{name:12s} {' '.join(argv)[:110]:110s} {base.get(name, '(no baseline)')}")
        return 0
    unknown = [r for r in args.ranges if r not in ranges]
    if unknown:
        sys.exit(f"unknown ranges: {unknown}")
    # (from a frozen copy of the code: edits during the run cannot mix in)
    sys.path.insert(0, str(ROOT / "scripts"))
    import snapshot
    snapshot.reexec(f"suite-{args.tag}")
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_belief as cb
    from ygoai.parallel import Crash, run_parallel
    SUITE_DIR.mkdir(parents=True, exist_ok=True)
    # all ranges' games in one pool, the slowest (by the baselines) first: no range
    # waits for its last few slow cases while the other workers are idle
    names = args.ranges or list(ranges)
    owner: dict[int, str] = {}
    left: dict[str, int] = {}
    outs, results, tasks = {}, {}, []
    for name in names:
        argv = list(ranges[name])
        if base.get(name):
            argv += ["--previous", str(ROOT / base[name])]
        cost = _cost(base.get(name))
        median = sorted(cost.values())[len(cost) // 2] if cost else 1.0
        quick = (_quick_cases(base.get(name), args.failed, sample=not args.regression)
                 if args.quick or args.failed or args.regression else None)
        for t in cb.build_tasks(cb.make_parser().parse_args(argv)):
            if quick is not None:
                cases = [(v, f) for v, f in t[1] if (t[0], v, f) in quick]
                if not cases:
                    continue
                t = (t[0], cases, *t[2:])
            owner[id(t)] = name
            tasks.append((cost.get(t[0], median), t))
        left[name] = sum(1 for _, t in tasks if owner[id(t)] == name)
        if not left[name]:
            continue
        outs[name] = (SUITE_DIR / f"{name}.{args.tag}.jsonl").open("w")
        results[name] = [0, 0]  # (consistent, all)
    tasks.sort(key=lambda ct: -ct[0])
    print(f"{len(tasks)} games of {len(names)} ranges", flush=True)
    t0, failed = time.time(), False
    for task, res in run_parallel(cb.run_task, [t for _, t in tasks], args.jobs):
        name = owner[id(task)]
        if isinstance(res, Crash):
            res = [{"seed": task[0], "viewer": v, "frac": f, "ok": False, "error": f"crash: {res.error}"}
                   for v, f in task[1]]
        res = [r for r in res if r is not None]
        results[name][0] += sum(r["ok"] for r in res)
        results[name][1] += len(res)
        for r in res:
            outs[name].write(json.dumps(r) + "\n")
        outs[name].flush()
        left[name] -= 1
        if left[name]:
            continue
        outs[name].close()
        out = SUITE_DIR / f"{name}.{args.tag}.jsonl"
        ok, total = results[name]
        line = f"{name}: {ok}/{total} consistent [{time.time() - t0:.0f}s]"
        old = base.get(name)
        if old:
            text, broke = compare(ROOT / old, out)
            line += "\n    " + text.replace("\n", "\n    ")
            other = re.search(r"(\d+) reconstructions are of games that were played differently", text)
            if other:
                # (not compared: not silently a baseline either)
                line += f"\n    ! {other.group(1)} reconstructions of other games: not compared, run not made the baseline"
            failed |= broke != 0 or bool(other)
            if (broke == 0 and not other or args.accept) and not (args.quick or args.failed or args.regression):
                store_baseline(name, out)
        elif not (args.quick or args.failed or args.regression):
            store_baseline(name, out)
        print(line, flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
