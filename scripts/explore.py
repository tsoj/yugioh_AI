#!/usr/bin/env python3
"""Exploration: belief checks of fresh games until the first failure.

Each seed (from 100000 on, continuing where the last run stopped) gets its
settings from the seed itself, so a game is reproducible from its seed alone:
the deck mode (random corpus decks, diverse, random cards, newer human decks,
human vs random cards, mixed human decks, the opening book), the bots and how
the unknown opponent cards are filled in. Games run in batches on all cores;
after a batch with a failure the run stops and reports it.

Every game played is logged (runs/explore/log.jsonl). The games worth checking
again - a failure, restarts, a slow or unfinished reconstruction - join the
regression set (tests/explore_seeds.json), which scripts/suite.py runs as
further ranges.

    python3 scripts/explore.py              # until the first failure
    python3 scripts/explore.py --games 2000 # at most this many games
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

STATE = ROOT / "runs" / "explore" / "state.json"
LOG = ROOT / "runs" / "explore" / "log.jsonl"
SEEDS = ROOT / "tests" / "explore_seeds.json"
FIRST_SEED = 100000
P = "setter-greedy,mixed-setter,greedy-mixed,setter-random,mixed-mixed,random-setter"

DECKS = [("new", 3), ("new-mixed", 2), ("new-cards", 2), ("diverse", 2), ("cards", 1), ("random", 1), ("book", 1)]
PAIRS = [(None, 1), (P, 2)]
COMPLETION = [("neural", 6), ("corpus", 2), ("vanilla", 1)]  # (neural: the learned deck models, the bot's)


def _pick(rng: random.Random, choices: list[tuple]):
    return rng.choices([c for c, _ in choices], weights=[w for _, w in choices])[0]


def settings(seed: int) -> list[str]:
    """The check_belief flags of a seed's game (drawn from the seed)."""
    rng = random.Random(f"explore {seed}")
    decks, pairs, completion = _pick(rng, DECKS), _pick(rng, PAIRS), _pick(rng, COMPLETION)
    flags = ["--book", "build/book.jsonl"] if decks == "book" else ["--decks", decks]
    if pairs:
        flags += ["--pairs", pairs]
    return flags + ["--completion", completion]


def interesting(r: dict) -> bool:
    return not r["ok"] or bool(r.get("restarts")) or (r.get("seconds") or 0) > 2 or bool(r.get("end_ignored"))


def solo_seconds(r: dict, flags: str, limit: int = 900) -> str:
    """Replays a timed-out reconstruction alone, with a longer limit: its time
    there, or whether it still runs out (a hang rather than load)."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "solo.jsonl"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "check_belief.py"), "--seeds", str(r["seed"]),
                        "--fractions", str(r["frac"]), "--turns", "", "--timeout", str(limit), "-j", "1",
                        "-o", str(out), *flags.split()], cwd=ROOT, capture_output=True)
        rows = [json.loads(line) for line in out.open()] if out.exists() else []
    row = next((x for x in rows if x.get("viewer") == r.get("viewer")), None)
    if row is None:
        return "could not run"
    if row.get("error") == "timeout":
        return f"timeout after {limit}s too"
    return f"{row.get('seconds', 0):.0f}s, {'consistent' if row['ok'] else (row.get('error') or '')[:80]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, help="first seed (default: where the last run stopped)")
    ap.add_argument("--games", type=int, default=10 ** 9, help="at most this many games")
    ap.add_argument("--batch", type=int, default=128, help="games per batch")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    import snapshot
    snapshot.reexec("explore")  # (from a frozen copy of the code: edits during the run cannot mix in)
    import check_belief as cb
    from compare_runs import reviewed
    from common.parallel import Crash, run_parallel

    LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        state = {"next": FIRST_SEED}
    seed = args.start if args.start is not None else state["next"]
    try:
        regression = json.loads(SEEDS.read_text())
    except (OSError, ValueError):
        regression = {}
    ok_pseudo = reviewed()
    played, t0 = 0, time.time()
    while played < args.games:
        n = min(args.batch, args.games - played)
        seeds = list(range(seed, seed + n))
        groups: dict[str, list[int]] = {}
        for s in seeds:
            groups.setdefault(" ".join(settings(s)), []).append(s)
        tasks, flags_of = [], {}
        for flags, ss in groups.items():
            argv = ["--seeds", ",".join(map(str, ss)), *flags.split()]
            for t in cb.build_tasks(cb.make_parser().parse_args(argv)):
                tasks.append(t)
                flags_of[id(t)] = flags
        failures = []
        with LOG.open("a") as log:
            for task, res in run_parallel(cb.run_task, tasks, args.jobs):
                flags = flags_of[id(task)]
                if isinstance(res, Crash):
                    res = [{"seed": task[0], "viewer": v, "frac": f, "ok": False, "error": f"crash: {res.error}"}
                           for v, f in task[1]]
                res = [r for r in res if r is not None]
                for r in res:
                    if not r["ok"] and ok_pseudo.get((r["seed"], r.get("viewer"), r.get("frac"))) == r.get("error"):
                        r["ok"] = True
                    log.write(json.dumps({"flags": flags, **r}) + "\n")
                    if not r["ok"]:
                        failures.append((flags, r))
                if any(interesting(r) for r in res):
                    lst = regression.setdefault(flags, [])
                    if task[0] not in lst:
                        lst.append(task[0])
        played += n
        seed += n
        state["next"] = seed
        STATE.write_text(json.dumps(state) + "\n")
        SEEDS.write_text(json.dumps({k: sorted(v) for k, v in sorted(regression.items())}, indent=1) + "\n")
        print(f"{played} games ({seeds[0]}-{seeds[-1]}), {time.time() - t0:.0f}s: {len(failures)} failing", flush=True)
        if failures:
            load = os.getloadavg()[0]
            for flags, r in failures:
                if r.get("error") == "timeout":
                    # (timeouts apart: the machine's load can cause them; the case alone,
                    # with a longer limit, shows how slow it really is)
                    alone = solo_seconds(r, flags)
                    print(f"  TIMEOUT seed={r['seed']} viewer={r.get('viewer')} frac={r.get('frac')} [{flags}] "
                          f"(load {load:.1f}; alone: {alone})")
                else:
                    print(f"  seed={r['seed']} viewer={r.get('viewer')} frac={r.get('frac')} [{flags}]: "
                          f"{(r.get('error') or '')[:150]}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
