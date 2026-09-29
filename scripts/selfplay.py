#!/usr/bin/env python3
"""Plays many games between built-in agents and reports statistics.

Optionally writes the game records (one JSON object per line) for later use,
e.g. by the belief replay tests.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.agents import make_agent  # noqa: E402
from ygoai.carddb import default_db  # noqa: E402
from ygoai.decks import DeckPool  # noqa: E402
from ygoai.game import new_record, play  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402

_pool: DeckPool | None = None


def run_one(args: tuple) -> dict:
    global _pool
    seed, agent_names, mirror = args
    if _pool is None:
        _pool = DeckPool(default_db())
    rng = random.Random(seed)
    d0 = _pool.sample(rng)
    d1 = d0 if mirror else _pool.sample(rng)
    record = new_record([d0, d1], seed)
    agents = [make_agent(agent_names[0], seed * 2), make_agent(agent_names[1], seed * 2 + 1)]
    t0 = time.monotonic()
    try:
        play(record, agents, time_limit=120)
        return {"seed": seed, "record": record.to_json(), "time": time.monotonic() - t0}
    except Exception as e:  # noqa: BLE001
        return {"seed": seed, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(),
                "decks": [d0.name, d1.name], "agents": agent_names}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", default="random,random", help="comma separated pair, or 'mixed'")
    ap.add_argument("--mirror-prob", type=float, default=0.2, help="probability of mirror matches")
    ap.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("-o", "--output", type=Path, help="write records as JSON lines")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pairs = [("random", "random"), ("random", "greedy"), ("greedy", "random"), ("greedy", "greedy")]
    tasks = []
    for i in range(args.games):
        if args.agents == "mixed":
            names = pairs[i % len(pairs)]
        else:
            names = tuple(args.agents.split(","))
        tasks.append((args.seed * 1_000_000 + i, names, rng.random() < args.mirror_prob))

    stats = collections.Counter()
    reasons = collections.Counter()
    errors = []
    total_time = 0.0
    out = args.output.open("w") if args.output else None
    t0 = time.monotonic()
    for task, res in run_parallel(run_one, tasks, args.jobs):
        if isinstance(res, Crash):
            res = {"seed": task[0], "error": res.error, "trace": "(worker process died)",
                   "decks": [], "agents": task[1]}
        if "error" in res:
            errors.append(res)
            stats["error"] += 1
            continue
        rec = res["record"]
        r = rec["result"]
        total_time += res["time"]
        stats["games"] += 1
        stats["decisions"] += r["decisions"]
        stats["turns"] += r["turns"]
        stats["retries"] += r["retries"]
        stats["script_errors"] += r["script_errors"]
        reasons[r["reason"]] += 1
        if out:
            out.write(json.dumps(rec) + "\n")
    if out:
        out.close()
    n = max(1, stats["games"])
    print(f"games: {stats['games']}  errors: {stats['error']}  wall: {time.monotonic() - t0:.1f}s  "
          f"cpu/game: {total_time / n:.3f}s")
    print(f"avg decisions: {stats['decisions'] / n:.0f}  avg turns: {stats['turns'] / n:.1f}  "
          f"retries: {stats['retries']}  script errors: {stats['script_errors']}")
    print("end reasons:", dict(reasons))
    for e in errors[:5]:
        print("----", e["seed"], e["decks"], e["agents"])
        print(e["trace"])
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
