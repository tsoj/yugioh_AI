#!/usr/bin/env python3
"""fastchess-style match between two Yu-Gi-Oh engines.

Examples:
  scripts/match.py --engine greedy=builtin:greedy --engine random=builtin:random -n 200
  scripts/match.py --engine new="python -m ygoai.engine greedy" --engine base=builtin:greedy \\
      --sprt 0 10 --tc 60+0.5 -n 5000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.match import (EngineSpec, MatchStats, close_engines, play_opening,  # noqa: E402
                         random_openings, sprt_bounds)
from ygoai.parallel import Crash, run_parallel  # noqa: E402

_args = None


def _init(args):
    global _args
    _args = args


def _run(opening):
    a, b = _args["engines"]
    try:
        return play_opening(opening, a, b, tc=_args["tc"], max_turns=_args["max_turns"])
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def parse_engine(text: str) -> EngineSpec:
    name, _, cmd = text.partition("=")
    if not cmd:
        raise argparse.ArgumentTypeError("expected NAME=COMMAND")
    return EngineSpec(name, cmd)


def parse_tc(text: str) -> tuple[float, float]:
    base, _, inc = text.partition("+")
    return float(base), float(inc or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", type=parse_engine, action="append", required=True,
                    help="NAME=COMMAND, COMMAND is builtin:<agent> or an engine protocol process")
    ap.add_argument("-n", "--openings", type=int, default=100, help="number of openings (2 games each)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mirror-prob", type=float, default=0.2)
    ap.add_argument("--deck-split", choices=["train", "test"],
                    help="only use decks of this corpus split (use 'test' for engines that "
                         "complete opponent decks from the 'train' corpus)")
    ap.add_argument("--tc", type=parse_tc, help="time control per game: BASE+INC in seconds")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--sprt", type=float, nargs=2, metavar=("ELO0", "ELO1"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("-c", "--concurrency", type=int, default=mp.cpu_count())
    ap.add_argument("--records", type=Path, help="write game records as JSON lines")
    args = ap.parse_args()
    if len(args.engine) != 2:
        ap.error("exactly two engines are required")

    from ygoai.carddb import default_db
    from ygoai.decks import DeckPool
    pool = DeckPool(default_db(), split=args.deck_split)
    openings = random_openings(args.openings, args.seed, args.mirror_prob, pool)
    stats = MatchStats()
    lo, hi = sprt_bounds(args.alpha, args.beta) if args.sprt else (None, None)
    a, b = args.engine
    print(f"{a.name} vs {b.name}: {len(openings)} openings x 2 games", flush=True)
    out = args.records.open("w") if args.records else None
    t0 = time.monotonic()
    errors = 0
    shared = {"engines": (a, b), "tc": args.tc, "max_turns": args.max_turns}
    results = run_parallel(_run, openings, args.concurrency, initializer=_init, initargs=(shared,))
    for i, (opening, res) in enumerate(results, 1):
        if isinstance(res, Crash):
            res = {"error": f"crash in opening {opening.seed}: {res.error}"}
        if isinstance(res, dict):
            errors += 1
            print("error:", res["error"], flush=True)
            continue
        stats.add_pair([g["score"] for g in res])
        if out:
            for g in res:
                out.write(json.dumps(g["record"]) + "\n")
        if i % 10 == 0 or i == len(openings):
            print(f"[{time.monotonic() - t0:6.0f}s] " +
                  stats.summary(*(args.sprt or (None, None))), flush=True)
        if args.sprt:
            llr = stats.llr(*args.sprt)
            if llr <= lo or llr >= hi:
                print(f"SPRT finished: {'H1 accepted' if llr >= hi else 'H0 accepted'} "
                      f"(LLR {llr:.2f}, bounds [{lo:.2f}, {hi:.2f}])")
                results.close()
                break
    close_engines()
    if out:
        out.close()
    print("final:", stats.summary(*(args.sprt or (None, None))), f"errors: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
