#!/usr/bin/env python3
"""Belief replay consistency test.

Plays games between the built-in agents (random / greedy, random deck pairs,
some mirror matches), then for both players and for several cut points of
the game (default: ~1/3, ~2/3 and the full game) reconstructs the duel from
that player's observations only and checks that the reconstruction produces
exactly the same observations and state views.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.belief.check import check_prefix, cut_points, make_game, observation_stream  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402

class Timeout(BaseException):  # not swallowed by check_prefix
    pass


def _alarm(*_):
    if os.getpid() != _main_pid:
        os._exit(3)  # a forked probe inherited the alarm
    # an exception raised inside e.g. an os.fork() callback is ignored: try again
    # a second later (the task cancels the alarm when it ends)
    signal.alarm(1)
    raise Timeout()


_main_pid = os.getpid()


_completion = None


def get_completion(name: str):
    global _completion
    if _completion is None:
        if name == "corpus":
            from ygoai.belief.corpus import CorpusCompletion
            _completion = CorpusCompletion()
        else:
            from ygoai.belief.completion import VanillaCompletion
            _completion = VanillaCompletion()
    return _completion


_games: dict = {}


def _game(seed: int, mirror_prob: float, pairs=None, decks="random", book=None):
    """The game and both observation streams (cached per worker process)."""
    key = (seed, mirror_prob, pairs, decks, book)
    if key not in _games:
        record = make_game(seed, mirror_prob, [tuple(p) for p in pairs] if pairs else None, decks, book)
        _games.clear()  # tasks of one seed tend to arrive together
        _games[key] = (record, [observation_stream(record, v) for v in (0, 1)])
    return _games[key]


def run_task(task: tuple) -> list[dict]:
    """One reconstruction: (seed, viewer, fraction, ...)."""
    seed, viewer, frac, mirror_prob, timeout, completion, pairs, decks, book = task
    global _main_pid
    _main_pid = os.getpid()
    signal.signal(signal.SIGALRM, _alarm)
    try:
        record, streams = _game(seed, mirror_prob, pairs, decks, book)
    except Exception as e:  # noqa: BLE001
        return [{"seed": seed, "viewer": viewer, "frac": frac, "ok": False, "error": f"game failed: {e}"}]
    items = streams[viewer]
    cut = cut_points(items, (frac,))[0]
    signal.alarm(timeout)
    t0 = time.monotonic()
    try:
        r = check_prefix(record, viewer, items, cut, completion=get_completion(completion))
        res = {"seed": seed, "viewer": viewer, "frac": frac, "cut": cut, "len": len(items),
               "ok": r.ok, "error": r.error, "seconds": r.seconds,
               "decisions": r.opponent_decisions, "probed": r.probed, "probes": r.probes,
               "restarts": r.restarts, "fixes": r.fixes, "repairs": r.repairs}
    except Timeout:
        res = {"seed": seed, "viewer": viewer, "frac": frac, "cut": cut, "len": len(items),
               "ok": False, "error": "timeout", "seconds": time.monotonic() - t0}
    finally:
        signal.alarm(0)
    return [res]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=50)
    ap.add_argument("--start", type=int, default=0, help="first game seed")
    ap.add_argument("--seeds", help="comma separated game seeds (instead of -n/--start)")
    ap.add_argument("--cases", type=Path,
                    help="rerun exactly the reconstructions (seed, viewer, frac) of an earlier output")
    ap.add_argument("--only-failed", action="store_true", help="with --cases: only the failed ones")
    ap.add_argument("--index", type=Path, help="game index of scripts/index_games.py (with --tags)")
    ap.add_argument("--tags", help="comma separated: only games from --index with any of these tags")
    ap.add_argument("--fractions", default="0.333,0.667,1.0")
    ap.add_argument("--mirror-prob", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=180, help="seconds per belief replay")
    ap.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("-o", "--output", type=Path, help="write all results as JSON lines")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--pairs", help="agent pairings, e.g. setter-greedy,mixed-random (game seed %% number "
                                    "of pairs picks one; default: random/greedy in all four combinations)")
    ap.add_argument("--decks", choices=["random", "diverse"], default="random",
                    help="random corpus decks, or decks ordered to cover as many different cards as possible")
    ap.add_argument("--book", help="opening book (scripts/make_book.py): game seed %% openings picks the decks")
    ap.add_argument("--completion", choices=["vanilla", "corpus"], default="vanilla",
                    help="how unknown opponent cards are guessed")
    args = ap.parse_args()
    fractions = tuple(float(f) for f in args.fractions.split(","))
    if args.cases:
        cases = []
        for line in args.cases.open():
            r = json.loads(line)
            if "viewer" in r and "frac" in r and (not args.only_failed or not r["ok"]):
                cases.append((r["seed"], r["viewer"], r["frac"]))
        cases.sort(key=lambda c: -c[2])
    else:
        if args.seeds:
            seeds = [int(x) for x in args.seeds.split(",")]
        elif args.index and args.tags:
            wanted = set(args.tags.split(","))
            seeds = [r["seed"] for r in map(json.loads, args.index.open()) if wanted & set(r["tags"])]
            seeds = seeds[:args.games]
            print(f"{len(seeds)} games with tags {sorted(wanted)}", flush=True)
        else:
            seeds = [args.start + i for i in range(args.games)]
        # one task per reconstruction, the long full-game ones first
        cases = [(seed, viewer, frac) for frac in sorted(fractions, reverse=True)
                 for seed in seeds for viewer in (0, 1)]
    pairs = tuple(tuple(p.split("-")) for p in args.pairs.split(",")) if args.pairs else None
    tasks = [(seed, viewer, frac, args.mirror_prob, args.timeout, args.completion, pairs, args.decks, args.book)
             for seed, viewer, frac in cases]
    results = []
    t0 = time.monotonic()
    out = args.output.open("w") if args.output else None  # streamed: large runs can be watched
    for task, res in run_parallel(run_task, tasks, args.jobs):
        if isinstance(res, Crash):
            res = [{"seed": task[0], "viewer": task[1], "frac": task[2], "ok": False,
                    "error": f"crash: {res.error}"}]
        results.extend(res)
        if out:
            for r in res:
                out.write(json.dumps(r) + "\n")
            out.flush()
        if args.verbose:
            for r in res:
                if not r["ok"]:
                    print(f"FAIL seed={r['seed']} viewer={r.get('viewer')} frac={r.get('frac')}: "
                          f"{r['error'][:200]}", flush=True)
    if out:
        out.close()
    ok = sum(r["ok"] for r in results)
    print(f"\n{ok}/{len(results)} reconstructions consistent "
          f"({100 * ok / max(1, len(results)):.1f}%), wall {time.monotonic() - t0:.0f}s")
    by_frac = collections.defaultdict(lambda: [0, 0])
    for r in results:
        if "frac" in r:
            by_frac[r["frac"]][0] += r["ok"]
            by_frac[r["frac"]][1] += 1
    for frac, (k, n) in sorted(by_frac.items()):
        print(f"  cut {frac:.2f}: {k}/{n}")
    times = [r["seconds"] for r in results if r.get("seconds")]
    if times:
        times.sort()
        print(f"  replay time: median {times[len(times) // 2]:.2f}s, max {times[-1]:.1f}s")
    fixes = collections.Counter()
    for r in results:
        for k, v in (r.get("fixes") or {}).items():
            fixes[k] += v
    if fixes:
        print("  repairs used:", dict(fixes))
    errs = collections.Counter()
    for r in results:
        if not r["ok"]:
            msg = r["error"]
            key = msg.split(": ", 1)[1] if msg.startswith("divergence at item") else msg
            errs[key[:100]] += 1
    for k, v in errs.most_common(25):
        print(f"  {v:4d}  {k}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
