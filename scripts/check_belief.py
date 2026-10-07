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
import ctypes
import gc
import gzip
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.belief.check import check_prefix, cut_point, game_decks, make_game_streams  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402

class Timeout(BaseException):  # not swallowed by check_prefix
    pass


def _alarm(_signum, frame):
    if os.getpid() != _main_pid:
        os._exit(3)  # a forked probe inherited the alarm
    f = frame
    while f is not None:
        if f.f_code.co_name in ("close", "__del__") and f.f_code.co_filename.endswith("core.py"):
            # (not while a duel is destroyed: an exception between the core's destroy
            # and forgetting the handle destroys it a second time, a crash)
            signal.alarm(1)
            return
        f = f.f_back
    # an exception raised inside e.g. an os.fork() callback is ignored: try again
    # a second later (the task cancels the alarm when it ends)
    signal.alarm(1)
    raise Timeout()


_main_pid = os.getpid()


_completions: dict = {}  # (per worker process: building one takes a while)


def get_completion(name: str):
    if name not in _completions:
        if name == "corpus":
            from ygoai.belief.corpus import CorpusCompletion
            _completions[name] = CorpusCompletion()
        else:
            from ygoai.belief.completion import VanillaCompletion
            _completions[name] = VanillaCompletion()
    return _completions[name]


def _replay_case(record, items, seed: int, viewer: int, frac: float, timeout: int, completion: str) -> dict | None:
    cut = cut_point(items, frac)
    if cut is None:
        return None  # (e.g. a turn the game did not reach)
    signal.alarm(timeout)
    t0 = time.monotonic()
    try:
        r = check_prefix(record, viewer, items, cut, completion=get_completion(completion))
        if time.monotonic() - t0 > timeout:
            # (the alarm came while the core ran one long step: the only Python code
            # running then is its log callback, which swallows the exception)
            raise Timeout()
        return {"seed": seed, "viewer": viewer, "frac": frac, "cut": cut, "len": len(items),
                "ok": r.ok, "error": r.error, "seconds": r.seconds,
                "decisions": r.opponent_decisions, "probed": r.probed, "probes": r.probes,
                "restarts": r.restarts, "fixes": r.fixes, "repairs": r.repairs, "end_ignored": r.end_ignored}
    except Timeout:
        return {"seed": seed, "viewer": viewer, "frac": frac, "cut": cut, "len": len(items),
                "ok": False, "error": "timeout", "seconds": time.monotonic() - t0}
    finally:
        signal.alarm(0)


def run_task(task: tuple) -> list[dict]:
    """All reconstructions of one game: (seed, [(viewer, cut), ...], ...). The game
    is played once, recording both players' observation streams on the way."""
    seed, cases, mirror_prob, timeout, completion, pairs, decks, book, deck_pair = task
    global _main_pid
    _main_pid = os.getpid()
    signal.signal(signal.SIGALRM, _alarm)
    try:
        record, streams = make_game_streams(seed, mirror_prob, [tuple(p) for p in pairs] if pairs else None,
                                            decks, book, deck_pair=deck_pair)
    except Exception as e:  # noqa: BLE001
        return [{"seed": seed, "viewer": v, "frac": f, "ok": False, "error": f"game failed: {e}"}
                for v, f in cases]
    out = []
    done = set()
    for viewer, frac in cases:
        cut = cut_point(streams[viewer], frac)
        if cut is None or (viewer, cut) in done:
            continue  # (the same prefix as another cut)
        done.add((viewer, cut))
        r = _replay_case(record, streams[viewer], seed, viewer, frac, timeout, completion)
        if r is not None:
            # (which game it was: runs are compared only on the same games)
            r["game"] = zlib.crc32(repr(streams[viewer]).encode())
        out.append(r)
    # (give the memory of the game's replays back: a worker runs many games, and the
    # heap would otherwise stay at the largest replay's peak)
    del record, streams
    gc.collect()
    _malloc_trim()
    return out


def _malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def make_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--turns", default="1,2,3",
                    help="also cut at the end of these turns (results show them as frac -N); '' for none")
    ap.add_argument("--turn-share", type=float, default=0.1,
                    help="share of the games that get the --turns cuts (picked by seed, the same in every "
                         "run; they rarely fail where the others pass), 1 for all")
    ap.add_argument("--previous", type=Path,
                    help="an earlier output: games whose reconstruction failed there (at any cut, for that "
                         "viewer) get the --turns cuts too")
    ap.add_argument("--mirror-prob", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=180, help="seconds per belief replay")
    ap.add_argument("--read-guided-ok", action="store_true",
                    help="count reconstructions the last-resort read-guided repair saved as consistent "
                         "(by default they count as failures, to be looked at)")
    ap.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("-o", "--output", type=Path, help="write all results as JSON lines")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--pairs", help="agent pairings, e.g. setter-greedy,mixed-random (game seed %% number "
                                    "of pairs picks one; default: random/greedy in all four combinations)")
    ap.add_argument("--decks", choices=["random", "diverse", "cards", "new", "new-cards", "new-mixed"], default="random",
                    help="random corpus decks, decks ordered to cover as many different cards as possible, "
                         "decks of random cards (\"cards\"), or the decks downloaded later in the diverse order "
                         "(\"new\")")
    ap.add_argument("--book", help="opening book (scripts/make_book.py): game seed %% openings picks the decks")
    ap.add_argument("--completion", choices=["vanilla", "corpus"], default="corpus",
                    help="how unknown opponent cards are guessed")
    return ap


def build_tasks(args: argparse.Namespace) -> list[tuple]:
    """One task per game (it is played once for all of its reconstructions)."""
    fractions = tuple(float(f) for f in args.fractions.split(",") if f)
    turn_cuts = tuple(-float(t) for t in args.turns.split(",") if t)
    failed_before: set[tuple[int, int]] = set()
    if args.previous:
        opener = gzip.open(args.previous, "rt") if args.previous.suffix == ".gz" else args.previous.open()
        failed_before = {(r["seed"], r["viewer"]) for r in map(json.loads, opener)
                         if "viewer" in r and not r["ok"]}

    def cuts(seed: int, viewer: int) -> tuple:
        # (a fixed share of the seeds, spread over the range by a hash)
        sampled = zlib.crc32(f"turn cuts {seed}".encode()) % 10000 < args.turn_share * 10000
        return fractions + (turn_cuts if sampled or (seed, viewer) in failed_before else ())
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
        cases = [(seed, viewer, frac) for seed in seeds for viewer in (0, 1)
                 for frac in sorted(cuts(seed, viewer), reverse=True)]
    pairs = tuple(tuple(p.split("-")) for p in args.pairs.split(",")) if args.pairs else None
    by_seed: dict[int, list] = {}
    for seed, viewer, frac in cases:
        by_seed.setdefault(seed, []).append((viewer, frac))
    # (the decks are chosen here: the workers then never load the deck corpus,
    # which keeps them small and the replay's probes (forks) cheap)
    return [(seed, cs, args.mirror_prob, args.timeout, args.completion, pairs, args.decks, args.book,
             game_decks(seed, args.mirror_prob, args.decks, args.book))
            for seed, cs in by_seed.items()]


def main() -> int:
    args = make_parser().parse_args()
    if args.read_guided_ok:
        os.environ["YGOAI_READ_GUIDED_OK"] = "1"
    tasks = build_tasks(args)
    results = []
    t0 = time.monotonic()
    out = args.output.open("w") if args.output else None  # streamed: large runs can be watched
    for task, res in run_parallel(run_task, tasks, args.jobs):
        if isinstance(res, Crash):
            res = [{"seed": task[0], "viewer": v, "frac": f, "ok": False, "error": f"crash: {res.error}"}
                   for v, f in task[1]]
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
        print(f"  cut {f'turn {-frac:.0f}' if frac < 0 else f'{frac:.2f}'}: {k}/{n}")
    pseudo = sum(1 for r in results if not r["ok"] and str(r.get("error", "")).startswith("read-guided repair"))
    if pseudo:
        print(f"  ({pseudo} of the failures are pseudo-failures: consistent with the last-resort read-guided "
              f"repair, to check that no leak is behind them)")
    ignored = sum(1 for r in results if r.get("end_ignored"))
    if ignored:
        print(f"  ({ignored} consistent up to the viewer's last decision: the game ended in an unfinished action)")
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
            if msg.startswith("read-guided repair"):
                key = "pseudo-failure (read-guided repair)"
            errs[key[:100]] += 1
    for k, v in errs.most_common(25):
        print(f"  {v:4d}  {k}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
