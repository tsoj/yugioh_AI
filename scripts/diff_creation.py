"""Leak detector: does anything the players observe depend on card creation order?

Card creation order (card ids) is hidden information: the arbiter creates the
cards in a random order, and our core patch processes cards in location order
wherever players can see the order. This replays recorded games with a
different random creation order and compares both players' observation
streams (messages, prompts and state views): any difference is an observable
effect of creation order, i.e. a leak (or a place the core patch misses).

    python scripts/diff_creation.py -n 500 --start 3000
"""

from __future__ import annotations

import argparse
import collections
import faulthandler
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.belief.check import make_game  # noqa: E402
from ygoai.game import GameError, ObservationLog, replay  # noqa: E402
from ygoai.parallel import run_parallel  # noqa: E402


def streams(record, creation_seed=None):
    logs = [ObservationLog(0), ObservationLog(1)]
    replay(record, logs, creation_seed=creation_seed).close()
    return [log.items for log in logs]


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


def diff(task: tuple, timeout: int = 300) -> list[tuple] | str:
    """Differences for (seed, agent pairings), or a string if the check itself failed."""
    seed, pairs, decks, book = task
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    # (a loop inside the core never returns to Python: the watchdog ends the
    # worker, which run_parallel reports as a crash of this seed)
    faulthandler.dump_traceback_later(timeout + 60, exit=True)
    try:
        return _diff(seed, pairs, decks, book)
    except _Timeout:
        return "timeout"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    finally:
        faulthandler.cancel_dump_traceback_later()
        signal.alarm(0)


def _diff(seed: int, pairs=None, decks="random", book=None) -> list[tuple]:
    record = make_game(seed, pairs=[tuple(p) for p in pairs] if pairs else None, decks=decks, book=book)
    base = streams(record)
    out = []
    for k in (1, 2):
        logs = [ObservationLog(0), ObservationLog(1)]
        try:
            replay(record, logs, creation_seed=record.core_seed[0] + 7919 * k).close()
        except _Timeout:
            raise
        except Exception:  # noqa: BLE001  (the game went differently: responses no longer fit)
            pass
        other = [log.items for log in logs]
        for viewer in (0, 1):
            a, b = base[viewer], other[viewer]
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    ctx = [it[1].get("type") if isinstance(it[1], dict) else it[0] for it in a[max(0, i - 4):i]]
                    out.append((seed, viewer, i, _describe(x, y), ctx))
                    break
            else:
                if len(a) != len(b):
                    out.append((seed, viewer, min(len(a), len(b)), "length differs", None))
    return out


def _describe(x, y) -> str:
    kind = x[0]
    t = x[1].get("type") if isinstance(x[1], dict) else kind
    if kind == "prompt" and x[1] == y[1]:
        t += " (state view)"
    return f"{kind} {t}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--pairs", help="agent pairings as in check_belief.py, e.g. setter-greedy,mixed-random")
    ap.add_argument("--decks", choices=["random", "diverse", "cards"], default="random")
    ap.add_argument("--book", help="opening book (scripts/make_book.py)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    pairs = tuple(tuple(p.split("-")) for p in args.pairs.split(",")) if args.pairs else None
    kinds = collections.Counter()
    games = leaky = 0
    failed = []
    tasks = [(seed, pairs, args.decks, args.book) for seed in range(args.start, args.start + args.n)]
    for (seed, _, _, _), res in run_parallel(diff, tasks, os.cpu_count()):
        if not isinstance(res, list):
            failed.append((seed, res))
            print(seed, "check failed:", res, flush=True)
            continue
        games += 1
        if res:
            leaky += 1
        for seed, viewer, i, what, ctx in res:
            kinds[what] += 1
            if args.verbose:
                print(seed, viewer, i, what, ctx, flush=True)
    print(f"{games} games, {leaky} with an observable effect of creation order"
          + (f" ({len(failed)} could not be checked)" if failed else ""))
    for k, v in kinds.most_common():
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
