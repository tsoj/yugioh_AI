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

from common.games import add_game_options, play_game
from common.parallel import run_parallel


def diff(task: tuple, timeout: int = 360) -> list[tuple] | str:
    """Differences for (seed, mirror_prob, pairs, decks, book), or a string if the check itself
    failed."""
    # (a loop inside the engine never returns to Python: the watchdog ends the worker, which
    # run_parallel reports as a crash of this seed)
    faulthandler.dump_traceback_later(timeout, exit=True)
    try:
        return _diff(*task)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    finally:
        faulthandler.cancel_dump_traceback_later()


def _diff(seed: int, mirror_prob: float, pairs, decks: str, book) -> list[tuple]:
    import bot_py_binding
    record, _ = play_game(seed, mirror_prob, pairs, decks, book)
    base, error = bot_py_binding.replay_streams(record)
    if error:
        raise RuntimeError(error)
    out = []
    for k in (1, 2):
        # (a replay that stops early: the game went differently, the responses no longer fit)
        other, _ = bot_py_binding.replay_streams(record, record["core_seed"][0] + 7919 * k)
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
    add_game_options(ap)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    kinds = collections.Counter()
    games = leaky = 0
    failed = []
    tasks = [(seed, args.mirror_prob, args.pairs, args.decks, args.book) for seed in range(args.start, args.start + args.n)]
    for (seed, *_), res in run_parallel(diff, tasks, os.cpu_count()):
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
