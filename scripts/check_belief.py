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
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.belief.games import add_game_options, agents, game_decks  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402

SEQUENCE_FRAC = 2.0  # (the "cut" of an agent sequence's result: the whole game, decision by decision)


def run_task(task: tuple) -> list[dict]:
    """All checks of one game (ygoai_rs.check_game: played once, then each case checked):
    (seed, [(viewer, frac), ...], agents, decks, completion, timeout, every, agent_mode)."""
    import ygoai_rs
    seed, cases, names, decks, completion, timeout, every, agent_mode = task
    try:
        out = ygoai_rs.check_game(seed, decks, names, cases, completion, timeout, every, agent_mode)
    except ValueError as e:
        out = [{"seed": seed, "viewer": v, "frac": f, "ok": False, "error": f"game failed: {e}"} for v, f in cases]
    # (give the memory of the game's replays back: a worker runs many games, and the heap would
    # otherwise stay at the largest replay's peak)
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
    ap.add_argument("--timeout", type=int, default=180, help="seconds per belief replay")
    ap.add_argument("--agent-mode", action="store_true",
                    help="reconstruct as an agent does: at the viewer's decisions, with the best effort "
                         "allowed (approximate states count separately, with their state differences)")
    ap.add_argument("--agent-sequence", type=int, metavar="K",
                    help="reconstruct as belief_mc does: at every K-th decision of the viewer, each inheriting "
                         "from the previous consistent one (one result per game and viewer: consistent if all "
                         "of them are; approximate ones are counted with their state differences)")
    ap.add_argument("--read-guided-ok", action="store_true",
                    help="count reconstructions the last-resort read-guided repair saved as consistent "
                         "(by default they count as failures, to be looked at)")
    ap.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("-o", "--output", type=Path, help="write all results as JSON lines")
    ap.add_argument("-v", "--verbose", action="store_true")
    add_game_options(ap)
    ap.add_argument("--completion", choices=["vanilla", "corpus", "neural"], default="corpus",
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
                 for frac in (sorted(cuts(seed, viewer), reverse=True) if not args.agent_sequence
                              else (SEQUENCE_FRAC,))]
    by_seed: dict[int, list] = {}
    for seed, viewer, frac in cases:
        by_seed.setdefault(seed, []).append((viewer, frac))
    # (the decks are chosen here: the workers then never load the deck corpus, which keeps them small)
    return [(seed, cs, agents(seed, args.pairs), [d.to_json() for d in game_decks(seed, args.mirror_prob, args.decks, args.book)],
             args.completion, args.timeout, args.agent_sequence or 0, args.agent_mode)
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
    approx = [r for r in results if r.get("approximate")]
    if args.agent_mode:
        diffs = sorted(r["state_diffs"] for r in approx)
        print(f"  agent mode: {len(approx)} approximate (state differences: "
              f"{'none' if not diffs else f'median {diffs[len(diffs) // 2]}, max {diffs[-1]}, exact {diffs.count(0)}'}), "
              f"{sum(1 for r in results if not r['ok'] and not r.get('approximate'))} without any state")
    if args.agent_sequence:
        seqs = [r for r in results if "reconstructions" in r]
        n = sum(r["reconstructions"] for r in seqs)
        print(f"  agent sequence (every {args.agent_sequence}. decision, inherited): {n} reconstructions, "
              f"{sum(r['consistent'] for r in seqs)} consistent, {sum(r['approximate_n'] for r in seqs)} approximate "
              f"(state differences {sum(r['diffs_total'] for r in seqs)}), {sum(r['failed'] for r in seqs)} without "
              f"any state, {sum(r.get('extended', 0) for r in seqs)} incremental, "
              f"{sum(r['retries'] for r in seqs)} fresh retries; "
              f"{1000 * sum(r['seconds'] for r in seqs) / max(1, n):.0f} ms per reconstruction")
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
