#!/usr/bin/env python3
"""Debugging aid: runs one belief replay and prints the context of a divergence.

Shows the repairs made, the divergence, the real game (god view) and the
reconstruction's own god view events around the divergence, both annotated
with the viewer's stream position.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.belief.check import cut_point, make_game, observation_stream  # noqa: E402
from ygoai.belief.replay import BeliefReplay, Divergence, first_difference  # noqa: E402
from ygoai.game import ObservationLog, Observer, init_info, replay  # noqa: E402
from ygoai.pretty import msg_str, prompt_str  # noqa: E402


def show(x):
    if isinstance(x, tuple):
        if x[0] == "msg":
            return msg_str(x[1])
        if x[0] == "prompt":
            return ">> " + prompt_str(x[1])
        return f"<< {x[1]}"
    if isinstance(x, dict) and "type" in x:
        if x["type"].startswith(("SELECT", "SORT", "ANNOUNCE")):
            return ">> " + prompt_str(x)
        return msg_str(x)
    return str(x)[:400]


class GodPrinter(Observer):
    """Prints god-view events annotated with the viewer's stream index."""

    def __init__(self, viewer, lo, hi):
        self.log = ObservationLog(viewer, with_state=False)
        self.lo, self.hi = lo, hi

    def _p(self, s):
        if self.lo <= len(self.log.items) <= self.hi:
            print(f"  [{len(self.log.items):5d}] {s}")

    def message(self, m):
        self._p(msg_str(m))
        self.log.message(m)

    def prompt(self, p, duel):
        self._p(">> " + prompt_str(p))
        self.log.prompt(p, duel)

    def response(self, player, r):
        self._p(f"<< P{player} {r}")
        self.log.response(player, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("viewer", type=int)
    ap.add_argument("--cut", type=float, default=1.0, help="fraction of the game, or -N: the end of turn N")
    ap.add_argument("--context", type=int, default=25)
    ap.add_argument("--after", type=int, default=5)
    ap.add_argument("--no-sim", action="store_true", help="do not print the reconstruction's events")
    ap.add_argument("--no-truth", action="store_true", help="do not print the real game's events")
    ap.add_argument("--max-restarts", type=int, help="stop after this many repairs (to inspect one attempt)")
    ap.add_argument("--pairs", help="agent pairings as in check_belief.py, e.g. setter-greedy,mixed-random")
    ap.add_argument("--decks", choices=["random", "diverse", "cards", "new", "new-cards", "new-mixed"], default="random")
    ap.add_argument("--book", help="opening book (scripts/make_book.py)")
    ap.add_argument("--completion", choices=["vanilla", "corpus"], default="vanilla")
    args = ap.parse_args()
    rec = make_game(args.seed, pairs=[tuple(p.split("-")) for p in args.pairs.split(",")] if args.pairs else None,
                    decks=args.decks, book=args.book)
    print(rec.decks[0].name, "vs", rec.decks[1].name, rec.agents, rec.result)
    items = observation_stream(rec, args.viewer)
    cut = cut_point(items, args.cut)
    if cut is None:
        sys.exit(f"the game has no cut {args.cut}")
    completion = None
    if args.completion == "corpus":
        from ygoai.belief.corpus import CorpusCompletion
        completion = CorpusCompletion()
    br = BeliefReplay(args.viewer, init_info(rec, args.viewer), items[:cut], completion=completion)
    if args.max_restarts is not None:
        br.max_restarts = args.max_restarts
    br.trace = []
    br.sim_log = []
    br.log = lambda text: print("  ..", text[:220], flush=True)
    try:
        br.run().close()
        print("OK", br.stats)
        return
    except Divergence as e:
        print("DIVERGENCE:", e)
        pos = e.pos
        if e.expected is not None:
            print("  expected:", show(e.expected)[:600])
        if e.got is not None:
            print("  got:     ", show(e.got)[:600])
        exp = e.expected[1] if isinstance(e.expected, tuple) else e.expected
        if isinstance(exp, dict) and isinstance(e.got, dict):
            print("  first difference:", first_difference(exp, e.got))
    if not args.no_truth:
        print("\n-- real game around the divergence (viewer stream position in brackets):")
        replay(rec, [GodPrinter(args.viewer, pos - args.context, pos + args.after)]).close()
    if not args.no_sim:
        print("\n-- reconstruction (last attempt) around the divergence:")
        for entry in br.sim_log:
            p = entry[0]
            if pos - args.context <= p <= pos + args.after:
                if entry[1] == "msg":
                    print(f"  [{p:5d}] {msg_str(entry[2])}   cp={entry[3]}")
                else:
                    print(f"  [{p:5d}] >> {prompt_str(entry[2])}")
                    print(f"  [{p:5d}] << P{entry[2]['player']} {entry[3]}")


if __name__ == "__main__":
    main()
