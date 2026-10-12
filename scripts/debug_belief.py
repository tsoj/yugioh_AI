#!/usr/bin/env python3
"""Debugging aid: runs one belief replay and prints the context of a divergence.

Shows the repairs made, the divergence (both sides and where they first differ) and the real game
(god view) around it, annotated with the viewer's stream position.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.belief.check import check_prefix, cut_point, make_game, observation_stream  # noqa: E402
from ygoai.game import ObservationLog, Observer, replay  # noqa: E402
from ygoai.pretty import msg_str, prompt_str  # noqa: E402


def show(x):
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
    ap.add_argument("--no-truth", action="store_true", help="do not print the real game's events")
    ap.add_argument("--pairs", help="agent pairings as in check_belief.py, e.g. setter-greedy,mixed-random")
    ap.add_argument("--decks", choices=["random", "diverse", "cards", "new", "new-cards", "new-mixed"], default="random")
    ap.add_argument("--book", help="opening book (scripts/make_book.py)")
    ap.add_argument("--completion", choices=["vanilla", "corpus", "neural"], default="corpus")
    args = ap.parse_args()
    rec = make_game(args.seed, pairs=[tuple(p.split("-")) for p in args.pairs.split(",")] if args.pairs else None,
                    decks=args.decks, book=args.book)
    print(rec.decks[0].name, "vs", rec.decks[1].name, rec.agents, rec.result)
    items = observation_stream(rec, args.viewer)
    cut = cut_point(items, args.cut)
    if cut is None:
        sys.exit(f"the game has no cut {args.cut}")
    completion = None  # (the replay's own default: vanilla)
    if args.completion == "corpus":
        from ygoai.belief.corpus import CorpusCompletion
        completion = CorpusCompletion()
    elif args.completion == "neural":
        from ygoai.belief.neural_completion import NeuralCompletion
        completion = NeuralCompletion()
    r = check_prefix(rec, args.viewer, items, cut, completion=completion)
    for text in r.repairs:
        print("  ..", text[:220])
    if r.ok:
        print(f"OK ({r.restarts} restarts, {r.probes} probes, fixes {r.fixes}, {r.seconds:.1f}s)")
        return
    print("DIVERGENCE:", r.error)
    for side in ("expected", "got"):
        if r.details.get(side):
            print(f"  {side + ':':9s}", show(r.details[side])[:600])
    if r.details.get("first_difference"):
        print("  first difference:", r.details["first_difference"])
    m = re.search(r"divergence at item (\d+)", r.error)
    if m and not args.no_truth:
        pos = int(m.group(1))
        print("\n-- real game around the divergence (viewer stream position in brackets):")
        replay(rec, [GodPrinter(args.viewer, pos - args.context, pos + args.after)]).close()


if __name__ == "__main__":
    main()
