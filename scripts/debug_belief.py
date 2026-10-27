#!/usr/bin/env python3
"""Debugging aid: runs one belief replay and prints the context of a divergence.

Shows the repairs made, the divergence (both sides and where they first differ) and the real game
(god view) around it, annotated with the viewer's stream position.
"""

import argparse
import re
import sys

from common.games import add_game_options, play_game
from common.pretty import msg_str, prompt_str


def show(x):
    if isinstance(x, dict) and "type" in x:
        if x["type"].startswith(("SELECT", "SORT", "ANNOUNCE")):
            return ">> " + prompt_str(x)
        return msg_str(x)
    return str(x)[:400]


def print_god_view(record: dict, viewer: int, lo: int, hi: int) -> None:
    """The real game's events (god view) where the viewer's stream is between `lo` and `hi`,
    annotated with the viewer's stream position."""
    import ygoai_rs
    events, positions = ygoai_rs.replay_events(record, viewer)
    for event, pos in zip(events, positions):
        if lo <= pos <= hi:
            kind = event[0]
            text = (msg_str(event[1]) if kind == "msg" else ">> " + prompt_str(event[1]) if kind == "prompt"
                    else f"<< P{event[1]} {event[2]}")
            print(f"  [{pos:5d}] {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("viewer", type=int)
    ap.add_argument("--cut", type=float, default=1.0, help="fraction of the game, or -N: the end of turn N")
    ap.add_argument("--context", type=int, default=25)
    ap.add_argument("--after", type=int, default=5)
    ap.add_argument("--no-truth", action="store_true", help="do not print the real game's events")
    add_game_options(ap)
    ap.add_argument("--completion", choices=["vanilla", "corpus", "neural"], default="corpus")
    args = ap.parse_args()
    import ygoai_rs
    rec, streams = play_game(args.seed, args.mirror_prob, args.pairs, args.decks, args.book)
    print(rec["decks"][0]["name"], "vs", rec["decks"][1]["name"], rec["agents"], rec["result"])
    items = streams[args.viewer]
    cut = ygoai_rs.cut_point(items, args.cut)
    if cut is None:
        sys.exit(f"the game has no cut {args.cut}")
    r = ygoai_rs.check_prefix(rec, args.viewer, items, cut, args.completion)
    for text in r["repairs"]:
        print("  ..", text[:220])
    if r["ok"]:
        print(f"OK ({r['restarts']} restarts, {r['probes']} probes, fixes {r['fixes']}, {r['seconds']:.1f}s)")
        return
    print("DIVERGENCE:", r["error"])
    details = r["details"]
    for side in ("expected", "got"):
        if details.get(side):
            print(f"  {side + ':':9s}", show(details[side])[:600])
    if details.get("first_difference"):
        print("  first difference:", details["first_difference"])
    m = re.search(r"divergence at item (\d+)", r["error"])
    if m and not args.no_truth:
        pos = int(m.group(1))
        print("\n-- real game around the divergence (viewer stream position in brackets):")
        print_god_view(rec, args.viewer, pos - args.context, pos + args.after)


if __name__ == "__main__":
    main()
