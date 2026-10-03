#!/usr/bin/env python3
"""Debugging aid: which hidden card or copy a failed belief replay has wrong.

Replays without repairs, with the core's read log on (OCG_ExtReadLog), and lists
the hidden cards the engine read in the last items before the first divergence:
the cards whose identity can have decided it. Then tries, one at a time (with the
usual repairs), a swap of such a card with another copy of it in the same place
and other codes for it (one per combination of the properties the engine read),
following a candidate that gets further for a few rounds. A candidate that makes
the replay consistent names the wrong card.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai import constants as C  # noqa: E402
from ygoai import core as core_mod  # noqa: E402
from ygoai.belief.check import cut_point, make_game, observation_stream  # noqa: E402
from ygoai.belief.replay import BeliefReplay, Divergence  # noqa: E402
from ygoai.belief.tracker import TrackerError  # noqa: E402
from ygoai.carddb import default_db  # noqa: E402
from ygoai.game import init_info  # noqa: E402

H, D, X, M, S = C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA, C.LOCATION_MZONE, C.LOCATION_SZONE
READS = {1: "code", 2: "type", 4: "level", 8: "rank", 16: "attribute", 32: "race", 64: "atk", 128: "def",
         256: "archetype", 512: "link", 1024: "flag", 2048: "effect", 4096: "count limit", 16384: "history"}
PROPERTIES = 0x3ff

_log = {"on": False, "pos": None, "steps": None}
_process = core_mod.Duel.process


def _logged_process(self):
    """Duel.process that records (stream position, reads, card locations) per step."""
    if not _log["on"]:
        return _process(self)
    steps = _log["steps"].setdefault(self._key, [])
    if not steps:
        self.read_log(True)
    status = _process(self)
    cards = self.all_cards()[0]
    where = {cid: (p, loc, i, code) for (p, loc), lst in cards.items() for i, (code, cid) in enumerate(lst) if cid}
    steps.append((_log["pos"](), self.read_log(True), where))
    return status


core_mod.Duel.process = _logged_process


def replay(rec, viewer, items, swaps, assumptions, restarts, log=False):
    """(divergence position or None, the replay, the main duel's logged steps)."""
    br = BeliefReplay(viewer, init_info(rec, viewer), items, max_restarts=restarts)
    br.swaps, br.assumptions = dict(swaps), list(assumptions)
    _log.update(on=log, pos=lambda: br.pos, steps={})
    try:
        br.run().close()
        pos = None
    except Divergence as e:
        pos = e.pos
    except TrackerError:
        pos = -1
    finally:
        _log["on"] = False
    steps = max(_log["steps"].values(), key=len) if _log["steps"] else []
    return pos, br, steps


def read_cards(steps, pos, opp, window):
    """{card id: (mask of what was read, stream position, location before that step)} of the
    hidden cards read in the last `window` items up to `pos`."""
    out, before = {}, {}
    for p, reads, where in steps:
        if pos - window <= p <= pos:
            for cid, mask in reads.items():
                loc = before.get(cid) or where.get(cid)
                if loc and (loc[1] in (H, D, X) or (loc[0] == opp and loc[1] in (M, S))):
                    old = out.get(cid, (0,))[0]
                    out[cid] = (old | mask, p, loc)
        before = where
    return out


def candidates(steps, read, opp, codes, db):
    def signature(code, mask):
        c = db.cards.get(code)
        if c is None:
            return None
        parts = [v for bit, v in ((2, c.type), (4, c.level), (16, c.attribute), (32, c.race), (64, c.attack),
                                  (128, c.defense), (256, tuple(c.setcodes))) if mask & bit]
        return tuple(parts) or (c.type, c.level, c.attribute, c.race)
    per_card = []
    for cid in sorted(read, key=lambda c: (not read[c][0] & PROPERTIES, -read[c][1])):
        mask, p, (con, loc, i, code) = read[cid]
        mine = []
        where = next((w for q, _, w in reversed(steps) if q < p), {})
        arrived = next((q for q, _, w in steps if q <= p and w.get(cid, (None, None))[:2] == (con, loc)), p)
        for oid, (c2, l2, i2, code2) in where.items():
            if oid != cid and (c2, l2, code2) == (con, loc, code):
                for at in dict.fromkeys((p, arrived)):
                    mine.append(({(at, 0): [(cid, oid)]}, [], f"swap the copies of {code} at P{con} loc {loc}"
                                                             f" seq {i} and {i2} at item {at}"))
        if con == opp and loc in (H, D, M, S):
            seen = {signature(code, mask)}
            for new in codes:
                sig = signature(new, mask)
                if new != code and sig is not None and sig not in seen:
                    seen.add(sig)
                    mine.append(({}, [(p, con, loc, i, new)], f"P{con} loc {loc} seq {i} at item {p}: "
                                                              f"{code} -> {new}"))
                if len(mine) >= 8:
                    break
        per_card.append(mine)
    return [lst[k] for k in range(max((len(x) for x in per_card), default=0)) for lst in per_card if k < len(lst)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("viewer", type=int)
    ap.add_argument("--cut", type=float, default=1.0, help="fraction of the game, or -N: the end of turn N")
    ap.add_argument("--pairs", help="agent pairings as in check_belief.py")
    ap.add_argument("--decks", choices=["random", "diverse", "cards"], default="random")
    ap.add_argument("--book", help="opening book (scripts/make_book.py)")
    ap.add_argument("--window", type=int, default=5, help="items before the divergence whose reads count")
    ap.add_argument("--candidates", type=int, default=20, help="candidates tried per round")
    ap.add_argument("--rounds", type=int, default=3, help="candidates that get further are followed this often")
    args = ap.parse_args()
    rec = make_game(args.seed, pairs=[tuple(p.split("-")) for p in args.pairs.split(",")] if args.pairs else None,
                    decks=args.decks, book=args.book)
    items = observation_stream(rec, args.viewer)
    cut = cut_point(items, args.cut)
    if cut is None:
        sys.exit(f"the game has no cut {args.cut}")
    items = items[:cut]
    opp, db = 1 - args.viewer, default_db()
    swaps, assumptions, path = {}, [], []
    for _ in range(args.rounds):
        pos, br, steps = replay(rec, args.viewer, items, swaps, assumptions, 0, log=True)
        if pos is None or pos < 0:
            print("no divergence without repairs" if pos is None else "the hypothesis is infeasible")
            return
        read = read_cards(steps, pos, opp, args.window)
        print(f"first divergence at item {pos}; hidden cards read in the {args.window} items before:")
        shown = sorted(read.items(), key=lambda x: (not x[1][0] & (PROPERTIES | 2048), -x[1][1]))
        for cid, (mask, p, (con, loc, i, code)) in shown[:15]:
            print(f"  card {cid} {code} P{con} loc {loc} seq {i} (item {p}): "
                  + ", ".join(name for bit, name in READS.items() if mask & bit))
        if len(shown) > 15:
            print(f"  ... and {len(shown) - 15} more")
        seen = [m["code"] for it in items if it[0] == "msg" for m in [it[1]] if m.get("code")]
        hyp = sorted(set(br.hypothesis.decks[opp]["main"])) if br.hypothesis else []
        codes = list(dict.fromkeys(seen + hyp))
        best = None
        for sw, asm, what in candidates(steps, read, opp, codes, db)[:args.candidates]:
            new_swaps = dict(swaps)
            new_swaps.update(sw)
            res, _, _ = replay(rec, args.viewer, items, new_swaps, assumptions + asm, 40)
            print(f"  {what}: {'CONSISTENT' if res is None else f'diverges at {res}'}", flush=True)
            if res is None:
                print("repair:", " + ".join(path + [what]))
                return
            if res > pos and (best is None or res > best[0]):
                best = (res, new_swaps, assumptions + asm, what)
        if best is None:
            break
        _, swaps, assumptions, what = best
        path.append(what)
        print(f"following: {what}")
    print("no repair found")


if __name__ == "__main__":
    main()
