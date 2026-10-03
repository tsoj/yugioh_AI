"""Last-resort repair of a belief replay: the hidden card the engine read.

When the replay's own repairs give up, the card whose identity is wrong is
usually one the engine read just before the divergence: a hidden card of the
opponent (another code) or a copy of a card that another copy should be (copies
differ in history: which one was revealed, drawn, used). The core's read log
(OCG_ExtReadLog) names the hidden cards the engine looked at, and the properties
it read; the candidates are swaps with same-name copies in the same place and
other codes that differ in those properties. Each candidate is tried as a whole
replay with the usual repairs; one that gets further is followed for a few rounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import constants as C
from .tracker import TrackerError

if TYPE_CHECKING:
    from ..core import Duel
    from .replay import BeliefReplay

H, D, X, M, S = C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_EXTRA, C.LOCATION_MZONE, C.LOCATION_SZONE
PROPERTIES = 0x3ff  # code, type, Level, Rank, attribute, race, ATK, DEF, archetypes, Link (see OCG_ExtReadLog)
WINDOW = 5  # items before the divergence whose reads count
CANDIDATES = 20  # replays per round
ROUNDS = 3
CODES_PER_CARD = 8


def _sub_replay(br: BeliefReplay, swaps: dict, assumptions: list, restarts: int) -> BeliefReplay:
    from .replay import BeliefReplay
    sub = BeliefReplay(br.viewer, br.init, br.items, db=br.db, completion=br.completion,
                       check_state=br.check_state, max_probes=br.max_probes, probe_items=br.probe_items,
                       max_depth=br.max_depth, max_restarts=restarts, probe_timeout=br.probe_timeout,
                       max_stall=br.max_stall, min_progress=br.min_progress, stopped=br.stopped)
    sub.read_guided = False
    sub.swaps = {k: list(v) for k, v in swaps.items()}
    sub.assumptions = list(assumptions)
    return sub


def _logged_run(br: BeliefReplay, swaps: dict, assumptions: list) -> tuple[int | None, list, BeliefReplay]:
    """A replay without repairs, logging per engine step (stream position, the
    card properties read, where each card was)."""
    from .replay import Divergence
    sub = _sub_replay(br, swaps, assumptions, 0)
    steps: list = []

    def note(duel):
        if sub.depth == 0:  # (not the probes' steps)
            steps.append((sub.pos, duel.read_log(True), _where(duel)))
    sub.on_process = note
    try:
        sub._run().close()
        return None, steps, sub
    except Divergence as e:
        return e.pos, steps, sub
    except TrackerError:
        return -1, steps, sub


def _where(duel: Duel) -> dict:
    cards = duel.all_cards()[0]
    return {cid: (p, loc, i, code) for (p, loc), lst in cards.items() for i, (code, cid) in enumerate(lst) if cid}


def _candidates(br: BeliefReplay, steps: list, pos: int, codes: list[int]) -> list[tuple]:
    opp = br.opponent
    read, before = {}, {}
    for p, reads, where in steps:
        if pos - WINDOW <= p <= pos:
            for cid, mask in reads.items():
                loc = before.get(cid) or where.get(cid)  # (where it was when read: the step may move it)
                if loc and (loc[1] in (H, D, X) or (loc[0] == opp and loc[1] in (M, S))):
                    read[cid] = (read.get(cid, (0,))[0] | mask, p, loc)
        before = where

    def signature(code, mask):
        c = br.db.get(code)
        if c is None:
            return None
        parts = [v for bit, v in ((2, c.type), (4, c.level), (16, c.attribute), (32, c.race), (64, c.attack),
                                  (128, c.defense), (256, tuple(c.setcodes))) if mask & bit]
        return tuple(parts) or (c.type, c.level, c.attribute, c.race)

    per_card = []
    # (the cards whose properties were read first, the latest first)
    for cid in sorted(read, key=lambda c: (not read[c][0] & PROPERTIES, -read[c][1])):
        mask, p, (con, loc, i, code) = read[cid]
        mine = []
        where = next((w for q, _, w in reversed(steps) if q < p), {})
        arrived = next((q for q, _, w in steps if q <= p and w.get(cid, (None, None))[:2] == (con, loc)), p)
        for other, (c2, l2, _, code2) in where.items():
            if other != cid and (c2, l2, code2) == (con, loc, code):
                for at in dict.fromkeys((p, arrived)):
                    mine.append(({(at, 0): [(cid, other)]}, [],
                                 f"copies of {code} at P{con} loc {loc} swapped at item {at}"))
        if con == opp and loc in (H, D, M, S):
            seen = {signature(code, mask)}
            for new in codes:
                sig = signature(new, mask)
                if new != code and sig is not None and sig not in seen:
                    seen.add(sig)
                    mine.append(({}, [(p, con, loc, i, new)], f"P{con} loc {loc} seq {i} at item {p}: {code} -> {new}"))
                if len(mine) >= CODES_PER_CARD:
                    break
        per_card.append(mine)
    # (breadth first: each card's first candidate, then the second ones, ...)
    return [lst[k] for k in range(max((len(x) for x in per_card), default=0)) for lst in per_card if k < len(lst)]


def read_guided_repair(br: BeliefReplay) -> Duel | None:
    """A duel reproducing the stream after a read-guided repair, or None."""
    from .replay import Divergence
    swaps = {k: list(v) for k, v in br.swaps.items()}
    assumptions = list(br.assumptions)
    # (codes the opponent's cards showed first, then the hypothesis's deck)
    seen = []
    for it in br.items:
        if it[0] == "msg" and it[1].get("code"):
            m = it[1]
            con = m.get("con", m.get("player", (m.get("from") or {}).get("con")))
            if con == br.opponent:
                seen.append(m["code"])
    seen = list(dict.fromkeys(seen))
    path: list[str] = []  # (the repairs followed so far)
    for _ in range(ROUNDS):
        pos, steps, first = _logged_run(br, swaps, assumptions)
        if pos is None or pos < 0:
            return None  # (consistent without repairs, or no hypothesis: nothing to add here)
        hyp = first.hypothesis
        codes = seen + sorted(set(hyp.decks[br.opponent]["main"])) if hyp is not None else seen
        best = None
        for sw, asm, what in _candidates(br, steps, pos, list(dict.fromkeys(codes)))[:CANDIDATES]:
            new_swaps = {k: list(v) for k, v in swaps.items()}
            for k, v in sw.items():
                new_swaps.setdefault(k, []).extend(v)
            sub = _sub_replay(br, new_swaps, assumptions + asm, br.max_restarts)
            br.stats.read_guided += 1
            try:
                duel = sub._run()
            except TrackerError:
                continue
            except Divergence as e:
                if e.pos > pos and (best is None or e.pos > best[0]):
                    best = (e.pos, new_swaps, assumptions + asm, what)
                continue
            br.stats.restarts += sub.stats.restarts
            br.hypothesis, br.swaps, br.assumptions = sub.hypothesis, new_swaps, assumptions + asm
            br.read_guided_fix = path + [what]
            return duel
        if best is None:
            return None
        _, swaps, assumptions, what = best
        path.append(what)
    return None
