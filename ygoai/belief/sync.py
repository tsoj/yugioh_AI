"""Re-synchronizing an approximate reconstruction with what its viewer observed.

When no consistent reconstruction exists, the lenient pass replays the best
hypothesis while re-aligning with the stream (BeliefReplay._lenient_run); after
an early difference its duel drifts away from the real one (cards the replay put
elsewhere, codes it guessed wrong, Life Points). At the viewer's Idle and Battle
commands the stream holds what the viewer saw there; `sync_to_view` moves the
engine's cards to match it (silently, see OCG_ExtPlaceCard), gives guessed cards
the codes that showed, and sets the Life Points. Only the viewer's observation is
used, so nothing leaks; hidden cards stay where they are when they can.
"""

from __future__ import annotations

from .. import constants as C
from ..state import LOCATIONS, PRIVATE_STATUS

_ZONES = C.LOCATION_MZONE | C.LOCATION_SZONE
_EXTRA_TYPES = C.EXTRA_DECK_TYPES
_NAME = {loc: name for name, loc in LOCATIONS}
_STATUS_MASK = 0xFFFFFFFF & ~PRIVATE_STATUS  # (the status bits a view shows)


def sync_to_view(duel, viewer: int, obs: dict, got: dict, db) -> int:
    """Changes the duel's state toward the observed view `obs` (`got`: the duel's
    current view of the viewer). Returns how many changes it made."""
    lists, _ = duel.all_cards()
    place: dict[int, tuple] = {}  # card id -> (player, location, index)
    code_of: dict[int, int] = {}
    view_of: dict[int, dict] = {}
    for p in (0, 1):
        for name, loc in LOCATIONS:
            for i, (code, cid) in enumerate(lists[p, loc]):
                if cid:
                    place[cid], code_of[cid] = (p, loc, i), code
                    view_of[cid] = got["players"][p][name][i]

    def is_extra(cid):
        card = db.get(code_of[cid])
        return card is not None and bool(card.type & _EXTRA_TYPES)

    def is_token(code):
        card = db.get(code)
        return card is not None and bool(card.type & C.TYPE_TOKEN)

    def owner(cid):
        return view_of[cid].get("owner", place[cid][0])

    slots = [(p, loc, i, e) for p in (0, 1) for name, loc in LOCATIONS
             for i, e in enumerate(obs["players"][p][name]) if e is not None]
    target: dict[tuple, int] = {}  # (player, location, index) -> card id
    used: set[int] = set()

    def engine_at(p, loc, i):
        lst = lists[p, loc]
        return lst[i][1] if i < len(lst) and lst[i][1] else None

    def assign(slot, cid):
        target[slot] = cid
        used.add(cid)

    # 1. cards already where they showed, as they showed
    for p, loc, i, e in slots:
        cid = engine_at(p, loc, i)
        if "code" in e and cid is not None and view_of[cid] == e:
            assign((p, loc, i), cid)
    # 2. the other visible cards: a copy the engine has elsewhere, else a hidden
    # card of their owner that becomes this code
    recreate: dict[int, int] = {}
    for p, loc, i, e in slots:
        if "code" not in e or (p, loc, i) in target:
            continue
        code, own = e["code"], e.get("owner", p)
        cands = [cid for cid in place if cid not in used and code_of[cid] == code and owner(cid) == own]
        if cands:
            # (the one in the same pile first, then one in a place it did not show, then a hidden one)
            def rank(cid):
                q, ql, _ = place[cid]
                return ((q, ql) != (p, loc), "code" in view_of[cid], place[cid])
            assign((p, loc, i), min(cands, key=rank))
            continue
        if is_token(code):
            cid = duel.create_card(own, code)  # (a token the replay did not make)
            place[cid], code_of[cid], view_of[cid] = (own, 0, -1), code, {"owner": own}
            assign((p, loc, i), cid)
            continue
        if own == viewer or db.get(code) is None:
            continue  # (the viewer's own cards are known: no guess to correct)
        extra = bool(db.get(code).type & _EXTRA_TYPES)
        pool = [cid for cid in place if cid not in used and owner(cid) == own and "code" not in view_of[cid]
                and place[cid][1] & (C.LOCATION_DECK | C.LOCATION_HAND | C.LOCATION_EXTRA)
                and is_extra(cid) == extra and not is_token(code_of[cid])]
        if pool:
            # (from the deck first: a guessed card in the hand stays there)
            cid = min(pool, key=lambda c: (place[c][1] != (C.LOCATION_EXTRA if extra else C.LOCATION_DECK), place[c]))
            recreate[cid] = code
            assign((p, loc, i), cid)
    # 3. hidden cards: the one already there, else one of their owner's
    for p, loc, i, e in slots:
        if "code" in e or (p, loc, i) in target:
            continue
        cid = engine_at(p, loc, i)
        if cid is not None and cid not in used and view_of[cid] == e:
            assign((p, loc, i), cid)
    for p, loc, i, e in slots:
        if "code" in e or (p, loc, i) in target:
            continue
        own = e.get("owner", p)
        want_extra = loc == C.LOCATION_EXTRA
        pool = [cid for cid in place if cid not in used and owner(cid) == own and not is_token(code_of[cid])
                and (is_extra(cid) == want_extra or not loc & (C.LOCATION_DECK | C.LOCATION_HAND | C.LOCATION_EXTRA))]
        if pool:
            # (the same pile first, then a card the view did not place, then the deck)
            def rank(cid):
                q, ql, _ = place[cid]
                return ((q, ql) != (p, loc), (q, ql) != (own, C.LOCATION_DECK), place[cid])
            assign((p, loc, i), min(pool, key=rank))

    # what moves: assigned cards in another place, or (zones) in another position
    moves = []
    for (p, loc, i), cid in target.items():
        e = obs["players"][p][_NAME[loc]][i]
        if place[cid] != (p, loc, i) or (loc & _ZONES and e.get("position") != view_of[cid].get("position")) \
                or cid in recreate:
            moves.append(((p, loc, i), cid, e))
    # cards no slot took that sit in a zone the view shows empty or another card in
    taken = {(p, loc, i) for p, loc, i in target if loc & _ZONES}
    stray = [cid for cid in place if cid not in used and place[cid][1] & _ZONES
             and (place[cid] in taken or obs["players"][place[cid][0]][_NAME[place[cid][1]]]
                  [place[cid][2]] is None)]
    changes = 0
    # leave the zones first (a card's zone can be another's target)
    for (p, loc, i), cid, e in moves:
        if place[cid][1] & _ZONES and place[cid] != (p, loc, i):
            duel.place_card(cid, owner(cid), C.LOCATION_DECK, 1, C.POS_FACEDOWN_DEFENSE)
    for cid in stray:
        # (a token the viewer did not see leaves play)
        duel.place_card(cid, owner(cid), 0 if is_token(code_of[cid]) else C.LOCATION_DECK, 1, C.POS_FACEDOWN_DEFENSE)
        changes += 1
    for cid, code in recreate.items():
        duel.lua_messages(f"local c=Duel.GetCardFromCardID({cid}) if c then "
                          f"c:Recreate({code},nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,true) end")
    for (p, loc, i), cid, e in sorted(moves, key=lambda m: m[0]):
        pos = e.get("position", C.POS_FACEDOWN_DEFENSE)
        seq = i if loc & _ZONES else 0
        if duel.place_card(cid, p, loc, seq, pos):
            changes += 1
    # what the cards show of their history: why they are there, their status
    moved = {cid for _, cid, _ in moves}
    for (p, loc, i), cid in target.items():
        e = obs["players"][p][_NAME[loc]][i]
        v = view_of[cid]
        reason = e["reason"] if "reason" in e and (cid in moved or e["reason"] != v.get("reason")) else None
        status = "status" in e and (cid in moved or e["status"] != v.get("status"))
        if reason is not None or status:
            duel.set_card_info(cid, reason, e.get("status", 0), _STATUS_MASK if status else 0)
            changes += 1
    # the order of the piles the viewer sees in order
    for p in (0, 1):
        for name, loc in LOCATIONS:
            if loc & (_ZONES | C.LOCATION_DECK):
                continue
            want = [target.get((p, loc, i)) for i in range(len(obs["players"][p][name]))]
            if None in want:
                continue
            now = [cid for _, cid in duel.location_cards(p, loc) if cid]
            rest = [cid for cid in now if cid not in set(want)]
            if now != want + rest and sorted(now) == sorted(want + rest):
                duel.set_location_order_by_id(p, loc, want + rest)
    if obs["turn"] != got["turn"]:
        duel.set_turn(obs["turn"])
        changes += 1
    for p in (0, 1):
        if obs["lp"][p] != got["lp"][p]:
            duel.set_lp(p, obs["lp"][p])
            changes += 1
    return changes
