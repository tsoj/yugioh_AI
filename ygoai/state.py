"""Full (god view) state snapshots of a duel and their per-player projection."""

from __future__ import annotations

import struct

from . import constants as C
from .core import Duel
from .observe import canon_pos

QUERY_ALL = (
    C.QUERY_CODE | C.QUERY_POSITION | C.QUERY_ALIAS | C.QUERY_TYPE | C.QUERY_LEVEL | C.QUERY_RANK
    | C.QUERY_ATTRIBUTE | C.QUERY_RACE | C.QUERY_ATTACK | C.QUERY_DEFENSE | C.QUERY_BASE_ATTACK
    | C.QUERY_BASE_DEFENSE | C.QUERY_REASON | C.QUERY_REASON_CARD | C.QUERY_EQUIP_CARD
    | C.QUERY_TARGET_CARD | C.QUERY_OVERLAY_CARD | C.QUERY_COUNTERS | C.QUERY_OWNER
    | C.QUERY_STATUS | C.QUERY_IS_PUBLIC | C.QUERY_LSCALE | C.QUERY_RSCALE | C.QUERY_LINK
    | C.QUERY_IS_HIDDEN | C.QUERY_COVER
)

# Cheaper flag set for piles where only identity and position matter.
QUERY_BASIC = C.QUERY_CODE | C.QUERY_POSITION | C.QUERY_OWNER | C.QUERY_IS_PUBLIC

LOCATIONS = [
    ("deck", C.LOCATION_DECK), ("hand", C.LOCATION_HAND), ("mzone", C.LOCATION_MZONE),
    ("szone", C.LOCATION_SZONE), ("grave", C.LOCATION_GRAVE), ("removed", C.LOCATION_REMOVED),
    ("extra", C.LOCATION_EXTRA),
]
LOC_NAME = {v: k for k, v in LOCATIONS}


def _loc_info(buf: bytes, pos: int) -> dict | None:
    con, loc, seq, cpos = struct.unpack_from("<BBII", buf, pos)
    if not loc:
        return None
    return {"con": con, "loc": loc, "seq": seq, "pos": cpos}


_HEADER = struct.Struct("<HI")
# simple scalar fields: flag -> (key, struct)
_SCALARS = {
    C.QUERY_CODE: ("code", struct.Struct("<I")),
    C.QUERY_POSITION: ("position", struct.Struct("<I")),
    C.QUERY_ALIAS: ("alias", struct.Struct("<I")),
    C.QUERY_TYPE: ("type", struct.Struct("<I")),
    C.QUERY_LEVEL: ("level", struct.Struct("<I")),
    C.QUERY_RANK: ("rank", struct.Struct("<I")),
    C.QUERY_ATTRIBUTE: ("attribute", struct.Struct("<I")),
    C.QUERY_RACE: ("race", struct.Struct("<Q")),
    C.QUERY_ATTACK: ("attack", struct.Struct("<i")),
    C.QUERY_DEFENSE: ("defense", struct.Struct("<i")),
    C.QUERY_BASE_ATTACK: ("base_attack", struct.Struct("<i")),
    C.QUERY_BASE_DEFENSE: ("base_defense", struct.Struct("<i")),
    C.QUERY_REASON: ("reason", struct.Struct("<I")),
    C.QUERY_OWNER: ("owner", struct.Struct("<B")),
    C.QUERY_STATUS: ("status", struct.Struct("<I")),
    C.QUERY_IS_PUBLIC: ("is_public", struct.Struct("<B")),
    C.QUERY_LSCALE: ("lscale", struct.Struct("<I")),
    C.QUERY_RSCALE: ("rscale", struct.Struct("<I")),
    C.QUERY_IS_HIDDEN: ("is_hidden", struct.Struct("<B")),
    C.QUERY_COVER: ("cover", struct.Struct("<I")),
}


def parse_card_query(buf: bytes, pos: int) -> tuple[dict | None, int]:
    """Parses one card query starting at `pos`; returns (card or None, new pos)."""
    if buf[pos] == 0 and buf[pos + 1] == 0:
        return None, pos + 2
    card: dict = {}
    header = _HEADER.unpack_from
    scalars = _SCALARS
    while True:
        size, flag = header(buf, pos)
        p = pos + 6
        pos += 2 + size
        sc = scalars.get(flag)
        if sc is not None:
            card[sc[0]] = sc[1].unpack_from(buf, p)[0]
        elif flag == C.QUERY_END:
            return card, pos
        elif flag == C.QUERY_REASON_CARD:
            card["reason_card"] = _loc_info(buf, p)
        elif flag == C.QUERY_EQUIP_CARD:
            card["equip_card"] = _loc_info(buf, p)
        elif flag == C.QUERY_TARGET_CARD:
            (n,) = struct.unpack_from("<I", buf, p)
            card["target_cards"] = [_loc_info(buf, p + 4 + 10 * i) for i in range(n)]
        elif flag == C.QUERY_OVERLAY_CARD:
            (n,) = struct.unpack_from("<I", buf, p)
            card["overlays"] = list(struct.unpack_from(f"<{n}I", buf, p + 4))
        elif flag == C.QUERY_COUNTERS:
            (n,) = struct.unpack_from("<I", buf, p)
            card["counters"] = list(struct.unpack_from(f"<{n}I", buf, p + 4))
        elif flag == C.QUERY_LINK:
            card["link"], card["link_marker"] = struct.unpack_from("<II", buf, p)


def parse_location_query(buf: bytes) -> list[dict | None]:
    if not buf:
        return []
    (total,) = struct.unpack_from("<I", buf, 0)
    pos = 4
    end = 4 + total
    out = []
    while pos < end:
        card, pos = parse_card_query(buf, pos)
        out.append(card)
    return out


def parse_field_query(buf: bytes) -> dict:
    """Parses OCG_DuelQueryField: we only keep the chain part (the rest is in the snapshot)."""
    pos = 4
    for _ in range(2):
        pos += 4
        for _ in range(7 + 8):
            present = buf[pos]
            pos += 1
            if present:
                pos += 1 + 4
        pos += 6 * 4
    (n,) = struct.unpack_from("<I", buf, pos)
    pos += 4
    chain = []
    for _ in range(n):
        code, hcon, hloc, hseq, hpos, tcon, tloc, tseq, desc = struct.unpack_from("<IBBIIBBIQ", buf, pos)
        pos += struct.calcsize("<IBBIIBBIQ")
        chain.append({"code": code, "handler": {"con": hcon, "loc": hloc, "seq": hseq, "pos": hpos},
                      "trigger_con": tcon, "trigger_loc": tloc, "trigger_seq": tseq, "desc": desc})
    return {"chain": chain}


# parsed location queries by their raw bytes (most locations are unchanged
# between two snapshots)
_parsed: dict[bytes, list[dict | None]] = {}


def _parse_location_cached(buf: bytes) -> list[dict | None]:
    cards = _parsed.get(buf)
    if cards is None:
        if len(_parsed) > 5000:
            _parsed.clear()
        cards = _parsed[buf] = parse_location_query(buf)
    # (fresh dicts: the caller adds the card ids; nested values are never changed)
    return [dict(c) if c is not None else None for c in cards]


def capture(duel: Duel) -> dict:
    """God view snapshot: every card of every location with full query data and card id."""
    info = duel.info()
    snap = {"turn": info["turn"], "turn_player": info["turn_player"], "phase": info["phase"],
            "lp": info["lp"], "players": []}
    all_ids, mats = duel.all_cards()
    for p in (0, 1):
        pl = {}
        for name, loc in LOCATIONS:
            flags = QUERY_BASIC if loc == C.LOCATION_DECK else QUERY_ALL
            cards = _parse_location_cached(duel.query_location_raw(p, loc, flags))
            ids = all_ids[p, loc]
            for i, card in enumerate(cards):
                if card is not None:
                    card["id"] = ids[i][1]
                    if card.get("overlays"):
                        card["overlay_ids"] = [cid for _, cid in mats.get((p, loc, i), ())]
            pl[name] = cards
        snap["players"].append(pl)
    snap.update(parse_field_query(duel.query_field_raw()))
    for link, related in zip(snap["chain"], duel.chain_related()):
        link["related"] = related
    return snap


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

PUBLIC_FIELDS = (
    "code", "position", "alias", "type", "level", "rank", "attribute", "race", "attack",
    "defense", "base_attack", "base_defense", "reason", "equip_card",
    "target_cards", "overlays", "counters", "owner", "status", "is_public", "lscale", "rscale",
    "link", "link_marker", "is_hidden", "cover",
)
HIDDEN_FIELDS = ("position", "owner", "counters", "equip_card", "target_cards", "overlays")
# internal status bits that record hidden decisions: choosing an attacker clears
# ATTACK_CANCELED even if the player then cancels the target selection
PRIVATE_STATUS = C.STATUS_ATTACK_CANCELED


# viewed locations by (raw query bytes, player, location, viewer): most locations
# are unchanged between two state checks
_viewed: dict[tuple, list] = {}


def viewer_view(duel: Duel, viewer: int) -> dict:
    """view(capture(duel), viewer), computed per location from the core's raw query
    (unchanged locations come from a cache; the card ids, which the view leaves
    out, are not looked up). The lists and dicts are shared: do not modify."""
    info = duel.info()
    snap = {"turn": info["turn"], "turn_player": info["turn_player"], "phase": info["phase"],
            "lp": info["lp"]}
    snap.update(parse_field_query(duel.query_field_raw()))
    for link, related in zip(snap["chain"], duel.chain_related()):
        link["related"] = related
    out = _view_header(snap, viewer)
    for p in (0, 1):
        vp = {}
        for name, loc in LOCATIONS:
            raw = duel.query_location_raw(p, loc, QUERY_BASIC if loc == C.LOCATION_DECK else QUERY_ALL)
            key = (raw, p, loc, viewer)
            cards = _viewed.get(key)
            if cards is None:
                if len(_viewed) > 5000:
                    _viewed.clear()
                cards = _viewed[key] = _view_location(parse_location_query(raw), p, loc, viewer)
            vp[name] = cards
        out["players"].append(vp)
    return out


def card_visible(card: dict, controller: int, location: int, viewer: int) -> bool:
    if location & (C.LOCATION_GRAVE | C.LOCATION_OVERLAY):
        return True
    if card.get("is_public") or (card.get("position", 0) & C.POS_FACEUP):
        return True
    if controller == viewer and location != C.LOCATION_DECK:
        return True
    return False


def _mask_ref(li: dict | None, viewer: int) -> dict | None:
    """References to cards in hidden piles must not reveal the card's position there."""
    if li is None:
        return None
    from .observe import loc_visible
    if loc_visible(li, viewer) and not li["loc"] & C.LOCATION_DECK:
        return li
    return {"con": li["con"], "loc": li["loc"], "seq": 0, "pos": canon_pos(li["loc"], li["pos"])}


def view(snap: dict, viewer: int) -> dict:
    """What `viewer` can see of a god view snapshot."""
    out = _view_header(snap, viewer)
    for p, pl in enumerate(snap["players"]):
        out["players"].append({name: _view_location(pl[name], p, loc, viewer) for name, loc in LOCATIONS})
    return out


def _view_header(snap: dict, viewer: int) -> dict:
    # (a link's card that left its place since, e.g. shuffled into the deck, is a
    # new object: which physical copy it is now cannot be known, even when a copy
    # of it is back on the field)
    chain = [{**{k: v for k, v in c.items() if k != "related"},
              "handler": _mask_ref(c["handler"], viewer) if c.get("related", True) else None}
             for c in snap["chain"]]
    return {"turn": snap["turn"], "turn_player": snap["turn_player"], "phase": snap["phase"],
            "lp": list(snap["lp"]), "chain": chain, "players": []}


def _view_location(pile: list, p: int, loc: int, viewer: int) -> list:
    """What `viewer` can see of player `p`'s cards in location `loc`."""
    cards = []
    for card in pile:
        if card is None:
            cards.append(None)
            continue
        if card_visible(card, p, loc, viewer):
            shown = {k: card[k] for k in PUBLIC_FIELDS if k in card}
            if not loc & (C.LOCATION_GRAVE | C.LOCATION_REMOVED):
                # cards in the hand or on the field keep the stale reason of
                # how they got there earlier; it would reveal hidden history
                # and make identical copies distinguishable
                shown.pop("reason", None)
            if not loc & (C.LOCATION_MZONE | C.LOCATION_SZONE):
                # status bits of cards in piles are stale leftovers from
                # earlier trips to the field; they would make identical
                # copies distinguishable after a shuffle
                shown.pop("status", None)
            elif "status" in shown:
                shown["status"] &= ~PRIVATE_STATUS
            # (reason_card is left out entirely: it points at whichever copy
            # caused the move, wherever that copy is hidden now)
            if "equip_card" in shown:
                shown["equip_card"] = _mask_ref(shown["equip_card"], viewer)
            if "target_cards" in shown:
                shown["target_cards"] = [_mask_ref(t, viewer) for t in shown["target_cards"]]
            cards.append(shown)
        else:
            hidden = {k: card[k] for k in HIDDEN_FIELDS if k in card}
            if "position" in hidden:
                hidden["position"] = canon_pos(loc, hidden["position"])
            if "equip_card" in hidden:
                hidden["equip_card"] = _mask_ref(hidden["equip_card"], viewer)
            if "target_cards" in hidden:
                hidden["target_cards"] = [_mask_ref(t, viewer) for t in hidden["target_cards"]]
            cards.append(hidden)
    return cards
