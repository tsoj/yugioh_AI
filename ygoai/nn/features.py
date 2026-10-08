"""Features for the policy and value networks (no card identities: only what a
card is, so a model carries over to cards it never saw).

A position is the full state of a duel (capture: in training the real game, in a
search the belief's), seen from the player to decide ("own" and "opponent"):

- its cards, each (static index, group, count, dynamic features): the static index
  points into a table of what the card is (static_features: types, attribute,
  race, level, base ATK/DEF; built from the card database, the same for every
  copy), the group is where it is (own hand, own monster zones, ..., the
  opponent's, the chain; the decks and Extra Decks as one row per distinct card
  with its count), the dynamic features how it is now (position, current
  ATK/DEF, overlays, counters, ...; stored as integers, see expand_dynamic);
- scalars (LP, phase, whose turn, pile sizes, ...).

A decision's candidates (actions) are one-hot parts (decision type, kind of
action, select hint, effect index) plus the card they are about, if any (its
static index, location, owner and dynamic features).
"""

from __future__ import annotations

from .. import constants as C
from ..actions import BATTLE_LISTS, IDLE_LISTS

# -- cards: what they are --------------------------------------------------

_TYPE_BITS = 27  # (TYPE_MONSTER .. TYPE_LINK)
_ATTRIBUTE_BITS = 7
_RACE_BITS = 26
_LEVELS = 14  # (level / rank / link rating 0..13)
_ATK_BUCKETS = (0, 1000, 1500, 2000, 2500, 3000)  # (lower bounds; one-hot, plus "no ATK" for non-monsters)
STATIC_DIM = _TYPE_BITS + _ATTRIBUTE_BITS + _RACE_BITS + _LEVELS + 2 * (len(_ATK_BUCKETS) + 1) + 2 + 1


def _bits(value: int, n: int) -> list[float]:
    return [float(value >> i & 1) for i in range(n)]


def _bucket(value: int | None) -> list[float]:
    out = [0.0] * (len(_ATK_BUCKETS) + 1)
    if value is None or value < 0:
        out[-1] = 1.0
    else:
        out[max(i for i, b in enumerate(_ATK_BUCKETS) if value >= b)] = 1.0
    return out


def static_features(card) -> list[float]:
    """What a card is (from the card database; zeros but the last for an unknown card)."""
    if card is None:
        return [0.0] * (STATIC_DIM - 1) + [1.0]
    monster = bool(card.type & C.TYPE_MONSTER)
    level = card.level if not card.type & C.TYPE_LINK else bin(card.link_marker).count("1")
    lv = [0.0] * _LEVELS
    if monster:
        lv[min(level, _LEVELS - 1)] = 1.0
    atk = card.attack if monster else None
    deff = card.defense if monster and not card.type & C.TYPE_LINK else None
    return (_bits(card.type, _TYPE_BITS) + _bits(card.attribute, _ATTRIBUTE_BITS) + _bits(card.race, _RACE_BITS)
            + lv + _bucket(atk) + _bucket(deff)
            + [max(atk or 0, 0) / 5000.0, max(deff or 0, 0) / 5000.0] + [0.0])


class CodeIndex:
    """Static indices for card codes, in order of first use (0: no card / unknown)."""

    def __init__(self, codes: list[int] | None = None):
        self.codes = [0] + [c for c in (codes or []) if c]
        self.index = {c: i for i, c in enumerate(self.codes)}

    def __call__(self, code: int | None) -> int:
        if not code:
            return 0
        i = self.index.get(code)
        if i is None:
            i = self.index[code] = len(self.codes)
            self.codes.append(code)
        return i

    def table(self, db) -> list[list[float]]:
        return [static_features(db.get(c) if c else None) for c in self.codes]


# -- cards: how and where they are ------------------------------------------

LOCATIONS = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_GRAVE,
             C.LOCATION_REMOVED, C.LOCATION_EXTRA, C.LOCATION_OVERLAY)
_LOC_INDEX = {loc: i for i, loc in enumerate(LOCATIONS)}
_POSITIONS = (C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_ATTACK, C.POS_FACEUP_DEFENSE, C.POS_FACEDOWN_DEFENSE)
# dynamic features, stored compactly as integers (dynamic_raw) and expanded for the networks (expand_dynamic):
# position (index into _POSITIONS + 1, 0: none), ATK, DEF, overlays, counters, flags (1 disabled, 2 public)
RAW_DIM = 6
DYN_DIM = len(_POSITIONS) + 6

# state groups: (own, opponent) x piles, and the chain
_PILES = ("hand", "mzone", "szone", "grave", "removed", "extra", "deck")
GROUPS = [f"own_{p}" for p in _PILES] + [f"opp_{p}" for p in _PILES] + ["chain"]
_GROUP = {name: i for i, name in enumerate(GROUPS)}
_PILED = {"extra", "deck"}  # (one row per distinct card, with its count)


def dynamic_raw(entry: dict | None) -> tuple[int, ...]:
    """How a card is now (a capture's card dict or a prompt's entry), as integers."""
    if entry is None:
        return (0, 0, 0, 0, 0, 0)
    pos = (entry.get("position", entry.get("pos", 0)) or 0) & 0xF
    status = entry.get("status", 0) or 0
    counters = entry.get("counters") or []
    return (_POSITIONS.index(pos) + 1 if pos in _POSITIONS else 0,
            max(entry.get("attack", 0) or 0, 0), max(entry.get("defense", 0) or 0, 0),
            len(entry.get("overlays") or ()), sum(counters[1::2]) if counters else 0,
            int(bool(status & C.STATUS_DISABLED)) | 2 * int(bool(entry.get("is_public"))))


def expand_dynamic(raw):
    """(n, RAW_DIM) integers -> (n, DYN_DIM) floats (numpy)."""
    import numpy as np
    raw = np.asarray(raw, dtype=np.float32).reshape(-1, RAW_DIM)
    pos = np.zeros((len(raw), len(_POSITIONS)), dtype=np.float32)
    idx = raw[:, 0].astype(int)
    pos[idx > 0, idx[idx > 0] - 1] = 1.0
    flags = raw[:, 5].astype(int)
    return np.concatenate([pos, raw[:, 1:2] / 5000.0, raw[:, 2:3] / 5000.0, raw[:, 3:4] / 5.0, raw[:, 4:5] / 5.0,
                           (flags & 1)[:, None].astype(np.float32), (flags >> 1 & 1)[:, None].astype(np.float32)],
                          axis=1)


# -- positions ----------------------------------------------------------------

_PHASES = (C.PHASE_DRAW, C.PHASE_STANDBY, C.PHASE_MAIN1, C.PHASE_BATTLE_START, C.PHASE_BATTLE_STEP,
           C.PHASE_DAMAGE, C.PHASE_DAMAGE_CAL, C.PHASE_BATTLE, C.PHASE_MAIN2, C.PHASE_END)
SCALAR_DIM = 2 + len(_PHASES) + 2 + 2 * 4 + 1


def position(snap: dict, me: int, codes) -> tuple[list[tuple], list[float]]:
    """The cards (static index, group, count, dynamic raw) and scalars of a capture,
    for `me` (`codes`: card code -> static index)."""
    from collections import Counter
    cards = []
    for side, p in (("own", me), ("opp", 1 - me)):
        pl = snap["players"][p]
        for pile in _PILES:
            g = _GROUP[f"{side}_{pile}"]
            if pile in _PILED:
                for code, n in sorted(Counter(c.get("code", 0) for c in pl[pile] if c is not None).items()):
                    cards.append((codes(code), g, n, (0, 0, 0, 0, 0, 0)))
                continue
            for c in pl[pile]:
                if c is not None:
                    cards.append((codes(c.get("code")), g, 1, dynamic_raw(c)))
    for link in snap.get("chain", ()):
        cards.append((codes(link.get("code")), _GROUP["chain"], 1, (0, 0, 0, 0, 0, 0)))
    lp = snap["lp"]
    phase = snap.get("phase", 0)
    scalars = [lp[me] / 8000.0, lp[1 - me] / 8000.0] + [float(phase == ph) for ph in _PHASES]
    scalars += [float(snap["turn_player"] == me), min(snap["turn"], 40) / 20.0]
    for p in (me, 1 - me):
        pl = snap["players"][p]
        scalars += [len(pl["deck"]) / 40.0, len(pl["hand"]) / 10.0, len(pl["extra"]) / 15.0,
                    sum(1 for c in pl["mzone"] if c is not None) / 5.0]
    scalars += [len(snap.get("chain", ())) / 5.0]
    return cards, scalars


# -- candidates ------------------------------------------------------------

PROMPTS = ("SELECT_IDLECMD", "SELECT_BATTLECMD", "SELECT_CHAIN", "SELECT_EFFECTYN", "SELECT_YESNO", "SELECT_OPTION",
           "SELECT_CARD")
KINDS = ("summon", "spsummon", "repos", "mset", "sset", "activate", "to_bp", "to_ep", "shuffle", "attack", "to_m2",
         "chain", "pass", "yes", "no", "option", "select")
HINTS = 81  # (select hints 500..579 and "none")
EFFECTS = 18  # (effect index 0..15 of a card, a system string, none)


def _effect(desc: int | None) -> int:
    if desc is None:
        return EFFECTS - 1
    return desc & 0xF if desc >> 20 else EFFECTS - 2


def _card_at(snap: dict, entry: dict) -> dict | None:
    """The capture's card for a prompt's (con, loc, seq) entry."""
    from ..state import LOC_NAME
    name = LOC_NAME.get(entry.get("loc", 0) & ~C.LOCATION_OVERLAY)
    if name is None:
        return None
    pile = snap["players"][entry["con"]][name]
    seq = entry.get("seq", 0)
    return pile[seq] if 0 <= seq < len(pile) else None


def actions(prompt: dict, cands: list[dict], snap: dict, me: int, hint: int | None,
            codes: CodeIndex) -> list[tuple]:
    """Per candidate: (prompt type, kind, hint, effect, card static index, card location (-1: none),
    card is own, dynamic raw)."""
    t = prompt["type"]
    pt = PROMPTS.index(t)
    h = HINTS - 1 if hint is None or not 500 <= hint < 500 + HINTS - 1 else hint - 500
    out = []
    for c in cands:
        entry, kind, desc = None, None, None
        if t in ("SELECT_IDLECMD", "SELECT_BATTLECMD"):
            kind = c["kind"]
            lst = (IDLE_LISTS if t == "SELECT_IDLECMD" else BATTLE_LISTS).get(kind)
            if lst is not None:
                entry = prompt[lst][c["index"]]
                desc = entry.get("desc")
        elif t == "SELECT_CHAIN":
            if c["index"] < 0:
                kind = "pass"
            else:
                kind, entry = "chain", prompt["chains"][c["index"]]
                desc = entry.get("desc")
        elif t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
            kind, desc = ("yes" if c["yes"] else "no"), prompt.get("desc")
            if t == "SELECT_EFFECTYN":
                entry = prompt
        elif t == "SELECT_OPTION":
            kind, desc = "option", prompt["options"][c["index"]]
        elif t == "SELECT_CARD":
            kind, entry = "select", prompt["cards"][c["indices"][0]]
        code = entry.get("code") if entry is not None else (desc >> 20 if desc and desc >> 20 else 0)
        here = _card_at(snap, entry) if entry is not None and "loc" in entry else None
        loc = entry.get("loc", 0) & ~C.LOCATION_OVERLAY if entry is not None else 0
        out.append((pt, KINDS.index(kind), h if t == "SELECT_CARD" else HINTS - 1, _effect(desc), codes(code),
                    _LOC_INDEX.get(loc, -1), float(entry is not None and entry.get("con", me) == me),
                    dynamic_raw(here if here is not None else entry)))
    return out
