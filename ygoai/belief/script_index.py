"""What a card's script is about, read statically from its text.

The scripts declare the archetypes and cards they refer to (``s.listed_series``,
``s.listed_names``); and some checks read values of several cards (how many
different Pendulum Scales, the highest Rank in the Extra Deck, a sum of Levels)
instead of asking each card a yes/no question. Neither needs the Lua to be
understood: the declarations narrow down which cards could satisfy a check, and
the engine decides whether one does.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..paths import CARD_SCRIPTS_DIR, FIXED_CARD_SCRIPTS_DIR

# calls that read values of a group of cards (a forced filter answer cannot
# satisfy them: the values themselves are compared)
_VALUE_CHECKS = re.compile(r"GetClassCount|GetMaxGroup|GetMinGroup|CheckWithSum|GetSum\b|"
                           r"aux\.dpcheck|aux\.dncheck|GetBinClassCount")
_LISTED = re.compile(r"s\.listed_(series|names)\s*=\s*\{([^}]*)\}")
_CONST = re.compile(r"^\s*((?:SET|CARD)_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+|\d+)", re.M)


@lru_cache(maxsize=1)
def _constants() -> dict[str, int]:
    out: dict[str, int] = {}
    for path in CARD_SCRIPTS_DIR.glob("*.lua"):
        for name, value in _CONST.findall(path.read_text(errors="replace")):
            out[name] = int(value, 0)
    return out


def _script_text(code: int) -> str:
    for path in [FIXED_CARD_SCRIPTS_DIR / f"c{code}.lua"] + [
            CARD_SCRIPTS_DIR / sub / f"c{code}.lua" for sub in ("official", "pre-release", "unofficial")]:
        if path.exists():
            return path.read_text(errors="replace")
    return ""


@lru_cache(maxsize=None)
def script_info(code: int) -> tuple[frozenset[int], frozenset[int], bool]:
    """(listed archetype codes, listed card codes, whether it checks card values)."""
    text = _script_text(code)
    consts = _constants()
    series: set[int] = set()
    names: set[int] = set()
    for kind, body in _LISTED.findall(text):
        for tok in (t.strip() for t in body.split(",")):
            if not tok:
                continue
            if tok == "id":
                value = code
            elif re.fullmatch(r"0x[0-9a-fA-F]+|\d+", tok):
                value = int(tok, 0)
            else:
                value = consts.get(tok)
            if value is not None:
                (series if kind == "series" else names).add(value)
    return frozenset(series), frozenset(names), bool(_VALUE_CHECKS.search(text))


def is_set_card(card_setcodes, setcode: int) -> bool:
    """Card.IsSetCard: the archetype matches, and the sub-archetype bits if any."""
    return any((s & 0xfff) == (setcode & 0xfff) and (s & setcode) == setcode for s in card_setcodes)


@lru_cache(maxsize=None)
def piles(code: int) -> frozenset[str]:
    """The hidden piles the script refers to ("deck", "extra")."""
    text = _script_text(code)
    return frozenset(name for name, loc in (("deck", "LOCATION_DECK"), ("extra", "LOCATION_EXTRA"))
                     if loc in text)


def candidates(code: int, db) -> list[int]:
    """Cards the script of `code` refers to (its archetypes and named cards),
    legal ones, by code."""
    series, names, _ = script_info(code)
    out = []
    for c in db.cards.values():
        if c.alias or not c.ot & 0x3 or c.code == code:
            continue
        if c.code in names or (series and any(is_set_card(c.setcodes, s) for s in series)):
            out.append(c.code)
    return sorted(out)


@lru_cache(maxsize=None)
def proc_cards(kind: str) -> frozenset[int]:
    """Cards whose scripts give them their own Normal Summon ("summon") or Set
    ("mset") procedure that can Tribute the opponent's monsters (e.g. Thestalos
    the Shadowfire Monarch: 1 monster the opponent controls and 1 Tribute
    Summoned monster)."""
    code_name = {"summon": "EFFECT_SUMMON_PROC", "mset": "EFFECT_SET_PROC"}[kind]
    out = set()
    for sub in ("official", "pre-release", "unofficial"):
        for path in (CARD_SCRIPTS_DIR / sub).glob("c*.lua"):
            text = path.read_text(errors="replace")
            if code_name in text and "1-tp" in text and ("Release" in text or "RELEASE" in text):
                try:
                    out.add(int(path.stem[1:]))
                except ValueError:
                    pass
    return frozenset(out)


@lru_cache(maxsize=1)
def discard_set_proc_cards() -> frozenset[int]:
    """Cards whose own Set procedure discards cards as its cost (e.g. Infernal
    Incinerator: "discarding your entire hand and Tributing 1 monster")."""
    out = set()
    for sub in ("official", "pre-release", "unofficial"):
        for path in (CARD_SCRIPTS_DIR / sub).glob("c*.lua"):
            text = path.read_text(errors="replace")
            if ("EFFECT_SET_PROC" in text or "EFFECT_LIMIT_SET_PROC" in text) and "REASON_DISCARD" in text:
                try:
                    out.add(int(path.stem[1:]))
                except ValueError:
                    pass
    return frozenset(out)


_SET_PROC = re.compile(r"AddNormalSetProcedure\(\s*c\s*,[^,]*,[^,]*,\s*(\d+)\s*,\s*(\d+)")


@lru_cache(maxsize=None)
def set_proc_tribute_cards(n: int) -> frozenset[int]:
    """Cards whose own Set procedure Tributes `n` monsters (e.g. The Wicked
    Avatar: 3; no Level asks for more than 2)."""
    out = set()
    for sub in ("official", "pre-release", "unofficial"):
        for path in (CARD_SCRIPTS_DIR / sub).glob("c*.lua"):
            text = path.read_text(errors="replace")
            for lo, hi in _SET_PROC.findall(text):
                if int(lo) <= n <= int(hi):
                    try:
                        out.add(int(path.stem[1:]))
                    except ValueError:
                        pass
    return frozenset(out)


@lru_cache(maxsize=None)
def reveals_itself(code: int) -> bool:
    """Whether the card's own procedure reveals it together with other cards
    (e.g. Evil Dragon Cards: "revealing 3 LIGHT monsters in your hand, including
    this card"), as opposed to revealing other cards only (Crimson Nova the Dark
    Cubic Lord: 3 other Cubic cards)."""
    text = _script_text(code)
    return bool(re.search(r"Group\.FromCards\(c\)|:IsContains\(c\)", text))


@lru_cache(maxsize=None)
def unsummonable(code: int) -> bool:
    """Whether the card cannot be Normal Summoned or Set by its own script (e.g. "Must
    be Special Summoned by ..."): Card.EnableUnsummonable, EFFECT_UNSUMMONABLE_CARD,
    and Card.EnableReviveLimit (on a Main Deck monster: Special Summon only)."""
    text = _script_text(code)
    return bool(re.search(r"EnableUnsummonable\(|EFFECT_UNSUMMONABLE_CARD|EnableReviveLimit\(", text))
