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
