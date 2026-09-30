"""Guessing the unknown part of the opponent's deck list."""

from __future__ import annotations

from collections import Counter

from .. import constants as C
from ..carddb import CardDB, default_db


def tribute_levels(tributes: int) -> range:
    return {0: range(0, 5), 1: range(5, 7)}.get(tributes, range(7, 13))


_monster_sset: set[int] | None = None


def monster_sset_codes() -> set[int]:
    """Monsters that can be Set in the spell/trap zone (EFFECT_MONSTER_SSET,
    e.g. the "Artifact" monsters), read from the card scripts."""
    global _monster_sset
    if _monster_sset is None:
        import re
        from ..paths import CARD_SCRIPTS_DIR
        _monster_sset = set()
        for sub in ("official", "pre-release", "unofficial"):
            d = CARD_SCRIPTS_DIR / sub
            if not d.is_dir():
                continue
            for f in d.glob("c*.lua"):
                try:
                    text = f.read_text(errors="replace")
                except OSError:
                    continue
                if "EFFECT_MONSTER_SSET" in text:
                    m = re.match(r"c(\d+)\.lua", f.name)
                    if m:
                        _monster_sset.add(int(m.group(1)))
    return _monster_sset


def fits_need(card, need: tuple) -> bool:
    """Whether a card can be the hidden card with requirement `need`:
    ("st",) Set in a spell/trap zone, ("field",) Set in the field zone,
    ("mset", tributes) Set in a monster zone, ("monster",) summoned face-down."""
    if card is None:
        return False
    kind = need[0]
    if kind == "st":
        if card.type & C.TYPE_MONSTER:
            return card.code in monster_sset_codes()
        return bool(card.type & (C.TYPE_SPELL | C.TYPE_TRAP)) and not card.type & C.TYPE_FIELD
    if kind == "field":
        return bool(card.type & C.TYPE_SPELL) and bool(card.type & C.TYPE_FIELD)
    if kind in ("mset", "monster"):
        if not card.type & C.TYPE_MONSTER or card.is_extra_deck:
            return False
        return kind == "monster" or card.level in tribute_levels(need[1])
    raise ValueError(f"unknown requirement {need}")


class DeckCompletion:
    """Interface: given the known cards of a deck, guess the missing ones."""

    def complete(self, known_main: list[int], known_extra: list[int], n_main: int,
                 n_extra: int) -> tuple[list[int], list[int]]:
        raise NotImplementedError

    def stand_ins(self, need: tuple) -> list[int]:
        """Codes for a hidden card of a known kind that was never revealed
        (effect-less where possible, so they cannot change the game)."""
        return _vanilla(getattr(self, "db", None)).stand_ins(need)


_vanilla_cache: dict = {}


def _vanilla(db) -> "VanillaCompletion":
    key = id(db)
    if key not in _vanilla_cache:
        _vanilla_cache[key] = VanillaCompletion(db)
    return _vanilla_cache[key]


def _fill(pool: list[int], known: list[int], n: int, db: CardDB) -> list[int]:
    counts = Counter((db.get(c).alias if db.get(c) and db.get(c).alias else c) for c in known)
    out: list[int] = []
    while len(out) < n:
        progressed = False
        for code in pool:
            if len(out) >= n:
                break
            if counts[code] < 3:
                counts[code] += 1
                out.append(code)
                progressed = True
        if not progressed:
            raise RuntimeError("filler pool exhausted")
    return out


class VanillaCompletion(DeckCompletion):
    """Fills unknown cards with effect-less cards (never interferes with the game)."""

    def __init__(self, db: CardDB | None = None):
        self.db = db or default_db()
        cards = self.db.cards.values()
        self.main_pool = sorted(c.code for c in cards if c.type == C.TYPE_MONSTER | C.TYPE_NORMAL
                                and not c.alias and c.ot & 0x3)[:60]
        self.extra_pool = sorted(c.code for c in cards if c.type == C.TYPE_MONSTER | C.TYPE_FUSION
                                 and not c.alias and c.ot & 0x3)[:30]
        # stand-ins for hidden cards that must have a certain kind
        self.settable_monsters = [c for c in self.main_pool if self.db.get(c).level <= 4][:10]
        vanillas = sorted(c.code for c in cards if c.type == C.TYPE_MONSTER | C.TYPE_NORMAL
                          and not c.alias and c.ot & 0x3)
        self.settable_monsters_all = []
        for levels in (range(1, 5), range(5, 7), range(7, 13)):
            self.settable_monsters_all += [c for c in vanillas if self.db.get(c).level in levels][:6]
        normal_st = sorted(c.code for c in cards if c.type in (C.TYPE_SPELL, C.TYPE_TRAP)
                           and not c.alias and c.ot & 0x3)
        # the lowest codes, plus some spread over the whole range (the code
        # decides the creation order, which is sometimes constrained)
        spread = normal_st[10::max(1, (len(normal_st) - 10) // 30)]
        self.settable_spells = normal_st[:10] + spread
        self.settable_spells += sorted(c.code for c in cards if c.type == C.TYPE_SPELL | C.TYPE_FIELD
                                       and not c.alias and c.ot & 0x3)[:5]

    def stand_ins(self, need: tuple) -> list[int]:
        if need[0] in ("st", "field"):
            pool = self.settable_spells
        elif need[0] == "mset":
            pool = self.settable_monsters_all
        else:
            pool = self.main_pool
        return [c for c in pool if fits_need(self.db.get(c), need)]

    def complete(self, known_main, known_extra, n_main, n_extra):
        return (_fill(self.main_pool, known_main, n_main, self.db),
                _fill(self.extra_pool, known_extra, n_extra, self.db))
