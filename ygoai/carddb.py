"""Card database backed by the BabelCDB sqlite files."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import constants as C
from .paths import BABEL_CDB_DIR

# Later files override earlier ones.
DEFAULT_CDBS = [
    "cards-unofficial.cdb",
    "prerelease-*.cdb",
    "release-*.cdb",
    "cards.cdb",
]
EXCLUDED_CDB_PARTS = ("rush", "skills", "goat")


@dataclass(frozen=True, slots=True)
class CardData:
    code: int
    alias: int
    setcodes: tuple[int, ...]
    type: int
    level: int
    attribute: int
    race: int
    attack: int
    defense: int
    lscale: int
    rscale: int
    link_marker: int
    ot: int
    category: int
    name: str
    desc: str
    strings: tuple[str, ...]

    @property
    def is_extra_deck(self) -> bool:
        return bool(self.type & C.TYPE_MONSTER) and bool(self.type & C.EXTRA_DECK_TYPES)

    @property
    def is_monster(self) -> bool:
        return bool(self.type & C.TYPE_MONSTER)

    @property
    def is_token(self) -> bool:
        return bool(self.type & C.TYPE_TOKEN)

    def string(self, index: int) -> str:
        return self.strings[index] if 0 <= index < len(self.strings) else ""


def _unpack_setcodes(packed: int) -> tuple[int, ...]:
    out = []
    for i in range(4):
        sc = (packed >> (16 * i)) & 0xFFFF
        if sc:
            out.append(sc)
    return tuple(out)


class CardDB:
    def __init__(self, cdb_dir: Path = BABEL_CDB_DIR, patterns: list[str] | None = None):
        self.cards: dict[int, CardData] = {}
        files: list[Path] = []
        for pattern in patterns or DEFAULT_CDBS:
            for f in sorted(cdb_dir.glob(pattern)):
                if any(part in f.name for part in EXCLUDED_CDB_PARTS):
                    continue
                if f not in files:
                    files.append(f)
        if not files:
            raise FileNotFoundError(f"no .cdb files found in {cdb_dir}")
        for f in files:
            self._load(f)

    def _load(self, path: Path) -> None:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = db.execute(
                "SELECT d.id, d.ot, d.alias, d.setcode, d.type, d.atk, d.def, d.level, d.race,"
                " d.attribute, d.category, t.name, t.desc, t.str1, t.str2, t.str3, t.str4,"
                " t.str5, t.str6, t.str7, t.str8, t.str9, t.str10, t.str11, t.str12, t.str13,"
                " t.str14, t.str15, t.str16"
                " FROM datas d LEFT JOIN texts t ON d.id = t.id"
            ).fetchall()
        finally:
            db.close()
        for row in rows:
            (code, ot, alias, setcode, ctype, atk, deff, level, race, attribute, category,
             name, desc, *strings) = row
            link_marker = 0
            if ctype & C.TYPE_LINK:
                link_marker = deff
                deff = 0
            self.cards[code] = CardData(
                code=code, alias=alias, setcodes=_unpack_setcodes(setcode or 0), type=ctype,
                level=level & 0xFF, attribute=attribute, race=race & 0xFFFFFFFFFFFFFFFF,
                attack=atk, defense=deff, lscale=(level >> 24) & 0xFF, rscale=(level >> 16) & 0xFF,
                link_marker=link_marker, ot=ot, category=category, name=name or "", desc=desc or "",
                strings=tuple(s or "" for s in strings),
            )

    def get(self, code: int) -> CardData | None:
        return self.cards.get(code)

    def __contains__(self, code: int) -> bool:
        return code in self.cards

    def name(self, code: int) -> str:
        card = self.cards.get(code)
        return card.name if card else f"#{code}"

    def describe_desc(self, desc: int) -> str:
        """Human readable form of an effect description (code << 4 | index)."""
        code, idx = desc >> 4, desc & 0xF
        card = self.cards.get(code)
        if card is None:
            return f"desc {desc}"
        text = card.string(idx)
        return f"{card.name}: {text}" if text else f"{card.name} [{idx}]"


@lru_cache(maxsize=1)
def default_db() -> CardDB:
    return CardDB()
