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

    @property
    def is_extra_deck(self) -> bool:
        return bool(self.type & C.TYPE_MONSTER) and bool(self.type & C.EXTRA_DECK_TYPES)

    @property
    def is_token(self) -> bool:
        return bool(self.type & C.TYPE_TOKEN)


def _unpack_setcodes(packed: int) -> tuple[int, ...]:
    out = []
    for i in range(4):
        sc = (packed >> (16 * i)) & 0xFFFF
        if sc:
            out.append(sc)
    return tuple(out)


class CardDB:
    """The cards' data; their texts (effect strings) are read only when asked for
    (`strings`): every process holds the database, and the texts are most of it."""

    def __init__(self, cdb_dir: Path = BABEL_CDB_DIR, patterns: list[str] | None = None):
        self.cards: dict[int, CardData] = {}
        self._strings: dict[int, tuple[str, ...]] | None = None
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
        self.files = files

    def _load(self, path: Path) -> None:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = db.execute(
                "SELECT d.id, d.ot, d.alias, d.setcode, d.type, d.atk, d.def, d.level, d.race,"
                " d.attribute, d.category, t.name"
                " FROM datas d LEFT JOIN texts t ON d.id = t.id"
            ).fetchall()
        finally:
            db.close()
        for row in rows:
            (code, ot, alias, setcode, ctype, atk, deff, level, race, attribute, category, name) = row
            link_marker = 0
            if ctype & C.TYPE_LINK:
                link_marker = deff
                deff = 0
            self.cards[code] = CardData(
                code=code, alias=alias, setcodes=_unpack_setcodes(setcode or 0), type=ctype,
                level=level & 0xFF, attribute=attribute, race=race & 0xFFFFFFFFFFFFFFFF,
                attack=atk, defense=deff, lscale=(level >> 24) & 0xFF, rscale=(level >> 16) & 0xFF,
                link_marker=link_marker, ot=ot, category=category, name=name or "",
            )

    def get(self, code: int) -> CardData | None:
        return self.cards.get(code)

    def __contains__(self, code: int) -> bool:
        return code in self.cards

    def strings(self, code: int) -> tuple[str, ...]:
        """The card's effect strings (str1..str16)."""
        if self._strings is None:
            self._strings = {}
            cols = ", ".join(f"str{i}" for i in range(1, 17))
            for f in self.files:
                db = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
                try:
                    for code_, *strs in db.execute(f"SELECT id, {cols} FROM texts"):
                        if code_ in self.cards:
                            self._strings[code_] = tuple(x or "" for x in strs)
                finally:
                    db.close()
        return self._strings.get(code, ())

    def name(self, code: int) -> str:
        card = self.cards.get(code)
        return card.name if card else f"#{code}"

    def describe_desc(self, desc: int) -> str:
        """Human readable form of an effect description (code << 4 | index)."""
        code, idx = desc >> 4, desc & 0xF
        card = self.cards.get(code)
        if card is None:
            return f"desc {desc}"
        strings = self.strings(code)
        text = strings[idx] if idx < len(strings) else ""
        return f"{card.name}: {text}" if text else f"{card.name} [{idx}]"


@lru_cache(maxsize=1)
def default_db() -> CardDB:
    return CardDB()
