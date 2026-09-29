"""Deck lists (.ydk) and deck selection."""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from . import constants as C
from .carddb import CardDB
from .core import _find_script
from .paths import DECKS_DIR


@dataclass
class Deck:
    main: list[int]
    extra: list[int] = field(default_factory=list)
    name: str = ""

    def to_json(self) -> dict:
        return {"name": self.name, "main": list(self.main), "extra": list(self.extra)}

    @staticmethod
    def from_json(d: dict) -> "Deck":
        return Deck(main=list(d["main"]), extra=list(d.get("extra", [])), name=d.get("name", ""))


def parse_ydk(path: Path) -> Deck:
    main, extra = [], []
    section = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = {"#main": main, "#extra": extra}.get(line)
            continue
        if line.startswith("!"):
            section = None
            continue
        if section is not None:
            try:
                section.append(int(line))
            except ValueError:
                pass
    return Deck(main=main, extra=extra, name=path.stem)


def is_alt_art(card) -> bool:
    return bool(card.alias) and abs(card.alias - card.code) < 10


def canonical_code(db: CardDB, code: int) -> int:
    """Alternate artworks behave exactly like the original card; use the original's code."""
    card = db.get(code)
    if card is not None and is_alt_art(card) and card.alias in db:
        return card.alias
    return code


def card_supported(db: CardDB, code: int) -> bool:
    card = db.get(code)
    if card is None or card.is_token:
        return False
    vanilla = card.type & C.TYPE_NORMAL and not card.type & (C.TYPE_PENDULUM | C.EXTRA_DECK_TYPES)
    # vanilla monsters have no script
    return bool(vanilla) or _find_script(f"c{code}.lua") is not None


def normalize(deck: Deck, db: CardDB) -> Deck | None:
    """Moves cards to the section their type requires; None if the deck is unusable."""
    main, extra = [], []
    for code in deck.main + deck.extra:
        code = canonical_code(db, code)
        card = db.get(code)
        if card is None or not card_supported(db, code):
            return None
        (extra if card.is_extra_deck else main).append(code)
    if not 40 <= len(main) <= 60 or len(extra) > 15:
        return None
    counts: dict[int, int] = {}
    for code in main + extra:
        base = db.get(code).alias or code
        counts[base] = counts.get(base, 0) + 1
        if counts[base] > 3:
            return None
    return Deck(main=main, extra=extra, name=deck.name)


def deck_split(path: Path) -> str:
    """Stable 90/10 split of the deck corpus: "train" (e.g. for deck completion
    models) and "test" (decks to play with when evaluating such models)."""
    return "test" if zlib.crc32(path.stem.encode()) % 10 == 0 else "train"


class DeckPool:
    """Lazily loads random usable decks from a directory of .ydk files."""

    def __init__(self, db: CardDB, decks_dir: Path = DECKS_DIR, split: str | None = None):
        self.db = db
        self.paths = sorted(p for p in decks_dir.glob("*.ydk") if split is None or deck_split(p) == split)
        if not self.paths:
            raise FileNotFoundError(f"no .ydk files in {decks_dir}")
        self._cache: dict[Path, Deck | None] = {}

    def load(self, path: Path) -> Deck | None:
        if path not in self._cache:
            self._cache[path] = normalize(parse_ydk(path), self.db)
        return self._cache[path]

    def sample(self, rng: random.Random) -> Deck:
        for _ in range(1000):
            deck = self.load(rng.choice(self.paths))
            if deck is not None:
                return deck
        raise RuntimeError("could not find a usable deck")
