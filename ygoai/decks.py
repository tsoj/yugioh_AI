"""Deck lists (.ydk) and deck selection."""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from . import constants as C
from .carddb import CardDB
from .paths import CARD_SCRIPTS_DIR, DECKS_DIR, FIXED_CARD_SCRIPTS_DIR, ROOT


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
    return bool(vanilla) or _has_script(code)


def _has_script(code: int) -> bool:
    """Whether the card has a script (the Rust cards are translated from these)."""
    name = f"c{code}.lua"
    return any(p.exists() for p in [FIXED_CARD_SCRIPTS_DIR / name] + [
        CARD_SCRIPTS_DIR / d / name for d in ("official", "pre-release", "pre-errata", "goat")] + [CARD_SCRIPTS_DIR / name])


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
        self.dir = decks_dir
        # (sorted by name: the same order as by path within one directory, much faster)
        self.paths = sorted((p for p in decks_dir.glob("*.ydk") if split is None or deck_split(p) == split),
                            key=lambda p: p.name)
        if not self.paths:
            raise FileNotFoundError(f"no .ydk files in {decks_dir}")
        self._cache: dict[Path, Deck | None] = {}

    def load(self, path: Path) -> Deck | None:
        if path not in self._cache:
            self._cache[path] = normalize(parse_ydk(path), self.db)
        return self._cache[path]

    def diverse(self, index: int) -> Deck:
        """The deck at `index` of the diverse order (see `diverse_order`), wrapping around."""
        if not hasattr(self, "_order"):
            name = "diverse_decks.json" if self.dir == DECKS_DIR else f"diverse_decks_{self.dir.name}.json"
            self._order = diverse_order(self, ROOT / "build" / name)
        return self.load(self._by_name()[self._order[index % len(self._order)]])

    def _by_name(self) -> dict[str, Path]:
        if not hasattr(self, "_names"):
            self._names = {p.name: p for p in self.paths}
        return self._names

    def sample(self, rng: random.Random) -> Deck:
        for _ in range(1000):
            deck = self.load(rng.choice(self.paths))
            if deck is not None:
                return deck
        raise RuntimeError("could not find a usable deck")


DIVERSE_HEAD = 20000  # decks ordered for coverage (the rest follow in random order)


def diverse_order(pool: DeckPool, cache: Path | None = None, seed: int = 0) -> list[str]:
    """Usable decks of the pool (file names) without duplicates, ordered so that
    the decks early in the order together contain as many different cards as
    possible: each next deck is the one adding the most cards not yet covered
    (ties at random); once every card is covered, a new round starts. Popular
    builds that differ in a few cards come late, rare cards early."""
    import heapq
    import json
    if cache is not None and cache.exists():
        data = json.loads(cache.read_text())
        if data.get("seed") == seed and data.get("n_paths") == len(pool.paths):
            return data["order"]
    rng = random.Random(seed)
    seen: set[tuple] = set()
    names: list[str] = []
    cards: list[frozenset] = []
    for path in pool.paths:
        deck = pool.load(path)
        if deck is None:
            continue
        key = tuple(sorted(deck.main)) + (0,) + tuple(sorted(deck.extra))
        if key in seen:
            continue
        seen.add(key)
        names.append(path.name)
        cards.append(frozenset(pool.db.get(c).alias or c for c in deck.main + deck.extra))
    pool._cache.clear()  # (the decks are loaded again when used)
    tiebreak = [rng.random() for _ in names]
    left = set(range(len(names)))
    order: list[int] = []
    while left and len(order) < DIVERSE_HEAD:
        covered: set = set()
        heap = [(-len(cards[i]), tiebreak[i], i) for i in left]
        heapq.heapify(heap)
        while heap and len(order) < DIVERSE_HEAD:
            neg, tb, i = heapq.heappop(heap)
            gain = len(cards[i] - covered)
            if gain < -neg:
                heapq.heappush(heap, (-gain, tb, i))  # (stale estimate)
                continue
            if gain == 0:
                break  # (every card covered: a new round)
            order.append(i)
            left.discard(i)
            covered |= cards[i]
    rest = sorted(left)
    rng.shuffle(rest)
    result = [names[i] for i in order + rest]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"seed": seed, "n_paths": len(pool.paths), "order": result}))
    return result


_random_card_pools: dict = {}


RANDOM_DECK_STYLES = ("uniform", "threes", "only_threes", "singleton")


def random_card_deck(db: CardDB, rng: random.Random, name: str = "random cards",
                     style: str = "uniform") -> Deck:
    """A deck of random legal cards: 40 main, 15 extra. No structure at all, unlike
    human decks: for edge cases of rarely combined cards. Copies per card by style:
    "uniform" 1-3, "threes" mostly 3, "only_threes" always 3 (as far as the count
    allows), "singleton" always 1 (many identical copies, or none at all: the
    source of copy identity questions in shuffles, hints and histories)."""
    key = id(db)
    if key not in _random_card_pools:
        codes = sorted(c.code for c in db.cards.values()
                       if c.ot & 0x3 and not is_alt_art(c) and not c.is_token and card_supported(db, c.code))
        _random_card_pools[key] = ([c for c in codes if not db.get(c).is_extra_deck],
                                   [c for c in codes if db.get(c).is_extra_deck])
    main_pool, extra_pool = _random_card_pools[key]

    def copies() -> int:
        if style == "singleton":
            return 1
        if style == "only_threes":
            return 3
        if style == "threes":
            return 3 if rng.random() < 0.6 else rng.randint(1, 2)
        return rng.randint(1, 3)

    def fill(pool, n):
        out: list[int] = []
        while len(out) < n:
            code = rng.choice(pool)
            if code not in out:
                out += [code] * min(copies(), n - len(out))
        return out
    return Deck(main=fill(main_pool, 40), extra=fill(extra_pool, 15), name=f"{name} ({style})")
