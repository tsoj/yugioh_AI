"""Opponent deck completion from a corpus of human deck lists.

The corpus (download_human_decks/decks/*.ydk) is compiled once into a compact
numpy index (build/deck_index.npz). Completion scores every corpus deck by
how many of the known opponent cards it contains and fills the unknown slots
with the cards of the best matching decks, weighted by their score.
"""

from __future__ import annotations

import random
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from ..carddb import CardDB, default_db
from ..decks import canonical_code, deck_split, parse_ydk
from ..paths import DECKS_DIR, ROOT
from .completion import DeckCompletion

INDEX_PATH = ROOT / "build" / "deck_index.npz"


def build_index(db: CardDB, decks_dir: Path = DECKS_DIR, out: Path = INDEX_PATH,
                limit: int | None = None, split: str | None = "train") -> Path:
    """Only the "train" split is indexed by default, so that games played with
    "test" split decks cannot look up the exact opponent deck list."""
    paths = sorted(p for p in decks_dir.glob("*.ydk") if split is None or deck_split(p) == split)
    if limit:
        paths = paths[:limit]
    deck_ids, codes, counts, extra_flags = [], [], [], []
    n = 0
    for path in paths:
        deck = parse_ydk(path)
        cnt = Counter(canonical_code(db, c) for c in deck.main + deck.extra)
        if not cnt or any(c not in db for c in cnt):
            continue
        for code, k in cnt.items():
            deck_ids.append(n)
            codes.append(code)
            counts.append(min(k, 3))
            extra_flags.append(db.get(code).is_extra_deck)
        n += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, deck=np.array(deck_ids, dtype=np.int32),
                        code=np.array(codes, dtype=np.int64), count=np.array(counts, dtype=np.int8),
                        extra=np.array(extra_flags, dtype=bool), n=np.array([n]))
    return out


class CorpusCompletion(DeckCompletion):
    def __init__(self, db: CardDB | None = None, index: Path = INDEX_PATH, top_k: int = 20,
                 seed: int = 0):
        self.db = db or default_db()
        if not index.exists():
            build_index(self.db, out=index)
        data = np.load(index)
        self.deck = data["deck"]
        self.code = data["code"]
        self.count = data["count"]
        self.extra = data["extra"]
        self.n = int(data["n"][0])
        order = np.argsort(self.code, kind="stable")
        self._sorted_codes = self.code[order]
        self._order = order
        self.top_k = top_k
        self.rng = random.Random(seed)

    def _rows_with(self, code: int) -> np.ndarray:
        lo = np.searchsorted(self._sorted_codes, code, "left")
        hi = np.searchsorted(self._sorted_codes, code, "right")
        return self._order[lo:hi]

    def _scores(self, known: Counter) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for code, k in known.items():
            rows = self._rows_with(code)
            scores[self.deck[rows]] += np.minimum(self.count[rows], k)
        return scores

    def complete(self, known_main, known_extra, n_main, n_extra, present=()):
        known = Counter(canonical_code(self.db, c) for c in known_main + known_extra)
        scores = self._scores(known)
        if scores.max() <= 0:
            # (drawn from the known cards: the same in every call for the same cards,
            # whatever was completed before in this process)
            rng = random.Random(zlib.crc32(repr(sorted(known.items())).encode()))
            best = np.array(rng.sample(range(self.n), self.top_k))
        else:
            # (stable: numpy's default sort of float32 arrays orders ties by machine, e.g. with SIMD)
            best = np.argsort(-scores, kind="stable")[:self.top_k]
        weights: Counter = Counter()
        mask = np.isin(self.deck, best)
        for d, code, k, is_extra in zip(self.deck[mask], self.code[mask], self.count[mask], self.extra[mask]):
            weights[(int(code), bool(is_extra))] += float(scores[d] + 1) * int(k)
        # (no guess adds a duel-wide effect that no known card has brought into the game:
        # by its mere presence it could change the game, see script_index.global_keys)
        from .script_index import global_keys
        have = set().union(*(global_keys(canonical_code(self.db, c))
                             for c in list(present) + known_main + known_extra))
        adds = lambda code: not global_keys(code) <= have
        fill_main = self._fill(weights, known, n_main, extra=False, skip=adds)
        fill_extra = self._fill(weights, known, n_extra, extra=True, skip=adds)
        return fill_main, fill_extra

    def _fill(self, weights: Counter, known: Counter, n: int, extra: bool, skip=lambda code: False) -> list[int]:
        from .script_index import win_condition_codes
        wins = win_condition_codes()  # (never guessed: e.g. Exodia pieces, a guessed hand would win)
        have = Counter(known)
        ranked = sorted(((w, code) for (code, is_extra), w in weights.items()
                         if is_extra == extra and code not in wins and not skip(code)), reverse=True)
        out: list[int] = []
        while len(out) < n:
            progressed = False
            for _, code in ranked:
                if len(out) >= n:
                    break
                if have[code] < 3:
                    have[code] += 1
                    out.append(code)
                    progressed = True
            if not progressed:
                break
        if len(out) < n:
            # the best decks did not have enough cards of this kind; fall back to vanillas
            from .completion import VanillaCompletion
            van = VanillaCompletion(self.db)
            pool = van.extra_pool if extra else van.main_pool
            for code in pool:
                while len(out) < n and have[code] < 3:
                    have[code] += 1
                    out.append(code)
        return out
