"""Test games for the belief checks: their decks and bots (the games themselves are played in
Rust: bot_py_binding.play_with_streams, bot_py_binding.check_game)."""

from __future__ import annotations

import json
import random

import bot_py_binding

from .carddb import default_db
from .decks import RANDOM_DECK_STYLES, Deck, DeckPool, random_card_deck
from .paths import DECKS_DIR, NEW_DECKS_DIR

PAIRS = [("random", "random"), ("random", "greedy"), ("greedy", "random"), ("greedy", "greedy")]
DECK_MODES = ("random", "diverse", "cards", "new", "new-cards", "new-mixed")
_deck_pools: dict = {}
_books: dict = {}


def pairs_of(text: str) -> list[tuple[str, str]]:
    """Bot pairings from "setter-greedy,mixed-random"."""
    return [tuple(p.split("-")) for p in text.split(",")]


def add_game_options(ap) -> None:
    """The options that choose the test games (argparse): --pairs, --decks, --book, --mirror-prob."""
    ap.add_argument("--pairs", type=pairs_of,
                    help="bot pairings, e.g. setter-greedy,mixed-random (game seed %% number of pairs picks one; "
                         "default: random/greedy in all four combinations)")
    ap.add_argument("--decks", choices=DECK_MODES, default="random",
                    help="random corpus decks, decks ordered to cover as many different cards as possible, "
                         "decks of random cards (\"cards\"), or the decks downloaded later in the diverse order "
                         "(\"new\")")
    ap.add_argument("--book", help="opening book (scripts/make_book.py): game seed %% openings picks the decks")
    ap.add_argument("--mirror-prob", type=float, default=0.2)


def load_book(path) -> list:
    """The openings (pairs of decks) of a book written by scripts/make_book.py."""
    key = str(path)
    if key not in _books:
        with open(path) as f:
            _books[key] = [[Deck.from_json(d) for d in json.loads(line)["decks"]] for line in f if line.strip()]
    return _books[key]


def game_decks(seed: int, mirror_prob: float = 0.2, decks: str = "random", book: str | None = None) -> tuple:
    """The deck pair of test game `seed`: random decks (sometimes a mirror match);
    with decks="diverse" from the order that covers as many different cards as
    possible (game `seed` uses its positions 2*seed and 2*seed + 1), with
    decks="new" the same among the decks downloaded later; with a `book`
    (see scripts/make_book.py) opening seed % (number of openings)."""
    if book:
        return tuple(load_book(book)[seed % len(load_book(book))])  # (the book decides the pairing)
    pool_name = "new" if decks.startswith("new") else decks
    if decks != "cards" and pool_name not in _deck_pools:
        _deck_pools[pool_name] = DeckPool(default_db(), NEW_DECKS_DIR if pool_name == "new" else DECKS_DIR)
    pool = _deck_pools.get(pool_name)
    rng = random.Random(seed)
    if decks == "cards":
        db = default_db()
        d0 = random_card_deck(db, rng, f"random cards {seed}a", rng.choice(RANDOM_DECK_STYLES))
        return d0, d0 if rng.random() < mirror_prob else random_card_deck(
            db, rng, f"random cards {seed}b", rng.choice(RANDOM_DECK_STYLES))
    if decks == "new-cards":
        # (a human deck against a deck of random cards, on either side)
        human = pool.diverse(seed)
        rand = random_card_deck(default_db(), rng, f"random cards {seed}", rng.choice(RANDOM_DECK_STYLES))
        return (human, rand) if rng.random() < 0.5 else (rand, human)
    if decks == "new-mixed":
        # (each deck mixed from two human decks)
        return (_mixed_deck(pool.diverse(4 * seed), pool.diverse(4 * seed + 1), rng),
                _mixed_deck(pool.diverse(4 * seed + 2), pool.diverse(4 * seed + 3), rng))
    if decks in ("diverse", "new"):
        d0 = pool.diverse(2 * seed)
        return d0, d0 if rng.random() < mirror_prob else pool.diverse(2 * seed + 1)
    d0 = pool.sample(rng)
    return d0, d0 if rng.random() < mirror_prob else pool.sample(rng)


def _mixed_deck(a, b, rng):
    """About half of each of two decks' Main Decks (at most 3 copies of a card, 40
    cards) and a sample of both Extra Decks (at most 15)."""
    db = default_db()
    counts: dict[int, int] = {}

    def take(cards, n):
        out = []
        for c in rng.sample(cards, len(cards)):
            base = db.original(c)
            if len(out) < n and counts.get(base, 0) < 3:
                counts[base] = counts.get(base, 0) + 1
                out.append(c)
        return out
    main = take(a.main, 20)
    main += take(b.main, 40 - len(main))
    main += take(a.main + b.main, 40 - len(main))
    extra = take(a.extra + b.extra, 15)
    return Deck(main=main, extra=extra, name=f"mix of {a.name} and {b.name}")


def bots(seed: int, pairs: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    """The bots of test game `seed`: pairs[seed % len(pairs)] (by default PAIRS)."""
    pairs = pairs or PAIRS
    return tuple(pairs[seed % len(pairs)])


def play_game(seed: int, mirror_prob: float = 0.2, pairs: list[tuple[str, str]] | None = None,
              decks: str = "random", book: str | None = None) -> tuple[dict, list[list[tuple]]]:
    """The test game for `seed` (the decks of game_decks, the bots of `bots`): its record and
    both players' observation streams."""
    deck_pair = game_decks(seed, mirror_prob, decks, book)
    return bot_py_binding.play_with_streams(seed, [d.to_json() for d in deck_pair], bots(seed, pairs))
