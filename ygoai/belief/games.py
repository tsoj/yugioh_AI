"""Test games for the belief checks: decks, agents, playing them and the
observation streams of both players. (Kept apart from the checks themselves:
the game cache key covers this module, so changing the checks does not make
every cached game be played again.)"""

from __future__ import annotations

from ..game import GameRecord, ObservationLog, new_record, play, replay


def observation_stream(record: GameRecord, viewer: int, with_state: bool = True) -> list[tuple]:
    log = ObservationLog(viewer, with_state=with_state)
    replay(record, [log]).close()
    return log.items


PAIRS = [("random", "random"), ("random", "greedy"), ("greedy", "random"), ("greedy", "greedy")]
_deck_pools: dict = {}


_books: dict = {}


def load_book(path) -> list:
    """The openings (pairs of decks) of a book written by scripts/make_book.py."""
    import json
    from ..decks import Deck
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
    import random
    from ..carddb import default_db
    from ..decks import DeckPool
    from ..paths import DECKS_DIR, NEW_DECKS_DIR
    if book:
        return tuple(load_book(book)[seed % len(load_book(book))])  # (the book decides the pairing)
    pool_name = "new" if decks.startswith("new") else decks
    if decks != "cards" and pool_name not in _deck_pools:
        _deck_pools[pool_name] = DeckPool(default_db(), NEW_DECKS_DIR if pool_name == "new" else DECKS_DIR)
    pool = _deck_pools.get(pool_name)
    rng = random.Random(seed)
    if decks == "cards":
        from ..decks import RANDOM_DECK_STYLES, random_card_deck
        db = default_db()
        d0 = random_card_deck(db, rng, f"random cards {seed}a", rng.choice(RANDOM_DECK_STYLES))
        return d0, d0 if rng.random() < mirror_prob else random_card_deck(
            db, rng, f"random cards {seed}b", rng.choice(RANDOM_DECK_STYLES))
    if decks == "new-cards":
        # (a human deck against a deck of random cards, on either side)
        from ..decks import RANDOM_DECK_STYLES, random_card_deck
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
    from ..carddb import default_db
    from ..decks import Deck
    db = default_db()
    counts: dict[int, int] = {}

    def take(cards, n):
        out = []
        for c in rng.sample(cards, len(cards)):
            base = db.get(c).alias or c
            if len(out) < n and counts.get(base, 0) < 3:
                counts[base] = counts.get(base, 0) + 1
                out.append(c)
        return out
    main = take(a.main, 20)
    main += take(b.main, 40 - len(main))
    main += take(a.main + b.main, 40 - len(main))
    extra = take(a.extra + b.extra, 15)
    return Deck(main=main, extra=extra, name=f"mix of {a.name} and {b.name}")


def make_game(seed: int, mirror_prob: float = 0.2, pairs: list[tuple[str, str]] | None = None,
              decks: str = "random", book: str | None = None, observers: list | None = None,
              deck_pair: tuple | None = None) -> GameRecord:
    """The test game for `seed`: the decks of game_decks (or `deck_pair`, e.g. chosen
    by a process that has the deck corpus loaded) and agent pairing
    pairs[seed % len(pairs)] (by default PAIRS)."""
    from ..agents import make_agent
    if observers is None:
        # (the game as checked: from the cache, e.g. a belief_mc game played by an
        # older version of the belief)
        return make_game_streams(seed, mirror_prob, pairs, decks, book, deck_pair)[0]
    d0, d1 = deck_pair or game_decks(seed, mirror_prob, decks, book)
    pairs = pairs or PAIRS
    names = pairs[seed % len(pairs)]
    record = new_record([d0, d1], seed)
    # (a time limit makes a game depend on the machine's speed: only for the slow
    # belief_mc agents, whose games the cache keeps)
    play(record, [make_agent(names[0], seed * 2), make_agent(names[1], seed * 2 + 1)],
         time_limit=120 if any("belief" in n for n in names) else None, observers=observers)
    return record


def make_game_streams(seed: int, mirror_prob: float = 0.2, pairs: list[tuple[str, str]] | None = None,
                      decks: str = "random", book: str | None = None,
                      deck_pair: tuple | None = None) -> tuple[GameRecord, list[list[tuple]]]:
    """make_game and both players' observation streams, recorded while the game
    is played (the same as observation_stream, without replaying the game).
    Games played before come from the on-disk cache (ygoai.game_cache)."""
    from .. import game_cache
    deck_pair = deck_pair or game_decks(seed, mirror_prob, decks, book)
    names = (pairs or PAIRS)[seed % len(pairs or PAIRS)]
    key = {"seed": seed, "pairs": [list(names)], "decks": [d.to_json() for d in deck_pair]}
    # (games of agents that use the belief are kept too when the belief changes: the
    # same games are checked again; they are played anew only when the game changes)
    with_belief = False
    cached = game_cache.load(key, with_belief)
    if cached is not None:
        return cached
    logs = [ObservationLog(v) for v in (0, 1)]
    record = make_game(seed, mirror_prob, pairs, decks, book, observers=logs, deck_pair=deck_pair)
    streams = []
    for log in logs:
        items = log.items
        # (a replay stops at the win; the game itself goes on to the end of its step)
        end = next((i + 1 for i, it in enumerate(items) if it[0] == "msg" and it[1]["type"] == "WIN"), len(items))
        streams.append(items[:end])
    game_cache.store(key, (record, streams), with_belief)
    return record, streams
