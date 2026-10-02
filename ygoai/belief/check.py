"""Consistency checks of the belief replay against recorded games."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

from ..game import GameRecord, ObservationLog, init_info, new_record, play, replay
from .replay import BeliefReplay, Divergence
from .tracker import TrackerError


def observation_stream(record: GameRecord, viewer: int, with_state: bool = True) -> list[tuple]:
    log = ObservationLog(viewer, with_state=with_state)
    replay(record, [log]).close()
    return log.items


def cut_point(items: list[tuple], frac: float) -> int | None:
    """Prefix length for a cut: a fraction of the game as in cut_points, or with
    a negative value -N the end of turn N (right after the viewer's last prompt
    before turn N + 1 starts; None if the game has no turn N + 1 or the viewer no
    prompt before it)."""
    if frac >= 0:
        return cut_points(items, (frac,))[0]
    turns = 0
    for i, it in enumerate(items):
        if it[0] == "msg" and it[1]["type"] == "NEW_TURN":
            turns += 1
            if turns == -frac + 1:
                return next((j + 1 for j in range(i - 1, -1, -1) if items[j][0] == "prompt"), None)
    return None


def cut_points(items: list[tuple], fractions: tuple[float, ...]) -> list[int]:
    """Prefix lengths ending right after a viewer prompt near each fraction (1.0 = everything)."""
    prompts = [i for i, it in enumerate(items) if it[0] == "prompt"]
    cuts = []
    for f in fractions:
        if f >= 1.0 or not prompts:
            cuts.append(len(items))
            continue
        target = f * len(items)
        best = min(prompts, key=lambda i: abs(i - target))
        cuts.append(best + 1)
    return cuts


@dataclass
class CheckResult:
    viewer: int
    cut: int
    length: int
    ok: bool
    stage: str = ""
    error: str = ""
    seconds: float = 0.0
    opponent_decisions: int = 0
    probed: int = 0
    probes: int = 0
    unknown_atoms: int = 0
    restarts: int = 0
    fixes: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    repairs: list = field(default_factory=list)  # "reason -> fix" of every restart
    end_ignored: bool = False  # the unfinished action the game ended in was not checked


def check_prefix(record: GameRecord, viewer: int, items: list[tuple], cut: int,
                 **replay_kwargs) -> CheckResult:
    prefix = items[:cut]
    res = CheckResult(viewer=viewer, cut=cut, length=len(items), ok=False)
    t0 = time.monotonic()
    stopped = cut == len(items) and (record.result or {}).get("reason") in ("turn_limit", "decision_limit", "time_limit")
    br = BeliefReplay(viewer, init_info(record, viewer), prefix, stopped=stopped, **replay_kwargs)
    br.log = lambda text: res.repairs.append(text) if text.startswith("restart") else None
    try:
        res.stage = "hypothesis"
        hyp = br.build_hypothesis()
        res.unknown_atoms = hyp.unknown_atoms
        res.stage = "replay"
        duel = br.run()
        duel.close()
        res.ok = True
        res.stage = "done"
    except Divergence as e:
        res.error = str(e)
        res.details = {"expected": e.expected, "got": e.got}
        last_prompt = max((i for i, it in enumerate(prefix) if it[0] == "prompt"), default=-1)
        if cut == len(items) and items and items[-1][0] == "msg" and items[-1][1]["type"] == "WIN" \
                and e.pos > last_prompt + 1 and last_prompt >= 0:
            # the game ended in the middle of an action (e.g. LP 0 or an empty deck
            # while its costs are paid), before what identifies its card shows:
            # check everything up to the viewer's last decision instead
            retry = BeliefReplay(viewer, init_info(record, viewer), items[:last_prompt + 1], stopped=True,
                                 **replay_kwargs)
            retry.log = br.log
            try:
                retry.build_hypothesis()
                retry.run().close()
                res.ok, res.stage, res.end_ignored = True, "done", True
                res.error, res.details = "", {}
                br = retry
            except (Divergence, TrackerError):
                pass
    except TrackerError as e:
        res.error = f"TrackerError: {e}"
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
    res.seconds = time.monotonic() - t0
    res.opponent_decisions = br.stats.opponent_decisions
    res.probed = br.stats.probed_decisions
    res.probes = br.stats.probes
    res.restarts = br.stats.restarts
    res.fixes = {k: getattr(br.stats, k) for k in ("forces", "assumptions", "swaps")
                 if getattr(br.stats, k)}
    return res


def check_game(record: GameRecord, fractions=(1 / 3, 2 / 3, 1.0), viewers=(0, 1),
               **replay_kwargs) -> list[CheckResult]:
    results = []
    for viewer in viewers:
        items = observation_stream(record, viewer)
        for cut in cut_points(items, fractions):
            results.append(check_prefix(record, viewer, items, cut, **replay_kwargs))
    return results


PAIRS = [("random", "random"), ("random", "greedy"), ("greedy", "random"), ("greedy", "greedy")]
_deck_pool = None


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
    possible (game `seed` uses its positions 2*seed and 2*seed + 1); with a `book`
    (see scripts/make_book.py) opening seed % (number of openings)."""
    import random
    from ..carddb import default_db
    from ..decks import DeckPool
    global _deck_pool
    if book:
        return tuple(load_book(book)[seed % len(load_book(book))])  # (the book decides the pairing)
    if _deck_pool is None and decks != "cards":
        _deck_pool = DeckPool(default_db())
    rng = random.Random(seed)
    if decks == "cards":
        from ..decks import RANDOM_DECK_STYLES, random_card_deck
        db = default_db()
        d0 = random_card_deck(db, rng, f"random cards {seed}a", rng.choice(RANDOM_DECK_STYLES))
        return d0, d0 if rng.random() < mirror_prob else random_card_deck(
            db, rng, f"random cards {seed}b", rng.choice(RANDOM_DECK_STYLES))
    if decks == "diverse":
        d0 = _deck_pool.diverse(2 * seed)
        return d0, d0 if rng.random() < mirror_prob else _deck_pool.diverse(2 * seed + 1)
    d0 = _deck_pool.sample(rng)
    return d0, d0 if rng.random() < mirror_prob else _deck_pool.sample(rng)


def make_game(seed: int, mirror_prob: float = 0.2, pairs: list[tuple[str, str]] | None = None,
              decks: str = "random", book: str | None = None, observers: list | None = None,
              deck_pair: tuple | None = None) -> GameRecord:
    """The test game for `seed`: the decks of game_decks (or `deck_pair`, e.g. chosen
    by a process that has the deck corpus loaded) and agent pairing
    pairs[seed % len(pairs)] (by default PAIRS)."""
    from ..agents import make_agent
    d0, d1 = deck_pair or game_decks(seed, mirror_prob, decks, book)
    pairs = pairs or PAIRS
    names = pairs[seed % len(pairs)]
    record = new_record([d0, d1], seed)
    play(record, [make_agent(names[0], seed * 2), make_agent(names[1], seed * 2 + 1)], time_limit=120,
         observers=observers)
    return record


def make_game_streams(seed: int, *args, **kwargs) -> tuple[GameRecord, list[list[tuple]]]:
    """make_game and both players' observation streams, recorded while the game
    is played (the same as observation_stream, without replaying the game)."""
    logs = [ObservationLog(v) for v in (0, 1)]
    record = make_game(seed, *args, observers=logs, **kwargs)
    streams = []
    for log in logs:
        items = log.items
        # (a replay stops at the win; the game itself goes on to the end of its step)
        end = next((i + 1 for i, it in enumerate(items) if it[0] == "msg" and it[1]["type"] == "WIN"), len(items))
        streams.append(items[:end])
    return record, streams
