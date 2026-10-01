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


def check_prefix(record: GameRecord, viewer: int, items: list[tuple], cut: int,
                 **replay_kwargs) -> CheckResult:
    prefix = items[:cut]
    res = CheckResult(viewer=viewer, cut=cut, length=len(items), ok=False)
    t0 = time.monotonic()
    stopped = cut == len(items) and (record.result or {}).get("reason") in ("turn_limit", "decision_limit")
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


def make_game(seed: int, mirror_prob: float = 0.2, pairs: list[tuple[str, str]] | None = None) -> GameRecord:
    """The test game for `seed`: random decks (sometimes a mirror match) and
    agent pairing pairs[seed % len(pairs)] (by default PAIRS)."""
    import random
    from ..agents import make_agent
    from ..carddb import default_db
    from ..decks import DeckPool
    global _deck_pool
    if _deck_pool is None:
        _deck_pool = DeckPool(default_db())
    rng = random.Random(seed)
    d0 = _deck_pool.sample(rng)
    d1 = d0 if rng.random() < mirror_prob else _deck_pool.sample(rng)
    pairs = pairs or PAIRS
    names = pairs[seed % len(pairs)]
    record = new_record([d0, d1], seed)
    play(record, [make_agent(names[0], seed * 2), make_agent(names[1], seed * 2 + 1)], time_limit=120)
    return record
