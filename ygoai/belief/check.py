"""Consistency checks of the belief replay against recorded games."""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field

from ..game import GameRecord, init_info
from .games import (PAIRS, _mixed_deck, game_decks, load_book, make_game,  # noqa: F401 (re-exported)
                    make_game_streams, observation_stream)
from .replay import BeliefReplay, Divergence
from .tracker import TrackerError


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
    last_prompt = max((i for i, it in enumerate(prefix) if it[0] == "prompt"), default=-1)
    if cut == len(items) and items and items[-1][0] == "msg" and items[-1][1]["type"] == "WIN" and last_prompt >= 0:
        br.tail_start = last_prompt + 2  # (see the retry below)
    try:
        res.stage = "hypothesis"
        hyp = br.build_hypothesis()
        res.unknown_atoms = hyp.unknown_atoms
        res.stage = "replay"
        duel = br.run()
        duel.close()
        res.ok = True
        res.stage = "done"
        if br.read_guided_fix and not os.environ.get("YGOAI_READ_GUIDED_OK"):
            # (a test failure until such repairs are known not to hide anything, e.g. a
            # leak that makes copies differ: the last resort is meant for the AI's use)
            res.ok = False
            res.error = "read-guided repair needed: " + "; ".join(br.read_guided_fix)
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
    res.fixes = {k: getattr(br.stats, k) for k in ("forces", "assumptions", "swaps", "read_guided")
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
