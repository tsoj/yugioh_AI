#!/usr/bin/env python3
"""Index the test games by the mechanics they use.

Plays (regenerates) the games of `check_belief.py` for a range of seeds and
writes one JSON line per game: decks, agents, length and a set of tags
(message and prompt types that occurred, plus derived ones like "xyz" or
"mirror"). `check_belief.py --index FILE --tags xyz,sort_card` then tests only
games with those mechanics.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai import constants as C  # noqa: E402
from ygoai.belief.check import make_game  # noqa: E402
from ygoai.game import Observer, replay  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402

# too common to be useful as a selection criterion
_COMMON = {"move", "hint", "new_turn", "new_phase", "draw", "chaining", "chained", "chain_solving",
           "chain_solved", "chain_end", "select_chain", "select_idlecmd", "select_battlecmd",
           "select_place", "select_card", "select_position", "summoning", "summoned", "spsummoning",
           "spsummoned", "set", "pos_change", "shuffle_deck", "shuffle_hand", "damage", "lpupdate",
           "attack", "battle", "damage_step_start", "damage_step_end", "win", "select_effectyn",
           "select_yesno", "confirm_cards", "retry", "select_option", "attack_disabled", "recover",
           "pay_lpcost", "missed_effect", "field_disabled", "player_hint", "card_selected"}


class Tagger(Observer):
    def __init__(self):
        self.tags: set[str] = set()

    def message(self, m: dict) -> None:
        t = m["type"]
        self.tags.add(t.lower())
        if t == "MOVE":
            if m["to"]["loc"] & C.LOCATION_OVERLAY:
                self.tags.add("xyz")
            if not m["from"]["loc"]:
                self.tags.add("created")
            if m["to"]["loc"] == C.LOCATION_REMOVED and not m["to"]["pos"] & C.POS_FACEUP:
                self.tags.add("banish_facedown")
            if m["from"]["loc"] == C.LOCATION_DECK and m["to"]["loc"] == C.LOCATION_DECK:
                self.tags.add("deck_reorder")
            if m["to"]["loc"] == C.LOCATION_SZONE and m["to"]["seq"] == 5:
                self.tags.add("field_spell")
        elif t == "SHUFFLE_EXTRA":
            self.tags.add("extra_shuffle")

    def prompt(self, p: dict, duel) -> None:
        self.tags.add(p["type"].lower())


def index_game(seed: int) -> dict:
    record = make_game(seed)
    tagger = Tagger()
    replay(record, [tagger]).close()
    tags = {t for t in tagger.tags if t not in _COMMON}
    if record.decks[0].name == record.decks[1].name:
        tags.add("mirror")
    turns = record.result.get("turns", 0)
    if turns >= 20:
        tags.add("long")
    return {"seed": seed, "decks": [d.name for d in record.decks], "agents": record.agents,
            "turns": turns, "decisions": len(record.responses), "tags": sorted(tags)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--games", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("-j", "--jobs", type=int, default=mp.cpu_count())
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    counts: dict[str, int] = {}
    with args.output.open("w") as f:
        for seed, res in run_parallel(index_game, range(args.start, args.start + args.games), args.jobs):
            if isinstance(res, Crash):
                res = {"seed": seed, "error": res.error, "tags": ["crash"]}
            for t in res["tags"]:
                counts[t] = counts.get(t, 0) + 1
            f.write(json.dumps(res) + "\n")
    for t, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{n:6d}  {t}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
