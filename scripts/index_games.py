#!/usr/bin/env python3
"""Index the test games by the mechanics they use.

Plays (regenerates) the games of `check_belief.py` for a range of seeds and
writes one JSON line per game: decks, bots, length and a set of tags
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

from common import constants as C
from common.games import play_game
from common.parallel import Crash, run_parallel

# too common to be useful as a selection criterion
_COMMON = {"move", "hint", "new_turn", "new_phase", "draw", "chaining", "chained", "chain_solving",
           "chain_solved", "chain_end", "select_chain", "select_idlecmd", "select_battlecmd",
           "select_place", "select_card", "select_position", "summoning", "summoned", "spsummoning",
           "spsummoned", "set", "pos_change", "shuffle_deck", "shuffle_hand", "damage", "lpupdate",
           "attack", "battle", "damage_step_start", "damage_step_end", "win", "select_effectyn",
           "select_yesno", "confirm_cards", "retry", "select_option", "attack_disabled", "recover",
           "pay_lpcost", "missed_effect", "field_disabled", "player_hint", "card_selected"}


def message_tags(m: dict) -> set[str]:
    """A message's type, and what it shows about the mechanics."""
    t = m["type"]
    tags = {t.lower()}
    if t == "MOVE":
        if m["to"]["loc"] & C.LOCATION_OVERLAY:
            tags.add("xyz")
        if not m["from"]["loc"]:
            tags.add("created")
        if m["to"]["loc"] == C.LOCATION_REMOVED and not m["to"]["pos"] & C.POS_FACEUP:
            tags.add("banish_facedown")
        if m["from"]["loc"] == C.LOCATION_DECK and m["to"]["loc"] == C.LOCATION_DECK:
            tags.add("deck_reorder")
        if m["to"]["loc"] == C.LOCATION_SZONE and m["to"]["seq"] == 5:
            tags.add("field_spell")
    elif t == "SHUFFLE_EXTRA":
        tags.add("extra_shuffle")
    return tags


def index_game(seed: int) -> dict:
    import bot_py_binding
    record, _ = play_game(seed)
    events, _ = bot_py_binding.replay_events(record)
    tags = set()
    for event in events:
        if event[0] == "msg":
            tags |= message_tags(event[1])
        elif event[0] == "prompt":
            tags.add(event[1]["type"].lower())
    tags -= _COMMON
    names = [d["name"] for d in record["decks"]]
    if names[0] == names[1]:
        tags.add("mirror")
    turns = record["result"].get("turns", 0)
    if turns >= 20:
        tags.add("long")
    return {"seed": seed, "decks": names, "agents": record["agents"], "turns": turns,
            "decisions": len(record["responses"]), "tags": sorted(tags)}


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
