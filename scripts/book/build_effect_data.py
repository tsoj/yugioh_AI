#!/usr/bin/env python3
"""Training data for the effect model (scripts/book/train_effect_model.py): what
happens after a decision, from recorded games.

  scripts/book/build_effect_data.py runs/sprt/*.jsonl -o runs/data/effects1.npz [--keep 0.5] [--games N]

Replays each game and reads the full state (both decks known: the model learns
the rules, it does not play) at the prompts, on copies of the duel (the replay
goes on exactly as the game was played). A sample is a decision with its state, the decision
(prompt type, kind, the cards it names), and what followed until the next
prompt: every card's change (by its id in the core: new location, position,
ATK/DEF), which cards the messages name (activated, targeted, attacking, ...),
which message types occurred, the LP changes and the next prompt.

Arrays (one row per card and sample, samples by offsets "row_off"):
  rows: [id, code index, con, loc, seq, pos, atk, def, level, overlays, counters, public,
         target: new loc class, pos changed, atk/def changed, message references (bits of REF_TYPES)]
  ... see FIELDS / TARGETS below.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # (scripts/: common)

from common import constants as C  # noqa: E402
from common.parallel import Crash, run_parallel  # noqa: E402

ZONES = [("deck", C.LOCATION_DECK), ("hand", C.LOCATION_HAND), ("mzone", C.LOCATION_MZONE),
         ("szone", C.LOCATION_SZONE), ("grave", C.LOCATION_GRAVE), ("removed", C.LOCATION_REMOVED),
         ("extra", C.LOCATION_EXTRA)]
LOC_CLASS = {loc: i for i, (_, loc) in enumerate(ZONES)}
LOC_CLASS[C.LOCATION_OVERLAY] = len(ZONES)
GONE = len(ZONES) + 1  # (target class: not in the next state, e.g. a token)
FIELDS = ["id", "code", "con", "loc", "seq", "pos", "atk", "def", "level", "overlays", "counters", "public",
          "chain", "equip", "target"]  # (chain: the chain link it activated (1, 2, ...) or 0; equip, target: ids or -1)
TARGETS = ["t_loc", "t_pos", "t_stat", "t_refs"]
# (messages that name cards, by role; a card's t_refs has the bit of each that names it)
REF_TYPES = ["CHAINING", "SUMMONING", "SPSUMMONING", "FLIPSUMMONING", "SET", "POS_CHANGE", "BECOME_TARGET",
             "CONFIRM_CARDS", "CARD_SELECTED", "ATTACK_attacker", "ATTACK_target", "EQUIP", "CARD_HINT"]
PROMPTS = ["SELECT_IDLECMD", "SELECT_BATTLECMD", "SELECT_CHAIN", "SELECT_EFFECTYN", "SELECT_YESNO", "SELECT_OPTION",
           "SELECT_CARD", "SELECT_TRIBUTE", "SELECT_SUM", "SELECT_UNSELECT_CARD", "SELECT_PLACE", "SELECT_DISFIELD",
           "SELECT_POSITION", "SELECT_COUNTER", "SORT_CARD", "SORT_CHAIN", "ANNOUNCE_RACE", "ANNOUNCE_ATTRIB",
           "ANNOUNCE_CARD", "ANNOUNCE_NUMBER", "ROCK_PAPER_SCISSORS", "OTHER"]
SCALARS = ["lp0", "lp1", "turn", "turn_player", "phase", "chain"]
# (the commands of an idle and a battle command, the action's kind 1 + its index, and the prompt's
# list of each command's cards)
IDLE_KINDS = ["summon", "spsummon", "repos", "mset", "sset", "activate", "to_bp", "to_ep", "shuffle"]
IDLE_LISTS = {"summon": "summonable", "spsummon": "spsummonable", "repos": "repositionable",
              "mset": "msetable", "sset": "ssetable", "activate": "activatable"}
BATTLE_KINDS = ["activate", "attack", "to_m2", "to_ep"]
BATTLE_LISTS = {"activate": "activatable", "attack": "attackable"}


def _ptype(t: str) -> int:
    return PROMPTS.index(t) if t in PROMPTS else len(PROMPTS) - 1


def _rows(snap: dict) -> list[tuple]:
    """FIELDS of every card (the relations as ids of cards in the same state)."""
    where = {}
    for con, pl in enumerate(snap["players"]):
        for name, loc in ZONES:
            for seq, c in enumerate(pl[name]):
                if c is not None:
                    where[(con, loc, seq)] = c["id"]

    def ref(x):
        return where.get((x["con"], x["loc"], x["seq"]), -1) if isinstance(x, dict) else -1
    links = {}
    for k, link in enumerate(snap.get("chain") or (), 1):
        h = link.get("handler")
        if isinstance(h, dict):
            links.setdefault(ref(h), k)
    out = []
    for con, pl in enumerate(snap["players"]):
        for name, loc in ZONES:
            for seq, c in enumerate(pl[name]):
                if c is None:
                    continue
                level = c.get("level") or c.get("rank") or c.get("link") or 0
                targets = c.get("target_cards") or []
                out.append((c["id"], c.get("code") or 0, con, loc, seq, c.get("position", 0),
                            c.get("attack", 0), c.get("defense", 0), level, len(c.get("overlays") or ()),
                            sum(n for _, n in c.get("counters") or ()) if c.get("counters") else 0,
                            int(bool(c.get("is_public"))), links.get(c["id"], 0), ref(c.get("equip_card")),
                            ref(targets[0]) if targets else -1))
                for oid, ocode in zip(c.get("overlay_ids") or (), c.get("overlays") or ()):
                    out.append((oid, ocode, con, C.LOCATION_OVERLAY, seq, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1))
    return out


def _scalars(snap: dict) -> list[int]:
    return [snap["lp"][0], snap["lp"][1], snap["turn"], snap["turn_player"], snap["phase"], len(snap["chain"])]


def _locs(m: dict, *keys) -> list[tuple]:
    """(con, loc, seq) of the cards a message names under `keys` (a dict with con/loc/seq or a list of them)."""
    out = []
    for k in keys:
        v = m.get(k)
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, dict) and "loc" in x:
                out.append((x.get("con", 0), x["loc"], x.get("seq", 0)))
    if not keys and "loc" in m:
        out.append((m.get("con", 0), m["loc"], m.get("seq", 0)))
    return out


def _message_refs(m: dict) -> list[tuple[str, tuple]]:
    t = m["type"]
    if t in ("CHAINING", "SUMMONING", "SPSUMMONING", "FLIPSUMMONING", "SET", "POS_CHANGE", "CARD_HINT"):
        return [(t, x) for x in _locs(m)]
    if t in ("BECOME_TARGET", "CONFIRM_CARDS", "CARD_SELECTED", "RANDOM_SELECTED"):
        return [("CARD_SELECTED" if t == "RANDOM_SELECTED" else t, x) for x in _locs(m, "cards")]
    if t in ("ATTACK", "BATTLE"):
        return [("ATTACK_attacker", x) for x in _locs(m, "attacker")] + [("ATTACK_target", x) for x in _locs(m, "target")]
    if t in ("EQUIP", "CARD_TARGET"):
        return [("EQUIP", x) for x in _locs(m, "card", "target")]
    return []


def _action(prompt: dict, resp: dict) -> tuple[list[int], list[tuple]]:
    """(prompt type, kind, yes/no or option, number of cards; the (con, loc, seq) of the cards the response names)."""
    t = prompt["type"]
    kind, flag, cards = 0, 0, []

    def ref(c):
        if isinstance(c, dict):
            h = c.get("handler") if isinstance(c.get("handler"), dict) else c
            if "loc" in h:
                cards.append((h.get("con", 0), h["loc"], h.get("seq", 0)))
    if t in ("SELECT_IDLECMD", "SELECT_BATTLECMD"):
        kinds, lists = (IDLE_KINDS, IDLE_LISTS) if t == "SELECT_IDLECMD" else (BATTLE_KINDS, BATTLE_LISTS)
        kind = 1 + kinds.index(resp["kind"]) if resp.get("kind") in kinds else 0
        lst = lists.get(resp.get("kind"))
        if lst and "index" in resp and resp["index"] < len(prompt.get(lst, ())):
            ref(prompt[lst][resp["index"]])
    elif t == "SELECT_CHAIN":
        i = resp.get("index", -1)
        flag = int(i >= 0)
        if 0 <= i < len(prompt["chains"]):
            ref(prompt["chains"][i])
    elif t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
        flag = int(bool(resp.get("yes")))
        if t == "SELECT_EFFECTYN":
            ref(prompt)
    elif t in ("SELECT_OPTION", "ANNOUNCE_NUMBER"):
        flag = min(resp.get("index", 0), 15)
    elif t in ("SELECT_CARD", "SELECT_TRIBUTE", "SELECT_SUM"):
        for i in resp.get("indices") or ():
            if i < len(prompt.get("cards", ())):
                ref(prompt["cards"][i])
        flag = int(bool(resp.get("cancel")))
    elif t == "SELECT_UNSELECT_CARD":
        i = resp.get("index")
        pool = list(prompt.get("select_cards", ())) + list(prompt.get("unselect_cards", ()))
        if i is not None and i < len(pool):
            ref(pool[i])
        flag = int(bool(resp.get("finish")))
    elif t == "SELECT_POSITION":
        flag = resp.get("position", 0) & 0xF
    return [_ptype(t), kind, flag, len(cards)], cards


def _decisions(events: list[tuple], keep: float, rng: random.Random) -> list[list]:
    """The prompts of a replay's god view (ygoai_rs.replay_events), each [prompt, core response,
    state rows, scalars, the messages after the response, kept]; the state only at the kept prompts
    and the ones after them."""
    items: list[list] = []
    for event in events:
        if event[0] == "msg":
            if items:
                items[-1][4].append(event[1])
        elif event[0] == "prompt":
            prompt, snap = event[1], event[2]
            # (chain windows are most prompts and mostly passed: a quarter as many of those)
            kept = rng.random() < keep * (0.25 if prompt["type"] == "SELECT_CHAIN" else 1.0)
            if kept or (items and items[-1][5]):
                items.append([prompt, None, _rows(snap), _scalars(snap), [], kept])
            else:
                items.append([prompt, None, None, None, [], False])
        elif items and items[-1][1] is None:
            _, _, response, core = event
            items[-1][1] = core if core is not None else response
    return items


def samples(task):
    import ygoai_rs
    line, keep, seed = task
    record = json.loads(line)
    events, _ = ygoai_rs.replay_events(record, None, True)
    items = _decisions(events, keep, random.Random(seed))
    out = []
    for a, b in zip(items, items[1:]):
        prompt, resp, rows, scalars, msgs, kept = a
        if not kept or resp is None or rows is None or b[2] is None:
            continue
        nxt_prompt, nrows, nscalars = b[0], b[2], b[3]
        where = {(r[2], r[3], r[4]): r[0] for r in rows if r[3] != C.LOCATION_OVERLAY}
        where_after = {(r[2], r[3], r[4]): r[0] for r in nrows if r[3] != C.LOCATION_OVERLAY}
        after = {r[0]: r for r in nrows}
        refs: dict[int, int] = {}
        mtypes = set()
        for m in msgs:
            mtypes.add(m["type"])
            for role, loc in _message_refs(m):
                cid = where.get(loc, where_after.get(loc))
                if cid is not None and role in REF_TYPES:
                    refs[cid] = refs.get(cid, 0) | (1 << REF_TYPES.index(role))
        act, act_locs = _action(prompt, resp)
        act_ids = [where[x] for x in act_locs if x in where][:4]
        trows = []
        for r in rows:
            n = after.get(r[0])
            if n is None:
                tl, tp, ts = GONE, 0, 0
            else:
                tl = LOC_CLASS.get(n[3], GONE)
                tp = int(n[5] != r[5])
                ts = int(n[6] != r[6] or n[7] != r[7])
            trows.append(r + (tl, tp, ts, refs.get(r[0], 0)))
        out.append({"player": prompt.get("player", 0), "act": act, "act_ids": act_ids, "rows": trows,
                    "scalars": scalars, "msgs": sorted(mtypes), "next": _ptype(nxt_prompt["type"]),
                    "next_same": int(nxt_prompt.get("player", 0) == prompt.get("player", 0)),
                    "dlp": [nscalars[0] - scalars[0], nscalars[1] - scalars[1]]})
    return record["seed"], out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--keep", type=float, default=0.5, help="share of the decisions kept (at random)")
    ap.add_argument("--games", type=int, default=None, help="at most this many games, a random sample")
    ap.add_argument("-j", "--jobs", type=int, default=24)
    args = ap.parse_args()
    lines = [line for path in args.records for line in path.open() if line.strip()]
    random.Random(0).shuffle(lines)
    lines = lines[:args.games]
    tasks = [(line, args.keep, i) for i, line in enumerate(lines)]
    codes: dict[int, int] = {0: 0}
    msg_types: dict[str, int] = {}
    cols = {k: [] for k in ("game", "game_seed", "player", "act", "act_ids", "scalars", "msgs", "next", "next_same",
                            "dlp", "nrows")}
    rows = []  # (per game: an int16 array, the codes as indices)
    games = failed = 0
    for _, res in run_parallel(samples, tasks, args.jobs):
        if isinstance(res, Crash) or res is None:
            failed += 1
            continue
        seed, out = res
        for s in out:
            cols["game"].append(games)
            cols["game_seed"].append(seed)
            cols["player"].append(s["player"])
            cols["act"].append(s["act"])
            cols["act_ids"].append((s["act_ids"] + [-1] * 4)[:4])
            cols["scalars"].append(s["scalars"])
            bits = 0
            for t in s["msgs"]:
                bits |= 1 << msg_types.setdefault(t, len(msg_types))
            cols["msgs"].append(bits)
            cols["next"].append(s["next"])
            cols["next_same"].append(s["next_same"])
            cols["dlp"].append(s["dlp"])
            cols["nrows"].append(len(s["rows"]))
        if out:
            a = np.asarray([r for s in out for r in s["rows"]], dtype=np.int64)
            a[:, 1] = [codes.setdefault(int(c), len(codes)) for c in a[:, 1]]
            rows.append(np.clip(a, -32768, 32767).astype(np.int16))
        games += 1
        if games % 1000 == 0:
            print(f"{games} games, {len(cols['game'])} samples", flush=True)
    assert len(msg_types) <= 63, "message types do not fit a 64-bit mask"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_a = np.concatenate(rows)
    np.savez(args.output, rows=rows_a, row_off=np.concatenate([[0], np.cumsum(cols["nrows"])]).astype(np.int64),
             codes=np.asarray(sorted(codes, key=codes.get), dtype=np.int64),
             msg_types=np.asarray(sorted(msg_types, key=msg_types.get)),
             fields=np.asarray(FIELDS + TARGETS), ref_types=np.asarray(REF_TYPES), prompts=np.asarray(PROMPTS),
             scalar_names=np.asarray(SCALARS),
             **{k: np.asarray(v, dtype=np.int64 if k in ("msgs", "game_seed") else np.int32)
                for k, v in cols.items() if k != "nrows"})
    print(f"{games} games ({failed} failed), {len(cols['game'])} samples, {len(rows_a)} card rows, "
          f"{len(codes) - 1} distinct cards, {len(msg_types)} message types -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
