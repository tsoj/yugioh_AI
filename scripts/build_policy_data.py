#!/usr/bin/env python3
"""Training data for the policy and value networks, from self-play records.

  scripts/build_policy_data.py runs/data/boot1.jsonl -o runs/data/boot1.npz [--games N] [--view]

Replays each game (gen_selfplay.py records: game records with "targets") and,
at each decision with a policy target (in a forked process, see _Decisions), takes the position from the deciding
player's side (ygoai/nn/features.py) and the decision's candidates, with the
index of the target among them, whether that player won, and the value's
features (ygoai/nn/value_features.py, always the full state). The position and
candidates: the full state of the real game and the core's prompt, or with
--view what the deciding player saw (its view of the state, the prompt as it got
it: hidden cards unknown). Card codes become static indices into the saved
table of static features (no identities reach the networks).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # (the workers fork per decision, see _Decisions)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai import constants as C  # noqa: E402
from ygoai.agents.belief_mc import candidates  # noqa: E402
from ygoai.carddb import default_db  # noqa: E402
from ygoai.game import GameRecord, Observer, replay  # noqa: E402
from ygoai.nn import features as F  # noqa: E402
from ygoai.nn import value_features as V  # noqa: E402
from ygoai.observe import filter_prompt, to_core_response  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402
from ygoai.state import capture, view  # noqa: E402


def _code(code):
    return code or 0


class _Decisions(Observer):
    def __init__(self, targets: dict[int, dict], player_view: bool):
        self.targets, self.player_view = targets, player_view
        self.i = 0  # (the response the next prompt gets)
        self.hint = [None, None]
        self.rows = []

    def message(self, msg: dict) -> None:
        if msg["type"] == "HINT" and msg["hint_type"] == C.HINT_SELECTMSG:
            self.hint[msg["player"]] = msg["data"]

    def prompt(self, prompt: dict, duel) -> None:
        i, self.i = self.i, self.i + 1
        me = prompt["player"]
        hint, self.hint[me] = self.hint[me], None
        target = self.targets.get(i)
        if target is None:
            return
        # (the queries in a forked process: reading the state changes the core's memory
        # allocations, and with them where the core iterates by address, so the game
        # could go on differently than it was played)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            data = b""
            try:
                snap = capture(duel)
                if self.player_view:
                    seen, shown = view(snap, me), filter_prompt(prompt, me)
                    chosen = target
                else:
                    seen, shown = snap, prompt
                    chosen = to_core_response(prompt, me, target)
                cands = candidates(shown)
                if chosen in cands:
                    data = pickle.dumps((F.position(seen, me, _code), F.actions(shown, cands, seen, me, hint, _code),
                                         cands.index(chosen), V.features(snap, me)))
            finally:
                with os.fdopen(w, "wb") as f:
                    f.write(data)
                os._exit(0)
        os.close(w)
        with os.fdopen(r, "rb") as f:
            data = f.read()
        os.waitpid(pid, 0)
        if data:
            (cards, scalars), acts, chosen, vfeat = pickle.loads(data)
            self.rows.append((me, cards, scalars, acts, chosen, vfeat))


def decisions(task):
    line, player_view = task
    d = json.loads(line)
    record = GameRecord.from_json(d)
    winner = record.result.get("winner")
    if winner not in (0, 1):
        return None
    obs = _Decisions({int(k): v for k, v in d.get("targets", {}).items()}, player_view)
    replay(record, [obs]).close()
    return winner, obs.rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--games", type=int, default=None, help="at most this many games")
    ap.add_argument("-j", "--jobs", type=int, default=24)
    ap.add_argument("--view", action="store_true", help="what the deciding player saw (else the full state)")
    args = ap.parse_args()
    lines = [(line, args.view) for path in args.records for line in path.open() if line.strip()][:args.games]
    codes = F.CodeIndex()
    dec = {k: [] for k in ("game", "player", "label", "target", "ncards", "nacts")}
    scal, vfeat, cards, acts = [], [], [], []  # (per game: numpy arrays, the codes as static indices)
    failed = games = 0
    for _, res in run_parallel(decisions, lines, args.jobs):
        if isinstance(res, Crash) or res is None:
            failed += isinstance(res, Crash)
            continue
        winner, rows = res
        if not rows:
            continue
        for me, cs, sc, ac, target, vf in rows:
            dec["game"].append(games)
            dec["player"].append(me)
            dec["label"].append(1.0 if me == winner else 0.0)
            dec["target"].append(target)
            dec["ncards"].append(len(cs))
            dec["nacts"].append(len(ac))
            scal.append(sc)
            vfeat.append(vf)
        c = np.asarray([(cd, g, n, *raw) for _, cs, _, _, _, _ in rows for cd, g, n, raw in cs], dtype=np.int64)
        a = np.asarray([(pt, k, h, e, cd, loc, own, *raw) for _, _, _, ac, _, _ in rows
                        for pt, k, h, e, cd, loc, own, raw in ac], dtype=np.int64)
        lookup = {code: codes(int(code)) for code in np.unique(np.concatenate([c[:, 0], a[:, 4]]))}
        c[:, 0] = [lookup[x] for x in c[:, 0]]
        a[:, 4] = [lookup[x] for x in a[:, 4]]
        cards.append(c.astype(np.int32))
        acts.append(a.astype(np.int32))
        games += 1
        if games % 2000 == 0:
            print(f"{games} games, {len(scal)} decisions", flush=True)
    cards = np.concatenate(cards)
    acts = np.concatenate(acts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             game=np.asarray(dec["game"], dtype=np.int32), player=np.asarray(dec["player"], dtype=np.int8),
             label=np.asarray(dec["label"], dtype=np.float32), target=np.asarray(dec["target"], dtype=np.int16),
             ncards=np.asarray(dec["ncards"], dtype=np.int16), nacts=np.asarray(dec["nacts"], dtype=np.int16),
             scalars=np.asarray(scal, dtype=np.float32), vfeat=np.asarray(vfeat, dtype=np.float32),
             vnames=np.asarray(V.NAMES), player_view=np.asarray(args.view),
             card_static=cards[:, 0], card_group=cards[:, 1].astype(np.int8), card_count=cards[:, 2].astype(np.int8),
             card_raw=cards[:, 3:].astype(np.int16),
             act_cat=acts[:, :4].astype(np.int8), act_static=acts[:, 4], act_loc=acts[:, 5].astype(np.int8),
             act_own=acts[:, 6].astype(np.int8), act_raw=acts[:, 7:].astype(np.int16),
             static_table=np.asarray(codes.table(default_db()), dtype=np.float32),
             codes=np.asarray(codes.codes, dtype=np.int64))
    print(f"{games} games, {len(scal)} decisions, {len(cards)} card rows, {len(acts)} candidates, "
          f"{len(codes.codes)} distinct cards ({failed} games failed) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
