"""Usage: scripts/eval_mover_features.py DATA.jsonl OUT.npz [--rust]

Mover-view features of the datagen rows into an npz (see FEATURES); with --rust the rows' own
Rust mover2 features instead (`mover2`, from ygoai-datagen or ygoai-refeature; `mover_x` in data
from before 3cf4fbf)."""
import json, sys
import numpy as np
sys.path.insert(0, "scripts")
from train_eval import side_features

PHASES = [(1,), (2,), (4,), (8, 16, 32, 64, 128), (256,), (512,)]
PROMPTS = ["SELECT_IDLECMD", "SELECT_CHAIN", "SELECT_CARD", "SELECT_BATTLECMD", "SELECT_EFFECTYN", "SELECT_YESNO", "SELECT_OPTION"]

def dominant(s, o):
    mons = [m for m in o["mzone"] if m is not None]
    best = max(((m[3] or 0) if m[2] else (m[4] or 0) for m in mons), default=-1)
    return sum(1 for m in s["mzone"] if m is not None and m[1] and m[2] and (m[3] or 0) > best)

def features(r):
    me, op = r["state"]
    f = side_features(me) + [dominant(me, op)] + side_features(op) + [dominant(op, me)]
    f.append(float(r["turn_player"] == r["player"]))
    f += [float(r["phase"] in ph) for ph in PHASES]
    f += [float(r["prompt"] == p) for p in PROMPTS]
    f.append(min(r["turn"], 30) / 10)
    return f

cols = {k: [] for k in ["x", "value", "static", "result", "game", "player", "turn", "depth"]}
for line in open(sys.argv[1]):
    r = json.loads(line)
    cols["x"].append(r.get("mover2", r.get("mover_x")) if "--rust" in sys.argv else features(r))
    for k in ["value", "static", "result", "game", "player", "turn", "depth"]:
        cols[k].append(r[k])
np.savez(sys.argv[2], **{k: np.asarray(v, np.float32 if k in ("x", "value", "static") else np.int64) for k, v in cols.items()})
