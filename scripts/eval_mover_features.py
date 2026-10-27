"""Usage: scripts/eval_mover_features.py DATA.jsonl OUT.npz [SET]

The datagen rows' Rust features (the set SET, default `mover2`, from bot-datagen or
bot-refeature; data from before 3cf4fbf names it `mover_x`) and their labels into an npz for
scripts/train_eval_mover.py."""
import json, sys
import numpy as np

key = sys.argv[3] if len(sys.argv) > 3 else "mover2"
labels = ["value", "static", "result", "game", "player", "turn", "depth"]
cols = {k: [] for k in ["x"] + labels}
for line in open(sys.argv[1]):
    r = json.loads(line)
    cols["x"].append(r[key])
    for k in labels:
        cols[k].append(r[k])
np.savez(sys.argv[2], **{k: np.asarray(v, np.float32 if k in ("x", "value", "static") else np.int64) for k, v in cols.items()})
