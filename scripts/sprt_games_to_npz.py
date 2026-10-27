"""rows_mover2.jsonl (ygoai-refeature of the SPRT games) + meta.jsonl + rounds 2+3's npz -> one npz
for train_eval_mover.py with explicit deck pairs: `pair`, `aseat` (the seat of the pair's first
deck), `valgame` (games held out within training pairs) and `src` (0: SPRT, 1/2: rounds 2/3).

  scripts/sprt_games_to_npz.py ROWS.jsonl META.jsonl ROUNDS23.npz OUT.npz
(ROWS: ygoai-refeature of scripts/collect_sprt_games.py's GAMES; OUT for scripts/train_eval_mover.py)"""
import json, sys, zlib
import numpy as np
import multiprocessing as mp
rows, meta, r23, out = sys.argv[1:5]
KEYS = ["value", "result", "game", "player", "turn", "depth"]

def parse(lines):
    n = len(lines); x = np.empty((n, 87), np.float32); c = {k: np.empty(n, np.float64 if k == "value" else np.int64) for k in KEYS}
    for i, l in enumerate(lines):
        r = json.loads(l); x[i] = r["mover2"]
        for k in KEYS: c[k][i] = r[k]
    return x, c

def chunks(path, size=50000):
    buf = []
    with open(path) as f:
        for l in f:
            buf.append(l)
            if len(buf) == size: yield buf; buf = []
    if buf: yield buf

with mp.get_context("fork").Pool(20) as p:
    parts = list(p.imap(parse, chunks(rows), chunksize=1))
X = np.concatenate([a for a, _ in parts]); C = {k: np.concatenate([b[k] for _, b in parts]) for k in KEYS}
del parts
print("sprt rows", len(X), flush=True)
M = [json.loads(l) for l in open(meta)]
keys = {}; gpair = np.array([keys.setdefault(m["pair"], len(keys)) for m in M]); gaseat = np.array([1 - m["order"] for m in M])
# (about 10% of each pair's games: validation games within the training pairs, by a hash of the game)
gval = np.array([zlib.crc32(str(i).encode()) % 10 == 0 for i in range(len(M))])
g = C["game"]
d = np.load(r23)
g23 = d["game"]; k23 = g23 % 20
off_game, off_pair = int(g23.max()) + 1, int(g23.max()) // 20 + 1
res = {
    "x": np.concatenate([X, d["x"]]),
    "value": np.concatenate([C["value"].astype(np.float32), d["value"]]),
    "static": np.concatenate([np.zeros(len(X), np.float32), d["static"]]),
    "result": np.concatenate([C["result"], d["result"]]),
    "game": np.concatenate([g + off_game, g23]),
    "player": np.concatenate([C["player"], d["player"]]),
    "turn": np.concatenate([C["turn"], d["turn"]]),
    "depth": np.concatenate([C["depth"], d["depth"]]),
    "pair": np.concatenate([gpair[g] + off_pair, g23 // 20]),
    "aseat": np.concatenate([gaseat[g], k23 % 2]),
    "valgame": np.concatenate([gval[g], k23 >= 18]),
    "src": np.concatenate([np.zeros(len(X), np.int64), d["src"] + 1]),
}
print({k: v.shape for k, v in res.items()}, "sprt pairs", len(keys), flush=True)
np.savez(out, **res)
