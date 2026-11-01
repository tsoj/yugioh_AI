"""bot-datagen's rows (mover2) and games -> an npz for train_eval_mover.py, put before an existing
one or alone (the newest data first: the export takes the first set's value scale): explicit
deck pairs (`pair`: the decks of a game, either order; `aseat`: the seat of the pair's first deck),
`valgame` (about 10% of the games by a hash) and `src` (0: these rows, the base's shifted by 1).
Several runs (ROWS:GAMES) are one data set: their games numbered on (the era replays: each
opening's second game, the same pair as its first).

  scripts/datagen_to_npz.py OUT.npz [--base BASE.npz] ROWS.jsonl:GAMES.jsonl [ROWS:GAMES ...]
"""
import json
import multiprocessing as mp
import sys
import zlib

import numpy as np

args = sys.argv[1:]
out, base = args.pop(0), None
if args[0] == "--base":
    base = args[1]; args = args[2:]
runs = [a.split(":") for a in args]
KEYS = ["value", "static", "result", "game", "player", "turn", "depth"]


def parse(lines):
    n = len(lines); x = np.empty((n, 87), np.float32); c = {k: np.empty(n, np.float64 if k in ("value", "static") else np.int64) for k in KEYS}
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


def deck_key(d):
    return json.dumps(sorted(d["main"]) + sorted(d["extra"]))


parts, offsets, off = [], [], 0
with mp.get_context("fork").Pool(8) as p:
    for rows, games in runs:
        part = list(p.imap(parse, chunks(rows), chunksize=1))
        for _, c in part:
            c["game"] += off
        parts += part; offsets.append(off)
        off += max(json.loads(l)["game"] for l in open(games)) + 1
X = np.concatenate([a for a, _ in parts]); C = {k: np.concatenate([b[k] for _, b in parts]) for k in KEYS}
del parts
# per game: its pair (both decks, either order) and the seat of the pair's first deck
keys, gpair, gaseat = {}, {}, {}
for (_, games), o in zip(runs, offsets):
    for l in open(games):
        r = json.loads(l)
        k = [deck_key(d) for d in r["record"]["decks"]]
        first = min(k)
        gpair[r["game"] + o] = keys.setdefault((first, max(k)), len(keys))
        gaseat[r["game"] + o] = k.index(first)
g = C["game"]
pair = np.array([gpair[i] for i in g]); aseat = np.array([gaseat[i] for i in g])
valgame = np.array([zlib.crc32(str(i).encode()) % 10 == 0 for i in g])
print("rows", len(X), "games", len(gpair), "pairs", len(keys), flush=True)

res = {"x": X, "value": C["value"].astype(np.float32), "static": C["static"].astype(np.float32), "result": C["result"], "game": g,
       "player": C["player"], "turn": C["turn"], "depth": C["depth"], "pair": pair, "aseat": aseat, "valgame": valgame,
       "src": np.zeros(len(g), np.int64)}
del X, C
if base:
    d = np.load(base)
    shift = {"game": int(g.max()) + 1, "pair": len(keys), "src": 1}
    res = {k: np.concatenate([v, d[k] + shift[k] if k in shift else d[k]]) for k, v in res.items()}
print({k: v.shape for k, v in res.items()}, flush=True)
np.savez(out, **res)
