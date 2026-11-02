"""bot-refeature's `cards` rows -> the card lists of an npz's rows (datagen_to_npz.py's, the same runs
matched by (run, game, decision); the datagen rows give the order), as flat arrays: row i's cards are entries off[i]..off[i+1]; per entry `cidx`
(an index into `codes`), `bucket` (0 hand, 1 GY, 2 field), `mine`, `negated`, `used`.
The matched rows' `value` must equal the npz's (the check of the alignment).

  scripts/eval_cards/cards_to_npz.py OUT.npz BASE.npz ROWS.jsonl:CARDS.jsonl [ROWS:CARDS ...]
"""
import json
import multiprocessing as mp
import sys

import numpy as np

out, base, runs = sys.argv[1], sys.argv[2], [a.split(":") for a in sys.argv[3:]]


def key(r):
    return r["game"] * 1_000_000 + r["decision"]


def parse(lines):
    n = [0] * len(lines); ent = []; val = np.empty(len(lines), np.float32); k = np.empty(len(lines), np.int64)
    for i, l in enumerate(lines):
        r = json.loads(l); n[i] = len(r["cards"]); ent += r["cards"]; val[i] = r["value"]; k[i] = key(r)
    return np.array(n, np.int64), np.array(ent, np.int64).reshape(-1, 5), val, k


def keys(lines):
    return np.array([key(json.loads(l)) for l in lines], np.int64)


def chunks(path, size=50000):
    buf = []
    with open(path) as f:
        for l in f:
            buf.append(l)
            if len(buf) == size: yield buf; buf = []
    if buf: yield buf


ns, Es, vals = [], [], []
with mp.get_context("fork").Pool(8) as p:
    for rows, cards in runs:
        want = np.concatenate(list(p.imap(keys, chunks(rows), chunksize=1)))
        parts = list(p.imap(parse, chunks(cards), chunksize=1))
        n = np.concatenate([a[0] for a in parts]); E = np.concatenate([a[1] for a in parts])
        val = np.concatenate([a[2] for a in parts]); k = np.concatenate([a[3] for a in parts])
        del parts
        order = np.argsort(k); pos = np.searchsorted(k[order], want)
        assert np.array_equal(k[order][pos], want), "a datagen row without a card row"
        src = order[pos]  # (the card row of each datagen row)
        start = np.zeros(len(n) + 1, np.int64); np.cumsum(n, out=start[1:])
        idx = np.concatenate([np.arange(start[i], start[i + 1]) for i in src]) if len(src) else np.zeros(0, np.int64)
        ns.append(n[src]); Es.append(E[idx]); vals.append(val[src])
n = np.concatenate(ns); E = np.concatenate(Es); val = np.concatenate(vals)
del ns, Es, vals
ref = np.load(base, mmap_mode="r")["value"]
assert len(ref) == len(val) and np.array_equal(ref, val), "the rows do not align with the npz's"
codes, cidx = np.unique(E[:, 0], return_inverse=True)
off = np.zeros(len(n) + 1, np.int64); np.cumsum(n, out=off[1:])
np.savez(out, off=off, codes=codes.astype(np.int64), cidx=cidx.astype(np.int32), bucket=E[:, 1].astype(np.int8),
         mine=E[:, 2].astype(np.int8), negated=E[:, 3].astype(np.int8), used=E[:, 4].astype(np.int8))
print("rows", len(n), "entries", len(E), "distinct codes", len(codes), flush=True)
