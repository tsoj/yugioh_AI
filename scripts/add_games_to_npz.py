"""More recorded games added to scripts/sprt_games_to_npz.py's npz and its card lists
(scripts/eval_cards/sprt_cards_to_npz.py): per set bot-refeature's rows (feature sets mover2 and
cards) and a meta file as scripts/collect_sprt_games.py writes it. Each set is a new `src` (its
search values in their own scale; rows of bot-refeature --unlabeled have the value NaN: the result
only). A set's deck pair that the base has (the same deck names) gets the base's pair number, so
held-out pairs stay held out; the others get new numbers after the base's. The base's rows keep
their numbers: train_eval_mover.py --held-below can draw the held-out pairs from the base alone.

  scripts/add_games_to_npz.py BASE.npz BASE_CARDS.npz BASE_META.jsonl OUT.npz OUT_CARDS.npz ROWS.jsonl META.jsonl [ROWS META ...]

(BASE_META: collect_sprt_games.py's meta of the base's SPRT games, for their pair numbers)
"""
import json
import multiprocessing as mp
import sys
import zipfile
import zlib

import numpy as np

base, base_cards, base_meta, out, out_cards = sys.argv[1:6]
sets = list(zip(sys.argv[6::2], sys.argv[7::2]))
KEYS = ["value", "result", "game", "player", "turn", "depth"]


def parse(lines):
    n = len(lines)
    x = np.empty((n, 87), np.float32)
    c = {k: np.empty(n, np.float64 if k == "value" else np.int64) for k in KEYS}
    cnt = np.empty(n, np.int64)
    codes, flags = [], []
    for i, l in enumerate(lines):
        r = json.loads(l)
        x[i] = r["mover2"]
        for k in KEYS:
            c[k][i] = np.nan if r[k] is None else r[k]
        cnt[i] = len(r["cards"])
        for code, b, m, ng, u in r["cards"]:
            codes.append(code)
            flags.append(b * 8 + m * 4 + ng * 2 + u)
    return x, c, cnt, np.array(codes, np.int32), np.array(flags, np.int8)


def chunks(path, size=50000):
    buf = []
    with open(path) as f:
        for l in f:
            buf.append(l)
            if len(buf) == size:
                yield buf
                buf = []
    if buf:
        yield buf


def write_member(z, name, parts, dtype, shape):
    """An .npy member of the npz written from parts (arrays along axis 0), not all in memory."""
    with z.open(name + ".npy", "w", force_zip64=True) as f:
        np.lib.format.write_array_header_2_0(f, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": shape})
        for p in parts:
            f.write(np.ascontiguousarray(p, dtype=dtype).tobytes())


d = np.load(base)
bsrc, bgame, bpair = d["src"], d["game"], d["pair"]
# (sprt_games_to_npz.py: the SPRT games' pair numbers are their keys' order in the meta, after the rounds')
off_pair = int(bpair[bsrc != 0].max()) + 1
keys = {}
for l in open(base_meta):
    keys.setdefault(json.loads(l)["pair"], len(keys))
pair_of = {k: i + off_pair for k, i in keys.items()}
next_pair, next_game, next_src = int(bpair.max()) + 1, int(bgame.max()) + 1, int(bsrc.max()) + 1
print("base rows", len(bsrc), "pairs", len(np.unique(bpair)), "(--held-below", int(bpair.max()) + 1, "for held-out pairs of the base only)", flush=True)

new = []  # per set: (x parts, columns, card counts, codes, flags)
for rows, meta in sets:
    M = [json.loads(l) for l in open(meta)]
    gp = []
    shared = 0
    for m in M:
        if m["pair"] not in pair_of:
            pair_of[m["pair"]] = next_pair
            next_pair += 1
        else:
            shared += m["pair"] in keys
        gp.append(pair_of[m["pair"]])
    gp = np.array(gp)
    gaseat = np.array([1 - m["order"] for m in M])
    gval = np.array([zlib.crc32(f"{rows}:{i}".encode()) % 10 == 0 for i in range(len(M))])
    with mp.get_context("fork").Pool(24) as p:
        parts = list(p.imap(parse, chunks(rows), chunksize=1))
    xs = [a[0] for a in parts]
    C = {k: np.concatenate([a[1][k] for a in parts]) for k in KEYS}
    cnt = np.concatenate([a[2] for a in parts]); codes = np.concatenate([a[3] for a in parts]); flags = np.concatenate([a[4] for a in parts])
    del parts
    g = C["game"]
    C.update(pair=gp[g], aseat=gaseat[g], valgame=gval[g], src=np.full(len(g), next_src), static=np.zeros(len(g), np.float32), game=g + next_game)
    print(f"{rows}: {len(g)} rows, {len(M)} games, src {next_src}, {len(np.unique(gp))} pairs ({shared} games in base pairs), "
          f"{np.isfinite(C['value']).sum()} rows with a search value", flush=True)
    new.append((xs, C, cnt, codes, flags))
    next_game += len(M)
    next_src += 1

# the npz: the base's arrays, then the sets' rows
n_new = sum(len(c["src"]) for _, c, *_ in new)
N = len(bsrc) + n_new
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
    for k in d.files:
        b = d[k]
        if k == "x":
            parts = [b] + [x for xs, *_ in new for x in xs]
        else:
            parts = [b] + [c[k] for _, c, *_ in new]
        write_member(z, k, parts, b.dtype, (N,) + b.shape[1:])
        del b
print("wrote", out, "rows", N, flush=True)

# the card lists: the base's, then the sets'
cb = np.load(base_cards)
assert len(cb["off"]) == len(bsrc) + 1
codes_all = np.unique(np.concatenate([cb["codes"]] + [s[3] for s in new]).astype(np.int64))
cidx = [np.searchsorted(codes_all, cb["codes"])[cb["cidx"]].astype(np.int32)]
bucket, mine, negated, used = [cb["bucket"]], [cb["mine"]], [cb["negated"]], [cb["used"]]
offs, last = [cb["off"][:-1]], int(cb["off"][-1])
for _, _, cnt, codes, flags in new:
    o = np.zeros(len(cnt), np.int64)
    np.cumsum(cnt[:-1], out=o[1:])
    offs.append(o + last)
    last += int(cnt.sum())
    cidx.append(np.searchsorted(codes_all, codes).astype(np.int32))
    bucket.append((flags >> 3).astype(np.int8)); mine.append(((flags >> 2) & 1).astype(np.int8))
    negated.append(((flags >> 1) & 1).astype(np.int8)); used.append((flags & 1).astype(np.int8))
off = np.concatenate(offs + [np.array([last])])
assert len(off) == N + 1 and off[-1] == sum(len(c) for c in cidx)
np.savez(out_cards, off=off, codes=codes_all, cidx=np.concatenate(cidx), bucket=np.concatenate(bucket), mine=np.concatenate(mine),
         negated=np.concatenate(negated), used=np.concatenate(used))
print("wrote", out_cards, "entries", off[-1], "codes", len(codes_all), flush=True)
