"""bot-refeature's `cards` rows of the SPRT games -> the card lists of scripts/sprt_games_to_npz.py's
npz (its SPRT rows, src 0; the rows of rounds 2 and 3 get no cards), in cards_to_npz.py's format:
row i's cards are entries off[i]..off[i+1]; per entry `cidx` (an index into `codes`), `bucket`
(0 hand, 1 GY, 2 field), `mine`, `negated`, `used`.

The npz's rows of a game are matched to the card rows of that game in decision order; a game whose
row count or search values differ (a game that does not replay the same now) gets no cards.

  scripts/eval_cards/sprt_cards_to_npz.py OUT.npz BASE.npz CARDS.jsonl
"""
import json
import multiprocessing as mp
import sys

import numpy as np

out, base, cards = sys.argv[1:4]


def parse(lines):
    n = len(lines)
    game = np.empty(n, np.int64); dec = np.empty(n, np.int64); val = np.empty(n, np.float32); cnt = np.empty(n, np.int64)
    codes, flags = [], []
    for i, l in enumerate(lines):
        r = json.loads(l)
        game[i] = r["game"]; dec[i] = r["decision"]; val[i] = r["value"]; cnt[i] = len(r["cards"])
        for c, b, m, ng, u in r["cards"]:
            codes.append(c); flags.append(b * 8 + m * 4 + ng * 2 + u)
    return game, dec, val, cnt, np.array(codes, np.int32), np.array(flags, np.int8)


def chunks(path, size=100000):
    buf = []
    with open(path) as f:
        for l in f:
            buf.append(l)
            if len(buf) == size:
                yield buf
                buf = []
    if buf:
        yield buf


with mp.get_context("fork").Pool(24) as p:
    parts = list(p.imap(parse, chunks(cards), chunksize=1))
cg, cd, cv, cn = (np.concatenate([a[k] for a in parts]) for k in range(4))
code = np.concatenate([a[4] for a in parts]); flag = np.concatenate([a[5] for a in parts])
del parts
print("card rows", len(cg), "entries", len(code), flush=True)
cstart = np.zeros(len(cn) + 1, np.int64); np.cumsum(cn, out=cstart[1:])

d = np.load(base, mmap_mode="r")
src = np.asarray(d["src"]); game = np.asarray(d["game"]); value = np.asarray(d["value"])
sprt = np.flatnonzero(src == 0)
off_game = int(game[src != 0].max()) + 1  # (sprt_games_to_npz.py: the SPRT games after the rounds')
g = game[sprt] - off_game
# the npz's SPRT rows by (game, position), the card rows by (game, decision): the k-th of a game to the k-th
o_n = np.lexsort((sprt, g)); o_c = np.lexsort((cd, cg))
gn, gc = g[o_n], cg[o_c]
games = np.union1d(gn, gc)
cnt_n = np.bincount(np.searchsorted(games, gn), minlength=len(games))
cnt_c = np.bincount(np.searchsorted(games, gc), minlength=len(games))
same = cnt_n == cnt_c
print(f"games {len(games)}: row counts differ in {(~same).sum()} ({cnt_n[~same].sum()} npz rows)", flush=True)
ok_n = same[np.searchsorted(games, gn)]; ok_c = same[np.searchsorted(games, gc)]
rows_n = o_n[ok_n]; rows_c = o_c[ok_c]  # (paired in order)
vmatch = value[sprt[rows_n]] == cv[rows_c]
bad_games = np.unique(g[rows_n[~vmatch]])
print(f"value mismatches: {(~vmatch).sum()} rows in {len(bad_games)} games (those games get no cards)", flush=True)
keep = ~np.isin(g[rows_n], bad_games)
rows_n, rows_c = rows_n[keep], rows_c[keep]

N = len(value)
n_of = np.zeros(N, np.int64); src_row = np.full(N, -1, np.int64)
n_of[sprt[rows_n]] = cn[rows_c]; src_row[sprt[rows_n]] = rows_c
print(f"rows with cards {np.count_nonzero(src_row >= 0)} of {N} (SPRT {len(sprt)})", flush=True)
off = np.zeros(N + 1, np.int64); np.cumsum(n_of, out=off[1:])
has = np.flatnonzero(src_row >= 0)
starts = cstart[src_row[has]]; lens = n_of[has]
ent = np.repeat(starts - off[has], lens) + np.arange(off[-1])  # (each output entry's index into code/flag)
E_code = code[ent]; E_flag = flag[ent]
codes, cidx = np.unique(E_code, return_inverse=True)
np.savez(out, off=off, codes=codes.astype(np.int64), cidx=cidx.astype(np.int32), bucket=(E_flag >> 3).astype(np.int8),
         mine=((E_flag >> 2) & 1).astype(np.int8), negated=((E_flag >> 1) & 1).astype(np.int8), used=(E_flag & 1).astype(np.int8))
print("rows", N, "entries", len(E_code), "distinct codes", len(codes), flush=True)
