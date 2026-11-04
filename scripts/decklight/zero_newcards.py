"""main's eval (the per-code card table) with decklight's new cards (runs/decklight/newcard_codes.json,
decklight's newcard_codes(): the non-staple cards of its decks/newcards) left out of the table, as if
the eval had never been trained on them (a code without a table value has strength 0), and its LP
anchor recomputed by bot-anchor on the held-out pairs' positions with the modified table.

  systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 .venv/bin/python runs/decklight/zero_newcards.py
"""
import json
import subprocess

import numpy as np

MODEL = "crates/bot/models/eval_mover2_sprt2r23os_cardstable_64.json"
OUT = "runs/decklight/eval_no_newcards.json"
ROWS = "runs/decklight/eval_no_newcards_anchor_rows.jsonl"
ANCHOR = "build/eval-cards-target/release/bot-anchor"
N_ROWS = 200_000

new = set(json.load(open("runs/decklight/newcard_codes.json")))
m = json.load(open(MODEL))
c = m["cards"]
keep = [i for i, code in enumerate(c["codes"]) if code not in new]
print(f"table: {len(c['codes'])} codes, {len(c['codes']) - len(keep)} new cards left out")
c["codes"] = [c["codes"][i] for i in keep]
c["strength"] = [c["strength"][i] for i in keep]
json.dump(m, open(OUT, "w"))

# (the held-out pairs as train_eval_mover.py chooses them: --held 400, rng 0)
d = np.load("runs/data/sprt_net2/sprt2_r23_m2.npz")
pair = d["pair"]
rng = np.random.default_rng(0)
held = set(rng.choice(np.unique(pair), 400, replace=False).tolist())
hp = np.flatnonzero(np.isin(pair, list(held)))
sel = np.sort(np.random.default_rng(1).choice(hp, min(N_ROWS, len(hp)), replace=False))
X = d["x"][sel]
print(f"held-out rows {len(hp)}, anchor rows {len(sel)}")

# (card_x of each row: sum over its cards of sign * mult[bucket, state] * strength[code, bucket], the
# strengths already in the output's points)
cd = np.load("runs/data/sprt_net2_cards/sprt2_cards.npz")
off, codes = cd["off"], cd["codes"]
s_of = {code: np.array(v) for code, v in zip(c["codes"], c["strength"])}
S = np.zeros((len(codes), 3))
for j, code in enumerate(codes.tolist()):
    if code in s_of:
        S[j] = s_of[code]
mult = np.array(c["mult"])
cidx, bucket, mine, neg, used = (cd[k] for k in ("cidx", "bucket", "mine", "negated", "used"))
with open(ROWS, "w") as f:
    for i, row in zip(sel, X):
        a, b = off[i], off[i + 1]
        bk = bucket[a:b].astype(int)
        val = np.where(mine[a:b] > 0, 1.0, -1.0) * mult[bk, neg[a:b].astype(int) * 2 + used[a:b].astype(int)] * S[cidx[a:b], bk]
        cx = np.bincount(bk, weights=val, minlength=3)
        f.write(json.dumps({"mover2": row.astype(float).tolist(), "card_x": cx.tolist()}) + "\n")
subprocess.run([ANCHOR, OUT, ROWS, "--write"], check=True)
print("wrote", OUT, "lp_anchor", {k: v for k, v in json.load(open(OUT))["lp_anchor"].items() if k != "data"})
print("main's anchor", {k: v for k, v in json.load(open(MODEL))["lp_anchor"].items() if k != "data"})
