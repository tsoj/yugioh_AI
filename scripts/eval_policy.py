"""Scores policy models (the card, set and place policies' JSON: crates/ygoai/src/search/cardpolicy.rs)
on `ygoai-card-refs` rows: held out (every HELD-th game, as the trainers split them) or all, per
type, for each model and the rules: how often the best candidate is first / in the first 2 / 3,
and the regret (LP) of the first and of the best of the first 2 / 3. Rows with "sets" score each
set as the sum of its items' scores + size * |set| + finish * [empty]; rows without (one-card
selections) each card; the rules: "rules" (an order of the cards, or the answer's index first and
the candidates' order after it).

    .venv/bin/python scripts/eval_policy.py REFS.jsonl [REFS.jsonl ...] --model M.json [--model M.json ...] [--all]
"""

import argparse
import json
import zlib
from collections import defaultdict

import numpy as np


def load_model(path):
    m = json.load(open(path))
    return dict(w1=np.array(m["w1"]), b1=np.array(m["b1"]), w2=np.array(m["w2"]), b2=m["b2"], size=m.get("size", 0.0), finish=m.get("finish", 0.0))


def item_scores(m, x):
    return np.maximum(x @ m["w1"].T + m["b1"], 0.0) @ m["w2"] + m["b2"]


def candidate_scores(m, r, x):
    s = item_scores(m, x)
    if "sets" not in r:
        return s
    return np.array([s[c].sum() + m["size"] * len(c) + m["finish"] * (len(c) == 0) for c in r["sets"]])


def rules_scores(r, n):
    if isinstance(r["rules"], list):
        score = np.zeros(n)
        score[np.array(r["rules"])] = -np.arange(n)
        return score
    score = -np.arange(n, dtype=np.float64)
    score[r["rules"]] = 1.0
    return score


def ranked_metrics(score, v):
    order = np.argsort(-score, kind="stable")
    best = v.max()
    out = {}
    for k in (1, 2, 3):
        top = v[order[:k]].max()
        out[f"top{k}"] = float(top >= best - 1e-6)
        out[f"regret{k}"] = best - top
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--held", type=int, default=10)
    ap.add_argument("--all", action="store_true", help="every row, not only the held-out games'")
    ap.add_argument("--clamp", type=float, default=4000.0)
    a = ap.parse_args()
    models = {p: load_model(p) for p in a.model}

    acc = defaultdict(lambda: defaultdict(list))
    for path in a.refs:
        for line in open(path):
            r = json.loads(line)
            if not a.all and zlib.crc32(f"{r['file']}:{r['line']}".encode()) % a.held != 0:
                continue
            v = np.array(r["values"], dtype=np.float64)
            if v.max() - v.min() < 1.0:
                continue
            v = np.maximum(v, v.max() - a.clamp)
            x = np.array(r["features"], dtype=np.float64)
            t = r.get("type", "SELECT_CARD (one)")
            acc[t]["rules"].append(ranked_metrics(rules_scores(r, len(v)), v))
            for p, m in models.items():
                acc[t][p].append(ranked_metrics(candidate_scores(m, r, x[:, : m["w1"].shape[1]]), v))

    for t, by in sorted(acc.items()):
        print(f"{t} ({len(by['rules'])}): top1/2/3, regret1/2/3")
        for name, ms in by.items():
            mean = {k: np.mean([m[k] for m in ms]) for k in ms[0]}
            print(f"  {mean['top1']:.3f}/{mean['top2']:.3f}/{mean['top3']:.3f}, {mean['regret1']:.0f}/{mean['regret2']:.0f}/{mean['regret3']:.0f}  {name}")


if __name__ == "__main__":
    main()
