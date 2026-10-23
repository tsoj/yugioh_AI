"""Per decision type, how much searching the decisions the rules answer below the root would
change: from `ygoai-card-refs --decisions` rows (each candidate's reference value, the rules'
answer's index).

    .venv/bin/python scripts/decision_refs_report.py runs/data/decision_refs/d1.jsonl

Columns: rows (games with the type, one row each), count (the type's decisions per such game),
cands (candidates), differ (the candidates' values differ by more than 50 LP), rules best (the
rules' answer within 50 LP of the best), regret (best minus the rules' answer, LP: mean, and
the share over 500 / 2000; a won or lost value counts as CLIP LP, mean regret with the clip),
mate (the best is a win the rules' answer is not, or the rules' answer a loss the best is not),
and regret x count (the expected loss per game with the type).
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--tol", type=float, default=50.0)
    ap.add_argument("--clip", type=float, default=8000.0)
    args = ap.parse_args()
    by_type = defaultdict(list)
    for path in args.refs:
        for line in open(path):
            r = json.loads(line)
            by_type[r["type"]].append(r)
    print(f"{'type':24} {'rows':>6} {'count':>6} {'cands':>6} {'differ':>7} {'rules best':>10} {'regret':>7} {'>500':>6} {'>2000':>6} {'mate':>6} {'x count':>8}")
    win = 5_000_000
    for t, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        v = [np.array(r["values"]) for r in rows]
        clip = [np.clip(x, -args.clip, args.clip) for x in v]
        regret = np.array([x.max() - x[r["rules"]] for x, r in zip(clip, rows)])
        mate = np.mean([(x.max() > win) != (x[r["rules"]] > win) or (x.max() < -win) != (x[r["rules"]] < -win) for x, r in zip(v, rows)])
        count = np.array([r.get("count", 1) for r in rows])
        differ = np.mean([x.max() - x.min() > args.tol for x in clip])
        print(
            f"{t:24} {len(rows):6d} {count.mean():6.2f} {np.mean([len(x) for x in v]):6.1f} {differ:7.3f} {np.mean(regret <= args.tol):10.3f}"
            f" {regret.mean():7.0f} {np.mean(regret > 500):6.3f} {np.mean(regret > 2000):6.3f} {mate:6.3f} {(regret * count).mean():8.0f}"
        )


if __name__ == "__main__":
    main()
