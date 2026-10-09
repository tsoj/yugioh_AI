#!/usr/bin/env python3
"""A linear value function: logistic regression of the game result on the
antisymmetric hand-made features (ygoai/nn/value_features.py).

  scripts/train_value_linear.py runs/data/boot1_view.npz -o runs/nn/value_lin.json [--only d_lp,...]

No bias and features scaled only (not centered), so the logit of the other
player is exactly minus this one (antisymmetric, as negamax needs). Adam on
minibatches; the last --val share of the games is held out (log loss, accuracy).
The export holds the weights on the raw features (names, weights); the value
of a position is the logit of "the player to decide wins".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--only", default=None, help="comma-separated feature names (default: all)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val", type=float, default=0.05)
    args = ap.parse_args()
    d = np.load(args.data)
    names = [str(n) for n in d["vnames"]]
    cols = list(range(len(names))) if args.only is None else [names.index(n) for n in args.only.split(",")]
    X = d["vfeat"][:, cols].astype(np.float32)
    y = d["label"].astype(np.float32)
    games = d["game"]
    cut = int(games.max() * (1 - args.val))
    tr, va = games < cut, games >= cut
    scale = X[tr].std(0) + 1e-6
    Xs = torch.as_tensor(X / scale)
    yt = torch.as_tensor(y)
    w = torch.zeros(len(cols), requires_grad=True)
    opt = torch.optim.Adam([w], lr=args.lr)
    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
    tri, vai = np.flatnonzero(tr), np.flatnonzero(va)
    rng = np.random.default_rng(1)

    def held_out():
        with torch.no_grad():
            z = Xs[vai] @ w
            return loss_fn(z, yt[vai]).item(), ((z > 0).float() == yt[vai]).float().mean().item()
    for epoch in range(args.epochs):
        perm = rng.permutation(tri)
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            loss = loss_fn(Xs[idx] @ w, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl, vacc = held_out()
        print(f"epoch {epoch + 1}: held-out log loss {vl:.4f} accuracy {vacc:.3f}", flush=True)
    weights = (w.detach().numpy() / scale).tolist()
    vl, vacc = held_out()
    out = {"names": [names[c] for c in cols], "weights": weights, "held_out_loss": vl, "held_out_accuracy": vacc,
           "data": str(args.data), "positions": int(len(y))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    order = sorted(range(len(cols)), key=lambda i: -abs(w[i].item()))
    print("largest weights (per std of the feature):",
          ", ".join(f"{names[cols[i]]} {w[i].item():+.3f}" for i in order[:10]))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
