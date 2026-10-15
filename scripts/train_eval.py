#!/usr/bin/env python3
"""Train the search's evaluation on self-play data (crates/ygoai/src/bin/ygoai-datagen.rs).

  scripts/train_eval.py DATA.jsonl [...] --model linear|mlp [--hidden 16] [--features eval|rich]
                        [--lam 0.5] [-o runs/nn/eval_X.json]

Each row is a searched position: the mover's eval terms (`eval`), a fuller state (`state`), the
search's root value (`value`, in eval points) and the game's result for the mover. The model
predicts the mover's logit (its value in eval points is K times it, as exported); the loss compares sigmoid(v / K) with the target
lam * sigmoid(value / K) + (1 - lam) * (result + 1) / 2 (lam = 1: the search's value alone, 0: the
result alone). K, the scale of eval points to a win probability, is fitted first on the search
values against the results.

Both models are antisymmetric, as negamax needs (the value of the other player is minus this
one): the features are differences (mine minus the opponent's), the linear model has no bias, and
the MLP is v(x) = g(x) - g(-x). Games are split into train and validation (the last --val share
of the games). The export (JSON) holds the feature names, the normalization (features are
divided by `scale` before the model) and the weights, with the output in eval points.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

EVAL_NAMES = ["lp", "monsters", "dominant", "hand", "backrow", "deck_out"]
# (the search's weights, crates/ygoai/src/search/evaluate.rs EVAL_WEIGHTS)
EVAL_WEIGHTS = [1 / 100, 4.3, 17.6, 9.1, 5.1, 20.0]


def side_features(s: dict) -> list[float]:
    """One player's side as numbers (see rich_features)."""
    mons = [m for m in s["mzone"] if m is not None]
    up = [m for m in mons if m[1]]
    atk_up = [m for m in up if m[2]]
    backrow = [c for c in s["szone"][:5] if c is not None]
    stats = sorted((m[3] or 0) if m[2] else (m[4] or 0) for m in up)
    return [
        s["lp"] / 1000,
        s["hand"], s["deck"] / 10, s["grave"] / 10, s["removed"] / 10, s["extra"] / 10,
        len(mons), len(up), len(atk_up), len(mons) - len(up),
        sum((m[3] or 0) for m in atk_up) / 1000,
        (stats[-1] if stats else 0) / 1000,
        sum(1 for c in backrow if c[1]), sum(1 for c in backrow if not c[1]),
        int(s["szone"][5] is not None) if len(s["szone"]) > 5 else 0,
        int(s["deck"] <= 2),
    ]


RICH_NAMES = ["lp", "hand", "deck", "grave", "removed", "extra", "monsters", "faceup", "attackers",
              "facedown_mons", "atk_sum", "best_stat", "backrow_down", "backrow_up", "field_spell", "deck_out"]


def rich_features(row: dict) -> list[float]:
    """The fuller state as differences, mine minus the opponent's (antisymmetric), plus the
    dominance terms of the eval (which compare the two sides)."""
    a, b = (side_features(s) for s in row["state"])
    return [x - y for x, y in zip(a, b)] + [row["eval"][2]]


def load(paths: list[Path], rich: bool):
    X, value, result, game = [], [], [], []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                X.append(rich_features(r) if rich else r["eval"])
                value.append(r["value"])
                result.append(r["result"])
                game.append(r["game"])
    return (np.asarray(X, np.float32), np.asarray(value, np.float32), np.asarray(result, np.float32),
            np.asarray(game, np.int64))


def fit_k(value: np.ndarray, result: np.ndarray) -> float:
    """The scale K with the least log loss of sigmoid(value / K) against the results (decisive
    games' positions, values short of a found win)."""
    keep = (result != 0) & (np.abs(value) < 1000)
    v, y = value[keep], (result[keep] + 1) / 2
    best = None
    for k in np.geomspace(1, 1000, 200):
        p = 1 / (1 + np.exp(-v / k))
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        if best is None or loss < best[0]:
            best = (loss, k)
    return float(best[1])


class Linear(torch.nn.Module):
    def __init__(self, n: int, init: list[float] | None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32) if init else torch.zeros(n))

    def forward(self, x):
        return x @ self.w


class Mlp(torch.nn.Module):
    """v(x) = g(x) - g(-x) with g one hidden ReLU layer: antisymmetric."""

    def __init__(self, n: int, hidden: int):
        super().__init__()
        self.l1 = torch.nn.Linear(n, hidden)
        self.l2 = torch.nn.Linear(hidden, 1, bias=False)

    def g(self, x):
        return self.l2(torch.relu(self.l1(x))).squeeze(-1)

    def forward(self, x):
        return self.g(x) - self.g(-x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, nargs="+")
    ap.add_argument("--model", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--features", choices=["eval", "rich"], default="eval")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("-o", "--output", type=Path)
    args = ap.parse_args()
    torch.manual_seed(0)
    rich = args.features == "rich"
    X, value, result, game = load(args.data, rich)
    names = RICH_NAMES + ["dominant"] if rich else EVAL_NAMES
    k = fit_k(value, result)
    print(f"{len(X)} positions of {len(np.unique(game))} games; K = {k:.1f} (eval points per logit)")
    # (a win found by the search is clipped to a sure value)
    target = args.lam / (1 + np.exp(-np.clip(value, -50 * k, 50 * k) / k)) + (1 - args.lam) * (result + 1) / 2
    games = np.unique(game)
    val_games = set(games[int(len(games) * (1 - args.val)):].tolist())
    is_val = np.array([g in val_games for g in game])
    scale = np.maximum(np.abs(X[~is_val]).std(0), 1e-6).astype(np.float32)  # (scaled, not centered: antisymmetric)
    Xs = torch.tensor(X / scale)
    T = torch.tensor(target, dtype=torch.float32)
    R = torch.tensor((result + 1) / 2, dtype=torch.float32)
    tr, va = torch.tensor(~is_val), torch.tensor(is_val)
    init = [w * s / k for w, s in zip(EVAL_WEIGHTS, scale)] if args.model == "linear" and not rich else None
    model = Linear(X.shape[1], init) if args.model == "linear" else Mlp(X.shape[1], args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def losses(mask):
        with torch.no_grad():
            p = torch.sigmoid(model(Xs[mask]))
            t, r = T[mask], R[mask]
            mse = torch.mean((p - t) ** 2).item()
            dec = r != 0.5
            ll = torch.nn.functional.binary_cross_entropy(p[dec].clamp(1e-6, 1 - 1e-6), r[dec]).item()
            acc = ((p[dec] > 0.5) == (r[dec] > 0.5)).float().mean().item()
            return mse, ll, acc

    def report(tag):
        a, b = losses(tr), losses(va)
        print(f"{tag}: train mse {a[0]:.5f} result-ll {a[1]:.4f} acc {a[2]:.3f} | val mse {b[0]:.5f} "
              f"result-ll {b[1]:.4f} acc {b[2]:.3f}")

    if args.model == "linear" and not rich:
        report("search's weights")
    idx = torch.nonzero(tr).squeeze(1)
    for epoch in range(args.epochs):
        perm = idx[torch.randperm(len(idx))]
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            p = torch.sigmoid(model(Xs[b]))
            loss = torch.mean((p - T[b]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            report(f"epoch {epoch + 1}")
    out = {"features": args.features, "names": names, "scale": scale.tolist(), "k": k, "lam": args.lam,
           "model": args.model}
    if args.model == "linear":
        w = (model.w.detach().numpy() / scale * k).tolist()
        out["weights"] = w
        print("weights on the raw features:", {n: round(x, 4) for n, x in zip(names, w)})
    else:
        out["hidden"] = args.hidden
        out["w1"] = model.l1.weight.detach().numpy().tolist()
        out["b1"] = model.l1.bias.detach().numpy().tolist()
        out["w2"] = (model.l2.weight.detach().numpy()[0] * k).tolist()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1))
        print("saved", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
