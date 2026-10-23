"""Trains a gate for zone selections: whether the choice matters (its candidates' reference values
spread more than --spread LP; the values clipped to +/- --clip), from the zones' features (`ygoai-card-refs --decisions` rows of
SELECT_PLACE / SELECT_DISFIELD with "features") pooled over the zones (their mean and max). The
search branches on a zone selection only where the gate opens. Held out: every HELD-th game.
Reports the AUC, and for thresholds the share of decisions the gate opens for, of those that
matter, and of the spread (LP) and the rules' regret they hold.

    .venv/bin/python scripts/train_place_gate.py OUT.json REFS.jsonl [REFS.jsonl ...] [--spread 50]
"""

import argparse
import json
import zlib

import numpy as np
import torch

TYPES = ("SELECT_PLACE", "SELECT_DISFIELD")


def load(paths, clip):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["type"] not in TYPES or "features" not in r:
                    continue
                x = np.array(r["features"], dtype=np.float32)
                v = np.clip(np.array(r["values"], dtype=np.float64), -clip, clip)
                pooled = np.concatenate([x.mean(0), x.max(0)])
                rows.append(dict(x=pooled, spread=v.max() - v.min(), regret=v.max() - v[r["rules"]], game=f"{r['file']}:{r['line']}"))
    return rows


def auc(score, label):
    order = np.argsort(-score)
    label = label[order]
    pos, neg = label.sum(), (1 - label).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    # (the share of positive-negative pairs ranked right)
    return float(((1 - label) * np.cumsum(label)).sum() / (pos * neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--spread", type=float, default=50.0, help="a choice matters above this spread (LP)")
    ap.add_argument("--clip", type=float, default=8000.0, help="values clipped to +/- this (won and lost: 8000 LP)")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--held", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--open", type=float, default=0.9, help="the threshold written: opening for this share of the held-out decisions that matter")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    rows = load(a.refs, a.clip)
    held = [r for r in rows if zlib.crc32(r["game"].encode()) % a.held == 0]
    train = [r for r in rows if zlib.crc32(r["game"].encode()) % a.held != 0]
    stack = lambda rs: (torch.tensor(np.stack([r["x"] for r in rs])), torch.tensor([float(r["spread"] > a.spread) for r in rs]))
    xt, yt = stack(train)
    xh, yh = stack(held)
    print(f"{len(rows)} decisions ({len(train)} train, {len(held)} held out); matter (spread > {a.spread:.0f}): train {yt.mean():.3f}, held out {yh.mean():.3f}")
    mu, sd = xt.mean(0), xt.std(0)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)

    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], a.hidden), torch.nn.ReLU(), torch.nn.Linear(a.hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    batch = 256
    spread_h = np.array([r["spread"] for r in held])
    regret_h = np.array([r["regret"] for r in held])
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            loss = loss_fn(net((xt[b] - mu) / sd).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if epoch % 10 == 9 or epoch == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                s = net((xh - mu) / sd).squeeze(1).numpy()
            print(f"epoch {epoch + 1}: train loss {total / len(xt):.4f}; held out AUC {auc(s, yh.numpy()):.3f}")

    # (the held-out trade-off: open for the top share of decisions by the gate's score)
    y = yh.numpy()
    order = np.argsort(-s)
    print("held out: opened share -> of those that matter / of the spread / of the rules' regret; threshold")
    for share in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        k = max(1, int(round(share * len(s))))
        top = order[:k]
        print(
            f"  {share:.2f} -> {y[top].sum() / max(y.sum(), 1):.3f} / {spread_h[top].sum() / max(spread_h.sum(), 1e-9):.3f} / "
            f"{regret_h[top].sum() / max(regret_h.sum(), 1e-9):.3f}; {s[order[k - 1]]:.3f}"
        )
    matter = np.sort(s[y > 0])[::-1]
    threshold = float(matter[min(len(matter) - 1, int(a.open * len(matter)))]) if len(matter) else 0.0
    print(f"threshold {threshold:.3f}: opens for {(s >= threshold).mean():.3f} of the held-out decisions, {(s[y > 0] >= threshold).mean():.3f} of those that matter")

    # (the standardization folded into the first layer)
    l1, l2 = net[0], net[2]
    w = l1.weight.detach() / sd
    b = l1.bias.detach() - (l1.weight.detach() * (mu / sd)).sum(1)
    model = {
        "features": xt.shape[1],
        "w1": w.tolist(),
        "b1": b.tolist(),
        "w2": l2.weight.detach()[0].tolist(),
        "b2": l2.bias.detach().item(),
        "threshold": threshold,
        "trained_on": a.refs,
        "args": vars(a),
    }
    with open(a.out, "w") as f:
        json.dump(model, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
