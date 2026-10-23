"""Trains the card policy (crates/ygoai/src/search/cardpolicy.rs): an MLP scoring each card of a
one-card selection from its features, on ygoai-card-refs' exact values of every card. Loss: the
cross-entropy of the cards' softmax against softmax(value / T) (values clamped to CLAMP LP below
the best). Held out: every HELD-th game. Reports, held out, for the policy and the rules' order:
how often the best card is first / in the first 3 (the search's width) / in the first 5 (the
root's), and the regret (LP) of the first card and of the best of the first 3.

    .venv/bin/python scripts/train_card_policy.py OUT.json REFS.jsonl [REFS.jsonl ...] [--hidden 32]
"""

import argparse
import json
import zlib

import numpy as np
import torch


def load(paths, clamp, features=None):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                v = np.array(r["values"], dtype=np.float64)
                if v.max() - v.min() < 1.0:
                    continue
                v = np.maximum(v, v.max() - clamp)
                rank = np.empty(len(v), dtype=np.int64)
                rank[np.array(r["rules"])] = np.arange(len(v))
                game = f"{r['file']}:{r['line']}"
                rows.append((np.array(r["features"], dtype=np.float32)[:, :features], v, rank, game, r["hint"]))
    return rows


def pad(rows):
    n = max(len(r[1]) for r in rows)
    k = rows[0][0].shape[1]
    x = np.zeros((len(rows), n, k), dtype=np.float32)
    v = np.zeros((len(rows), n), dtype=np.float32)
    rank = np.full((len(rows), n), 10**6, dtype=np.int64)
    mask = np.zeros((len(rows), n), dtype=bool)
    for i, (f, vals, rk, _, _) in enumerate(rows):
        m = len(vals)
        x[i, :m], v[i, :m], rank[i, :m], mask[i, :m] = f, vals, rk, True
    return torch.tensor(x), torch.tensor(v), torch.tensor(rank), torch.tensor(mask)


def metrics(score, v, mask):
    """score, v: [B, N] (masked cards: score -inf). Best-in-top-k rates and regrets."""
    v = v.masked_fill(~mask, -1e9)
    best = v.max(1).values
    order = score.argsort(1, descending=True)
    ranked = v.gather(1, order)
    out = {}
    for k in (1, 3, 5):
        top = ranked[:, :k].max(1).values
        out[f"top{k}"] = (top >= best - 1e-6).float().mean().item()
        out[f"regret{k}"] = (best - top).mean().item()
    return out


class Net(torch.nn.Module):
    def __init__(self, k, hidden):
        super().__init__()
        self.l1 = torch.nn.Linear(k, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x))).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--temp", type=float, default=500.0, help="T: the target's softmax temperature (LP)")
    ap.add_argument("--clamp", type=float, default=4000.0)
    ap.add_argument("--held", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--features", type=int, help="only each card's first N features (an older feature set's)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    rows = load(a.refs, a.clamp, a.features)
    held = [r for r in rows if zlib.crc32(r[3].encode()) % a.held == 0]
    train = [r for r in rows if zlib.crc32(r[3].encode()) % a.held != 0]
    print(f"{len(rows)} positions with a spread ({len(train)} train, {len(held)} held out), cards per position {np.mean([len(r[1]) for r in rows]):.1f}")
    xt, vt, rt, mt = pad(train)
    xh, vh, rh, mh = pad(held)
    flat = xt[mt]
    mu, sd = flat.mean(0), flat.std(0)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)

    def rules_score(rank):
        return -rank.float()

    print("rules held out:", {k: round(v, 3) for k, v in metrics(rules_score(rh).masked_fill(~mh, -1e9), vh, mh).items()})
    net = Net(xt.shape[2], a.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    target = torch.softmax(((vt - vt.masked_fill(~mt, -1e9).max(1, keepdim=True).values) / a.temp).masked_fill(~mt, -1e9), 1)
    batch = 256
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            s = net((xt[b] - mu) / sd).masked_fill(~mt[b], -1e9)
            loss = -(target[b] * torch.log_softmax(s, 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if epoch % 10 == 9 or epoch == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                sh = net((xh - mu) / sd).masked_fill(~mh, -1e9)
                st = net((xt - mu) / sd).masked_fill(~mt, -1e9)
            m = {k: round(v, 3) for k, v in metrics(sh, vh, mh).items()}
            print(f"epoch {epoch + 1}: train loss {total / len(xt):.4f}, train top1 {metrics(st, vt, mt)['top1']:.3f}, held out {m}")

    # (the standardization folded into the first layer)
    w = net.l1.weight.detach() / sd
    b = net.l1.bias.detach() - (net.l1.weight.detach() * (mu / sd)).sum(1)
    model = {
        "features": xt.shape[2],
        "w1": w.tolist(),
        "b1": b.tolist(),
        "w2": net.l2.weight.detach()[0].tolist(),
        "b2": net.l2.bias.detach().item(),
        "trained_on": a.refs,
        "args": vars(a),
    }
    with open(a.out, "w") as f:
        json.dump(model, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
