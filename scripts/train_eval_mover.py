"""Usage: scripts/train_eval_mover.py FEATURES.npz [--hidden 16] [--lam 0.5] [-o runs/nn/X.json]
(FEATURES.npz from scripts/eval_mover_features.py; games in blocks of 20 per deck pair, alternating
seat order, as runs/books/pairs500x20.jsonl).

Mover-view eval with a deck-pair term (experiment).

logit P(mover wins) = sgn * a[pair, seat order] + v(x); the search value's target is sigmoid(v(x))
alone (the search knows nothing of deck strength). a is dropped at inference.
"""
import argparse, math, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--model", default="mlp")
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--sigma", type=float, default=1.0, help="prior sd of the pair term (logits)")
ap.add_argument("--nopair", action="store_true")
ap.add_argument("--drop", default="", help="feature indices to zero, comma list")
ap.add_argument("-o", "--output")
ap.add_argument("--save-v", help="npz of the final v on all rows (for paired comparisons)")
args = ap.parse_args()
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

d = np.load(args.data)
X, value, static, result, game, player, turn = (d[k] for k in ["x", "value", "static", "result", "game", "player", "turn"])
if args.drop:
    X = X.copy(); X[:, [int(i) for i in args.drop.split(",")]] = 0
pair, k = game // 20, game % 20
ps = pair * 2 + (k % 2)
sgn = np.where(player == 0, 1.0, -1.0).astype(np.float32)

def fit_k(v, r):
    keep = (r != 0) & (np.abs(v) < 1000)
    v, y = v[keep].astype(np.float64), (r[keep] + 1) / 2
    best = None
    for kk in np.geomspace(1, 1000, 200):
        z = -v / kk
        p = 1 / (1 + np.exp(np.clip(z, -50, 50)))
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        if best is None or loss < best[0]:
            best = (loss, kk)
    return best[1]
K = fit_k(value, result)

rng = np.random.default_rng(0)
held_pairs = set(rng.choice(np.unique(pair), 50, replace=False).tolist())
is_hp = np.isin(pair, list(held_pairs))
is_vin = ~is_hp & (k >= 18)
is_tr = ~is_hp & ~is_vin
mu, sd = X[is_tr].mean(0), X[is_tr].std(0) + 1e-6
Z = torch.tensor((X - mu) / sd, device=dev)
Yr = torch.tensor((result + 1) / 2, dtype=torch.float32, device=dev)
Yv = torch.tensor(1 / (1 + np.exp(-np.clip(value, -50 * K, 50 * K) / K)), dtype=torch.float32, device=dev)
PS = torch.tensor(ps, device=dev); SG = torch.tensor(sgn, device=dev)
dec = torch.tensor(result != 0, device=dev)

n = X.shape[1]
if args.model == "linear":
    net = torch.nn.Linear(n, 1)
else:
    net = torch.nn.Sequential(torch.nn.Linear(n, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, 1))
net = net.to(dev)
a = torch.zeros(1000, device=dev, requires_grad=True)
opt = torch.optim.Adam([{"params": net.parameters(), "weight_decay": args.wd}, {"params": [a]}], lr=args.lr)
tr_idx = torch.tensor(np.nonzero(is_tr)[0], device=dev)
N = len(tr_idx)
bce = torch.nn.functional.binary_cross_entropy_with_logits

def v_of(idx):
    return net(Z[idx]).squeeze(-1)

def loss_of(idx, use_a=True):
    v = v_of(idx)
    lr_ = SG[idx] * a[PS[idx]] + v if (use_a and not args.nopair) else v
    m = dec[idx]
    l_res = bce(lr_[m], Yr[idx][m]) if m.any() else torch.zeros((), device=dev)
    l_val = bce(v, Yv[idx])
    return l_res, l_val

def ranking(vals, mask):
    """Within-pair concordance on the rows of mask: same pair, seat order, mover and turn; the
    game the mover won vs one they lost: P(mean v higher in the won game)."""
    import collections
    sel = np.nonzero(mask & (result != 0))[0]
    acc = collections.defaultdict(lambda: [0.0, 0])
    for i in sel:
        key = (ps[i], player[i], turn[i], game[i], result[i])
        e = acc[key]; e[0] += vals[i]; e[1] += 1
    groups = collections.defaultdict(lambda: ([], []))
    for (p, pl, t, g, r), (s, c) in acc.items():
        groups[(p, pl, t)][0 if r > 0 else 1].append(s / c)
    conc = tot = 0.0
    for w, l in groups.values():
        if not w or not l: continue
        w, l = np.array(w), np.array(l)
        cmp = (w[:, None] > l[None, :]).sum() + 0.5 * (w[:, None] == l[None, :]).sum()
        conc += cmp; tot += len(w) * len(l)
    return conc / max(tot, 1)

def all_v():
    with torch.no_grad():
        return torch.cat([v_of(torch.arange(i, min(i + 200000, len(X)), device=dev)) for i in range(0, len(X), 200000)]).cpu().numpy()

def report(tag):
    with torch.no_grad():
        out = []
        for name, m, ua in [("train", is_tr, True), ("val-games", is_vin, True), ("held-pairs", is_hp, False)]:
            idx = torch.tensor(np.nonzero(m)[0], device=dev)
            lr_, lv = loss_of(idx, ua)
            out.append(f"{name} res-ll {lr_.item():.4f} val-bce {lv.item():.4f}")
    v = all_v()
    print(f"{tag}: " + " | ".join(out) + f" | rank held-pairs {ranking(v, is_hp):.4f}", flush=True)
    return v

print(f"{len(X)} rows, K {K:.1f}; train {is_tr.sum()} val-games {is_vin.sum()} held-pairs {is_hp.sum()}")
print(f"baselines rank held-pairs: static {ranking(static, is_hp):.4f} search {ranking(value, is_hp):.4f}")
B = 8192
for ep in range(args.epochs):
    perm = tr_idx[torch.randperm(N, device=dev)]
    for i in range(0, N, B):
        b = perm[i:i + B]
        l_res, l_val = loss_of(b)
        loss = (1 - args.lam) * l_res + args.lam * l_val + (a ** 2).sum() / (2 * args.sigma ** 2) / N
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 3 == 2 or ep == args.epochs - 1:
        report(f"epoch {ep + 1}")
print("pair term: sd", a.detach().std().item())
if args.save_v:
    np.savez(args.save_v, v=all_v())
if args.output:
    import json
    assert args.model == "mlp"
    st = {kk: vv.detach().cpu().numpy().astype(float) for kk, vv in net.state_dict().items()}
    json.dump({"features": "mover2" if n == 87 else "mover", "model": "mover_mlp", "mu": mu.astype(float).tolist(), "scale": sd.astype(float).tolist(),
               "w1": st["0.weight"].tolist(), "b1": st["0.bias"].tolist(), "w2": (st["2.weight"][0] * K).tolist(),
               "b2": float(st["2.bias"][0] * K), "k": float(K), "lam": args.lam, "hidden": args.hidden}, open(args.output, "w"))
    print("saved", args.output)
