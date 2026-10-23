"""Trains a set policy: the card policy's MLP (crates/ygoai/src/search/cardpolicy.rs) scoring each
card of a card selection whose candidates are sets of cards (several cards, a sum, tributes; one
at a time with finishing as the empty set), a set's score the sum of its cards' plus a weight per
card in it (and one for finishing), on `ygoai-card-refs --decisions` rows with "features" and
"sets". Loss: the cross-entropy of the sets' softmax against softmax(value / T) (values clamped
to CLAMP LP below the best). Held out: every HELD-th game. Reports, held out and per type, for
the policy, the candidates' order (the rules' cards first) and the rules' answer: how often the
best set is first / in the first 3 (the search's width), and the regret (LP) of the first set
and of the best of the first 3.

    .venv/bin/python scripts/train_set_policy.py OUT.json REFS.jsonl [REFS.jsonl ...] [--hidden 16]
"""

import argparse
import json
import zlib

import numpy as np
import torch


def load(paths, clamp, types):
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "sets" not in r or (types and r["type"] not in types):
                    continue
                v = np.array(r["values"], dtype=np.float64)
                if v.max() - v.min() < 1.0:
                    continue
                v = np.maximum(v, v.max() - clamp)
                x = np.array(r["features"], dtype=np.float32)
                member = np.zeros((len(r["sets"]), len(x)), dtype=np.float32)
                for s, cards in enumerate(r["sets"]):
                    member[s, cards] = 1.0
                rows.append(dict(x=x, v=v, member=member, rules=r["rules"], type=r["type"], game=f"{r['file']}:{r['line']}"))
    return rows


def pad(rows):
    b = len(rows)
    n = max(len(r["x"]) for r in rows)
    s = max(len(r["v"]) for r in rows)
    k = rows[0]["x"].shape[1]
    x = np.zeros((b, n, k), dtype=np.float32)
    member = np.zeros((b, s, n), dtype=np.float32)
    v = np.zeros((b, s), dtype=np.float32)
    mask = np.zeros((b, s), dtype=bool)
    cmask = np.zeros((b, n), dtype=bool)
    for i, r in enumerate(rows):
        nc, ns = r["member"].shape[1], len(r["v"])
        x[i, :nc], member[i, :ns, :nc], v[i, :ns], mask[i, :ns], cmask[i, :nc] = r["x"], r["member"], r["v"], True, True
    return tuple(torch.tensor(a) for a in (x, member, v, mask, cmask))


def metrics(score, v, mask):
    """score, v: [B, S] (masked sets: score -1e9). Best-in-top-k rates and regrets."""
    v = v.masked_fill(~mask, -1e9)
    best = v.max(1).values
    order = score.argsort(1, descending=True)
    ranked = v.gather(1, order)
    out = {}
    for k in (1, 2, 3):
        top = ranked[:, :k].max(1).values
        out[f"top{k}"] = (top >= best - 1e-6).float().mean().item()
        out[f"regret{k}"] = (best - top).mean().item()
    return out


class Net(torch.nn.Module):
    def __init__(self, k, hidden):
        super().__init__()
        self.l1 = torch.nn.Linear(k, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)
        # (per card in the set, and for finishing: the empty set)
        self.size = torch.nn.Parameter(torch.zeros(()))
        self.finish = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, member, mask):
        card = self.l2(torch.relu(self.l1(x))).squeeze(-1)
        count = member.sum(2)
        s = (member * card.unsqueeze(1)).sum(2) + self.size * count + self.finish * (count == 0).float()
        return s.masked_fill(~mask, -1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("refs", nargs="+")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--temp", type=float, default=500.0, help="T: the target's softmax temperature (LP)")
    ap.add_argument("--clamp", type=float, default=4000.0)
    ap.add_argument("--held", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zero", help="features (indices, comma-separated) set to 0, unused by the policy (e.g. 95,96: the rules' rank)")
    ap.add_argument("--types", help="only these types (comma-separated, as the rows' \"type\")")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    rows = load(a.refs, a.clamp, a.types.split(",") if a.types else None)
    zero = [int(i) for i in a.zero.split(",")] if a.zero else []
    for r in rows:
        r["x"][:, zero] = 0.0
    held = [r for r in rows if zlib.crc32(r["game"].encode()) % a.held == 0]
    train = [r for r in rows if zlib.crc32(r["game"].encode()) % a.held != 0]
    types = sorted({r["type"] for r in rows})
    print(f"{len(rows)} decisions with a spread ({len(train)} train, {len(held)} held out): " + ", ".join(f"{t} {sum(r['type'] == t for r in rows)}" for t in types))
    xt, mt, vt, kt, ct = pad(train)
    flat = xt[ct]
    mu, sd = flat.mean(0), torch.nan_to_num(flat.std(0), nan=1.0)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    held_by_type = {t: pad([r for r in held if r["type"] == t]) for t in types if any(r["type"] == t for r in held)}

    def baselines(t, rs):
        _, _, vh, kh, _ = held_by_type[t]
        first = (-torch.arange(vh.shape[1]).float()).expand_as(vh).masked_fill(~kh, -1e9)
        rules = first.clone()
        for i, r in enumerate(rs):
            rules[i, r["rules"]] = 1.0
        return {k: round(v, 3) for k, v in metrics(first, vh, kh).items()}, {k: round(v, 3) for k, v in metrics(rules, vh, kh).items()}

    for t in held_by_type:
        order, rules = baselines(t, [r for r in held if r["type"] == t])
        print(f"{t} held out ({len(held_by_type[t][2])}): candidates' order {order}, rules' answer {rules}")

    net = Net(xt.shape[2], a.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    target = torch.softmax(((vt - vt.masked_fill(~kt, -1e9).max(1, keepdim=True).values) / a.temp).masked_fill(~kt, -1e9), 1)
    batch = 256
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            s = net((xt[b] - mu) / sd, mt[b], kt[b])
            loss = -(target[b] * torch.log_softmax(s, 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        if epoch % 10 == 9 or epoch == a.epochs - 1:
            net.eval()
            report = []
            with torch.no_grad():
                for t, (xh, mh, vh, kh, _) in held_by_type.items():
                    m = metrics(net((xh - mu) / sd, mh, kh), vh, kh)
                    report.append(f"{t} {m['top1']:.3f}/{m['top2']:.3f}/{m['top3']:.3f}/{m['regret1']:.0f}/{m['regret2']:.0f}/{m['regret3']:.0f}")
            print(f"epoch {epoch + 1}: train loss {total / len(xt):.4f}; held out top1/2/3, regret1/2/3: " + ", ".join(report))

    # (the standardization folded into the first layer)
    w = net.l1.weight.detach() / sd
    w[:, zero] = 0.0  # (inputs always 0 in training: no effect there)
    b = net.l1.bias.detach() - (net.l1.weight.detach() * (mu / sd)).sum(1)
    model = {
        "features": xt.shape[2],
        "w1": w.tolist(),
        "b1": b.tolist(),
        "w2": net.l2.weight.detach()[0].tolist(),
        "b2": net.l2.bias.detach().item(),
        "size": net.size.item(),
        "finish": net.finish.item(),
        "trained_on": a.refs,
        "args": vars(a),
    }
    with open(a.out, "w") as f:
        json.dump(model, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
