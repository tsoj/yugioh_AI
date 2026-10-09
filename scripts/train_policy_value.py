#!/usr/bin/env python3
"""Trains the policy/value network (ygoai/nn/model.py) on build_policy_data.py data.

  scripts/train_policy_value.py runs/data/boot1.npz -o runs/nn/boot1.npz [--epochs 10] [--dim 64]

Policy: cross-entropy of the target among each decision's candidates (scores
are dot products of the trunk's query with the candidates' embeddings). Value:
binary cross-entropy of "the player to decide won". The last --val share of
the games is held out: top-1 accuracy of the policy, loss and accuracy of the
value. The best epoch (by held-out loss) is exported for numpy inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.nn import features as F  # noqa: E402
from ygoai.nn.model import ACT_CATS, N_LOC  # noqa: E402


def expand_dynamic(raw: torch.Tensor) -> torch.Tensor:
    """The torch twin of features.expand_dynamic."""
    raw = raw.float()
    idx = raw[:, 0].long()
    pos = torch.zeros(len(raw), 4, device=raw.device)
    m = idx > 0
    pos[m, idx[m] - 1] = 1.0
    flags = raw[:, 5].long()
    return torch.cat([pos, raw[:, 1:2] / 5000.0, raw[:, 2:3] / 5000.0, raw[:, 3:4] / 5.0, raw[:, 4:5] / 5.0,
                      (flags & 1)[:, None].float(), (flags >> 1 & 1)[:, None].float()], dim=1)


class Net(nn.Module):
    def __init__(self, static_table: torch.Tensor, dim: int, hidden: int):
        super().__init__()
        self.register_buffer("static_table", static_table)
        card_in = F.STATIC_DIM + F.DYN_DIM
        self.card = nn.Sequential(nn.Linear(card_in, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        self.trunk = nn.Sequential(nn.Linear(len(F.GROUPS) * dim + F.SCALAR_DIM, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.query = nn.Linear(hidden, dim)
        self.act_cat = nn.ModuleList(nn.Embedding(n, dim) for n in ACT_CATS)
        self.act_loc = nn.Embedding(N_LOC, dim)
        self.act_own = nn.Embedding(2, dim)
        self.act_card = nn.Linear(card_in, dim, bias=False)
        self.dim = dim

    def forward(self, b):
        B = b["scalars"].shape[0]
        x = torch.cat([self.static_table[b["card_static"]], expand_dynamic(b["card_raw"])], dim=1)
        emb = self.card(x) * b["card_count"][:, None]
        pooled = torch.zeros(B * len(F.GROUPS), self.dim, device=x.device)
        pooled.index_add_(0, b["card_slot"], emb)
        h = self.trunk(torch.cat([pooled.view(B, -1), b["scalars"]], dim=1))
        value = self.value(h)[:, 0]
        q = self.query(h)
        cat = b["act_cat"]
        e = sum(self.act_cat[i](cat[:, i]) for i in range(4))
        e = e + self.act_loc(b["act_loc"] + 1) + self.act_own(b["act_own"])
        has = (b["act_static"] != 0).float()[:, None]
        ax = torch.cat([self.static_table[b["act_static"]], expand_dynamic(b["act_raw"])], dim=1)
        e = e + has * self.act_card(ax)
        scores = (e * q[b["act_dec"]]).sum(1)
        return value, scores


class Data:
    def __init__(self, path: Path, device):
        d = np.load(path)
        self.d = {k: d[k] for k in d.files}
        self.device = device
        self.card_off = np.concatenate([[0], np.cumsum(self.d["ncards"].astype(np.int64))])
        self.act_off = np.concatenate([[0], np.cumsum(self.d["nacts"].astype(np.int64))])
        self.n = len(self.d["target"])

    @staticmethod
    def _ranges(off, lens, ids):
        starts = off[ids]
        n = lens[ids].astype(np.int64)
        total = int(n.sum())
        rep = np.repeat(np.arange(len(ids)), n)
        return np.repeat(starts - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(total), rep

    def batch(self, ids: np.ndarray) -> dict:
        d, dev = self.d, self.device
        ci, cdec = self._ranges(self.card_off, d["ncards"], ids)
        ai, adec = self._ranges(self.act_off, d["nacts"], ids)
        t = lambda a, dtype=torch.long: torch.as_tensor(a, dtype=dtype, device=dev)  # noqa: E731
        nacts = d["nacts"][ids].astype(np.int64)
        return {"scalars": t(d["scalars"][ids], torch.float32),
                "card_static": t(d["card_static"][ci]), "card_raw": t(d["card_raw"][ci]),
                "card_count": t(d["card_count"][ci], torch.float32),
                "card_slot": t(cdec * len(F.GROUPS) + d["card_group"][ci]),
                "act_cat": t(d["act_cat"][ai]), "act_static": t(d["act_static"][ai]), "act_loc": t(d["act_loc"][ai]),
                "act_own": t(d["act_own"][ai]), "act_raw": t(d["act_raw"][ai]), "act_dec": t(adec),
                "nacts": nacts, "target": d["target"][ids].astype(np.int64), "label": t(d["label"][ids], torch.float32)}


def policy_loss(scores: torch.Tensor, b: dict):
    """Cross-entropy of the targets, top-1 hits (scores padded per decision)."""
    nacts = b["nacts"]
    B, A = len(nacts), int(nacts.max())
    pos = np.arange(len(scores)) - np.repeat(np.concatenate([[0], np.cumsum(nacts)[:-1]]), nacts)
    padded = torch.full((B, A), -1e9, device=scores.device)
    padded[b["act_dec"], torch.as_tensor(pos, device=scores.device)] = scores
    target = torch.as_tensor(b["target"], device=scores.device)
    loss = nn.functional.cross_entropy(padded, target)
    hits = (padded.argmax(1) == target).float().sum()
    return loss, hits


def evaluate(net, data, ids, batch):
    net.eval()
    tot = {"p": 0.0, "v": 0.0, "hit": 0.0, "vacc": 0.0}
    with torch.no_grad():
        for s in range(0, len(ids), batch):
            b = data.batch(ids[s:s + batch])
            value, scores = net(b)
            pl, hits = policy_loss(scores, b)
            vl = nn.functional.binary_cross_entropy_with_logits(value, b["label"], reduction="sum")
            n = len(b["nacts"])
            tot["p"] += pl.item() * n
            tot["v"] += vl.item()
            tot["hit"] += hits.item()
            tot["vacc"] += ((value > 0).float() == b["label"]).float().sum().item()
    net.train()
    return {k: v / len(ids) for k, v in tot.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init", type=Path, help="start from these weights (an export; its static table is not used)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(args.data, device)
    games = data.d["game"]
    cut = int(games.max() * (1 - args.val))
    train_ids, val_ids = np.flatnonzero(games < cut), np.flatnonzero(games >= cut)
    net = Net(torch.as_tensor(data.d["static_table"], device=device), args.dim, args.hidden).to(device)
    if args.init is not None:
        w = np.load(args.init)
        net.load_state_dict({k: torch.as_tensor(w[k], device=device) for k in w.files
                             if k not in ("codes", "static_table")}, strict=False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(args.seed)
    print(f"{device}: {len(train_ids)} train / {len(val_ids)} held-out decisions", flush=True)
    best, best_state = None, None
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = rng.permutation(train_ids)
        for s in range(0, len(perm), args.batch):
            b = data.batch(perm[s:s + args.batch])
            value, scores = net(b)
            pl, _ = policy_loss(scores, b)
            vl = nn.functional.binary_cross_entropy_with_logits(value, b["label"])
            loss = pl + args.value_weight * vl
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        m = evaluate(net, data, val_ids, 4096)
        score = m["p"] + args.value_weight * m["v"]
        print(f"epoch {epoch + 1}: held-out policy loss {m['p']:.4f} top-1 {m['hit']:.3f} | value loss {m['v']:.4f} "
              f"acc {m['vacc']:.3f} [{time.time() - t0:.0f}s]", flush=True)
        if best is None or score < best[0]:
            best = (score, epoch + 1, m)
            best_state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, codes=data.d["codes"], **best_state)
    meta = {"data": str(args.data), "epoch": best[1], "held_out": best[2], "dim": args.dim, "hidden": args.hidden,
            "epochs": args.epochs, "lr": args.lr,
            "init": None if args.init is None else str(args.init)}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"best epoch {best[1]}: {best[2]} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
