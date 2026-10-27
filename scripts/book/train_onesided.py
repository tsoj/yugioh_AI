"""Predicts how often an opening's result is decided by the seat (the same seat wins both games of
a pair, whoever plays it), from its cards: both decks (main, extra) and the top cards of each
deck (the opening hands and first draws), each card its text embedding (scripts/book/embed_cards.py:
cards no training game had get one too) or, as a control, a learned per-card embedding (unseen
cards: none). Data: scripts/book/build_onesided_data.py (per opening the pairs played and how many
the seat decided); loss: the binomial likelihood of those counts.

Splits: --split none (every opening trains: the model to use, --out), random (every --held-th
opening held out) or archetype: archetypes (setcodes) that are
rare (in at most 3% of the decks) are held out at random until --test openings have a deck with
at least a quarter of its cards from them; training openings have no card of them at all (and
--match-random gives a random split with as many training openings, the comparison).

Inputs (--inputs): deck+top (the decks as sets, the top cards with their draw positions), deck
(the decks only), top (the top cards only), seq (each main deck as its draw sequence, every card
with its position; the extra deck as a set). Models (--model): attn (attention over all cards of
both decks), set (each group's cards pooled: mean and max), pool (all of a seat's cards pooled),
phipool (pool, each card first through an MLP with its group and position: added before the
pooling without one, the positions cancel out in the mean).

Reports, held out: the likelihood gain per pair over the training rate, the correlation of the
predicted and the observed rate (openings with >= 8 pairs), and as a filter: the observed
seat-decided share of the held-out openings left after dropping the 10/25/50% predicted most
one-sided.

    .venv/bin/python scripts/book/train_onesided.py OPENINGS.jsonl EMB.npz [--model attn|set|pool] [--inputs deck+top|deck|top|seq]
        [--cards text|id] [--split random|archetype]
"""

import argparse
import json
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # (scripts/: common)

MAIN, EXTRA, TOP = 60, 15, 8
# (per input mode, each seat's parts: (tokens, part, positions: the draw order counts))
INPUTS = {
    "deck+top": ((MAIN, "main", False), (EXTRA, "extra", False), (TOP, "top", True)),
    "deck": ((MAIN, "main", False), (EXTRA, "extra", False)),
    "top": ((TOP, "top", True),),
    "seq": ((MAIN, "order", True), (EXTRA, "extra", False)),
}


def tokens(rows, index, parts):
    """[N, 2 * tokens of parts] card indices (0: none), with each token's group (seat, part) and
    position (in draw order, for the parts that have one)."""
    n_tok = 2 * sum(k for k, _, _ in parts)
    idx = np.zeros((len(rows), n_tok), dtype=np.int64)
    group = np.zeros(n_tok, dtype=np.int64)
    pos = np.zeros(n_tok, dtype=np.int64)
    for i, r in enumerate(rows):
        t = 0
        for s in (0, 1):
            for p, (k, part, positional) in enumerate(parts):
                cards = r["decks"][s][part] if part in ("main", "extra") else r[part][s]
                for j in range(k):
                    if j < len(cards):
                        idx[i, t] = index.get(cards[j], 0)
                    group[t], pos[t] = len(parts) * s + p, j if positional else 0
                    t += 1
    return idx, group, pos


class Attention(torch.nn.Module):
    """(attention written out: ROCm's SDPA gives NaN for padded batches)"""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.out = torch.nn.Linear(d, d)

    def forward(self, x, mask):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)) / (d // self.h) ** 0.5
        a = a.masked_fill(~mask[:, None, None, :], -1e4).softmax(-1)
        return self.out((a @ v).transpose(1, 2).reshape(b, n, d))


class Net(torch.nn.Module):
    def __init__(self, card_emb, n_cards, model, n_parts, d=128, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.model = model
        self.n_parts = n_parts
        if card_emb is not None:
            self.table = torch.nn.Embedding.from_pretrained(card_emb, freeze=True, padding_idx=0)
            self.proj = torch.nn.Sequential(torch.nn.Linear(card_emb.shape[1], d), torch.nn.GELU(), torch.nn.Linear(d, d))
        else:
            self.table = torch.nn.Embedding(n_cards, d, padding_idx=0)
            self.proj = torch.nn.Identity()
        self.group = torch.nn.Embedding(2 * n_parts, d)
        self.pos = torch.nn.Embedding(MAIN, d)
        self.drop = torch.nn.Dropout(dropout)
        if model == "attn":
            self.cls = torch.nn.Parameter(torch.zeros(1, 1, d))
            self.blocks = torch.nn.ModuleList()
            for _ in range(layers):
                self.blocks.append(torch.nn.ModuleList([torch.nn.LayerNorm(d), Attention(d, heads), torch.nn.LayerNorm(d),
                                                        torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.GELU(), torch.nn.Linear(2 * d, d))]))
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 1))
        else:
            if model == "phipool":
                self.phi = torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.GELU(), torch.nn.Linear(2 * d, d))
            n_pooled = 2 * n_parts if model == "set" else 2
            self.head = torch.nn.Sequential(torch.nn.Linear(2 * n_pooled * d, d), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(d, 1))

    def forward(self, idx, group, pos):
        mask = idx > 0
        x = self.proj(self.table(idx)) * (self.table.weight.shape[1] != 0)
        # (an unseen card's learned embedding: none)
        x = x + self.group(group)[None] + self.pos(pos)[None]
        x = self.drop(x) * mask[..., None]
        if self.model == "phipool":
            x = self.phi(x) * mask[..., None]
        if self.model == "attn":
            x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
            m = torch.cat([torch.ones(len(x), 1, dtype=torch.bool, device=x.device), mask], 1)
            for ln1, att, ln2, ff in self.blocks:
                x = x + att(ln1(x), m)
                x = x + ff(ln2(x))
            return self.head(x[:, 0]).squeeze(-1)
        pooled = []
        # (set: per group; pool: per seat, all its cards)
        pools = [group == g for g in range(2 * self.n_parts)] if self.model == "set" else [group // self.n_parts == s for s in (0, 1)]
        for sel in pools:
            sel = sel[None, :] & mask
            xs = x * sel[..., None]
            pooled.append(xs.sum(1) / sel.sum(1, keepdim=True).clamp(min=1))
            pooled.append(xs.masked_fill(~sel[..., None], -1e4).max(1).values.clamp(min=-10))
        return self.head(torch.cat(pooled, -1)).squeeze(-1)


def binomial_nll(logit, n, k):
    return -(k * torch.nn.functional.logsigmoid(logit) + (n - k) * torch.nn.functional.logsigmoid(-logit))


def archetype_split(rows, seed, test_openings):
    from common.carddb import default_db

    db = default_db()
    cache = {}

    def arch(code):
        if code not in cache:
            c = db.get(code)
            cache[code] = [s & 0xFFF for s in (c.setcodes or [])] if c else []
        return cache[code]

    deck_arch = [[{a for c in d["main"] + d["extra"] for a in arch(c)} for d in r["decks"]] for r in rows]
    counts = Counter(a for da in deck_arch for s in da for a in s)
    cand = [a for a in sorted(counts) if counts[a] <= 0.03 * 2 * len(rows)]
    random.Random(seed).shuffle(cand)

    def share(d, held):
        cs = d["main"] + d["extra"]
        return sum(any(a in held for a in arch(c)) for c in cs) / len(cs)

    held = set()
    for a in cand:
        held.add(a)
        test = [i for i, r in enumerate(rows) if max(share(d, held) for d in r["decks"]) >= 0.25]
        if len(test) >= test_openings:
            break
    train = [i for i, da in enumerate(deck_arch) if not (da[0] & held or da[1] & held)]
    return train, test, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("openings")
    ap.add_argument("emb")
    ap.add_argument("--model", default="attn", choices=["attn", "set", "pool", "phipool"])
    ap.add_argument("--inputs", default="deck+top", choices=sorted(INPUTS))
    ap.add_argument("--cards", default="text", choices=["text", "id"])
    ap.add_argument("--split", default="random", choices=["none", "random", "archetype"])
    ap.add_argument("--match-random", action="store_true", help="a random split with the archetype split's sizes")
    ap.add_argument("--held", type=int, default=5)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ensemble", type=int, default=1, help="nets trained with seeds seed, seed+1, ... (and their own early-stopping tenth), their logits averaged")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="the model's state (torch)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [json.loads(line) for line in open(a.openings)]
    if a.split == "archetype" or a.match_random:
        train_i, test_i, held = archetype_split(rows, a.seed, a.test)
        print(f"archetype split: {len(held)} archetypes held out; {len(train_i)} train openings (none of their cards), {len(test_i)} test")
        if a.match_random:
            perm = list(range(len(rows)))
            random.Random(a.seed).shuffle(perm)
            test_i, train_i = perm[: len(test_i)], perm[len(test_i) : len(test_i) + len(train_i)]
            print(f"random split of the same sizes: {len(train_i)} train, {len(test_i)} test")
    elif a.split == "none":
        train_i, test_i = list(range(len(rows))), []
    else:
        test_i = [i for i, r in enumerate(rows) if zlib.crc32(str(r["seed"]).encode()) % a.held == 0]
        train_i = [i for i, r in enumerate(rows) if zlib.crc32(str(r["seed"]).encode()) % a.held != 0]
    train_set = set(train_i)
    seen = {c for i in train_i for d in rows[i]["decks"] for c in d["main"] + d["extra"]}
    unc = [0.0] if not test_i else [sum(c not in seen for c in d["main"] + d["extra"]) / len(d["main"] + d["extra"]) for i in test_i for d in rows[i]["decks"]]
    print(f"test decks' cards unseen in training: mean {np.mean(unc):.3f}, decks with any {np.mean([u > 0 for u in unc]):.3f}")

    z = np.load(a.emb)
    codes = z["codes"]
    if a.cards == "text":
        index = {int(c): i + 1 for i, c in enumerate(codes)}
        emb = torch.cat([torch.zeros(1, z["emb"].shape[1]), torch.tensor(z["emb"])])
    else:
        # (the training openings' cards only: an unseen card is none)
        index = {c: i + 1 for i, c in enumerate(sorted(seen))}
        emb = None
    idx, group, pos = tokens(rows, index, INPUTS[a.inputs])
    idx_t = torch.tensor(idx, device=dev)
    group_t, pos_t = torch.tensor(group, device=dev), torch.tensor(pos, device=dev)
    n = torch.tensor([r["pairs"] for r in rows], dtype=torch.float32, device=dev)
    k = torch.tensor([r["seat_decided"] for r in rows], dtype=torch.float32, device=dev)
    test_t = torch.tensor(test_i, device=dev)
    base = (k[torch.tensor(train_i, device=dev)].sum() / n[torch.tensor(train_i, device=dev)].sum()).item()
    base_logit = float(np.log(base / (1 - base)))

    def nll_gain(sel, logit):
        const = binomial_nll(torch.full_like(logit, base_logit), n[sel], k[sel]).sum()
        return ((const - binomial_nll(logit, n[sel], k[sel]).sum()) / n[sel].sum()).item()

    def train_one(seed):
        torch.manual_seed(seed)
        net = Net(emb, len(index) + 1, a.model, len(INPUTS[a.inputs]), d=a.d, layers=a.layers, heads=a.heads, dropout=a.dropout).to(dev)
        # (a tenth of the training openings for early stopping)
        tr = list(train_i)
        random.Random(seed + 1000).shuffle(tr)
        val, tr = tr[: len(tr) // 10], tr[len(tr) // 10 :]
        tr_t, val_t = torch.tensor(tr, device=dev), torch.tensor(val, device=dev)
        with torch.no_grad():
            net.head[-1].bias.fill_(base_logit)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)

        def predict(sel):
            net.eval()
            with torch.no_grad():
                return torch.cat([net(idx_t[b], group_t, pos_t) for b in sel.split(256)])

        best, best_state, bad = -1e9, None, 0
        for epoch in range(a.epochs):
            net.train()
            perm = tr_t[torch.randperm(len(tr_t), device=dev)]
            for b in perm.split(a.batch):
                loss = binomial_nll(net(idx_t[b], group_t, pos_t), n[b], k[b]).sum() / n[b].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            g = nll_gain(val_t, predict(val_t))
            if g > best:
                best, best_state, bad = g, {kk: v.detach().clone() for kk, v in net.state_dict().items()}, 0
            else:
                bad += 1
            if bad >= 8:
                break
        net.load_state_dict(best_state)
        print(f"seed {seed}: val gain {best * 1000:.2f} mnats/pair at epoch {epoch + 1 - bad}")
        return net, best, predict

    nets, vals, preds = [], [], []
    for e in range(a.ensemble):
        net, best, predict = train_one(a.seed + e)
        nets.append(net)
        vals.append(best)
        if test_i:
            preds.append(predict(test_t))
    print(f"val gain mean {np.mean(vals) * 1000:.2f} mnats/pair")
    if a.out:
        torch.save({"states": [m.state_dict() for m in nets], "args": vars(a), "base_logit": base_logit}, a.out)
    if not test_i:
        return
    logit = torch.stack(preds).mean(0)
    p = torch.sigmoid(logit).cpu().numpy()
    obs = (k[test_t] / n[test_t]).cpu().numpy()
    many = (n[test_t] >= 8).cpu().numpy()
    r = np.corrcoef(p[many], obs[many])[0, 1]
    nn_, kk_ = n[test_t].cpu().numpy(), k[test_t].cpu().numpy()
    order = np.argsort(-p)
    drops = []
    for frac in (0.1, 0.25, 0.5):
        keep = order[int(frac * len(order)) :]
        drops.append(f"drop {frac:.0%}: {kk_[keep].sum() / nn_[keep].sum():.3f}")
    print(f"test ({len(test_i)} openings, {int(nn_.sum())} pairs; train rate {base:.3f}, test {kk_.sum() / nn_.sum():.3f}): "
          f"gain {nll_gain(test_t, logit) * 1000:.2f} mnats/pair, r {r:.3f} ({many.sum()} with >= 8 pairs); " + ", ".join(drops))


if __name__ == "__main__":
    main()
