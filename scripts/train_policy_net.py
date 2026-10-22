"""Usage: scripts/train_policy_net.py NODES.jsonl [-o runs/nn/policy_net.json] [--epochs 6] [--hidden 64,32]

A move-ordering network for the search's inner nodes, trained on ygoai-datagen --node-log rows
(their "position": the node's mover2 features, "moves": the linear policy's move features,
"history": the moves' history scores). The first layer takes position and move together, split so
that the search computes the position's part once per node:

    h1(move) = relu(Wp z + sum_f v_f E[f] + Wh hist(move) + b1),   z = (position - mu) / sd

(E: an embedding per move feature, weighted by its value; hist: sign(h) ln(1 + |h|), its difference
to the node's greatest and whether it is the greatest), then hidden layers relu(W h + b) and a
score w . h + c. Loss: the cross-entropy over each node's moves (the best: the cutoff's move or the
exact value's). Without the hash table's move (searched first anyway) and the nodes where it was
the best. Trained on the nodes with an exact value only (--cutoffs: the cutoffs too): a cutoff is
logged only when its first move did not cut (one searched move is no data), so the cutoffs are the
order's failures alone, and the moves after the cutoff's were not searched (read for the held-out
games only then). Held out: every 10th game by a hash. Reports top-1 against the search's own order
(history, "first") and the rules' order, at the nodes with an exact value (all moves searched: the
fair comparison) and at the cutoffs. The parsed data are cached next to NODES as .npz.
"""
import argparse, collections, json, os, zlib
import multiprocessing as mp
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("rows")
ap.add_argument("-o", "--output")
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--hidden", default="64,32")
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--cutoffs", action="store_true")
ap.add_argument("--no-position", action="store_true", help="(ablation) without the position features")
ap.add_argument("--no-history", action="store_true", help="(ablation) without the history inputs")
args = ap.parse_args()
hidden = [int(x) for x in args.hidden.split(",")]

def hist_inputs(h):
    hs = np.sign(h) * np.log1p(np.abs(h))
    return np.stack([hs, hs - hs.max(), (hs == hs.max()).astype(np.float64)], 1)

def held_game(g):
    return zlib.crc32(str(g).encode()) % 10 == 0

def parse(lines):
    """A chunk's nodes as arrays, without the hash table's move (the cutoffs of training games
    only with --cutoffs): games, prompts, positions, moves per node, the move features' names,
    values and count per move, hist inputs, chosen, rules' first, cut."""
    games, prompts_, pos, nmoves, keys, vals, nfeat, hist, chosen, rules, cut = ([] for _ in range(11))
    for l in lines:
        r = json.loads(l)
        p = r["policy"]
        if p["cut"] and not args.cutoffs and not held_game(r["game"]):
            continue
        moves, ch, tt = p["moves"], p["chosen"], p.get("tt_move")
        keep = [j for j in range(len(moves)) if j != tt]
        if ch not in keep or len(keep) < 2:
            continue
        new = {j: i for i, j in enumerate(keep)}
        hist.append(hist_inputs(np.array([p["history"][j] for j in keep], dtype=np.float64)))
        games.append(r["game"]); prompts_.append(p["prompt"]); pos.append(p["position"]); nmoves.append(len(keep))
        for j in keep:
            m = moves[j]
            keys.extend(m.keys()); vals.extend(m.values()); nfeat.append(len(m))
        chosen.append(new[ch]); rules.append([new[j] for j in p["rules"] if j in new][0]); cut.append(p["cut"])
    if not games:
        return None
    return (games, prompts_, np.array(pos, dtype=np.float32), np.array(nmoves, dtype=np.int64), keys,
            np.array(vals, dtype=np.float32), np.array(nfeat, dtype=np.int64), np.concatenate(hist).astype(np.float32),
            np.array(chosen), np.array(rules), np.array(cut))

def chunks(path, size=5000):
    buf = []
    with open(path) as f:
        for l in f:
            buf.append(l)
            if len(buf) == size: yield buf; buf = []
    if buf: yield buf

cache = os.path.splitext(args.rows)[0] + (".cutoffs" if args.cutoffs else "") + ".policynet2.npz"
if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(args.rows):
    c = np.load(cache, allow_pickle=True)
    pos, mfirst, fidx, fval, foff, hist, dgame, dprompt, dchosen, drules, dcut = (c[k] for k in
        ["pos", "mfirst", "fidx", "fval", "foff", "hist", "dgame", "dprompt", "dchosen", "drules", "dcut"])
    vocab = {k: i for i, k in enumerate(c["vocab"])}; prompts = {k: i for i, k in enumerate(c["prompts"])}
else:
    vocab, prompts = {}, {}
    parts = collections.defaultdict(list)
    with mp.get_context("fork").Pool(24) as pool:
        for part in pool.imap(parse, chunks(args.rows), chunksize=1):
            if part is None: continue
            games, ps, position, nmoves, keys, vals, nfeat, h, chosen, rules, cut = part
            parts["dgame"].append(np.array(games)); parts["dprompt"].append(np.array([prompts.setdefault(p, len(prompts)) for p in ps]))
            parts["pos"].append(position); parts["nmoves"].append(nmoves); parts["nfeat"].append(nfeat)
            parts["fidx"].append(np.fromiter((vocab.setdefault(k, len(vocab)) for k in keys), np.int64, len(keys)))
            parts["fval"].append(vals); parts["hist"].append(h)
            parts["dchosen"].append(chosen); parts["drules"].append(rules); parts["dcut"].append(cut)
    cat = {k: np.concatenate(v) for k, v in parts.items()}; del parts
    pos, fidx, fval, hist = cat["pos"], cat["fidx"], cat["fval"], cat["hist"]
    dgame, dprompt, dchosen, drules, dcut = cat["dgame"], cat["dprompt"], cat["dchosen"], cat["drules"], cat["dcut"]
    mfirst = np.concatenate([[0], np.cumsum(cat["nmoves"])]).astype(np.int64)
    foff = np.concatenate([[0], np.cumsum(cat["nfeat"])]).astype(np.int64)
    del cat
    np.savez(cache, pos=pos, mfirst=mfirst, fidx=fidx, fval=fval, foff=foff, hist=hist, dgame=dgame, dprompt=dprompt,
             dchosen=dchosen, drules=drules, dcut=dcut, vocab=np.array(list(vocab)), prompts=np.array(list(prompts)))
nd, nm, nf = len(dgame), int(mfirst[-1]), len(vocab)
print(f"{nd} nodes ({int(dcut.sum())} cutoffs), {nm} moves, {nf} move features, prompts {dict(collections.Counter(dprompt.tolist()))} = {prompts}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
held = np.array([held_game(g) for g in dgame])
mu = pos[~held].mean(0); sd = pos[~held].std(0) + 1e-3
T = lambda x, **k: torch.tensor(x, device=dev, **k)
z = T((pos - mu) / sd); H = T(hist); FI = T(fidx); FV = T(fval); FO = T(foff); MF = T(mfirst); CH = T(dchosen)
if args.no_position: z = torch.zeros_like(z)
if args.no_history: H = torch.zeros_like(H)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wp = torch.nn.Linear(z.shape[1], hidden[0], bias=False)
        self.emb = torch.nn.Parameter(torch.randn(nf, hidden[0], device=dev) * 0.1)
        self.wh = torch.nn.Linear(H.shape[1], hidden[0])
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(hidden, hidden[1:]))
        self.out = torch.nn.Linear(hidden[-1], 1)

    def forward(self, ds):
        """The scores of the moves of nodes ds, and each move's node (0..len(ds))."""
        starts = MF[ds]; lens = MF[ds + 1] - starts
        local = torch.repeat_interleave(torch.arange(len(ds), device=dev), lens)
        ms = torch.repeat_interleave(starts - torch.cumsum(lens, 0) + lens, lens) + torch.arange(int(lens.sum()), device=dev)
        fs, fl = FO[ms], FO[ms + 1] - FO[ms]
        mloc = torch.repeat_interleave(torch.arange(len(ms), device=dev), fl)
        fp = torch.repeat_interleave(fs - torch.cumsum(fl, 0) + fl, fl) + torch.arange(int(fl.sum()), device=dev)
        moves = torch.zeros(len(ms), hidden[0], device=dev).index_add_(0, mloc, self.emb[FI[fp]] * FV[fp, None])
        h = torch.relu(self.wp(z[ds])[local] + moves + self.wh(H[ms]))
        for l in self.layers:
            h = torch.relu(l(h))
        return self.out(h)[:, 0], local, starts

def node_ce(s, local, n, target):
    mx = torch.full((n,), -1e30, device=dev).scatter_reduce(0, local, s.detach(), "amax")
    lse = torch.zeros(n, device=dev).index_add_(0, local, torch.exp(s - mx[local])).log() + mx
    return lse - s[target]

net = Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
tr = T(np.nonzero(~held & (~dcut | args.cutoffs))[0]); te = np.nonzero(held)[0]

def report(tag):
    net.eval()
    best, ce = np.zeros(nd, dtype=np.int64), []
    with torch.no_grad():
        for i in range(0, len(te), 65536):
            ds = T(te[i:i + 65536])
            s, local, _ = net(ds)
            # (the chosen move's place in s: its node's offset in the batch plus chosen)
            off = torch.cumsum(MF[ds + 1] - MF[ds], 0) - (MF[ds + 1] - MF[ds])
            ce.append(node_ce(s, local, len(ds), off + CH[ds]).cpu().numpy())
            am = torch.full((len(ds),), -1e30, device=dev).scatter_reduce(0, local, s, "amax")
            idx = torch.arange(len(s), device=dev)
            first_hit = torch.full((len(ds),), len(s), device=dev).scatter_reduce(0, local, torch.where(s >= am[local], idx, len(s)), "amin")
            best[te[i:i + 65536]] = (first_hit - off).cpu().numpy()
    net.train()
    ce = np.concatenate(ce)
    names = {v: k for k, v in prompts.items()}
    lines = []
    for label, sel in (("exact", ~dcut[te]), ("cutoff", dcut[te])):
        d = te[sel]
        if not len(d): continue
        pol, rul, fst = (best[d] == dchosen[d]).mean(), (drules[d] == dchosen[d]).mean(), (dchosen[d] == 0).mean()
        per = []
        for p in sorted(set(dprompt[d].tolist()), key=lambda p: -(dprompt[d] == p).sum()):
            q = d[dprompt[d] == p]
            per.append(f"{names[p]} {(best[q] == dchosen[q]).mean():.3f}/{(drules[q] == dchosen[q]).mean():.3f}/{(dchosen[q] == 0).mean():.3f} ({len(q)})")
        lines.append(f"{label} {pol:.3f}/{rul:.3f}/{fst:.3f} over {len(d)} [" + ", ".join(per) + "]")
    print(f"{tag}: held-out CE {ce.mean():.4f}; top-1 policy/rules/first: " + "; ".join(lines), flush=True)

report("start")
for ep in range(args.epochs):
    perm = tr[torch.randperm(len(tr), device=dev)]
    for i in range(0, len(perm), args.batch):
        ds = perm[i:i + args.batch]
        s, local, starts = net(ds)
        off = torch.cumsum(MF[ds + 1] - MF[ds], 0) - (MF[ds + 1] - MF[ds])
        loss = node_ce(s, local, len(ds), off + CH[ds]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    report(f"epoch {ep + 1}")

if args.output:
    sd_ = {k: v.detach().cpu().numpy().tolist() for k, v in net.state_dict().items()}
    json.dump({
        "hidden": hidden, "mu": mu.tolist(), "sd": sd.tolist(),
        "wp": sd_["wp.weight"], "wh": sd_["wh.weight"], "b1": sd_["wh.bias"],
        "emb": {k: sd_["emb"][i] for k, i in vocab.items()},
        "layers": [{"w": sd_[f"layers.{j}.weight"], "b": sd_[f"layers.{j}.bias"]} for j in range(len(hidden) - 1)],
        "out": {"w": sd_["out.weight"][0], "b": sd_["out.bias"][0]},
    }, open(args.output, "w"))
    print("saved", args.output)
