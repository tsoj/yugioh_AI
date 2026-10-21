"""Usage: scripts/train_policy_linear.py ROWS.jsonl [-o runs/nn/policy_linear.json] [--epochs 4]

A linear move policy (search::policy): a weight per move feature, a move's score the sum of its
features' weights times their values, trained per move towards 1 for the move the search chose and
0 for the others (logistic regression). ROWS: ygoai-refeature ... --features policy. Reports, on
held-out games (every 10th by a hash of the game), how often the best-scored move is the search's
choice, against the rules' order (the first of `rules`), per prompt.
"""
import argparse, collections, json, zlib
import multiprocessing as mp
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("rows")
ap.add_argument("-o", "--output")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=65536)
ap.add_argument("--max-rows", type=int, default=0)
args = ap.parse_args()

def parse(lines):
    out = []
    for l in lines:
        r = json.loads(l)
        p = r.get("policy")
        if p:
            out.append((r["game"], p["prompt"], p["moves"], p["chosen"], p["rules"][0] if p["rules"] else -1))
    return out

def chunks(path, size=20000):
    buf, n = [], 0
    with open(path) as f:
        for l in f:
            buf.append(l); n += 1
            if len(buf) == size: yield buf; buf = []
            if args.max_rows and n >= args.max_rows: break
    if buf: yield buf

vocab = {}
feat_idx, feat_val, move_off, move_dec, target = [], [], [0], [], []
dec_game, dec_prompt, dec_chosen, dec_rules, dec_first = [], [], [], [], []
prompts = {}
with mp.get_context("fork").Pool(20) as pool:
    for part in pool.imap(parse, chunks(args.rows), chunksize=1):
        for game, prompt, moves, chosen, rules in part:
            d = len(dec_game)
            dec_game.append(game); dec_prompt.append(prompts.setdefault(prompt, len(prompts)))
            dec_chosen.append(chosen); dec_rules.append(rules); dec_first.append(len(move_dec))
            for j, m in enumerate(moves):
                for k, v in m.items():
                    feat_idx.append(vocab.setdefault(k, len(vocab))); feat_val.append(v)
                move_off.append(len(feat_idx)); move_dec.append(d); target.append(float(j == chosen))
dec_first.append(len(move_dec))
nd, nm, nf = len(dec_game), len(move_dec), len(vocab)
print(f"{nd} decisions, {nm} moves, {nf} features; prompts {dict(collections.Counter(dec_prompt))} = {prompts}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
fi = torch.tensor(feat_idx, dtype=torch.int64, device=dev); fv = torch.tensor(feat_val, dtype=torch.float32, device=dev)
off = torch.tensor(move_off, dtype=torch.int64, device=dev); y = torch.tensor(target, dtype=torch.float32, device=dev)
mdec = np.array(move_dec); held_dec = np.array([zlib.crc32(str(g).encode()) % 10 == 0 for g in dec_game])
held_move = held_dec[mdec]
tr = torch.tensor(np.nonzero(~held_move)[0], device=dev); te_np = np.nonzero(held_move)[0]

w = torch.zeros(nf, device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
opt = torch.optim.Adam([w, b], lr=args.lr)

def scores(moves):
    """The scores of moves (indices), by their feature ranges."""
    starts, ends = off[moves], off[moves + 1]
    lens = ends - starts
    rep = torch.repeat_interleave(torch.arange(len(moves), device=dev), lens)
    pos = torch.repeat_interleave(starts - torch.cumsum(lens, 0) + lens, lens) + torch.arange(int(lens.sum()), device=dev)
    return torch.zeros(len(moves), device=dev).index_add_(0, rep, w[fi[pos]] * fv[pos])

def report(tag):
    with torch.no_grad():
        s = torch.cat([scores(torch.arange(i, min(i + 1 << 20, nm), device=dev)) for i in range(0, nm, 1 << 20)]).cpu().numpy()
        te = torch.tensor(te_np, device=dev)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(s[te_np], device=dev) + b, y[te]).item()
    by = collections.defaultdict(lambda: [0, 0, 0])
    first = np.array(dec_first)
    for d in np.nonzero(held_dec)[0]:
        ss = s[first[d]:first[d + 1]]
        e = by[dec_prompt[d]]
        e[0] += 1; e[1] += int(np.argmax(ss) == dec_chosen[d]); e[2] += int(dec_rules[d] == dec_chosen[d])
    names = {v: k for k, v in prompts.items()}
    tot = np.sum([v for v in by.values()], axis=0)
    print(f"{tag}: held-out move BCE {bce:.4f}; top-1 policy {tot[1] / tot[0]:.3f} rules {tot[2] / tot[0]:.3f} over {tot[0]} decisions; "
          + ", ".join(f"{names[p]} {e[1] / e[0]:.3f}/{e[2] / e[0]:.3f} ({e[0]})" for p, e in sorted(by.items(), key=lambda kv: -kv[1][0])), flush=True)

report("start")
for ep in range(args.epochs):
    perm = tr[torch.randperm(len(tr), device=dev)]
    for i in range(0, len(perm), args.batch):
        m = perm[i:i + args.batch]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores(m) + b, y[m])
        opt.zero_grad(); loss.backward(); opt.step()
    report(f"epoch {ep + 1}")

top = sorted(((w[i].item(), k) for k, i in vocab.items()), reverse=True)
print("largest weights:", ", ".join(f"{k} {v:+.2f}" for v, k in top[:12]))
print("smallest weights:", ", ".join(f"{k} {v:+.2f}" for v, k in top[-12:]))
if args.output:
    json.dump({"weights": {k: w[i].item() for k, i in vocab.items()}, "bias": b.item()}, open(args.output, "w"), indent=0, sort_keys=True)
    print("saved", args.output)
