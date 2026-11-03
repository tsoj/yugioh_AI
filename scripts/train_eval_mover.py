"""Usage: scripts/train_eval_mover.py FEATURES.npz [--hidden 16] [--lam 0.5] [-o runs/nn/X.json]
(FEATURES.npz from scripts/eval_mover_features.py; games in blocks of 20 per deck pair, alternating
seat order, as books/pairs500x20.jsonl.zst, or explicit deck pairs: `pair`, `aseat`, `valgame`).

Mover-view eval with a deck-pair term (experiment).

logit P(mover wins) = sgn * a[pair, seat order] + v(x); the search value's target is sigmoid(v(x))
alone (the search knows nothing of deck strength). a is dropped at inference.

With --cards (scripts/eval_cards/cards_to_npz.py, the rows' card lists) and --effects
(scripts/eval_cards/embed_effects.py) the card term: per bucket b (hand, GY, field) a card's
strength s_b = w_b . e + c_b (e: its effect text's embedding, the mean over the pseudo-name seeds;
0 for a card without an effect), times a learned multiplier m_b by the card's state (negated, used
up; 1 for neither), summed over the cards, the mover's minus the opponent's: 3 more inputs of the
network. With --card-out instead their sum is added to the network's output (the logit).
--card-pca K: e reduced to its first K principal components (of every card with an effect,
whitened); --card-hidden H1,H2,..: hidden layers (ReLU) before the strengths (s = W relu(.. relu(W1 e + b1)) + c);
--card-no-gy: only the cards in the hand and on the field (the GY's strengths 0); --card-decorr W:
a penalty on the card term's correlation with the pair term (the decks' strength). A
card's strengths depend on its code only: the export is a table in both cases.
"""
import argparse
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--model", default="mlp")
ap.add_argument("--hidden", type=int, default=32)
ap.add_argument("--layers", type=int, default=1, help="hidden layers")
ap.add_argument("--dims", help="the hidden layers' widths, comma list (e.g. 8,32), instead of --hidden and --layers")
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--sigma", type=float, default=1.0, help="prior sd of the pair term (logits)")
ap.add_argument("--nopair", action="store_true")
ap.add_argument("--held", type=int, default=50, help="deck pairs held out for validation")
ap.add_argument("--held-below", type=int, default=0, help="choose them (or the folds) among the pairs below this one (0: all)")
ap.add_argument("--drop", default="", help="feature indices to zero, comma list")
ap.add_argument("-o", "--output")
ap.add_argument("--save-v", help="npz of the final v on all rows (for paired comparisons)")
ap.add_argument("--seed", type=int, default=0, help="the training's seed (not the held-out pairs')")
ap.add_argument("--indicator", action="store_true", help="the last column marks rows of older data whose newer features are filled in (by their means): 0 in play, folded into the export's first bias")
ap.add_argument("--drop-onesided", action="store_true", help="leave out of training the deck pairs where one deck won every game (held-out pairs stay); --min-wins 1")
ap.add_argument("--min-wins", type=int, default=0, help="leave out of training the deck pairs where a deck won fewer games than this (held-out pairs stay)")
ap.add_argument("--balance-wins", action="store_true", help="weight each training game so that both decks of a pair have as much weight in won games: a game the pair's winner of W of its games won counts min(W, L) / W (a one-sided pair: 0)")
ap.add_argument("--cards", help="the rows' card lists (cards_to_npz.py): the card term")
ap.add_argument("--effects", help="the effect embeddings (embed_effects.py), with --cards")
ap.add_argument("--card-free", action="store_true", help="the card term's strengths a free learned value per code and bucket instead of w_b . e (no embeddings; the export: the codes of the data)")
ap.add_argument("--card-out", action="store_true", help="the card term added to the output instead of 3 more inputs")
ap.add_argument("--card-wd", type=float, default=0.0, help="weight decay of the card term's w_b (or first layer)")
ap.add_argument("--card-pca", type=int, default=0, help="the effect embeddings reduced to their first K principal components (whitened; 0: all dimensions)")
ap.add_argument("--card-no-gy", action="store_true", help="no card term for the cards in the GY (their strengths 0)")
ap.add_argument("--card-decorr", type=float, default=0.0, help="weight of a penalty on the squared correlation of the card term's sum with the pair term (the mover's side, its gradient stopped): the card term is not to tell the decks' strength (--card-out)")
ap.add_argument("--card-holdout", type=float, default=0.0, help="(diagnostic) this share of the codes with an effect left out of the card term in training; the reports also with them in: does the card term carry over to unseen cards")
ap.add_argument("--card-hidden", default="", help="hidden layers (ReLU) before the card term's strengths, their widths as a comma list (e.g. 32,32; empty: none)")
ap.add_argument("--threads", type=int, default=0, help="torch's CPU threads (0: its default)")
ap.add_argument("--fold",help="I/N: hold out the I-th of N folds of the deck pairs instead of --held of them")
args = ap.parse_args()
hidden = [int(w) for w in args.dims.split(",")] if args.dims else [args.hidden] * args.layers
args.layers = len(hidden)
torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if args.threads:
    torch.set_num_threads(args.threads)

d = np.load(args.data)
X, value, static, result, game, player, turn = (d[k] for k in ["x", "value", "static", "result", "game", "player", "turn"])
if args.drop:
    X = X.copy(); X[:, [int(i) for i in args.drop.split(",")]] = 0
# (explicit deck pairs if the npz has them, e.g. match records: `pair`, `aseat` the seat of the
# pair's first deck, `valgame` the validation games within training pairs; else blocks of 20 games
# per pair with alternating seat order, the last 2 of each for validation)
if "pair" in d.files:
    pair, aseat, valgame = d["pair"], d["aseat"], d["valgame"].astype(bool)
else:
    pair, k = game // 20, game % 20
    aseat, valgame = k % 2, k >= 18
ps = pair * 2 + aseat
sgn = np.where(player == 0, 1.0, -1.0).astype(np.float32)

def fit_k(v, r):
    # (no wins and losses: +/-100000 before the values were in LP, +/-1e7 since)
    keep = (r != 0) & (np.abs(v) < 90000)
    # (on the device: millions of rows times 400 scales)
    v = torch.tensor(v[keep], dtype=torch.float64, device=dev)
    y = torch.tensor((r[keep] + 1) / 2, dtype=torch.float64, device=dev)
    best = None
    for kk in np.geomspace(1, 1e5, 400):
        p = torch.sigmoid((v / kk).clamp(-50, 50))
        loss = -(y * torch.log(p + 1e-9) + (1 - y) * torch.log(1 - p + 1e-9)).mean().item()
        if best is None or loss < best[0]:
            best = (loss, kk)
    return best[1]
# (a merged npz has `src` per row: each data set's search values in its own scale; the export
# uses the first set's, so put the newest data first)
src = d["src"] if "src" in d.files else np.zeros(len(X), np.int64)
Ks = {s_: fit_k(value[src == s_], result[src == s_]) for s_ in np.unique(src)}
K, Kr = Ks[src[0]], np.array([Ks.get(i, 1.0) for i in range(src.max() + 1)])[src]

rng = np.random.default_rng(0)
if args.fold:
    fi, fn = map(int, args.fold.split("/"))
    held_pairs = set(np.array_split(rng.permutation(np.unique(pair[pair < args.held_below] if args.held_below else pair)), fn)[fi].tolist())
else:
    held_pairs = set(rng.choice(np.unique(pair[pair < args.held_below] if args.held_below else pair), args.held, replace=False).tolist())
is_hp = np.isin(pair, list(held_pairs))
is_vin = ~is_hp & valgame
is_tr = ~is_hp & ~is_vin
W = np.ones(len(X), np.float32)
min_wins = max(args.min_wins, 1 if args.drop_onesided else 0)
if min_wins or args.balance_wins:
    # (the pair's first deck sits in seat aseat; per game: 0 or 1 the deck that won, 2 a draw)
    won = np.where(result == 0, 2, (np.where(result > 0, player, 1 - player) != aseat).astype(np.int64))
    games_of = {}
    for g_, p_, w_ in zip(game, pair, won):
        games_of.setdefault(p_, {})[g_] = w_
    wins = {q: (sum(w == 0 for w in gs.values()), sum(w == 1 for w in gs.values())) for q, gs in games_of.items()}
    if min_wins:
        dropped = np.array([q for q, (w0, w1) in wins.items() if min(w0, w1) < min_wins])
        is_os = np.isin(pair, dropped)
        print(f"--min-wins {min_wins}: {len(dropped)} of {len(wins)} pairs, {int((is_tr & is_os).sum())} training rows dropped")
        is_vin &= ~is_os
        is_tr &= ~is_os
    if args.balance_wins:
        w0 = np.array([wins[q][0] for q in pair], np.float32); w1 = np.array([wins[q][1] for q in pair], np.float32)
        lo = np.minimum(w0, w1)
        W = np.where(won == 2, 1.0, lo / np.maximum(np.where(won == 0, w0, w1), 1)).astype(np.float32)
        print(f"--balance-wins: training rows' weight {W[is_tr].sum():.0f} of {is_tr.sum()}")
# (normalized on the device in chunks: no float copies of X in memory)
Z = torch.tensor(X, device=dev); TR = torch.tensor(is_tr, device=dev)
s1 = torch.zeros(X.shape[1], dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1)
for i in range(0, len(X), 1 << 20):
    z = Z[i:i + (1 << 20)][TR[i:i + (1 << 20)]].double()
    s1 += z.sum(0); s2 += (z * z).sum(0)
n_tr = int(is_tr.sum()); m_ = s1 / n_tr
mu, sd = m_.cpu().numpy().astype(np.float32), ((s2 / n_tr - m_ * m_).clamp_min(0).sqrt().cpu().numpy() + 1e-6).astype(np.float32)
Z.sub_(torch.tensor(mu, device=dev)).div_(torch.tensor(sd, device=dev))
del TR
Yr = torch.tensor((result + 1) / 2, dtype=torch.float32, device=dev)
Yv = torch.tensor(1 / (1 + np.exp(-np.clip(value / Kr, -50, 50))), dtype=torch.float32, device=dev)
PS = torch.tensor(ps, device=dev); SG = torch.tensor(sgn, device=dev)
dec = torch.tensor(result != 0, device=dev)
Wt = torch.tensor(W, device=dev)

n = X.shape[1]
N_BUCKETS, N_STATES = 3, 4
if args.cards:
    cd = np.load(args.cards); ef = np.load(args.effects)
    assert len(cd["off"]) == len(X) + 1, "the card lists are not the npz's rows"
    # (each data code's effect embedding: the mean over the seeds; none: 0)
    at = {c: i for i, c in enumerate(ef["codes"].tolist())}
    emb = ef["emb"].astype(np.float32).mean(0) * ef["has_effect"][:, None]
    if args.card_pca:
        # (the principal components of the cards with an effect; the cards without stay 0)
        E = emb[ef["has_effect"] > 0]; mean = E.mean(0)
        ev, vec = np.linalg.eigh(np.cov(E - mean, rowvar=False))
        top = np.argsort(ev)[::-1][:args.card_pca]
        print(f"card term: PCA {args.card_pca} of {E.shape[1]}: {ev[top].sum() / ev.sum():.3f} of the variance", flush=True)
        emb = (((emb - mean) @ vec[:, top]) / np.sqrt(ev[top])).astype(np.float32) * ef["has_effect"][:, None]
    rows_of = np.array([at.get(c, -1) for c in cd["codes"].tolist()])
    print(f"card term: {len(rows_of)} codes, {(rows_of < 0).sum()} without an embedding, "
          f"{(ef['has_effect'][rows_of[rows_of >= 0]] == 0).sum()} without an effect")
    EC = np.zeros((len(rows_of), emb.shape[1]), np.float32); EC[rows_of >= 0] = emb[rows_of[rows_of >= 0]]
    HE = np.zeros(len(rows_of), np.float32); HE[rows_of >= 0] = ef["has_effect"][rows_of[rows_of >= 0]]
    EC_t = torch.tensor(EC, device=dev); HE_t = torch.tensor(HE, device=dev)
    EF_codes, EF_emb, EF_has = ef["codes"], torch.tensor(emb), torch.tensor(ef["has_effect"].astype(np.float32))  # (every card: the export's table)
    OFF = torch.tensor(cd["off"], device=dev)
    DATA_CODES = cd["codes"]
    # (compact on the device: 500M entries of the SPRT data; widened per batch)
    CI = torch.tensor(cd["cidx"].astype(np.int32), device=dev)
    BK = torch.tensor(cd["bucket"].astype(np.int8), device=dev)
    ST = torch.tensor((cd["negated"] * 2 + cd["used"]).astype(np.int8), device=dev)
    SGN = torch.tensor(np.where(cd["mine"] > 0, 1, -1).astype(np.int8), device=dev)
    del cd, ef, emb
    emb_dim = EC.shape[1]
    card_hid = [int(h) for h in args.card_hidden.split(",") if h]
    if card_hid:
        card_layers = [torch.nn.Linear(a_, b_).to(dev) for a_, b_ in zip([emb_dim] + card_hid, card_hid)]
        card_w = torch.zeros(card_hid[-1], N_BUCKETS, device=dev, requires_grad=True)
        card_first = [p_ for l_ in card_layers for p_ in l_.parameters()]
    else:
        card_w = torch.zeros(emb_dim, N_BUCKETS, device=dev, requires_grad=True)
        card_first = [card_w]
    card_c = torch.zeros(N_BUCKETS, device=dev, requires_grad=True)
    card_m = torch.zeros(N_BUCKETS, N_STATES - 1, device=dev, requires_grad=True)  # (the multipliers of the states but "neither", minus 1)
    card_T = torch.zeros(len(rows_of), N_BUCKETS, device=dev, requires_grad=True)
    # (--card-holdout: the codes left out of the card term in training)
    HOLD = torch.zeros(len(rows_of), dtype=torch.bool, device=dev)
    if args.card_holdout:
        cand = np.flatnonzero(HE > 0)
        HOLD[torch.tensor(np.random.default_rng(1).choice(cand, int(len(cand) * args.card_holdout), replace=False), device=dev)] = True
        print(f"--card-holdout: {int(HOLD.sum())} of {len(cand)} codes with an effect left out; {HOLD[CI.long()].float().mean().item():.3f} of the card entries", flush=True)
    use_hold = [bool(args.card_holdout)]
    card_params = [card_w, card_c, card_m] + ([card_T] if args.card_free else []) + (card_first if card_hid else [])
    n_x = n if args.card_out else n + N_BUCKETS
else:
    card_params, n_x = [], n

def card_strength(E, has):
    """The strengths per bucket of the cards with embeddings E (on card_w's device or not)."""
    dv = E.device
    on = torch.tensor([1.0, 0.0 if args.card_no_gy else 1.0, 1.0], device=dv)  # (hand, GY, field)
    if args.card_free:
        return (card_T + card_c) * has[:, None] * on
    h = E
    for l_ in (card_layers if card_hid else []):
        h = torch.relu(torch.nn.functional.linear(h, l_.weight.to(dv), l_.bias.to(dv)))
    return (h @ card_w.to(dv) + card_c.to(dv)) * has[:, None] * on

def card_x(idx):
    """The card term's inputs of rows idx: (len(idx), N_BUCKETS)."""
    s = card_strength(EC_t, HE_t)  # (each code's strength per bucket)
    m = torch.cat([torch.ones(N_BUCKETS, 1, device=dev), 1 + card_m], 1)
    lo, hi = OFF[idx], OFF[idx + 1]
    cnt = hi - lo
    row = torch.repeat_interleave(torch.arange(len(idx), device=dev), cnt)
    ent = torch.repeat_interleave(lo - torch.cumsum(cnt, 0) + cnt, cnt) + torch.arange(int(cnt.sum()), device=dev)
    b = BK[ent].long()
    val = SGN[ent].float() * m[b, ST[ent].long()] * s[CI[ent].long(), b]
    if use_hold[0]:
        val = val * (~HOLD[CI[ent].long()]).float()
    out = torch.zeros(len(idx) * N_BUCKETS, device=dev)
    out.index_add_(0, row * N_BUCKETS + b, val)
    return out.view(len(idx), N_BUCKETS)

n_in = n
n = n_x
if args.model == "linear":
    net = torch.nn.Linear(n, 1)
else:
    dims = [n] + hidden
    net = torch.nn.Sequential(*[m for a_, b_ in zip(dims, dims[1:]) for m in (torch.nn.Linear(a_, b_), torch.nn.ReLU())], torch.nn.Linear(dims[-1], 1))
net = net.to(dev)
a = torch.zeros(int(ps.max()) + 1, device=dev, requires_grad=True)
groups = [{"params": net.parameters(), "weight_decay": args.wd}, {"params": [a]}]
if card_params:
    first = [card_T] if args.card_free else ([card_layers[0].weight] if card_hid else [card_w])
    rest = [card_c, card_m] + ([p_ for p_ in card_first if all(p_ is not f_ for f_ in first)] + [card_w] if card_hid else [])
    groups += [{"params": first, "weight_decay": args.card_wd}, {"params": rest}]
opt = torch.optim.Adam(groups, lr=args.lr)
tr_idx = torch.tensor(np.nonzero(is_tr)[0], device=dev)
N = len(tr_idx)
bce = torch.nn.functional.binary_cross_entropy_with_logits

def v_of(idx):
    if card_params and args.card_out:
        return net(Z[idx]).squeeze(-1) + card_x(idx).sum(1)
    if card_params:
        return net(torch.cat([Z[idx], card_x(idx)], 1)).squeeze(-1)
    return net(Z[idx]).squeeze(-1)

def loss_of(idx, use_a=True, weighted=False, v=None):
    v = v_of(idx) if v is None else v
    lr_ = SG[idx] * a[PS[idx]] + v if (use_a and not args.nopair) else v
    m = dec[idx]
    if weighted:
        w = Wt[idx]
        l_res = (bce(lr_[m], Yr[idx][m], reduction="none") * w[m]).sum() / w[m].sum().clamp_min(1e-9) if m.any() else torch.zeros((), device=dev)
        l_val = (bce(v, Yv[idx], reduction="none") * w).sum() / w.sum().clamp_min(1e-9)
        return l_res, l_val
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
    # (v of all rows in chunks first: the network on all training rows at once does not fit on the GPU)
    v = all_v()
    with torch.no_grad():
        V = torch.tensor(v, device=dev)
        out = []
        for name, m, ua in [("train", is_tr, True), ("val-games", is_vin, True), ("held-pairs", is_hp, False)]:
            idx = torch.tensor(np.nonzero(m)[0], device=dev)
            lr_, lv = loss_of(idx, ua, v=V[idx])
            out.append(f"{name} res-ll {lr_.item():.4f} val-bce {lv.item():.4f}")
        del V
    print(f"{tag}: " + " | ".join(out) + f" | rank held-pairs {ranking(v, is_hp):.4f} | held-pairs fit res-ll {fitted_res_ll(v, is_hp):.4f}", flush=True)
    if args.cards and args.card_holdout:
        # (the same with the held-out codes in, their strengths from the trained card term)
        use_hold[0] = False
        v2 = all_v()
        use_hold[0] = True
        with torch.no_grad():
            idx = torch.tensor(np.nonzero(is_hp)[0], device=dev)
            lr2, lv2 = loss_of(idx, False, v=torch.tensor(v2[np.nonzero(is_hp)[0]], device=dev))
        print(f"{tag} with the held-out codes: held-pairs res-ll {lr2.item():.4f} val-bce {lv2.item():.4f} | rank held-pairs {ranking(v2, is_hp):.4f} | held-pairs fit res-ll {fitted_res_ll(v2, is_hp):.4f}", flush=True)
    return v

def fitted_res_ll(vals, mask, iters=12):
    """The result log-loss of the rows of mask with an offset per pair and seat order fitted on the
    pair's other games (cross-fitted by the game's parity, prior sd --sigma): what v tells within a
    pair, not which deck is stronger."""
    sel = np.nonzero(mask & (result != 0))[0]
    q = np.unique(ps[sel], return_inverse=True)[1]
    V, Q = torch.tensor(vals[sel], dtype=torch.float64, device=dev), torch.tensor(q, device=dev)
    S, Y = SG[torch.tensor(sel, device=dev)].double(), Yr[torch.tensor(sel, device=dev)].double()
    fold = torch.tensor(game[sel] % 2, device=dev)
    n_q, tot = int(q.max()) + 1, 0.0
    for f in (0, 1):
        fit, ev = fold != f, fold == f
        b = torch.zeros(n_q, dtype=torch.float64, device=dev)
        for _ in range(iters):  # (Newton per offset)
            p = torch.sigmoid(S[fit] * b[Q[fit]] + V[fit])
            g = torch.zeros_like(b).index_add_(0, Q[fit], S[fit] * (p - Y[fit])) + b / args.sigma ** 2
            h = torch.zeros_like(b).index_add_(0, Q[fit], p * (1 - p)) + 1 / args.sigma ** 2
            b -= g / h
        tot += bce(S[ev] * b[Q[ev]] + V[ev], Y[ev], reduction="sum").item()
    return tot / len(sel)

print(f"{len(X)} rows, K {K:.1f} (per set: {', '.join(f'{v_:.1f}' for v_ in Ks.values())}); train {is_tr.sum()} val-games {is_vin.sum()} held-pairs {is_hp.sum()}")
print(f"baselines rank held-pairs: static {ranking(static, is_hp):.4f} search {ranking(value, is_hp):.4f}")
B = 8192
for ep in range(args.epochs):
    perm = tr_idx[torch.randperm(N, device=dev)]
    for i in range(0, N, B):
        b = perm[i:i + B]
        if args.card_decorr:
            cs = card_x(b).sum(1)
            l_res, l_val = loss_of(b, weighted=args.balance_wins, v=net(Z[b]).squeeze(-1) + cs)
            t = SG[b] * a.detach()[PS[b]]
            cs_, t_ = cs - cs.mean(), t - t.mean()
            corr2 = (cs_ * t_).mean() ** 2 / ((cs_ ** 2).mean() * (t_ ** 2).mean()).clamp_min(1e-12)
        else:
            l_res, l_val = loss_of(b, weighted=args.balance_wins)
            corr2 = 0.0
        loss = (1 - args.lam) * l_res + args.lam * l_val + (a ** 2).sum() / (2 * args.sigma ** 2) / N + args.card_decorr * corr2
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 3 == 2 or ep == args.epochs - 1:
        report(f"epoch {ep + 1}")
print("pair term: sd", a.detach().std().item())
if card_params:
    with torch.no_grad():
        s = card_strength(EC_t, HE_t).cpu().numpy()
        print("card term: multipliers (bucket x [neither, used, negated, both])\n", torch.cat([torch.ones(N_BUCKETS, 1, device=dev), 1 + card_m], 1).cpu().numpy().round(3))
        print("card term: strength sd per bucket", s[HE > 0].std(0).round(4), *(() if args.card_out else ("first layer's weights of the card inputs", net[0].weight[:, n_in:].norm(dim=0).cpu().numpy().round(3))))
        if args.output:
            np.savez(args.output + ".cards.npz", codes=np.load(args.cards)["codes"], strength=s, mult=torch.cat([torch.ones(N_BUCKETS, 1, device=dev), 1 + card_m], 1).cpu().numpy())
if args.save_v:
    np.savez(args.save_v, v=all_v())
def card_table():
    """The card term for the bot: every card's strength per bucket (the cards with an effect), the
    multipliers per bucket and state. With --card-out (`direct`) the strengths in the output's
    points (times K, as w2)."""
    with torch.no_grad():
        m = torch.cat([torch.ones(N_BUCKETS, 1), 1 + card_m.cpu()], 1).numpy()
        if args.card_free:  # (the codes of the data only: a code never seen has no value)
            s, codes, keep = card_strength(EC_t, HE_t).cpu().numpy(), DATA_CODES, HE > 0
        else:
            s, codes, keep = card_strength(EF_emb, EF_has).numpy(), EF_codes, EF_has.numpy() > 0
    if args.card_out:
        s = s * K
    return {"codes": codes[keep].tolist(), "strength": s[keep].astype(float).tolist(), "mult": m.astype(float).tolist(),
            **({"direct": True} if args.card_out else {})}

if args.output:
    import json
    assert args.model == "mlp"
    FEATURE_SETS = {87: "mover2", 99: "mover5", 104: "mover3", 122: "mover4"}
    st = {kk: vv.detach().cpu().numpy().astype(float) for kk, vv in net.state_dict().items()}
    if args.drop:  # (dropped inputs were 0 in training but not at inference: their weights out)
        st["0.weight"][:, [int(i) for i in args.drop.split(",")]] = 0
        sd[[int(i) for i in args.drop.split(",")]] = 1
    if args.indicator:  # (the indicator is 0 in play: its input (0 - mu) / sd joins the bias)
        st["0.bias"] = st["0.bias"] + st["0.weight"][:, -1] * (0 - mu[-1]) / sd[-1]
        assert not card_params
        st["0.weight"], mu, sd, n = st["0.weight"][:, :-1], mu[:-1], sd[:-1], n - 1
    json.dump({"features": FEATURE_SETS.get(n_in, "mover"), "model": "mover_mlp", "mu": mu.astype(float).tolist(), "scale": sd.astype(float).tolist(),
               "w1": st["0.weight"].tolist(), "b1": st["0.bias"].tolist(),
               "layers": [{"w": st[f"{2 * i}.weight"].tolist(), "b": st[f"{2 * i}.bias"].tolist()} for i in range(1, args.layers)],
               "w2": (st[f"{2 * args.layers}.weight"][0] * K).tolist(), "b2": float(st[f"{2 * args.layers}.bias"][0] * K),
               "k": float(K), "lam": args.lam, "hidden": hidden[0], "dims": hidden, "n_layers": args.layers,
               **({"cards": card_table()} if card_params else {})}, open(args.output, "w"))
    print("saved", args.output)
    # the LP anchor (bot-anchor) on the held-out pairs' rows, so that the bot's values are in LP
    import os, subprocess, tempfile
    anchor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "target", "release", "bot-anchor")
    if os.path.exists(anchor):
        hp = np.nonzero(is_hp)[0]
        if card_params:
            with torch.no_grad():
                cx = torch.cat([card_x(torch.tensor(hp[i:i + 100000], device=dev)) for i in range(0, len(hp), 100000)]).cpu().numpy()
                if args.card_out:
                    cx = cx * K
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for j, row in enumerate(d["x"][hp][:, :n_in]):
                r = {FEATURE_SETS.get(n_in, "mover"): row.astype(float).tolist()}
                if card_params:
                    r["card_x"] = cx[j].astype(float).tolist()
                f.write(json.dumps(r) + "\n")
        subprocess.run([anchor, args.output, f.name, "--write"], check=True)
        os.unlink(f.name)
    else:
        print(f"warning: {anchor} not built: {args.output} has no LP anchor, the bot refuses it until bot-anchor MODEL ROWS --write")
