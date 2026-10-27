"""Candidate openings for a book, screened by the one-sided predictor (train_onesided.py --split
none --out): random openings (deck pairs from the corpus, seeds), each with its main decks' draw
order as the game shuffles them (ygoai.game.new_record), then scored; the KEEP share predicted
least often decided by the seat goes on to the policy games (ygoai-book-test).

    .venv/bin/python scripts/book/onesided_candidates.py generate ALL.jsonl -n N [--seed S]
    .venv/bin/python scripts/book/embed_cards.py EMB.npz ALL.jsonl          (the candidates' cards)
    .venv/bin/python scripts/book/onesided_candidates.py score ALL.jsonl KEPT.jsonl --model M.pt --emb EMB.npz [--keep 0.5]

Kept rows keep their candidate order and get "onesided": the predicted share of pairs the seat
decides.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TOP = 8


def random_openings(n: int, seed: int) -> list[tuple[int, list]]:
    """`n` openings (a seed and a pair of decks) of random corpus decks."""
    import random

    from ygoai.carddb import default_db
    from ygoai.decks import DeckPool
    pool = DeckPool(default_db())
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d0 = pool.sample(rng)
        rng.random()  # (the draw for a mirror match, never one: the candidates of match2 drew it)
        out.append((seed * 1_000_003 + i, [d0, pool.sample(rng)]))
    return out


def generate(a):
    import ygoai_rs

    with open(a.out, "w") as f:
        for seed, decks in random_openings(a.n, a.seed):
            decks = [d.to_json() for d in decks]
            # (deck_orders: bottom card first; the top cards are drawn first)
            order = [list(reversed(o)) for o in ygoai_rs.new_record(decks, seed)["deck_orders"]]
            f.write(json.dumps({"seed": seed, "decks": decks, "top": [o[:TOP] for o in order], "order": order}) + "\n")
    print(f"{a.n} candidates -> {a.out}")


def score(a):
    from train_onesided import INPUTS, Net, tokens

    ck = torch.load(a.model, map_location="cpu")
    args = ck["args"]
    assert args["cards"] == "text", "card ids cannot score unseen cards"
    z = np.load(a.emb)
    index = {int(c): i + 1 for i, c in enumerate(z["codes"])}
    emb = torch.cat([torch.zeros(1, z["emb"].shape[1]), torch.tensor(z["emb"])])
    parts = INPUTS[args["inputs"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = []
    for state in ck["states"]:
        net = Net(emb, len(index) + 1, args["model"], len(parts), d=args["d"], layers=args["layers"], heads=args["heads"], dropout=args["dropout"])
        # (the card table: this file's, with the candidates' cards)
        missing, unexpected = net.load_state_dict({k: v for k, v in state.items() if k != "table.weight"}, strict=False)
        assert missing == ["table.weight"] and not unexpected, (missing, unexpected)
        nets.append(net.to(dev).eval())
    rows = [json.loads(line) for line in open(a.candidates)]
    missing_cards = {c for r in rows for d in r["decks"] for c in d["main"] + d["extra"] if c not in index}
    assert not missing_cards, f"{len(missing_cards)} cards without an embedding (embed_cards.py on the candidates)"
    idx, group, pos = tokens(rows, index, parts)
    idx_t, group_t, pos_t = torch.tensor(idx, device=dev), torch.tensor(group, device=dev), torch.tensor(pos, device=dev)
    with torch.no_grad():
        logit = torch.stack([torch.cat([net(b, group_t, pos_t) for b in idx_t.split(512)]) for net in nets]).mean(0)
    p = torch.sigmoid(logit).cpu().numpy()
    threshold = np.quantile(p, a.keep)
    kept = 0
    with open(a.out, "w") as f:
        for r, q in zip(rows, p):
            if q <= threshold:
                f.write(json.dumps({"seed": r["seed"], "decks": r["decks"], "onesided": round(float(q), 4)}) + "\n")
                kept += 1
    print(f"{len(rows)} candidates, predicted seat-decided mean {p.mean():.3f}; kept {kept} (<= {threshold:.3f}, mean {p[p <= threshold].mean():.3f}) -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("out")
    g.add_argument("-n", type=int, required=True)
    g.add_argument("--seed", type=int, default=91)
    s = sub.add_parser("score")
    s.add_argument("candidates")
    s.add_argument("out")
    s.add_argument("--model", required=True)
    s.add_argument("--emb", required=True)
    s.add_argument("--keep", type=float, default=0.5)
    a = ap.parse_args()
    generate(a) if a.cmd == "generate" else score(a)


if __name__ == "__main__":
    main()
