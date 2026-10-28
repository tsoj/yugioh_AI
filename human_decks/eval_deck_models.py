#!/usr/bin/env python3
"""Evaluates deck model checkpoints on held-out decks, with the trainers' own metrics: for the
existence model the recall@20 of the unseen cards with K cards visible, for the counts model the
accuracy of the copies. The deck sets:

  val   the training's validation split of decks/ (random_split, seed 0, as the trainers make it)
  new   decks_new/ (downloaded later, never trained on)
  meta  yugiohmeta-decklists/ (tournament decks of another site), if present

Cards outside a model's vocabulary are left out of its decks, as in training.

    ../.venv/bin/python eval_deck_models.py CHECKPOINT.pt [...] [--decks 2000] [--device cuda]
"""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

import train_deck_counts as tc
import train_deck_existence as te

HERE = Path(__file__).resolve().parent
KS = (1, 3, 10, 20)


def deck_sets(n):
    """The paths of each set: (name, paths, is the training split)."""
    sets = [("val", sorted((HERE / "decks").glob("*.ydk")), True),
            ("new", random.Random(0).sample(sorted((HERE / "decks_new").glob("*.ydk")), n), False)]
    meta = sorted((HERE / "yugiohmeta-decklists").glob("*.ydk"))
    if meta:
        sets.append(("meta", random.Random(1).sample(meta, min(n, len(meta))), False))
    return sets


def decks_of(dataset_type, paths, vocab, split, n):
    ds = dataset_type(paths, vocab)
    if not split:
        return ds.decks
    n_val = max(1, len(ds) // 50)
    _, val = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    return [ds.decks[i] for i in val.indices][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--decks", type=int, default=2000, help="decks per set")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    sets = deck_sets(args.decks)
    for path in args.checkpoints:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        existence = "slots_scale" in ck["config"]
        model = (te.ExistenceModel if existence else tc.CountModel)(**ck["config"])
        model.load_state_dict(ck["state_dict"])
        model.to(args.device).eval()
        params = sum(p.numel() for p in model.parameters())
        cells = []
        with torch.no_grad():
            for name, paths, split in sets:
                if existence:
                    decks = decks_of(te.DeckDataset, paths, ck["vocab"], split, args.decks)
                    for k in KS:
                        random.seed(0)
                        r = te.eval_at_k(model, decks, len(ck["vocab"]), args.device, k, batch_size=256)
                        cells.append(f"{name} k{k} {r['r@20']:.3f}")
                else:
                    decks = decks_of(tc.DeckCountsDataset, paths, ck["vocab"], split, args.decks)
                    random.seed(0)
                    a = tc.eval_accuracy(model, DataLoader(decks, batch_size=256, collate_fn=tc.collate_counts), args.device)
                    cells.append(f"{name} {a['overall']:.3f}")
        kind = "existence r@20" if existence else "counts acc"
        print(f"{Path(path).name}: {params / 1e6:.2f}M params, {kind}: " + ", ".join(cells), flush=True)


if __name__ == "__main__":
    main()
