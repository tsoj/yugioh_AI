#!/usr/bin/env python3
"""Evaluates deck model checkpoints on held-out decks, with the trainers' own metrics: for the
existence model the recall@20 of the unseen cards with K cards visible and the loss (binary cross
entropy, mean over the K: it measures the calibration too), for the counts model the accuracy of the
copies. The deck sets:

  val   the training's validation split of decks/ (random_split, seed 0, as the trainers make it)
  new   decks_new/ (downloaded later, never trained on)
  meta  yugiohmeta-decklists/ (tournament decks of another site), if present

Cards outside a model's vocabulary are left out of its decks, as in training.

The size is that of the weights as stored, compressed with zstd -19. With --bits N (16, 8, 6, 4),
each matrix is stored as N-bit integers with a 16-bit scale per row, the other weights as 16-bit
floats, and the model is evaluated with the weights it would get back from that.

    ../.venv/bin/python eval_deck_models.py CHECKPOINT.pt [...] [--decks 2000] [--bits 8] [--device cuda]
"""

import argparse
import random
from compression import zstd
from pathlib import Path

import numpy as np

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


def stored(model, bits):
    """Quantizes the model's weights in place (bits < 32) and returns their compressed size."""
    parts = []
    with torch.no_grad():
        for name, w in model.state_dict().items():
            if name.endswith("feat_ids") or name == "shared_table":
                continue  # (from the card database at load time; the existence model's)
            if bits >= 32:
                parts.append(w.cpu().numpy().astype(np.float32).tobytes())
            elif bits == 16 or w.dim() != 2:
                h = w.half()
                parts.append(h.cpu().numpy().tobytes())
                w.copy_(h.float())
            else:
                top = 2 ** (bits - 1) - 1
                # (clamped after the conversion: a row of zeros would get a scale of 0 in 16 bits)
                scale = (w.abs().amax(dim=1, keepdim=True) / top).half().float().clamp(min=1e-7)
                q = (w / scale).round().clamp(-top, top)
                parts.append(q.to(torch.int8).cpu().numpy().tobytes() + scale.half().cpu().numpy().tobytes())
                w.copy_(q * scale)
    return len(zstd.compress(b"".join(parts), level=19))


def decks_of(dataset_type, paths, vocab, split, n, min_cards=0):
    ds = dataset_type(paths, vocab)
    decks = ds.decks
    if split:
        n_val = max(1, len(ds) // 50)
        _, val = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
        decks = [ds.decks[i] for i in val.indices][:n]
    return [d for d in decks if len(d) >= min_cards]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--decks", type=int, default=2000, help="decks per set")
    ap.add_argument("--min-cards", type=int, default=0,
                    help="only decks with at least this many cards in the vocabulary (existence: the "
                         "bot's decks have 40 or more)")
    ap.add_argument("--bits", type=int, default=32, choices=[32, 16, 8, 6, 4])
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
        size = stored(model, args.bits)
        cells = []
        with torch.no_grad():
            for name, paths, split in sets:
                if existence:
                    decks = decks_of(te.DeckDataset, paths, ck["vocab"], split, args.decks, args.min_cards)
                    losses = []
                    for k in KS:
                        random.seed(0)
                        r = te.eval_at_k(model, decks, len(ck["vocab"]), args.device, k, batch_size=256)
                        cells.append(f"{name} k{k} {r['r@20']:.3f}")
                        losses.append(r["loss"])
                    cells.append(f"{name} loss {1e3 * sum(losses) / len(losses):.3f}e-3")
                else:
                    decks = decks_of(tc.DeckCountsDataset, paths, ck["vocab"], split, args.decks)
                    random.seed(0)
                    a = tc.eval_accuracy(model, DataLoader(decks, batch_size=256, collate_fn=tc.collate_counts), args.device)
                    cells.append(f"{name} {a['overall']:.3f}")
        kind = "existence r@20" if existence else "counts acc"
        print(f"{Path(path).name} ({args.bits} bits): {params / 1e6:.2f}M params, {size / 1e6:.2f} MB, {kind}: " + ", ".join(cells), flush=True)


if __name__ == "__main__":
    main()
