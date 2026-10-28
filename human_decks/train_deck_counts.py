#!/usr/bin/env python3
"""Train a deck-count model on the human .ydk corpus.

Given the *complete set* of unique cards in a deck (just the binary
set — no count info), predict how many copies of each card the deck
contains, ∈ {1, 2, 3}, formulated as 3-way classification (one logit
per count class).  At inference the predicted count is argmax + 1.

This is the count head of a two-stage deck pipeline: the existence
model (train_deck_existence.py) predicts which cards are in the deck,
and this model fills in the per-card multiplicity.

Encoder: each card in the set becomes a token, those tokens are
sum-pooled into a single context vector, every token is then concat'd
with the context vector and run through an FC to produce a context-
conditioned embedding, and a second sum-pool produces the deck
representation.  For each input token the deck rep is concatenated
with the card's input embedding and the normalized deck-size scalar,
then fed through an MLP trunk + 3-way classification head.  Loss:
cross-entropy over the in-set tokens only.

Vocab and cards.cdb handling mirrors train_deck_existence.py so the
two checkpoints can be paired at inference time.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from tqdm import tqdm

from deck_common import (
    ContextRefiner,
    build_extra_flags,
    load_or_build_vocab,
    parse_ydk_deck,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class DeckCountsDataset(Dataset):
    """Each example: (unique_card_indices, counts) for one .ydk."""

    def __init__(self, paths, vocab):
        self.V = len(vocab)
        self.decks = []
        n_skipped = 0
        for p in paths:
            idxs = [vocab[c] for c in parse_ydk_deck(p) if c in vocab]
            if not idxs:
                continue
            counts = Counter(idxs)
            if max(counts.values()) > 3:
                n_skipped += 1
                continue
            uniques = list(counts.keys())
            cnts = [counts[c] for c in uniques]
            self.decks.append((uniques, cnts))
        if n_skipped:
            print(f"skipped {n_skipped} decks with >3 copies of some card")

    def __len__(self):
        return len(self.decks)

    def __getitem__(self, i):
        return self.decks[i]


def collate_counts(batch):
    """Pack a batch of (uniques, counts) into padded tensors.

    Returns:
        card_ids          (B, L_max)  long, padded with 0
        targets           (B, L_max)  long, class index ∈ {0, 1, 2} for counts {1, 2, 3}
        key_padding_mask  (B, L_max)  bool, True at padding
    """
    B = len(batch)
    L_max = max(len(cards) for cards, _ in batch)
    L_max = max(L_max, 1)
    card_ids = torch.zeros(B, L_max, dtype=torch.long)
    targets = torch.zeros(B, L_max, dtype=torch.long)
    key_padding_mask = torch.ones(B, L_max, dtype=torch.bool)
    for b, (cards, cnts) in enumerate(batch):
        # Shuffle so the model can't memorize a positional ordering.
        order = list(range(len(cards)))
        random.shuffle(order)
        for i, j in enumerate(order):
            card_ids[b, i] = cards[j]
            targets[b, i] = cnts[j] - 1
            key_padding_mask[b, i] = False
    return card_ids, targets, key_padding_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CountModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden=512, n_trunk_layers=2):
        super().__init__()
        self.V = vocab_size
        self.d_model = d_model
        self.card_emb = nn.Embedding(vocab_size, d_model)
        self.refiner = ContextRefiner(d_model)

        # Per-token head: concat(card_emb, deck_rep) → MLP → 3-class logits.
        layers, in_dim = [], 2 * d_model
        for _ in range(n_trunk_layers):
            layers += [nn.Linear(in_dim, hidden), nn.GELU()]
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 3)

    def forward(self, card_ids, key_padding_mask):
        """
        card_ids          (B, L) long — card indices in the deck, 0-padded
        key_padding_mask  (B, L) bool — True at padding positions

        Returns logits (B, L, 3) — one logit per count class {1, 2, 3}.
        """
        B, L = card_ids.shape
        tok = self.card_emb(card_ids)                          # (B, L, d)
        valid = (~key_padding_mask).float()                    # (B, L)
        pooled, _ = self.refiner(tok, valid)                   # (B, d)

        # Per-token head: concat each input card emb with deck rep.
        deck_rep = pooled.unsqueeze(1).expand(-1, L, -1)       # (B, L, d)
        z = torch.cat([tok, deck_rep], dim=-1)                 # (B, L, 2d)
        h = self.trunk(z)                                      # (B, L, hidden)
        return self.head(h)                                    # (B, L, 3)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def count_loss(logits, targets, key_padding_mask):
    """3-way cross-entropy over in-set tokens; padding excluded.

    logits  : (B, L, 3) class logits
    targets : (B, L) long in {0, 1, 2}
    """
    valid = (~key_padding_mask).float()
    ce = F.cross_entropy(logits.reshape(-1, 3),
                         targets.reshape(-1),
                         reduction="none").view_as(targets)
    return (ce * valid).sum() / valid.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, opt=None,
              log_losses=None, mid_callbacks=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    callbacks = mid_callbacks or {}
    with torch.set_grad_enabled(train):
        for step, (card_ids, targets, kpm) in enumerate(tqdm(loader)):
            if step in callbacks:
                callbacks[step]()
                model.train(train)
            card_ids = card_ids.to(device)
            targets = targets.to(device)
            kpm = kpm.to(device)
            pred = model(card_ids, kpm)
            loss = count_loss(pred, targets, kpm)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            # Weight by number of valid tokens so the mean is per-card, not per-deck.
            bs = (~kpm).sum().item()
            l = loss.item()
            total += l * bs
            n += bs
            if log_losses is not None:
                log_losses.append(l)
    return total / max(n, 1)


@torch.no_grad()
def eval_accuracy(model, loader, device):
    """Per-class and overall token-level accuracy.

    Predicted count is argmax(logits) + 1.
    """
    model.eval()
    n_per_class = [0, 0, 0]
    correct_per_class = [0, 0, 0]
    for card_ids, targets, kpm in loader:
        card_ids = card_ids.to(device)
        targets = targets.to(device)
        kpm = kpm.to(device)
        pred_cls = model(card_ids, kpm).argmax(dim=-1)
        valid = ~kpm
        for c in range(3):
            mask = valid & (targets == c)
            n_per_class[c] += mask.sum().item()
            correct_per_class[c] += ((pred_cls == c) & mask).sum().item()
    total_n = sum(n_per_class)
    total_correct = sum(correct_per_class)
    out = {"overall": total_correct / max(total_n, 1),
           "n": total_n}
    for c in range(3):
        out[f"acc_{c+1}"] = correct_per_class[c] / max(n_per_class[c], 1)
        out[f"n_{c+1}"] = n_per_class[c]
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--decks-dir", default=here / "decks")
    ap.add_argument("--cdb",
                    default=here.parent / "external" / "BabelCDB" / "cards.cdb",
                    help="path to cards.cdb (used to flag extra-deck cards)")
    ap.add_argument("--vocab-cache", default=here / "vocab.pkl")
    ap.add_argument("--save-path", default=here / "deck_counts.pt",
                    help="where to save the trained model (use '' to skip)")
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                    help="cosine: decays the learning rate to 0 over --epochs (per epoch)")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-trunk-layers", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--metrics-out", default="",
                    help="path to write per-epoch metrics as JSON")
    ap.add_argument("--mid-epoch-vals", type=int, default=0,
                    help="number of additional val passes per epoch (evenly spaced)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.save_path:
        save_path = Path(args.save_path)
        if save_path.exists():
            sys.exit(f"error: {save_path} already exists; remove it or pass "
                     f"a different --save-path")

    # Vocab — shared cache with the existence trainer.
    vocab, paths = load_or_build_vocab(args.vocab_cache, args.decks_dir,
                                       min_freq=args.min_freq)
    V = len(vocab)
    print(f"vocab size = {V:,}   decks = {len(paths):,}")

    is_extra = build_extra_flags(vocab, args.cdb)

    print("loading decks into RAM ...")
    ds = DeckCountsDataset(paths, vocab)
    n_val = max(1, len(ds) // 50)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_counts, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_counts, num_workers=args.num_workers)

    model = CountModel(
        V,
        d_model=args.d_model,
        hidden=args.hidden,
        n_trunk_layers=args.n_trunk_layers,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.lr_schedule == "cosine" else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params:,}")

    best_val = float("inf")
    best_epoch = -1
    best_state = None

    def save_best(epoch, val):
        nonlocal best_state
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        if not args.save_path:
            return
        torch.save({
            "state_dict": best_state,
            "config": {
                "vocab_size": V,
                "d_model": args.d_model,
                "hidden": args.hidden,
                "n_trunk_layers": args.n_trunk_layers,
            },
            "vocab": vocab,
            "is_extra": is_extra,
            "val_loss": val,
            "epoch": epoch,
        }, save_path)
        print(f"saved best checkpoint (epoch {epoch}, val={val:.4f}) "
              f"to {save_path}")

    metrics = {
        "config": {
            "lr": args.lr,
            "lr_schedule": args.lr_schedule,
            "d_model": args.d_model,
            "hidden": args.hidden,
            "n_trunk_layers": args.n_trunk_layers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "mid_epoch_vals": args.mid_epoch_vals,
            "vocab_size": V,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_params": n_params,
        },
        "epochs": [],
    }

    def write_metrics():
        if args.metrics_out:
            Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))

    for epoch in range(args.epochs):
        t0 = time.time()
        loss_traj = []
        mid_vals = []
        callbacks = {}
        if args.mid_epoch_vals > 0:
            n_batches = len(train_loader)
            for i in range(args.mid_epoch_vals):
                step = int(n_batches * (i + 1) / (args.mid_epoch_vals + 1))
                def _make_cb(s):
                    def cb():
                        vl = run_epoch(model, val_loader, args.device, None)
                        mid_vals.append({"step_frac": s / n_batches,
                                         "val_loss": vl})
                    return cb
                callbacks[step] = _make_cb(step)

        train_loss = run_epoch(model, train_loader, args.device, opt,
                               log_losses=loss_traj, mid_callbacks=callbacks)
        if sched is not None:
            sched.step()
        val_loss = run_epoch(model, val_loader, args.device, None)
        marker = ""
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            marker = "  *"
            save_best(best_epoch, best_val)
        elapsed = time.time() - t0
        print(f"epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  "
              f"({elapsed:.1f}s){marker}")

        metrics["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mid_vals": mid_vals,
            "loss_trajectory": loss_traj,
            "wall_time": elapsed,
        })
        write_metrics()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nfinal accuracy on best model "
          f"(epoch {best_epoch}, val={best_val:.4f}):")
    acc = eval_accuracy(model, val_loader, args.device)
    print(f"  overall:  {acc['overall']:.4f}  (n={acc['n']:,})")
    for c in (1, 2, 3):
        print(f"  count={c}: {acc[f'acc_{c}']:.4f}  (n={acc[f'n_{c}']:,})")
    metrics["final_accuracy"] = acc
    write_metrics()


if __name__ == "__main__":
    main()
