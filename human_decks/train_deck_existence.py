#!/usr/bin/env python3
"""Train a deck-existence model on the human .ydk corpus.

Given a random *visible* subset of a deck (main + extra combined),
predict for every card in the vocabulary whether it appears anywhere
in the full deck (visible + unseen).  Counts are predicted by a
separate model (see train_deck_counts.py).

Encoder: each unique visible card becomes a token, those tokens are
sum-pooled (count-weighted) into a single context vector, every token
is then concat'd with the context vector and run through an FC to
produce a context-conditioned embedding, and a second count-weighted
sum-pool produces the final deck representation.  That representation
feeds an MLP trunk and a per-card presence head (single logit per
vocab card).  Loss: binary cross-entropy.

The model treats main and extra as one big multiset.  At inference
the two sections are recovered by card type (Fusion/Synchro/XYZ/Link
→ extra, everything else → main), looked up once at training time
from cards.cdb and stored alongside the weights.

After each epoch we also evaluate at a sweep of *fixed* visible-card
counts (k = 1, 2, 3, 5, 10, 20, 30) to characterize how well the model
identifies the deck early-game vs. late-game.
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
    CardTable,
    ContextRefiner,
    baked_state,
    build_extra_flags,
    card_feature_ids,
    quantize_aware,
    load_or_build_vocab,
    parse_ydk_deck,
    print_eval_table,
)


class DeckDataset(Dataset):
    """Holds each deck as a pre-tokenized list of card indices.

    Masking is done in `collate_mask` so every epoch sees a fresh random
    reveal of the same deck.
    """

    def __init__(self, paths, vocab):
        self.V = len(vocab)
        self.decks = []
        n_skipped = 0
        for p in paths:
            idxs = [vocab[c] for c in parse_ydk_deck(p) if c in vocab]
            if not idxs:
                continue
            # Yu-Gi-Oh allows max 3 copies of any card.  A small number of
            # corrupt .ydk files violate this; the MLM loss masks count classes
            # > 3 − c_vis, so an illegal target produces inf and trashes
            # training.  Drop them defensively.
            counts = Counter(idxs)
            if max(counts.values()) > 3:
                n_skipped += 1
                continue
            self.decks.append(idxs)
        if n_skipped:
            print(f"skipped {n_skipped} decks with >3 copies of some card")

    def __len__(self):
        return len(self.decks)

    def __getitem__(self, i):
        return self.decks[i]


def build_token_tensors(per_batch_visible):
    """Pack a per-batch list of [(card_idx, count), ...] into padded tensors.

    Returns (card_ids, card_counts, key_padding_mask) with shape (B, L_max).
    L_max is at least 1, so attention always sees at least one (possibly
    fully-masked) position alongside the model's context token.
    """
    B = len(per_batch_visible)
    L_max = max((len(v) for v in per_batch_visible), default=0)
    L_max = max(L_max, 1)
    card_ids = torch.zeros(B, L_max, dtype=torch.long)
    card_counts = torch.zeros(B, L_max, dtype=torch.long)
    key_padding_mask = torch.ones(B, L_max, dtype=torch.bool)
    for b, cards in enumerate(per_batch_visible):
        for i, (cid, cnt) in enumerate(cards):
            card_ids[b, i] = cid
            card_counts[b, i] = cnt
            key_padding_mask[b, i] = False
    return card_ids, card_counts, key_padding_mask


class Collate:
    """A DataLoader collate_fn (a class: worker processes must pickle it) that masks a random
    visible subset of each deck. The cards are lists of (card, count), not dense (B, V) count
    vectors: those would be ~7 MB per batch to pass between processes (`dense_counts` makes them on
    the device). Returns:
        card_ids          (B, L_max)  long, the visible cards, padded with 0
        card_counts       (B, L_max)  long, their counts, padded with 0
        key_padding_mask  (B, L_max)  bool, True at padding
        unseen_ids        (B, U_max)  long, the unseen cards, padded with 0
        unseen_counts     (B, U_max)  long, their counts (at most 3), padded with 0
        slots_in_deck     (B,)        float, total deck size (visible + unseen)
    """

    def __init__(self, V, min_reveal=0.05, max_reveal=0.95):
        self.V, self.min_reveal, self.max_reveal = V, min_reveal, max_reveal

    def __call__(self, batch):
        B = len(batch)
        slots_in_deck = torch.zeros(B, dtype=torch.float32)
        per_batch_visible, per_batch_unseen = [], []
        for b, idxs in enumerate(batch):
            idxs = list(idxs)
            random.shuffle(idxs)
            r = random.uniform(self.min_reveal, self.max_reveal)
            k = max(1, int(round(r * len(idxs))))
            vis_counter = Counter()
            for j in idxs[:k]:
                if vis_counter[j] < 3:
                    vis_counter[j] += 1
            unseen_counter = Counter()
            for j in idxs[k:]:
                if unseen_counter[j] < 3:
                    unseen_counter[j] += 1
            slots_in_deck[b] = len(idxs)
            per_batch_visible.append(list(vis_counter.items()))
            per_batch_unseen.append(list(unseen_counter.items()))
        card_ids, card_counts, key_padding_mask = build_token_tensors(per_batch_visible)
        unseen_ids, unseen_counts, _ = build_token_tensors(per_batch_unseen)
        return card_ids, card_counts, key_padding_mask, unseen_ids, unseen_counts, slots_in_deck


def dense_counts(ids, counts, V):
    """(B, V) count vectors from padded (card, count) lists (padding: count 0)."""
    return torch.zeros(ids.shape[0], V, dtype=torch.long, device=ids.device).scatter_add_(1, ids, counts)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ExistenceModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, hidden=512, n_trunk_layers=2,
                 slots_scale=1.0, emb_buckets=None, features=None):
        super().__init__()
        self.V = vocab_size
        self.d_model = d_model
        # Pre-divide the raw slots_in_deck scalar before concatenation so the
        # trunk sees an O(1) feature instead of a 0–60 bare integer.
        self.slots_scale = float(slots_scale)
        # (`features`: (S, n) for a checkpoint, whose ids the state dict brings, or the ids and n)
        if features is not None and not torch.is_tensor(features[0]):
            features = (torch.zeros(vocab_size, features[0], dtype=torch.long), features[1])
        if emb_buckets is None and features is None:
            self.card_emb = nn.Embedding(vocab_size, d_model)
        else:
            self.card_table = CardTable(vocab_size, d_model, emb_buckets or [(0, d_model)], features)
        # Per-count pool weight, init to identity (0, 1, 2, 3).
        self.count_weight = nn.Embedding(4, 1)
        with torch.no_grad():
            self.count_weight.weight.copy_(torch.arange(4).float().unsqueeze(-1))
        self.refiner = ContextRefiner(d_model)

        layers, in_dim = [], d_model + 1
        for _ in range(n_trunk_layers):
            layers += [nn.Linear(in_dim, hidden), nn.GELU()]
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.head_proj = nn.Linear(hidden, d_model)
        self.presence_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, card_ids, card_counts, key_padding_mask, slots_in_deck):
        """
        card_ids          (B, L) long — visible card indices, 0-padded
        card_counts       (B, L) long — count per token, 0 at padding
        key_padding_mask  (B, L) bool — True at padding positions
        slots_in_deck     (B,) float — total deck size (visible + unseen)
        """
        table = self.card_emb.weight if hasattr(self, "card_emb") else self.card_table()  # (V, d)
        tok = F.embedding(card_ids, table)                     # (B, L, d)
        w = self.count_weight(card_counts).squeeze(-1)         # (B, L)
        w = w.masked_fill(key_padding_mask, 0.0)
        pooled, _ = self.refiner(tok, w)                       # (B, d)

        slots_norm = slots_in_deck.float() / self.slots_scale
        z = torch.cat([pooled, slots_norm.unsqueeze(-1)], dim=-1)
        h = self.trunk(z)
        q = self.head_proj(h)                                  # (B, d)
        logits = q @ table.T + self.presence_bias              # (B, V)
        return logits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def presence_loss(logits, c_vis, c_unseen):
    """Binary cross-entropy: is each card present in the full deck?

    logits : (B, V)  — per-card presence logit
    target : (B, V)  — 1 iff (c_vis + c_unseen) > 0

    Cards with c_vis > 0 are excluded from the loss — their presence is
    a trivial given, so training on them is wasted signal.
    """
    target = ((c_vis + c_unseen) > 0).float()
    mask = (c_vis == 0).float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * mask).sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, opt=None,
              log_losses=None, mid_callbacks=None):
    """Run one epoch.

    log_losses     : if a list, per-batch BCE losses are appended.
    mid_callbacks  : dict {step_idx: fn()} fired before that step.  Used
                     to inject mid-epoch val passes from the train loop.
                     The callback may put the model in eval mode; train
                     mode is restored after it returns.
    """
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    callbacks = mid_callbacks or {}
    with torch.set_grad_enabled(train):
        for step, (card_ids, card_counts, kpm, unseen_ids, unseen_counts, slots_in_deck) in enumerate(tqdm(loader)):
            if step in callbacks:
                callbacks[step]()
                model.train(train)
            card_ids = card_ids.to(device)
            card_counts = card_counts.to(device)
            kpm = kpm.to(device)
            c_vis = dense_counts(card_ids, card_counts, model.V)
            c_unseen = dense_counts(unseen_ids.to(device), unseen_counts.to(device), model.V)
            slots_in_deck = slots_in_deck.to(device)
            logits = model(card_ids, card_counts, kpm, slots_in_deck)
            loss = presence_loss(logits, c_vis, c_unseen)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = c_vis.shape[0]
            l = loss.item()
            total += l * bs
            n += bs
            if log_losses is not None:
                log_losses.append(l)
    return total / max(n, 1)


@torch.no_grad()
def eval_at_k(model, decks, V, device, k, batch_size=128, top_ks=(10, 20, 50)):
    """Evaluate the model when exactly k cards are visible.

    Returns dict with mean BCE loss and recall@K for each K in top_ks.
    Recall is over the set of *new* cards in the unseen part (cards with
    c_unseen > 0 AND c_vis == 0).  Already-visible cards are excluded from
    both ranking and truth — the existence model cannot say anything new
    about cards we've already seen.
    """
    model.eval()
    eligible = [d for d in decks if len(d) > k]
    if not eligible:
        return None

    total_loss = 0.0
    total_recall = {tk: 0.0 for tk in top_ks}
    n = 0
    for i in range(0, len(eligible), batch_size):
        batch = eligible[i:i + batch_size]
        B = len(batch)
        c_vis = torch.zeros(B, V, dtype=torch.long)
        c_unseen = torch.zeros(B, V, dtype=torch.long)
        slots_in_deck = torch.zeros(B, dtype=torch.float32)
        per_batch_visible = []
        for b, idxs in enumerate(batch):
            idxs = list(idxs)
            random.shuffle(idxs)
            vis_counter = Counter()
            for j in idxs[:k]:
                if vis_counter[j] < 3:
                    vis_counter[j] += 1
            for j, n_copies in vis_counter.items():
                c_vis[b, j] = n_copies
            for j in idxs[k:]:
                if c_unseen[b, j] < 3:
                    c_unseen[b, j] += 1
            slots_in_deck[b] = len(idxs)
            per_batch_visible.append(list(vis_counter.items()))
        card_ids, card_counts, kpm = build_token_tensors(per_batch_visible)

        card_ids = card_ids.to(device)
        card_counts = card_counts.to(device)
        kpm = kpm.to(device)
        c_vis = c_vis.to(device)
        c_unseen = c_unseen.to(device)
        slots_in_deck = slots_in_deck.to(device)
        logits = model(card_ids, card_counts, kpm, slots_in_deck)

        total_loss += presence_loss(logits, c_vis, c_unseen).item() * B

        # Rank by P(presence).  Cards already visible are trivially present
        # and contribute no signal — mask them out of both ranking and truth.
        p_pos = torch.sigmoid(logits)
        p_pos = p_pos.masked_fill(c_vis > 0, -1.0)
        truth = (c_unseen > 0) & (c_vis == 0)
        n_pos = truth.float().sum(dim=-1).clamp(min=1.0)
        for tk in top_ks:
            top = p_pos.topk(tk, dim=-1).indices
            hits = truth.gather(1, top).float().sum(dim=-1)
            total_recall[tk] += (hits / n_pos).sum().item()
        n += B

    return {
        "k": k,
        "n": n,
        "loss": total_loss / n,
        **{f"r@{tk}": total_recall[tk] / n for tk in top_ks},
    }


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
    ap.add_argument("--eval-ks", type=int, nargs="+",
                    default=[1, 2, 3, 5, 10, 20, 30],
                    help="visible-card counts to evaluate per epoch")
    ap.add_argument("--eval-max-decks", type=int, default=2000,
                    help="cap on val decks used per fixed-k eval")
    ap.add_argument("--save-path", default=here / "deck_existence.pt",
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
    ap.add_argument("--emb-buckets", default="",
                    help="card embedding dimensions by frequency, e.g. 2000:128,6000:64,0:32 (0: the rest)")
    ap.add_argument("--qat-bits", type=int, default=0,
                    help="train with the matrices rounded to this many bits (as eval_deck_models.py --bits stores them)")
    ap.add_argument("--card-features", action="store_true",
                    help="add embeddings of the cards' set codes, types, race, attribute and level")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--metrics-out", default="",
                    help="path to write per-epoch metrics as JSON")
    ap.add_argument("--fixed-k-eval-min-gap", type=int, default=0,
                    help="during training, only run the fixed-k eval when "
                         "more than this many epochs have passed since the "
                         "last eval AND the current epoch produced a new "
                         "best val loss. A final eval on the best model is "
                         "always run after training.")
    ap.add_argument("--mid-epoch-vals", type=int, default=0,
                    help="number of additional val passes per epoch (evenly spaced)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Bail out early if the save target already exists — never silently
    # overwrite a trained checkpoint.
    if args.save_path:
        save_path = Path(args.save_path)
        if save_path.exists():
            sys.exit(f"error: {save_path} already exists; remove it or pass "
                     f"a different --save-path")

    # Vocab
    vocab, paths = load_or_build_vocab(args.vocab_cache, args.decks_dir,
                                       min_freq=args.min_freq)
    V = len(vocab)
    print(f"vocab size = {V:,}   decks = {len(paths):,}")

    # Per-card is_extra flag — looked up once from cards.cdb and stored in
    # the checkpoint so inference can split the predicted deck without
    # needing the cdb itself.
    is_extra = build_extra_flags(vocab, args.cdb)

    # Data
    print("loading decks into RAM ...")
    ds = DeckDataset(paths, vocab)
    n_val = max(1, len(ds) // 50)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    collate = Collate(V)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)

    # Model
    buckets = [tuple(int(x) for x in b.split(":")) for b in args.emb_buckets.split(",")] if args.emb_buckets else None
    features = card_feature_ids(vocab, args.cdb) if args.card_features else None
    model = ExistenceModel(
        V,
        d_model=args.d_model,
        hidden=args.hidden,
        n_trunk_layers=args.n_trunk_layers,
        slots_scale=60.0,
        emb_buckets=buckets,
        features=features,
    ).to(args.device)
    if args.qat_bits:
        quantize_aware(model, args.qat_bits)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.lr_schedule == "cosine" else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params:,}")

    # A frozen subset of val decks used for the fixed-k diagnostic, so the
    # numbers across epochs are directly comparable.
    val_decks = [val_ds[i] for i in range(min(len(val_ds), args.eval_max_decks))]

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    last_fixed_k_epoch = -1

    def save_best(epoch, val):
        nonlocal best_state
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        if not args.save_path:
            return
        torch.save({
            "state_dict": baked_state(model) if args.qat_bits else best_state,
            "qat_bits": args.qat_bits,
            "config": {
                "vocab_size": V,
                "d_model": args.d_model,
                "hidden": args.hidden,
                "n_trunk_layers": args.n_trunk_layers,
                "slots_scale": 60.0,
                "emb_buckets": buckets,
                "features": [features[0].shape[1], features[1]] if features else None,
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

    # Train
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

        if improved and epoch - last_fixed_k_epoch > args.fixed_k_eval_min_gap:
            rows = [eval_at_k(model, val_decks, V, args.device, k)
                    for k in args.eval_ks]
            print_eval_table([r for r in rows if r is not None])
            last_fixed_k_epoch = epoch

    # Always run a fixed-k eval at the end on the best model, regardless
    # of whether we just ran one in the inner loop.
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nfinal fixed-k eval on best model "
          f"(epoch {best_epoch}, val={best_val:.4f}):")
    rows = [eval_at_k(model, val_decks, V, args.device, k)
            for k in args.eval_ks]
    rows = [r for r in rows if r is not None]
    print_eval_table(rows)
    metrics["final_fixed_k"] = rows
    write_metrics()


if __name__ == "__main__":
    main()
