"""Shared utilities for the deck existence/count models."""

import pickle
import sqlite3
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Type flags from ygopro-core: a card is an extra-deck card iff its
# `type` column in cards.cdb has any of these bits set.
EXTRA_TYPE_MASK = 0x40 | 0x2000 | 0x800000 | 0x4000000  # Fusion | Synchro | XYZ | Link

# Bump whenever the vocab-cache layout or its semantics change (e.g. when
# we started including extra-deck cards), so stale caches are auto-rebuilt.
VOCAB_CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def parse_ydk_deck(path):
    """Return the list of card codes in main + extra (side deck ignored)."""
    cards = []
    section = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line == "#main":
                    section = "main"
                elif line == "#extra":
                    section = "extra"
                else:
                    section = None
                continue
            if line.startswith("!"):
                section = None
                continue
            if section is not None:
                try:
                    cards.append(int(line))
                except ValueError:
                    pass
    return cards


def build_vocab(decks_dir, min_freq=5):
    """Scan every .ydk under decks_dir, return (card_id → idx, list[Path])."""
    counts = Counter()
    paths = sorted(Path(decks_dir).glob("*.ydk"))
    for p in paths:
        for cid in parse_ydk_deck(p):
            counts[cid] += 1
    vocab = {cid: i for i, (cid, c) in enumerate(counts.most_common())
             if c >= min_freq}
    return vocab, paths


def build_extra_flags(vocab, cdb_path):
    """Return a list[bool] of length |vocab|: True iff that card is an extra-deck card."""
    db = sqlite3.connect(f"file:{cdb_path}?mode=ro", uri=True)
    flags = [False] * len(vocab)
    unknown = 0
    for cid, idx in vocab.items():
        row = db.execute("SELECT type FROM datas WHERE id = ?", (cid,)).fetchone()
        if row is None:
            unknown += 1
            continue
        if row[0] & EXTRA_TYPE_MASK:
            flags[idx] = True
    db.close()
    n_extra = sum(flags)
    print(f"is_extra: {n_extra:,} extra-deck cards, {len(vocab) - n_extra:,} main-deck cards"
          + (f" ({unknown} not found in cdb)" if unknown else ""))
    return flags


def load_or_build_vocab(cache_path, decks_dir, min_freq=5):
    """Load the vocab from cache_path, rebuilding from decks_dir on miss/stale."""
    cache = Path(cache_path)
    cached = None
    if cache.exists():
        with open(cache, "rb") as f:
            cached = pickle.load(f)
    if (isinstance(cached, dict)
            and cached.get("version") == VOCAB_CACHE_VERSION):
        print(f"loaded vocab from {cache}")
        return cached["vocab"], cached["paths"]
    if cached is not None:
        print(f"vocab cache at {cache} is stale (version mismatch); rebuilding")
    print(f"building vocab from {decks_dir} ...")
    vocab, paths = build_vocab(decks_dir, min_freq=min_freq)
    with open(cache, "wb") as f:
        pickle.dump({"version": VOCAB_CACHE_VERSION,
                     "vocab": vocab, "paths": paths}, f)
    return vocab, paths


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class ContextRefiner(nn.Module):
    """One round of (pool → broadcast context → per-token refine), then a final pool.

    The refine layer is concat(token, pooled) → Linear → GELU, zero-initialized
    so the module starts as a plain DeepSets pool and the refinement contribution
    grows in during training (stable warmup).

    Token `weights` handle both padding (set weight to 0) and arbitrary per-token
    multiplicities (e.g. count-weighted pooling).
    """

    def __init__(self, d_model):
        super().__init__()
        self.refine = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, tok, weights):
        """
        tok      : (B, L, d)  — input token embeddings
        weights  : (B, L)     — token weights (0 at padding/masked)

        Returns:
            pooled  : (B, d)        — final weighted pool over refined tokens
            tokens  : (B, L, d)     — refined per-token embeddings
        """
        w = weights.unsqueeze(-1)
        pooled = (tok * w).sum(dim=1)
        ctx = pooled.unsqueeze(1).expand_as(tok)
        tok = F.gelu(self.refine(torch.cat([tok, ctx], dim=-1)))
        pooled = (tok * w).sum(dim=1)
        return pooled, tok


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def print_eval_table(rows):
    """Pretty-print a list of dicts (each with a 'k' key) as an aligned table."""
    if not rows:
        return
    keys = [k for k in rows[0] if k != "k"]
    header = "  ".join(["k"] + [f"{key:>8}" for key in keys])
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = [f"{r['k']:>2}"]
        for key in keys:
            v = r[key]
            cells.append(f"{v:>8.4f}" if isinstance(v, float) else f"{v:>8d}")
        print("  ".join(cells))
