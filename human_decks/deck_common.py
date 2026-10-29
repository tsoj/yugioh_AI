"""Shared utilities for the deck existence/count models."""

import pickle
import sqlite3
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


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

def card_feature_ids(vocab, cdb_path):
    """Per vocabulary card the ids of its features in the card database: its set codes
    (archetypes), type bits, race, attribute and level, all in one id space. Returns ((V, S) ids,
    padded with n) and the number of ids n. The bot has the card database, so these cost no
    stored bytes."""
    db = sqlite3.connect(f"file:{cdb_path}?mode=ro", uri=True)
    rows = {r[0]: r[1:] for r in db.execute("SELECT id, setcode, type, race, attribute, level FROM datas")}
    db.close()
    keys = {}

    def key(k):
        return keys.setdefault(k, len(keys))

    feats = [[] for _ in vocab]
    for code, idx in vocab.items():
        row = rows.get(code)
        if row is None:
            continue
        setcode, typ, race, attr, level = row
        f = feats[idx]
        f += [key(("set", (setcode >> (16 * i)) & 0xFFFF)) for i in range(4) if (setcode >> (16 * i)) & 0xFFFF]
        f += [key(("type", b)) for b in range(32) if typ >> b & 1]
        f += [key(("race", b)) for b in range(64) if race >> b & 1]
        f += [key(("attribute", b)) for b in range(8) if attr >> b & 1]
        f.append(key(("level", level & 0xFF)))
    n = len(keys)
    ids = torch.full((len(vocab), max(len(f) for f in feats)), n, dtype=torch.long)
    for i, f in enumerate(feats):
        ids[i, :len(f)] = torch.tensor(f, dtype=torch.long)
    return ids, n


def quantize_rows(w, bits):
    """The matrix as `bits`-bit integers with a 16-bit scale per row: (int8 values, float16 scales),
    the matrix back: values * scales."""
    top = 2 ** (bits - 1) - 1
    # (clamped after the conversion: a row of zeros would get a scale of 0 in 16 bits)
    scale = (w.abs().amax(dim=1, keepdim=True) / top).half().float().clamp(min=1e-7)
    return (w / scale).round().clamp(-top, top).to(torch.int8), scale.half()


class FakeQuant(nn.Module):
    """A parametrization: the matrix rounded to `bits`-bit integers with a 16-bit scale per row
    (`quantize_rows`, as the models are stored); gradients pass through the rounding
    (straight-through)."""

    def __init__(self, bits):
        super().__init__()
        self.top = 2 ** (bits - 1) - 1

    def forward(self, w):
        # (clamped after the conversion: a row of zeros would get a scale of 0 in 16 bits)
        scale = (w.detach().abs().amax(dim=1, keepdim=True) / self.top).half().float().clamp(min=1e-7)
        q = (w / scale).round().clamp(-self.top, self.top) * scale
        return w + (q - w).detach()


def quantize_aware(model, bits):
    """Trains `model` with its matrices rounded to `bits` bits (FakeQuant on each 2-D weight)."""
    for m in list(model.modules()):
        w = getattr(m, "weight", None)
        if isinstance(w, nn.Parameter) and w.dim() == 2 and w.shape[1] > 1:
            parametrize.register_parametrization(m, "weight", FakeQuant(bits))


def baked_state(model):
    """The weights the model computes with (rounded ones as rounded) under their plain names."""
    out = {}
    for k, v in model.state_dict().items():
        if ".parametrizations." in k:
            if k.endswith(".original"):
                path = k.split(".parametrizations.")[0]
                out[f"{path}.weight"] = model.get_submodule(path).weight.detach().cpu().clone()
        else:
            out[k] = v.detach().cpu().clone()
    return out


class CardTable(nn.Module):
    """The card embeddings (V, d), computed: the vocabulary (indices by frequency) in buckets with a
    dimension of their own, projected to d (`buckets`: [(n, dim)], n 0 for the rest; dim 0: no
    embedding of their own), plus with `features` ((V, S) ids from `card_feature_ids`, their number)
    the sum of the features' embeddings."""

    def __init__(self, V, d, buckets, features=None):
        super().__init__()
        sizes, left = [], V
        for n, dim in buckets:
            n = left if n == 0 else min(n, left)
            sizes.append((n, dim))
            left -= n
        if left:
            raise ValueError(f"the buckets leave {left} cards without an embedding")
        if features is None and any(dim == 0 for _, dim in sizes):
            raise ValueError("cards without an embedding of their own need --card-features")
        self.d = d
        self.sizes = sizes
        self.parts = nn.ModuleList(nn.Embedding(n, dim) for n, dim in sizes if dim)
        self.projs = nn.ModuleList(nn.Identity() if dim == d else nn.Linear(dim, d, bias=False) for _, dim in sizes if dim)
        self.feat_emb = None
        if features is not None:
            ids, n = features
            self.register_buffer("feat_ids", ids)
            self.feat_emb = nn.Embedding(n + 1, d, padding_idx=n)
            with torch.no_grad():
                self.feat_emb.weight.mul_(0.3)
                self.feat_emb.weight[n].zero_()

    def forward(self):
        own = iter(zip(self.parts, self.projs))
        rows = []
        for n, dim in self.sizes:
            if dim:
                e, proj = next(own)
                rows.append(proj(e.weight))
            else:
                rows.append(self.feat_emb.weight.new_zeros(n, self.d))
        t = torch.cat(rows)
        if self.feat_emb is not None:
            t = t + self.feat_emb(self.feat_ids).sum(1)
        return t


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
