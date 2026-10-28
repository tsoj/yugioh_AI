#!/usr/bin/env python3
"""Exports the deck models for the Rust side (crates/deckmodels) into one file, deck_models.bin:
the existence model, the counts model on its card table (train_deck_counts.py --shared-table), the
vocabulary, and test cases for crates/deckmodels' tests.

The format: zstd (level 19) of

  b"YGODECK1", u32 LE header length, header (JSON), tensor data

The header: "avg_copies" (the corpus's average copies per distinct card), "existence" and "counts"
(the models' configuration), "tensors" (in data order: "name", "shape", "dtype"), "checks" (test
cases). The dtypes, all little-endian:

  f16          float16 values
  qN           a matrix as N-bit integers in int8 values, then a float16 scale per row (the values:
               integer * scale; as eval_deck_models.py --bits stores them)
  u8, u16, u32 unsigned integers

The tensors: "codes" (u32, the vocabulary's card codes by index), "is_extra" (u8, by index), and the
models' weights under the PyTorch state dict's names with the prefix "existence." or "counts.".
Matrices are stored with --bits (--bits-for: other bits for some matrices), the other weights as
f16, the card table's feature ids as u16. The integer tensors come after the weights (zstd
compresses the weights better without them in between).

    ../.venv/bin/python export_deck_models.py [--existence deck_existence.pt] [--counts deck_counts.pt] \\
        [--bits 6] [--bits-for existence.card_table.parts.1.weight=4 ...] [--out deck_models.bin]
"""

import argparse
import json
import struct
from compression import zstd
from pathlib import Path

import numpy as np
import torch

from complete_deck import compute_avg_copies_per_card, load_model
from deck_common import quantize_rows
from train_deck_counts import CountModel
from train_deck_existence import ExistenceModel

HERE = Path(__file__).resolve().parent
MAGIC = b"YGODECK1"
TOP_LOGITS = 20


def encode(prefix, model, bits, bits_for, tensors):
    """Appends the model's weights to the tensors ((entry, bytes)), and sets the model's weights to
    the stored ones."""
    with torch.no_grad():
        for name, w in model.state_dict().items():
            if name == "shared_table":
                continue  # (the existence model's card table)
            full = f"{prefix}.{name}"
            if name.endswith("feat_ids"):
                dtype, raw = "u16", w.numpy().astype("<u2").tobytes()
            elif w.dim() == 2 and bits_for.get(full, bits) < 16:
                b = bits_for.get(full, bits)
                q, scale = quantize_rows(w, b)
                dtype, raw = f"q{b}", q.numpy().tobytes() + scale.numpy().tobytes()
                w.copy_(q.float() * scale.float())
            else:
                h = w.half()
                dtype, raw = "f16", h.numpy().tobytes()
                w.copy_(h.float())
            tensors.append(({"name": full, "shape": list(w.shape), "dtype": dtype}, raw))


def checks(ex, cnt, V):
    """Test cases: inputs and the stored models' outputs (the existence model's highest logits and
    its mean logit; the count logits)."""
    cases = [([], 55.0), ([(0, 3), (5, 1), (17, 2), (250, 1), (4000, 3)], 40.0),
             ([(i * 37 % V, 1 + i % 3) for i in range(20)], 60.0)]
    out = {"existence": [], "counts": []}
    with torch.no_grad():
        for cards, slots in cases:
            l = max(len(cards), 1)
            ids, counts, kpm = torch.zeros(1, l, dtype=torch.long), torch.zeros(1, l, dtype=torch.long), torch.ones(1, l, dtype=torch.bool)
            for i, (j, c) in enumerate(cards):
                ids[0, i], counts[0, i], kpm[0, i] = j, c, False
            logits = ex(ids, counts, kpm, torch.tensor([slots]))[0]
            top = logits.topk(TOP_LOGITS)
            out["existence"].append({"cards": cards, "slots": slots, "mean": logits.mean().item(),
                                     "top": [[i.item(), x.item()] for x, i in zip(top.values, top.indices)]})
        for cards, _ in cases[1:]:
            ids = torch.tensor([[j for j, _ in cards]])
            logits = cnt(ids, torch.zeros_like(ids, dtype=torch.bool))[0]
            out["counts"].append({"set": ids[0].tolist(), "logits": logits.flatten().tolist()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--existence", type=Path, default=HERE / "deck_existence.pt")
    ap.add_argument("--counts", type=Path, default=HERE / "deck_counts.pt")
    ap.add_argument("--bits", type=int, default=6, choices=[16, 8, 6, 4])
    ap.add_argument("--bits-for", action="append", default=[], metavar="NAME=BITS",
                    help="other bits for a matrix, e.g. existence.card_table.parts.1.weight=4")
    ap.add_argument("--out", type=Path, default=HERE / "deck_models.bin")
    args = ap.parse_args()
    bits_for = {n: int(b) for n, b in (x.split("=") for x in args.bits_for)}
    ex, ex_ckpt = load_model(ExistenceModel, args.existence, "cpu")
    cnt, cnt_ckpt = load_model(CountModel, args.counts, "cpu")
    if ex_ckpt["vocab"] != cnt_ckpt["vocab"] or ex_ckpt["is_extra"] != cnt_ckpt["is_extra"]:
        raise SystemExit("the deck models were trained on different vocabularies")
    if not hasattr(ex, "card_table") or not hasattr(cnt, "shared_table"):
        raise SystemExit("the export needs an existence model with a card table (--emb-buckets or --card-features) "
                         "and a counts model on it (train_deck_counts.py --shared-table)")
    ex.eval()
    cnt.eval()
    with torch.no_grad():
        if not torch.equal(cnt.shared_table, ex.card_table()):
            raise SystemExit(f"{args.counts} was not trained on the card table of {args.existence}")
    vocab = ex_ckpt["vocab"]
    codes = [None] * len(vocab)
    for code, i in vocab.items():
        codes[i] = code
    ex_cfg, cnt_cfg = ex_ckpt["config"], cnt_ckpt["config"]
    header = {
        "avg_copies": compute_avg_copies_per_card(HERE / "decks", vocab),
        "existence": {"d_model": ex_cfg["d_model"], "n_trunk_layers": ex_cfg["n_trunk_layers"],
                      "slots_scale": ex.slots_scale, "buckets": [list(s) for s in ex.card_table.sizes],
                      "features": ex.card_table.feat_emb is not None},
        "counts": {"d_model": cnt_cfg["d_model"], "n_trunk_layers": cnt_cfg["n_trunk_layers"],
                   "count_bias": cnt.count_bias is not None},
    }
    tensors = [({"name": "codes", "shape": [len(codes)], "dtype": "u32"}, np.array(codes, dtype="<u4").tobytes()),
               ({"name": "is_extra", "shape": [len(codes)], "dtype": "u8"},
                np.array(ex_ckpt["is_extra"], dtype=np.uint8).tobytes())]
    encode("existence", ex, args.bits, bits_for, tensors)
    encode("counts", cnt, args.bits, bits_for, tensors)
    tensors.sort(key=lambda t: t[0]["dtype"].startswith("u"))
    header["tensors"] = [entry for entry, _ in tensors]
    with torch.no_grad():
        cnt.shared_table.copy_(ex.card_table())
    header["checks"] = checks(ex, cnt, len(codes))
    head = json.dumps(header).encode()
    raw = MAGIC + struct.pack("<I", len(head)) + head + b"".join(data for _, data in tensors)
    args.out.write_bytes(zstd.compress(raw, level=19))
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.2f} MB (V={len(codes)}, avg copies {header['avg_copies']:.4f})")


if __name__ == "__main__":
    main()
