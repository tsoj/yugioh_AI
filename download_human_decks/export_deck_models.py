#!/usr/bin/env python3
"""Exports the deck models for the Rust side (crates/deckmodels, burn-onnx):

  onnx_export/deck_existence.onnx   ExistenceModel: (card_ids, card_counts, key_padding_mask, slots) -> logits (1, V)
  onnx_export/deck_counts.onnx      CountModel: (card_ids, key_padding_mask) -> logits (1, L, 3)
  onnx_export/deck_models.json      the vocabulary (card codes in index order), is_extra per index, and the
                             average copies per unique card of the corpus (complete_deck.py's)

The number of tokens L is a dynamic dimension; the batch is 1. Run after training:
  ../.venv/bin/python export_deck_models.py [--out onnx_export]
"""

import argparse
import json
from pathlib import Path

import torch

from complete_deck import compute_avg_copies_per_card, load_model
from train_deck_counts import CountModel
from train_deck_existence import ExistenceModel

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "onnx_export")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ex, ex_ckpt = load_model(ExistenceModel, HERE / "deck_existence.pt", "cpu")
    cnt, cnt_ckpt = load_model(CountModel, HERE / "deck_counts.pt", "cpu")
    if ex_ckpt["vocab"] != cnt_ckpt["vocab"] or ex_ckpt["is_extra"] != cnt_ckpt["is_extra"]:
        raise SystemExit("the deck models were trained on different vocabularies")
    ex.eval()
    cnt.eval()
    L = 7
    ids = torch.arange(1, L + 1, dtype=torch.long).unsqueeze(0)
    counts = torch.ones(1, L, dtype=torch.long)
    kpm = torch.zeros(1, L, dtype=torch.bool)
    slots = torch.tensor([55.0])
    with torch.no_grad():
        torch.onnx.export(ex, (ids, counts, kpm, slots), args.out / "deck_existence.onnx",
                          input_names=["card_ids", "card_counts", "key_padding_mask", "slots"],
                          output_names=["logits"], opset_version=args.opset, dynamo=False,
                          dynamic_axes={"card_ids": {1: "L"}, "card_counts": {1: "L"}, "key_padding_mask": {1: "L"}})
        torch.onnx.export(cnt, (ids, kpm), args.out / "deck_counts.onnx",
                          input_names=["card_ids", "key_padding_mask"], output_names=["logits"],
                          opset_version=args.opset, dynamo=False,
                          dynamic_axes={"card_ids": {1: "L"}, "key_padding_mask": {1: "L"}, "logits": {1: "L"}})
    vocab = ex_ckpt["vocab"]
    codes = [None] * len(vocab)
    for code, i in vocab.items():
        codes[i] = code
    avg = compute_avg_copies_per_card(HERE / "decks", vocab)
    (args.out / "deck_models.json").write_text(json.dumps(
        {"codes": codes, "is_extra": [bool(x) for x in ex_ckpt["is_extra"]], "avg_copies": avg}))
    print(f"exported to {args.out} (V={len(codes)}, avg copies {avg:.4f})")


if __name__ == "__main__":
    main()
