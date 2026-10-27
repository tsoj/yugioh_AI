"""Embeds card texts with the effect model's encoder (scripts/book/train_effect_model.py's OUT/encoder,
sentence-transformers): every card of the decks in the given books or opening files, its text
rendered by scripts/book/card_text.py with one set of pseudo-names for all (the same name, the same
made-up word: links between cards kept, real names not). A card's embedding depends on its text
only, so cards no training game had get one too.

    .venv/bin/python scripts/book/embed_cards.py OUT.npz DECKS.jsonl [DECKS.jsonl ...] [--encoder runs/nn/effect1/encoder]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from card_text import Names  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("decks", nargs="+", help="JSONL rows with \"decks\": [{main, extra}, ...]")
    ap.add_argument("--encoder", default="runs/nn/effect1/encoder")
    ap.add_argument("--names-seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    codes = set()
    for path in a.decks:
        for line in open(path):
            for d in json.loads(line)["decks"]:
                codes.update(d["main"])
                codes.update(d["extra"])
    codes = sorted(codes)
    names = Names(a.names_seed)
    texts = [names.render(c) for c in codes]
    print(f"{len(codes)} cards, {sum(t == 'Unknown card.' for t in texts)} unknown")

    from sentence_transformers import SentenceTransformer

    # (eager attention: ROCm's SDPA gives NaN for padded batches)
    model = SentenceTransformer(a.encoder, device="cuda" if torch.cuda.is_available() else "cpu", model_kwargs={"attn_implementation": "eager"})
    with torch.no_grad():
        emb = model.encode(texts, batch_size=a.batch, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    assert not np.isnan(emb).any(), "NaN embeddings"
    np.savez(a.out, codes=np.array(codes, dtype=np.int64), emb=emb.astype(np.float32))
    print(f"wrote {a.out}: {emb.shape}")


if __name__ == "__main__":
    main()
