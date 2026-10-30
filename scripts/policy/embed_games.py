"""Card text embeddings per game for the policy data (bot-policy-data's games.jsonl): every game's
cards rendered with that game's own pseudo-names (scripts/book/card_text.py, Names(seed) with a
seed of the game's number), embedded by EmbeddingGemma 2 (frozen, the "document" prompt), so a
network cannot learn cards by their names, not even across games.

    .venv/bin/python scripts/policy/embed_games.py DATA_DIR [--salt S] [--dim D] [--batch B]

Writes DATA_DIR/emb_S.npy (rows: the games' codes, game after game in games.jsonl's order, float16)
and DATA_DIR/emb_S_index.npz (`game`, `start`: a game's first row; a code index i of game g is
row start[g] + i - 1). Another salt gives every game other names (augmentation).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # (scripts/: common)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "book"))

from card_text import Names  # noqa: E402

_MARK = re.compile("\x00([^\x00]*)\x00")


class _Marked(Names):
    """Renders a card with its name words marked, a template for any game's names."""

    def word(self, w: str) -> str:
        return f"\x00{w}\x00"


def templates(codes):
    m = _Marked(0)
    return {c: m.render(c) for c in codes}


def render(template: str, names: Names) -> str:
    return _MARK.sub(lambda x: names.word(x.group(1)), template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--salt", type=int, default=0)
    ap.add_argument("--dim", type=int, default=768, help="Matryoshka truncation (768, 512, 256, 128)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--encoder", default="google/embeddinggemma-2")
    ap.add_argument("--max-tokens", type=int, default=512, help="texts cut after this many tokens (few cards have more)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N games (a speed test)")
    a = ap.parse_args()

    games = [json.loads(line) for line in open(os.path.join(a.data, "games.jsonl"))]
    games.sort(key=lambda g: g["game"])
    if a.limit:
        games = games[:a.limit]
    tpl = templates({c for g in games for c in g["codes"]})
    starts, texts = [], []
    for g in games:
        starts.append(len(texts))
        names = Names(a.salt * 1_000_003 + g["game"])
        texts.extend(render(tpl[c], names) for c in g["codes"])
    print(f"{len(games)} games, {len(texts)} texts, {len(tpl)} cards", flush=True)

    from sentence_transformers import SentenceTransformer

    # (eager attention: ROCm's SDPA gives NaN for padded batches)
    model = SentenceTransformer(a.encoder, device="cuda", model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.bfloat16},
                                truncate_dim=a.dim)
    model.max_seq_length = a.max_tokens
    # (longest first: batches of similar lengths pad little)
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = np.lib.format.open_memmap(os.path.join(a.data, f"emb_{a.salt}.npy"), mode="w+", dtype=np.float16, shape=(len(texts), a.dim))
    t0 = time.time()
    chunk = a.batch * 64
    with torch.no_grad():
        for s in range(0, len(texts), chunk):
            idx = order[s:s + chunk]
            e = model.encode([texts[i] for i in idx], prompt_name="document", batch_size=a.batch, convert_to_numpy=True,
                             show_progress_bar=False, normalize_embeddings=True)
            assert not np.isnan(e).any(), "NaN embeddings"
            out[idx] = e.astype(np.float16)
            done = s + len(idx)
            print(f"  {done}/{len(texts)} texts, {done / (time.time() - t0):.0f}/s", flush=True)
    out.flush()
    np.savez(os.path.join(a.data, f"emb_{a.salt}_index.npz"), game=np.array([g["game"] for g in games]), start=np.array(starts))
    print(f"wrote emb_{a.salt}.npy: {out.shape}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
