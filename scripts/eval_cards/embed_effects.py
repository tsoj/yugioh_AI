"""The card term's effect embeddings: every card's effect text alone (no name, no type line, no
stats), card and archetype names replaced by the pseudo-names of each of --seeds seeds
(card_text.Names), embedded with EmbeddingGemma 2. Cards without an effect (Normal Monsters that
are not Pendulum Monsters, tokens) get no text and the flag has_effect 0.

  scripts/eval_cards/embed_effects.py OUT.npz [--seeds 4] [--device cpu] [--codes CARDS.npz]

OUT.npz: codes (n), has_effect (n), emb (seeds, n, dim) float16 (zero rows without an effect).
--codes: only the codes of a cards_to_npz.py file (default: every card of the database).
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # (scripts/: common)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "book"))

from card_text import Names, _texts  # noqa: E402
from common import constants_gen as G  # noqa: E402
from common.carddb import default_db  # noqa: E402


def encode(model, texts, batch):
    """The texts' embeddings (normalized, float16), the longest first in batches."""
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.truncate_dim or model.get_sentence_embedding_dimension()), np.float16)
    with torch.no_grad():
        for s in range(0, len(texts), batch * 64):
            idx = order[s:s + batch * 64]
            e = model.encode([texts[i] for i in idx], prompt_name="document", batch_size=batch, convert_to_numpy=True,
                             show_progress_bar=False, normalize_embeddings=True)
            assert not np.isnan(e).any(), "NaN embeddings"
            out[idx] = e.astype(np.float16)
    return out


def has_effect(card) -> bool:
    t = card.type
    if t & G.TYPE_TOKEN:
        return False
    return not (t & G.TYPE_MONSTER and t & G.TYPE_NORMAL and not t & G.TYPE_PENDULUM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=20_000_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--codes", default=None)
    ap.add_argument("--encoder", default="google/embeddinggemma-2")
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    db = default_db()
    if a.codes:
        codes = [int(c) for c in np.load(a.codes)["codes"]]
    else:
        codes = sorted({row[0] for f in db.files for row in sqlite3.connect(f"file:{f}?mode=ro", uri=True).execute("SELECT id FROM datas")})
    flags = np.array([db.get(c) is not None and has_effect(db.get(c)) and bool(_texts().get(c, ("", ""))[1]) for c in codes])
    idx = np.flatnonzero(flags)
    print(f"{len(codes)} cards, {len(idx)} with an effect, {a.seeds} seeds, {a.device}", flush=True)

    from sentence_transformers import SentenceTransformer

    kw = {"attn_implementation": "eager", "torch_dtype": torch.bfloat16 if a.device != "cpu" else torch.float32}
    model = SentenceTransformer(a.encoder, device=a.device, model_kwargs=kw, truncate_dim=a.dim)
    model.max_seq_length = a.max_tokens
    emb = np.zeros((a.seeds, len(codes), a.dim), np.float16)
    t0 = time.time()
    for k in range(a.seeds):
        names = Names(a.first_seed + k)
        emb[k, idx] = encode(model, [names.effect(_texts()[codes[i]][1]) for i in idx], a.batch)
        print(f"  seed {k}: {time.time() - t0:.0f} s", flush=True)
    np.savez(a.out, codes=np.array(codes, np.int64), has_effect=flags.astype(np.int8), emb=emb)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
