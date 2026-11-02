"""The policy bot's text embeddings (bot::imitation): every card of the database and every card
string, under each of --seeds pseudo-name seeds (a game draws one: names consistent within it,
other in each), as embed_games.py and embed_strings.py embed a game's.

    .venv/bin/python scripts/policy/embed_tables.py OUT.bin [--seeds 8] [--first-seed 10000000]

OUT.bin: b"YGOPTAB1", u32 header length, the JSON header {"dim", "seeds", "codes", "strings":
[[code, string number]...]}, then per seed the cards' rows (codes' order) and the strings' rows,
float16.
"""

import argparse
import json
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # (scripts/: common)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "book"))

from card_text import Names, _texts  # noqa: E402
from common.carddb import default_db  # noqa: E402
from embed_games import render, templates  # noqa: E402
from embed_strings import card_strings  # noqa: E402


def encode(model, texts, batch):
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.get_sentence_embedding_dimension() if model.truncate_dim is None else model.truncate_dim), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(texts), batch * 64):
            idx = order[s:s + batch * 64]
            e = model.encode([texts[i] for i in idx], prompt_name="document", batch_size=batch, convert_to_numpy=True,
                             show_progress_bar=False, normalize_embeddings=True)
            assert not np.isnan(e).any(), "NaN embeddings"
            out[idx] = e.astype(np.float16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--first-seed", type=int, default=10_000_000, help="(the training data's games used seeds below this)")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--encoder", default="google/embeddinggemma-2")
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    db = default_db()
    codes = sorted({row[0] for f in db.files for row in sqlite3.connect(f"file:{f}?mode=ro", uri=True).execute("SELECT id FROM datas")})
    tpl = templates(codes)
    strings = card_strings()
    keys = [(c, i) for c in codes for i, s in enumerate(strings.get(c, [])) if s]
    print(f"{len(codes)} cards, {len(keys)} strings, {a.seeds} seeds", flush=True)

    from sentence_transformers import SentenceTransformer

    # (eager attention: ROCm's SDPA gives NaN for padded batches)
    model = SentenceTransformer(a.encoder, device="cuda", model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.bfloat16},
                                truncate_dim=a.dim)
    model.max_seq_length = a.max_tokens
    seeds = list(range(a.first_seed, a.first_seed + a.seeds))
    header = {"dim": a.dim, "seeds": seeds, "codes": codes, "strings": keys}
    head = json.dumps(header).encode()
    t0 = time.time()
    with open(a.out, "wb") as f:
        f.write(b"YGOPTAB1" + struct.pack("<I", len(head)) + head)
        for s in seeds:
            names = Names(s)
            cards = encode(model, [render(tpl[c], names) for c in codes], a.batch)
            texts = [f"{names.name(_texts().get(c, ('', ''))[0])}: {names.effect(strings[c][i])}" for c, i in keys]
            strs = encode(model, texts, 2 * a.batch)
            f.write(cards.astype("<f2").tobytes() + strs.astype("<f2").tobytes())
            print(f"  seed {s}: {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
