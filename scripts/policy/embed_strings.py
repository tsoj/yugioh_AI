"""Embeddings of the card strings the policy data refers to (an option's effect, a selection hint,
a yes/no question: a code index and a string number below S_CARD), per game with that game's
pseudo-names like embed_games.py: "<the card's pseudo-name>: <its string>".

    .venv/bin/python scripts/policy/embed_strings.py DATA_DIR [--salt S] [--dim D] [--batch B]

Writes DATA_DIR/str_S.npy (float16) and DATA_DIR/str_S_index.npz (`key`, sorted: string_key(game,
code index, string number); row i of str_S.npy is key[i]'s).
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # (scripts/: common)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "book"))

from card_text import Names, _texts  # noqa: E402
from common.carddb import default_db  # noqa: E402  (the database files)
from data import H, O, S_CARD, PolicyData  # noqa: E402


def string_key(game, code, string):
    """A (game, code index, string number) as one sortable integer (numpy arrays or ints)."""
    return (np.asarray(game, dtype=np.int64) * 4096 + code) * 32 + string


def card_strings() -> dict[int, list[str]]:
    """code -> its 16 strings (the first database file wins, as in card_text)."""
    out: dict[int, list[str]] = {}
    for f in default_db().files:
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        try:
            cols = ", ".join(f"str{i}" for i in range(1, 17))
            for row in con.execute(f"SELECT id, {cols} FROM texts"):
                out.setdefault(row[0], [s or "" for s in row[1:]])
        finally:
            con.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--salt", type=int, default=0)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--encoder", default="google/embeddinggemma-2")
    a = ap.parse_args()

    d = PolicyData(a.data)
    h, o = d.hdr, d.opt
    game_of_opt = np.repeat(h[:, H["GAME"]], h[:, H["N_OPT"]])
    keys = [string_key(game_of_opt, o[:, O["DESC_CODE"]], o[:, O["DESC_STR"]])[(o[:, O["DESC_CODE"]] > 0) & (o[:, O["DESC_STR"]] < S_CARD)]]
    for code, string in [("HINT_CODE", "HINT_STR"), ("DESC_CODE", "DESC_STR")]:
        m = (h[:, H[code]] > 0) & (h[:, H[string]] < S_CARD)
        keys.append(string_key(h[m, H["GAME"]], h[m, H[code]], h[m, H[string]]))
    keys = np.unique(np.concatenate(keys))
    strings = card_strings()
    texts = []
    names_of: dict[int, Names] = {}
    for k in keys.tolist():
        game, rest = divmod(k, 4096 * 32)
        code_index, string = divmod(rest, 32)
        code = d.games[game]["codes"][code_index - 1]
        names = names_of.get(game)
        if names is None:
            names = names_of[game] = Names(a.salt * 1_000_003 + game)
        name = names.name(_texts().get(code, ("", ""))[0])
        s = strings.get(code, [""] * 16)[string]
        texts.append(f"{name}: {names.effect(s) if s else 'an effect'}")
    print(f"{len(keys)} strings of {len(names_of)} games; e.g. {texts[:3]}", flush=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(a.encoder, device="cuda", model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.bfloat16},
                                truncate_dim=a.dim)
    order = np.argsort([-len(t) for t in texts], kind="stable")
    out = np.lib.format.open_memmap(os.path.join(a.data, f"str_{a.salt}.npy"), mode="w+", dtype=np.float16, shape=(len(texts), a.dim))
    t0 = time.time()
    chunk = a.batch * 64
    with torch.no_grad():
        for s in range(0, len(texts), chunk):
            idx = order[s:s + chunk]
            e = model.encode([texts[i] for i in idx], prompt_name="document", batch_size=a.batch, convert_to_numpy=True,
                             show_progress_bar=False, normalize_embeddings=True)
            assert not np.isnan(e).any(), "NaN embeddings"
            out[idx] = e.astype(np.float16)
            print(f"  {s + len(idx)}/{len(texts)} strings, {(s + len(idx)) / (time.time() - t0):.0f}/s", flush=True)
    out.flush()
    np.savez(os.path.join(a.data, f"str_{a.salt}_index.npz"), key=keys)
    print(f"wrote str_{a.salt}.npy: {out.shape}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
