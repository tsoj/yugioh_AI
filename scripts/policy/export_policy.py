"""Exports a train.py model for the policy bot (crates/policynet, burn on the CPU).

    .venv/bin/python scripts/policy/export_policy.py RUN_DIR/model.pt OUT.bin [--checks 8]

OUT.bin, zstd: b"YGOPOL01", u32 header length, the JSON header {"args", "consts", "tensors":
[{"name", "shape"}...], "checks"}, then the tensors (float16, state_dict order). A check is one
held-out decision of the model's data: the batch's rows (train.Batcher, one decision), the text
vectors its rows refer to (null: none) and the model's scores, which the Rust port must give.
"""

import argparse
import json
import os
import struct
import sys
from compression import zstd
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import LOCS, N_KINDS, PolicyData  # noqa: E402
from train import MAX_OPTIONS, MAX_TOKENS, N_PROMPTS, SYS_BUCKETS, Batcher, Policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--checks", type=int, default=8)
    ap.add_argument("--data", default=None, help="the checks' data (default: the model's; a small one with the same embeddings' names loads faster)")
    a = ap.parse_args()
    ck = torch.load(a.model, map_location="cpu")
    args = ck["args"]
    data_dir = a.data or args["data"]
    emb = np.load(os.path.join(data_dir, f"{args['emb']}.npy"), mmap_mode="r")
    model = Policy(emb.shape[1], args["dim"], args["layers"], args["heads"], strings=bool(args["strings"]))
    model.load_state_dict(ck["model"])
    model.eval()
    tensors, raw = weights(model)

    # checks: decisions of the data with their text vectors
    d = PolicyData(data_dir)
    ix = np.load(os.path.join(data_dir, f"{args['emb']}_index.npz"))
    start = {int(g): int(s) for g, s in zip(ix["game"], ix["start"])}
    semb, str_keys = None, None
    if args["strings"]:
        semb = np.load(os.path.join(data_dir, f"{args['strings']}.npy"), mmap_mode="r")
        str_keys = np.load(os.path.join(data_dir, f"{args['strings']}_index.npz"))["key"]
    batcher = Batcher(d, start, str_keys)
    rng = np.random.default_rng(0)
    checks = []
    for i in rng.choice(len(d.hdr), a.checks, replace=False):
        b = batcher(np.array([i]))
        vec = lambda rows, table: [None if r < 0 else table[r].astype(np.float32).round(6).tolist() for r in rows.reshape(-1).tolist()]
        texts = {k: vec(b[k].numpy(), emb) for k in ("tok_row", "ref_row", "opt_desc_row", "glob_rows")}
        if semb is not None:
            texts.update({k: vec(b[k].numpy(), semb) for k in ("opt_str_row", "glob_str_rows")})
        with torch.no_grad():
            sc = model(b, torch.from_numpy(np.asarray(emb)), None if semb is None else torch.from_numpy(np.asarray(semb)))
        checks.append({"hdr": b["hdr"][0].tolist(), "tok": b["tok"][0].tolist(), "opt": b["opt"][0].tolist(), "texts": texts,
                       "scores": sc[0].tolist()})
    write(a.out, args, int(emb.shape[1]), tensors, raw, checks)
    print(f"{a.out}: {Path(a.out).stat().st_size / 1e6:.1f} MB, {len(tensors)} tensors, {len(checks)} checks")


def weights(model) -> tuple[list[dict], list[bytes]]:
    """The model's tensors as float16 (the model keeps the rounded weights: it computes as the
    export will)."""
    tensors, raw = [], []
    with torch.no_grad():
        for name, w in model.state_dict().items():
            h = w.half()
            w.copy_(h.float())
            tensors.append({"name": name, "shape": list(w.shape)})
            raw.append(h.cpu().numpy().astype("<f2").tobytes())
    return tensors, raw


def write(out, args: dict, emb_dim: int, tensors: list[dict], raw: list[bytes], checks: list[dict]):
    consts = {"n_prompts": N_PROMPTS, "sys_buckets": SYS_BUCKETS, "max_tokens": MAX_TOKENS, "max_options": MAX_OPTIONS,
              "locs": len(LOCS), "kinds": N_KINDS, "emb_dim": emb_dim}
    head = json.dumps({"args": args, "consts": consts, "tensors": tensors, "checks": checks}).encode()
    blob = b"YGOPOL01" + struct.pack("<I", len(head)) + head + b"".join(raw)
    Path(out).write_bytes(zstd.compress(blob, level=10))


if __name__ == "__main__":
    main()
