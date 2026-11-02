"""Imitation of the bot's choices (bot-policy-data + embed_games.py): a transformer over the
decider's view (a global token, a token per card, a token per option) scores the options; trained
on the chosen option(s) (softmax cross-entropy; a selection of several cards: the mean over its
cards). Held out: the deck pairs whose hash falls in --val-share (no held-out deck pair is
trained on), and their games' card names were never seen either (embed_games.py: names per game).

    .venv/bin/python scripts/policy/train.py DATA_DIR OUT_DIR [--layers 4] [--dim 256] ...

Reports, held out, the top-1 accuracy (a selection of k cards: its k best are the chosen ones)
per prompt type, for searched decisions (the search's move) and all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import H, LOCS, N_KINDS, O, S_CARD, T, PolicyData  # noqa: E402
from embed_strings import string_key  # noqa: E402

# (categorical sizes)
N_PROMPTS = 17
SYS_BUCKETS = 1024  # system strings (hints, descs), by number
MAX_TOKENS = 192
MAX_OPTIONS = 64


def deck_pair_key(g: dict) -> str:
    decks = sorted(json.dumps(sorted(d["main"]) + sorted(d["extra"])) for d in g["decks"])
    return hashlib.blake2b("|".join(decks).encode(), digest_size=8).hexdigest()


class Batcher:
    """Padded batches of decisions as tensors (numpy gathers; the text embeddings stay rows)."""

    def __init__(self, data: PolicyData, emb_index: dict[int, int], str_keys: np.ndarray | None = None):
        self.d = data
        self.start = emb_index  # game -> first embedding row
        self.str_keys = str_keys  # embed_strings.py's sorted keys (their rows), or None

    def str_rows(self, game, code, string):
        """The rows of card strings (embed_strings.py), -1 for none."""
        key = string_key(game, code, string)
        i = np.searchsorted(self.str_keys, key).clip(0, len(self.str_keys) - 1)
        return np.where((self.str_keys[i] == key) & (code > 0) & (string < S_CARD), i, -1)

    def __call__(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        d = self.d
        h = d.hdr[idx]
        b = len(idx)
        n_tok = np.minimum(h[:, H["N_TOK"]], MAX_TOKENS)
        n_opt = np.minimum(h[:, H["N_OPT"]], MAX_OPTIONS)
        lt, lo = int(n_tok.max()), int(n_opt.max())
        base = np.array([self.start[g] - 1 for g in h[:, H["GAME"]]], dtype=np.int64)  # code index i -> row base + i

        def ranges(starts, counts):
            rows = np.repeat(starts, counts) + (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts))
            pos = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
            which = np.repeat(np.arange(len(counts)), counts)
            return rows, which, pos

        rows, which, pos = ranges(h[:, H["TOK_START"]], n_tok)
        tok = np.zeros((b, lt, len(T)), dtype=np.int64)
        tok[which, pos] = d.tok[rows]
        tok[:, :, T["REF"]][tok[:, :, T["REF"]] >= MAX_TOKENS] = -1
        tok_mask = np.zeros((b, lt), dtype=bool)
        tok_mask[which, pos] = True
        # text embedding rows: -1 for none
        code = tok[:, :, T["CODE"]]
        tok_row = np.where(code > 0, base[:, None] + code, -1)
        ref = tok[:, :, T["REF"]]
        ref_code = np.where(ref >= 0, np.take_along_axis(code, np.maximum(ref, 0), axis=1), 0)
        ref_row = np.where(ref_code > 0, base[:, None] + ref_code, -1)

        rows, which, pos = ranges(h[:, H["OPT_START"]], n_opt)
        opt = np.zeros((b, lo, len(O)), dtype=np.int64)
        opt[which, pos] = d.opt[rows]
        opt_mask = np.zeros((b, lo), dtype=bool)
        opt_mask[which, pos] = True
        opt[:, :, O["TOKEN"]][opt[:, :, O["TOKEN"]] >= lt] = -1
        dcode = opt[:, :, O["DESC_CODE"]]
        opt_desc_row = np.where(dcode > 0, base[:, None] + dcode, -1)

        def hrow(col):
            c = h[:, H[col]]
            return np.where(c > 0, base + c, -1)

        glob_rows = np.stack([hrow("HINT_CODE"), hrow("DESC_CODE"), hrow("CODE"), hrow("PREV_CODE")], axis=1)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x))
        out = {
            "hdr": t(h), "tok": t(tok), "tok_mask": t(tok_mask), "tok_row": t(tok_row), "ref_row": t(ref_row),
            "opt": t(opt), "opt_mask": t(opt_mask), "opt_desc_row": t(opt_desc_row), "glob_rows": t(glob_rows),
        }
        if self.str_keys is not None:
            g = h[:, H["GAME"]]
            out["opt_str_row"] = t(self.str_rows(g[:, None], dcode, opt[:, :, O["DESC_STR"]]))
            out["glob_str_rows"] = t(np.stack([self.str_rows(g, h[:, H[c]], h[:, H[s]]) for c, s in (("HINT_CODE", "HINT_STR"), ("DESC_CODE", "DESC_STR"))], 1))
        return out


def bucket(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.clamp(0, n - 1)


class Policy(nn.Module):
    def __init__(self, emb_dim: int, dim: int = 256, layers: int = 4, heads: int = 8, ff: int = 4, dropout: float = 0.0, strings: bool = False):
        super().__init__()
        self.dim = dim
        if strings:
            self.str_opt = nn.Linear(emb_dim, dim)
            self.str_glob = nn.ModuleList(nn.Linear(emb_dim, dim) for _ in range(2))
        self.text = nn.Linear(emb_dim, dim)
        self.text_ref = nn.Linear(emb_dim, dim)
        self.text_glob = nn.ModuleList(nn.Linear(emb_dim, dim) for _ in range(4))
        self.text_desc = nn.Linear(emb_dim, dim)
        self.unknown = nn.Parameter(torch.zeros(dim))
        E = lambda n: nn.Embedding(n, dim)
        # tokens
        self.con, self.owner, self.locseq, self.pos = E(2), E(3), E(len(LOCS) * 32), E(8)
        self.level, self.scale, self.counters, self.overlays, self.count = E(16), E(16), E(8), E(8), E(8)
        self.status = nn.Linear(8, dim)
        self.stats = nn.Linear(8, dim)
        self.has_ref = E(2)
        # global
        self.g_kind = nn.Parameter(torch.zeros(dim))
        self.prompt, self.tp, self.phase, self.turn = E(N_PROMPTS), E(2), E(11), E(32)
        self.minmax, self.flags = E(2 * 8), E(8)
        self.sys_hint, self.sys_desc, self.str_hint, self.str_desc = E(SYS_BUCKETS), E(SYS_BUCKETS), E(32), E(32)
        self.prev_kind = E(N_KINDS + 1)
        self.g_num = nn.Linear(5, dim)
        # options
        self.o_kind = E(N_KINDS)
        self.o_aux = E(N_KINDS * 64)
        self.o_sys, self.o_str = E(SYS_BUCKETS), E(32)
        self.o_marker = nn.Parameter(torch.zeros(dim))
        layer = nn.TransformerEncoderLayer(dim, heads, ff * dim, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, b: dict[str, torch.Tensor], emb: torch.Tensor, semb: torch.Tensor | None = None) -> torch.Tensor:
        """Option scores (batch, options); padding -inf. `semb`: the card strings' embeddings
        (a model made with strings)."""
        dev = emb.device

        def text(rows, lin, table=emb):
            # rows -1: zero vectors
            e = table[rows.clamp(min=0)].float()
            return lin(e) * (rows >= 0).unsqueeze(-1)

        tok, h, opt = b["tok"], b["hdr"], b["opt"]
        c = lambda col: tok[..., T[col]]
        x = text(b["tok_row"], self.text) + self.unknown * (b["tok_row"] < 0).unsqueeze(-1)
        x = x + self.con(c("CON").clamp(0, 1)) + self.owner(c("OWNER").clamp(0, 2))
        x = x + self.locseq(c("LOC").clamp(0, len(LOCS) - 1) * 32 + c("SEQ").clamp(0, 31)) + self.pos(c("POS").clamp(0, 7))
        x = x + self.level(bucket(c("LEVEL"), 16)) + self.scale(bucket(c("SCALE"), 16))
        x = x + self.counters(bucket(c("COUNTERS"), 8)) + self.overlays(bucket(c("OVERLAYS"), 8)) + self.count(bucket(c("COUNT"), 8))
        bits = ((c("STATUS").unsqueeze(-1) >> torch.arange(8, device=dev)) & 1).float()
        x = x + self.status(bits)
        st = torch.stack([c(k) for k in ("ATK", "DEF", "BASE_ATK", "BASE_DEF")], -1).float()
        known = (st >= 0).float()
        x = x + self.stats(torch.cat([st.clamp(min=0) / 3000.0, known], -1))
        x = x + text(b["ref_row"], self.text_ref) + self.has_ref((c("REF") >= 0).long())

        g = self.g_kind + self.prompt(h[:, H["PROMPT"]].clamp(0, N_PROMPTS - 1)) + self.tp(h[:, H["TURN_PLAYER"]].clamp(0, 1))
        phase = torch.log2(h[:, H["PHASE"]].clamp(min=1).float()).long().clamp(0, 10)
        g = g + self.phase(phase) + self.turn(bucket(h[:, H["TURN"]], 32))
        g = g + self.minmax(bucket(h[:, H["MIN"]], 8)) + self.minmax(8 + bucket(h[:, H["MAX"]], 8)) + self.flags(bucket(h[:, H["FLAGS"]], 8))
        hint_sys = (h[:, H["HINT_CODE"]] == 0) & (h[:, H["HINT_STR"]] > 0)
        g = g + self.sys_hint(h[:, H["HINT_STR"]] % SYS_BUCKETS) * hint_sys.unsqueeze(-1) + self.str_hint(bucket(h[:, H["HINT_STR"]], 32)) * (~hint_sys).unsqueeze(-1)
        desc_sys = (h[:, H["DESC_CODE"]] == 0) & (h[:, H["DESC_STR"]] > 0)
        g = g + self.sys_desc(h[:, H["DESC_STR"]] % SYS_BUCKETS) * desc_sys.unsqueeze(-1) + self.str_desc(bucket(h[:, H["DESC_STR"]], 32)) * (~desc_sys).unsqueeze(-1)
        g = g + self.prev_kind(h[:, H["PREV_KIND"]] + 1)
        num = torch.stack([h[:, H["LP_ME"]] / 8000.0, h[:, H["LP_OPP"]] / 8000.0, h[:, H["DECK_ME"]] / 40.0, h[:, H["DECK_OPP"]] / 40.0, h[:, H["EXTRA_OPP"]] / 15.0], -1).float()
        g = g + self.g_num(num)
        for i, lin in enumerate(self.text_glob):
            g = g + text(b["glob_rows"][:, i], lin)
        if semb is not None:
            for i, lin in enumerate(self.str_glob):
                g = g + text(b["glob_str_rows"][:, i], lin, semb)

        o = lambda col: opt[..., O[col]]
        kind = o("KIND").clamp(0, N_KINDS - 1)
        ref = o("TOKEN")
        ox = self.o_marker + self.o_kind(kind) + self.o_aux(kind * 64 + o("AUX").clamp(0, 63))
        osys = (o("DESC_CODE") == 0) & (o("DESC_STR") > 0)
        ox = ox + self.o_sys(o("DESC_STR") % SYS_BUCKETS) * osys.unsqueeze(-1) + self.o_str(bucket(o("DESC_STR"), 32)) * (~osys).unsqueeze(-1)
        ox = ox + text(b["opt_desc_row"], self.text_desc)
        if semb is not None:
            ox = ox + text(b["opt_str_row"], self.str_opt, semb)
        ref_in = torch.gather(x, 1, ref.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.dim)) * (ref >= 0).unsqueeze(-1)
        ox = ox + ref_in

        seq = torch.cat([g.unsqueeze(1), x, ox], 1)
        mask = torch.cat([torch.ones_like(b["tok_mask"][:, :1]), b["tok_mask"], b["opt_mask"]], 1)
        y = self.norm(self.trunk(seq, src_key_padding_mask=~mask))
        lt = x.shape[1]
        ty, oy = y[:, 1:1 + lt], y[:, 1 + lt:]
        ref_out = torch.gather(ty, 1, ref.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.dim)) * (ref >= 0).unsqueeze(-1)
        s = self.head(torch.cat([oy, ref_out], -1)).squeeze(-1)
        return s.masked_fill(~b["opt_mask"], float("-inf"))


def losses_and_hits(scores: torch.Tensor, b: dict[str, torch.Tensor]):
    chosen = (b["opt"][..., O["CHOSEN"]] > 0) & b["opt_mask"]
    k = chosen.sum(1).clamp(min=1)
    logp = F.log_softmax(scores.float(), -1)
    loss = -(logp.masked_fill(~chosen, 0).sum(1) / k)
    # hit: the k best options are the chosen ones
    order = scores.argsort(-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
    hit = ((ranks < k.unsqueeze(1)) == chosen).masked_fill(~b["opt_mask"], True).all(1)
    return loss, hit


def evaluate(model, batcher, idx, emb, semb, dev, batch, prompts):
    model.eval()
    tot = {}
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            b = {k: v.to(dev, non_blocking=True) for k, v in batcher(idx[s:s + batch]).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc = model(b, emb, semb)
            loss, hit = losses_and_hits(sc, b)
            h = b["hdr"]
            for key, sel in [("all", torch.ones_like(hit))] + [(p, h[:, H["PROMPT"]] == i) for i, p in enumerate(prompts)]:
                for tag, m in [("", sel), ("searched ", sel & (h[:, H["DEPTH"]] >= 0))]:
                    n = int(m.sum())
                    if n:
                        a = tot.setdefault(tag + key, [0, 0.0, 0.0])
                        a[0] += n
                        a[1] += float(hit[m].float().sum())
                        a[2] += float(loss[m].sum())
    model.train()
    return {k: (v[0], v[1] / v[0], v[2] / v[0]) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--emb", default="emb_0")
    ap.add_argument("--strings", default="", help="card string embeddings (embed_strings.py), e.g. str_0")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--val-share", type=float, default=0.05)
    ap.add_argument("--val-max", type=int, default=100000)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    t0 = time.time()
    data = PolicyData(a.data)
    ix = np.load(os.path.join(a.data, f"{a.emb}_index.npz"))
    start = dict(zip(ix["game"].tolist(), ix["start"].tolist()))
    have = np.array([g in start for g in data.hdr[:, H["GAME"]]])
    if a.max_games:
        have &= data.hdr[:, H["GAME"]] < a.max_games
    val_pairs = {g for g, x in data.games.items() if int(deck_pair_key(x), 16) % 10000 < a.val_share * 10000}
    is_val = np.array([g in val_pairs for g in data.hdr[:, H["GAME"]]])
    train_idx = np.where(have & ~is_val)[0]
    val_idx = np.where(have & is_val)[0]
    val_idx = val_idx[rng.permutation(len(val_idx))[:a.val_max]]
    print(f"{len(data.hdr)} decisions; train {len(train_idx)}, val {len(val_idx)} ({len(val_pairs)} of {len(data.games)} games' pairs held out); "
          f"load {time.time() - t0:.0f} s", flush=True)

    dev = torch.device("cuda")
    emb = torch.from_numpy(np.load(os.path.join(a.data, f"{a.emb}.npy"))).to(dev)
    semb, str_keys = None, None
    if a.strings:
        semb = torch.from_numpy(np.load(os.path.join(a.data, f"{a.strings}.npy"))).to(dev)
        str_keys = np.load(os.path.join(a.data, f"{a.strings}_index.npz"))["key"]
    batcher = Batcher(data, start, str_keys)
    model = Policy(emb.shape[1], a.dim, a.layers, a.heads, dropout=a.dropout, strings=semb is not None).to(dev)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    # (batches built by worker threads ahead of the GPU)
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(6)
    perm = rng.permutation(train_idx)
    pos = 0

    def next_idx():
        nonlocal perm, pos
        if pos + a.batch > len(perm):
            perm, pos = rng.permutation(train_idx), 0
        pos += a.batch
        return perm[pos - a.batch:pos]

    pending = [pool.submit(batcher, next_idx()) for _ in range(12)]
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    run_loss, run_hit, t1 = 0.0, 0.0, time.time()
    for step in range(1, a.steps + 1):
        b = pending.pop(0).result()
        pending.append(pool.submit(batcher, next_idx()))
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sc = model(b, emb, semb)
        loss, hit = losses_and_hits(sc, b)
        loss = loss.mean()
        if not torch.isfinite(loss):
            raise SystemExit(f"step {step}: loss {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += loss.item()
        run_hit += hit.float().mean().item()
        if step % 200 == 0:
            print(f"step {step}: loss {run_loss / 200:.4f} top1 {run_hit / 200:.4f} lr {sched.get_last_lr()[0]:.2e} {200 * a.batch / (time.time() - t1):.0f} dec/s", flush=True)
            run_loss, run_hit, t1 = 0.0, 0.0, time.time()
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(model, batcher, val_idx, emb, semb, dev, 512, data.prompts)
            print(f"== step {step} val: " + ", ".join(f"{k} {v[1]:.4f}" for k, v in r.items() if k in ("all", "searched all")), flush=True)
            for k, v in sorted(r.items(), key=lambda kv: -kv[1][0]):
                if not k.startswith("searched") and v[0] >= 200:
                    print(f"   {k:22s} n {v[0]:7d} top1 {v[1]:.4f} loss {v[2]:.4f}")
            log.write(json.dumps({"step": step, "val": r, "args": vars(a)}) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "args": vars(a)}, os.path.join(a.out, "model.pt"))
    print(f"done, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
