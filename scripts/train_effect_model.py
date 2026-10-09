#!/usr/bin/env python3
"""The effect model: a text encoder for card texts (EmbeddingGemma 2, fine-tuned),
trained by predicting what happens after a decision (build_effect_data.py).

  scripts/train_effect_model.py runs/data/effects1.npz -o runs/nn/effect1 [--steps 20000]
      [--freeze-encoder] [--train-layers N] [--real-names] [--encoding v1|v2|v3]

Each card of the position is a token: its text's embedding (name, type line,
effect: ygoai/nn/card_text.py, with per-game pseudo-names unless --real-names)
plus its state (own/opponent, zone, sequence, position, ATK/DEF/level, materials,
counters, public, whether the decision names it); one more token holds LP, turn,
phase, chain and the decision (prompt type, kind, yes/no or option). A
transformer over them; heads (discarded later):
  per card: moves, where to, position change, ATK/DEF change, which messages name it
  global: which message types occur, the next prompt (type, same player), LP changes
Card position (--encoding): v1 a slot embedding per sequence number; v2 board
coordinates instead (row and column as the deciding player sees the field: the
opponent's zones mirrored, Extra Monster Zones in columns 1 and 3, materials at
their monster's place, cards off the field none); v3 also the chain link a card
activated and where the card it equips or targets is (needs build_effect_data.py
data with the chain/equip/target columns).
A batch: a few games, many decisions each (the texts are encoded once per game:
the pseudo-names differ between games). The encoder is what it is for: saved
to OUT/encoder (sentence-transformers), the rest to OUT/heads.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ygoai.nn.card_text import Names  # noqa: E402

ENCODER = "google/embeddinggemma-2"
N_LOC = 10  # (deck, hand, mzone, szone, grave, removed, extra, overlay, gone, other)
CHUNK = 32  # (texts per checkpointed encoder pass)


class Data:
    def __init__(self, path: Path, val: float):
        d = np.load(path)
        self.d = {k: d[k] for k in d.files}
        self.f = {n: i for i, n in enumerate(str(x) for x in self.d["fields"])}
        self.off = self.d["row_off"]
        self.codes = self.d["codes"]
        game = self.d["game"]
        n_games = int(game.max()) + 1
        cut = int(n_games * (1 - val))
        order = np.argsort(game, kind="stable")
        starts = np.searchsorted(game[order], np.arange(n_games + 1))
        self.by_game = [order[starts[g]:starts[g + 1]] for g in range(n_games)]
        self.train_games = [g for g in range(cut) if len(self.by_game[g])]
        self.val_games = [g for g in range(cut, n_games) if len(self.by_game[g])]
        self.n_msg = len(self.d["msg_types"])
        self.n_ref = len(self.d["ref_types"])
        self.n_prompt = len(self.d["prompts"])

    def sample(self, rng, games, k: int) -> np.ndarray:
        out = []
        for g in games:
            ids = self.by_game[g]
            out.append(ids if len(ids) <= k else rng.choice(ids, k, replace=False))
        return np.concatenate(out)

    def batch(self, ids: np.ndarray, device) -> dict:
        d, f = self.d, self.f
        n = (self.off[ids + 1] - self.off[ids]).astype(np.int64)
        B, N = len(ids), int(n.max())
        rows = np.zeros((B, N, d["rows"].shape[1]), dtype=np.int64)
        mask = np.zeros((B, N), dtype=bool)
        for i, s in enumerate(ids):
            r = d["rows"][self.off[s]:self.off[s + 1]]
            rows[i, :len(r)] = r
            mask[i, :len(r)] = True
        # (texts: one per game and card)
        seeds = d["game_seed"][ids]
        keys = {}
        tidx = np.zeros((B, N), dtype=np.int64)
        for i in range(B):
            for j in range(int(n[i])):
                key = (int(seeds[i]), int(rows[i, j, f["code"]]))
                tidx[i, j] = keys.setdefault(key, len(keys))
        player = d["player"][ids].astype(np.int64)
        act_ids = d["act_ids"][ids]
        is_act = np.zeros((B, N), dtype=np.int64)
        for i in range(B):
            for a in act_ids[i]:
                if a >= 0:
                    is_act[i] |= rows[i, :, f["id"]] == a
        cur_loc = _loc_class(rows[:, :, f["loc"]])
        own = rows[:, :, f["con"]] == player[:, None]
        brow, bcol = _board(rows[:, :, f["loc"]], rows[:, :, f["seq"]], own)
        # (relations: the chain link a card activated; where the card it equips / targets is)
        rel = {}
        if "chain" in f:
            rel["chain"] = np.clip(rows[:, :, f["chain"]], 0, 8)
            for name in ("equip", "target"):
                rr, rc = np.full((B, N), 5, dtype=np.int64), np.full((B, N), 7, dtype=np.int64)
                for i, j in zip(*np.nonzero(rows[:, :, f[name]] >= 0)):
                    k = np.flatnonzero(rows[i, :n[i], f["id"]] == rows[i, j, f[name]])
                    if len(k):
                        rr[i, j], rc[i, j] = brow[i, k[0]], bcol[i, k[0]]
                rel[name] = (rr, rc)
        t_loc = rows[:, :, f["t_loc"]]
        msgs = d["msgs"][ids].astype(np.int64)
        t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=device)  # noqa: E731
        sc = d["scalars"][ids].astype(np.float64)
        own_lp = np.where(player == 0, sc[:, 0], sc[:, 1])
        opp_lp = np.where(player == 0, sc[:, 1], sc[:, 0])
        dlp = d["dlp"][ids].astype(np.float64)
        own_dlp = np.where(player == 0, dlp[:, 0], dlp[:, 1])
        opp_dlp = np.where(player == 0, dlp[:, 1], dlp[:, 0])
        return {
            "keys": list(keys), "tidx": t(tidx), "mask": t(mask, torch.bool),
            "own": t(own), "loc": t(cur_loc), "seq": t(np.clip(rows[:, :, f["seq"]], 0, 63)),
            "brow": t(brow), "bcol": t(bcol),
            **({"chain": t(rel["chain"]), "eq_row": t(rel["equip"][0]), "eq_col": t(rel["equip"][1]),
                "tg_row": t(rel["target"][0]), "tg_col": t(rel["target"][1])} if rel else {}),
            "pos": t(rows[:, :, f["pos"]] & 0xF), "public": t(rows[:, :, f["public"]] != 0), "is_act": t(is_act),
            "num": t(np.stack([rows[:, :, f["atk"]] / 1000, rows[:, :, f["def"]] / 1000, rows[:, :, f["level"]] / 12,
                               rows[:, :, f["overlays"]] / 5, rows[:, :, f["counters"]] / 5], -1), torch.float32),
            "scal": t(np.stack([own_lp / 8000, opp_lp / 8000, sc[:, 2] / 20, (sc[:, 3] == player), sc[:, 5] / 5], -1),
                      torch.float32),
            "phase": t(np.log2(np.maximum(sc[:, 4], 1)).astype(np.int64).clip(0, 15)),
            "act": t(np.clip(d["act"][ids], 0, [self.n_prompt - 1, 15, 15, 7])),
            # targets
            "moved": t(mask & (t_loc != cur_loc), torch.float32), "dest": t(t_loc),
            "t_pos": t(rows[:, :, f["t_pos"]], torch.float32), "t_stat": t(rows[:, :, f["t_stat"]], torch.float32),
            "t_refs": t((rows[:, :, f["t_refs"]][..., None] >> np.arange(self.n_ref)) & 1, torch.float32),
            "msgs": t((msgs[:, None] >> np.arange(self.n_msg)) & 1, torch.float32),
            "next": t(d["next"][ids]), "next_same": t(d["next_same"][ids], torch.float32),
            "dlp": t(np.stack([np.clip(own_dlp / 1000, -8, 8), np.clip(opp_dlp / 1000, -8, 8)], -1), torch.float32),
        }


N_ROW, N_COL = 6, 8  # (rows: opponent S/T, opponent monsters, Extra Monster Zones, own monsters, own S/T, off field;
# columns 0-4, 5 field zone, 6 other S/T slots, 7 off field)


def _board(loc: np.ndarray, seq: np.ndarray, own: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of cards on the field as the deciding player sees it (materials: their monster's)."""
    row = np.full(loc.shape, 5, dtype=np.int64)
    col = np.full(loc.shape, 7, dtype=np.int64)
    mon = (loc == 0x4) | (loc == 0x80)
    main = mon & (seq < 5)
    row[main] = np.where(own[main], 3, 1)
    col[main] = np.where(own[main], seq[main], 4 - seq[main])
    emz = mon & (seq >= 5)
    row[emz] = 2
    col[emz] = np.where((seq[emz] == 5) == own[emz], 1, 3)
    st = loc == 0x8
    row[st] = np.where(own[st], 4, 0)
    col[st] = np.where(seq[st] < 5, np.where(own[st], seq[st], 4 - seq[st]), np.where(seq[st] == 5, 5, 6))
    return row, col


def _loc_class(loc: np.ndarray) -> np.ndarray:
    out = np.full(loc.shape, N_LOC - 1, dtype=np.int64)
    for i, v in enumerate((0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80)):
        out[loc == v] = i
    return out


class EffectNet(nn.Module):
    def __init__(self, encoder, n_msg: int, n_ref: int, n_prompt: int, dim: int = 256, layers: int = 4,
                 encoding: str = "v1"):
        super().__init__()
        self.encoder = encoder
        self.encoding = encoding
        if encoding != "v1":
            self.brow, self.bcol = nn.Embedding(N_ROW, dim), nn.Embedding(N_COL, dim)
        if encoding == "v3":
            self.chain = nn.Embedding(9, dim)
            self.eq_row, self.eq_col = nn.Embedding(N_ROW, dim), nn.Embedding(N_COL, dim)
            self.tg_row, self.tg_col = nn.Embedding(N_ROW, dim), nn.Embedding(N_COL, dim)
        self.text = nn.Linear(768, dim)
        self.own, self.loc, self.seq = nn.Embedding(2, dim), nn.Embedding(N_LOC, dim), nn.Embedding(64, dim)
        self.pos, self.public, self.is_act = nn.Embedding(16, dim), nn.Embedding(2, dim), nn.Embedding(2, dim)
        self.num = nn.Linear(5, dim)
        self.scal = nn.Linear(5, dim)
        self.phase = nn.Embedding(16, dim)
        self.act = nn.ModuleList([nn.Embedding(n_prompt, dim), nn.Embedding(16, dim), nn.Embedding(16, dim),
                                  nn.Embedding(8, dim)])
        self.body = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, 8, 4 * dim, 0.0, batch_first=True,
                                                                     norm_first=True), layers,
                                          enable_nested_tensor=False)
        self.card_head = nn.Linear(dim, 1 + N_LOC + 2 + n_ref)
        self.global_head = nn.Linear(dim, n_msg + n_prompt + 1 + 2)
        self.n_ref, self.n_msg, self.n_prompt = n_ref, n_msg, n_prompt

    def encode_texts(self, texts: list[str], device, train_encoder: bool) -> torch.Tensor:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        tok = self.encoder.preprocess if hasattr(self.encoder, "preprocess") else self.encoder.tokenize
        parts = []
        for s in range(0, len(order), CHUNK):
            feats = tok([texts[i] for i in order[s:s + CHUNK]])
            ids, att = feats["input_ids"].to(device), feats["attention_mask"].to(device)

            def f(ids, att):
                return self.encoder({"input_ids": ids, "attention_mask": att})["sentence_embedding"]
            if train_encoder:
                parts.append(checkpoint(f, ids, att, use_reentrant=False))
            else:
                with torch.no_grad():
                    parts.append(f(ids, att))
        emb = torch.cat(parts)
        out = torch.empty_like(emb)
        out[torch.as_tensor(order, device=device)] = emb
        return out

    def forward(self, b: dict, texts: torch.Tensor):
        x = self.text(texts[b["tidx"]].to(self.text.weight.dtype))
        x = x + self.own(b["own"]) + self.loc(b["loc"]) + self.pos(b["pos"]) \
            + self.public(b["public"]) + self.is_act(b["is_act"]) + self.num(b["num"])
        if self.encoding == "v1":
            x = x + self.seq(b["seq"])
        else:
            x = x + self.brow(b["brow"]) + self.bcol(b["bcol"])
        if self.encoding == "v3":
            x = x + self.chain(b["chain"]) + self.eq_row(b["eq_row"]) + self.eq_col(b["eq_col"]) \
                + self.tg_row(b["tg_row"]) + self.tg_col(b["tg_col"])
        a = b["act"]
        g = self.scal(b["scal"]) + self.phase(b["phase"]) + sum(e(a[:, i]) for i, e in enumerate(self.act))
        h = self.body(torch.cat([g[:, None], x], 1),
                      src_key_padding_mask=~torch.cat([torch.ones_like(b["mask"][:, :1]), b["mask"]], 1))
        return self.card_head(h[:, 1:]), self.global_head(h[:, 0])


def losses(net: EffectNet, card: torch.Tensor, glob: torch.Tensor, b: dict) -> dict:
    bce = nn.functional.binary_cross_entropy_with_logits
    m = b["mask"]
    nr = net.n_ref
    moved_logit, dest_logit = card[..., 0], card[..., 1:1 + N_LOC]
    pos_l, stat_l, refs_l = card[..., 1 + N_LOC], card[..., 2 + N_LOC], card[..., 3 + N_LOC:3 + N_LOC + nr]
    moved = b["moved"] > 0
    out = {
        "moved": bce(moved_logit[m], b["moved"][m], pos_weight=torch.tensor(10.0, device=card.device)),
        "dest": nn.functional.cross_entropy(dest_logit[moved], b["dest"][moved]) if moved.any() else card.sum() * 0,
        "pos": bce(pos_l[m], b["t_pos"][m], pos_weight=torch.tensor(10.0, device=card.device)),
        "stat": bce(stat_l[m], b["t_stat"][m], pos_weight=torch.tensor(10.0, device=card.device)),
        "refs": bce(refs_l[m], b["t_refs"][m], pos_weight=torch.tensor(10.0, device=card.device)),
    }
    nm, npr = net.n_msg, net.n_prompt
    out["msgs"] = bce(glob[:, :nm], b["msgs"])
    out["next"] = nn.functional.cross_entropy(glob[:, nm:nm + npr], b["next"])
    out["next_same"] = bce(glob[:, nm + npr], b["next_same"])
    out["dlp"] = nn.functional.smooth_l1_loss(glob[:, nm + npr + 1:], b["dlp"])
    return out


@torch.no_grad()
def evaluate(net, data: Data, batches: list, device, train_encoder: bool) -> dict:
    net.eval()
    tot: dict[str, float] = {}
    counts = {"moved_tp": 0, "moved_fp": 0, "moved_fn": 0, "dest_hit": 0, "dest_n": 0, "refs_tp": 0, "refs_fp": 0,
              "refs_fn": 0, "msgs_hit": 0, "msgs_n": 0, "next_hit": 0, "n": 0}
    for ids, seeds in batches:
        b = data.batch(ids, device)
        texts = net.encode_texts(_texts(b["keys"], data, seeds), device, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            card, glob = net(b, texts)
        card, glob = card.float(), glob.float()
        for k, v in losses(net, card, glob, b).items():
            tot[k] = tot.get(k, 0.0) + v.item() * len(ids)
        m = b["mask"]
        pm, tm = (card[..., 0] > 0) & m, b["moved"] > 0
        counts["moved_tp"] += (pm & tm).sum().item()
        counts["moved_fp"] += (pm & ~tm).sum().item()
        counts["moved_fn"] += (~pm & tm & m).sum().item()
        counts["dest_hit"] += (card[..., 1:1 + N_LOC].argmax(-1) == b["dest"])[tm].sum().item()
        counts["dest_n"] += tm.sum().item()
        nr = net.n_ref
        pr, tr = (card[..., 3 + N_LOC:3 + N_LOC + nr] > 0) & m[..., None], b["t_refs"] > 0
        counts["refs_tp"] += (pr & tr).sum().item()
        counts["refs_fp"] += (pr & ~tr).sum().item()
        counts["refs_fn"] += (~pr & tr).sum().item()
        nm, npr = net.n_msg, net.n_prompt
        counts["msgs_hit"] += ((glob[:, :nm] > 0) == (b["msgs"] > 0)).all(1).sum().item()
        counts["msgs_n"] += len(ids)
        counts["next_hit"] += (glob[:, nm:nm + npr].argmax(-1) == b["next"]).sum().item()
        counts["n"] += len(ids)
    net.train()
    c = counts
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)  # noqa: E731
    out = {k: v / c["n"] for k, v in tot.items()}
    out.update({"moved_f1": f1(c["moved_tp"], c["moved_fp"], c["moved_fn"]), "dest_acc": c["dest_hit"] / max(c["dest_n"], 1),
                "refs_f1": f1(c["refs_tp"], c["refs_fp"], c["refs_fn"]), "msgs_exact": c["msgs_hit"] / c["msgs_n"],
                "next_acc": c["next_hit"] / c["n"]})
    return out


_NAMES: dict[int, Names] = {}
REAL_NAMES = False


def _texts(keys: list, data: Data, seeds=None) -> list[str]:
    from ygoai.nn.card_text import _texts as raw, type_line
    from ygoai.carddb import default_db
    out = []
    for seed, ci in keys:
        code = int(data.codes[ci])
        if REAL_NAMES:
            card = default_db().get(code)
            name, desc = raw().get(code, ("", ""))
            out.append(f"{name}. {type_line(card)}. {desc}" if card else "Unknown card.")
            continue
        names = _NAMES.get(seed)
        if names is None:
            if len(_NAMES) > 64:
                _NAMES.clear()
            names = _NAMES[seed] = Names(seed)
        out.append(names.render(code))
    return out


def main() -> int:
    global REAL_NAMES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--games-per-batch", type=int, default=2)
    ap.add_argument("--per-game", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--freeze-encoder", action="store_true", help="the pretrained encoder as it is (a baseline)")
    ap.add_argument("--train-layers", type=int, default=None, help="only the encoder's last N layers train")
    ap.add_argument("--real-names", action="store_true", help="the cards' real names (an ablation)")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--encoding", choices=("v1", "v2", "v3"), default="v1")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    REAL_NAMES = args.real_names
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    from sentence_transformers import SentenceTransformer
    # (eager attention: on ROCm, SDPA gives NaN for every padded batch)
    enc = SentenceTransformer(ENCODER, config_kwargs={"vision_config": None, "audio_config": None}, device="cuda",
                              model_kwargs={"attn_implementation": "eager"})
    enc.max_seq_length = args.max_tokens
    layers = enc[0].model.language_model.layers
    for n, p in enc.named_parameters():
        train = not args.freeze_encoder and "embed" not in n
        if train and args.train_layers is not None:
            train = any(f"layers.{i}." in n for i in range(len(layers) - args.train_layers, len(layers)))
        p.requires_grad_(train)
    train_encoder = any(p.requires_grad for p in enc.parameters())
    data = Data(args.data, args.val)
    net = EffectNet(enc, data.n_msg, data.n_ref, data.n_prompt, encoding=args.encoding).to(device)
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    head_params = [p for n, p in net.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    rng = np.random.default_rng(args.seed)
    vrng = np.random.default_rng(12345)
    val_batches = []
    for s in range(0, min(len(data.val_games), 60), args.games_per_batch):
        games = data.val_games[s:s + args.games_per_batch]
        val_batches.append((data.sample(vrng, games, args.per_game), None))
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")
    print(f"{len(data.train_games)} train / {len(data.val_games)} val games; encoder trains: {train_encoder} "
          f"({sum(p.numel() for p in enc_params) / 1e6:.0f}M params)", flush=True)
    t0 = time.time()
    run: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        frac = step / args.steps
        scale = min(1.0, step / 500) * 0.5 * (1 + math.cos(math.pi * frac))
        for g, lr in zip(opt.param_groups, base):
            g["lr"] = lr * scale
        games = rng.choice(data.train_games, args.games_per_batch, replace=False)
        b = data.batch(data.sample(rng, games, args.per_game), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            texts = net.encode_texts(_texts(b["keys"], data), device, train_encoder)
            card, glob = net(b, texts)
            parts = losses(net, card.float(), glob.float(), b)
            loss = sum(parts.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        for k, v in parts.items():
            run[k] = 0.98 * run.get(k, v.item()) + 0.02 * v.item()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(net, data, val_batches, device, train_encoder)
            rec = {"step": step, "seconds": round(time.time() - t0), "train": {k: round(v, 4) for k, v in run.items()},
                   "val": {k: round(v, 4) for k, v in ev.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step} [{rec['seconds']}s] val: " + " ".join(f"{k} {v:.3f}" for k, v in ev.items()), flush=True)
            if train_encoder:
                enc.save(str(args.out / "encoder"))
            torch.save({k: v for k, v in net.state_dict().items() if not k.startswith("encoder.")}, args.out / "heads.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
