"""Reinforcement learning of the imitation policy (train.py's model) from its own games.

    .venv/bin/python scripts/policy/rl.py runs/policy/train/l8d512_str/model.pt runs/policy/rl/NAME [--iterations 100]

Each iteration: bot-rl-games plays --count openings of --book (the next ones each iteration), both
seats, --repeat times each, the learner sampling at --temp against itself, --jobs games at a time
(their decisions scored here, on the GPU, in batches: bot-rl-games --remote); every decision of a
game player gets its result z (+1 won, -1 lost, 0 a draw) less the mean z of the same opening and
player (a group baseline: the deck pair's and seat's strength drop out) as its advantage. Then
--epochs passes of the clipped objective -min(r A, clip(r, 1 - eps, 1 + eps) A) + --kl KL(pi ||
pi_init), r = pi(chosen) / pi_played(chosen) (pi = softmax(score / T); a multi-card choice: the
product over its cards). The policy is exported for the bot (export_policy.py's format; the
model keeps the float16-rounded weights, as it plays). Every --test-every iterations it plays
the initial policy (both their best options) on --test-count openings of --test-book, both seats.

The text embeddings come from the bot's tables (embed_tables.py): each game player's own
pseudo-name seed, as it played. OUT/log.jsonl: a line per iteration; OUT/iterN.bin (the bot's), OUT/iterN.pt; OUT/state.pt resumes.
"""

import argparse
import copy
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import types
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import H, O, T, PolicyData  # noqa: E402
from embed_strings import string_key  # noqa: E402
from export_policy import weights, write  # noqa: E402
from train import MAX_OPTIONS, Batcher, Policy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent


class Tables:
    """embed_tables.py's file on the GPU, plus a zero row (a code not in the tables)."""

    def __init__(self, path: str, dev):
        raw = Path(path).read_bytes()
        assert raw[:8] == b"YGOPTAB1", path
        n = int.from_bytes(raw[8:12], "little")
        h = json.loads(raw[12:12 + n])
        self.dim, self.seeds = h["dim"], len(h["seeds"])
        self.card_row = {c: i for i, c in enumerate(h["codes"])}
        self.string_row = {(c, s): len(h["codes"]) + i for i, (c, s) in enumerate(h["strings"])}
        self.per_seed = len(h["codes"]) + len(h["strings"])
        v = np.frombuffer(raw, dtype="<f2", offset=12 + n).reshape(-1, self.dim)
        assert len(v) == self.seeds * self.per_seed
        self.values = torch.cat([torch.from_numpy(v.copy()), torch.zeros(1, self.dim, dtype=torch.float16)]).to(dev)
        self.zero = len(self.values) - 1

    def embeddings(self, data: PolicyData):
        """(emb, start, semb, str_keys) for a games directory, as embed_games.py and embed_strings.py
        give them for train.Batcher: each game player's codes and strings under its names."""
        rows, start, skeys, srows = [], {}, [], []
        for g, x in sorted(data.games.items()):
            base = x["names"] * self.per_seed
            start[g] = len(rows)
            for i, c in enumerate(x["codes"]):
                r = self.card_row.get(c)
                rows.append(self.zero if r is None else base + r)
                for s in range(16):
                    r = self.string_row.get((c, s))
                    if r is not None:
                        skeys.append(int(string_key(g, i + 1, s)))
                        srows.append(base + r)
        order = np.argsort(skeys)
        dev = self.values.device
        emb = self.values[torch.tensor(rows, device=dev)]
        semb = self.values[torch.tensor(np.asarray(srows)[order], device=dev)]
        return emb, start, semb, np.asarray(skeys, dtype=np.int64)[order]


def log_probs(scores: torch.Tensor, b: dict, temp: float):
    """(log pi(chosen) per decision, log pi per option): pi = softmax(score / temp) over the options."""
    lp = F.log_softmax(scores.float() / temp, -1)
    chosen = (b["opt"][..., O["CHOSEN"]] > 0) & b["opt_mask"]
    return lp.masked_fill(~chosen, 0).sum(1), lp


def answer(read, nets: list, tables: Tables, batch: int = 512) -> bytes:
    """The answer to a batch of bot::policy_remote's protocol (after its u32 1; `read(k)`: its
    next k bytes): its decisions scored by `nets` (network i: nets[i], in eval mode)."""
    u32 = lambda: struct.unpack("<I", read(4))[0]
    n = u32()
    _, net, nt, no, nc, ns = np.frombuffer(read(24 * n), "<u4").reshape(n, 6).astype(np.int64).T
    hdr = np.frombuffer(read(8 * len(H) * n), "<i8").reshape(n, len(H)).copy()
    tok = np.frombuffer(read(2 * len(T) * nt.sum()), "<i2").reshape(-1, len(T))
    opt = np.frombuffer(read(4 * len(O) * no.sum()), "<i4").reshape(-1, len(O))
    code_rows = np.frombuffer(read(4 * nc.sum()), "<i4").astype(np.int64)
    str_rows = np.frombuffer(read(12 * ns.sum()), "<i4").reshape(-1, 3).astype(np.int64)
    # (as a games directory's: decision i is game i, its code index 1 at start[i])
    hdr[:, H["GAME"]] = np.arange(n)
    hdr[:, H["TOK_START"]] = np.cumsum(nt) - nt
    hdr[:, H["OPT_START"]] = np.cumsum(no) - no
    start = dict(enumerate((np.cumsum(nc) - nc).tolist()))
    dev = tables.values.device
    emb = tables.values[torch.from_numpy(np.where(code_rows < 0, tables.zero, code_rows)).to(dev)]
    # (a key no string has first: the keys are never empty)
    keys = np.concatenate([[-1], string_key(np.repeat(np.arange(n), ns), str_rows[:, 0], str_rows[:, 1])])
    rows = np.concatenate([[tables.zero], str_rows[:, 2]])
    order = np.argsort(keys)
    semb = tables.values[torch.from_numpy(rows[order]).to(dev)]
    batcher = Batcher(types.SimpleNamespace(hdr=hdr, tok=tok, opt=opt), start, keys[order])
    ids, scores = [], []
    with torch.no_grad():
        for k, m in enumerate(nets):
            idx = np.where(net == k)[0]
            for s in range(0, len(idx), batch):
                j = idx[s:s + batch]
                b = {key: v.to(dev, non_blocking=True) for key, v in batcher(j).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sc = m(b, emb, semb if hasattr(m, "str_opt") else None).float().cpu().numpy()
                ids.append(j)
                scores += [sc[i, :min(no[x], MAX_OPTIONS)] for i, x in enumerate(j)]
    ids = np.concatenate(ids)
    return (struct.pack("<I", n) + ids.astype("<u4").tobytes() + np.array([len(x) for x in scores], "<u4").tobytes()
            + np.concatenate(scores).astype("<f4").tobytes())


def play(cmd: list[str], nets: list, tables: Tables, log, batch: int = 512):
    """Runs bot-rl-games --remote, scoring its decisions with `nets` (network i: nets[i]); its
    summary. (bot::policy_remote's protocol.)"""
    print(" ".join(cmd), file=log, flush=True)
    proc = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=24G", "-p", "MemorySwapMax=0"] + cmd,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)

    def read(n: int) -> bytes:
        b = proc.stdout.read(n)
        if len(b) != n:
            raise SystemExit(f"{cmd[0]} failed ({proc.wait()})")
        return b

    u32 = lambda: struct.unpack("<I", read(4))[0]
    for m in nets:
        m.eval()
    while u32() == 1:
        proc.stdin.write(answer(read, nets, tables, batch))
        proc.stdin.flush()
    summary = json.loads(read(u32()))
    if proc.wait():
        raise SystemExit(f"{cmd[0]} failed ({proc.returncode})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--book", default="runs/books/era_data.jsonl")
    ap.add_argument("--first", type=int, default=30000, help="(the book's first openings: the eval's datagen)")
    ap.add_argument("--count", type=int, default=256, help="openings per iteration")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tables", default="runs/policy/tables/tables8.bin")
    ap.add_argument("--jobs", type=int, default=256, help="games at a time")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl", type=float, default=0.02, help="weight of KL(pi || initial policy)")
    ap.add_argument("--test-book", default="runs/books/era_sprt.jsonl")
    ap.add_argument("--test-count", type=int, default=200)
    ap.add_argument("--test-every", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=5, help="(iterN.pt; iterN.bin: every iteration)")
    ap.add_argument("--keep-games", action="store_true", help="keep each iteration's games (else only the last)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "run.log", "a")
    # (the binary as it is now: a rebuild during the run changes nothing)
    games_bin = out / "bot-rl-games"
    shutil.copy2(ROOT / "target/release/bot-rl-games", games_bin)

    dev = torch.device("cuda")
    tables = Tables(a.tables, dev)
    ck = torch.load(a.model, map_location="cpu")
    args = ck["args"]
    model = Policy(tables.dim, args["dim"], args["layers"], args["heads"], strings=bool(args["strings"])).to(dev)
    model.load_state_dict(ck["model"])
    start_it = 0
    state = out / "state.pt"
    if state.exists():
        st = torch.load(state, map_location="cpu")
        model.load_state_dict(st["model"])
        start_it = st["iteration"]
    initial = Policy(tables.dim, args["dim"], args["layers"], args["heads"], strings=bool(args["strings"])).to(dev)
    initial.load_state_dict(ck["model"])
    initial.eval()
    init_bin, init_weights = out / "iter0.bin", weights(initial)  # (float16-rounded, as the bot's)
    if not init_bin.exists():
        write(init_bin, args, tables.dim, *init_weights, [])
    for p in initial.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.98))
    if state.exists() and "opt" in st:
        opt.load_state_dict(st["opt"])
    cur_bin, cur_weights = out / f"iter{start_it}.bin", weights(model)
    if not cur_bin.exists():
        write(cur_bin, args, tables.dim, *cur_weights, [])
    rlog = open(out / "log.jsonl", "a")

    for it in range(start_it + 1, a.iterations + 1):
        t0 = time.time()
        gdir = out / "games" / str(it)
        first = a.first + (it - 1) * a.count
        g = play([str(games_bin), str(gdir), "--remote", "--tables", a.tables, "--book", a.book,
                  "--first", str(first), "--count", str(a.count), "--repeat", str(a.repeat), "--temp", str(a.temp),
                  "--jobs", str(a.jobs), "--seed", str(it)], [model], tables, log)
        t_games = time.time() - t0
        data = PolicyData(str(gdir))
        emb, start, semb, str_keys = tables.embeddings(data)
        batcher = Batcher(data, start, str_keys)

        # advantages: z less the mean z of the same opening and player
        z = {gg: x["result"] for gg, x in data.games.items()}
        groups = {}
        for gg, x in data.games.items():
            groups.setdefault((x["opening"], x["player"]), []).append(z[gg])
        adv_game = {gg: z[gg] - float(np.mean(groups[(x["opening"], x["player"])])) for gg, x in data.games.items()}
        learner = {gg for gg, x in data.games.items() if x["learner"]}
        gcol = data.hdr[:, H["GAME"]]
        idx = np.array([i for i in range(len(gcol)) if gcol[i] in learner])
        adv = np.array([adv_game[gcol[i]] for i in idx], dtype=np.float32)
        keep = adv != 0  # (a decision whose group all ended alike teaches nothing)
        idx, adv = idx[keep], adv[keep]

        # the playing policy's log probabilities of its choices
        model.eval()
        old = np.zeros(len(idx), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                b = {k: v.to(dev, non_blocking=True) for k, v in batcher(idx[s:s + 1024]).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sc = model(b, emb, semb)
                old[s:s + 1024] = log_probs(sc, b, a.temp)[0].cpu().numpy()
        model.train()

        rng = np.random.default_rng(it)
        stats = {"loss": 0.0, "clipped": 0.0, "kl_init": 0.0, "entropy": 0.0, "approx_kl": 0.0, "n": 0}
        for _ in range(a.epochs):
            perm = rng.permutation(len(idx))
            for s in range(0, len(perm), a.batch):
                j = perm[s:s + a.batch]
                b = {k: v.to(dev, non_blocking=True) for k, v in batcher(idx[j]).items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sc = model(b, emb, semb)
                    with torch.no_grad():
                        sc0 = initial(b, emb, semb)
                new, lp = log_probs(sc, b, a.temp)
                lp0 = F.log_softmax(sc0.float() / a.temp, -1)
                mask = b["opt_mask"]
                p = lp.exp().masked_fill(~mask, 0)
                kl0 = (p * (lp - lp0).masked_fill(~mask, 0)).sum(1)
                ent = -(p * lp.masked_fill(~mask, 0)).sum(1)
                A = torch.from_numpy(adv[j]).to(dev)
                o = torch.from_numpy(old[j]).to(dev)
                r = (new - o).exp()
                surr = torch.minimum(r * A, r.clamp(1 - a.clip, 1 + a.clip) * A)
                loss = (-surr + a.kl * kl0).mean()
                if not torch.isfinite(loss):
                    raise SystemExit(f"iteration {it}: loss {loss}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                n = len(j)
                stats["loss"] += float(loss.detach()) * n
                stats["clipped"] += float(((r.detach() - 1).abs() > a.clip).float().sum())
                stats["kl_init"] += float(kl0.detach().sum())
                stats["entropy"] += float(ent.detach().sum())
                stats["approx_kl"] += float((o - new).detach().sum())
                stats["n"] += n
        n = max(stats.pop("n"), 1)
        stats = {k: round(v / n, 5) for k, v in stats.items()}

        cur_bin = out / f"iter{it}.bin"
        write(cur_bin, args, tables.dim, *weights(model), [])
        if it % a.save_every == 0:
            torch.save({"args": args, "model": model.state_dict(), "iteration": it}, out / f"iter{it}.pt")
        torch.save({"args": args, "model": model.state_dict(), "opt": opt.state_dict(), "iteration": it}, state)
        if not a.keep_games and it > 1:
            prev = out / "games" / str(it - 1)
            if prev.exists():
                shutil.rmtree(prev)
        rec = {"iteration": it, "games": g["games"], "decisions": int(len(idx)), "openings": [first, first + a.count],
               "seconds_games": round(t_games), "seconds": round(time.time() - t0), **stats}
        if it % a.test_every == 0:
            t = play([str(games_bin), "--test", "--remote", "--versus", "--tables", a.tables,
                      "--book", a.test_book, "--first", "0", "--count", str(a.test_count), "--jobs", str(a.jobs)], [model, initial], tables, log)
            rec["test"] = {"games": t["games"], "score": round(t["score"], 4), "elo": round(t["elo"], 1)}
        print(json.dumps(rec), file=rlog, flush=True)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
