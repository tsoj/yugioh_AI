#!/usr/bin/env python3
"""Reinforcement learning of the policy from its own games (policy gradient, PPO-style clipping).

  scripts/rl_policy.py --init runs/nn/boot1_policy.npz -o runs/nn/rl1 [--iterations 50] [--games 2000]

Each iteration: games of the current policy (sampling, softmax(score / T)) on
random corpus deck pairs, against itself (both sides learn), earlier versions or
the heuristic player; each of the learner's decisions gets the game's result z
(+1 won, -1 lost) minus the batch mean as its advantage; --epochs passes of the
clipped objective -min(r * A, clip(r, 1 - eps, 1 + eps) * A) - entropy_bonus *
entropy, r = pi(chosen) / pi_played(chosen) (an update cannot move a move's
probability far from the policy that played; plain REINFORCE collapsed after 42
iterations), gradient norm clipped. Then the policy alone (argmax) plays the
heuristic player (both seats); an iteration --rollback Elo below the best so far
is undone (the best weights again). Each iteration's policy is saved
(OUT/iterN.npz, a train_policy_value.py export) with a log line in OUT/log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# (torch and the training code: imported in main only; the game workers import this module)
torch = Data = Net = None

from ygoai.agents.heuristic_agents import HeuristicAgent  # noqa: E402
from ygoai.agents.policy_agent import PolicyAgent, load_policy  # noqa: E402
from ygoai.carddb import default_db  # noqa: E402
from ygoai.game import new_record, play  # noqa: E402
from ygoai.match import random_openings  # noqa: E402
from ygoai.nn import features as F  # noqa: E402
from ygoai.parallel import Crash, run_parallel  # noqa: E402


# -- games (in the workers) ---------------------------------------------------

MAX_DECISIONS = 3000  # (a game is ~150-400; one that loops ends as a draw, z = 0)

def _agent(spec: str, seed: int, temperature: float, record: bool):
    if spec == "heuristic":
        return HeuristicAgent(seed)
    return PolicyAgent(seed, model=load_policy(spec), temperature=temperature, record=record)


def selfplay_game(task):
    """(opening, learner model, opponent spec, learner's seat, T) -> [(decisions, z)] of the learner's sides."""
    opening, learner, opponent, seat, temperature = task
    record = new_record(opening.decks, opening.seed)
    agents = [None, None]
    agents[seat] = _agent(learner, opening.seed * 2 + seat, temperature, True)
    agents[1 - seat] = _agent(learner if opponent == "self" else opponent, opening.seed * 2 + 1 - seat,
                              temperature, opponent == "self")
    play(record, agents, max_turns=100, max_decisions=MAX_DECISIONS)
    winner = record.result.get("winner")
    out = []
    for p, a in enumerate(agents):
        if isinstance(a, PolicyAgent) and a.decisions is not None:
            z = 0.0 if winner not in (0, 1) else (1.0 if winner == p else -1.0)
            out.append((a.decisions, z))
    return out, winner


def eval_game(task):
    """(opening, model, seat) -> 1 if the policy (argmax) won."""
    opening, model, seat = task
    record = new_record(opening.decks, opening.seed)
    agents = [None, None]
    agents[seat] = PolicyAgent(opening.seed, model=load_policy(model))
    agents[1 - seat] = HeuristicAgent(opening.seed + 1)
    play(record, agents, max_turns=100, max_decisions=MAX_DECISIONS)
    w = record.result.get("winner")
    return 0.5 if w not in (0, 1) else float(w == seat)


# -- training -------------------------------------------------------------------

class Codes:
    """Static indices of card codes, growing; the table of their static features."""

    def __init__(self, codes, table):
        self.index = F.CodeIndex([int(c) for c in codes])
        self.table = [row for row in np.asarray(table, dtype=np.float32)]

    def __call__(self, code: int) -> int:
        n = len(self.index.codes)
        i = self.index(code)
        if i >= n:
            card = default_db().get(code) if code else None
            self.table.append(np.asarray(F.static_features(card), dtype=np.float32))
        return i


def to_data(samples: list[tuple], codes: Codes) -> dict:
    """[(cards, scalars, acts, chosen, logp, advantage)] (codes as codes) -> Data's arrays."""
    d = {"game": [], "ncards": [], "nacts": [], "target": [], "label": [], "scalars": [], "oldlogp": []}
    cards, acts = [], []
    for i, (cs, sc, ac, k, lp, adv) in enumerate(samples):
        d["oldlogp"].append(lp)
        d["game"].append(i)
        d["ncards"].append(len(cs))
        d["nacts"].append(len(ac))
        d["target"].append(k)
        d["label"].append(adv)
        d["scalars"].append(sc)
        cards.extend((codes(c), g, n, *raw) for c, g, n, raw in cs)
        acts.extend((pt, kk, h, e, codes(c), loc, own, *raw) for pt, kk, h, e, c, loc, own, raw in ac)
    c = np.asarray(cards, dtype=np.int32).reshape(-1, 3 + F.RAW_DIM)
    a = np.asarray(acts, dtype=np.int32).reshape(-1, 7 + F.RAW_DIM)
    return {"game": np.asarray(d["game"], dtype=np.int32), "ncards": np.asarray(d["ncards"], dtype=np.int16),
            "nacts": np.asarray(d["nacts"], dtype=np.int16), "target": np.asarray(d["target"], dtype=np.int16),
            "label": np.asarray(d["label"], dtype=np.float32), "scalars": np.asarray(d["scalars"], dtype=np.float32),
            "oldlogp": np.asarray(d["oldlogp"], dtype=np.float32),
            "card_static": c[:, 0], "card_group": c[:, 1], "card_count": c[:, 2], "card_raw": c[:, 3:],
            "act_cat": a[:, :4], "act_static": a[:, 4], "act_loc": a[:, 5], "act_own": a[:, 6], "act_raw": a[:, 7:]}


def pg_loss(net, b, entropy_bonus: float, clip: float):
    """Clipped policy gradient (per decision, averaged), see the module's doc."""
    _, scores = net(b)
    nacts = b["nacts"]
    B, A = len(nacts), int(nacts.max())
    pos = np.arange(len(scores)) - np.repeat(np.concatenate([[0], np.cumsum(nacts)[:-1]]), nacts)
    padded = torch.full((B, A), -1e9, device=scores.device)
    padded[b["act_dec"], torch.as_tensor(pos, device=scores.device)] = scores
    logp = torch.log_softmax(padded, dim=1)
    target = torch.as_tensor(b["target"], device=scores.device)
    chosen = logp.gather(1, target[:, None])[:, 0]
    probs = logp.exp()
    entropy = -(probs * torch.where(probs > 0, logp, torch.zeros_like(logp))).sum(1)
    ratio = torch.exp(chosen - b["oldlogp"])
    adv = b["label"]
    objective = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    return -objective.mean() - entropy_bonus * entropy.mean(), entropy.mean().item()


def export(net, codes: Codes, path: Path) -> None:
    state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    state["static_table"] = np.stack(codes.table)
    np.savez(path, codes=np.asarray(codes.index.codes, dtype=np.int64), **state)


def load(path: Path, device, dim: int, hidden: int):
    w = np.load(path)
    codes = Codes(w["codes"], w["static_table"])
    net = Net(torch.as_tensor(w["static_table"], device=device), dim, hidden).to(device)
    net.load_state_dict({k: torch.as_tensor(w[k], device=device) for k in w.files if k != "codes"})
    return net, codes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=2, help="passes over each iteration's games")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--rollback", type=float, default=50.0, help="Elo below the best that undoes an iteration")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--opponents", default="self:0.5,past:0.25,heuristic:0.25")
    ap.add_argument("--eval-openings", type=int, default=500)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=500000)
    ap.add_argument("-j", "--jobs", type=int, default=24)
    args = ap.parse_args()
    global torch, Data, Net
    import torch
    from train_policy_value import Data, Net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, codes = load(args.init, device, args.dim, args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    current = args.out / "iter0.npz"
    export(net, codes, current)
    pool = [current]
    mix = [(k, float(v)) for k, v in (x.split(":") for x in args.opponents.split(","))]
    rng = random.Random(args.seed)
    log = (args.out / "log.jsonl").open("a")
    best = None  # (Elo, model path, weights) of the best iteration so far
    eval_ops = random_openings(args.eval_openings, args.seed - 1, 0.0)  # (the same every iteration)
    for it in range(1, args.iterations + 1):
        t0 = time.time()
        openings = random_openings(args.games, args.seed + it * 100003, 0.0)
        tasks = []
        for op in openings:
            kind = rng.choices([k for k, _ in mix], [w for _, w in mix])[0]
            opp = {"self": "self", "heuristic": "heuristic", "past": str(rng.choice(pool))}[kind]
            tasks.append((op, str(current), opp, rng.randrange(2), args.temperature))
        samples, results = [], {"self": [], "heuristic": [], "past": []}
        for task, res in run_parallel(selfplay_game, tasks, args.jobs):
            if isinstance(res, Crash):
                continue
            sides, _ = res
            opp = task[2]
            kind = "self" if opp == "self" else "heuristic" if opp == "heuristic" else "past"
            for decisions, z in sides:
                if kind != "self":
                    results[kind].append(z)
                samples.extend((cs, sc, ac, k, lp, z) for cs, sc, ac, k, lp in decisions)
        t_play = time.time() - t0
        mean_z = float(np.mean([s[5] for s in samples])) if samples else 0.0
        samples = [(cs, sc, ac, k, lp, z - mean_z) for cs, sc, ac, k, lp, z in samples]
        data = Data.__new__(Data)
        data.d, data.device = to_data(samples, codes), device
        data.card_off = np.concatenate([[0], np.cumsum(data.d["ncards"].astype(np.int64))])
        data.act_off = np.concatenate([[0], np.cumsum(data.d["nacts"].astype(np.int64))])
        data.n = len(samples)
        net.static_table = torch.as_tensor(np.stack(codes.table), device=device)
        ent = []
        for epoch in range(args.epochs):
            perm = np.random.default_rng(it * 100 + epoch).permutation(data.n)
            for s in range(0, len(perm), args.batch):
                ids = perm[s:s + args.batch]
                b = data.batch(ids)
                b["oldlogp"] = torch.as_tensor(data.d["oldlogp"][ids], device=device)
                loss, e = pg_loss(net, b, args.entropy, args.clip)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                opt.step()
                ent.append(e)
        current = args.out / f"iter{it}.npz"
        export(net, codes, current)
        pool.append(current)
        evals = [r for _, r in run_parallel(eval_game, [(op, str(current), seat) for op in eval_ops for seat in (0, 1)],
                                            args.jobs) if not isinstance(r, Crash)]
        score = float(np.mean(evals))
        elo = -400 * math.log10(1 / min(max(score, 1e-3), 1 - 1e-3) - 1)
        rolled_back = False
        if best is None or elo > best[0]:
            best = (elo, current, {k: v.detach().clone() for k, v in net.state_dict().items() if k != "static_table"})
        elif elo < best[0] - args.rollback:
            # (a collapse: back to the best weights; the next games are the best policy's)
            net.load_state_dict(best[2], strict=False)  # (the static table only grows: kept)
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            pool.remove(current)  # (no opponent either)
            current = best[1]
            rolled_back = True
        row = {"iteration": it, "decisions": data.n, "entropy": float(np.mean(ent)),
               "win_vs_heuristic_sampled": float(np.mean([z > 0 for z in results["heuristic"]]))
               if results["heuristic"] else None,
               "win_vs_past_sampled": float(np.mean([z > 0 for z in results["past"]])) if results["past"] else None,
               "eval_vs_heuristic_score": score, "eval_elo": elo, "eval_games": len(evals), "rolled_back": rolled_back,
               "best": [best[0], str(best[1])],
               "seconds_play": round(t_play), "seconds": round(time.time() - t0)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
