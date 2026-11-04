"""decklight on its C++ core (edo9300) vs on the Rust core (ygopro-core-rust): the same deals played in lockstep by
its raw policy (forge_r1, argmax), which decides on the C++ env; the Rust env takes the option of the same meaning.
At each decision: the player, the prompt type and the options compared (each option's kind, code, description,
argument and card place). A difference is "order" when the options are the same but listed or placed differently
(e.g. another sequence in a pile), "rules" otherwise (then the game stops). Per game: decisions, the first
difference, both winners and turn counts.

  cd external/decklight && .venv/bin/python ../../runs/decklight/equiv.py OUT.jsonl [GAMES_PER_DECK] [PROCS]
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
BOOK = Path(__file__).resolve().parent / "gauntlet_mirror.jsonl"
DECKS = Path(__file__).resolve().parent / "client" / "decks"

_W = {}


def init_worker():
    import torch
    torch.set_num_threads(1)
    from decklight.engine.duel import Engine
    from decklight.play.agents import RawPolicyAgent
    _W["cpp"] = Engine()
    _W["rust"] = Engine(lib_path=ROOT / "build/rust/libocgcore_rs.so", cards=_W["cpp"].cards)
    _W["agent"] = RawPolicyAgent(ckpt="data/cache/forge_r1/best.pt")


def key(o):
    c = o.card
    return (int(o.kind), o.code, o.desc, o.arg, None if c is None else (c.code, c.controller, c.location, c.sequence, c.position))


def loose(o):
    c = o.card
    return (int(o.kind), o.code, o.desc, o.arg, None if c is None else (c.code, c.controller, c.location))


def contents(env):
    """Per player: the codes in each location as multisets (hand and deck together), and the order of the piles."""
    from collections import Counter
    out, order = {}, {}
    for p in (0, 1):
        for name, loc in (("field", 0x0C), ("grave", 0x10), ("removed", 0x20), ("extra", 0x40), ("hand+deck", 0x03)):
            codes = []
            for l in ((0x04, 0x08) if loc == 0x0C else (0x01, 0x02) if loc == 0x03 else (loc,)):
                cs = [c.code if c is not None else None for c in env.duel.query_location(p, l)]
                codes += [c for c in cs if c]
                order[f"{p}/{l}"] = cs
            out[f"{p} {name}"] = sorted(Counter(codes).items())
    return out, order


def classify(a, b):
    """At a difference: "order" if every location holds the same cards on both cores (hand and deck together: a draw
    of another card after another deck order), else "rules" with the locations that differ."""
    ca, oa = contents(a)
    cb, ob = contents(b)
    differ = [k for k in ca if ca[k] != cb[k]]
    piles = [k for k in oa if oa[k] != ob[k]]
    return ("order" if not differ else "rules"), differ, piles


def play(job):
    deck, seed = job
    from decklight.decks import load_ydk
    from decklight.engine.duel import DuelConfig
    from decklight.engine.env import PyEnv
    from decklight.play.view import EngineView
    d = load_ydk(DECKS / f"{deck}.ydk")
    cfg = DuelConfig(d, d, seed=seed)
    a, b = PyEnv(_W["cpp"], cfg), PyEnv(_W["rust"], cfg)
    out = {"deck": deck, "seed": seed, "decisions": 0, "order_diffs": 0, "first_order": None, "rules": None}
    t0 = time.time()
    try:
        a.reset(); b.reset()
        while not a.done and not b.done:
            if a.player != b.player or a.dec.msg.type != b.dec.msg.type:
                out["rules"] = {"at": out["decisions"], "what": "prompt", "cpp": [a.player, a.dec.msg.type], "rust": [b.player, b.dec.msg.type]}
                out["rules"]["kind"], out["rules"]["differ"], out["rules"]["piles"] = classify(a, b)
                break
            ka, kb = [key(o) for o in a.opts], [key(o) for o in b.opts]
            if ka != kb:
                if sorted(map(repr, [loose(o) for o in a.opts])) == sorted(map(repr, [loose(o) for o in b.opts])):
                    out["order_diffs"] += 1
                    if out["first_order"] is None:
                        out["first_order"] = {"at": out["decisions"], "msg": a.dec.msg.type, "cpp": [list(map(str, x)) for x in ka[:6]], "rust": [list(map(str, x)) for x in kb[:6]]}
                else:
                    out["rules"] = {"at": out["decisions"], "what": "options", "msg": a.dec.msg.type, "cpp": [str(x) for x in ka[:8]], "rust": [str(x) for x in kb[:8]]}
                    out["rules"]["kind"], out["rules"]["differ"], out["rules"]["piles"] = classify(a, b)
                    break
            i = _W["agent"].act(EngineView(a))
            j = kb.index(ka[i]) if ka[i] in kb else [loose(o) for o in b.opts].index(loose(a.opts[i]))
            a.step(i); b.step(j)
            out["decisions"] += 1
            if out["decisions"] > 3000:
                out["rules"] = {"at": out["decisions"], "what": "decision limit"}
                break
        out.update(done=[a.done, b.done], winner=[a.winner if a.done else None, b.winner if b.done else None],
                   turns=[a.duel.turn, b.duel.turn])
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    out["secs"] = round(time.time() - t0, 1)
    return out


def main():
    outp, per, procs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 16
    jobs, count = [], {}
    for line in open(BOOK):
        r = json.loads(line)
        d = r["decks"][0]["name"]
        if count.get(d, 0) < per:
            count[d] = count.get(d, 0) + 1
            jobs.append((d, r["seed"] % 2**31))
    with mp.get_context("fork").Pool(procs, initializer=init_worker) as pool, open(outp, "w") as f:
        for res in pool.imap_unordered(play, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(res["deck"], res["seed"], res["decisions"], "order", res["order_diffs"], (res["rules"] or {}).get("kind", ""), res.get("error", ""), res.get("winner"), flush=True)


if __name__ == "__main__":
    main()
