#!/usr/bin/env python3
"""Rebuilds a case of build/tee_mismatches.jsonl (YGOAI_ENGINE=tee) on the C++ core and the Rust
engine separately, up to the call before the mismatch, and shows what both engines say there.

  .venv/bin/python scripts/tee_repro.py [LOG] [--entry N] [--upto K]

Then the duels are left in `a` (C++) and `b` (Rust) for -i inspection:
  .venv/bin/python -i scripts/tee_repro.py ...
"""
from __future__ import annotations

import json
import os
import sys

os.environ.pop("YGOAI_ENGINE", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ygoai import core  # noqa: E402
from ygoai.core import DuelConfig  # noqa: E402,F401  (for eval)
from ygoai.core_rust import Duel as RustDuel  # noqa: E402

CONTINUE = 2


def rebuild(entry: dict, upto: int):
    calls = entry["calls"]
    _, args = calls[0]
    seed, config = eval(args)
    a, b = core.CppDuel(seed, config), RustDuel(seed, config)
    na = nb = 0  # (messages sent since both were at the same point)
    rust_status = CONTINUE
    for name, args in calls[1:upto]:
        args = eval(args)
        if name == "process":
            # (as the tee: the C++ core one step, the Rust engine on as far)
            s = a.process()
            na += len(a.get_messages())
            while rust_status == CONTINUE and (s != CONTINUE or nb < na):
                rust_status = b.process()
                nb += len(b.get_messages())
            n = min(na, nb)
            na, nb = na - n, nb - n
            if s != CONTINUE:
                rust_status = CONTINUE
            rebuild.ahead = (na, nb, rust_status)
            continue
        if isinstance(args, tuple) and len(args) == 2 and isinstance(args[1], dict):
            args, kwargs = args
        else:
            kwargs = {}
        getattr(a, name)(*args, **kwargs)
        getattr(b, name)(*args, **kwargs)
    return a, b


def run_to_question(d) -> tuple[int, list[bytes]]:
    msgs = []
    while True:
        s = d.process()
        msgs += d.get_messages()
        if s != CONTINUE:
            return s, msgs


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = lambda k, dflt: type(dflt)(args[args.index(k) + 1]) if k in args else dflt  # noqa: E731
    log = next((x for x in args if x.endswith(".jsonl")), "build/tee_mismatches.jsonl")
    entries = [json.loads(line) for line in open(log)]
    entry = entries[opt("--entry", 0)]
    upto = opt("--upto", entry["call"])
    print(entry["what"][:300])
    a, b = rebuild(entry, upto)
    print(f"rebuilt up to call {upto} ({entry['calls'][upto][0]} next); unmatched messages C++/Rust, Rust status: {getattr(rebuild, 'ahead', None)}")
    for name, d in (("C++ ", a), ("Rust", b)):
        print(f"{name}: info {d.info()}, select_chains {d.select_chains()}, pending_triggers {d.pending_triggers()}")
    sa, ma = run_to_question(a)
    sb, mb = run_to_question(b)
    print("C++  ->", sa, [m.hex() for m in ma][:12])
    print("Rust ->", sb, [m.hex() for m in mb][:12])
