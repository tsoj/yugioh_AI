#!/usr/bin/env python3
"""SPRT of one engine version against another, both given as git refs.

  scripts/sprt.py NEW [BASE] [--agent alphabeta] [--tc 30+0.3] [--elo 0 10]

Each ref's code (ygoai/, scripts/, core_ext/, its core library) is exported to
build/snapshots/engine-<sha>; the engines run from there as engine protocol
processes (python -m ygoai.engine AGENT), so both versions play in one match
(scripts/match.py: random deck pairs from the corpus, both seats, Fischer clock
per decision). BASE defaults to main. The log and records go to runs/sprt/, a
summary line to runs/sprt/results.jsonl.

The usual loop: a feature on a branch, `scripts/sprt.py <branch>` against main;
if H1 is accepted, merge it into main.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import snapshot  # noqa: E402

OUT = ROOT / "runs" / "sprt"


def resolve(ref: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"], check=True,
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new")
    ap.add_argument("base", nargs="?", default="main")
    ap.add_argument("--agent", default="alphabeta", help="the built-in agent both versions run")
    ap.add_argument("--base-agent", help="the base version's agent (default: --agent)")
    ap.add_argument("--tc", default="30+0.3")
    ap.add_argument("--elo", type=float, nargs=2, default=(0.0, 10.0), metavar=("ELO0", "ELO1"))
    ap.add_argument("-n", "--openings", type=int, default=5000, help="at most (the SPRT usually ends earlier)")
    ap.add_argument("-c", "--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    shas = {}
    for side, ref in (("new", args.new), ("base", args.base)):
        sha = resolve(ref)
        shas[side] = sha
        if not (snapshot.SNAPSHOTS / f"engine-{sha[:12]}" / "ygoai").exists():
            print(f"exporting {ref} ({sha[:12]})", flush=True)
            snapshot.make(f"engine-{sha[:12]}", sha)
    python = ROOT / ".venv" / "bin" / "python"
    agents = {"new": args.agent, "base": args.base_agent or args.agent}

    def engine(side: str) -> str:
        return (f"{side}=env -C {snapshot.SNAPSHOTS / ('engine-' + shas[side][:12])} {python} "
                f"-m ygoai.engine {agents[side]}")
    name = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', args.new)}_vs_{re.sub(r'[^A-Za-z0-9_.-]', '_', args.base)}_" \
           f"{time.strftime('%Y%m%d-%H%M%S')}"
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{name}.log"
    cmd = [str(python), str(ROOT / "scripts" / "match.py"), "--engine", engine("new"), "--engine", engine("base"),
           "-n", str(args.openings), "-c", str(args.concurrency), "--mirror-prob", "0", "--tc", args.tc,
           "--sprt", str(args.elo[0]), str(args.elo[1]), "--seed", str(args.seed),
           "--records", str(OUT / f"{name}.jsonl")]
    print(f"{args.new} ({shas['new'][:12]}) vs {args.base} ({shas['base'][:12]}): log {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        lines = []
        for line in proc.stdout:
            f.write(line)
            f.flush()
            lines.append(line.rstrip())
            if line.startswith(("SPRT", "final")) or "games" in line and len(lines) % 10 == 0:
                print(line.rstrip(), flush=True)
        proc.wait()
    final = next((x for x in reversed(lines) if x.startswith("final")), "")
    verdict = next((x for x in reversed(lines) if x.startswith("SPRT")), "SPRT not finished")
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps({"new": args.new, "new_sha": shas["new"], "base": args.base, "base_sha": shas["base"],
                            "agent": agents, "tc": args.tc, "elo": args.elo, "verdict": verdict, "final": final,
                            "seconds": round(time.time() - t0), "log": str(log)}) + "\n")
    print(verdict)
    return 0 if "H1 accepted" in verdict else 1


if __name__ == "__main__":
    sys.exit(main())
