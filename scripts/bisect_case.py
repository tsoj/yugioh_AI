#!/usr/bin/env python3
"""Is a failing reconstruction a regression? Checks it at older commits.

The case runs in a git worktree (build/bisect-worktree) at each commit tried,
with that commit's core built there; the card scripts, databases and decks are
the current ones. If one of the older commits tried passes, the commit that
broke the case is searched by bisection between it and the newest failing one.

    python3 scripts/bisect_case.py SEED VIEWER FRAC [check_belief flags...]
    python3 scripts/bisect_case.py 100060 0 1.0 --decks random --pairs ... --completion corpus

By default the commits tried first are 5, 15, 40 and 100 commits back.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKTREE = ROOT / "build" / "bisect-worktree"


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def passes(commit: str, seed: int, viewer: int, frac: float, flags: list[str], timeout: int) -> bool | None:
    """Whether the case is consistent at `commit` (None: it could not be run there)."""
    if not WORKTREE.exists():
        git("worktree", "add", "-f", str(WORKTREE), commit)
    git("checkout", "-q", "-f", commit, cwd=WORKTREE)
    core = WORKTREE / "external" / "ygopro-core"
    if core.is_dir() and not core.is_symlink() and not any(core.iterdir()):
        core.rmdir()
    if not core.exists():
        core.symlink_to(ROOT / "external" / "ygopro-core")
    if subprocess.run(["bash", "scripts/build_core.sh"], cwd=WORKTREE, capture_output=True).returncode != 0:
        return None
    env = dict(os.environ, YGOAI_CORE_LIB=str(WORKTREE / "build" / "libocgcore.so"),
               YGOAI_SCRIPTS=str(ROOT / "external" / "CardScripts"), YGOAI_CDB_DIR=str(ROOT / "external" / "BabelCDB"),
               YGOAI_DECKS=str(ROOT / "download_human_decks" / "decks"),
               YGOAI_NEW_DECKS=str(ROOT / "download_human_decks" / "decks_new"), YGOAI_GAME_CACHE="0")
    out = WORKTREE / "bisect_case.jsonl"
    out.unlink(missing_ok=True)
    flags = [str(ROOT / f) if f.endswith(".jsonl") and not f.startswith("/") else f for f in flags]
    r = subprocess.run([sys.executable, "scripts/check_belief.py", "--seeds", str(seed), "--fractions", str(frac),
                        "--turns", "", "--timeout", str(timeout), "-j", "1", "-o", str(out), *flags],
                       cwd=WORKTREE, env=env, capture_output=True, text=True)
    if not out.exists():
        return None
    rows = [json.loads(l) for l in out.open()]
    row = next((x for x in rows if x.get("viewer") == viewer), None)
    return None if row is None else bool(row["ok"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("viewer", type=int)
    ap.add_argument("frac", type=float)
    ap.add_argument("--back", default="5,15,40,100", help="commits back to try first")
    ap.add_argument("--timeout", type=int, default=180)
    args, flags = ap.parse_known_args()  # (the rest: check_belief's flags of the case)
    args.flags = flags
    commits = git("rev-list", "--first-parent", "HEAD").split()
    run = lambda c: passes(c, args.seed, args.viewer, args.frac, args.flags, args.timeout)  # noqa: E731
    bad = 0  # index into commits (0 = HEAD), known failing
    if run(commits[0]) is not False:
        print("passes (or cannot run) at HEAD: nothing to bisect")
        return 0
    good = None
    for back in (int(b) for b in args.back.split(",")):
        if back >= len(commits):
            break
        ok = run(commits[back])
        print(f"{back} commits back ({commits[back][:8]}): {'passes' if ok else 'fails' if ok is False else 'cannot run'}",
              flush=True)
        if ok:
            good = back
            break
        if ok is False:
            bad = back
    if good is None:
        print("not a regression (fails at every older commit tried): a new failure")
        return 1
    while good - bad > 1:
        mid = (good + bad) // 2
        ok = run(commits[mid])
        print(f"{commits[mid][:8]}: {'passes' if ok else 'fails' if ok is False else 'cannot run'}", flush=True)
        if ok:
            good = mid
        else:
            bad = mid  # (a commit that cannot run counts as failing: the search stays sound on its side)
    print(f"regression: broken by {git('log', '-1', '--format=%h %s', commits[bad])}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
