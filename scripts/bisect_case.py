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


def passes(commit: str, seed: int, viewer: int, frac: float, flags: list[str], timeout: int,
           game: dict | None = None) -> bool | None:
    """Whether the case is consistent at `commit` (None: it could not be run there,
    or that commit plays another game for the seed, e.g. after a core change: then
    its result says nothing about this case). `game` collects the game's identity
    when empty and is compared with otherwise."""
    wt = WORKTREE
    if not wt.exists():
        git("worktree", "add", "-f", str(wt), commit)
    git("checkout", "-q", "-f", commit, cwd=wt)
    core = wt / "external" / "ygopro-core"
    if core.is_dir() and not core.is_symlink() and not any(core.iterdir()):
        core.rmdir()
    if not core.exists():
        core.symlink_to(ROOT / "external" / "ygopro-core")
    if subprocess.run(["bash", "scripts/build_core.sh"], cwd=wt, capture_output=True).returncode != 0:
        return None
    env = dict(os.environ, YGOAI_CORE_LIB=str(wt / "build" / "libocgcore.so"),
               YGOAI_SCRIPTS=str(ROOT / "external" / "CardScripts"), YGOAI_CDB_DIR=str(ROOT / "external" / "BabelCDB"),
               YGOAI_DECKS=str(ROOT / "download_human_decks" / "decks"),
               YGOAI_NEW_DECKS=str(ROOT / "download_human_decks" / "decks_new"), YGOAI_GAME_CACHE="0")
    out = wt / "bisect_case.jsonl"
    out.unlink(missing_ok=True)
    # (only this reconstruction: the case's viewer at the case's cut)
    cases = wt / "bisect_cases.jsonl"
    cases.write_text(json.dumps({"seed": seed, "viewer": viewer, "frac": frac}) + "\n")
    flags = [str(ROOT / f) if f.endswith(".jsonl") and not f.startswith("/") else f for f in flags]
    r = subprocess.run([sys.executable, "scripts/check_belief.py", "--cases", str(cases),
                        "--timeout", str(timeout), "-j", "1", "-o", str(out), *flags],
                       cwd=wt, env=env, capture_output=True, text=True)
    if not out.exists():
        return None
    rows = [json.loads(l) for l in out.open()]
    row = next((x for x in rows if x.get("viewer") == viewer), None)
    if row is None:
        return None
    ident = {k: row[k] for k in ("game", "len") if k in row}
    if game is not None:
        if not game:
            game.update(ident)
        elif any(game.get(k) != v for k, v in ident.items() if k in game):
            return None  # (another game: no evidence either way)
    return bool(row["ok"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("viewer", type=int)
    ap.add_argument("frac", type=float)
    ap.add_argument("--back", default="5,15,40,100", help="commits back to try first")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--at", default="HEAD", help="the commit the case fails at (default HEAD)")
    ap.add_argument("--worktree", type=Path, help="the git worktree to use (several bisections at once)")
    args, flags = ap.parse_known_args()  # (the rest: check_belief's flags of the case)
    global WORKTREE
    if args.worktree is not None:
        WORKTREE = args.worktree.resolve()
    args.flags = flags
    commits = git("rev-list", "--first-parent", args.at).split()
    game: dict = {}
    run = lambda c: passes(c, args.seed, args.viewer, args.frac, args.flags, args.timeout, game)  # noqa: E731
    bad = 0  # index into commits (0 = HEAD), known failing
    if run(commits[0]) is not False:
        print(f"passes (or cannot run) at {args.at}: nothing to bisect")
        return 0
    good = None
    for back in (int(b) for b in args.back.split(",")):
        if back >= len(commits):
            break
        ok = run(commits[back])
        print(f"{back} commits back ({commits[back][:8]}): {'passes' if ok else 'fails' if ok is False else 'cannot run / another game'}",
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
        print(f"{commits[mid][:8]}: {'passes' if ok else 'fails' if ok is False else 'cannot run / another game'}", flush=True)
        if ok:
            good = mid
        else:
            bad = mid  # (a commit that cannot run counts as failing: the search stays sound on its side)
    print(f"regression: broken by {git('log', '-1', '--format=%h %s', commits[bad])}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
