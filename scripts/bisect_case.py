#!/usr/bin/env python3
"""Is a failing reconstruction a regression? Checks it at older commits.

The case runs from a snapshot of each commit tried (scripts/snapshot.py: the
commit's code, with its own build of the Rust binding where its Rust sources
differ); the card scripts, databases and decks are the current ones. If one of the older commits tried passes, the commit that
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
sys.path.insert(0, str(ROOT / "scripts"))
import snapshot  # noqa: E402


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def passes(commit: str, seed: int, viewer: int, frac: float, flags: list[str], timeout: int,
           game: dict | None = None) -> bool | None:
    """Whether the case is consistent at `commit` (None: it could not be run there,
    or that commit plays another game for the seed, e.g. after a core change: then
    its result says nothing about this case). `game` collects the game's identity
    when empty and is compared with otherwise."""
    try:
        wt = snapshot.make(f"bisect-{commit[:12]}", commit)
    except (subprocess.CalledProcessError, OSError):
        return None
    # (YGOAI_GAME_CACHE: commits from before the test games were played in Rust kept them in a cache)
    env = dict(os.environ, YGOAI_SNAPSHOT=str(ROOT),  # (it is one: no snapshot of the snapshot)
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
    args, flags = ap.parse_known_args()  # (the rest: check_belief's flags of the case)
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
