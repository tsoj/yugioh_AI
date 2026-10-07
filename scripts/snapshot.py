"""Long test runs from a frozen copy of the code.

A suite or exploration run takes an hour or more and its worker processes are
recycled, importing the code again: an edit during the run would mix versions.
`reexec` copies the code (ygoai/, scripts/, core_ext/, the core library) to
build/snapshots/NAME and runs the calling script again from there; the data
(external/, the decks, tests/, runs/ and the rest of build/, e.g. the game cache)
are symlinks to the checkout's, so the results land where they always do and the
game cache is shared (its fingerprint covers file contents, which are the same).
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "build" / "snapshots"
_CODE = ("ygoai", "scripts", "core_ext")
_SHARED = ("external", "download_human_decks", "tests", "runs")
_NOT_SHARED = ("snapshots", "ocgcore", "ocgcore-src", "ocgcore-exp")
_ENV = "YGOAI_SNAPSHOT"  # (set in the snapshot's process: the checkout it was taken from)
_KEEP_DAYS = 3


def reexec(name: str) -> None:
    """Runs the calling script again from a fresh snapshot of the code (nothing
    to do when already running from one). Keeps the process id."""
    if os.environ.get(_ENV):
        return
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    for old in SNAPSHOTS.iterdir():
        if old.is_dir() and time.time() - old.stat().st_mtime > _KEEP_DAYS * 86400:
            shutil.rmtree(old, ignore_errors=True)
    dst = SNAPSHOTS / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in _CODE:
        shutil.copytree(ROOT / d, dst / d, ignore=ignore, symlinks=True)
    for d in _SHARED:
        if (ROOT / d).exists():
            (dst / d).symlink_to(ROOT / d)
    (dst / "build").mkdir()
    for p in (ROOT / "build").iterdir():
        if p.name in _NOT_SHARED or p.name.startswith("bisect"):
            continue
        if p.name.startswith("libocgcore"):
            shutil.copy2(p, dst / "build" / p.name)  # (the core is code too)
        else:
            (dst / "build" / p.name).symlink_to(p)
    script = dst / Path(sys.argv[0]).resolve().relative_to(ROOT)
    print(f"running from the snapshot {dst}", flush=True)
    os.execve(sys.executable, [sys.executable, str(script), *sys.argv[1:]], dict(os.environ, **{_ENV: str(ROOT)}))
