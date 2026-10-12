"""On-disk cache of played test games (record and both observation streams).

Playing a test game is a large part of a belief check run, and regression runs
play the same games again and again. A game is determined by its seed, settings
and decks and by everything the duel and the agents run on: the Rust engine and
binding sources, the card database, the card scripts, the Python code that plays and observes
games and a few environment variables. The cache key covers all of them (files
by content, the many card scripts by size and modification time), so a change
to any of them plays the games anew. YGOAI_GAME_CACHE=0 turns the cache off.

Games stored under an older fingerprint are never loaded again (unless the code
goes back to that state, e.g. to compare with the previous commit), so only the
KEEP most recently used fingerprints are kept; the others are deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
import zlib
from pathlib import Path

from .paths import BABEL_CDB_DIR, CARD_SCRIPTS_DIR, FIXED_CARD_SCRIPTS_DIR, ROOT

CACHE_DIR = Path(os.environ.get("YGOAI_GAME_CACHE_DIR", ROOT / "build" / "game_cache"))
_ENV = ("YGOAI_SCRIPTS", "YGOAI_CDB_DIR", "YGOAI_DECKS", "YGOAI_NEW_DECKS")
_fingerprints: dict[bool, str] = {}
KEEP = 3  # most recently used fingerprints kept
_used: set[str] = set()  # fingerprint directories this process has marked as used


def enabled() -> bool:
    return os.environ.get("YGOAI_GAME_CACHE", "1") != "0"


def _python_sources(with_belief: bool) -> list[Path]:
    pkg = ROOT / "ygoai"
    out = []
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).parts
        # (the belief replay does not take part in playing a game, except for the
        # module that plays the test games and the belief_mc agent)
        if rel[0] == "belief" and not with_belief and path.name != "games.py":
            continue
        out.append(path)
    return out


def fingerprint(with_belief: bool = False) -> str:
    """A hash of everything a played game depends on besides its own settings."""
    if with_belief in _fingerprints:
        return _fingerprints[with_belief]
    h = hashlib.sha256()
    for path in _python_sources(with_belief):
        h.update(str(path).encode())
        h.update(path.read_bytes() if path.exists() else b"-")
    # (the Rust code: hashes of its sources, made when the binding was built; the belief modules
    # only for games of agents that use them)
    import ygoai_rs
    engine, belief = ygoai_rs.source_hashes()
    h.update(f"rust {engine} {belief if with_belief else ''}\n".encode())
    for d in (CARD_SCRIPTS_DIR, FIXED_CARD_SCRIPTS_DIR, BABEL_CDB_DIR):
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames.sort()
            for name in sorted(filenames):
                st = os.stat(os.path.join(dirpath, name))
                h.update(f"{dirpath}/{name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    h.update(json.dumps({k: os.environ.get(k) for k in _ENV}).encode())
    _fingerprints[with_belief] = h.hexdigest()[:20]
    return _fingerprints[with_belief]


def _path(key: dict, with_belief: bool) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:24]
    d = CACHE_DIR / fingerprint(with_belief)
    if d.name not in _used:
        _used.add(d.name)
        _mark_used_and_prune(d)
    return d / f"{key['seed']}-{digest}.pkl.z"


def _mark_used_and_prune(d: Path) -> None:
    """Marks fingerprint directory `d` as just used (once per process) and deletes
    all but the KEEP most recently used ones."""
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / ".used").write_text(str(time.time()))
        dirs = [x for x in CACHE_DIR.iterdir() if x.is_dir()]
    except OSError:
        return

    def last_used(x: Path) -> float:
        try:
            return (x / ".used").stat().st_mtime
        except OSError:
            try:
                return x.stat().st_mtime
            except OSError:
                return 0.0
    for old in sorted(dirs, key=last_used, reverse=True)[KEEP:]:
        if old.name not in _used:
            shutil.rmtree(old, ignore_errors=True)  # (another process may be pruning it too)


def load(key: dict, with_belief: bool = False):
    """The cached value for `key`, or None."""
    if not enabled():
        return None
    path = _path(key, with_belief)
    try:
        return pickle.loads(zlib.decompress(path.read_bytes()))
    except (OSError, zlib.error, pickle.UnpicklingError, EOFError):
        return None


def store(key: dict, value, with_belief: bool = False) -> None:
    if not enabled():
        return
    path = _path(key, with_belief)
    try:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(zlib.compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), 6))
        tmp.replace(path)  # (atomic: parallel workers never read half a file)
    except OSError:
        pass
