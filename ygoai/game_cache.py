"""On-disk cache of played test games (record and both observation streams).

Playing a test game is a large part of a belief check run, and regression runs
play the same games again and again. A game is determined by its seed, settings
and decks and by everything the duel and the agents run on: the core library,
the card database, the card scripts, the Python code that plays and observes
games and a few environment variables. The cache key covers all of them (files
by content, the many card scripts by size and modification time), so a change
to any of them plays the games anew. YGOAI_GAME_CACHE=0 turns the cache off.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import zlib
from pathlib import Path

from .paths import BABEL_CDB_DIR, CARD_SCRIPTS_DIR, CORE_LIB, ROOT

CACHE_DIR = Path(os.environ.get("YGOAI_GAME_CACHE_DIR", ROOT / "build" / "game_cache"))
_ENV = ("YGOAI_NO_OVERRIDES", "YGOAI_CORE_LIB", "YGOAI_SCRIPTS", "YGOAI_CDB_DIR", "YGOAI_DECKS",
        "YGOAI_NEW_DECKS")
_fingerprints: dict[bool, str] = {}


def enabled() -> bool:
    return os.environ.get("YGOAI_GAME_CACHE", "1") != "0"


def _python_sources(with_belief: bool) -> list[Path]:
    pkg = ROOT / "ygoai"
    out = []
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).parts
        # (the belief replay does not take part in playing a game, except for the
        # function that plays the test games and the belief_mc agent)
        if rel[0] == "belief" and not with_belief and path.name != "check.py":
            continue
        out.append(path)
    return out


def fingerprint(with_belief: bool = False) -> str:
    """A hash of everything a played game depends on besides its own settings."""
    if with_belief in _fingerprints:
        return _fingerprints[with_belief]
    h = hashlib.sha256()
    for path in [CORE_LIB] + _python_sources(with_belief):
        h.update(str(path).encode())
        h.update(path.read_bytes() if path.exists() else b"-")
    for d in (CARD_SCRIPTS_DIR, ROOT / "core_ext" / "lua", BABEL_CDB_DIR):
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
    return CACHE_DIR / fingerprint(with_belief) / f"{key['seed']}-{digest}.pkl.z"


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
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(zlib.compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), 6))
        tmp.replace(path)  # (atomic: parallel workers never read half a file)
    except OSError:
        pass
