"""Long test runs from a frozen copy of the code.

A suite or exploration run takes an hour or more and its worker processes are
recycled, importing the code again: an edit during the run would mix versions.
`reexec` copies the code (ygoai/, scripts/, lua_fixes/, the Rust engine's binding
ygoai_rs) to
build/snapshots/NAME and runs the calling script again from there; the data
(external/, the decks, tests/, runs/ and the rest of build/, e.g. the game cache)
are symlinks to the checkout's, so the results land where they always do and the
game cache is shared (its fingerprint covers file contents, which are the same).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "build" / "snapshots"
_CODE = ("ygoai", "scripts", "lua_fixes")
_SHARED = ("external", "download_human_decks", "tests", "runs")
_NOT_SHARED = ("snapshots",)
_RUST = ("crates", "ygoai_rs", "Cargo.toml", "Cargo.lock")  # (the binding's sources besides the rust/ submodule)
_ENV = "YGOAI_SNAPSHOT"  # (set in the snapshot's process: the checkout it was taken from)
_KEEP_DAYS = 3


def reexec(name: str, ref: str | None = None) -> None:
    """Runs the calling script again from a fresh snapshot of the code, of the
    checkout or of the git commit `ref` (nothing to do when already running from
    one). Keeps the process id."""
    if os.environ.get(_ENV):
        return
    dst = make(name, ref)
    script = dst / Path(sys.argv[0]).resolve().relative_to(ROOT)
    print(f"running from the snapshot {dst}", flush=True)
    os.execve(sys.executable, [sys.executable, str(script), *sys.argv[1:]], dict(os.environ, **{_ENV: str(ROOT)}))


def make(name: str, ref: str | None = None) -> Path:
    """A snapshot of the code in build/snapshots/NAME: of the checkout, or of a
    git commit `ref` (with its own build of the Rust binding if its Rust sources
    differ from the checkout's)."""
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    for old in SNAPSHOTS.iterdir():
        if old.is_dir() and time.time() - old.stat().st_mtime > _KEEP_DAYS * 86400:
            shutil.rmtree(old, ignore_errors=True)
    dst = SNAPSHOTS / name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    if ref is None:
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for d in _CODE:
            shutil.copytree(ROOT / d, dst / d, ignore=ignore, symlinks=True)
    else:
        code = [d for d in _CODE if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{ref}:{d}"],
                                                     capture_output=True).returncode == 0]
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", ref, *code], check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(dst)], input=archive, check=True)
    for d in _SHARED:
        if (ROOT / d).exists():
            (dst / d).symlink_to(ROOT / d)
    (dst / "build").mkdir()
    for p in (ROOT / "build").iterdir():
        if p.name in _NOT_SHARED or p.name.startswith("bisect"):
            continue
        (dst / "build" / p.name).symlink_to(p)
    _rust_binding(dst, ref)
    return dst


def _installed_binding() -> Path | None:
    try:
        import importlib.util
        spec = importlib.util.find_spec("ygoai_rs")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith("__init__.py"):
        return None
    return Path(spec.origin).parent


def _rust_binding(dst: Path, ref: str | None) -> None:
    """The Rust engine's binding in the snapshot (python -m from there imports it first): the
    installed one, or for a commit whose Rust sources differ from the checkout's, its own build."""
    installed = _installed_binding()
    if installed is None:
        return
    shutil.copytree(installed, dst / "ygoai_rs", ignore=shutil.ignore_patterns("__pycache__"))
    if ref is None:
        return
    sub = lambda r: subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{r}:rust"], capture_output=True,  # noqa: E731
                                   text=True).stdout.strip()
    same = sub(ref) == sub("HEAD") and not subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", ref, "--", *_RUST]).returncode
    if same or not sub(ref):
        return
    # (its own: the sources of the commit, the engine at its submodule commit, built in a capped scope)
    src = SNAPSHOTS / f"{dst.name}-src"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir()
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", ref, *_RUST], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(src)], input=archive, check=True)
    (src / "rust").mkdir()
    archive = subprocess.run(["git", "-C", str(ROOT / "rust"), "archive", sub(ref)], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(src / "rust")], input=archive, check=True)
    target = ROOT / "build" / "rust-snapshot-target"
    cargo = ["cargo", "build", "--release", "-q", "-p", "ygoai_rs"]
    if shutil.which("systemd-run"):
        cargo = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=20G", "-p", "MemorySwapMax=0", *cargo]
    subprocess.run(cargo, cwd=src, check=True, env={**os.environ, "CARGO_TARGET_DIR": str(target), "CARGO_BUILD_JOBS": "8"})
    shutil.copy2(target / "release" / "libygoai_rs.so", dst / "ygoai_rs" / "ygoai_rs.abi3.so")
    shutil.rmtree(src)
