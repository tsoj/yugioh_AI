"""Live table of the matches: every arbiter log (SPRTs, WindBot and decklight matches) and every
ygo-edopro-client result file in runs/ changed in the last hours, redrawn in place under the prompt
(no full-screen mode, no scrolling): running matches first, then the latest finished ones.

  python3 scripts/live.py [--since HOURS] [--interval SECONDS] [--rows N] [--all] [--once] [FILE ...]

Arbiter logs (*.log with "[   12s] games N: ..." lines): games, Elo +/- its interval, LLR, the SPRT
verdict. Client results (*.jsonl with "won" and "tag" per duel; an opening's two seat orders paired):
duels, score and Elo with a 95% interval over the complete openings. A match is running while its file
changed in the last 2 minutes.
"""

import argparse
import collections
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = re.compile(r"^\[\s*(\d+)s\] games (\d+): .*?elo ([-+]?[0-9.]+) \+/- ([0-9.]+)(?:.*?LLR ([-+]?[0-9.]+))?")
FINAL = re.compile(r"^final: games (\d+): .*?elo ([-+]?[0-9.]+) \+/- ([0-9.]+)(?:.*?LLR ([-+]?[0-9.]+))?")
VERDICT = re.compile(r"^SPRT finished: (H[01]) accepted")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RUNNING_S = 120


class C:
    on = sys.stdout.isatty()

    @staticmethod
    def wrap(code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if C.on else s


green = lambda s: C.wrap("32", s)
red = lambda s: C.wrap("31", s)
yellow = lambda s: C.wrap("33", s)
dim = lambda s: C.wrap("2", s)
bold = lambda s: C.wrap("1", s)


@dataclass
class Row:
    name: str
    age: float
    games: int
    elo: float | None = None
    err: float | None = None  # (+/- of the Elo)
    lo: float | None = None
    hi: float | None = None
    llr: float | None = None
    verdict: str | None = None
    done: bool = False
    live: bool = False  # (a process works on it: its records or results path on a command line)

    @property
    def running(self) -> bool:
        return not self.done and (self.live or self.age < RUNNING_S)


def elo_of(s: float) -> float:
    s = min(max(s, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / s - 1)


def tail_lines(path: Path, size: int = 65536) -> list[str]:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        n = f.tell()
        f.seek(max(0, n - size))
        return f.read().decode(errors="replace").splitlines()


def arbiter_row(path: Path, name: str, age: float) -> Row | None:
    row = None
    verdict = None
    for line in tail_lines(path):
        if m := PROGRESS.match(line):
            row = Row(name, age, int(m[2]), float(m[3]), float(m[4]), llr=float(m[5]) if m[5] else None)
        elif m := FINAL.match(line):
            row = Row(name, age, int(m[1]), float(m[2]), float(m[3]), llr=float(m[4]) if m[4] else None, done=True)
        elif m := VERDICT.match(line):
            verdict = m[1]
    if row is not None and verdict:
        row.verdict, row.done = verdict, True
    return row


def client_row(path: Path, name: str, age: float) -> Row | None:
    pairs = collections.defaultdict(list)
    n = 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict) or "won" not in r or "tag" not in r:
                return None
            n += 1
            if r["won"] is not None:
                *opening, _order = r["tag"].split()
                pairs[tuple(opening)].append(float(r["won"]))
    if n == 0:
        return None
    # (no end marker in these files: finished when no process writes it and every opening has both seat orders)
    row = Row(name, age, n, done=bool(pairs) and all(len(v) == 2 for v in pairs.values()))
    sc = [sum(v) / len(v) for v in pairs.values() if len(v) == 2]
    if len(sc) >= 2:
        m = sum(sc) / len(sc)
        se = (sum((x - m) ** 2 for x in sc) / (len(sc) - 1)) ** 0.5 / len(sc) ** 0.5
        row.elo, row.lo, row.hi = elo_of(m), elo_of(m - 1.96 * se), elo_of(m + 1.96 * se)
    return row


def command_lines() -> list[str]:
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                out.append((d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
            except OSError:
                pass
    return out


def is_live(p: Path, cmds: list[str]) -> bool:
    """A process names this file's path without its suffix (an arbiter its records next to its log, the
    EDOPro client its results), followed by a suffix."""
    stems = {str(p.resolve().with_suffix(""))}
    if p.resolve().is_relative_to(ROOT):
        stems.add(str(p.resolve().relative_to(ROOT).with_suffix("")))
    return any(f"{st}." in c for c in cmds for st in stems)


def collect(args) -> list[Row]:
    now = time.time()
    cmds = command_lines()
    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        cutoff = now - args.since * 3600
        paths = []
        for p in (ROOT / "runs").rglob("*"):
            # (test directories, e.g. runs/decklight/search_test/: only with --all)
            if p.suffix in (".log", ".jsonl") and (args.all or not any("test" in part for part in p.relative_to(ROOT / "runs").parts[:-1])):
                try:
                    if p.is_file() and p.stat().st_mtime >= cutoff:
                        paths.append(p)
                except OSError:
                    pass
    rows = []
    for p in paths:
        try:
            age = now - p.stat().st_mtime
            rel = p.resolve().relative_to(ROOT / "runs") if p.resolve().is_relative_to(ROOT / "runs") else p
            name = str(rel.with_suffix(""))
            row = arbiter_row(p, name, age) if p.suffix == ".log" else client_row(p, name, age)
        except OSError:
            continue
        if row is not None:
            row.live = is_live(p, cmds)
            if row.live and p.suffix == ".jsonl":
                row.done = False
            rows.append(row)
    running = sorted((r for r in rows if r.running), key=lambda r: r.name)
    # (idle: neither running nor finished, e.g. a stopped match or an old test's results: --all shows them)
    rest = sorted((r for r in rows if r not in running and (r.done or args.all)), key=lambda r: r.age)
    return (running + rest)[: args.rows]


def ago(s: float) -> str:
    return f"{s:.0f}s" if s < 60 else f"{s / 60:.0f}m" if s < 3600 else f"{s / 3600:.1f}h"


def fmt(row: Row, width_name: int) -> str:
    running = row.running
    state = yellow("● run ") if running else (dim("✓ done") if row.done else dim("· idle"))
    name = row.name if len(row.name) <= width_name else "…" + row.name[-(width_name - 1):]
    name = f"{name:<{width_name}}"
    name = bold(name) if running else name
    if row.elo is None:
        elo = dim(f"{'-':>8}")
        ci = " " * 18
    else:
        sig = (row.lo is not None and (row.lo > 0 or row.hi < 0)) or (row.err is not None and abs(row.elo) > row.err)
        col = (green if row.elo > 0 else red) if sig else (lambda s: s)
        elo = col(f"{row.elo:+8.1f}")
        ci = f"[{row.lo:+5.0f}, {row.hi:+5.0f}]" if row.lo is not None else f"± {row.err:<6.1f}"
        ci = f"{ci:<18}"
    llr = ""
    if row.verdict:
        llr = (green if row.verdict == "H1" else red)(f"{row.verdict:>8}")
    elif row.llr is not None:
        llr = f"LLR {row.llr:+5.2f}"
    return f"{state} {name} {row.games:>6} {elo} {ci} {llr:<9} {dim(ago(row.age)):>5}"


def render(args) -> list[str]:
    rows = collect(args)
    cols = shutil.get_terminal_size((120, 24)).columns
    width_name = max(20, min(56, cols - 62))
    head = f"{'':6} {'match':<{width_name}} {'games':>6} {'elo':>8} {'95% / ±':<18} {'LLR':<9} {'age':>5}"
    lines = [dim(time.strftime("%H:%M:%S") + f"  matches changed in the last {args.since:g} h (Ctrl-C quits)"), dim(head)]
    lines += [fmt(r, width_name) for r in rows] or [dim("no matches found")]
    out = []
    for line in lines:  # (cut to the terminal's width: a wrapped line would break the redraw)
        visible = ANSI.sub("", line)
        if len(visible) > cols - 1:
            line = visible[: cols - 2] + "…"
        out.append(line)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="only these files (default: runs/ changed in the last --since hours)")
    ap.add_argument("--since", type=float, default=6.0, help="hours (default 6)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between redraws (default 5)")
    ap.add_argument("--rows", type=int, default=12, help="at most this many matches (default 12)")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--all", action="store_true", help="also the idle ones (neither running nor finished)")
    args = ap.parse_args()
    if args.once or not sys.stdout.isatty():
        print("\n".join(render(args)))
        return
    drawn = 0
    sys.stdout.write("\x1b[?25l")  # (cursor hidden while drawing)
    try:
        while True:
            lines = render(args)
            # (back to the table's first line, its old lines cleared, the new ones drawn)
            sys.stdout.write((f"\x1b[{drawn}F" if drawn else "") + "\x1b[J" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            drawn = len(lines)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
