#!/usr/bin/env python3
"""Move .ydk decks that reference unknown card codes (or violate basic
size constraints) into a quarantine subdirectory.

Mirrors the validation duelfish does at load time (DeckBook +
CardDb::lookup): main 40-60, extra 0-15, side 0-15, and every card
code must resolve in cards.cdb. Decks that fail any check are moved
to `decks_invalid/` next to `decks/` (or `--invalid-dir DIR`).

Usage:
    ./filter_decks.py                          # use defaults
    ./filter_decks.py --decks-dir decks --cdb ../external/BabelCDB/cards.cdb
    ./filter_decks.py --dry-run                # report counts, move nothing
"""

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path


def parse_ydk(path):
    """Parse a .ydk file into (main, extra, side) lists of card codes."""
    main, extra, side = [], [], []
    section = main
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line == "#main":
                    section = main
                elif line == "#extra":
                    section = extra
                continue
            if line.startswith("!"):
                if line == "!side":
                    section = side
                continue
            try:
                section.append(int(line))
            except ValueError:
                raise ValueError(f"line {raw!r} is not a card code")
    return main, extra, side


def size_check(main, extra, side):
    if not (40 <= len(main) <= 60):
        return f"main deck size {len(main)} outside [40, 60]"
    if len(extra) > 15:
        return f"extra deck size {len(extra)} exceeds 15"
    if len(side) > 15:
        return f"side deck size {len(side)} exceeds 15"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--decks-dir", default=here / "decks",
                    help="directory of .ydk files (default: decks/)")
    ap.add_argument("--invalid-dir", default=None,
                    help="quarantine directory (default: <decks-dir>_invalid/)")
    ap.add_argument("--cdb", default=Path(__file__).resolve().parent.parent / "external" / "BabelCDB" / "cards.cdb",
                    help="path to cards.cdb")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be moved, don't actually move")
    args = ap.parse_args()

    decks_dir = Path(args.decks_dir)
    if not decks_dir.is_dir():
        sys.exit(f"decks dir not found: {decks_dir}")
    invalid_dir = (Path(args.invalid_dir) if args.invalid_dir
                   else decks_dir.with_name(decks_dir.name + "_invalid"))
    cdb_path = Path(args.cdb)
    if not cdb_path.is_file():
        sys.exit(f"cards.cdb not found: {cdb_path}")

    # Slurp every known card id into a set — much faster than per-row lookups
    # for the ~170 K decks the human corpus has.
    db = sqlite3.connect(f"file:{cdb_path}?mode=ro", uri=True)
    known = {row[0] for row in db.execute("SELECT id FROM datas")}
    db.close()
    print(f"loaded {len(known):,} card codes from {cdb_path}")

    ydk_files = sorted(decks_dir.glob("*.ydk"))
    print(f"scanning {len(ydk_files):,} decks in {decks_dir}")

    n_ok = n_parse_err = n_size_err = n_unknown_card = 0
    invalid = []

    for path in ydk_files:
        try:
            main_d, extra, side = parse_ydk(path)
        except Exception as e:
            invalid.append((path, f"parse: {e}"))
            n_parse_err += 1
            continue
        size_err = size_check(main_d, extra, side)
        if size_err:
            invalid.append((path, size_err))
            n_size_err += 1
            continue
        bad = None
        for code in main_d + extra + side:
            if code not in known:
                bad = code
                break
        if bad is not None:
            invalid.append((path, f"unknown card code {bad}"))
            n_unknown_card += 1
            continue
        n_ok += 1

    print(f"valid:                   {n_ok:>7,}")
    print(f"invalid (parse error):   {n_parse_err:>7,}")
    print(f"invalid (size):          {n_size_err:>7,}")
    print(f"invalid (unknown card):  {n_unknown_card:>7,}")
    print(f"invalid (total):         {len(invalid):>7,}")

    if not invalid:
        print("nothing to quarantine")
        return

    if args.dry_run:
        print(f"\n--dry-run: would move {len(invalid)} decks to {invalid_dir}")
        for path, reason in invalid[:20]:
            print(f"  {path.name}: {reason}")
        if len(invalid) > 20:
            print(f"  ... and {len(invalid) - 20} more")
        return

    invalid_dir.mkdir(exist_ok=True)
    for path, _reason in invalid:
        dest = invalid_dir / path.name
        # If there's a name collision (re-run with the same files), skip.
        if dest.exists():
            continue
        shutil.move(str(path), str(dest))
    print(f"moved {len(invalid)} decks to {invalid_dir}")


if __name__ == "__main__":
    main()
