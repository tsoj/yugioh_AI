#!/usr/bin/env python3
"""Replays recorded games with YGOAI_ENGINE=tee: the C++ core and the Rust engine in lockstep
(ygoai/core_tee.py), compared at every question and every other call.

  .venv/bin/python scripts/tee_check.py RECORDS.jsonl [more.jsonl ...] [--first N] [--every K]

Mismatches are listed and logged (with every call, to rebuild the case) to YGOAI_TEE_LOG
(default build/tee_mismatches.jsonl).
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["YGOAI_ENGINE"] = "tee"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ygoai.core_tee import TeeMismatch  # noqa: E402
from ygoai.game import GameRecord, replay  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d  # noqa: E731
    first, every = opt("--first", 10**9), opt("--every", 1)
    n = bad = other = unsynced = 0
    t0 = time.time()
    for path in (a for a in args if a.endswith(".jsonl")):
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= first:
                    break
                if i % every:
                    continue
                d = json.loads(line)
                if "responses" not in d:
                    continue
                n += 1
                try:
                    duel = replay(GameRecord.from_json(d), [])
                    unsynced += duel.unsynced_calls
                    duel.close()
                except TeeMismatch as e:
                    bad += 1
                    print(f"  {os.path.basename(path)}:{i}: {e}", flush=True)
                except Exception as e:  # noqa: BLE001
                    other += 1
                    print(f"  {os.path.basename(path)}:{i}: replay failed: {e!r:.200}", flush=True)
    print(f"{n} games in {time.time() - t0:.0f}s: {n - bad - other} identical, {bad} differ, "
          f"{other} failed to replay; {unsynced} calls between questions (not compared)")


if __name__ == "__main__":
    main()
