#!/usr/bin/env python3
"""Checks the Rust port of the ygoai package (crates/ygoai, through ygoai_rs) against the Python
modules on test games: every message of a game is parsed and filtered for both players by both,
every prompt is filtered, ordered and answered (the game's response, translated and encoded) by
both, and at every prompt both players' state views and the god view snapshot are compared.

  .venv/bin/python scripts/rs_diff.py [--games N] [--start S] [--decks random|diverse|new|cards|new-mixed|new-cards] [-j J]

The games are played on the Rust engine (YGOAI_ENGINE=rust is set here). Differences are printed
(the first few per kind) and the exit status is 1 if there are any.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

os.environ["YGOAI_ENGINE"] = "rust"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import ygoai_rs  # noqa: E402

from ygoai import actions, observe, state  # noqa: E402
from ygoai.carddb import default_db  # noqa: E402
from ygoai.game import setup_duel  # noqa: E402
from ygoai.messages import ParseError, parse  # noqa: E402

PAIRS = [("random", "random"), ("random", "greedy"), ("greedy", "random"), ("greedy", "greedy")]


def norm(v) -> str:
    return json.dumps(v, sort_keys=True, default=lambda b: b.hex() if isinstance(b, bytes) else repr(b))


class Diff:
    def __init__(self, seed: int):
        self.seed = seed
        self.counts: Counter = Counter()
        self.checked: Counter = Counter()
        self.examples: list[str] = []
        self.types: Counter = Counter()

    def eq(self, kind: str, py, rs, ctx) -> None:
        self.checked[kind] += 1
        if norm(py) != norm(rs):
            self.counts[kind] += 1
            if self.counts[kind] <= 2:
                self.examples.append(f"seed {self.seed} {kind} {ctx}:\n  py {norm(py)[:600]}\n  rs {norm(rs)[:600]}")

    def call(self, kind: str, f_py, f_rs, ctx) -> None:
        """Both results (or both errors)."""
        try:
            py = ("ok", f_py())
        except (ParseError, KeyError, IndexError, ValueError) as e:
            py = ("error", type(e).__name__)
        try:
            rs = ("ok", f_rs())
        except (ValueError, KeyError, IndexError, OverflowError) as e:
            rs = ("error", "ParseError" if py[0] == "error" else repr(e))
        if py[0] == "error" and rs[0] == "error":
            rs = py
        self.eq(kind, py, rs, ctx)


def check_prompt(d: Diff, raw: bytes, msg: dict, response: dict | None, player: int, duel) -> None:
    ctx = msg["type"]
    for v in (0, 1):
        d.call("filter_prompt", lambda: observe.filter_prompt(msg, v), lambda: ygoai_rs.filter_prompt(raw, v), ctx)
        d.call("option_order", lambda: observe.option_order(msg, v), lambda: ygoai_rs.option_order(raw, v), ctx)
    d.eq("is_hidden_prompt", observe.is_hidden_prompt(msg), ygoai_rs.is_hidden_prompt(raw), ctx)
    if msg["type"] in ("SELECT_IDLECMD", "SELECT_BATTLECMD"):
        opts = actions.idle_options(msg) if msg["type"] == "SELECT_IDLECMD" else actions.battle_options(msg)
        d.eq("command_options", opts, ygoai_rs.command_options(raw), ctx)
        for o in opts:
            d.call("encode", lambda: actions.encode(msg, o), lambda: ygoai_rs.encode_response(raw, o), ctx)
    if response is not None:
        d.call("to_core_response", lambda: observe.to_core_response(msg, player, response),
               lambda: ygoai_rs.to_core_response(raw, player, response), f"{ctx} {response}")
        core = observe.to_core_response(msg, player, response)
        d.call("encode", lambda: actions.encode(msg, core), lambda: ygoai_rs.encode_response(raw, core), f"{ctx} {core}")
        if msg["type"] == "SELECT_SUM" and "indices" in core:
            d.eq("check_sum", actions.check_sum(msg, core["indices"]), ygoai_rs.check_sum(raw, core["indices"]), ctx)
    for v in (0, 1):
        d.eq("viewer_view", state.viewer_view(duel, v), duel._d.viewer_view(v), ctx)
    d.eq("capture", state.capture(duel), duel._d.capture(), ctx)


def check_messages(d: Diff, raws: list[bytes]) -> list[dict]:
    out = []
    for raw in raws:
        d.call("parse", lambda: parse(raw), lambda: ygoai_rs.parse_message(raw), raw[:1].hex())
        msg = parse(raw)
        d.types[msg["type"]] += 1
        out.append(msg)
        if not msg_is_prompt(msg):
            for v in (0, 1):
                d.call("filter_message", lambda: observe.filter_message(msg, v),
                       lambda: ygoai_rs.filter_message(raw, v), msg["type"])
    return out


def msg_is_prompt(msg: dict) -> bool:
    return msg["type"] in observe._PROMPTS


def check_game(seed: int, decks: str) -> Diff:
    from ygoai.belief.games import make_game
    d = Diff(seed)
    record = make_game(seed, pairs=PAIRS, decks=decks, observers=[])
    db = default_db()
    duel = setup_duel(record.decks, record.core_seed, record.duel_config(), record.deck_orders, db,
                      copy_seed=record.core_seed[0])
    try:
        duel.start()
        prompt_raw = prompt = None
        for entry in record.responses + [None]:
            while True:
                status = duel.process()
                raws = duel.get_messages()
                msgs = check_messages(d, raws)
                if status == 0:
                    return d
                if status == 1:
                    if msgs and msgs[-1]["type"] != "RETRY":
                        prompt_raw, prompt = raws[-1], msgs[-1]
                    break
            if entry is None:
                check_prompt(d, prompt_raw, prompt, None, prompt["player"], duel)
                return d
            check_prompt(d, prompt_raw, prompt, entry["response"], entry["player"], duel)
            duel.set_response(actions.encode(prompt, observe.to_core_response(prompt, entry["player"], entry["response"])))
    finally:
        duel.close()
    return d


class _Invent(Exception):
    pass


def fuzz_message(mtype: int, rng) -> bytes | None:
    """A random valid message of type `mtype`: the Python parser is run with a reader that
    invents each value it reads (small numbers, locations, positions, flags) and records the
    bytes."""
    import struct

    from ygoai import messages as M
    vals = [0, 1, 2, 3, 0, 1, 2, 1, 4, 5, 8, 10, 0x10, 0x20, 0x40, 0x80, 0x84, 0x0c, 0x02, 0x0e]
    out = bytearray([mtype])

    class R(M.Reader):
        def _unpack(self, fmt, size):
            if len(out) > 600:
                raise _Invent()
            r = rng.random()
            if r < 0.85:
                v = rng.choice(vals)
            elif r < 0.95:
                v = rng.getrandbits(8 * size)
            else:
                v = rng.choice(vals) | (1 << (8 * size - 1))
            if fmt[1] in "bhiq":
                v = struct.unpack(fmt, struct.pack(fmt.upper(), v))[0]
            out.extend(struct.pack(fmt, v))
            return v

        def raw(self, n):
            if n > 64:
                raise _Invent()
            b = bytes(rng.getrandbits(8) for _ in range(n))
            out.extend(b)
            return b

    try:
        M._PARSERS[mtype](R(b"", 0))
    except (_Invent, KeyError, IndexError):
        return None
    return bytes(out)


def fuzz(seed: int, n: int) -> Diff:
    import random
    from ygoai import messages as M
    rng = random.Random(seed)
    d = Diff(seed)
    for _ in range(n):
        mtype = rng.choice(sorted(M._PARSERS))
        raw = fuzz_message(mtype, rng)
        if raw is None:
            continue
        if rng.random() < 0.05:  # (broken messages: both must refuse them)
            raw = raw[:-1] if rng.random() < 0.5 else raw + b"\0"
        check_messages_fuzz(d, raw, rng)
    return d


def check_messages_fuzz(d: Diff, raw: bytes, rng) -> None:
    d.call("parse", lambda: parse(raw), lambda: ygoai_rs.parse_message(raw), raw[:1].hex())
    try:
        msg = parse(raw)
    except ParseError:
        return
    d.types[msg["type"]] += 1
    if not msg_is_prompt(msg):
        for v in (0, 1):
            d.call("filter_message", lambda: observe.filter_message(msg, v),
                   lambda: ygoai_rs.filter_message(raw, v), msg["type"])
        return
    ctx = msg["type"]
    for v in (0, 1):
        d.call("filter_prompt", lambda: observe.filter_prompt(msg, v), lambda: ygoai_rs.filter_prompt(raw, v), ctx)
        d.call("option_order", lambda: observe.option_order(msg, v), lambda: ygoai_rs.option_order(raw, v), ctx)
    d.eq("is_hidden_prompt", observe.is_hidden_prompt(msg), ygoai_rs.is_hidden_prompt(raw), ctx)
    if msg["type"] in ("SELECT_IDLECMD", "SELECT_BATTLECMD"):
        opts = actions.idle_options(msg) if msg["type"] == "SELECT_IDLECMD" else actions.battle_options(msg)
        d.eq("command_options", opts, ygoai_rs.command_options(raw), ctx)
    if msg["type"] == "SELECT_SUM" and len(msg["cards"]) + len(msg["must_cards"]) > 14:
        return  # (the sum checks are exponential in the number of cards)
    try:
        resp = actions.random_response(msg, rng, default_db())
    except (ValueError, IndexError, KeyError):
        return
    import struct
    try:
        actions.encode(msg, resp)
    except struct.error:  # (an invented player number above 1: no such response exists)
        return
    player = msg["player"]
    d.call("to_core_response", lambda: observe.to_core_response(msg, player, resp),
           lambda: ygoai_rs.to_core_response(raw, player, resp), f"{ctx} {resp}")
    d.call("encode", lambda: actions.encode(msg, resp), lambda: ygoai_rs.encode_response(raw, resp), f"{ctx} {resp}")
    if msg["type"] == "SELECT_SUM" and "indices" in resp:
        d.eq("check_sum", actions.check_sum(msg, resp["indices"]), ygoai_rs.check_sum(raw, resp["indices"]), ctx)


def play_both(seed: int, decks: str) -> Diff:
    """One game of two random agents played by Python (game.play) and by Rust (game::play): the
    records must be equal; then both players' observation streams of the game (game.replay with
    ObservationLog), replayed by both."""
    from ygoai.agents.random_agent import RandomAgent
    from ygoai.belief.games import game_decks
    from ygoai.game import GameRecord, ObservationLog, new_record, play, replay
    d = Diff(seed)
    record = new_record(list(game_decks(seed, decks=decks)), seed)
    blank = json.dumps(record.to_json())
    play(record, [RandomAgent(seed * 2), RandomAgent(seed * 2 + 1)])
    rs = json.loads(ygoai_rs.play_game(blank, [("random", seed * 2), ("random", seed * 2 + 1)]))
    py = record.to_json()
    for key in ("core_seed", "deck_orders", "agents", "result"):
        d.eq(f"record.{key}", py[key], rs[key], "")
    d.eq("record.responses", py["responses"], rs["responses"], f"{len(py['responses'])} / {len(rs['responses'])} responses")
    record = GameRecord.from_json(rs)
    logs = [ObservationLog(v) for v in (0, 1)]
    replay(record, logs).close()
    streams = json.loads(ygoai_rs.observation_streams(json.dumps(rs)))
    for v in (0, 1):
        items = [list(it) for it in logs[v].items]
        d.eq("observation_stream", items, streams[v], f"viewer {v}, {len(items)} / {len(streams[v])} items")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--decks", default="random")
    ap.add_argument("-j", type=int, default=1)
    ap.add_argument("--fuzz", type=int, default=0, help="instead of games: N random messages per seed")
    ap.add_argument("--play", action="store_true", help="instead: whole games of random agents played by both")
    a = ap.parse_args()
    seeds = range(a.start, a.start + a.games)
    if a.play:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.j, maxtasksperchild=20) as pool:
            results = pool.starmap(play_both, [(s, a.decks) for s in seeds], chunksize=1)
    elif a.fuzz:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.j) as pool:
            results = pool.starmap(fuzz, [(s, a.fuzz) for s in seeds], chunksize=1)
    elif a.j > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.j, maxtasksperchild=20) as pool:
            results = pool.starmap(check_game, [(s, a.decks) for s in seeds], chunksize=1)
    else:
        results = [check_game(s, a.decks) for s in seeds]
    counts: Counter = Counter()
    checked: Counter = Counter()
    types: Counter = Counter()
    for r in results:
        types.update(r.types)
        counts.update(r.counts)
        checked.update(r.checked)
        for e in r.examples[:4]:
            print(e)
    print(f"{len(results)} games ({a.decks}, seeds {a.start}..{a.start + a.games - 1})")
    for k in sorted(checked):
        print(f"  {k:18} {checked[k]:8} checked, {counts[k]} different")
    from ygoai import constants as C
    if types:
        print("  message types not seen:", " ".join(sorted(set(C.MSG_IDS) - set(types))))
    sys.exit(1 if counts else 0)


if __name__ == "__main__":
    main()
