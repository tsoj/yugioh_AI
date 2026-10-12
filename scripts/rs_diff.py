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
from ygoai import constants as C  # noqa: E402
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


PLAY_PAIRS = [("random", "random"), ("random", "greedy"), ("greedy", "heuristic"), ("heuristic", "random"),
              ("greedy", "greedy"), ("heuristic", "heuristic")]


def play_both(seed: int, decks: str) -> Diff:
    """One game of two built-in agents (PLAY_PAIRS by seed) played by Python (game.play) and by
    Rust (game::play): the records must be equal; then both players' observation streams of the
    game (game.replay with ObservationLog), replayed by both."""
    from ygoai.agents import make_agent
    from ygoai.belief.games import game_decks
    from ygoai.game import GameRecord, ObservationLog, new_record, play, replay
    d = Diff(seed)
    record = new_record(list(game_decks(seed, decks=decks)), seed)
    blank = json.dumps(record.to_json())
    names = PLAY_PAIRS[seed % len(PLAY_PAIRS)]
    play(record, [make_agent(names[0], seed * 2), make_agent(names[1], seed * 2 + 1)])
    rs = json.loads(ygoai_rs.play_game(blank, [(names[0], seed * 2), (names[1], seed * 2 + 1)]))
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


def py_tracker_dump(t) -> dict:
    """The Python tracker's state in the form of the Rust tracker's dump (ygoai_rs/src/belief.rs)."""
    from ygoai.belief.tracker import Var

    def need(n):
        if n is None:
            return None
        return f"mset{n[1]}" if n[0] == "mset" else n[0]

    def mark(mk):
        return f"p{mk[1]}" if mk[0] == "public" else f"h{mk[0]}:{mk[1]}"

    def vid(v):
        return v.id if isinstance(v, Var) else v
    vars_ = [{"owner": v.owner, "kind": v.kind, "fixed": v.fixed_code, "code": v.code,
              "pool": v.pool.index if v.pool is not None else None, "faceup": v.faceup,
              "overlays": [x.id for x in v.overlays], "excluded": sorted(v.excluded), "need": need(v.need),
              "marks": sorted(mark(m) for m in v.marks),
              "allowed": sorted(v.allowed) if v.allowed is not None else None,
              "same_as": vid(v.same_as), "not_same": [vid(x) for x in v.not_same], "prefer": [vid(x) for x in v.prefer],
              "seen_owner": v.seen_owner} for v in t.all_vars]
    pools = [{"player": p.player, "location": p.location, "members": [m.id for m in p.members],
              "claims": [c.id for c in p.claims], "zones": p.zones} for p in t.pools]
    from ygoai.belief.tracker import LIST_LOCS
    return {"vars": vars_, "pools": pools,
            "lists": [[[v.id for v in t.lists[p][loc]] for loc in LIST_LOCS] for p in (0, 1)],
            "mzones": [[vid(v) for v in t.zones[p][C.LOCATION_MZONE]] for p in (0, 1)],
            "szones": [[vid(v) for v in t.zones[p][C.LOCATION_SZONE]] for p in (0, 1)],
            "observations": [[v.id, c] for v, c in t.observations], "searched": [[v.id, c] for v, c in t.searched],
            "exclusions": [v.id for v in t.exclusions], "needs": [v.id for v in t.needs],
            "contains": sorted([list(x) for x in t.contains]), "shuffle_events": [p.index for p in t.shuffle_events],
            "set_shuffle_events": [p.index for p in t.set_shuffle_events], "turn_player": t.turn_player}


def first_diff(a, b, path="") -> str:
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return f"{path}: {str(a)[:200]} vs {str(b)[:200]}"
    if isinstance(a, dict):
        for k in a:
            if k not in b or a[k] != b[k]:
                return first_diff(a[k], b.get(k), f"{path}.{k}")
        return f"{path}: keys {set(b) - set(a)}"
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return first_diff(x, y, f"{path}[{i}]")
        return f"{path}: length {len(a)} vs {len(b)}"
    return f"{path}: {a!r} vs {b!r}"


def tracker_both(seed: int, decks: str) -> Diff:
    """Both trackers (belief/tracker.py, crates/ygoai/src/belief/tracker.rs) over both players'
    observation streams of a test game: the same errors at the same items and the same state
    (every 25 items and at the end)."""
    from ygoai.belief.games import make_game_streams
    from ygoai.belief.tracker import Tracker, TrackerError
    from ygoai.game import init_info
    d = Diff(seed)
    record, streams = make_game_streams(seed, decks=decks, pairs=PAIRS)
    for viewer in (0, 1):
        init = init_info(record, viewer)
        py, rs = Tracker(viewer, init), ygoai_rs.Tracker(viewer, init)
        for i, item in enumerate(streams[viewer]):
            try:
                py.process(item)
                pe = None
            except TrackerError as e:
                pe = str(e)
            re_ = rs.process(item)
            d.eq("tracker.error", pe is not None, re_ is not None, f"viewer {viewer} item {i}: py {pe} / rs {re_}")
            if pe is not None or re_ is not None:
                break
            if i % 25 == 0 or i == len(streams[viewer]) - 1:
                a, b = py_tracker_dump(py), json.loads(rs.dump())
                d.checked["tracker.state"] += 1
                if a != b:
                    d.counts["tracker.state"] += 1
                    if d.counts["tracker.state"] <= 2:
                        d.examples.append(f"seed {seed} viewer {viewer} item {i} {item[0]} "
                                          f"{item[1].get('type') if isinstance(item[1], dict) else ''}: {first_diff(a, b)}")
                    break
    return d


def py_hypothesis(h) -> dict:
    return {"decks": h.decks, "deck_orders": h.deck_orders, "extra_orders": h.extra_orders,
            "shuffles": [{"player": s["player"], "location": s["location"], "codes": s["codes"], "ranks": s["ranks"],
                          "owners": s["owners"]} for s in h.shuffles],
            "set_card_shuffles": h.set_card_shuffles,
            "set_card_shuffle_ranks": [list(x) for x in h.set_card_shuffle_ranks],
            "unknown_atoms": h.unknown_atoms, "var_codes": sorted([k, v] for k, v in h.var_codes.items()),
            "observed": sorted(h.observed), "deck_ranks": h.deck_ranks, "extra_ranks": h.extra_ranks}


_py_completions: dict = {}


def flow_both(seed: int, decks: str, completion: str = "corpus") -> Diff:
    """Both trackers over prefixes of both players' streams, then Tracker.solve (the flow model and
    the deck completion): the same hypotheses (or the same failures)."""
    from ygoai.belief.games import make_game_streams
    from ygoai.belief.tracker import Tracker, TrackerError
    from ygoai.carddb import default_db
    from ygoai.game import init_info
    if completion not in _py_completions:
        if completion == "corpus":
            from ygoai.belief.corpus import CorpusCompletion
            _py_completions[completion] = CorpusCompletion()
        else:
            from ygoai.belief.completion import VanillaCompletion
            _py_completions[completion] = VanillaCompletion()
    comp = _py_completions[completion]
    d = Diff(seed)
    record, streams = make_game_streams(seed, decks=decks, pairs=PAIRS)
    for viewer in (0, 1):
        items = streams[viewer]
        prompts = [i for i, it in enumerate(items) if it[0] == "prompt"]
        cuts = sorted({prompts[int(f * (len(prompts) - 1))] + 1 for f in (0.25, 0.5, 0.75, 1.0)}) if prompts else []
        for cut in cuts:
            init = init_info(record, viewer)
            py, rs = Tracker(viewer, init), ygoai_rs.Tracker(viewer, init)
            try:
                for it in items[:cut]:
                    py.process(it)
            except TrackerError:
                break
            for it in items[:cut]:
                rs.process(it)
            try:
                a = py_hypothesis(py.solve(comp, default_db()))
            except TrackerError as e:
                a = f"error: {e}"
            b = rs.solve(completion)
            b = b if b.startswith("error: ") else json.loads(b)
            d.checked["flow.ok"] += 1
            if isinstance(a, str) != isinstance(b, str):
                d.counts["flow.ok"] += 1
                d.examples.append(f"seed {seed} viewer {viewer} cut {cut}: py {str(a)[:200]} / rs {str(b)[:200]}")
                continue
            if isinstance(a, str):
                continue
            d.checked["flow.hypothesis"] += 1
            if a != b:
                d.counts["flow.hypothesis"] += 1
                if d.counts["flow.hypothesis"] <= 2:
                    d.examples.append(f"seed {seed} viewer {viewer} cut {cut}: {first_diff(a, b)}")
    return d


_INFER_NAMES = ("idle", "battle", "chain", "effectyn", "option", "place", "position", "announce", "cards",
                "unselect", "counter", "rps", "sort")


def _strip_none(v):
    if isinstance(v, dict):
        return {k: _strip_none(x) for k, x in v.items() if x is not None}
    if isinstance(v, (list, tuple)):
        return [_strip_none(x) for x in v]
    return v


def infer_both(seed: int, decks: str) -> Diff:
    """Belief replays of a test game (both players, the whole stream) on the Python modules; every
    call of a belief/infer.py inference is made of the Rust one too, with the same inputs (the
    duel's cards as they are), and the results compared."""
    import inspect
    from ygoai.belief import infer
    from ygoai.belief.check import check_prefix
    from ygoai.belief.corpus import CorpusCompletion
    from ygoai.belief.games import make_game_streams
    d = Diff(seed)
    pid = os.getpid()
    originals = {n: getattr(infer, f"infer_{n}") for n in _INFER_NAMES}

    def wrap(name, fn):
        sig = inspect.signature(fn)

        def call(*args, **kwargs):
            out = fn(*args, **kwargs)
            if os.getpid() != pid:
                return out  # (in a probe's fork: not counted)
            b = sig.bind(*args, **kwargs)
            b.apply_defaults()
            a = dict(b.arguments)
            prompt, up, duel = a.pop("prompt"), a.pop("up"), a.pop("duel", None)
            a.pop("db", None)
            raw = duel.all_cards_raw() if duel is not None else None
            plan = [(p, l, list(ids) if ids is not None else None) for p, l, ids in (getattr(duel, "shuffle_plan", None) or ())] \
                if duel is not None else None
            try:
                rs = ygoai_rs.infer(name, prompt, list(up), a, raw, plan)
                rs = (rs[0], rs[1])
            except Exception as e:  # noqa: BLE001
                rs = ("error", repr(e))
            d.eq(f"infer.{name}", _strip_none([out[0], out[1]]), _strip_none([rs[0], rs[1]]),
                 f"prompt {prompt.get('type')} up {[m.get('type') for m in up[:6]]}")
            return out
        return call
    for n, fn in originals.items():
        setattr(infer, f"infer_{n}", wrap(n, fn))
    try:
        record, streams = make_game_streams(seed, decks=decks, pairs=PAIRS)
        comp = _py_completions.setdefault("corpus", None) or CorpusCompletion()
        _py_completions["corpus"] = comp
        for viewer in (0, 1):
            try:
                check_prefix(record, viewer, streams[viewer], len(streams[viewer]), completion=comp)
            except Exception as e:  # noqa: BLE001
                d.examples.append(f"seed {seed} viewer {viewer}: replay raised {e!r:.200}")
    finally:
        for n, fn in originals.items():
            setattr(infer, f"infer_{n}", fn)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--decks", default="random")
    ap.add_argument("-j", type=int, default=1)
    ap.add_argument("--fuzz", type=int, default=0, help="instead of games: N random messages per seed")
    ap.add_argument("--play", action="store_true", help="instead: whole games of built-in agents played by both")
    ap.add_argument("--tracker", action="store_true", help="instead: the belief trackers over the test games' streams")
    ap.add_argument("--flow", choices=["vanilla", "corpus"], help="instead: the trackers' hypotheses (flow model) with this completion")
    ap.add_argument("--infer", action="store_true", help="instead: the inferences of Python belief replays, made by both")
    a = ap.parse_args()
    seeds = range(a.start, a.start + a.games)
    if a.infer:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.j, maxtasksperchild=10) as pool:
            results = pool.starmap(infer_both, [(s, a.decks) for s in seeds], chunksize=1)
    elif a.flow:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.j, maxtasksperchild=20) as pool:
            results = pool.starmap(flow_both, [(s, a.decks, a.flow) for s in seeds], chunksize=1)
    elif a.tracker:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.j, maxtasksperchild=20) as pool:
            results = pool.starmap(tracker_both, [(s, a.decks) for s in seeds], chunksize=1)
    elif a.play:
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
