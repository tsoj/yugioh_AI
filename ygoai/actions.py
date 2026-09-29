"""Responses to selection messages.

A response is a small JSON-friendly dict whose shape depends on the message
type (see `encode`). This module turns responses into the binary format the
core expects and can generate uniformly random valid responses, which is what
the random agent uses.
"""

from __future__ import annotations

import random
import struct

from . import constants as C
from .carddb import CardDB

IDLE_KINDS = ["summon", "spsummon", "repos", "mset", "sset", "activate", "to_bp", "to_ep", "shuffle"]
IDLE_LISTS = {"summon": "summonable", "spsummon": "spsummonable", "repos": "repositionable",
              "mset": "msetable", "sset": "ssetable", "activate": "activatable"}
BATTLE_KINDS = ["activate", "attack", "to_m2", "to_ep"]
BATTLE_LISTS = {"activate": "activatable", "attack": "attackable"}


class InvalidResponse(ValueError):
    pass


def encode(msg: dict, resp: dict) -> bytes:
    t = msg["type"]
    if t == "SELECT_IDLECMD":
        kind = IDLE_KINDS.index(resp["kind"])
        return struct.pack("<i", (resp.get("index", 0) << 16) | kind)
    if t == "SELECT_BATTLECMD":
        kind = BATTLE_KINDS.index(resp["kind"])
        return struct.pack("<i", (resp.get("index", 0) << 16) | kind)
    if t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
        return struct.pack("<i", 1 if resp["yes"] else 0)
    if t in ("SELECT_OPTION", "ANNOUNCE_NUMBER"):
        return struct.pack("<i", resp["index"])
    if t in ("SELECT_CARD", "SELECT_TRIBUTE", "SELECT_SUM"):
        if resp.get("cancel"):
            return struct.pack("<i", -1)
        idx = resp["indices"]
        return struct.pack(f"<iI{len(idx)}I", 0, len(idx), *idx)
    if t == "SELECT_UNSELECT_CARD":
        if resp.get("finish"):
            return struct.pack("<i", -1)
        return struct.pack("<ii", 1, resp["index"])
    if t == "SELECT_CHAIN":
        return struct.pack("<i", resp["index"])
    if t in ("SELECT_PLACE", "SELECT_DISFIELD"):
        return b"".join(struct.pack("<BBB", p, loc, seq) for p, loc, seq in resp["places"])
    if t == "SELECT_POSITION":
        return struct.pack("<i", resp["position"])
    if t == "SELECT_COUNTER":
        return b"".join(struct.pack("<h", c) for c in resp["counts"])
    if t in ("SORT_CARD", "SORT_CHAIN"):
        order = resp.get("order")
        if order is None:
            return struct.pack("<b", -1)
        return b"".join(struct.pack("<b", o) for o in order)
    if t == "ANNOUNCE_RACE":
        return struct.pack("<Q", resp["mask"])
    if t == "ANNOUNCE_ATTRIB":
        return struct.pack("<I", resp["mask"])
    if t == "ANNOUNCE_CARD":
        return struct.pack("<i", resp["code"])
    if t == "ROCK_PAPER_SCISSORS":
        return struct.pack("<i", resp["hand"])
    raise InvalidResponse(f"cannot encode response for {t}")


# ---------------------------------------------------------------------------
# Enumeration helpers
# ---------------------------------------------------------------------------

def idle_options(msg: dict) -> list[dict]:
    opts = []
    for kind in IDLE_KINDS:
        lst = IDLE_LISTS.get(kind)
        if lst:
            opts.extend({"kind": kind, "index": i} for i in range(len(msg[lst])))
        elif msg.get(kind if kind != "shuffle" else "can_shuffle"):
            opts.append({"kind": kind})
    return opts


def battle_options(msg: dict) -> list[dict]:
    opts = []
    for kind in BATTLE_KINDS:
        lst = BATTLE_LISTS.get(kind)
        if lst:
            opts.extend({"kind": kind, "index": i} for i in range(len(msg[lst])))
        elif msg.get(kind):
            opts.append({"kind": kind})
    return opts


def available_places(msg: dict) -> list[tuple[int, int, int]]:
    player = msg["player"]
    flag = msg["flag"]
    places = []
    for own in (True, False):
        p = player if own else 1 - player
        for loc, nseq, shift in ((C.LOCATION_MZONE, 7, 0), (C.LOCATION_SZONE, 8, 8)):
            for seq in range(nseq):
                bit = 1 << (seq + shift + (0 if own else 16))
                if not flag & bit:
                    places.append((p, loc, seq))
    return places


def _sum_options(param: int) -> tuple[int, ...]:
    o1, o2 = param & 0xFFFF, param >> 16
    return (o1, o2) if o2 else (o1,)


def _check_exact(params: list[int], acc: int) -> bool:
    """Mirror of select_sum_check1 in playerop.cpp."""
    def rec(i: int, acc: int) -> bool:
        if acc == 0 or i == len(params):
            return False
        o1 = params[i] & 0xFFFF
        o2 = params[i] >> 16
        if i == len(params) - 1:
            return acc == o1 or acc == o2
        return (acc > o1 and rec(i + 1, acc - o1)) or (o2 > 0 and acc > o2 and rec(i + 1, acc - o2))
    return rec(0, acc)


def _check_atleast(params: list[int], acc: int) -> bool:
    total = mx = 0
    mn = 0x7FFFFFFF
    for op in params:
        o1, o2 = op & 0xFFFF, op >> 16
        ms = o2 if (o2 and o2 < o1) else o1
        total += ms
        mx += o2 if o2 > o1 else o1
        mn = min(mn, ms)
    return not (mx < acc or total - mn >= acc)


def check_sum(msg: dict, indices: list[int]) -> bool:
    must = [c["param"] for c in msg["must_cards"]]
    sel = [msg["cards"][i]["param"] for i in indices]
    acc = msg["acc"]
    if msg["mode"] == 0:
        if not (msg["min"] <= len(indices) <= msg["max"]):
            return False
        return _check_exact(must + sel, acc)
    return _check_atleast(must + sel, acc)


def random_sum_selection(msg: dict, rng: random.Random, budget: int = 20000) -> list[int] | None:
    n = len(msg["cards"])
    order = list(range(n))
    rng.shuffle(order)
    steps = 0

    def low(param: int) -> int:
        o1, o2 = param & 0xFFFF, param >> 16
        return min(o1, o2) if o2 else o1

    must_low = sum(low(c["param"]) for c in msg["must_cards"])

    def dfs(start: int, chosen: list[int], low_sum: int) -> list[int] | None:
        nonlocal steps
        steps += 1
        if steps > budget:
            return None
        if chosen and check_sum(msg, chosen):
            return list(chosen)
        if msg["mode"] == 0 and (len(chosen) >= msg["max"] or low_sum >= msg["acc"]):
            return None  # exact sums cannot shrink by adding cards
        for k in range(start, n):
            chosen.append(order[k])
            res = dfs(k + 1, chosen, low_sum + low(msg["cards"][order[k]]["param"]))
            chosen.pop()
            if res is not None:
                return res
        return None

    return dfs(0, [], must_low)


# ---------------------------------------------------------------------------
# ANNOUNCE_CARD filter evaluation (mirror of is_declarable in playerop.cpp)
# ---------------------------------------------------------------------------

_BIN = {
    C.OPCODE_ADD: lambda a, b: a + b, C.OPCODE_SUB: lambda a, b: a - b,
    C.OPCODE_MUL: lambda a, b: a * b, C.OPCODE_DIV: lambda a, b: int(a / b) if b else 0,
    C.OPCODE_AND: lambda a, b: int(bool(a) and bool(b)), C.OPCODE_OR: lambda a, b: int(bool(a) or bool(b)),
    C.OPCODE_BAND: lambda a, b: a & b, C.OPCODE_BOR: lambda a, b: a | b,
    C.OPCODE_BXOR: lambda a, b: a ^ b, C.OPCODE_LSHIFT: lambda a, b: a << b,
    C.OPCODE_RSHIFT: lambda a, b: a >> b,
}
CARD_MARINE_DOLPHIN = 78734254
CARD_TWINKLE_MOSS = 13857930


def is_declarable(card, opcodes: list[int]) -> bool:
    stack: list[int] = []
    alias = token = False
    for op in opcodes:
        if op in _BIN:
            if len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(_BIN[op](a, b))
        elif op == C.OPCODE_NEG:
            if stack:
                stack.append(-stack.pop())
        elif op == C.OPCODE_NOT:
            if stack:
                stack.append(int(not stack.pop()))
        elif op == C.OPCODE_BNOT:
            if stack:
                stack.append(~stack.pop())
        elif op == C.OPCODE_ISCODE:
            if stack:
                stack.append(int(card.code == (stack.pop() & 0xFFFFFFFF)))
        elif op == C.OPCODE_ISTYPE:
            if stack:
                stack.append(card.type & stack.pop())
        elif op == C.OPCODE_ISRACE:
            if stack:
                stack.append(card.race & stack.pop())
        elif op == C.OPCODE_ISATTRIBUTE:
            if stack:
                stack.append(card.attribute & stack.pop())
        elif op == C.OPCODE_GETCODE:
            stack.append(card.code)
        elif op == C.OPCODE_GETTYPE:
            stack.append(card.type)
        elif op == C.OPCODE_GETRACE:
            stack.append(card.race)
        elif op == C.OPCODE_GETATTRIBUTE:
            stack.append(card.attribute)
        elif op == C.OPCODE_ISSETCARD:
            if stack:
                set_code = stack.pop() & 0xFFFFFFFF
                settype, subtype = set_code & 0xFFF, set_code & 0xF000
                stack.append(int(any((sc & 0xFFF) == settype and (sc & 0xF000 & subtype) == subtype
                                     for sc in card.setcodes)))
        elif op == C.OPCODE_ALLOW_ALIASES:
            alias = True
        elif op == C.OPCODE_ALLOW_TOKENS:
            token = True
        else:
            stack.append(op)
    if len(stack) != 1 or stack[0] == 0:
        return False
    if card.code in (CARD_MARINE_DOLPHIN, CARD_TWINKLE_MOSS):
        return True
    is_token_monster = (card.type & (C.TYPE_MONSTER | C.TYPE_TOKEN)) == (C.TYPE_MONSTER | C.TYPE_TOKEN)
    return (alias or not card.alias) and (token or not is_token_monster)


def declarable_codes(db: CardDB, opcodes: list[int]) -> list[int]:
    return [code for code, card in db.cards.items() if is_declarable(card, opcodes)]


# ---------------------------------------------------------------------------
# Random responses
# ---------------------------------------------------------------------------

def random_response(msg: dict, rng: random.Random, db: CardDB | None = None,
                    cancel_prob: float = 0.05) -> dict:
    """Uniformly-ish random valid response to a selection message."""
    t = msg["type"]
    if t == "SELECT_IDLECMD":
        return rng.choice(idle_options(msg))
    if t == "SELECT_BATTLECMD":
        return rng.choice(battle_options(msg))
    if t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
        return {"yes": rng.random() < 0.5}
    if t == "SELECT_OPTION":
        return {"index": rng.randrange(len(msg["options"]))}
    if t == "ANNOUNCE_NUMBER":
        return {"index": rng.randrange(len(msg["values"]))}
    if t == "SELECT_CARD":
        if msg["cancelable"] and rng.random() < cancel_prob:
            return {"cancel": True}
        n = len(msg["cards"])
        lo, hi = msg["min"], min(msg["max"], n)
        k = rng.randint(max(lo, 1) if hi >= 1 else 0, hi) if hi >= lo else lo
        return {"indices": sorted(rng.sample(range(n), k))}
    if t == "SELECT_TRIBUTE":
        if msg["cancelable"] and rng.random() < cancel_prob:
            return {"cancel": True}
        order = list(range(len(msg["cards"])))
        rng.shuffle(order)
        chosen, total = [], 0
        for i in order:
            if total >= msg["min"] and chosen:
                break
            chosen.append(i)
            total += msg["cards"][i]["release_param"]
        return {"indices": sorted(chosen[:max(1, msg["max"])])}
    if t == "SELECT_SUM":
        sel = random_sum_selection(msg, rng)
        if sel is None:
            sel = []
        return {"indices": sorted(sel)}
    if t == "SELECT_UNSELECT_CARD":
        if (msg["finishable"] or msg["cancelable"]) and (not msg["select_cards"] or rng.random() < 0.5):
            return {"finish": True}
        if msg["select_cards"]:
            return {"index": rng.randrange(len(msg["select_cards"]))}
        return {"index": len(msg["select_cards"]) + rng.randrange(len(msg["unselect_cards"]))}
    if t == "SELECT_CHAIN":
        n = len(msg["chains"])
        if msg["forced"]:
            return {"index": rng.randrange(n)}
        choice = rng.randrange(n + 1) - 1
        return {"index": choice}
    if t in ("SELECT_PLACE", "SELECT_DISFIELD"):
        places = available_places(msg)
        count = min(msg["count"], len(places))
        return {"places": [list(p) for p in rng.sample(places, count)]}
    if t == "SELECT_POSITION":
        options = [p for p in (C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_ATTACK, C.POS_FACEUP_DEFENSE,
                               C.POS_FACEDOWN_DEFENSE) if msg["positions"] & p]
        return {"position": rng.choice(options)}
    if t == "SELECT_COUNTER":
        remaining = msg["count"]
        counts = [0] * len(msg["cards"])
        while remaining > 0:
            cands = [i for i, c in enumerate(msg["cards"]) if counts[i] < c["counters"]]
            i = rng.choice(cands)
            counts[i] += 1
            remaining -= 1
        return {"counts": counts}
    if t in ("SORT_CARD", "SORT_CHAIN"):
        order = list(range(len(msg["cards"])))
        rng.shuffle(order)
        return {"order": order}
    if t == "ANNOUNCE_RACE":
        bits = [1 << i for i in range(64) if msg["available"] & (1 << i)]
        return {"mask": sum(rng.sample(bits, min(msg["count"], len(bits))))}
    if t == "ANNOUNCE_ATTRIB":
        bits = [1 << i for i in range(32) if msg["available"] & (1 << i)]
        return {"mask": sum(rng.sample(bits, min(msg["count"], len(bits))))}
    if t == "ANNOUNCE_CARD":
        if db is None:
            raise InvalidResponse("ANNOUNCE_CARD needs a card database")
        codes = declarable_codes(db, msg["values"])
        return {"code": rng.choice(codes) if codes else 0}
    if t == "ROCK_PAPER_SCISSORS":
        return {"hand": rng.randint(1, 3)}
    raise InvalidResponse(f"no random response for {t}")
