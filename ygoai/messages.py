"""Parsing of ygopro-core binary messages into plain (JSON friendly) dicts.

Every message becomes a dict with a "type" key holding the message name
without the MSG_ prefix (e.g. "MOVE"). Locations are dicts with the keys
con/loc/seq/pos exactly as the core writes them. Parsing is strict: every
byte of a message must be consumed, otherwise ParseError is raised.
"""

from __future__ import annotations

import struct

from . import constants as C


class ParseError(ValueError):
    pass


class Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes, pos: int = 0):
        self.buf = buf
        self.pos = pos

    def _unpack(self, fmt: str, size: int):
        if self.pos + size > len(self.buf):
            raise ParseError("message truncated")
        (v,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += size
        return v

    def u8(self) -> int:
        return self._unpack("<B", 1)

    def u16(self) -> int:
        return self._unpack("<H", 2)

    def u32(self) -> int:
        return self._unpack("<I", 4)

    def i32(self) -> int:
        return self._unpack("<i", 4)

    def u64(self) -> int:
        return self._unpack("<Q", 8)

    def raw(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise ParseError("message truncated")
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def loc(self) -> dict:
        return {"con": self.u8(), "loc": self.u8(), "seq": self.u32(), "pos": self.u32()}

    def remaining(self) -> int:
        return len(self.buf) - self.pos


def _card_loc(r: Reader) -> dict:
    """code u32 followed by a full loc_info."""
    d = {"code": r.u32()}
    d.update(r.loc())
    return d


def _card_short(r: Reader, seq_size: int = 4) -> dict:
    """code u32, con u8, loc u8, seq u32/u8."""
    code = r.u32()
    con = r.u8()
    loc = r.u8()
    seq = r.u32() if seq_size == 4 else r.u8()
    return {"code": code, "con": con, "loc": loc, "seq": seq}


def _list(r: Reader, fn) -> list:
    return [fn(r) for _ in range(r.u32())]


# ---------------------------------------------------------------------------
# Per message parsers
# ---------------------------------------------------------------------------

def _p_hint(r):
    return {"hint_type": r.u8(), "player": r.u8(), "data": r.u64()}


def _p_win(r):
    return {"player": r.u8(), "reason": r.u8()}


def _chain_option_short(r):
    d = _card_short(r)
    d["desc"] = r.u64()
    d["client_mode"] = r.u8()
    return d


def _p_select_battlecmd(r):
    d = {"player": r.u8()}
    d["activatable"] = _list(r, _chain_option_short)

    def attacker(r):
        a = _card_short(r, seq_size=1)
        a["direct"] = r.u8()
        return a

    d["attackable"] = _list(r, attacker)
    d["to_m2"] = r.u8()
    d["to_ep"] = r.u8()
    return d


def _p_select_idlecmd(r):
    d = {"player": r.u8()}
    d["summonable"] = _list(r, _card_short)
    d["spsummonable"] = _list(r, _card_short)
    d["repositionable"] = _list(r, lambda r: _card_short(r, seq_size=1))
    d["msetable"] = _list(r, _card_short)
    d["ssetable"] = _list(r, _card_short)
    d["activatable"] = _list(r, _chain_option_short)
    d["to_bp"] = r.u8()
    d["to_ep"] = r.u8()
    d["can_shuffle"] = r.u8()
    return d


def _p_select_effectyn(r):
    d = {"player": r.u8(), "code": r.u32()}
    d.update(r.loc())
    d["desc"] = r.u64()
    return d


def _p_select_yesno(r):
    return {"player": r.u8(), "desc": r.u64()}


def _p_select_option(r):
    player = r.u8()
    n = r.u8()
    return {"player": player, "options": [r.u64() for _ in range(n)]}


def _p_select_card(r):
    d = {"player": r.u8(), "cancelable": r.u8(), "min": r.u32(), "max": r.u32()}
    d["cards"] = _list(r, _card_loc)
    return d


def _p_select_chain(r):
    d = {"player": r.u8(), "spe_count": r.u8(), "forced": r.u8(),
         "hint_timing": r.u32(), "hint_timing_other": r.u32()}

    def opt(r):
        o = _card_loc(r)
        o["desc"] = r.u64()
        o["client_mode"] = r.u8()
        return o

    d["chains"] = _list(r, opt)
    return d


def _p_select_place(r):
    return {"player": r.u8(), "count": r.u8(), "flag": r.u32()}


def _p_select_position(r):
    return {"player": r.u8(), "code": r.u32(), "positions": r.u8()}


def _p_select_tribute(r):
    d = {"player": r.u8(), "cancelable": r.u8(), "min": r.u32(), "max": r.u32()}

    def trib(r):
        t = _card_short(r)
        t["release_param"] = r.u8()
        return t

    d["cards"] = _list(r, trib)
    return d


def _p_sort(r):
    d = {"player": r.u8()}

    def c(r):
        return {"code": r.u32(), "con": r.u8(), "loc": r.u32(), "seq": r.u32()}

    d["cards"] = _list(r, c)
    return d


def _p_select_counter(r):
    d = {"player": r.u8(), "counter_type": r.u16(), "count": r.u16()}

    def c(r):
        e = _card_short(r, seq_size=1)
        e["counters"] = r.u16()
        return e

    d["cards"] = _list(r, c)
    return d


def _p_select_sum(r):
    d = {"player": r.u8(), "mode": r.u8(), "acc": r.u32(), "min": r.u32(), "max": r.u32()}

    def c(r):
        e = _card_loc(r)
        e["param"] = r.u32()
        return e

    d["must_cards"] = _list(r, c)
    d["cards"] = _list(r, c)
    return d


def _p_select_unselect_card(r):
    d = {"player": r.u8(), "finishable": r.u8(), "cancelable": r.u8(), "min": r.u32(),
         "max": r.u32()}
    d["select_cards"] = _list(r, _card_loc)
    d["unselect_cards"] = _list(r, _card_loc)
    return d


def _p_confirm(r):
    d = {"player": r.u8()}
    d["cards"] = _list(r, _card_short)
    return d


def _p_player(r):
    return {"player": r.u8()}


def _p_shuffle_codes(r):
    d = {"player": r.u8()}
    raw = _list(r, lambda r: r.u32())
    # (the patched core marks face-up cards with the top bit)
    d["codes"] = [c & 0x7fffffff for c in raw]
    d["public"] = [bool(c & 0x80000000) for c in raw]
    return d


def _p_shuffle_hand(r):
    d = _p_shuffle_codes(r)
    # (the patched core adds each card's position before the shuffle: its owner
    # knows where each of its cards went)
    d["prev"] = [r.u8() for _ in d["codes"]]
    return d


def _p_swap_grave_deck(r):
    d = {"player": r.u8(), "main_count": r.u32()}
    n = r.u32()
    d["extra_mask"] = r.raw(n).hex()
    return d


def _p_shuffle_set_card(r):
    loc = r.u8()
    n = r.u8()
    old = [r.loc() for _ in range(n)]
    new = [r.loc() for _ in range(n)]
    # (ocgext: the zone each card went to)
    return {"location": loc, "old": old, "overlay": new, "new": [r.u8() for _ in range(n)]}


def _p_empty(r):
    return {}


def _p_deck_top(r):
    return {"player": r.u8(), "seq": r.u32(), "code": r.u32(), "pos": r.u32()}


def _p_new_turn(r):
    return {"player": r.u8()}


def _p_new_phase(r):
    return {"phase": r.u16()}


def _p_move(r):
    return {"code": r.u32(), "from": r.loc(), "to": r.loc(), "reason": r.u32()}


def _p_pos_change(r):
    return {"code": r.u32(), "con": r.u8(), "loc": r.u8(), "seq": r.u8(), "prev_pos": r.u8(),
            "pos": r.u8()}


def _p_code_loc(r):
    d = {"code": r.u32()}
    d.update(r.loc())
    return d


def _p_swap(r):
    return {"code1": r.u32(), "loc1": r.loc(), "code2": r.u32(), "loc2": r.loc()}


def _p_field_disabled(r):
    return {"flag": r.u32()}


def _p_chaining(r):
    d = {"code": r.u32()}
    d.update(r.loc())
    d["trigger_con"] = r.u8()
    d["trigger_loc"] = r.u8()
    d["trigger_seq"] = r.u32()
    d["desc"] = r.u64()
    d["chain_count"] = r.u32()
    return d


def _p_chain_u8(r):
    return {"chain_count": r.u8()}


def _p_locs(r):
    return {"cards": _list(r, lambda r: r.loc())}


def _p_random_selected(r):
    d = {"player": r.u8()}
    d["cards"] = _list(r, lambda r: r.loc())
    return d


def _p_draw(r):
    d = {"player": r.u8()}
    d["cards"] = _list(r, lambda r: {"code": r.u32(), "pos": r.u32()})
    return d


def _p_player_amount(r):
    return {"player": r.u8(), "amount": r.u32()}


def _p_two_locs(r):
    return {"card": r.loc(), "target": r.loc()}


def _p_counter(r):
    return {"counter_type": r.u16(), "con": r.u8(), "loc": r.u8(), "seq": r.u8(), "count": r.u16()}


def _p_attack(r):
    return {"attacker": r.loc(), "target": r.loc()}


def _p_battle(r):
    return {"attacker": r.loc(), "attacker_atk": r.i32(), "attacker_def": r.i32(),
            "attacker_destroyed": r.u8(), "target": r.loc(), "target_atk": r.i32(),
            "target_def": r.i32(), "target_destroyed": r.u8()}


def _p_missed_effect(r):
    d = r.loc()
    d["code"] = r.u32()
    return d


def _p_toss(r):
    player = r.u8()
    n = r.u8()
    return {"player": player, "results": [r.u8() for _ in range(n)]}


def _p_hand_res(r):
    return {"result": r.u8()}


def _p_announce_race(r):
    return {"player": r.u8(), "count": r.u8(), "available": r.u64()}


def _p_announce_attrib(r):
    return {"player": r.u8(), "count": r.u8(), "available": r.u32()}


def _p_announce_u64_list(r):
    player = r.u8()
    n = r.u8()
    return {"player": player, "values": [r.u64() for _ in range(n)]}


def _p_card_hint(r):
    d = r.loc()
    d["hint_type"] = r.u8()
    d["value"] = r.u64()
    return d


def _p_player_hint(r):
    return {"player": r.u8(), "hint_type": r.u8(), "value": r.u64()}


def _p_match_kill(r):
    return {"code": r.u32()}


def _p_string(r):
    n = r.u16()
    s = r.raw(n).decode(errors="replace")
    r.u8()
    return {"text": s}


def _p_tag_swap(r):
    d = {"player": r.u8(), "main_count": r.u32()}
    extra_count = r.u32()
    d["extra_p_count"] = r.u32()
    hand_count = r.u32()
    d["top_code"] = r.u32()
    d["hand"] = [{"code": r.u32(), "pos": r.u32()} for _ in range(hand_count)]
    d["extra"] = [{"code": r.u32(), "pos": r.u32()} for _ in range(extra_count)]
    return d


def _p_lpupdate(r):
    return {"player": r.u8(), "lp": r.u32()}


def _p_reload_field(r):
    d = {"options": r.u32(), "players": []}
    for _ in range(2):
        p = {"lp": r.u32(), "mzone": [], "szone": []}
        for key, n in (("mzone", 7), ("szone", 8)):
            for _ in range(n):
                if r.u8():
                    p[key].append({"pos": r.u8(), "overlays": r.u32()})
                else:
                    p[key].append(None)
        p["deck"], p["hand"], p["grave"], p["removed"], p["extra"], p["extra_p"] = (
            r.u32(), r.u32(), r.u32(), r.u32(), r.u32(), r.u32())
        d["players"].append(p)
    chains = []
    for _ in range(r.u32()):
        c = {"code": r.u32(), "handler": r.loc(), "trigger_con": r.u8(), "trigger_loc": r.u8(),
             "trigger_seq": r.u32(), "desc": r.u64()}
        chains.append(c)
    d["chains"] = chains
    return d


_PARSERS = {
    C.MSG_RETRY: _p_empty,
    C.MSG_HINT: _p_hint,
    C.MSG_WIN: _p_win,
    C.MSG_SELECT_BATTLECMD: _p_select_battlecmd,
    C.MSG_SELECT_IDLECMD: _p_select_idlecmd,
    C.MSG_SELECT_EFFECTYN: _p_select_effectyn,
    C.MSG_SELECT_YESNO: _p_select_yesno,
    C.MSG_SELECT_OPTION: _p_select_option,
    C.MSG_SELECT_CARD: _p_select_card,
    C.MSG_SELECT_CHAIN: _p_select_chain,
    C.MSG_SELECT_PLACE: _p_select_place,
    C.MSG_SELECT_POSITION: _p_select_position,
    C.MSG_SELECT_TRIBUTE: _p_select_tribute,
    C.MSG_SORT_CHAIN: _p_sort,
    C.MSG_SELECT_COUNTER: _p_select_counter,
    C.MSG_SELECT_SUM: _p_select_sum,
    C.MSG_SELECT_DISFIELD: _p_select_place,
    C.MSG_SORT_CARD: _p_sort,
    C.MSG_SELECT_UNSELECT_CARD: _p_select_unselect_card,
    C.MSG_CONFIRM_DECKTOP: _p_confirm,
    C.MSG_CONFIRM_CARDS: _p_confirm,
    C.MSG_SHUFFLE_DECK: _p_player,
    C.MSG_SHUFFLE_HAND: _p_shuffle_hand,
    C.MSG_SWAP_GRAVE_DECK: _p_swap_grave_deck,
    C.MSG_SHUFFLE_SET_CARD: _p_shuffle_set_card,
    C.MSG_REVERSE_DECK: _p_empty,
    C.MSG_DECK_TOP: _p_deck_top,
    C.MSG_SHUFFLE_EXTRA: _p_shuffle_codes,
    C.MSG_NEW_TURN: _p_new_turn,
    C.MSG_NEW_PHASE: _p_new_phase,
    C.MSG_CONFIRM_EXTRATOP: _p_confirm,
    C.MSG_MOVE: _p_move,
    C.MSG_POS_CHANGE: _p_pos_change,
    C.MSG_SET: _p_code_loc,
    C.MSG_SWAP: _p_swap,
    C.MSG_FIELD_DISABLED: _p_field_disabled,
    C.MSG_SUMMONING: _p_code_loc,
    C.MSG_SUMMONED: _p_empty,
    C.MSG_SPSUMMONING: _p_code_loc,
    C.MSG_SPSUMMONED: _p_empty,
    C.MSG_FLIPSUMMONING: _p_code_loc,
    C.MSG_FLIPSUMMONED: _p_empty,
    C.MSG_CHAINING: _p_chaining,
    C.MSG_CHAINED: _p_chain_u8,
    C.MSG_CHAIN_SOLVING: _p_chain_u8,
    C.MSG_CHAIN_SOLVED: _p_chain_u8,
    C.MSG_CHAIN_END: _p_empty,
    C.MSG_CHAIN_NEGATED: _p_chain_u8,
    C.MSG_CHAIN_DISABLED: _p_chain_u8,
    C.MSG_CARD_SELECTED: _p_locs,
    C.MSG_RANDOM_SELECTED: _p_random_selected,
    C.MSG_BECOME_TARGET: _p_locs,
    C.MSG_DRAW: _p_draw,
    C.MSG_DAMAGE: _p_player_amount,
    C.MSG_RECOVER: _p_player_amount,
    C.MSG_EQUIP: _p_two_locs,
    C.MSG_LPUPDATE: _p_lpupdate,
    C.MSG_CARD_TARGET: _p_two_locs,
    C.MSG_CANCEL_TARGET: _p_two_locs,
    C.MSG_PAY_LPCOST: _p_player_amount,
    C.MSG_ADD_COUNTER: _p_counter,
    C.MSG_REMOVE_COUNTER: _p_counter,
    C.MSG_ATTACK: _p_attack,
    C.MSG_BATTLE: _p_battle,
    C.MSG_ATTACK_DISABLED: _p_empty,
    C.MSG_DAMAGE_STEP_START: _p_empty,
    C.MSG_DAMAGE_STEP_END: _p_empty,
    C.MSG_MISSED_EFFECT: _p_missed_effect,
    C.MSG_TOSS_COIN: _p_toss,
    C.MSG_TOSS_DICE: _p_toss,
    C.MSG_ROCK_PAPER_SCISSORS: _p_player,
    C.MSG_HAND_RES: _p_hand_res,
    C.MSG_ANNOUNCE_RACE: _p_announce_race,
    C.MSG_ANNOUNCE_ATTRIB: _p_announce_attrib,
    C.MSG_ANNOUNCE_CARD: _p_announce_u64_list,
    C.MSG_ANNOUNCE_NUMBER: _p_announce_u64_list,
    C.MSG_CARD_HINT: _p_card_hint,
    C.MSG_TAG_SWAP: _p_tag_swap,
    C.MSG_RELOAD_FIELD: _p_reload_field,
    C.MSG_AI_NAME: _p_string,
    C.MSG_SHOW_HINT: _p_string,
    C.MSG_PLAYER_HINT: _p_player_hint,
    C.MSG_MATCH_KILL: _p_match_kill,
    C.MSG_REMOVE_CARDS: _p_locs,
}


def parse(msg: bytes) -> dict:
    if not msg:
        raise ParseError("empty message")
    mtype = msg[0]
    fn = _PARSERS.get(mtype)
    if fn is None:
        raise ParseError(f"unknown message type {mtype}")
    r = Reader(msg, 1)
    d = fn(r)
    if r.remaining():
        raise ParseError(f"{C.MSG_NAMES.get(mtype, mtype)}: {r.remaining()} trailing bytes")
    d["type"] = C.MSG_NAMES[mtype]
    return d


def is_select(msg: dict) -> bool:
    return C.MSG_IDS[msg["type"]] in C.SELECT_MESSAGES


def select_player(msg: dict) -> int:
    """Player that has to answer a selection message."""
    if msg["type"] == "ROCK_PAPER_SCISSORS":
        return msg["player"]
    return msg["player"]
