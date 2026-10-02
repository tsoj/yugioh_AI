"""ctypes binding to the (patched) ygopro-core shared library."""

from __future__ import annotations

import ctypes as ct
import os
import itertools
import struct
from dataclasses import dataclass
from pathlib import Path

from . import constants as C
from .carddb import CardDB, default_db
from .paths import CARD_SCRIPTS_DIR, CORE_LIB, FIXED_CARD_SCRIPTS_DIR, OVERRIDE_SCRIPTS

_ALL_LOCS = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_MZONE, C.LOCATION_SZONE,
             C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_EXTRA)


def parse_all_cards(raw: bytes) -> tuple[dict, dict]:
    """Parses Duel.all_cards_raw (6 uint32 per entry: player, location, sequence,
    material index + 1, code, card id)."""
    flat = memoryview(raw).cast("I").tolist()
    lists: dict = {(p, loc): [] for p in (0, 1) for loc in _ALL_LOCS}
    mats: dict = {}
    for i in range(0, len(flat), 6):
        p, loc, seq, mat, code, cid = flat[i:i + 6]
        if mat:
            mats.setdefault((p, loc, seq), []).append((code, cid))
        else:
            lists[p, loc].append((code, cid))
    return lists, mats


class OCG_CardData(ct.Structure):
    _fields_ = [
        ("code", ct.c_uint32),
        ("alias", ct.c_uint32),
        ("setcodes", ct.POINTER(ct.c_uint16)),
        ("type", ct.c_uint32),
        ("level", ct.c_uint32),
        ("attribute", ct.c_uint32),
        ("race", ct.c_uint64),
        ("attack", ct.c_int32),
        ("defense", ct.c_int32),
        ("lscale", ct.c_uint32),
        ("rscale", ct.c_uint32),
        ("link_marker", ct.c_uint32),
    ]


class OCG_Player(ct.Structure):
    _fields_ = [
        ("startingLP", ct.c_uint32),
        ("startingDrawCount", ct.c_uint32),
        ("drawCountPerTurn", ct.c_uint32),
    ]


DataReader = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_uint32, ct.POINTER(OCG_CardData))
DataReaderDone = ct.CFUNCTYPE(None, ct.c_void_p, ct.POINTER(OCG_CardData))
ScriptReader = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_char_p)
LogHandler = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_char_p, ct.c_int)


class OCG_DuelOptions(ct.Structure):
    _fields_ = [
        ("seed", ct.c_uint64 * 4),
        ("flags", ct.c_uint64),
        ("team1", OCG_Player),
        ("team2", OCG_Player),
        ("cardReader", DataReader),
        ("payload1", ct.c_void_p),
        ("scriptReader", ScriptReader),
        ("payload2", ct.c_void_p),
        ("logHandler", LogHandler),
        ("payload3", ct.c_void_p),
        ("cardReaderDone", DataReaderDone),
        ("payload4", ct.c_void_p),
        ("enableUnsafeLibraries", ct.c_uint8),
    ]


class OCG_NewCardInfo(ct.Structure):
    _fields_ = [
        ("team", ct.c_uint8),
        ("duelist", ct.c_uint8),
        ("code", ct.c_uint32),
        ("con", ct.c_uint8),
        ("loc", ct.c_uint32),
        ("seq", ct.c_uint32),
        ("pos", ct.c_uint32),
    ]


class OCG_QueryInfo(ct.Structure):
    _fields_ = [
        ("flags", ct.c_uint32),
        ("con", ct.c_uint8),
        ("loc", ct.c_uint32),
        ("seq", ct.c_uint32),
        ("overlay_seq", ct.c_uint32),
    ]


def _load_library(path: Path) -> ct.CDLL:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/build_core.sh")
    lib = ct.CDLL(str(path))
    vp, u8, u32, i32 = ct.c_void_p, ct.c_uint8, ct.c_uint32, ct.c_int
    sig = {
        "OCG_GetVersion": (None, [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]),
        "OCG_CreateDuel": (i32, [ct.POINTER(vp), ct.POINTER(OCG_DuelOptions)]),
        "OCG_DestroyDuel": (None, [vp]),
        "OCG_DuelNewCard": (None, [vp, ct.POINTER(OCG_NewCardInfo)]),
        "OCG_StartDuel": (None, [vp]),
        "OCG_DuelProcess": (i32, [vp]),
        "OCG_DuelGetMessage": (vp, [vp, ct.POINTER(u32)]),
        "OCG_DuelSetResponse": (None, [vp, ct.c_char_p, u32]),
        "OCG_LoadScript": (i32, [vp, ct.c_char_p, u32, ct.c_char_p]),
        "OCG_DuelQueryCount": (u32, [vp, u8, u32]),
        "OCG_DuelQuery": (vp, [vp, ct.POINTER(u32), ct.POINTER(OCG_QueryInfo)]),
        "OCG_DuelQueryLocation": (vp, [vp, ct.POINTER(u32), ct.POINTER(OCG_QueryInfo)]),
        "OCG_DuelQueryField": (vp, [vp, ct.POINTER(u32)]),
        "OCG_ExtPushForcedInts": (None, [vp, i32, ct.POINTER(ct.c_int32), u32]),
        "OCG_ExtPushForcedShuffle": (None, [vp, u8, u32, ct.POINTER(u32), u32]),
        "OCG_ExtPushForcedShuffleById": (None, [vp, u8, u32, ct.POINTER(u32), ct.POINTER(u32), u32]),
        "OCG_ExtPushForcedRandomSelect": (None, [vp, ct.POINTER(u32), u32]),
        "OCG_ExtPushForcedSetCardShuffle": (None, [vp, ct.POINTER(u32), u32]),
        "OCG_ExtClearForced": (None, [vp]),
        "OCG_ExtPendingForced": (u32, [vp]),
        "OCG_ExtGetErrors": (u32, [vp]),
        "OCG_ExtSetLocationOrder": (i32, [vp, u8, u32, ct.POINTER(u32), u32]),
        "OCG_ExtSetLocationOrderById": (i32, [vp, u8, u32, ct.POINTER(u32), u32]),
        "OCG_ExtGetLocationCards": (u32, [vp, u8, u32, ct.POINTER(u32), ct.POINTER(u32), u32]),
        "OCG_ExtGetOverlayCards": (u32, [vp, u8, u32, u32, ct.POINTER(u32), ct.POINTER(u32), u32]),
        "OCG_ExtGetAllCards": (u32, [vp, ct.POINTER(u32), u32]),
        "OCG_ExtChainRelated": (u32, [vp, ct.POINTER(u8), u32]),
        "OCG_ExtGetInfo": (None, [vp, ct.POINTER(u32)]),
        "OCG_ExtForceActivatable": (None, [vp, u8, u32, ct.c_uint64]),
        "OCG_ExtClearForceActivatable": (None, [vp]),
        "OCG_ExtSwapCards": (i32, [vp, u32, u32]),
        "OCG_ExtSetForcedMatch": (None, [vp, ct.POINTER(u32), u32]),
        "OCG_ExtSetForcedNoMatch": (None, [vp, ct.POINTER(u32), u32]),
        "OCG_ExtDebugPendingTriggers": (u32, [vp, ct.POINTER(ct.c_uint64), u32]),
        "OCG_ExtSelectChains": (u32, [vp, ct.POINTER(ct.c_uint64), u32]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype = res
        fn.argtypes = args
    return lib


_lib: ct.CDLL | None = None


def lib() -> ct.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_library(CORE_LIB)
    return _lib


# ---------------------------------------------------------------------------
# Callbacks. They are process-global: card data comes from a single CardDB and
# log messages are routed to the owning Duel through the payload pointer.
# ---------------------------------------------------------------------------

_card_db: CardDB | None = None
_setcode_arrays: dict[int, ct.Array] = {}
_script_cache: dict[str, bytes | None] = {}
_duels_by_key: dict[int, "Duel"] = {}
_duel_keys = itertools.count(1)
_EMPTY_SETCODES = (ct.c_uint16 * 1)(0)


def _find_script(name: str) -> bytes | None:
    if name in _script_cache:
        return _script_cache[name]
    base = Path(name).name
    if base == OVERRIDE_SCRIPTS.name:
        data = _script_cache[name] = OVERRIDE_SCRIPTS.read_bytes()
        return data
    candidates = []
    if base.startswith("c") and base[1:-4].isdigit():
        candidates = [] if os.environ.get("YGOAI_NO_OVERRIDES") else [FIXED_CARD_SCRIPTS_DIR / base]
        candidates += [CARD_SCRIPTS_DIR / "official" / base, CARD_SCRIPTS_DIR / "pre-release" / base,
                      CARD_SCRIPTS_DIR / "pre-errata" / base, CARD_SCRIPTS_DIR / "goat" / base]
    candidates.append(CARD_SCRIPTS_DIR / base)
    data = None
    for path in candidates:
        if path.exists():
            data = path.read_bytes()
            break
    _script_cache[name] = data
    return data


@DataReader
def _card_reader(payload, code, out):
    card = _card_db.get(code) if _card_db else None
    data = out.contents
    data.code = code
    if card is None:
        data.alias = 0
        data.setcodes = ct.cast(_EMPTY_SETCODES, ct.POINTER(ct.c_uint16))
        data.type = data.level = data.attribute = data.race = 0
        data.attack = data.defense = 0
        data.lscale = data.rscale = data.link_marker = 0
        return
    arr = _setcode_arrays.get(code)
    if arr is None:
        arr = (ct.c_uint16 * (len(card.setcodes) + 1))(*card.setcodes, 0)
        _setcode_arrays[code] = arr
    data.alias = card.alias
    data.setcodes = ct.cast(arr, ct.POINTER(ct.c_uint16))
    data.type = card.type
    data.level = card.level
    data.attribute = card.attribute
    data.race = card.race
    data.attack = card.attack
    data.defense = card.defense
    data.lscale = card.lscale
    data.rscale = card.rscale
    data.link_marker = card.link_marker


@DataReaderDone
def _card_reader_done(payload, data):
    pass


@ScriptReader
def _script_reader(payload, duel_ptr, name):
    name = name.decode()
    data = _find_script(name)
    if data is None:
        return 0
    return 1 if lib().OCG_LoadScript(duel_ptr, data, len(data), name.encode()) else 0


@LogHandler
def _log_handler(payload, text, log_type):
    duel = _duels_by_key.get(payload or 0)
    if duel is not None:
        duel.logs.append((log_type, text.decode(errors="replace")))


class CoreError(RuntimeError):
    pass


@dataclass
class DuelConfig:
    flags: int = C.DUEL_MODE_MR5
    start_lp: int = 8000
    start_hand: int = 5
    draw_per_turn: int = 1


class Duel:
    """Thin object-oriented wrapper around one OCG_Duel."""

    def __init__(self, seed: tuple[int, int, int, int], config: DuelConfig | None = None,
                 card_db: CardDB | None = None):
        global _card_db
        L = lib()
        if _card_db is None:
            _card_db = card_db or default_db()
        self.config = config or DuelConfig()
        self.logs: list[tuple[int, str]] = []
        self._key = next(_duel_keys)
        opts = OCG_DuelOptions()
        for i, s in enumerate(seed):
            opts.seed[i] = s & 0xFFFFFFFFFFFFFFFF
        if not any(opts.seed):
            opts.seed[0] = 1
        opts.flags = self.config.flags
        for team in (opts.team1, opts.team2):
            team.startingLP = self.config.start_lp
            team.startingDrawCount = self.config.start_hand
            team.drawCountPerTurn = self.config.draw_per_turn
        opts.cardReader = _card_reader
        opts.scriptReader = _script_reader
        opts.logHandler = _log_handler
        opts.payload3 = self._key
        opts.cardReaderDone = _card_reader_done
        opts.enableUnsafeLibraries = 0
        handle = ct.c_void_p()
        status = L.OCG_CreateDuel(ct.byref(handle), ct.byref(opts))
        if status != 0:
            raise CoreError(f"OCG_CreateDuel failed with status {status}")
        self.handle = handle
        _duels_by_key[self._key] = self
        scripts = ["constant.lua", "utility.lua"]
        if not os.environ.get("YGOAI_NO_OVERRIDES"):  # (for comparing against the original helpers)
            scripts.append(OVERRIDE_SCRIPTS.name)
        for script in scripts:
            if not _script_reader(None, self.handle, script.encode()):
                raise CoreError(f"failed to load {script}")

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self.handle:
            lib().OCG_DestroyDuel(self.handle)
            self.handle = None
            _duels_by_key.pop(self._key, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def new_card(self, owner: int, code: int, location: int, controller: int | None = None,
                 sequence: int = 0, position: int = C.POS_FACEDOWN_DEFENSE) -> None:
        info = OCG_NewCardInfo(team=owner, duelist=0, code=code,
                               con=owner if controller is None else controller,
                               loc=location, seq=sequence, pos=position)
        lib().OCG_DuelNewCard(self.handle, ct.byref(info))

    def start(self) -> None:
        lib().OCG_StartDuel(self.handle)

    # -- processing ----------------------------------------------------------
    def process(self) -> int:
        return lib().OCG_DuelProcess(self.handle)

    def get_messages(self) -> list[bytes]:
        length = ct.c_uint32()
        ptr = lib().OCG_DuelGetMessage(self.handle, ct.byref(length))
        if not length.value:
            return []
        buf = ct.string_at(ptr, length.value)
        msgs = []
        pos = 0
        n = len(buf)
        while pos < n:
            (size,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            msgs.append(buf[pos:pos + size])
            pos += size
        return msgs

    def set_response(self, data: bytes) -> None:
        lib().OCG_DuelSetResponse(self.handle, data, len(data))

    # -- queries -------------------------------------------------------------
    def query_count(self, player: int, location: int) -> int:
        return lib().OCG_DuelQueryCount(self.handle, player, location)

    def query_location_raw(self, player: int, location: int, flags: int) -> bytes:
        info = OCG_QueryInfo(flags=flags, con=player, loc=location, seq=0, overlay_seq=0)
        length = ct.c_uint32()
        ptr = lib().OCG_DuelQueryLocation(self.handle, ct.byref(length), ct.byref(info))
        return ct.string_at(ptr, length.value) if length.value else b""

    def query_card_raw(self, player: int, location: int, sequence: int, flags: int,
                       overlay_seq: int = 0) -> bytes:
        info = OCG_QueryInfo(flags=flags, con=player, loc=location, seq=sequence,
                             overlay_seq=overlay_seq)
        length = ct.c_uint32()
        ptr = lib().OCG_DuelQuery(self.handle, ct.byref(length), ct.byref(info))
        return ct.string_at(ptr, length.value) if length.value else b""

    def query_field_raw(self) -> bytes:
        length = ct.c_uint32()
        ptr = lib().OCG_DuelQueryField(self.handle, ct.byref(length))
        return ct.string_at(ptr, length.value) if length.value else b""

    # -- extension API -------------------------------------------------------
    def location_cards(self, player: int, location: int) -> list[tuple[int, int]]:
        """(code, card id) for each slot of a location; (0, 0) for empty zones."""
        cap = 256
        codes = (ct.c_uint32 * cap)()
        ids = (ct.c_uint32 * cap)()
        n = lib().OCG_ExtGetLocationCards(self.handle, player, location, codes, ids, cap)
        return [(codes[i], ids[i]) for i in range(min(n, cap))]

    def overlay_cards(self, player: int, location: int, sequence: int) -> list[tuple[int, int]]:
        cap = 64
        codes = (ct.c_uint32 * cap)()
        ids = (ct.c_uint32 * cap)()
        n = lib().OCG_ExtGetOverlayCards(self.handle, player, location, sequence, codes, ids, cap)
        return [(codes[i], ids[i]) for i in range(min(n, cap))]

    _all_buf = None

    def all_cards_raw(self) -> bytes:
        """Every slot of both players in one call, unparsed (see parse_all_cards)."""
        cap = 2048
        buf = Duel._all_buf
        if buf is None:
            buf = Duel._all_buf = (ct.c_uint32 * (6 * cap))()
        n = lib().OCG_ExtGetAllCards(self.handle, buf, cap)
        if n > cap:
            buf = (ct.c_uint32 * (6 * n))()
            n = lib().OCG_ExtGetAllCards(self.handle, buf, n)
        return ct.string_at(buf, 24 * n)

    def chain_related(self) -> list[bool]:
        """For each current chain link: whether its card is still the activated
        object (it has not left its place since)."""
        buf = (ct.c_uint8 * 64)()
        n = lib().OCG_ExtChainRelated(self.handle, buf, 64)
        if n > 64:
            buf = (ct.c_uint8 * n)()
            n = lib().OCG_ExtChainRelated(self.handle, buf, n)
        return [bool(buf[i]) for i in range(n)]

    def all_cards(self) -> tuple[dict, dict]:
        """Every slot of both players in one call: ({(player, location): [(code,
        card id), ...]}, {(player, location, sequence): materials as (code, id)})."""
        return parse_all_cards(self.all_cards_raw())

    def info(self) -> dict:
        out = (ct.c_uint32 * 5)()
        lib().OCG_ExtGetInfo(self.handle, out)
        return {"turn": out[0], "turn_player": out[1], "phase": out[2],
                "lp": [ct.c_int32(out[3]).value, ct.c_int32(out[4]).value]}

    def set_location_order(self, player: int, location: int, codes: list[int]) -> bool:
        arr = (ct.c_uint32 * len(codes))(*codes)
        return bool(lib().OCG_ExtSetLocationOrder(self.handle, player, location, arr, len(codes)))

    def set_location_order_by_id(self, player: int, location: int, ids: list[int]) -> bool:
        arr = (ct.c_uint32 * len(ids))(*ids)
        return bool(lib().OCG_ExtSetLocationOrderById(self.handle, player, location, arr, len(ids)))

    def push_forced_coins(self, results: list[int]) -> None:
        arr = (ct.c_int32 * len(results))(*results)
        lib().OCG_ExtPushForcedInts(self.handle, 0, arr, len(results))

    def push_forced_dice(self, results: list[int]) -> None:
        arr = (ct.c_int32 * len(results))(*results)
        lib().OCG_ExtPushForcedInts(self.handle, 1, arr, len(results))

    def push_forced_shuffle(self, player: int, location: int, codes: list[int]) -> None:
        arr = (ct.c_uint32 * max(1, len(codes)))(*codes)
        lib().OCG_ExtPushForcedShuffle(self.handle, player, location, arr, len(codes))

    def push_forced_shuffle_ids(self, player: int, location: int, ids: list[int], codes: list[int]) -> None:
        """The next shuffle of the pile yields exactly these cards (unique ids, bottom
        first), or if the pile holds other copies, these codes."""
        arr = (ct.c_uint32 * max(1, len(ids)))(*ids)
        carr = (ct.c_uint32 * max(1, len(codes)))(*codes)
        lib().OCG_ExtPushForcedShuffleById(self.handle, player, location, arr, carr, len(ids))

    def push_forced_random_select(self, locs: list[tuple[int, int, int, int]]) -> None:
        flat = [v for loc in locs for v in loc]
        arr = (ct.c_uint32 * max(1, len(flat)))(*flat)
        lib().OCG_ExtPushForcedRandomSelect(self.handle, arr, len(locs))

    def push_forced_set_card_shuffle(self, codes: list[int]) -> None:
        arr = (ct.c_uint32 * max(1, len(codes)))(*codes)
        lib().OCG_ExtPushForcedSetCardShuffle(self.handle, arr, len(codes))

    def force_activatable(self, player: int, code: int, desc: int) -> None:
        lib().OCG_ExtForceActivatable(self.handle, player, code, desc)

    def clear_force_activatable(self) -> None:
        lib().OCG_ExtClearForceActivatable(self.handle)

    def select_chains(self) -> list[tuple[int, int]]:
        """(code, description) of the effects the pending selection is about."""
        cap = 64
        out = (ct.c_uint64 * (cap * 2))()
        n = lib().OCG_ExtSelectChains(self.handle, out, cap)
        return [(out[2 * i], out[2 * i + 1]) for i in range(n)]

    def pending_triggers(self) -> list[dict]:
        """Debugging: triggers waiting to be chained (core.new_fchain/new_ochain)."""
        cap = 64
        out = (ct.c_uint64 * (cap * 6))()
        n = lib().OCG_ExtDebugPendingTriggers(self.handle, out, cap)
        return [dict(zip(("forced", "code", "desc", "con", "loc", "seq"), out[i * 6:i * 6 + 6]))
                for i in range(min(n, cap))]

    def set_forced_match(self, ids: list[int]) -> None:
        arr = (ct.c_uint32 * max(1, len(ids)))(*ids)
        lib().OCG_ExtSetForcedMatch(self.handle, arr, len(ids))

    def set_forced_nomatch(self, ids: list[int]) -> None:
        arr = (ct.c_uint32 * max(1, len(ids)))(*ids)
        lib().OCG_ExtSetForcedNoMatch(self.handle, arr, len(ids))

    def swap_cards(self, id_a: int, id_b: int) -> bool:
        return bool(lib().OCG_ExtSwapCards(self.handle, id_a, id_b))

    def clear_forced(self) -> None:
        lib().OCG_ExtClearForced(self.handle)

    def pending_forced(self) -> int:
        return lib().OCG_ExtPendingForced(self.handle)

    def forced_errors(self) -> int:
        return lib().OCG_ExtGetErrors(self.handle)


def core_version() -> tuple[int, int]:
    major, minor = ct.c_int(), ct.c_int()
    lib().OCG_GetVersion(ct.byref(major), ct.byref(minor))
    return major.value, minor.value
