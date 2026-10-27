"""The duel engine: the Rust port of ygopro-core and its card scripts (ygopro-core-rust/),
through the ygoai_rs binding.

The Rust engine reproduces the (patched) C++ core message for message (ygopro-core-rust/ygo-diff, with the C++
core as the oracle in ygopro-core-rust/oracle/), including the extension hooks the belief replay uses
(ygopro-core-rust/ygo-core/src/ext.rs). Duel.clone() copies a duel cheaply, for searches.
"""

from __future__ import annotations

from dataclasses import dataclass

import ygoai_rs

from . import constants as C
from .carddb import CardDB
from .paths import BABEL_CDB_DIR

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


class CoreError(RuntimeError):
    pass


@dataclass
class DuelConfig:
    flags: int = C.DUEL_MODE_MR5
    start_lp: int = 8000
    start_hand: int = 5
    draw_per_turn: int = 1


_inited = False


def _init() -> None:
    global _inited
    if not _inited:
        ygoai_rs.init(str(BABEL_CDB_DIR))
        _inited = True


class Duel:
    """One duel of the Rust engine (with the interface ygopro-core's C API has)."""

    def __init__(self, seed: tuple[int, int, int, int], config: DuelConfig | None = None,
                 card_db: CardDB | None = None):
        _init()
        self.config = config or DuelConfig()
        cfg = self.config
        self._d = ygoai_rs.RustDuel([s & 0xFFFFFFFFFFFFFFFF for s in seed], cfg.flags, cfg.start_lp,
                                    cfg.start_hand, cfg.draw_per_turn)
        self.handle = True
        self._logs: list[tuple[int, str]] = []

    @property
    def logs(self) -> list[tuple[int, str]]:
        self._logs.extend(self._d.take_logs())
        return self._logs

    @logs.setter
    def logs(self, value: list) -> None:
        self._d.take_logs()
        self._logs = value

    def set_fine_steps(self, on: bool) -> None:
        """Whether process() stops inside card functions where the C++ core does (default on:
        callers such as the belief replay change the duel between process() calls; off is faster)."""
        self._d.set_fine_steps(on)

    @classmethod
    def from_rust(cls, d, config=None) -> "Duel":
        """A Duel around a ygoai_rs.RustDuel (e.g. a reconstruction of the belief replay)."""
        new = object.__new__(cls)
        new.config = config or DuelConfig()
        new._d = d
        new.handle = True
        new._logs = []
        return new

    def clone(self) -> "Duel":
        """A copy of the duel: both go on independently (cheap: cards and effects are shared until
        changed)."""
        new = object.__new__(Duel)
        new.config = self.config
        new._d = self._d.clone()
        new.handle = True
        new._logs = list(self.logs)
        return new

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self._d = None
        self.handle = None

    def new_card(self, owner: int, code: int, location: int, controller: int | None = None,
                 sequence: int = 0, position: int = C.POS_FACEDOWN_DEFENSE) -> None:
        self._d.new_card(owner, code, location, owner if controller is None else controller, sequence, position)

    def start(self) -> None:
        self._d.start()

    # -- processing ----------------------------------------------------------
    def process(self) -> int:
        return self._d.process()

    def get_messages(self) -> list[bytes]:
        return self._d.get_messages()

    def set_response(self, data: bytes) -> None:
        self._d.set_response(bytes(data))

    # -- queries -------------------------------------------------------------
    def query_count(self, player: int, location: int) -> int:
        return self._d.query_count(player, location)

    def query_location_raw(self, player: int, location: int, flags: int) -> bytes:
        return self._d.query_location_raw(player, location, flags)

    def query_card_raw(self, player: int, location: int, sequence: int, flags: int,
                       overlay_seq: int = 0) -> bytes:
        return self._d.query_card_raw(player, location, sequence, flags, overlay_seq)

    def query_field_raw(self) -> bytes:
        return self._d.query_field_raw()

    # -- extension API -------------------------------------------------------
    def location_cards(self, player: int, location: int) -> list[tuple[int, int]]:
        return self._d.location_cards(player, location)

    def overlay_cards(self, player: int, location: int, sequence: int) -> list[tuple[int, int]]:
        return self._d.overlay_cards(player, location, sequence)

    def all_cards_raw(self) -> bytes:
        return self._d.all_cards_raw()

    def chain_related(self) -> list[bool]:
        return self._d.chain_related()

    def read_log(self, enable: bool = True) -> dict[int, int]:
        return self._d.read_log(enable)

    def all_cards(self) -> tuple[dict, dict]:
        return parse_all_cards(self.all_cards_raw())

    def info(self) -> dict:
        turn, turn_player, phase, lp0, lp1 = self._d.info()
        signed = lambda v: v - (1 << 32) if v >= 1 << 31 else v  # noqa: E731
        return {"turn": turn, "turn_player": turn_player, "phase": phase, "lp": [signed(lp0), signed(lp1)]}

    def set_location_order(self, player: int, location: int, codes: list[int]) -> bool:
        return self._d.set_location_order(player, location, list(codes))

    def set_location_order_by_id(self, player: int, location: int, ids: list[int]) -> bool:
        return self._d.set_location_order_by_id(player, location, list(ids))

    def push_forced_coins(self, results: list[int]) -> None:
        self._d.push_forced_coins(list(results))

    def push_forced_dice(self, results: list[int]) -> None:
        self._d.push_forced_dice(list(results))

    def push_forced_shuffle(self, player: int, location: int, codes: list[int]) -> None:
        self._d.push_forced_shuffle(player, location, list(codes))

    def push_forced_shuffle_ids(self, player: int, location: int, ids: list[int], codes: list[int]) -> None:
        self._d.push_forced_shuffle_ids(player, location, list(ids), list(codes))

    def push_forced_random_select(self, locs: list[tuple[int, int, int, int]]) -> None:
        self._d.push_forced_random_select([tuple(loc) for loc in locs])

    def push_forced_set_card_shuffle(self, codes: list[int]) -> None:
        self._d.push_forced_set_card_shuffle(list(codes))

    def push_forced_set_card_shuffle_ids(self, ids: list[int], codes: list[int]) -> None:
        self._d.push_forced_set_card_shuffle_ids(list(ids), list(codes))

    def force_activatable(self, player: int, code: int, desc: int) -> None:
        self._d.force_activatable(player, code, desc)

    def clear_force_activatable(self) -> None:
        self._d.clear_force_activatable()

    def select_chains(self) -> list[tuple[int, int]]:
        return self._d.select_chains()

    def pending_triggers(self) -> list[dict]:
        return [dict(zip(("forced", "code", "desc", "con", "loc", "seq"), t)) for t in self._d.pending_triggers()]

    def set_forced_match(self, ids: list[int]) -> None:
        self._d.set_forced_match(list(ids))

    def set_forced_nomatch(self, ids: list[int]) -> None:
        self._d.set_forced_nomatch(list(ids))

    def take_forced_used(self) -> set[tuple[int, int]]:
        return set(self._d.take_forced_used())

    def swap_cards(self, id_a: int, id_b: int) -> bool:
        return self._d.swap_cards(id_a, id_b)

    def place_card(self, card_id: int, controller: int, location: int, sequence: int, position: int) -> bool:
        return self._d.place_card(card_id, controller, location, sequence, position)

    def set_lp(self, player: int, lp: int) -> None:
        self._d.set_lp(player, lp)

    def create_card(self, owner: int, code: int) -> int:
        return self._d.create_card(owner, code)

    def set_card_info(self, card_id: int, reason: int | None, status: int, mask: int) -> bool:
        return self._d.set_card_info(card_id, 0xFFFFFFFF if reason is None else reason, status, mask)

    def set_turn(self, turn: int) -> None:
        self._d.set_turn(turn)

    def restart_command(self) -> bool:
        return self._d.restart_command()

    def clear_forced(self) -> None:
        self._d.clear_forced()

    def pending_forced(self) -> int:
        return self._d.pending_forced()

    def forced_errors(self) -> int:
        return self._d.forced_errors()

    # -- questions about cards as the card scripts see them ---------------------------------------
    def recreate_card(self, card_id: int, code: int) -> None:
        self._d.recreate_card(card_id, code)

    def reason_effect_is_own(self, con: int, loc: int, seq: int) -> bool:
        return self._d.reason_effect_is_own(con, loc, seq)

    def has_extra_release(self, con: int, seq: int) -> bool:
        return self._d.has_extra_release(con, seq)

    def eligible_as(self, li: tuple, kind: str, src: tuple, cands: list[tuple]) -> dict:
        return self._d.eligible_as(tuple(li), kind, tuple(src), [tuple(c) for c in cands])

    def settable_as(self, li: tuple, sumtype: int, trib_locs: list[tuple], cands: list) -> dict:
        return self._d.settable_as(tuple(li), sumtype, [tuple(t) for t in trib_locs],
                                   [None if c is None else tuple(c) for c in cands])
