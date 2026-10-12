"""YGOAI_ENGINE=tee: every duel runs on the C++ core AND the Rust engine in lockstep, and any
difference raises TeeMismatch (logged to YGOAI_TEE_LOG, default build/tee_mismatches.jsonl). The
caller sees the C++ core's results, so a run behaves exactly as with the C++ core.

Messages are compared at the points where the C++ core stops for a question or ends: a Rust card
function asks its question in the same process() call in which it sends the messages before it,
where a C++ script yields first (process() returns "continue" with those messages), so the
messages between two questions can be split into process() calls differently. Every other call
(responses, queries, the ext API) goes to both engines and the results must be equal.
"""

from __future__ import annotations

import json
import os
import time

from .paths import ROOT

CONTINUE = 2
TEE_LOG = os.environ.get("YGOAI_TEE_LOG", str(ROOT / "build" / "tee_mismatches.jsonl"))


class TeeMismatch(BaseException):
    """The engines differ (a BaseException: generic `except Exception` handlers must not hide it)."""


def _short(v, n: int = 400) -> str:
    if isinstance(v, list) and v and isinstance(v[0], bytes):
        v = [m.hex() for m in v]
    elif isinstance(v, bytes):
        v = v.hex()
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


class Duel:
    journal: list | None = None

    def __init__(self, seed, config=None, card_db=None):
        from . import core
        from .core_rust import Duel as RustDuel
        self.config = config or core.DuelConfig()
        self.seed = tuple(seed)
        self.a = core.CppDuel(seed, config, card_db)
        self.b = RustDuel(seed, config, card_db)
        self.handle = True
        self._last: list[bytes] = []
        self._pend_a: list[bytes] = []
        self._pend_b: list[bytes] = []
        self._synced = True
        self._rust_status = CONTINUE
        self.unsynced_calls = 0
        # every call, to rebuild the case (C++ results' digests are not needed: the C++ core is
        # deterministic)
        self.calls: list = [("__init__", (self.seed, config))]
        self.journal = [("__init__", (tuple(seed), config), None, None)] if core._journaling else None

    @property
    def logs(self):
        return self.a.logs

    def _mismatch(self, what: str) -> None:
        entry = {"time": time.time(), "seed": [s & 0xFFFFFFFFFFFFFFFF for s in self.seed],
                 "config": vars(self.config), "call": len(self.calls) - 1, "what": what,
                 "calls": [(n, repr(args)) for n, args in self.calls]}
        try:
            os.makedirs(os.path.dirname(TEE_LOG), exist_ok=True)
            with open(TEE_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
        raise TeeMismatch(f"engines differ at call {len(self.calls) - 1}: {what}")

    def _both(self, name: str, *args, **kwargs):
        self.calls.append((name, args if not kwargs else (args, kwargs)))
        if not self._synced:
            self.unsynced_calls += 1
        ea = eb = None
        try:
            ra = getattr(self.a, name)(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            ea, ra = e, None
        try:
            rb = getattr(self.b, name)(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # noqa: BLE001 - (also a panic of the Rust engine: PanicException)
            eb, rb = e, None
        if (ea is None) != (eb is None):
            self._mismatch(f"{name}{_short(args)}: C++ {'raised ' + repr(ea) if ea else _short(ra)}, "
                           f"Rust {'raised ' + repr(eb) if eb else _short(rb)}")
        if ea is not None:
            raise ea
        if self._synced and ra != rb:
            self._mismatch(f"{name}{_short(args)}: C++ {_short(ra)}, Rust {_short(rb)}")
        return ra

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self.a is not None:
            self.a.close()
            self.b.close()
        self.handle = None

    # (no clone(): the C++ core cannot copy a duel, and the agents check for the method)

    def new_card(self, owner, code, location, controller=None, sequence=0, position=None):
        from . import constants as C
        pos = C.POS_FACEDOWN_DEFENSE if position is None else position
        return self._both("new_card", owner, code, location, controller, sequence, pos)

    def start(self):
        return self._both("start")

    # -- processing ----------------------------------------------------------
    def process(self) -> int:
        self.calls.append(("process", ()))
        s = self.a.process()
        m = self.a.get_messages()
        self._last = m
        self._pend_a += m
        # the Rust engine on as far as the C++ core: until it sent as many messages, or (when
        # the C++ core stops) to its own stop; outside card functions both stop at the same
        # points, inside one the Rust engine can get ahead (it asks without yielding first)
        while self._rust_status == CONTINUE and (s != CONTINUE or len(self._pend_b) < len(self._pend_a)):
            try:
                self._rust_status = self.b.process()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as e:  # noqa: BLE001 - (a panic of the Rust engine)
                self._mismatch(f"process: the Rust engine failed: {e!r:.300}")
            self._pend_b += self.b.get_messages()
        n = min(len(self._pend_a), len(self._pend_b))
        if self._pend_a[:n] != self._pend_b[:n] or (s != CONTINUE and (s != self._rust_status or len(self._pend_a) != len(self._pend_b))):
            k = next((i for i, (x, y) in enumerate(zip(self._pend_a, self._pend_b)) if x != y), n)
            self._mismatch(f"messages: status C++ {s}, Rust {self._rust_status}; first difference at message {k}: "
                           f"C++ {_short(self._pend_a[k:k + 2])}, Rust {_short(self._pend_b[k:k + 2])}")
        # (in step where both sent the same messages; else the Rust engine is ahead)
        self._synced = len(self._pend_a) == len(self._pend_b) and (s == CONTINUE) == (self._rust_status == CONTINUE)
        del self._pend_a[:n]
        del self._pend_b[:n]
        if s != CONTINUE:
            self._rust_status = CONTINUE
        return s

    def get_messages(self) -> list[bytes]:
        return self._last

    def lua_messages(self, code: str) -> list[str]:
        raise NotImplementedError("the Rust engine runs no Lua: use the named queries instead")

    def all_cards(self) -> tuple[dict, dict]:
        from .core import parse_all_cards
        return parse_all_cards(self.all_cards_raw())


def _forward(name: str):
    def call(self, *args, **kwargs):
        return self._both(name, *args, **kwargs)
    call.__name__ = name
    return call


for _name in ("set_response", "query_count", "query_location_raw", "query_card_raw", "query_field_raw",
              "location_cards", "overlay_cards", "all_cards_raw", "chain_related", "read_log", "info",
              "set_location_order", "set_location_order_by_id", "push_forced_coins", "push_forced_dice",
              "push_forced_shuffle", "push_forced_shuffle_ids", "push_forced_random_select",
              "push_forced_set_card_shuffle", "push_forced_set_card_shuffle_ids", "force_activatable",
              "clear_force_activatable", "select_chains", "pending_triggers", "set_forced_match",
              "set_forced_nomatch", "take_forced_used", "swap_cards", "place_card", "set_lp", "create_card",
              "set_card_info", "set_turn", "restart_command", "clear_forced", "pending_forced", "forced_errors",
              "recreate_card", "reason_effect_is_own", "has_extra_release", "eligible_as", "settable_as"):
    setattr(Duel, _name, _forward(_name))
