"""Leak audit of the observation filter, independent of the filter's rules.

The game is replayed in god view one core step at a time. After every step
the full state is captured and every card identity visible to the viewer is
recorded (plus cards explicitly confirmed to the viewer). Afterwards, any
card code of the opponent's deck that appears anywhere in the viewer's
observation stream (card codes, effect descriptions, hints) but was never
visible to the viewer is reported as a potential leak.

Cards that are revealed and hidden again within a single core step would be
reported as false positives, so findings need a manual look.
"""

from __future__ import annotations

from collections import Counter

from .actions import encode
from .game import GameRecord, setup_duel
from .observe import to_core_response
from .messages import parse
from .observe import filter_message, filter_prompt
from .state import capture, view

CODE_KEYS = ("code", "code1", "code2")
DESC_KEYS = ("desc", "value")


def _codes_in(obj, out: set[int]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in CODE_KEYS and isinstance(v, int) and v:
                out.add(v)
            elif k in DESC_KEYS and isinstance(v, int) and v >= 0x100000:
                out.add(v >> 4)  # effect descriptions are code << 4 | index
            elif k == "codes" and isinstance(v, list):
                out.update(c for c in v if c)
            elif k == "options" and isinstance(v, list):
                out.update(o >> 4 for o in v if o >= 0x100000)
            else:
                _codes_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _codes_in(v, out)


def _visible_codes(state_view: dict, out: set[int]) -> None:
    for pl in state_view["players"]:
        for cards in pl.values():
            for c in cards:
                if c:
                    if c.get("code"):
                        out.add(c["code"])
                    out.update(c.get("overlays", []))
    for ch in state_view["chain"]:
        out.add(ch["code"])


def audit(record: GameRecord, viewer: int) -> list[str]:
    duel = setup_duel(record.decks, record.core_seed, record.duel_config(), record.deck_orders,
                      copy_seed=record.core_seed[0])
    own = set(record.decks[viewer].main) | set(record.decks[viewer].extra)
    opponent = set(record.decks[1 - viewer].main) | set(record.decks[1 - viewer].extra)
    seen: set[int] = set(own)
    shown: Counter = Counter()
    examples: dict[int, dict] = {}
    try:
        duel.start()
        responses = iter(record.responses)
        prompt = None
        while True:
            status = duel.process()
            msgs = [parse(m) for m in duel.get_messages()]
            _visible_codes(view(capture(duel), viewer), seen)
            if status == 1 and msgs and msgs[-1]["type"] != "RETRY":
                prompt = msgs.pop()
            ended = False
            for m in msgs:
                f = filter_message(m, viewer)
                if f is None:
                    continue
                if m["type"] == "CONFIRM_CARDS" and m["player"] == viewer:
                    seen.update(c["code"] for c in m["cards"])
                if m["type"] in ("CONFIRM_DECKTOP", "CONFIRM_EXTRATOP"):
                    seen.update(c["code"] for c in m["cards"])
                if m["type"] == "HINT" and m["hint_type"] in (8,):  # announced card names
                    seen.add(m["data"])
                found: set[int] = set()
                _codes_in(f, found)
                for code in found:
                    shown[code] += 1
                    examples.setdefault(code, f)
                if m["type"] == "WIN":
                    ended = True
            if ended or status == 0:
                break
            if status == 2:
                continue
            if prompt["player"] == viewer:
                found = set()
                _codes_in(filter_prompt(prompt, viewer), found)
                for code in found:
                    shown[code] += 1
                    examples.setdefault(code, prompt)
            entry = next(responses, None)
            if entry is None:
                break
            duel.set_response(encode(prompt, to_core_response(prompt, entry["player"], entry["response"])))
    finally:
        duel.close()
    findings = []
    for code in shown:
        if code in opponent and code not in seen:
            findings.append(f"code {code} shown {shown[code]}x but never visible, e.g. {examples[code]}")
    return findings
