"""Human readable rendering of messages, prompts and responses (for debugging)."""

from __future__ import annotations

from . import constants as C
from .carddb import CardDB, default_db

LOC_SHORT = {C.LOCATION_DECK: "deck", C.LOCATION_HAND: "hand", C.LOCATION_MZONE: "mzone",
             C.LOCATION_SZONE: "szone", C.LOCATION_GRAVE: "grave", C.LOCATION_REMOVED: "banished",
             C.LOCATION_EXTRA: "extra", 0: "-"}


def loc_str(li: dict) -> str:
    loc = li.get("loc", 0)
    if loc & C.LOCATION_OVERLAY:
        base = LOC_SHORT.get(loc & ~C.LOCATION_OVERLAY, hex(loc))
        return f"P{li['con']}:{base}{li['seq']}.ov{li.get('pos', 0)}"
    name = LOC_SHORT.get(loc, hex(loc))
    pos = li.get("pos")
    fd = ""
    if pos is not None and loc in (C.LOCATION_MZONE, C.LOCATION_SZONE, C.LOCATION_REMOVED, C.LOCATION_EXTRA):
        fd = "(fd)" if pos & C.POS_FACEDOWN else ""
    return f"P{li.get('con', '?')}:{name}{li.get('seq', '')}{fd}"


def card_str(code: int, db: CardDB | None = None) -> str:
    if not code:
        return "?"
    db = db or default_db()
    return f"{db.name(code)}[{code}]"


def msg_str(m: dict, db: CardDB | None = None) -> str:
    db = db or default_db()
    t = m["type"]
    if t == "MOVE":
        return f"MOVE {card_str(m['code'], db)} {loc_str(m['from'])} -> {loc_str(m['to'])} reason={m['reason']:#x}"
    if t == "DRAW":
        return f"DRAW P{m['player']} " + ", ".join(card_str(c["code"], db) for c in m["cards"])
    if t in ("SUMMONING", "SPSUMMONING", "FLIPSUMMONING", "SET"):
        return f"{t} {card_str(m['code'], db)} at {loc_str(m)}"
    if t == "CHAINING":
        return (f"CHAINING #{m['chain_count']} {card_str(m['code'], db)} at {loc_str(m)} "
                f"trig=P{m['trigger_con']}:{LOC_SHORT.get(m['trigger_loc'], m['trigger_loc'])}{m['trigger_seq']} "
                f"desc={db.describe_desc(m['desc']) if m['desc'] else 0}")
    if t == "NEW_PHASE":
        return f"NEW_PHASE {m['phase']:#x}"
    if t == "HINT":
        return f"HINT type={m['hint_type']} P{m['player']} data={m['data']}"
    if t in ("SHUFFLE_HAND", "SHUFFLE_EXTRA"):
        return f"{t} P{m['player']} " + ", ".join(card_str(c, db) for c in m["codes"])
    if t in ("CONFIRM_CARDS", "CONFIRM_DECKTOP", "CONFIRM_EXTRATOP"):
        return f"{t} to P{m['player']}: " + ", ".join(
            f"{card_str(c['code'], db)}@{loc_str(c)}" for c in m["cards"])
    if t == "POS_CHANGE":
        return f"POS_CHANGE {card_str(m['code'], db)} at {loc_str(m)} {m['prev_pos']:#x}->{m['pos']:#x}"
    if t.startswith("SELECT") or t.startswith("SORT") or t.startswith("ANNOUNCE"):
        return prompt_str(m, db)
    rest = {k: v for k, v in m.items() if k != "type"}
    return f"{t} {rest}" if rest else t


def prompt_str(p: dict, db: CardDB | None = None) -> str:
    db = db or default_db()
    t = p["type"]
    parts = [f"{t} P{p.get('player')}"]
    for key in ("summonable", "spsummonable", "repositionable", "msetable", "ssetable",
                "activatable", "attackable", "chains", "cards", "select_cards", "unselect_cards",
                "must_cards"):
        if p.get(key):
            parts.append(f"{key}=[" + ", ".join(
                f"{card_str(c['code'], db)}@{loc_str(c)}" for c in p[key]) + "]")
    for key in ("min", "max", "options", "values", "count", "flag", "positions", "desc",
                "forced", "to_bp", "to_ep", "to_m2"):
        if key in p:
            parts.append(f"{key}={p[key]}")
    return " ".join(parts)
