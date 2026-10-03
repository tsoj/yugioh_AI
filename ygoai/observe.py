"""Per-player filtering of god-view core messages (the "fastchess protocol" view).

`filter_message(msg, viewer)` returns what `viewer` is allowed to see of a
parsed core message, or None if the message is not shown to them at all.
The rules follow EDOPro's server (Multirole) and are stricter where Multirole
leaks information (card hints, confirmations shown to the other player,
face-down position changes, ...). Every message type is handled explicitly;
an unknown type raises, so that new messages cannot leak silently.

Hidden card codes are replaced by 0; effect descriptions of hidden cards
(which encode the card code) are replaced by 0 as well.
"""

from __future__ import annotations

from . import constants as C


class UnknownMessage(KeyError):
    pass


_PILES = C.LOCATION_DECK | C.LOCATION_HAND | C.LOCATION_EXTRA | C.LOCATION_REMOVED


def canon_pos(loc: int, pos: int) -> int:
    """The core gives face-down cards in piles different positions depending on
    how they got there (e.g. 8 for the starting deck, 10 for cards returned to
    it). That would let players track cards inside shuffled piles, so every
    face-down card in a pile gets the same position."""
    if loc & _PILES and not loc & C.LOCATION_OVERLAY and not pos & C.POS_FACEUP:
        return C.POS_FACEDOWN_DEFENSE
    return pos


def _canon(obj):
    """Canonicalizes the positions of all loc_info-like dicts in a message."""
    if isinstance(obj, dict):
        out = {k: _canon(v) for k, v in obj.items()}
        if "loc" in out and "pos" in out and isinstance(out["loc"], int):
            out["pos"] = canon_pos(out["loc"], out["pos"])
        return out
    if isinstance(obj, list):
        return [_canon(v) for v in obj]
    return obj


def loc_visible(li: dict, viewer: int) -> bool:
    """Whether the identity of a card at loc_info `li` is visible to `viewer`."""
    loc = li["loc"]
    if not loc:
        return True
    if loc & (C.LOCATION_GRAVE | C.LOCATION_OVERLAY):
        return True
    if li.get("pos", 0) & C.POS_FACEUP:
        return True
    if li["con"] == viewer and not loc & C.LOCATION_DECK:
        return True
    return False


def _strip_prompt_cards(cards: list[dict], viewer: int, has_pos: bool = True) -> None:
    """Selection prompts: keep own cards and public cards, hide everything else."""
    for c in cards:
        if c["con"] == viewer:
            continue
        if has_pos and loc_visible(c, viewer):
            continue
        c["code"] = 0
        if "desc" in c:
            c["desc"] = 0


# Hint types shown only to the named player / to everyone else.
_HINT_PRIVATE = {C.HINT_EVENT, C.HINT_MESSAGE, C.HINT_SELECTMSG, C.HINT_EFFECT}
_HINT_OTHERS = {C.HINT_OPSELECTED, C.HINT_RACE, C.HINT_ATTRIB, C.HINT_CODE, C.HINT_NUMBER,
                C.HINT_ZONE}

_PUBLIC = frozenset({
    "WIN", "SHUFFLE_DECK", "SWAP_GRAVE_DECK", "SHUFFLE_SET_CARD", "REVERSE_DECK", "DECK_TOP",
    "NEW_TURN", "NEW_PHASE", "FIELD_DISABLED", "SUMMONING", "SUMMONED", "SPSUMMONED",
    "FLIPSUMMONING", "FLIPSUMMONED", "CHAINING", "CHAINED", "CHAIN_SOLVING", "CHAIN_SOLVED",
    "CHAIN_END", "CHAIN_NEGATED", "CHAIN_DISABLED", "CARD_SELECTED", "RANDOM_SELECTED",
    "BECOME_TARGET", "DAMAGE", "RECOVER", "EQUIP", "LPUPDATE", "CARD_TARGET", "CANCEL_TARGET",
    "PAY_LPCOST", "ADD_COUNTER", "REMOVE_COUNTER", "ATTACK", "BATTLE", "ATTACK_DISABLED",
    "DAMAGE_STEP_START", "DAMAGE_STEP_END", "TOSS_COIN", "TOSS_DICE", "HAND_RES",
    "PLAYER_HINT", "MATCH_KILL", "REMOVE_CARDS", "CONFIRM_DECKTOP", "CONFIRM_EXTRATOP",
})

_PROMPTS = frozenset({
    "SELECT_BATTLECMD", "SELECT_IDLECMD", "SELECT_EFFECTYN", "SELECT_YESNO", "SELECT_OPTION",
    "SELECT_CARD", "SELECT_CHAIN", "SELECT_PLACE", "SELECT_POSITION", "SELECT_TRIBUTE",
    "SORT_CHAIN", "SELECT_COUNTER", "SELECT_SUM", "SELECT_DISFIELD", "SORT_CARD",
    "SELECT_UNSELECT_CARD", "ROCK_PAPER_SCISSORS", "ANNOUNCE_RACE", "ANNOUNCE_ATTRIB",
    "ANNOUNCE_CARD", "ANNOUNCE_NUMBER", "RETRY",
})


def is_hidden_prompt(msg: dict) -> bool:
    """Prompts the arbiter answers itself, without the player seeing them.

    An empty, non-mandatory chain window leaves no choice, but whether such a
    window opens can depend on hidden cards (e.g. a trigger of a card in the
    opponent's deck that turns out not to be activatable). Clients auto-pass
    these windows; here the player does not see them at all.
    """
    return msg["type"] == "SELECT_CHAIN" and not msg["chains"] and not msg["forced"]


AUTO_RESPONSE = {"index": -1}


# Option lists the core sorts by effect id (or builds from effect containers):
# their order reflects hidden history (card creation order, when effects were
# last enabled), so the arbiter presents them in a canonical order and maps the
# chosen index back (to_core_response).
_ORDERED = {"SELECT_IDLECMD": ("summonable", "spsummonable", "repositionable", "msetable", "ssetable",
                               "activatable"),
            "SELECT_BATTLECMD": ("activatable", "attackable"),
            "SELECT_CHAIN": ("chains",),
            # card selections list their cards in group order (a script may sort them)
            "SELECT_CARD": ("cards",),
            "SELECT_TRIBUTE": ("cards",),
            "SELECT_UNSELECT_CARD": ("select_cards", "unselect_cards")}


def _option_key(c: dict) -> tuple:
    return (c.get("con", 0), c.get("loc", 0), c.get("seq", 0), c.get("pos", 0) if c.get("loc", 0) & C.LOCATION_OVERLAY
            else 0, c.get("code", 0), c.get("desc", 0))


def option_order(msg: dict, viewer: int) -> dict[str, list[int]]:
    """For each ordered option list of a prompt: the core indices in the order
    the player sees them."""
    if msg["type"] == "ANNOUNCE_NUMBER":
        # (scripts may collect the numbers from cards, in their order)
        return {"values": sorted(range(len(msg["values"])), key=lambda i: (msg["values"][i], i))}
    stripped = _strip_prompt(msg, viewer)
    return {key: sorted(range(len(stripped[key])), key=lambda i: (_option_key(stripped[key][i]), i))
            for key in _ORDERED.get(msg["type"], ()) if key in stripped}


def to_core_response(msg: dict, viewer: int, resp: dict) -> dict:
    """Translates a response to the prompt as the player saw it (filter_prompt)
    into the core's option order."""
    from .actions import BATTLE_LISTS, IDLE_LISTS
    t = msg["type"]
    if t in ("SELECT_CARD", "SELECT_TRIBUTE"):
        if "indices" not in resp:
            return resp
        order = option_order(msg, viewer)["cards"]
        return {**resp, "indices": [order[i] for i in resp["indices"]]}
    if t == "SELECT_UNSELECT_CARD":
        if "index" not in resp:
            return resp
        order = option_order(msg, viewer)
        n = len(order["select_cards"])
        i = resp["index"]
        return {**resp, "index": order["select_cards"][i] if i < n else n + order["unselect_cards"][i - n]}
    if t == "ANNOUNCE_NUMBER" and "index" in resp:
        return {**resp, "index": option_order(msg, viewer)["values"][resp["index"]]}
    if t not in _ORDERED or "index" not in resp:
        return resp
    if t == "SELECT_CHAIN":
        key = "chains" if resp["index"] >= 0 else None
    else:
        key = (IDLE_LISTS if t == "SELECT_IDLECMD" else BATTLE_LISTS).get(resp.get("kind"))
    if key is None:
        return resp
    order = option_order(msg, viewer)[key]
    return {**resp, "index": order[resp["index"]]}


def filter_prompt(msg: dict, viewer: int) -> dict:
    """Strips a selection prompt addressed to `viewer` (option lists in canonical order)."""
    m = _strip_prompt(msg, viewer)
    for key, order in option_order(msg, viewer).items():
        m[key] = [m[key][i] for i in order]
    return m


def _strip_prompt(msg: dict, viewer: int) -> dict:
    m = _canon(msg)
    t = m["type"]
    if t == "SELECT_IDLECMD":
        for key in ("summonable", "spsummonable", "repositionable", "msetable", "ssetable",
                    "activatable"):
            _strip_prompt_cards(m[key], viewer, has_pos=False)
    elif t == "SELECT_BATTLECMD":
        _strip_prompt_cards(m["activatable"], viewer, has_pos=False)
        _strip_prompt_cards(m["attackable"], viewer, has_pos=False)
    elif t == "SELECT_CHAIN":
        _strip_prompt_cards(m["chains"], viewer)
        # (the core's timing hints for the client's auto-pass: what happened since
        # the windows were last passed, including windows only the opponent's
        # hidden cards opened, e.g. a hand trap's; a player at a table sees none)
        m.pop("hint_timing", None)
        m.pop("hint_timing_other", None)
        # (spe_count counts the offered effects whose hint timing matches those
        # timings: the same information, a hint for clients only)
        m.pop("spe_count", None)
    elif t == "SELECT_POSITION":
        # (the code of the card whose position is chosen: it can be an opponent's
        # face-down card, e.g. "change a monster's battle position", and the prompt
        # does not say where the card is; a summoned card is named by the hint
        # before the prompt anyway)
        m["code"] = 0
    elif t == "SELECT_EFFECTYN":
        if m["con"] != viewer and not loc_visible(m, viewer):
            m["code"] = 0
            m["desc"] = 0
    elif t in ("SELECT_CARD",):
        _strip_prompt_cards(m["cards"], viewer)
    elif t == "SELECT_UNSELECT_CARD":
        _strip_prompt_cards(m["select_cards"], viewer)
        _strip_prompt_cards(m["unselect_cards"], viewer)
    elif t == "SELECT_SUM":
        _strip_prompt_cards(m["must_cards"], viewer)
        _strip_prompt_cards(m["cards"], viewer)
    elif t in ("SELECT_TRIBUTE", "SELECT_COUNTER", "SORT_CARD", "SORT_CHAIN"):
        _strip_prompt_cards(m["cards"], viewer, has_pos=False)
    elif t == "ANNOUNCE_CARD":
        m["values"] = canonical_opcodes(m["values"])
    return m


# ANNOUNCE_CARD filters are RPN programs (see actions.is_declarable) that scripts
# often assemble by iterating over cards (e.g. one "code ISCODE OR" term per
# card name in the deck, by way of a Lua table whose iteration order follows
# insertion order). Which names can be declared is what the player sees; the
# order of the terms would tell the order of the cards. Chains of the
# commutative boolean operators are flattened and their operands sorted.
_UNARY_OPS = {C.OPCODE_NEG, C.OPCODE_NOT, C.OPCODE_BNOT, C.OPCODE_ISCODE, C.OPCODE_ISTYPE,
              C.OPCODE_ISRACE, C.OPCODE_ISATTRIBUTE, C.OPCODE_ISSETCARD}
_BINARY_OPS = {C.OPCODE_ADD, C.OPCODE_SUB, C.OPCODE_MUL, C.OPCODE_DIV, C.OPCODE_AND, C.OPCODE_OR,
               C.OPCODE_BAND, C.OPCODE_BOR, C.OPCODE_BXOR, C.OPCODE_LSHIFT, C.OPCODE_RSHIFT}
_FLAG_OPS = {C.OPCODE_ALLOW_ALIASES, C.OPCODE_ALLOW_TOKENS}
_COMMUTATIVE = {C.OPCODE_AND, C.OPCODE_OR}


def canonical_opcodes(values: list[int]) -> list[int]:
    """An equivalent ANNOUNCE_CARD filter in canonical form (unchanged if it is
    not a well formed program)."""
    stack: list[tuple] = []  # expression trees: (op, operands...) or (value,)
    flags = []
    for op in values:
        if op in _FLAG_OPS:
            flags.append(op)
        elif op in _BINARY_OPS:
            if len(stack) < 2:
                return values
            b, a = stack.pop(), stack.pop()
            if op in _COMMUTATIVE:
                operands = [x for e in (a, b) for x in (e[1:] if e[0] == op and len(e) > 2 else (e,))]
                stack.append((op, *sorted(operands)))
            else:
                stack.append((op, a, b))
        elif op in _UNARY_OPS:
            if not stack:
                return values
            stack.append((op, stack.pop()))
        else:  # (a constant or a GET operator)
            stack.append((op,))
    if len(stack) != 1:
        return values
    out: list[int] = []

    def emit(e: tuple) -> None:
        op, *operands = e
        if not operands:
            out.append(op)
        elif op in _COMMUTATIVE:
            emit(operands[0])
            for x in operands[1:]:
                emit(x)
                out.append(op)
        else:
            for x in operands:
                emit(x)
            out.append(op)
    emit(stack[0])
    return out + sorted(set(flags))


# Multi-card listings: the viewer gets them sorted by location (the patched core
# lists them in location order too; this keeps the order independent of it).
_LISTINGS = {"CONFIRM_CARDS", "RANDOM_SELECTED", "CARD_SELECTED", "BECOME_TARGET", "REMOVE_CARDS"}


def _loc_key(c: dict) -> tuple:
    return (c["con"], c["loc"], c["seq"], c.get("pos", 0))


def filter_message(msg: dict, viewer: int) -> dict | None:
    """Filters a non-prompt god-view message for `viewer` (None = not shown)."""
    out = _filter_message(msg, viewer)
    if out is None:
        return None
    out = _canon(out)
    if out["type"] in _LISTINGS:
        out["cards"] = sorted(out["cards"], key=_loc_key)
    elif out["type"] == "SHUFFLE_SET_CARD":
        # (where each card went only their controller knows)
        new = out.pop("new", None)
        if not out["old"] or out["old"][0]["con"] != viewer:
            new = None
        pairs = sorted(zip(out["old"], out["overlay"], new or out["old"]), key=lambda p: _loc_key(p[0]))
        out["old"] = [a for a, _, _ in pairs]
        out["overlay"] = [b for _, b, _ in pairs]
        if new is not None:
            out["new"] = [c for _, _, c in pairs]
    return out


def _filter_message(msg: dict, viewer: int) -> dict | None:
    t = msg["type"]
    if t == "DECK_TOP":
        # (the position of a face-down deck card tells how it got there, see canon_pos)
        return {**msg, "pos": canon_pos(C.LOCATION_DECK, msg["pos"])}
    if t in _PUBLIC:
        return msg
    if t in _PROMPTS:
        # prompts are delivered separately, only to the deciding player
        return None
    if t == "HINT":
        ht = msg["hint_type"]
        if ht in _HINT_PRIVATE:
            return msg if msg["player"] == viewer else None
        if ht in _HINT_OTHERS:
            return msg if msg["player"] != viewer else None
        return msg
    if t == "MOVE":
        to = msg["to"]
        if to["loc"] & (C.LOCATION_MZONE | C.LOCATION_SZONE) and not msg["reason"] & C.REASON_SPSUMMON:
            # Placing a card on the field (summon, set, activation, equipping,
            # moving it to another zone) does not update its reason
            # (move_to_field), so the message would carry the stale reason of
            # how the card entered e.g. the hand (drawn by rule or by an effect,
            # searched, ...), even much later when it moves on from the field.
            # The viewer cannot know that for hidden cards, and for its own hand
            # it tells apart copies of a card after a shuffle. (A special summon
            # sets its own reason.)
            msg = {**msg, "reason": 0}
        frm = msg["from"]
        if (frm["con"], frm["loc"]) == (to["con"], to["loc"]) and not loc_visible(frm, viewer):
            # reordering within a hidden pile (e.g. placing a card on top of
            # the deck) reports the card's stale reason as well
            msg = {**msg, "reason": 0}
        if to["con"] == viewer or loc_visible(to, viewer):
            return msg
        if frm["loc"] == C.LOCATION_HAND and to["loc"] & (C.LOCATION_DECK | C.LOCATION_EXTRA) \
                and msg["reason"] & (C.REASON_COST | C.REASON_MATERIAL):
            # a card returned from the hand to the Deck as a cost or material has
            # to meet a requirement: it is shown to the opponent first (the core
            # moves it face-down), e.g. the materials of Mementoal Tecuhtlica
            return msg
        return {**msg, "code": 0}
    if t == "DRAW":
        cards = [c if c["pos"] & C.POS_FACEUP else {**c, "pos": C.POS_FACEDOWN_DEFENSE}
                 for c in msg["cards"]]
        if msg["player"] != viewer:
            cards = [c if c["pos"] & C.POS_FACEUP else {**c, "code": 0} for c in cards]
        return {**msg, "cards": cards}
    if t in ("SHUFFLE_HAND", "SHUFFLE_EXTRA"):
        if msg["player"] == viewer:
            return msg
        # (a face-up card, e.g. one kept revealed in the hand, stays visible; where the
        # cards went only their owner knows)
        public = msg.get("public") or [False] * len(msg["codes"])
        out = {k: v for k, v in msg.items() if k != "prev"}
        return {**out, "codes": [c if pub else 0 for c, pub in zip(msg["codes"], public)]}
    if t == "SET":
        if msg["con"] == viewer:
            return msg
        return {**msg, "code": 0}
    if t == "SPSUMMONING":
        if msg["con"] == viewer or not msg["pos"] & C.POS_FACEDOWN:
            return msg
        return {**msg, "code": 0}
    if t == "POS_CHANGE":
        if msg["con"] == viewer or (msg["pos"] & C.POS_FACEUP) or (msg["prev_pos"] & C.POS_FACEUP):
            return msg
        return {**msg, "code": 0}
    if t == "SWAP":
        out = dict(msg)
        for code_key, loc_key, other in (("code1", "loc1", "loc2"), ("code2", "loc2", "loc1")):
            li = msg[loc_key]
            if not (loc_visible(li, viewer) or msg[other]["con"] == viewer):
                out[code_key] = 0
        return out
    if t == "CONFIRM_CARDS":
        if msg["player"] == viewer:
            return msg
        cards = []
        for c in msg["cards"]:
            # a player sees its own cards it reveals (even from the deck: it takes
            # them out to show them)
            known = c["con"] == viewer
            cards.append(c if known else {**c, "code": 0})
        return {**msg, "cards": cards}
    if t == "CARD_HINT":
        # hints emitted while cards are created (location 0) would reveal deck contents
        if msg["loc"] and loc_visible(msg, viewer):
            return msg
        return None
    if t == "MISSED_EFFECT":
        return msg if msg["con"] == viewer else None
    if t in ("AI_NAME", "SHOW_HINT", "TAG_SWAP", "RELOAD_FIELD"):
        return None
    raise UnknownMessage(t)
