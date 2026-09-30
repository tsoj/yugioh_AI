"""Inference of the opponent's hidden decisions from the upcoming observations.

During a belief replay the opponent's prompts are not part of the viewer's
stream; their answers have to be deduced from what the viewer observed next.
Each function returns (candidates, certain): candidate responses ranked by
plausibility, and whether the first one is known to be right. Uncertain
decisions are verified by the replay by probing candidates in forked
processes.
"""

from __future__ import annotations

from .. import constants as C
from ..actions import available_places, battle_options, idle_options

# Messages that may appear between an activation decision and its CHAINING
# (the card is moved to the spell zone / flipped face-up first).
_ACTIVATION_PRELUDE = {"MOVE", "POS_CHANGE", "HINT", "CARD_HINT"}

LIST_LOCS = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_GRAVE, C.LOCATION_REMOVED,
             C.LOCATION_EXTRA)


def same_loc(a: dict, b: dict) -> bool:
    return a["con"] == b["con"] and a["loc"] == b["loc"] and a["seq"] == b["seq"]


def prelude_fits(before: list[dict], ch: dict) -> bool:
    """Whether the messages before CHAINING `ch` belong to that activation: the
    card is placed in its zone or flipped face-up there (and a field spell it
    replaces leaves). Anything else is an earlier, different action."""
    for m in before:
        t = m["type"]
        if t == "HINT" and m["hint_type"] == C.HINT_EVENT:
            return False  # a timing event starts a new chain window: a later one
        if t in ("HINT", "CARD_HINT"):
            continue
        if t == "POS_CHANGE" and same_loc(m, ch) and m["prev_pos"] & C.POS_FACEDOWN \
                and m["pos"] & C.POS_FACEUP:
            continue  # (a change between face-up positions is a separate command)
        if t == "MOVE" and (same_loc(m["to"], ch) or (m["from"]["loc"] == C.LOCATION_SZONE
                                                      and m["from"]["seq"] == 5)):
            continue
        return False
    return True


def source_of(dest: dict, before: list[dict]) -> dict | None:
    """Location the card at `dest` came from, undoing shifts caused by the
    earlier moves in `before` (messages since the decision, oldest first)."""
    idx = None
    for i in range(len(before) - 1, -1, -1):
        m = before[i]
        if m["type"] == "MOVE" and same_loc(m["to"], dest):
            idx = i
            break
    if idx is None:
        return None
    src = dict(before[idx]["from"])
    if src["loc"] in LIST_LOCS:
        # undo the shifts of earlier moves out of / into the same pile
        for m in reversed(before[:idx]):
            if m["type"] != "MOVE":
                continue
            f, t = m["from"], m["to"]
            if t["con"] == src["con"] and t["loc"] == src["loc"] and t["seq"] <= src["seq"]:
                src["seq"] -= 1
            if f["con"] == src["con"] and f["loc"] == src["loc"] and f["seq"] <= src["seq"]:
                src["seq"] += 1
    return src


def match_chaining(options: list[dict], ch: dict, before: list[dict]) -> list[int]:
    """Indices of activation options that produce CHAINING `ch`."""
    cands = [i for i, o in enumerate(options)
             if o["code"] == ch["code"] and o.get("desc", 0) == ch["desc"]]
    if len(cands) <= 1:
        return cands
    # a card that moved into the activation's zone first (e.g. a field spell from
    # the hand replacing one Set in the field zone) was activated from where it came
    src = source_of(ch, before)
    if src is not None:
        from_src = [i for i in cands if same_loc(options[i], src)]
        if from_src:
            return from_src + [i for i in cands if i not in from_src]
    # the activated card's location identifies the copy
    exact = [i for i in cands if same_loc(options[i], ch)]
    if len(exact) == 1:
        return exact
    if exact:
        return exact + [i for i in cands if i not in exact]
    return cands


def _first_defining(up: list[dict], skip: set[str]) -> int | None:
    for i, m in enumerate(up):
        if m["type"] not in skip:
            return i
    return None


# timing events of the window the non-turn player gets when the turn player
# leaves the Main Phase (23) or the Battle Step (29)
PHASE_LEAVE_EVENTS = (23, 29)


def phase_leave(prompt: dict, m: dict, up_ext: list[dict], moves: dict[int, str]) -> tuple[list[dict], bool] | None:
    """Candidates if `m` is the window opened by a phase-change command. No
    other command shows a timing event before its own defining message."""
    if not (m["type"] == "HINT" and m["hint_type"] == C.HINT_EVENT and m["data"] in PHASE_LEAVE_EVENTS):
        return None
    kinds = [k for k in moves.values() if prompt[k]]
    if not kinds:
        return None
    # the phase entered afterwards (if the window restarts the command instead,
    # the choice has no effect)
    target = next((moves.get(x["phase"]) for x in up_ext if x["type"] == "NEW_PHASE"), None)
    if target in kinds:
        kinds.remove(target)
        kinds.insert(0, target)
    return [{"kind": k} for k in kinds], True


def infer_idle(prompt: dict, up: list[dict], up_ext: list[dict] | None = None) -> tuple[list[dict], bool]:
    opts = idle_options(prompt)
    for i, m in enumerate(up):
        t = m["type"]
        before = up[:i]
        leave = phase_leave(prompt, m, up_ext or [], {C.PHASE_BATTLE_START: "to_bp", C.PHASE_END: "to_ep"})
        if leave:
            return leave
        if t == "CHAINING":
            idx = match_chaining(prompt["activatable"], m, before)
            if idx:
                return [{"kind": "activate", "index": j} for j in idx], len(idx) == 1
            break
        if t in ("SUMMONING", "SPSUMMONING", "SET", "FLIPSUMMONING"):
            key = {"SUMMONING": ["summon"], "SPSUMMONING": ["spsummon"], "FLIPSUMMONING": ["repos"],
                   "SET": ["mset"] if m["loc"] == C.LOCATION_MZONE else ["sset"]}[t]
            lst = {"summon": "summonable", "spsummon": "spsummonable", "repos": "repositionable",
                   "mset": "msetable", "sset": "ssetable"}[key[0]]
            src = m if t == "FLIPSUMMONING" else source_of(m, before)
            entries = prompt[lst]
            cands = []
            if src is not None:
                cands = [j for j, e in enumerate(entries) if same_loc(e, src)]
            if not cands and m["code"]:
                cands = [j for j, e in enumerate(entries) if e["code"] == m["code"]]
            if cands:
                return [{"kind": key[0], "index": j} for j in cands], len(cands) == 1
            break
        if t == "POS_CHANGE" and m["loc"] == C.LOCATION_MZONE:
            cands = [j for j, e in enumerate(prompt["repositionable"]) if same_loc(e, m)]
            if cands:
                return [{"kind": "repos", "index": j} for j in cands], len(cands) == 1
            break
        if t == "NEW_PHASE":
            if m["phase"] == C.PHASE_BATTLE_START and prompt["to_bp"]:
                return [{"kind": "to_bp"}], True
            if prompt["to_ep"]:
                return [{"kind": "to_ep"}], True
            break
        if t == "SHUFFLE_HAND" and prompt["can_shuffle"] and m["player"] == prompt["player"] and i == 0:
            # the shuffle command (IdleCommand, ctype 8) shuffles right away; a
            # shuffle after other messages belongs to another action (e.g. a reveal)
            return [{"kind": "shuffle"}] + [o for o in opts if o["kind"] != "shuffle"], True
        if t in ("MOVE", "HINT", "CARD_HINT", "POS_CHANGE", "PAY_LPCOST", "CONFIRM_CARDS",
                 "SHUFFLE_HAND", "SHUFFLE_DECK", "LPUPDATE", "DAMAGE", "RECOVER", "BECOME_TARGET",
                 "CARD_SELECTED", "RANDOM_SELECTED", "REMOVE_COUNTER", "ADD_COUNTER",
                 "FIELD_DISABLED", "TOSS_COIN", "TOSS_DICE", "PLAYER_HINT", "CHAIN_SOLVING"):
            continue
        break
    phase = next_phase(up_ext or [])
    if phase == C.PHASE_BATTLE_START and prompt["to_bp"]:
        return [{"kind": "to_bp"}], True
    if phase in (C.PHASE_END, C.PHASE_MAIN2) and prompt["to_ep"]:
        return [{"kind": "to_ep"}], True
    return opts, False


def next_phase(up_ext: list[dict]) -> int | None:
    """The next phase change, looking past the viewer's chain-window prompts."""
    for m in up_ext:
        if m["type"] == "NEW_PHASE":
            return m["phase"]
        if m["type"] not in ("HINT", "CARD_HINT", "PLAYER_HINT"):
            return None
    return None


def infer_battle(prompt: dict, up: list[dict], up_ext: list[dict] | None = None) -> tuple[list[dict], bool]:
    opts = battle_options(prompt)
    for i, m in enumerate(up):
        t = m["type"]
        leave = phase_leave(prompt, m, up_ext or [], {C.PHASE_MAIN2: "to_m2", C.PHASE_END: "to_ep"})
        if leave:
            return leave
        if t == "CHAINING":
            idx = match_chaining(prompt["activatable"], m, up[:i])
            if idx:
                return [{"kind": "activate", "index": j} for j in idx], len(idx) == 1
            break
        if t == "ATTACK":
            cands = [j for j, e in enumerate(prompt["attackable"]) if same_loc(e, m["attacker"])]
            if cands:
                return [{"kind": "attack", "index": j} for j in cands], len(cands) == 1
            break
        if t == "NEW_PHASE":
            if m["phase"] == C.PHASE_MAIN2 and prompt["to_m2"]:
                return [{"kind": "to_m2"}], True
            if prompt["to_ep"]:
                return [{"kind": "to_ep"}], True
            break
        if t in _ACTIVATION_PRELUDE:
            continue
        break
    phase = next_phase(up_ext or [])
    if phase == C.PHASE_MAIN2 and prompt["to_m2"]:
        return [{"kind": "to_m2"}], True
    if phase == C.PHASE_END and prompt["to_ep"]:
        return [{"kind": "to_ep"}], True
    return opts, False


def infer_chain(prompt: dict, up: list[dict]) -> tuple[list[dict], bool]:
    n = len(prompt["chains"])
    if up and up[0]["type"] == "CARD_HINT":
        # an option that is no chain link (a continuous effect among the forced
        # triggers) acts right away, e.g. updating a turn counter on its card
        ch = next((m for m in up if m["type"] == "CHAINING"), None)
        own = [j for j, o in enumerate(prompt["chains"]) if same_loc(o, up[0])]
        if own and (ch is None or not same_loc(ch, up[0])):
            return [{"index": j} for j in own], len(own) == 1
    for i, m in enumerate(up):
        t = m["type"]
        if t == "CHAINING":
            if not prelude_fits(up[:i], m):
                break
            idx = match_chaining(prompt["chains"], m, up[:i])
            if idx:
                rest = [] if prompt["forced"] else [{"index": -1}]
                return [{"index": j} for j in idx] + rest, len(idx) == 1
            break
        if t in _ACTIVATION_PRELUDE:
            continue
        break
    if prompt["forced"]:
        return [{"index": j} for j in range(n)], n == 1
    return [{"index": -1}] + [{"index": j} for j in range(n)], True


def infer_effectyn(prompt: dict, up: list[dict], descs: list[int] | None = None) -> tuple[list[dict], bool]:
    """`descs`: descriptions of the effects the prompt may be about (if known)."""
    for i, m in enumerate(up):
        t = m["type"]
        if t == "CHAINING":
            if m["code"] == prompt["code"] and prelude_fits(up[:i], m):
                if descs and m["desc"] not in descs:
                    # another effect of the same card is activated next
                    return [{"yes": False}, {"yes": True}], True
                return [{"yes": True}, {"yes": False}], True
            break
        if t in _ACTIVATION_PRELUDE:
            continue
        break
    return [{"yes": False}, {"yes": True}], False


def infer_option(prompt: dict, up: list[dict], player: int) -> tuple[list[dict], bool]:
    opts = prompt["options"]
    for m in up[:12]:
        if m["type"] == "HINT" and m["hint_type"] == C.HINT_OPSELECTED and m["data"] in opts:
            i = opts.index(m["data"])
            return [{"index": i}] + [{"index": j} for j in range(len(opts)) if j != i], True
    return [{"index": j} for j in range(len(opts))], len(opts) == 1


def infer_hint_value(prompt: dict, up: list[dict], hint_type: int) -> int | None:
    for m in up[:12]:
        if m["type"] == "HINT" and m["hint_type"] == hint_type:
            return m["data"]
    return None


def infer_place(prompt: dict, up: list[dict], up_ext: list[dict] | None = None) -> tuple[list[dict], bool]:
    avail = set(available_places(prompt))
    places = []
    if prompt["type"] == "SELECT_PLACE" and not _field_move_follows(up_ext if up_ext is not None else up):
        # a card is put there before the action is over (visibly, even face-down):
        # no such move, no such prompt
        return [], False
    for i, m in enumerate(up):
        if prompt["type"] == "SELECT_DISFIELD" and not places and i < 12 \
                and m["type"] == "HINT" and m["hint_type"] == C.HINT_ZONE:
            # the script announces the selected zones (before anything uses them)
            zone = m["data"]
            if m["player"] != prompt["player"]:
                zone = ((zone & 0xffff) << 16) | ((zone >> 16) & 0xffff)  # (relative to the hinted player)
            return _zone_places(prompt, zone), True
        if m["type"] == "MOVE":
            to = m["to"]
            key = (to["con"], to["loc"], to["seq"])
            if to["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE) and key in avail and key not in places:
                places.append(key)
                if len(places) == prompt["count"]:
                    return [{"places": [list(p) for p in places]}], True
    # fallback: every single place (only sensible for count == 1)
    return [{"places": [list(p)]} for p in sorted(avail)][:40], False


def _field_move_follows(msgs: list[dict]) -> bool:
    for m in msgs:
        t = m["type"]
        if t in ("CHAIN_SOLVED", "CHAIN_END", "NEW_PHASE", "NEW_TURN") \
                or (t == "HINT" and m["hint_type"] == C.HINT_EVENT):
            return False
        if (t == "MOVE" and m["to"]["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE)) or t == "SWAP":
            return True  # (a control swap places both cards by SWAP)
    return True  # (the known stream ends first: no evidence either way)


def _zone_places(prompt: dict, zone: int) -> list[dict]:
    player = prompt["player"]
    places = []
    for bit in range(32):
        if not zone & (1 << bit):
            continue
        p = player if bit < 16 else 1 - player
        b = bit % 16
        loc, seq = (C.LOCATION_MZONE, b) if b < 8 else (C.LOCATION_SZONE, b - 8)
        places.append([p, loc, seq])
    return [{"places": places[:prompt["count"]]}]


def infer_position(prompt: dict, up: list[dict]) -> tuple[list[dict], bool]:
    allowed = [p for p in (C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_ATTACK, C.POS_FACEUP_DEFENSE,
                           C.POS_FACEDOWN_DEFENSE) if prompt["positions"] & p]
    for m in up:
        pos = None
        if m["type"] == "MOVE" and m["to"]["loc"] == C.LOCATION_MZONE and m["code"] and prompt["code"] \
                and m["code"] != prompt["code"] and m["from"]["loc"] != C.LOCATION_MZONE:
            # another card is put onto the field next: this card is not being placed
            return [], False
        if m["type"] == "MOVE" and m["to"]["loc"] == C.LOCATION_MZONE and (
                not m["code"] or not prompt["code"] or m["code"] == prompt["code"]):
            pos = m["to"]["pos"]
        elif m["type"] == "POS_CHANGE" and m["loc"] == C.LOCATION_MZONE and (
                not m["code"] or not prompt["code"] or m["code"] == prompt["code"]):
            pos = m["pos"]  # (Duel.ChangePosition with a choice)
        if pos is not None:
            if pos in allowed:
                return [{"position": pos}] + [{"position": p} for p in allowed if p != pos], True
            break
    return [{"position": p} for p in allowed], len(allowed) == 1


def infer_announce(prompt: dict, up: list[dict], db) -> tuple[list[dict], bool]:
    t = prompt["type"]
    if t == "ANNOUNCE_CARD":
        code = infer_hint_value(prompt, up, C.HINT_CODE)
        if code is not None:
            return [{"code": code}], True
        return [], False
    if t == "ANNOUNCE_NUMBER":
        v = infer_hint_value(prompt, up, C.HINT_NUMBER)
        if v is not None and v in prompt["values"]:
            return [{"index": prompt["values"].index(v)}], True
        return [{"index": i} for i in range(len(prompt["values"]))], False
    if t in ("ANNOUNCE_RACE", "ANNOUNCE_ATTRIB"):
        v = infer_hint_value(prompt, up, C.HINT_RACE if t == "ANNOUNCE_RACE" else C.HINT_ATTRIB)
        if v is not None:
            return [{"mask": v}], True
        return [], False
    return [], False


class LocMap:
    """Follows sim card ids through upcoming messages to resolve references."""

    def __init__(self, duel, lazy: bool = True):
        """Locations are read from the duel when first needed; `lazy=False` reads
        everything now (for snapshots that outlive the duel's current state)."""
        self.duel = duel
        self.lists: dict[tuple[int, int], list[int]] = {}
        self.zones: dict[tuple[int, int], list[int]] = {}
        self.overlays: dict[int, list[int]] = {}
        self.unknown: set[tuple[int, int]] = set()
        self.codes: dict[int, int] = {}  # card id -> code (of the locations read so far)
        self._loaded: set[tuple[int, int]] = set()
        # the results of the upcoming forced shuffles, (player, location, card ids
        # bottom first) in stream order, if the caller knows them (the replay does)
        self.plan: list[tuple] = list(getattr(duel, "shuffle_plan", None) or ())
        if not lazy:
            for p in (0, 1):
                for loc in (*LIST_LOCS, C.LOCATION_MZONE, C.LOCATION_SZONE):
                    self._load(p, loc)
            self.duel = None

    def __deepcopy__(self, memo):
        import copy as _copy
        dup = LocMap.__new__(LocMap)
        dup.duel = self.duel  # shared: only snapshots (fully read, no duel) are copied
        for name in ("lists", "zones", "overlays", "unknown", "codes", "_loaded", "plan"):
            setattr(dup, name, _copy.deepcopy(getattr(self, name, []), memo))
        return dup

    def _load(self, p: int, loc: int) -> None:
        key = (p, loc)
        if key in self._loaded or self.duel is None or p not in (0, 1):
            return
        self._loaded.add(key)
        duel = self.duel
        if loc in LIST_LOCS:
            cards = duel.location_cards(p, loc)
            self.lists[key] = [cid for _, cid in cards]
            self.codes.update((cid, code) for code, cid in cards)
        elif loc in (C.LOCATION_MZONE, C.LOCATION_SZONE):
            cards = duel.location_cards(p, loc)
            self.zones[key] = [cid for _, cid in cards]
            self.codes.update((cid, code) for code, cid in cards if cid)
            for seq, (_, cid) in enumerate(cards):
                if cid:
                    ov = duel.overlay_cards(p, loc, seq)
                    self.overlays[cid] = [oid for _, oid in ov]
                    self.codes.update((oid, code) for code, oid in ov)

    def at(self, li: dict) -> int | None:
        con, loc, seq = li["con"], li["loc"], li["seq"]
        if not loc:
            return None
        self._load(con, loc & ~C.LOCATION_OVERLAY)
        if loc & C.LOCATION_OVERLAY:
            host = self.at({"con": con, "loc": loc & ~C.LOCATION_OVERLAY, "seq": seq})
            ov = self.overlays.get(host, []) if host else []
            pos = li.get("pos", 0)
            return ov[pos] if pos < len(ov) else None
        key = (con, loc)
        if key in self.unknown:
            return None
        if key in self.zones:
            z = self.zones[key]
            return z[seq] if seq < len(z) else None
        lst = self.lists.get(key)
        if lst is None or seq >= len(lst):
            return None
        return lst[seq]

    def move(self, m: dict) -> int | None:
        f, t = m["from"], m["to"]
        self._load(t["con"], t["loc"] & ~C.LOCATION_OVERLAY)
        cid = self.at(f)
        if f["loc"] & C.LOCATION_OVERLAY:
            host = self.at({"con": f["con"], "loc": f["loc"] & ~C.LOCATION_OVERLAY, "seq": f["seq"]})
            if host in self.overlays and cid in self.overlays[host]:
                self.overlays[host].remove(cid)
        elif (f["con"], f["loc"]) in self.zones:
            self.zones[(f["con"], f["loc"])][f["seq"]] = 0
        elif (f["con"], f["loc"]) in self.lists and f["loc"] and (f["con"], f["loc"]) not in self.unknown:
            lst = self.lists[(f["con"], f["loc"])]
            if f["seq"] < len(lst):
                lst.pop(f["seq"])
        if t["loc"] & C.LOCATION_OVERLAY:
            host = self.at({"con": t["con"], "loc": t["loc"] & ~C.LOCATION_OVERLAY, "seq": t["seq"]})
            if host is not None:
                self.overlays.setdefault(host, []).insert(t["pos"], cid)
        elif (t["con"], t["loc"]) in self.zones:
            self.zones[(t["con"], t["loc"])][t["seq"]] = cid or 0
        elif (t["con"], t["loc"]) in self.lists and (t["con"], t["loc"]) not in self.unknown:
            lst = self.lists[(t["con"], t["loc"])]
            lst.insert(min(t["seq"], len(lst)), cid)
        return cid

    def references(self, up: list[dict]) -> list[int]:
        """Card ids referenced by the upcoming messages, in order."""
        refs: list[int] = []
        self.released: list[int] = []  # cards moved with REASON_RELEASE, in order

        def add(cid):
            if cid and cid not in refs:
                refs.append(cid)

        for m in up:
            t = m["type"]
            if t == "MOVE":
                cid = self.at(m["from"])
                add(cid)
                if cid and m.get("reason", 0) & C.REASON_RELEASE and cid not in self.released:
                    self.released.append(cid)
                self.move(m)
            elif t in ("BECOME_TARGET", "CARD_SELECTED", "RANDOM_SELECTED"):
                for li in m["cards"]:
                    add(self.at(li))
            elif t in ("CONFIRM_CARDS", "CONFIRM_DECKTOP", "CONFIRM_EXTRATOP"):
                for c in m["cards"]:
                    add(self.at(c))
            elif t in ("POS_CHANGE", "SET", "SPSUMMONING", "SUMMONING", "FLIPSUMMONING", "CHAINING",
                       "ADD_COUNTER", "REMOVE_COUNTER"):
                add(self.at(m))
            elif t in ("EQUIP", "CARD_TARGET", "CANCEL_TARGET"):
                add(self.at(m["card"]))
                add(self.at(m["target"]))
            elif t == "ATTACK":
                add(self.at(m["target"]))
            elif t == "DRAW":
                p = m["player"]
                self._load(p, C.LOCATION_DECK)
                self._load(p, C.LOCATION_HAND)
                deck = self.lists[(p, C.LOCATION_DECK)]
                for _ in m["cards"]:
                    if deck:
                        self.lists[(p, C.LOCATION_HAND)].append(deck.pop())
            elif t in ("SHUFFLE_HAND", "SHUFFLE_DECK", "SHUFFLE_EXTRA"):
                loc = {"SHUFFLE_HAND": C.LOCATION_HAND, "SHUFFLE_DECK": C.LOCATION_DECK,
                       "SHUFFLE_EXTRA": C.LOCATION_EXTRA}[t]
                key = (m["player"], loc)
                forced = self.plan.pop(0) if self.plan else None
                if forced is not None and forced[:2] == key and forced[2] is not None \
                        and key not in self.unknown:
                    self._load(*key)
                    cur = self.lists[key]
                    if None not in cur and None not in forced[2] and sorted(cur) == sorted(forced[2]):
                        self.lists[key] = list(forced[2])  # the replay decides the result
                        continue
                self.unknown.add(key)
            elif t == "SWAP":
                a, b = self.at(m["loc1"]), self.at(m["loc2"])  # (loads both zones)
                self.zones[(m["loc1"]["con"], m["loc1"]["loc"])][m["loc1"]["seq"]] = b or 0
                self.zones[(m["loc2"]["con"], m["loc2"]["loc"])][m["loc2"]["seq"]] = a or 0
        return refs


PROMPT_LISTS = ("summonable", "spsummonable", "repositionable", "msetable", "ssetable", "activatable",
                "attackable", "chains", "cards", "select_cards", "unselect_cards", "must_cards")


def prompt_card_ids(duel, prompt: dict) -> list[int]:
    """Card ids of all cards a prompt lists."""
    lm = LocMap(duel)
    out = []
    for key in PROMPT_LISTS:
        for c in prompt.get(key, ()):
            if "loc" in c and c["loc"]:
                cid = lm.at(dict(c, pos=c.get("pos", 0)))
                if cid:
                    out.append(cid)
    if prompt["type"] == "SELECT_EFFECTYN":
        cid = lm.at(prompt)
        if cid:
            out.append(cid)
    return out


def response_card_ids(duel, prompt: dict, resp: dict | None) -> list[int]:
    """Card ids a response selects (the card objects the core will act on)."""
    from ..actions import BATTLE_LISTS, IDLE_LISTS
    if not resp:
        return []
    t = prompt["type"]
    cards: list[dict] = []
    if t in ("SELECT_IDLECMD", "SELECT_BATTLECMD"):
        lists = IDLE_LISTS if t == "SELECT_IDLECMD" else BATTLE_LISTS
        lst = lists.get(resp.get("kind"))
        if lst and "index" in resp:
            cards = [prompt[lst][resp["index"]]]
    elif t == "SELECT_CHAIN":
        if resp.get("index", -1) >= 0:
            cards = [prompt["chains"][resp["index"]]]
    elif t in ("SELECT_CARD", "SELECT_TRIBUTE", "SELECT_SUM"):
        cards = [prompt["cards"][i] for i in resp.get("indices", ())]
        if t == "SELECT_SUM":
            cards += prompt.get("must_cards", [])
    elif t == "SELECT_UNSELECT_CARD":
        if "index" in resp:
            allc = prompt["select_cards"] + prompt["unselect_cards"]
            cards = [allc[resp["index"]]]
    elif t in ("SELECT_EFFECTYN",):
        cards = [prompt]
    if not cards:
        return []
    lm = LocMap(duel)
    out = []
    for c in cards:
        if c.get("loc"):
            cid = lm.at(dict(c, pos=c.get("pos", 0)))
            if cid:
                out.append(cid)
    return out


def card_ids(duel, cards: list[dict]) -> list[int | None]:
    lm = LocMap(duel)
    return [lm.at(c) for c in cards]


# enumerated alternatives per decision (only a few are probed anyway)
_MAX_CANDIDATES = 64

# messages that end the action a selection belongs to: moves after them are
# consequences of later decisions
_WINDOW_END = frozenset(("CHAINING", "CHAINED", "CHAIN_SOLVED", "CHAIN_END", "NEW_PHASE", "NEW_TURN", "WIN",
                         # a summon's materials/costs move before the summon itself
                         "SUMMONING", "SPSUMMONING", "FLIPSUMMONING"))


def selection_window(up: list[dict]) -> list[dict]:
    for i, m in enumerate(up):
        if m["type"] in _WINDOW_END:
            return up[:i]
    return up


def infer_cards(prompt: dict, up: list[dict], duel, ids: list | None = None) -> tuple[list[dict], bool]:
    """SELECT_CARD / SELECT_TRIBUTE / SELECT_SUM."""
    t = prompt["type"]
    cards = prompt["cards"]
    if ids is None:
        ids = card_ids(duel, cards)
    lm = LocMap(duel)
    refs = lm.references(up)
    chosen = []
    for cid in refs:
        if cid in ids:
            chosen.append(ids.index(cid))
    lo, hi = prompt["min"], prompt["max"]
    n = len(cards)
    cands: list[dict] = []
    certain = False
    first = next((m for m in up if m["type"] not in ("HINT", "CARD_HINT")), None)
    if t == "SELECT_CARD" and first is not None and first["type"] == "CARD_SELECTED":
        # the core announces exactly the selected cards
        sel = [i for i, c in enumerate(cards) if any(same_loc(c, x) for x in first["cards"])]
        if len(sel) == len(first["cards"]) and lo <= len(sel) <= hi:
            # (not certain: the same cards can be announced by another path of
            # the script, so the other candidates stay available)
            cands.append({"indices": sel})
    if t == "SELECT_TRIBUTE":
        # valid: at most max cards whose release params sum to at least min
        def valid(sel):
            return 0 < len(sel) <= hi and sum(cards[i]["release_param"] for i in sel) >= lo
        released = [ids.index(cid) for cid in lm.released if cid in ids]
        if released and valid(released):
            cands.append({"indices": sorted(released)})
            certain = True
        for k in range(1, len(chosen) + 1):
            c = {"indices": sorted(chosen[:k])}
            if valid(chosen[:k]) and c not in cands:
                cands.append(c)
                break
        import itertools
        for k in range(1, min(hi, n) + 1):
            for sub in itertools.islice(itertools.combinations(range(n), k), _MAX_CANDIDATES):
                c = {"indices": list(sub)}
                if valid(sub) and c not in cands:
                    cands.append(c)
    elif t == "SELECT_SUM":
        from ..actions import check_sum, random_sum_selection
        import itertools
        import random as _random
        # smallest valid prefix first, then any valid subset of the references
        for k in range(1, len(chosen) + 1):
            if check_sum(prompt, sorted(chosen[:k])):
                cands.append({"indices": sorted(chosen[:k])})
                certain = k == len(chosen)
                break
        if not cands:
            for k in range(1, min(len(chosen), 5) + 1):
                for sub in itertools.islice(itertools.combinations(chosen, k), _MAX_CANDIDATES):
                    if check_sum(prompt, sorted(sub)):
                        cands.append({"indices": sorted(sub)})
        rng = _random.Random(0)
        for _ in range(8):
            sel = random_sum_selection(prompt, rng)
            if sel and {"indices": sorted(sel)} not in cands:
                cands.append({"indices": sorted(sel)})
    elif chosen:
        # the cards referenced before the action ends are the selection; later
        # references may belong to other decisions
        near = [ids.index(cid) for cid in LocMap(duel).references(selection_window(up)) if cid in ids]
        kn = min(len(near), hi)
        if near and kn >= lo:
            cands.append({"indices": sorted(near[:kn])})
            # certain only if exactly the referenced cards can be selected and no
            # later reference could be part of the selection too
            certain = kn == len(near) and (kn == hi or len(chosen) == len(near))
        k = min(len(chosen), hi)
        if k >= lo and {"indices": sorted(chosen[:k])} not in cands:
            cands.append({"indices": sorted(chosen[:k])})
        if k < len(chosen):
            import itertools
            # (only the first few candidates are ever probed: the number of
            # combinations can be astronomical for large selections)
            for sub in itertools.islice(itertools.combinations(chosen, k), _MAX_CANDIDATES):
                c = {"indices": sorted(sub)}
                if c not in cands:
                    cands.append(c)
    if t == "SELECT_CARD" and (hi == 1 or lo == 1):
        for i in range(n):
            c = {"indices": [i]}
            if c not in cands:
                cands.append(c)
    if not cands and lo <= n:
        cands.append({"indices": list(range(lo))})
    if prompt.get("cancelable"):
        cands.append({"cancel": True})
    return cands, certain


def infer_unselect(prompt: dict, up: list[dict], duel, ids: list | None = None) -> tuple[list[dict], bool]:
    sel, unsel = prompt["select_cards"], prompt["unselect_cards"]
    all_cards = sel + unsel
    if ids is None:
        ids = card_ids(duel, all_cards)
    unsel_ids = set(ids[len(sel):])
    near = LocMap(duel).references(selection_window(up))
    refs = LocMap(duel).references(up)
    for window, certain in ((near, True), (refs, False)):
        for cid in window:
            if cid in ids[:len(sel)] and cid not in unsel_ids:
                i = ids.index(cid)
                rest = [{"index": j} for j in range(len(sel)) if j != i]
                fin = [{"finish": True}] if (prompt["finishable"] or prompt["cancelable"]) else []
                return [{"index": i}] + fin + rest, certain
    cands = []
    if prompt["finishable"] or prompt["cancelable"]:
        cands.append({"finish": True})
    cands += [{"index": j} for j in range(len(sel))]
    return cands, bool(cands) and (prompt["finishable"] or prompt["cancelable"]) and bool(near)


def infer_counter(prompt: dict, up: list[dict]) -> tuple[list[dict], bool]:
    counts = [0] * len(prompt["cards"])
    for m in up:
        if m["type"] in ("HINT", "CARD_HINT"):
            continue
        # the removal follows right away (later ones belong to other actions); the
        # messages carry the counter's flags, e.g. 0x1000, the prompt may not
        if m["type"] != "REMOVE_COUNTER" or (m["counter_type"] & 0xfff) != (prompt["counter_type"] & 0xfff):
            break
        for i, c in enumerate(prompt["cards"]):
            if c["con"] == m["con"] and c["loc"] == m["loc"] and c["seq"] == m["seq"]:
                counts[i] += m["count"]
    if sum(counts) == prompt["count"]:
        return [{"counts": counts}], True
    from ..actions import random_response
    import random
    return [random_response(prompt, random.Random(0))], False


def infer_rps(prompt: dict, up: list[dict]) -> tuple[list[dict], bool]:
    for m in up:
        if m["type"] == "HAND_RES":
            r = m["result"]
            hand = (r & 3) if prompt["player"] == 0 else (r >> 2) & 3
            if 1 <= hand <= 3:
                return [{"hand": hand}], True
    return [{"hand": h} for h in (1, 2, 3)], False


def _infer_deck_sort(prompt: dict, up: list[dict]) -> list[int] | None:
    """Sorting the top/bottom cards of a deck (Duel.SortDecktop/SortDeckbottom).

    Mirrors Processors::SortDeck: tc[order[i]] = card i, then the cards are
    moved to the top (or bottom) one by one, which the viewer sees as MOVE
    messages. The permutation reproducing those moves is the answer.
    """
    import itertools
    cards = prompt["cards"]
    count = len(cards)
    if count == 0 or count > 6 or any(c["loc"] != C.LOCATION_DECK for c in cards):
        return None
    con = cards[0]["con"]
    seqs = [c["seq"] for c in cards]
    bottom = seqs[0] == 0
    observed = []
    for m in up:
        if m["type"] == "MOVE" and m["from"]["loc"] == C.LOCATION_DECK and m["to"]["loc"] == C.LOCATION_DECK \
                and m["from"]["con"] == con:
            observed.append((m["from"]["seq"], m["to"]["seq"]))
            if len(observed) == count:
                break
        elif m["type"] not in ("HINT", "CARD_HINT"):
            break
    if len(observed) != count:
        return None
    size = (max(seqs) + 1) if not bottom else None
    for perm in itertools.permutations(range(count)):
        tc = [None] * count
        for i, r in enumerate(perm):
            tc[r] = seqs[i]
        deck = list(range(size if size is not None else max(seqs) + 1 + 64))
        moves = []
        for i in range(count):
            card = tc[i] if bottom else tc[count - i - 1]
            frm = deck.index(card)
            deck.pop(frm)
            if bottom:
                deck.insert(0, card)
            else:
                deck.append(card)
            moves.append((frm, deck.index(card)))
        if moves == observed:
            return list(perm)
    return None


def infer_sort(prompt: dict, up: list[dict]) -> tuple[list[dict], bool]:
    import itertools
    n = len(prompt["cards"])
    cands = []
    if prompt["type"] == "SORT_CHAIN":
        # the triggers are chained in the chosen order (order[i] = rank of card i)
        ranks: list[int | None] = [None] * n
        k = 0
        for m in up:
            if m["type"] != "CHAINING":
                continue
            for i, c in enumerate(prompt["cards"]):
                if ranks[i] is None and c["code"] == m["code"] and c["con"] == m["trigger_con"] \
                        and c["loc"] == m["trigger_loc"] and c["seq"] == m["trigger_seq"]:
                    ranks[i] = k
                    k += 1
                    break
            if k == n:
                break
        if k:
            rest = iter(range(k, n))
            order = [r if r is not None else next(rest) for r in ranks]
            cands.append({"order": order})
            if k == n:
                return cands, True
    if prompt["type"] == "SORT_CARD":
        order = _infer_deck_sort(prompt, up)
        if order is not None:
            return [{"order": order}], True
    cands.append({"order": None})
    if n <= 5:
        cands += [{"order": list(p)} for p in itertools.permutations(range(n))]
    return cands, False
