"""Baseline players that do not read card text.

- SummonAttackAgent ("summon_attack"): the random player with three habits: it
  Normal Summons whenever it can, attacks whenever it can, and summons in face-up
  Attack Position. Everything else is random.
- HeuristicAgent ("heuristic"): a fixed order of priorities in the Main Phase
  (activate, Special Summon, Normal Summon when its field is empty or the monster
  beats the opponent's strongest face-up monster, set a monster when behind on an
  empty field, change the position of a monster in Defense Position, go to battle
  with an attacker, set a Spell/Trap, Normal Summon anyway, end the turn); in
  battle it activates, then attacks, then goes to Main Phase 2 and ends the turn.
  It chains only above 2000 LP, says yes to optional effects in its own turn only,
  and chooses cards by what the selection is for (the core's hint before it, see
  `_card_order`): the opponent's strongest cards to destroy, banish or bounce, its
  own weakest to pay costs, tribute or use as material, its own strongest to
  summon or equip, a level 4 or lower monster rather than a big one to add to
  the hand, the opponent's weakest monster to attack.

Repeated activations of the same effect are capped per turn (as in GreedyAgent)
so that effects that can be used without limit cannot loop forever.
"""

from __future__ import annotations

import random

from .. import constants as C
from ..actions import available_places, battle_options, idle_options, random_response
from ..carddb import default_db
from .base import Agent

MAX_SAME_ACTIVATION_PER_TURN = 3
CHAIN_MIN_LP = 2000

# select hints (HINTMSG_* in the card scripts' constant.lua)
_RELEASE, _DISCARD, _DESTROY, _REMOVE, _TOGRAVE, _RTOHAND, _ATOHAND, _TODECK = 500, 501, 502, 503, 504, 505, 506, 507
_SUMMON, _SPSUMMON, _FMATERIAL, _SMATERIAL, _XMATERIAL, _FACEUP, _FACEDOWN = 508, 509, 511, 512, 513, 514, 515
_EQUIP, _REMOVEXYZ, _CONTROL, _TOFIELD, _POSCHANGE, _TRIBUTE, _DEATTACHFROM, _LMATERIAL = 518, 519, 520, 527, 528, 531, 532, 533
_ATTACKTARGET, _TARGET, _NEGATE = 549, 551, 575
# own cards: the best first (else the least valuable first: costs, materials)
_OWN_BEST = {_ATOHAND, _SUMMON, _SPSUMMON, _FACEUP, _EQUIP, _TOFIELD}
# the opponent's cards: the least valuable first (else the most valuable: removal)
_OPP_WORST = {_EQUIP}
# hints with a rule (the others, or none: as listed)
_RULED = {_RELEASE, _DISCARD, _DESTROY, _REMOVE, _TOGRAVE, _RTOHAND, _ATOHAND, _TODECK, _SUMMON, _SPSUMMON,
          _FMATERIAL, _SMATERIAL, _XMATERIAL, _FACEUP, _FACEDOWN, _EQUIP, _REMOVEXYZ, _CONTROL, _TOFIELD,
          _POSCHANGE, _TRIBUTE, _DEATTACHFROM, _LMATERIAL, _ATTACKTARGET, _TARGET, _NEGATE}
UNKNOWN_VALUE = 1500  # (a face-down or unknown card)


class _Capped(Agent):
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.activations: dict[tuple, int] = {}

    def observe(self, msg: dict) -> None:
        if msg["type"] == "NEW_TURN":
            self.activations.clear()

    def _allowed(self, entry: dict) -> bool:
        key = (entry["code"], entry["loc"], entry["seq"], entry.get("desc"))
        return self.activations.get(key, 0) < MAX_SAME_ACTIVATION_PER_TURN

    def _note(self, entry: dict) -> None:
        key = (entry["code"], entry["loc"], entry["seq"], entry.get("desc"))
        self.activations[key] = self.activations.get(key, 0) + 1


class SummonAttackAgent(_Capped):
    name = "summon_attack"

    def respond(self, prompt: dict, state: dict | None) -> dict:
        t = prompt["type"]
        if t == "SELECT_IDLECMD":
            summons = [o for o in idle_options(prompt) if o["kind"] == "summon"]
            if summons:
                return self.rng.choice(summons)
        elif t == "SELECT_BATTLECMD":
            attacks = [o for o in battle_options(prompt) if o["kind"] == "attack"]
            if attacks:
                return self.rng.choice(attacks)
        elif t == "SELECT_POSITION" and prompt["positions"] & C.POS_FACEUP_ATTACK:
            return {"position": C.POS_FACEUP_ATTACK}
        return random_response(prompt, self.rng, default_db())


class HeuristicAgent(_Capped):
    name = "heuristic"
    wants_state = True

    def __init__(self, seed: int | None = None):
        super().__init__(seed)
        self.hint = None  # (the core's last select hint to this player)

    def observe(self, msg: dict) -> None:
        super().observe(msg)
        if msg["type"] == "HINT" and msg["hint_type"] == C.HINT_SELECTMSG and msg["player"] == self.player:
            self.hint = msg["data"]

    def respond(self, prompt: dict, state: dict | None) -> dict:
        try:
            return self._respond(prompt, state)
        finally:
            self.hint = None  # (a hint belongs to the selection right after it)

    def _respond(self, prompt: dict, state: dict | None) -> dict:
        t = prompt["type"]
        me = self.player
        own_turn = state is not None and state["turn_player"] == me
        if t == "SELECT_IDLECMD":
            return self._main(prompt, state)
        if t == "SELECT_BATTLECMD":
            opts = battle_options(prompt)
            for o in opts:
                if o["kind"] == "activate" and self._allowed(prompt["activatable"][o["index"]]):
                    self._note(prompt["activatable"][o["index"]])
                    return o
            for kind in ("attack", "to_m2", "to_ep"):
                for o in opts:
                    if o["kind"] == kind:
                        return o
            return opts[0]
        if t == "SELECT_CHAIN":
            if prompt["forced"]:
                return {"index": 0}
            if state is not None and state["lp"][me] > CHAIN_MIN_LP:
                for i, c in enumerate(prompt["chains"]):
                    if self._allowed(c):
                        self._note(c)
                        return {"index": i}
            return {"index": -1}
        if t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
            return {"yes": own_turn}
        if t == "SELECT_POSITION":
            card = default_db().get(prompt["code"])
            order = (C.POS_FACEUP_DEFENSE, C.POS_FACEDOWN_DEFENSE, C.POS_FACEUP_ATTACK) \
                if card is not None and card.attack <= 0 else \
                (C.POS_FACEUP_ATTACK, C.POS_FACEUP_DEFENSE, C.POS_FACEDOWN_DEFENSE)
            for pos in order + (C.POS_FACEDOWN_ATTACK,):
                if prompt["positions"] & pos:
                    return {"position": pos}
        return self._first(prompt)

    def _main(self, prompt: dict, state: dict | None) -> dict:
        me = self.player
        opts = idle_options(prompt)
        by_kind: dict[str, list[dict]] = {}
        for o in opts:
            by_kind.setdefault(o["kind"], []).append(o)
        if state is not None and state["phase"] == C.PHASE_MAIN2 and by_kind.get("to_ep"):
            return by_kind["to_ep"][0]  # (after the battle: end the turn)
        own = [c for c in (state["players"][me]["mzone"] if state else []) if c]
        opp_up = [c for c in (state["players"][1 - me]["mzone"] if state else [])
                  if c and c.get("position", 0) & C.POS_FACEUP]
        opp_best = max((c.get("attack", 0) for c in opp_up), default=-1)
        db = default_db()

        def atk(entry):
            card = db.get(entry["code"])
            return card.attack if card is not None else 0
        # a. activate a card or effect
        for o in by_kind.get("activate", ()):
            if self._allowed(prompt["activatable"][o["index"]]):
                self._note(prompt["activatable"][o["index"]])
                return o
        # b. Special Summon
        for o in by_kind.get("spsummon", ()):
            if self._allowed(prompt["spsummonable"][o["index"]]):
                self._note(prompt["spsummonable"][o["index"]])
                return o
        # c. Normal Summon: on an empty field, or a monster stronger than the opponent's best
        for o in by_kind.get("summon", ()):
            if not own or atk(prompt["summonable"][o["index"]]) > opp_best:
                return o
        # d. set a monster when behind on an empty field
        behind = state is not None and state["lp"][me] < state["lp"][1 - me]
        if behind and not own and by_kind.get("mset"):
            return by_kind["mset"][0]
        # e. change the position of a monster stuck in Defense Position
        for o in by_kind.get("repos", ()):
            c = prompt["repositionable"][o["index"]]
            here = state["players"][me]["mzone"][c["seq"]] if state and c["loc"] == C.LOCATION_MZONE else None
            if here is not None and here.get("position", 0) & C.POS_DEFENSE:
                return o
        # f. battle, with an attacker that has ATK
        attackers = [c for c in own if c.get("position", 0) & C.POS_FACEUP_ATTACK and c.get("attack", 0) > 0]
        if attackers and by_kind.get("to_bp"):
            return by_kind["to_bp"][0]
        # g. set a Spell/Trap, h. Normal Summon anyway, i. end the turn
        for kind in ("sset", "summon", "to_ep"):
            if by_kind.get(kind):
                return by_kind[kind][0]
        return opts[0]

    def _card_order(self, cards: list[dict]) -> list[int]:
        """The cards' indices, the preferred first, by the select hint: own cards
        the best first to summon, add to the hand, equip, else the least valuable
        first (costs, tributes, materials); the opponent's the most valuable first
        (removal), the weakest first as attack targets."""
        hint = self.hint
        if hint not in _RULED:
            return list(range(len(cards)))
        db = default_db()

        def key(i):
            c = cards[i]
            card = db.get(c.get("code") or 0)
            own = c.get("con", self.player) == self.player
            if hint == _ATTACKTARGET:
                return _battle_value(c, card)
            v = _to_hand_value(card) if hint == _ATOHAND else _card_value(c, card)
            if own:
                return (1, -v if hint in _OWN_BEST else v)
            return (0, v if hint in _OPP_WORST else -v)
        return sorted(range(len(cards)), key=key)

    def _first(self, prompt: dict) -> dict:
        """The preferred cards (`_card_order`), else the first place or option
        listed; random where that does not make a valid answer (sums, counters)."""
        t = prompt["type"]
        if t == "SELECT_CARD" and prompt["cards"] and prompt["max"] >= 1:
            k = min(max(prompt["min"], 1), prompt["max"], len(prompt["cards"]))
            return {"indices": sorted(self._card_order(prompt["cards"])[:k])}
        if t == "SELECT_UNSELECT_CARD":
            if prompt["select_cards"]:
                return {"index": self._card_order(prompt["select_cards"])[0]}
            if prompt["finishable"] or prompt["cancelable"]:
                return {"finish": True}
        if t == "SELECT_TRIBUTE" and prompt["cards"]:
            # (the least valuable first, until enough is released)
            chosen, total = [], 0
            for i in sorted(range(len(prompt["cards"])), key=lambda i: _card_value(
                    prompt["cards"][i], default_db().get(prompt["cards"][i].get("code") or 0))):
                if total >= prompt["min"] and chosen:
                    break
                chosen.append(i)
                total += prompt["cards"][i]["release_param"]
            if prompt["min"] <= total and len(chosen) <= max(1, prompt["max"]):
                return {"indices": sorted(chosen)}
        if t in ("SELECT_PLACE", "SELECT_DISFIELD"):
            places = available_places(prompt)
            return {"places": [list(p) for p in places[:min(prompt["count"], len(places))]]}
        if t in ("SELECT_OPTION", "ANNOUNCE_NUMBER"):
            return {"index": 0}
        if t in ("SORT_CARD", "SORT_CHAIN"):
            return {"order": list(range(len(prompt["cards"])))}
        return random_response(prompt, self.rng, default_db(), cancel_prob=0.0)


def _card_value(entry: dict, card) -> float:
    """How much a card is worth (in ATK points): a monster its ATK (DEF if higher),
    a Spell/Trap a fixed amount, a face-down or unknown card UNKNOWN_VALUE."""
    if card is None or not entry.get("code"):
        return UNKNOWN_VALUE
    if card.type & C.TYPE_MONSTER:
        return max(card.attack, card.defense, 0)
    return UNKNOWN_VALUE + 300 if card.type & C.TYPE_SPELL else UNKNOWN_VALUE


def _to_hand_value(card) -> float:
    """A card to add to the hand: a monster by its ATK, less the more tributes its
    Normal Summon needs (none for the Extra Deck's, which are not added)."""
    if card is None:
        return UNKNOWN_VALUE
    if card.type & C.TYPE_MONSTER:
        tributes = 0 if card.level <= 4 else 1 if card.level <= 6 else 2
        return max(card.attack, 0) - 1200 * tributes
    return UNKNOWN_VALUE + 300 if card.type & C.TYPE_SPELL else UNKNOWN_VALUE


def _battle_value(entry: dict, card) -> float:
    """An attack target's stat to beat: ATK in Attack Position, DEF in Defense."""
    if card is None or not entry.get("code") or entry.get("pos", 0) & C.POS_FACEDOWN:
        return UNKNOWN_VALUE
    return card.defense if entry.get("pos", 0) & C.POS_DEFENSE else card.attack
