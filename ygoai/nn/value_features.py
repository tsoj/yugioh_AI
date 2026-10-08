"""Hand-designed features for a linear value function (weights trained on game results:
scripts/train_value_linear.py).

`features(snap, me)` takes the full state (capture: in training the real game, in a
search the belief's duel) and returns, for each per-side feature, the decider's
value minus the opponent's, and whose turn it is as +1/-1: so the features of the
other player are exactly the negation, and a linear model without bias is
antisymmetric (the value of one player is minus the other's, as negamax needs).

Per side: Life Points; cards in hand, deck, Extra Deck, Graveyard, banished;
monsters on the field by position; of the face-up ones the best, worst, sum and
mean ATK, best and sum DEF, sum of Levels/Ranks/Links, how many are Normal,
Effect, Extra Deck or Tuner monsters, Xyz materials, monsters stronger than
everything the other side shows; Spell/Trap zone: set cards, face-up Spells,
face-up Traps, a Field Spell, Pendulum Scales; the damage the face-up attackers
could deal directly; in hand: monsters, Effect and Normal monsters, low (Level
<= 4) and high Level monsters, Spells, Quick-Play Spells, Traps, sum of Levels,
best ATK.
"""

from __future__ import annotations

from .. import constants as C

_EXTRA = C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK
_FIELD_ZONE = 5  # (index of the Field Zone in the Spell/Trap zone list)

_SIDE = ["lp", "hand", "deck", "extra", "grave", "removed",
         "mon", "mon_up_atk", "mon_up_def", "mon_down",
         "atk_max", "atk_min", "atk_sum", "atk_mean", "def_max", "def_sum", "level_sum",
         "mon_normal", "mon_effect", "mon_extra", "mon_tuner", "xyz_materials", "mon_dominant",
         "st_set", "spell_up", "trap_up", "field_spell", "pend_scales", "direct_damage"]
_HAND = ["hand_mon", "hand_effect", "hand_normal", "hand_low", "hand_high", "hand_spell",
         "hand_quickplay", "hand_trap", "hand_level_sum", "hand_atk_max"]


def _level(c: dict) -> int:
    t = c.get("type", 0)
    if t & C.TYPE_LINK:
        return c.get("link", 0)
    if t & C.TYPE_XYZ:
        return c.get("rank", 0)
    return c.get("level", 0)


def _battle_stat(c: dict) -> int:
    return c.get("defense", 0) if c.get("position", 0) & C.POS_DEFENSE else c.get("attack", 0)


def _side(pl: dict, other: dict, lp: int, other_lp: int) -> list[float]:
    mons = [c for c in pl["mzone"] if c is not None]
    up = [c for c in mons if c.get("position", 0) & C.POS_FACEUP]
    up_atk = [c for c in up if c.get("position", 0) & C.POS_ATTACK]
    atks = [c.get("attack", 0) for c in up]
    defs = [c.get("defense", 0) for c in up]
    other_mons = [c for c in other["mzone"] if c is not None]
    other_up = [c for c in other_mons if c.get("position", 0) & C.POS_FACEUP]
    best_other = max((_battle_stat(c) for c in other_up), default=-1)
    if len(other_up) < len(other_mons):
        best_other = max(best_other, 1500)  # (a face-down monster: some unknown DEF)
    st = [c for i, c in enumerate(pl["szone"]) if c is not None and i != _FIELD_ZONE]
    st_up = [c for c in st if c.get("position", 0) & C.POS_FACEUP]
    field = pl["szone"][_FIELD_ZONE] if len(pl["szone"]) > _FIELD_ZONE else None
    direct = sum(c.get("attack", 0) for c in up_atk) if not other_mons else 0
    return [
        lp / 8000, len(pl["hand"]), len(pl["deck"]) / 10, len(pl["extra"]) / 5, len(pl["grave"]) / 5,
        len(pl["removed"]) / 5,
        len(mons), len(up_atk), len(up) - len(up_atk), len(mons) - len(up),
        max(atks, default=0) / 1000, min(atks, default=0) / 1000, sum(atks) / 1000,
        (sum(atks) / len(atks) if atks else 0) / 1000, max(defs, default=0) / 1000, sum(defs) / 1000,
        sum(_level(c) for c in up) / 4,
        sum(1 for c in up if c.get("type", 0) & C.TYPE_NORMAL),
        sum(1 for c in up if c.get("type", 0) & C.TYPE_EFFECT),
        sum(1 for c in up if c.get("type", 0) & _EXTRA),
        sum(1 for c in up if c.get("type", 0) & C.TYPE_TUNER),
        sum(len(c.get("overlays") or ()) for c in up),
        sum(1 for c in up_atk if c.get("attack", 0) > best_other),
        len(st) - len(st_up),
        sum(1 for c in st_up if c.get("type", 0) & C.TYPE_SPELL and not c.get("type", 0) & C.TYPE_PENDULUM),
        sum(1 for c in st_up if c.get("type", 0) & C.TYPE_TRAP),
        1.0 if field is not None and field.get("position", 0) & C.POS_FACEUP else 0.0,
        sum(1 for c in st_up if c.get("type", 0) & C.TYPE_PENDULUM),
        min(direct / max(other_lp, 1), 2.0),
    ]


def _hand(hand: list) -> list[float]:
    cards = [c for c in hand if c is not None and c.get("code")]
    mons = [c for c in cards if c.get("type", 0) & C.TYPE_MONSTER]
    return [
        len(mons),
        sum(1 for c in mons if c.get("type", 0) & C.TYPE_EFFECT),
        sum(1 for c in mons if c.get("type", 0) & C.TYPE_NORMAL),
        sum(1 for c in mons if _level(c) <= 4),
        sum(1 for c in mons if _level(c) >= 5),
        sum(1 for c in cards if c.get("type", 0) & C.TYPE_SPELL),
        sum(1 for c in cards if c.get("type", 0) & C.TYPE_QUICKPLAY),
        sum(1 for c in cards if c.get("type", 0) & C.TYPE_TRAP),
        sum(_level(c) for c in mons) / 4,
        max((c.get("attack", 0) for c in mons), default=0) / 1000,
    ]


NAMES = [f"d_{n}" for n in _SIDE] + [f"d_{n}" for n in _HAND] + ["turn_sign"]


def side_features(snap: dict, p: int) -> list[float]:
    pls, lp = snap["players"], snap["lp"]
    return _side(pls[p], pls[1 - p], lp[p], lp[1 - p]) + _hand(pls[p]["hand"])


def features(snap: dict, me: int) -> list[float]:
    """The decider's features minus the opponent's, and +1/-1 for whose turn it is."""
    mine, theirs = side_features(snap, me), side_features(snap, 1 - me)
    return [a - b for a, b in zip(mine, theirs)] + [1.0 if snap["turn_player"] == me else -1.0]
