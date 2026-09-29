"""ygopro-core constants (generated) plus a few helpers used across the package."""

from .constants_gen import *  # noqa: F401,F403
from . import constants_gen as _gen

EXTRA_DECK_TYPES = _gen.TYPE_FUSION | _gen.TYPE_SYNCHRO | _gen.TYPE_XYZ | _gen.TYPE_LINK

# Messages that require a response from a player.
SELECT_MESSAGES = frozenset({
    _gen.MSG_SELECT_BATTLECMD, _gen.MSG_SELECT_IDLECMD, _gen.MSG_SELECT_EFFECTYN,
    _gen.MSG_SELECT_YESNO, _gen.MSG_SELECT_OPTION, _gen.MSG_SELECT_CARD, _gen.MSG_SELECT_CHAIN,
    _gen.MSG_SELECT_PLACE, _gen.MSG_SELECT_POSITION, _gen.MSG_SELECT_TRIBUTE,
    _gen.MSG_SORT_CHAIN, _gen.MSG_SELECT_COUNTER, _gen.MSG_SELECT_SUM, _gen.MSG_SELECT_DISFIELD,
    _gen.MSG_SORT_CARD, _gen.MSG_SELECT_UNSELECT_CARD, _gen.MSG_ROCK_PAPER_SCISSORS,
    _gen.MSG_ANNOUNCE_RACE, _gen.MSG_ANNOUNCE_ATTRIB, _gen.MSG_ANNOUNCE_CARD,
    _gen.MSG_ANNOUNCE_NUMBER,
})

MSG_NAMES = {v: k[4:] for k, v in vars(_gen).items() if k.startswith("MSG_")}
MSG_IDS = {v: k for k, v in MSG_NAMES.items()}
