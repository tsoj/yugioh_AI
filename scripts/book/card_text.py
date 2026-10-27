"""Card texts for a text encoder: the card's name, a type line and its effect text.

Names are replaced by pseudo-names that are consistent within one game and
different across games (`Names(seed)`): every word of a card or archetype name
(the card's own name, and names quoted in effect texts: other cards, archetypes)
becomes a made-up word, the same one everywhere in that game. Relations survive
("Blue-Eyes" in "Blue-Eyes White Dragon" and in '"Blue-Eyes" monster' get the same
substitute), but a network cannot learn cards by their names: it has to read the
effects, which is what should carry over to cards it has never seen.

  texts = Names(game_seed)
  texts.render(code)  # -> 'Zorvak Pale Wyrm. LIGHT Dragon monster, Level 8, Normal, ATK 3000 DEF 2500. This ...'
"""

from __future__ import annotations

import hashlib
import random
import re
import sqlite3
from functools import lru_cache

from ygoai import constants_gen as G
from ygoai.carddb import default_db

_UNOFFICIAL = re.compile(r"\n*\* The above text is unofficial.*$", re.S)
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _flag_names(prefix: str, pretty: dict[str, str]) -> list[tuple[int, str]]:
    out = []
    for k, v in vars(G).items():
        if k.startswith(prefix) and isinstance(v, int) and v and v & (v - 1) == 0 and not k.endswith(("_ALL", "_MAX")):
            name = k[len(prefix):]
            out.append((v, pretty.get(name, name.replace("_", " ").title())))
    return out


_ATTRIBUTES = _flag_names("ATTRIBUTE_", {})
_RACES = _flag_names("RACE_", {"WINGEDBEAST": "Winged Beast", "BEASTWARRIOR": "Beast-Warrior",
                               "SEASERPENT": "Sea Serpent", "CREATORGOD": "Creator God",
                               "MAGICALKNIGHT": "Magical Knight", "HIGHDRAGON": "High Dragon",
                               "OMEGAPSYCHIC": "Omega Psychic", "CELESTIALWARRIOR": "Celestial Warrior"})
_MONSTER_TYPES = [(f, n) for f, n in _flag_names("TYPE_", {"TRAPMONSTER": "Trap Monster", "SPSUMMON": "Special Summon only"})
                  if f not in (G.TYPE_MONSTER, G.TYPE_SPELL, G.TYPE_TRAP, G.TYPE_QUICKPLAY, G.TYPE_CONTINUOUS,
                               G.TYPE_EQUIP, G.TYPE_FIELD, G.TYPE_COUNTER)]
_ARROWS = [(G.LINK_MARKER_TOP_LEFT, "top-left"), (G.LINK_MARKER_TOP, "top"), (G.LINK_MARKER_TOP_RIGHT, "top-right"),
           (G.LINK_MARKER_LEFT, "left"), (G.LINK_MARKER_RIGHT, "right"),
           (G.LINK_MARKER_BOTTOM_LEFT, "bottom-left"), (G.LINK_MARKER_BOTTOM, "bottom"),
           (G.LINK_MARKER_BOTTOM_RIGHT, "bottom-right")]


@lru_cache(maxsize=1)
def _texts() -> dict[int, tuple[str, str]]:
    """code -> (name, effect text), from the card databases (the first file wins, as in CardDB)."""
    out: dict[int, tuple[str, str]] = {}
    for f in default_db().files:
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        try:
            for code, name, desc in con.execute("SELECT id, name, desc FROM texts"):
                out.setdefault(code, (name or "", _UNOFFICIAL.sub("", desc or "").strip()))
        finally:
            con.close()
    return out


@lru_cache(maxsize=1)
def _name_pattern() -> re.Pattern:
    """Quoted names in effect texts: every card name and every other quoted string
    (archetypes, series), the longest first (names can contain quotes: 'Maxx "C"')."""
    names = {n for n, _ in _texts().values() if n}
    names |= {q for _, d in _texts().values() for q in re.findall(r'"([^"\n]+)"', d)}
    alts = sorted(names, key=len, reverse=True)
    return re.compile('"(' + "|".join(re.escape(n) for n in alts) + ')"')


def type_line(card) -> str:
    t = card.type
    if t & G.TYPE_SPELL or t & G.TYPE_TRAP:
        kind = "Spell" if t & G.TYPE_SPELL else "Trap"
        sub = [n for f, n in ((G.TYPE_QUICKPLAY, "Quick-Play"), (G.TYPE_CONTINUOUS, "Continuous"),
                              (G.TYPE_EQUIP, "Equip"), (G.TYPE_FIELD, "Field"), (G.TYPE_COUNTER, "Counter"),
                              (G.TYPE_RITUAL, "Ritual")) if t & f]
        return " ".join(sub + [kind]) if sub else f"Normal {kind}"
    attribute = next((n for f, n in _ATTRIBUTES if card.attribute & f), "")
    race = next((n for f, n in _RACES if card.race & f), "")
    types = [n for f, n in _MONSTER_TYPES if t & f]
    parts = [f"{attribute} {race} monster".strip()]
    if t & G.TYPE_LINK:
        arrows = [n for f, n in _ARROWS if card.link_marker & f]
        parts.append(f"Link-{card.level}, arrows {', '.join(arrows)}")
    else:
        parts.append(f"{'Rank' if t & G.TYPE_XYZ else 'Level'} {card.level}")
    if t & G.TYPE_PENDULUM:
        parts.append(f"Pendulum Scale {card.lscale}")
    parts.append(", ".join(types))
    stats = f"ATK {card.attack if card.attack >= 0 else '?'}"
    if not t & G.TYPE_LINK:
        stats += f" DEF {card.defense if card.defense >= 0 else '?'}"
    parts.append(stats)
    return ", ".join(p for p in parts if p)


class Names:
    """The pseudo-names of one game (seeded): the same word always becomes the same
    made-up word, keeping its capitalization."""

    _SYL = ["ka", "ri", "zo", "mel", "tha", "vor", "en", "lu", "dra", "qui", "sor", "ba", "nyx", "el", "go",
            "fi", "rum", "ta", "ves", "or", "ji", "pa", "skal", "un", "wen", "ix", "mo", "dal", "cy", "ru"]

    def __init__(self, seed: int):
        self.seed = seed
        self._words: dict[str, str] = {}
        self._cache: dict[int, str] = {}

    def word(self, w: str) -> str:
        key = w.lower()
        out = self._words.get(key)
        if out is None:
            h = hashlib.blake2b(f"{self.seed}:{key}".encode(), digest_size=8).digest()
            rng = random.Random(int.from_bytes(h, "little"))
            n = max(1, min(4, round(len(key.replace("-", "")) / 3)))
            out = self._words[key] = "".join(rng.choice(self._SYL) for _ in range(n))
        if w.isupper() and len(w) > 1:
            return out.upper()
        return out.capitalize() if w[:1].isupper() else out

    def name(self, name: str) -> str:
        return _WORD.sub(lambda m: "-".join(self.word(p) for p in m.group().split("-") if p)
                         if "-" in m.group() else self.word(m.group()), name)

    def effect(self, text: str) -> str:
        return _name_pattern().sub(lambda m: '"' + self.name(m.group(1)) + '"', text)

    def render(self, code: int) -> str:
        """The card's text for the encoder: pseudo-name, type line, effect."""
        text = self._cache.get(code)
        if text is None:
            card = default_db().get(code)
            name, desc = _texts().get(code, ("", ""))
            if card is None:
                text = "Unknown card."
            else:
                text = f"{self.name(name)}. {type_line(card)}. {self.effect(desc)}"
            self._cache[code] = text
        return text
