"""What a card's script is about, read statically from its text.

The scripts declare the archetypes and cards they refer to (``s.listed_series``,
``s.listed_names``); and some checks read values of several cards (how many
different Pendulum Scales, the highest Rank in the Extra Deck, a sum of Levels)
instead of asking each card a yes/no question. Neither needs the Lua to be
understood: the declarations narrow down which cards could satisfy a check, and
the engine decides whether one does.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..paths import CARD_SCRIPTS_DIR, FIXED_CARD_SCRIPTS_DIR, ROOT

# Scans of all ~38000 card scripts take ~0.1-0.2 s each, in every new process:
# their results are kept on disk, valid while the scripts and this code stay the
# same (the scripts' git commit, the corrected scripts' times, these modules).
_SCANS = ROOT / "build" / "script_scans.pkl"
_scans: dict | None = None


def _scripts_fingerprint() -> str:
    import hashlib
    import subprocess
    h = hashlib.sha1()
    try:
        h.update(subprocess.run(["git", "-C", str(CARD_SCRIPTS_DIR), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.encode())
        h.update(subprocess.run(["git", "-C", str(CARD_SCRIPTS_DIR), "status", "--porcelain"], capture_output=True,
                                text=True, timeout=30).stdout.encode())
    except (OSError, subprocess.SubprocessError):
        h.update(b"no git")
    for path in sorted(FIXED_CARD_SCRIPTS_DIR.glob("*.lua")):
        h.update(f"{path.name}:{path.stat().st_mtime_ns}".encode())
    here = __import__("pathlib").Path(__file__).parent
    for name in ("script_index.py", "completion.py"):
        h.update((here / name).read_bytes())
    return h.hexdigest()


def disk_cached(fn):
    """A whole-corpus scan, remembered on disk (and in this process)."""
    import functools

    @functools.lru_cache(maxsize=None)
    def wrapper(*args):
        global _scans
        import os
        import pickle
        if _scans is None:
            fp = _scripts_fingerprint()
            try:
                with open(_SCANS, "rb") as f:
                    stored = pickle.load(f)
                _scans = stored if stored.get("_fingerprint") == fp else {"_fingerprint": fp}
            except (OSError, EOFError, pickle.UnpicklingError, ValueError):
                _scans = {"_fingerprint": fp}
        key = (fn.__name__, args)
        if key not in _scans:
            _scans[key] = fn(*args)
            try:
                _SCANS.parent.mkdir(parents=True, exist_ok=True)
                tmp = _SCANS.with_name(f"{_SCANS.name}.{os.getpid()}")
                with open(tmp, "wb") as f:
                    pickle.dump(_scans, f)
                os.replace(tmp, _SCANS)  # (atomic: workers may write at once)
            except OSError:
                pass
        return _scans[key]
    return wrapper

# calls that read values of a group of cards (a forced filter answer cannot
# satisfy them: the values themselves are compared)
_VALUE_CHECKS = re.compile(r"GetClassCount|GetMaxGroup|GetMinGroup|CheckWithSum|GetSum\b|"
                           r"aux\.dpcheck|aux\.dncheck|GetBinClassCount|AnnounceNumber\(")
_LISTED = re.compile(r"s\.listed_(series|names)\s*=\s*\{([^}]*)\}")
_CONST = re.compile(r"^\s*((?:SET|CARD|RACE)_[A-Z0-9_]+)\s*=\s*(0x[0-9a-fA-F]+|\d+)", re.M)
_RACE_CHECK = re.compile(r"IsRace,?\s*\(?\s*(?:c\s*,\s*)?((?:RACE_[A-Z_]+\s*[|+]?\s*)+)")


@disk_cached
def _constants() -> dict[str, int]:
    out: dict[str, int] = {}
    for path in CARD_SCRIPTS_DIR.glob("*.lua"):
        for name, value in _CONST.findall(path.read_text(errors="replace")):
            out[name] = int(value, 0)
    return out


def _script_text(code: int) -> str:
    for path in [FIXED_CARD_SCRIPTS_DIR / f"c{code}.lua"] + [
            CARD_SCRIPTS_DIR / sub / f"c{code}.lua" for sub in ("official", "pre-release", "unofficial")]:
        if path.exists():
            return path.read_text(errors="replace")
    return ""


@lru_cache(maxsize=None)
def script_info(code: int) -> tuple[frozenset[int], frozenset[int], bool]:
    """(listed archetype codes, listed card codes, whether it checks card values)."""
    text = _script_text(code)
    consts = _constants()
    series: set[int] = set()
    names: set[int] = set()
    for kind, body in _LISTED.findall(text):
        for tok in (t.strip() for t in body.split(",")):
            if not tok:
                continue
            if tok == "id":
                value = code
            elif re.fullmatch(r"0x[0-9a-fA-F]+|\d+", tok):
                value = int(tok, 0)
            else:
                value = consts.get(tok)
            if value is not None:
                (series if kind == "series" else names).add(value)
    return frozenset(series), frozenset(names), bool(_VALUE_CHECKS.search(text))


def is_set_card(card_setcodes, setcode: int) -> bool:
    """Card.IsSetCard: the archetype matches, and the sub-archetype bits if any."""
    return any((s & 0xfff) == (setcode & 0xfff) and (s & setcode) == setcode for s in card_setcodes)


@lru_cache(maxsize=None)
def piles(code: int) -> frozenset[str]:
    """The hidden piles the script refers to ("deck", "extra")."""
    text = _script_text(code)
    return frozenset(name for name, loc in (("deck", "LOCATION_DECK"), ("extra", "LOCATION_EXTRA"))
                     if loc in text)


def candidates(code: int, db) -> list[int]:
    """Cards the script of `code` refers to (its archetypes and named cards),
    legal ones, by code."""
    series, names, _ = script_info(code)
    out = []
    for c in db.cards.values():
        if c.alias or not c.ot & 0x3 or c.code == code:
            continue
        if c.code in names or (series and any(is_set_card(c.setcodes, s) for s in series)):
            out.append(c.code)
    return sorted(out)


@disk_cached
def proc_cards(kind: str) -> frozenset[int]:
    """Cards whose scripts give them their own Normal Summon ("summon") or Set
    ("mset") procedure that can Tribute the opponent's monsters (e.g. Thestalos
    the Shadowfire Monarch: 1 monster the opponent controls and 1 Tribute
    Summoned monster)."""
    code_name = {"summon": "EFFECT_SUMMON_PROC", "mset": "EFFECT_SET_PROC"}[kind]
    out = set()
    for sub in ("official", "pre-release", "unofficial"):
        for path in (CARD_SCRIPTS_DIR / sub).glob("c*.lua"):
            text = path.read_text(errors="replace")
            if code_name in text and "1-tp" in text and ("Release" in text or "RELEASE" in text):
                try:
                    out.add(int(path.stem[1:]))
                except ValueError:
                    pass
    return frozenset(out)


@disk_cached
def discard_set_proc_cards() -> frozenset[int]:
    """Cards whose own Set procedure discards cards as its cost (e.g. Infernal
    Incinerator: "discarding your entire hand and Tributing 1 monster")."""
    out = set()
    for sub in ("official", "pre-release", "unofficial"):
        for path in (CARD_SCRIPTS_DIR / sub).glob("c*.lua"):
            text = path.read_text(errors="replace")
            if ("EFFECT_SET_PROC" in text or "EFFECT_LIMIT_SET_PROC" in text) and "REASON_DISCARD" in text:
                try:
                    out.add(int(path.stem[1:]))
                except ValueError:
                    pass
    return frozenset(out)


_SET_PROC = re.compile(r"AddNormalSetProcedure\(\s*c\s*,[^,]*,[^,]*,\s*(\d+)\s*,\s*(\d+)")


@disk_cached
def set_proc_tribute_cards(n: int) -> frozenset[int]:
    """Cards whose own Set procedure Tributes `n` monsters (e.g. The Wicked
    Avatar: 3; no Level asks for more than 2)."""
    out = set()
    for sub in ("official", "pre-release", "unofficial"):
        for path in (CARD_SCRIPTS_DIR / sub).glob("c*.lua"):
            text = path.read_text(errors="replace")
            for lo, hi in _SET_PROC.findall(text):
                if int(lo) <= n <= int(hi):
                    try:
                        out.add(int(path.stem[1:]))
                    except ValueError:
                        pass
    return frozenset(out)


@lru_cache(maxsize=None)
def reveals_itself(code: int) -> bool:
    """Whether the card's own procedure reveals it together with other cards
    (e.g. Evil Dragon Cards: "revealing 3 LIGHT monsters in your hand, including
    this card"), as opposed to revealing other cards only (Crimson Nova the Dark
    Cubic Lord: 3 other Cubic cards)."""
    text = _script_text(code)
    return bool(re.search(r"Group\.FromCards\(c\)|:IsContains\(c\)", text))


@lru_cache(maxsize=None)
def has_trigger(code: int) -> bool:
    """Whether the card's script has a trigger effect (EFFECT_TYPE_TRIGGER_O/F)."""
    return "EFFECT_TYPE_TRIGGER_" in _script_text(code)


@lru_cache(maxsize=None)
def unsummonable(code: int) -> bool:
    """Whether the card cannot be Normal Summoned or Set by its own script (e.g. "Must
    be Special Summoned by ..."): Card.EnableUnsummonable, EFFECT_UNSUMMONABLE_CARD,
    and Card.EnableReviveLimit (on a Main Deck monster: Special Summon only)."""
    text = _script_text(code)
    return bool(re.search(r"EnableUnsummonable\(|EFFECT_UNSUMMONABLE_CARD|EnableReviveLimit\(", text))


@disk_cached
def win_condition_codes() -> frozenset[int]:
    """Cards that can win the game by themselves (their script calls Duel.Win, e.g.
    Exodia the Forbidden One, Destiny Board) and the cards those scripts name (e.g.
    the Forbidden One pieces): as stand-ins for unknown cards they would change the
    game with no player action (a hand of guessed pieces wins)."""
    out: set[int] = set()
    paths = [p for sub in ("official", "pre-release", "unofficial") for p in (CARD_SCRIPTS_DIR / sub).glob("c*.lua")]
    paths += list(FIXED_CARD_SCRIPTS_DIR.glob("c*.lua"))
    for path in paths:
        text = path.read_text(errors="replace")
        if "Duel.Win(" in text:
            out.add(int(path.stem[1:]))
            out.update(int(x) for x in re.findall(r"\b(\d{6,9})\b", text))
    return frozenset(out)


_FUNCTION = re.compile(r"(?ms)^(?:local\s+)?function\s+([\w.:]+)\s*\(.*?^end\b")
_GLOBAL_REGISTRATION = re.compile(r"Effect\.GlobalEffect\(|GlobalCheck\(")


@lru_cache(maxsize=1)
def _global_helpers() -> frozenset[str]:
    """Functions of the shared scripts that register a duel-wide effect once per
    duel for whichever card calls them first (e.g. aux.EnableCheckReincarnation),
    by the last part of their name."""
    out: set[str] = set()
    for path in CARD_SCRIPTS_DIR.glob("*.lua"):
        for m in _FUNCTION.finditer(path.read_text(errors="replace")):
            name = re.split(r"[.:]", m.group(1))[-1]
            if name != "GlobalCheck" and _GLOBAL_REGISTRATION.search(m.group(0)):
                out.add(name)
    return frozenset(out)


@lru_cache(maxsize=None)
def global_keys(code: int) -> frozenset:
    """The duel-wide effects a card's script registers when the card is created:
    its own (aux.GlobalCheck, Effect.GlobalEffect: once per script, keyed by the
    code) and a shared helper's (once per duel, keyed by the helper's name). Such an
    effect is in the game as soon as one card with it is, wherever the card is, and
    it can change how the engine runs (e.g. Salamangreat's grant of a Reincarnation
    check to every Fusion, Ritual, Synchro, Xyz and Link Monster adds an adjustment
    pass whenever one of them moves)."""
    text = _script_text(code)
    out: set = set()
    if _GLOBAL_REGISTRATION.search(text):
        out.add(code)
    out.update(h for h in _global_helpers() if re.search(r"\b" + h + r"\s*\(", text))
    return frozenset(out)


@lru_cache(maxsize=None)
def races(code: int) -> int:
    """The races the script of `code` checks for (IsRace(RACE_...)), as a mask."""
    consts = _constants()
    out = 0
    for group in _RACE_CHECK.findall(_script_text(code)):
        for name in re.findall(r"RACE_[A-Z_]+", group):
            out |= consts.get(name, 0)
    return out
