"""An effect text without what tells its card's summoning type: the material line of Fusion,
Synchro, Xyz and Link Monsters, the summoning conditions and procedures of monsters ("Must be
Special Summoned ...", "Cannot be Normal Summoned/Set.", "You can Ritual Summon this card with
...", "You can also Xyz Summon this card ..."), the Pendulum section headings and the flavor text
of Pendulum Normal Monsters. The effects themselves stay (both effects of a Pendulum Monster).

  clean_effect(text, card_type) -> (text, removed parts)
"""

import re

TYPE_MONSTER, TYPE_FUSION, TYPE_RITUAL, TYPE_SYNCHRO, TYPE_XYZ, TYPE_PENDULUM, TYPE_LINK = (
    0x1, 0x40, 0x80, 0x2000, 0x800000, 0x1000000, 0x4000000)
EXTRA = TYPE_FUSION | TYPE_SYNCHRO | TYPE_XYZ | TYPE_LINK

_HEADING = re.compile(r"^\[ (Pendulum Effect|Monster Effect|Flavor Text) \]$")
_SEPARATOR = re.compile(r"^-{10,}$")
_QUOTE = re.compile(r'"[^"\n]*"')
# (a sentence's end: a period before the next sentence or the line's end, not an abbreviation's
# period inside a sentence: "(min. 2)")
_END = r"\.(?=[ \t]+[A-Z\x00(●]|[ \t]*$)"
# (sentences of a monster's text, quotes masked; each from the sentence's start to its end)
# (what belongs to a summoning procedure after it: "(You do not use "Polymerization".)", "You can
# only Special Summon "X" once per turn this way.")
_AFTER = r"(?:[ \t]+\(You do not use [^)\n]*\))?(?:[ \t]+You can only Special Summon \x00\d+\x01 once per turn this way(?:, no matter which method you use)?\.)?"
_CONDITIONS = [
    re.compile(r"(?:This card |This monster )?[Mm]ust (?:first )?be (?:either )?(?:\w+ )?Summoned\b[^\n]*?" + _END + _AFTER),
    re.compile(r"(?:This card |This monster )?[Cc]annot be (?:Normal Summoned|Special Summoned|Fusion Summoned|Synchro Summoned|Xyz Summoned|Link Summoned|Ritual Summoned)\b[^\n]*?" + _END),
    re.compile(r"This card can only be Special Summoned\b[^\n]*?" + _END + _AFTER),
    re.compile(r"You can Ritual Summon this card with\b[^\n]*?" + _END),
    re.compile(r"(?:(?:Once per turn|During [^.,\n]*?), )?(?:you|You) can also (?:Fusion|Synchro|Xyz|Link) Summon this card\b[^\n]*?" + _END
               + r"(?: \(Transfer its materials to this card\.\))?"),
]


def _mask(text):
    """The text with its quoted parts replaced by numbered placeholders (periods inside names:
    "U.A."), and the quotes."""
    quotes = []
    def put(m):
        quotes.append(m.group(0))
        return f"\x00{len(quotes) - 1}\x01"
    return _QUOTE.sub(put, text), quotes


def _unmask(text, quotes):
    return re.sub("\x00(\\d+)\x01", lambda m: quotes[int(m.group(1))], text)


def _sections(text):
    """[(heading or None, lines)]: a Pendulum text's sections, any other text one section."""
    out, head, lines = [], None, []
    for line in text.split("\n"):
        m = _HEADING.match(line.strip())
        if m:
            if head is not None or any(l.strip() for l in lines):
                out.append((head, lines))
            head, lines = m.group(1), []
        elif _SEPARATOR.match(line.strip()):
            continue
        else:
            lines.append(line)
    out.append((head, lines))
    return out


def _drop_conditions(text, removed):
    masked, quotes = _mask(text)
    # (a sentence starts at a line's start or after ". " or ") ")
    for pat in _CONDITIONS:
        def cut(m):
            removed.append(_unmask(m.group(0).strip(), quotes))
            return ""
        masked = re.sub(r"(?:^|(?<=\. )|(?<=\) ))" + pat.pattern + r"[ \t]*", cut, masked, flags=re.M)
    return _unmask(masked, quotes)


def clean_effect(text, card_type):
    removed = []
    parts = []
    sections = _sections(text.replace("\r\n", "\n").replace("\r", "\n"))
    if any(head is not None for head, _ in sections):
        removed.append("[headings]")
    for head, lines in sections:
        if head == "Flavor Text":
            removed.append("[flavor] " + " ".join(l.strip() for l in lines if l.strip()))
            continue
        body = "\n".join(lines).strip()
        if card_type & TYPE_MONSTER and head != "Pendulum Effect":
            if card_type & EXTRA and body:
                # (the material line: the first line, no sentence in it (no period at its end or
                # inside, no colon at its end), after the lines in parentheses: "(This card is
                # always treated as ...)")
                lines = body.split("\n")
                i = 0
                while i < len(lines) and lines[i].strip().startswith("("):
                    i += 1
                if i < len(lines) and not re.search(r"[.:]$|\. [A-Z\"]", lines[i].rstrip()):
                    removed.append("[materials] " + lines[i].strip())
                    body = "\n".join(lines[:i] + lines[i + 1:]).strip()
            body = _drop_conditions(body, removed)
        body = re.sub(r"[ \t]{2,}", " ", body)
        body = "\n".join(l.strip() for l in body.split("\n") if l.strip())
        if body:
            parts.append(body)
    # (a text with nothing to remove stays as it is, its whitespace too)
    return ("\n".join(parts), removed) if removed else (text, removed)
