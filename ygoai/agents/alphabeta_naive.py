"""Alpha-beta on a naive guess of the hidden cards: how much the belief replay's reconstruction
is worth.

It reconstructs the duel exactly as AlphaBetaAgent does (the same work, so the same time), then
searches a copy of it in which every card of the opponent that is hidden right now (the deck, the
hand cards not revealed, face-down cards on the field, the face-down Extra Deck) gets a fresh code:
the deck completion's guess given only the cards visible now. What the belief replay knows from
the game's history (cards seen before and shuffled back, cards searched or revealed, what a
face-down card can be) is dropped; the visible state (places, positions, LP, phase, chain, the
pending prompt) stays as it is. Face-down cards get a code of their kind (a monster in a Monster
Zone, a Spell/Trap in a Spell & Trap Zone, a Field Spell in the Field Zone); the other guesses go
to the hidden hand and deck slots in random order.
"""

from __future__ import annotations

from .. import constants as C
from ..carddb import default_db
from ..state import LOCATIONS, viewer_view
from .alphabeta import AlphaBetaAgent

_FIELD = C.LOCATION_MZONE | C.LOCATION_SZONE


def _need(loc: int, seq: int) -> tuple:
    if loc == C.LOCATION_MZONE:
        return ("monster",)
    return ("field",) if seq == 5 else ("st",)


def _fits(card, need: tuple) -> bool:
    if card is None:
        return False
    if need == ("monster",):
        return bool(card.type & C.TYPE_MONSTER) and not card.is_extra_deck
    if need == ("field",):
        return bool(card.type & C.TYPE_FIELD)
    return bool(card.type & (C.TYPE_SPELL | C.TYPE_TRAP)) and not card.type & C.TYPE_FIELD


def naive_hidden_cards(duel, me: int, completion: str, rng) -> dict[int, int]:
    """New codes for the opponent's currently hidden cards (card id -> code), guessed from the
    cards visible now only (`completion`: the name of the deck completion, as the belief's)."""
    import ygoai_rs
    db = default_db()
    opp = 1 - me
    seen = viewer_view(duel, me)
    lists, _ = duel.all_cards()
    visible: list[int] = []  # (both players' visible codes)
    hidden_field, hidden_main, hidden_extra = [], [], []  # (card ids: (cid, need) / cid)
    for p in (0, 1):
        for name, loc in LOCATIONS:
            for i, entry in enumerate(seen["players"][p][name]):
                if entry is None:
                    continue
                if entry.get("code"):
                    visible.append(entry["code"])
                    continue
                if p != opp:
                    continue
                cid = lists[p, loc][i][1]
                if not cid:
                    continue
                if loc & _FIELD:
                    hidden_field.append((cid, _need(loc, i)))
                elif loc == C.LOCATION_EXTRA:
                    hidden_extra.append(cid)
                else:
                    hidden_main.append(cid)
    opp_visible = [e["code"] for name, _ in LOCATIONS for e in seen["players"][opp][name] if e and e.get("code")]
    is_extra = lambda c: db.get(c) is not None and db.get(c).is_extra_deck  # noqa: E731
    known_main = [c for c in opp_visible if not is_extra(c)]
    known_extra = [c for c in opp_visible if is_extra(c)]
    n_main, n_extra = len(hidden_field) + len(hidden_main), len(hidden_extra)
    try:
        fill_main, fill_extra = ygoai_rs.complete(completion, known_main, known_extra, n_main, n_extra, visible)
    except ValueError:
        fill_main, fill_extra = ygoai_rs.complete("vanilla", known_main, known_extra, n_main, n_extra, visible)
    fill_main, fill_extra = list(fill_main), list(fill_extra)
    out: dict[int, int] = {}
    # face-down cards on the field: a guess of their kind (else the completion's stand-in for it)
    for cid, need in hidden_field:
        k = next((j for j, c in enumerate(fill_main) if _fits(db.get(c), need)), None)
        if k is not None:
            out[cid] = fill_main.pop(k)
            continue
        stand = [c for c in ygoai_rs.stand_ins(completion, need, known_main) if _fits(db.get(c), need)]
        if stand:
            out[cid] = stand[0]
            if fill_main:
                fill_main.pop()  # (the guesses stay as many as the hidden cards)
    rng.shuffle(fill_main)
    out.update(zip(hidden_main, fill_main))
    out.update(zip(hidden_extra, fill_extra))
    return out


class AlphaBetaNaiveAgent(AlphaBetaAgent):
    name = "alphabeta_naive"

    def _select(self, duel, core_prompt: dict, core_cands: list[dict], t0: float, budget: float | None) -> int | None:
        codes = naive_hidden_cards(duel, self.player, self.completion, self.rng)
        naive = duel.clone()
        for cid, code in codes.items():
            naive.recreate_card(cid, code)
        return super()._select(naive, core_prompt, core_cands, t0, budget)
