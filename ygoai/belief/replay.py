"""Belief replay: reconstruct a duel that looks identical from the viewer's side.

Given the viewer's observation stream (as produced by the arbiter) this builds
a hypothesis of all hidden information (`tracker.py`), creates a duel from it
and replays it in lockstep with the stream:

* shuffles, coin tosses, dice rolls and random selections are forced to the
  observed / hypothesized outcome via the core extension,
* the viewer's prompts must match the stream exactly and are answered with the
  viewer's recorded responses,
* the opponent's prompts are answered by inferring the decision from what the
  viewer observed next (`infer.py`); ambiguous decisions are verified by
  probing the candidates in forked processes.

Every message the replay produces is filtered for the viewer and compared to
the stream, and at every viewer prompt the viewer's view of the full state is
compared as well.

Some divergences can be repaired, after which the replay restarts:

* "force": the opponent activated an effect that is not activatable in the
  replay because its cost/target check depends on hidden cards (e.g. a search
  that got negated, so the searched card was never revealed). The effect is
  made activatable regardless of those checks.
* "assume": the opponent took a hidden card for an effect (e.g. set a card
  from the deck face-down) but the hypothesis has a card there that the
  effect cannot select. A valid choice is assumed to have been there, which
  becomes an extra observation for the hypothesis.
* "swap": two copies of the same card have to swap places (e.g. a script
  took the copy at another position of a pile than the hypothesis has).
* "match"/"matchpile": a filter (of a selection, or of a check that decides
  which options exist) accepted a hidden card in the real game that the
  replay's card does not satisfy: during that step, the card (or every card of
  the opponent's hidden piles) matches every filter.

Any other difference raises `Divergence`.
"""

from __future__ import annotations

import copy
from collections import Counter
import os
import signal
import sys
import time
from dataclasses import dataclass

from .. import constants as C
from ..actions import InvalidResponse, encode
from ..carddb import CardDB, default_db
from ..core import Duel, DuelConfig
from ..decks import Deck
from ..game import setup_duel
from ..messages import parse
from ..observe import AUTO_RESPONSE, filter_message, filter_prompt, is_hidden_prompt, to_core_response
from ..state import LOCATIONS, capture, view
from . import infer
from .completion import DeckCompletion, VanillaCompletion
from . import tracker as tracker_mod
from .tracker import Hypothesis, Tracker, TrackerError

STATUS_END, STATUS_AWAITING, STATUS_CONTINUE = 0, 1, 2
PROBE_OK, PROBE_FAILED, PROBE_TIMEOUT, PROBE_REPAIRABLE, PROBE_PROGRESSED = 0, 1, 3, 4, 5
PROBE_REPAIRABLE_AT_ONCE = 6  # repairable, but at the very first item after the decision
# messages after which moves no longer result from the current selection
# (a summon ends the selection of what it summons: later moves are other actions)
# (and a cost's selection ends when the chain link is complete)
_SELECTION_END = frozenset(("CHAINING", "CHAINED", "CHAIN_SOLVING", "CHAIN_SOLVED", "CHAIN_END", "NEW_PHASE",
                            "NEW_TURN", "WIN", "SUMMONING", "SPSUMMONING", "FLIPSUMMONING"))
HIDDEN_PILES = (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_EXTRA)
# where an action's consequences are over (a probe decides only after one)
_BOUNDARIES = frozenset(("CHAIN_SOLVED", "CHAIN_END", "NEW_PHASE", "NEW_TURN", "SUMMONED", "SPSUMMONED",
                         "FLIPSUMMONED", "CHAINED"))
# messages by which the real game goes on after an effect did (nothing) more
_NOTHING_MORE = frozenset(("CHAIN_SOLVED", "CHAIN_END", "PLAYER_HINT", "NEW_PHASE", "NEW_TURN"))
UNORDERED_HINTS = ("PLAYER_HINT", "CARD_HINT")
_SHUFFLE_LOCS = {"SHUFFLE_HAND": C.LOCATION_HAND, "SHUFFLE_DECK": C.LOCATION_DECK,
                 "SHUFFLE_EXTRA": C.LOCATION_EXTRA}
REVEALED = "revealed"  # identity: one of the cards revealed before the shuffle (see _revealed_fix)
CARD_PROMPTS = ("SELECT_CARD", "SELECT_UNSELECT_CARD", "SELECT_TRIBUTE", "SELECT_SUM")


class _ProbeSuccess(Exception):
    """Raised inside a probe once it has verified enough of the stream."""


class Divergence(RuntimeError):
    def __init__(self, pos: int, reason: str, expected=None, got=None, fix: tuple | None = None):
        super().__init__(f"divergence at item {pos}: {reason}")
        self.pos = pos
        self.reason = reason
        self.expected = expected
        self.got = got
        self.fix = fix
        self.fallback: tuple | None = None  # a repair for when no decision change helps


def _stat_mismatch(e: Divergence) -> bool:
    """A message that shows stats (e.g. the ATK of an attack) differs only in
    them: like a state view, it shows the result of every decision since the
    last one (e.g. which monster an effect raised)."""
    exp = e.expected[1] if isinstance(e.expected, tuple) and len(e.expected) > 1 else e.expected
    got = e.got
    if not isinstance(exp, dict) or not isinstance(got, dict) or exp.get("type") != got.get("type"):
        return False
    diff = {k for k in set(exp) | set(got) if exp.get(k) != got.get(k)}
    return bool(diff) and all(isinstance(k, str) and k.endswith(("_atk", "_def")) for k in diff)


# messages that leave every card where it was (the rolling snapshot stays valid)
_NO_MOVES = frozenset((
    "HINT", "CARD_HINT", "PLAYER_HINT", "CHAINING", "CHAINED", "CHAIN_SOLVING", "CHAIN_SOLVED",
    "CHAIN_END", "CHAIN_NEGATED", "CHAIN_DISABLED", "NEW_PHASE", "NEW_TURN", "POS_CHANGE",
    "SUMMONING", "SUMMONED", "SPSUMMONING", "SPSUMMONED", "FLIPSUMMONING", "FLIPSUMMONED",
    "BECOME_TARGET", "CONFIRM_CARDS", "DAMAGE", "RECOVER", "PAY_LPCOST", "LPUPDATE", "ATTACK",
    "BATTLE", "ATTACK_DISABLED", "DAMAGE_STEP_START", "DAMAGE_STEP_END", "ADD_COUNTER",
    "REMOVE_COUNTER", "FIELD_DISABLED", "CARD_SELECTED", "RANDOM_SELECTED", "SET"))


def _all_responses(prompt: dict) -> list[dict]:
    """The legal answers to an opponent prompt that inference may have left out
    (card selections only one card at a time; zones and announcements, whose
    choices the stream shows directly, not at all)."""
    from ..actions import battle_options, idle_options
    t = prompt["type"]
    if t == "SELECT_IDLECMD":
        return idle_options(prompt)
    if t == "SELECT_BATTLECMD":
        return battle_options(prompt)
    if t == "SELECT_CHAIN":
        return ([] if prompt["forced"] else [{"index": -1}]) + [{"index": j} for j in range(len(prompt["chains"]))]
    if t in ("SELECT_EFFECTYN", "SELECT_YESNO"):
        return [{"yes": True}, {"yes": False}]
    if t == "SELECT_OPTION":
        return [{"index": j} for j in range(len(prompt["options"]))]
    if t == "SELECT_POSITION":
        return [{"position": p} for p in (C.POS_FACEUP_ATTACK, C.POS_FACEDOWN_ATTACK, C.POS_FACEUP_DEFENSE,
                                          C.POS_FACEDOWN_DEFENSE) if prompt["positions"] & p]
    if t in ("SELECT_CARD", "SELECT_TRIBUTE") and prompt["min"] <= 1 <= prompt["max"]:
        return [{"indices": [j]} for j in range(len(prompt["cards"]))]
    if t == "SELECT_UNSELECT_CARD":
        fin = [{"finish": True}] if prompt.get("finishable") or prompt.get("cancelable") else []
        return [{"index": j} for j in range(len(prompt["select_cards"]))] + fin
    return []


_DECISION_WINDOW = 10  # stream positions a decision stays a candidate for retrying
_DECISION_TRIES = 24  # alternatives tried for one mismatch


def first_difference(a, b, path: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side ({a.get(k)!r} vs {b.get(k)!r})"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


@dataclass
class ReplayStats:
    opponent_decisions: int = 0
    probed_decisions: int = 0
    probes: int = 0
    restarts: int = 0
    forces: int = 0
    assumptions: int = 0
    swaps: int = 0


def _loc_of(msg: dict) -> dict | None:
    """The location a (god view or filtered) message is primarily about."""
    if msg["type"] == "MOVE":
        return msg["from"]
    if "loc" in msg and "con" in msg and "seq" in msg:
        return msg
    return None


class BeliefReplay:
    def __init__(self, viewer: int, init: dict, items: list[tuple], db: CardDB | None = None,
                 completion: DeckCompletion | None = None, check_state: bool = True,
                 max_probes: int = 24, probe_items: int = 60, max_depth: int = 4,
                 max_restarts: int = 40, probe_timeout: int = 20, max_stall: int = 300,
                 min_progress: int = 4, stopped: bool = False):
        self.viewer = viewer
        # the stream is a whole game that the arbiter stopped (turn or decision
        # limit) at the prompt after its last item
        self.stopped = stopped
        self.opponent = 1 - viewer
        self.init = init
        self.items = list(items)
        self.db = db or default_db()
        self.completion = completion or VanillaCompletion(self.db)
        self.check_state = check_state
        self.max_probes = max_probes
        self.probe_items = probe_items
        self.max_depth = max_depth
        self.max_restarts = max_restarts
        self.probe_timeout = probe_timeout
        self.max_stall = max_stall  # opponent decisions without new observations
        self.min_progress = min_progress  # items a probe must reproduce before it defers
        # a probe also checks the viewer's first state view if it comes within this
        # many items (a decision may show only there)
        self.probe_state_window = int(os.environ.get("YGOAI_PROBE_STATE_WINDOW", 8))
        self.stats = ReplayStats()
        self.trace: list[tuple] | None = None  # (pos, prompt, response, certain, n_candidates)
        self.log = None  # optional callable(str) for debugging
        self.sim_log = None  # optional list: god view events of the latest attempt (debugging)
        # repairs, persistent across restarts
        # Every effect the opponent was seen activating may be forced to be
        # activatable (skipping cost/target checks, which may depend on hidden
        # cards, when they fail) around its activation (see force_windows);
        # "unforce" repairs the rare case where that makes a mandatory effect
        # activate when it did not.
        self.forces: set[tuple[int, int, int]] = {
            (self.opponent, it[1]["code"], it[1]["desc"]) for it in self.items
            if it[0] == "msg" and it[1]["type"] == "CHAINING" and it[1]["trigger_con"] == self.opponent}
        self.unforced: set[tuple[int, int, int]] = set()
        self.unforced_windows: set[tuple] = set()  # force windows switched off by "unforce"
        # the forcing applies only around the activation: from the start of its
        # phase until just after it (elsewhere a forced mandatory effect could
        # activate where the real one could not)
        self.force_windows: list[tuple[int, int, tuple]] = []
        start = 0
        chaining = None
        for i, it in enumerate(self.items):
            if it[0] != "msg":
                continue
            m = it[1]
            if m["type"] in ("NEW_PHASE", "NEW_TURN"):
                start = i
            elif m["type"] == "CHAINING" and m["trigger_con"] == self.opponent:
                chaining = m["code"]
                self.force_windows.append((start, i + 1, (self.opponent, m["code"], m["desc"])))
            elif m["type"] == "CHAINED":
                chaining = None
            elif m["type"] == "HINT" and m["hint_type"] == C.HINT_OPSELECTED and m["player"] == self.opponent \
                    and chaining is not None and m["data"] >> 20 == chaining:
                # a continuous trap activated together with one of its effects
                # (field::get_cteffect): that effect was activatable
                key = (self.opponent, chaining, m["data"])
                self.forces.add(key)
                self.force_windows.append((start, i + 1, key))
        self.nomatch_id_spans: set[tuple] = set()  # (from, to, card id): matches no filter then
        self.assumptions: list[tuple[int, int, int, int, int]] = []  # (pos, con, loc, seq, code)
        self.exclusions: list[tuple] = []  # (pos, con, loc, seqs, code)
        # (pos, slot, var id): the card in that slot is the physical card of that
        # (earlier) var, as the replay's scripts showed
        self.identities: list[tuple] = []
        self.failed_identities: set[tuple] = set()  # identities that made the model infeasible
        self._folded_swaps: set = set()
        self.swaps: dict[int, list[tuple[int, int]]] = {}
        self.matches: dict[int, list[int]] = {}  # control point -> card ids matching all filters
        self.match_piles: set[tuple[int, int]] = set()  # stream positions (from, to): hidden piles match
        # (start, end, slot): the opponent's hidden card in that slot matches no filter
        self.nomatch_cards: set[tuple] = set()
        self.nomatch_piles: set[tuple[int, int]] = set()  # ... hidden piles match nothing
        self.snapshot_cps: set[int] = set()
        self.failed_assumptions: set[tuple] = set()
        self.failed_exclusions: set[tuple] = set()
        self.failed_swaps: set[tuple] = set()  # (cp, a, b) swaps that made an earlier observation differ
        self._state_bt: dict | None = None  # backtracking over the decisions before a state mismatch
        self._last_swap: tuple | None = None  # (fix, position it was for)
        # one-decision backtracking: (stream position, index among the decisions
        # at that position) -> candidate to use instead of the inferred one
        self.overrides: dict[tuple[int, int], int] = {}
        # decisions answered with a response of their own ("respond" repairs: a card
        # selection changed to include a card that later shows to have been in it)
        self.forced_resp: dict[tuple[int, int], dict] = {}
        self.tried_resp: set[tuple] = set()
        self._last_hyp_fix: str | None = None
        # opponent decisions of earlier attempts: (pos, prompt) -> response
        self.memo: dict[tuple, tuple] = {}  # -> (answer, (candidates, index) for the decision search)
        self.use_memo = True
        self.hypothesis: Hypothesis | None = None
        self._reset()

    def _simlog(self, *entry) -> None:
        if self.sim_log is not None and self.depth == 0:
            self.sim_log.append((self.pos, *entry))

    def _reset(self) -> None:
        if getattr(self, "sim_log", None) is not None:
            self.sim_log = []
        self.pos = 0
        # control points identify calls of duel.process() as (stream position,
        # number of calls since that position was reached), which stays valid
        # when a repair changes how many steps happen earlier in the replay
        self._cp_pos = -1
        self._cp_n = 0
        self.carry_swaps: list[tuple[int, int]] = []
        # card id -> (control point, (con, pile)) of its latest move from a pile to the field
        self.left_pile: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}
        self.last_listed: dict[int, tuple[int, int]] = {}  # card id -> step of the latest prompt listing it
        self.prompt_cp = None  # the step that produced the most recent prompt
        self.depth = 0
        self.probe_start = 0
        self.probe_state_at: int | None = None  # (probes) the state view a probe must check
        self.horizon = len(self.items)
        self.stall = 0
        self.last_shuffle_cp: dict[tuple[int, int], int] = {}
        self.card_shuffled: dict[int, list] = {}  # card id -> cps of the shuffles of its pile
        self._shuffle_count = 0  # shuffles so far (index into the hypothesis' shuffles)
        self._event_hint = None  # (stream position, data) of the opponent's latest timing event
        self._decisions_at: dict[int, int] = {}  # stream position -> decisions made there
        # (key, candidates, chosen) of the recent decisions: those of the last few
        # stream positions (a decision's consequence can show a few items later,
        # e.g. after the hint of a replacement effect)
        self._recent: list[tuple] = []
        self._since_state: list[tuple] = []  # the same, since the last state view that matched
        self._cancelled: set[tuple] = set()  # (stream position, prompt) answered by cancelling
        self._opp_prompt: dict | None = None  # the opponent's latest prompt
        # where the previous attempt shuffled (for folding swaps into shuffles)
        self.prev_shuffle_cps = getattr(self, "shuffle_cps", {})
        self.shuffle_cps: dict[tuple[int, int], list[int]] = {}
        self.snapshots: dict[int, infer.LocMap] = {}
        self._roll_cp = None  # the latest main-line step's snapshot (kept only until the next)
        self._card_decisions: dict[tuple, dict[int, int]] = {}  # decision -> card id -> answer index
        self._window_choices: list[tuple] = []  # (decision, activation answer) passed in leave windows
        self._hidden_picks: list[tuple] = []  # (position, span) of opponent choices among its hidden cards
        self._multi_decisions: dict[tuple, tuple] = {}  # decision -> (card ids, chosen response)
        self._roll_dirty = True  # whether a card may have moved since that snapshot
        self.lockstep = Tracker(self.viewer, self.init)

    # -- setup -----------------------------------------------------------------
    def build_hypothesis(self) -> Hypothesis:
        tracker = Tracker(self.viewer, self.init)
        pending = sorted(self.assumptions)
        excl = sorted(self.exclusions)
        k = j = 0
        for i, item in enumerate(self.items):
            while k < len(pending) and pending[k][0] == i:
                _, con, loc, seq, code = pending[k]
                tracker.observe(tracker.var_at(con, loc, seq), code)
                k += 1
            while j < len(excl) and excl[j][0] == i:
                _, con, loc, seqs, code = excl[j]
                for seq in seqs:
                    tracker.exclude(tracker.var_at(con, loc, seq), code)
                j += 1
            revealed = []
            for pos, (con, loc, seq), same in self.identities:
                if pos == i:
                    if same == REVEALED:
                        revealed.append(tracker.var_at(con, loc, seq))
                        continue
                    if isinstance(same, tuple) and same[0] == "not":
                        byid = {v.id: v for v in tracker.all_vars}
                        v = tracker.var_at(con, loc, seq)
                        if v is not None and same[1] in byid:
                            v.not_same.append(byid[same[1]])
                        continue
                    byid = {v.id: v for v in tracker.all_vars}
                    if same in byid:
                        tracker._identify(tracker.var_at(con, loc, seq), byid[same])
            tracker.process(item)
            for v in revealed:
                # the card moved from that slot is one of the cards revealed before
                # the shuffle (the tracker's preference, made a constraint)
                if v is not None and v.kind == "claim" and v.pool is not None and v.prefer:
                    ids = frozenset(m.id for m in v.pool.members if any(m is x for x in v.prefer))
                    if not ids:
                        # (shuffled more than once since: the revealed cards themselves,
                        # which the flow follows through the shuffles)
                        ids = frozenset(x.id for x in v.prefer)
                    if ids:
                        v.allowed = ids if v.allowed is None else (v.allowed & ids) or v.allowed
        self.hypothesis = tracker.solve(self.completion, self.db)
        self._solved = tracker
        return self.hypothesis

    def build_duel(self) -> Duel:
        hyp = self.hypothesis or self.build_hypothesis()
        decks = [Deck(main=hyp.decks[p]["main"], extra=hyp.decks[p]["extra"]) for p in (0, 1)]
        cfg = DuelConfig(**self.init["config"])
        duel = setup_duel(decks, [0x1234, 0x5678, 0x9ABC, 0xDEF0], cfg, hyp.deck_orders, self.db)
        # which physical copy is where: copies of a code are created in rank order
        id_of: dict[tuple[int, int, int], int] = {}
        for p in (0, 1):
            for loc in (C.LOCATION_DECK, C.LOCATION_EXTRA):
                n: dict[int, int] = {}
                for code, cid in sorted(duel.location_cards(p, loc), key=lambda x: x[1]):
                    id_of[(p, code, n.get(code, 0))] = cid
                    n[code] = n.get(code, 0) + 1

        def ids(player, codes, ranks):
            if ranks is None:
                return None
            out = [id_of.get((player, c, r)) for c, r in zip(codes, ranks)]
            return None if None in out else out

        # a swap at the very start exchanges two copies' roles for the whole duel
        # (every placement by card id included)
        roles = self._roles(self.swaps.get((0, 0), ()))

        def swap_roles(lst):
            return None if lst is None else [roles.get(x, x) for x in lst]
        for p in (0, 1):
            deck_ids = swap_roles(ids(p, hyp.deck_orders[p], hyp.deck_ranks[p] if hyp.deck_ranks else None))
            if deck_ids is not None and not duel.set_location_order_by_id(p, C.LOCATION_DECK, deck_ids):
                duel.close()
                raise Divergence(0, f"cannot set deck order of player {p} by card id")
            if hyp.extra_orders[p]:
                extra_ids = swap_roles(ids(p, hyp.extra_orders[p], hyp.extra_ranks[p] if hyp.extra_ranks else None))
                ok = (duel.set_location_order_by_id(p, C.LOCATION_EXTRA, extra_ids) if extra_ids is not None
                      else duel.set_location_order(p, C.LOCATION_EXTRA, hyp.extra_orders[p]))
                if not ok:
                    duel.close()
                    raise Divergence(0, f"cannot set extra deck order of player {p}")
        shuffle_ids = [swap_roles(ids(sh["player"], sh["codes"], sh.get("ranks"))) for sh in hyp.shuffles]
        # a copy swap scheduled for the step that shuffles the pile exchanges the
        # two cards in that shuffle's result
        self._folded_swaps = {((0, 0), a, b) for a, b in self.swaps.get((0, 0), ())}
        shuffle_cp = {k: c for c, ks in self.prev_shuffle_cps.items() for k in ks}
        for cp in sorted(c for c in self.swaps if c != (0, 0)):
            pairs = self.swaps[cp]
            here = set(self.prev_shuffle_cps.get(cp, ()))
            for k in here:
                lst = shuffle_ids[k] if k < len(shuffle_ids) else None
                if lst is None:
                    continue
                for a, b in pairs:
                    if a in lst and b in lst:
                        i, j = lst.index(a), lst.index(b)
                        lst[i], lst[j] = lst[j], lst[i]
                        self._folded_swaps.add((cp, a, b))
            # from then on each of the two copies continues the other's path: the
            # later shuffles place them the other way round
            for k, lst in enumerate(shuffle_ids):
                # (shuffles the previous attempt did not reach come later)
                if lst is None or k in here or (k in shuffle_cp and shuffle_cp[k] <= cp):
                    continue
                for a, b in pairs:
                    shuffle_ids[k] = lst = [b if x == a else a if x == b else x for x in lst]
        self._shuffle_ids = shuffle_ids
        for sh, sh_ids in zip(hyp.shuffles, shuffle_ids):
            if sh_ids is not None:
                duel.push_forced_shuffle_ids(sh["player"], sh["location"], sh_ids, sh["codes"])
            else:
                duel.push_forced_shuffle(sh["player"], sh["location"], sh["codes"])
        for codes in hyp.set_card_shuffles:
            duel.push_forced_set_card_shuffle(codes)
        for item in self.items:
            if item[0] != "msg":
                continue
            m = item[1]
            if m["type"] == "TOSS_COIN":
                duel.push_forced_coins(m["results"])
            elif m["type"] == "TOSS_DICE":
                duel.push_forced_dice(m["results"])
            elif m["type"] == "RANDOM_SELECTED":
                duel.push_forced_random_select(
                    [(c["con"], c["loc"], c["seq"], c["pos"] if c["loc"] & C.LOCATION_OVERLAY else 0)
                     for c in m["cards"]])
        self._active_forces = frozenset()
        return duel

    # -- main loop ---------------------------------------------------------------
    def run(self) -> Duel:
        """Replays the whole stream; returns the duel positioned at the stream end."""
        last = None
        for _ in range(self.max_restarts + 1):
            self._reset()
            if self.hypothesis is None:
                try:
                    self.build_hypothesis()
                except TrackerError as te:
                    if self.identities and self._last_hyp_fix in ("identify", "identify_all"):
                        n = self._last_identify_count if self._last_hyp_fix == "identify_all" else 1
                        for _ in range(n):
                            self.failed_identities.add(self.identities.pop())
                        self._last_hyp_fix = None
                        if self.log:
                            self.log(f"restart {self.stats.restarts}: hypothesis failed ({te}) -> drop identity")
                        self.stats.restarts += 1
                        continue
                    if self.exclusions and self._last_hyp_fix == "exclude":
                        # the last exclusion cannot be satisfied: drop it
                        self.failed_exclusions.add(self.exclusions.pop())
                        self._last_hyp_fix = None
                        if self.log:
                            self.log(f"restart {self.stats.restarts}: hypothesis failed ({te}) -> drop exclusion")
                        self.stats.restarts += 1
                        continue
                    if not self.assumptions:
                        raise
                    # the last assumption is impossible (e.g. too many copies)
                    self.failed_assumptions.add(self.assumptions.pop())
                    if self.log:
                        self.log(f"restart {self.stats.restarts}: hypothesis failed ({te}) -> drop assumption")
                    self.stats.restarts += 1
                    continue
            duel = self.build_duel()
            try:
                duel.start()
                self._loop(duel, None, None)
                return duel
            except Divergence as e:
                duel.close()
                last = e
                bt = self._state_bt
                if bt is not None and e.pos == bt["pos"] and not bt["state"] and e.fix is not None \
                        and self._add_fix(e.fix):
                    # (searching the decisions for a message mismatch: a repair at the
                    # same point applies on top of the alternative being tried)
                    if self.log:
                        self.log(f"restart {self.stats.restarts}: {e.reason} -> {e.fix} (with decision "
                                 f"{bt['key']} candidate {self.overrides.get(bt['key'])})")
                    self.stats.restarts += 1
                    continue
                if bt is not None and (e.pos < bt["pos"] or e.pos == bt["pos"] and (
                        not bt["state"] or e.reason.startswith("state mismatch") or _stat_mismatch(e))):
                    # the decision changed for a mismatch made an earlier observation
                    # differ (or did not get past it): its next alternative, else an
                    # earlier decision (within a budget)
                    bt["tries"] += 1
                    if bt["tries"] > _DECISION_TRIES or not bt["queue"]:
                        for kk, _, _ in bt["list"]:
                            self.overrides.pop(kk, None)
                        self._state_bt = None
                        if bt.get("fallback") is not None and self._add_fix(bt["fallback"]):
                            if self.log:
                                self.log(f"restart {self.stats.restarts}: {bt['reason']} -> {bt['fallback']} "
                                         f"(no decision helped)")
                            self.stats.restarts += 1
                            continue
                        raise Divergence(bt["pos"], bt["reason"], None, None)
                    self.overrides.pop(bt["key"], None)
                    bt["key"], self.overrides[bt["key"]] = bt["queue"].pop(0)
                    if self.log:
                        self.log(f"restart {self.stats.restarts}: {e.reason} -> decision {bt['key']} "
                                 f"candidate {self.overrides[bt['key']]} (for the mismatch at {bt['pos']})")
                    self.stats.restarts += 1
                    continue
                if bt is not None and e.pos > bt["pos"]:
                    self._state_bt = None
                if self._last_swap is not None and e.pos < self._last_swap[1]:
                    # the latest copy swap made things worse (an earlier observation
                    # now differs): undo it, so another point is tried
                    (_, at, a, b), _ = self._last_swap
                    self._last_swap = None
                    lst = self.swaps.get(at, [])
                    if (a, b) in lst:
                        del lst[len(lst) - 1 - lst[::-1].index((a, b))]
                        for later in [c for c in self.swaps if c > at and c != (0, 0)]:
                            self.swaps[later] = [(b if x == a else a if x == b else x,
                                                  b if y == a else a if y == b else y)
                                                 for x, y in self.swaps[later]]
                    self.failed_swaps.add((at, a, b))
                    if self.log:
                        self.log(f"restart {self.stats.restarts}: {e.reason} -> undo swap {(at, a, b)}")
                    self.stats.restarts += 1
                    continue
                if e.fix is not None and e.fix[0] == "swap" and (tuple(e.fix[1:]) in self.failed_swaps
                                                                 or (e.fix[1], e.fix[3], e.fix[2]) in self.failed_swaps):
                    e.fix = None
                self._last_swap = None
                if e.fix is not None and e.fix[0] == "swap" and self._add_fix(e.fix):
                    self._last_swap = (e.fix, e.pos)
                    if self.log:
                        self.log(f"restart {self.stats.restarts}: {e.reason} -> {e.fix}")
                    self.stats.restarts += 1
                    continue
                if e.fix is None or not self._add_fix(e.fix):
                    # the decisions made since the last observed item are the ones
                    # whose consequence diverged: try the latest one differently
                    # (a state view shows the result of every decision since the
                    # previous one, e.g. an answer to "lower its Level?")
                    state = e.reason.startswith("state mismatch") or _stat_mismatch(e)
                    recent = self._since_state if state else self._recent
                    # the other candidates of the decisions, one decision at a time (the
                    # others as before): first each decision's next best candidate, the
                    # latest decision first, then the ones after that
                    todo = [(k, [j for j in range(n) if j != i]) for k, n, i in reversed(recent)
                            if k not in self.overrides]
                    queue = [(k, alts[d]) for d in range(max((len(a) for _, a in todo), default=0))
                             for k, alts in todo if d < len(alts)]
                    alt = queue.pop(0) if queue else None
                    fallback = getattr(e, "fallback", None)
                    if alt is None and fallback is not None and self._add_fix(fallback):
                        if self.log:
                            self.log(f"restart {self.stats.restarts}: {e.reason} -> {fallback} (no decision to try)")
                        self.stats.restarts += 1
                        continue
                    if alt is not None:
                        self.overrides[alt[0]] = alt[1]
                        if self._state_bt is None:
                            self._state_bt = {"pos": e.pos, "reason": e.reason, "list": list(recent),
                                              "key": alt[0], "queue": queue, "state": state, "tries": 0,
                                              "fallback": fallback}
                        if self.log:
                            self.log(f"restart {self.stats.restarts}: {e.reason} -> decision {alt[0]} candidate {alt[1]}")
                        self.stats.restarts += 1
                        continue
                    if self.memo:
                        # a remembered decision may be wrong under the repairs made
                        # since; try once more deciding everything afresh
                        if self.log:
                            self.log(f"restart {self.stats.restarts}: {e.reason} -> forget remembered decisions")
                        self.memo = {}
                        self.use_memo = False
                        self.stats.restarts += 1
                        continue
                    raise
                if self.log:
                    exp = e.expected
                    what = ""
                    if isinstance(exp, tuple) and len(exp) > 1 and isinstance(exp[1], dict):
                        what = f" [{exp[0]} {exp[1].get('type')}]"
                    elif isinstance(exp, dict) and "type" in exp:
                        what = f" [{exp['type']}]"
                    self.log(f"restart {self.stats.restarts}: {e.reason}{what} -> {e.fix}")
                self.stats.restarts += 1
            except BaseException:
                duel.close()
                raise
        raise Divergence(self.pos, f"too many restarts (last: {last.reason})", last.expected, last.got)

    def _add_fix(self, fix: tuple) -> bool:
        kind = fix[0]
        if kind == "force":
            key = fix[1:]
            if key in self.forces or key in self.unforced:
                return False
            self.forces.add(key)
            self.stats.forces += 1
            return True
        if kind == "unforce":
            # (only the windows around the problem: another effect of the card with
            # the same description, e.g. desc 0, can need the forcing elsewhere)
            key = fix[1:]
            windows = {w for w in self.force_windows if w[2] == key and w[0] <= self.pos <= w[1]}
            if key not in self.forces or not windows - self.unforced_windows:
                return False
            self.unforced_windows |= windows
            self.stats.forces += 1
            return True
        if kind == "assume":
            key = fix[1:]
            if key in self.assumptions:
                return False
            # a later assumption for the same card replaces an earlier one (which
            # then counts as tried)
            self.failed_assumptions.update(a for a in self.assumptions if a[:4] == key[:4])
            self.assumptions = [a for a in self.assumptions if a[:4] != key[:4]] + [key]
            self.hypothesis = None
            self.stats.assumptions += 1
            self._last_hyp_fix = "assume"
            return True
        if kind == "exclude":
            key = fix[1:]
            if key in self.exclusions or key in self.failed_exclusions:
                return False
            self.exclusions.append(key)
            self.hypothesis = None
            self.stats.assumptions += 1
            self._last_hyp_fix = "exclude"
            return True
        if kind == "swap":
            _, cp, a, b = fix
            if cp == (0, 0) and not self._role_swap_is_new(a, b):
                return False
            lst = self.swaps.setdefault(cp, [])
            if cp != (0, 0) and ((a, b) in lst or (b, a) in lst):
                return False
            lst.append((a, b))
            # swaps recorded later in the duel named the cards as they were
            # without this one: from here on each continues the other's path
            for later in [c for c in self.swaps if c > cp and c != (0, 0)]:
                self.swaps[later] = [(b if x == a else a if x == b else x, b if y == a else a if y == b else y)
                                     for x, y in self.swaps[later]]
            self.stats.swaps += 1
            return True
        if kind == "identify_all":
            self.identities.extend(fix[1:])
            self.hypothesis = None
            self._last_hyp_fix = "identify_all"
            self._last_identify_count = len(fix) - 1
            self.stats.assumptions += 1
            return True
        if kind == "identify":
            entry = fix[1:]
            if entry in self.identities:
                return False
            self.identities.append(entry)
            self.hypothesis = None  # (rebuilt with the new constraint)
            self._last_hyp_fix = "identify"
            self.stats.assumptions += 1
            return True
        if kind == "nomatchcard":
            entry = fix[1:]
            if entry in self.nomatch_cards:
                return False
            self.nomatch_cards.add(entry)
            self.stats.assumptions += 1
            return True
        if kind == "nomatchpile":
            span = fix[1:]
            if span in self.nomatch_piles:
                return False
            self.nomatch_piles.add(span)
            self.stats.assumptions += 1
            return True
        if kind == "matchpile":
            span = fix[1:]
            if span in self.match_piles:
                return False
            self.match_piles.add(span)
            self.stats.assumptions += 1
            return True
        if kind == "nomatchid":
            entry = fix[1:]
            if entry in self.nomatch_id_spans:
                return False
            self.nomatch_id_spans.add(entry)
            self.stats.assumptions += 1
            return True
        if kind == "respond":
            _, dkey, indices = fix
            self.tried_resp.add((dkey, indices))
            self.forced_resp[dkey] = {"indices": list(indices)}
            return True
        if kind == "decide":
            _, dkey, k = fix
            if self.overrides.get(dkey) == k:
                return False
            self.overrides[dkey] = k
            return True
        if kind == "match":
            _, cp, cid = fix
            lst = self.matches.setdefault(cp[0], [])
            if cid in lst:
                return False
            lst.append(cid)
            self.stats.assumptions += 1
            return True
        if kind == "snapshot":
            if fix[1] in self.snapshot_cps:
                return False
            self.snapshot_cps.add(fix[1])
            return True
        return False

    def _done(self) -> bool:
        return self.pos >= self.horizon

    def _advance(self) -> None:
        it = self.items[self.pos]
        if self.depth > 0 and (it[0] == "prompt" or it[1].get("type") in _BOUNDARIES):
            self.probe_boundary = True
            self.probe_after_boundary = 0
        elif self.depth > 0 and self.probe_boundary and it[0] == "msg" \
                and it[1].get("type") not in ("HINT", "CARD_HINT", "PLAYER_HINT"):
            self.probe_after_boundary += 1
        self.lockstep.process(it)
        self.pos += 1
        self.stall = 0
        self._recent = [d for d in self._recent if d[0][0] >= self.pos - _DECISION_WINDOW]

    def _loop(self, duel: Duel, prompt: dict | None, response: dict | None) -> None:
        pending = None  # the prompt whose response the next step processes
        pending_resp = None
        if response is not None:
            duel.set_response(encode(prompt, response))
            pending, pending_resp = prompt, response
        while True:
            if self.pos != self._cp_pos:
                self._cp_pos, self._cp_n = self.pos, 0
            else:
                self._cp_n += 1
            cp = (self._cp_pos, self._cp_n)
            todo = self.carry_swaps + [(a, b) for a, b in self.swaps.get(cp, ())
                                       if (cp, a, b) not in self._folded_swaps]
            if todo:
                self._roll_dirty = True
                # the pending response selected card objects: never move those
                busy = set(infer.response_card_ids(duel, pending, pending_resp)) if pending else set()
                carry = []
                for a, b in todo:
                    if a in busy or b in busy or not duel.swap_cards(a, b):
                        carry.append((a, b))  # not possible yet: try again next step
                self.carry_swaps = carry
            self._apply_forces(duel)
            if cp not in self.snapshots and (self.depth == 0 or cp in self.snapshot_cps):
                # (the main line keeps the latest step's positions, which most repairs
                # need, so that finding them takes no restart of its own)
                if self.depth == 0:
                    prev = self.snapshots.get(self._roll_cp)
                    if self._roll_cp is not None and self._roll_cp not in self.snapshot_cps:
                        self.snapshots.pop(self._roll_cp, None)
                    self._roll_cp = cp
                    if prev is not None and not self._roll_dirty:
                        self.snapshots[cp] = prev  # (no card moved since)
                    else:
                        self.snapshots[cp] = infer.LocMap(duel, lazy=False)
                    self._roll_dirty = False
                else:
                    self.snapshots[cp] = infer.LocMap(duel, lazy=False)
            # active for every step at this stream position: the selection may
            # happen in a step that shows the viewer nothing
            forced_match = self.matches.get(cp[0])
            if any(a <= cp[0] <= b for a, b in self.match_piles):
                forced_match = list(forced_match or ()) + self._hidden_ids(duel)
            if forced_match:
                duel.set_forced_match(forced_match)
            nomatch = self._hidden_ids(duel) if any(a <= cp[0] <= b for a, b in self.nomatch_piles) else []
            nomatch = nomatch + [cid for a, b, cid in self.nomatch_id_spans if a <= cp[0] <= b]
            if self.nomatch_cards:
                lm = None
                for a, b, slot in self.nomatch_cards:
                    if a <= cp[0] <= b:
                        lm = lm or infer.LocMap(duel)
                        cid = lm.at({"con": slot[0], "loc": slot[1], "seq": slot[2]})
                        if cid:
                            nomatch.append(cid)
            if nomatch:
                duel.set_forced_nomatch(nomatch)
            answered, answered_resp = pending, pending_resp  # (the response this step processes)
            status = duel.process()
            if nomatch:
                duel.set_forced_nomatch([])
            if forced_match:
                duel.set_forced_match([])
            msgs = [parse(m) for m in duel.get_messages()]
            if not self._roll_dirty and any(m["type"] not in _NO_MOVES for m in msgs):
                self._roll_dirty = True
            for m in msgs:
                self._simlog("msg", m, cp)
            # a swap must happen before the prompt that led to the move was built
            # (the prompt refers to the card objects)
            self._note_field_arrivals(cp, msgs, duel)
            if status == STATUS_AWAITING:
                self.prompt_cp = cp
                if msgs and msgs[-1]["type"] != "RETRY":
                    # a copy swap must happen before the prompt that chooses the card
                    for cid in infer.prompt_card_ids(duel, msgs[-1]):
                        self.last_listed[cid] = cp
            # reported after the step's messages: an earlier mismatch in them is
            # the cause (and may have a repair), the failed outcome a consequence
            forced_error = duel.forced_errors()

            def check_forced():
                if forced_error:
                    raise Divergence(self.pos, "a forced shuffle/random outcome could not be applied "
                                     "(the replay's cards differ from the hypothesis)", None, msgs)
            if status == STATUS_AWAITING and msgs and msgs[-1]["type"] != "RETRY":
                prompt = msgs.pop()
            for m in msgs:
                if m["type"] == "HINT" and m["hint_type"] == C.HINT_EVENT and m["player"] == self.opponent:
                    # the (private) timing of the opponent's next chain window
                    self._event_hint = (self.pos, m["data"])
            pending = prompt if status == STATUS_AWAITING else None
            pending_resp = None
            for idx, m in enumerate(msgs):
                if m["type"] in _SHUFFLE_LOCS:
                    self.last_shuffle_cp[(m["player"], _SHUFFLE_LOCS[m["type"]])] = cp
                    if m["type"] in ("SHUFFLE_DECK", "SHUFFLE_HAND"):
                        for _, cid in duel.location_cards(m["player"], _SHUFFLE_LOCS[m["type"]]):
                            self.card_shuffled.setdefault(cid, []).append(cp)
                    self.shuffle_cps.setdefault(cp, []).append(self._shuffle_count)
                    self._shuffle_count += 1
                if m["type"] == "RETRY":
                    if answered is not None and answered["type"] == "ANNOUNCE_CARD" \
                            and answered["player"] == self.opponent and answered_resp:
                        # the announcement must name a card in the deck (e.g. Crossout
                        # Designator): the deck holds a copy somewhere
                        n = len(duel.location_cards(self.opponent, C.LOCATION_DECK))
                        for seq in range(n):
                            self._assume({"con": self.opponent, "loc": C.LOCATION_DECK, "seq": seq},
                                         [answered_resp["code"]], "the opponent announced a card of its deck")
                    raise Divergence(self.pos, "core rejected a response")
                o = filter_message(m, self.viewer)
                if o is None:
                    continue
                if self.pos >= len(self.items):
                    if self.depth > 0:
                        check_forced()
                        return
                    raise Divergence(self.pos, "replay produced more messages than observed", None, o)
                item = self.items[self.pos]
                if item[0] != "msg" or item[1] != o:
                    if not self._match_unordered_hint(o):
                        raise self._mismatch(cp, msgs, idx, item, o, duel)
                self._advance()
                if o["type"] == "WIN":
                    check_forced()
                    if self.pos < len(self.items) and self.depth == 0:
                        raise Divergence(self.pos, "stream continues after the game ended")
                    return
                if self.depth > 0 and self._done():
                    check_forced()
                    return
            check_forced()
            if status == STATUS_END:
                if self.pos < len(self.items):
                    raise Divergence(self.pos, "replay ended early", self.items[self.pos])
                return
            if status == STATUS_CONTINUE:
                continue
            if prompt is None:
                raise Divergence(self.pos, "core awaits a response without a prompt")
            if prompt["player"] == self.viewer and is_hidden_prompt(prompt):
                # answered by the arbiter, not part of the viewer's stream
                self._simlog("prompt", prompt, AUTO_RESPONSE)
                duel.set_response(encode(prompt, AUTO_RESPONSE))
                pending, pending_resp = prompt, AUTO_RESPONSE
                continue
            if self.stopped and self.depth == 0 and self.pos >= len(self.items):
                return
            if prompt["player"] == self.viewer:
                if self.pos >= len(self.items):
                    if self.depth > 0:
                        return
                    raise Divergence(self.pos, "stream ended before the viewer's prompt")
                item = self.items[self.pos]
                fp = filter_prompt(prompt, self.viewer)
                if item[0] != "prompt" or item[1] != fp:
                    fix = self._force_fix(item[1]) if item[0] == "msg" else None
                    if fix is None and item[0] == "prompt":
                        fix = self._prompt_copy_fix(item[1], fp, duel, cp) \
                            or self._hidden_candidates_fix(item[1], fp, duel) \
                            or self._granted_copy_fix(item[1], fp, duel) \
                            or self._enabler_fix(item[1], fp, duel)
                    what = item[1].get("type") if item[0] == "prompt" else item[1].get("type", item[0])
                    raise Divergence(self.pos, f"prompt mismatch [{what} vs {fp['type']}]", item, fp, fix)
                if (self.check_state or self.pos == self.probe_state_at) and item[2] is not None:
                    got = view(capture(duel), self.viewer)
                    if got != item[2]:
                        raise Divergence(self.pos, "state mismatch: " + str(first_difference(item[2], got)),
                                         item[2], got, self._state_copy_fix(item[2], got, duel, cp))
                    if self.depth == 0:
                        self._since_state = []
                self._advance()
                if self.pos >= len(self.items):
                    self.final_prompt = prompt  # (the core's version: see to_core_response)
                    return  # positioned at the viewer's decision
                resp = self.items[self.pos]
                if resp[0] != "response":
                    raise Divergence(self.pos, "expected the viewer's response")
                self._advance()
                self._simlog("prompt", prompt, resp[1])
                duel.set_response(encode(prompt, to_core_response(prompt, self.viewer, resp[1])))
                pending_resp = resp[1]
                if self.depth > 0 and self._done():
                    return
            else:
                self._opp_prompt = prompt
                resp = self._decide(duel, prompt)
                self._simlog("prompt", prompt, resp)
                duel.set_response(encode(prompt, resp))
                pending_resp = resp

    def _apply_forces(self, duel: Duel) -> None:
        active = frozenset(k for start, end, k in self.force_windows
                           if start <= self.pos <= end and k in self.forces
                           and (start, end, k) not in self.unforced_windows)
        if active != self._active_forces:
            duel.clear_force_activatable()
            for key in active:
                duel.force_activatable(*key)
            self._active_forces = active

    def _match_unordered_hint(self, o: dict) -> bool:
        """Runs of hint messages come out of hash sets in the core, so their order
        depends on memory addresses and carries no information: accept any order."""
        if o["type"] not in UNORDERED_HINTS:
            return False
        j = self.pos
        while j < len(self.items) and self.items[j][0] == "msg" and self.items[j][1]["type"] in UNORDERED_HINTS:
            if self.items[j][1] == o:
                self.items[self.pos], self.items[j] = self.items[j], self.items[self.pos]
                return True
            j += 1
        return False

    # -- diagnosis -----------------------------------------------------------------
    def _force_fix(self, expected: dict) -> tuple | None:
        if expected["type"] == "CHAINING" and expected["trigger_con"] == self.opponent:
            key = (self.opponent, expected["code"], expected["desc"])
            if key not in self.forces and key not in self.unforced:
                return ("force", *key)
        return None

    def _include_fix(self, locs: list[dict], duel: Duel) -> tuple | None:
        """A recent card selection of the opponent must have included the card at
        each of `locs`: answer it with that card in place of one of its picks."""
        live = infer.LocMap(duel)
        for li in locs:
            cid = live.at(li)
            if not cid:
                continue
            for dkey, _, _ in reversed(self._recent):
                ids, resp = self._multi_decisions.get(dkey, (None, None))
                if not ids or cid not in ids:
                    continue
                j = ids.index(cid)
                picks = list(resp["indices"])
                if j in picks:
                    continue
                for p in reversed(range(len(picks))):
                    new = tuple(sorted(picks[:p] + [j] + picks[p + 1:]))
                    if (dkey, new) not in self.tried_resp:
                        return ("respond", dkey, new)
        return None

    def _forced_cost_fix(self) -> tuple | None:
        """The opponent's activation is forced (its cost or target check failed in
        the replay) and still being activated: the check depends on hidden cards,
        and so does its cost or target selection (e.g. "reveal a card whose other
        copy is in the deck"): let the hidden cards qualify during the activation."""
        start = self._chain_start()
        if start is None or (start, self.pos) in self.match_piles:
            return None
        ch = self.items[start][1]
        if (self.opponent, ch["code"], ch["desc"]) not in self._active_forces:
            return None
        if any(it[0] == "msg" and it[1]["type"] == "CHAINED" for it in self.items[start:self.pos]):
            return None
        return ("matchpile", start, self.pos)

    def _unforce_fix(self, got: dict) -> tuple | None:
        """The replay activated a forced effect that the real game did not."""
        if got["type"] == "CHAINING" and got["trigger_con"] == self.opponent:
            key = (self.opponent, got["code"], got["desc"])
            if key in self._active_forces:
                return ("unforce", *key)
        return None

    def _mismatch(self, cp: int, msgs: list[dict], idx: int, item: tuple, got: dict, duel: Duel) -> Divergence:
        fix = None
        if item[0] == "msg":
            exp = item[1]
            fix = self._force_fix(exp)
            if fix is None and exp["type"] in infer._ACTIVATION_PRELUDE:
                # the start of an activation (the card placed or flipped face-up):
                # the replay did not activate it (the core may not even have offered
                # a chain window when nothing was activatable)
                ch = self._next_activation(self._upcoming())
                if ch is not None:
                    fix = self._force_fix(ch)
            if fix is None and not (exp["type"] == "CHAINING" and exp.get("code") == got.get("code")
                                    and exp.get("desc") == got.get("desc")):
                fix = self._unforce_fix(got)
            if fix is None and got["type"] == "CARD_HINT" and exp["type"] == "CARD_HINT":
                fix = self._hint_swap_fix(cp, got, duel)
            hinted = exp if exp["type"] == "CARD_HINT" and got["type"] != "CARD_HINT" else \
                got if got["type"] == "CARD_HINT" and exp["type"] != "CARD_HINT" else None
            if fix is None and hinted is not None:
                # the hint belongs to another copy than the one it is on in the
                # replay (e.g. which copy, the one added to the hand this turn or an
                # older one, was discarded)
                live = infer.LocMap(duel)
                a = live.at(hinted)
                if a:
                    others = [cid for code, cid in duel.location_cards(hinted["con"], C.LOCATION_HAND)
                              if cid != a and code == live.codes.get(a)]
                    for b in others:
                        fix = self._history_swap(a, b, cp)
                        if fix is not None:
                            break
                if fix is None and hinted is got and got["hint_type"] == C.CHINT_DESC_REMOVE:
                    fix = self._unhinted_fix(got)
            if fix is None and exp["type"] == got["type"] and isinstance(exp.get("cards"), list) \
                    and isinstance(got.get("cards"), list) and len(exp["cards"]) == len(got["cards"]):
                # the same card in another place of a pile: another copy was chosen
                for e, g in zip(exp["cards"], got["cards"]):
                    if e == g:
                        continue
                    if e.get("code") == g.get("code") and e["con"] == g["con"] and e["loc"] == g["loc"] \
                            and e["seq"] != g["seq"] \
                            and e["loc"] in (*HIDDEN_PILES, C.LOCATION_GRAVE, C.LOCATION_REMOVED):
                        fix = self._swap_fix(cp, msgs, idx, e, g, duel)
                    break
            if fix is None and exp["type"] == got["type"] == "MOVE" \
                    and (not exp.get("code") or exp.get("code") != got.get("code")) \
                    and exp["from"]["con"] == got["from"]["con"] and exp["from"]["loc"] == got["from"]["loc"] \
                    and exp["from"]["seq"] != got["from"]["seq"] and exp["from"]["loc"] in HIDDEN_PILES \
                    and exp["to"] == got["to"]:
                # a hidden card moved from elsewhere in the pile: the replay's script
                # moved specific physical cards (e.g. ones it revealed and shuffled
                # back), and in the real game the card at that slot is one of them
                fix = self._revealed_fix(exp["from"])
            if fix is None and exp["type"] == got["type"] and exp.get("code") == got.get("code"):
                le, lg = _loc_of(exp), _loc_of(got)
                if (le is not None and lg is not None and le["con"] == lg["con"]
                        and le["loc"] == lg["loc"]
                        and le["loc"] in (*HIDDEN_PILES, C.LOCATION_GRAVE, C.LOCATION_REMOVED)):
                    if le["seq"] != lg["seq"]:
                        fix = self._swap_fix(cp, msgs, idx, le, lg, duel)
                        if (fix is None or self._has_swap(fix)) and lg["seq"] < le["seq"]:
                            # the replay's scan hit this code lower down than the real one
                            fix = self._exclude_fix(le, msgs[idx].get("code"))
                    elif exp.get("code"):
                        # right card, wrong copy (the copies differ in hidden history)
                        fix = self._other_copy_fix(cp, msgs, idx, le)
            if fix is None and exp["type"] == got["type"] == "MOVE" and got["from"]["con"] == self.opponent \
                    and exp["from"]["con"] == got["from"]["con"] and exp["from"]["loc"] == got["from"]["loc"] \
                    and exp["from"]["seq"] != got["from"]["seq"] and got["from"]["loc"] in HIDDEN_PILES \
                    and exp.get("code") and got.get("code") and exp["code"] != got["code"]:
                # the replay's operation took one more card, a hidden one the viewer
                # knows nothing about (e.g. "send all monsters from the hand"): in the
                # real game it does not fit (a filler card of the wrong kind)
                fix = self._nomatch_card_fix(got["from"], duel, got["to"])
            if fix is None and got["type"] == "MOVE" and got["from"]["con"] == self.opponent \
                    and got["from"]["loc"] in HIDDEN_PILES and not (
                        exp["type"] == "MOVE" and exp["from"]["con"] == got["from"]["con"]
                        and exp["from"]["loc"] == got["from"]["loc"]):
                # the replay took a hidden card of the opponent where the real game
                # took none from there (e.g. "each player adds a Normal Monster from
                # their Deck" with none in the real Deck): none of them fits
                start = self._chain_start()
                span = (min(self._link_start(), start if start is not None else self.pos), self.pos)
                if span not in self.nomatch_piles and span not in self.match_piles:
                    fix = ("nomatchpile", *span)
                else:
                    fix = self._nomatch_card_fix(got["from"], duel, got["to"])
            if fix is None and exp["type"] == got["type"] == "MOVE" and exp["from"]["con"] == got["from"]["con"] \
                    and exp["from"]["loc"] == got["from"]["loc"] and exp["from"]["seq"] != got["from"]["seq"] \
                    and exp["from"]["loc"] == C.LOCATION_HAND and exp["to"] == got["to"]:
                # the replay's script moved a specific physical card (e.g. the card it
                # revealed earlier) from where the hypothesis shuffled it: in the real
                # game that card is at the slot it was moved from
                fix = self._identify_fix(exp["from"], got["from"])
            if fix is None and exp["type"] == "MOVE" and exp.get("code") and exp["from"]["loc"] in HIDDEN_PILES \
                    and self._opp_prompt is not None:
                # the opponent chose among cards offered to it, one of them a copy
                # of the card the real game took from elsewhere in the pile: that
                # copy is the real one (e.g. revealed cards shuffled back, then picked)
                cards = self._opp_prompt.get("cards") or self._opp_prompt.get("select_cards") or []
                le = exp["from"]
                for c in cards:
                    if c.get("code") == exp["code"] and c["con"] == le["con"] and c["loc"] == le["loc"] \
                            and c["seq"] != le["seq"]:
                        fix = self._swap_fix(cp, msgs, idx, le, c, duel)
                        if fix is not None:
                            break
            if fix is None and exp["type"] == "MOVE" and got["type"] == "MOVE" and got.get("code") \
                    and exp.get("code") != got["code"] and exp["from"]["loc"] == got["from"]["loc"] \
                    and got["from"]["loc"] in HIDDEN_PILES:
                # the replay moved a copy of a card that the real game moves later
                # from elsewhere in the same operation: the copies are swapped
                fix = self._later_copy_fix(cp, msgs, idx, got, duel)
            if fix is None and exp["type"] == got["type"] == "RANDOM_SELECTED":
                # the real random pick took a card the replay's group did not have:
                # an earlier selection (e.g. "pick 4 cards of the other's deck")
                # included it
                fix = self._include_fix([c for c in exp["cards"] if c not in got["cards"]], duel)
            if fix is None and exp["type"] == "HINT" and exp["hint_type"] == C.HINT_OPSELECTED \
                    and exp["player"] == self.opponent and exp != got:
                # the opponent chose an option the replay did not offer or take:
                # which options exist can depend on hidden cards (e.g. "set a trap
                # from the deck"); let them qualify from the activation on
                # (first from just after the activation's message: the chain window
                # before it sees the real cards, e.g. whether a hand trap may respond)
                start = self._chain_start()
                if start is not None:
                    fix = next((("matchpile", a, self.pos) for a in (min(start + 1, self.pos), start)
                                if (a, self.pos) not in self.match_piles), None)
            revealed = None
            if got["type"] == "POS_CHANGE" and got["prev_pos"] & C.POS_FACEDOWN and got["pos"] & C.POS_FACEUP:
                revealed = got
            elif got["type"] == "MOVE" and got["from"]["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE) \
                    and got["from"].get("pos", 0) & C.POS_FACEDOWN:
                revealed = got["from"]
            if fix is None and revealed is not None and exp["type"] != got["type"] \
                    and revealed["con"] == self.opponent and got.get("code"):
                # the replay showed the opponent's face-down card (flipped it, or sent
                # it away) where the real game did not (e.g. an effect that destroys
                # only traps): the card is not the replay's
                key = (self.pos, revealed["con"], revealed["loc"], (revealed["seq"],), got["code"])
                if key not in self.exclusions and key not in self.failed_exclusions:
                    fix = ("exclude", *key)
            if fix is None and exp["type"] == "CONFIRM_CARDS" and exp["cards"] \
                    and exp["cards"][0]["con"] == self.opponent \
                    and (got["type"] != "CONFIRM_CARDS" or {(c["con"], c["loc"], c["seq"]) for c in exp["cards"]}
                         - {(c["con"], c["loc"], c["seq"]) for c in got["cards"]}):
                # the opponent revealed a card that the replay's filter rejects (the
                # filter depends on hidden cards, e.g. a matching card in the deck)
                shown = [(c["con"], c["loc"], c["seq"]) for c in got["cards"]] if got["type"] == "CONFIRM_CARDS" else []
                # (a forced activation's reveal: its cost depends on hidden cards, e.g.
                # "reveal a card whose other copy is in the deck", not the revealed card)
                fix = self._forced_cost_fix()
                for c in exp["cards"] if fix is None else ():
                    if (c["con"], c["loc"], c["seq"]) not in shown:
                        fix = self._match_fix(cp, msgs, idx, c)
                        if fix is not None:
                            break
                if fix is None:
                    start = self._chain_start()
                    if start is not None and (start, self.pos) not in self.match_piles:
                        fix = ("matchpile", start, self.pos)
            if fix is None and exp["type"] == got["type"] == "BECOME_TARGET" \
                    and len(exp["cards"]) == len(got["cards"]) == 1:
                # the same card in another place of a pile: another copy was chosen
                e, g = exp["cards"][0], got["cards"][0]
                if e["con"] == g["con"] and e["loc"] == g["loc"] and e["seq"] != g["seq"] \
                        and e["loc"] in (C.LOCATION_GRAVE, C.LOCATION_REMOVED, *HIDDEN_PILES):
                    fix = self._swap_fix(cp, msgs, idx, e, g, duel)
            if fix is None and exp["type"] in _NOTHING_MORE and got["type"] not in _NOTHING_MORE \
                    and got["type"] != exp["type"]:
                # the replay did something more where the real game went on (e.g. an
                # effect found a fitting card among the hidden filler cards that
                # the real deck does not have): let the hidden cards match nothing
                start = self._chain_start()
                span = (min(self._link_start(), start if start is not None else self.pos), self.pos)
                if span not in self.nomatch_piles and span not in self.match_piles:
                    fix = ("nomatchpile", *span)
            if fix is None and exp["type"] == got["type"] == "HINT" and exp["hint_type"] == got["hint_type"] \
                    == C.HINT_SELECTMSG and exp["player"] == got["player"] == self.viewer and exp["data"] != got["data"]:
                # the viewer's own selection hint names the card it handles (e.g. "select
                # the zone for <card>"): the replay's card of that name is that card
                fix = self._hint_code_fix(exp["data"], got["data"], duel)
                if fix is None:
                    # (or the order the operation handles its cards in: two copies of
                    # the replay's card are the other way round, e.g. which one was
                    # revealed before the hand was shuffled)
                    for loc in (C.LOCATION_HAND, C.LOCATION_DECK, C.LOCATION_GRAVE):
                        ids = [cid for code, cid in duel.location_cards(self.viewer, loc) if code == got["data"]]
                        for i, a in enumerate(ids):
                            for b in ids[i + 1:]:
                                fix = fix or self._history_swap(a, b, cp)
                        if fix is not None:
                            break
            if fix is None and got["type"] == "HINT" and got["hint_type"] == C.HINT_OPSELECTED \
                    and got["player"] == self.opponent and exp != got \
                    and not (exp["type"] == "HINT" and exp["hint_type"] == C.HINT_OPSELECTED):
                # the replay offered the opponent a choice of options that the real
                # game did not (it had only one option): hidden (filler) cards
                # satisfy a condition of the others, so let them match nothing
                start = self._chain_start()
                span = (min(self._link_start(), start if start is not None else self.pos), self.pos)
                if span not in self.nomatch_piles and span not in self.match_piles:
                    fix = ("nomatchpile", *span)
            if fix is None and exp["type"] == "BECOME_TARGET" and exp["cards"] \
                    and (got["type"] != "BECOME_TARGET" or got["cards"] != exp["cards"]):
                # the opponent targeted a card that the replay's filter rejects
                # (the filter depends on hidden cards, e.g. a fitting card in the
                # deck): let it match
                fix = self._match_fix(cp, msgs, idx, exp["cards"][0], duel)
                start = self.pos
                while start > 0 and self.items[start - 1][0] == "msg" \
                        and self.items[start - 1][1]["type"] == "BECOME_TARGET":
                    start -= 1
                if fix is not None and fix[0] == "match" and start < self.pos:
                    # (the targets of one selection are announced one by one: the
                    # selection happened before the first of them)
                    fix = None if fix[2] in self.matches.get(start, ()) else ("match", (start, 0), fix[2])
            if fix is None and exp["type"] == "MOVE" and exp["from"]["con"] == self.opponent \
                    and exp["from"]["loc"] in HIDDEN_PILES and exp != got:
                fix = self._kind_fix(cp, msgs, idx, exp) or self._match_fix(cp, msgs, idx, exp["from"])
                if fix is None and not exp.get("code"):
                    # several hidden cards must qualify together (e.g. differently
                    # named cards set as a group): let all of them qualify (a card
                    # whose code shows can be assumed instead)
                    start = self._chain_start()
                    if start is not None and (start, self.pos) not in self.match_piles:
                        fix = ("matchpile", start, self.pos)
            elif fix is None and exp["type"] == "MOVE" and exp["from"]["con"] == self.opponent \
                    and exp["from"]["loc"] and not exp["from"]["loc"] & C.LOCATION_OVERLAY and exp != got:
                # a public card of the opponent the replay did not take: whether it
                # qualified can depend on hidden cards (e.g. a cost that needs a
                # matching card in the deck), so let it match the filters
                fix = self._match_fix(cp, msgs, idx, exp["from"])
            if fix is None:
                fix = self._forced_cost_fix()
            if fix is None and got["type"] in _NOTHING_MORE | {"CHAINED"} \
                    and exp["type"] not in _NOTHING_MORE | {"CHAINED"}:
                # (last resort) the real game did more before going on (e.g. a cost
                # whose size depends on fitting cards in the deck): let the hidden
                # cards qualify
                start = self._chain_start()
                if start is not None and (start, self.pos) not in self.match_piles \
                        and (start, self.pos) not in self.nomatch_piles:
                    fix = ("matchpile", start, self.pos)
        if fix is None and item[0] == "prompt" and item[1]["type"] in ("SELECT_BATTLECMD", "SELECT_IDLECMD") \
                and got.get("type") == "NEW_PHASE":
            # the replay skipped a phase the real game did not: an effect that ends a
            # phase lingers where it should not (e.g. activated in a later window than
            # the real one, after the step whose end it resets at)
            for dkey, act in reversed(self._window_choices):
                if self.overrides.get(dkey) != act:
                    fix = ("decide", dkey, act)
                    break
        if fix is None and item[0] == "prompt":
            # the viewer's prompt comes next in the real game, but the replay
            # activated a forced effect first (e.g. a mandatory trigger whose
            # target check the forcing skipped)
            fix = self._unforce_fix(got)
        fallback = None
        if fix is None:
            fallback, at = self._hidden_pick_fallback()
            if fallback is not None and not any(k[0] > at and n > 1 for k, n, _ in self._recent):
                # (no later decision could explain the consequences instead)
                fix, fallback = fallback, None
        what = item[1].get("type") if item[0] == "msg" else item[0]
        d = Divergence(self.pos, f"message mismatch [{what} vs {got.get('type')}]", item, got, fix)
        d.fallback = fallback
        return d

    def _hidden_pick_fallback(self) -> tuple[tuple | None, int]:
        """(Last resort, once the decision search found nothing.) The opponent
        recently chose among its hidden cards (e.g. "Normal Summon a Normal
        Monster from your hand"): in the real game none of the replay's (filler)
        cards there may have fit, the consequences showing only later. Returns
        the repair and where the choice was."""
        for at, span in reversed(self._hidden_picks):
            if at < self.pos - 40:
                break
            if span not in self.nomatch_piles and span not in self.match_piles:
                return ("nomatchpile", *span), at
        return None, -1

    def _kind_fix(self, cp: int, msgs: list[dict], idx: int, exp: dict) -> tuple | None:
        """A hidden card set into the spell/trap zone must be a spell or trap
        (a field spell for the field zone): assume one if the replay's card is not."""
        to = exp["to"]
        if to["loc"] != C.LOCATION_SZONE or not to["pos"] & C.POS_FACEDOWN or to["seq"] > 5:
            return None
        if cp not in self.snapshots:
            return None if cp in self.snapshot_cps else ("snapshot", cp)
        lm = copy.deepcopy(self.snapshots[cp])
        lm.references(msgs[:idx])
        cid = lm.at(exp["from"])
        field_zone = to["seq"] == 5

        def fits(code):
            card = self.db.get(code)
            return card is not None and bool(card.type & (C.TYPE_SPELL | C.TYPE_TRAP)) \
                and bool(card.type & C.TYPE_FIELD) == field_zone
        if cid and fits(lm.codes.get(cid, 0)):
            return None
        codes = [c for c in self._filler().settable_spells if fits(c)]
        try:
            self._assume(exp["from"], codes, "a hidden card was set from a pile")
        except Divergence as d:
            return d.fix
        return None

    def _match_fix(self, cp: int, msgs: list[dict], idx: int, loc: dict, duel: Duel | None = None) -> tuple | None:
        """The opponent took a hidden card for an effect, but in the replay no card
        at that position satisfies the effect's filter (the hypothesis could not
        know what the effect needed): let that card match every filter during
        the step."""
        if duel is not None and loc["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE) \
                and not any(m["type"] == "MOVE" for m in msgs[:idx]):
            # (a card on the field that nothing moved in this step: no snapshot needed)
            cid = infer.LocMap(duel).at(loc)
            if not cid or cid in self.matches.get(cp[0], ()):
                return None
            return ("match", cp, cid)
        if cp not in self.snapshots:
            return None if cp in self.snapshot_cps else ("snapshot", cp)
        lm = copy.deepcopy(self.snapshots[cp])
        lm.references(msgs[:idx])
        cid = lm.at(loc)
        if not cid or cid in self.matches.get(cp[0], ()):
            return None
        return ("match", cp, cid)

    def _note_field_arrivals(self, cp, msgs: list[dict], duel: Duel) -> None:
        """Remembers when cards went from a pile to the field (for copy swaps)."""
        for m in msgs:
            if m["type"] != "MOVE":
                continue
            f, t = m["from"], m["to"]
            if f["loc"] in (*HIDDEN_PILES, C.LOCATION_GRAVE, C.LOCATION_REMOVED) \
                    and t["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE):
                cards = duel.location_cards(t["con"], t["loc"])
                if t["seq"] < len(cards) and cards[t["seq"]][1]:
                    cid = cards[t["seq"]][1]
                    # swap before the pile's last shuffle if there was one (the
                    # forced shuffle is stable, so that decides where copies go),
                    # else before the prompt that chose the card
                    at = self.last_shuffle_cp.get((f["con"], f["loc"]), self.last_listed.get(cid, cp))
                    self.left_pile[cid] = (at, (f["con"], f["loc"]))

    def _hint_code_fix(self, code: int, replay_code: int, duel: Duel) -> tuple | None:
        """The opponent's hidden card the replay handles (the only one with the
        code the replay's hint named) has the code of the real hint."""
        if self.db.get(code) is None or self.db.get(replay_code) is None:
            return None
        from ..state import parse_location_query
        found = []
        for loc in (*HIDDEN_PILES, C.LOCATION_MZONE, C.LOCATION_SZONE):
            cards = duel.location_cards(self.opponent, loc)
            pos = parse_location_query(duel.query_location_raw(self.opponent, loc, C.QUERY_POSITION)) \
                if loc in (C.LOCATION_MZONE, C.LOCATION_SZONE) else None
            for seq, (c, cid) in enumerate(cards):
                hidden = pos is None or (pos[seq] is not None and not pos[seq].get("position", 0) & C.POS_FACEUP)
                if cid and c == replay_code and hidden:
                    found.append({"con": self.opponent, "loc": loc, "seq": seq})
        field = [f for f in found if f["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE)]
        if len(field) > 1:
            # (several: the one the chain link being resolved targeted)
            start = next((j for j in range(self.pos - 1, max(-1, self.pos - 30), -1)
                          if self.items[j][0] == "msg" and self.items[j][1]["type"] == "CHAINING"), self.pos)
            targets = [c for it in self.items[start:self.pos]
                       if it[0] == "msg" and it[1]["type"] == "BECOME_TARGET" for c in it[1]["cards"]]
            field = [f for f in field if any(infer.same_loc(f, t) for t in targets)]
        if len(field) == 1:
            found = field  # (a face-down card on the field, e.g. the target, before filler copies)
        if len(found) != 1:
            return None
        li = found[0]
        key = (li["con"], li["loc"], li["seq"])
        try:
            var = self.lockstep.var_at(*key)
        except TrackerError:
            return None
        tried = {a[4] for a in self.assumptions if a[1:4] == key} | {
            a[4] for a in self.failed_assumptions if a[1:4] == key}
        if var is None or code in tried:
            return None
        return ("assume", self.pos, *key, code)

    def _unhinted_fix(self, got: dict) -> tuple | None:
        """The replay's card lost a hint as it moved from a hidden pile to a
        public place, the real one did not: the real card is another copy than
        the one the hypothesis has there (e.g. not the copy added to the hand this
        turn, whose hint the viewer cannot see)."""
        prev = self.items[self.pos - 1] if self.pos else None
        if prev is None or prev[0] != "msg" or prev[1]["type"] != "MOVE":
            return None
        m = prev[1]
        if (m["to"]["con"], m["to"]["loc"], m["to"]["seq"]) != (got["con"], got["loc"], got["seq"]) \
                or m["from"]["loc"] not in HIDDEN_PILES:
            return None
        solved = getattr(self, "_solved", None)
        v = self.lockstep.var_at(got["con"], got["loc"], got["seq"])
        sv = next((x for x in solved.all_vars if x.id == v.id), None) if solved and v else None
        member = sv.assigned if sv is not None and sv.kind == "claim" else None
        if member is None:
            return None
        f = m["from"]
        entry = (self.pos - 1, (f["con"], f["loc"], f["seq"]), ("not", member.id))
        if entry in self.identities or entry in self.failed_identities:
            return None
        return ("identify", *entry)

    def _nomatch_card_fix(self, slot: dict, duel: Duel | None = None, dest: dict | None = None) -> tuple | None:
        try:
            v = self.lockstep.var_at(slot["con"], slot["loc"], slot["seq"])
        except TrackerError:
            return None
        if v is None or v.code is not None or v.fixed_code is not None:
            return None
        start = self._chain_start()
        if start is None:
            # (the link being resolved may be the viewer's: e.g. "banish all monsters
            # in both hands" collected the cards when its operation started)
            start = next((j for j in range(self.pos - 1, max(-1, self.pos - 30), -1)
                          if self.items[j][0] == "msg" and self.items[j][1]["type"] in ("CHAINING", "CHAIN_SOLVING")),
                         self.pos)
        # (the card the replay moved is at the move's destination now)
        cid = infer.LocMap(duel).at(dest) if duel is not None and dest is not None else None
        if cid:
            # (by card id: the slots shift while the operation moves the cards one by one)
            entry = (start, self.pos, cid)
            return None if entry in self.nomatch_id_spans else ("nomatchid", *entry)
        entry = (start, self.pos, (slot["con"], slot["loc"], slot["seq"]))
        return None if entry in self.nomatch_cards else ("nomatchcard", *entry)

    def _revealed_fix(self, le: dict) -> tuple | None:
        """Every card this operation moves from the pile (the MOVEs from there to
        the same place that follow) is one of the cards revealed before it was
        shuffled."""
        try:
            v = self.lockstep.var_at(le["con"], le["loc"], le["seq"])
        except TrackerError:
            return None
        if v is None or v.kind != "claim" or not self.lockstep._referenced \
                or (le["con"], le["loc"]) not in self.lockstep._referenced_piles:
            return None
        first = self.items[self.pos][1]
        entries = []
        for k, item in enumerate(self.items[self.pos:self.pos + len(self.lockstep._referenced)]):
            m = item[1]
            if item[0] != "msg" or m["type"] != "MOVE" or m["to"] != first["to"] \
                    or (m["from"]["con"], m["from"]["loc"]) != (le["con"], le["loc"]):
                break
            entries.append((self.pos + k, (le["con"], le["loc"], m["from"]["seq"]), REVEALED))
        entries = [e for e in entries if e not in self.identities]
        if not entries or entries[0] in self.failed_identities:
            return None
        return ("identify_all", *entries)

    def _identify_fix(self, le: dict, lg: dict) -> tuple | None:
        solved = getattr(self, "_solved", None)
        if solved is None:
            return None
        try:
            vg = self.lockstep.var_at(lg["con"], lg["loc"], lg["seq"])
            ve = self.lockstep.var_at(le["con"], le["loc"], le["seq"])
        except TrackerError:
            return None
        if vg is None or ve is None or ve.kind != "claim" or ve.same_as is not None:
            return None
        sv = next((v for v in solved.all_vars if v.id == vg.id), None)
        member = sv.assigned if sv is not None and sv.kind == "claim" else None
        if member is None:
            return None
        entry = (self.pos, (le["con"], le["loc"], le["seq"]), member.id)
        if entry in self.identities or entry in self.failed_identities:
            return None
        return ("identify", *entry)

    def _later_copy_fix(self, cp, msgs: list[dict], idx: int, got: dict, duel: Duel) -> tuple | None:
        earlier = []
        for item in self.items[self.pos:self.pos + 12]:
            if item[0] != "msg" or item[1]["type"] not in ("MOVE", "CARD_HINT", "HINT"):
                break
            m = item[1]
            if m["type"] != "MOVE":
                continue
            if m["code"] == got["code"] and m["from"]["con"] == got["from"]["con"] \
                    and m["from"]["loc"] == got["from"]["loc"]:
                # where that card was before the earlier moves of the operation
                src = dict(m["from"])
                for e in reversed(earlier):
                    f = e["from"]
                    if f["con"] == src["con"] and f["loc"] == src["loc"] and f["seq"] <= src["seq"]:
                        src["seq"] += 1
                return self._swap_fix(cp, msgs, idx, src, got["from"], duel)
            earlier.append(m)
        return None

    def _history_swap(self, a: int, b: int, cp: tuple) -> tuple | None:
        """Exchanges two copies whose positions or histories are visible: where
        they were last shuffled together (a hand shuffle can still show the
        difference through card hints, so earlier ones are tried next), else
        from the start of the duel."""
        shared = sorted(set(self.card_shuffled.get(a, ())) & set(self.card_shuffled.get(b, ())), reverse=True)
        for at in shared:
            if at <= cp and (a, b) not in self.swaps.get(at, ()) and (b, a) not in self.swaps.get(at, ()) \
                    and (at, a, b) not in self.failed_swaps and (at, b, a) not in self.failed_swaps:
                return ("swap", at, a, b)
        if self._role_swap_is_new(a, b):
            return ("swap", (0, 0), a, b)
        return None

    def _field_copy_fix(self, a: int, b: int) -> tuple | None:
        """Two copies on the field are the wrong way round: swap them just before
        the first of them left the pile they shared."""
        if a not in self.left_pile or b not in self.left_pile:
            return None
        (cpa, pile_a), (cpb, pile_b) = self.left_pile[a], self.left_pile[b]
        if pile_a != pile_b:
            if pile_a[0] != pile_b[0] or {pile_a[1], pile_b[1]} & {C.LOCATION_EXTRA}:
                return None
            # (e.g. one was drawn first) they were together in the deck before
            return self._history_swap(a, b, max(cpa, cpb))
        return ("swap", min(cpa, cpb), a, b)

    @staticmethod
    def _roles(pairs) -> dict[int, int]:
        """Slot id -> id placed there after the start-of-duel swaps."""
        roles: dict[int, int] = {}
        for a, b in pairs:
            # (each swap exchanges the ids as the previous ones placed them)
            roles.setdefault(a, a)
            roles.setdefault(b, b)
            for k, v in roles.items():
                roles[k] = b if v == a else a if v == b else v
        return roles

    def _role_swap_is_new(self, a: int, b: int) -> bool:
        pairs = list(self.swaps.get((0, 0), ()))
        tried = {frozenset((k, v) for k, v in self._roles(pairs[:i]).items() if k != v)
                 for i in range(len(pairs) + 1)}
        new = frozenset((k, v) for k, v in self._roles(pairs + [(a, b)]).items() if k != v)
        return new not in tried

    def _enabler_fix(self, exp: dict, got: dict, duel: Duel) -> tuple | None:
        """The viewer is offered an effect the replay does not offer: the opponent's
        hidden cards can enable it, most directly a copy of the same card (e.g. one
        whose watcher, active in every location, raises the event both copies
        respond to). Assume one copy at the bottom of the opponent's deck, once."""
        key = "chains" if "chains" in exp else "activatable"
        if key not in exp or (exp["type"] != got["type"] and exp["type"] != "SELECT_CHAIN"):
            return None  # (no chain window at all in the replay when nothing could respond)
        have = Counter((o["code"], o.get("desc", 0)) for o in got.get(key, ())) \
            if exp["type"] == got["type"] else Counter()
        n = len(duel.location_cards(self.opponent, C.LOCATION_DECK))
        for o in exp[key]:
            if have[(o["code"], o.get("desc", 0))] > 0 or o["con"] != self.viewer or not n:
                continue
            if any(a[4] == o["code"] and a[1:3] == (self.opponent, C.LOCATION_DECK)
                   for a in [*self.assumptions, *self.failed_assumptions]):
                continue
            try:
                self._assume({"con": self.opponent, "loc": C.LOCATION_DECK, "seq": 0}, [o["code"]],
                             "the viewer's option needs a card of the opponent")
            except Divergence as d:
                return d.fix
        return None

    def _granted_copy_fix(self, exp: dict, got: dict, duel: Duel) -> tuple | None:
        """The viewer's prompt offers an effect granted by Xyz materials a different
        number of times: copies of a material differ in history (which one's effect
        was used this turn; the choice among identical entries follows the card
        ids), so an attached copy and one that left trade roles from the start."""
        if exp["type"] != got["type"] or "activatable" not in exp:
            return None
        ce = Counter((o["code"], o.get("desc", 0)) for o in exp["activatable"])
        cg = Counter((o["code"], o.get("desc", 0)) for o in got.get("activatable", ()))
        for (code, desc), n in sorted(ce.items()):
            if cg[(code, desc)] == n or not desc >> 20 or desc >> 20 == code:
                continue
            src = desc >> 20
            o = next(o for o in exp["activatable"] if (o["code"], o.get("desc", 0)) == (code, desc))
            if not o["loc"] & C.LOCATION_MZONE:
                continue
            attached = [cid for c, cid in duel.overlay_cards(o["con"], C.LOCATION_MZONE, o["seq"]) if c == src]
            others = [cid for p in (0, 1) for loc in (C.LOCATION_GRAVE, C.LOCATION_REMOVED, C.LOCATION_HAND,
                                                      C.LOCATION_DECK, C.LOCATION_EXTRA)
                      for c, cid in duel.location_cards(p, loc) if c == src]
            for a in attached:
                for b in others:
                    if self._role_swap_is_new(a, b) and ((0, 0), a, b) not in self.failed_swaps \
                            and ((0, 0), b, a) not in self.failed_swaps:
                        return ("swap", (0, 0), a, b)
        return None

    def _prompt_copy_fix(self, exp: dict, got: dict, duel: Duel, cp: int) -> tuple | None:
        """The viewer's prompt lists a different copy of a card (copies can differ in
        hidden history, e.g. which one may be activated this turn): swap the copies
        before the pile was last shuffled, so they end up the other way round."""
        live = infer.LocMap(duel)
        # (a prompt about one card, e.g. SELECT_EFFECTYN, names it directly)
        pairs = [(exp, got)] if all(k in exp and k in got for k in ("con", "loc", "seq")) else []
        for key in ("summonable", "spsummonable", "repositionable", "msetable", "ssetable",
                    "activatable", "attackable", "chains", "cards", "select_cards", "unselect_cards"):
            pairs += zip(exp.get(key, ()), got.get(key, ()))
        for e, g in pairs:
            if e == g:
                continue
            if e.get("code") != g.get("code") or e["con"] != g["con"] or e["loc"] != g["loc"] \
                    or e["seq"] == g["seq"]:
                return None
            a, b = live.at(g), live.at(e)
            if not a or not b or live.codes.get(a) != live.codes.get(b):
                return None
            if g["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE):
                return self._field_copy_fix(a, b)
            if g["loc"] in (C.LOCATION_GRAVE, C.LOCATION_REMOVED) and self._role_swap_is_new(a, b):
                # the positions and histories of public cards show: exchange the
                # copies' roles from the start instead of the cards in place
                return ("swap", (0, 0), a, b)
            at = self.last_shuffle_cp.get((g["con"], g["loc"]), cp)
            if (a, b) in self.swaps.get(at, ()) or (b, a) in self.swaps.get(at, ()):
                at = cp
            return ("swap", at, a, b)
        return None

    def _state_copy_fix(self, exp: dict, got: dict, duel: Duel, cp: tuple[int, int]) -> tuple | None:
        """A card in the state view differs from the real one, but another copy
        of it in the replay looks exactly like it (the copies differ in history,
        e.g. only one had its level changed): swap the two copies."""
        live = infer.LocMap(duel)
        for le, lg in zip(exp.get("chain") or (), got.get("chain") or ()):
            he, hg = le.get("handler"), lg.get("handler")
            if le.get("code") != lg.get("code") or not he or not hg or infer.same_loc(he, hg):
                continue
            # the chain link's card is another copy than the replay's (e.g. the
            # copy that stayed in the deck when the shuffled cards were excavated):
            # swap it with a copy where the real one is
            a = live.at(hg)
            cands = [live.at(he)] + [cid for code, cid in duel.location_cards(he["con"], he["loc"] & 0xff)
                                     if code == le["code"]]
            for b in dict.fromkeys(c for c in cands if c and c != a):
                if live.codes.get(b, le["code"]) != le["code"]:
                    continue
                fix = self._history_swap(a, b, cp) if a else None
                if fix is not None:
                    return fix
        for p in (0, 1):
            for name, loc in LOCATIONS:
                e_cards, g_cards = exp["players"][p][name], got["players"][p][name]
                if len(e_cards) != len(g_cards):
                    return None
                for i, (e, g) in enumerate(zip(e_cards, g_cards)):
                    if e == g:
                        continue
                    if not e or not g or not e.get("code") or e.get("code") != g.get("code"):
                        return None
                    if loc in (C.LOCATION_MZONE, C.LOCATION_SZONE) \
                            and {k for k in set(e) | set(g) if e.get(k) != g.get(k)} == {"status"}:
                        # only the card's history differs (e.g. a position change while
                        # it was in the deck): another copy from a hidden pile
                        a = live.at({"con": p, "loc": loc, "seq": i})
                        for hloc in (C.LOCATION_DECK, C.LOCATION_HAND, C.LOCATION_EXTRA):
                            for code, b in duel.location_cards(p, hloc):
                                if code == e["code"] and a and b != a:
                                    fix = self._history_swap(a, b, cp)
                                    if fix is not None:
                                        return fix
                    for j, g2 in enumerate(g_cards):
                        if j == i or g2 != e or e_cards[j] == g2:
                            continue
                        a = live.at({"con": p, "loc": loc, "seq": i})
                        b = live.at({"con": p, "loc": loc, "seq": j})
                        if not a or not b:
                            return None
                        if loc in (C.LOCATION_MZONE, C.LOCATION_SZONE):
                            return self._field_copy_fix(a, b)
                        if loc in (C.LOCATION_GRAVE, C.LOCATION_REMOVED):
                            return self._history_swap(a, b, cp)
                        at = self._copy_swap_cp((p, loc), cp)
                        if (a, b) in self.swaps.get(at, ()) or (b, a) in self.swaps.get(at, ()):
                            at = cp
                        return ("swap", at, a, b)
                    if loc in (C.LOCATION_GRAVE, C.LOCATION_REMOVED):
                        # no copy looks exactly like it (e.g. the copies are being
                        # processed one by one in creation order): try the copies
                        # of the same pile the other way round
                        a = live.at({"con": p, "loc": loc, "seq": i})
                        for j, g2 in enumerate(g_cards):
                            if j != i and g2 and g2.get("code") == e.get("code"):
                                b = live.at({"con": p, "loc": loc, "seq": j})
                                if a and b:
                                    fix = self._history_swap(a, b, cp)
                                    if fix is not None:
                                        return fix
                    return None
        return None

    def _hidden_candidates_fix(self, exp: dict, got: dict, duel: Duel | None = None) -> tuple | None:
        """The viewer selects among the opponent's hidden cards (e.g. in their deck):
        the prompt shows which positions hold eligible cards, so the hypothesis
        must put eligible cards exactly there."""
        key = "cards" if "cards" in exp else "select_cards"
        if exp["type"] != got["type"] or key not in exp or key not in got:
            return None

        def hidden(c):
            return c["con"] == self.opponent and not c.get("code") and (
                c["loc"] in HIDDEN_PILES or (c["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE)
                                             and c.get("pos", 0) & C.POS_FACEDOWN))
        want = {(c["con"], c["loc"], c["seq"]) for c in exp[key] if hidden(c)}
        have = {(c["con"], c["loc"], c["seq"]) for c in got[key] if hidden(c)}
        if want == have:
            return None
        for loc in sorted(have - want):
            if loc[1] in (C.LOCATION_MZONE, C.LOCATION_SZONE):
                # a face-down card that is not eligible: not the replay's card
                var = self.lockstep.var_at(*loc)
                code = self.hypothesis.var_codes.get(var.id) if var is not None else None
                entry = (self.pos, loc[0], loc[1], (loc[2],), code)
                if code and entry not in self.exclusions and entry not in self.failed_exclusions:
                    return ("exclude", *entry)
        want = {x for x in want if x[1] in HIDDEN_PILES}
        have = {x for x in have if x[1] in HIDDEN_PILES}
        if have - want and duel is not None:
            # slots the real game leaves out: a recent hidden selection of the
            # opponent took that card (e.g. each player picks a card of their own
            # deck, then cards of the other's deck except that one)
            live = infer.LocMap(duel)
            for loc in sorted(have - want):
                cid = live.at({"con": loc[0], "loc": loc[1], "seq": loc[2]})
                for dkey, _, i in reversed(self._recent):
                    k = self._card_decisions.get(dkey, {}).get(cid)
                    if k is not None and k != i:
                        return ("decide", dkey, k)
        # an eligible code, taken from one of the replay's eligible cards
        codes = []
        for c in have - want:
            var = self.lockstep.var_at(*c)
            code = self.hypothesis.var_codes.get(var.id) if var is not None else None
            if code:
                codes.append(code)
        try:
            for loc in sorted(want - have):
                self._assume({"con": loc[0], "loc": loc[1], "seq": loc[2]}, codes,
                             "the viewer's selection shows an eligible hidden card")
            for loc in sorted(have - want):
                self._assume({"con": loc[0], "loc": loc[1], "seq": loc[2]}, self._filler().main_pool,
                             "the viewer's selection shows a hidden card is not eligible")
        except Divergence as d:
            return d.fix
        return None

    def _other_copy_fix(self, cp: int, msgs: list[dict], idx: int, loc: dict) -> tuple | None:
        """Swap the card at `loc` with another copy of it from the same pile."""
        if cp not in self.snapshots:
            return None if cp in self.snapshot_cps else ("snapshot", cp)
        lm = copy.deepcopy(self.snapshots[cp])
        lm.references(msgs[:idx])
        a = lm.at(loc)
        pile = lm.lists.get((loc["con"], loc["loc"]), [])
        at = self._copy_swap_cp((loc["con"], loc["loc"]), cp)
        tried = {frozenset(p) for p in self.swaps.get(at, ())}
        for b in pile:
            if b and b != a and self.snapshots[cp].codes.get(b) == self.snapshots[cp].codes.get(a) \
                    and frozenset((a, b)) not in tried:
                return ("swap", at, a, b)
        return None

    def _copy_swap_cp(self, pile: tuple[int, int], cp: tuple[int, int]) -> tuple[int, int]:
        """Copies are best exchanged right before their pile was last shuffled
        (the forced shuffle is stable); otherwise at the current step."""
        at = self.last_shuffle_cp.get(pile)
        return at if at is not None and at <= cp else cp

    def _hint_swap_fix(self, cp: int, got: dict, duel: Duel) -> tuple | None:
        """A card hint sits on the wrong copy of a card (e.g. which of two identical
        hand cards was added this turn): swap it with the copy the real game hinted."""
        live = infer.LocMap(duel)
        a = live.at(got)
        if not a:
            return None
        j = self.pos
        while j < len(self.items) and self.items[j][0] == "msg" and self.items[j][1]["type"] == "CARD_HINT":
            e = self.items[j][1]
            if (e["hint_type"], e["value"]) == (got["hint_type"], got["value"]) and not infer.same_loc(e, got):
                b = live.at(e)
                if b and b != a and live.codes.get(b) == live.codes.get(a):
                    return ("swap", cp, a, b)
            j += 1
        return None

    def _has_swap(self, fix: tuple) -> bool:
        if fix[0] != "swap":
            return False
        _, at, a, b = fix
        lst = self.swaps.get(at, ())
        return (a, b) in lst or (b, a) in lst

    def _exclude_fix(self, loc: dict, code: int) -> tuple | None:
        """The real game took the card at `loc` although the replay has the same
        card further down the pile: scans for the first matching card
        (Duel.GetFirstMatchingCard) run from the bottom of the pile, so no card
        below `loc` can be this card."""
        if not code or loc["loc"] not in HIDDEN_PILES or loc["seq"] == 0:
            return None
        fix = ("exclude", self.pos, loc["con"], loc["loc"], tuple(range(loc["seq"])), code)
        return None if fix[1:] in self.failed_exclusions or fix[1:] in self.exclusions else fix

    def _swap_fix(self, cp: int, msgs: list[dict], idx: int, loc_exp: dict, loc_got: dict,
                  duel: Duel) -> tuple | None:
        """Two copies of the same card are processed in the wrong order.

        The copies are identified in a snapshot taken before the step that
        produced the mismatch (a restart takes that snapshot). If the step
        shuffled the pile in between, the current positions are used: a
        swap before the (stable, forced) shuffle exchanges where the copies
        end up.
        """
        if cp not in self.snapshots:
            return None if cp in self.snapshot_cps else ("snapshot", cp)
        lm = copy.deepcopy(self.snapshots[cp])
        lm.references(msgs[:idx])
        a, b = lm.at(loc_got), lm.at(loc_exp)
        if a is None or b is None:
            now = infer.LocMap(duel)
            a, b = now.at(loc_got), now.at(loc_exp)
        codes = self.snapshots[cp].codes
        if not a or not b or a == b or codes.get(a) != codes.get(b):
            return None  # only copies of the same card may trade places
        if loc_exp["loc"] in (C.LOCATION_GRAVE, C.LOCATION_REMOVED):
            # (public history: see _prompt_copy_fix)
            fix = self._history_swap(a, b, cp)
            if fix is not None:
                return fix
        at = self._copy_swap_cp((loc_exp["con"], loc_exp["loc"]), cp)
        if (a, b) in self.swaps.get(at, ()) or (b, a) in self.swaps.get(at, ()):
            at = cp
        return ("swap", at, a, b)

    # -- opponent decisions --------------------------------------------------------
    def _upcoming(self) -> list[dict]:
        out = []
        for item in self.items[self.pos:]:
            if item[0] != "msg":
                break
            out.append(item[1])
        return out

    def _upcoming_ext(self, limit: int = 40) -> list[dict]:
        """Upcoming messages, skipping over the viewer's prompts and responses."""
        return [item[1] for item in self.items[self.pos:self.pos + limit] if item[0] == "msg"]

    def _selection_window(self, up: list[dict]) -> list[dict]:
        """The messages that show what the opponent selected: when the viewer
        decides something right after it (e.g. "each player sends 3 cards"), the
        operation on the selected cards comes after both decisions."""
        if any(m["type"] not in ("HINT", "CARD_HINT") for m in up):
            return up
        i = self.pos + len(up)
        if i >= len(self.items) or self.items[i][0] != "prompt":
            return up
        # (any prompt of the viewer: e.g. "each player may reveal a card from their
        # deck" asks both before revealing both)
        return self._upcoming_ext()

    @staticmethod
    def _next_activation(up: list[dict]) -> dict | None:
        for i, m in enumerate(up):
            if m["type"] == "CHAINING":
                return m if infer.prelude_fits(up[:i], m) else None
            if m["type"] not in infer._ACTIVATION_PRELUDE:
                return None
        return None

    def _check_force(self, prompt: dict, up: list[dict]) -> None:
        """Raises a force fix if the opponent's next activation is not available."""
        t = prompt["type"]
        if t not in ("SELECT_IDLECMD", "SELECT_BATTLECMD", "SELECT_CHAIN", "SELECT_EFFECTYN"):
            return
        ch = self._next_activation(up)
        if t == "SELECT_CHAIN" and prompt["forced"]:
            matched = ch is not None and any(o["code"] == ch["code"] and o["desc"] == ch["desc"]
                                             for o in prompt["chains"])
            if not matched:
                for o in prompt["chains"]:
                    key = (self.opponent, o["code"], o["desc"])
                    if key in self._active_forces:
                        raise Divergence(self.pos, "a forced effect became mandatory",
                                         fix=("unforce", *key))
        if ch is None or ch["trigger_con"] != self.opponent:
            return
        if t == "SELECT_EFFECTYN":
            options = [prompt]
        elif t == "SELECT_CHAIN":
            options = prompt["chains"]
        else:
            options = prompt["activatable"]
        if any(o["code"] == ch["code"] and o.get("desc", ch["desc"]) == ch["desc"] for o in options):
            return
        fix = self._force_fix(ch)
        key = (self.opponent, ch["code"], ch["desc"])
        if fix is None and t in ("SELECT_IDLECMD", "SELECT_BATTLECMD") and key in self._active_forces \
                and not any(o.get("code") == ch["code"] for o in options):
            # forced, and still not available (not even as the chain windows ended):
            # a check beyond its cost and target
            # depends on hidden cards (e.g. "can be activated the turn it was Set"
            # if the deck has a fitting card); let them qualify until it is chained
            end = next((j for j in range(self.pos, len(self.items)) if self.items[j][0] == "msg"
                        and self.items[j][1]["type"] == "CHAINED"), None)
            if end is not None and (self.pos, end) not in self.match_piles:
                fix = ("matchpile", self.pos, end)
        if fix is not None:
            raise Divergence(self.pos, "opponent activation is not available in the replay", fix=fix)

    def _check_assumption(self, duel: Duel, prompt: dict, up: list[dict], ids: list) -> None:
        """The opponent took a hidden card that is no valid choice in the replay."""
        cards = prompt["cards"] if "cards" in prompt else prompt["select_cards"]
        lm = infer.LocMap(duel)
        lm.plan = []  # (past a shuffle, the moves belong to other selections)
        earlier: list[dict] = []
        for mi, m in enumerate(up):
            if m["type"] in _SELECTION_END:
                return  # later moves belong to other decisions
            if m["type"] == "CARD_SELECTED":
                # the announced selection: a hidden card in it must be a valid option
                for li in m["cards"]:
                    cid = lm.at(li)
                    if cid is None or cid in ids or li["con"] != self.opponent or li["loc"] not in HIDDEN_PILES:
                        continue
                    # (first codes not among the other selected cards: selections of
                    # differently named cards are common)
                    others = {lm.codes.get(lm.at(x)) for x in m["cards"] if x is not li}
                    codes = []
                    for c in sorted(cards, key=lambda c: (c["code"] in others, c["loc"] != li["loc"])):
                        if c["code"] and c["code"] not in codes:
                            codes.append(c["code"])
                    self._assume(li, codes, "opponent selected a card the hypothesis does not provide")
                return
            if m["type"] != "MOVE":
                lm.references([m])
                continue
            f = m["from"]
            cid = lm.at(f)
            lm.move(m)
            earlier.append(m)
            if cid is None or cid in ids:
                continue
            if f["con"] != self.opponent or f["loc"] not in HIDDEN_PILES:
                continue
            # (a card revealed in the destination right after, e.g. a searched card
            # shown after the hand was shuffled, most likely is this one)
            codes = [c["code"] for x in up[mi + 1:mi + 8] if x["type"] == "CONFIRM_CARDS"
                     for c in x["cards"] if c["code"] and (c["con"], c["loc"]) == (m["to"]["con"], m["to"]["loc"])]
            codes = list(dict.fromkeys(codes))
            for c in cards:
                if c["con"] == f["con"] and c["loc"] == f["loc"] and c["code"] and c["code"] not in codes \
                        and self._fits_destination(c["code"], m["to"]):
                    codes.append(c["code"])
            # (the slot as it is now, before the earlier moves shifted the pile)
            src = dict(f)
            for e in reversed(earlier[:-1]):
                ef = e["from"]
                if ef["con"] == src["con"] and ef["loc"] == src["loc"] and ef["seq"] <= src["seq"]:
                    src["seq"] += 1
            self._assume(src, codes, "opponent selected a card the hypothesis does not provide")
            return

    def _fits_destination(self, code: int, to: dict) -> bool:
        """Whether a card with `code` can be put face-down where `to` says (the
        field zone takes field spells, the other spell/trap zones spells, traps
        or cards Set as such)."""
        card = self.db.get(code)
        if card is None or to["loc"] != C.LOCATION_SZONE or not to["pos"] & C.POS_FACEDOWN:
            return True
        if to["seq"] == 5:
            return bool(card.type & C.TYPE_FIELD)
        return not card.type & C.TYPE_FIELD

    def _assume(self, li: dict, codes: list[int], reason: str) -> None:
        """Raises an "assume" fix for the first untried code, if the slot is free."""
        try:
            var = self.lockstep.var_at(li["con"], li["loc"], li["seq"])
        except TrackerError:
            return
        key = (li["con"], li["loc"], li["seq"])
        if var is None or (var.id in self.hypothesis.observed
                           and not any(a[:4] == (self.pos, *key) for a in self.assumptions)):
            return  # (observed, other than by an assumption at this slot and point)
        tried = {a[4] for a in self.assumptions if a[1:4] == key} | {
            a[4] for a in self.failed_assumptions if a[1:4] == key}
        for code in codes:
            if code and code not in tried:
                raise Divergence(self.pos, reason, fix=("assume", self.pos, *key, code))

    def _opp_seen_codes(self) -> list[int]:
        """Codes of the opponent's cards the stream showed, in order of appearance."""
        if not hasattr(self, "_seen_cache"):
            out: dict[int, None] = {}
            for it in self.items:
                if it[0] != "msg":
                    continue
                m = it[1]
                li = m["from"] if m["type"] == "MOVE" else m
                if m.get("code") and li.get("con") == self.opponent:
                    out.setdefault(m["code"], None)
            self._seen_cache = [c for c in out if self.db.get(c) is not None]
        return self._seen_cache

    def _filler(self) -> VanillaCompletion:
        if not hasattr(self, "_filler_cache"):
            self._filler_cache = VanillaCompletion(self.db)
        return self._filler_cache

    def _check_idle_assumption(self, prompt: dict, up: list[dict]) -> None:
        """The opponent set a hidden hand card that has the wrong kind in the hypothesis."""
        for i, m in enumerate(up):
            if m["type"] in infer._ACTIVATION_PRELUDE or m["type"] in ("PAY_LPCOST",):
                continue
            if m["type"] != "SET":
                return
            src = infer.source_of(m, up[:i])
            if src is None or src["con"] != self.opponent or src["loc"] not in HIDDEN_PILES:
                return
            kind = "msetable" if m["loc"] == C.LOCATION_MZONE else "ssetable"
            filler = self._filler()
            if kind == "msetable":
                tributes = sum(1 for x in up[:i] if x["type"] == "MOVE"
                               and (x["reason"] & (C.REASON_RELEASE | C.REASON_COST)
                                    or tracker_mod._leaves_field(x)))
                levels = {0: range(1, 5), 1: range(5, 13)}.get(tributes, range(7, 13))

                def fits(code):
                    card = self.db.get(code)
                    return card is not None and card.level in levels
            else:
                field_zone = m["seq"] == 5

                def fits(code):
                    card = self.db.get(code)
                    return card is not None and bool(card.type & C.TYPE_FIELD) == field_zone
            entries = [e for e in prompt[kind] if infer.same_loc(e, src)]
            if entries and all(fits(e["code"]) for e in entries):
                return
            usual = {0: range(1, 5), 1: range(5, 7)}.get(tributes, range(7, 13)) if kind == "msetable" else None
            codes = [e["code"] for e in prompt[kind] if fits(e["code"])]
            # (most likely a copy of a card the opponent has shown, e.g. one whose
            # Tribute the field allows: then the generic stand-ins)
            want = C.TYPE_MONSTER if kind == "msetable" else C.TYPE_SPELL | C.TYPE_TRAP
            codes += [c for c in self._opp_seen_codes() if fits(c) and self.db.get(c).type & want
                      and not self.db.get(c).type & (C.TYPE_FUSION | C.TYPE_SYNCHRO | C.TYPE_XYZ | C.TYPE_LINK)]
            if kind == "msetable":
                # (then stand-ins of the races the opponent's monsters have, e.g. a
                # Zombie where only Zombies may be Tributed for)
                races = Counter(self.db.get(c).race for c in self._opp_seen_codes()
                                if self.db.get(c).type & C.TYPE_MONSTER)
                for race, _ in races.most_common(3):
                    codes += [c for c in filler.vanillas if self.db.get(c).race == race and fits(c)][:2]
            codes += [c for c in (filler.settable_monsters_all if kind == "msetable"
                                  else filler.settable_spells) if fits(c)]
            if usual is not None:
                # (the usual Levels for the Tributes first)
                codes.sort(key=lambda c: self.db.get(c).level not in usual)
            self._assume(src, codes, "opponent set a card the hypothesis cannot set")
            return

    def _check_summon_condition(self, duel: Duel, prompt: dict, up: list[dict]) -> None:
        """The opponent special summoned a hand card by its own procedure, which
        the replay does not offer although it has that card there: the
        procedure's condition depends on hidden cards (e.g. materials in the
        hand), so let the hidden cards qualify until the summon."""
        for i, m in enumerate(up):
            if m["type"] in ("MOVE", "HINT", "CARD_HINT", "CARD_SELECTED", "CONFIRM_CARDS", "PAY_LPCOST",
                             "SHUFFLE_HAND", "SHUFFLE_DECK", "SHUFFLE_EXTRA", "LPUPDATE"):
                continue
            if m["type"] != "SPSUMMONING":
                return
            src = infer.source_of(m, up[:i])
            if src is None or src["con"] != self.opponent or src["loc"] not in (C.LOCATION_HAND, C.LOCATION_EXTRA):
                return
            if src["loc"] == C.LOCATION_EXTRA:
                # (e.g. a Synchro Summon whose check involves hidden cards; by code:
                # positions in a hidden Extra Deck say nothing; a face-up card comes
                # by Pendulum Summon, offered through the scale)
                if src.get("pos", 0) & C.POS_FACEUP or any(
                        e["loc"] == C.LOCATION_SZONE or (e["loc"] == C.LOCATION_EXTRA and e["code"] == m["code"])
                        for e in prompt["spsummonable"]):
                    return
                if m["code"] not in {code for code, _ in duel.location_cards(self.opponent, C.LOCATION_EXTRA)}:
                    return
            else:
                if any(infer.same_loc(e, src) or e["loc"] == C.LOCATION_SZONE for e in prompt["spsummonable"]):
                    return  # (a Pendulum Summon is offered through the scale)
                lm = infer.LocMap(duel)
                cid = lm.at(src)
                if not cid or lm.codes.get(cid) != m["code"]:
                    return
            span = (self.prompt_cp[0], self.pos + i)
            if span not in self.match_piles:
                raise Divergence(self.pos, "opponent summoned a card whose condition the replay does not meet",
                                 fix=("matchpile", *span))
            return

    def _candidates(self, duel: Duel, prompt: dict, up: list[dict], ids) -> tuple[list[dict], bool]:
        t = prompt["type"]
        if t == "SELECT_IDLECMD":
            return infer.infer_idle(prompt, up, self._upcoming_ext())
        if t == "SELECT_BATTLECMD":
            return infer.infer_battle(prompt, up, self._upcoming_ext())
        if t == "SELECT_CHAIN":
            cands, certain = infer.infer_chain(prompt, up)
            if cands and cands[0].get("index", -1) >= 0 and not prompt["forced"] and self._leaving_phase():
                if not self._restarts_after_chain():
                    # the phase did change: the activation came in a later window
                    cands = [{"index": -1}] + [c for c in cands if c.get("index", -1) >= 0]
            elif cands and cands[0].get("index", -1) >= 0 and not prompt["forced"] \
                    and duel.info()["phase"] == C.PHASE_BATTLE_START \
                    and self._next_state_phase() == C.PHASE_BATTLE_STEP:
                # the Battle Step starts without a message: an activation seen
                # in it came from the battle command, not the Start Step window
                cands = [{"index": -1}] + [c for c in cands if c.get("index", -1) >= 0]
                certain = True
            return cands, certain
        if t == "SELECT_EFFECTYN":
            descs = [d for code, d in duel.select_chains() if code == prompt["code"]]
            return infer.infer_effectyn(prompt, up, descs)
        if t == "SELECT_YESNO":
            return [{"yes": True}, {"yes": False}], False
        if t == "SELECT_OPTION":
            return infer.infer_option(prompt, up, self.opponent)
        if t in CARD_PROMPTS:
            up = self._selection_window(up)
        if t in ("SELECT_CARD", "SELECT_TRIBUTE", "SELECT_SUM"):
            return infer.infer_cards(prompt, up, duel, ids)
        if t == "SELECT_UNSELECT_CARD":
            return infer.infer_unselect(prompt, up, duel, ids)
        if t in ("SELECT_PLACE", "SELECT_DISFIELD"):
            return infer.infer_place(prompt, up, self._upcoming_ext())
        if t == "SELECT_POSITION":
            return infer.infer_position(prompt, up)
        if t in ("ANNOUNCE_CARD", "ANNOUNCE_NUMBER", "ANNOUNCE_RACE", "ANNOUNCE_ATTRIB"):
            return infer.infer_announce(prompt, up, self.db)
        if t == "SELECT_COUNTER":
            return infer.infer_counter(prompt, up)
        if t == "ROCK_PAPER_SCISSORS":
            return infer.infer_rps(prompt, up)
        if t in ("SORT_CARD", "SORT_CHAIN"):
            return infer.infer_sort(prompt, up)
        raise Divergence(self.pos, f"no inference for opponent prompt {t}")

    # if anything is chained in a phase-leave window (infer.PHASE_LEAVE_EVENTS),
    # the core resolves it and gives the turn player the command again
    # (IdleCommand steps 9-10, BattleCommand steps 39-40)

    def _hidden_ids(self, duel: Duel) -> list[int]:
        """Card ids of the opponent's cards the viewer cannot see (hidden piles
        and face-down cards on the field)."""
        from ..state import parse_location_query
        out = [cid for loc in HIDDEN_PILES for _, cid in duel.location_cards(self.opponent, loc)]
        for loc in (C.LOCATION_MZONE, C.LOCATION_SZONE):
            cards = parse_location_query(duel.query_location_raw(self.opponent, loc, C.QUERY_POSITION))
            for (_, cid), card in zip(duel.location_cards(self.opponent, loc), cards):
                if cid and card is not None and not card.get("position", 0) & C.POS_FACEUP:
                    out.append(cid)
        return out

    def _chain_start(self) -> int | None:
        """Stream position of the opponent's latest CHAINING (shortly before)."""
        for j in range(self.pos - 1, max(-1, self.pos - 30), -1):
            it = self.items[j]
            if it[0] == "msg" and it[1]["type"] in ("CHAIN_END", "NEW_PHASE", "NEW_TURN"):
                return None  # (no chain is being built or resolved)
            if it[0] == "msg" and it[1]["type"] == "CHAINING":
                return j if it[1]["trigger_con"] == self.opponent else None
        return None

    def _link_start(self) -> int:
        """Stream position where the chain link being built or resolved began
        (either player's; the current position outside of chains)."""
        for j in range(self.pos - 1, max(-1, self.pos - 30), -1):
            it = self.items[j]
            if it[0] == "msg" and it[1]["type"] in ("CHAIN_END", "NEW_PHASE", "NEW_TURN", "CHAIN_SOLVED"):
                return self.pos
            if it[0] == "msg" and it[1]["type"] in ("CHAINING", "CHAIN_SOLVING"):
                return j
        return self.pos

    def _next_state_phase(self) -> int | None:
        """The phase in the viewer's next state view while the upcoming chain is
        built or resolved (the phase in which it was started), or at a trigger
        right after it (its triggers are offered in the same phase)."""
        ended = False
        for item in self.items[self.pos:self.pos + 400]:
            if item[0] == "msg":
                t = item[1]["type"]
                if t in ("NEW_PHASE", "NEW_TURN"):
                    return None
                if t == "CHAIN_END":
                    ended = True
                elif ended and t not in ("HINT", "CARD_HINT", "PLAYER_HINT"):
                    return None
            elif item[0] == "prompt" and item[2] is not None:
                p = item[1]
                if ended and not (p["type"] == "SELECT_EFFECTYN" or (p["type"] == "SELECT_CHAIN" and p["forced"])):
                    return None
                return item[2]["phase"]
        return None

    def _leaving_phase(self) -> bool:
        hint = getattr(self, "_event_hint", None)
        return hint is not None and hint[0] == self.pos and hint[1] in infer.PHASE_LEAVE_EVENTS

    def _command_target(self) -> int | None:
        """The phase the viewer's phase command right before this window asked for
        (None if the window does not follow one, e.g. in the opponent's turn)."""
        j = self.pos - 1
        while j >= 1 and self.items[j][0] == "msg" and self.items[j][1]["type"] == "HINT":
            j -= 1
        it, before = self.items[j], self.items[j - 1]
        if it[0] != "response" or before[0] != "prompt" \
                or before[1]["type"] not in ("SELECT_IDLECMD", "SELECT_BATTLECMD") or not isinstance(it[1], dict):
            return None
        return {"to_ep": C.PHASE_END, "to_m2": C.PHASE_MAIN2, "to_bp": C.PHASE_BATTLE_START}.get(it[1].get("kind"))

    def _restarts_after_chain(self) -> bool:
        """Whether the stream shows the turn player's command again after the
        upcoming chain (instead of the phase change)."""
        depth, started = 0, False
        window = self.items[self.pos:self.pos + 600]
        for n, item in enumerate(window):
            if item[0] == "msg":
                t = item[1]["type"]
                if t == "CHAINING":
                    started = True
                elif t == "CHAIN_END" and started:
                    depth = 1
                elif t == "NEW_PHASE" and depth:
                    # (unless the phase reached is another than the command asked
                    # for: then the activation came in this window and changed the
                    # phase, e.g. "end the Battle Phase" after the player chose the
                    # End Phase; the core passes through phases on the way, e.g.
                    # Main Phase 2 on the way to the End Phase)
                    target = self._command_target()
                    reached = item[1]["phase"]
                    for x in window[n + 1:]:
                        if x[0] != "msg" or x[1]["type"] not in ("NEW_PHASE", "HINT"):
                            break
                        if x[1]["type"] == "NEW_PHASE":
                            reached = x[1]["phase"]
                    return target is not None and reached != target
                elif t == "WIN":
                    return True
            elif item[0] == "prompt" and depth and item[1]["type"] in ("SELECT_IDLECMD", "SELECT_BATTLECMD"):
                return True
        return True

    def _decide(self, duel: Duel, prompt: dict) -> dict:
        self.stats.opponent_decisions += 1
        self.stall += 1
        if self.stall > self.max_stall:
            raise Divergence(self.pos, "opponent decisions make no progress")
        up = self._upcoming()
        # where the upcoming shuffles put the cards (for resolving references)
        duel.shuffle_plan = [(sh["player"], sh["location"], ids) for sh, ids in
                             zip(self.hypothesis.shuffles[self._shuffle_count:],
                                 self._shuffle_ids[self._shuffle_count:])]
        self._check_force(prompt, up)
        if prompt["type"] == "SELECT_IDLECMD":
            self._check_idle_assumption(prompt, up)
            self._check_summon_condition(duel, prompt, up)
        if self.depth == 0:
            n = self._decisions_at.get(self.pos, 0)
            self._decisions_at[self.pos] = n + 1
            self._dkey = (self.pos, n)
        else:
            self._dkey = None
        key = None
        if self.depth == 0 and self.use_memo and self._dkey not in self.overrides:
            key = (self.pos, self.stall, repr(prompt))
            hit = self.memo.get(key)
            if hit is not None and prompt["type"] not in CARD_PROMPTS:
                resp, rec = hit
                if rec is not None:
                    # (still a decision the search can revisit)
                    self._recent.append((self._dkey, *rec))
                    self._since_state.append((self._dkey, *rec))
                return resp
        n = len(self._recent)
        resp = self._decide_fresh(duel, prompt, up)
        if key is not None:
            rec = self._recent[-1][1:] if len(self._recent) > n and self._recent[-1][0] == self._dkey \
                and self._recent[-1][2] >= 0 else None
            self.memo[key] = (resp, rec)
        return resp

    def _decide_fresh(self, duel: Duel, prompt: dict, up: list[dict]) -> dict:
        ids = None
        if prompt["type"] in CARD_PROMPTS:
            cards = prompt["cards"] if "cards" in prompt else prompt["select_cards"] + prompt["unselect_cards"]
            ids = infer.card_ids(duel, cards)
            self._check_assumption(duel, prompt, up, ids)
        cands, certain = self._candidates(duel, prompt, up, ids)
        # inference only orders: every other legal answer stays available behind
        # its candidates (for backtracking, and when none of them works)
        self._inferred = list(cands)
        if self.depth == 0:  # (backtracking happens on the main line only)
            cands = list(cands) + [c for c in _all_responses(prompt) if c not in cands]
        valid = []
        sig = (self.pos, repr(sorted(prompt.items())))
        if self.depth > 0 and sig == getattr(self, "probe_sig", None) \
                and prompt["type"] in ("SELECT_IDLECMD", "SELECT_BATTLECMD", "SELECT_CHAIN"):
            # back at the probed prompt without any new observation: a cycle
            # (e.g. a summon procedure that was cancelled; a card selection can
            # repeat on purpose, e.g. two cards chosen one by one from the same hand)
            raise Divergence(self.pos, "probe returned to its prompt")
        for c in cands:
            try:
                encode(prompt, c)
            except (InvalidResponse, KeyError, TypeError, ValueError):
                continue
            if c.get("cancel") and sig in self._cancelled:
                continue  # (cancelling it before led back here without progress)
            if c not in valid:
                valid.append(c)
        if not valid:
            fix = None
            if prompt["type"] in ("SELECT_PLACE", "SELECT_POSITION"):
                # the replay's effect is about to put a card on the field that the
                # real game did not (e.g. it found a fitting filler card): let the
                # hidden cards match nothing (from the opponent's latest activation,
                # even if its chain just ended: e.g. a summon it granted)
                start = self._chain_start()
                if start is None:
                    start = next((j for j in range(self.pos - 1, max(-1, self.pos - 12), -1)
                                  if self.items[j][0] == "msg" and self.items[j][1]["type"] == "CHAINING"
                                  and self.items[j][1]["trigger_con"] == self.opponent), None)
                span = (min(self._link_start(), start if start is not None else self.pos), self.pos)
                if span not in self.nomatch_piles and span not in self.match_piles:
                    fix = ("nomatchpile", *span)
            raise Divergence(self.pos, f"no candidate response for opponent {prompt['type']}", fix=fix)
        dkey = self._dkey if self.depth == 0 else None
        if dkey is not None and ids is not None:
            # (which card each single-card answer takes, for "decide" repairs)
            self._card_decisions[dkey] = {ids[c["indices"][0]]: k for k, c in enumerate(valid)
                                          if len(c.get("indices") or ()) == 1 and c["indices"][0] < len(ids)}
        if dkey is not None and dkey in self.forced_resp:
            r = self.forced_resp[dkey]
            try:
                encode(prompt, r)
                ok = True
            except (InvalidResponse, KeyError, TypeError, ValueError):
                ok = False
            if ok:
                if ids is not None:
                    self._multi_decisions[dkey] = (ids, r)
                self._recent.append((dkey, len(valid), -1))
                self._trace(prompt, r, False, len(valid))
                return r
        if dkey is not None and dkey in self.overrides and self.overrides[dkey] < len(valid):
            i = self.overrides[dkey]
            self._recent.append((dkey, len(valid), i))
            self._since_state.append((dkey, len(valid), i))
            self._trace(prompt, valid[i], False, len(valid))
            return valid[i]
        resp = self._choose(duel, prompt, valid, certain)
        if resp.get("cancel"):
            self._cancelled.add(sig)
        if dkey is not None and prompt["type"] == "SELECT_CHAIN" and resp.get("index", -1) == -1 \
                and self._leaving_phase():
            # (passed in a phase-leave window in favour of a later one: which window
            # it was can show much later, e.g. an effect ending the Battle Phase that
            # lingers into the next one when activated after the Battle Step)
            act = next((k for k, c in enumerate(valid) if c.get("index", -1) >= 0), None)
            if act is not None:
                self._window_choices.append((dkey, act))
        if dkey is not None and ids is not None and resp.get("indices"):
            self._multi_decisions[dkey] = (ids, resp)
        if dkey is not None and prompt["type"] in CARD_PROMPTS and any(
                c["con"] == self.opponent and c["loc"] & (C.LOCATION_DECK | C.LOCATION_HAND | C.LOCATION_EXTRA)
                for c in prompt.get("cards") or prompt.get("select_cards") or []):
            self._hidden_picks.append((self.pos, (self._link_start(), self.pos)))
        if dkey is not None:
            self._recent.append((dkey, len(valid), valid.index(resp)))
            self._since_state.append((dkey, len(valid), valid.index(resp)))
        return resp

    def _choose(self, duel: Duel, prompt: dict, valid: list[dict], certain: bool) -> dict:
        n_inferred = sum(1 for c in valid if c in self._inferred)
        if certain or len(valid) == 1 or n_inferred == 1:
            # (one inferred answer: the others are there for backtracking only)
            self._trace(prompt, valid[0], True, len(valid))
            return valid[0]
        if self.depth > 0 and self.pos - self.probe_start >= self.min_progress and self.probe_boundary \
                and self.probe_after_boundary > 0 \
                and (self.probe_state_at is None or self.pos > self.probe_state_at):
            # (and something substantive after the latest boundary: a consequence
            # of the decision can show only once its effect is over, e.g. "then you
            # can Normal Summon" right after the chain ends)
            # the probed decision already reproduced enough new observations;
            # the next ambiguous decision is the parent's business
            raise _ProbeSuccess()
        if self.depth >= self.max_depth:
            return valid[0]
        self.stats.probed_decisions += 1
        # a choice among face-up cards on the field shows in the next state view
        # (e.g. which Pendulum Scale an effect raised), even a later one; so does
        # a yes/no or option that changes stats without any message (e.g. "increase
        # its Level by 4?")
        cards = prompt.get("cards") or prompt.get("select_cards") or []
        self._probe_wide = prompt["type"] in ("SELECT_YESNO", "SELECT_OPTION") or (
            prompt["type"] in CARD_PROMPTS and bool(cards) and all(
                c["loc"] in (C.LOCATION_MZONE, C.LOCATION_SZONE) and c.get("pos", C.POS_FACEUP) & C.POS_FACEUP
                for c in cards))
        # No candidate may reproduce the stream as is. Then the best is the one
        # that reproduced the most of it before running into a problem, a
        # repairable one (e.g. two copies of a card that have to trade places)
        # before one the replay cannot repair; one that only ran out of time
        # without contradicting the stream comes after those.
        best = None  # (reached, repairable, candidate)
        unrefuted = None
        for k, c in enumerate(valid[:self.max_probes]):
            if k >= n_inferred and (best is not None or unrefuted is not None):
                break  # (the answers inference left out only when its own all failed)
            self.stats.probes += 1
            outcome = self._probe(duel, prompt, c)
            reached = self.last_probe_reached
            if outcome == PROBE_OK:
                self._trace(prompt, c, False, len(valid))
                return c
            if outcome in (PROBE_REPAIRABLE, PROBE_REPAIRABLE_AT_ONCE, PROBE_PROGRESSED):
                key = (reached, outcome != PROBE_PROGRESSED)
                if best is None or key > best[:2]:
                    best = (*key, c)
            if outcome == PROBE_TIMEOUT and unrefuted is None:
                unrefuted = c
        choice = best[2] if best is not None else unrefuted
        if choice is not None:
            self._trace(prompt, choice, False, len(valid))
            return choice
        fix = None
        if prompt["type"] == "SELECT_OPTION" and not any(
                m["type"] == "HINT" and m["hint_type"] == C.HINT_OPSELECTED and m["player"] == self.opponent
                for m in self._upcoming()[:12]):
            # the real game shows no option choice here: it had only one option (a
            # script offers just the available ones), the replay has more because
            # hidden (filler) cards satisfy a condition: let them match nothing
            start = self._chain_start()
            span = (min(self._link_start(), start if start is not None else self.pos), self.pos)
            if span not in self.nomatch_piles and span not in self.match_piles:
                fix = ("nomatchpile", *span)
        cards = prompt.get("cards") or prompt.get("select_cards") or []
        if fix is None and prompt["type"] in CARD_PROMPTS and cards and all(
                c["con"] == self.opponent and (c["loc"] & (C.LOCATION_DECK | C.LOCATION_HAND | C.LOCATION_EXTRA)
                                               or not c.get("pos", C.POS_FACEUP) & C.POS_FACEUP)
                for c in cards):
            # no answer fits: the real game had nothing to choose from, the
            # replay's hidden (filler) cards fit the selection (e.g. "each player
            # adds a Normal Monster from their Deck" with none in that Deck)
            start = self._chain_start()
            span = (min(self._link_start(), start if start is not None else self.pos), self.pos)
            if span not in self.nomatch_piles and span not in self.match_piles:
                fix = ("nomatchpile", *span)
        d = Divergence(self.pos, f"no candidate for opponent {prompt['type']} survived probing",
                       None, prompt, fix)
        if fix is None:
            d.fallback = self._hidden_pick_fallback()[0]
        raise d

    def _trace(self, prompt: dict, response: dict, certain: bool, n: int) -> None:
        if self.trace is not None and self.depth == 0:
            self.trace.append((self.pos, prompt, response, certain, n))

    def _probe(self, duel: Duel, prompt: dict, response: dict) -> int:
        """Checks a candidate response in a forked copy of the process.

        Returns PROBE_OK if the candidate reproduced the stream (up to the
        probe horizon or the next ambiguous decision after enough progress),
        PROBE_REPAIRABLE if it ran into a divergence the replay can repair,
        and PROBE_FAILED otherwise. Choosing the wrong copy of a card looks
        like a repairable divergence, which is why clean successes win.
        """
        rfd, wfd = os.pipe()  # (the probe reports how far it reproduced the stream)
        pid = os.fork()
        if pid == 0:
            code = 2
            reached = self.pos
            os.close(rfd)
            try:
                signal.signal(signal.SIGALRM, lambda *_: os._exit(3))
                # nested probes get less time, so they cannot eat their parent's time
                signal.alarm(max(2, self.probe_timeout >> self.depth))  # (depth of the parent)
                self.depth += 1
                self.check_state = False  # the main line checks states; probes only choose
                # (a probe making many decisions without reproducing anything loops)
                self.max_stall = min(self.max_stall, 50)
                self.probe_start = self.pos
                self.probe_boundary = False  # (the probed action's effect is over)
                self.probe_after_boundary = 0  # (non-hint messages reproduced since the latest one)
                self.probe_sig = (self.pos, repr(sorted(prompt.items())))
                self.horizon = min(len(self.items), self.pos + self.probe_items)
                # ... except the first one: a decision may show only there (e.g.
                # "lower its Level?"), so the probe must get past it
                # (only if it comes soon: farther on, unrelated problems would
                # decide the probe)
                self.probe_state_at = next(
                    (j for j in range(self.pos, min(self.horizon, self.pos + (
                        self.probe_items if getattr(self, "_probe_wide", False) else self.probe_state_window) + 1))
                     if self.items[j][0] == "prompt" and self.items[j][2] is not None), None)
                self._loop(duel, prompt, response)
                code = 0
                reached = self.pos
            except _ProbeSuccess:
                code = PROBE_OK
                reached = self.pos
            except Divergence as e:
                reached = e.pos
                code = (PROBE_REPAIRABLE if e.pos > self.probe_start else PROBE_REPAIRABLE_AT_ONCE) \
                    if e.fix is not None else \
                    PROBE_PROGRESSED if e.pos - self.probe_start >= self.min_progress else PROBE_FAILED
                if os.environ.get("YGOAI_PROBE_DEBUG"):
                    from ..pretty import msg_str
                    exp = e.expected[1] if isinstance(e.expected, tuple) else e.expected
                    show = lambda m: (msg_str(m) if isinstance(m, dict) and "type" in m else str(m))[:160]
                    print(f"   probe {response} @{self.probe_start}: {e} fix={e.fix} | exp {show(exp)} | got {show(e.got)}",
                          file=sys.stderr)
            except BaseException:
                code = 2
                if os.environ.get("YGOAI_PROBE_DEBUG"):
                    import traceback
                    traceback.print_exc()
            finally:
                try:
                    os.write(wfd, reached.to_bytes(4, "little"))
                finally:
                    os._exit(code)
        os.close(wfd)
        try:
            _, status = os.waitpid(pid, 0)
            # (non-blocking: a probe that timed out wrote nothing, and its own
            # probes may still hold the pipe open)
            os.set_blocking(rfd, False)
            try:
                data = os.read(rfd, 4)
            except BlockingIOError:
                data = b""
        except BaseException:
            # e.g. a timeout of the caller: never leave the probe running
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise
        finally:
            os.close(rfd)
        self.last_probe_reached = int.from_bytes(data, "little") if len(data) == 4 else self.pos
        code = os.waitstatus_to_exitcode(status)
        if self.log and self.depth == 0:
            self.log(f"probe @{self.pos} {prompt['type']} {response} -> exit {code}")
        return code if code in (PROBE_OK, PROBE_REPAIRABLE, PROBE_TIMEOUT, PROBE_PROGRESSED,
                                PROBE_REPAIRABLE_AT_ONCE) else PROBE_FAILED


def belief_replay(viewer: int, init: dict, items: list[tuple], **kwargs) -> tuple[Duel, BeliefReplay]:
    br = BeliefReplay(viewer, init, items, **kwargs)
    duel = br.run()
    return duel, br
