# yugioh_AI_2

A testing framework for Yu-Gi-Oh! AIs built on [ygopro-core](https://github.com/edo9300/ygopro-core)
(the EDOPro rules engine), with the card scripts from
[ProjectIgnis/CardScripts](https://github.com/ProjectIgnis/CardScripts) and the card database
from [ProjectIgnis/BabelCDB](https://github.com/ProjectIgnis/BabelCDB).

It has three parts:

1. **Arbiter and match runner** (the [fastchess](https://github.com/Disservin/fastchess)
   counterpart): plays games between AIs, gives every AI exactly the information its player is
   allowed to have, records games, and reports W/D/L, Elo and SPRT results.
2. **Belief replay**: a library for AIs with imperfect information. From a player's observations
   alone it reconstructs a complete ygopro-core duel that is indistinguishable from the real one
   from that player's point of view, so search algorithms (MCTS, alpha-beta, ...) can simulate
   ahead, e.g. with `fork()`.
3. **Consistency tests** for the belief replay: random and greedy AIs play thousands of games
   with human deck lists; for every game, both players and several cut points, the
   reconstruction must reproduce the player's observation stream exactly.

## Setup

```sh
git submodule update --init --recursive   # ygopro-core (+lua), CardScripts, BabelCDB
scripts/build_core.sh                      # builds build/libocgcore.so (patched core)
python3 scripts/gen_constants.py           # only needed after updating ygopro-core
```

Requirements: a C++17 compiler, meson, ninja, Python >= 3.11 with numpy and scipy. The human deck lists
are expected in `download_human_decks/decks/*.ydk` (override with `YGOAI_DECKS`).

## Running things

```sh
# self-play statistics between built-in agents
python3 scripts/selfplay.py -n 1000 --agents mixed

# a match, fastchess style (NAME=COMMAND, builtin:<agent> or any engine process)
python3 scripts/match.py --engine greedy=builtin:greedy \
    --engine "random=python3 -m ygoai.engine random" -n 200 --sprt 0 10 --tc 60+0.5

# belief replay consistency test (both players, cut at 1/3, 2/3 and the full game);
# every reconstruction is a separate task, results are streamed to the JSON lines file
python3 scripts/check_belief.py -n 1000 --start 1000 -o runs/check.jsonl

# quick iteration: rerun only what failed before, then diff the two runs
python3 scripts/check_belief.py --cases runs/check.jsonl --only-failed -o runs/retry.jsonl
python3 scripts/compare_runs.py runs/check.jsonl runs/retry.jsonl

# test only games with certain mechanics (tags: xyz, sort_card, toss_coin, shuffle_set_card, ...)
python3 scripts/index_games.py -n 1000 --start 1000 -o runs/index.jsonl
python3 scripts/check_belief.py --index runs/index.jsonl --tags shuffle_set_card,toss_coin -o runs/t.jsonl

# look at one reconstruction in detail (--max-restarts N stops after N repairs)
python3 scripts/debug_belief.py SEED VIEWER [--cut 0.5]

# leak check: does the (secret) card creation order show in the observations?
python3 scripts/diff_creation.py -n 1000 --start 5000

# unit and integration tests
python3 -m unittest discover tests
```

## The engine protocol

An AI is a process speaking JSON lines on stdin/stdout (see `ygoai/protocol.py`):

```
-> {"type": "init", "player": 0, "deck": {"main": [...], "extra": [...]}, "extra_order": [...],
    "opponent_deck_size": 40, "opponent_extra_size": 15, "config": {...}}
-> {"type": "prompt", "obs": [...messages since the last prompt...], "prompt": {...}, "state": {...}}
<- {"response": {...}}
-> {"type": "end", "obs": [...], "result": {"winner": 1, "reason": 1, ...}}
```

`obs` are the core's messages as parsed by `ygoai/messages.py`, filtered for the player by
`ygoai/observe.py`: the player sees what an EDOPro client would show, minus several leaks that
EDOPro has (hidden-card effect descriptions in card hints, stale "reason" fields, deck positions
that distinguish returned cards, ...). A `prompt` is a core selection message, and the response
format for each kind is documented in `ygoai/actions.py`. `state` is the player's view of the
whole field. `python -m ygoai.engine greedy` runs a built-in agent as an engine, which doubles as a
minimal example.

## How the belief replay works (`ygoai/belief/`)

* `tracker.py` follows every card slot of both players through the observation stream,
  mirroring the core's pile semantics exactly. Unknown identities are variables; a shuffle turns
  the cards of a pile into a *pool* whose members are assigned to the new slots in an unknown
  way. Everything the stream implies becomes a constraint of this model:
  * revealed cards (observations);
  * the kind of hidden cards: a card Set face-down in a spell/trap zone is a spell/trap (or a
    monster that can be Set there), a Set monster has a level matching its tributes;
  * negative information: a scan for the first matching card (`Duel.GetFirstMatchingCard`)
    passed over the cards below the one it took;
  * which physical card is which. Each rule below is taken from the core or script source:
    * card hints are shown again after a hand shuffle (`field::shuffle`);
    * the chain shows where an activated card is now;
    * after an effect activated from the hand, the hand is shuffled right after CHAIN_SOLVED if the
      card is still there (`SolveChain`), or right before it if the card moved (the
      `proc_workaround.lua` handler), so the viewer can tell whether a card that left the hand was
      the activated one;
    * the hand card the viewer's own command summons or Sets is the card that moves to the
      field next, even if the hand is shuffled in between (e.g. after a reveal as a cost);
    * cards kept revealed in a hand stay revealed through a shuffle.
* `flow.py` solves that model exactly: every slot carries one card code and every shuffle
  conserves the multiset of codes, which makes it an integer multi-commodity flow over the DAG
  of shuffles (a few thousand variables, solved by HiGHS through `scipy.optimize.milp`; no search
  budget, no random restarts). Unknown opponent cards stay typed wildcards ("some spell/trap")
  until a `DeckCompletion` fills them (effect-less vanilla cards, or the corpus-based
  `CorpusCompletion` which uses the most similar human decks). Identity constraints enter the
  model where they imply something about codes (a slot has the code of the card it is known to
  be, or its code comes from a card other than the one it is known not to be); copy identity is
  then fixed by matching claims to members per shuffle (copies nothing tells apart are
  interchangeable), and the duel is built by card id (deck order and every shuffle result).
* `replay.py` creates a duel from that hypothesis and replays it in lockstep with the stream,
  using a small core extension (`core_ext/`) to force shuffle results, coin/dice results and
  random selections. The viewer's prompts are answered with the recorded responses; the
  opponent's hidden decisions are inferred from what the viewer saw next (`infer.py`), and
  ambiguous ones are verified by probing candidates in forked processes.
* Opponent effects that the stream shows being activated are made activatable regardless of
  hidden costs and targets (e.g. a negated search whose target was never revealed), but only
  from the start of the phase in which they were activated until the activation, and only if
  the normal check fails (target checks can have side effects scripts rely on).
* Some hidden decisions are decided by rules of the core rather than by the next visible message,
  e.g. an activation in the window the non-turn player gets when the turn player leaves the Main
  Phase or the Battle Step makes the core go back to the command (IdleCommand/BattleCommand), so
  the stream after the chain tells in which window the opponent activated.
* The goal is a single pass. When the replay still cannot reproduce an observation it repairs
  the hypothesis and restarts; the check scripts report how many reconstructions needed no
  restart at all. The repairs (see the docstring of `replay.py`):
  * force an effect the stream shows (also one a continuous trap was activated together with);
  * assume a card of a certain kind or code where the opponent took a hidden card;
  * let a hidden card satisfy every filter during one step, or, when the opponent chose an
    option or used a summon procedure whose availability depends on its hidden cards, every
    card of its hidden piles from the activation until the choice; conversely, when the real
    game did nothing more where the replay found a fitting card among its filler cards, let
    the hidden cards match no filter for that span (or, when the replay's operation took one
    card more, e.g. "send all monsters from the hand", let just that card match nothing);
  * exchange two copies of a card, at the step where they last shared a shuffled pile or, for
    copies whose history is public (graveyard, banished), from the start of the duel;
  * identify physical cards: the card a script moves out of a shuffled pile is the one it
    revealed before the shuffle (it keeps them in a group). Where the codes show, the matching
    of copies prefers these cards anyway; where they do not, the moved cards are constrained to
    be among the revealed ones;
  * retry a different candidate for one recent opponent decision.

### Using the belief replay in an AI

An AI keeps its observation stream as a list of items `("msg", message)`, `("prompt", prompt,
state)` and `("response", response)`, exactly as the engine protocol delivers them. At a
decision it reconstructs the duel and searches from it with `fork()`:

```python
from ygoai.belief.replay import BeliefReplay, Divergence
from ygoai.belief.corpus import CorpusCompletion   # or VanillaCompletion
from ygoai.actions import encode

replay = BeliefReplay(player, init, items, completion=CorpusCompletion())
duel = replay.run()             # a live duel, waiting for this player's response
pid = os.fork()                 # e.g. one child per candidate response
if pid == 0:
    duel.set_response(encode(prompt, candidate))
    ...                         # simulate on, evaluate, report through a pipe
```

`ygoai/agents/belief_mc.py` is a complete (deliberately simple) example: flat Monte Carlo with
greedy rollouts on the reconstructed duel. The reconstruction is rebuilt from scratch at every
searched decision; making it incremental is an obvious next step.

## Current results

Measured with `scripts/check_belief.py -n 1000` (random and greedy agents, random human deck
pairs from the corpus, 20% mirror matches; both players; cut after ~1/3, ~2/3 and at the end of
the game, 6000 reconstructions per run; the reconstruction must reproduce every filtered
message, every prompt of the viewer and the viewer's state view at every prompt). Seeds
1000-1999 were used while developing; seeds 2000-2999 are the check for generalization:

| cut      | seeds 1000-1999 | seeds 2000-2999 |
|----------|-----------------|-----------------|
| ~1/3     | 99.7%           | 99.8%           |
| ~2/3     | 99.5%           | 99.4%           |
| full game| 99.2%           | 99.3%           |
| overall  | 99.5%           | 99.5%           |

About 96% of all reconstructions need no repair restart at all. The remaining failures are mostly
effects whose behaviour depends on hidden cards in ways no repair covers yet (e.g. an
excavation whose result decides what the opponent can pick), and copy identity puzzles.

Median reconstruction time is about 0.3s (a few long games take up to the 90s limit because
of repeated repair restarts). Plain self-play is fast: about 0.2s CPU per game, ~650 decisions per game.

First match of the example search agent (`belief_mc`: belief replay + flat Monte Carlo with
greedy rollouts, vanilla deck completion) against `greedy`, 40 openings with swapped seats:
+52 =0 -28, Elo +107 +/- 67.

Open design questions (from the remaining failures):

* **Physical identity in the solver.** The flow model reasons about codes; identity facts enter
  it only where they imply code facts, and the per-shuffle matching applies the rest. An exact
  treatment would make every physical card its own commodity (bigger model), or bind hidden
  cards lazily, only when they are used, choosing among the candidates by search.
* **Hidden decisions with late consequences.** Opponent decisions are inferred from the next
  visible messages and verified by short probes; when the consequence only shows much later
  (e.g. after a whole chain resolved), a probe can accept a wrong choice. A backtracking search
  over hidden decisions (with fork() checkpoints) would handle these uniformly.
* **Hidden option availability.** Some script-level choices (e.g. which effect of a card can be
  chosen) depend on hidden cards; forcing activation and forcing a card to satisfy a filter
  cover the common cases, but not options built inside a script.

## Information leaks

`ygoai/observe.py` hides everything a player must not know, and the belief replay doubles as a
leak detector: it rebuilds the game with *wrong* hidden cards, so any hidden detail that leaks
into the observations makes the streams differ. This found several leaks that EDOPro itself has:
positions of face-down cards in piles (a card returned to the deck has a different position
value than the original deck cards), stale "reason" fields (how a card got into the hand,
visible when it is later played or reordered), card hints emitted while cards are created (they
reveal deck contents), `reason_card` pointers into hidden piles, status bits of pile cards, and
multi-card listings in card creation order. `ygoai/audit.py` is an independent check that
replays games step by step and flags opponent card codes that appear in a stream without ever
having been visible; it reports nothing on the current filter (and does catch deliberately
planted leaks).

Further leaks fixed since: empty chain windows (the core offers them when the *opponent* has a
pending optional trigger, so they are answered by the arbiter and not shown), the
ATTACK_CANCELED status bit (choosing an attacker clears it even if the attack is then cancelled,
so it revealed hidden attack attempts), status bits that a card kept from an earlier stint on the
field (they told physical copies apart; fixed in the core), and the physical copy in each deck
slot (the arbiter now places copies of a card in random order, instead of in creation order),
and the order of options in command, chain and card selection prompts (the core sorts them by
effect id, which depends on card creation order and on when effects were last enabled, or lists
cards in creation order, which tells physical copies apart; the arbiter shows them in a
canonical order, by location, and maps the chosen indices back, see
`observe.to_core_response`),
the stale reason on every move onto the field except special summons (equipping a monster
or moving it to another zone showed how the card had once entered the hand), and the position
value in DECK_TOP (it showed whether the card had been returned to the deck). Conversely, a
player now sees its own cards when it reveals them from its deck.

Card creation order: the core gives every card an id in creation order and processes groups
(a script's loop over a group, the moves of one operation, target announcements, counters
placed on several cards, ...) in id order. With cards created in a fixed order (sorted by code),
the order of the messages of several hidden cards revealed their relative codes, and the order
of copies of one card told them apart across shuffles. Now the arbiter creates the cards in a
secret random order, and the patched core processes groups in location order (controller, then
location, then sequence) wherever the order can show, so the creation order has no observable
effect at all. `scripts/diff_creation.py` checks exactly that: it replays recorded games with
other creation orders and compares both players' observation streams (no difference in 2,000
games). Location order adds nothing the messages do not show anyway, except where a script
turns the iteration order into something without locations: a card-name declaration built by
iterating over the deck lists the names in deck order, so the arbiter shows ANNOUNCE_CARD
filters in a canonical form (and ANNOUNCE_NUMBER options sorted).

## Changes to ygopro-core

`core_ext/ocgcore.patch` (applied by `scripts/build_core.sh` to a copy of the submodule) plus
`core_ext/ocgext.cpp`:

* Hooks for forcing outcomes, all no-ops unless the `OCG_Ext*` API is used: shuffle results
  (deck, hand, extra, `Duel.ShuffleSetCard` and the zone assignment of a group SSet), coin and
  dice results, `Group.RandomSelect`,
  "treat this effect as activatable" (skips cost/target checks), "treat this card as
  matching every filter" and "treat this card as matching no filter".
* Extra API: reorder a pile (by code or by card id), force shuffle results by card id, swap two
  cards within a pile, read card ids per location, clear the forced activations.
* A card entering the hand, deck or Extra Deck forgets its position and activation history
  (JUST_POS, CONTINUOUS_POS, ATTACK_CANCELED, ACT_FROM_HAND status bits): it is a new object by
  the rules, and the stale bits made physical copies distinguishable.
* Groups are processed in location order instead of card creation order (card id): iterating a
  group from Lua (`GetFirst`/`GetNext`, `Filter`, `RandomSelect`, ...), the loops of the
  operations (moves, summons, sets, effect enabling, control changes), target and counter
  announcements, and effects expiring together (sorted by effect id). See the information leaks
  above.
* A hand card that becomes public (`EFFECT_PUBLIC`, e.g. a card revealed when drawn) or private
  again is announced as a position change. The core only changed its position silently, so
  players saw revealed cards only in their next state view, and a revealed card that was hidden
  again before that was never seen at all.
* A card an effect Sets (`Duel.SSet`, e.g. from the Deck) is confirmed to the other player. The
  core named the player who Sets it as the one who looks (pointless); the EDOPro server makes
  up for it by showing every confirmation of cards outside the Deck to both players, which
  would also show a player's private look at its own face-down card to the opponent.
* Queries do not change the duel. Evaluating an effect's condition gives the effect a new
  timestamp when the condition has become true (`effect::is_available`), and the timestamps
  order continuous effects; a state query (e.g. of a card's level) evaluated conditions at other
  moments than the game itself, so observing the state could change what happened next (about
  1 in 25 games with the upstream core diverged when every prompt queried levels).
* A step that runs more than 10^9 Lua instructions raises a script error instead of never
  returning: some scripts take exponential time in rare situations (e.g.
  `aux.SelectUnselectGroup` over 24 cards whose condition no subset can meet any more), which
  would hang a match. The count is deterministic, so replays stop at the same point.
* A fix for an upstream crash: `field::reset_phase`/`reset_chain` iterated a pointer-keyed hash
  set while removing effects, which could invalidate the iterator (roughly 1 in 20,000
  random games segfaulted) and made the order of the resulting messages depend on memory
  addresses. They now iterate a snapshot sorted by effect id.

The command line tools run games in a crash tolerant process pool (`ygoai/parallel.py`): a
worker that dies is reported with its seed instead of hanging the run. `scripts/build_core.sh`
replaces the library atomically, so running test processes are not affected by a rebuild.

Known upstream issue: `c87257460.lua` (Allure Queen LV3) raises Lua errors with the current
CardScripts; games using it still run.

## Layout

```
core_ext/      patch + extension for ygopro-core (forced outcomes, reordering, forcing)
external/      submodules: ygopro-core, CardScripts, BabelCDB
scripts/       command line tools
tests/         unit and integration tests
ygoai/         the Python package
  core.py        ctypes binding          messages.py   message parser
  actions.py     responses              observe.py    per-player filtering
  state.py       state snapshots        game.py       arbiter, records, replays
  protocol.py    engine protocol        match.py      matches, Elo, SPRT
  agents/        random and greedy AIs  belief/       tracker, replay, inference
```
