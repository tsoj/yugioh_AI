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

Requirements: a C++17 compiler, meson, ninja, Python >= 3.11 with numpy and scipy. The learned deck
completion (`--completion neural`, ygoai/belief/neural_completion.py) also needs PyTorch and the models in
`download_human_decks/` (`deck_existence.pt`, `deck_counts.pt`); the project environment is made with uv:

```sh
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python numpy==2.4.6 scipy==1.18.0 tqdm
uv pip install --python .venv/bin/python "torch==2.13.0+rocm7.1" --index-url https://download.pytorch.org/whl/rocm7.1
```

and scripts are run with `.venv/bin/python` (inference on the CPU, deterministic; the GPU is for training). The human deck lists
are expected in `download_human_decks/decks/*.ydk` (override with `YGOAI_DECKS`).

## Running things

```sh
# self-play statistics between built-in agents
python3 scripts/selfplay.py -n 1000 --agents mixed

# a match, fastchess style (NAME=COMMAND, builtin:<agent> or any engine process)
python3 scripts/match.py --engine greedy=builtin:greedy \
    --engine "random=python3 -m ygoai.engine random" -n 200 --sprt 0 10 --tc 60+0.5

# belief replay consistency test (both players, cut at the end of turns 1, 2 and 3, at 1/3,
# 2/3 and the full game; turn cuts show as frac -1, -2, -3; --fractions / --turns choose);
# one task per game (played once, both observation streams recorded while it is played),
# results are streamed to the JSON lines file
python3 scripts/check_belief.py -n 1000 --start 1000 -o runs/check.jsonl
# ... with other agent pairings (seed % number of pairs picks one; agents: random, greedy,
# setter = sets cards face-down and responds in chain windows, mixed = greedy with 30% random)
python3 scripts/check_belief.py -n 1000 --start 4000 --pairs setter-greedy,mixed-setter -o runs/c.jsonl
# ... with decks chosen for card variety: the corpus without duplicates, ordered so that
# each next deck adds the most cards not covered yet (game seed s uses positions 2s, 2s+1;
# the order is cached in build/diverse_decks.json); popular builds are near duplicates
python3 scripts/check_belief.py -n 1000 --start 5000 --decks diverse -o runs/c.jsonl
# ... or an opening book covering every card of the corpus (see below)
python3 scripts/make_book.py -o build/book.jsonl
python3 scripts/check_belief.py -n 1143 --start 7000 --book build/book.jsonl -o runs/c.jsonl
# decks of random legal cards (no human deck structure: edge cases); each deck gets a
# style: 1-3 copies uniformly, mostly 3, only 3, or singleton
python3 scripts/check_belief.py -n 1000 --start 40000 --decks cards -o runs/r.jsonl
python3 scripts/card_coverage.py -n 1143 --start 7000 --book build/book.jsonl

# quick iteration: rerun only what failed before, then diff the two runs
python3 scripts/check_belief.py --cases runs/check.jsonl --only-failed -o runs/retry.jsonl
python3 scripts/compare_runs.py runs/check.jsonl runs/retry.jsonl

# test only games with certain mechanics (tags: xyz, sort_card, toss_coin, shuffle_set_card, ...)
python3 scripts/index_games.py -n 1000 --start 1000 -o runs/index.jsonl
python3 scripts/check_belief.py --index runs/index.jsonl --tags shuffle_set_card,toss_coin -o runs/t.jsonl

# look at one reconstruction in detail (--max-restarts N stops after N repairs)
python3 scripts/debug_belief.py SEED VIEWER [--cut 0.5]   # (--cut -2: the end of turn 2)

# which hidden card or copy a failed reconstruction has wrong: the hidden cards the
# engine read before the first divergence (core read log), and trial swaps/codes
python3 scripts/suggest_repair.py SEED VIEWER [--cut 0.5] [--pairs ...] [--decks ...]

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
  * exchange the roles of two copies of an Xyz material from the start of the duel when the
    viewer is offered an effect they grant a different number of times (which copy used it);
  * exclude a code where the replay flipped or offered a face-down card the real game did not;
  * retry a different candidate for one recent opponent decision (also when a message that
    shows stats, e.g. an attack's ATK, differs: like a state view, it shows the result of every
    decision since the last one).

  The main line keeps the positions of the latest step, which most repairs need, so finding a
  repair takes no restart of its own.

  When these repairs give up, a last resort (`read_guided.py`): replayed without repairs, the
  core's read log names the hidden cards the engine read just before the first divergence;
  each is tried as another copy of its name in the same place (copies differ in history: which
  one was revealed, drawn, used) or with other codes that differ in the properties it read,
  each candidate a whole replay with the usual repairs. It covers the wrong hidden card no rule
  addresses (a deck card whose ATK an effect compares, a face-down monster that keeps the
  viewer from attacking); it runs only where the replay would fail. It can also hide a problem
  that should be fixed instead (a leak that makes copies differ showed as such a case), so the
  tests count what it saves as failures ("read-guided repair needed: ..."; `check_belief.py
  --read-guided-ok` counts them as consistent).

### Using the belief replay in an AI

An AI keeps its observation stream as a list of items `("msg", message)`, `("prompt", prompt,
state)` and `("response", response)`, exactly as the engine protocol delivers them. At a
decision it reconstructs the duel and searches from it with `fork()`:

```python
from ygoai.belief.agent import AgentBelief
from ygoai.belief.neural_completion import neural_completion   # or CorpusCompletion()
from ygoai.actions import encode
from ygoai.observe import to_core_response

belief = AgentBelief(player, init, completion=neural_completion())   # one per game
...
replay, duel = belief.reconstruct(items)   # a live duel, waiting for this player's response
pid = os.fork()                            # e.g. one child per candidate response
if pid == 0:
    core_prompt = replay.final_prompt   # (the core's version of the prompt)
    duel.set_response(encode(core_prompt, to_core_response(core_prompt, player, candidate)))
    ...                                    # simulate on, evaluate, report through a pipe
```

`AgentBelief` reconstructs decision by decision: the previous consistent duel goes on with the
new observations (`BeliefReplay.extend`; the viewer's own draws and hand shuffles are arranged
just in time, as a hypothesis of the longer stream would arrange them); when that runs into a
mismatch, a full replay inherits the previous one's repairs and decisions
(`BeliefReplay.inherit`); when that fails too, one from scratch. The duels belong to the
`AgentBelief` (search in forks, never change or close them). When no consistent reconstruction
exists, `replay.approximate` is set: the best effort re-aligned with the stream, its public state
re-synchronized with what the player saw at each of its Idle/Battle commands (`belief/sync.py`).

`ygoai/agents/belief_mc.py` is a complete (deliberately simple) example: flat Monte Carlo with
greedy rollouts on the reconstructed duel. `scripts/check_belief.py --agent-sequence K` checks
reconstructions made this way (every K-th decision of each player; suite range
`agentseq50000`), `--agent-mode` single reconstructions as an agent sees them (with the state
differences of approximate ones).

## Current results

Measured with `scripts/check_belief.py -n 1000` (both players; cut at the end of turns 1, 2 and 3,
after ~1/3, ~2/3 and at the end of the game, ~9,800 reconstructions per run (a turn cut that gives
the same prefix as another one is skipped); the reconstruction must reproduce every filtered
message, every prompt of the viewer and the viewer's state view at every prompt). A run takes
about 2 minutes on 32 threads (i9-13900K).
Seeds 1000-4999 use random corpus deck pairs (20% mirror matches); 5000-6999 use the decks
ordered for card variety (`--decks diverse`). Agents: random and greedy, or (marked "other
agents") the `setter` and `mixed` agents against each other and the others (`--pairs
setter-greedy,mixed-setter,greedy-mixed,setter-random,mixed-mixed,random-setter`). By now
all of them have been used while developing (5000-6999 started out fresh, at 99.43% / 99.32%; 7000-7999
at 99.79%):

| cut      | 1000-1999 | 2000-2999 | 3000-3999 | 4000-4999 (other agents) | 5000-5999 (diverse decks) | 6000-6999 (diverse, other agents) | 7000-7999 (diverse, other agents) |
|----------|-----------|-----------|-----------|--------------------------|---------------------------|-----------------------------------|-----------------------------------|
| turn 1   | 100.0%    | 100.0%    | 100.0%    | 100.0%                   | 100.0%                    | 100.0%                            | 100.0%                            |
| turn 2   | 100.0%    | 100.0%    | 100.0%    | 100.0%                   | 100.0%                    | 100.0%                            | 100.0%                            |
| turn 3   | 100.0%    | 100.0%    | 100.0%    | 100.0%                   | 100.0%                    | 100.0%                            | 100.0%                            |
| ~1/3     | 100.0%    | 100.0%    | 100.0%    | 100.0%                   | 100.0%                    | 100.0%                            | 100.0%                            |
| ~2/3     | 100.0%    | 100.0%    | 100.0%    | 100.0%                   | 100.0%                    | 100.0%                            | 100.0%                            |
| full game| 100.0%    | 100.0%    | 100.0%    | 100.0%                   | 100.0%                    | 100.0%                            | 100.0%                            |
| overall  | 100.0%    | 100.0%    | 100.0%    | 100.0%                   | 100.0%                    | 100.0%                            | 100.0%                            |

A game can end in the middle of an action: a player loses (LP 0, or a draw from an empty deck)
while the action's costs are paid, before the message that identifies its card (e.g. a Tribute
Set interrupted by Amorphous Persona's draw from an empty deck). The stream then shows only the
costs, which depend on a card that is never shown. When the full-game reconstruction diverges
only after the viewer's last decision in a game that ended with a win, the check is repeated up
to that decision, and the reconstruction counts as consistent ("unfinished end ignored", reported
separately by `check_belief.py`; 5 of the ~79,000 reconstructions in the table and the book).

Fresh seeds are the honest measure: the corpus has ~164k decks, and new decks bring new card
mechanics (the earlier ranges started between 99.3% and 99.4% when they were fresh, 7000-7999 at
99.79%). About 97-98% of all reconstructions need no repair restart at all. The latest fresh
test (seeds never used in development: 10000-13999 with all deck and agent variants, all 381
book openings twice with new shuffles, 300 `belief_mc` games; ~50,000 reconstructions) started
at 99.92-100% and is at 99.98-100% after the fixes it prompted; what remains is the class below
(hidden deck contents with a specific property) and one 93-turn game over the time limit.

For broad tests, `scripts/make_book.py` builds an opening book whose 762 decks (381 openings)
together contain every card of the usable corpus decks (13,869); `--book build/book.jsonl` plays
opening seed % 381. Three passes (1143 games) show ~90% of the cards in a game and activate
~55% (`scripts/card_coverage.py`; many cards never chain, e.g. Normal Monsters).

Median reconstruction time is about 0.3-0.6s (a few long games take up to the 90s limit because
of repeated repair restarts). Plain self-play is fast: about 0.2s CPU per game, ~650 decisions per game.

First match of the example search agent (`belief_mc`: belief replay + flat Monte Carlo with
greedy rollouts, vanilla deck completion) against `greedy`, 40 openings with swapped seats:
+52 =0 -28, Elo +107 +/- 67.

Open design questions (from the remaining failures):

* **Hidden deck contents with a specific property.** Forcing hidden cards to match a filter
  covers "the deck holds a fitting card", but not checks that compare values of several cards,
  e.g. two Pendulum Scales that differ (Solfachord Happiness) or an Xyz monster a given number of
  Ranks higher in the Extra Deck (Rank-Up-Magic), never revealed. `belief/script_index.py` reads
  from each script which archetypes and cards it refers to (`listed_series`, `listed_names`),
  which piles it names and whether it compares values (`GetClassCount`, `GetMaxGroup`, ...); the
  replay then places such cards (one per distinct Scale/Level/Rank, the opponent's seen cards
  first) in hidden slots and lets the engine decide. Open: cards whose scripts list nothing.

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
player now sees its own cards when it reveals them from its deck. Two more told physical copies
apart: the status bits of cards that Exchange of the Spirit swaps from the graveyard into the
deck (the core's swap bypassed the reset for cards entering a hidden pile), and the chain view's
pointer to a link's card after that card left its place (e.g. Morphing Jar #2 shuffled into
the deck and a copy summoned back: the view showed whether it was the same physical card). A
chain link now shows its card only while the card is still the activated object (the core's
relation between card and chain link, `OCG_ExtChainRelated`). And the timing of hand and deck
shuffles: inside an operation they are done when it ends, unless an adjustment runs in
between, which hidden cards can cause (Exodia the Forbidden One in the hand checks for a win
whenever any card is added to a hand, so the opponent's search shuffled before its reveal
instead of after it); the patched core leaves them to the end of the operation. The order in
which continuous effects apply: an effect with a condition got a new id (its "timestamp")
whenever its condition was evaluated and found true again, but conditions are evaluated lazily,
whenever something asks (e.g. a hidden card's activation check reading Levels), so a Level that
"becomes 2" before or after a "+2 in the Main Phase" depended on the opponent's hidden cards. The
patched core checks the conditions at fixed points instead (every adjustment, in id order, see
`field::refresh_effect_conditions`); only these checks give new timestamps, as the official rules
do when an effect starts applying anew (`tests/test_core_rules.py`). And conversely, a card kept revealed in the hand now
stays visible through hand shuffles (the shuffle message marks face-up cards), as it would at the
table.

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
filters in a canonical form (and ANNOUNCE_NUMBER options sorted). One more place showed it: the
cards that left the field together had their own "when this card leaves the field" effects
handled in id order (two Double Hooking leaving at once destroyed their equipped monsters in
that order); these single events are raised in location order now too.

## Changes to ygopro-core

`core_ext/ocgcore.patch` (applied by `scripts/build_core.sh` to a copy of the submodule) plus
`core_ext/ocgext.cpp`:

* Hooks for forcing outcomes, all no-ops unless the `OCG_Ext*` API is used: shuffle results
  (deck, hand, extra, `Duel.ShuffleSetCard` and the zone assignment of a group SSet), coin and
  dice results, `Group.RandomSelect`,
  "treat this effect as activatable" (skips cost/target checks), "treat this card as
  matching every filter" and "treat this card as matching no filter". The forced (non-)matching
  decides the filters of card scripts only; the filters of the rule helpers in
  `proc_workaround.lua` (e.g. which cards must be released because of an extra-release effect)
  keep telling the truth.
* Extra API: reorder a pile (by code or by card id), force shuffle results by card id, swap two
  cards within a pile, read card ids per location (or of every location of both players in
  one call: `OCG_ExtGetAllCards`), clear the forced activations.
* `core_ext/lua/ygoai_overrides.lua`, loaded after `utility.lua`: script helpers with exactly
  the same results, but faster. `aux.SelectUnselectGroup`'s search visited every group of k
  cards in up to k! orders (Rainbow Dark Dragon's check of 7 DARK monsters with different
  names took seconds at every idle prompt); it now remembers per group whether the rest can
  complete it. Likewise the Fusion material search for "N materials + X or more"
  (`proc_fusion.lua`), which reached each state in every order both when extending the chosen
  materials and when assigning selected ones to the material slots (Frightfur monsters with
  many Fluffals ran into the Lua budget, 22 s per check); each state is answered once per
  check, with every intermediate group still checked along every order as before.
  `YGOAI_NO_OVERRIDES=1` loads the original helpers: 780 games compared (seeds 1000-1099,
  5000-5099, 6000-6099, 6900-6999 and the whole opening book) are identical, except one game
  from the point where the original ran into the budget.
* `core_ext/lua/cards/`: corrected copies of card scripts with bugs, used instead of the
  CardScripts originals (e.g. Oracle of the Herald imposed a chain limit when its activation
  was only checked, so a set copy kept the opponent from responding to unrelated events).
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
* Observations: a card returned from the hand to the Deck or Extra Deck as a cost or material
  is shown to the opponent (it has to meet a requirement; the core moves it face-down).
* A card an effect Sets (`Duel.SSet`, e.g. from the Deck) is confirmed to the other player. The
  core named the player who Sets it as the one who looks (pointless); the EDOPro server makes
  up for it by showing every confirmation of cards outside the Deck to both players, which
  would also show a player's private look at its own face-down card to the opponent.
* Queries do not change the duel. Evaluating an effect's condition gave the effect a new
  timestamp when the condition had become true (`effect::is_available`), and the timestamps
  order continuous effects; a state query (e.g. of a card's level) evaluated conditions at other
  moments than the game itself, so observing the state could change what happened next (about
  1 in 25 games with the upstream core diverged when every prompt queried levels). Now only the
  checks at fixed points give new timestamps (see the information leaks above).
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
  agents/        simple AIs (random, greedy, setter, mixed)  belief/       tracker, replay, inference
```
