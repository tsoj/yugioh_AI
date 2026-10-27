# The belief replay

The bot sees only what its player may see. To search, it needs a complete duel. The belief replay
(`crates/bot/src/belief/`) builds one from the player's observations alone: a duel that the player
cannot tell apart from the real one. The bot then searches on copies of that duel.

The belief checks test it: bots play many games, and at several points of each game the
reconstruction must reproduce the player's observations exactly (`scripts/check_belief.py`, the
regression suite `scripts/suite.py`).

## Running the checks

The scripts need the Python of `.venv` (the binding `bot_py_binding` is installed there).

```sh
PY=.venv/bin/python
# the regression suite: all ranges against their baselines (tests/baselines/), from a snapshot of
# the code; --regression runs only the cases where regressions show
$PY scripts/suite.py --list
$PY scripts/suite.py --regression TAG [RANGE ...]

# one run: both players, cut at the end of turns 1, 2 and 3, at 1/3, 2/3 and at the end
# (turn cuts show as frac -1, -2, -3; --fractions / --turns choose)
$PY scripts/check_belief.py -n 1000 --start 1000 -o runs/check.jsonl
# other bot pairings (seed % number of pairs picks one): random, greedy, setter (Sets cards and
# responds in chain windows), mixed (greedy with 30% random)
$PY scripts/check_belief.py -n 1000 --start 4000 --pairs setter-greedy,mixed-setter -o runs/c.jsonl
# decks ordered for card variety (cached in build/diverse_decks.json)
$PY scripts/check_belief.py -n 1000 --start 5000 --decks diverse -o runs/c.jsonl
# an opening book that covers every card of the corpus
$PY scripts/make_book.py -o build/book.jsonl
$PY scripts/check_belief.py -n 1143 --start 7000 --book build/book.jsonl -o runs/c.jsonl
$PY scripts/card_coverage.py -n 1143 --start 7000 --book build/book.jsonl
# decks of random legal cards (edge cases)
$PY scripts/check_belief.py -n 1000 --start 40000 --decks cards -o runs/r.jsonl
# reconstructions as the bot makes them: every K-th decision, inherited
$PY scripts/check_belief.py -n 100 --start 50000 --decks new --completion neural --bot-sequence 3 -o runs/s.jsonl

# rerun only what failed, then compare the two runs
$PY scripts/check_belief.py --cases runs/check.jsonl --only-failed -o runs/retry.jsonl
$PY scripts/compare_runs.py runs/check.jsonl runs/retry.jsonl
# only games with certain mechanics (tags: xyz, sort_card, toss_coin, shuffle_set_card, ...)
$PY scripts/index_games.py -n 1000 --start 1000 -o runs/index.jsonl
$PY scripts/check_belief.py --index runs/index.jsonl --tags shuffle_set_card,toss_coin -o runs/t.jsonl
# one reconstruction in detail: its repairs, the divergence and the real game around it
$PY scripts/debug_belief.py SEED VIEWER [--cut 0.5]
# is a failure a regression? the case at older commits, then bisection
$PY scripts/bisect_case.py SEED VIEWER FRAC [check_belief flags...]
# leak check: does the secret card creation order show in the observations?
$PY scripts/diff_creation.py -n 1000 --start 5000
```

## How it works

The belief replay is Rust; the belief checks (`check.rs`: `check_belief.py` calls them through
`bot_py_binding`) play its test games (`games.rs`) and reconstruct each player's view at several cuts.

* `tracker.rs` follows every card slot of both players through the observation stream,
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
* `flow.rs` solves that model exactly: every slot carries one card code and every shuffle
  conserves the multiset of codes, which makes it an integer multi-commodity flow over the DAG
  of shuffles (a few thousand variables, solved by HiGHS; no search budget, no random
  restarts). Unknown opponent cards stay typed wildcards ("some spell/trap")
  until a `DeckCompletion` fills them (effect-less vanilla cards, or the corpus-based
  `CorpusCompletion` which uses the most similar human decks). Identity constraints enter the
  model where they imply something about codes (a slot has the code of the card it is known to
  be, or its code comes from a card other than the one it is known not to be); copy identity is
  then fixed by matching claims to members per shuffle (copies nothing tells apart are
  interchangeable), and the duel is built by card id (deck order and every shuffle result).
* `replay/` creates a duel from that hypothesis and replays it in lockstep with the stream,
  using the engine's extension hooks (`ygopro-core-rust/ygo-core/src/ext.rs`) to force shuffle results,
  coin/dice results and random selections. The viewer's prompts are answered with the recorded
  responses; the opponent's hidden decisions are inferred from what the viewer saw next
  (`infer.rs`), and ambiguous ones are verified by probing candidates on copies of the replay and
  the duel.
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
  restart at all. The repairs (see `replay/mod.rs`):
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

  When these repairs give up, a last resort (`read_guided.rs`): replayed without repairs, the
  engine's read log names the hidden cards the engine read just before the first divergence;
  each is tried as another copy of its name in the same place (copies differ in history: which
  one was revealed, drawn, used) or with other codes that differ in the properties it read,
  each candidate a whole replay with the usual repairs. It covers the wrong hidden card no rule
  addresses (a deck card whose ATK an effect compares, a face-down monster that keeps the
  viewer from attacking); it runs only where the replay would fail. It can also hide a problem
  that should be fixed instead (a leak that makes copies differ showed as such a case), so the
  tests count what it saves as failures ("read-guided repair needed: ..."; `check_belief.py
  --read-guided-ok` counts them as consistent).

## Using it in a bot

A bot keeps its observation stream as a list of items `("msg", message)`, `("prompt", prompt,
state)` and `("response", response)`, exactly as the bot protocol delivers them. At a
decision it reconstructs the duel and searches from it on copies (`duel.clone()`, microseconds):

```rust
let mut belief = BotBelief::new(player, init, completion::by_name("neural")?);   // one per game
...
let (r, Some(duel)) = belief.reconstruct(&items, timeout) else { ... };   // a duel waiting for this player's response
let core_prompt = r.final_prompt.unwrap_or(prompt);    // (the core's version of the prompt)
for candidate in &candidates {
    let mut copy = duel.d.clone();
    copy.ocg_set_response(&encode(&core_prompt, &to_core_response(&core_prompt, player, candidate)?)?);
    // ... simulate on, evaluate
}
```

`BotBelief` reconstructs decision by decision: the previous consistent duel goes on with the
new observations (`BeliefReplay::extend`; the viewer's own draws and hand shuffles are arranged
just in time, as a hypothesis of the longer stream would arrange them); when that runs into a
mismatch, a full replay inherits the previous one's repairs and decisions
(`BeliefReplay::inherit`); when that fails too, one from scratch. The duel it returns is a copy,
the caller's to change. When no consistent reconstruction exists, `r.approximate` is set: the
best effort re-aligned with the stream, its public state re-synchronized with what the player saw
at each of its Idle/Battle commands (`belief/sync.rs`).

`BeliefMcBot` (`belief_bots.rs`) is a complete (deliberately simple) example: flat Monte
Carlo with greedy rollouts on the reconstructed duel; `BeliefSearchBot` searches it with
alpha-beta. `scripts/check_belief.py --bot-sequence K` checks
reconstructions made this way (every K-th decision of each player; suite range
`botseq50000`), `--bot-mode` single reconstructions as a bot sees them (with the state
differences of approximate ones).

## Results

Measured with `scripts/check_belief.py -n 1000` (both players; cut at the end of turns 1, 2 and 3,
after ~1/3, ~2/3 and at the end of the game, ~9,800 reconstructions per run (a turn cut that gives
the same prefix as another one is skipped); the reconstruction must reproduce every filtered
message, every prompt of the viewer and the viewer's state view at every prompt). A run takes
about 2 minutes on 32 threads (i9-13900K).
Seeds 1000-4999 use random corpus deck pairs (20% mirror matches); 5000-6999 use the decks
ordered for card variety (`--decks diverse`). Bots: random and greedy, or (marked "other
bots") the `setter` and `mixed` bots against each other and the others (`--pairs
setter-greedy,mixed-setter,greedy-mixed,setter-random,mixed-mixed,random-setter`). By now
all of them have been used while developing (5000-6999 started out fresh, at 99.43% / 99.32%; 7000-7999
at 99.79%):

| cut      | 1000-1999 | 2000-2999 | 3000-3999 | 4000-4999 (other bots) | 5000-5999 (diverse decks) | 6000-6999 (diverse, other bots) | 7000-7999 (diverse, other bots) |
|----------|-----------|-----------|-----------|------------------------|---------------------------|---------------------------------|---------------------------------|
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
test (seeds never used in development: 10000-13999 with all deck and bot variants, all 381
book openings twice with new shuffles, 300 `belief_mc` games; ~50,000 reconstructions) started
at 99.92-100% and is at 99.98-100% after the fixes it prompted; what remains is the class below
(hidden deck contents with a specific property) and one 93-turn game over the time limit.

For broad tests, `scripts/make_book.py` builds an opening book whose 762 decks (381 openings)
together contain every card of the usable corpus decks (13,869); `--book build/book.jsonl` plays
opening seed % 381. Three passes (1143 games) show ~90% of the cards in a game and activate
~55% (`scripts/card_coverage.py`; many cards never chain, e.g. Normal Monsters).

Median reconstruction time is about 0.3-0.6s (a few long games take up to the 90s limit because
of repeated repair restarts). Plain self-play is fast: about 0.2s CPU per game, ~650 decisions per game.

First match of the example search bot (`belief_mc`: belief replay + flat Monte Carlo with
greedy rollouts, vanilla deck completion) against `greedy`, 40 openings with swapped seats:
+52 =0 -28, Elo +107 +/- 67.

Open design questions (from the remaining failures):

* **Hidden deck contents with a specific property.** Forcing hidden cards to match a filter
  covers "the deck holds a fitting card", but not checks that compare values of several cards,
  e.g. two Pendulum Scales that differ (Solfachord Happiness) or an Xyz monster a given number of
  Ranks higher in the Extra Deck (Rank-Up-Magic), never revealed. `belief/script_index.rs` reads
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
  over hidden decisions (with duel copies as checkpoints) would handle these uniformly.
* **Hidden option availability.** Some script-level choices (e.g. which effect of a card can be
  chosen) depend on hidden cards; forcing activation and forcing a card to satisfy a filter
  cover the common cases, but not options built inside a script.

## Information leaks it found

The arbiter's `observe.rs` (`ygopro-core-rust/ygo-arbiter/src/`) hides everything a player must not know, and the belief replay doubles as a
leak detector: it rebuilds the game with *wrong* hidden cards, so any hidden detail that leaks
into the observations makes the streams differ. This found several leaks that EDOPro itself has:
positions of face-down cards in piles (a card returned to the deck has a different position
value than the original deck cards), stale "reason" fields (how a card got into the hand,
visible when it is later played or reordered), card hints emitted while cards are created (they
reveal deck contents), `reason_card` pointers into hidden piles, status bits of pile cards, and
multi-card listings in card creation order.

Further leaks fixed since: empty chain windows (the core offers them when the *opponent* has a
pending optional trigger, so they are answered by the arbiter and not shown), the
ATTACK_CANCELED status bit (choosing an attacker clears it even if the attack is then cancelled,
so it revealed hidden attack attempts), status bits that a card kept from an earlier stint on the
field (they told physical copies apart; fixed in the core), and the physical copy in each deck
slot (the arbiter now places copies of a card in random order, instead of in creation order),
and the order of options in command, chain and card selection prompts (the core sorts them by
effect id, which depends on card creation order and on when effects were last enabled, or lists
cards in creation order, which tells physical copies apart; the arbiter shows them in a
canonical order, by location, and maps the chosen indices back, see `to_core_response` in
`ygopro-core-rust/ygo-arbiter/src/observe.rs`),
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
do when an effect starts applying anew (`crates/bot/tests/core_rules.rs`). And conversely, a card kept revealed in the hand now
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
