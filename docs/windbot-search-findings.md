# Search failure modes from WindBot matches

Investigation: 2026-11-08. Current search baseline: `fbe5850d`, OpenBench bench
signature **186381**. The recorded matches used earlier versions; probes explicitly
labelled current main below were rebuilt on `fbe5850d`.

## How to interpret the evidence

These are diagnostic positions and hypotheses, not Elo results for proposed patches.
Earlier failed experiments are evidence about their implementations and settings,
not final conclusions about the techniques. A winning WindBot takeover changes many
decisions: its first difference is not necessarily the original losing move.

`bot-analyze --exact` gives full-window root searches, but the search remains selective
and width-capped. It uses the true duel, fresh history, and one shared node budget
for the root candidates. A displayed best move can come from a partial iteration,
while the printed values are from the last completed iteration. `--bot` reconstructs
hidden cards, but its seed and earlier 100-node searches need not reproduce the
recorded bot's history. None of these is a game-theoretically exact reference.

Recorded response indices are in the player's canonical option order. Raw replay
prompts use the core's order: use `to_core_response` (or replay events' core response)
before interpreting indices. Labels are indexed by their `decision` field, not by
their position in the labels array. Generated `--line` records retain old labels;
those labels do not describe the new continuation.

## Match results

`runs/windbot/hint_main_x2/n5000.jsonl`: 640 games, mirrors of eight selected weak
decks, 5,000 nodes, eight opponent-deck cards supplied as a hint. Its run script
identifies main `ced2a3f9` plus the hint patch. Overall **268–2–370**,
**-55.8 +/- 24.0 Elo**, zero reported match errors.

| Deck | Alpha-beta wins | Draws | Losses |
|---|---:|---:|---:|
| Exosister | 26 | 0 | 54 |
| Brave | 28 | 0 | 52 |
| Dogmatika | 30 | 0 | 50 |
| Trickstar | 33 | 0 | 47 |
| Witchcraft | 36 | 0 | 44 |
| ABC | 37 | 0 | 43 |
| ChainBurn | 38 | 2 | 40 |
| Blue-Eyes | 40 | 0 | 40 |

This is a selected weak-deck set. The older, broader
`runs/windbot/fix/mirror.jsonl` run was **581–0–419**, +56.8 +/- 21.3 Elo.
The newer no-hint `main_ced2a3f9/n5000.jsonl` stopped after 84 games and is incomplete.

The older node-scaling runs (`runs/windbot/nodes/run.sh`: `ae98c959` plus the
timing-hint fix, same 160 openings) were:

| Nodes | Wins | Draws | Losses |
|---|---:|---:|---:|
| 5,000 | 115 | 1 | 204 |
| 20,000 | 116 | 3 | 201 |

This sample does not establish that depth cannot help, but motivates investigating
candidate omissions, horizon effects, and evaluation rather than just more nodes.

## 1. ABC: an excluded candidate blocks a winning combo

Source: `runs/windbot/hint/n5000.jsonl`, zero-based game **14**, decision **31**.
Union Hangar chooses a monster to equip. There are eight card instances, but just
four codes: A-Assault Core, B-Buster Drake, Union Driver, Heavy Mech Support Armor.
The default five searched choices contain only copies of A and B. Union Driver,
canonical response `{"indices":[1]}`, is excluded.

Current-main true-state probes:

| Width / nodes | Result |
|---|---|
| 5 / 5,000 | Driver unsearched; best completed value about -237 LP |
| 8 / 5,000 | Driver selected, +3,962 LP |
| 8 / 50,000 | Driver selected, +4,103 LP |

Controlled continuations on current main, 5,000 nodes, same prefix and seeds,
alpha-beta versus WindBot:

- Force Driver (`{"indices":[1]}`): alpha-beta wins on turn 3, 360 decisions.
- Force the original B-Buster Drake (`{"indices":[3]}`): loses on turn 4, 357 decisions.

Records: `runs/windbot/abc-main-driver.jsonl`, `abc-main-buster.jsonl`.
This is one recovered loss, not a strength test. WindBot's
`ABCExecutor.cs::UnionHangarEquip` explicitly prioritizes Driver on its own turn;
Driver enables a further equip and the Union summon sequence.

### Hypothesis: diversity before the width cap

Prioritize one representative per code/controller/location in deck selections,
then retain duplicate instances as fallbacks. This spends the existing width on
more distinct choices without assuming instances are truly interchangeable.
Start with deck selections: field cards, materials, effects and per-instance
history can make copies meaningfully different. The policy's first choice can
remain first, including for horizon responses.

Audit of the newer 80-game-per-deck run (counts are raw option groups, not proven
equivalence classes or policy top-five coverage):

| Deck | One-card selections with >5 choices | Of those, <=5 code/controller/location groups |
|---|---:|---:|
| ABC | 307 | 209 |
| Exosister | 257 | 114 |
| Dogmatika | 440 | 119 |
| Brave | 81 | 19 |

## 2. The combo construction gap

Special-summon milestones in the newer mirror run, games containing the summon:

| Deck / monster | Alpha-beta | WindBot |
|---|---:|---:|
| ABC / ABC-Dragon Buster | 39 | 58 |
| ABC / Cyber Dragon Infinity | 9 | 35 |
| Brave / Borreload Savage Dragon | 6 | 35 |
| Brave / Thunder Dragon Colossus | 10 | 45 |
| Exosister / Exosisters Magnifica | 23 | 63 |

In ABC game 14's recorded first turn, alpha-beta ends with C-Crush Wyvern and an
equipped B-Buster Drake. WindBot's winning takeover builds ABC-Dragon Buster,
Cyber Dragon Infinity and Union Carrier. In Brave game 112, alpha-beta builds a
plausible board, but WindBot follows a much longer Halqifibrax/Auroradon sequence
into Savage Dragon, Colossus and Destroyer Phoenix Enforcer.

These counts alone do not establish optimal play. However, the stronger monsters
often never become available to alpha-beta: Infinity was offered in only nine
ABC games, Colossus in eleven Brave games. Intermediate decisions deserve attention.

The search counts selected prompts as plies; players need not alternate. Some
material decisions are still heuristic below the root. A combo can pay several
decision plies and lose material before its benefit appears. Candidate omissions
and action completion should be measured separately.

## 3. Exosister: a selective-search preference, not a proven lost win

Source: hint game **192**, decision **83**. Magnifica can be summoned, and Returnia
is in hand, but the recorded bot ends its turn. Current main first sets Returnia.
Following that choice produces decision **87** in
`runs/windbot/exosister-main-line.jsonl`:

| Current-main true-state search | Best Magnifica value | End turn |
|---|---:|---:|
| 5,000 nodes | +4,020 | +4,507 |
| 50,000 nodes | +3,770 | +4,071 |
| 50,000, futility and LMR disabled | +5,019 | +4,625 |

Disabling either alone did not reverse the preference in these probes. Both
forced continuations won against WindBot with current main: Magnifica on turn 3
(219 decisions), passing on turn 5 (512). Records:
`exosister-main-magnifica.jsonl`, `exosister-main-pass.jsonl`.

Across the newer Exosister games Magnifica was offered in 40 games, but never
summoned in 17. This is a useful family for tracing reductions, pruning and leaf
evaluations. It does not justify a card-specific summon bonus or conclude that
the evaluation alone is wrong.

## 4. Dogmatika: the set policy cannot rank a missing set

Source: hint game **174**, decision **250**, choose four cards to send away.
All five generated candidates include Fleurdelis from hand. WindBot preserves it
and sends four Extra Deck cards (`{"indices":[1,2,7,3]}`), a missing candidate.
`card_sets` currently lists sets close to the first ranked cards; the set policy
only reorders those sets. More search cannot recover a set that was not generated.

Forcing WindBot's set alone did not save the game on current main: both tested
continuations lost on turn 6 (`dogmatika-main-wind-set.jsonl` and
`dogmatika-main-ab-set.jsonl`). This establishes an option-coverage limitation,
not that this one substitution is sufficient to win.

Hypothesis: keep the original best set, but reserve candidate slots for best
sets excluding each of its cards. This tests whether the near-neighbour pool
overcommits to its first cards, without a wholesale policy-generated pool.

## Takeovers and experimental priorities

`runs/windbot/takeover/points_wind.jsonl` has 1,287 completed turn-boundary
playouts of 192 original losses. WindBot recovered 98 at some boundary.
`decisive.jsonl`, `compare.json`, `diverge_5k.jsonl`, and `diverge_50k.jsonl`
contain the continuations and comparisons. Of 92 first divergences analysed at
5,000 nodes: WindBot's choice scored higher in 27, alpha-beta's in 39, 24 tied,
and two were missing/unsearched. The 50,000-node subset was selected from the
unequal numeric comparisons, not a fresh unbiased sample.

Priority experiments:

1. Deck-selection diversity before single-card caps (strongest direct evidence).
2. Small, diversified multi-card pools, preserving the original first candidate.
3. A bounded action-completion extension for decisions during an already resolving
   effect, rather than extending complete turns or all chain response windows.
4. Instrument combo lines to separate candidate coverage, heuristic materials,
   pruning, reductions, hidden-card guesses and static evaluation.

Run each patch separately against the same main in OpenBenchYGO. Passing these
positions is diagnostic only; the final strength test should remain on the broad
book. Historical PVS, set-pool, inner-decision and horizon experiments in `TODO.md`
should guide the scope, not veto targeted retries.

## Reproduction

### Implemented experiment: `search-card-diversity`

In one-card selections entirely from the deciding player's main deck, the policy's
first instance of each nonzero code precedes repeated copies. All instances remain
in the list; root width 5 and tree width 3 are unchanged. The first policy choice
and hence the horizon answer remain unchanged. Other locations and mixed-location
selections retain their order.

Release library tests pass. Bench **180640** (base 186381); 3,000-node mean depth
**13.02** (base 12.97), zero broken lines. The ABC game 14 / decision 31 probe now
selects Union Driver at **+3,962 LP**, at the default width and 5,000 nodes.
Strength remains untested.

The records and local helper scripts live in ignored `runs/windbot/`; they are
not distributed with this document. The analyzer source is tracked at
`crates/bot/src/bin/bot-analyze.rs`.

```sh
CARGO_TARGET_DIR=ygopro-core-rust/target cargo build --release -p bot --bin bot-analyze
ygopro-core-rust/target/release/bot-analyze runs/windbot/hint/n5000.jsonl 14 31 --nodes 5000 --exact
BOT_TUNE=root_card_width=8 ygopro-core-rust/target/release/bot-analyze runs/windbot/hint/n5000.jsonl 14 31 --nodes 5000 --exact

# Substitute indices [3] for the control; output is appended if the file exists.
BOT_ANALYZE_WINDBOT="$PWD/external/windbot-src/build" \
BOT_ANALYZE_SEATS=ab,wind BOT_ANALYZE_THREADS=1 \
BOT_ANALYZE_RECORDS=runs/windbot/abc-driver-recheck.jsonl \
ygopro-core-rust/target/release/bot-analyze runs/windbot/hint/n5000.jsonl 14 31 \
  --nodes 5000 --playout 1 --force '{"indices":[1]}'
```
