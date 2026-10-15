# Which decisions the alpha-beta search makes

The search (`crates/ygoai/src/search/`) branches only on some of the core's prompts. Every
other prompt along a line is answered by the rules' player, the heuristic in
`crates/ygoai/src/heuristic.rs`, inside the same ply. In the search a move like "Tribute Summon
X" is then one branch, and the heuristic picks the tributes, the position and the zone.

Two lists say what is searched:

- `is_searched` / `is_inner` (which prompts become search nodes);
- `candidates_in` (which answers to such a prompt the search tries).

Prompts with only one candidate are answered without a node.

State as of commit 3886146 (2026-10-13). The default features are `DEFAULT_FEATURES = "lmr,predict"`.
The decisions inside a move (`unselect`, `tribute`, `multicard`, `sum`, `position`, `place`)
are off by default and under SPRT; see the second column.

## Overview

| Prompt | Searched by default | With the feature | Otherwise the heuristic's answer |
|---|---|---|---|
| `SelectIdlecmd` (summon, set, activate, reposition, to BP/EP, shuffle) | yes, all options | – | – |
| `SelectBattlecmd` (attack, activate, to M2/EP) | yes, all options | – | – |
| `SelectChain` (respond with an effect) | yes, pass + every effect | – | – |
| `SelectEffectyn`, `SelectYesno` | yes, both | – | – |
| `SelectOption` | yes, every option | – | – |
| `SelectCard`, exactly 1 card | yes, the 3 best cards (`CARD_WIDTH`) | – | best card by the select hint |
| `SelectCard`, several cards (min/max ≠ 1) | **no** | `multicard` | the first max(min, 1) cards by the hint |
| `SelectUnselectCard` (card by card: most Link/Xyz materials, many modern effects) | **no** | `unselect` | the best card by the hint, finish only when none is left |
| `SelectTribute` | **no** | `tribute` | the least valuable monsters until enough is released |
| `SelectSum` (Synchro, Ritual and other level sums) | **no** | `sum` | a random valid selection |
| `SelectPosition` | **no** | `position` | face-up Attack (face-up Defense if ATK ≤ 0) |
| `SelectPlace`, 1 zone | **no** | `place` | the first free zone |
| `SelectPlace`, several zones | **no** | – | the first free zones |
| `SelectDisfield` (zones to disable) | **no** | – | the first free zones |
| `SelectCounter` (remove counters) | **no** | – | random valid distribution |
| `SortCard` (order cards, e.g. the deck top) | **no** | – | as listed |
| `SortChain` (order simultaneous triggers) | **no** | – | as listed |
| `AnnounceNumber` | **no** | – | the first value |
| `AnnounceCard` (declare a card name) | **no** | – | random declarable card |
| `AnnounceRace`, `AnnounceAttrib` | **no** | – | random |
| `RockPaperScissors` | **no** | – | random |
| `SelectChain` with no effect to activate | (not a decision) | – | pass, automatically |

Coin tosses and dice are random in the core and are not decisions.

## Searched, but not every answer

Even where the search branches, it does not try every legal answer:

- **Per node:** at most 12 children after ordering (`MAX_CHILDREN`). Options past the 12th are
  never searched (e.g. a big idle command list).
- **At the root:** at most 16 candidates (`MAX_CANDIDATES` in `belief_agents.rs`).
- **One-card selections:** only the 3 best cards by the heuristic's order for the select hint
  (`CARD_WIDTH`, `ROOT_CARD_WIDTH`). The other cards are never chosen by the search.
- **Cancel** is never a candidate in any selection (`SelectCard`, `SelectTribute`,
  `SelectUnselectCard`), and neither is unselecting a card already selected.

With the inner-decision features on, their candidates are capped too, so that they stay the
heuristic's first choices:

| Feature | Candidates | Not considered |
|---|---|---|
| `multicard` | combinations of the 4 best cards (`MULTI_CARDS`); the heuristic's count first, then larger counts; at most 6 (`MULTI_WIDTH`) | cards beyond the 4 best; smaller counts than the heuristic's; most combinations once 6 are listed |
| `unselect` | the 3 best cards (`UNSELECT_WIDTH`) and finish (if allowed) | the other cards; unselecting; cancel |
| `tribute` | the 4 least valuable sets (`TRIBUTE_WIDTH`) that release enough with no card to spare | sets with a spare card; more valuable sets; cards past the 10th |
| `sum` | the first 4 valid sets (`SUM_WIDTH`) found by a depth-first search over the cards from least valuable up, within 5000 steps (`SUM_STEPS`) | sets found later; a valid set plus more cards on top |
| `position` | every allowed position | – |
| `place` | every free zone (only when 1 zone is chosen) | choices of several zones |

## At the horizon

At depth ≤ 0, a decision inside a move is always the heuristic's, even with its feature on.
That applies to card selections and the inner decisions above, so the move (or effect)
completes before the static eval. The other searched prompts get the static eval at the
horizon. With `qsearch` (off by default) battle commands with an attack and chain responses
are searched up to 6 plies further.

## The biggest gaps, roughly by expected impact

1. **Materials:** card-by-card selections (`unselect`) and sums (`sum`) decide what goes to
   the GY and what stays on the field. They drive most modern combos.
2. **Tributes** (`tribute`).
3. **Several-card selections** (`multicard`), e.g. "send 2", "add 2", "destroy up to 3".
4. **Zones** (`place`, `SelectDisfield`): Link arrows, the Extra Monster Zones, column
   effects, zones to disable.
5. **Position** (`position`): face-down Defense versus face-up.
6. **Declarations** (`AnnounceCard`, `AnnounceRace`, `AnnounceAttrib`, `AnnounceNumber`): a
   random declared card name is often a wasted effect.
7. **Orders and counters** (`SortCard`, `SortChain`, `SelectCounter`).

The goal is that every decision is represented, with move ordering, LMR and other reductions
keeping the tree thin enough that the unimportant choices cost little.
