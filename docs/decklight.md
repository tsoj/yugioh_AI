# decklight: local changes for matches against our alpha-beta bot

This document gives the changes that we made to decklight (github.com/chronchi/decklight,
models of release v0.2.0). With these changes, decklight can play against our alpha-beta bot.
The data is correct on 2026-11-09.

## Setup

We used two match modes:

- **Raw policy (no search).** Our arbiter is the host of the duel. decklight is the client.
  It connects with `scripts/edopro_join.py --agent raw` through `ygo-windbot`.
  `ygo-windbot` is our EDOPro server for clients that operate like WindBot.
- **Search (`--opponent product`).** decklight is the host. It uses `scripts/edopro_host.py`.
  Its search must have the true engine state, thus decklight must be the host.
  Our bot is the client. It connects as an EDOPro client.

In both modes, our Rust core (ygopro-core-rust) runs the duel and the search.
decklight uses the Rust core through its shim `libocgcore_rs.so`. The shim was already in decklight.

## Changes

One patch contains all the changes. In the code, each change has the comment
`(local patch, yugioh_AI_2: ...)`.

| File | Change | Reason |
|---|---|---|
| `third_party/ygopro-core-rust` | We changed the submodule from `3ba0a87` to `0478c44`. | `0478c44` is the commit of our engine that we used. The shim compiles without changes. |
| `src/decklight/engine/resources.py` | When `YGO_CORE=rust`, `CORE_LIB` is `build/rust/libocgcore_rs.so`. | Before the change, `YGO_CORE=rust` set only the core of the search (`cpp_env.py`). The `Engine` of the host used the C++ core. Now the host and the search use the same core. |
| `src/decklight/edopro/protocol.py` | `parse_join_game` accepts a HostInfo of 56 bytes. It sets the missing fields to zero. | `ygo-windbot` sends the older HostInfo of WindBot. This HostInfo has no deck limits. The client does not use the deck limits. |
| `src/decklight/edopro/client.py` | After the client sends the deck, `MSG_START` starts the duel. | `ygo-windbot` does not send `STOC_DUEL_START`. WindBot does not expect it. |
| `src/decklight/edopro/filter.py` | A confirmation of Deck cards or Extra Deck cards also goes to the controller of the cards. | The controller knows these cards. An example: the opponent sees an Extra Deck card of the controller. Our client expects this message. |
| `src/decklight/edopro/filter.py` | The opponent sees a card that goes from the hand to the Deck or Extra Deck as a cost or a material. | In the real game, the player reveals this card. Our bot uses this data. |
| `src/decklight/engine/inverse.py` | In `_sort_path`, a sort of one card is one pick. | **Bug.** For a sort of one card, the function gave no pick. Thus the decoding of the response failed. |
| `src/decklight/play/stream.py` | `StreamView.take` calls `self.state._demote_deck()`. | **Bug.** `StreamView` does not have the method `_demote_deck`. The state has this method. `take_auto` calls it correctly. |

The two bugs also occur without our setup.

To compile decklight after the patch, do these steps:

1. Run `scripts/build_rust_core.sh`.
2. Run `YGO_CORE=rust scripts/build_ext.sh`.

## Why the host uses the Rust core

At first, decklight was the host on the C++ core (edo9300). In approximately 10% of the duels,
our client could not reconstruct the game state. The cause is the order of the cards in a pile.

When a group of cards moves (to the GY, to banishment or to the Xyz materials), the two cores
put the cards in different orders:

- edo9300 sorts the cards by their card ID.
- Our core puts the cards of the turn player first. Then it sorts the cards by their location.

We could not convert one order into the other in all cases.

## Equivalence check: C++ core and Rust core

We did a check to make sure that the Rust core is fair to decklight. Its raw policy (forge_r1,
argmax) played the same deals on the two cores at the same time. At each decision, we compared
the player, the prompt type and the options. We used 124 deals with the gauntlet decks.

| Result | Deals |
|---|---|
| The two duels are the same. | 92 |
| The two duels have the same result. Only the order of the options or of a pile is different. | 18 |
| The order of a pile is different. Thus the duels have different results. An example: the Deck order is different, thus the player draws a different card. | 13 |
| The raw policy repeats the same actions without end on the two cores. | 1 |

We found no difference in the rules.

## Note added after the matches (2026-11-09): possible information leaks

We wrote this section after the matches. The sections above describe the setup as we used it in
the matches. We did not change them.

After the matches, we compared the two changes in `filter.py` with the official rules. Each change
can give our bot more information than the rules allow. These changes affect only the messages to
the client seat, thus only our bot can get the extra information. We did not measure the effect
on the results. We think that it is small, because only a small number of cards cause these cases.

### Confirmation of Deck cards to the controller

This change is correct for Extra Deck cards. The owner can look at its Extra Deck at all times.

This change is not always correct for Main Deck cards. The core sends the same message in two
cases:

- The owner reveals its own Main Deck card to the opponent. The owner knows the card, because it
  selected the card. In this case, the change is correct.
- The opponent looks at Main Deck cards of the owner (for example, the top card of the Deck).
  In the real game, the owner does not see these cards. In this case, the change gave our bot the
  codes and the positions of its own Deck cards. **This is a leak.**

The message does not tell the two cases apart.

Correction: our arbiter now hides the codes of the own Main Deck cards in this message
(ygopro-core-rust `510b8c5`). Our EDOPro client applies the arbiter's filter to the messages from
the host. When the client is built on this commit, the leak is closed for our bot, also with the
patch unchanged.

### A card from the hand to the Deck as a cost or material

This change is correct when the selection has a requirement. Examples:

- Materials: the player must show that the summon is legal.
- A cost with a requirement, for example "shuffle 1 Spellcaster monster from your hand into the
  Deck": the player must show that the card meets the requirement.

This change is possibly not correct for a cost without a requirement, for example "shuffle 1 card
from your hand into the Deck". We think that the player does not reveal the card in this case,
but we did not find an official ruling. The change shows the card to the opponent in all cases.
**This is a possible leak.**

Status: not corrected. We examine a check that finds if a selection has a requirement.
