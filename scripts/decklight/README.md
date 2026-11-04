# Matches against decklight

decklight (github.com/chronchi/decklight, AGPL-3.0) is a learned Yu-Gi-Oh! bot: a transformer policy, a
value net and a search over sampled worlds. These scripts play our bot against it.

## Setup

1. Clone decklight into `external/decklight/` (git-ignored) with its submodules.
2. In `external/decklight/`:
   - `uv sync --group dev --group ml`
   - `scripts/build_core.sh` and `scripts/build_ext.sh` (its C++ core)
   - `bash scripts/fetch_release.sh` (the models of release v0.2.0, sha256-checked)
3. Apply `decklight-local.patch` (`git apply`). It contains:
   - the submodule `third_party/ygopro-core-rust` at our engine's commit;
   - `YGO_CORE=rust` also for the host (not only for the search);
   - protocol fixes for our clients (the short HostInfo of ygo-windbot, MSG_START as the duel start,
     a sort of one card, deck confirmations to the controller, a cost card shown);
   - a fix in `StreamView.take` (`self.state._demote_deck()`).
4. Build decklight on our engine: `scripts/build_rust_core.sh`, then `YGO_CORE=rust scripts/build_ext.sh`.
5. Build `ygo-edopro-client` (ygopro-core-rust, branch `edopro-client`).
6. The scripts use `runs/decklight/` for data and results: copy `client_WindBot` to
   `runs/decklight/client/WindBot`, and the gauntlet decks (`external/decklight/decks/gauntlet/*.ydk`)
   to `runs/decklight/client/decks/`. The held-out decks go there as `ho-NAME.ydk`, the new-card decks
   as `nc-NAME.ydk`.

## Matches

| Script | What |
|---|---|
| `run_raw.sh` | decklight's raw policy (no search) as a WindBot-like client of our arbiter (ygo-windbot + `client_WindBot`) |
| `run_search.sh N` | decklight with its search hosts the duel (`edopro_host.py --opponent product`, on our engine); our bot joins with `ygo-edopro-client --native-core`; N duels at once |
| `run_search_fullhint.sh N` | the same, our bot knows the opponent's whole deck (`BOT_HINT_MIRROR=all`, branch `windbot-hint`) |
| `run_search_holdout.sh N` | decklight's 16 held-out deck lists (`holdout_mirror.jsonl`) |
| `run_search_newcards.sh N` | decklight's 14 new-card decks (`newcards_mirror.jsonl`); our eval without those cards (`zero_newcards.py`) |

The books (`*_mirror.jsonl`) hold mirror matches: 5 openings per deck, each in both seat orders.
`newcard_codes.json`: decklight's `newcard_codes()` (the non-staple cards of its new-card decks).
`equiv.py`: decklight's raw policy on its C++ core and on our engine, the same deals in lockstep (the check
that our engine is a fair stage for it). `python3 scripts/live.py` shows the running matches.

## Results (2026-11-04, our bot: main ced2a3f9 + the 8-card hint, 5,000 nodes)

| Opponent | Decks | Our bot (Elo, 95%) |
|---|---|---|
| raw policy | gauntlet (31, trained on) | +130 [+93, +170] |
| search | gauntlet | −4.5 [−40, +31] |
| search, our bot knows the deck list | gauntlet | +42.8 [+7, +80] |
| search | held-out (16) | +102.8 [+56, +153] |
| search, our table without its new cards | new cards (14) | +190.8 [+128, +268] |
