# Yu-Gi-Oh! bot

A bot that plays Yu-Gi-Oh! duels with hidden information, and the tools to develop and test it.

The duels run on [ygopro-core-rust](https://github.com/tsoj/ygopro-core-rust) (submodule
`ygopro-core-rust/`): a Rust port of [ygopro-core](https://github.com/edo9300/ygopro-core), the
EDOPro rules engine, and of the card scripts of
[ProjectIgnis/CardScripts](https://github.com/ProjectIgnis/CardScripts). The card database is
[ProjectIgnis/BabelCDB](https://github.com/ProjectIgnis/BabelCDB).

## Parts

| Part | Where | What it does |
|---|---|---|
| bot | `crates/bot` | Plays a duel: it reconstructs the hidden cards, then searches. |
| arbiter | `ygopro-core-rust/ygo-arbiter` | Runs matches between bots, like [fastchess](https://github.com/Disservin/fastchess) for chess engines. Gives each bot only what its player may see. Reports W/D/L, Elo and SPRT results. |
| bot protocol | `ygopro-core-rust/ygo-arbiter/PROTOCOL.md` | How the arbiter and a bot communicate (JSON lines on stdin/stdout). Any program that speaks it can play. |
| belief replay | `crates/bot/src/belief` | Builds a complete duel from one player's observations. See [docs/belief-replay.md](docs/belief-replay.md). |
| engine | `ygopro-core-rust` | Applies the rules. It is tested message by message against the C++ core. |

## How the bot plays

At each decision, the bot (`bot alphabeta`) does three steps:

1. **Reconstruct.** The belief replay builds a duel that agrees with everything the player saw.
   A deck completion model fills the opponent's unknown cards (`crates/deckmodels`, trained on
   human deck lists).
2. **Search.** Alpha-beta search on copies of that duel: iterative deepening, a transposition
   table, a history heuristic, futility pruning, late move reductions and node-based time
   management. The search branches only on some decisions; a rule-based player answers the others
   ([docs/search-decisions.md](docs/search-decisions.md)). In card selections, trained policies
   order the candidates: single cards, or sets of cards such as materials and sums. The search
   tries the first few.
3. **Evaluate.** A small trained network evaluates positions in Life Points, from features of the
   position.

`bot NAME [SEED]` runs one strategy as a bot. NAME is one of:

| NAME | Strategy |
|---|---|
| `alphabeta` | The bot described above. |
| `alphabeta_naive` | The same search, but the opponent's hidden cards are guessed from the visible cards only. |
| `mcts`, `belief_mc` | Monte Carlo searches on the reconstructed duel. |
| `heuristic` | The rule-based player. |
| `greedy`, `setter`, `mixed`, `random` | Simple players for tests. |

Environment variables `BOT_*` change settings, for example `BOT_SEARCH_NODES` (a node limit per
decision) and `BOT_COMPLETION` (`vanilla`, `corpus` or `neural`, the default).

## Setup

Requirements: Rust (stable), cmake and a C++ compiler (for HiGHS), and [uv](https://docs.astral.sh/uv/).

```sh
git submodule update --init external/CardScripts external/BabelCDB external/HiGHS ygopro-core-rust

# Python for the scripts (torch only for training)
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python numpy==2.4.6 scipy==1.18.0 tqdm maturin
uv pip install --python .venv/bin/python "torch==2.13.0+rocm7.1" --index-url https://download.pytorch.org/whl/rocm7.1

# the bot and its tools (one target directory for both workspaces)
CARGO_TARGET_DIR=ygopro-core-rust/target cargo build --release -p bot
# the arbiter and the reference bot ygo-random-bot
cargo build --release --manifest-path ygopro-core-rust/Cargo.toml -p ygo-arbiter
# the Python binding of the bot (for the belief checks and other scripts)
cd crates/bot-py-binding && CARGO_TARGET_DIR=../../ygopro-core-rust/target VIRTUAL_ENV=../../.venv ../../.venv/bin/maturin develop --release
```

Data that is not in git:

- The human deck lists, in `human_decks/decks/` and `human_decks/decks_new/`.
  `human_decks/README.md` tells how to download them and train the deck models.
- The deck models' export, `human_decks/onnx_export/`. The build compiles it into the bot. Without
  it, the bot builds without the `neural` completion; then set `BOT_COMPLETION=corpus`.
- The index of the `corpus` completion, `build/deck_index.npz`: `scripts/build_deck_index.py`.

## Development

A change to the bot is a branch. An SPRT tests it against `main`, and it is merged if H1 is accepted:

```sh
scripts/sprt_branch.sh my-branch main
```

The script builds both commits into `build/bots/<sha>/` and runs the arbiter with the opening book
`runs/books/match2.jsonl`, time control 10+0.1 and Elo bounds [0, 5]. The log and the records go
to `runs/sprt/`, a summary line to `runs/sprt/results.jsonl`. `NODES=N` gives fixed-node searches
instead; the script's header lists the other settings.

A match between any two bots:

```sh
ygopro-core-rust/target/release/ygo-arbiter --bot "new=ygopro-core-rust/target/release/bot alphabeta" \
    --bot "base=ygopro-core-rust/target/release/ygo-random-bot" --book runs/books/match2.jsonl -n 200 --tc 10+0.1
```

Other checks:

| Check | Command | When |
|---|---|---|
| Search behavior | `bot-bench-search runs/books/match1.jsonl 6 10d` and `... 3000n`: the checksums must not change | after a refactor of the search |
| Belief replay | `.venv/bin/python scripts/suite.py --regression TAG` (see [docs/belief-replay.md](docs/belief-replay.md)) | after a change to the belief replay or the arbiter |
| Engine | differential tests against the C++ core (`ygopro-core-rust/README.md`); `ygopro-core-rust/tools/replay_records.py` replays SPRT records on both engines | after a change to the engine |
| Unit tests | `.venv/bin/python -m unittest discover tests`; `cargo test --release -p bot`; `cargo test --release --manifest-path ygopro-core-rust/Cargo.toml -p ygo-arbiter` | always |

Long runs (the suite, SPRTs) work from snapshots of the code (`scripts/snapshot.py`), so a rebuild
during a run does not change it.

## Models

The bot's models are in `crates/bot/models/` and compiled in. To make them again:

- The evaluation `eval_mover2_sprt2r23os_64.json`: SPRT records → `scripts/collect_sprt_games.py`
  → `bot-refeature --features mover2` → `scripts/sprt_games_to_npz.py` (with rounds 2 and 3:
  `bot-datagen` → `scripts/eval_mover_features.py`, in `runs/data/rounds23/`) →
  `scripts/train_eval_mover.py --dims 64 --held 400 --drop-onesided --lam 0.5`.
- The card policy `card_policy_cp4_h16.json`: `bot-card-refs --independent` →
  `scripts/train_card_policy.py`.
- The set policy `set_policy_s2_h16.json`: `bot-card-refs --independent --decisions` →
  `scripts/train_set_policy.py`.
- The deck models: `human_decks/README.md`.
- The SPRT opening book `runs/books/match2.jsonl`: `scripts/book/README.md`.

## Changes to the rules engine

The engine differs from upstream ygopro-core in a few ways. A bot must not see hidden information,
and the belief replay needs games whose visible course never depends on hidden cards.
`ygopro-core-rust/README.md` lists the differences. `lua_fixes/` has corrected copies of card
scripts with bugs; the engine's cards are translated from them.

## Layout

```
crates/bot/             the bot (bin bot) and its tools (bin bot-*)
crates/bot-py-binding/  the bot's Python binding (import bot_py_binding)
crates/deckmodels/      the deck completion models (their ONNX export, compiled in)
crates/highs/           HiGHS (external/HiGHS), for the belief replay's flow model
ygopro-core-rust/       submodule: the engine (ygo-core), the cards (ygo-cards), the arbiter and the
                        bot protocol (ygo-arbiter), the differential tests (ygo-diff, oracle/)
human_decks/            the deck corpus, its download and the deck models' training
external/               submodules: CardScripts, BabelCDB, HiGHS
lua_fixes/              corrected card scripts
scripts/                command line tools; common/: their shared code; book/: the SPRT opening book
tests/                  unit tests; baselines/: the belief suite's baselines
docs/                   the belief replay; which decisions the search makes
```
