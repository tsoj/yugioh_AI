# TODO

Planned work, roughly in order. SPRTs run at 30+0.3, SPRT [0, 15], with book
`runs/books/match1.jsonl`, and are read by LLR. Results go to `runs/sprt/results.jsonl`.

## Running now

- [ ] SPRT `lmr,unselect` vs `lmr` (first of the decision-feature queue below).
- [ ] Self-play data for the eval: `ygoai-datagen`, 2000 nodes per search, 9996 games, into
      `runs/data/selfplay_2000n.jsonl`.

## Search: represent every decision

See `docs/search-decisions.md` for what is and isn't searched.

- [ ] SPRT each decision feature vs `lmr` (queued in this order): `unselect`, `tribute`,
      `multicard`, `position`, `place`, then all six (`unselect,tribute,multicard,sum,position,place`).
      `sum` is too rare for an SPRT of its own.
- [ ] Make the winners default (`DEFAULT_FEATURES`).
- [ ] Search the decisions that have no feature yet:
  - [ ] `SelectDisfield` (zones to disable)
  - [ ] `SelectPlace` with several zones
  - [ ] `AnnounceCard`, `AnnounceRace`, `AnnounceAttrib`, `AnnounceNumber`
  - [ ] `SortCard`, `SortChain`
  - [ ] `SelectCounter`
  - [ ] cancel in selections, and unselecting in `SelectUnselectCard`
- [ ] Widen the capped candidate lists once ordering and reductions can take it:
      `CARD_WIDTH`, `MAX_CHILDREN`, `MULTI_CARDS`, `TRIBUTE_WIDTH`, `SUM_WIDTH`, `UNSELECT_WIDTH`.
- [ ] Better move ordering for the new decisions, e.g. history by decision type and card. The
      goal is that LMR and other reductions thin out the unimportant choices.

## Search: features

Already in: iterative deepening, heuristic move ordering, TT with TT move and bounds,
butterfly-style history, hard/soft time bounds, LMR (default since f783119).

- [ ] Queued retests vs `lmr`: `lmr_noresearch` (as a replacement for LMR), `lmr,pv_ext`.
- [ ] LMR the chess way: a log formula over depth and move number, adjusted by history,
      PV/non-PV and cut nodes; with PVS, a null-window reduced search.
- [ ] Not tried yet (from the chess progression in memory): aspiration windows, null move
      pruning, killer moves, late move pruning, internal iterative reduction, improving
      heuristic, SEE-like pruning, singular extensions / multicut, double/negative extensions,
      static eval correction history.
- [ ] Time management: node-based scaling of the soft bound, best-move stability, eval stability.
- [ ] qsearch, if retried: a TT in quiescence, cheaper chain windows (pass plus the best few
      answers), pruning of hopeless attacks.
- Retested at the Rust speed and stopped near zero: PVS (H0), qsearch, futility, conthist,
  PVS+RFP, RFP by kind (stopped at 2220 games, +4.5, LLR -1.50) (all on top of LMR).

## Evaluation

- [ ] Train on the self-play data with `scripts/train_eval.py`:
  - [ ] linear on the current eval terms (retuned weights)
  - [ ] 1 hidden layer, 16 neurons, the same eval terms
  - [ ] linear and MLP on the richer features (`--features rich`)
- [ ] Compare on validation, then SPRT the best against the current eval (`YGOAI_EVAL=model.json`,
      engine built at 8bc727b or later).
- [ ] Optionally add the SPRT games' records as extra data (result only, no search values).
- [ ] More data if needed: more self-play, or a faster fixed-node setting.
- [ ] A bigger eval with more of the state as features.
- [ ] Later: the learned policy/value nets and effect-text embeddings (see memory).

## Engine and core correctness

- [ ] The 4 mismatching games against the C++ core (found 2026-10-13, older than the recent
      changes): pair17349 game 1 decision 947, pair19366 game 2 decision 1880, pair8112 game 0
      decision 89, pair9854 game 1 decision 467 (`build/regress_pairs/`).
- [ ] Last Turn activated during an attack: its forced battle leaves the outer attack without an
      attacker, and the core panics (C++ dereferences null). This is an upstream bug. The search now
      avoids such lines (9c3af1d); decide whether to guard it in the core too.
- [ ] Seed the engine processes in matches (pass the seed to `ygoai-engine`), so a crashed
      game can be reproduced exactly.

## Housekeeping

- [ ] Repo cleanup: what is still needed of `ygoai/` (Python), `ygoai_rs/` and the other folders.
- [ ] Remove the worktrees `wt-old` (main repo) and `wt-old-rust` (rust repo) in the scratchpad.
- [ ] Push the rust submodule commits 2b7f988, 1354870, 935783d, 658b359 (ask first).
