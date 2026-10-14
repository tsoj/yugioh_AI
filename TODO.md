# TODO

Planned work, roughly in order. SPRTs run at 30+0.3, SPRT [0, 15], with book
`runs/books/match1.jsonl`, and are read by LLR. Results go to `runs/sprt/results.jsonl`.

## Running now

- [ ] SPRT: `MAX_DEPTH` 100 vs 12 (both `lmr`).
- [x] Self-play data for the eval: `ygoai-datagen`, 2000 nodes per search, 9996 games, in
      `runs/data/selfplay_2000n.jsonl` (1.42M rows).

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
- [ ] **Next, before adding more decision types: budget search.** Replace depth by a node
      budget: an approximate number of nodes the subtree may use. A node costs 1 node of it, and
      it gives each child, before searching, the share it expects that child to need: the
      expected cost includes the chance of a cutoff before a move (by move number) and of a
      re-search. These probabilities come from online statistics by node type (prompt, width) and
      from the TT (an entry, its flag, a TT move). Iterative deepening grows the root budget.
      Not a hard limit; a soft allocation in expectation.
  - [x] Measure the widths (done: about 45% of the nodes are 2-way; `YGOAI_STATS` in
        `ygoai-bench-search`).
  - [ ] Measure cutoffs by move index, TT state and prompt; LMR re-search rates; subtree sizes
        by depth (`YGOAI_STATS`).
  - [ ] Implement the budget search as a feature; SPRT vs `lmr`.
  - [ ] LMR in budget terms, calibrated to the statistics: skewed shares when the node is
        confident (a TT move, a cut node by the TT flag), near uniform when only the static
        ordering is known. Adapt Stockfish's log(depth) * log(move number) reduction (in budget
        terms: shares like (i+1)^-a, with a growing with log of the budget and with confidence).
  - [ ] Retest the decision features (`unselect` etc.) on top of it (`unselect` failed with one
        ply per node: H0, -19.8 Elo, 492 games).
- [ ] `MAX_DEPTH` was 12: the depth search (`lmr`) stopped there whatever its time (mean depth
      10.8 already at 3000 nodes). Now 100, a safety bound only; SPRT vs the cap of 12.
- [ ] Try skipping "tactical" decisions at low budget: answer some decisions with the heuristic
      when the budget is small, as the horizon does now for the inner decisions. Which ones count
      as "tactical" is open: e.g. zones, positions, chain responses that do not change the eval,
      battle details.
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
