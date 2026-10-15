# TODO

Planned work, roughly in order. SPRTs run at 30+0.3, SPRT [0, 15], with book
`runs/books/match1.jsonl`, and are read by LLR. Results go to `runs/sprt/results.jsonl`.

## Running now

- [ ] Self-play data with repeated deck pairs (500 pairs x 20 games, both seat orders):
      `runs/books/pairs500x20.jsonl` into `runs/data/selfplay_pairs_2000n*.jsonl`.
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
- [x] `MAX_DEPTH` was 12: the depth search (`lmr`) stopped there whatever its time (mean depth
      10.8 already at 3000 nodes). Now 100, a safety bound only (SPRT vs 12 stopped at 1420 games:
      +2.9 +/- 13.0, LLR -0.47).
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
- [x] Predictive stopping (`predict`, default since 2026-10-14 without a pass: +3.8 +/- 11.7,
      LLR -0.35 at 1760 games): the next iteration only if it is expected to end within the
      budget. It uses about 25% more of the budget than the old rule (start while under half).
- [ ] Time management: node-based scaling of the soft bound, best-move stability, eval stability;
      the split between early and late moves (`MOVES_TO_GO` = 20).
- [ ] qsearch, if retried: a TT in quiescence, cheaper chain windows (pass plus the best few
      answers), pruning of hopeless attacks.
- Retested at the Rust speed and stopped near zero: PVS (H0), qsearch, futility, conthist,
  PVS+RFP, RFP by kind (stopped at 2220 games, +4.5, LLR -1.50) (all on top of LMR).

## Evaluation

- [x] Learned eval, 2026-10-15: the mover-view MLP16 (`features: "mover"`, 49 features;
      scripts/train_eval_mover.py with a deck-pair term in training) on
      runs/data/pairs500x20_2000n.jsonl: SPRT vs the hand-written eval H1, +28.5 +/- 17.6 at 952
      games. `YGOAI_EVAL=model.json` loads another model, `YGOAI_EVAL=hand` the hand-written eval.
  - Earlier baselines on runs/data/selfplay_2000n.jsonl (each game its own deck pair, so deck
    strength and position quality mix): scripts/train_eval.py, linear/MLP on eval and rich features.
- [x] Round 2 (default since 2026-10-15): datagen with the learned eval on
      runs/books/pairs500x20_b.jsonl (runs/data/pairs500x20b_2000n.jsonl, 6.1 h on 28 threads), MLP16
      on the 87 mover2 features (crates/ygoai/models/eval_mover2_mlp16.json): SPRT vs the 49-feature
      model H1, +31.9 +/- 18.7 at 896 games. Bench checksum 12ea7bf38d028206 (3352f7a7796dd607 with
      `hand`, 8e06890beed3e7fd with runs/nn/eval_mover_mlp16.json).
  - The within-pair ranking did not predict it (mover2 -0.005 +/- 0.004 vs 49 features; it moves
    by as much between epochs); the held-out losses did (value BCE 0.5107 vs 0.5129, result
    log-loss 0.4798 vs 0.4850). Feature-group ablations were all within noise.
- [ ] Deeper MLPs (`--layers`, the Rust eval takes any number): 2x64 ranks +0.007 +/- 0.004 over
      1x16 but overfits after a few epochs (10k games); SPRT 2x64 vs the default stopped at 380
      games (-7.3 +/- 24.9, LLR -0.77). Retry once more data rounds exist.
- [x] Data are games (45ab3d4): ygoai-datagen writes OUT.games.jsonl (GameRecord + search value and
      depth per searched decision); `ygoai-refeature GAMES OUT --features ...` computes rows for any
      feature set (`ygoai::datarow`) by replay, in seconds. Rounds 1 and 2 predate it (no moves
      recorded; their features change only with a 6 h search rerun).
- [ ] More features: normal summon used, LP thresholds, ...
- [ ] Later: the learned policy/value nets and effect-text embeddings (see memory).

## Engine and core correctness

- [ ] A datagen game that never finishes: opening 14478 of `match1.jsonl` (`selfplay_2000n`)
      kept several threads busy for hours. With the old datagen binary (before ab5521c) it still
      runs after 10 min, inside the Synchro material check (`procs::synchro::hand_checks`,
      `check_p42`, `get_card_effect`): likely an exponential material search. The current binary
      plays the game in 51 s (its search goes elsewhere), so the slow check is still there. Find
      the position (old binary: `scratchpad/datagen-bin`) and bound or speed up the check.

- [x] The 4 mismatching games against the C++ core (found 2026-10-13, older than the recent
      changes): pair17349 game 1 decision 947, pair19366 game 2 decision 1880, pair8112 game 0
      decision 89, pair9854 game 1 decision 467 (`build/regress_pairs/`). Fixed (rust 745abcb);
      script bugs are now fixed in `lua_fixes/` instead of emulated (rust 0e3b15d, 46 cards).
- [x] Collect the Lua errors the C++ core really raises over the recorded deck pairs
      (`YGO_DIFF_LOG=1`): identical games hide script bugs the port copies (e.g. Spell Gear).
      Done 2026-10-15 (all 19470 pairs, 3 games, seed 1): 5 card scripts fixed (rust 81c49f0); no mismatches.
- [ ] Left from that scan: the Freezing Chains of the Ice Barrier / Safe Zone "C stack overflow"
      (endless IsSetCard/IsExistingMatchingCard recursion, pairs 9160 and 12763; Gozen Match's
      error there follows it), and Alba System Dogmatikalamity's fusion check running out of the
      core's script budget (pairs 2394 9294 15254 16976, those games skipped).
- [ ] Last Turn activated during an attack: its forced battle leaves the outer attack without an
      attacker, and the core panics (C++ dereferences null). This is an upstream bug. The search now
      avoids such lines (9c3af1d); decide whether to guard it in the core too.
- [ ] Seed the engine processes in matches (pass the seed to `ygoai-engine`), so a crashed
      game can be reproduced exactly.

## Housekeeping

- [ ] Repo cleanup: what is still needed of `ygoai/` (Python), `ygoai_rs/` and the other folders.
- [ ] Remove the worktrees `wt-old` (main repo) and `wt-old-rust` (rust repo) in the scratchpad.
- [ ] Push the rust submodule commits 2b7f988, 1354870, 935783d, 658b359 (ask first).
