# TODO

Planned work, roughly in order. SPRTs run at 30+0.3, SPRT [0, 15], with book
`runs/books/match1.jsonl`, and are read by LLR. Results go to `runs/sprt/results.jsonl`.

## Next (user's order, 2026-10-16)

1. Different card guesses per iteration (below; the first-child extension failed).
2. A new data round with the current engine (labels from match-style records), train the same
   architecture (MLP16, mover2), SPRT it.
3. More input features.
4. Enough diverse openings for tests and datagen: match1.jsonl has 19996 lines (15535 pass
   book_min 2); SPRTs draw from the first 10000 (shuffled); the eval data used deck pairs from
   lines 10000-10999 (pairs500x20*.jsonl). Keep test and training openings apart and check how
   often SPRTs reuse openings (and decks) the nets were trained on.

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
- [x] Budget search (node budgets instead of depth, shares by online cutoff statistics): every
      variant failed or was stopped near zero (runs/sprt/results.jsonl, 297c001..f8ed04f); removed
      in the search/ split. `YGOAI_STATS` (widths, cutoffs, LMR re-searches, subtree sizes) stays.
  - [ ] Retest the decision features (`unselect` etc.) on the current search (`unselect` failed
        with one ply per node: H0, -19.8 Elo, 492 games).
- [x] `MAX_DEPTH` was 12: the depth search (`lmr`) stopped there whatever its time (mean depth
      10.8 already at 3000 nodes). Now 100, a safety bound only (SPRT vs 12 stopped at 1420 games:
      +2.9 +/- 13.0, LLR -0.47).
- [ ] Try skipping "tactical" decisions at low depth: answer some decisions with the heuristic
      when little depth is left, as the horizon does now for the inner decisions. Which ones count
      as "tactical" is open: e.g. zones, positions, chain responses that do not change the eval,
      battle details.
- [ ] Better move ordering for the new decisions, e.g. history by decision type and card. The
      goal is that LMR and other reductions thin out the unimportant choices.

## Search: features

Already in: iterative deepening, heuristic move ordering, TT with TT move and bounds,
butterfly-style history, hard/soft time bounds, LMR (default since f783119).

- [x] Search laid out like a chess engine (Nalwald/Stockfish names): crates/ygoai/src/search/
      (search, tt, history, movegen, movepick, evaluate, params). Removed with it: the node-budget
      search, `lmr_noresearch`, `rfp_kind`, `pv_ext` (failed or near zero; git history has them).
- [ ] LMR the chess way: a log formula over depth and move number, adjusted by history,
      PV/non-PV and cut nodes; with PVS, a null-window reduced search. SPRTs at 1000 nodes
      (TC 60+1), 2026-10-16:
  - [x] `lmrzw` (the reduced search with a null window; depth at 3000 nodes 12.71 -> 12.84): H1,
        +11.1 +/- 8.8, LLR +3.05 at 3074 games; default since a6e2819 (checksum 91ffa3dab8e8055b).
  - `lmrlog` (max(1, round(0.5 + ln depth ln(i+1) / 2)) plies; 13.22): stopped at 4280 games,
    +5.4 +/- 7.7, LLR +0.28. `rfpall` (RFP at all nodes below the root by futility's margins; 14.88):
    stopped at 3840 games, +3.6 +/- 8.0, LLR -0.82. Both on top of `lmrzw` (15.29): stopped at
    2880 games, +1.4 +/- 9.4, LLR -1.54. `pvs` on top of `lmrzw` (12.65): H0, -9.2 +/- 13.5 at
    1290 games.
  - 30+0.3 is about 5000-8000 nodes per move by the games' depths (13-13.8 vs 8.6-9.1 at 1000
    nodes), but the speed depends on the load: 15600 nodes/s idle, 3800 with three SPRTs running.
  - 74% of the game pairs are split (the same deck wins both); openings the engine decides would
    make SPRTs 3-4x faster.
- [ ] Not tried yet (from the chess progression in memory): aspiration windows, null move
      pruning, killer moves, late move pruning, internal iterative reduction, improving
      heuristic, SEE-like pruning, singular extensions / multicut, double/negative extensions,
      static eval correction history.
- [x] Predictive stopping (`predict`, default since 2026-10-14 without a pass: +3.8 +/- 11.7,
      LLR -0.35 at 1760 games): the next iteration only if it is expected to end within the
      budget. It uses about 25% more of the budget than the old rule (start while under half).
- [ ] Time management: node-based scaling of the soft bound, best-move stability, eval stability;
      the split between early and late moves (`MOVES_TO_GO` = 20).
- [x] Futility pruning again (Stockfish's parent-node form): the moves after the first at depth
      <= 3 whose static eval + 45 + 4 * depth cannot raise alpha (margins measured with the mover2
      eval: the 99th percentile of the gap over the moves that raised alpha, 49/52/55/72 at depth
      1-4). Mean depth at 3000 nodes 9.60 -> 9.91 (up to depth 1: 9.74, 2: 9.83, 4: 10.02). SPRT
      vs `lmr,predict` stopped at 280 games (+2.5 +/- 28.0, LLR -0.12).
  - Smaller margins (`YGOAI_FUTILITY=margin,per_ply,max_depth`), mean depth at 3000 nodes (15
    openings, base 9.65): 25,2,3 10.29; 15,2,4 10.80; 10,2,6 11.54; 5,2,8 12.37; 0,2,all 14.41;
    20,0,all 13.52; 10,0,all 15.00; 0,0,all 17.85; -10,0,all 20.80; -20,0,all 25.66. SPRT of
    5,2,8 (+2.7 plies) vs `lmr,predict`: H1, +19.4 +/- 13.5, LLR +3.03 at 1364 games; default since
    2026-10-16 (in LP: 583.4 + 233.4 * depth up to depth 8). Bench checksum 7ff0dd5b80995d11
    (1a362d951150a90d with `hand`), mean depth 12.71 at 3000 nodes (was 10.31).
  - [ ] Tune further (margins, max depth, per move kind) now that the units are LP.
- [ ] Extend the first child when it is the TT move and the node has only 1 or 2 children: by 1 ply,
      or by a fraction of a ply (e.g. 0.5; needs fractional depth). `narrowext` (1 ply, up to
      height 2 * the iteration's depth): mean depth at 3000 nodes 12.71 -> 10.13 (in games 10.9 ->
      9.5); SPRT vs the default stopped heading to H0 at 1120 games, -6.5 +/- 14.6, LLR -2.07.
      Untried: the fractional (0.5 ply) version, or only at nodes with exactly 1 child.
- [ ] qsearch, if retried: a TT in quiescence, cheaper chain windows (pass plus the best few
      answers), pruning of hopeless attacks.
- Retested at the Rust speed and stopped near zero: PVS (H0), qsearch, futility, conthist,
  PVS+RFP, RFP by kind (stopped at 2220 games, +4.5, LLR -1.50) (all on top of LMR).

## Hidden information

- [ ] Different card guesses and deck fillers in each iteration of the iterative deepening (the
      opponent's hidden cards re-sampled per depth), instead of one guess for the whole search.
      Feature `resample` (c0342ea): the guessed cards still hidden get codes drawn from the deck
      model (presence-weighted, copies from the count model) within the observations'
      requirements, ~1.7 ms per draw; SPRT vs the default running. TT and history unchanged.
      Stopped (2026-10-16): 30+0.3 at 480 games -14.5 +/- 22.7 (LLR -1.46); 1000 nodes at 960
      games -5.4 +/- 16.6 (LLR -1.46); T = 0.3 (`YGOAI_RESAMPLE_T`, 28ddb41) at 1000 nodes, 2300
      games +3.2 +/- 10.7 (LLR -0.62).
- [x] `deckpick` (73d477b): after the root's move the opponent picks the worst of 3 guessed decks
      (the reconstruction's + 2 draws at T = 0.3), a pseudo-move with cutoffs, the deck in the TT
      key; 2 plies less depth at 1000 nodes (10.0 -> 8.0). H0 at 1000 nodes: -5.0 +/- 11.3, LLR
      -3.00 at 2006 games.
- Measurements (2026-10-16): `ygoai-tree-compare` (50 positions, 3000 nodes): trees under
  different guesses share ~21% of their edges (68% one ply below the root, 8% at height 12);
  on shared edges 92% same view, ~65% same value; 40% of edges come after an unknown card was
  revealed; root move = the truth's 78-80%. `ygoai-guess-scaling` (508 positions, reference 50000
  nodes with the true cards): the reference's move picked at 300 / 1000 / 3000 / 10000 nodes:
  true cards 67.9 / 69.5 / 74.8 / 81.3%, drawn decks 65.0 / 66.0 / 68.0 / 70.6%, the likeliest deck
  64.0 / 65.0 / 65.2 / 70.3% (+-2): depth helps on guesses about half as much; the likeliest deck
  is no better than draws.
  - [ ] If it fails, or after: a TT key without the guessed cards' codes while they are hidden
        (zone cards' codes and stats, the codes in the opponent's prompts), so that entries carry
        over between guesses. History: keep the codes (an entry for a card matters in every
        guess that has it).

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
- [x] Values in LP (2026-10-16): a learned eval's output times its LP anchor (`lp_anchor` in the
      model JSON, measured by `ygoai-anchor MODEL ROWS --write`: 1 / the mean output change per LP
      of the LP difference, positions with both players at 2000+ LP; the default mover2 MLP16: 1 point
      = 116.7 LP). train_eval_mover.py -o anchors each new net on its held-out rows; the engine
      warns about a net without an anchor or with one measured on other weights (hash). The
      hand eval, NULL_WINDOW, the RFP and futility margins and MCTS's VALUE_K are in LP too; WIN
      is 1e7. Bench checksums unchanged (default, futility, the round-1 net).
- [x] Match games as data (2026-10-16): engines send their search's root value and depth with each
      response ("info"), and ygoai-match's records carry them as `labels`; ygoai-refeature reads
      such records (the game number is the line). The labels are the engine's view (its
      reconstruction), the features the true duel's, as in the datagen.
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
