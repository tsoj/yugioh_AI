# TODO

Planned work, roughly in order. SPRTs run at 20+0.2 (since 2026-10-17; 30+0.3 before), SPRT [0, 10], with book
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
- [ ] Not tried yet (from the chess progression in memory): aspiration windows, late move
      pruning, internal iterative reduction, improving heuristic, SEE-like pruning, singular
      extensions / multicut, double/negative extensions.
- Tried 2026-10-16/17 (features in the code, off by default):
  - `killers` (two per height, after the TT move; depth at 3000 nodes 12.84 -> 12.93), 1000
    nodes: H0, -2.8 +/- 10.1 at 1972 games.
  - `online` (the user's correction history: no table, one normalized gradient step of all the
    eval's weights toward each node's search value, share min(1, lr * depth^exp), the net reset
    per decision; error before each step ~15-16 points vs 17.2 without), 20+0.2: lr 0.003 stopped
    at 860 games (-3.6 +/- 16.4, LLR -1.23), lr 0.01 stopped at 860 (-5.3 +/- 16.6, LLR -1.44).
  - `nullmove` (ending the phase as the null move at main/battle decisions with static eval >=
    beta, null window, R = NULL_R + depth / NULL_R_DIV), 20+0.2: R = 3 + d/4 (depth 13.12, 3.8% of
    cuts wrong) H0, -24.4 +/- 19.6 at 600 games; R = 4 + d/3 (13.21, 5.1%) H0, -0.8 +/- 8.7 at
    3224 games.
- [x] Predictive stopping (`predict`, default since 2026-10-14 without a pass: +3.8 +/- 11.7,
      LLR -0.35 at 1760 games): the next iteration only if it is expected to end within the
      budget. It uses about 25% more of the budget than the old rule (start while under half).
- [ ] Time management: node-based scaling of the soft bound, best-move stability, eval stability;
      the split between early and late moves (`MOVES_TO_GO` = 20).
- [x] Clock handling (2026-10-18, taken without a pass: 20+0.2 stopped at 460 games, -10.6 +/-
      22.8, LLR -1.16; uses ~48 s of the clock per game vs 42, depth 12.51 vs 12.26): no
      MIN_SEARCH_TIME (2 s) floor any more, under which the rules' player answered (at 1+0.01 that
      was ~90% of the moves). The budget is (remaining - reserve) / MOVES_TO_GO + the clock's
      measured mean gain between two searches (the in-between decisions' increments less their
      overhead) instead of one increment; the reserve is 2x the game's slowest reconstruction; the
      hard limit is at most half the clock and TIME_BUFFER (50 ms) before its end; the
      reconstruction has that deadline too (checked before each restart, the approximate pass and
      the second attempt). 1+0.01 self-play: ~100 searched moves per game (~190 nodes), 1 time
      forfeit in 960 games (single hypothesis builds of ~0.2 s are not interruptible).
      Overhead outside the search 6.4 -> 3.3 ms per decision (16 parallel games): corpus, regexes
      and whole-corpus script scans load at "isready" (each game is a new engine process); the
      corpus completion indexes rows by deck; one regex for all global helpers. Left: the replay
      per decision (~38%), the deck model (~30%, only on new reveals), HiGHS (~12%).
- [x] Search cleanup (2026-10-18): every feature not on by default removed, with its parameters,
      environment variables and tests: pvs, rfp, rfpall, conthist, qsearch, narrowext, killers,
      nullmove, online, lmrlog, resample, deckpick, passrisk, and searching the decisions inside a
      move (unselect, tribute, multicard, sum, position, place); `ygoai-tree-compare` (it needed
      the search's trace). lmr (with a null window), futility and the predictive iteration start
      are always on ($YGOAI_SEARCH is gone). The features' results stay in this file; their code
      in the git history before the cleanup commit. Bench checksum unchanged (573fe24d6e988ea6).
      The evaluation too: only mover2 MLPs load (the hand-written eval, YGOAI_EVAL=hand, the linear
      and antisymmetric models and nets on mover/mover3/mover4/rich features are gone; datagen
      still writes those feature sets). mover3 without one-sided pairs, NPS-adjusted 200 nodes
      (`ygoai-match --nps-nodes`): H0, -1.1 +/- 8.9 at 3182 games (it searched ~4% slower); at
      20+0.2 +5.0 at 620 games (stopped): not merged.
- [x] SPSA of the search parameters on positions (`ygoai-spsa`, 2026-10-18): fishtest's/OpenBench's
      update rule, theta+ vs theta- on the same batch of 64 positions (one random searched decision
      per game, true state, shuffled across 4000 games); loss = regret of the 1000-node search's
      move under a 2000-node-depth reference with exact values per root move + 0.1 * value error.
      Run 1 (3000 iterations, ~2.7 h): validation loss 538 -> ~500 (regret 438 -> ~400 LP, value
      error unchanged); root card width 3 -> 5, futility margins 583 -> 547 and 233 -> 228 LP,
      the rest unchanged. SPRT at 20+0.2 H1, +17.8 +/- 12.8 at 1484 games: merged (bench checksum
      fb0d2430d4f8e25f). Next: more iterations / other parameters (time management, history decay,
      the move ordering), a deeper reference.
      Run 2 (2026-10-18; fractional depths; tuned: lmr_min_depth, lmr_min_move_counter,
      lmr_reduction, lmr_reduction_per_move (new, from 0), futility_*, hist_exponent (new),
      hist_malus (new); 1000 nodes vs a 4000-node reference, refs ~20 min, 3000 iterations ~1.6 h):
      no gain. Final lmr_reduction 1.24, per move 0.03, the rest near the defaults; on all 4500
      positions (`--compare`) +9.3 +/- 6.9 loss vs the defaults (592.7); SPRT stopped at 240 games
      (+0.0 +/- 34.2, LLR -0.17): not merged.
      One validation position can move the mean by 16 (regret clamp 8000 / 500): the validation
      now prints the paired difference with its standard error; batches take the same number of
      positions from each worker (the iteration waited for the unluckiest one, ~75% CPU).
- [x] Every decision searched at the root (`root_candidates`, 2026-10-18): besides the searched
      types, several cards (subsets of the rules' first 6), select/unselect (the rules' first 5 +
      finish), tributes (the rules' answer, then the least valuable minimal sets), sums, places and
      opponent's zones, counters, card/chain orders (permutations, the given order first): the
      rules' answer first, at most 5 (ROOT_COMBO_WIDTH); positions, numbers, races and attributes:
      all; card names and rock-paper-scissors stay with the rules. Inside the tree the rules answer
      as before. In 32 test games per side: 330 place, 111 position, 7 multi-card, 2 order
      searches. SPRT at 20+0.2 vs 120eb1b: H1, +11.5 +/- 9.2 at 2836 games: merged (bench
      checksum unchanged, fb0d2430d4f8e25f: the bench has only the old types). Next: places
      rarely matter (Links, Pendulum Zones, columns): measure how often a place search changes the
      rules' choice, and gate it on Link/Pendulum/column cards in the duel.
  - Measured (`YGOAI_ROOT_LOG=path`: a JSON line per root search, 200 games at 10+0.1, 42965
    searches): places 16% of the root's search time; the search took another zone than the rules
    in 17.6% (gap > 200 LP: 6.6%), as often in duels without a Link/Pendulum/column card
    (`zone_sensitive`; 90% of the place decisions have one, mostly Links in the Extra Deck: a gate
    on that would gate nothing). Places answered by the rules: SPRT at 10+0.1 stopped at 1940 games
    (+1.3 +/- 11.4, LLR -1.11): not merged. Open: the Extra Monster Zone choice vs noise.
- [x] The root's 16-candidate cap removed (2026-10-19): it cut in the listed order, so an idle
      decision with 16+ options (2.8%) lost its activations and Battle/End Phase, a card selection
      of 16+ cards (1.6%) the rules' first cards. SPRT at 500 nodes stopped at 1040 games (+0.3 +/-
      8.4, LLR -2.53): merged anyway as a fix (user). ygoai-spsa's references of positions with 16+
      moves change: recompute them.
- [x] No cap on the children below the root (MAX_CHILDREN 12 -> 1000, `YGOAI_TUNE=max_children=1000`,
      2026-10-19): SPRT at 500 nodes H0, -3.7 +/- 10.5 at 1220 games; removed anyway (user):
      MAX_CHILDREN and the tunable max_children are gone. Bench checksum 66e2903f46a8bd1e.
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
- [x] Round 3 (default since 2026-10-17): datagen with the round-2 net on runs/books/pairs500x20_c.jsonl
      (lines 11000-11499 of match1, 20 games per pair; all 10000 games in
      runs/data/pairs500x20c_2000n_all.games.jsonl, the ones the OOM below lost replayed). MLP16 on
      the mover2 features of rounds 2 and 3 (2.45M positions; crates/ygoai/models/eval_mover2_r23_mlp16.json)
      vs the round-2 net: stopped at +6.7 +/- 10.0 after 2540 games (LLR +0.66), kept as unlikely
      to be worse. Bench checksum 77d370772eda5977 (mean depth at 3000 nodes 13.29).
  - Without the one-sided deck pairs (`--drop-onesided`: 153 of 1000 pairs where one deck won all
    20 games, 15% of the rows), default since 2026-10-17 (models/eval_mover2_r23os_mlp16.json):
    SPRT H1, +21.0 +/- 14.4 at 1242 games (LLR +3.00) vs the same on all pairs, although its CV
    was worse in all 5 folds (result log-loss 0.4839 vs 0.4823): the held-out losses do not
    predict this. Bench checksum 573fe24d6e988ea6 (mean depth 13.15). Next: other data filters
    (also drop nearly one-sided pairs, e.g. 19-1 / 18-2? weight pairs by balance?) and retest
    mover3 and 2x32 on the filtered data.
  - Then mover3 vs mover2, both MLP16 on rounds 2+3: stopped at +3.3 +/- 12.6 after 1480 games
    (LLR -0.41). 5-fold CV (held out: round-3 pairs): mover3 0.4792 alone, 0.4789 with round 2
    (value BCE 0.4897 / 0.4877).
  - Deeper, mover3 on rounds 2+3 (CV result log-loss): MLP16 0.4789 (4 epochs 0.4785), 2x32
    0.4774 (4 epochs 0.4764, better in all 5 folds), 3x32 0.4782, 2x64 0.4825 (4 epochs 0.4768:
    bigger nets overfit after ~3 epochs), 2x64 wd 1e-3 0.4789. SPRT 2x32 (4 epochs) vs mover3
    MLP16: stopped at -13.0 +/- 20.5 after 560 games (LLR -1.65).
- [x] Feature sets mover3 (104: summons/attacks this turn, ready attackers, low-Level hand
      monsters, free zones, direct lethal, chain owner) and mover4 (122: effects with uses left on
      the field / in GY / in hand, set Traps and Quick-Plays, Extra Deck kinds, more battle stats).
      5-fold CV over deck pairs on 3800 round-3 games (MLP16, `--fold I/5`), held-out result
      log-loss / value BCE: mover2 0.4832 / 0.4929, mover3 0.4798 / 0.4920 (better in 4 of 5
      folds), mover4 0.4800 / 0.4937 (its groups ablated: all within 0.4795-0.4800). For mover3
      the MLP16 beat h8 (0.4837), h32 (0.4809), 2x16 (0.4811), 6 epochs (0.4829); wd 1e-4 equal.
      Data-bound; on the full round 3 see Round 3 above.
- [ ] More features: LP thresholds, ...
- [ ] Later: the learned policy/value nets and effect-text embeddings (see memory).

## Engine and core correctness

- [ ] A datagen game that never finishes: opening 14478 of `match1.jsonl` (`selfplay_2000n`)
      kept several threads busy for hours. With the old datagen binary (before ab5521c) it still
      runs after 10 min, inside the Synchro material check (`procs::synchro::hand_checks`,
      `check_p42`, `get_card_effect`): likely an exponential material search. The current binary
      plays the game in 51 s (its search goes elsewhere), so the slow check is still there. Find
      the position (old binary: `scratchpad/datagen-bin`) and bound or speed up the check.

- [ ] A Kaiju loop: in round 3's game 2898 (pairs500x20_c.jsonl, 2000 nodes) a duel the search
      reached at decision 596 never asks anything again: process_adjust ->
      adjust_self_destroy_set -> check_unique_code (Kaiju's unique procedure), memory growing ~1 GB a
      minute; the OOM killer stopped the whole datagen at 20G. run_until_prompt now gives up after
      MAX_PROCESS_CALLS = 100000 (b01e459; normal steps < 1000). The looping duel (turn 7, End
      Phase): player 1 controls Thunder King (Kaiju, unique on field), Ha-Re x2, Ni-Ni and player
      0's Block Dragon (Rock monsters it controls are indestructible unless by battle, so also by
      rule); every process call hints Block Dragon, i.e. a destruction it prevents is retried by each
      adjust. The C++ destroy applies EFFECT_INDESTRUCTABLE to REASON_RULE alike, so likely upstream
      (or a script: Block Dragon's `(r&REASON_BATTLE)==0` also covers rule destruction). Next: a
      record of the line ($YGOAI_DEBUG_BROKEN prints the field at the cap) to replay against C++.
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
