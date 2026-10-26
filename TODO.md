# TODO

Planned work, roughly in order. SPRTs (`scripts/sprt_branch.sh`) run at 10+0.1, bounds [0, 5] and at
most 10000 openings (20000 games) since 2026-10-25 ([0, 10] and 5000 before; 20+0.2 2026-10-17 to
-19, 30+0.3 before), with book `runs/books/match2.jsonl` (lines 0..49999; match1.jsonl before
2026-10-24), and are read by LLR. Results go to `runs/sprt/results.jsonl`.

Since 2026-10-19 (OpenBench/fishtest style): main holds only what passed, always on, without flags
to switch search methods; every experiment is a branch with the feature on by default, tested with
`scripts/sprt_branch.sh <branch> [main]` (10+0.1 by default, `NODES=N` for fixed nodes). Failed
experiments stay as branches: futility-reductions, inner-decisions, pvext-open-window, pv-half-ply,
pv-move-half-ply, lmr-linear.

Where we stand (2026-10-19, current alphabeta at 10+0.1, 100 games each, shuffle 7):
the rules' player (Rust) 93-7 (+449 +/- 155); the rules' player at 1b5e063 (Python) 95-5 (+512 +/-
215); the RL policy (ygoai/nn/policy.npz, rl4 iteration 71) 90-10 (+382 +/- 119); the first
alpha-beta (1b5e063, Python on the C++ core, rebuilt in build/snapshots/first-ab-1b5e063) 86-14
(+315 +/- 96). Logs: runs/sprt/vsold-*. The first run against the rules' player had a game of 12332
decisions (alphabeta activated Zoodiac Whiptail's attach effect, negated every time, ~3000 times in
one turn, then lost on time): fixed by capping the search agent's own activations per turn (1c731b1;
the rerun: longest game 941 decisions).

## Next (user's order, 2026-10-16)

1. [x] Different card guesses per iteration (below): `resample` stopped near zero (2026-10-16).
2. [x] A new data round with the current engine (labels from match-style records), train the same
   architecture (MLP16, mover2), SPRT it: done with the SPRT games as data (2026-10-21, H1 +17.2).
3. More input features (see Evaluation: mover3/mover5 were tested on 2.45M / ~14M positions; the
   SPRT data is now ~30M positions, ~200k games, so a retest is due; the engine evaluates only
   mover2 since the cleanup, so a feature set needs its code back in `ygoai::datarow`'s eval).
4. [x] Enough diverse openings for tests and datagen (match2.jsonl, ~55k, since 2026-10-24): match1.jsonl has 19996 lines (15535 pass
   book_min 2); SPRTs draw from the first 10000 (shuffled); the eval data used deck pairs from
   lines 10000-10999 (pairs500x20*.jsonl). Keep test and training openings apart and check how
   often SPRTs reuse openings (and decks) the nets were trained on. Urgent since 2026-10-21: the
   SPRT-data net and the MLP 64 were trained on SPRT games, i.e. on the openings the SPRTs draw
   from (the first 10000). Check their gains on unused openings (lines 12000+) before more of it.
   [x] Checked 2026-10-22 (main's MLP 64 vs the MLP16 r23os before both SPRT-data nets, 1000
   openings each, 10+0.1, sprt_branch.sh ELO=none, branch `eval-r23os`): first 10000 +29.8 +/-
   11.4, lines 12000+ (runs/books/match1_from12000.jsonl) +20.2 +/- 11.5. The gain holds on
   unseen openings; the gap (9.6 +/- 16) is within noise. (Both below the SPRTs' +17.2 and +35.8
   summed: early-stopped H1s overestimate.) Keep test and training openings apart anyway.

## Running now

- Nothing. For review: branch `tidy` (2026-10-25, the code cleanup after Nalwald's style: same
  search, bench checksums unchanged; the belief regression set equal to a fresh build of main),
  and branch `py-cleanup` (the legacy Python removed).
- `tt-keep` (77cd2b5, on `tidy`: the TT kept across decisions, no generations; its best move by
  move id, not by index) SPRT vs main [0, 5]: stopped by the user at 3220 games, +1.7 +/- 9.3, LLR
  -0.17 (bench: depth 10 265691 nodes, main 362311; move ids alone, the TT still reset per
  decision: 363004). Not merged: the gain is small, and a TT reset per decision keeps searches
  independent (easier debugging).
- `improving2` (643545c: improving rebased; +3.1 at 2720 games before) SPRT vs main [0, 5]:
  stopped by the user at 1340 games, -3.1 +/- 14.2, LLR -0.53 (bench: depth 10 361130 nodes, main
  362311; 3000n depth 13.25, main 13.05; same NPS). Not merged.
- `hist-decay` (3f780cf: history decay 0.5 -> 0.75, for the finer keys) SPRT vs main [0, 5]:
  stopped by the user at 4820 games, +3.6 +/- 7.6, LLR +0.37 (bench: depth 10 362662 nodes, main
  362311; 3000n depth 13.25, main 13.05; same NPS). Not merged.
- `lmrlog` (697e4a7: LMR round(0.5 + ln depth ln(i+1) / 2) plies, at least 1; at 1000 nodes it was
  +5.4 +/- 7.7 at 4280 games) SPRT vs main [0, 5]: stopped by the user at 1140 games, -16.5 +/-
  15.8, LLR -1.47 (bench: depth 10 324415 nodes, main 362311; 3000n depth 13.71, main 13.05; same
  NPS). Not merged.
- Queued (2026-10-25, one at a time, each benched first; rebased if main changes):
  `probcut2` (9fadc3d: ProbCut rebased; +2.8 at 5380 games before). Like the ones above, a
  candidate whose earlier run ended slightly positive under [0, 10] and 5000 openings.

## Search: represent every decision

See `docs/search-decisions.md` for what is and isn't searched.

- [x] SPRT each decision feature vs `lmr` (`unselect`, `tribute`, `multicard`, `position`,
      `place`, `sum`): obsolete, the features were removed with the other failed ones in the
      2026-10-18 cleanup (see the retest item below).
- [ ] In-tree decisions in small steps (user, 2026-10-23; order: position, tribute,
      announcements, then several-card selections and unselect, zones/sorts/counters last).
      `ygoai-card-refs --decisions` (main, 45e4833): at the decisions the rules answer below the
      root, every root candidate's reference value and the rules' answer (per type, with the
      game's count). Measurement d1 (12119 positions from the card-policy and horizon-action
      SPRT records, one decision of each type per game, `scripts/decision_refs_report.py`; won/lost
      values clipped at 8000 LP): per type, games with it of ~3200, decisions per such game, rules'
      answer within 50 LP of the best, mean regret LP, regret x count per game:
      place 3170 / 26.0 / 0.967 / 39 / 1380; position 3140 / 10.3 / 0.945 / 54 / 614;
      unselect 2931 / 7.2 / 0.946 / 102 / 1002; several cards 1324 / 1.8 / 0.807 / 503 / 935;
      sort 504 / 1.6 / 0.956 / 60 / 117; number 411 / 1.2 / 0.786 / 621 / 768; tribute
      289 / 1.2 / 0.941 / 109 / 112; sum 167 / 1.6 / 0.743 / 939 / 1267; attribute 95 / 2.1 /
      0.926 / 350 / 369; disfield, race, counter, sort chain: rare.
  - [x] Position (branch `tree-position`, 73a42e5): a searched decision, the rules' order; at
        the horizon the rules answer it like a card selection (inside a move), on the main line
        searched one ply deeper. Bench: depth 10 344964 nodes in 388 searches (main 343056 in 369:
        the bench searches positions too), 322.6G instructions; 3000n depth 13.49. SPRT after
        horizon-action-ext2: H0, -2.1 +/- 9.6 at 2604 games (LLR -2.95). Then tribute,
        announcements, cards, unselect, each against main (chained).
  - [x] Tributes (branch `tree-tribute`, 2ec2a5f): the root's candidates in the tree (the rules'
        answer, then the least valuable sets that release enough, at most ROOT_COMBO_WIDTH),
        inside a move at the horizon. (`heuristic::tribute_answer` shared.) Bench: depth 10 342951
        nodes in 370 searches, 331.5G instructions; 3000n depth 13.44. SPRT H0: +0.9 +/- 7.3 at
        4336 games (LLR -2.96).
  - [x] Announcements (branch `tree-announce`, 2083990): a number, one type or one attribute,
        each a move (the rules: the first number, a random type or attribute), inside a move at
        the horizon. Bench: depth 10 344893 nodes in 369 searches, 334.4G instructions (+0.9%);
        3000n depth 13.41. SPRT H0: -2.5 +/- 9.8 at 2198 games (LLR -3.04).
  - [x] Several cards (branch `tree-cards`, 11eaa7a): the root's sets (`movegen::card_sets`)
        in the tree, ordered by the rules' order of the cards, the first CARD_WIDTH (d1: the best
        set is among the first 3 in 0.949, the first 2 0.903, the first 0.807). Bench: depth 10
        342760 nodes in 370 searches, 331.1G instructions; 3000n depth 13.41. SPRT H1: +14.8 +/-
        11.2 at 1696 games (LLR +3.02): merged (20d7b31).
  - [x] One card at a time (branch `tree-unselect`, 63d3099): the first CARD_WIDTH cards by the
        rules' order and finishing, inside a move at the horizon (d1: the best among the first 3
        cards 0.987). The root as before (at most ROOT_COMBO_WIDTH cards). Rebased onto main with
        tree-cards (2e5ee60): depth 10 334139 nodes in 384 searches (main 342760 in 370), 323.6G
        instructions (331.1G); 3000n depth 13.56 (13.41). SPRT H0: -23.0 +/- 18.9 at 666 games
        (LLR -3.04), as with one ply per node (-19.8).
  - [x] Sums (branch `tree-sum`, deacbd2, on main with tree-cards): the root's valid selections
        (least valuable cards first) in the tree, the first CARD_WIDTH, inside a move at the
        horizon (d1: rules' answer best 0.743, 939 LP mean regret, in ~5% of games). Bench: the
        same tree as main (no sum in the bench's searches). SPRT paused (user) at 3780 games,
        +3.8 +/- 7.6 (LLR -0.83; +13.9 at 1000 games) for the set policy: if it passes, restart
        from scratch with the policy's sum order against the new main; else resume with these
        counts (RESUME=runs/sprt/tree-sum_20261023-124136.jsonl); but if the set policy fails
        with positive Elo, `tree-sum-policy` from scratch against the current main instead
        (user, 16:59). Set policy H0 (+1.5 +/- 6.6, 4842 games) → `tree-sum-policy` SPRT vs main
        running since 17:15: **H1, +16.0 +/- 11.9 at 1438 games (LLR +2.98) → merged**. Ready for all: `tree-sum`
        rebased onto main (c2fbafd, the search unchanged), `tree-sum-policy` (6a857de) = set-policy
        + sums in the tree ordered by the set policy.
- [x] Set policy (user, 2026-10-23): the card policy for selections whose candidates are sets of
      cards (several cards, sums, tributes): a set's score is its cards' scores summed (plus a
      weight per card). Data: `ygoai-card-refs --decisions --types ...` rows now carry each
      card's features, the rules' order and each candidate's cards (c644ee2); trainer
      scripts/train_set_policy.py. Refs s1 (60k decisions of SELECT_CARD/SUM/TRIBUTE from the
      SPRT records since 10-19): 60083 rows in 37 min. Held out (scripts/train_set_policy.py,
      h16, all types; h32 and several-cards-only no better), best first / in the first 3 /
      regret of the first / of the best of the first 3: several cards 0.513 / 0.872 / 825 / 164
      (candidates' order 0.345 / 0.825 / 1287 / 265), sums 0.639 / 0.967 / 604 / 65 (rules'
      random answer 0.431 / 0.947 / 1255 / 94), tributes 0.676 (0.676: not used). Branch
      `set-policy` (727cf9f): several cards ordered in the tree and answered at the horizon,
      sums answered, by the policy. Bench: depth 10 342737 nodes (342760), +0.1% instructions;
      3000n depth 13.41. SPRT: H0, +1.5 +/- 6.6 at 4842 games (not merged; kept in `tree-sum-policy`).
  - [x] Zones (branch `tree-place`, 6a8df12): the root's places (as listed, the rules take the
        first) in the tree, the first CARD_WIDTH, inside a move at the horizon (d1: 26 per game,
        rules' answer best 0.967, 39 LP mean regret). Rebased onto c644ee2 (35a1522): depth 10
        520481 nodes in 446 searches (main 342760 in 370, +52%), 522.8G instructions (+58%);
        3000n depth 12.88 (13.41). Rebased onto the merged sums (42ef645; the same bench: 520443
        nodes, 523.0G instructions vs main 342737 / 331.3G; 3000n 12.88 vs 13.41). SPRT: H0,
        -7.6 +/- 12.8 at 1454 games.
  - [x] Zones with a place policy (user, 2026-10-23): narrow in the tree (the policy's first
        PLACE_WIDTH = 2), the policy's answer at the horizon, the root all zones by the policy.
        Features per zone (`search::placepolicy`, on main 03796be): whose and which zone, the
        column, what is in the column and next to it, the link arrows pointing to it, the card
        placed (the hint's code: types, link arrows) and where its arrows point from there. d1:
        20% of zone decisions have a spread, among them the rules' answer best 53%. Data
        runs/data/place_refs/p1.jsonl (50k, --per-game 6, seed 4): 8.3% spread > 50 LP (values
        clipped at 8000); policy h16 held out top1/2 0.54/0.83 (rules 0.55/0.82), regret of the
        best of 2 175 LP (206); gate (user: predict whether the zone matters; the zones'
        features' mean and max, scripts/train_place_gate.py) AUC 0.70. But the refs' shared
        search biases the first candidate: of single odd ones out 1090 the first, 762 another
        (~410 expected). `--independent` (6e0115f, each candidate in its own search) on 3000 of
        the same decisions: spread 4.3% (9.1%), the bias gone. p2.jsonl (50k, --independent)
        (50192, 1670 s): spread > 50 4.4%; gate (g2_s50) held out AUC 0.86, opening for 25% catches
        82% of those that matter; policy h16 (p2_h16) top1/2 0.60/0.80 (rules 0.63/0.73), regret
        of the best of 2 200 LP (286); the rules' answer + the policy's best other 0.80 / 206.
        Branch `tree-place-policy` (187b6dd): below the root a node only where the gate opens,
        there the rules' answer and the policy's best other; else and at the horizon the rules;
        the root every zone. Bench: depth 10 441109 nodes in 446 searches (main 342737 in 370;
        +7% per search), 438.7G instructions; 3000n depth 13.27 (13.41; tree-place 12.88). SPRT
        stopped by the user at 2520 games, +0.6 +/- 9.6, LLR -1.84 (for the refs below).
  - [x] The same first-candidate bias was in the card and set refs (r1, s1: main's policies').
        Regenerated with --independent (one-card mode now honours it, f0bcd8a; --decisions keeps
        sums, searched since tree-sum-policy and so missing from s2 at first, a4967c7):
        runs/data/card_refs/r3_indep.jsonl (60062, 46 min), set_refs/s2_indep.jsonl (60073 cards
        and tributes, 66 min) + s2_indep_sums.jsonl (8053). scripts/eval_policy.py scores models on
        the same held-out rows. Card (one), held out 4739, best first / in first 3 / regret of the
        best of 3: rules 0.442/0.813/353, main's cp1 0.477/0.840/275, new h16 e12 0.495/0.854/240
        (h16 e60, h32 the same in the first 3). Set: several cards s1 0.539/0.879/147 vs s2 h16
        0.552/0.877/153; sums 0.590/0.948/93 vs 0.596/0.957/62; tributes equal. Branch
        `indep-policies` (a1854c0: cp4_h16 + s2_h16). Bench: 10d 356184 nodes (342737), 3000n depth
        13.34 (13.41). SPRT H1: +26.7 +/- 16.7 at 822 games, LLR +3.00 -> merged.
- [x] The policies in place of the rules (user, 2026-10-24: where there is a policy, no rules).
      Branch `set-pool` (e623952): the set policy builds its own pool from its cards' order (every
      set of its first 8 cards; sums searched in its order, up to 64; every valid tribute set) and
      chooses from it at the root (from the agent's reconstruction, `root_set_moves`, mapped back to
      the agent's indices), in a node (best 3) and at the horizon; tributes now too (below the root
      the set policy's answer). Before: the rules' first 5 sets (27% of several-card selections
      have more than 8 cards). Bench: 10d 354983 nodes (356184), 3000n depth 13.33 (13.34); smoke
      128 games, 0 errors. SPRT stopped by the user at 5458 games, +2.2 +/- 6.5, LLR -2.55 (pentanomial
      [326, 0, 2059, 1, 343]: 75% of the opening pairs 1-1). `set-pool-notrib` (a363f9b, rebased:
      tributes back to the rules) on match2: H0, -10.8 +/- 14.3 at 1288 games (LLR -2.99, penta
      [101, 0, 462, 0, 81]). The set policy's own pool does not help: sets stay with the rules'
      first 5 (indep-policies on main).
- [x] Does the book's filter hold for alpha-beta? (user, 2026-10-24; yes: see the strength ladder below) match1.jsonl was filtered by
      the old Python policy nets (make_match_book.py: up to 20 games per opening, both seats one
      net at temperature 1; --book-min 2 keeps 15535 of 19996), yet ~75% of the SPRTs' opening
      pairs go 1-1 (the same seat wins both). Test: main vs main at 10+0.1 on 500 openings the
      filter removed and 500 it kept, lines 12000+ (runs/books/filtertest_{removed,kept}.jsonl,
      results runs/sprt/filtertest_*.log), the share of seat-decided pairs. If the same: no filter,
      random pairs and seat order each time; else a new book filtered with alpha-beta, more decks.
      The rules' rank as an input feature kept: without it (--zero 95,96) held out card top3 0.846
      (0.854), regret3 262 (240); sets about equal; tributes top1 0.645 (0.666).
      Remaining with a policy but rules: none on main (zones: the place policy only on
      `tree-place-policy`, stopped).
- [x] Predict one-sided openings from their cards (user, 2026-10-24; used for match2, see below). scripts/build_onesided_data.py:
      10385 SPRT openings (132417 pairs, ~13 each; 73.7% seat-decided; per opening 16% always, only
      14.5% at most half). The old filter barely predicts it (book balance >= 0.3: 0.719, else 0.760).
      Cards: scripts/embed_cards.py (effect1 encoder, one set of pseudo-names; 13748 cards of
      match1.jsonl; eager attention: SDPA gave NaN). scripts/train_onesided.py: binomial likelihood;
      splits random and archetype (41 rare archetypes held out: 5108 train openings without their
      cards, 1017 test with >= 25% of a deck from them, 28% of the test decks' cards unseen).
      Test r (3-seed ensembles; random / archetype): plain pooling (mean+max of a seat's cards)
      deck+top 0.439 / 0.351, seq 0.433 / 0.357, deck 0.433 / 0.340, top 0.380 / 0.296; attention
      0.405-0.423 / 0.280-0.326; grouped pooling unstable on archetypes (0.15-0.19); card ids (control)
      0.33 / 0.21. Draw order adds nothing; hyperparameters within noise (sweep1). As a filter:
      dropping the 25% predicted most one-sided 0.734 -> 0.695, 50% -> 0.675 (random split).
      Ceiling from ~13 pairs per opening: r ~0.8. Order-aware pooling (phipool: each card through an
      MLP with its draw position before the pooling; plain pooling cancels the positions in the
      mean): seq 0.431 / 0.343, deck+top 0.432 / 0.357, deck 0.425 / 0.348: the draw order adds
      nothing.
- [x] New book (user, 2026-10-24, while away): runs/books/match2.jsonl, ~50k lines. 130000 random
      candidates (seed 91, scripts/onesided_candidates.py generate), the final predictor
      (runs/nn/onesided/final_pool_decktop.pt: pool, deck+top, all 10385 openings, 5 seeds) keeps
      the less one-sided half, then the old policy games (make_match_book.py --from, rl4/rl3/rl2/
      rl1/boot1, book-min 2 at load). SPRTs then use match2 lines 0..49999 (sprt_branch.sh
      defaults); the rest for eval data and tests. Then set-pool-notrib on it, then an eval retrain
      on all SPRT games (scripts/collect_sprt_games.py from 2026-10-17 23:22 + scripts/sprt_games_to_npz.py + train_eval_mover.py
      --dims 64 --held 400 --drop-onesided --lam 0.5), SPRT, merge on H1.
      Built with ygoai-book-test in 763s: 64998 tested (2 panicked), 620389 games; less winning
      seat 0: 5085, 1: 5323, 2+: 54590 (book-min 2 keeps 54590; SPRTs draw from the first 50000,
      the rows are in finishing order). First SPRT on it (set-pool-notrib): 462 of 644 pairs
      seat-decided, 71.7% (+/- 1.8) vs 74.7-75.4% on match1: the predictor's expected 0.68 does
      not carry over, a gain of ~3 points at most. Next book: the predictor stage is optional.
- [x] RL policy in Rust (user, 2026-10-24): crates/ygoai/src/rl_policy.rs (the self-play policy/value
      network of runs/nn/rl*, boot1 and its agent; not the search's move-ordering policies) and
      ygoai-book-test (make_match_book.py's test games on threads, each candidate written as it
      finishes, resumable; make_match_book.py now also appends and resumes). Same games as the
      Python agents: 410 of 410 identical (40 candidates). One thread: 12.9s vs 60s; the book build
      86 candidates/s on 30 threads vs ~6.7/s in Python. Time now: engine ~51%, network ~23%, views
      and allocation ~24%. match2 was rebuilt with it (the Python run was stopped at 39000/65000).
- [x] Rust panic in the policy games of the match2 build (rust/ygo-cards/src/prelude.rs:100,
      effect_card_value: a value function called without a card). It ended the first build after
      16500 candidates (run_parallel now turns panics into Crash, 07f8739). Diagnosed 2026-10-25:
      the prelude.rs panics are caught by `Duel::protected_call` (lua.rs: a panic with a message
      inside a card function is a script error, as a failing `lua_pcall` in C++: logged, the call
      gives nil, the game goes on), so they only print. The 2 candidates the Rust build lost
      (seeds 91070843, 91125503; both have Last Turn) panic at processor3.rs:387
      (`core.attacker.unwrap()` in the battle command's step 9): the Last Turn bug below. Left:
      which card's value function gets no card (harmless; the C++ script would error the same).
- [x] Strength ladder (user, 2026-10-24): main at more vs fewer nodes on the 500 removed / 500 kept
      openings (runs/sprt/ladder_*, --tc 60+1, 32 games at once). Pentanomial [weak both, -,
      seat-decided, -, strong both]:
      | rung      | removed                       | kept                         |
      | 1000v500  | [28,0,434,0,38] +6.9          | [62,0,370,0,68] +4.2         |
      | 1500v1000 | [34,0,436,0,30] -2.8          | [54,0,374,0,72] +12.5        |
      | 2500v1500 | [26,0,448,0,26] 0.0           | [48,0,377,0,75] +18.8        |
      | 4000v2500 | [56,0,409,0,35] -14.6         | [81,0,357,0,62] -13.2        |
      4000v2500 is spoiled by the clock: 64 / 70 time forfeits, 47 / 57 lost by the 4000-node side
      (2500v1500: 19 / 21). Leaving out pairs with a forfeit, summed over the rungs: removed 115
      weak-both vs 124 strong-both (no strength signal; 87% seat-decided), kept 208 vs 275 (about
      3 sigma; 74% seat-decided). So the old policy filter works: the openings it removes are
      seat-decided even between engines of clearly different strength, the kept ones show it.
      Fixed-node games are nearly deterministic (13 of 16 identical when replayed).
- [ ] Search the decisions that have no feature yet:
  - [ ] `SelectDisfield` (zones to disable)
  - [ ] `SelectPlace` with several zones
  - [ ] `AnnounceCard`, `AnnounceRace`, `AnnounceAttrib`, `AnnounceNumber`
  - [ ] `SortCard`, `SortChain`
  - [ ] `SelectCounter`
  - [ ] cancel in selections, and unselecting in `SelectUnselectCard`
- [ ] Widen the capped candidate lists once ordering and reductions can take it: `CARD_WIDTH` (3,
      card selections and sums below the root), `ROOT_CARD_WIDTH` (5), `ROOT_COMBO_WIDTH` /
      `ROOT_COMBO_CARDS` (5 / 6: the sets of several cards, also in the tree). MAX_CHILDREN is gone
      (2026-10-19); the old per-decision widths went with the 2026-10-18 cleanup.
- [x] Budget search (node budgets instead of depth, shares by online cutoff statistics): every
      variant failed or was stopped near zero (runs/sprt/results.jsonl, 297c001..f8ed04f); removed
      in the search/ split. `YGOAI_STATS` (widths, cutoffs, LMR re-searches, subtree sizes) stays.
  - [x] Retest the decision features (`unselect` etc.) on the current search (done as the
        tree-* round 2026-10-23, see "In-tree decisions in small steps") (`unselect` failed
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
  - [x] Finer history keys (user, 2026-10-24; merged 5dbcbac by the user's call). Collisions measured first (ygoai-bench-search with
        YGOAI_STATS, 8220394 on main: move identities sharing a history slot, 6 openings at depth
        10): main within a search 0.3% of identities / 0.1% of lookups, within a game (~250
        identities) 1.6% / 1.1%. Branch `hist-keys` (b5e8ac8 + 2a374f9): an option by its effect
        (was its index), a card selection by its select hint and the chosen cards' controller and
        location (was their codes), a chain pass by the chain's last link (card, whose; was one
        slot for all passes), activations and chain links by their location; slots from all 128
        bits of the identity (splitmix64; before only the low 64 bits were mixed, which tripled
        the collisions with the new keys). Per game ~340 identities: 2.5% / 0.8% at 2^14 (2^15:
        1.1% / 0.2%), so the table stays 2^14. Bench: depth 10 362311 nodes (main 369443), 3000n
        depth 13.05 (13.20). SPRT vs main (tc 10+0.1, match2): ended at the 5000-opening cap, 10000
        games, +5.7 +/- 5.4, penta [751, 0, 3415, 1, 833], LLR +0.96 [0, 10] (would be +2.11 with
        [0, 5], +1.24 with [0, 2]): undecided, likely a small gain; merged. Next (by the result): the context in the key (own
        turn, phase), continuation history, separate or bigger tables.
  - [x] All three at once (user, 2026-10-25): branch `hist-more` (58a9981), the history plus
        two tables added up in the move order: the move in its turn and phase (mover is the turn
        player, phase; 2^15) and the move after the one that led to the node (continuation,
        2^16); same bonus, malus, decay. Bench: depth 10 366742 nodes (main 362311), 3000n depth
        13.29 (13.05), same NPS. SPRT vs main (resumed with [0, 5] and 10000 openings at 902 games): ended at the cap, 20000
        games, +1.3 +/- 3.7, LLR -1.64 (was +8.6 at 4722 games, LLR +2.07): no gain, not merged.
  - [x] Card selections by history (user, 2026-10-25): branch `hist-cards` (8bd99ba). Below the
        root a card selection's card_width (3) searched choices are the first by the TT move and
        the history among all its choices (CARD_POOL 127: the TT keeps an i8 index into them; sets
        and sums have at most 5 anyway), not the policies' first 3; at the horizon the answer is that first choice (TT move,
        else best history, the policy's order among equals) instead of the policy's first. Bench: depth 10 361955 nodes, main 362311; 3000n depth 13.27, main
        13.05; same NPS. SPRT vs main [0, 5]: stopped by the user at 9320 games, -0.1 +/- 5.6, LLR
        -1.58 (was +9.9 at 840 games, then flat at 0): no gain, not merged.

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
- [x] Not tried yet (from the chess progression in memory): aspiration windows, late move
      pruning, internal iterative reduction, improving heuristic, SEE-like pruning, singular
      extensions / multicut, double/negative extensions. All but these tried in the 2026-10-20
      round below (all H0 or stopped near 0): SEE-like pruning (no exchange notion here yet),
      double/negative extensions (SE itself failed), aspiration (not run: bench depth fell).
- Technique round 2026-10-20 (branches from main, implementations after Stockfish, Reckless,
  Plenty, Stormphrax, Pawnocchio; SPRT 10+0.1 vs main; bench depth at 3000 nodes, main 13.00):
  - `se` (singular extensions: TT move, depth >= 4, tt.depth >= depth - 3, not an all node,
    singular beta tt.value - 800 LP * depth, verification at half depth; +1, +2 if 300 LP below,
    multicut, -1 when tt.value >= beta; 14% singular, depth 12.59): H0, -16.2 +/- 16.6 at 900
    games. `se-400` (margin 400, depth 11.35) dropped with it.
  - `tm-nodes` (Reckless's node-based time management: the budget scaled by max(0.546, 3.18 -
    2.66 * the best move's share of the root's nodes) * max(0.716, 1.29 - 0.044 * the iterations it
    has stayed the best), times 0.556 for a mean of 1 on the bench): H1, +16.4 +/- 12.1 at 1612
    games; merged 2426943.
  - `iir` (internal iterative reduction: not on the main line, from depth 3, without a TT entry or
    one 4.5 plies too shallow, 1 ply less; depth 13.49): stopped by accident at 5640 games, +3.2 +/-
    6.7, LLR -1.54; rerun against main with tm-nodes stopped at 460 games, -18.1 +/- 23.7.
  - `probcut` (off the main line from depth 5, TT value (else static eval) >= beta: the first 2
    moves 3 plies less against beta + 1000 LP; depth 14.01): stopped by accident at 3280 games, +3.0
    +/- 8.6, LLR -1.06. Both rerun against the new main (with tm-nodes); the rerun stopped at 460 games, -19.7 +/- 24.2.
  - `razoring` (off the main line up to depth 4, a static eval 1000 LP per ply below alpha is the
    value; depth 13.57): stopped at 1640 games, -7.6 +/- 13.1, LLR -2.84.
  - INVALID: every test started after the tm-nodes merge (lmp, razoring, the iir and probcut reruns,
    improving, hist-prune, corrhist) ran a branch from before the merge against main with it, so
    without tm-nodes' gain (about -16 Elo each). All branches rebased onto main, all rerun
    (2026-10-20 18:22); sprt_branch.sh now refuses a branch that lacks BASE's last code commit.
  - Reruns (rebased onto main with tm-nodes): `iir` H0, -13.4 +/- 15.4 at 934 games; `corrhist`
    stopped at 2220 games, -2.2, LLR about -2.4; `razoring` stopped at 380 games, -14.6 +/- 26.8; `hist-prune` H0, -4.3 +/- 10.9 at 1954 games;
    `probcut` stopped at 5380 games, +2.8 +/- 6.6 (LLR -1.91): a small gain at best; `improving`
    stopped at 2720 games, +3.1 +/- 9.1 (LLR -0.90); `tm-eval` (on top of tm-nodes) stopped at about
    1180 games, -3.5 +/- 14.3 (LLR -1.60); `probcut-aggr2` (from depth 4, first 3 moves, margin 0;
    depth 15.05) stopped at about 180 games, -29.0 +/- 34.7; `hist-lmr-bad` (1 ply more LMR for
    moves with negative history; depth 13.05) stopped at 1080 games, -2.6 +/- 14.9 (LLR -1.31).
  - Finished 2026-10-21 (vs main with tm-nodes): `probcut-aggr` (from depth 3, first 3 moves, margin
    300 LP; depth 14.76) H0, +1.7 +/- 6.5 at 5870 games (was +18 at 540); `hist-gravity` (history
    bonus with gravity, HIST_MAX 2000; depth 13.09) H0, -0.0 +/- 8.0 at 3860 games; `hist-lmr` (LMR
    reduction from history; depth 12.46) H0, -22.8 +/- 18.9 at 702 games.
  - `probcut-improving` (both together, ProbCut on improving's static eval; depth 13.92): H0, +0.4
    +/- 7.7 at 4098 games.
  - `lmp` (late move pruning, Stockfish's/Reckless's shape: off the main line, a node stops after
    2.75 + 1.32 * depth^2 moves unless it is losing; depth 13.15): H0, -24.7 +/- 19.8 at 704 games. Rebased rerun: H0, +0.5 +/- 7.7 at 4388 games.
  - `aspiration` (root window around the iterations' mean, delta 300 LP, x1.5 per fail): not run,
    the bench depth fell at every delta (100: 12.39, 300: 12.67, 600: 12.79, 1500: 12.86).
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
- [ ] Time management: node-based scaling of the soft bound and best-move stability done
      (`tm-nodes`, H1 +16.4, 2026-10-20); eval stability failed (`tm-eval` stopped near 0). Left: the
      split between early and late moves (`MOVES_TO_GO` = 20), tuning tm-nodes' constants.
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
- [x] Linear LMR (`lmr_reduction_per_move`, the reduction 1 + x * (move - 2), 2026-10-19, without
      the 12 cap): x = 0.1 at 500 nodes stopped at 680 games (-3.6 +/- 17.7, LLR -1.05); x = 0.25 at
      1000 nodes stopped at 1160 games (-1.5 +/- 13.9, LLR -1.29): not merged.
- [x] Futility reductions instead of the pruning (Nalwald's, the static eval after the move in place
      of static eval + SEE, `futility_reduction_div`, 58ce4b1, 2026-10-19). Mean depth at bench
      3000n (default 13.05), fractional: div 300 15.50, 500 14.75, 800 13.82, 1200 13.26, 2400
      12.20, 4000 11.41; NPS ~4% lower. Div 800 at 10+0.1: H0, -3.7 +/- 10.6 at 2360 games.
      Div 500 at 10+0.1: H0, -20.4 +/- 18.3 at 562 games. Not merged (the pruning stays).
- [x] The decisions inside a move in the tree again, all types (`inner_width` 3, `is_inner`,
      `tree_candidates`, the rules at the horizon, 58ce4b1): bench depth 13.65 (inner decisions
      count as plies), ~12% lower NPS. SPRT at 10+0.1: H0, -23.1 +/- 19.1 at 724 games. Not
      merged (inner_width stays 0; the code stays, off).
- [x] Main-line horizon extension (`pv_extension`, user's idea, 2026-10-19): on the main line (the
      first move at every node) a decision inside a move (a card selection; the `is_inner` ones if
      searched) is searched one ply deep at the horizon too, so the search finishes the move. Bench
      mean depth 13.05 -> 13.00, same NPS. SPRT at 10+0.1 H1, +25.4 +/- 16.4 at 890 games: default
      since 2026-10-19, bench checksum ba6b42dff5abf893.
  - [x] At every node whose window is wider than the null window (`pv_extension=2`, user): bench
        depth 13.08, same NPS; SPRT vs the main line at 10+0.1 H0, -15.8 +/- 16.3 at 838 games.
  - [x] Every main-line decision costs 0.5 ply (`pv_ply=0.5,pv_move_ply=0.5`): at 3000 nodes the
        iteration depth 13.00 -> 11.69, the main line's height 12.98 -> 22.32, the greatest height
        14.12 -> 24.44. SPRT at 10+0.1 stopped by the user at 2880 games (+1.9 +/- 9.4, LLR -1.35).
  - [x] Only the main line's decisions inside a move cost 0.5 (`pv_move_ply=0.5`): iteration depth
        12.86, main line 14.26, greatest height 15.42. SPRT at 10+0.1 stopped by the user at 2720
        games (+3.2 +/- 9.4, LLR -0.79).
  - [x] What it does (bench, 20 openings, 1239 searches at 3000 nodes): 0.06% of the nodes; it
        fires in 833 searches, extends the last iteration's main line in 150; at 40% of its decided
        nodes the best card is not the rules' first (gains when better: median 400-1500 LP, 3
        wins found); 72% are the root player's own selections. Root moves differ from no extension
        (branch no-pvext) in 70 of 1239 searches, only 9 of them with the last main line extended
        (the rest: earlier iterations' move order, TT and history).
  - [x] On every root move's own main line (branch pvext-root-all): H0, -36.4 +/- 23.5 at 402 games
        (10+0.1). So the gain is the previous best's advantage (its line valued with searched card
        choices, the alternatives with the rules'), not the accuracy of values.
  - [x] A bonus for the previous best root move instead (branch prevbest-bonus, 100 LP, no
        extension) vs main: H0, -6.5 +/- 12.1 at 1712 games. Control: no extension at all (branch
        no-pvext) vs main: H0, -8.6 +/- 13.3 at 1296 games. So the extension is worth less than
        its first SPRT said (+25.4 at 890 games), perhaps ~+5-10; the bonus does not clearly
        replace it.
  - [ ] Next ideas: with `inner_width` 3 on the main line only; the first k moves at main-line nodes.
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
- [x] Extend the first child when it is the TT move and the node has only 1 or 2 children: by 1 ply,
      or by a fraction of a ply (e.g. 0.5; needs fractional depth). `narrowext` (1 ply, up to
      height 2 * the iteration's depth): mean depth at 3000 nodes 12.71 -> 10.13 (in games 10.9 ->
      9.5); SPRT vs the default stopped heading to H0 at 1120 games, -6.5 +/- 14.6, LLR -2.07.
      Untried: the fractional (0.5 ply) version, or only at nodes with exactly 1 child.
      Branches (user, 2026-10-24): `tt-ext` (9eb7d1e: the TT move of a node with at most 2 moves a
      ply deeper; not at the horizon's extension, below twice the iteration's depth) and
      `tt-ext-half` (da61632: half a ply). Bench (6 openings): depth 10 356184 nodes (main) ->
      832600 (+134%) / 557839 (+57%); at 3000 nodes mean nominal depth 13.34 -> 10.40 (tt-ext).
      Two moves are common here (every yes/no, a chain of one card plus pass). tt-ext-half
      (rebased, 7a464b6) SPRT stopped by the user at 1720 games: +0.4 +/- 12.5, LLR -1.13. Untried:
      only at nodes with exactly 1 child.
- [ ] qsearch, if retried: a TT in quiescence, cheaper chain windows (pass plus the best few
      answers), pruning of hopeless attacks.
- Retested at the Rust speed and stopped near zero: PVS (H0), qsearch, futility, conthist,
  PVS+RFP, RFP by kind (stopped at 2220 games, +4.5, LLR -1.50) (all on top of LMR).

## Hidden information

- [x] Different card guesses and deck fillers in each iteration of the iterative deepening (the
      opponent's hidden cards re-sampled per depth), instead of one guess for the whole search.
      Feature `resample` (c0342ea): the guessed cards still hidden get codes drawn from the deck
      model (presence-weighted, copies from the count model) within the observations'
      requirements, ~1.7 ms per draw. TT and history unchanged. Removed in the 2026-10-18 cleanup.
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
- [ ] Deeper / wider MLPs (`--dims`, the Rust eval takes any shape): on rounds 2+3, 2x64 ranked
      +0.007 +/- 0.004 over 1x16 but overfit after a few epochs; SPRT 2x64 stopped at 380 games
      (-7.3). On the SPRT data (2026-10-21, 6x more): MLP 64 H1 +35.8 vs 16 (default); against
      the 64: 32x32 H0 (+0.9 +/- 7.3, 4380 games), 64x32 stopped (-0.0, 1720 games, LLR -1.30),
      8x32 stopped (-4.6, 600 games, LLR -1.09). The net's size costs no NPS (the features dominate).
      Next on this axis: more data rather than more shape.
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
- [x] SPRT games as data (default since 2026-10-21): MLP16 mover2 on the SPRT games with search
      values from the runs started after the round-2+3 net became the default (2026-10-17 23:22
      to 2026-10-21, without vsold-*: 86594 games, 13.2M positions; runs/data/sprt_net/, built by
      collecting the records + ygoai-refeature + an npz with explicit deck pairs) and rounds 2+3,
      without one-sided deck pairs (2544 of 8737), `--held 400`
      (crates/ygoai/models/eval_mover2_sprtr23os_mlp16.json, LP factor 1.105): SPRT H1 +17.2 +/-
      12.6 at 1680 games (LLR +2.97) vs the round-2+3 net. Bench mean depth 13.35, checksum
      44336e139f6541ec. Next: mover3/mover4 and deeper nets on this data (the engine evaluates only
      mover2 since the cleanup), and keep adding SPRT games as data.
  - All SPRT games (user, 2026-10-24): 184522 games since 2026-10-17 23:22 (28.1M positions,
    runs/data/sprt_net2/: scripts/collect_sprt_games.py + ygoai-refeature, which now streams its input (all games at
    once ran out of memory) + scripts/sprt_games_to_npz.py with rounds 2+3 (runs/data/rounds23/all_m2_r23.npz): 30.6M rows, 10395 pairs). MLP 64, --held
    400 --drop-onesided --lam 0.5 (train_eval_mover.py's report now chunked: the GPU ran out at
    30M rows): held-pairs rank 0.7209 (search values 0.7348; other held-out pairs than before, not
    comparable), LP factor 1.021. Branch `eval-sprt2` (models/eval_mover2_sprt2r23os_64.json): SPRT
    stopped at 3960 games, +3.2 +/- 8.8 (LLR -0.92; penta [316, 0, 1330, 0, 334]); merged anyway
    (user: no big effect, unlikely worse), default since e642eb3.
  - Shapes (2026-10-21, mover2 on the same data, `--dims`; the same NPS for all, the features cost
    the time): MLP 64 H1 +35.8 +/- 20.0 at 702 games vs the MLP16, default since 5fd9188
    (models/eval_mover2_sprtr23os_64.json). 32x32 was at +19.8 +/- 17.4 (860 games, LLR +1.89) and
    64x32 had just started when it passed: both stopped, rerun against the 64 with 8x32: 32x32
    H0 (+0.9 +/- 7.3 at 4380 games), 64x32 stopped at 1720 games (-0.0 +/- 12.2, LLR -1.30), 8x32
    (bottleneck: 8 then 32) stopped at 600 games (-4.6 +/- 18.5, LLR -1.09).
  - mover5 (branch `eval-mover5`, 99 features: mover2 and per side the ready attackers' ATK, the
    Normal Summons left this turn (turn player before the End Phase, a free Main Monster Zone, the
    mover a monster in hand to summon, the opponent assumed to have one), and mover4's field / GY
    effects with uses left, second best battle stat, weakest attacker's ATK), MLP16 on the same SPRT
    games and round 3 (round 2 has no moves): H0, -8.6 +/- 13.3 at 1368 games. Same NPS.
- [x] Round 3 (default 2026-10-17 to 2026-10-21): datagen with the round-2 net on runs/books/pairs500x20_c.jsonl
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
- [x] Linear move policy (2026-10-21, branch `policy-linear`, search::policy): a weight per move
      feature (kind one-hot + ATK/DEF/level, attack margins, chain context), trained by
      scripts/train_policy_linear.py on the SPRT games' search choices (ygoai-refeature --features
      policy), orders the moves of the decisions that are not card selections (after TT move and
      history). Held-out top-1 0.594 vs the rules' 0.350 (idle 0.364/0.276, chain 0.747/0.222,
      battle 0.786/0.727, effect yes/no 0.814/0.745). SPRT H0, -27.2 +/- 20.6 at 538 games (penta
      44 lost pairs vs 23 won); not slower (bench depth 13.42 vs 13.35). `policy-tree` (the
      policy inside the tree only, the root in the rules' order, which decides ties and an
      unfinished iteration's move): H0, -4.8 +/- 11.3 at 1888 games; same depth (8.97 vs 9.02 in
      the games). So the root order cost most of the -27, and in the tree the policy buys no depth:
      trained on the search's choices on the game's line, not on what cuts off inside the tree. Retrained without shuffle moves (--drop idle:shuffle): top-1 0.617 vs
      0.366, branch `policy-linear-ns` on noshuffle, untested.
- [x] No hand shuffle (merged b1aec37, user's call): the idle command "shuffle hand" is never
      offered to the search; the search picked it in 10.6% of idle decisions (~4 per game). Bench
      depth 13.73 vs 13.35. SPRT stopped at 5660 games, +2.8 +/- 6.4, LLR -2.02 (a simplification,
      merged as neutral).
- [x] Policy network on the search's inner nodes (2026-10-22, branch `policy-net`): node log
      (`ygoai-datagen --node-log 8`: every 8th node with a policy prompt, position mover2 features,
      move features, history, best move), 6000 games at 2000 nodes per decision from openings
      12000+ (runs/data/policy_nodes/n2000, 8.2M rows). Net: first layer = position part (once
      per node) + move feature embeddings + history inputs, 64,32, per-node softmax on the
      exact-value nodes (cutoffs logged only when the first move failed: biased), after the TT
      move, at depth >= 2. Held-out top-1 at exact nodes 0.623 vs history 0.566, rules 0.419
      (1000 games: 0.609). Node statistics: ~60% of nodes do not cut off, and of the cutting
      ones ~85-95% cut at the first move, so ordering saves few nodes: depth-10 nodes -0.4%;
      cost +2.8% instructions per node after f32 + columns + ReLU-zero skipping (was +3.8%;
      ablation: net arithmetic 2.1%, move features 0.5%). Any gain must come from quality (LMR).
      SPRT H0: -1.1 +/- 8.8 at 3260 games (LLR -3.00). Better top-1 than history buys neither
      nodes nor quality: the order matters at few nodes, and there history is about as good.
- [x] Leaf rollout (user's idea, 2026-10-22, branch `leaf-rollout`): at depth <= 0 the decisions
      up to the next turn are made without search, each the first move of the ordering (TT move,
      history, rules' order), then the static eval. 5.5x the instructions to depth 10; at 0.5 s
      per search mean depth 10.37 vs 13.41. SPRT H0: -132.4 +/- 48.0 at 154 games (LLR -2.96).
      At equal nodes (NODES=2000, the rollouts not counted, 600 games): -47.8 +/- 23.3, so the
      rollout's values are worse, not only its cost: the ordering's first move is what cuts off,
      not a good move to play out a turn with.
- [x] Greedy quiescence (user's idea, 2026-10-22, branch `quiesce-greedy`): at depth <= 0 the
      decision's player stands pat (static eval) or makes the ordering's first move (TT move,
      history, rules' order), the same at the next decisions, up to the next turn; beta cutoff on
      the stand pat. +41% instructions to depth 10. SPRT stopped at 6320 games, +6.5 +/- 6.3,
      LLR +1.42, as contaminated: its gain came in one window, games 3400-5400 (lost/won pairs
      99/159), while eval training also loaded the CPU under 32 games on 32 cores; before (232/233)
      and after (61/58) even. Less time per move seems to favour it. Rerun on a quiet machine
      (2026-10-22, next to the card-policy SPRT): stopped by the user at 2460 games, +1.8 +/- 10.1,
      LLR -1.18 (penta 164 lost / 171 won pairs), for the strict "action under way" horizon.
- [x] Eval training target (user's question, 2026-10-22): the default MLP 64 uses --lam 0.5 (half
      the game result with the deck-pair term, half the search value). Same data and settings:
      `eval-lam0` (results alone; held-out pairs res-ll 0.5306, overfits the training games:
      0.446 vs 0.654 on held-out games) H0, -12.1 +/- 14.9 at 1262 games (LLR -2.98).
      `eval-lam1` (search values alone; held-out pairs res-ll 0.5277) H0, -2.0 +/- 9.4 at 2806
      games (LLR -3.04). The mix stays.
- [x] Card policy (user's idea, 2026-10-22, branch `card-policy`): a small MLP (16 hidden) scores the
      cards of a one-card selection (`search::cardpolicy`) in place of the rules' order by the
      select hint: it decides which cards the search tries (card_width 3, root 5) and the horizon's
      pick. Features: the card's types, attribute, level, ATK/DEF, location, its effects (kinds,
      ranges, the scripts' categories), copies with its player, the rules' rank, the hint, a little
      of the position; no card identity. Data: ygoai-card-refs, every card searched with the full
      window (spsa-style references, 2000 nodes) at 20093 selections of the evalcheck-from12000
      and noshuffle games (18 min); trained by scripts/train_card_policy.py (softmax over cards vs
      softmax(value/500)). Held out (1466 with a spread): best card first 0.482 (rules 0.428), in
      the first 3 0.849 (0.808), first 5 0.932 (0.898); regret of the first 3's best 251 LP (342).
      Hidden 16/32/64, temperature, wd: all within ~0.01 (data-bound). Off (YGOAI_CARD_POLICY=none)
      = main's checksum; on: depth-10 bench 4.6% fewer instructions, +0.9% per node at 3000 nodes
      (depth 13.42 vs 13.34). SPRT H1: +8.5 +/- 6.7 at 5348 games (LLR +3.01), merged (578c0ce).
      Main's depth-10 bench is now 343056 nodes, 331.3G instructions. Next: more card-ref data
      (the model is data-bound).
  - [x] Duel-state features (user, 2026-10-23; branch `card-policy-state`): what asks (the effect at
        work: `core.reason_effect`, else the resolving link; its card's kind, categories, effect
        type; cost or resolution; chain length), the duel (turn, both sides' best battle stat,
        best attacker, face-down monsters, backrow, hand/grave/banished/deck/extra, Normal Summon
        left) and the card against it (the asking card or its name, ATK now, beats the opponent's
        best, could be Normal Summoned, duplicates among the cards): 210 features (137).
        `ygoai-card-refs --refeature` recomputes r1's features (20093 rows, identical old ones).
        On r1 alone no gain (data-bound): held out top1 0.480 / top3 0.855 / regret3 255 (old
        features 0.482 / 0.849 / 251); train fit better. More refs (r2_state, 60k from the SPRT
        records since 2026-10-19) in the gap after the wide test, then retrain and SPRT.
        r2_state: 60061 rows (65 min). r1 + r2 (58644 with a spread; held out every 10th game,
        epoch 60): old features h16 0.495 / 0.845 / 260, state h16 0.493 / 0.853 / 243, state h32
        0.490 / 0.851 / 246 (epoch 50: 0.491 / 0.851 / 246, 0.486 / 0.846 / 259, 0.491 / 0.847 /
        252: the differences are within the epoch-to-epoch noise). Neither the state features nor
        4x the data move the held-out numbers: not SPRTed (models runs/nn/card/cp3_*).
- [x] Horizon by action, not by decision type (user, 2026-10-22, branch `horizon-action`): at the
      horizon `do_move` let the rules answer only card selections, so a yes/no or option inside a
      resolution stopped the move and the eval came mid-effect. `action_under_way` (search::movegen)
      reads the core's unit stack: a chain link resolving (`chain_solving`) or any unit behind
      the question other than Turn/IdleCommand/BattleCommand/PointEvent/PhaseEvent/QuickEffect/
      SolveChain/DamageStep/ForcedBattle/bookkeeping (AddChain = cost/targets, summon procedures,
      DiscardHand, OperationReplace, SolveContinuous ...). At the horizon such decisions are answered
      by the rules, also on the main line, whose extension stays with card selections. The strict
      variant: an open chain's response window still stops. (A first version also extended those
      decisions on the main line, now branch `horizon-action-ext`: stopped at 0 games so that the
      stopping rule is tested alone first, user.) Stacks at searched decisions in 100
      replayed games: idle/battle commands, chain windows and trigger yes/no all on boundary stacks;
      resolution: 847 card / 92 yes-no / 45 option, activation (AddChain) ~500 card / 28 option.
      Bench: depth 10 355577 nodes (main 353436), +1.9% instructions; 3000n depth 13.33 (13.34).
      (horizon-action-ext: 353580 nodes, +1.7%; 13.35.) SPRT against the old main stopped at
      2850 games, +7.6 +/- 9.0 (LLR +1.21), once the card policy was merged: rebased onto it
      (51e423a; at the horizon one-card selections take the policy's first card, inside an action
      too, the rest the rules) and continued against the new main with those 1425 pairs counted
      (user's call, knowingly mixed: RESUME=), 32 at a time: stopped by the user at 8570 games,
      +3.8 +/- 5.2 (LLR -1.76), drifting to H0 (the 2860 new games about even).
  - [x] Wide variant (branch `horizon-action-wide`, 184d9e4, user 2026-10-23): an open chain counts
        as under way too, so at the horizon the rules answer its response windows until it has
        resolved. Bench: depth 10 324175 nodes (main 343056, -5.5%), 353.6G instructions (+6.7%);
        3000n depth 13.43 (13.42). SPRT H0: +1.1 +/- 7.1 at 4392 games (LLR -2.95).
  - [x] Then with the main-line extension: branch `horizon-action-ext2` (c27e70e) = horizon-action
        rebased onto main + the main-line extension of under-way decisions (search one ply deep
        on the main line instead of the rules' answer); against main (the strict variant, since
        wide was even). Bench: depth 10 332434 nodes (main 343056, -3.1%), 324.5G instructions
        (-2.0%); 3000n depth 13.41 (13.42). SPRT H0: -5.5 +/- 11.6 at 1762 games (LLR -3.00).
        With the strict variant drifting to H0 and the wide one H0, the horizon-by-action
        direction is done for now.
- [ ] Eval net forward in f32 / columns (branch `eval-f32`): the same search tree (depth-10
      bench checksum eb7f41e9580dc935 as main then), 1.0% fewer instructions. Too small for an
      SPRT; the user's call whether to merge as a pure speedup (needs a rebase: main's eval changed).
- [x] Chain quiescence (branch `quiesce-chain`): at depth <= 0 with a chain open, the rules
      resolve the chain before the static eval. Depth-10 nodes -2.2%, +12.9% instructions per
      node. SPRT H0: -6.5 +/- 12.2 at 1436 games (LLR -2.98).

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
      avoids such lines (9c3af1d); decide whether to guard it in the core too. It also ends whole
      games outside the search: the 2 candidates lost in the match2 book build (processor3.rs:387,
      seeds 91070843 and 91125503 of runs/data/onesided/book2_kept.jsonl reproduce it in 1 s with
      ygoai-book-test -j 1).
- [ ] Seed the engine processes in matches (pass the seed to `ygoai-engine`), so a crashed
      game can be reproduced exactly.

## Housekeeping

- [ ] Repo cleanup: what is still needed of `ygoai/` (Python), `ygoai_rs/` and the other folders.
  - Branch `py-cleanup` (fd916cf + 82e70a4, 2026-10-25, for the user's review, not merged): the
    Python play stack the Rust engine replaced removed (-3934 lines: the Python alpha-beta, MCTS,
    PUCT, policy and heuristic agents, the Python engine process and protocol, match play,
    scripts/match.py, sprt.py, make_match_book.py, the RL self-play scripts and nets, the Python
    deck completions, tests/test_integration.py; README and doc comments updated). Kept: the
    belief checks' agents (random, greedy, setter, mixed, belief_mc) and modules, suite.py and
    its tools, the data and training scripts. Every module and the scripts' imports load; the
    unit tests pass; check_belief.py 36/36 on the suite's setter/mixed pairs.
  - Eval data tools were only in a temporary directory: now scripts/collect_sprt_games.py and
    scripts/sprt_games_to_npz.py, the rounds 2+3 npz in runs/data/rounds23/ (done on main).
- [x] Remove the worktrees `wt-old` (main repo) and `wt-old-rust` (rust repo) in the scratchpad (gone).
- [x] Push the rust submodule commits 2b7f988, 1354870, 935783d, 658b359 (on origin/main).
- [x] Resumable SPRTs (idea, 2026-10-22; done 04fa423): `ygoai-match --resume RECORDS` (the finished pairs of the
      records into the pentanomial, their openings skipped, same shuffle seed, records appended) and
      `RESUME=<log name>` in sprt_branch.sh (the old run's seed and binaries). Only under the same
      conditions (commits, tc, no load).
