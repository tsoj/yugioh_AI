# The SPRT opening book

How runs/books/match2.jsonl was built: deck pairs with their shuffles ("openings") that the seat
does not decide too often. Each step's usage is in its script's docstring.

1. Candidates: `onesided_candidates.py generate` draws random openings from the deck corpus.
2. (Optional, it gained ~3 points of seat-decided pairs at most) keep the half that the one-sided
   predictor finds least one-sided:
   - card text embeddings: `build_effect_data.py` (what effects do in the SPRT records) ->
     `train_effect_model.py` (the encoder) -> `embed_cards.py` (the candidates' cards);
     `card_text.py` renders the card texts;
   - the predictor: `build_onesided_data.py` (per opening of the SPRT records, how often the seat
     decided it) -> `train_onesided.py`;
   - `onesided_candidates.py score` keeps the less one-sided candidates.
3. Test games: `bot-book-test CANDIDATES runs/books/match2_tested.jsonl --policies ...` plays each
   candidate with the RL policy nets (runs/nn/rl1-rl4, boot1: given, their training was removed
   with the Python play stack) and writes the wins of each seat.
4. The book: `select_openings.py runs/books/match2_tested.jsonl runs/books/match2.jsonl` keeps the
   openings in which each seat won at least 2 test games (54,590 of 64,998), without the test
   results. SPRTs draw from its first 42,049 openings (`scripts/sprt_branch.sh`: POOL).

Both files are in git as `books/match2.jsonl.zst` and `books/match2_tested.jsonl.zst`.

## The era books (OpenBench: `books/openbench/era_sprt.zip`, `era_sprt2.zip`)

Same-era deck pairs, so that fewer openings are one-sided.

1. Candidates: `era_candidates.py OUT --part sprt -n N --seed S`: random deck pairs whose eras (a
   deck's newest card's first release, `card_dates.py`) are at most 365 days apart. Each deck is
   in one part, sprt (16% of the decks) or data: the SPRT books' decks are never in training data.
2. Test games: `bot-book-test CANDIDATES TESTED --policies
   runs/nn/rl4/iter71.npz,runs/nn/rl3/iter100.npz,runs/nn/rl2/iter95.npz,runs/nn/rl1/iter28.npz,runs/nn/boot1_policy.npz -j 30`.
3. The book: `select_openings.py TESTED [TESTED ...] BOOK` (each seat won at least 2 test games;
   each deck pair once).

- `era_sprt`: 24,000 candidates (seed 92), 18,224 openings (63 deck pairs twice).
- `era_sprt2`: `select_openings.py runs/books/era/tested_sprt.jsonl runs/books/era/tested_sprt2.jsonl`,
  the second file 110,000 candidates of seed 94: 100,072 openings, each deck pair once. Its first
  18,161 openings are `era_sprt`'s, without the repeated pairs.
