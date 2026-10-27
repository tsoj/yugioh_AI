# Deck completion models

The human deck lists and the learned deck completion (the engine's `--completion neural`):

```bash
python download_decks.py                  # the deck lists into decks/
python filter_decks.py                    # unknown cards or bad sizes into decks_invalid/
python train_deck_existence.py --epochs 200 --fixed-k-eval-min-gap 10
python train_deck_counts.py --epochs 200
python export_deck_models.py              # onnx_export/: what crates/deckmodels compiles in
```

`deck_common.py` reads the deck lists; `complete_deck.py` is the models' completion (the Rust
engine's `belief/neural.rs` does the same on the export).
