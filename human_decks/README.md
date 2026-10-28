# Deck completion models

The human deck lists and the learned deck completion (the bot's `--completion neural`):

```bash
python download_decks.py                  # the deck lists into decks/
python filter_decks.py                    # unknown cards or bad sizes into decks_invalid/
python train_deck_existence.py --d-model 128 --hidden 256 --batch-size 128 --lr 4e-3 \
    --lr-schedule cosine --warmup-steps 2000 --epochs 80 --fixed-k-eval-min-gap 1000 \
    --emb-buckets 2000:128,6000:64,0:32 --card-features
python train_deck_counts.py --shared-table deck_existence.pt --d-model 64 --hidden 256 \
    --n-trunk-layers 2 --batch-size 128 --lr 2e-3 --lr-schedule cosine --epochs 30 --count-bias \
    --qat-bits 6
python export_deck_models.py --bits-for existence.card_table.parts.1.weight=4 \
    --bits-for existence.card_table.parts.2.weight=4   # deck_models.bin: what crates/deckmodels embeds
```

The counts model computes on the existence model's card table, so it is trained after it, on it.
`eval_deck_models.py` measures the models (`--bits 6 --bits-for card_table.parts.1.weight=4
--bits-for card_table.parts.2.weight=4 --min-cards 40` for the weights as exported, on decks of the
bot's sizes).

`deck_common.py` reads the deck lists; `complete_deck.py` is the models' completion (the Rust
bot's `belief/neural.rs` does the same on the export).
