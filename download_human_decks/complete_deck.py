#!/usr/bin/env python3
"""Complete a deck given a partial set of observed cards.

Two-stage pipeline:
  1. Existence model (deck_existence.pt) — predicts which cards are in
     the full deck given the observed visible subset.
  2. Count model (deck_counts.pt) — given the set of cards in the deck,
     predicts how many copies of each (1, 2, or 3).

Set selection (resolves the chicken-and-egg between |set| and counts):
  1. Compute the corpus-wide average copies-per-unique-card from the
     human .ydk decks (or override with --avg-copies).
  2. Forward the existence model on the observed cards → P(presence).
  3. Start K = round(target_total / avg_copies) as the guess for the
     total number of unique cards in the deck.
  4. Build candidate set: observed cards + top-(K - num_observed_unique)
     non-observed cards by P(presence).  Run the count model and
     bucketize to {1, 2, 3}; observed cards are forced to keep at least
     their observed count.
  5. If the resulting total < target, K += 1.  If > target, K -= 1.
     Stop on exact match, on a repeat K, or on the second oscillation
     (over→under→over→under).
  6. If no K produced an exact match, pick the K whose count-model
     output is cheapest to *force* onto target_total — cost = sum of
     squared deviations between the model's raw output and the bucket
     value of the adjusted count.  Adjustment is greedy: at each step
     bump/drop the card whose squared-error delta is smallest.
  7. Per section (main vs extra), greedy-fill in P(presence) order with
     observed cards forced in first.  The last card may be truncated to
     fit the section budget.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

from deck_common import parse_ydk_deck
from train_deck_counts import CountModel
from train_deck_existence import ExistenceModel


def parse_observed(spec):
    """Return a list of card codes from either a .ydk path or a comma list."""
    p = Path(spec)
    if p.exists():
        return parse_ydk_deck(p)
    return [int(x) for x in spec.split(",") if x.strip()]


def compute_avg_copies_per_card(decks_dir, vocab, sample=2000):
    """Mean of (total copies / unique cards) across human decks."""
    paths = sorted(Path(decks_dir).glob("*.ydk"))[:sample]
    ratios = []
    for p in paths:
        cards = [vocab[c] for c in parse_ydk_deck(p) if c in vocab]
        if not cards:
            continue
        cnt = Counter(cards)
        if max(cnt.values()) > 3:
            continue
        ratios.append(sum(cnt.values()) / len(cnt))
    if not ratios:
        return 1.6  # fallback when --decks-dir is unavailable
    return sum(ratios) / len(ratios)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _existence_tokens(c_vis, device):
    """Pack the dense visible-count vector into (1, L) tokens."""
    nz = (c_vis > 0).nonzero(as_tuple=True)[0].tolist()
    L = max(1, len(nz))
    card_ids = torch.zeros(1, L, dtype=torch.long, device=device)
    card_counts = torch.zeros(1, L, dtype=torch.long, device=device)
    kpm = torch.ones(1, L, dtype=torch.bool, device=device)
    for i, j in enumerate(nz):
        card_ids[0, i] = j
        card_counts[0, i] = int(c_vis[j].item())
        kpm[0, i] = False
    return card_ids, card_counts, kpm


def _count_tokens(set_indices, device):
    """Pack a list of card indices into (1, L) tokens for the count model."""
    L = max(1, len(set_indices))
    card_ids = torch.zeros(1, L, dtype=torch.long, device=device)
    kpm = torch.ones(1, L, dtype=torch.bool, device=device)
    for i, j in enumerate(set_indices):
        card_ids[0, i] = j
        kpm[0, i] = False
    return card_ids, kpm


# ---------------------------------------------------------------------------
# Model forwards
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_presence(ex_model, c_vis, slots_in_deck, device):
    """Return (V,) presence probabilities given the observed counts."""
    card_ids, card_counts, kpm = _existence_tokens(c_vis, device)
    slots = torch.tensor([float(slots_in_deck)], device=device)
    logits = ex_model(card_ids, card_counts, kpm, slots)  # (1, V)
    return torch.sigmoid(logits.squeeze(0))


@torch.no_grad()
def predict_counts(cnt_model, set_indices, device):
    """Return (raw, discrete) — `raw` is the expected count under the
    predicted class distribution (a float in [1, 3]); `discrete` is the
    argmax class + 1 ∈ {1, 2, 3}.  Both lists are aligned with set_indices.

    `raw` exists so downstream adjustment code can rank cards by how
    confidently the model wants each one at a given count — cards whose
    expected count is close to an adjacent integer are the cheapest to
    bump up or down."""
    if not set_indices:
        return [], []
    card_ids, kpm = _count_tokens(set_indices, device)
    logits = cnt_model(card_ids, kpm).squeeze(0)[:len(set_indices)]  # (L, 3)
    probs = torch.softmax(logits, dim=-1)
    counts = torch.tensor([1.0, 2.0, 3.0], device=device)
    raw = (probs * counts).sum(dim=-1)                               # (L,) ∈ [1, 3]
    discrete = (logits.argmax(dim=-1) + 1).tolist()
    return raw.tolist(), discrete


# ---------------------------------------------------------------------------
# Adjustment: force counts to sum exactly to target_total
# ---------------------------------------------------------------------------

def force_total(cand_indices, raw_pred, discrete, c_vis, target_total):
    """Greedy +/-1 adjustments to make sum(counts) == target_total.

    Returns (adjusted_discrete, cost) where cost is the squared deviation
    between raw_pred (expected count) and the final integer counts.
    Visible cards stay >= observed count and counts are clamped to [1, 3].
    """
    current = list(discrete)
    total = sum(current)
    vmin = [max(1, int(c_vis[j].item())) for j in cand_indices]

    def delta_cost(i, sign):
        # Squared-error change for moving current[i] by sign ∈ {-1, +1}.
        old = (raw_pred[i] - current[i]) ** 2
        new = (raw_pred[i] - (current[i] + sign)) ** 2
        return new - old

    while total != target_total:
        sign = -1 if total > target_total else +1
        best_i, best_delta = -1, float("inf")
        for i in range(len(cand_indices)):
            if sign > 0 and current[i] >= 3:
                continue
            if sign < 0 and current[i] <= vmin[i]:
                continue
            d = delta_cost(i, sign)
            if d < best_delta:
                best_delta = d
                best_i = i
        if best_i < 0:
            break  # can't adjust any further (everyone clamped)
        current[best_i] += sign
        total += sign

    cost = sum((raw_pred[i] - current[i]) ** 2
               for i in range(len(cand_indices)))
    return current, cost


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

@torch.no_grad()
def complete_deck(ex_model, cnt_model, vocab, is_extra, observed_ids,
                  main_size, extra_size, avg_copies, max_iter, device,
                  verbose=False):
    V = len(vocab)
    inv_vocab = {i: c for c, i in vocab.items()}
    is_extra_t = torch.tensor(is_extra, dtype=torch.bool, device=device)

    # Observed → c_vis
    c_vis = torch.zeros(V, dtype=torch.long, device=device)
    skipped = []
    for cid in observed_ids:
        j = vocab.get(cid)
        if j is None:
            skipped.append(cid)
            continue
        if c_vis[j] < 3:
            c_vis[j] += 1

    main_obs = int(c_vis[~is_extra_t].sum().item())
    extra_obs = int(c_vis[is_extra_t].sum().item())
    if main_obs > main_size:
        sys.exit(f"observed {main_obs} main-deck cards but "
                 f"--main-size is {main_size}")
    if extra_obs > extra_size:
        sys.exit(f"observed {extra_obs} extra-deck cards but "
                 f"--extra-size is {extra_size}")

    target_total = main_size + extra_size

    # 1) Existence
    ex_model.eval()
    presence = predict_presence(ex_model, c_vis, target_total, device)

    # 2) Sort non-visible candidates by P(presence) descending
    nv_mask = (c_vis == 0)
    p_masked = presence.clone()
    p_masked[~nv_mask] = -1.0
    n_nv = int(nv_mask.sum().item())
    sorted_nv = (p_masked.topk(n_nv).indices.tolist() if n_nv > 0 else [])

    visible_indices = (c_vis > 0).nonzero(as_tuple=True)[0].tolist()

    # 3) Initial K guess
    K_init = max(1, round(target_total / max(avg_copies, 1e-6)))
    K_max = len(visible_indices) + len(sorted_nv)
    K_init = max(K_init, len(visible_indices))
    K_init = min(K_init, K_max)

    # 4) Iterative search for K
    cnt_model.eval()
    history = {}   # K -> (cand_indices, raw_pred, discrete, total)
    K = K_init
    prev_sign = 0
    osc_count = 0

    for _ in range(max_iter):
        if K in history or K < 1 or K > K_max:
            break
        n_nv_use = max(0, K - len(visible_indices))
        n_nv_use = min(n_nv_use, len(sorted_nv))
        cand = visible_indices + sorted_nv[:n_nv_use]
        raw_pred, discrete = predict_counts(cnt_model, cand, device)
        adj = [max(c, int(c_vis[j].item())) for j, c in zip(cand, discrete)]
        total = sum(adj)
        history[K] = (cand, raw_pred, adj, total)
        if verbose:
            print(f"  K={K}: total={total} (diff {total - target_total:+d})",
                  file=sys.stderr)
        if total == target_total:
            break
        new_sign = 1 if total > target_total else -1
        if prev_sign and new_sign != prev_sign:
            osc_count += 1
            if osc_count >= 2:
                break
        prev_sign = new_sign
        K += -1 if total > target_total else 1

    # 5) Pick best K — prefer exact match (smallest K), else greedy adjustment.
    exact = [(K_, v) for K_, v in history.items() if v[3] == target_total]
    if exact:
        exact.sort(key=lambda x: x[0])
        chosen_K, (final_cand, _, final_discrete, _) = exact[0]
        if verbose:
            print(f"  chose K={chosen_K} (exact match)", file=sys.stderr)
    else:
        best_cost = float("inf")
        best = None
        chosen_K = None
        for K_, (cand, raw_pred, adj, _total) in history.items():
            adj_new, cost = force_total(cand, raw_pred, adj, c_vis, target_total)
            if cost < best_cost:
                best_cost = cost
                best = (cand, adj_new)
                chosen_K = K_
        if best is None:
            sys.exit("error: search produced no valid candidates")
        final_cand, final_discrete = best
        if verbose:
            print(f"  chose K={chosen_K} after adjustment (cost={best_cost:.3f})",
                  file=sys.stderr)

    # 6) Per-section fill.  Phase A: place cards in P(presence) order
    # (visible first), truncating per section.  Phase B: if a section is
    # under-filled (the count model is unaware of the main/extra split),
    # bump existing section cards' counts toward 3 in raw-pred order, then
    # add fresh section candidates from sorted_nv until full.
    main_counts, extra_counts = {}, {}
    main_total, extra_total = 0, 0
    per_card = dict(zip(final_cand, final_discrete))
    raw_by_idx = dict(zip(final_cand,
                          history[chosen_K][1] if chosen_K is not None else final_discrete))
    visible_set = set(visible_indices)
    nv_in_order = [j for j in final_cand if j not in visible_set]

    def place(j):
        nonlocal main_total, extra_total
        cnt = per_card[j]
        if bool(is_extra_t[j].item()):
            room = extra_size - extra_total
            c = max(0, min(cnt, room))
            if c > 0:
                extra_counts[j] = c
                extra_total += c
        else:
            room = main_size - main_total
            c = max(0, min(cnt, room))
            if c > 0:
                main_counts[j] = c
                main_total += c

    for j in visible_indices:
        place(j)
    for j in nv_in_order:
        if main_total >= main_size and extra_total >= extra_size:
            break
        place(j)

    def balance(counts, current, target, section_is_extra):
        if current >= target:
            return current
        # Phase B1: bump cards already in section, prefer ones with highest raw.
        already = [j for j in counts
                   if bool(is_extra_t[j].item()) == section_is_extra]
        already.sort(key=lambda j: -raw_by_idx.get(j, 0.0))
        for j in already:
            while counts[j] < 3 and current < target:
                counts[j] += 1
                current += 1
            if current >= target:
                return current
        # Phase B2: add fresh section candidates by P(presence).
        for j in sorted_nv:
            if current >= target:
                break
            if j in counts:
                continue
            if bool(is_extra_t[j].item()) != section_is_extra:
                continue
            room = target - current
            c = min(3, room)
            counts[j] = c
            current += c
        return current

    main_total = balance(main_counts, main_total, main_size, False)
    extra_total = balance(extra_counts, extra_total, extra_size, True)
    if verbose:
        print(f"  final main={main_total}/{main_size}  "
              f"extra={extra_total}/{extra_size}", file=sys.stderr)

    main_cards, extra_cards = [], []
    for j, n in main_counts.items():
        main_cards.extend([inv_vocab[j]] * n)
    for j, n in extra_counts.items():
        extra_cards.extend([inv_vocab[j]] * n)
    return main_cards, extra_cards, skipped


def write_ydk(out, main_cards, extra_cards):
    lines = ["#created by complete_deck.py", "#main"]
    lines.extend(str(c) for c in main_cards)
    lines.append("#extra")
    lines.extend(str(c) for c in extra_cards)
    lines.append("!side")
    text = "\n".join(lines) + "\n"
    if out == "-":
        sys.stdout.write(text)
    else:
        Path(out).write_text(text)


def load_model(model_cls, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_cls(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    ap.add_argument("--observed", required=True,
                    help="path to a partial .ydk OR comma-separated card codes")
    ap.add_argument("--existence-model", default=str(here / "deck_existence.pt"),
                    help="path to the trained existence checkpoint")
    ap.add_argument("--count-model", default=str(here / "deck_counts.pt"),
                    help="path to the trained count checkpoint")
    ap.add_argument("--decks-dir", default=str(here / "decks"),
                    help="human-deck corpus used to compute the corpus-wide "
                         "average copies-per-unique-card (ignored if "
                         "--avg-copies is set)")
    ap.add_argument("--avg-copies", type=float, default=0.0,
                    help="override the computed average copies/unique-card. "
                         "0 = compute from --decks-dir")
    ap.add_argument("--main-size", type=int, default=40)
    ap.add_argument("--extra-size", type=int, default=15)
    ap.add_argument("--max-iter", type=int, default=60,
                    help="hard cap on K-search iterations")
    ap.add_argument("--output", default="-",
                    help="path for the completed .ydk, or - for stdout")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-K search trace to stderr")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # weights_only=False is needed because the checkpoint also carries the
    # vocab dict.  Only load checkpoints you trust.
    ex_model, ex_ckpt = load_model(ExistenceModel, args.existence_model, args.device)
    cnt_model, cnt_ckpt = load_model(CountModel, args.count_model, args.device)
    if ex_ckpt["vocab"] != cnt_ckpt["vocab"]:
        sys.exit("error: existence and count checkpoints have different vocabs")
    if ex_ckpt["is_extra"] != cnt_ckpt["is_extra"]:
        sys.exit("error: existence and count checkpoints have different is_extra flags")
    vocab = ex_ckpt["vocab"]
    is_extra = ex_ckpt["is_extra"]

    avg_copies = args.avg_copies
    if avg_copies <= 0:
        avg_copies = compute_avg_copies_per_card(args.decks_dir, vocab)
        print(f"corpus avg copies per unique card = {avg_copies:.3f}",
              file=sys.stderr)

    observed = parse_observed(args.observed)
    print(f"observed {len(observed)} cards", file=sys.stderr)

    main_cards, extra_cards, skipped = complete_deck(
        ex_model, cnt_model, vocab, is_extra, observed,
        main_size=args.main_size,
        extra_size=args.extra_size,
        avg_copies=avg_copies,
        max_iter=args.max_iter,
        device=args.device,
        verbose=args.verbose,
    )

    if skipped:
        preview = ", ".join(str(c) for c in skipped[:5])
        more = f" (+{len(skipped) - 5} more)" if len(skipped) > 5 else ""
        print(f"warning: {len(skipped)} observed cards out-of-vocab "
              f"and dropped: {preview}{more}", file=sys.stderr)

    write_ydk(args.output, main_cards, extra_cards)
    print(f"completed: {len(main_cards)} main + {len(extra_cards)} extra",
          file=sys.stderr)


if __name__ == "__main__":
    main()
