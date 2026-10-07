#!/usr/bin/env python3
"""One-axis-at-a-time hyperparameter sweep for the deck trainers.

Trains a baseline plus a small set of variants where one hyperparameter
at a time is perturbed around the baseline.  Each run is short (4
epochs by default) with one mid-epoch val measurement.

Pick which trainer to sweep with --trainer {existence,counts}.  Both
trainers share the same CLI surface (--lr/--d-model/--hidden/
--n-trunk-layers/--batch-size/--epochs/--metrics-out/--save-path/
--mid-epoch-vals), so the sweep is identical except for which script
it invokes and where the results land.

Per-run metrics land in --out-dir as JSON; existing JSONs are reused so
the sweep is resumable.  At the end we print a comparison table and a
crude 1/n extrapolation of val loss to spot configs that are still
improving fast.

Usage:
    ./sweep_hyperparams.py --trainer existence
    ./sweep_hyperparams.py --trainer counts --epochs 6
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Baseline matches the CLI defaults of train_deck_existence.py and
# train_deck_counts.py (which are the same).
BASE = {
    "lr": 1e-3,
    "d_model": 384,
    "hidden": 512,
    "n_trunk_layers": 3,
    "batch_size": 32,
}

# For each axis, the off-baseline values to try.  Keep it small — this is
# meant as a first-pass survey, not a real search.
SWEEP = {
    "lr": [5e-4, 2e-3],
    "d_model": [256, 512],
    "hidden": [256, 1024],
    "n_trunk_layers": [2, 4],
    "batch_size": [16, 64],
}

CLI_FLAG = {
    "lr": "--lr",
    "d_model": "--d-model",
    "hidden": "--hidden",
    "n_trunk_layers": "--n-trunk-layers",
    "batch_size": "--batch-size",
}

TRAINER_SCRIPTS = {
    "existence": "train_deck_existence.py",
    "counts": "train_deck_counts.py",
}

# Extra flags to forward per trainer.  For existence, we want to skip the
# (expensive) fixed-k eval during training; pushing the min-gap above the
# epoch count makes the inner-loop eval never fire.  The trainer still
# runs a final eval on the best model at the end.
def trainer_extra_args(trainer, epochs):
    if trainer == "existence":
        return ["--fixed-k-eval-min-gap", str(epochs + 1)]
    return []


# Non-CLI keys that can appear in an explicit-config dict: these override
# sweep-level defaults instead of being forwarded as --flag.
META_KEYS = {"epochs"}


def configs():
    """Yield (slug, cfg) — baseline + one-axis variants."""
    yield "baseline", dict(BASE)
    for axis, values in SWEEP.items():
        for v in values:
            cfg = dict(BASE)
            cfg[axis] = v
            yield f"{axis}_{v}", cfg


def extrapolate_asymptote(val_losses):
    """Fit val_loss = a + b/n by least squares, return (a, val_at_n=20)."""
    n = len(val_losses)
    if n < 3:
        return float("nan"), float("nan")
    xs = [1.0 / (i + 1) for i in range(n)]
    ys = list(val_losses)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    return a, a + b / 20.0


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer", choices=sorted(TRAINER_SCRIPTS),
                    required=True,
                    help="which trainer script to sweep")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to sweep_results_<trainer>")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mid-epoch-vals", type=int, default=1)
    ap.add_argument("--train-script", default=None,
                    help="override the trainer script path "
                         "(defaults to the canonical one for --trainer)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run, don't execute")
    args = ap.parse_args()

    train_script = (Path(args.train_script) if args.train_script
                    else here / TRAINER_SCRIPTS[args.trainer])
    out_dir = Path(args.out_dir) if args.out_dir \
        else here / f"sweep_results_{args.trainer}"
    out_dir.mkdir(exist_ok=True, parents=True)

    runs = list(configs())
    print(f"sweep [{args.trainer}]: {len(runs)} runs, "
          f"{args.epochs} epochs each → {out_dir}")

    for slug, cfg in runs:
        metrics_path = out_dir / f"{slug}.json"
        if metrics_path.exists():
            print(f"[skip] {slug} (output exists)")
            continue
        epochs = cfg.get("epochs", args.epochs)
        cmd = [
            sys.executable, str(train_script),
            "--epochs", str(epochs),
            "--mid-epoch-vals", str(args.mid_epoch_vals),
            "--metrics-out", str(metrics_path),
            "--save-path", "",
        ]
        cmd += trainer_extra_args(args.trainer, epochs)
        for k, v in cfg.items():
            if k in META_KEYS:
                continue
            cmd += [CLI_FLAG[k], str(v)]
        print(f"\n=== {slug} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[fail] {slug} exited {rc}", file=sys.stderr)

    if args.dry_run:
        return

    # Aggregate
    results = []
    for slug, cfg in runs:
        metrics_path = out_dir / f"{slug}.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        results.append((slug, cfg, m))

    if not results:
        print("no results to aggregate", file=sys.stderr)
        return

    print(f"\n\n=== SWEEP SUMMARY [{args.trainer}] ===")
    header = (f"{'config':<22} {'mid_e0':>8} {'val_e0':>8} {'val_eN':>8} "
              f"{'best':>8} {'asym':>8} {'extrap20':>9} {'sec/ep':>8}")
    print(header)
    print("-" * len(header))
    rows = []
    for slug, cfg, m in results:
        epochs = m["epochs"]
        if not epochs:
            continue
        vals = [e["val_loss"] for e in epochs]
        mid_e0 = (epochs[0]["mid_vals"][0]["val_loss"]
                  if epochs[0].get("mid_vals") else float("nan"))
        asym, extrap20 = extrapolate_asymptote(vals)
        sec = sum(e["wall_time"] for e in epochs) / len(epochs)
        rows.append((slug, mid_e0, vals[0], vals[-1], min(vals),
                     asym, extrap20, sec))

    rows.sort(key=lambda r: r[6])  # sort by extrapolated val
    for slug, mid, v0, vN, best, asym, ex20, sec in rows:
        print(f"{slug:<22} {mid:>8.4f} {v0:>8.4f} {vN:>8.4f} "
              f"{best:>8.4f} {asym:>8.4f} {ex20:>9.4f} {sec:>8.1f}")

    print("\nLegend:")
    print("  mid_e0   = val loss at midpoint of epoch 0 (early signal)")
    print("  val_e0   = val loss at end of epoch 0")
    print("  val_eN   = val loss at end of final epoch")
    print("  best     = best val loss across all measured points")
    print("  asym     = a in fit val ≈ a + b/n  (extrapolated asymptote)")
    print("  extrap20 = fit's predicted val at epoch 20")
    print("  sec/ep   = wall time per epoch")


if __name__ == "__main__":
    main()
