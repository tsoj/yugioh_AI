#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$SCRIPT_DIR/output"

mkdir -p "$OUTPUT_DIR"

for seed in "$SCRIPT_DIR"/*.ydk; do
    name="$(basename "$seed" .ydk)"
    out="$OUTPUT_DIR/${name}.ydk"
    echo "completing $name ..."
    python3 "$REPO_DIR/complete_deck.py" \
        --observed "$seed" \
        --output "$out" \
        "$@"
    echo "  -> $out"
done
