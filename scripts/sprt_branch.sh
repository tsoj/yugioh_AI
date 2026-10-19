#!/bin/bash
# SPRT of the Rust engine at one git ref against another (OpenBench/fishtest style: a feature is a
# branch, its defaults are the feature; main is what passed).
#
#   scripts/sprt_branch.sh NEW [BASE]        (BASE defaults to main)
#
# Each ref is built once into build/engines/<sha>/ (ygoai-engine, ygoai-match): the working tree
# is checked out at the ref for the build (tracked files must be committed; submodules stay as
# they are) and back afterwards. The match uses BASE's ygoai-match. Log and records go to
# runs/sprt/<name>_<time>.*, a summary line to runs/sprt/results.jsonl.
#
# Environment: TC (default 10+0.1); NODES=N for fixed-node searches (YGOAI_SEARCH_NODES, --tc
# 60+1); CONC (16 games at once); OPENINGS (5000); ELO ("0 10"); NAME (default NEW); SHUF (the
# book's shuffle seed, random by default); BUILD_ONLY=1 builds and stops.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
NEW=$1
BASE=${2:-main}
TC=${TC:-10+0.1}
CONC=${CONC:-16}
OPENINGS=${OPENINGS:-5000}
ELO=${ELO:-0 10}
NAME=${NAME:-$(echo "$NEW" | tr '/' '-')}
SHUF=${SHUF:-$(od -An -N4 -tu4 /dev/urandom | tr -d " ")}

# The binaries of `ref` in build/engines/<sha>/ (built if missing); prints the directory.
build() {
    local sha dir orig
    sha=$(git rev-parse --short=10 "$1^{commit}")
    dir=build/engines/$sha
    if [ ! -x "$dir/ygoai-engine" ] || [ ! -x "$dir/ygoai-match" ]; then
        if ! git diff --quiet --ignore-submodules || ! git diff --cached --quiet --ignore-submodules; then
            echo "uncommitted changes to tracked files: commit or stash them first" >&2
            exit 1
        fi
        orig=$(git symbolic-ref -q --short HEAD || git rev-parse HEAD)
        trap 'git checkout -q "$orig"' EXIT
        git checkout -q --detach "$sha"
        echo "$(date) building $1 ($sha)" >&2
        systemd-run --user --scope -q -p MemoryMax=28G -p MemorySwapMax=0 nice -n 10 \
            env CARGO_TARGET_DIR="$ROOT/rust/target" CARGO_BUILD_JOBS=6 \
            cargo build --release -p ygoai --bin ygoai-engine --bin ygoai-match >&2
        mkdir -p "$dir"
        cp rust/target/release/ygoai-engine rust/target/release/ygoai-match "$dir/"
        git checkout -q "$orig"
        trap - EXIT
    fi
    echo "$dir"
}

new_dir=$(build "$NEW")
base_dir=$(build "$BASE")
[ "${BUILD_ONLY:-0}" = 1 ] && { echo "$new_dir $base_dir"; exit 0; }

new_sha=$(basename "$new_dir")
base_sha=$(basename "$base_dir")
env_prefix=""
if [ -n "${NODES:-}" ]; then
    env_prefix="env YGOAI_SEARCH_NODES=$NODES "
    TC=60+1
fi
T=$(date +%Y%m%d-%H%M%S)
log=runs/sprt/${NAME}_$T.log
mkdir -p runs/sprt
echo "$(date) start $NAME: $NEW ($new_sha) vs $BASE ($base_sha), tc $TC${NODES:+, $NODES nodes} -> $log"
"$base_dir/ygoai-match" --engine "new=$env_prefix$new_dir/ygoai-engine alphabeta" --engine "base=$env_prefix$base_dir/ygoai-engine alphabeta" \
    --book runs/books/match1.jsonl --pool 10000 --shuffle "$SHUF" -n "$OPENINGS" -c "$CONC" --tc "$TC" --sprt $ELO \
    --records "runs/sprt/${NAME}_$T.jsonl" > "$log" 2>&1 || true
final=$(grep -E "^(SPRT finished|final:)" "$log" | tr '\n' ' ' || true)
[ -z "$final" ] && final="stopped: $(grep "games" "$log" | tail -1)"
python3 -c "import json,sys; a=sys.argv; print(json.dumps({'new': a[2], 'new_sha': a[3], 'base': a[4], 'base_sha': a[5], 'agent': {'new': 'alphabeta', 'base': 'alphabeta'}, 'tc': a[6], 'nodes': int(a[7]) if a[7] else None, 'elo': [float(x) for x in a[8].split()], 'verdict': a[1], 'engine': 'rust', 'shuffle': int(a[9]), 'pool': 10000}))" \
    "$final" "$NAME" "$new_sha" "$BASE" "$base_sha" "$TC" "${NODES:-}" "$ELO" "$SHUF" >> runs/sprt/results.jsonl
echo "$(date) done $NAME: $final"
