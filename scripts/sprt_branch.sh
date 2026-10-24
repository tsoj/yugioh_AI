#!/bin/bash
# SPRT of the Rust engine at one git ref against another (OpenBench/fishtest style: a feature is a
# branch, its defaults are the feature; main is what passed).
#
#   scripts/sprt_branch.sh NEW [BASE]        (BASE defaults to main)
#
# Each ref is built once into build/engines/<sha>/ (ygoai-engine, ygoai-match, ygoai-bench-search), in a worktree of
# its own (build/bench-tree, target build/bench-target; its submodules at the ref's commits, the
# deck models the checkout's): the checkout itself is never switched. Each build's engines run with
# YGOAI_ROOT=build/engines/<sha>/root: the ref's lua_fixes/, the checkout's data. The match uses
# BASE's ygoai-match. Log and records go to runs/sprt/<name>_<time>.*, a summary line to
# runs/sprt/results.jsonl.
#
# Environment: ALLOW_DIVERGED=1 (NEW need not contain BASE); TC (default 10+0.1); NODES=N for fixed-node searches (YGOAI_SEARCH_NODES, --tc
# 60+1); CONC (16 games at once); OPENINGS (10000: 20000 games); ELO ("0 5"; "none": all OPENINGS, no SPRT);
# NAME (default NEW); SHUF (the book's shuffle seed, random by default); BOOK (runs/books/match2.jsonl)
# and POOL (50000: the openings drawn from are the book's first POOL); BUILD_ONLY=1 builds and stops;
# RESUME=runs/sprt/<name>_<time>.jsonl continues that stopped run (its finished pairs count, its
# shuffle seed; ygoai-match --resume): under the same conditions, or knowingly not.
# (all in one function, read whole before it runs, and the call and exit on one line: editing the
# file during a run cannot change it; still, replace it by a rename rather than in place)
main() {
set -euo pipefail
# (also inside $(...): a failed build must not go on to copy older binaries)
shopt -s inherit_errexit
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
NEW=$1
BASE=${2:-main}
TC=${TC:-10+0.1}
CONC=${CONC:-16}
OPENINGS=${OPENINGS:-10000}
ELO=${ELO:-0 5}
BOOK=${BOOK:-runs/books/match2.jsonl}
POOL=${POOL:-50000}
NAME=${NAME:-$(echo "$NEW" | tr '/' '-')}
RESUME=${RESUME:-}
if [ -n "$RESUME" ]; then
    SHUF=$(sed -n 's/^book shuffled with seed //p' "${RESUME%.jsonl}.log")
fi
SHUF=${SHUF:-$(od -An -N4 -tu4 /dev/urandom | tr -d " ")}

TREE=$ROOT/build/bench-tree
TARGET=$ROOT/build/bench-target

# The worktree for builds (made once), with real submodule checkouts at the ref's commits, cloned
# and fetched from the checkout's submodules (url.*.insteadOf: nothing is downloaded): an ordinary
# checkout, nothing to commit there.
submodules() {
    local args=(-c protocol.file.allow=always) name url
    while read -r name url; do
        args+=(-c "url.$ROOT/${name}.insteadOf=$url")
    done < <(git config -f "$ROOT/.gitmodules" --get-regexp '^submodule\..*\.url$' | sed -E 's/^submodule\.(.*)\.url /\1 /')
    git -C "$TREE" "${args[@]}" submodule update --init -q
}

setup_tree() {
    [ -e "$TREE/Cargo.toml" ] || git worktree add -q --detach "$TREE" HEAD
}

# The binaries of `ref` in build/engines/<sha>/ (built if missing); prints the directory.
build() {
    local sha dir
    sha=$(git rev-parse --short=10 "$1^{commit}")
    dir=$ROOT/build/engines/$sha
    # (YGOAI_ROOT: refs from before it would read the worktree's data, which it lacks)
    if ! git merge-base --is-ancestor 6d49e8f "$sha"; then
        echo "$1 lacks 6d49e8f (YGOAI_ROOT): rebase it onto main first" >&2
        exit 1
    fi
    if [ ! -x "$dir/ygoai-engine" ] || [ ! -x "$dir/ygoai-match" ] || [ ! -x "$dir/ygoai-bench-search" ] || [ ! -d "$dir/root" ]; then
        setup_tree
        git -C "$TREE" checkout -q --detach "$sha"
        submodules
        echo "$(date) building $1 ($sha)" >&2
        systemd-run --user --scope -q -p MemoryMax=28G -p MemorySwapMax=0 nice -n 10 \
            env CARGO_TARGET_DIR="$TARGET" CARGO_BUILD_JOBS=6 YGOAI_DECK_MODELS_ONNX="$ROOT/download_human_decks/onnx_export" \
            cargo build --release --manifest-path "$TREE/Cargo.toml" -p ygoai --bin ygoai-engine --bin ygoai-match --bin ygoai-bench-search >&2 \
            || { echo "build of $1 failed" >&2; exit 1; }
        mkdir -p "$dir/root"
        cp "$TARGET/release/ygoai-engine" "$TARGET/release/ygoai-match" "$TARGET/release/ygoai-bench-search" "$dir/"
        # (the ref's corrected scripts; the data is the checkout's)
        git archive "$sha" lua_fixes | tar -x -C "$dir/root"
        for d in external download_human_decks build; do
            [ -e "$dir/root/$d" ] || ln -s "$ROOT/$d" "$dir/root/$d"
        done
    fi
    echo "$dir"
}

# (NEW must contain BASE's engine, its last commit to the code: a branch from before a merge into
# BASE would also be tested without that merge's gain; ALLOW_DIVERGED=1 to compare anyway)
base_code=$(git log -1 --format=%H "$BASE" -- crates lua_fixes Cargo.toml Cargo.lock rust external)
if [ "${ALLOW_DIVERGED:-0}" != 1 ] && ! git merge-base --is-ancestor "$base_code" "$NEW"; then
    echo "$NEW lacks $BASE's code ($base_code): rebase it first (or ALLOW_DIVERGED=1)" >&2
    exit 1
fi
# (one build at a time: runs started together share the worktree and build/engines)
exec 8>"$ROOT/build/engines.lock"
flock 8
new_dir=$(build "$NEW")
base_dir=$(build "$BASE")
flock -u 8
[ "${BUILD_ONLY:-0}" = 1 ] && { echo "$new_dir $base_dir"; exit 0; }

new_sha=$(basename "$new_dir")
base_sha=$(basename "$base_dir")
new_env="env YGOAI_ROOT=$new_dir/root"
base_env="env YGOAI_ROOT=$base_dir/root"
if [ -n "${NODES:-}" ]; then
    new_env="$new_env YGOAI_SEARCH_NODES=$NODES"
    base_env="$base_env YGOAI_SEARCH_NODES=$NODES"
    TC=60+1
fi
T=$(date +%Y%m%d-%H%M%S)
log=runs/sprt/${NAME}_$T.log
mkdir -p runs/sprt
echo "$(date) start $NAME: $NEW ($new_sha) vs $BASE ($base_sha), tc $TC${NODES:+, $NODES nodes}${RESUME:+, resuming $RESUME} -> $log"
YGOAI_ROOT=$base_dir/root "$base_dir/ygoai-match" --engine "new=$new_env $new_dir/ygoai-engine alphabeta" --engine "base=$base_env $base_dir/ygoai-engine alphabeta" \
    --book "$BOOK" --pool "$POOL" --shuffle "$SHUF" -n "$OPENINGS" -c "$CONC" --tc "$TC" $([ "$ELO" = none ] || echo --sprt $ELO) \
    --records "runs/sprt/${NAME}_$T.jsonl" ${RESUME:+--resume "$RESUME"} > "$log" 2>&1 || true
final=$(grep -E "^(SPRT finished|final:)" "$log" | tr '\n' ' ' || true)
[ -z "$final" ] && final="stopped: $(grep "games" "$log" | tail -1)"
python3 -c "import json,sys; a=sys.argv; print(json.dumps({'new': a[2], 'new_sha': a[3], 'base': a[4], 'base_sha': a[5], 'agent': {'new': 'alphabeta', 'base': 'alphabeta'}, 'tc': a[6], 'nodes': int(a[7]) if a[7] else None, 'elo': [float(x) for x in a[8].split()] if a[8] != 'none' else None, 'verdict': a[1], 'engine': 'rust', 'shuffle': int(a[9]), 'pool': int(a[11]), **({'resumed': a[12]} if a[12] else {}), **({'book': a[10]} if a[10] != 'runs/books/match1.jsonl' else {})}))" \
    "$final" "$NAME" "$new_sha" "$BASE" "$base_sha" "$TC" "${NODES:-}" "$ELO" "$SHUF" "$BOOK" "$POOL" "$RESUME" >> runs/sprt/results.jsonl
echo "$(date) done $NAME: $final"
}
main "$@"; exit
