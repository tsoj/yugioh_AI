#!/usr/bin/env bash
# Builds build/libocgcore.so: upstream ygopro-core (external/ygopro-core)
# plus core_ext/ocgcore.patch and the extension sources in core_ext/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/build/ocgcore-src"
OUT="$ROOT/build/ocgcore"
BUILDTYPE="${BUILDTYPE:-release}"

if [ ! -f "$ROOT/external/ygopro-core/lua/src/lua.h" ]; then
	echo "lua submodule missing, run: git submodule update --init --recursive" >&2
	exit 1
fi

rm -rf "$SRC"
mkdir -p "$SRC"
# copy without the .git metadata
(cd "$ROOT/external/ygopro-core" && tar --exclude=.git -cf - .) | (cd "$SRC" && tar -xf -)
(cd "$SRC" && patch -p1 --quiet < "$ROOT/core_ext/ocgcore.patch")
cp "$ROOT/core_ext/ocgext.h" "$ROOT/core_ext/ocgext.cpp" "$SRC/"

if [ ! -f "$OUT/build.ninja" ]; then
	meson setup "$OUT" "$SRC" --buildtype="$BUILDTYPE" -Ddefault_library=shared -Db_ndebug=if-release >/dev/null
fi
ninja -C "$OUT" >"$OUT/build.log" 2>&1 || { grep -E "error|Error" "$OUT/build.log" >&2; exit 1; }
# replace atomically: running processes keep their mapping of the old library
cp "$OUT/libocgcore.so" "$ROOT/build/libocgcore.so.new"
mv -f "$ROOT/build/libocgcore.so.new" "$ROOT/build/libocgcore.so"
echo "built $ROOT/build/libocgcore.so"
