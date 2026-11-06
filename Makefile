# The bot's build for OpenBench (make -j EXE=PATH): the binary `bot`, profile-guided (as
# scripts/sprt_branch.sh: an instrumented bot plays `bot bench`, then the bot is built with that
# profile, see PROFILE; ~25% more nodes/s). The submodules must be present: ygopro-core-rust and
# external/HiGHS to build, external/CardScripts and external/BabelCDB to run (the bot runs with
# YGO_ROOT set to this directory).
#
#   make [EXE=bot] [PGO=1] [PROFILE=PATH]   the bot
#   make arbiter [ARBITER=ygo-arbiter]
#
# CARGO_TARGET_DIR (default target/) can be shared between builds of different commits.
# LLVM_PROFDATA must be of rustc's LLVM version (rustc -vV).

EXE ?= bot
ARBITER ?= ygo-arbiter
PGO ?= 1
LLVM_PROFDATA ?= llvm-profdata
CARGO_TARGET_DIR ?= $(CURDIR)/target

# PROFILE.profdata: the PGO profile. A profile that exists is used again: only the first build
# profiles (an instrumented build plays `bot bench`), later builds use its profile (LLVM ignores
# it for functions that changed since). Delete it or give another PROFILE for a new profile (the
# OpenBench client gives one per ygopro-core-rust commit). Fixed paths: RUSTFLAGS stay the same from
# commit to commit, so that the dependencies of an earlier build are used again. One build at a time.
PROFILE ?= $(CARGO_TARGET_DIR)/pgo/profile

# HiGHS (C++) is built once per HiGHS version, in this directory (crates/highs/build.rs)
export HIGHS_BUILD_DIR ?= $(CARGO_TARGET_DIR)/highs

.PHONY: all arbiter

ifeq ($(PGO),1)
all: $(PROFILE).profdata
	RUSTFLAGS="-Cprofile-use=$(PROFILE).profdata" CARGO_TARGET_DIR=$(CARGO_TARGET_DIR)/pgo-use \
		cargo build --release -p bot --bin bot
	cp $(CARGO_TARGET_DIR)/pgo-use/release/bot $(EXE)

$(PROFILE).profdata:
	rm -rf $(PROFILE)
	mkdir -p $(PROFILE)
	RUSTFLAGS="-Cprofile-generate=$(PROFILE)" CARGO_TARGET_DIR=$(CARGO_TARGET_DIR)/pgo-gen \
		cargo build --release -p bot --bin bot
	rm -f $(PROFILE)/*.profraw
	YGO_ROOT=$(CURDIR) $(CARGO_TARGET_DIR)/pgo-gen/release/bot bench
	$(LLVM_PROFDATA) merge -o $@.tmp $(PROFILE)/*.profraw
	mv $@.tmp $@
else
all:
	CARGO_TARGET_DIR=$(CARGO_TARGET_DIR) cargo build --release -p bot --bin bot
	cp $(CARGO_TARGET_DIR)/release/bot $(EXE)
endif

arbiter:
	CARGO_TARGET_DIR=$(CARGO_TARGET_DIR)/arbiter \
		cargo build --release --manifest-path ygopro-core-rust/Cargo.toml -p ygo-arbiter --bin ygo-arbiter
	cp $(CARGO_TARGET_DIR)/arbiter/release/ygo-arbiter $(ARBITER)
