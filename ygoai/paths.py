"""Filesystem locations of the external resources."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "external"
CARD_SCRIPTS_DIR = Path(os.environ.get("YGOAI_SCRIPTS", EXTERNAL / "CardScripts"))
# corrected card scripts (bugs in CardScripts), read instead of the originals (the Rust cards are
# translated from them; the belief's script scans read them)
FIXED_CARD_SCRIPTS_DIR = ROOT / "lua_fixes"
BABEL_CDB_DIR = Path(os.environ.get("YGOAI_CDB_DIR", EXTERNAL / "BabelCDB"))
DECKS_DIR = Path(os.environ.get("YGOAI_DECKS", ROOT / "download_human_decks" / "decks"))
# decks downloaded later (newer than the corpus): test games only, the corpus stays as is
NEW_DECKS_DIR = Path(os.environ.get("YGOAI_NEW_DECKS", ROOT / "download_human_decks" / "decks_new"))
