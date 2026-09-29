"""Filesystem locations of the external resources."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "external"
CORE_LIB = Path(os.environ.get("YGOAI_CORE_LIB", ROOT / "build" / "libocgcore.so"))
CARD_SCRIPTS_DIR = Path(os.environ.get("YGOAI_SCRIPTS", EXTERNAL / "CardScripts"))
BABEL_CDB_DIR = Path(os.environ.get("YGOAI_CDB_DIR", EXTERNAL / "BabelCDB"))
DECKS_DIR = Path(os.environ.get("YGOAI_DECKS", ROOT / "download_human_decks" / "decks"))
