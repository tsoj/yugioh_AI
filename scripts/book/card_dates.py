"""Cards' first release dates (the earlier of the OCG and TCG release; YGOPRODeck's card data,
misc_info), for eras of decks (era_candidates.py).

    .venv/bin/python scripts/book/card_dates.py      (downloads the card data to runs/carddates/)

`dates()`: code -> "YYYY-MM-DD", alternate artworks included.
"""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CARD_INFO = ROOT / "runs" / "carddates" / "cardinfo.json"
URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php?misc=yes"


def download():
    CARD_INFO.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=300) as r:
        CARD_INFO.write_bytes(r.read())


def dates() -> dict[int, str]:
    if not CARD_INFO.exists():
        download()
    out = {}
    for c in json.load(open(CARD_INFO))["data"]:
        m = (c.get("misc_info") or [{}])[0]
        # (1990-01-01: a placeholder of unreleased cards)
        released = [d for d in (m.get("ocg_date"), m.get("tcg_date")) if d and d > "1990-01-01"]
        if released:
            for code in [c["id"]] + [im["id"] for im in c.get("card_images", [])]:
                out[code] = min(released)
    return out


if __name__ == "__main__":
    download()
    print(f"{len(dates())} cards with a release date -> {CARD_INFO}")
