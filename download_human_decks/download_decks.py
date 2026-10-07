#!/usr/bin/env python3
"""Download all decks from ygoprodeck.com in YDK format.

Uses the /api/decks/getDecks.php endpoint, which returns 20 decks per page
with main/extra/side as JSON-encoded arrays of card IDs — exactly what's
needed to reconstruct YDK files. No per-deck HTTP requests required.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

API = "https://ygoprodeck.com/api/decks/getDecks.php"
PAGE_SIZE = 20
RATE_LIMIT_PER_SEC = 15
MIN_INTERVAL = 1.0 / RATE_LIMIT_PER_SEC
UA = "Mozilla/5.0 (deck-archiver script)"
# Rough upper bound discovered via binary search on offsets (API 400s past this).
ESTIMATED_TOTAL_DECKS = 173422


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h: return f"{h}h{m:02d}m"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def sanitize(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name).strip()
    return name[:120] or "deck"


def fetch_page(sort: str, offset: int, retries: int = 8):
    qs = urlencode({"sort": sort, "offset": offset})
    url = f"{API}?{qs}"
    backoff = 1.0
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except HTTPError as e:
            last_err = e
            if e.code == 400:
                # API returns 400 past the last real page — signal end of results.
                return []
            # 429 / 5xx / etc — back off and retry
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last_err = e
        log(f"  retry {attempt+1}/{retries} for offset={offset} after {backoff:.0f}s: {last_err}")
        time.sleep(backoff)
        backoff = min(backoff * 2, 30.0)
    raise RuntimeError(f"failed to fetch offset={offset} after {retries} attempts: {last_err}")


def build_ydk(deck: dict) -> str:
    def ids(field):
        raw = deck.get(field)
        if not raw:
            return []
        try:
            return [str(x) for x in json.loads(raw)]
        except (json.JSONDecodeError, TypeError):
            return []

    lines = ["#created by " + (deck.get("username") or "unknown"), "#main"]
    lines.extend(ids("main_deck"))
    lines.append("#extra")
    lines.extend(ids("extra_deck"))
    lines.append("!side")
    lines.extend(ids("side_deck"))
    return "\r\n".join(lines) + "\r\n"


def write_deck(deck: dict, out_dir: Path) -> Path | None:
    main = json.loads(deck.get("main_deck") or "[]")
    if not main:
        return None
    deck_num = deck.get("deckNum")
    name = sanitize(deck.get("deck_name") or "deck")
    path = out_dir / f"{deck_num}_{name}.ydk"
    if path.exists():
        return path
    path.write_text(build_ydk(deck), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("decks"),
                    help="output directory (default: ./decks)")
    ap.add_argument("--sort", default="Deck Views",
                    help="sort order (default: 'Deck Views')")
    ap.add_argument("--start", type=int, default=0, help="starting offset")
    ap.add_argument("--max", type=int, default=None,
                    help="stop after writing this many decks")
    ap.add_argument("--after", type=int, default=None,
                    help="only decks with a higher deck number (newer than the ones already downloaded)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in args.out.glob("*.ydk"))
    log(f"output dir: {args.out.resolve()} (already contains {existing} .ydk files)")
    log(f"sort='{args.sort}', start offset={args.start}, "
        f"max={'unlimited' if args.max is None else args.max}")
    log(f"rate limit: {RATE_LIMIT_PER_SEC} req/s "
        f"(min {MIN_INTERVAL*1000:.0f}ms between requests)")
    log(f"estimated total decks available: ~{ESTIMATED_TOTAL_DECKS}")

    offset = args.start
    written = 0
    reused = 0           # file already existed on disk
    skipped_empty = 0    # deck had empty main_deck
    skipped_pages = 0    # whole pages we gave up on after retries
    pages = 0
    last_req = 0.0
    start_t = time.monotonic()
    last_progress_t = start_t

    while True:
        wait = MIN_INTERVAL - (time.monotonic() - last_req)
        if wait > 0:
            time.sleep(wait)
        last_req = time.monotonic()

        log(f"fetching page offset={offset} "
            f"(~deck {offset+1}/{ESTIMATED_TOTAL_DECKS})...")
        t0 = time.monotonic()
        try:
            decks = fetch_page(args.sort, offset)
        except Exception as e:
            log(f"  SKIP page offset={offset} (gave up): {e}")
            skipped_pages += 1
            offset += PAGE_SIZE
            continue
        fetch_ms = (time.monotonic() - t0) * 1000

        if not decks:
            log(f"  empty/past-end response at offset={offset} — finished")
            break

        page_new = 0
        page_reused = 0
        for deck in decks:
            if args.after is not None and int(deck.get("deckNum") or 0) <= args.after:
                continue
            path = write_deck(deck, args.out)
            if path is None:
                skipped_empty += 1
                continue
            if path.stat().st_mtime < t0:
                page_reused += 1
                reused += 1
            else:
                page_new += 1
                written += 1
            if args.max is not None and written >= args.max:
                log(f"  hit --max ({written} new this run); stopping")
                _print_summary(start_t, pages+1, written, reused,
                               skipped_empty, skipped_pages, offset)
                return

        pages += 1
        elapsed = time.monotonic() - start_t
        rate = (written + reused) / elapsed if elapsed > 0 else 0
        remaining = max(0, ESTIMATED_TOTAL_DECKS - (offset + len(decks)))
        eta = remaining / rate if rate > 0 else 0
        log(f"  page {pages} done in {fetch_ms:.0f}ms — "
            f"got {len(decks)} decks ({page_new} new, {page_reused} already on disk), "
            f"run totals: {written} new, {reused} reused, {skipped_empty} empty")
        if time.monotonic() - last_progress_t > 30 or pages % 25 == 0:
            log(f"  progress: ~{(offset+len(decks))/ESTIMATED_TOTAL_DECKS*100:.1f}% "
                f"of estimated total, {rate:.1f} decks/s, ETA {fmt_duration(eta)}")
            last_progress_t = time.monotonic()

        if len(decks) < PAGE_SIZE:
            log("  short page received — assuming end of results")
            break
        offset += PAGE_SIZE

    _print_summary(start_t, pages, written, reused,
                   skipped_empty, skipped_pages, offset)


def _print_summary(start_t, pages, written, reused, skipped_empty,
                   skipped_pages, last_offset):
    elapsed = time.monotonic() - start_t
    log("=" * 60)
    log(f"SUMMARY  elapsed={fmt_duration(elapsed)}  pages_fetched={pages}")
    log(f"  decks written (new):     {written}")
    log(f"  decks reused (existing): {reused}")
    log(f"  decks skipped (empty):   {skipped_empty}")
    log(f"  pages skipped (errors):  {skipped_pages}")
    log(f"  last offset processed:   {last_offset}")
    if skipped_pages:
        log(f"  re-run with --start <offset> to retry skipped pages")


if __name__ == "__main__":
    main()
