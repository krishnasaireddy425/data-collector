"""
Verify locally-recorded winners against Polymarket's actual settlement.

For every CSV in price_collector/data/, this script:
  1. Reads the locally-computed winner (from the '# RESULT' line we wrote)
  2. Asks Polymarket's Gamma API for the official settlement
  3. Compares them — and prints any mismatch

It does NOT modify any files. Read-only check.

Usage:
    python3 scripts/verify_winners.py
"""

import argparse
import os
import re
import ssl
import sys
import time
from pathlib import Path

import urllib.request
import urllib.parse
import urllib.error
import json

# Use certifi's CA bundle so macOS Python can verify HTTPS certs
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "price_collector" / "data"
GAMMA_API = "https://gamma-api.polymarket.com/markets"

# Polite delay between API calls so we don't get rate-limited
REQUEST_DELAY = 0.05  # 50ms = 20 req/sec
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 1.5


# ----------------------------------------------------------------------
# Local CSV winner extraction
# ----------------------------------------------------------------------

def extract_local_winner(csv_path):
    """Read the trailing '# RESULT' line and pull out winner=... value.

    Returns 'up' / 'down' / 'flat' / 'unknown' / None.
    """
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 4096)
            f.seek(-chunk, os.SEEK_END)
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            if "# RESULT" in line and "winner=" in line:
                m = re.search(r"winner=(\w+)", line)
                if m:
                    return m.group(1).lower()
    except Exception as e:
        print(f"  warn: could not read winner from {csv_path.name}: {e}")
    return None


# ----------------------------------------------------------------------
# Polymarket actual winner
# ----------------------------------------------------------------------

def fetch_polymarket_winner(slug):
    """Query Polymarket Gamma API for the actual settled winner of a slug.

    Returns:
        'up' / 'down' / 'flat' if resolved
        None if not resolved or not found
    """
    # closed=true is REQUIRED for settled 5-min markets — without it,
    # Gamma hides them and returns an empty list
    url = (
        f"{GAMMA_API}?slug={urllib.parse.quote(slug)}&closed=true"
    )
    last_err = None

    for attempt in range(RETRY_ATTEMPTS):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=10, context=SSL_CTX) as resp:
                if resp.status != 200:
                    last_err = f"HTTP {resp.status}"
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            last_err = f"network: {e}"
            time.sleep(RETRY_BACKOFF * (attempt + 1))
            continue
        except Exception as e:
            last_err = str(e)
            time.sleep(RETRY_BACKOFF * (attempt + 1))
            continue

        if not data:
            return None

        market = data[0] if isinstance(data, list) else data

        # The market is settled when outcomePrices show 1.0 / 0.0.
        # outcomePrices is sometimes a JSON-encoded string, sometimes a list.
        outcome_prices_raw = market.get("outcomePrices")
        outcomes_raw = market.get("outcomes")

        try:
            if isinstance(outcome_prices_raw, str):
                outcome_prices = json.loads(outcome_prices_raw)
            else:
                outcome_prices = outcome_prices_raw or []

            if isinstance(outcomes_raw, str):
                outcomes = json.loads(outcomes_raw)
            else:
                outcomes = outcomes_raw or []
        except (json.JSONDecodeError, TypeError):
            return None

        if not outcome_prices or not outcomes:
            return None

        # Find the side that resolved to 1.0
        for outcome_name, price_str in zip(outcomes, outcome_prices):
            try:
                price = float(price_str)
            except (TypeError, ValueError):
                continue
            if price >= 0.5:
                name = (outcome_name or "").strip().lower()
                if name in ("up", "yes"):
                    return "up"
                if name in ("down", "no"):
                    return "down"
                return name
        return None

    print(f"  warn: gamma API failed for {slug}: {last_err}")
    return None


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reverse", action="store_true",
                    help="Walk newest → oldest (default: oldest → newest)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only check the first N files AFTER sorting/reversing")
    args = ap.parse_args()

    if not DATA_DIR.exists():
        print(f"error: data directory {DATA_DIR} does not exist")
        return 1

    all_files = sorted(DATA_DIR.glob("btc-updown-5m-*.csv"))
    if not all_files:
        print(f"error: no CSV files found in {DATA_DIR}")
        return 1

    if args.reverse:
        all_files = list(reversed(all_files))
    files = all_files[:args.limit] if args.limit else all_files

    direction = "newest → oldest" if args.reverse else "oldest → newest"
    print(f"Verifying {len(files)} of {len(all_files)} CSV files "
          f"({direction}) against Polymarket Gamma API...")
    print(f"(50ms delay between requests, {RETRY_ATTEMPTS} retries on failure)")
    print()

    mismatches = []
    unresolved = []
    no_local = []
    checked = 0

    for csv_path in files:
        slug = csv_path.stem
        local_winner = extract_local_winner(csv_path)
        actual_winner = fetch_polymarket_winner(slug)
        time.sleep(REQUEST_DELAY)
        checked += 1

        if local_winner is None:
            no_local.append(slug)
        elif actual_winner is None:
            unresolved.append(slug)
        elif local_winner != actual_winner:
            mismatches.append((slug, local_winner, actual_winner))
            print(
                f"  MISMATCH  {slug}  "
                f"local={local_winner}  polymarket={actual_winner}"
            )

        # Progress indicator every 50 windows
        if checked % 50 == 0:
            print(f"  ... {checked}/{len(files)} checked, "
                  f"{len(mismatches)} mismatches so far")

    print()
    print("=" * 60)
    print(f"SUMMARY")
    print("=" * 60)
    print(f"Total checked:           {checked}")
    print(f"Mismatches:              {len(mismatches)}")
    print(f"Local winner missing:    {len(no_local)}")
    print(f"Unresolved on Polymarket: {len(unresolved)}")

    if mismatches:
        print()
        print("All mismatches:")
        for slug, local, actual in mismatches:
            print(f"  {slug}  local={local}  polymarket={actual}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
