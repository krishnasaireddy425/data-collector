"""
Backfill the '# RESULT' line for CSVs that are missing it.

When a workflow run is cancelled mid-way through asyncio.gather, the
result-writing loop in main() never executes — so completed CSVs end up
without their trailing '# RESULT,winner=...,slug=...,ticks=...' line.

This script:
  1. Finds every CSV in price_collector/data/ that has NO '# RESULT' line
  2. Determines the winner using the same logic as get_winner_from_csv
     (boundary-closest tick if btc_oracle_ts is present, else first/last)
  3. Appends a '# RESULT,winner=...,slug=...,ticks=...' line

Read-only on the existing CSV rows. Only appends one line per file.

Usage:
    python3 scripts/backfill_results.py
"""

import csv as csv_mod
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "price_collector" / "data"


def has_result_line(csv_path):
    """Return True if the CSV already contains a '# RESULT' comment."""
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 4096)
            f.seek(-chunk, os.SEEK_END)
            tail = f.read().decode("utf-8", errors="replace")
        return "# RESULT" in tail
    except Exception:
        return False


def slug_to_open_epoch(slug):
    """Extract the open epoch from a slug like btc-updown-5m-1775898300."""
    m = re.search(r"-(\d{10,})$", slug)
    return int(m.group(1)) if m else None


def compute_winner_and_count(csv_path):
    """Mirror of get_winner_from_csv() in price_collector/main.py.

    Uses btc_oracle_ts to pick boundary-closest tick if present,
    else falls back to first/last row. Tie -> 'Up'.

    Returns (winner_label, tick_count).
    """
    slug = csv_path.stem
    open_epoch = slug_to_open_epoch(slug)
    if open_epoch is None:
        return "unknown", 0

    open_ms = int(open_epoch * 1000)
    close_ms = int((open_epoch + 300) * 1000)

    tick_count = 0
    first_price = None
    last_price = None
    best_open = None  # (distance_ms, price)
    best_close = None  # (distance_ms, price)
    has_ts = False

    with open(csv_path, "r") as f:
        reader = csv_mod.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return "unknown", 0

        try:
            btc_col = header.index("btc_price")
        except ValueError:
            return "unknown", 0

        try:
            ts_col = header.index("btc_oracle_ts")
            has_ts = True
        except ValueError:
            ts_col = None
            has_ts = False

        for row in reader:
            if not row or row[0].startswith("#"):
                continue

            try:
                price = float(row[btc_col])
            except (ValueError, IndexError):
                tick_count += 1
                continue

            tick_count += 1

            if price <= 0:
                continue

            if first_price is None:
                first_price = price
            last_price = price

            if has_ts:
                try:
                    ts = float(row[ts_col])
                except (ValueError, IndexError):
                    continue
                if ts <= 0:
                    continue
                od = abs(ts - open_ms)
                cd = abs(ts - close_ms)
                if best_open is None or od < best_open[0]:
                    best_open = (od, price)
                if best_close is None or cd < best_close[0]:
                    best_close = (cd, price)

    if has_ts and best_open is not None and best_close is not None:
        open_price = best_open[1]
        close_price = best_close[1]
    elif first_price is not None and last_price is not None:
        open_price = first_price
        close_price = last_price
    else:
        return "unknown", tick_count

    # Tie -> Up (matches Polymarket's '>=' rule)
    if close_price >= open_price:
        return "Up", tick_count
    else:
        return "Down", tick_count


def append_result_line(csv_path, winner, tick_count):
    """Append the '# RESULT' line to the CSV."""
    slug = csv_path.stem
    with open(csv_path, "a", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow([])
        w.writerow([
            "# RESULT",
            f"winner={winner}",
            f"slug={slug}",
            f"ticks={tick_count}",
        ])


def main():
    if not DATA_DIR.exists():
        print(f"error: {DATA_DIR} does not exist")
        return 1

    files = sorted(DATA_DIR.glob("btc-updown-5m-*.csv"))
    print(f"Scanning {len(files)} CSV files...")

    missing = []
    for csv_path in files:
        if not has_result_line(csv_path):
            missing.append(csv_path)

    if not missing:
        print("All CSVs already have a '# RESULT' line. Nothing to do.")
        return 0

    print(f"Found {len(missing)} CSV(s) missing '# RESULT'")
    print()

    backfilled = 0
    skipped = 0

    for csv_path in missing:
        winner, tick_count = compute_winner_and_count(csv_path)

        if winner == "unknown" or tick_count == 0:
            print(f"  SKIP   {csv_path.name}  (no usable BTC data)")
            skipped += 1
            continue

        append_result_line(csv_path, winner, tick_count)
        print(f"  +RESULT  {csv_path.name}  winner={winner}  ticks={tick_count}")
        backfilled += 1

    print()
    print("=" * 60)
    print(f"Backfilled: {backfilled}")
    print(f"Skipped:    {skipped}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
