"""
Build viewer/index.json — a static index of all CSV files in
price_collector/data/. The viewer reads this index to know which
files exist and renders them as paginated charts.

Runs both:
  - Locally: python3 viewer/build_index.py  (before bash viewer/run.sh)
  - In GitHub Actions: as the last step before git commit, so the
    index always stays in sync with the committed CSVs.

The index records minimal metadata per window so the viewer can
filter by winner without having to fetch every CSV upfront.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "price_collector" / "data"
INDEX_PATH = Path(__file__).resolve().parent / "index.json"


def extract_winner(csv_path):
    """Pull the winner from the trailing '# RESULT' comment line, if any.

    The collector writes a comment like:
        # RESULT,winner=Up,slug=btc-updown-5m-XXX,ticks=569
    We read the file backwards (last few KB) so this stays fast even on
    very large CSVs.
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
                    return m.group(1)
    except Exception as e:
        print(f"  warn: could not read winner from {csv_path.name}: {e}")
    return None


def slug_to_epoch(slug):
    """Extract the open epoch from the slug for sorting."""
    m = re.search(r"-(\d{10,})$", slug)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0


def main():
    if not DATA_DIR.exists():
        print(f"  error: {DATA_DIR} does not exist")
        return 1

    files = sorted(DATA_DIR.glob("btc-updown-5m-*.csv"))
    print(f"Scanning {len(files)} CSV files in {DATA_DIR}...")

    windows = []
    for csv_path in files:
        slug = csv_path.stem
        winner = extract_winner(csv_path)
        windows.append({
            "slug": slug,
            "filename": csv_path.name,
            "winner": winner,
            "epoch": slug_to_epoch(slug),
        })

    # Newest first
    windows.sort(key=lambda w: w["epoch"], reverse=True)

    index = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "count": len(windows),
        "windows": windows,
    }

    INDEX_PATH.write_text(json.dumps(index, indent=2))
    print(f"Wrote {INDEX_PATH} ({len(windows)} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
