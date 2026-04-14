"""
Empirical test of the "buy cheap side" / mean-reversion hypothesis.

Approach:
  - Walk every CSV second-by-second.
  - At each sampled time t, identify the CHEAP side (whichever of up_ask
    or down_ask is lower) and its price.
  - Check if that cheap side ends up winning the window.
  - Bucket by entry price range; compute win rate per bucket.
  - Compare win rate vs implied probability (= entry price).
  - Compute expected value per share: win_rate - entry_price.

No ML. No prediction. Just: does buying cheap historically pay off?
"""

import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "price_collector" / "data"

# Sample times within the window
SAMPLE_TIMES = list(range(30, 280, 10))  # 30, 40, 50, ..., 270

# Price buckets for the cheap side's ask
BUCKETS = [
    (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25),
    (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45),
    (0.45, 0.50), (0.50, 0.50)   # 50c = exact tie
]


def read_winner(csv_path):
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(-min(size, 4096), os.SEEK_END)
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            if "# RESULT" in line and "winner=" in line:
                m = re.search(r"winner=(\w+)", line)
                if m:
                    w = m.group(1)
                    return w if w in ("Up", "Down") else None
    except Exception:
        return None
    return None


def resample(df, max_t=300):
    for col in ("up_bid", "up_ask", "down_bid", "down_ask", "btc_price",
                "elapsed_sec"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["elapsed_sec", "btc_price"])
    df = df.drop_duplicates(subset=["elapsed_sec"], keep="last")
    df = df.sort_values("elapsed_sec").reset_index(drop=True)
    if df.empty:
        return None
    new_idx = np.arange(0, max_t + 1, 1.0)
    df_r = df.set_index("elapsed_sec")
    df_1s = df_r.reindex(df_r.index.union(new_idx)).sort_index()
    df_1s = df_1s.ffill().bfill()
    df_1s = df_1s.loc[new_idx].reset_index()
    df_1s.rename(columns={"index": "elapsed_sec"}, inplace=True)
    return df_1s


def main():
    files = sorted(glob.glob(str(DATA_DIR / "btc-updown-5m-*.csv")))
    print(f"Loading {len(files)} windows...")

    # Collect one record per (window, sample_time)
    records = []
    loaded = 0
    for f in files:
        winner = read_winner(f)
        if winner not in ("Up", "Down"):
            continue
        try:
            raw = pd.read_csv(f, comment="#")
        except Exception:
            continue
        if len(raw) < 10:
            continue
        df = resample(raw)
        if df is None:
            continue
        loaded += 1

        for t in SAMPLE_TIMES:
            row = df[df["elapsed_sec"] == t]
            if row.empty:
                continue
            r = row.iloc[0]
            up_ask = float(r["up_ask"]) if pd.notna(r["up_ask"]) else None
            down_ask = float(r["down_ask"]) if pd.notna(r["down_ask"]) else None
            if up_ask is None or down_ask is None:
                continue
            if up_ask <= 0 or down_ask <= 0:
                continue

            if up_ask <= down_ask:
                cheap_side, cheap_price = "Up", up_ask
                expensive_price = down_ask
            else:
                cheap_side, cheap_price = "Down", down_ask
                expensive_price = up_ask
            cheap_won = 1 if cheap_side == winner else 0

            # How much did BTC move relative to open by time t?
            btc_open = float(df.iloc[0]["btc_price"])
            btc_now = float(r["btc_price"])
            btc_chg_pct = (
                (btc_now - btc_open) / btc_open if btc_open > 0 else 0
            )

            records.append({
                "slug": Path(f).stem,
                "elapsed_sec": t,
                "cheap_side": cheap_side,
                "cheap_price": cheap_price,
                "expensive_price": expensive_price,
                "implied_prob": cheap_price,
                "cheap_won": cheap_won,
                "winner": winner,
                "btc_chg_pct": btc_chg_pct,
            })

    print(f"Loaded {loaded} windows with winners")
    df = pd.DataFrame(records)
    print(f"Total (window, sample) records: {len(df):,}")
    print()

    # --- 1. Overall: win rate by cheap_price bucket ---
    print("=" * 90)
    print("WIN RATE OF 'CHEAP SIDE' BY ENTRY PRICE BUCKET")
    print("(Implied prob = price; Actual = empirical win rate; Edge = Actual − Implied)")
    print("=" * 90)
    print(f"{'bucket':<12} {'n':>7} {'implied':>9} {'actual':>9} {'edge':>9} {'EV/share':>10}")
    print("-" * 90)

    for lo, hi in BUCKETS:
        if lo == hi:
            sub = df[(df["cheap_price"] >= lo - 0.005) &
                     (df["cheap_price"] <= hi + 0.005)]
            label = f"≈{lo:.2f}"
        else:
            sub = df[(df["cheap_price"] >= lo) & (df["cheap_price"] < hi)]
            label = f"{lo:.2f}-{hi:.2f}"
        n = len(sub)
        if n < 20:
            continue
        actual = sub["cheap_won"].mean()
        implied = sub["cheap_price"].mean()
        edge = actual - implied
        # EV per share = actual*(1-price) - (1-actual)*price = actual - price
        ev_per_share = edge
        print(f"{label:<12} {n:>7,} {implied:>9.3f} {actual:>9.3f} "
              f"{edge:>+9.3f} ${ev_per_share:>+8.3f}")

    # --- 2. Win rate by (bucket × time of entry) ---
    print()
    print("=" * 100)
    print("WIN RATE BY ENTRY PRICE × TIME OF ENTRY")
    print("(Is the mispricing worse at certain times?)")
    print("=" * 100)
    print(f"{'price':<12}", end="")
    time_bins = [(30, 90), (90, 150), (150, 210), (210, 280)]
    for lo, hi in time_bins:
        print(f" t={lo}-{hi}s".center(16), end="")
    print()
    print("-" * 100)

    for p_lo, p_hi in [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
                        (0.40, 0.50)]:
        row_label = f"{p_lo:.2f}-{p_hi:.2f}"
        print(f"{row_label:<12}", end="")
        for t_lo, t_hi in time_bins:
            sub = df[
                (df["cheap_price"] >= p_lo) & (df["cheap_price"] < p_hi) &
                (df["elapsed_sec"] >= t_lo) & (df["elapsed_sec"] < t_hi)
            ]
            n = len(sub)
            if n < 10:
                print(f"{'—':^16}", end="")
            else:
                winrate = sub["cheap_won"].mean()
                edge = winrate - sub["cheap_price"].mean()
                print(f"{winrate:.1%} n={n:<5} {edge:+.2f}".center(16), end="")
        print()

    # --- 3. Simulation: "buy the cheap side ONCE per window at chosen time/price" ---
    print()
    print("=" * 90)
    print("SIMULATED STRATEGY: buy cheap side once per window if price in range")
    print("Holding to settlement. PnL per share = winner? (1-price) : -price")
    print("=" * 90)

    for p_lo, p_hi in [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
                        (0.10, 0.30), (0.20, 0.40), (0.10, 0.40)]:
        # For each window, find the FIRST second where cheap_price is in range
        sub = df[(df["cheap_price"] >= p_lo) & (df["cheap_price"] < p_hi)]
        if sub.empty:
            continue
        # First-opportunity per window
        first = sub.sort_values("elapsed_sec").groupby("slug").first().reset_index()
        n_windows_entered = len(first)
        n_windows_total = loaded
        pnl_per_share = np.where(
            first["cheap_won"] == 1,
            1.0 - first["cheap_price"],
            -first["cheap_price"],
        )
        total_pnl_per_share = pnl_per_share.sum()
        avg_entry_price = first["cheap_price"].mean()
        avg_entry_time = first["elapsed_sec"].mean()
        winrate = first["cheap_won"].mean()

        print(f"\n  Cheap price in [{p_lo:.2f}, {p_hi:.2f}):")
        print(f"    Windows entered:      {n_windows_entered} / {n_windows_total} "
              f"({100*n_windows_entered/n_windows_total:.1f}%)")
        print(f"    Avg entry price:      ${avg_entry_price:.3f}")
        print(f"    Avg entry time:       {avg_entry_time:.1f}s")
        print(f"    Win rate:             {winrate:.1%}")
        print(f"    Edge (win−price):     {winrate - avg_entry_price:+.3f}")
        print(f"    Total PnL per share:  ${total_pnl_per_share:+.2f} "
              f"(${total_pnl_per_share/n_windows_entered:+.3f}/entry)")


if __name__ == "__main__":
    main()
