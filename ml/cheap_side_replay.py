"""
Tick-by-tick replay of the "buy cheap side" strategy.

This is the HONEST version: we walk the raw CSV rows in the order the
collector recorded them (simulating a live WebSocket stream). At each
tick we only see data up to that moment — no future peeking, no
pre-resampled grid.

Rule tested (baseline):
  If cheap_side_ask is in [p_lo, p_hi] at this tick AND we haven't
  entered yet AND elapsed_sec is within [t_min, t_max] → BUY.
  Hold to settlement.

We also test three refinements:
  B. Wait for cheap price to tick UP 2c from its local minimum (reversal confirm)
  C. Filter by BTC momentum (only enter if BTC is decelerating)
  D. Time-gate entries (avoid very early + very late)

Usage: python3 -m ml.cheap_side_replay
"""

import csv
import glob
import os
import re
from collections import deque
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "price_collector" / "data"
SHARES = 100   # per entry; change here to rescale PnL


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


def _parse_row(row):
    try:
        t = float(row["elapsed_sec"])
        up_ask = float(row["up_ask"])
        down_ask = float(row["down_ask"])
        btc = float(row["btc_price"])
    except (TypeError, ValueError, KeyError):
        return None
    if not (up_ask > 0 and down_ask > 0 and btc > 0):
        return None
    return t, up_ask, down_ask, btc


def replay_baseline(csv_path, p_lo, p_hi, t_min=30, t_max=240):
    """Rule A: enter on the first tick where cheap_ask is in [p_lo, p_hi]
    and t is in [t_min, t_max]. Hold to settlement.
    Returns (entered, entry_side, entry_price, entry_t, winner, pnl_per_share)."""
    winner = read_winner(csv_path)
    if winner not in ("Up", "Down"):
        return None

    raw = pd.read_csv(csv_path, comment="#")
    if len(raw) < 10:
        return None

    for _, row in raw.iterrows():
        parsed = _parse_row(row)
        if parsed is None:
            continue
        t, up_ask, down_ask, btc = parsed
        if t < t_min or t > t_max:
            continue
        if up_ask <= down_ask:
            cheap_side, cheap_price = "Up", up_ask
        else:
            cheap_side, cheap_price = "Down", down_ask
        if p_lo <= cheap_price <= p_hi:
            won = (cheap_side == winner)
            pnl = (1.0 - cheap_price) if won else -cheap_price
            return {
                "entered": True, "side": cheap_side,
                "price": cheap_price, "t": t,
                "winner": winner, "pnl_per_share": pnl,
            }
    return {"entered": False, "winner": winner, "pnl_per_share": 0.0}


def replay_reversal_confirm(csv_path, p_lo, p_hi, t_min=30, t_max=240,
                             uptick_c=0.02):
    """Rule B: same as baseline but REQUIRE the cheap side to tick up by
    uptick_c from its local minimum before entering. This filters
    'catching a falling knife'.
    """
    winner = read_winner(csv_path)
    if winner not in ("Up", "Down"):
        return None

    raw = pd.read_csv(csv_path, comment="#")
    if len(raw) < 10:
        return None

    running_min_up = 1.0
    running_min_down = 1.0
    for _, row in raw.iterrows():
        parsed = _parse_row(row)
        if parsed is None:
            continue
        t, up_ask, down_ask, btc = parsed
        running_min_up = min(running_min_up, up_ask)
        running_min_down = min(running_min_down, down_ask)
        if t < t_min or t > t_max:
            continue
        if up_ask <= down_ask:
            cheap_side, cheap_price = "Up", up_ask
            local_min = running_min_up
        else:
            cheap_side, cheap_price = "Down", down_ask
            local_min = running_min_down
        if p_lo <= cheap_price <= p_hi and cheap_price >= local_min + uptick_c:
            won = (cheap_side == winner)
            pnl = (1.0 - cheap_price) if won else -cheap_price
            return {
                "entered": True, "side": cheap_side,
                "price": cheap_price, "t": t,
                "winner": winner, "pnl_per_share": pnl,
            }
    return {"entered": False, "winner": winner, "pnl_per_share": 0.0}


def replay_btc_decel(csv_path, p_lo, p_hi, t_min=30, t_max=240,
                      decel_threshold=0.0):
    """Rule C: enter only when BTC momentum is DECELERATING.
    Deceleration = (btc_now - btc_30s_ago) moves toward zero compared to
    (btc_30s_ago - btc_60s_ago). Using a rolling 60-tick buffer of btc prices.
    """
    winner = read_winner(csv_path)
    if winner not in ("Up", "Down"):
        return None

    raw = pd.read_csv(csv_path, comment="#")
    if len(raw) < 10:
        return None

    btc_buffer = deque(maxlen=120)   # last ~40s at 3 writes/sec
    for _, row in raw.iterrows():
        parsed = _parse_row(row)
        if parsed is None:
            continue
        t, up_ask, down_ask, btc = parsed
        btc_buffer.append((t, btc))
        if t < t_min or t > t_max:
            continue
        if up_ask <= down_ask:
            cheap_side, cheap_price = "Up", up_ask
        else:
            cheap_side, cheap_price = "Down", down_ask
        if not (p_lo <= cheap_price <= p_hi):
            continue

        # Need at least 60 seconds of BTC buffer to measure deceleration.
        if len(btc_buffer) < 60:
            continue
        btc_now = btc
        btc_30 = None
        btc_60 = None
        for bt, bp in btc_buffer:
            if btc_30 is None and t - bt <= 30:
                btc_30 = bp
                break
        for bt, bp in btc_buffer:
            if t - bt >= 60:
                btc_60 = bp
                break
        if btc_30 is None or btc_60 is None:
            continue
        mom_recent = abs(btc_now - btc_30)
        mom_older = abs(btc_30 - btc_60)
        if mom_recent <= mom_older + decel_threshold:
            won = (cheap_side == winner)
            pnl = (1.0 - cheap_price) if won else -cheap_price
            return {
                "entered": True, "side": cheap_side,
                "price": cheap_price, "t": t,
                "winner": winner, "pnl_per_share": pnl,
            }
    return {"entered": False, "winner": winner, "pnl_per_share": 0.0}


def summarise(results, label, total_windows):
    entered = [r for r in results if r and r.get("entered")]
    if not entered:
        print(f"\n  {label}: NO ENTRIES")
        return
    n = len(entered)
    wins = sum(1 for r in entered if r["pnl_per_share"] > 0)
    total_pnl_per_share = sum(r["pnl_per_share"] for r in entered)
    avg_price = sum(r["price"] for r in entered) / n
    avg_t = sum(r["t"] for r in entered) / n
    winrate = wins / n
    edge = winrate - avg_price
    print(f"\n  {label}:")
    print(f"    entered:       {n:>4} / {total_windows}  "
          f"({100*n/total_windows:.1f}%)")
    print(f"    win rate:      {winrate:.1%}   "
          f"avg entry: ${avg_price:.3f}   avg t: {avg_t:.1f}s")
    print(f"    edge:          {edge:+.3f} per share")
    print(f"    total PnL:     ${total_pnl_per_share:+.2f}/share-unit "
          f"(× {SHARES} shares = ${total_pnl_per_share * SHARES:+.2f})")


def main():
    files = sorted(glob.glob(str(DATA_DIR / "btc-updown-5m-*.csv")))
    print(f"Walking {len(files)} CSVs tick-by-tick (no lookahead, no resample)...")

    total = 0
    for f in files:
        if read_winner(f) in ("Up", "Down"):
            total += 1
    print(f"  {total} have valid winners\n")

    # Parameter sets to test
    configs = [
        # (label, fn, kwargs)
        ("A1  baseline $0.20-$0.30, t=30-240",
         replay_baseline, dict(p_lo=0.20, p_hi=0.30, t_min=30, t_max=240)),
        ("A2  baseline $0.30-$0.40, t=30-240",
         replay_baseline, dict(p_lo=0.30, p_hi=0.40, t_min=30, t_max=240)),
        ("A3  baseline $0.20-$0.40, t=30-240",
         replay_baseline, dict(p_lo=0.20, p_hi=0.40, t_min=30, t_max=240)),
        ("A4  baseline $0.30-$0.40, t=60-180",
         replay_baseline, dict(p_lo=0.30, p_hi=0.40, t_min=60, t_max=180)),
        ("A5  baseline $0.30-$0.40, t=90-210",
         replay_baseline, dict(p_lo=0.30, p_hi=0.40, t_min=90, t_max=210)),
        ("B1  reversal-confirm $0.30-$0.40, 2c uptick",
         replay_reversal_confirm, dict(p_lo=0.30, p_hi=0.40, uptick_c=0.02)),
        ("B2  reversal-confirm $0.20-$0.40, 2c uptick",
         replay_reversal_confirm, dict(p_lo=0.20, p_hi=0.40, uptick_c=0.02)),
        ("C1  BTC decelerating $0.30-$0.40",
         replay_btc_decel, dict(p_lo=0.30, p_hi=0.40)),
        ("C2  BTC decelerating $0.20-$0.40",
         replay_btc_decel, dict(p_lo=0.20, p_hi=0.40)),
    ]

    for label, fn, kwargs in configs:
        results = []
        for f in files:
            r = fn(f, **kwargs)
            if r is not None:
                results.append(r)
        summarise(results, label, total)

    print("\n" + "=" * 72)
    print("All numbers above are TICK-BY-TICK replays. No lookahead.")
    print("No bfill. No full-window resample. Just: read row, decide, hold.")
    print("=" * 72)


if __name__ == "__main__":
    main()
