"""
Fast EV backtest. ONE pass per window: compute (ev_up, ev_down, up_ask,
down_ask) at every integer second from MIN_ENTRY..MAX_ENTRY. Store the
trace. Post-process cheaply for:

  - Threshold sweep on the EV model
  - Cheap-side A1/B1 strategies on the same windows (apples-to-apples)
  - Half-1 vs Half-2 stability check

10x faster than the original backtest because the expensive per-tick model
inference only happens once per window, not once per threshold.
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .ev_predictor import EVPredictor

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "price_collector" / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SHARES = 10
EVENTS_PER_DAY = 288
MIN_ENTRY = 30
MAX_ENTRY = 240
THRESHOLDS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07]


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


def _parse_row(r):
    try:
        t = float(r["elapsed_sec"])
        up_ask = float(r["up_ask"])
        down_ask = float(r["down_ask"])
        btc = float(r["btc_price"])
    except (TypeError, ValueError, KeyError):
        return None
    if not (up_ask > 0 and down_ask > 0 and btc > 0):
        return None
    up_bid = r.get("up_bid", 0.0); down_bid = r.get("down_bid", 0.0)
    up_bid = float(up_bid) if pd.notna(up_bid) else 0.0
    down_bid = float(down_bid) if pd.notna(down_bid) else 0.0
    return t, up_bid, up_ask, down_bid, down_ask, btc


def build_trace(csv_path, predictor):
    """Walk CSV tick-by-tick once. At each integer-second boundary in
    [MIN_ENTRY, MAX_ENTRY] record (t, up_ask, down_ask, ev_up, ev_down).
    Returns (winner, list_of_trace_rows) or None."""
    predictor.reset()
    winner = read_winner(csv_path)
    if winner not in ("Up", "Down"):
        return None
    try:
        raw = pd.read_csv(csv_path, comment="#")
    except Exception:
        return None
    if len(raw) < 10:
        return None

    next_predict = 1
    trace = []
    for _, r in raw.iterrows():
        parsed = _parse_row(r)
        if parsed is None:
            continue
        t, up_bid, up_ask, down_bid, down_ask, btc = parsed
        predictor.add_tick(t, up_bid, up_ask, down_bid, down_ask, btc)

        while next_predict <= t and next_predict <= MAX_ENTRY:
            tp = next_predict
            next_predict += 1
            if tp < MIN_ENTRY:
                continue
            ev_up, ev_down = predictor.predict(tp)
            if ev_up is None:
                continue
            trace.append((tp, up_ask, down_ask, ev_up, ev_down))
    return winner, trace


def decide_ev(trace, threshold):
    """Pick first tick where max(ev_up, ev_down) > threshold."""
    for t, up_ask, down_ask, ev_up, ev_down in trace:
        if ev_up >= ev_down:
            side, price, ev = "Up", up_ask, ev_up
        else:
            side, price, ev = "Down", down_ask, ev_down
        if ev > threshold:
            return {"side": side, "price": price, "t": t, "ev": ev}
    return None


def decide_cheap(trace, p_lo, p_hi, t_min=MIN_ENTRY, t_max=MAX_ENTRY):
    """Pick first tick where cheap side's ask is in [p_lo, p_hi]."""
    for t, up_ask, down_ask, _, _ in trace:
        if t < t_min or t > t_max:
            continue
        if up_ask <= down_ask:
            side, price = "Up", up_ask
        else:
            side, price = "Down", down_ask
        if p_lo <= price <= p_hi:
            return {"side": side, "price": price, "t": t, "ev": None}
    return None


def decide_cheap_reversal(trace, p_lo, p_hi, uptick_c=0.02):
    """Cheap side in [p_lo, p_hi] AND tick UP uptick_c from local min."""
    running_min_up = 1.0
    running_min_down = 1.0
    for t, up_ask, down_ask, _, _ in trace:
        running_min_up = min(running_min_up, up_ask)
        running_min_down = min(running_min_down, down_ask)
        if up_ask <= down_ask:
            side, price, rmin = "Up", up_ask, running_min_up
        else:
            side, price, rmin = "Down", down_ask, running_min_down
        if p_lo <= price <= p_hi and price >= rmin + uptick_c:
            return {"side": side, "price": price, "t": t, "ev": None}
    return None


def settle(entry, winner):
    if entry is None:
        return {"entered": False, "pnl_per_share": 0.0}
    won = (entry["side"] == winner)
    pnl = (1.0 - entry["price"]) if won else -entry["price"]
    return {
        "entered": True, "side": entry["side"],
        "price": entry["price"], "t": entry["t"],
        "ev": entry["ev"],
        "won": won, "pnl_per_share": pnl,
    }


def summarise(rows, total_windows):
    entered = [r for r in rows if r.get("entered")]
    if not entered:
        return {"n_entered": 0, "n_total": total_windows,
                "win_rate": None, "avg_price": None, "avg_t": None,
                "edge": None, "daily_pnl": 0.0, "yearly_pnl": 0.0}
    n = len(entered)
    wins = sum(1 for r in entered if r["won"])
    pnl = sum(r["pnl_per_share"] for r in entered)
    avg_p = sum(r["price"] for r in entered) / n
    avg_t = sum(r["t"] for r in entered) / n
    win_rate = wins / n
    entry_rate = n / total_windows
    daily_entries = EVENTS_PER_DAY * entry_rate
    daily_pnl = daily_entries * (pnl / n) * SHARES
    return {
        "n_entered": n, "n_total": total_windows,
        "win_rate": win_rate, "avg_price": avg_p, "avg_t": avg_t,
        "edge": win_rate - avg_p, "pnl_per_share_total": pnl,
        "daily_pnl": daily_pnl, "yearly_pnl": daily_pnl * 365,
    }


def print_row(label, s):
    if s["n_entered"] == 0:
        print(f"  {label:<32}  0/{s['n_total']}    —       —         —       —      $0        $0")
        return
    print(f"  {label:<32} {s['n_entered']:>3}/{s['n_total']}  "
          f"{s['win_rate']*100:>5.1f}%  "
          f"${s['avg_price']:>5.3f}  "
          f"{s['avg_t']:>5.1f}s  "
          f"{s['edge']:>+6.3f}  "
          f"${s['daily_pnl']:>+7.2f}  "
          f"${s['yearly_pnl']:>+8.0f}")


def main():
    stats_path = MODEL_DIR / "ev_training_stats.json"
    if not stats_path.exists():
        print("ERROR: run `python3 -m ml.ev_train` first.")
        return
    stats = json.loads(stats_path.read_text())
    test_files = [DATA_DIR / f for f in stats["test_files"]]
    n_total = len(test_files)
    print(f"Test set: {n_total} windows (one day of data)")

    pred = EVPredictor()
    print("Building traces (one tick-by-tick pass per window)...")

    traces = []
    t0 = time.time()
    for i, f in enumerate(test_files):
        r = build_trace(str(f), pred)
        if r is None:
            continue
        winner, trace = r
        traces.append({"slug": Path(f).stem, "winner": winner, "trace": trace})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_total} done  ({time.time() - t0:.1f}s)")
    print(f"  Built {len(traces)} traces in {time.time() - t0:.1f}s")

    # ---------- 1. EV threshold sweep ----------
    print("\n" + "=" * 96)
    print("1. EV MODEL — threshold sweep on the full 288-window test set")
    print("=" * 96)
    print(f"  {'label':<32} {'entries':>8}  {'win%':>6}  {'price':>5}    {'t':>5}     "
          f"{'edge':>7}  {'daily':>9}  {'yearly':>9}")
    print("-" * 96)

    ev_summaries = {}
    for th in THRESHOLDS:
        rows = [settle(decide_ev(tr["trace"], th), tr["winner"]) for tr in traces]
        s = summarise(rows, len(traces))
        ev_summaries[th] = s
        print_row(f"EV th>={th:.3f}", s)

    # ---------- 2. Cheap-side strategies on SAME test set ----------
    print("\n" + "=" * 96)
    print("2. CHEAP-SIDE rules — SAME test set (apples-to-apples)")
    print("=" * 96)
    print(f"  {'label':<32} {'entries':>8}  {'win%':>6}  {'price':>5}    {'t':>5}     "
          f"{'edge':>7}  {'daily':>9}  {'yearly':>9}")
    print("-" * 96)

    cheap_configs = [
        ("A1 $0.20-$0.30 baseline",      decide_cheap, dict(p_lo=0.20, p_hi=0.30)),
        ("A2 $0.30-$0.40 baseline",      decide_cheap, dict(p_lo=0.30, p_hi=0.40)),
        ("A3 $0.20-$0.40 baseline",      decide_cheap, dict(p_lo=0.20, p_hi=0.40)),
        ("B1 $0.30-$0.40 reversal+2c",   decide_cheap_reversal, dict(p_lo=0.30, p_hi=0.40, uptick_c=0.02)),
        ("B2 $0.20-$0.40 reversal+2c",   decide_cheap_reversal, dict(p_lo=0.20, p_hi=0.40, uptick_c=0.02)),
    ]
    for label, fn, kwargs in cheap_configs:
        rows = [settle(fn(tr["trace"], **kwargs), tr["winner"]) for tr in traces]
        s = summarise(rows, len(traces))
        print_row(label, s)

    # ---------- 3. Half-1 vs Half-2 stability ----------
    print("\n" + "=" * 96)
    print("3. STABILITY CHECK — Half-1 vs Half-2 of the 288-window test")
    print("=" * 96)
    mid = len(traces) // 2
    half1 = traces[:mid]
    half2 = traces[mid:]

    # Focus on the two candidate strategies: EV@0.02 and cheap-side A1
    def run_ev(subset, th):
        rows = [settle(decide_ev(tr["trace"], th), tr["winner"]) for tr in subset]
        return summarise(rows, len(subset))

    def run_cheap_a1(subset):
        rows = [settle(decide_cheap(tr["trace"], 0.20, 0.30), tr["winner"])
                for tr in subset]
        return summarise(rows, len(subset))

    def run_cheap_b1(subset):
        rows = [settle(decide_cheap_reversal(tr["trace"], 0.30, 0.40, 0.02),
                        tr["winner"]) for tr in subset]
        return summarise(rows, len(subset))

    print(f"  Half-1: {len(half1)} windows")
    print(f"  Half-2: {len(half2)} windows\n")
    print(f"  {'strategy':<32} {'H1 daily':>10}  {'H1 edge':>8}  "
          f"{'H2 daily':>10}  {'H2 edge':>8}  {'stable?':>8}")
    print("-" * 96)
    for label, strategies in [
        ("EV th=0.02", [run_ev(half1, 0.02), run_ev(half2, 0.02)]),
        ("EV th=0.03", [run_ev(half1, 0.03), run_ev(half2, 0.03)]),
        ("Cheap-side A1", [run_cheap_a1(half1), run_cheap_a1(half2)]),
        ("Cheap-side B1", [run_cheap_b1(half1), run_cheap_b1(half2)]),
    ]:
        s1, s2 = strategies
        daily1 = s1["daily_pnl"]; daily2 = s2["daily_pnl"]
        edge1 = s1["edge"]; edge2 = s2["edge"]
        stable = "YES" if (daily1 > 0 and daily2 > 0) else "NO"
        e1 = f"{edge1:+.3f}" if edge1 is not None else "—"
        e2 = f"{edge2:+.3f}" if edge2 is not None else "—"
        print(f"  {label:<32} ${daily1:>+7.2f}    {e1:>7}   "
              f"${daily2:>+7.2f}    {e2:>7}      {stable:>6}")

    # ---------- Save ----------
    out = {
        "n_test_windows": len(traces),
        "ev_threshold_sweep": {f"{th:.3f}": ev_summaries[th] for th in THRESHOLDS},
    }
    (RESULTS_DIR / "ev_backtest_fast.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved summary to {RESULTS_DIR / 'ev_backtest_fast.json'}")


if __name__ == "__main__":
    main()
