"""
Tick-by-tick replay of the EV strategy on the 170 held-out test windows.

At each integer second, predict EV_Up and EV_Down from the buffer so far.
If max(EV) > threshold and we haven't entered → buy that side. Hold.

Reports daily PnL @ 10 shares @ 288 events/day across a threshold sweep.
Also compares to the cheap-side A1 baseline for reference.
"""

import json
import os
import re
from pathlib import Path

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

THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]


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
    up_bid = r.get("up_bid", 0.0)
    down_bid = r.get("down_bid", 0.0)
    up_bid = float(up_bid) if pd.notna(up_bid) else 0.0
    down_bid = float(down_bid) if pd.notna(down_bid) else 0.0
    return t, up_bid, up_ask, down_bid, down_ask, btc


def replay_window(csv_path, predictor, threshold):
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
    entry = None
    for _, r in raw.iterrows():
        parsed = _parse_row(r)
        if parsed is None:
            continue
        elapsed, up_bid, up_ask, down_bid, down_ask, btc = parsed
        predictor.add_tick(elapsed, up_bid, up_ask, down_bid, down_ask, btc)

        if entry is not None:
            continue

        while next_predict <= elapsed and next_predict <= MAX_ENTRY:
            t = next_predict
            next_predict += 1
            if t < MIN_ENTRY:
                continue
            ev_up, ev_down = predictor.predict(t)
            if ev_up is None:
                continue
            # Pick the best side. Require > threshold to enter.
            if ev_up >= ev_down:
                best_side, best_ev, best_price = "Up", ev_up, up_ask
            else:
                best_side, best_ev, best_price = "Down", ev_down, down_ask
            if best_ev > threshold:
                entry = {
                    "side": best_side, "price": best_price,
                    "t": t, "ev": best_ev,
                }
                break

    if entry is None:
        return {"slug": Path(csv_path).stem, "winner": winner,
                "entered": False, "pnl_per_share": 0.0}
    won = (entry["side"] == winner)
    pnl = (1.0 - entry["price"]) if won else -entry["price"]
    return {
        "slug": Path(csv_path).stem, "winner": winner,
        "entered": True, "side": entry["side"],
        "price": entry["price"], "t": entry["t"],
        "predicted_ev": entry["ev"],
        "pnl_per_share": pnl, "correct": won,
    }


def summarise(rows, label, total_windows):
    entered = [r for r in rows if r and r.get("entered")]
    if not entered:
        return {
            "label": label, "entries": 0, "total": total_windows,
            "win_rate": None, "edge": None,
            "daily_pnl_10sh": 0.0, "yearly_pnl_10sh": 0.0,
        }
    n = len(entered)
    wins = sum(1 for r in entered if r["correct"])
    total_pnl = sum(r["pnl_per_share"] for r in entered)
    avg_price = sum(r["price"] for r in entered) / n
    avg_t = sum(r["t"] for r in entered) / n
    avg_ev = sum(r["predicted_ev"] for r in entered) / n
    entry_rate = n / total_windows
    daily_entries = EVENTS_PER_DAY * entry_rate
    pnl_per_entry = total_pnl / n
    daily_pnl = daily_entries * pnl_per_entry * SHARES
    return {
        "label": label,
        "entries": n, "total": total_windows,
        "entry_rate_pct": 100 * entry_rate,
        "win_rate_pct": 100 * wins / n,
        "avg_price": avg_price, "avg_t": avg_t,
        "avg_predicted_ev": avg_ev,
        "edge_per_share": wins / n - avg_price,
        "total_pnl_per_share": total_pnl,
        "daily_pnl_10sh": daily_pnl,
        "yearly_pnl_10sh": daily_pnl * 365,
    }


def main():
    manifest = MODEL_DIR / "ev_training_stats.json"
    if not manifest.exists():
        print("ERROR: ev_training_stats.json not found. Run `python3 -m ml.ev_train` first.")
        return
    stats = json.loads(manifest.read_text())
    test_files = [DATA_DIR / f for f in stats["test_files"]]
    print(f"Test set: {len(test_files)} windows")

    pred = EVPredictor()
    print(f"Loaded EV models.")

    all_rows_by_thresh = {}
    print("\n" + "=" * 96)
    print("EV STRATEGY @ 10 shares, 288 events/day — tick-by-tick replay")
    print("=" * 96)
    hdr = f"{'thresh':>7}  {'entries':>8}  {'win%':>6}  {'avg_price':>10}  {'avg_t':>6}  {'edge':>7}  {'daily':>9}  {'yearly':>9}"
    print(hdr)
    print("-" * 96)

    for th in THRESHOLDS:
        rows = []
        for f in test_files:
            r = replay_window(str(f), pred, th)
            if r is not None:
                rows.append(r)
        all_rows_by_thresh[f"{th:.3f}"] = rows
        s = summarise(rows, f"th={th:.3f}", len(test_files))
        if s["entries"] == 0:
            print(f"  {th:>5.3f}   0/{s['total']}   —       —         —      —     $0       $0")
            continue
        print(f"  {th:>5.3f}   {s['entries']:>3}/{s['total']}  "
              f"{s['win_rate_pct']:>5.1f}  "
              f"${s['avg_price']:>8.3f}  "
              f"{s['avg_t']:>5.1f}s  "
              f"{s['edge_per_share']:>+6.3f}  "
              f"${s['daily_pnl_10sh']:>+7.2f}  "
              f"${s['yearly_pnl_10sh']:>+8.0f}")

    print("=" * 96)

    # Per-entry distribution inspection at threshold 0 (maximum entries)
    best_th = max(THRESHOLDS, key=lambda t: summarise(
        all_rows_by_thresh[f"{t:.3f}"], "", len(test_files)
    )["daily_pnl_10sh"])
    best_rows = all_rows_by_thresh[f"{best_th:.3f}"]
    best_entered = [r for r in best_rows if r.get("entered")]

    if best_entered:
        print(f"\nBEST THRESHOLD: {best_th:.3f}")
        print(f"  Side breakdown:")
        side_counts = {}
        for r in best_entered:
            side_counts[r["side"]] = side_counts.get(r["side"], 0) + 1
        for side, cnt in side_counts.items():
            wins = sum(1 for r in best_entered if r["side"] == side and r["correct"])
            print(f"    {side}: {cnt} entries, {wins} wins ({100*wins/cnt:.1f}%)")

        print(f"\n  Entry price distribution:")
        bins = [(0.0, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
                (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.0)]
        for lo, hi in bins:
            sub = [r for r in best_entered if lo <= r["price"] < hi]
            if not sub:
                continue
            wins = sum(1 for r in sub if r["correct"])
            pnl = sum(r["pnl_per_share"] for r in sub)
            print(f"    ${lo:.2f}-${hi:.2f}: n={len(sub):>3}, "
                  f"win%={100*wins/len(sub):.1f}, pnl/share=${pnl/len(sub):+.4f}")

        print(f"\n  Entry time distribution:")
        time_bins = [(30, 60), (60, 90), (90, 120), (120, 150),
                      (150, 180), (180, 210), (210, 240)]
        for lo, hi in time_bins:
            sub = [r for r in best_entered if lo <= r["t"] < hi]
            if not sub:
                continue
            wins = sum(1 for r in sub if r["correct"])
            pnl = sum(r["pnl_per_share"] for r in sub)
            print(f"    t={lo}-{hi}s: n={len(sub):>3}, "
                  f"win%={100*wins/len(sub):.1f}, pnl/share=${pnl/len(sub):+.4f}")

    # Save everything
    out = {}
    for th, rows in all_rows_by_thresh.items():
        s = summarise(rows, th, len(test_files))
        out[th] = s
    (RESULTS_DIR / "ev_backtest_summary.json").write_text(json.dumps(out, indent=2))
    flat_rows = []
    for th, rows in all_rows_by_thresh.items():
        for r in rows:
            flat_rows.append({**r, "threshold": th})
    pd.DataFrame(flat_rows).to_csv(
        RESULTS_DIR / "ev_backtest_windows.csv", index=False
    )
    print(f"\nWrote {RESULTS_DIR / 'ev_backtest_summary.json'}")
    print(f"Wrote {RESULTS_DIR / 'ev_backtest_windows.csv'}")


if __name__ == "__main__":
    main()
