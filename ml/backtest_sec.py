"""
Orchestrator: replay the 170 held-out test CSVs through the trained
second-level XGBoost predictor and report honest accuracy + PnL.

For a range of confidence thresholds, computes:
  - coverage (how many windows produced an entry)
  - accuracy (% of entries that matched the winner)
  - PnL (total $ gained/lost on simulated entries at 10 shares each)
  - average entry price, entry elapsed
  - calibration bucketing (confidence vs actual accuracy)

Outputs:
  ml/results/backtest_summary.json
  ml/results/backtest_windows.csv    (one row per window, per threshold)

Usage:
  python3 -m ml.backtest_sec
"""

import json
from pathlib import Path

import pandas as pd

from .paper_replay import replay_window
from .predictor_sec import PredictorSec


REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "price_collector" / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def summarise(rows):
    """rows: list of dict results (entry_side may be None if skipped)."""
    n_total = len(rows)
    entered = [r for r in rows if not r["skipped"]]
    n_entered = len(entered)
    if n_entered == 0:
        return {
            "n_total": n_total, "n_entered": 0, "coverage_pct": 0.0,
            "accuracy_pct": None, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "avg_pnl_per_entry": None,
            "avg_entry_price": None, "avg_entry_elapsed": None,
        }
    wins = sum(1 for r in entered if r["correct"])
    losses = n_entered - wins
    total_pnl = sum(r["pnl"] for r in entered)
    total_shares = sum(r.get("shares", 0) for r in entered)
    return {
        "n_total": n_total,
        "n_entered": n_entered,
        "coverage_pct": round(100.0 * n_entered / n_total, 2),
        "accuracy_pct": round(100.0 * wins / n_entered, 2),
        "wins": wins,
        "losses": losses,
        "total_pnl": round(total_pnl, 2),
        "total_shares": total_shares,
        "avg_pnl_per_entry": round(total_pnl / n_entered, 4),
        "avg_entry_price": round(
            sum(r["entry_price"] for r in entered) / n_entered, 4
        ),
        "avg_entry_elapsed": round(
            sum(r["entry_elapsed"] for r in entered) / n_entered, 1
        ),
    }


def calibration(rows, n_buckets=10):
    """Bucketed (confidence, accuracy) — uses peak confidence regardless of entry."""
    buckets = [[] for _ in range(n_buckets)]
    for r in rows:
        # Use entry_confidence if we entered, else the higher of the two peak
        # probabilities as the model's "best shot" at calling this window.
        conf = r.get("entry_confidence")
        side = r.get("entry_side")
        if conf is None:
            # No entry: use the peak probability the model expressed, on the
            # side it was leaning toward.
            if r["peak_prob_up"] >= r["peak_prob_down"]:
                conf = r["peak_prob_up"]
                side = "Up"
            else:
                conf = r["peak_prob_down"]
                side = "Down"
        bucket_idx = min(int(conf * n_buckets), n_buckets - 1)
        correct = side == r["winner"]
        buckets[bucket_idx].append(correct)
    out = []
    for i, b in enumerate(buckets):
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        if not b:
            out.append({"conf_range": [lo, hi], "n": 0, "accuracy": None})
        else:
            out.append({
                "conf_range": [round(lo, 2), round(hi, 2)],
                "n": len(b),
                "accuracy": round(sum(b) / len(b) * 100, 2),
            })
    return out


def main():
    # Load test split from the training manifest
    manifest_path = MODEL_DIR / "split_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run `python3 -m ml.train_sec` first.")
        return
    manifest = json.loads(manifest_path.read_text())
    test_files = [DATA_DIR / f for f in manifest["test_files"]]
    print(f"Test set: {len(test_files)} windows "
          f"(epochs {manifest['test_epoch_range'][0]}..{manifest['test_epoch_range'][1]})")

    predictor = PredictorSec()
    print(f"Loaded predictor: model={MODEL_DIR / 'xgb_sec.joblib'}")

    # For each threshold, run the full replay. Predictor is stateful per window
    # (reset inside replay_window) so we can reuse it.
    all_window_rows = []   # one row per (window, threshold)
    summary_by_thresh = {}

    # First pass: compute per-window results ONCE at the lowest threshold.
    # At lower thresholds, more windows enter — but replaying once per
    # threshold is cheap (each window is ~few hundred ticks). Simpler and
    # harder to get wrong than caching predictions.
    for thresh in THRESHOLDS:
        print(f"\n-- Threshold {thresh:.2f} --")
        rows = []
        for i, f in enumerate(test_files):
            r = replay_window(str(f), predictor, conf_threshold=thresh)
            if r is None:
                continue
            r["threshold"] = thresh
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  replayed {i + 1}/{len(test_files)}")
        summ = summarise(rows)
        summary_by_thresh[f"{thresh:.2f}"] = summ
        print(f"  coverage: {summ['coverage_pct']}%  "
              f"acc: {summ['accuracy_pct']}%  "
              f"pnl: ${summ['total_pnl']}  "
              f"avg_entry_price: {summ['avg_entry_price']}  "
              f"avg_entry_t: {summ['avg_entry_elapsed']}s")
        all_window_rows.extend(rows)

    # Calibration at threshold=0.5 rows (includes skipped windows' peak probs)
    baseline = [r for r in all_window_rows if r["threshold"] == THRESHOLDS[0]]
    calib = calibration(baseline, n_buckets=10)

    # --- Write outputs ---
    summary = {
        "test_window_count": len(test_files),
        "model_path": str(MODEL_DIR / "xgb_sec.joblib"),
        "by_threshold": summary_by_thresh,
        "calibration_on_peak_prob": calib,
    }
    (RESULTS_DIR / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    pd.DataFrame(all_window_rows).to_csv(
        RESULTS_DIR / "backtest_windows.csv", index=False
    )
    print(f"\nWrote {RESULTS_DIR / 'backtest_summary.json'}")
    print(f"Wrote {RESULTS_DIR / 'backtest_windows.csv'}")

    # --- Human-readable recap ---
    print("\n" + "=" * 84)
    print("  thresh  entries  acc     pnl       shares  avg_price  avg_t")
    print("-" * 84)
    for thresh in THRESHOLDS:
        s = summary_by_thresh[f"{thresh:.2f}"]
        print(f"   {thresh:.2f}  {s['n_entered']:>5}/{s['n_total']}  "
              f"{s['accuracy_pct'] if s['accuracy_pct'] is not None else '—':>5}%  "
              f"${s['total_pnl']:>7}  "
              f"{s['total_shares']:>6}  "
              f"{s['avg_entry_price'] if s['avg_entry_price'] is not None else '—':>8}   "
              f"{s['avg_entry_elapsed'] if s['avg_entry_elapsed'] is not None else '—'}s")
    print("=" * 84)
    print("\nCalibration (peak-confidence bucket vs actual winner agreement):")
    for b in calib:
        if b["n"] == 0:
            continue
        lo, hi = b["conf_range"]
        print(f"  [{lo:.1f}-{hi:.1f})  n={b['n']:>4}  acc={b['accuracy']}%")


if __name__ == "__main__":
    main()
