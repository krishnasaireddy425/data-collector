"""
V3 Exit Classifier — a precision-focused model for early-exiting losing V3 trades.

Goal
----
Given V3 entered a window at (t_entry, side, entry_price), at each post-entry
tick predict whether this entry will be a LOSER at settlement. Tune the
decision threshold for HIGH PRECISION (≥99%) on the "EXIT" class. Accept
low recall — catching 20% of losers at 99% precision is the target.

Data
----
  analysis/supabase_export/v3_windows_rows.csv       — 1,565 V3 window records
  price_collector/data/*.csv                         — per-tick price series
  ml/models/xgb_sec.joblib + calibrator.joblib       — existing V3 model (for
                                                        re-prediction feature)

Only rows where entry_made=true are used. Split is chronological (60/20/20
by open_epoch) on the entries.

Features (at each post-entry tick)
----------------------------------
  * V3's 44 continuous-time features at the current tick (FEATURES in
    features_sec.py).
  * V3's own re-predicted probability + its delta vs entry_prob_up.
  * Entry context: entry_elapsed_sec, entry_price, entry_side_is_up,
    entry_prob_up, entry_confidence.
  * Diff features: our_side_ask − entry_price, our_side_bid − entry_price,
    btc_price − btc_at_entry.
  * Time: t_since_entry, t_remaining.

Label: 1 if entry_side != winner (entry will LOSE at settlement), else 0.

Output
------
  ml/models/v3_exit_xgb.joblib          — trained classifier
  ml/models/v3_exit_calibrator.joblib   — isotonic calibrator (val)
  ml/models/v3_exit_thresholds.json     — thresholds picked on val for 99/98/95 pct
  ml/results/v3_exit_report.md          — full metrics + PnL simulation

Reports: for each precision target (0.99, 0.98, 0.95, 0.90):
  - Threshold value (picked on VAL)
  - TEST precision, recall at that threshold
  - # exits, # correct-exits (losers caught), # false-positives (winners exited)
  - Total $ savings on caught losers
  - Total $ lost on false-positive winners exited
  - Net PnL impact vs no-exit baseline
  - Average exit time (t)

Usage
-----
  cd /Users/krishnasaireddypeddinti/data-collector
  python3 -m ml.v3_exit_classifier
"""

import csv
import json
import math
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, average_precision_score

from .features_sec import FEATURES, compute_features_batch, resample_window


REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "price_collector" / "data"
MODEL_DIR = REPO / "ml" / "models"
RESULTS_DIR = REPO / "ml" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
V3_WINDOWS_CSV = REPO / "analysis" / "supabase_export" / "v3_windows_rows.csv"

SHARES = 10
MAX_EXIT_T = 270           # last second at which we'll try to exit
TICK_STEP = 1              # one sample per second after entry
HYGIENE_FIRST_TICK_MAX = 5 # skip windows whose first raw tick > this second

# Precision targets for threshold tuning
PRECISION_TARGETS = [0.99, 0.98, 0.95, 0.90]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def epoch_of(slug_or_path):
    s = str(slug_or_path)
    m = re.search(r"-(\d{10,})(?:\.csv)?$", s)
    return int(m.group(1)) if m else 0


def load_v3_entries():
    """Return list of entry dicts from v3_windows_rows.csv, filtered to
    entry_made=True and sorted chronologically."""
    rows = []
    with open(V3_WINDOWS_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if (r.get("entry_made") or "").lower() != "true":
                continue
            try:
                slug = r["slug"]
                open_epoch = int(r["open_epoch"])
                entry_elapsed = float(r["entry_elapsed_sec"])
                entry_side = r["entry_side"].strip()
                entry_price = float(r["entry_price"])
                entry_prob_up = float(r["entry_prob_up"]) if r.get("entry_prob_up") else None
                entry_confidence = float(r["entry_confidence"]) if r.get("entry_confidence") else None
                winner = (r.get("winner") or "").strip()
                correct = (r.get("correct") or "").lower() == "true"
                pnl = float(r["pnl"]) if r.get("pnl") else None
            except (KeyError, ValueError, TypeError):
                continue
            if entry_side.lower() not in ("up", "down"):
                continue
            if winner.lower() not in ("up", "down"):
                continue
            # Skip entries near the window boundary — can't evaluate exits
            if entry_elapsed >= MAX_EXIT_T - 5:
                continue
            rows.append({
                "slug": slug,
                "open_epoch": open_epoch,
                "entry_elapsed": entry_elapsed,
                "entry_side": entry_side.capitalize(),  # "Up" / "Down"
                "entry_price": entry_price,
                "entry_prob_up": entry_prob_up,
                "entry_confidence": entry_confidence,
                "winner": winner.capitalize(),
                "correct": correct,
                "pnl": pnl,
            })
    rows.sort(key=lambda r: r["open_epoch"])
    return rows


def load_window_ticks(slug):
    """Load and resample a window CSV. Return (df_1s, btc_open) or None."""
    path = DATA_DIR / f"{slug}.csv"
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path, comment="#")
    except Exception:
        return None
    if len(raw) < 10:
        return None
    raw["elapsed_sec"] = pd.to_numeric(raw["elapsed_sec"], errors="coerce")
    rc = raw.dropna(subset=["elapsed_sec"])
    if rc.empty:
        return None
    if float(rc["elapsed_sec"].min()) > HYGIENE_FIRST_TICK_MAX:
        return None
    df = resample_window(raw, max_t=300)
    if df is None or df.empty or len(df) < 301:
        return None
    btc_open = float(df.iloc[0]["btc_price"])
    if btc_open <= 0:
        return None
    return df, btc_open


# ---------------------------------------------------------------------------
# V3 model loader (for re-prediction feature)
# ---------------------------------------------------------------------------

class V3Reprobe:
    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "xgb_sec.joblib")
        cal_path = MODEL_DIR / "calibrator.joblib"
        self.calibrator = joblib.load(cal_path) if cal_path.exists() else None

    def predict_calibrated(self, X):
        p = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.asarray(p, dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature construction per entry
# ---------------------------------------------------------------------------

# Extra features appended to the 44 V3 features. Order matters for the final
# matrix; keep consistent.
EXTRA_COLS = [
    "entry_elapsed_sec_f",     # constant per entry
    "entry_price_f",
    "entry_side_is_up",
    "entry_prob_up_f",
    "entry_confidence_f",
    "t_since_entry",
    "t_remaining",
    "v3_prob_up_now",
    "v3_prob_delta",           # now - entry
    "our_side_prob_now",       # side==Up ? prob_up : 1 - prob_up
    "our_side_prob_delta",
    "our_side_ask_now",
    "our_side_bid_now",
    "opposite_side_ask_now",
    "our_ask_minus_entry",
    "our_bid_minus_entry",
    "btc_since_entry",
]

ALL_COLS = list(FEATURES) + list(EXTRA_COLS)


def build_entry_rows(entry, df, btc_open, v3_feats_all, v3_prob_all):
    """For one V3 entry, build per-post-tick feature rows.

    v3_feats_all : (301, 44) matrix of V3 features at every second
    v3_prob_all  : (301,)    V3 calibrated prob_up at every second

    Returns:
      X        (n, len(ALL_COLS))   float32
      y        (n,)                 int8   label: 1 if entry will lose
      meta     list of dicts with t, our_bid, our_ask, pnl_exit, pnl_hold
    """
    t_entry = entry["entry_elapsed"]
    entry_side = entry["entry_side"]
    entry_side_is_up = 1.0 if entry_side == "Up" else 0.0
    entry_price = entry["entry_price"]
    entry_prob_up = entry["entry_prob_up"] if entry["entry_prob_up"] is not None else float("nan")
    entry_confidence = entry["entry_confidence"] if entry["entry_confidence"] is not None else float("nan")
    winner = entry["winner"]
    will_lose = 1 if winner != entry_side else 0

    # BTC value at entry tick (use the resampled grid value)
    t_e_int = int(math.ceil(t_entry))
    if t_e_int < 0 or t_e_int >= len(df):
        return None
    btc_at_entry = float(df.iloc[t_e_int]["btc_price"])

    start_t = t_e_int + TICK_STEP
    end_t = MAX_EXIT_T
    if start_t > end_t:
        return None

    n = end_t - start_t + 1
    X = np.zeros((n, len(ALL_COLS)), dtype=np.float32)
    y = np.full(n, will_lose, dtype=np.int8)
    meta = []

    for i, t in enumerate(range(start_t, end_t + 1)):
        if t >= len(df):
            break
        # Base 44 features
        X[i, : len(FEATURES)] = v3_feats_all[t]

        # Current tick state
        row = df.iloc[t]
        up_bid = float(row["up_bid"]) if pd.notna(row.get("up_bid")) else 0.0
        up_ask = float(row["up_ask"]) if pd.notna(row.get("up_ask")) else 0.0
        down_bid = float(row["down_bid"]) if pd.notna(row.get("down_bid")) else 0.0
        down_ask = float(row["down_ask"]) if pd.notna(row.get("down_ask")) else 0.0
        btc_now = float(row["btc_price"]) if pd.notna(row.get("btc_price")) else btc_open

        if entry_side == "Up":
            our_bid = up_bid
            our_ask = up_ask
            opp_ask = down_ask
        else:
            our_bid = down_bid
            our_ask = down_ask
            opp_ask = up_ask

        v3p = float(v3_prob_all[t])
        our_prob = v3p if entry_side == "Up" else (1.0 - v3p)

        extra_offset = len(FEATURES)
        X[i, extra_offset + 0] = float(t_entry)
        X[i, extra_offset + 1] = float(entry_price)
        X[i, extra_offset + 2] = float(entry_side_is_up)
        X[i, extra_offset + 3] = 0.0 if math.isnan(entry_prob_up) else float(entry_prob_up)
        X[i, extra_offset + 4] = 0.0 if math.isnan(entry_confidence) else float(entry_confidence)
        X[i, extra_offset + 5] = float(t - t_entry)
        X[i, extra_offset + 6] = float(300.0 - t)
        X[i, extra_offset + 7] = v3p
        X[i, extra_offset + 8] = v3p - (0.0 if math.isnan(entry_prob_up) else entry_prob_up)
        X[i, extra_offset + 9] = our_prob
        X[i, extra_offset + 10] = our_prob - (0.0 if math.isnan(entry_confidence) else entry_confidence)
        X[i, extra_offset + 11] = our_ask
        X[i, extra_offset + 12] = our_bid
        X[i, extra_offset + 13] = opp_ask
        X[i, extra_offset + 14] = our_ask - entry_price
        X[i, extra_offset + 15] = our_bid - entry_price
        X[i, extra_offset + 16] = btc_now - btc_at_entry

        # Meta for simulation
        pnl_hold = SHARES * (1.0 - entry_price) if entry_side == winner \
                   else (-SHARES * entry_price)
        if our_bid > 0 and our_bid < 1.0:
            pnl_exit = SHARES * (our_bid - entry_price)
        else:
            pnl_exit = None  # cannot exit (no bid)

        meta.append({
            "t": t, "our_bid": our_bid, "our_ask": our_ask,
            "pnl_exit": pnl_exit, "pnl_hold": pnl_hold,
        })

    X = X[: len(meta)]
    y = y[: len(meta)]
    return X, y, meta


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def chronological_split(entries, train_frac=0.60, val_frac=0.20):
    n = len(entries)
    n_tr = int(round(n * train_frac))
    n_vl = int(round(n * val_frac))
    return (
        entries[:n_tr],
        entries[n_tr:n_tr + n_vl],
        entries[n_tr + n_vl:],
    )


def build_matrices(entries, v3, label):
    """Process all entries in a set, return stacked X/y and per-entry meta.

    per_entry is a list aligned with entries:
      {entry, X_indices (start, end), meta (list of per-tick dicts),
       label (1 if lose), pnl_hold}
    """
    Xs, ys = [], []
    per_entry = []
    total_ticks = 0
    skipped = 0
    for e in entries:
        loaded = load_window_ticks(e["slug"])
        if loaded is None:
            skipped += 1
            continue
        df, btc_open = loaded
        feats = compute_features_batch(df, btc_open).to_numpy(dtype=np.float32)
        v3_probs = v3.predict_calibrated(feats)
        built = build_entry_rows(e, df, btc_open, feats, v3_probs)
        if built is None:
            skipped += 1
            continue
        X, y, meta = built
        start = total_ticks
        end = start + len(X)
        Xs.append(X)
        ys.append(y)
        per_entry.append({
            "entry": e,
            "start": start,
            "end": end,
            "meta": meta,
            "label": 1 if e["winner"] != e["entry_side"] else 0,
            "pnl_hold": (SHARES * (1.0 - e["entry_price"])) if e["correct"] else (-SHARES * e["entry_price"]),
        })
        total_ticks += len(X)
    if not Xs:
        return None
    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    print(f"  [{label}] {len(per_entry)} entries ({skipped} skipped), {X_all.shape[0]:,} ticks")
    return X_all, y_all, per_entry


def train_classifier(X_tr, y_tr, X_vl, y_vl):
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    # NO scale_pos_weight — we want well-calibrated probabilities for
    # precision-focused threshold tuning. Weighting distorts calibration.
    print(f"  train positives (losers): {n_pos:,}  negatives (winners): {n_neg:,}")
    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.02,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",  # area-under-PR, ideal for precision-focused imbalanced task
        tree_method="hist",
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    print(f"  fit: {time.time()-t0:.1f}s  best_iter={getattr(model, 'best_iteration', 0)}")
    # Print top features
    imp = sorted(zip(ALL_COLS, model.feature_importances_.tolist()),
                 key=lambda kv: kv[1], reverse=True)
    print(f"  Top 15 features:")
    for k, v in imp[:15]:
        print(f"    {k:<28} {v:.4f}")
    return model


def calibrate(probs_vl, y_vl):
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probs_vl, y_vl)
    return cal


def find_thresholds_for_precision(y_true, y_prob, targets):
    """Return {target: (threshold, precision, recall)} using a high→low scan."""
    out = {}
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thresholds of length n-1 matched to
    # precisions[1:], recalls[1:]. Walk from high-threshold (end) down.
    for tgt in targets:
        best = None
        # Iterate from highest threshold to lowest
        for i in range(len(thresholds) - 1, -1, -1):
            thr = thresholds[i]
            p = precisions[i + 1]
            r = recalls[i + 1]
            if p >= tgt and r > 0:
                best = (float(thr), float(p), float(r))
                break
        out[tgt] = best  # may be None if unreachable
    return out


def simulate_exits(per_entry, probs, threshold):
    """Walk each entry's ticks; exit at first one exceeding threshold.

    Returns per-entry outcome dicts + aggregate numbers.
    """
    outcomes = []
    for rec in per_entry:
        entry = rec["entry"]
        label = rec["label"]           # 1 = loser
        pnl_hold = rec["pnl_hold"]
        start, end = rec["start"], rec["end"]
        prob_slice = probs[start:end]
        meta = rec["meta"]
        exited = False
        exit_t = exit_pnl = None
        for j in range(len(meta)):
            if prob_slice[j] >= threshold:
                m = meta[j]
                if m["pnl_exit"] is None:
                    continue  # can't exit here (no bid)
                exited = True
                exit_t = m["t"]
                exit_pnl = m["pnl_exit"]
                break
        if exited:
            final_pnl = exit_pnl
        else:
            final_pnl = pnl_hold

        savings = final_pnl - pnl_hold  # positive if exit saved us
        true_positive = exited and label == 1    # loser correctly exited
        false_positive = exited and label == 0   # winner incorrectly exited

        outcomes.append({
            "slug": entry["slug"],
            "label_loser": label,
            "exited": exited,
            "exit_t": exit_t,
            "exit_pnl": exit_pnl,
            "pnl_hold": pnl_hold,
            "final_pnl": final_pnl,
            "savings": savings,
            "tp": 1 if true_positive else 0,
            "fp": 1 if false_positive else 0,
        })
    return outcomes


def summarise_outcomes(outcomes, label):
    n = len(outcomes)
    n_exits = sum(o["exited"] for o in outcomes)
    n_tp = sum(o["tp"] for o in outcomes)
    n_fp = sum(o["fp"] for o in outcomes)
    n_losers = sum(o["label_loser"] for o in outcomes)
    n_winners = n - n_losers
    precision = n_tp / n_exits if n_exits else None
    recall = n_tp / n_losers if n_losers else None

    total_hold = sum(o["pnl_hold"] for o in outcomes)
    total_final = sum(o["final_pnl"] for o in outcomes)
    net_savings = total_final - total_hold

    tp_savings = sum(o["savings"] for o in outcomes if o["tp"])
    fp_cost = sum(o["savings"] for o in outcomes if o["fp"])  # negative

    tp_exits = [o for o in outcomes if o["tp"]]
    avg_exit_t = (sum(o["exit_t"] for o in tp_exits) / len(tp_exits)) if tp_exits else None
    avg_tp_savings = (tp_savings / len(tp_exits)) if tp_exits else None

    return {
        "label": label,
        "n_entries": n, "n_exits": n_exits,
        "n_losers": n_losers, "n_winners": n_winners,
        "n_tp": n_tp, "n_fp": n_fp,
        "precision": precision, "recall": recall,
        "total_hold_pnl": total_hold,
        "total_final_pnl": total_final,
        "net_savings": net_savings,
        "tp_savings": tp_savings,
        "fp_cost": fp_cost,
        "avg_exit_t_on_tp": avg_exit_t,
        "avg_savings_per_tp": avg_tp_savings,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(split_epochs, class_balance, results_by_precision, baseline):
    lines = ["# V3 Exit Classifier — Report\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    lines.append("## Split\n")
    for k, v in split_epochs.items():
        lines.append(f"- {k}: {v['n']} entries, epochs {v['epoch_range']}")
    lines.append("")
    lines.append("## Class balance\n")
    for k, v in class_balance.items():
        lines.append(f"- {k}: losers={v['losers']}/{v['total']} ({100*v['losers']/max(v['total'],1):.1f}%)")
    lines.append("")
    lines.append("## Baseline — V3 hold-to-settlement on TEST\n")
    lines.append(f"- Entries: {baseline['n_entries']}")
    lines.append(f"- Losers: {baseline['n_losers']}  Winners: {baseline['n_winners']}")
    lines.append(f"- Total hold PnL: **${baseline['total_hold_pnl']:+.2f}**")
    lines.append(f"- Per entry: ${baseline['total_hold_pnl']/max(baseline['n_entries'],1):+.3f}")
    lines.append("")
    lines.append("## Exit-classifier results on TEST (walk-forward)\n")
    lines.append("| Target precision | Threshold (val) | Test precision | Test recall | # exits | TP (losers caught) | FP (winners exited) | Avg exit t (TP) | Avg savings/TP | Total savings | FP cost | **Net PnL vs hold** |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tgt, entry in results_by_precision.items():
        if entry is None:
            lines.append(f"| {tgt:.2f} | *not reachable* | — | — | — | — | — | — | — | — | — | — |")
            continue
        r = entry["summary"]
        net_new = baseline["total_hold_pnl"] + r["net_savings"]
        lines.append(
            f"| {tgt:.2f} | {entry['threshold']:.4f} | "
            f"{(r['precision']*100 if r['precision'] is not None else 0):.1f}% | "
            f"{(r['recall']*100 if r['recall'] is not None else 0):.1f}% | "
            f"{r['n_exits']} | "
            f"{r['n_tp']} | {r['n_fp']} | "
            f"{(r['avg_exit_t_on_tp'] or 0):.1f} | "
            f"${(r['avg_savings_per_tp'] or 0):+.2f} | "
            f"${r['tp_savings']:+.2f} | ${r['fp_cost']:+.2f} | "
            f"**${r['net_savings']:+.2f}**  ({'+' if net_new >= baseline['total_hold_pnl'] else '−'}{abs(net_new-baseline['total_hold_pnl'])/max(abs(baseline['total_hold_pnl']),1)*100:.0f}% vs baseline) |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("V3 Exit Classifier — training + threshold tuning + PnL simulation")
    print("=" * 72)

    # 1. Load V3 entries
    entries = load_v3_entries()
    print(f"\nLoaded {len(entries)} V3 entries (entry_made=true, valid winner)")
    if len(entries) < 200:
        print("ERROR: too few entries for training.")
        return

    # 2. Chronological split
    train_e, val_e, test_e = chronological_split(entries, 0.60, 0.20)
    print(f"Split (chronological):  train={len(train_e)}  val={len(val_e)}  test={len(test_e)}")
    split_epochs = {
        "train": {"n": len(train_e), "epoch_range": (train_e[0]["open_epoch"], train_e[-1]["open_epoch"])},
        "val":   {"n": len(val_e),   "epoch_range": (val_e[0]["open_epoch"], val_e[-1]["open_epoch"])},
        "test":  {"n": len(test_e),  "epoch_range": (test_e[0]["open_epoch"], test_e[-1]["open_epoch"])},
    }

    # 3. V3 model for re-prediction feature
    print("\nLoading V3 model for re-prediction...")
    v3 = V3Reprobe()

    # 4. Build feature matrices
    print("\nBuilding feature matrices...")
    tr = build_matrices(train_e, v3, "train")
    vl = build_matrices(val_e, v3, "val")
    te = build_matrices(test_e, v3, "test")
    if tr is None or vl is None or te is None:
        print("ERROR: feature matrix construction failed.")
        return
    X_tr, y_tr, per_tr = tr
    X_vl, y_vl, per_vl = vl
    X_te, y_te, per_te = te

    class_balance = {
        "train": {"losers": int((y_tr == 1).sum()), "total": len(y_tr)},
        "val":   {"losers": int((y_vl == 1).sum()), "total": len(y_vl)},
        "test":  {"losers": int((y_te == 1).sum()), "total": len(y_te)},
    }
    print(f"\nClass balance (per-tick labels):")
    for k, v in class_balance.items():
        print(f"  {k}: {v['losers']:>6}/{v['total']:>7}  "
              f"({100*v['losers']/max(v['total'],1):.2f}% losers)")

    # 5. Train
    print("\nTraining XGBoost...")
    model = train_classifier(X_tr, y_tr, X_vl, y_vl)

    # 6. Calibrate on val
    probs_vl_raw = model.predict_proba(X_vl)[:, 1]
    cal = calibrate(probs_vl_raw, y_vl)
    probs_vl = cal.predict(probs_vl_raw)

    # 7. Threshold tuning — entry-level precision on val.
    # We scan a fine grid. For each threshold, simulate exits and compute
    # entry-level precision/recall. Also dump a diagnostic sweep.
    print("\nThreshold diagnostic sweep on VAL (entry-level):")
    thr_grid = sorted(set(
        [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
         0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
    ), reverse=True)
    print(f"  {'thr':<6}{'#exits':<8}{'#TP':<6}{'#FP':<6}"
          f"{'precision':<12}{'recall':<10}")
    for thr in thr_grid:
        outs = simulate_exits(per_vl, probs_vl, thr)
        s = summarise_outcomes(outs, "val")
        if s["n_exits"] == 0:
            continue
        p = s["precision"] or 0
        r = s["recall"] or 0
        print(f"  {thr:<6.3f}{s['n_exits']:<8}{s['n_tp']:<6}"
              f"{s['n_fp']:<6}{p*100:>6.1f}%     {r*100:>6.1f}%")

    # Raw prob distribution stats
    print(f"\n  Val prob distribution (losers vs winners):")
    pos_mask = y_vl == 1
    print(f"    losers  (y=1): min={probs_vl[pos_mask].min():.4f}  "
          f"med={np.median(probs_vl[pos_mask]):.4f}  "
          f"max={probs_vl[pos_mask].max():.4f}  n={pos_mask.sum()}")
    print(f"    winners (y=0): min={probs_vl[~pos_mask].min():.4f}  "
          f"med={np.median(probs_vl[~pos_mask]):.4f}  "
          f"max={probs_vl[~pos_mask].max():.4f}  n={(~pos_mask).sum()}")

    print("\nThreshold selection for precision targets on VAL...")
    fine_grid = sorted(set(
        list(np.linspace(0.20, 0.995, 160))
    ), reverse=True)
    chosen = {}
    for tgt in PRECISION_TARGETS:
        best = None
        for thr in fine_grid:
            outs = simulate_exits(per_vl, probs_vl, thr)
            s = summarise_outcomes(outs, "val")
            if s["n_exits"] < 1:
                continue
            if s["precision"] is None or s["precision"] < tgt:
                continue
            if best is None or s["n_tp"] > best[1]["n_tp"]:
                best = (thr, s)
        if best is None:
            chosen[tgt] = None
            print(f"  precision ≥ {tgt:.2f}: UNREACHABLE on val")
        else:
            thr, s = best
            chosen[tgt] = {"threshold": thr, "val_summary": s}
            print(f"  precision ≥ {tgt:.2f}:  θ={thr:.4f}  "
                  f"val_precision={s['precision']*100:.1f}%  val_recall={s['recall']*100:.1f}%  "
                  f"val_TP={s['n_tp']}  val_FP={s['n_fp']}")

    # 8. Apply chosen thresholds on TEST
    print("\nApplying thresholds on TEST (final evaluation)...")
    probs_te_raw = model.predict_proba(X_te)[:, 1]
    probs_te = cal.predict(probs_te_raw)

    # Baseline (hold-to-settlement)
    base_outs = [{
        "slug": rec["entry"]["slug"], "label_loser": rec["label"],
        "exited": False, "exit_t": None, "exit_pnl": None,
        "pnl_hold": rec["pnl_hold"], "final_pnl": rec["pnl_hold"],
        "savings": 0.0, "tp": 0, "fp": 0,
    } for rec in per_te]
    baseline = summarise_outcomes(base_outs, "test_baseline")

    results_by_precision = {}
    for tgt in PRECISION_TARGETS:
        picked = chosen.get(tgt)
        if picked is None:
            results_by_precision[tgt] = None
            continue
        thr = picked["threshold"]
        outs = simulate_exits(per_te, probs_te, thr)
        s = summarise_outcomes(outs, f"test@{tgt}")
        results_by_precision[tgt] = {"threshold": thr, "summary": s}
        print(f"  precision ≥ {tgt:.2f}:  θ={thr:.4f}  "
              f"test_precision={(s['precision']*100 if s['precision'] else 0):.1f}%  "
              f"test_recall={(s['recall']*100 if s['recall'] else 0):.1f}%  "
              f"test_TP={s['n_tp']}  test_FP={s['n_fp']}  "
              f"net_savings=${s['net_savings']:+.2f}")

    # 9. Save artifacts
    joblib.dump(model, MODEL_DIR / "v3_exit_xgb.joblib")
    joblib.dump(cal, MODEL_DIR / "v3_exit_calibrator.joblib")
    (MODEL_DIR / "v3_exit_thresholds.json").write_text(json.dumps({
        str(tgt): {
            "threshold": picked["threshold"],
            "val_precision": picked["val_summary"]["precision"],
            "val_recall": picked["val_summary"]["recall"],
        } if picked else None
        for tgt, picked in chosen.items()
    }, indent=2))
    report = format_report(split_epochs, class_balance, results_by_precision, baseline)
    (RESULTS_DIR / "v3_exit_report.md").write_text(report)

    print(f"\nSaved:")
    print(f"  {MODEL_DIR / 'v3_exit_xgb.joblib'}")
    print(f"  {MODEL_DIR / 'v3_exit_calibrator.joblib'}")
    print(f"  {MODEL_DIR / 'v3_exit_thresholds.json'}")
    print(f"  {RESULTS_DIR / 'v3_exit_report.md'}")
    print("\n--- Report ---")
    print(report)


if __name__ == "__main__":
    main()
