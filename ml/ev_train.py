"""
Train expected-value (EV) regression models.

Instead of "who wins", predict expected PnL per share of buying each
side at each tick. The model directly learns where the market is
mispricing the probability — positive prediction = +EV entry.

Labels at row t with window winner W:
  pnl_up_at_t   = (1 − up_ask_t)   if W == "Up"   else −up_ask_t
  pnl_down_at_t = (1 − down_ask_t) if W == "Down" else −down_ask_t

Features: same 44 as xgb_sec (fully leak-safe — only shift() and rolling()
which look backward). So a single batched feature pass per window is valid
and massively faster than re-computing per tick.

Walks 797 train CSVs (chronological split from split_manifest.json if it
exists, else proportional fallback). Trains two XGBoost regressors.

Outputs:
  ml/models/ev_up.joblib
  ml/models/ev_down.joblib
  ml/models/ev_feature_names.json
  ml/models/ev_training_stats.json

Usage:
  python3 -m ml.ev_train
"""

import glob
import json
import os
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from .features_sec import FEATURES, compute_features_batch, resample_window

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "price_collector" / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

N_TEST = 288          # a full day at 5-min cadence
VAL_TAIL = 80          # held-out tail of TRAIN for early-stopping/calibration


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


def epoch_of(path):
    m = re.search(r"-(\d{10,})\.csv$", path)
    return int(m.group(1)) if m else 0


def process_window(csv_path):
    """Return (X[300×F], y_up[300], y_down[300]) or None."""
    winner = read_winner(csv_path)
    if winner not in ("Up", "Down"):
        return None
    try:
        raw = pd.read_csv(csv_path, comment="#")
    except Exception:
        return None
    if len(raw) < 10:
        return None
    df = resample_window(raw, max_t=300)
    if df is None or df.empty:
        return None

    btc_open = float(df.iloc[0]["btc_price"])
    feats = compute_features_batch(df, btc_open).values.astype(np.float32)

    up_ask = df["up_ask"].values.astype(np.float32)
    down_ask = df["down_ask"].values.astype(np.float32)
    if winner == "Up":
        y_up = 1.0 - up_ask
        y_down = -down_ask
    else:
        y_up = -up_ask
        y_down = 1.0 - down_ask

    return feats, y_up.astype(np.float32), y_down.astype(np.float32)


def stack(files):
    X_parts, Yu_parts, Yd_parts, groups = [], [], [], []
    ok = 0
    for f in files:
        r = process_window(f)
        if r is None:
            continue
        X, yu, yd = r
        X_parts.append(X)
        Yu_parts.append(yu)
        Yd_parts.append(yd)
        groups.append(np.full(len(yu), ok, dtype=np.int32))
        ok += 1
    if not X_parts:
        return None, None, None, None, 0
    return (np.vstack(X_parts),
            np.concatenate(Yu_parts),
            np.concatenate(Yd_parts),
            np.concatenate(groups),
            ok)


def _fit_ev_model(X_tr, y_tr, w_tr, X_val, y_val, w_val, label):
    model = xgb.XGBRegressor(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=40,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )
    fit_s = time.time() - t0
    tr_pred = model.predict(X_tr)
    val_pred = model.predict(X_val)
    print(f"\n  [{label}] fit in {fit_s:.1f}s, best_iter={model.best_iteration}")
    print(f"    train MAE: {mean_absolute_error(y_tr, tr_pred):.5f}")
    print(f"    val   MAE: {mean_absolute_error(y_val, val_pred):.5f}")
    print(f"    val R²:    {r2_score(y_val, val_pred):.4f}")
    # Expected-PnL soundness: on val, what fraction of predicted positive
    # EV entries actually turn out positive on realized PnL?
    pos_mask = val_pred > 0
    if pos_mask.sum() > 0:
        hit_rate = (y_val[pos_mask] > 0).mean()
        realised = y_val[pos_mask].mean()
        print(f"    predicted +EV n={int(pos_mask.sum())}, "
              f"true hit rate={hit_rate:.1%}, "
              f"avg realised PnL/share=${realised:+.4f}")
    return model


def main():
    files = sorted(glob.glob(str(DATA_DIR / "btc-updown-5m-*.csv")),
                   key=epoch_of)
    valid = [f for f in files if read_winner(f) in ("Up", "Down")]
    n = len(valid)
    print(f"Found {len(files)} CSVs, {n} valid (Up/Down)")

    # Hold out the LAST N_TEST windows (chronologically). They represent
    # ~24 hours of the most recent market — closest thing to a forward test.
    test_files = valid[-N_TEST:]
    train_files = valid[:-N_TEST]
    from datetime import datetime, timezone
    def _fmt(f):
        return datetime.fromtimestamp(epoch_of(f), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"  split: {len(train_files)} train / {len(test_files)} test")
    print(f"  train span: {_fmt(train_files[0])} → {_fmt(train_files[-1])} UTC")
    print(f"  test  span: {_fmt(test_files[0])} → {_fmt(test_files[-1])} UTC")

    t0 = time.time()
    print("\nBuilding EV training matrix...")
    X, y_up, y_down, groups, n_ok = stack(train_files)
    if X is None:
        print("ERROR: no usable train windows")
        return
    print(f"  windows used: {n_ok}")
    print(f"  samples: {len(y_up):,}   features: {X.shape[1]}")
    print(f"  build time: {time.time() - t0:.1f}s")
    print(f"  y_up  stats: mean={y_up.mean():+.4f} std={y_up.std():.4f}")
    print(f"  y_down stats: mean={y_down.mean():+.4f} std={y_down.std():.4f}")

    cutoff = max(0, n_ok - VAL_TAIL)
    train_mask = groups < cutoff
    val_mask = ~train_mask
    X_tr = X[train_mask]; X_val = X[val_mask]
    yu_tr, yu_val = y_up[train_mask], y_up[val_mask]
    yd_tr, yd_val = y_down[train_mask], y_down[val_mask]
    print(f"  inner train: {len(yu_tr):,}  inner val (last {VAL_TAIL} windows): {len(yu_val):,}")

    elapsed_col = FEATURES.index("elapsed_sec")
    w_tr = 0.5 + (X_tr[:, elapsed_col] / 300.0)
    w_val = 0.5 + (X_val[:, elapsed_col] / 300.0)

    print("\nFitting EV_Up regressor...")
    m_up = _fit_ev_model(X_tr, yu_tr, w_tr, X_val, yu_val, w_val, "EV_Up")
    print("\nFitting EV_Down regressor...")
    m_down = _fit_ev_model(X_tr, yd_tr, w_tr, X_val, yd_val, w_val, "EV_Down")

    joblib.dump(m_up, MODEL_DIR / "ev_up.joblib")
    joblib.dump(m_down, MODEL_DIR / "ev_down.joblib")
    (MODEL_DIR / "ev_feature_names.json").write_text(json.dumps(FEATURES))

    # Feature importances
    imp_up = sorted(zip(FEATURES, m_up.feature_importances_.tolist()),
                    key=lambda kv: kv[1], reverse=True)
    imp_down = sorted(zip(FEATURES, m_down.feature_importances_.tolist()),
                      key=lambda kv: kv[1], reverse=True)
    print("\n  Top EV_Up features:")
    for k, v in imp_up[:8]:
        print(f"    {k:<28} {v:.4f}")
    print("\n  Top EV_Down features:")
    for k, v in imp_down[:8]:
        print(f"    {k:<28} {v:.4f}")

    (MODEL_DIR / "ev_training_stats.json").write_text(json.dumps({
        "n_train_samples": int(train_mask.sum()),
        "n_val_samples": int(val_mask.sum()),
        "n_features": X.shape[1],
        "up_best_iter": int(m_up.best_iteration),
        "down_best_iter": int(m_down.best_iteration),
        "up_train_mae": float(mean_absolute_error(yu_tr, m_up.predict(X_tr))),
        "up_val_mae": float(mean_absolute_error(yu_val, m_up.predict(X_val))),
        "down_train_mae": float(mean_absolute_error(yd_tr, m_down.predict(X_tr))),
        "down_val_mae": float(mean_absolute_error(yd_val, m_down.predict(X_val))),
        "up_feat_imp": dict(imp_up),
        "down_feat_imp": dict(imp_down),
        "test_files": [os.path.basename(f) for f in test_files],
    }, indent=2))
    print(f"\nSaved: {MODEL_DIR / 'ev_up.joblib'}, {MODEL_DIR / 'ev_down.joblib'}")
    print("Next: run `python3 -m ml.ev_backtest`")


if __name__ == "__main__":
    main()
