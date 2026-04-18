"""
Train a SINGLE XGBoost model on second-level continuous samples.

Design:
  - Chronological split by CSV epoch. First 800 = train, next 170 = test.
  - For each training window, resample to 1-sec grid, ffill, build ~20 features
    at every elapsed_sec ∈ [0, 300]. All seconds share the window's winner.
  - Stack into one big (samples × features) matrix. Fit XGBoost.
  - Early stopping uses the tail of the training set (last 80 windows) — the
    170 test windows are NEVER touched at train time.

Outputs:
  ml/models/xgb_sec.joblib
  ml/models/feature_names_sec.json
  ml/models/split_manifest.json     (train files + test files for reproducibility)
  ml/models/training_stats_sec.json

Usage:
  python3 -m ml.train_sec
"""

import glob
import json
import os
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from .features_sec import FEATURES, compute_features_batch, resample_window


def _tune_xgb(X_tr, y_tr, w_tr, X_val, y_val, w_val, n_trials=40):
    """Bayesian hyperparameter search. Maximises negative logloss on val."""
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        }
        mdl = xgb.XGBClassifier(
            n_estimators=400,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            early_stopping_rounds=25,
            n_jobs=-1,
            **params,
        )
        mdl.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )
        probs = mdl.predict_proba(X_val)[:, 1]
        return log_loss(y_val, probs, sample_weight=w_val)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "price_collector" / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

N_TRAIN = 1000
N_TEST = 267
VAL_TAIL = 100    # last 100 train windows used as early-stopping validation


def read_winner(csv_path):
    """Return 'Up' / 'Down' from the trailing `# RESULT` line, else None."""
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


def load_and_featurise(csv_path):
    """Return (X[300×F], y[300], btc_open, winner) or None if unusable."""
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
    feats = compute_features_batch(df, btc_open)
    y = np.ones(len(feats), dtype=np.int8) if winner == "Up" else np.zeros(len(feats), dtype=np.int8)
    return feats.values.astype(np.float32), y, btc_open, winner


def stack(window_files):
    X_parts, y_parts, groups = [], [], []
    ok = 0
    for i, f in enumerate(window_files):
        out = load_and_featurise(f)
        if out is None:
            continue
        X, y, _, _ = out
        X_parts.append(X)
        y_parts.append(y)
        groups.append(np.full(len(y), ok, dtype=np.int32))
        ok += 1
    if not X_parts:
        return None, None, None, 0
    return (np.vstack(X_parts), np.concatenate(y_parts),
            np.concatenate(groups), ok)


def main():
    files = sorted(glob.glob(str(DATA_DIR / "btc-updown-5m-*.csv")),
                   key=epoch_of)
    print(f"Found {len(files)} CSVs (sorted by epoch)")

    # Filter to CSVs with valid winner
    valid = [f for f in files if read_winner(f) in ("Up", "Down")]
    skipped = len(files) - len(valid)
    if skipped:
        print(f"  (skipped {skipped} CSVs without a valid # RESULT winner)")

    n = len(valid)
    if n < N_TRAIN + N_TEST:
        # Proportional fallback if we somehow don't have 970
        train_cut = int(n * (N_TRAIN / (N_TRAIN + N_TEST)))
        train_files = valid[:train_cut]
        test_files = valid[train_cut:]
        print(f"  WARN: only {n} valid; splitting "
              f"{len(train_files)} train / {len(test_files)} test")
    else:
        train_files = valid[:N_TRAIN]
        test_files = valid[N_TRAIN:N_TRAIN + N_TEST]
        print(f"  train: {len(train_files)} windows "
              f"({epoch_of(train_files[0])}..{epoch_of(train_files[-1])})")
        print(f"  test:  {len(test_files)} windows "
              f"({epoch_of(test_files[0])}..{epoch_of(test_files[-1])})")

    # --- Build training matrix ---
    t0 = time.time()
    print("\nBuilding training matrix...")
    X, y, groups, n_ok = stack(train_files)
    if X is None:
        print("ERROR: no usable training windows")
        return
    print(f"  windows used: {n_ok}")
    print(f"  samples: {len(y):,}   features: {X.shape[1]}")
    print(f"  class balance: Up={y.mean():.1%}  Down={1 - y.mean():.1%}")
    print(f"  build time: {time.time() - t0:.1f}s")

    # --- Chronological train/val split inside the training set ---
    cutoff_window = max(0, n_ok - VAL_TAIL)
    train_mask = groups < cutoff_window
    val_mask = ~train_mask
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"\n  inner train: {len(y_tr):,}  inner val (last {VAL_TAIL} windows): {len(y_val):,}")

    # --- Sample weights: linear ramp from 0.5 at t=0 to 1.5 at t=300 ---
    # Late-window samples matter more (more signal, closer to resolution).
    # elapsed_sec is the first column in FEATURES.
    elapsed_col = FEATURES.index("elapsed_sec")
    w_tr = 0.5 + (X_tr[:, elapsed_col] / 300.0)
    w_val = 0.5 + (X_val[:, elapsed_col] / 300.0)
    print(f"  sample weight range: [{w_tr.min():.2f} .. {w_tr.max():.2f}] "
          f"(linear ramp by elapsed_sec)")

    # --- Hyperparameters (Optuna disabled — see notes) ---
    skip_optuna = os.environ.get("SKIP_OPTUNA", "1") == "1"
    if HAS_OPTUNA and not skip_optuna:
        print("\nBayesian hyperparameter search (40 trials)...")
        t_opt = time.time()
        best_params, best_logloss = _tune_xgb(
            X_tr, y_tr, w_tr, X_val, y_val, w_val, n_trials=40
        )
        print(f"  best val logloss: {best_logloss:.4f}")
        print(f"  best params: {best_params}")
        print(f"  tuning time: {time.time() - t_opt:.1f}s")
    else:
        print("\n  Using default hyperparams (SKIP_OPTUNA=1)")
        best_params = {
            "max_depth": 6, "learning_rate": 0.05,
            "min_child_weight": 5, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.0,
            "reg_lambda": 1.0, "gamma": 0.0,
        }

    # --- Fit final XGBoost with best hyperparams ---
    model = xgb.XGBClassifier(
        n_estimators=800,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=40,
        n_jobs=-1,
        **best_params,
    )
    print("\nFitting final XGBoost with best hyperparams...")
    t1 = time.time()
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=50,
    )
    print(f"  fit time: {time.time() - t1:.1f}s")
    best_iter = getattr(model, "best_iteration", model.n_estimators)
    print(f"  best_iteration: {best_iter}")

    # --- In-sample and val accuracy (sanity; real judgement is test replay) ---
    train_preds = (model.predict_proba(X_tr)[:, 1] >= 0.5).astype(int)
    val_preds = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
    train_acc = (train_preds == y_tr).mean()
    val_acc = (val_preds == y_val).mean()
    print(f"\n  inner-train acc: {train_acc:.1%}")
    print(f"  inner-val   acc: {val_acc:.1%}  (held-out tail of train)")

    # --- Feature importance ---
    imp = dict(zip(FEATURES, model.feature_importances_.tolist()))
    imp_sorted = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
    print("\n  Top features:")
    for k, v in imp_sorted[:10]:
        print(f"    {k:<22} {v:.4f}")

    # --- Fit isotonic calibrator on inner-val raw probabilities ---
    val_probs_raw = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_probs_raw, y_val)
    val_probs_cal = calibrator.predict(val_probs_raw)
    # Compare calibration quality: mean abs error between prob and outcome
    brier_raw = float(np.mean((val_probs_raw - y_val) ** 2))
    brier_cal = float(np.mean((val_probs_cal - y_val) ** 2))
    print(f"\n  Calibration (Brier score, lower=better):")
    print(f"    raw:        {brier_raw:.4f}")
    print(f"    calibrated: {brier_cal:.4f}  "
          f"(Δ = {brier_cal - brier_raw:+.4f})")

    # --- Save ---
    joblib.dump(model, MODEL_DIR / "xgb_sec.joblib")
    joblib.dump(calibrator, MODEL_DIR / "calibrator.joblib")
    (MODEL_DIR / "feature_names_sec.json").write_text(json.dumps(FEATURES))
    (MODEL_DIR / "split_manifest.json").write_text(json.dumps({
        "n_train_windows": len(train_files),
        "n_test_windows": len(test_files),
        "train_files": [os.path.basename(f) for f in train_files],
        "test_files": [os.path.basename(f) for f in test_files],
        "train_epoch_range": [epoch_of(train_files[0]), epoch_of(train_files[-1])],
        "test_epoch_range": [epoch_of(test_files[0]), epoch_of(test_files[-1])],
    }, indent=2))
    (MODEL_DIR / "training_stats_sec.json").write_text(json.dumps({
        "n_train_samples": int(len(y_tr)),
        "n_val_samples": int(len(y_val)),
        "class_balance_up_pct": float(y.mean()),
        "inner_train_acc": float(train_acc),
        "inner_val_acc": float(val_acc),
        "best_iteration": int(best_iter),
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "feature_importance": imp,
        "sample_weighting": "linear_elapsed_0.5_to_1.5",
        "best_hyperparams": best_params,
        "n_features": len(FEATURES),
    }, indent=2))
    print(f"\nSaved model to {MODEL_DIR / 'xgb_sec.joblib'}")
    print("Next: run `python3 -m ml.backtest_sec` to evaluate on the 170 test windows.")


if __name__ == "__main__":
    main()
