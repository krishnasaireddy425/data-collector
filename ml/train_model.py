"""
Train ML models on 90 CSV datasets and save for production use.

Outputs:
  - analysis/models/rf_model_t{time}.joblib  (one per timepoint)
  - analysis/models/feature_names_t{time}.json
  - analysis/models/training_stats.json

Usage: python3 analysis/train_model.py
"""

import glob
import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "price_collector", "data")
CSV_PATTERN = os.path.join(DATA_DIR, "btc-updown-5m-*.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_all_windows():
    files = sorted(glob.glob(CSV_PATTERN))
    print(f"Found {len(files)} CSV files")

    windows = []
    for f in files:
        try:
            df = pd.read_csv(f, comment="#")
            if len(df) < 10:
                continue
            for col in ["up_bid", "up_ask", "down_bid", "down_ask", "btc_price", "elapsed_sec"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["elapsed_sec", "btc_price", "up_ask", "down_ask"])

            btc_open = df.iloc[0]["btc_price"]
            btc_close = df.iloc[-1]["btc_price"]

            if btc_close > btc_open:
                winner = "up"
            elif btc_close < btc_open:
                winner = "down"
            else:
                winner = "flat"

            epoch = int(os.path.basename(f).replace("btc-updown-5m-", "").replace(".csv", ""))

            # Resample to 1s grid
            df_r = df.set_index("elapsed_sec").sort_index()
            new_idx = np.arange(0, 301, 1.0)
            df_1s = df_r.reindex(df_r.index.union(new_idx)).sort_index()
            df_1s = df_1s.interpolate(method="index", limit_direction="both")
            df_1s = df_1s.loc[new_idx].reset_index()
            df_1s.rename(columns={"index": "elapsed_sec"}, inplace=True)

            windows.append({
                "epoch": epoch,
                "winner": winner,
                "df": df_1s,
                "btc_open": btc_open,
                "btc_close": btc_close,
            })
        except Exception as e:
            print(f"  Error loading {f}: {e}")
            continue

    return [w for w in windows if w["winner"] != "flat"]


def engineer_features(df, t, btc_open):
    """Engineer features from market data up to time t."""
    mask = df["elapsed_sec"] <= t
    sub = df[mask].copy()
    if len(sub) < 5:
        return None

    row = sub.iloc[-1]
    feat = {}

    # Current prices
    feat["up_ask"] = row["up_ask"]
    feat["up_bid"] = row["up_bid"]
    feat["down_ask"] = row["down_ask"]
    feat["down_bid"] = row["down_bid"]
    feat["up_mid"] = (row["up_ask"] + row["up_bid"]) / 2
    feat["down_mid"] = (row["down_ask"] + row["down_bid"]) / 2
    feat["up_spread"] = row["up_ask"] - row["up_bid"]
    feat["down_spread"] = row["down_ask"] - row["down_bid"]
    feat["book_imbalance"] = feat["up_ask"] - feat["down_ask"]

    # BTC features
    feat["btc_chg"] = row["btc_price"] - btc_open
    feat["btc_chg_pct"] = feat["btc_chg"] / btc_open if btc_open > 0 else 0

    # Market leader
    feat["market_leader_up"] = 1.0 if row["up_ask"] > row["down_ask"] else 0.0
    feat["btc_dir_up"] = 1.0 if row["btc_price"] > btc_open else 0.0
    feat["market_agrees_btc"] = 1.0 if feat["market_leader_up"] == feat["btc_dir_up"] else 0.0

    # Momentum features at various lookbacks
    for lb in [10, 20, 30, 60]:
        t_start = t - lb
        if t_start < 0:
            t_start = 0
        past_mask = df["elapsed_sec"] == t_start
        past_rows = df[past_mask]
        if len(past_rows) > 0:
            past = past_rows.iloc[0]
            feat[f"up_ask_mom_{lb}s"] = row["up_ask"] - past["up_ask"]
            feat[f"down_ask_mom_{lb}s"] = row["down_ask"] - past["down_ask"]
            feat[f"btc_mom_{lb}s"] = row["btc_price"] - past["btc_price"]
            feat[f"btc_mom_pct_{lb}s"] = feat[f"btc_mom_{lb}s"] / btc_open if btc_open > 0 else 0
        else:
            feat[f"up_ask_mom_{lb}s"] = 0
            feat[f"down_ask_mom_{lb}s"] = 0
            feat[f"btc_mom_{lb}s"] = 0
            feat[f"btc_mom_pct_{lb}s"] = 0

    # Volatility features
    for lb in [10, 20, 30]:
        window = sub[sub["elapsed_sec"] >= t - lb]
        if len(window) > 2:
            feat[f"up_ask_vol_{lb}s"] = window["up_ask"].std()
            feat[f"btc_vol_{lb}s"] = window["btc_price"].std()
        else:
            feat[f"up_ask_vol_{lb}s"] = 0
            feat[f"btc_vol_{lb}s"] = 0

    # Range features
    for lb in [30, 60]:
        window = sub[sub["elapsed_sec"] >= t - lb]
        if len(window) > 2:
            feat[f"up_ask_max_{lb}s"] = window["up_ask"].max()
            feat[f"up_ask_min_{lb}s"] = window["up_ask"].min()
            feat[f"up_ask_range_{lb}s"] = feat[f"up_ask_max_{lb}s"] - feat[f"up_ask_min_{lb}s"]
        else:
            feat[f"up_ask_max_{lb}s"] = row["up_ask"]
            feat[f"up_ask_min_{lb}s"] = row["up_ask"]
            feat[f"up_ask_range_{lb}s"] = 0

    # Integral features (area under price curve = sustained momentum)
    for lb in [30, 60]:
        window = sub[sub["elapsed_sec"] >= t - lb]
        if len(window) > 2:
            feat[f"up_ask_integral_{lb}s"] = (window["up_ask"] - 0.5).sum() / len(window)
        else:
            feat[f"up_ask_integral_{lb}s"] = 0

    # BTC-market correlation
    for lb in [20, 30]:
        window = sub[sub["elapsed_sec"] >= t - lb]
        if len(window) > 5:
            btc_returns = window["btc_price"].diff().dropna()
            ua_returns = window["up_ask"].diff().dropna()
            if len(btc_returns) > 3 and btc_returns.std() > 0 and ua_returns.std() > 0:
                feat[f"btc_ua_corr_{lb}s"] = btc_returns.corr(ua_returns)
            else:
                feat[f"btc_ua_corr_{lb}s"] = 0
        else:
            feat[f"btc_ua_corr_{lb}s"] = 0

    # BTC acceleration
    for lb in [30]:
        t_mid = t - lb // 2
        t_start = t - lb
        curr_mask = df["elapsed_sec"] == t
        mid_mask = df["elapsed_sec"] == t_mid
        start_mask = df["elapsed_sec"] == t_start
        cr = df[curr_mask]
        mr = df[mid_mask]
        sr = df[start_mask]
        if len(cr) > 0 and len(mr) > 0 and len(sr) > 0:
            mom1 = cr.iloc[0]["btc_price"] - mr.iloc[0]["btc_price"]
            mom2 = mr.iloc[0]["btc_price"] - sr.iloc[0]["btc_price"]
            feat[f"btc_accel_{lb}s"] = mom1 - mom2
        else:
            feat[f"btc_accel_{lb}s"] = 0

    # BTC position in range
    for lb in [60]:
        window = sub[sub["elapsed_sec"] >= t - lb]
        if len(window) > 2:
            btc_max = window["btc_price"].max()
            btc_min = window["btc_price"].min()
            if btc_max > btc_min:
                feat[f"btc_position_in_range_{lb}s"] = (row["btc_price"] - btc_min) / (btc_max - btc_min)
            else:
                feat[f"btc_position_in_range_{lb}s"] = 0.5
        else:
            feat[f"btc_position_in_range_{lb}s"] = 0.5

    # Leader flip count
    for lb in [60, 90]:
        window = sub[sub["elapsed_sec"] >= t - lb]
        if len(window) > 2:
            leaders = (window["up_ask"] > window["down_ask"]).astype(int)
            feat[f"leader_flips_{lb}s"] = (leaders.diff().abs().sum())
        else:
            feat[f"leader_flips_{lb}s"] = 0

    # Replace NaN/inf
    for k, v in feat.items():
        if not np.isfinite(v):
            feat[k] = 0.0

    return feat


def build_dataset(windows, t):
    """Build feature matrix for all windows at time t."""
    X_rows = []
    y = []
    for w in windows:
        feat = engineer_features(w["df"], t, w["btc_open"])
        if feat is None:
            continue
        X_rows.append(feat)
        y.append(1 if w["winner"] == "up" else 0)

    if not X_rows:
        return None, None, None

    X = pd.DataFrame(X_rows)
    feature_names = list(X.columns)
    return X.values, np.array(y), feature_names


def train_and_evaluate(windows, timepoints):
    """Train RF models at each timepoint, evaluate with LOO CV."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    stats = {}

    for t in timepoints:
        print(f"\n--- Training at t={t}s ---")
        X, y, feature_names = build_dataset(windows, t)
        if X is None:
            print(f"  Skipped (no data)")
            continue

        n = len(y)
        print(f"  Samples: {n}, Features: {X.shape[1]}")

        # LOO CV to estimate accuracy
        loo = LeaveOneOut()
        preds = np.zeros(n)
        probas = np.zeros(n)

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds[test_idx] = model.predict(X_test)
            probas[test_idx] = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y, preds)
        print(f"  LOO accuracy: {acc:.1%}")

        # High confidence stats
        for thresh in [0.60, 0.65, 0.70, 0.75]:
            confident = np.abs(probas - 0.5) >= (thresh - 0.5)
            if confident.sum() > 0:
                cacc = accuracy_score(y[confident], preds[confident])
                print(f"    conf>={thresh:.0%}: {cacc:.1%} on {confident.sum()}/{n}")

        # Train final model on ALL data
        final_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
        )
        final_model.fit(X, y)

        # Save model and feature names
        model_path = os.path.join(MODEL_DIR, f"rf_model_t{t}.joblib")
        feat_path = os.path.join(MODEL_DIR, f"feature_names_t{t}.json")
        joblib.dump(final_model, model_path)
        with open(feat_path, "w") as f:
            json.dump(feature_names, f)

        print(f"  Saved: {model_path}")

        # Feature importance
        importances = final_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:8]
        print("  Top features:")
        for i in top_idx:
            print(f"    {feature_names[i]:>30s}: {importances[i]:.4f}")

        stats[str(t)] = {
            "accuracy": round(acc, 4),
            "n_samples": n,
            "n_features": X.shape[1],
        }

    # Save training stats
    stats_path = os.path.join(MODEL_DIR, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nTraining stats saved to {stats_path}")

    return stats


def main():
    print("=" * 60)
    print("TRAINING ML MODELS FOR BTC 5-MIN PREDICTION")
    print("=" * 60)

    windows = load_all_windows()
    print(f"\nLoaded {len(windows)} windows (UP: {sum(1 for w in windows if w['winner']=='up')}, "
          f"DOWN: {sum(1 for w in windows if w['winner']=='down')})")

    # Train at key timepoints that showed best accuracy
    timepoints = [60, 90, 120, 150, 180, 210, 240]
    stats = train_and_evaluate(windows, timepoints)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    for t_str, s in sorted(stats.items(), key=lambda x: int(x[0])):
        print(f"  t={t_str:>3s}s: LOO acc={s['accuracy']:.1%} ({s['n_samples']} samples, {s['n_features']} features)")

    best_t = max(stats.items(), key=lambda x: x[1]["accuracy"])
    print(f"\n  BEST: t={best_t[0]}s with {best_t[1]['accuracy']:.1%} accuracy")
    print("=" * 60)


if __name__ == "__main__":
    main()
