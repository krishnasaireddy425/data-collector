"""
PredictionEngine — extracted from btc/prod_ml_strategy.py.

Loads RandomForest models trained at multiple timepoints (t=60..240s),
buffers tick data, computes the 50-feature vector, and produces an
ensemble prediction combining ML + rule signals.

NO real-order code. NO trading logic. Pure prediction.
"""

import json
import os
from collections import deque

import numpy as np

try:
    import joblib
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# ----------------------------------------------------------------------
# Constants (match prod_ml_strategy.py exactly)
# ----------------------------------------------------------------------

PREDICTION_TIMES = [60, 90, 120, 150, 180, 210, 240]

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# ----------------------------------------------------------------------
# Reversal tracker (used by paper_strategy too)
# ----------------------------------------------------------------------

class ReversalTracker:
    """Tracks leader flips with hysteresis (same as prod_ml_strategy)."""

    def __init__(self, window_sec=90.0, max_reversals=3, min_gap=0.02):
        self.window_sec = window_sec
        self.max_reversals = max_reversals
        self.min_gap = min_gap
        self.reversals = deque()
        self.last_leader = None

    def update(self, timestamp, up_ask, down_ask):
        if up_ask is None or down_ask is None:
            return
        if self.last_leader is None:
            current_leader = "up" if up_ask > down_ask else "down"
        elif self.last_leader == "up":
            current_leader = (
                "down" if down_ask > up_ask + self.min_gap else "up"
            )
        else:
            current_leader = (
                "up" if up_ask > down_ask + self.min_gap else "down"
            )

        if (self.last_leader is not None
                and current_leader != self.last_leader):
            self.reversals.append(timestamp)
        self.last_leader = current_leader

        cutoff = timestamp - self.window_sec
        while self.reversals and self.reversals[0] < cutoff:
            self.reversals.popleft()

    def is_stable(self):
        return len(self.reversals) <= self.max_reversals

    def count(self):
        return len(self.reversals)


# ----------------------------------------------------------------------
# Prediction engine
# ----------------------------------------------------------------------

class PredictionEngine:
    """Loads trained RF models and produces ensemble predictions."""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.feature_names = {}
        self.data_buffer = []  # (elapsed, up_bid, up_ask, down_bid, down_ask, btc_price)
        self.btc_open = None
        self._load_models()

    def _load_models(self):
        if not ML_AVAILABLE:
            print("  [ML] joblib/pandas not available — using rule-based only")
            return

        for t in PREDICTION_TIMES:
            model_path = os.path.join(self.model_dir, f"rf_model_t{t}.joblib")
            feat_path = os.path.join(self.model_dir, f"feature_names_t{t}.json")
            if os.path.exists(model_path) and os.path.exists(feat_path):
                self.models[t] = joblib.load(model_path)
                with open(feat_path) as f:
                    self.feature_names[t] = json.load(f)

        if self.models:
            print(f"  [ML] Loaded {len(self.models)} models: "
                  f"t={sorted(self.models.keys())}")
        else:
            print(f"  [ML] WARNING: no models loaded from {self.model_dir}")

    def reset(self):
        """Clear state for a new window."""
        self.data_buffer = []
        self.btc_open = None

    def add_tick(self, elapsed, up_bid, up_ask, down_bid, down_ask, btc_price):
        """Add a market tick to the buffer."""
        if self.btc_open is None and btc_price:
            self.btc_open = btc_price
        self.data_buffer.append(
            (elapsed, up_bid, up_ask, down_bid, down_ask, btc_price)
        )

    def _build_dataframe(self):
        if not ML_AVAILABLE or len(self.data_buffer) < 5:
            return None

        df = pd.DataFrame(
            self.data_buffer,
            columns=["elapsed_sec", "up_bid", "up_ask", "down_bid",
                     "down_ask", "btc_price"],
        )
        df = df.sort_values("elapsed_sec").drop_duplicates(
            subset=["elapsed_sec"], keep="last"
        )

        max_t = df["elapsed_sec"].max()
        new_idx = np.arange(0, int(max_t) + 1, 1.0)
        df_r = df.set_index("elapsed_sec")
        df_1s = df_r.reindex(df_r.index.union(new_idx)).sort_index()
        # Forward-fill only — matches train_model.py preprocessing.
        # Never uses future data to fill gaps at time t.
        df_1s = df_1s.ffill().bfill()
        df_1s = df_1s.loc[new_idx].reset_index()
        df_1s.rename(columns={"index": "elapsed_sec"}, inplace=True)
        return df_1s

    def _engineer_features(self, df, t):
        """Compute the 50-feature vector at time t (matches train_model.py)."""
        mask = df["elapsed_sec"] <= t
        sub = df[mask].copy()
        if len(sub) < 5:
            return None

        row = sub.iloc[-1]
        feat = {}
        btc_open = self.btc_open or row["btc_price"]

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
        feat["btc_chg_pct"] = (
            feat["btc_chg"] / btc_open if btc_open and btc_open > 0 else 0
        )

        # Market leader
        feat["market_leader_up"] = (
            1.0 if row["up_ask"] > row["down_ask"] else 0.0
        )
        feat["btc_dir_up"] = 1.0 if row["btc_price"] > btc_open else 0.0
        feat["market_agrees_btc"] = (
            1.0 if feat["market_leader_up"] == feat["btc_dir_up"] else 0.0
        )

        # Momentum
        for lb in [10, 20, 30, 60]:
            t_start = max(0, t - lb)
            past_rows = df[df["elapsed_sec"] == t_start]
            if len(past_rows) > 0:
                past = past_rows.iloc[0]
                feat[f"up_ask_mom_{lb}s"] = row["up_ask"] - past["up_ask"]
                feat[f"down_ask_mom_{lb}s"] = (
                    row["down_ask"] - past["down_ask"]
                )
                feat[f"btc_mom_{lb}s"] = (
                    row["btc_price"] - past["btc_price"]
                )
                feat[f"btc_mom_pct_{lb}s"] = (
                    feat[f"btc_mom_{lb}s"] / btc_open
                    if btc_open and btc_open > 0 else 0
                )
            else:
                for prefix in ["up_ask_mom", "down_ask_mom",
                               "btc_mom", "btc_mom_pct"]:
                    feat[f"{prefix}_{lb}s"] = 0

        # Volatility
        for lb in [10, 20, 30]:
            window = sub[sub["elapsed_sec"] >= t - lb]
            if len(window) > 2:
                feat[f"up_ask_vol_{lb}s"] = window["up_ask"].std()
                feat[f"btc_vol_{lb}s"] = window["btc_price"].std()
            else:
                feat[f"up_ask_vol_{lb}s"] = 0
                feat[f"btc_vol_{lb}s"] = 0

        # Range
        for lb in [30, 60]:
            window = sub[sub["elapsed_sec"] >= t - lb]
            if len(window) > 2:
                feat[f"up_ask_max_{lb}s"] = window["up_ask"].max()
                feat[f"up_ask_min_{lb}s"] = window["up_ask"].min()
                feat[f"up_ask_range_{lb}s"] = (
                    feat[f"up_ask_max_{lb}s"] - feat[f"up_ask_min_{lb}s"]
                )
            else:
                feat[f"up_ask_max_{lb}s"] = row["up_ask"]
                feat[f"up_ask_min_{lb}s"] = row["up_ask"]
                feat[f"up_ask_range_{lb}s"] = 0

        # Integral
        for lb in [30, 60]:
            window = sub[sub["elapsed_sec"] >= t - lb]
            if len(window) > 2:
                feat[f"up_ask_integral_{lb}s"] = (
                    (window["up_ask"] - 0.5).sum() / len(window)
                )
            else:
                feat[f"up_ask_integral_{lb}s"] = 0

        # Correlation
        for lb in [20, 30]:
            window = sub[sub["elapsed_sec"] >= t - lb]
            if len(window) > 5:
                btc_ret = window["btc_price"].diff().dropna()
                ua_ret = window["up_ask"].diff().dropna()
                if (len(btc_ret) > 3 and btc_ret.std() > 0
                        and ua_ret.std() > 0):
                    feat[f"btc_ua_corr_{lb}s"] = btc_ret.corr(ua_ret)
                else:
                    feat[f"btc_ua_corr_{lb}s"] = 0
            else:
                feat[f"btc_ua_corr_{lb}s"] = 0

        # Acceleration
        for lb in [30]:
            t_mid = t - lb // 2
            t_start = t - lb
            cr = df[df["elapsed_sec"] == t]
            mr = df[df["elapsed_sec"] == t_mid]
            sr = df[df["elapsed_sec"] == t_start]
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
                    feat[f"btc_position_in_range_{lb}s"] = (
                        (row["btc_price"] - btc_min) / (btc_max - btc_min)
                    )
                else:
                    feat[f"btc_position_in_range_{lb}s"] = 0.5
            else:
                feat[f"btc_position_in_range_{lb}s"] = 0.5

        # Leader flips
        for lb in [60, 90]:
            window = sub[sub["elapsed_sec"] >= t - lb]
            if len(window) > 2:
                leaders = (window["up_ask"] > window["down_ask"]).astype(int)
                feat[f"leader_flips_{lb}s"] = (
                    leaders.diff().abs().sum()
                )
            else:
                feat[f"leader_flips_{lb}s"] = 0

        # Replace NaN/inf
        for k, v in feat.items():
            if not np.isfinite(v):
                feat[k] = 0.0

        return feat

    def _compute_features_safe(self, elapsed):
        """Compute features at elapsed time without running the model.
        Returns feature dict or None. Used by V3 for regime detection."""
        try:
            return self._engineer_features(self._build_dataframe(), int(elapsed))
        except Exception:
            return None

    def _get_ml_prob_raw(self, elapsed):
        """Return raw ML probability of UP without ensemble weighting.
        Used by V4 which bypasses the ensemble entirely."""
        df = self._build_dataframe()
        if df is None:
            return None

        t = max([tp for tp in PREDICTION_TIMES if tp <= elapsed], default=None)
        if t is None or t not in self.models:
            return None

        feat = self._engineer_features(df, t)
        if feat is None:
            return None

        fn = self.feature_names[t]
        X = np.array([[feat.get(name, 0.0) for name in fn]])
        try:
            return float(self.models[t].predict_proba(X)[0][1])
        except Exception:
            return None

    def predict(self, elapsed):
        """
        Run ensemble prediction at the current elapsed time.

        Returns: (side, confidence, signals_dict)
          side       — 'up' / 'down' / None
          confidence — 0.0..1.0
          signals    — dict of all signal contributions for logging
        """
        df = self._build_dataframe()
        if df is None:
            return None, 0.0, {}

        # Pick the closest trained timepoint <= elapsed
        t = max([tp for tp in PREDICTION_TIMES if tp <= elapsed], default=None)
        if t is None:
            return None, 0.0, {}

        signals = {"model_t": t}
        votes_up = 0.0
        total_weight = 0.0

        # --- Signal 1: ML model ---
        ml_conf = 0.5
        if t in self.models:
            feat = self._engineer_features(df, t)
            if feat is not None:
                feat_names = self.feature_names[t]
                X = np.array([[feat.get(fn, 0.0) for fn in feat_names]])
                proba = self.models[t].predict_proba(X)[0]
                ml_conf = float(proba[1])  # P(UP)
                signals["ml_prob_up"] = round(ml_conf, 4)

                ml_weight = 3.0
                votes_up += ml_conf * ml_weight
                total_weight += ml_weight

        # --- Signal 2: Market leader ---
        row = df[df["elapsed_sec"] <= elapsed].iloc[-1]
        up_ask = float(row["up_ask"])
        down_ask = float(row["down_ask"])
        market_leader_up = up_ask > down_ask
        leader_strength = abs(up_ask - down_ask)
        signals["market_leader"] = "up" if market_leader_up else "down"
        signals["leader_strength"] = round(leader_strength, 4)

        market_weight = 2.0
        votes_up += (1.0 if market_leader_up else 0.0) * market_weight
        total_weight += market_weight

        # --- Signal 3: BTC direction ---
        btc_open = self.btc_open or float(df.iloc[0]["btc_price"])
        btc_now = float(row["btc_price"])
        btc_up = btc_now > btc_open
        btc_chg_pct = (
            (btc_now - btc_open) / btc_open if btc_open > 0 else 0
        )
        signals["btc_dir"] = "up" if btc_up else "down"
        signals["btc_chg_pct"] = round(btc_chg_pct * 100, 4)

        btc_weight = 1.5
        votes_up += (1.0 if btc_up else 0.0) * btc_weight
        total_weight += btc_weight

        # --- Signal 4: Agreement bonus ---
        if market_leader_up == btc_up:
            agree_weight = 1.0
            votes_up += (1.0 if btc_up else 0.0) * agree_weight
            total_weight += agree_weight
            signals["btc_market_agree"] = True
        else:
            signals["btc_market_agree"] = False

        # --- Signal 5: Ask threshold strength ---
        leader_ask = up_ask if market_leader_up else down_ask
        if leader_ask >= 0.80:
            ask_strength = 1.0
        elif leader_ask >= 0.75:
            ask_strength = 0.9
        elif leader_ask >= 0.70:
            ask_strength = 0.8
        elif leader_ask >= 0.65:
            ask_strength = 0.7
        elif leader_ask >= 0.60:
            ask_strength = 0.6
        else:
            ask_strength = 0.0
        signals["leader_ask"] = round(leader_ask, 4)
        signals["ask_strength"] = ask_strength

        ask_weight = 1.5
        votes_up += (
            (ask_strength if market_leader_up else (1 - ask_strength))
            * ask_weight
        )
        total_weight += ask_weight

        # Compute ensemble probability
        ensemble_prob_up = (
            votes_up / total_weight if total_weight > 0 else 0.5
        )

        if ensemble_prob_up >= 0.5:
            side = "up"
            confidence = ensemble_prob_up
        else:
            side = "down"
            confidence = 1.0 - ensemble_prob_up

        signals["ensemble_prob_up"] = round(ensemble_prob_up, 4)
        signals["confidence"] = round(confidence, 4)

        return side, confidence, signals
