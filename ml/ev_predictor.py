"""
Live-compatible EV predictor. Same pattern as predictor_sec.py but loads
both EV_Up and EV_Down regressors and returns predicted PnL for each
side given the current buffer.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features_sec import FEATURES, compute_features_batch, resample_window

MODEL_DIR = Path(__file__).resolve().parent / "models"


class EVPredictor:
    def __init__(self, up_path=None, down_path=None, feature_names_path=None):
        up_path = up_path or (MODEL_DIR / "ev_up.joblib")
        down_path = down_path or (MODEL_DIR / "ev_down.joblib")
        feature_names_path = feature_names_path or (
            MODEL_DIR / "ev_feature_names.json"
        )
        self.model_up = joblib.load(up_path)
        self.model_down = joblib.load(down_path)
        self.feature_names = json.loads(Path(feature_names_path).read_text())
        assert self.feature_names == FEATURES, "feature_names mismatch"
        self.reset()

    def reset(self):
        self._ticks = []
        self._btc_open = None

    def add_tick(self, elapsed_sec, up_bid, up_ask, down_bid, down_ask,
                 btc_price):
        if btc_price is None or up_ask is None or down_ask is None:
            return
        if self._btc_open is None:
            self._btc_open = float(btc_price)
        self._ticks.append({
            "elapsed_sec": float(elapsed_sec),
            "up_bid": float(up_bid if up_bid is not None else 0.0),
            "up_ask": float(up_ask),
            "down_bid": float(down_bid if down_bid is not None else 0.0),
            "down_ask": float(down_ask),
            "btc_price": float(btc_price),
        })

    def predict(self, elapsed_sec):
        """Returns (ev_up, ev_down) expected PnL per share, or (None, None)
        if not enough data."""
        if not self._ticks or self._btc_open is None:
            return None, None
        elapsed_sec = float(elapsed_sec)
        if elapsed_sec < 1:
            return None, None

        df = pd.DataFrame(self._ticks)
        df = df[df["elapsed_sec"] <= elapsed_sec]
        if len(df) < 2:
            return None, None

        max_t = int(elapsed_sec)
        df_1s = resample_window(df, max_t=max_t)
        if df_1s is None or df_1s.empty:
            return None, None

        feats = compute_features_batch(df_1s, self._btc_open)
        X = feats.iloc[-1][FEATURES].values.reshape(1, -1).astype(np.float32)
        ev_up = float(self.model_up.predict(X)[0])
        ev_down = float(self.model_down.predict(X)[0])
        return ev_up, ev_down
