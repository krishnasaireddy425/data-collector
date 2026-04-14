"""
Live-compatible predictor for the second-level continuous XGBoost model.

API mirrors how the live collector works: feed ticks one at a time via
add_tick(); at any time call predict(elapsed_sec) to get P(Up) given the
buffer so far. The same feature function used in training is applied to
the buffer (resampled to 1-sec grid, ffill) and the last row is read.

If an isotonic calibrator.joblib exists next to the model it is applied
automatically — raw XGBoost probabilities are remapped so the reported
confidence matches actual historical accuracy (fixes over-confidence).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features_sec import FEATURES, compute_features_batch, resample_window

MODEL_DIR = Path(__file__).resolve().parent / "models"


class PredictorSec:
    def __init__(self, model_path=None, feature_names_path=None,
                 calibrator_path=None):
        model_path = model_path or (MODEL_DIR / "xgb_sec.joblib")
        feature_names_path = feature_names_path or (
            MODEL_DIR / "feature_names_sec.json"
        )
        calibrator_path = calibrator_path or (
            MODEL_DIR / "calibrator.joblib"
        )
        self.model = joblib.load(model_path)
        self.feature_names = json.loads(Path(feature_names_path).read_text())
        assert self.feature_names == FEATURES, (
            "feature_names mismatch between training and predictor"
        )
        # Calibrator is optional — older models may not have one.
        self.calibrator = None
        if Path(calibrator_path).exists():
            try:
                self.calibrator = joblib.load(calibrator_path)
            except Exception:
                self.calibrator = None
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
        """Return (prob_up, feature_dict) using the buffer up to elapsed_sec.

        prob_up is post-calibration if a calibrator is loaded, else raw XGB
        output. Returns (None, {}) if not enough data.
        """
        if not self._ticks or self._btc_open is None:
            return None, {}
        elapsed_sec = float(elapsed_sec)
        if elapsed_sec < 1:
            return None, {}

        df_ticks = pd.DataFrame(self._ticks)
        df_ticks = df_ticks[df_ticks["elapsed_sec"] <= elapsed_sec]
        if len(df_ticks) < 2:
            return None, {}

        max_t = int(elapsed_sec)
        df_1s = resample_window(df_ticks, max_t=max_t)
        if df_1s is None or df_1s.empty:
            return None, {}

        feats = compute_features_batch(df_1s, self._btc_open)
        row = feats.iloc[-1]
        X = row[FEATURES].values.reshape(1, -1).astype(np.float32)
        raw = float(self.model.predict_proba(X)[0, 1])
        if self.calibrator is not None:
            prob = float(self.calibrator.predict([raw])[0])
        else:
            prob = raw
        return prob, row.to_dict()
