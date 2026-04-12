"""
XGBoost variant of PredictionEngine.

Identical to PredictionEngine except it loads xgb_model_t*.joblib
instead of rf_model_t*.joblib. Same features, same ensemble logic.
Used by V2 paper strategy.
"""

import json
import os

from .prediction_engine import PredictionEngine, DEFAULT_MODEL_DIR, PREDICTION_TIMES

try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class PredictionEngineXGB(PredictionEngine):
    """Loads XGBoost models instead of RandomForest. Everything else inherited."""

    def _load_models(self):
        if not ML_AVAILABLE:
            print("  [ML-XGB] joblib not available — disabled")
            return

        for t in PREDICTION_TIMES:
            # Load XGBoost model instead of RF
            mp = os.path.join(self.model_dir, f"xgb_model_t{t}.joblib")
            fp = os.path.join(self.model_dir, f"feature_names_t{t}.json")
            if os.path.exists(mp) and os.path.exists(fp):
                self.models[t] = joblib.load(mp)
                with open(fp) as f:
                    self.feature_names[t] = json.load(f)

        if self.models:
            print(f"  [ML-XGB] Loaded {len(self.models)} XGBoost models: "
                  f"t={sorted(self.models.keys())}")
        else:
            print(f"  [ML-XGB] WARNING: no XGBoost models found in "
                  f"{self.model_dir}")
