"""
V2 Paper Strategy: XGBoost model.

Identical to V1 (paper_strategy.py) except the PredictionEngine
loads XGBoost models (xgb_model_t*.joblib) instead of RandomForest.

Same features, same ensemble logic, same thresholds, same entry
timing, same hedge/stop-loss. ONLY the underlying ML algorithm differs.

This isolates the question: "Does XGBoost predict better than RF?"
"""

from .paper_strategy import PaperStrategy


class PaperStrategyV2(PaperStrategy):
    """V2: Uses XGBoost models. Everything else identical to V1."""
    pass
    # No overrides needed — the difference is which PredictionEngine
    # is passed in at construction time. The collector passes
    # PredictionEngineXGB instead of PredictionEngine.
    # The strategy logic is 100% identical to V1.
