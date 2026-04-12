"""
V3 Paper Strategy: Regime-gated.

Same RF model as V1. Same ensemble, same thresholds. The ONLY
difference: before the entry phase even runs the model, a regime
classifier checks if the current window is "tradeable."

If the window looks choppy, low-volatility, or indecisive, the
strategy SKIPS the window entirely — no entry, no prediction, no risk.

This isolates the question: "Does skipping bad windows improve accuracy?"

Evidence backing this approach:
  - All 8 losses in the deep research were reversal cases in low-vol windows
  - Loss BTC vol (30s) avg = 4.72 vs win avg = 7.12
  - Reddit post: "Start with regime detection, not indicator tuning"
"""

from .paper_strategy import (
    PaperStrategy,
    PREDICTION_TIMES,
    MIN_ELAPSED,
    MAX_ENTRY_ELAPSED,
    MAX_SPREAD,
    _conf_threshold,
)
from .prediction_engine import PredictionEngine

# Regime thresholds — derived from loss analysis
# (losses had btc_vol_30s < 5.0, book_imbalance < 0.10, leader_flips > 2)
REGIME_MIN_BTC_VOL_30S = 3.0       # skip if BTC volatility is too low
REGIME_MAX_LEADER_FLIPS = 4        # skip if market is too choppy
REGIME_MIN_BOOK_IMBALANCE = 0.08   # skip if market is undecided


class PaperStrategyV3(PaperStrategy):
    """V3: Adds regime pre-filter before entry. Everything else = V1."""

    def _handle_entry_phase(self, elapsed, up_bid, up_ask, down_bid, down_ask, btc_price):
        """Override: check regime BEFORE running the normal entry logic."""
        if elapsed < MIN_ELAPSED or elapsed > MAX_ENTRY_ELAPSED:
            return

        # ---- REGIME CHECK (V3 addition) ----
        # Compute regime features from the predictor's data buffer.
        # These are the SAME features the model uses, just checked
        # as a pre-filter before the model even runs.
        feat = self.predictor._compute_features_safe(elapsed)
        if feat is not None:
            btc_vol = feat.get("btc_vol_30s", 0)
            leader_flips = feat.get("leader_flips_60s", 0)
            book_imb = abs(feat.get("book_imbalance", 0))

            # Skip if regime is unfavorable
            if btc_vol < REGIME_MIN_BTC_VOL_30S:
                self._log_skip(
                    elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                    side=None, ask=None, spread=None,
                    reason="regime_low_volatility",
                    spread_passed=None, reversal_passed=None,
                )
                return

            if leader_flips > REGIME_MAX_LEADER_FLIPS:
                self._log_skip(
                    elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                    side=None, ask=None, spread=None,
                    reason="regime_choppy",
                    spread_passed=None, reversal_passed=None,
                )
                return

            if book_imb < REGIME_MIN_BOOK_IMBALANCE:
                self._log_skip(
                    elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                    side=None, ask=None, spread=None,
                    reason="regime_indecisive",
                    spread_passed=None, reversal_passed=None,
                )
                return

        # ---- PASSED REGIME CHECK — run normal V1 entry logic ----
        super()._handle_entry_phase(
            elapsed, up_bid, up_ask, down_bid, down_ask, btc_price
        )
