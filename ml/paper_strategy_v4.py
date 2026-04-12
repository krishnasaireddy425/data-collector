"""
V4 Paper Strategy: Late entry + raw ML probability (no ensemble).

Same RF model as V1. Two changes:
  1. MIN_ELAPSED = 210 (don't enter before 3.5 minutes)
  2. Use raw ML probability for side/confidence (bypass ensemble)

This isolates TWO findings from the deep research:
  - Walk-forward accuracy jumps from 56% at t=60 to 80% at t=210
  - Raw ML beats ensemble by 3-5% at t=210-240

Everything else is identical: same thresholds, same hedge logic,
same stop-loss, same features.
"""

from .paper_strategy import (
    PaperStrategy,
    PREDICTION_TIMES,
    MAX_ENTRY_ELAPSED,
    MAX_SPREAD,
    ENTRY_MAX_PRICE,
    ENTRY_CONFIRM_SEC,
    TIER3_MAX_COMBINED,
    MIN_ORDER_DOLLARS,
    POSITION_SIZE,
    _conf_threshold,
)

# V4 override: only enter at t=210+
V4_MIN_ELAPSED = 210.0


class PaperStrategyV4(PaperStrategy):
    """V4: Enter at t=210+ using raw ML probability (no ensemble)."""

    def _handle_entry_phase(self, elapsed, up_bid, up_ask, down_bid, down_ask, btc_price):
        """Override: wait until t=210, use raw ML prob instead of ensemble."""
        # V4: don't even look before t=210
        if elapsed < V4_MIN_ELAPSED or elapsed > MAX_ENTRY_ELAPSED:
            return

        up_spread = (up_ask - (up_bid or 0)) if up_bid else 0.10
        down_spread = (down_ask - (down_bid or 0)) if down_bid else 0.10

        if up_ask > down_ask:
            lead_side = "up"
            lead_ask = up_ask
            lead_spread = up_spread
        else:
            lead_side = "down"
            lead_ask = down_ask
            lead_spread = down_spread

        # Spread filter (same as V1)
        if lead_spread > MAX_SPREAD:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                lead_side, lead_ask, lead_spread,
                reason="spread_too_wide", spread_passed=False,
                reversal_passed=self.rev_tracker.is_stable(),
            )
            return

        # Reversal filter (same as V1)
        if not self.rev_tracker.is_stable():
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                lead_side, lead_ask, lead_spread,
                reason="reversal_blocked", spread_passed=True,
                reversal_passed=False,
            )
            return

        # Throttle predictions (same as V1)
        should_predict = (
            elapsed - self.last_prediction_elapsed >= 5.0
            or any(abs(elapsed - tp) < 1.0 for tp in PREDICTION_TIMES)
        )
        if not should_predict:
            return

        self.last_prediction_elapsed = elapsed

        # ---- V4 DIFFERENCE: get RAW ML probability, NOT ensemble ----
        ml_prob = self.predictor._get_ml_prob_raw(elapsed)
        if ml_prob is None:
            return

        # Side and confidence from RAW ML (no ensemble weighting)
        if ml_prob >= 0.5:
            pred_side = "up"
            confidence = ml_prob
        else:
            pred_side = "down"
            confidence = 1.0 - ml_prob

        signals = {
            "ml_prob_up": round(ml_prob, 4),
            "confidence": round(confidence, 4),
            "method": "raw_ml",
        }

        # Log prediction
        self._log_prediction(
            elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
            pred_side, confidence, signals,
        )

        # Threshold check (same thresholds as V1, just applies at t=210+)
        conf_thresh = _conf_threshold(elapsed)
        if confidence < conf_thresh:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, lead_ask, lead_spread,
                reason="confidence_too_low", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=ml_prob,
                threshold=conf_thresh,
            )
            return

        # Entry price filter (same as V1)
        entry_ask = up_ask if pred_side == "up" else down_ask
        if entry_ask > ENTRY_MAX_PRICE:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, entry_ask, lead_spread,
                reason="price_too_high", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=ml_prob,
            )
            return

        # Confirmation (same as V1)
        if self.confirm_side != pred_side:
            self.confirm_side = pred_side
            self.confirm_start = elapsed
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, entry_ask, lead_spread,
                reason="confirm_started", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=ml_prob,
                confirm_elapsed=0.0,
            )
            return

        confirm_dur = elapsed - self.confirm_start
        if confirm_dur < ENTRY_CONFIRM_SEC:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, entry_ask, lead_spread,
                reason="confirm_pending", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=ml_prob,
                confirm_elapsed=confirm_dur,
            )
            return

        # Hedgeability (same as V1)
        hedge_room = TIER3_MAX_COMBINED - entry_ask
        if hedge_room * POSITION_SIZE < MIN_ORDER_DOLLARS:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, entry_ask, lead_spread,
                reason="hedge_impossible", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=ml_prob,
                hedgeable=False,
            )
            return

        # ===== EXECUTE SIMULATED ENTRY =====
        self._simulate_entry(
            elapsed, pred_side, entry_ask, confidence, signals,
            up_bid, up_ask, down_bid, down_ask, btc_price,
        )
