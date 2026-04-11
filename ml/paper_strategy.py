"""
PaperStrategy — 1:1 mirror of btc/prod_ml_strategy.py state machine.

Same constants, same entry filters, same hedge tier logic, same stop-loss,
same confirmation period, same reversal filter. The ONLY difference is
that orders are simulated locally — no real Polymarket orders are placed,
no API keys, no wallet, nothing that touches money.

Designed to be driven by an external tick stream (the price collector
already has the WS connections — we hook in via on_tick).

Every decision is logged to the DecisionLogger (Supabase).
"""

import time
from collections import deque

from .prediction_engine import PredictionEngine, ReversalTracker, PREDICTION_TIMES


# ----------------------------------------------------------------------
# Strategy parameters (must match prod_ml_strategy.py)
# ----------------------------------------------------------------------

POSITION_SIZE = 11
MIN_ENTRY_SHARES = 11
ENTRY_MAX_PRICE = 0.85
MAX_SPREAD = 0.04
MIN_ELAPSED = 75.0
MAX_ENTRY_ELAPSED = 240.0

CONFIDENCE_THRESHOLDS = {
    75: 0.80, 90: 0.75, 120: 0.70, 150: 0.65,
    180: 0.60, 210: 0.55, 240: 0.55,
}

TIER1_MAX_COMBINED = 0.88
TIER2_MAX_COMBINED = 0.93
TIER3_MAX_COMBINED = 0.97
TIER4_MAX_COMBINED = 0.99

STOP_LOSS_PCT = 0.20

REVERSAL_WINDOW_SEC = 90.0
MAX_REVERSALS = 3
REVERSAL_MIN_GAP = 0.02

ENTRY_CONFIRM_SEC = 3.0
HEDGE_COOLDOWN_SEC = 1.0
MIN_ORDER_DOLLARS = 1.00

SL_SUSTAIN_FIRST = 5.0
SL_SUSTAIN_AFTER = 3.0


def _conf_threshold(elapsed):
    """Confidence threshold for the given elapsed time."""
    thresh = 0.70
    for tp in sorted(CONFIDENCE_THRESHOLDS.keys()):
        if elapsed >= tp:
            thresh = CONFIDENCE_THRESHOLDS[tp]
    return thresh


# ----------------------------------------------------------------------
# Strategy state machine — one instance per market window
# ----------------------------------------------------------------------

class PaperStrategy:
    """
    Mirror of prod_ml_strategy.trade_one_window state machine.

    Driven externally via on_tick(). The collector pushes every order-book
    update + every BTC price update into this object, and the strategy
    decides what to do (predict, enter, hedge, stop-loss).
    """

    def __init__(self, slug, open_epoch, close_epoch, predictor: PredictionEngine,
                 logger):
        self.slug = slug
        self.open_epoch = open_epoch
        self.close_epoch = close_epoch
        self.predictor = predictor
        self.logger = logger

        self.predictor.reset()

        # Reversal tracker
        self.rev_tracker = ReversalTracker(
            window_sec=REVERSAL_WINDOW_SEC,
            max_reversals=MAX_REVERSALS,
            min_gap=REVERSAL_MIN_GAP,
        )

        # Position state (simulated)
        self.entry_side = None
        self.entry_shares = 0.0
        self.entry_spent = 0.0
        self.entry_avg = 0.0
        self.entry_elapsed_sec = None
        self.entry_confidence = None
        self.entry_ml_prob = None

        self.hedge_shares = 0.0
        self.hedge_spent = 0.0
        self.hedge_tier = None
        self.hedge_elapsed_sec = None
        self.hedge_combined_cost = None
        self.hedge_opp_ask = None
        self.hedge_confidence = None

        # Confirmation
        self.confirm_side = None
        self.confirm_start = None

        # Stop-loss
        self.stopped_out = False
        self.sl_first_trigger = None
        self.sl_sustained = False
        self.stop_loss_elapsed_sec = None
        self.stop_loss_price = None
        self.sl_sell_attempts = 0
        self.stop_loss_received = 0.0

        # Cooldowns
        self.last_hedge_attempt = 0.0
        self.last_prediction_elapsed = -999.0

        # Logged window-start
        self._window_logged = False
        self._first_btc = None

    # ------------------------------------------------------------------
    # Public entry point: called by collector for every tick
    # ------------------------------------------------------------------

    def on_tick(self, elapsed_sec, up_bid, up_ask, down_bid, down_ask, btc_price):
        """Process one market tick (called from the collector loop)."""
        if up_ask is None or down_ask is None:
            return

        # First tick: register window
        if not self._window_logged:
            self._first_btc = btc_price
            self.logger.log_window_start(
                slug=self.slug,
                open_epoch=self.open_epoch,
                close_epoch=self.close_epoch,
                btc_open=btc_price,
            )
            self._window_logged = True

        # Feed predictor
        self.predictor.add_tick(
            elapsed_sec, up_bid or 0.0, up_ask, down_bid or 0.0, down_ask,
            btc_price or 0.0,
        )

        # Reversal tracker uses wall-clock-style timestamps (we use elapsed)
        self.rev_tracker.update(elapsed_sec, up_ask, down_ask)

        # Route based on state
        if self.entry_side is None and not self.stopped_out:
            self._handle_entry_phase(
                elapsed_sec, up_bid, up_ask, down_bid, down_ask, btc_price
            )
        elif self.entry_side is not None and not self.stopped_out:
            self._handle_post_entry(
                elapsed_sec, up_bid, up_ask, down_bid, down_ask, btc_price
            )

    # ------------------------------------------------------------------
    # Entry phase
    # ------------------------------------------------------------------

    def _handle_entry_phase(self, elapsed, up_bid, up_ask, down_bid, down_ask, btc_price):
        if elapsed < MIN_ELAPSED or elapsed > MAX_ENTRY_ELAPSED:
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

        # Spread filter
        if lead_spread > MAX_SPREAD:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                lead_side, lead_ask, lead_spread,
                reason="spread_too_wide", spread_passed=False,
                reversal_passed=self.rev_tracker.is_stable(),
            )
            return

        # Reversal filter
        if not self.rev_tracker.is_stable():
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                lead_side, lead_ask, lead_spread,
                reason="reversal_blocked", spread_passed=True,
                reversal_passed=False,
            )
            return

        # Throttle predictions: every 5s OR at trained timepoints
        should_predict = (
            elapsed - self.last_prediction_elapsed >= 5.0
            or any(abs(elapsed - tp) < 1.0 for tp in PREDICTION_TIMES)
        )
        if not should_predict:
            return

        self.last_prediction_elapsed = elapsed

        pred_side, confidence, signals = self.predictor.predict(elapsed)
        if pred_side is None:
            return

        # Always log the prediction
        self._log_prediction(
            elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
            pred_side, confidence, signals,
        )

        # Threshold check
        conf_thresh = _conf_threshold(elapsed)
        if confidence < conf_thresh:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, lead_ask, lead_spread,
                reason="confidence_too_low", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=signals.get("ml_prob_up"),
                threshold=conf_thresh,
            )
            return

        # Entry price filter
        entry_ask = up_ask if pred_side == "up" else down_ask
        if entry_ask > ENTRY_MAX_PRICE:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, entry_ask, lead_spread,
                reason="price_too_high", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=signals.get("ml_prob_up"),
            )
            return

        # Confirmation: must sustain for ENTRY_CONFIRM_SEC
        if self.confirm_side != pred_side:
            self.confirm_side = pred_side
            self.confirm_start = elapsed
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, entry_ask, lead_spread,
                reason="confirm_started", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=signals.get("ml_prob_up"),
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
                confidence=confidence, ml_prob=signals.get("ml_prob_up"),
                confirm_elapsed=confirm_dur,
            )
            return

        # Hedgeability check
        hedge_room = TIER3_MAX_COMBINED - entry_ask
        if hedge_room * POSITION_SIZE < MIN_ORDER_DOLLARS:
            self._log_skip(
                elapsed, up_bid, up_ask, down_bid, down_ask, btc_price,
                pred_side, entry_ask, lead_spread,
                reason="hedge_impossible", spread_passed=True,
                reversal_passed=True,
                confidence=confidence, ml_prob=signals.get("ml_prob_up"),
                hedgeable=False,
            )
            return

        # ===== EXECUTE SIMULATED ENTRY =====
        self._simulate_entry(
            elapsed, pred_side, entry_ask, confidence, signals,
            up_bid, up_ask, down_bid, down_ask, btc_price,
        )

    def _simulate_entry(self, elapsed, side, ask, confidence, signals,
                        up_bid, up_ask, down_bid, down_ask, btc_price):
        """Simulate filling POSITION_SIZE shares at the current ask."""
        self.entry_side = side
        self.entry_shares = float(POSITION_SIZE)
        self.entry_spent = float(POSITION_SIZE) * float(ask)
        self.entry_avg = float(ask)
        self.entry_elapsed_sec = elapsed
        self.entry_confidence = confidence
        self.entry_ml_prob = signals.get("ml_prob_up")

        print(f"  [PAPER] >> ENTRY {side.upper()} @ t={elapsed:.1f}s "
              f"ask=${ask:.4f} conf={confidence:.1%}")

        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed,
            event_type="entry",
            up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            up_spread=(up_ask - (up_bid or 0)) if up_bid else None,
            down_spread=(down_ask - (down_bid or 0)) if down_bid else None,
            btc_price=btc_price,
            btc_change_from_open=(
                (btc_price or 0) - (self._first_btc or 0)
                if self._first_btc else None
            ),
            ml_prob_up=signals.get("ml_prob_up"),
            ml_model_t=signals.get("model_t"),
            ensemble_confidence=confidence,
            predicted_side=side,
            market_leader_signal=signals.get("market_leader"),
            btc_direction_signal=signals.get("btc_dir"),
            btc_market_agree=signals.get("btc_market_agree"),
            ask_strength=signals.get("ask_strength"),
            action="enter",
            side=side,
            shares=int(POSITION_SIZE),
            price=ask,
            reason="threshold_met_simulated_fill",
        )

    # ------------------------------------------------------------------
    # Post-entry phase: hedge + stop-loss
    # ------------------------------------------------------------------

    def _handle_post_entry(self, elapsed, up_bid, up_ask, down_bid, down_ask, btc_price):
        if self.hedge_tier is not None:
            return  # nothing to do — fully hedged

        opp_ask = down_ask if self.entry_side == "up" else up_ask
        opp_bid = down_bid if self.entry_side == "up" else up_bid
        entry_bid_now = up_bid if self.entry_side == "up" else down_bid

        need = max(0.0, self.entry_shares - self.hedge_shares)
        if need < 1:
            return

        # ----- Stop-loss check -----
        if (entry_bid_now is not None
                and entry_bid_now < self.entry_avg * (1 - STOP_LOSS_PCT)):
            if self.sl_first_trigger is None:
                self.sl_first_trigger = elapsed
                self.sl_sustained = False

            sustain_needed = (
                SL_SUSTAIN_FIRST if self.sl_sell_attempts == 0
                else SL_SUSTAIN_AFTER
            )
            if elapsed - self.sl_first_trigger >= sustain_needed:
                self.sl_sustained = True

            if self.sl_sustained:
                self._simulate_stop_loss(
                    elapsed, entry_bid_now,
                    up_bid, up_ask, down_bid, down_ask, btc_price,
                )
                return
        else:
            self.sl_first_trigger = None
            self.sl_sustained = False

        # ----- Hedge cooldown -----
        if elapsed - self.last_hedge_attempt < HEDGE_COOLDOWN_SEC:
            return

        # ----- Hedge tier check -----
        combined = self.entry_avg + opp_ask if opp_ask else 999.0
        tier = None
        if combined < TIER1_MAX_COMBINED:
            tier = 1
        elif combined < TIER2_MAX_COMBINED:
            tier = 2
        elif combined < TIER3_MAX_COMBINED:
            tier = 3
        elif combined < TIER4_MAX_COMBINED:
            tier = 4

        if tier is None:
            return

        self.last_hedge_attempt = elapsed

        # ===== Simulated hedge fill =====
        self._simulate_hedge(
            elapsed, tier, opp_ask, combined,
            up_bid, up_ask, down_bid, down_ask, btc_price,
        )

    def _simulate_hedge(self, elapsed, tier, opp_ask, combined,
                        up_bid, up_ask, down_bid, down_ask, btc_price):
        """Simulate filling the hedge at opp_ask."""
        need = self.entry_shares - self.hedge_shares
        self.hedge_shares = self.entry_shares  # assume full fill in paper mode
        self.hedge_spent = self.hedge_shares * float(opp_ask)
        self.hedge_tier = tier
        self.hedge_elapsed_sec = elapsed
        self.hedge_combined_cost = combined
        self.hedge_opp_ask = opp_ask

        # Snapshot current confidence at hedge moment
        try:
            _, conf, _ = self.predictor.predict(elapsed)
            self.hedge_confidence = conf
        except Exception:
            self.hedge_confidence = None

        guaranteed = 1.0 - combined
        print(f"  [PAPER]    HEDGE T{tier} @ t={elapsed:.1f}s "
              f"opp=${opp_ask:.4f} combined=${combined:.4f} "
              f"guaranteed=${guaranteed:.4f}/sh")

        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed,
            event_type="hedge",
            up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            btc_price=btc_price,
            btc_change_from_open=(
                (btc_price or 0) - (self._first_btc or 0)
                if self._first_btc else None
            ),
            action="hedge",
            side="down" if self.entry_side == "up" else "up",
            shares=int(need),
            price=opp_ask,
            reason=f"tier_{tier}_hit",
            hedge_tier=tier,
            combined_cost=combined,
            opp_ask=opp_ask,
            guaranteed_profit=guaranteed,
        )

    def _simulate_stop_loss(self, elapsed, bid,
                            up_bid, up_ask, down_bid, down_ask, btc_price):
        """Simulate selling at the bid to stop the loss."""
        self.stopped_out = True
        self.stop_loss_elapsed_sec = elapsed
        self.stop_loss_price = bid
        self.stop_loss_received = self.entry_shares * float(bid)

        print(f"  [PAPER]    STOP-LOSS @ t={elapsed:.1f}s bid=${bid:.4f}")

        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed,
            event_type="stop_loss",
            up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            btc_price=btc_price,
            btc_change_from_open=(
                (btc_price or 0) - (self._first_btc or 0)
                if self._first_btc else None
            ),
            action="stop_loss",
            side=self.entry_side,
            shares=int(self.entry_shares),
            price=bid,
            reason="entry_bid_dropped_20pct_sustained",
        )

    # ------------------------------------------------------------------
    # Settlement (called by the collector at window close)
    # ------------------------------------------------------------------

    def settle(self, btc_open, btc_close, winner):
        """Compute final outcome and write the windows row + settlement event."""
        action_type = "no_entry"
        correct = None
        pnl = 0.0

        if self.entry_side is None:
            action_type = "no_entry"
        elif self.stopped_out:
            action_type = "stopped_out"
            pnl = self.stop_loss_received - self.entry_spent
            correct = (winner == self.entry_side)
        elif self.hedge_tier is not None:
            action_type = "full_hedge"
            # fully hedged: payout = entry_shares (winner pays $1/share),
            # cost = entry_spent + hedge_spent
            pnl = self.entry_shares - (self.entry_spent + self.hedge_spent)
            correct = (winner == self.entry_side)
        else:
            action_type = "full_ride"
            if winner == self.entry_side:
                # entry pays $1/share
                pnl = self.entry_shares - self.entry_spent
                correct = True
            elif winner in ("up", "down"):
                pnl = -self.entry_spent
                correct = False
            else:
                correct = None

        # Final settlement event
        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=300.0,
            event_type="settlement",
            btc_price=btc_close,
            action="settlement",
            reason=f"winner={winner}",
        )

        # Update windows row with everything
        self.logger.log_window_settlement(
            slug=self.slug,
            btc_close=btc_close,
            winner=winner,
            entry_made=self.entry_side is not None,
            entry_elapsed_sec=self.entry_elapsed_sec,
            entry_side=self.entry_side,
            entry_ask=self.entry_avg if self.entry_side else None,
            entry_shares=int(self.entry_shares) if self.entry_side else None,
            entry_confidence=self.entry_confidence,
            entry_ml_prob=self.entry_ml_prob,
            action_type=action_type,
            hedge_made=self.hedge_tier is not None,
            hedge_elapsed_sec=self.hedge_elapsed_sec,
            hedge_tier=self.hedge_tier,
            hedge_combined_cost=self.hedge_combined_cost,
            hedge_opp_ask=self.hedge_opp_ask,
            hedge_confidence=self.hedge_confidence,
            stopped_out=self.stopped_out,
            stop_loss_elapsed_sec=self.stop_loss_elapsed_sec,
            stop_loss_price=self.stop_loss_price,
            correct=correct,
            pnl=round(pnl, 4),
        )

        print(f"  [PAPER] {self.slug} | action={action_type} "
              f"winner={winner} correct={correct} pnl=${pnl:+.2f}")

        return {
            "action_type": action_type,
            "correct": correct,
            "pnl": pnl,
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_prediction(self, elapsed, up_bid, up_ask, down_bid, down_ask,
                        btc_price, side, confidence, signals):
        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed,
            event_type="prediction",
            up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            up_spread=(up_ask - (up_bid or 0)) if up_bid else None,
            down_spread=(down_ask - (down_bid or 0)) if down_bid else None,
            btc_price=btc_price,
            btc_change_from_open=(
                (btc_price or 0) - (self._first_btc or 0)
                if self._first_btc else None
            ),
            ml_prob_up=signals.get("ml_prob_up"),
            ml_model_t=signals.get("model_t"),
            ensemble_confidence=confidence,
            predicted_side=side,
            market_leader_signal=signals.get("market_leader"),
            btc_direction_signal=signals.get("btc_dir"),
            btc_market_agree=signals.get("btc_market_agree"),
            ask_strength=signals.get("ask_strength"),
            action="predict",
            side=side,
            reversal_count=self.rev_tracker.count(),
            reversal_passed=self.rev_tracker.is_stable(),
        )

    def _log_skip(self, elapsed, up_bid, up_ask, down_bid, down_ask,
                  btc_price, side, ask, spread, reason,
                  spread_passed=None, reversal_passed=None,
                  confidence=None, ml_prob=None, threshold=None,
                  confirm_elapsed=None, hedgeable=None):
        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed,
            event_type="skip",
            up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            up_spread=(up_ask - (up_bid or 0)) if up_bid else None,
            down_spread=(down_ask - (down_bid or 0)) if down_bid else None,
            btc_price=btc_price,
            btc_change_from_open=(
                (btc_price or 0) - (self._first_btc or 0)
                if self._first_btc else None
            ),
            ensemble_confidence=confidence,
            ml_prob_up=ml_prob,
            predicted_side=side,
            action="skip",
            side=side,
            price=ask,
            reason=reason,
            spread_value=spread,
            spread_passed=spread_passed,
            reversal_count=self.rev_tracker.count(),
            reversal_passed=reversal_passed,
            confirm_elapsed=confirm_elapsed,
            hedgeable=hedgeable,
            # Unknown keys land in details_json automatically
            threshold=threshold,
        )
