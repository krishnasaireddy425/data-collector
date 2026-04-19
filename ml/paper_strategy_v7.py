"""
PaperStrategyV7 — COMBINED: V6 entry + V5 entry + V5 exit on V6 disagree.

V7 internally runs both V5's cheap-side rule AND V6's 44f-XGB-at-0.70
logic. When V6 enters a side opposite to V5's existing entry, V7 sells
V5 at current market bid; V6 holds to settlement.

Backtest (1,260 CSVs from epoch >= 1776156900):
  combined PnL +$667.30 over ~4.4 days = ~$152/day

Tables: v7_windows, v7_events (schema in sql/006_create_v7_clean.sql)
"""

from .predictor_sec import PredictorSec


SHARES = 10

# V5 (cheap-side) config — matches paper_strategy_v5.py
V5_MIN_ENTRY = 30
V5_MAX_ENTRY = 240
V5_PRICE_LO = 0.20
V5_PRICE_HI = 0.40
V5_UPTICK = 0.02

# V6 (V3 model at 0.70) config — matches paper_strategy_v6.py
V6_MIN_ENTRY = 60
V6_MAX_ENTRY = 240
V6_CONF_THRESHOLD = 0.70
V6_PREDICT_EVERY_SEC = 1.0

# Columns that v7_events accepts (sql/006_create_v7_clean.sql)
V7_EVENTS_COLS = {
    "up_bid", "up_ask", "down_bid", "down_ask",
    "up_spread", "down_spread",
    "btc_price", "btc_change_from_open",
    "prob_up", "predicted_side",
    "action", "side", "shares", "price",
    "cheap_side", "cheap_price", "local_min", "uptick_amount",
}


class PaperStrategyV7:
    def __init__(self, slug, open_epoch, close_epoch, predictor: PredictorSec, logger):
        self.slug = slug
        self.open_epoch = open_epoch
        self.close_epoch = close_epoch
        self.predictor = predictor
        self.logger = logger
        self.predictor.reset()

        # V5 state
        self.v5_entry_side = None
        self.v5_entry_price = None
        self.v5_entry_elapsed = None
        self.v5_entry_local_min = None
        self.v5_entry_uptick = None
        self._v5_min_up_ask = 1.0
        self._v5_min_dn_ask = 1.0

        # V6 state
        self.v6_entry_side = None
        self.v6_entry_price = None
        self.v6_entry_elapsed = None
        self.v6_entry_prob_up = None
        self.v6_entry_confidence = None
        self._v6_last_predict_t = -999.0

        # V5 exit (triggered by V6 disagreement)
        self.v5_exit_made = False
        self.v5_exit_bid = None
        self.v5_exit_elapsed = None

        self._window_logged = False
        self._first_btc = None

    def on_tick(self, elapsed_sec, up_bid, up_ask, down_bid, down_ask, btc_price):
        if up_ask is None or down_ask is None:
            return

        if not self._window_logged:
            self._first_btc = btc_price
            self.logger.log_window_start(
                slug=self.slug,
                open_epoch=self.open_epoch,
                close_epoch=self.close_epoch,
                btc_open=btc_price,
            )
            self._window_logged = True

        # Update V5 running minima (always — even pre-MIN_ENTRY)
        if up_ask < self._v5_min_up_ask: self._v5_min_up_ask = up_ask
        if down_ask < self._v5_min_dn_ask: self._v5_min_dn_ask = down_ask

        # Feed V6 predictor on every tick
        self.predictor.add_tick(
            elapsed_sec, up_bid or 0.0, up_ask, down_bid or 0.0, down_ask,
            btc_price or 0.0,
        )

        # ---- V5 entry check (if not yet entered) ----
        if self.v5_entry_side is None:
            self._check_v5_entry(elapsed_sec, up_bid, up_ask, down_bid, down_ask, btc_price)

        # ---- V6 entry check (if not yet entered) ----
        if self.v6_entry_side is None:
            self._check_v6_entry(elapsed_sec, up_bid, up_ask, down_bid, down_ask, btc_price)

        # ---- V6 just entered? Check V5 exit condition ----
        # (handled inline after V6 entry is recorded)

    def _check_v5_entry(self, t, ub, ua, db, da, btc):
        if ua <= da:
            cheap_side, cheap_px, local_min = "Up", ua, self._v5_min_up_ask
        else:
            cheap_side, cheap_px, local_min = "Down", da, self._v5_min_dn_ask
        uptick = cheap_px - local_min

        if t < V5_MIN_ENTRY or t > V5_MAX_ENTRY: return
        if not (V5_PRICE_LO <= cheap_px <= V5_PRICE_HI): return
        if uptick < V5_UPTICK: return

        # Guard: if V6 already entered on the OPPOSITE side, skip V5.
        # The V6-disagree exit only fires at V6 entry time, so a V5 that
        # enters AFTER V6 on the wrong side would just sit against V6.
        if self.v6_entry_side is not None and cheap_side != self.v6_entry_side:
            return

        # Fire V5 entry
        self.v5_entry_side = cheap_side
        self.v5_entry_price = float(cheap_px)
        self.v5_entry_elapsed = float(t)
        self.v5_entry_local_min = float(local_min)
        self.v5_entry_uptick = float(uptick)
        up_spread = (ua - (ub or 0)) if ub else None
        down_spread = (da - (db or 0)) if db else None
        btc_change = btc - self._first_btc if (btc and self._first_btc) else None
        print(f"  [V7] {self.slug} V5 ENTRY {cheap_side} x{SHARES} @ ${cheap_px:.3f}  t={t:.1f}s")
        self.logger.log_event(
            slug=self.slug, elapsed_sec=t, event_type="v5_entry",
            action="buy", side=cheap_side.lower(), shares=SHARES, price=cheap_px,
            up_bid=ub, up_ask=ua, down_bid=db, down_ask=da,
            up_spread=up_spread, down_spread=down_spread,
            btc_price=btc, btc_change_from_open=btc_change,
            cheap_side=cheap_side.lower(), cheap_price=cheap_px,
            local_min=local_min, uptick_amount=uptick,
        )

    def _check_v6_entry(self, t, ub, ua, db, da, btc):
        if t < V6_MIN_ENTRY or t > V6_MAX_ENTRY: return
        if t - self._v6_last_predict_t < V6_PREDICT_EVERY_SEC: return
        self._v6_last_predict_t = t

        prob_up, _ = self.predictor.predict(t)
        if prob_up is None: return

        up_spread = (ua - (ub or 0)) if ub else None
        down_spread = (da - (db or 0)) if db else None
        btc_change = btc - self._first_btc if (btc and self._first_btc) else None
        predicted_side = "up" if prob_up >= 0.5 else "down"
        self.logger.log_event(
            slug=self.slug, elapsed_sec=t, event_type="prediction",
            up_bid=ub, up_ask=ua, down_bid=db, down_ask=da,
            up_spread=up_spread, down_spread=down_spread,
            btc_price=btc, btc_change_from_open=btc_change,
            prob_up=prob_up, predicted_side=predicted_side,
        )

        side = None
        conf = None
        if prob_up >= V6_CONF_THRESHOLD:
            side, price, conf = "Up", ua, prob_up
        elif (1.0 - prob_up) >= V6_CONF_THRESHOLD:
            side, price, conf = "Down", da, 1.0 - prob_up

        if side is None: return

        self.v6_entry_side = side
        self.v6_entry_price = float(price)
        self.v6_entry_elapsed = float(t)
        self.v6_entry_prob_up = float(prob_up)
        self.v6_entry_confidence = float(conf)
        print(f"  [V7] {self.slug} V6 ENTRY {side} x{SHARES} @ ${price:.3f}  t={t:.1f}s  conf={conf:.3f}")
        self.logger.log_event(
            slug=self.slug, elapsed_sec=t, event_type="v6_entry",
            action="buy", side=side.lower(), shares=SHARES, price=price,
            up_bid=ub, up_ask=ua, down_bid=db, down_ask=da,
            up_spread=up_spread, down_spread=down_spread,
            btc_price=btc, btc_change_from_open=btc_change,
            prob_up=prob_up, predicted_side=side.lower(),
        )

        # V6 just entered. If V5 is open and V6 took OPPOSITE side → exit V5 now.
        if self.v5_entry_side is not None and side != self.v5_entry_side and not self.v5_exit_made:
            v5_bid = ub if self.v5_entry_side == "Up" else db
            if v5_bid is not None and v5_bid > 0:
                self.v5_exit_made = True
                self.v5_exit_bid = float(v5_bid)
                self.v5_exit_elapsed = float(t)
                print(f"  [V7] {self.slug} V5 EXIT @ ${v5_bid:.3f}  t={t:.1f}s  (V6 disagreed)")
                self.logger.log_event(
                    slug=self.slug, elapsed_sec=t, event_type="v5_exit",
                    action="sell", side=self.v5_entry_side.lower(), shares=SHARES, price=v5_bid,
                    up_bid=ub, up_ask=ua, down_bid=db, down_ask=da,
                    btc_price=btc, btc_change_from_open=btc_change,
                )

    def settle(self, btc_open=None, btc_close=None, winner=None):
        winner_norm = (winner or "unknown").capitalize() if winner else "unknown"
        winner_low = winner_norm.lower()

        # V5 leg PnL
        v5_correct = None; v5_pnl = 0.0
        if self.v5_entry_side is not None:
            if self.v5_exit_made:
                # Sold V5 early — PnL = 10 * (exit_bid - entry_price)
                v5_pnl = SHARES * (self.v5_exit_bid - self.v5_entry_price)
                v5_correct = (v5_pnl > 0)
            else:
                v5_correct = (self.v5_entry_side == winner_norm)
                v5_pnl = (SHARES * (1.0 - self.v5_entry_price)) if v5_correct else (-SHARES * self.v5_entry_price)

        # V6 leg PnL
        v6_correct = None; v6_pnl = 0.0
        if self.v6_entry_side is not None:
            v6_correct = (self.v6_entry_side == winner_norm)
            v6_pnl = (SHARES * (1.0 - self.v6_entry_price)) if v6_correct else (-SHARES * self.v6_entry_price)

        combined = v5_pnl + v6_pnl
        print(f"  [V7] {self.slug} SETTLE  winner={winner_norm}  v5_pnl=${v5_pnl:+.2f}  v6_pnl=${v6_pnl:+.2f}  combined=${combined:+.2f}")

        self.logger.log_window_settlement(
            slug=self.slug,
            btc_open=btc_open, btc_close=btc_close,
            winner=winner_low,
            v5_entry_made=self.v5_entry_side is not None,
            v5_entry_elapsed_sec=self.v5_entry_elapsed,
            v5_entry_side=(self.v5_entry_side or "").lower() or None,
            v5_entry_price=self.v5_entry_price,
            v5_entry_shares=SHARES if self.v5_entry_side else 0,
            v5_entry_local_min=self.v5_entry_local_min,
            v5_entry_uptick_amount=self.v5_entry_uptick,
            v6_entry_made=self.v6_entry_side is not None,
            v6_entry_elapsed_sec=self.v6_entry_elapsed,
            v6_entry_side=(self.v6_entry_side or "").lower() or None,
            v6_entry_price=self.v6_entry_price,
            v6_entry_shares=SHARES if self.v6_entry_side else 0,
            v6_entry_prob_up=self.v6_entry_prob_up,
            v6_entry_confidence=self.v6_entry_confidence,
            v5_exit_made=self.v5_exit_made,
            v5_exit_elapsed_sec=self.v5_exit_elapsed,
            v5_exit_bid=self.v5_exit_bid,
            v5_exit_shares=SHARES if self.v5_exit_made else 0,
            v5_correct=v5_correct,
            v6_correct=v6_correct,
            v5_pnl=round(v5_pnl, 4),
            v6_pnl=round(v6_pnl, 4),
            combined_pnl=round(combined, 4),
        )
