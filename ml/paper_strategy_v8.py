"""
PaperStrategyV8 — V5 cheap-side entry + V6 disagreement signal used to EXIT V5.

V8 is NOT two entries like V7. V8 is V5 alone: the only position that
ever gets opened is the V5 cheap-side entry. V6 (V3 model @ 0.70) runs
in parallel as a *signal*: when V6 would have bought the OPPOSITE side,
V8 sells the V5 position at market bid.

Backtest (1,260 CSVs from epoch >= 1776156900):
  V5 alone + V6-triggered exits = +$306.00 over ~4.4 days = ~$70/day

Tables: v8_windows, v8_events (schema in sql/007_create_v8_clean.sql)
"""

from .predictor_sec import PredictorSec


SHARES = 10

# V5 (cheap-side) config — same as paper_strategy_v5.py
V5_MIN_ENTRY = 30
V5_MAX_ENTRY = 240
V5_PRICE_LO = 0.20
V5_PRICE_HI = 0.40
V5_UPTICK = 0.02

# V6 exit-signal config — same model as V3, threshold 0.70
V6_MIN_SIGNAL = 60
V6_MAX_SIGNAL = 240
V6_CONF_THRESHOLD = 0.70
V6_PREDICT_EVERY_SEC = 1.0

# Columns v8_events accepts (sql/007_create_v8_clean.sql)
V8_EVENTS_COLS = {
    "up_bid", "up_ask", "down_bid", "down_ask",
    "up_spread", "down_spread",
    "btc_price", "btc_change_from_open",
    "prob_up", "predicted_side",
    "action", "side", "shares", "price",
    "cheap_side", "cheap_price", "local_min", "uptick_amount",
}


class PaperStrategyV8:
    def __init__(self, slug, open_epoch, close_epoch, predictor: PredictorSec, logger):
        self.slug = slug
        self.open_epoch = open_epoch
        self.close_epoch = close_epoch
        self.predictor = predictor
        self.logger = logger
        self.predictor.reset()

        # V5 state (the only position we ever open)
        self.entry_side = None
        self.entry_price = None
        self.entry_elapsed = None
        self.entry_local_min = None
        self.entry_uptick = None
        self._min_up_ask = 1.0
        self._min_dn_ask = 1.0

        # Exit state (triggered by V6 opposite-side signal)
        self.exit_made = False
        self.exit_bid = None
        self.exit_elapsed = None
        self.exit_prob_up = None

        self._last_predict_t = -999.0
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

        # Update running minima (always)
        if up_ask < self._min_up_ask: self._min_up_ask = up_ask
        if down_ask < self._min_dn_ask: self._min_dn_ask = down_ask

        # Feed predictor on every tick (needed once we're in a V5 position)
        self.predictor.add_tick(
            elapsed_sec, up_bid or 0.0, up_ask, down_bid or 0.0, down_ask,
            btc_price or 0.0,
        )

        # ---- V5 entry check ----
        if self.entry_side is None:
            self._check_entry(elapsed_sec, up_bid, up_ask, down_bid, down_ask, btc_price)
            return

        # ---- In a V5 position: run V6 exit signal ----
        if not self.exit_made:
            self._check_v6_exit_signal(elapsed_sec, up_bid, up_ask, down_bid, down_ask, btc_price)

    def _check_entry(self, t, ub, ua, db, da, btc):
        if ua <= da:
            cheap_side, cheap_px, local_min = "Up", ua, self._min_up_ask
        else:
            cheap_side, cheap_px, local_min = "Down", da, self._min_dn_ask
        uptick = cheap_px - local_min

        if t < V5_MIN_ENTRY or t > V5_MAX_ENTRY: return
        if not (V5_PRICE_LO <= cheap_px <= V5_PRICE_HI): return
        if uptick < V5_UPTICK: return

        self.entry_side = cheap_side
        self.entry_price = float(cheap_px)
        self.entry_elapsed = float(t)
        self.entry_local_min = float(local_min)
        self.entry_uptick = float(uptick)

        up_spread = (ua - (ub or 0)) if ub else None
        down_spread = (da - (db or 0)) if db else None
        btc_change = btc - self._first_btc if (btc and self._first_btc) else None
        print(f"  [V8] {self.slug} ENTRY {cheap_side} x{SHARES} @ ${cheap_px:.3f}  t={t:.1f}s")
        self.logger.log_event(
            slug=self.slug, elapsed_sec=t, event_type="entry",
            action="buy", side=cheap_side.lower(), shares=SHARES, price=cheap_px,
            up_bid=ub, up_ask=ua, down_bid=db, down_ask=da,
            up_spread=up_spread, down_spread=down_spread,
            btc_price=btc, btc_change_from_open=btc_change,
            cheap_side=cheap_side.lower(), cheap_price=cheap_px,
            local_min=local_min, uptick_amount=uptick,
        )

    def _check_v6_exit_signal(self, t, ub, ua, db, da, btc):
        if t < V6_MIN_SIGNAL or t > V6_MAX_SIGNAL: return
        if t - self._last_predict_t < V6_PREDICT_EVERY_SEC: return
        self._last_predict_t = t

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

        # V6 "would-buy" logic
        v6_side = None
        if prob_up >= V6_CONF_THRESHOLD:
            v6_side = "Up"
        elif (1.0 - prob_up) >= V6_CONF_THRESHOLD:
            v6_side = "Down"

        if v6_side is None or v6_side == self.entry_side:
            return

        # V6 fires on the OPPOSITE side → exit V5 now at bid
        exit_bid = ub if self.entry_side == "Up" else db
        if exit_bid is None or exit_bid <= 0:
            return

        self.exit_made = True
        self.exit_bid = float(exit_bid)
        self.exit_elapsed = float(t)
        self.exit_prob_up = float(prob_up)
        print(f"  [V8] {self.slug} EXIT @ ${exit_bid:.3f}  t={t:.1f}s  "
              f"(V6 signal {v6_side}, prob_up={prob_up:.3f})")
        self.logger.log_event(
            slug=self.slug, elapsed_sec=t, event_type="exit",
            action="sell", side=self.entry_side.lower(), shares=SHARES, price=exit_bid,
            up_bid=ub, up_ask=ua, down_bid=db, down_ask=da,
            up_spread=up_spread, down_spread=down_spread,
            btc_price=btc, btc_change_from_open=btc_change,
            prob_up=prob_up, predicted_side=v6_side.lower(),
        )

    def settle(self, btc_open=None, btc_close=None, winner=None):
        winner_norm = (winner or "unknown").capitalize() if winner else "unknown"

        if self.entry_side is None:
            print(f"  [V8] {self.slug} SKIPPED  winner={winner_norm}")
            self.logger.log_window_settlement(
                slug=self.slug,
                btc_open=btc_open, btc_close=btc_close,
                winner=winner_norm.lower(),
                entry_made=False,
                pnl=0.0,
            )
            return

        if self.exit_made:
            pnl = SHARES * (self.exit_bid - self.entry_price)
            correct = (pnl > 0)
        else:
            correct = (self.entry_side == winner_norm)
            pnl = (SHARES * (1.0 - self.entry_price)) if correct else (-SHARES * self.entry_price)

        print(f"  [V8] {self.slug} SETTLE  side={self.entry_side}  winner={winner_norm}  "
              f"{'EXIT' if self.exit_made else ('WIN' if correct else 'LOSS')}  pnl=${pnl:+.2f}")

        self.logger.log_window_settlement(
            slug=self.slug,
            btc_open=btc_open, btc_close=btc_close,
            winner=winner_norm.lower(),
            entry_made=True,
            entry_elapsed_sec=self.entry_elapsed,
            entry_side=self.entry_side.lower(),
            entry_price=self.entry_price,
            entry_shares=SHARES,
            entry_local_min=self.entry_local_min,
            entry_uptick_amount=self.entry_uptick,
            exit_made=self.exit_made,
            exit_elapsed_sec=self.exit_elapsed,
            exit_bid=self.exit_bid,
            exit_shares=(SHARES if self.exit_made else 0),
            exit_prob_up=self.exit_prob_up,
            correct=correct,
            pnl=round(pnl, 4),
        )
