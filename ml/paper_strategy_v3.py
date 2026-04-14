"""
PaperStrategyV3 — 44-feature continuous XGB probability model (sample-
weighted + isotonic calibrated). Simple entry rule, no hedging.

Mirrors the backtest that produced +$28.65/day at threshold 0.90 on the
170-window test set.

Entry rule:
  At each tick with elapsed_sec in [MIN_ENTRY, MAX_ENTRY]:
    1. Feed tick into PredictorSec
    2. Call predict(elapsed_sec) → calibrated P(Up)
    3. If P(Up) >= CONF_THRESHOLD          → buy UP at up_ask
       elif (1 - P(Up)) >= CONF_THRESHOLD  → buy DOWN at down_ask
  Once entered: HOLD to settlement. No exits.

Supabase tables (schema in sql/003_recreate_v3_v4_clean.sql):
  v3_windows, v3_events
"""

from .predictor_sec import PredictorSec


SHARES = 10
MIN_ENTRY = 60
MAX_ENTRY = 240
CONF_THRESHOLD = 0.90
PREDICT_EVERY_SEC = 1.0

# Column whitelist for v3_events — matches sql/003_recreate_v3_v4_clean.sql
V3_EVENTS_COLS = {
    "up_bid", "up_ask", "down_bid", "down_ask",
    "up_spread", "down_spread",
    "btc_price", "btc_change_from_open",
    "prob_up", "predicted_side",
    "action", "side", "shares", "price",
}


class PaperStrategyV3:
    def __init__(self, slug, open_epoch, close_epoch, predictor: PredictorSec, logger):
        self.slug = slug
        self.open_epoch = open_epoch
        self.close_epoch = close_epoch
        self.predictor = predictor
        self.logger = logger

        self.predictor.reset()

        self.entry_side = None
        self.entry_shares = 0
        self.entry_price = None
        self.entry_elapsed_sec = None
        self.entry_prob_up = None
        self.entry_confidence = None

        self._window_logged = False
        self._first_btc = None
        self._last_predict_t = -999.0

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

        self.predictor.add_tick(
            elapsed_sec, up_bid or 0.0, up_ask, down_bid or 0.0, down_ask,
            btc_price or 0.0,
        )

        if self.entry_side is not None:
            return

        if elapsed_sec < MIN_ENTRY or elapsed_sec > MAX_ENTRY:
            return

        if elapsed_sec - self._last_predict_t < PREDICT_EVERY_SEC:
            return
        self._last_predict_t = elapsed_sec

        prob_up, _ = self.predictor.predict(elapsed_sec)
        if prob_up is None:
            return

        up_spread = (up_ask - (up_bid or 0)) if up_bid else None
        down_spread = (down_ask - (down_bid or 0)) if down_bid else None
        btc_change_from_open = (
            btc_price - self._first_btc if (btc_price and self._first_btc) else None
        )
        predicted_side = "up" if prob_up >= 0.5 else "down"

        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed_sec,
            event_type="prediction",
            up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            up_spread=up_spread, down_spread=down_spread,
            btc_price=btc_price,
            btc_change_from_open=btc_change_from_open,
            prob_up=prob_up,
            predicted_side=predicted_side,
        )

        book = {
            "up_bid": up_bid, "up_ask": up_ask,
            "down_bid": down_bid, "down_ask": down_ask,
            "up_spread": up_spread, "down_spread": down_spread,
            "btc_price": btc_price,
            "btc_change_from_open": btc_change_from_open,
            "prob_up": prob_up,
        }

        if prob_up >= CONF_THRESHOLD:
            self._enter("Up", up_ask, elapsed_sec, prob_up, book)
        elif (1.0 - prob_up) >= CONF_THRESHOLD:
            self._enter("Down", down_ask, elapsed_sec, 1.0 - prob_up, book)

    def _enter(self, side, price, elapsed, confidence, book):
        self.entry_side = side
        self.entry_shares = SHARES
        self.entry_price = float(price)
        self.entry_elapsed_sec = float(elapsed)
        self.entry_confidence = float(confidence)
        self.entry_prob_up = float(book["prob_up"])
        print(f"  [V3] {self.slug} ENTRY {side} x{SHARES} @ ${price:.3f}  "
              f"t={elapsed:.1f}s  conf={confidence:.3f}")
        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed,
            event_type="entry",
            action="buy",
            side=side.lower(),
            shares=SHARES,
            price=price,
            up_bid=book["up_bid"], up_ask=book["up_ask"],
            down_bid=book["down_bid"], down_ask=book["down_ask"],
            up_spread=book["up_spread"], down_spread=book["down_spread"],
            btc_price=book["btc_price"],
            btc_change_from_open=book["btc_change_from_open"],
            prob_up=book["prob_up"],
            predicted_side=side.lower(),
        )

    def settle(self, btc_open=None, btc_close=None, winner=None):
        winner_norm = (winner or "unknown").capitalize() if winner else "unknown"

        if self.entry_side is None:
            print(f"  [V3] {self.slug} SKIPPED  winner={winner_norm}")
            self.logger.log_window_settlement(
                slug=self.slug,
                btc_open=btc_open, btc_close=btc_close,
                winner=winner_norm.lower(),
                entry_made=False,
                pnl=0.0,
            )
            return

        correct = (self.entry_side == winner_norm)
        pnl = (SHARES * (1.0 - self.entry_price)) if correct else (-SHARES * self.entry_price)
        print(f"  [V3] {self.slug} SETTLE  side={self.entry_side}  winner={winner_norm}  "
              f"{'WIN' if correct else 'LOSS'}  pnl=${pnl:+.2f}")
        self.logger.log_window_settlement(
            slug=self.slug,
            btc_open=btc_open, btc_close=btc_close,
            winner=winner_norm.lower(),
            entry_made=True,
            entry_elapsed_sec=self.entry_elapsed_sec,
            entry_side=self.entry_side.lower(),
            entry_price=self.entry_price,
            entry_shares=self.entry_shares,
            entry_prob_up=self.entry_prob_up,
            entry_confidence=self.entry_confidence,
            correct=correct,
            pnl=round(pnl, 4),
        )
