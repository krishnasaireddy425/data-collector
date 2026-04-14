"""
PaperStrategyV4 — Expected-Value regression model. Two XGBoost regressors
predict dollars-per-share PnL of buying each side.

Mirrors the backtest that produced +$117.20/day at threshold 0.025 on the
288-window held-out day (stable across halves).

Entry rule:
  At each tick with elapsed_sec in [MIN_ENTRY, MAX_ENTRY]:
    1. Feed tick into EVPredictor
    2. Call predict(elapsed_sec) → (ev_up, ev_down) in dollars/share
    3. Pick best_side = whichever has higher EV
    4. If max(ev_up, ev_down) > EV_THRESHOLD → buy best_side at its ask
  Once entered: HOLD to settlement.

Supabase tables (schema in sql/003_recreate_v3_v4_clean.sql):
  v4_windows, v4_events
"""

from .ev_predictor import EVPredictor


SHARES = 10
MIN_ENTRY = 30
MAX_ENTRY = 240
EV_THRESHOLD = 0.025
PREDICT_EVERY_SEC = 1.0

# Column whitelist for v4_events — matches sql/003_recreate_v3_v4_clean.sql
V4_EVENTS_COLS = {
    "up_bid", "up_ask", "down_bid", "down_ask",
    "up_spread", "down_spread",
    "btc_price", "btc_change_from_open",
    "ev_up", "ev_down", "predicted_side",
    "action", "side", "shares", "price",
}


class PaperStrategyV4:
    def __init__(self, slug, open_epoch, close_epoch, predictor: EVPredictor, logger):
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
        self.entry_ev_up = None
        self.entry_ev_down = None
        self.entry_predicted_ev = None

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

        ev_up, ev_down = self.predictor.predict(elapsed_sec)
        if ev_up is None:
            return

        up_spread = (up_ask - (up_bid or 0)) if up_bid else None
        down_spread = (down_ask - (down_bid or 0)) if down_bid else None
        btc_change_from_open = (
            btc_price - self._first_btc if (btc_price and self._first_btc) else None
        )

        if ev_up >= ev_down:
            best_side, best_ev, best_price = "Up", ev_up, up_ask
        else:
            best_side, best_ev, best_price = "Down", ev_down, down_ask

        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed_sec,
            event_type="prediction",
            up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            up_spread=up_spread, down_spread=down_spread,
            btc_price=btc_price,
            btc_change_from_open=btc_change_from_open,
            ev_up=ev_up, ev_down=ev_down,
            predicted_side=best_side.lower(),
        )

        if best_ev > EV_THRESHOLD:
            book = {
                "up_bid": up_bid, "up_ask": up_ask,
                "down_bid": down_bid, "down_ask": down_ask,
                "up_spread": up_spread, "down_spread": down_spread,
                "btc_price": btc_price,
                "btc_change_from_open": btc_change_from_open,
            }
            self._enter(best_side, best_price, elapsed_sec, best_ev,
                        ev_up, ev_down, book)

    def _enter(self, side, price, elapsed, predicted_ev, ev_up, ev_down, book):
        self.entry_side = side
        self.entry_shares = SHARES
        self.entry_price = float(price)
        self.entry_elapsed_sec = float(elapsed)
        self.entry_ev_up = float(ev_up)
        self.entry_ev_down = float(ev_down)
        self.entry_predicted_ev = float(predicted_ev)
        print(f"  [V4] {self.slug} ENTRY {side} x{SHARES} @ ${price:.3f}  "
              f"t={elapsed:.1f}s  predicted_EV=${predicted_ev:+.4f}  "
              f"(EV_up=${ev_up:+.4f} EV_down=${ev_down:+.4f})")
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
            ev_up=ev_up, ev_down=ev_down,
            predicted_side=side.lower(),
        )

    def settle(self, btc_open=None, btc_close=None, winner=None):
        winner_norm = (winner or "unknown").capitalize() if winner else "unknown"

        if self.entry_side is None:
            print(f"  [V4] {self.slug} SKIPPED  winner={winner_norm}")
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
        print(f"  [V4] {self.slug} SETTLE  side={self.entry_side}  winner={winner_norm}  "
              f"{'WIN' if correct else 'LOSS'}  pnl=${pnl:+.2f}  "
              f"predicted_EV/share=${self.entry_predicted_ev:+.4f}  "
              f"realised=${pnl / SHARES:+.4f}")
        self.logger.log_window_settlement(
            slug=self.slug,
            btc_open=btc_open, btc_close=btc_close,
            winner=winner_norm.lower(),
            entry_made=True,
            entry_elapsed_sec=self.entry_elapsed_sec,
            entry_side=self.entry_side.lower(),
            entry_price=self.entry_price,
            entry_shares=self.entry_shares,
            entry_ev_up=self.entry_ev_up,
            entry_ev_down=self.entry_ev_down,
            entry_predicted_ev=self.entry_predicted_ev,
            entry_threshold=EV_THRESHOLD,
            correct=correct,
            pnl=round(pnl, 4),
        )
