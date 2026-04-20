"""
PaperStrategyV4Plus — V4's EV regression strategy with one added gate:
skip entry if V3's probability on V4's chosen side is below V3_PROB_FLOOR.

Data-backed rationale (from 7 days of production V4 data):
  - V4 entries where V3 prob on chosen side < 0.30: 39 entries, 10.3%
    accuracy, −$50.50 total PnL. This is V4's worst bleeder.
  - Filtering those out: +$50.50 saved (+24% vs V4 baseline).
  - Everything else (including V3 silent): V4 fires normally.

Everything else is IDENTICAL to V4:
  - Same 44 features, same ev_up.joblib / ev_down.joblib
  - EV_THRESHOLD = 0.025
  - MIN_ENTRY = 30, MAX_ENTRY = 240
  - SHARES = 10
  - Hold to settlement, no exits

One rule added:
  if best_ev > EV_THRESHOLD AND v3_prob_on_side is not None AND v3_prob_on_side < V3_PROB_FLOOR:
      skip (log 'blocked_v3_lt30' event)
  else:
      fire as V4 would

Supabase tables (schema in sql/008_create_v4plus_clean.sql):
  v4plus_windows, v4plus_events
"""

from .ev_predictor import EVPredictor
from .predictor_sec import PredictorSec


SHARES = 10
MIN_ENTRY = 30
MAX_ENTRY = 240
EV_THRESHOLD = 0.025           # unchanged from V4
PREDICT_EVERY_SEC = 1.0

# NEW: V3 probability filter
V3_PROB_FLOOR = 0.30           # skip entry if V3 prob on our side < this

# Column whitelist for v4plus_events — matches sql/008_create_v4plus_clean.sql
V4PLUS_EVENTS_COLS = {
    "up_bid", "up_ask", "down_bid", "down_ask",
    "up_spread", "down_spread",
    "btc_price", "btc_change_from_open",
    "ev_up", "ev_down", "predicted_side",
    "v3_prob_up", "v3_prob_on_side",
    "action", "side", "shares", "price",
    "filter_reason",
}


class PaperStrategyV4Plus:
    def __init__(self, slug, open_epoch, close_epoch,
                 ev_predictor: EVPredictor,
                 v3_predictor: PredictorSec,
                 logger):
        self.slug = slug
        self.open_epoch = open_epoch
        self.close_epoch = close_epoch
        self.ev_predictor = ev_predictor
        self.v3_predictor = v3_predictor
        self.logger = logger

        self.ev_predictor.reset()
        self.v3_predictor.reset()

        self.entry_side = None
        self.entry_shares = 0
        self.entry_price = None
        self.entry_elapsed_sec = None
        self.entry_ev_up = None
        self.entry_ev_down = None
        self.entry_predicted_ev = None
        self.entry_v3_prob_up = None
        self.entry_v3_prob_on_side = None

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

        # Feed BOTH predictors every tick (both use 44f continuous features,
        # so each needs its own running buffer). They share nothing — each has
        # its own PredictorSec / EVPredictor instance.
        self.ev_predictor.add_tick(
            elapsed_sec, up_bid or 0.0, up_ask, down_bid or 0.0, down_ask,
            btc_price or 0.0,
        )
        self.v3_predictor.add_tick(
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

        ev_up, ev_down = self.ev_predictor.predict(elapsed_sec)
        if ev_up is None:
            return

        # Also get V3's probability for the filter
        v3_prob_up, _ = self.v3_predictor.predict(elapsed_sec)

        up_spread = (up_ask - (up_bid or 0)) if up_bid else None
        down_spread = (down_ask - (down_bid or 0)) if down_bid else None
        btc_change_from_open = (
            btc_price - self._first_btc if (btc_price and self._first_btc) else None
        )

        if ev_up >= ev_down:
            best_side, best_ev, best_price = "Up", ev_up, up_ask
        else:
            best_side, best_ev, best_price = "Down", ev_down, down_ask

        # V3's probability on OUR entry side
        v3_prob_on_side = None
        if v3_prob_up is not None:
            v3_prob_on_side = float(v3_prob_up) if best_side == "Up" \
                else float(1.0 - v3_prob_up)

        # Log prediction event (every tick, like V4 does)
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
            v3_prob_up=v3_prob_up,
            v3_prob_on_side=v3_prob_on_side,
        )

        # Entry gate 1: EV threshold (same as V4)
        if best_ev <= EV_THRESHOLD:
            return

        # Entry gate 2 (NEW): V3 probability floor
        # If V3 predicted and prob on our side < floor → skip (log reason)
        if v3_prob_on_side is not None and v3_prob_on_side < V3_PROB_FLOOR:
            self.logger.log_event(
                slug=self.slug,
                elapsed_sec=elapsed_sec,
                event_type="blocked_by_v3",
                action="skip",
                side=best_side.lower(),
                price=best_price,
                up_bid=up_bid, up_ask=up_ask,
                down_bid=down_bid, down_ask=down_ask,
                up_spread=up_spread, down_spread=down_spread,
                btc_price=btc_price,
                btc_change_from_open=btc_change_from_open,
                ev_up=ev_up, ev_down=ev_down,
                predicted_side=best_side.lower(),
                v3_prob_up=v3_prob_up,
                v3_prob_on_side=v3_prob_on_side,
                filter_reason=f"v3_prob_lt_{V3_PROB_FLOOR:.2f}",
            )
            print(f"  [V4+] {self.slug} SKIP {best_side} @ ${best_price:.3f}  "
                  f"t={elapsed_sec:.1f}s  EV=${best_ev:+.4f}  "
                  f"v3_prob_on_side={v3_prob_on_side:.3f} < {V3_PROB_FLOOR:.2f}")
            return

        # All gates passed → fire entry (V4's normal entry logic)
        book = {
            "up_bid": up_bid, "up_ask": up_ask,
            "down_bid": down_bid, "down_ask": down_ask,
            "up_spread": up_spread, "down_spread": down_spread,
            "btc_price": btc_price,
            "btc_change_from_open": btc_change_from_open,
            "v3_prob_up": v3_prob_up,
            "v3_prob_on_side": v3_prob_on_side,
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
        self.entry_v3_prob_up = (
            float(book["v3_prob_up"]) if book["v3_prob_up"] is not None else None
        )
        self.entry_v3_prob_on_side = (
            float(book["v3_prob_on_side"])
            if book["v3_prob_on_side"] is not None else None
        )
        v3_tag = (f"v3_prob_on_side={self.entry_v3_prob_on_side:.3f}"
                  if self.entry_v3_prob_on_side is not None else "v3=silent")
        print(f"  [V4+] {self.slug} ENTRY {side} x{SHARES} @ ${price:.3f}  "
              f"t={elapsed:.1f}s  EV=${predicted_ev:+.4f}  {v3_tag}")
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
            v3_prob_up=book["v3_prob_up"],
            v3_prob_on_side=book["v3_prob_on_side"],
            filter_reason="passed",
        )

    def settle(self, btc_open=None, btc_close=None, winner=None):
        winner_norm = (winner or "unknown").capitalize() if winner else "unknown"

        if self.entry_side is None:
            print(f"  [V4+] {self.slug} SKIPPED  winner={winner_norm}")
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
        print(f"  [V4+] {self.slug} SETTLE  side={self.entry_side}  winner={winner_norm}  "
              f"{'WIN' if correct else 'LOSS'}  pnl=${pnl:+.2f}  "
              f"predicted_EV/share=${self.entry_predicted_ev:+.4f}")
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
            entry_v3_prob_up=self.entry_v3_prob_up,
            entry_v3_prob_on_side=self.entry_v3_prob_on_side,
            v3_prob_floor=V3_PROB_FLOOR,
            correct=correct,
            pnl=round(pnl, 4),
        )
