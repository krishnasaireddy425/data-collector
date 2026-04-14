"""
PaperStrategyV5 — pure rule-based cheap-side mean-reversion.

NO ML model. NO predictor class. Just reads up_ask / down_ask from the
WebSocket stream and enters when the cheap side has bounced 2c off its
local minimum while still priced in [0.20, 0.40].

Mirrors the B2 backtest that produced +$70.10/day on the 288-window
held-out test (stable across halves).

Entry rule:
  1. Track running local-min of up_ask and down_ask across the window.
  2. At each tick, identify cheap_side = whichever of (up_ask, down_ask)
     is lower, and cheap_price = that ask.
  3. If ALL of:
        - MIN_ENTRY  <= elapsed_sec <= MAX_ENTRY
        - PRICE_LO   <= cheap_price <= PRICE_HI
        - cheap_price >= local_min[cheap_side] + UPTICK_REQUIRED
        - not already entered
     → BUY SHARES of cheap_side at cheap_price.
  4. Hold to settlement. No exits, no hedging.

Supabase tables (see sql/004_create_v5_clean.sql):
  v5_windows, v5_events
"""


SHARES = 10
MIN_ENTRY = 30
MAX_ENTRY = 240
PRICE_LO = 0.20
PRICE_HI = 0.40
UPTICK_REQUIRED = 0.02      # cents of bounce needed from local min
LOG_PREDICT_EVERY_SEC = 5.0  # don't spam events table

# Column whitelist for v5_events — matches sql/004_create_v5_clean.sql
V5_EVENTS_COLS = {
    "up_bid", "up_ask", "down_bid", "down_ask",
    "up_spread", "down_spread",
    "btc_price", "btc_change_from_open",
    "cheap_side", "cheap_price", "local_min", "uptick_amount",
    "action", "side", "shares", "price",
}


class PaperStrategyV5:
    def __init__(self, slug, open_epoch, close_epoch, predictor=None, logger=None):
        # `predictor` accepted for signature parity with V1-V4 but unused.
        self.slug = slug
        self.open_epoch = open_epoch
        self.close_epoch = close_epoch
        self.logger = logger

        self.entry_side = None
        self.entry_shares = 0
        self.entry_price = None
        self.entry_elapsed_sec = None
        self.entry_local_min = None
        self.entry_uptick_amount = None

        # Running minima per side
        self._min_up_ask = 1.0
        self._min_down_ask = 1.0

        self._window_logged = False
        self._first_btc = None
        self._last_log_t = -999.0

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

        # Update running minima on EVERY tick (even before MIN_ENTRY, to
        # build an accurate history of the cheap side's floor).
        if up_ask < self._min_up_ask:
            self._min_up_ask = up_ask
        if down_ask < self._min_down_ask:
            self._min_down_ask = down_ask

        # Stop processing once we've entered — just hold to settlement.
        if self.entry_side is not None:
            return

        # Identify cheap side and its local-min
        if up_ask <= down_ask:
            cheap_side, cheap_price, local_min = "Up", up_ask, self._min_up_ask
        else:
            cheap_side, cheap_price, local_min = "Down", down_ask, self._min_down_ask

        uptick = cheap_price - local_min

        # Log periodically for visibility (every 5s, not every tick)
        if elapsed_sec - self._last_log_t >= LOG_PREDICT_EVERY_SEC:
            self._last_log_t = elapsed_sec
            up_spread = (up_ask - (up_bid or 0)) if up_bid else None
            down_spread = (down_ask - (down_bid or 0)) if down_bid else None
            btc_change_from_open = (
                btc_price - self._first_btc
                if (btc_price and self._first_btc) else None
            )
            self.logger.log_event(
                slug=self.slug,
                elapsed_sec=elapsed_sec,
                event_type="check",
                up_bid=up_bid, up_ask=up_ask,
                down_bid=down_bid, down_ask=down_ask,
                up_spread=up_spread, down_spread=down_spread,
                btc_price=btc_price,
                btc_change_from_open=btc_change_from_open,
                cheap_side=cheap_side.lower(),
                cheap_price=cheap_price,
                local_min=local_min,
                uptick_amount=uptick,
            )

        # Entry conditions
        if elapsed_sec < MIN_ENTRY or elapsed_sec > MAX_ENTRY:
            return
        if not (PRICE_LO <= cheap_price <= PRICE_HI):
            return
        if uptick < UPTICK_REQUIRED:
            return

        # All conditions met — enter
        up_spread = (up_ask - (up_bid or 0)) if up_bid else None
        down_spread = (down_ask - (down_bid or 0)) if down_bid else None
        btc_change_from_open = (
            btc_price - self._first_btc
            if (btc_price and self._first_btc) else None
        )
        self._enter(
            cheap_side, cheap_price, elapsed_sec, local_min, uptick,
            up_bid, up_ask, down_bid, down_ask,
            up_spread, down_spread, btc_price, btc_change_from_open,
        )

    def _enter(self, side, price, elapsed, local_min, uptick,
               up_bid, up_ask, down_bid, down_ask,
               up_spread, down_spread, btc_price, btc_change_from_open):
        self.entry_side = side
        self.entry_shares = SHARES
        self.entry_price = float(price)
        self.entry_elapsed_sec = float(elapsed)
        self.entry_local_min = float(local_min)
        self.entry_uptick_amount = float(uptick)
        print(f"  [V5] {self.slug} ENTRY {side} x{SHARES} @ ${price:.3f}  "
              f"t={elapsed:.1f}s  local_min=${local_min:.3f}  "
              f"uptick=${uptick:+.3f}")
        self.logger.log_event(
            slug=self.slug,
            elapsed_sec=elapsed,
            event_type="entry",
            action="buy",
            side=side.lower(),
            shares=SHARES,
            price=price,
            up_bid=up_bid, up_ask=up_ask,
            down_bid=down_bid, down_ask=down_ask,
            up_spread=up_spread, down_spread=down_spread,
            btc_price=btc_price,
            btc_change_from_open=btc_change_from_open,
            cheap_side=side.lower(),
            cheap_price=price,
            local_min=local_min,
            uptick_amount=uptick,
        )

    def settle(self, btc_open=None, btc_close=None, winner=None):
        winner_norm = (winner or "unknown").capitalize() if winner else "unknown"

        if self.entry_side is None:
            print(f"  [V5] {self.slug} SKIPPED  winner={winner_norm}")
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
        print(f"  [V5] {self.slug} SETTLE  side={self.entry_side}  winner={winner_norm}  "
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
            entry_local_min=self.entry_local_min,
            entry_uptick_amount=self.entry_uptick_amount,
            correct=correct,
            pnl=round(pnl, 4),
        )
