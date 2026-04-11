"""
Supabase decision logger for paper trading.

Reads SUPABASE_URL and SUPABASE_KEY from environment variables.
Provides simple methods to insert rows into the `windows` and `events`
tables created by sql/001_create_tables.sql.

All inserts are wrapped in try/except — a Supabase outage should NEVER
take down the collector. The strategy keeps running, only logging stops.
"""

import json
import os
import time
import threading
from queue import Queue, Empty
from datetime import datetime, timezone

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None  # type: ignore


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _now_iso():
    return datetime.now(tz=timezone.utc).isoformat()


def _clean_row(row: dict) -> dict:
    """Drop None values so Supabase doesn't overwrite defaults with nulls
    on partial inserts. Convert numpy/pandas types to plain Python."""
    cleaned = {}
    for k, v in row.items():
        if v is None:
            continue
        # Convert numpy scalars
        if hasattr(v, "item"):
            try:
                v = v.item()
            except Exception:
                pass
        cleaned[k] = v
    return cleaned


# ----------------------------------------------------------------------
# Decision logger
# ----------------------------------------------------------------------

class DecisionLogger:
    """Logs paper trading decisions to Supabase.

    Operations:
      - log_window_start(slug, open_epoch, close_epoch)
      - log_event(...)             — single decision/prediction event
      - log_window_settlement(...) — final outcome at window close

    All writes happen on a background worker thread so the trading loop
    is never blocked by network latency.
    """

    def __init__(self, url=None, key=None):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self.enabled = False
        self.client: Client = None
        self._queue: Queue = Queue()
        self._stop = threading.Event()
        self._worker: threading.Thread = None

        if not SUPABASE_AVAILABLE:
            print("  [LOG] supabase package not installed — logging disabled")
            return
        if not self.url or not self.key:
            print("  [LOG] SUPABASE_URL or SUPABASE_KEY not set — "
                  "logging disabled")
            return

        try:
            self.client = create_client(self.url, self.key)
            self.enabled = True
            self._start_worker()
            print(f"  [LOG] Connected to Supabase ({self.url[:32]}...)")
        except Exception as e:
            print(f"  [LOG] Failed to connect to Supabase: {e}")
            self.enabled = False

    # ------------------------------------------------------------------
    # Public API — strategy calls these
    # ------------------------------------------------------------------

    def log_window_start(self, slug, open_epoch, close_epoch, btc_open=None):
        """Insert (or upsert) a row into `windows` for a new market."""
        if not self.enabled:
            return
        row = {
            "slug": slug,
            "open_epoch": int(open_epoch),
            "close_epoch": int(close_epoch),
            "btc_open": btc_open,
            "recorded_at": _now_iso(),
        }
        self._enqueue("upsert", "windows", _clean_row(row))

    def log_window_settlement(self, slug, **fields):
        """Update the existing windows row with settlement results.

        Common fields: btc_close, winner, entry_made, entry_elapsed_sec,
        entry_side, entry_ask, entry_shares, entry_confidence,
        entry_ml_prob, action_type, hedge_made, hedge_elapsed_sec,
        hedge_tier, hedge_combined_cost, hedge_opp_ask, hedge_confidence,
        stopped_out, stop_loss_elapsed_sec, stop_loss_price,
        emergency_hedge_made, emergency_hedge_elapsed_sec,
        emergency_hedge_reason, correct, pnl
        """
        if not self.enabled:
            return
        row = {"slug": slug, **fields}
        self._enqueue("update_windows", "windows", _clean_row(row))

    def log_event(self, slug, elapsed_sec, event_type, **fields):
        """Insert a row into `events`.

        Required: slug, elapsed_sec, event_type
        Optional fields are passed through as columns. Unknown keys go
        into the details_json blob.
        """
        if not self.enabled:
            return

        # Known column whitelist for the events table
        known_cols = {
            "up_bid", "up_ask", "down_bid", "down_ask",
            "up_spread", "down_spread", "btc_price", "btc_change_from_open",
            "ml_prob_up", "ml_model_t", "ensemble_confidence",
            "predicted_side", "market_leader_signal", "btc_direction_signal",
            "btc_market_agree", "ask_strength",
            "action", "side", "shares", "price", "reason",
            "spread_value", "spread_passed", "reversal_count",
            "reversal_passed", "confirm_elapsed", "hedgeable",
            "hedge_tier", "combined_cost", "opp_ask", "guaranteed_profit",
        }

        row = {
            "slug": slug,
            "elapsed_sec": float(elapsed_sec),
            "event_type": event_type,
            "recorded_at": _now_iso(),
        }
        extras = {}
        for k, v in fields.items():
            if k in known_cols:
                row[k] = v
            else:
                extras[k] = v
        if extras:
            row["details_json"] = json.dumps(_clean_row(extras), default=str)

        self._enqueue("insert", "events", _clean_row(row))

    def shutdown(self, timeout=10.0):
        """Drain the queue and stop the worker. Call before exit."""
        if not self.enabled:
            return
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _start_worker(self):
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="supabase-logger"
        )
        self._worker.start()

    def _enqueue(self, op, table, row):
        try:
            self._queue.put_nowait((op, table, row))
        except Exception:
            pass  # never block the strategy

    def _worker_loop(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                op, table, row = self._queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                if op == "insert":
                    self.client.table(table).insert(row).execute()
                elif op == "upsert":
                    self.client.table(table).upsert(row).execute()
                elif op == "update_windows":
                    slug = row.pop("slug")
                    self.client.table(table).update(row).eq(
                        "slug", slug
                    ).execute()
            except Exception as e:
                # Log but never crash. Rate-limit prints.
                msg = str(e)[:200]
                print(f"  [LOG] {op} {table} failed: {msg}")
                # brief backoff before next item
                time.sleep(0.5)
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass
