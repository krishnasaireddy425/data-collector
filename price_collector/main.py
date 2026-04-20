"""
BTC 5m Price Tick Collector

Records every price change for BTC updown-5m markets.
One CSV file per market window.

Usage:
    python3 -m price_collector.main --windows 3
"""

import asyncio
import csv
import json
import ssl
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import certifi
import websockets

# Paper trading — 5 versions running simultaneously on the same data.
# V1: old RF 7-snapshot + full strategy (baseline)
# V2: old XGB 7-snapshot + full strategy (baseline)
# V3: new 44f continuous XGB probability model + simple entry
# V4: new EV (expected-value) regression model + simple entry
# V5: pure rule-based cheap-side mean-reversion (no ML)
# V6: V3 clone with CONF_THRESHOLD=0.70 (aggressive variant)
# V7: V5 entry + V6 entry + V5 exits at bid when V6 disagrees
# V8: V5 entry alone + V6 disagreement signal used as exit trigger
# V4+: V4's EV regression + V3 probability floor (skip entries where V3's
#      prob on V4's chosen side < 0.30, data-validated to save ~$50/week)
# Imports are best-effort: if ml/ deps are missing, the collector still works.
try:
    from ml.prediction_engine import PredictionEngine
    from ml.prediction_engine_xgb import PredictionEngineXGB
    from ml.paper_strategy import PaperStrategy
    from ml.paper_strategy_v2 import PaperStrategyV2
    from ml.predictor_sec import PredictorSec
    from ml.ev_predictor import EVPredictor
    from ml.paper_strategy_v3 import PaperStrategyV3
    from ml.paper_strategy_v4 import PaperStrategyV4
    from ml.paper_strategy_v5 import PaperStrategyV5
    from ml.paper_strategy_v6 import PaperStrategyV6
    from ml.paper_strategy_v7 import PaperStrategyV7
    from ml.paper_strategy_v8 import PaperStrategyV8
    from ml.paper_strategy_v4_plus import PaperStrategyV4Plus
    from ml.paper_strategy_v9 import PaperStrategyV9
    from ml.decision_logger import DecisionLogger
    PAPER_TRADING_AVAILABLE = True
except Exception as _e:
    PAPER_TRADING_AVAILABLE = False
    print(f"  [PAPER] Paper trading disabled: {_e}")

DATA_DIR = Path(__file__).parent / "data"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CHAINLINK_WS = "wss://ws-live-data.polymarket.com"
CHAINLINK_PING_INTERVAL = 4  # seconds

# Global loggers — one per strategy version.
# Created lazily in main() if PAPER_TRADING_AVAILABLE.
LOGGERS = {}  # {"v1": DecisionLogger, "v2": ..., "v3": ..., "v4": ..., "v5": ...}


def ssl_ctx():
    return ssl.create_default_context(cafile=certifi.where())


class MarketWindow:
    """Tracks one 5-minute market window."""

    def __init__(self, slug, condition_id, token_id_up, token_id_down, open_epoch):
        self.slug = slug
        self.condition_id = condition_id
        self.token_id_up = token_id_up
        self.token_id_down = token_id_down
        self.open_epoch = open_epoch
        self.close_epoch = open_epoch + 300

        self.csv_path = DATA_DIR / f"{slug}.csv"
        self.csv_file = None
        self.csv_writer = None
        self.tick_count = 0
        self.winner = None
        self._last_write = 0

    def open_csv(self):
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # btc_oracle_ts: the millisecond Chainlink observation timestamp
        # for the btc_price in this row. Used to detect stale rows in
        # post-processing and to pick the boundary tick in get_winner.
        self.csv_writer.writerow([
            "timestamp", "elapsed_sec",
            "up_bid", "up_ask", "down_bid", "down_ask",
            "up_spread", "down_spread",
            "btc_price", "btc_oracle_ts",
        ])
        print(f"  CSV opened: {self.csv_path.name}")

    def write_tick(self, event_type, state, btc_state=None):
        now = time.time()
        elapsed = round(now - self.open_epoch, 3)

        # Capture from t=0.5 to t=303 — 3 extra seconds past the boundary
        # to catch deviation ticks that arrive 1-2 seconds late.
        if elapsed < 0.5 or elapsed > 303.0:
            return

        # Rate limit: max 3 writes per second.
        # Applies to both CLOB book events AND Chainlink ticks (Fix #1
        # routes Chainlink ticks through this same path).
        if now - self._last_write < 0.333:
            return
        self._last_write = now

        up_spread = None
        if state["up_bid"] is not None and state["up_ask"] is not None:
            up_spread = round(state["up_ask"] - state["up_bid"], 4)
        down_spread = None
        if state["down_bid"] is not None and state["down_ask"] is not None:
            down_spread = round(state["down_ask"] - state["down_bid"], 4)

        btc = btc_state or {}
        self.csv_writer.writerow([
            round(now, 3), elapsed,
            state["up_bid"], state["up_ask"],
            state["down_bid"], state["down_ask"],
            up_spread, down_spread,
            btc.get("price"), btc.get("oracle_ts"),
        ])
        self.csv_file.flush()
        self.tick_count += 1

    def close_csv(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    @property
    def seconds_until_open(self):
        return max(0, self.open_epoch - time.time())

    @property
    def seconds_until_close(self):
        return max(0, self.close_epoch - time.time())


async def fetch_market_by_slug(session, slug):
    url = f"{GAMMA_API}/markets"
    params = {"slug": slug}
    try:
        async with session.get(url, params=params,
                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if not data:
                return None
            return data[0] if isinstance(data, list) else data
    except Exception as e:
        print(f"  API error for {slug}: {e}")
        return None


async def fetch_best_bid_ask(session, token_id):
    """Fetch current best bid/ask from REST orderbook."""
    url = f"{CLOB_API}/book"
    params = {"token_id": token_id}
    try:
        async with session.get(url, params=params,
                               timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None, None
            data = await resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            best_bid = max((float(b["price"]) for b in bids), default=None) if bids else None
            best_ask = min((float(a["price"]) for a in asks), default=None) if asks else None
            return best_bid, best_ask
    except Exception:
        return None, None


async def fetch_last_price(session, token_id):
    """Fetch last trade price from REST API."""
    url = f"{CLOB_API}/last-trade-price"
    params = {"token_id": token_id}
    try:
        async with session.get(url, params=params,
                               timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return float(data.get("price", 0)) or None
    except Exception:
        return None


async def discover_next_markets(count):
    """Generate deterministic slugs and query Gamma API for each."""
    ssl_context = ssl_ctx()
    markets = []
    now = time.time()
    interval = 300

    current_boundary = int(now // interval) * interval

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context),
        headers={"User-Agent": "Mozilla/5.0"},
    ) as session:
        for i in range(100):
            epoch = current_boundary + interval * i
            if epoch + 300 < now:
                continue

            slug = f"btc-updown-5m-{epoch}"
            market_data = await fetch_market_by_slug(session, slug)
            if not market_data:
                continue

            cid = market_data.get("conditionId", "")
            if not cid:
                continue

            token_id_up = token_id_down = None
            tokens = market_data.get("tokens", [])
            if tokens and len(tokens) >= 2:
                for tok in tokens:
                    outcome = tok.get("outcome", "")
                    tid = tok.get("token_id", "")
                    if outcome.lower() in ("up", "yes"):
                        token_id_up = tid
                    elif outcome.lower() in ("down", "no"):
                        token_id_down = tid

            if not token_id_up or not token_id_down:
                clob_raw = market_data.get("clobTokenIds", "")
                outcomes_raw = market_data.get("outcomes", "")
                if isinstance(clob_raw, str) and clob_raw:
                    try:
                        clob_list = json.loads(clob_raw)
                        outcomes_list = json.loads(outcomes_raw) if outcomes_raw else []
                        if len(clob_list) >= 2:
                            # Map using outcomes field if available
                            if len(outcomes_list) >= 2:
                                for idx, outcome in enumerate(outcomes_list):
                                    if outcome.lower() in ("up", "yes"):
                                        token_id_up = str(clob_list[idx])
                                    elif outcome.lower() in ("down", "no"):
                                        token_id_down = str(clob_list[idx])
                            else:
                                # Assume index 0=Up, 1=Down
                                token_id_up = str(clob_list[0])
                                token_id_down = str(clob_list[1])
                    except (json.JSONDecodeError, IndexError):
                        pass

            if not token_id_up or not token_id_down:
                continue

            markets.append(MarketWindow(
                slug=slug, condition_id=cid,
                token_id_up=token_id_up, token_id_down=token_id_down,
                open_epoch=epoch,
            ))
            if len(markets) >= count + 2:
                break

    markets.sort(key=lambda m: m.open_epoch)
    return markets[:count + 2]


# --- OLD: get_winner via CryptoCompare klines ---
# async def get_winner(market):
#     """Determine winner using CryptoCompare BTC klines."""
#     ssl_context = ssl_ctx()
#     async with aiohttp.ClientSession(
#         connector=aiohttp.TCPConnector(ssl=ssl_context),
#         headers={"User-Agent": "Mozilla/5.0"},
#     ) as session:
#         try:
#             url = f"{CRYPTOCOMPARE_API}?fsym=BTC&tsym=USD&limit=5&toTs={market.close_epoch}"
#             async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
#                 data = await resp.json()
#             klines = data.get("Data", {}).get("Data", [])
#             if not klines:
#                 return "unknown"
#             open_price = None
#             close_price = None
#             for k in klines:
#                 if k["time"] <= market.open_epoch and (open_price is None or k["time"] >= open_price):
#                     open_price = k["open"]
#                 if k["time"] + 60 >= market.close_epoch:
#                     close_price = k["close"]
#                     break
#             if open_price is None:
#                 open_price = klines[0]["open"]
#             if close_price is None:
#                 close_price = klines[-1]["close"]
#             if close_price > open_price:
#                 return "Up"
#             elif close_price < open_price:
#                 return "Down"
#             else:
#                 return "Flat"
#         except Exception as e:
#             print(f"  CryptoCompare failed: {e}")
#             return "unknown"
# --- END OLD ---


def get_winner_from_csv(market):
    """Determine winner from recorded Chainlink oracle prices in CSV.

    Fix #6: pick the row whose btc_oracle_ts is closest to the open and
    close boundary timestamps (instead of just first/last). This avoids
    boundary deviation ticks (price spikes 1-2s after t=300) flipping the
    winner. Falls back to first/last row if btc_oracle_ts column is missing
    (older CSVs).

    Fix #7: ties resolve to 'Up' (matches Polymarket's '>=' rule). There
    is no 'Flat' outcome on-chain.
    """
    try:
        import csv as csv_mod

        open_ms = int(market.open_epoch * 1000)
        close_ms = int(market.close_epoch * 1000)

        with open(market.csv_path, "r") as f:
            reader = csv_mod.reader(f)
            header = next(reader)
            try:
                btc_col = header.index("btc_price")
            except ValueError:
                print(f"  No btc_price column in {market.slug}")
                return "unknown"
            try:
                ts_col = header.index("btc_oracle_ts")
                has_ts = True
            except ValueError:
                ts_col = None
                has_ts = False

            best_open = None   # (distance_ms, price)
            best_close = None  # (distance_ms, price)
            first_price = None
            last_price = None

            for row in reader:
                # Skip empty rows and comment rows
                if not row or row[0].startswith("#"):
                    continue
                try:
                    price = float(row[btc_col])
                except (ValueError, IndexError):
                    continue
                if price <= 0:
                    continue

                # Track first/last for the fallback path
                if first_price is None:
                    first_price = price
                last_price = price

                if has_ts:
                    try:
                        ts = float(row[ts_col])
                    except (ValueError, IndexError):
                        continue
                    if ts <= 0:
                        continue
                    od = abs(ts - open_ms)
                    cd = abs(ts - close_ms)
                    if best_open is None or od < best_open[0]:
                        best_open = (od, price)
                    if best_close is None or cd < best_close[0]:
                        best_close = (cd, price)

        # Prefer the boundary-closest tick if we have oracle timestamps
        if has_ts and best_open is not None and best_close is not None:
            open_price = best_open[1]
            close_price = best_close[1]
            print(
                f"  Oracle: open=${open_price:,.2f} close=${close_price:,.2f} "
                f"(open_dist={best_open[0]:.0f}ms "
                f"close_dist={best_close[0]:.0f}ms)"
            )
        elif first_price is not None and last_price is not None:
            # Fallback for older CSVs without btc_oracle_ts
            open_price = first_price
            close_price = last_price
            print(f"  Oracle: open=${open_price:,.2f} close=${close_price:,.2f} "
                  f"(no oracle_ts, fallback to first/last)")
        else:
            print(f"  No BTC prices in CSV for {market.slug}")
            return "unknown"

        # Fix #7: ties resolve to "Up" (matches Polymarket's >= rule)
        if close_price >= open_price:
            return "Up"
        else:
            return "Down"
    except Exception as e:
        print(f"  CSV winner check failed: {e}")
        return "unknown"


async def seed_initial_state(market, state):
    """Fetch initial prices and orderbook via REST so CSV starts with full data."""
    ssl_context = ssl_ctx()
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context),
    ) as session:
        # Fetch orderbooks for both sides
        up_bid, up_ask = await fetch_best_bid_ask(session, market.token_id_up)
        down_bid, down_ask = await fetch_best_bid_ask(session, market.token_id_down)

        state["up_bid"] = up_bid
        state["up_ask"] = up_ask
        state["down_bid"] = down_bid
        state["down_ask"] = down_ask

        # Derive prices from mid or last trade
        if up_bid and up_ask:
            state["up_price"] = round((up_bid + up_ask) / 2, 4)
        if down_bid and down_ask:
            state["down_price"] = round((down_bid + down_ask) / 2, 4)

        # If we got up price, derive down (and vice versa)
        if state["up_price"] and not state["down_price"]:
            state["down_price"] = round(1.0 - state["up_price"], 4)
        elif state["down_price"] and not state["up_price"]:
            state["up_price"] = round(1.0 - state["down_price"], 4)

    print(f"  Initial state: Up={state['up_price']} ({state['up_bid']}/{state['up_ask']}), "
          f"Down={state['down_price']} ({state['down_bid']}/{state['down_ask']})")


# --- OLD: CryptoCompare REST polling (different price source than Polymarket oracle) ---
# CRYPTOCOMPARE_PRICE = "https://min-api.cryptocompare.com/data/price"
# CRYPTOCOMPARE_API_KEY = "b6dccec93723bc35ca805eec6fd5ce971919ab825ad5e932fc680e2e262ae826"
#
# async def cryptocompare_listener(btc_state, stop_event):
#     """Poll CryptoCompare REST API every 1s for BTC/USD price."""
#     url = f"{CRYPTOCOMPARE_PRICE}?fsym=BTC&tsyms=USD&api_key={CRYPTOCOMPARE_API_KEY}"
#     print("  CryptoCompare BTC feed started")
#     async with aiohttp.ClientSession() as session:
#         while not stop_event.is_set():
#             try:
#                 async with session.get(url, ssl=ssl_ctx()) as resp:
#                     if resp.status == 200:
#                         data = await resp.json()
#                         price = data.get("USD")
#                         if price:
#                             btc_state["price"] = price
#             except Exception as e:
#                 print(f"  CryptoCompare error: {e}")
#             await asyncio.sleep(1)
# --- END OLD ---


async def chainlink_btc_listener(btc_state, stop_event, on_tick=None):
    """Stream BTC/USD from Polymarket's Chainlink RTDS WebSocket.
    This is the exact oracle price Polymarket resolves 5-min markets against.

    on_tick: optional callback called with the new btc price after every
    fresh Chainlink update. Used by record_market() to trigger CSV writes
    on Chainlink ticks (Fix #1) so we don't have to wait for a CLOB book
    event to capture a fresh oracle price.
    """
    ssl_context = ssl_ctx()
    print("  Chainlink BTC oracle feed starting...")

    while not stop_event.is_set():
        try:
            async with websockets.connect(
                CHAINLINK_WS, ssl=ssl_context, ping_interval=None
            ) as ws:
                # Send PING immediately to establish keepalive
                await ws.send("PING")

                # Subscribe with empty filter (server-side filter breaks streaming)
                sub_msg = {
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices_chainlink",
                        "type": "update",
                        "filters": "",
                    }],
                }
                await ws.send(json.dumps(sub_msg))
                print("  Chainlink BTC oracle feed connected")

                last_ping = time.time()

                while not stop_event.is_set():
                    # Send PING every 4 seconds
                    now_t = time.time()
                    if now_t - last_ping >= CHAINLINK_PING_INTERVAL:
                        await ws.send("PING")
                        last_ping = now_t

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue

                    raw_str = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                    stripped = raw_str.strip()
                    if not stripped or stripped.upper() in ("PONG", "PING"):
                        continue

                    try:
                        msg = json.loads(raw_str)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(msg, dict):
                        continue

                    payload = msg.get("payload", {})
                    topic = msg.get("topic", "")

                    if topic != "crypto_prices_chainlink":
                        continue
                    if payload.get("symbol", "") != "btc/usd":
                        continue

                    price = payload.get("value")
                    oracle_ts = payload.get("timestamp")  # milliseconds
                    if price is not None:
                        btc_state["price"] = round(price, 2)
                        if oracle_ts:
                            btc_state["oracle_ts"] = oracle_ts
                        # Fix #1: notify the consumer that we have a fresh
                        # tick so they can write a CSV row immediately
                        # without waiting for a CLOB book event.
                        if on_tick is not None:
                            try:
                                on_tick()
                            except Exception as cb_err:
                                print(f"  on_tick error: {cb_err}")

        except Exception as e:
            if stop_event.is_set():
                break
            print(f"  Chainlink WS error: {e}, reconnecting...")
            await asyncio.sleep(2)


async def record_market(market):
    """Subscribe to WS and record all price ticks for one market window."""
    ssl_context = ssl_ctx()

    # Wait until 10s before market opens
    wait_time = market.seconds_until_open - 10
    if wait_time > 0:
        print(f"  Waiting {wait_time:.0f}s for {market.slug} to open...")
        await asyncio.sleep(wait_time)

    market.open_csv()

    # State — all fields filled from the start via REST seed
    state = {
        "up_price": None, "down_price": None,
        "up_bid": None, "up_ask": None,
        "down_bid": None, "down_ask": None,
    }

    # Shared BTC price state (updated by Chainlink oracle WS)
    btc_state = {"price": None, "oracle_ts": None}

    # Fix #1: when a new Chainlink tick arrives, write a CSV row
    # immediately using whatever the order book state currently holds.
    # This guarantees every Chainlink update is captured (within the
    # 333ms rate limit), instead of being lost when no CLOB event
    # follows it.
    def on_btc_tick():
        if market.csv_writer is None:
            return
        market.write_tick("btc", state, btc_state)

    # Start Chainlink BTC/USD oracle feed in background
    cc_stop = asyncio.Event()
    cc_task = asyncio.create_task(
        chainlink_btc_listener(btc_state, cc_stop, on_tick=on_btc_tick)
    )

    # ---------- Paper trading: 4 strategy versions per window ----------
    # All 4 versions receive the SAME tick data simultaneously.
    # Each version has its own predictor + logger.
    papers = {}  # {"v1": PaperStrategy, "v2": ..., "v3": ..., "v4": ...}
    btc_open_first = None
    if PAPER_TRADING_AVAILABLE and LOGGERS:
        try:
            # V1: current RF + ensemble (baseline)
            if LOGGERS.get("v1") and LOGGERS["v1"].enabled:
                eng_v1 = PredictionEngine()
                papers["v1"] = PaperStrategy(
                    slug=market.slug, open_epoch=market.open_epoch,
                    close_epoch=market.close_epoch,
                    predictor=eng_v1, logger=LOGGERS["v1"],
                )

            # V2: XGBoost + ensemble (same strategy, different model)
            if LOGGERS.get("v2") and LOGGERS["v2"].enabled:
                eng_v2 = PredictionEngineXGB()
                papers["v2"] = PaperStrategyV2(
                    slug=market.slug, open_epoch=market.open_epoch,
                    close_epoch=market.close_epoch,
                    predictor=eng_v2, logger=LOGGERS["v2"],
                )

            # V3: 44f continuous XGB probability model + simple entry
            if LOGGERS.get("v3") and LOGGERS["v3"].enabled:
                try:
                    eng_v3 = PredictorSec()
                    papers["v3"] = PaperStrategyV3(
                        slug=market.slug, open_epoch=market.open_epoch,
                        close_epoch=market.close_epoch,
                        predictor=eng_v3, logger=LOGGERS["v3"],
                    )
                except Exception as e_v3:
                    print(f"  [V3] init failed: {e_v3}")

            # V4: EV regression model + simple entry
            if LOGGERS.get("v4") and LOGGERS["v4"].enabled:
                try:
                    eng_v4 = EVPredictor()
                    papers["v4"] = PaperStrategyV4(
                        slug=market.slug, open_epoch=market.open_epoch,
                        close_epoch=market.close_epoch,
                        predictor=eng_v4, logger=LOGGERS["v4"],
                    )
                except Exception as e_v4:
                    print(f"  [V4] init failed: {e_v4}")

            # V5: cheap-side mean-reversion (pure rule, no ML)
            if LOGGERS.get("v5") and LOGGERS["v5"].enabled:
                try:
                    papers["v5"] = PaperStrategyV5(
                        slug=market.slug, open_epoch=market.open_epoch,
                        close_epoch=market.close_epoch,
                        predictor=None, logger=LOGGERS["v5"],
                    )
                except Exception as e_v5:
                    print(f"  [V5] init failed: {e_v5}")

            # V6: same 44f XGB as V3 but CONF_THRESHOLD=0.70 (aggressive)
            if LOGGERS.get("v6") and LOGGERS["v6"].enabled:
                try:
                    eng_v6 = PredictorSec()
                    papers["v6"] = PaperStrategyV6(
                        slug=market.slug, open_epoch=market.open_epoch,
                        close_epoch=market.close_epoch,
                        predictor=eng_v6, logger=LOGGERS["v6"],
                    )
                except Exception as e_v6:
                    print(f"  [V6] init failed: {e_v6}")

            # V7: V5 entry + V6 entry + V5 exits when V6 disagrees
            if LOGGERS.get("v7") and LOGGERS["v7"].enabled:
                try:
                    eng_v7 = PredictorSec()
                    papers["v7"] = PaperStrategyV7(
                        slug=market.slug, open_epoch=market.open_epoch,
                        close_epoch=market.close_epoch,
                        predictor=eng_v7, logger=LOGGERS["v7"],
                    )
                except Exception as e_v7:
                    print(f"  [V7] init failed: {e_v7}")

            # V8: V5 entry alone + V6 disagree signal used as exit
            if LOGGERS.get("v8") and LOGGERS["v8"].enabled:
                try:
                    eng_v8 = PredictorSec()
                    papers["v8"] = PaperStrategyV8(
                        slug=market.slug, open_epoch=market.open_epoch,
                        close_epoch=market.close_epoch,
                        predictor=eng_v8, logger=LOGGERS["v8"],
                    )
                except Exception as e_v8:
                    print(f"  [V8] init failed: {e_v8}")

            # V4+: V4's EV regression + V3 probability floor (skip <0.30)
            # Needs BOTH EVPredictor (for V4 signal) and PredictorSec (for V3 prob).
            if LOGGERS.get("v4plus") and LOGGERS["v4plus"].enabled:
                try:
                    eng_v4plus_ev = EVPredictor()
                    eng_v4plus_v3 = PredictorSec()
                    papers["v4plus"] = PaperStrategyV4Plus(
                        slug=market.slug, open_epoch=market.open_epoch,
                        close_epoch=market.close_epoch,
                        ev_predictor=eng_v4plus_ev,
                        v3_predictor=eng_v4plus_v3,
                        logger=LOGGERS["v4plus"],
                    )
                except Exception as e_v4plus:
                    print(f"  [V4+] init failed: {e_v4plus}")

            # V9: V4 with stricter EV threshold (0.04). No V3 filter.
            if LOGGERS.get("v9") and LOGGERS["v9"].enabled:
                try:
                    eng_v9 = EVPredictor()
                    papers["v9"] = PaperStrategyV9(
                        slug=market.slug, open_epoch=market.open_epoch,
                        close_epoch=market.close_epoch,
                        predictor=eng_v9, logger=LOGGERS["v9"],
                    )
                except Exception as e_v9:
                    print(f"  [V9] init failed: {e_v9}")

            if papers:
                print(f"  [PAPER] {len(papers)} strategies initialized: "
                      f"{list(papers.keys())}")
        except Exception as e:
            print(f"  [PAPER] Failed to init strategies: {e}")
            papers = {}

    # Seed initial values from REST
    await seed_initial_state(market, state)

    # Wait for window to open before writing first tick
    sleep_until_open = market.open_epoch - time.time()
    if sleep_until_open > 0:
        await asyncio.sleep(sleep_until_open)

    # Write initial snapshot
    market.write_tick("initial", state, btc_state)

    token_map = {
        market.token_id_up: "up",
        market.token_id_down: "down",
    }

    try:
        async with websockets.connect(
            CLOB_WS, ssl=ssl_context, ping_interval=20
        ) as ws:
            # Subscribe to BOTH tokens
            sub_msg = {
                "type": "market",
                "assets_ids": [market.token_id_up, market.token_id_down],
            }
            await ws.send(json.dumps(sub_msg))
            print(f"  WS subscribed for {market.slug}")

            # Record until market closes
            while time.time() < market.close_epoch + 1:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    msgs = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if not isinstance(msgs, list):
                    msgs = [msgs]

                for msg in msgs:
                    event_type = msg.get("event_type", "")
                    asset_id = msg.get("asset_id", "")

                    if asset_id not in token_map:
                        continue

                    side = token_map[asset_id]  # "up" or "down"

                    if event_type == "book":
                        bids = msg.get("bids", [])
                        asks = msg.get("asks", [])
                        if bids:
                            state[f"{side}_bid"] = max(float(b["price"]) for b in bids)
                        if asks:
                            state[f"{side}_ask"] = min(float(a["price"]) for a in asks)
                        market.write_tick("book", state, btc_state)

                        # Feed the same tick into ALL paper strategies
                        if papers:
                            elapsed = time.time() - market.open_epoch
                            if 0 <= elapsed <= 300:
                                btc_now = btc_state.get("price")
                                if btc_open_first is None and btc_now:
                                    btc_open_first = btc_now
                                for ver, strat in papers.items():
                                    try:
                                        strat.on_tick(
                                            elapsed_sec=elapsed,
                                            up_bid=state["up_bid"],
                                            up_ask=state["up_ask"],
                                            down_bid=state["down_bid"],
                                            down_ask=state["down_ask"],
                                            btc_price=btc_now,
                                        )
                                    except Exception as pe:
                                        print(f"  [PAPER-{ver}] on_tick error: {pe}")

    except Exception as e:
        print(f"  WS error for {market.slug}: {e}")
    finally:
        cc_stop.set()
        cc_task.cancel()
        try:
            await cc_task
        except asyncio.CancelledError:
            pass
        market.close_csv()
        print(f"  Recorded {market.tick_count} ticks for {market.slug}")

        # ---------- Paper trading settlement (all versions) ----------
        if papers:
            try:
                winner_str = get_winner_from_csv(market)
                winner_norm = (winner_str or "unknown").lower()
                btc_close = btc_state.get("price") or btc_open_first
                for ver, strat in papers.items():
                    try:
                        strat.settle(
                            btc_open=btc_open_first,
                            btc_close=btc_close,
                            winner=winner_norm,
                        )
                    except Exception as se:
                        print(f"  [PAPER-{ver}] Settlement error: {se}")
            except Exception as e:
                print(f"  [PAPER] Winner determination error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="BTC 5m Price Tick Collector")
    parser.add_argument("--windows", type=int, default=3,
                        help="Number of 5-minute windows to record (default: 3)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize one DecisionLogger per strategy version.
    # Each writes to its own pair of Supabase tables.
    global LOGGERS
    if PAPER_TRADING_AVAILABLE:
        # Each version gets its own events-columns whitelist so DecisionLogger
        # only sends fields that exist in that table. V1/V2 use the legacy
        # schema (default whitelist + details_json); V3/V4 use clean schemas
        # from sql/003_recreate_v3_v4_clean.sql (no details_json).
        from ml.paper_strategy_v3 import V3_EVENTS_COLS
        from ml.paper_strategy_v4 import V4_EVENTS_COLS
        from ml.paper_strategy_v5 import V5_EVENTS_COLS
        from ml.paper_strategy_v6 import V6_EVENTS_COLS
        from ml.paper_strategy_v7 import V7_EVENTS_COLS
        from ml.paper_strategy_v8 import V8_EVENTS_COLS
        from ml.paper_strategy_v4_plus import V4PLUS_EVENTS_COLS
        from ml.paper_strategy_v9 import V9_EVENTS_COLS
        version_configs = [
            ("v1", "windows", "events", None),
            ("v2", "v2_windows", "v2_events", None),
            ("v3", "v3_windows", "v3_events", V3_EVENTS_COLS),
            ("v4", "v4_windows", "v4_events", V4_EVENTS_COLS),
            ("v5", "v5_windows", "v5_events", V5_EVENTS_COLS),
            ("v6", "v6_windows", "v6_events", V6_EVENTS_COLS),
            ("v7", "v7_windows", "v7_events", V7_EVENTS_COLS),
            ("v8", "v8_windows", "v8_events", V8_EVENTS_COLS),
            ("v4plus", "v4plus_windows", "v4plus_events", V4PLUS_EVENTS_COLS),
            ("v9", "v9_windows", "v9_events", V9_EVENTS_COLS),
        ]
        for ver, win_tbl, evt_tbl, cols in version_configs:
            try:
                LOGGERS[ver] = DecisionLogger(
                    windows_table=win_tbl,
                    events_table=evt_tbl,
                    version_label=ver,
                    events_cols=cols,
                )
            except Exception as e:
                print(f"  [PAPER-{ver}] Could not init logger: {e}")
                LOGGERS[ver] = None

    print("=" * 50)
    print("BTC 5m Price Tick Collector (Chainlink Oracle)")
    print("=" * 50)
    now_dt = datetime.now(timezone.utc)
    print(f"Start time: {now_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Windows to record: {args.windows}")
    enabled_versions = [v for v, l in LOGGERS.items() if l and l.enabled]
    print(f"Paper trading: {len(enabled_versions)} versions active: {enabled_versions}")

    print("\nDiscovering upcoming BTC 5m markets...")
    markets = await discover_next_markets(args.windows)
    if not markets:
        print("ERROR: No upcoming markets found!")
        return

    print(f"Found {len(markets)} upcoming markets:")
    for m in markets:
        t = datetime.fromtimestamp(m.open_epoch, tz=timezone.utc).strftime('%H:%M:%S')
        secs = m.seconds_until_open
        status = "LIVE NOW" if secs == 0 else f"opens in {secs:.0f}s"
        print(f"  {m.slug} ({t} UTC, {status})")

    # Pick windows that haven't started yet (or just started)
    now = time.time()
    to_record = [m for m in markets if m.open_epoch > now - 10][:args.windows]
    if not to_record:
        to_record = markets[-args.windows:]

    print(f"\nRecording: {[m.slug for m in to_record]}")

    # Launch all windows concurrently — each manages its own timing
    # Window 2 starts setup while window 1 is still recording
    for i, market in enumerate(to_record):
        open_str = datetime.fromtimestamp(market.open_epoch, tz=timezone.utc).strftime('%H:%M:%S')
        close_str = datetime.fromtimestamp(market.close_epoch, tz=timezone.utc).strftime('%H:%M:%S')
        print(f"  Window {i+1}: {market.slug} ({open_str} → {close_str} UTC)")

    tasks = [asyncio.create_task(record_market(m)) for m in to_record]
    await asyncio.gather(*tasks)

    # Determine winners from recorded Chainlink oracle prices (no API call needed)
    print(f"\nDetermining winners from Chainlink oracle data...")

    for market in to_record:
        winner = get_winner_from_csv(market)
        market.winner = winner
        print(f"  {market.slug} → Winner: {winner}")

        # Append result to CSV
        with open(market.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([])
            w.writerow(["# RESULT", f"winner={winner}", f"slug={market.slug}",
                         f"ticks={market.tick_count}"])

    print("\n" + "=" * 50)
    print("COLLECTION COMPLETE")
    print("=" * 50)
    for m in to_record:
        print(f"  {m.slug}: {m.tick_count} ticks, winner={m.winner}")
        print(f"    {m.csv_path}")
    print("=" * 50)

    # Drain all decision logger queues before exiting.
    for ver, logger in LOGGERS.items():
        if logger is not None and logger.enabled:
            print(f"\nFlushing {ver} decisions to Supabase...")
            logger.shutdown(timeout=15.0)


if __name__ == "__main__":
    asyncio.run(main())
