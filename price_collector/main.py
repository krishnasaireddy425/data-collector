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

# Paper trading (1:1 mirror of prod_ml_strategy) — runs alongside data collection.
# Imports are best-effort: if ml/ deps are missing, the collector still works.
try:
    from ml.prediction_engine import PredictionEngine
    from ml.paper_strategy import PaperStrategy
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

# Globals: one logger + one prediction engine per process (shared across windows).
# These are created lazily in main() if PAPER_TRADING_AVAILABLE.
DECISION_LOGGER = None


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
        self.csv_writer.writerow([
            "timestamp", "elapsed_sec",
            "up_bid", "up_ask", "down_bid", "down_ask",
            "up_spread", "down_spread",
            "btc_price",
        ])
        print(f"  CSV opened: {self.csv_path.name}")

    def write_tick(self, event_type, state, btc_state=None):
        now = time.time()
        elapsed = round(now - self.open_epoch, 3)

        # Skip ticks outside 0-300s window
        if elapsed < 0.5 or elapsed > 301.0:
            return

        # Rate limit: max 3 writes per second
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
            btc.get("price"),
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
    Reads first and last btc_price from the CSV — these are the exact
    oracle prices Polymarket uses for settlement."""
    try:
        import csv as csv_mod
        with open(market.csv_path, "r") as f:
            reader = csv_mod.reader(f)
            header = next(reader)
            btc_col = header.index("btc_price")

            first_price = None
            last_price = None
            for row in reader:
                # Skip empty rows and comment rows
                if not row or row[0].startswith("#"):
                    continue
                try:
                    price = float(row[btc_col])
                    if price > 0:
                        if first_price is None:
                            first_price = price
                        last_price = price
                except (ValueError, IndexError):
                    continue

        if first_price is None or last_price is None:
            print(f"  No BTC prices in CSV for {market.slug}")
            return "unknown"

        print(f"  Oracle: open=${first_price:,.2f} close=${last_price:,.2f}")

        if last_price > first_price:
            return "Up"
        elif last_price < first_price:
            return "Down"
        else:
            return "Flat"
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


async def chainlink_btc_listener(btc_state, stop_event):
    """Stream BTC/USD from Polymarket's Chainlink RTDS WebSocket.
    This is the exact oracle price Polymarket resolves 5-min markets against."""
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

    # Start Chainlink BTC/USD oracle feed in background
    cc_stop = asyncio.Event()
    cc_task = asyncio.create_task(chainlink_btc_listener(btc_state, cc_stop))

    # ---------- Paper trading: one PaperStrategy instance per window ----------
    # Each window gets its own PredictionEngine + PaperStrategy. They share the
    # global DECISION_LOGGER which writes to Supabase on a background thread.
    paper = None
    btc_open_first = None
    if PAPER_TRADING_AVAILABLE and DECISION_LOGGER is not None:
        try:
            engine = PredictionEngine()
            paper = PaperStrategy(
                slug=market.slug,
                open_epoch=market.open_epoch,
                close_epoch=market.close_epoch,
                predictor=engine,
                logger=DECISION_LOGGER,
            )
            print(f"  [PAPER] Strategy initialized for {market.slug}")
        except Exception as e:
            print(f"  [PAPER] Failed to init strategy for {market.slug}: {e}")
            paper = None

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

                        # Feed the same tick into the paper strategy
                        if paper is not None:
                            elapsed = time.time() - market.open_epoch
                            if 0 <= elapsed <= 300:
                                btc_now = btc_state.get("price")
                                if btc_open_first is None and btc_now:
                                    btc_open_first = btc_now
                                try:
                                    paper.on_tick(
                                        elapsed_sec=elapsed,
                                        up_bid=state["up_bid"],
                                        up_ask=state["up_ask"],
                                        down_bid=state["down_bid"],
                                        down_ask=state["down_ask"],
                                        btc_price=btc_now,
                                    )
                                except Exception as pe:
                                    print(f"  [PAPER] on_tick error: {pe}")

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

        # ---------- Paper trading settlement ----------
        if paper is not None:
            try:
                # Determine winner from the recorded CSV (Chainlink oracle data)
                winner_str = get_winner_from_csv(market)
                winner_norm = (winner_str or "unknown").lower()
                btc_close = btc_state.get("price") or btc_open_first
                paper.settle(
                    btc_open=btc_open_first,
                    btc_close=btc_close,
                    winner=winner_norm,
                )
            except Exception as se:
                print(f"  [PAPER] Settlement error for {market.slug}: {se}")


async def main():
    parser = argparse.ArgumentParser(description="BTC 5m Price Tick Collector")
    parser.add_argument("--windows", type=int, default=3,
                        help="Number of 5-minute windows to record (default: 3)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize the global decision logger (Supabase) for paper trading.
    # If SUPABASE_URL/SUPABASE_KEY are not set, logging is silently disabled.
    global DECISION_LOGGER
    if PAPER_TRADING_AVAILABLE:
        try:
            DECISION_LOGGER = DecisionLogger()
        except Exception as e:
            print(f"  [PAPER] Could not init DecisionLogger: {e}")
            DECISION_LOGGER = None

    print("=" * 50)
    print("BTC 5m Price Tick Collector (Chainlink Oracle)")
    print("=" * 50)
    now_dt = datetime.now(timezone.utc)
    print(f"Start time: {now_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Windows to record: {args.windows}")
    print(f"Paper trading: {'ENABLED' if (DECISION_LOGGER and DECISION_LOGGER.enabled) else 'disabled'}")

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

    # Drain the decision logger queue before exiting so all events get pushed.
    if DECISION_LOGGER is not None and DECISION_LOGGER.enabled:
        print("\nFlushing paper trading decisions to Supabase...")
        DECISION_LOGGER.shutdown(timeout=20.0)


if __name__ == "__main__":
    asyncio.run(main())
