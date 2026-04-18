# Project Context: Polymarket BTC 5-min Trading System

## What this project does

We build and run algorithmic trading strategies against Polymarket's 5-minute UP/DOWN crypto prediction markets. The repo has three parts:

1. **Data collection** — `price_collector/main.py` runs in GitHub Actions (self-chaining, ~5h per run), subscribes to Polymarket WebSockets and Chainlink oracle, and records one CSV per 5-minute window.
2. **ML training + backtesting** — `ml/` directory. Models predict UP/DOWN outcome from market state + BTC price features.
3. **Paper trading** — 5 strategy versions (V1–V5) run concurrently on live data inside the collector, logging decisions to Supabase.

Related repos:
- `../data-collector-2/` — same thing for DOGE/HYPE/ETH/BNB (multi-market)
- `../Algos/dashboard/` — React + Flask dashboard analyzing account activity

## Polymarket fundamentals (verify from docs if unsure)

**Docs:** https://docs.polymarket.com

**Binary prediction markets on Polygon.** Users trade YES/NO tokens (or UP/DOWN for crypto markets). Each token pays $1 USDC if its side wins, $0 if it loses. Tokens trade on a CLOB (central limit order book).

**Conditional Token Framework (CTF):**
- **SPLIT**: deposit $1 USDC → mint 1 YES + 1 NO token
- **MERGE**: burn 1 YES + 1 NO → receive $1 USDC (inverse of SPLIT)
- **REDEEM** (after settlement): winning tokens → $1 each, losing tokens → $0
- **CONVERSION** (neg-risk markets only): convert NO shares across mutually-exclusive outcomes

**Activity types from Polymarket's `/activity` API:**
`TRADE, SPLIT, MERGE, REDEEM, REWARD, CONVERSION, MAKER_REBATE`

For a REDEEM record:
- `size` = number of winning shares redeemed
- `usdcSize` = USDC received (equals size because $1/share)
- `price` = 0, `outcome` = empty (REDEEM doesn't identify the winning side)

**5-minute UP/DOWN markets specifically:**
- Market slug format: `{asset}-updown-5m-{unix_epoch}` (e.g., `btc-updown-5m-1776247500`)
- Opens ~10 min before settlement, closes at `open_epoch + 300`
- Settles against a Chainlink oracle price at the boundary
- Rule: close_price ≥ open_price → UP wins (ties resolve UP, per Polymarket's `>=` rule)
- Oracle: Polymarket's Chainlink RTDS feed, topic `crypto_prices_chainlink`
  - WebSocket: `wss://ws-live-data.polymarket.com`
  - BTC ticks every 1-2s (very active). Altcoins (HYPE/DOGE/BNB) tick every 4-10s

## The collector architecture

**Key file:** `price_collector/main.py`

Flow per window:
1. Discover upcoming 5m markets via Gamma API (`gamma-api.polymarket.com/markets?slug=...`)
2. 10s pre-open: seed order book via REST (`/book?token_id=...`)
3. At open: subscribe to CLOB WebSocket (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) for the UP and DOWN tokens
4. Concurrent Chainlink listener streams oracle price ticks
5. Rate limit: 3 writes/sec (0.333s) to CSV
6. At close: read CSV oracle ticks, determine winner by closest timestamps to open/close boundaries

**Parallel scheduling:** All windows in a run use `asyncio.gather`. Each window runs its own 10s pre-open buffer independently (BTC repo already does this; data-collector-2 was recently fixed to match).

**CSV columns (BTC, flat top-of-book):**
```
timestamp, elapsed_sec, up_bid, up_ask, down_bid, down_ask,
up_spread, down_spread, btc_price, btc_oracle_ts
```

**CSV columns (data-collector-2, BOOK_DEPTH=2 with sizes):**
```
timestamp, elapsed_sec,
up_bid_p1, up_bid_s1, up_bid_p2, up_bid_s2,
up_ask_p1, up_ask_s1, up_ask_p2, up_ask_s2,
down_bid_p1, down_bid_s1, down_bid_p2, down_bid_s2,
down_ask_p1, down_ask_s1, down_ask_p2, down_ask_s2,
up_spread, down_spread, {coin}_price, {coin}_oracle_ts
```

## Paper trading versions (V1-V5)

Each runs simultaneously on the same tick stream, logs to its own Supabase tables:

| Ver | Strategy | Model | Live status |
|---|---|---|---|
| V1 | RF 7-snapshot + full strategy (baseline) | `rf_model_t{60..240}.joblib` | Lost money (-$163 historically) |
| V2 | XGB 7-snapshot + full strategy | `xgb_model_t{60..240}.joblib` | Lost money (-$161) |
| V3 | **44-feature continuous XGB (probability model)** | `xgb_sec.joblib` + `calibrator.joblib` | **THE LIVE WINNER** — 89.7% live matches 89.8% backtest at conf≥0.90 |
| V4 | EV regression (EV_up + EV_down in $) | `ev_up.joblib` + `ev_down.joblib` | Stabilized after ~100 samples, tracks backtest |
| V5 | Cheap-side mean-reversion rule (no ML) | — | STRUCTURALLY BROKEN — 31% live vs 35% backtest. Don't trust it. |

**V3 training script:** `ml/train_sec.py`. Features: `ml/features_sec.py` (44 features: core book state, BTC momentum at 3s-180s horizons, rolling volatility, book imbalance, Kaufman ER, rolling correlations, explicit interactions).

**V3 operating threshold (important!):** Backtest on 267 held-out windows showed:
- At conf≥0.90: 89.94% accuracy but **-$11.40 PnL** (entering at $0.91 avg, too expensive)
- At conf≥0.75: 80.56% accuracy, **+$47.24 PnL** (sweet spot — avg entry $0.77 at t=120s)
- **Switch from P≥0.90 to P≥0.75** based on backtest.

## Known patterns (see `actionable_patterns.md`)

We discovered 1,859 unique high-accuracy patterns across 967 BTC windows. Top rules:
- §7.1 gap≥0.70 at t=80 — 80 matches at 98% accuracy
- §5.1 BTC Kaufman ER at t=262 — 56 matches
- §3.1 rising 15s at t=238 — 51 matches

Cascade union of top 13 canonical rules = 282/967 = 29.2% coverage at 98.2% accuracy. Feeds into V6 cascade strategy (not yet deployed).

**Contrarian patterns** (finding losing sides to take the cheap winner):
- Max achievable ~66% accuracy
- Best: `hit≥0.63 now≤0.50 at t=46` → 62.7% at $0.43 entry ≈ $60/day PnL
- WARNING: cheap-side strategies structurally fail in live (V5 lesson)

## Dashboard project context (`../Algos/dashboard/`)

Analyzes Polymarket account activity. Key insight about the `/activity` API:

**The API returns incomplete trade data for high-volume wallets.** Some fills don't appear in the response. Validation: in a binary market with no merges/sells, `redeemed_shares` MUST equal one outcome's `net_bought_shares` (CTF invariant). If neither side matches, data is partial.

For wallet `0xb0f85baa97990910a3e8ac2b4a58a322f01ecef5`:
- Raw API gave $55,821 PnL for 7 days
- After strict per-outcome filter: ~$7,377 (real number)
- Polymarket's own dashboard showed $53,720 for the 30-day month — consistent with ~$2K/day average

## Key user preferences (learned from interaction)

1. **Be honest about tradeoffs.** Don't oversell. If a strategy might lose money, say so before shipping it.
2. **Never make code changes without explicit user permission.** Propose, wait for approval, then change.
3. **NEVER place real orders on Polymarket.** No trades, no test orders, no balance checks that spend USDC.
4. **Backtest numbers aren't live reality.** V5 had +$70/day backtest → -$37/day live. Cheap-side strategies in particular do NOT survive live. Verify before deploying.
5. **The BTC data ceiling at 10 shares/trade is ~$60-80/day.** To scale beyond, either increase share size or add new markets (why data-collector-2 was built).
6. **Use TaskCreate/TaskUpdate for multi-step work.** Keep the user informed of progress.

## When in doubt, verify

This space moves fast. Before trusting anything in this doc:
- Polymarket API semantics → WebFetch `https://docs.polymarket.com/developers/misc-endpoints/data-api-activity` and the CTF overview
- Current market state → `gamma-api.polymarket.com/markets?slug=...`
- Live oracle → subscribe to `crypto_prices_chainlink` on the Polymarket RTDS WS
- Wallet activity → `data-api.polymarket.com/activity?user={wallet}&limit=...`

If you see something that contradicts this doc, TRUST THE LIVE DATA. Flag the discrepancy and update the doc.

## Starter checklist for any new task

Before touching code:
1. Read the relevant files end-to-end — no shortcuts
2. State what you understand and what you're going to do
3. Wait for user confirmation
4. Implement in small increments; verify each

Before deploying any strategy:
1. Backtest on held-out test set
2. Report accuracy AND PnL (accuracy alone is misleading at high-price entries)
3. Look at PnL across threshold range 0.55 → 0.95 to find the sweet spot
4. Remember: 90% accuracy at $0.91 entry = likely negative PnL. Cheaper entries + moderate accuracy win.
