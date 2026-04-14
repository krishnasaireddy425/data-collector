# Decision Log — Changes and Why

This document records every significant change made to the data-collector
system, the reasoning behind it, and the evidence that motivated it.
Written so we can review past decisions without guessing why they were made.

---

## 2026-04-14: V3/V4/V5 Rewrite — Continuous Prediction + EV Model + Cheap-Side Rule

### Summary
The old V3 (regime filter on RF) and V4 (late+raw ML) were built on top of
the flawed 7-snapshot training. Rewrote V3/V4 around proper continuous
prediction + a new EV regression framework, and added V5 as a pure
rule-based cheap-side strategy (no ML). All three coexist with V1/V2
baselines for a 5-way A/B test.

### The big design shift
V1/V2 train 7 separate models (one per timepoint: 60, 90, 120, 150, 180,
210, 240s) and query the nearest one at live time. This throws away ~99%
of every window's tick data. Real quant firms train ONE model on every
tick as a sample with `elapsed_sec` as a feature — "continuous prediction."

### Files added
| File | Purpose |
|------|---------|
| `ml/features_sec.py` | 44-feature function shared by train + predict. Leak-safe (only `shift()` / `rolling()`) |
| `ml/train_sec.py` | Trains one continuous XGB probability model on ~240K samples (797 windows × 300s). Sample-weighted by elapsed_sec. Fits isotonic calibrator on inner-val |
| `ml/predictor_sec.py` | Live probability predictor. Loads `xgb_sec.joblib` + `calibrator.joblib` |
| `ml/backtest_sec.py` | Tick-by-tick replay — reads held-out CSVs in event order, predicts at every integer second. Same code path as live |
| `ml/paper_replay.py` | Shared replay utilities |
| `ml/ev_train.py` | Trains two XGBoost **regressors** (EV_Up, EV_Down) predicting dollars/share PnL of buying each side |
| `ml/ev_predictor.py` | Live EV predictor. Returns `(ev_up, ev_down)` per tick |
| `ml/ev_backtest.py` / `ev_backtest_fast.py` | EV backtest with threshold sweep + H1/H2 stability check |
| `ml/analyze_cheap_side.py` | Offline cheap-side hypothesis test. Numbers inflated by lookahead bug — caught + documented |
| `ml/cheap_side_replay.py` | Honest tick-by-tick replay of 7 cheap-side rule variants |
| `ml/paper_strategy_v5.py` | Rule-based cheap-side V5 strategy (no ML) |
| `sql/003_recreate_v3_v4_clean.sql` | Drops old V3/V4 tables (12 unused hedge/stop-loss cols) and recreates with fields each strategy actually uses |
| `sql/004_create_v5_clean.sql` | v5_windows + v5_events schema tailored to the cheap-side rule |

### Files modified
| File | What changed | Why |
|------|-------------|-----|
| `ml/paper_strategy_v3.py` | Rewrote: uses `PredictorSec`, enters on calibrated `P(side) >= 0.90`, no hedging | Old V3 was regime-filtered RF; new V3 is continuous XGB probability |
| `ml/paper_strategy_v4.py` | Rewrote: uses `EVPredictor`, enters when `max(EV_up, EV_down) > $0.025`, no hedging | Old V4 was late-entry+raw-ML; new V4 is EV regression |
| `ml/decision_logger.py` | Added `events_cols=` parameter for per-version column whitelists; `details_json` only emitted where the target table has that column | Clean V3/V4/V5 schemas don't have `details_json` |
| `price_collector/main.py` | Registers 5 versions instead of 4; imports `PredictorSec`, `EVPredictor`, `PaperStrategyV5` | Full 5-way A/B test |

### The 5 versions

**V1 (baseline): RF 7-snapshot + full strategy (hedging, stop-loss).** Unchanged. 4-week production baseline.  → `windows`, `events`

**V2 (baseline): XGB 7-snapshot + full strategy.** Unchanged. XGB twin of V1.  → `v2_windows`, `v2_events`

**V3 (new): 44-feature continuous XGB probability.** Sample-weighted + isotonic-calibrated. Enters first tick calibrated `P(side) >= 0.90` in [60, 240]s. Hold to settlement. Best honest backtest: **+$28.65/day** at conf=0.90 (170-test).  → `v3_windows`, `v3_events`

**V4 (new): Expected-Value regression.** Two XGBoost regressors predict dollars/share PnL for each side. Enters first tick `max(EV_up, EV_down) > $0.025` in [30, 240]s. Hold to settlement. Best honest backtest: **+$117.20/day** at threshold 0.025 (288-test, stable across halves). Known risk: 93% Up-biased in training regime — untested on bearish days.  → `v4_windows`, `v4_events`

**V5 (new): pure rule-based cheap-side — NO ML.** At each tick: track local-min of each side; buy 10 shares of cheap side when `0.20 <= cheap_price <= 0.40` AND `cheap_price >= local_min + 0.02` AND `30 <= t <= 240`. Best honest backtest: **+$70.10/day** (288-test, stable). Direction-neutral (46-47% Up entries) — works in any regime.  → `v5_windows`, `v5_events`

### Key empirical findings

1. **Continuous prediction closes the backtest-vs-live gap.** Old 7-snapshot: 89.5% "backtest" → 71.5% live (18-point gap). New tick-replay: 70.1% at conf≥0.75 — matches live.

2. **Aggressive hyperparameter tuning hurts at this data size.** Optuna (40 trials) improved inner-val 0.6% but test PnL dropped 2-5% at important thresholds. Classic val-set memorization. Using XGBoost defaults.

3. **EV framework beats probability prediction 4x.** At 10 shares/288 events: EV@0.025 = +$117/day vs probability@0.90 = +$28.65/day. EV directly optimizes dollars, no hand-tuned threshold indirection.

4. **Market over-reaction is real but small.** Empirical: at $0.30-$0.40 cheap-side entries, win rate is 38% vs implied 35%. That's real +3% mispricing, worth ~$70/day at 10 shares.

5. **BTC 5-min direction on public data caps around 70-75%.** Multiple architectures (RF, XGB 7-snapshot, XGB continuous) hit the same ceiling on honest tests. This is the market's own accuracy on the same public data. Breaking through requires different information (Binance tick, on-chain, cross-market).

### Compare versions after ~500 live windows

```sql
SELECT 'v1' AS v, COUNT(*) FILTER (WHERE entry_made) AS entries,
       SUM(pnl) FILTER (WHERE entry_made) AS pnl,
       ROUND(100.0*AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) FILTER (WHERE entry_made), 1) AS acc
FROM windows
UNION ALL SELECT 'v2', COUNT(*) FILTER (WHERE entry_made), SUM(pnl) FILTER (WHERE entry_made),
       ROUND(100.0*AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) FILTER (WHERE entry_made), 1) FROM v2_windows
UNION ALL SELECT 'v3', COUNT(*) FILTER (WHERE entry_made), SUM(pnl) FILTER (WHERE entry_made),
       ROUND(100.0*AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) FILTER (WHERE entry_made), 1) FROM v3_windows
UNION ALL SELECT 'v4', COUNT(*) FILTER (WHERE entry_made), SUM(pnl) FILTER (WHERE entry_made),
       ROUND(100.0*AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) FILTER (WHERE entry_made), 1) FROM v4_windows
UNION ALL SELECT 'v5', COUNT(*) FILTER (WHERE entry_made), SUM(pnl) FILTER (WHERE entry_made),
       ROUND(100.0*AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) FILTER (WHERE entry_made), 1) FROM v5_windows
ORDER BY v;
```

### Deployment steps
1. Run `sql/003_recreate_v3_v4_clean.sql` in Supabase (drops old, creates clean V3/V4).
2. Run `sql/004_create_v5_clean.sql` in Supabase (creates V5 tables).
3. Commit staged changes; next collector run populates all 5 versions.

---

## 2026-04-11: A/B Paper Trading Test (4 Versions)

### What changed
- Added 3 new paper trading strategies running alongside V1 (baseline)
- Each version logs to separate Supabase tables
- Added XGBoost model training alongside RandomForest

### Files added
| File | Purpose |
|------|---------|
| `ml/prediction_engine_xgb.py` | XGBoost variant of PredictionEngine — loads `xgb_model_t*.joblib` |
| `ml/paper_strategy_v2.py` | V2 strategy: uses XGBoost models, everything else identical to V1 |
| `ml/paper_strategy_v3.py` | V3 strategy: adds regime pre-filter (skip low-vol/choppy windows) |
| `ml/paper_strategy_v4.py` | V4 strategy: enters only at t=210+, uses raw ML prob (no ensemble) |
| `ml/models/xgb_model_t{60-240}.joblib` | 7 trained XGBoost models |
| `sql/002_create_version_tables.sql` | Creates v2_windows, v2_events, v3_windows, v3_events, v4_windows, v4_events |

### Files modified
| File | What changed | Why |
|------|-------------|-----|
| `ml/train_model.py` | Trains XGBoost alongside RF; both use walk-forward validation | V2 needs XGBoost models |
| `ml/prediction_engine.py` | Added `_compute_features_safe()` and `_get_ml_prob_raw()` | V3 needs feature access for regime check; V4 needs raw ML prob without ensemble |
| `ml/prediction_engine.py` | Changed `interpolate(method="index", limit_direction="both")` → `ffill().bfill()` | Fixes train/live preprocessing mismatch (see "Interpolation fix" below) |
| `ml/decision_logger.py` | Accepts configurable table names + version label | Each version writes to its own Supabase tables |
| `price_collector/main.py` | Runs 4 paper traders per window instead of 1 | A/B test requires all versions seeing the same data |
| `requirements.txt` | Added `xgboost` | V2 needs it |

### The 4 versions and WHY each exists

**V1 (baseline): Current RF + ensemble**
- No changes. This is the control group.
- Logs to: `windows`, `events` (existing tables)

**V2 (XGBoost): Different algorithm, same everything else**
- Hypothesis: "XGBoost extracts more signal from the same features than RandomForest"
- Evidence: Deep research on 117 OOS windows showed XGBoost beat RF by +3.4% at t=210s and +4.3% at t=180s
- Walk-forward validation on 490 samples confirmed: XGB 83.0% vs RF 80.3% at t=210s (+2.7%)
- What changes: ONLY the model files. Same features, thresholds, timing, ensemble logic.
- Logs to: `v2_windows`, `v2_events`

**V3 (regime-gated): Skip bad windows**
- Hypothesis: "Most losses come from choppy/low-volatility windows. Skipping them improves accuracy."
- Evidence: All 8 losses in the optimal strategy were reversal cases. Loss BTC vol (30s) avg = 4.72 vs win avg = 7.12. Low-vol windows are noise-dominated.
- What changes: Before entry, checks btc_vol_30s, leader_flips_60s, book_imbalance. If window looks bad → skip.
- Thresholds: btc_vol_30s < 3.0 → skip. leader_flips > 4 → skip. |book_imbalance| < 0.08 → skip.
- Same RF model, same timing, same thresholds for everything else.
- Logs to: `v3_windows`, `v3_events`

**V4 (late entry + raw ML): Enter at t=210+, bypass ensemble**
- Hypothesis: "Entering later with raw ML (not ensemble) gives higher accuracy"
- Evidence (timing): Walk-forward accuracy: t=60s=56.4%, t=210s=80.3%. Current paper trader enters at t=75-90s where accuracy is worst.
- Evidence (ensemble): On 117 OOS windows, raw ML beat ensemble by +4.3% at t=210s and +2.5% at t=240s. The ensemble adds noise from market signals that the ML model already incorporates.
- What changes: MIN_ELAPSED raised to 210. Uses ml_prob_up directly instead of going through ensemble weighting.
- Same RF model, same thresholds, same hedge/stop-loss logic.
- Logs to: `v4_windows`, `v4_events`

### How to compare results after 100 windows

```sql
SELECT 'v1' as version, COUNT(*) as entries,
       SUM(CASE WHEN correct THEN 1 ELSE 0 END) as wins,
       ROUND(100.0 * AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END), 1) as accuracy
FROM windows WHERE entry_made = TRUE
UNION ALL
SELECT 'v2', COUNT(*), SUM(CASE WHEN correct THEN 1 ELSE 0 END),
       ROUND(100.0 * AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END), 1)
FROM v2_windows WHERE entry_made = TRUE
UNION ALL
SELECT 'v3', COUNT(*), SUM(CASE WHEN correct THEN 1 ELSE 0 END),
       ROUND(100.0 * AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END), 1)
FROM v3_windows WHERE entry_made = TRUE
UNION ALL
SELECT 'v4', COUNT(*), SUM(CASE WHEN correct THEN 1 ELSE 0 END),
       ROUND(100.0 * AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END), 1)
FROM v4_windows WHERE entry_made = TRUE
ORDER BY version;
```

---

## 2026-04-11: Walk-Forward Validation (replaces LOO)

### What changed
- `ml/train_model.py`: Replaced Leave-One-Out cross-validation with walk-forward (expanding window) validation

### Why
LOO was inflating accuracy by 2-11% because:
1. LOO assumes samples are i.i.d. — our time-ordered windows are NOT independent
2. LOO trains on future data when predicting past windows (temporal leakage)
3. Adjacent windows share regime, volatility, and session characteristics

Empirical proof: trained on 314 samples, tested on 117 truly unseen windows.
LOO claimed 76.4% at t=180s. Real accuracy was 65.0%. That's 11.4% inflation.

### The walk-forward approach
```
Sort all N samples chronologically
Initial training set = 60% of N
Fold size = 50 windows
Purge gap = 2 windows between train and test

Fold 1: Train 1..294, SKIP 295-296, Test 297-346
Fold 2: Train 1..346, SKIP 347-348, Test 349-398
... until end of data
```

Each fold trains ONLY on past data and predicts future data. Honest numbers.

### Walk-forward results on 490 samples (honest numbers)
```
t= 60s: 56.4%   (LOO was claiming ~60%)
t= 90s: 64.9%
t=120s: 70.2%
t=150s: 73.9%
t=180s: 74.5%   (LOO was claiming ~74%, but LOO overstated at 180s by 11% on smaller data)
t=210s: 80.3%   (LOO was close here — 82.6%)
t=240s: 85.1%   (LOO was close — 84.5%)
```

### Validated by external review
Both Gemini and GPT 5.4 confirmed walk-forward is the correct approach for
time-series financial data. GPT also caught the interpolation leak (see below).

---

## 2026-04-11: Interpolation Leak Fix

### What changed
- `ml/train_model.py`: Changed `df_1s.interpolate(method="index", limit_direction="both")` → `df_1s.ffill().bfill()`
- `ml/prediction_engine.py`: Same change (to match training)

### Why
The old `interpolate(method="index", limit_direction="both")` used BIDIRECTIONAL
interpolation. When computing features at time t=180s, the interpolated value
could be influenced by actual ticks at t=182s or later. That's future data
relative to the prediction point.

In the LIVE system, this matters less (there's less future data available in
the buffer). But in TRAINING, the full 0-300s grid was pre-computed, so the
interpolation at t=180 could use data from t=300. This inflated training accuracy.

### The fix
`ffill().bfill()` = forward-fill first, then back-fill for the very start.
At any time t, the value is the LAST KNOWN value from t or before. Never uses
future values. This is the standard "last known price" approach in finance.

### Both training and live now match
Before the fix:
- Training: ffill().bfill() (after the train_model fix)
- Live: interpolate(... both) (old code, had future leakage)
- → MISMATCH between what model learned and what it saw live

After the fix:
- Training: ffill().bfill()
- Live: ffill().bfill()
- → MATCH. Walk-forward accuracy is now an honest estimate of live performance.

### Caught by external review (GPT 5.4)
This was flagged by another model during code review. I missed it initially.

---

## 2026-04-11: Retrain Key Fix

### What changed
- `ml/retrain.py`: Changed `v["accuracy"]` → `v["wf_accuracy"]`

### Why
When train_model.py was updated to walk-forward, the stats key was renamed
from "accuracy" to "wf_accuracy" to reflect the methodology change. But
retrain.py still read the old key name, which would cause a KeyError when
running automated retraining.

---

## 2026-04-10: Chainlink Oracle Integration

### What changed
- Replaced CryptoCompare BTC price feed with Polymarket's Chainlink RTDS WebSocket
- Feed: `wss://ws-live-data.polymarket.com`, topic: `crypto_prices_chainlink`

### Why
CryptoCompare is a DIFFERENT price source than what Polymarket uses for settlement.
The Chainlink Data Streams feed is the EXACT oracle Polymarket settles against.
Using a different price source caused ~5.7% of training labels to be wrong
(17 out of 300 windows had mismatched winners).

### Verified
Ran `scripts/verify_winners.py` against all 300+ windows after switching.
After deleting the 17 mismatched windows: 0 mismatches remaining.

---

## 2026-04-10: Data Collection Fixes

### Fix 1: Chainlink-driven CSV writes
- `price_collector/main.py`: Chainlink listener now directly calls `write_tick()`
- Before: BTC price only written to CSV when a CLOB book event happened
- After: BTC price written when Chainlink ticks (1/sec), in addition to CLOB events
- Why: We were missing BTC price updates between CLOB events, causing stale open/close prices

### Fix 2: btc_oracle_ts column
- Added `btc_oracle_ts` (millisecond Chainlink timestamp) to CSV header
- Why: Allows post-processing to detect stale BTC values and pick the boundary-closest tick

### Fix 3: Extended capture window
- Changed `> 301.0` → `> 303.0` in `write_tick()`
- Why: Captures deviation ticks that arrive 1-2 seconds after the boundary

### Fix 4: Tie → Up
- `get_winner_from_csv()` now returns "Up" when close >= open (not "Flat")
- Why: Polymarket's official rules say ">= resolves to Up". There is no Flat on-chain.

### Fix 5: Boundary-closest winner
- `get_winner_from_csv()` picks the tick whose `btc_oracle_ts` is closest to the boundary
- Why: Previously used first/last row, which could be stale by several seconds

---

## Key Numbers to Remember

| Metric | Value | Source |
|--------|-------|--------|
| Training samples | 490 | Collected 2026-04-10 to 2026-04-11 |
| Walk-forward accuracy at t=210s (RF) | 80.3% | 188 test predictions, 4 folds |
| Walk-forward accuracy at t=210s (XGB) | 83.0% | Same test set |
| Walk-forward accuracy at t=240s (RF) | 85.1% | 188 test predictions |
| Walk-forward accuracy at t=240s (XGB) | 85.6% | Same test set |
| Breakeven accuracy at $0.80 entry | ~80% | Math: 0.80 × win - 0.20 × loss = 0 |
| Breakeven accuracy at $0.85 entry | ~85% | Higher price → need higher accuracy |
| Paper trading V1 accuracy (first 120 windows) | 64.6% | Supabase actuals |
| Paper trading V1 PnL (first 120 windows) | -$37.84 | Supabase actuals |
