-- Version tables for A/B paper trading test.
-- V1 uses the existing `windows` and `events` tables (unchanged).
-- V2, V3, V4 each get their own pair of tables with identical schema.
--
-- Run this in Supabase SQL Editor once.

-- ============================================================
-- V2: XGBoost model (same features, different algorithm)
-- ============================================================

CREATE TABLE IF NOT EXISTS v2_windows (
    slug                          TEXT PRIMARY KEY,
    open_epoch                    BIGINT NOT NULL,
    close_epoch                   BIGINT NOT NULL,
    btc_open                      DOUBLE PRECISION,
    btc_close                     DOUBLE PRECISION,
    winner                        TEXT,
    entry_made                    BOOLEAN DEFAULT FALSE,
    entry_elapsed_sec             DOUBLE PRECISION,
    entry_side                    TEXT,
    entry_ask                     DOUBLE PRECISION,
    entry_shares                  INTEGER,
    entry_confidence              DOUBLE PRECISION,
    entry_ml_prob                 DOUBLE PRECISION,
    action_type                   TEXT,
    hedge_made                    BOOLEAN DEFAULT FALSE,
    hedge_elapsed_sec             DOUBLE PRECISION,
    hedge_tier                    INTEGER,
    hedge_combined_cost           DOUBLE PRECISION,
    hedge_opp_ask                 DOUBLE PRECISION,
    hedge_confidence              DOUBLE PRECISION,
    stopped_out                   BOOLEAN DEFAULT FALSE,
    stop_loss_elapsed_sec         DOUBLE PRECISION,
    stop_loss_price               DOUBLE PRECISION,
    emergency_hedge_made          BOOLEAN DEFAULT FALSE,
    emergency_hedge_elapsed_sec   DOUBLE PRECISION,
    emergency_hedge_reason        TEXT,
    correct                       BOOLEAN,
    pnl                           DOUBLE PRECISION,
    recorded_at                   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS v2_events (
    id                BIGSERIAL PRIMARY KEY,
    slug              TEXT NOT NULL REFERENCES v2_windows(slug) ON DELETE CASCADE,
    elapsed_sec       DOUBLE PRECISION NOT NULL,
    event_type        TEXT NOT NULL,
    recorded_at       TIMESTAMPTZ DEFAULT NOW(),
    up_bid            DOUBLE PRECISION,
    up_ask            DOUBLE PRECISION,
    down_bid          DOUBLE PRECISION,
    down_ask          DOUBLE PRECISION,
    up_spread         DOUBLE PRECISION,
    down_spread       DOUBLE PRECISION,
    btc_price         DOUBLE PRECISION,
    btc_change_from_open DOUBLE PRECISION,
    ml_prob_up           DOUBLE PRECISION,
    ml_model_t           INTEGER,
    ensemble_confidence  DOUBLE PRECISION,
    predicted_side       TEXT,
    market_leader_signal TEXT,
    btc_direction_signal TEXT,
    btc_market_agree     BOOLEAN,
    ask_strength         DOUBLE PRECISION,
    action            TEXT,
    side              TEXT,
    shares            INTEGER,
    price             DOUBLE PRECISION,
    reason            TEXT,
    spread_value      DOUBLE PRECISION,
    spread_passed     BOOLEAN,
    reversal_count    INTEGER,
    reversal_passed   BOOLEAN,
    confirm_elapsed   DOUBLE PRECISION,
    hedgeable         BOOLEAN,
    hedge_tier        INTEGER,
    combined_cost     DOUBLE PRECISION,
    opp_ask           DOUBLE PRECISION,
    guaranteed_profit DOUBLE PRECISION,
    details_json      JSONB
);

CREATE INDEX IF NOT EXISTS idx_v2_events_slug ON v2_events(slug);
CREATE INDEX IF NOT EXISTS idx_v2_events_type ON v2_events(event_type);
CREATE INDEX IF NOT EXISTS idx_v2_events_slug_elapsed ON v2_events(slug, elapsed_sec);

-- ============================================================
-- V3: Regime-gated (same RF model, skip bad windows)
-- ============================================================

CREATE TABLE IF NOT EXISTS v3_windows (
    slug                          TEXT PRIMARY KEY,
    open_epoch                    BIGINT NOT NULL,
    close_epoch                   BIGINT NOT NULL,
    btc_open                      DOUBLE PRECISION,
    btc_close                     DOUBLE PRECISION,
    winner                        TEXT,
    entry_made                    BOOLEAN DEFAULT FALSE,
    entry_elapsed_sec             DOUBLE PRECISION,
    entry_side                    TEXT,
    entry_ask                     DOUBLE PRECISION,
    entry_shares                  INTEGER,
    entry_confidence              DOUBLE PRECISION,
    entry_ml_prob                 DOUBLE PRECISION,
    action_type                   TEXT,
    hedge_made                    BOOLEAN DEFAULT FALSE,
    hedge_elapsed_sec             DOUBLE PRECISION,
    hedge_tier                    INTEGER,
    hedge_combined_cost           DOUBLE PRECISION,
    hedge_opp_ask                 DOUBLE PRECISION,
    hedge_confidence              DOUBLE PRECISION,
    stopped_out                   BOOLEAN DEFAULT FALSE,
    stop_loss_elapsed_sec         DOUBLE PRECISION,
    stop_loss_price               DOUBLE PRECISION,
    emergency_hedge_made          BOOLEAN DEFAULT FALSE,
    emergency_hedge_elapsed_sec   DOUBLE PRECISION,
    emergency_hedge_reason        TEXT,
    correct                       BOOLEAN,
    pnl                           DOUBLE PRECISION,
    recorded_at                   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS v3_events (
    id                BIGSERIAL PRIMARY KEY,
    slug              TEXT NOT NULL REFERENCES v3_windows(slug) ON DELETE CASCADE,
    elapsed_sec       DOUBLE PRECISION NOT NULL,
    event_type        TEXT NOT NULL,
    recorded_at       TIMESTAMPTZ DEFAULT NOW(),
    up_bid            DOUBLE PRECISION,
    up_ask            DOUBLE PRECISION,
    down_bid          DOUBLE PRECISION,
    down_ask          DOUBLE PRECISION,
    up_spread         DOUBLE PRECISION,
    down_spread       DOUBLE PRECISION,
    btc_price         DOUBLE PRECISION,
    btc_change_from_open DOUBLE PRECISION,
    ml_prob_up           DOUBLE PRECISION,
    ml_model_t           INTEGER,
    ensemble_confidence  DOUBLE PRECISION,
    predicted_side       TEXT,
    market_leader_signal TEXT,
    btc_direction_signal TEXT,
    btc_market_agree     BOOLEAN,
    ask_strength         DOUBLE PRECISION,
    action            TEXT,
    side              TEXT,
    shares            INTEGER,
    price             DOUBLE PRECISION,
    reason            TEXT,
    spread_value      DOUBLE PRECISION,
    spread_passed     BOOLEAN,
    reversal_count    INTEGER,
    reversal_passed   BOOLEAN,
    confirm_elapsed   DOUBLE PRECISION,
    hedgeable         BOOLEAN,
    hedge_tier        INTEGER,
    combined_cost     DOUBLE PRECISION,
    opp_ask           DOUBLE PRECISION,
    guaranteed_profit DOUBLE PRECISION,
    details_json      JSONB
);

CREATE INDEX IF NOT EXISTS idx_v3_events_slug ON v3_events(slug);
CREATE INDEX IF NOT EXISTS idx_v3_events_type ON v3_events(event_type);
CREATE INDEX IF NOT EXISTS idx_v3_events_slug_elapsed ON v3_events(slug, elapsed_sec);

-- ============================================================
-- V4: Late entry + raw ML (enter at t=210+, no ensemble)
-- ============================================================

CREATE TABLE IF NOT EXISTS v4_windows (
    slug                          TEXT PRIMARY KEY,
    open_epoch                    BIGINT NOT NULL,
    close_epoch                   BIGINT NOT NULL,
    btc_open                      DOUBLE PRECISION,
    btc_close                     DOUBLE PRECISION,
    winner                        TEXT,
    entry_made                    BOOLEAN DEFAULT FALSE,
    entry_elapsed_sec             DOUBLE PRECISION,
    entry_side                    TEXT,
    entry_ask                     DOUBLE PRECISION,
    entry_shares                  INTEGER,
    entry_confidence              DOUBLE PRECISION,
    entry_ml_prob                 DOUBLE PRECISION,
    action_type                   TEXT,
    hedge_made                    BOOLEAN DEFAULT FALSE,
    hedge_elapsed_sec             DOUBLE PRECISION,
    hedge_tier                    INTEGER,
    hedge_combined_cost           DOUBLE PRECISION,
    hedge_opp_ask                 DOUBLE PRECISION,
    hedge_confidence              DOUBLE PRECISION,
    stopped_out                   BOOLEAN DEFAULT FALSE,
    stop_loss_elapsed_sec         DOUBLE PRECISION,
    stop_loss_price               DOUBLE PRECISION,
    emergency_hedge_made          BOOLEAN DEFAULT FALSE,
    emergency_hedge_elapsed_sec   DOUBLE PRECISION,
    emergency_hedge_reason        TEXT,
    correct                       BOOLEAN,
    pnl                           DOUBLE PRECISION,
    recorded_at                   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS v4_events (
    id                BIGSERIAL PRIMARY KEY,
    slug              TEXT NOT NULL REFERENCES v4_windows(slug) ON DELETE CASCADE,
    elapsed_sec       DOUBLE PRECISION NOT NULL,
    event_type        TEXT NOT NULL,
    recorded_at       TIMESTAMPTZ DEFAULT NOW(),
    up_bid            DOUBLE PRECISION,
    up_ask            DOUBLE PRECISION,
    down_bid          DOUBLE PRECISION,
    down_ask          DOUBLE PRECISION,
    up_spread         DOUBLE PRECISION,
    down_spread       DOUBLE PRECISION,
    btc_price         DOUBLE PRECISION,
    btc_change_from_open DOUBLE PRECISION,
    ml_prob_up           DOUBLE PRECISION,
    ml_model_t           INTEGER,
    ensemble_confidence  DOUBLE PRECISION,
    predicted_side       TEXT,
    market_leader_signal TEXT,
    btc_direction_signal TEXT,
    btc_market_agree     BOOLEAN,
    ask_strength         DOUBLE PRECISION,
    action            TEXT,
    side              TEXT,
    shares            INTEGER,
    price             DOUBLE PRECISION,
    reason            TEXT,
    spread_value      DOUBLE PRECISION,
    spread_passed     BOOLEAN,
    reversal_count    INTEGER,
    reversal_passed   BOOLEAN,
    confirm_elapsed   DOUBLE PRECISION,
    hedgeable         BOOLEAN,
    hedge_tier        INTEGER,
    combined_cost     DOUBLE PRECISION,
    opp_ask           DOUBLE PRECISION,
    guaranteed_profit DOUBLE PRECISION,
    details_json      JSONB
);

CREATE INDEX IF NOT EXISTS idx_v4_events_slug ON v4_events(slug);
CREATE INDEX IF NOT EXISTS idx_v4_events_type ON v4_events(event_type);
CREATE INDEX IF NOT EXISTS idx_v4_events_slug_elapsed ON v4_events(slug, elapsed_sec);
