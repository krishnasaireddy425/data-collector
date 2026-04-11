-- Paper Trading Decision Database
-- Run this in Supabase SQL Editor (Project → SQL Editor → New query → paste → Run)
--
-- Creates two tables:
--   1. windows  - one row per market window (the summary)
--   2. events   - many rows per window (every decision the strategy makes)

-- ============================================================
-- Table 1: windows  (one row per 5-min market window)
-- ============================================================
CREATE TABLE IF NOT EXISTS windows (
    slug                          TEXT PRIMARY KEY,
    open_epoch                    BIGINT NOT NULL,
    close_epoch                   BIGINT NOT NULL,

    -- BTC oracle prices (Chainlink)
    btc_open                      DOUBLE PRECISION,
    btc_close                     DOUBLE PRECISION,
    winner                        TEXT,                    -- 'up' / 'down' / 'flat' / 'unknown'

    -- Entry decision
    entry_made                    BOOLEAN DEFAULT FALSE,
    entry_elapsed_sec             DOUBLE PRECISION,        -- exact second within window (0-300)
    entry_side                    TEXT,                    -- 'up' / 'down'
    entry_ask                     DOUBLE PRECISION,
    entry_shares                  INTEGER,
    entry_confidence              DOUBLE PRECISION,        -- ensemble confidence at entry
    entry_ml_prob                 DOUBLE PRECISION,        -- raw ML probability at entry
    action_type                   TEXT,                    -- 'full_ride' / 'partial_ride' / 'full_hedge' / 'no_entry'

    -- Hedge decision
    hedge_made                    BOOLEAN DEFAULT FALSE,
    hedge_elapsed_sec             DOUBLE PRECISION,
    hedge_tier                    INTEGER,                 -- 1 / 2 / 3 / 4
    hedge_combined_cost           DOUBLE PRECISION,        -- entry_avg + opp_ask
    hedge_opp_ask                 DOUBLE PRECISION,
    hedge_confidence              DOUBLE PRECISION,        -- confidence when hedge was triggered

    -- Stop loss
    stopped_out                   BOOLEAN DEFAULT FALSE,
    stop_loss_elapsed_sec         DOUBLE PRECISION,
    stop_loss_price               DOUBLE PRECISION,

    -- Emergency hedge
    emergency_hedge_made          BOOLEAN DEFAULT FALSE,
    emergency_hedge_elapsed_sec   DOUBLE PRECISION,
    emergency_hedge_reason        TEXT,                    -- 'conf_drop' / 'side_flip' / 'time_safety'

    -- Settlement
    correct                       BOOLEAN,                 -- did entry_side == winner?
    pnl                           DOUBLE PRECISION,        -- simulated P&L for this window

    recorded_at                   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_windows_open_epoch ON windows(open_epoch);
CREATE INDEX IF NOT EXISTS idx_windows_winner ON windows(winner);
CREATE INDEX IF NOT EXISTS idx_windows_correct ON windows(correct);
CREATE INDEX IF NOT EXISTS idx_windows_action_type ON windows(action_type);


-- ============================================================
-- Table 2: events  (many rows per window — the decision trail)
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    id                BIGSERIAL PRIMARY KEY,
    slug              TEXT NOT NULL REFERENCES windows(slug) ON DELETE CASCADE,

    -- WHEN
    elapsed_sec       DOUBLE PRECISION NOT NULL,           -- 0-300, exact second within window
    event_type        TEXT NOT NULL,                       -- 'prediction' / 'entry' / 'hedge' /
                                                           -- 'stop_loss' / 'emergency_hedge' /
                                                           -- 'monitor' / 'skip' / 'settlement'
    recorded_at       TIMESTAMPTZ DEFAULT NOW(),

    -- Market state at this moment
    up_bid            DOUBLE PRECISION,
    up_ask            DOUBLE PRECISION,
    down_bid          DOUBLE PRECISION,
    down_ask          DOUBLE PRECISION,
    up_spread         DOUBLE PRECISION,
    down_spread       DOUBLE PRECISION,
    btc_price         DOUBLE PRECISION,
    btc_change_from_open DOUBLE PRECISION,                 -- delta from window's BTC open

    -- Model state
    ml_prob_up           DOUBLE PRECISION,                 -- raw RF model probability (0-1)
    ml_model_t           INTEGER,                          -- which timepoint model (60/90/120/.../240)
    ensemble_confidence  DOUBLE PRECISION,                 -- final ensemble confidence
    predicted_side       TEXT,                             -- 'up' / 'down'

    -- Individual signal contributions (for debugging signal weighting)
    market_leader_signal TEXT,                             -- 'up' / 'down'
    btc_direction_signal TEXT,                             -- 'up' / 'down'
    btc_market_agree     BOOLEAN,
    ask_strength         DOUBLE PRECISION,                 -- 0.0-1.0

    -- Action taken
    action            TEXT,                                -- 'predict' / 'enter' / 'skip' / 'hedge' /
                                                           -- 'stop_loss' / 'monitor'
    side              TEXT,                                -- 'up' / 'down'
    shares            INTEGER,
    price             DOUBLE PRECISION,
    reason            TEXT,                                -- 'above_threshold' / 'spread_too_wide' /
                                                           -- 'reversal_blocked' / 'confidence_too_low' /
                                                           -- 'price_too_high' / 'hedge_impossible' /
                                                           -- 'confirm_pending' / etc.

    -- Filter state (mostly populated on 'skip' events)
    spread_value      DOUBLE PRECISION,
    spread_passed     BOOLEAN,
    reversal_count    INTEGER,
    reversal_passed   BOOLEAN,
    confirm_elapsed   DOUBLE PRECISION,
    hedgeable         BOOLEAN,

    -- Hedge-specific (for 'hedge' events)
    hedge_tier        INTEGER,
    combined_cost     DOUBLE PRECISION,
    opp_ask           DOUBLE PRECISION,
    guaranteed_profit DOUBLE PRECISION,

    -- Free-form blob for anything else
    details_json      JSONB
);

CREATE INDEX IF NOT EXISTS idx_events_slug ON events(slug);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_slug_elapsed ON events(slug, elapsed_sec);
CREATE INDEX IF NOT EXISTS idx_events_action ON events(action);


-- ============================================================
-- Sanity checks: confirm tables exist
-- ============================================================
-- Run these after the CREATE TABLEs to verify everything was created:
--
-- SELECT table_name FROM information_schema.tables
--   WHERE table_schema = 'public' AND table_name IN ('windows', 'events');
--
-- SELECT column_name, data_type FROM information_schema.columns
--   WHERE table_name = 'windows' ORDER BY ordinal_position;
--
-- SELECT column_name, data_type FROM information_schema.columns
--   WHERE table_name = 'events' ORDER BY ordinal_position;
