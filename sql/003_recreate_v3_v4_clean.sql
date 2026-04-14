-- Recreate V3 and V4 tables with CLEAN schemas for the new strategies:
--   V3 = 44f continuous XGB probability model (simple entry, no hedging)
--   V4 = EV regression model (expected-value entry)
--
-- Run this ONCE in Supabase SQL Editor. It drops the old V3/V4 tables
-- (which had hedging/stop-loss columns unused by the new strategies) and
-- recreates with only the fields these strategies actually use.
--
-- WARNING: this destroys any existing v3_* / v4_* data. V1/V2 tables are
-- untouched.

-- ============================================================
-- DROP OLD V3/V4
-- ============================================================
DROP TABLE IF EXISTS v3_events CASCADE;
DROP TABLE IF EXISTS v3_windows CASCADE;
DROP TABLE IF EXISTS v4_events CASCADE;
DROP TABLE IF EXISTS v4_windows CASCADE;


-- ============================================================
-- V3: 44-feature continuous XGB probability model
-- ============================================================

CREATE TABLE v3_windows (
    slug                  TEXT PRIMARY KEY,
    open_epoch            BIGINT NOT NULL,
    close_epoch           BIGINT NOT NULL,
    btc_open              DOUBLE PRECISION,
    btc_close             DOUBLE PRECISION,
    winner                TEXT,
    entry_made            BOOLEAN DEFAULT FALSE,
    entry_elapsed_sec     DOUBLE PRECISION,
    entry_side            TEXT,
    entry_price           DOUBLE PRECISION,
    entry_shares          INTEGER,
    entry_prob_up         DOUBLE PRECISION,   -- calibrated P(Up) at entry
    entry_confidence      DOUBLE PRECISION,   -- prob on the side we bought
    correct               BOOLEAN,
    pnl                   DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE v3_events (
    id                    BIGSERIAL PRIMARY KEY,
    slug                  TEXT NOT NULL REFERENCES v3_windows(slug) ON DELETE CASCADE,
    elapsed_sec           DOUBLE PRECISION NOT NULL,
    event_type            TEXT NOT NULL,
    up_bid                DOUBLE PRECISION,
    up_ask                DOUBLE PRECISION,
    down_bid              DOUBLE PRECISION,
    down_ask              DOUBLE PRECISION,
    up_spread             DOUBLE PRECISION,
    down_spread           DOUBLE PRECISION,
    btc_price             DOUBLE PRECISION,
    btc_change_from_open  DOUBLE PRECISION,
    prob_up               DOUBLE PRECISION,   -- calibrated P(Up) at this tick
    predicted_side        TEXT,
    action                TEXT,
    side                  TEXT,
    shares                INTEGER,
    price                 DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_v3_events_slug ON v3_events(slug);
CREATE INDEX idx_v3_events_type ON v3_events(event_type);
CREATE INDEX idx_v3_events_slug_elapsed ON v3_events(slug, elapsed_sec);


-- ============================================================
-- V4: Expected-Value regression model
-- ============================================================

CREATE TABLE v4_windows (
    slug                  TEXT PRIMARY KEY,
    open_epoch            BIGINT NOT NULL,
    close_epoch           BIGINT NOT NULL,
    btc_open              DOUBLE PRECISION,
    btc_close             DOUBLE PRECISION,
    winner                TEXT,
    entry_made            BOOLEAN DEFAULT FALSE,
    entry_elapsed_sec     DOUBLE PRECISION,
    entry_side            TEXT,
    entry_price           DOUBLE PRECISION,
    entry_shares          INTEGER,
    entry_ev_up           DOUBLE PRECISION,   -- predicted $ buying Up at entry
    entry_ev_down         DOUBLE PRECISION,   -- predicted $ buying Down at entry
    entry_predicted_ev    DOUBLE PRECISION,   -- chosen side's EV
    entry_threshold       DOUBLE PRECISION,   -- EV threshold in effect
    correct               BOOLEAN,
    pnl                   DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE v4_events (
    id                    BIGSERIAL PRIMARY KEY,
    slug                  TEXT NOT NULL REFERENCES v4_windows(slug) ON DELETE CASCADE,
    elapsed_sec           DOUBLE PRECISION NOT NULL,
    event_type            TEXT NOT NULL,
    up_bid                DOUBLE PRECISION,
    up_ask                DOUBLE PRECISION,
    down_bid              DOUBLE PRECISION,
    down_ask              DOUBLE PRECISION,
    up_spread             DOUBLE PRECISION,
    down_spread           DOUBLE PRECISION,
    btc_price             DOUBLE PRECISION,
    btc_change_from_open  DOUBLE PRECISION,
    ev_up                 DOUBLE PRECISION,   -- predicted $ buying Up at this tick
    ev_down               DOUBLE PRECISION,   -- predicted $ buying Down at this tick
    predicted_side        TEXT,
    action                TEXT,
    side                  TEXT,
    shares                INTEGER,
    price                 DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_v4_events_slug ON v4_events(slug);
CREATE INDEX idx_v4_events_type ON v4_events(event_type);
CREATE INDEX idx_v4_events_slug_elapsed ON v4_events(slug, elapsed_sec);
