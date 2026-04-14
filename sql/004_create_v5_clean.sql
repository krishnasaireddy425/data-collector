-- V5: cheap-side mean-reversion strategy with reversal-confirm rule.
-- No ML model. Pure rule-based entry when cheap_side_ask is in [0.20, 0.40]
-- AND has ticked up 2c from its local minimum.
--
-- Run this ONCE in Supabase SQL Editor.

CREATE TABLE IF NOT EXISTS v5_windows (
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
    entry_local_min       DOUBLE PRECISION,   -- lowest ask seen on chosen side before entry
    entry_uptick_amount   DOUBLE PRECISION,   -- how many cents above local_min at entry
    correct               BOOLEAN,
    pnl                   DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS v5_events (
    id                    BIGSERIAL PRIMARY KEY,
    slug                  TEXT NOT NULL REFERENCES v5_windows(slug) ON DELETE CASCADE,
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
    cheap_side            TEXT,               -- 'up' or 'down'
    cheap_price           DOUBLE PRECISION,   -- ask on the cheap side
    local_min             DOUBLE PRECISION,   -- running min for the cheap side
    uptick_amount         DOUBLE PRECISION,   -- cheap_price - local_min (positive = bounce)
    action                TEXT,
    side                  TEXT,
    shares                INTEGER,
    price                 DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v5_events_slug ON v5_events(slug);
CREATE INDEX IF NOT EXISTS idx_v5_events_type ON v5_events(event_type);
CREATE INDEX IF NOT EXISTS idx_v5_events_slug_elapsed ON v5_events(slug, elapsed_sec);
