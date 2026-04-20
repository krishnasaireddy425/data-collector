-- V4+ and V9 schemas in a single migration.
--
-- V4+: V4's EV regression + V3 probability floor (skip <0.30).
--      Expected vs V4: +24% PnL, +5pp accuracy, -6% entries.
--
-- V9:  V4 with stricter EV threshold (0.04 vs V4's 0.025). No V3 filter.
--      Pure ablation of the EV-threshold dimension.
--
-- Both paper-trade alongside V4 (baseline unchanged) for clean comparison.
--
-- Run ONCE in Supabase SQL Editor.

-- ============================================================
-- V4+ — V3 probability filter on top of V4
-- ============================================================
DROP TABLE IF EXISTS v4plus_events CASCADE;
DROP TABLE IF EXISTS v4plus_windows CASCADE;

CREATE TABLE v4plus_windows (
    slug                      TEXT PRIMARY KEY,
    open_epoch                BIGINT NOT NULL,
    close_epoch               BIGINT NOT NULL,
    btc_open                  DOUBLE PRECISION,
    btc_close                 DOUBLE PRECISION,
    winner                    TEXT,
    entry_made                BOOLEAN DEFAULT FALSE,
    entry_elapsed_sec         DOUBLE PRECISION,
    entry_side                TEXT,
    entry_price               DOUBLE PRECISION,
    entry_shares              INTEGER,
    entry_ev_up               DOUBLE PRECISION,   -- V4 prediction at entry
    entry_ev_down             DOUBLE PRECISION,
    entry_predicted_ev        DOUBLE PRECISION,
    entry_threshold           DOUBLE PRECISION,   -- EV threshold (0.025)
    -- V4+ specific: V3 probability at entry time
    entry_v3_prob_up          DOUBLE PRECISION,   -- V3's prob_up at entry tick
    entry_v3_prob_on_side     DOUBLE PRECISION,   -- V3's prob on OUR entry side
    v3_prob_floor             DOUBLE PRECISION,   -- filter threshold (0.30)
    correct                   BOOLEAN,
    pnl                       DOUBLE PRECISION,
    recorded_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE v4plus_events (
    id                    BIGSERIAL PRIMARY KEY,
    slug                  TEXT NOT NULL REFERENCES v4plus_windows(slug) ON DELETE CASCADE,
    elapsed_sec           DOUBLE PRECISION NOT NULL,
    event_type            TEXT NOT NULL,
        -- 'prediction' | 'entry' | 'blocked_by_v3'
    up_bid                DOUBLE PRECISION,
    up_ask                DOUBLE PRECISION,
    down_bid              DOUBLE PRECISION,
    down_ask              DOUBLE PRECISION,
    up_spread             DOUBLE PRECISION,
    down_spread           DOUBLE PRECISION,
    btc_price             DOUBLE PRECISION,
    btc_change_from_open  DOUBLE PRECISION,
    ev_up                 DOUBLE PRECISION,
    ev_down               DOUBLE PRECISION,
    predicted_side        TEXT,
    -- V4+ specific: V3 probability + filter tracking
    v3_prob_up            DOUBLE PRECISION,
    v3_prob_on_side       DOUBLE PRECISION,
    filter_reason         TEXT,       -- 'passed' | 'v3_prob_lt_0.30'
    action                TEXT,
    side                  TEXT,
    shares                INTEGER,
    price                 DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_v4plus_events_slug         ON v4plus_events(slug);
CREATE INDEX idx_v4plus_events_type         ON v4plus_events(event_type);
CREATE INDEX idx_v4plus_events_slug_elapsed ON v4plus_events(slug, elapsed_sec);

-- ============================================================
-- V9 — V4 with stricter EV threshold only
-- ============================================================
DROP TABLE IF EXISTS v9_events CASCADE;
DROP TABLE IF EXISTS v9_windows CASCADE;

CREATE TABLE v9_windows (
    slug                      TEXT PRIMARY KEY,
    open_epoch                BIGINT NOT NULL,
    close_epoch               BIGINT NOT NULL,
    btc_open                  DOUBLE PRECISION,
    btc_close                 DOUBLE PRECISION,
    winner                    TEXT,
    entry_made                BOOLEAN DEFAULT FALSE,
    entry_elapsed_sec         DOUBLE PRECISION,
    entry_side                TEXT,
    entry_price               DOUBLE PRECISION,
    entry_shares              INTEGER,
    entry_ev_up               DOUBLE PRECISION,
    entry_ev_down             DOUBLE PRECISION,
    entry_predicted_ev        DOUBLE PRECISION,
    entry_threshold           DOUBLE PRECISION,   -- EV threshold (0.04 for V9)
    correct                   BOOLEAN,
    pnl                       DOUBLE PRECISION,
    recorded_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE v9_events (
    id                    BIGSERIAL PRIMARY KEY,
    slug                  TEXT NOT NULL REFERENCES v9_windows(slug) ON DELETE CASCADE,
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
    ev_up                 DOUBLE PRECISION,
    ev_down               DOUBLE PRECISION,
    predicted_side        TEXT,
    action                TEXT,
    side                  TEXT,
    shares                INTEGER,
    price                 DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_v9_events_slug         ON v9_events(slug);
CREATE INDEX idx_v9_events_type         ON v9_events(event_type);
CREATE INDEX idx_v9_events_slug_elapsed ON v9_events(slug, elapsed_sec);
