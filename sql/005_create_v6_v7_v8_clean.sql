-- V6 + V7 + V8 schemas in a single migration.
--
-- V6: V3 clone with CONF_THRESHOLD=0.70 (aggressive variant).
--     Backtest: +$361.30 over 1,260 CSVs.
--
-- V7: V5 cheap-side entry + V6 (V3 @ 0.70) entry + V5 exits at bid when
--     V6 enters the opposite side.
--     Backtest: +$667.30 (~$152/day).
--
-- V8: V5 cheap-side entry alone. V6 runs only as an exit signal — when V6
--     would have bought the opposite side, V8 sells V5 at market bid.
--     Backtest: +$306.00 (~$70/day).
--
-- Run ONCE in Supabase SQL Editor.

-- ============================================================
-- V6 — V3 clone @ 0.70 threshold
-- ============================================================
DROP TABLE IF EXISTS v6_events CASCADE;
DROP TABLE IF EXISTS v6_windows CASCADE;

CREATE TABLE v6_windows (
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
    entry_prob_up         DOUBLE PRECISION,
    entry_confidence      DOUBLE PRECISION,
    correct               BOOLEAN,
    pnl                   DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE v6_events (
    id                    BIGSERIAL PRIMARY KEY,
    slug                  TEXT NOT NULL REFERENCES v6_windows(slug) ON DELETE CASCADE,
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
    prob_up               DOUBLE PRECISION,
    predicted_side        TEXT,
    action                TEXT,
    side                  TEXT,
    shares                INTEGER,
    price                 DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_v6_events_slug         ON v6_events(slug);
CREATE INDEX idx_v6_events_type         ON v6_events(event_type);
CREATE INDEX idx_v6_events_slug_elapsed ON v6_events(slug, elapsed_sec);

-- ============================================================
-- V7 — V5 entry + V6 entry + V5-exit-on-V6-disagree
-- ============================================================
DROP TABLE IF EXISTS v7_events CASCADE;
DROP TABLE IF EXISTS v7_windows CASCADE;

CREATE TABLE v7_windows (
    slug                      TEXT PRIMARY KEY,
    open_epoch                BIGINT NOT NULL,
    close_epoch               BIGINT NOT NULL,
    btc_open                  DOUBLE PRECISION,
    btc_close                 DOUBLE PRECISION,
    winner                    TEXT,

    v5_entry_made             BOOLEAN DEFAULT FALSE,
    v5_entry_elapsed_sec      DOUBLE PRECISION,
    v5_entry_side             TEXT,
    v5_entry_price            DOUBLE PRECISION,
    v5_entry_shares           INTEGER,
    v5_entry_local_min        DOUBLE PRECISION,
    v5_entry_uptick_amount    DOUBLE PRECISION,

    v6_entry_made             BOOLEAN DEFAULT FALSE,
    v6_entry_elapsed_sec      DOUBLE PRECISION,
    v6_entry_side             TEXT,
    v6_entry_price            DOUBLE PRECISION,
    v6_entry_shares           INTEGER,
    v6_entry_prob_up          DOUBLE PRECISION,
    v6_entry_confidence       DOUBLE PRECISION,

    v5_exit_made              BOOLEAN DEFAULT FALSE,
    v5_exit_elapsed_sec       DOUBLE PRECISION,
    v5_exit_bid               DOUBLE PRECISION,
    v5_exit_shares            INTEGER,

    v5_correct                BOOLEAN,
    v6_correct                BOOLEAN,
    v5_pnl                    DOUBLE PRECISION,
    v6_pnl                    DOUBLE PRECISION,
    combined_pnl              DOUBLE PRECISION,

    recorded_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE v7_events (
    id                    BIGSERIAL PRIMARY KEY,
    slug                  TEXT NOT NULL REFERENCES v7_windows(slug) ON DELETE CASCADE,
    elapsed_sec           DOUBLE PRECISION NOT NULL,
    event_type            TEXT NOT NULL,   -- 'prediction' | 'v5_entry' | 'v6_entry' | 'v5_exit'
    up_bid                DOUBLE PRECISION,
    up_ask                DOUBLE PRECISION,
    down_bid              DOUBLE PRECISION,
    down_ask              DOUBLE PRECISION,
    up_spread             DOUBLE PRECISION,
    down_spread           DOUBLE PRECISION,
    btc_price             DOUBLE PRECISION,
    btc_change_from_open  DOUBLE PRECISION,
    prob_up               DOUBLE PRECISION,
    predicted_side        TEXT,
    action                TEXT,
    side                  TEXT,
    shares                INTEGER,
    price                 DOUBLE PRECISION,
    cheap_side            TEXT,
    cheap_price           DOUBLE PRECISION,
    local_min             DOUBLE PRECISION,
    uptick_amount         DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_v7_events_slug         ON v7_events(slug);
CREATE INDEX idx_v7_events_type         ON v7_events(event_type);
CREATE INDEX idx_v7_events_slug_elapsed ON v7_events(slug, elapsed_sec);

-- ============================================================
-- V8 — V5 entry alone + V6-disagree signal as exit trigger
-- ============================================================
DROP TABLE IF EXISTS v8_events CASCADE;
DROP TABLE IF EXISTS v8_windows CASCADE;

CREATE TABLE v8_windows (
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
    entry_local_min           DOUBLE PRECISION,
    entry_uptick_amount       DOUBLE PRECISION,

    exit_made                 BOOLEAN DEFAULT FALSE,
    exit_elapsed_sec          DOUBLE PRECISION,
    exit_bid                  DOUBLE PRECISION,
    exit_shares               INTEGER,
    exit_prob_up              DOUBLE PRECISION,

    correct                   BOOLEAN,
    pnl                       DOUBLE PRECISION,

    recorded_at               TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE v8_events (
    id                    BIGSERIAL PRIMARY KEY,
    slug                  TEXT NOT NULL REFERENCES v8_windows(slug) ON DELETE CASCADE,
    elapsed_sec           DOUBLE PRECISION NOT NULL,
    event_type            TEXT NOT NULL,   -- 'prediction' | 'entry' | 'exit'
    up_bid                DOUBLE PRECISION,
    up_ask                DOUBLE PRECISION,
    down_bid              DOUBLE PRECISION,
    down_ask              DOUBLE PRECISION,
    up_spread             DOUBLE PRECISION,
    down_spread           DOUBLE PRECISION,
    btc_price             DOUBLE PRECISION,
    btc_change_from_open  DOUBLE PRECISION,
    prob_up               DOUBLE PRECISION,
    predicted_side        TEXT,
    action                TEXT,
    side                  TEXT,
    shares                INTEGER,
    price                 DOUBLE PRECISION,
    cheap_side            TEXT,
    cheap_price           DOUBLE PRECISION,
    local_min             DOUBLE PRECISION,
    uptick_amount         DOUBLE PRECISION,
    recorded_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_v8_events_slug         ON v8_events(slug);
CREATE INDEX idx_v8_events_type         ON v8_events(event_type);
CREATE INDEX idx_v8_events_slug_elapsed ON v8_events(slug, elapsed_sec);
