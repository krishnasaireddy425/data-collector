"""
Tick-by-tick replay of one CSV through PredictorSec. Mirrors how the live
collector would feed WebSocket ticks into the model.

For each second boundary (t=1, 2, ..., 299), we compute P(Up). We make a
SIMPLE entry decision (no hedging, no stop-loss) so the test measures
pure prediction quality:

    First second in [MIN_ENTRY, MAX_ENTRY] where:
        prob_up >= CONF_THRESHOLD      -> BUY Up at up_ask
        prob_up <= 1 - CONF_THRESHOLD  -> BUY Down at down_ask
    Hold to settlement. PnL = 10 * ($1 - entry_price) if correct, else -10 * entry_price.

If no entry ever triggers the window is "skipped" — not counted toward accuracy.

This file returns a dict of per-window results; backtest_sec.py runs it on
all 170 test CSVs and aggregates.
"""

import re
from pathlib import Path

import pandas as pd

from .predictor_sec import PredictorSec

MIN_ENTRY = 60
MAX_ENTRY = 240

# Probability-weighted position sizing.
# Entry confidence is the side's probability (P(Up) if we bought Up,
# 1-P(Up) if we bought Down). Bet bigger when we're more confident.
def _shares_for_confidence(conf):
    if conf >= 0.90:
        return 12
    if conf >= 0.80:
        return 8
    if conf >= 0.70:
        return 5
    if conf >= 0.60:
        return 3
    return 1


def read_winner(csv_path):
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(-min(size, 4096), 2)
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            if "# RESULT" in line and "winner=" in line:
                m = re.search(r"winner=(\w+)", line)
                if m:
                    w = m.group(1)
                    return w if w in ("Up", "Down") else None
    except Exception:
        return None
    return None


def _clean_row(row):
    """Convert a DataFrame row to (elapsed_sec, up_bid, up_ask, down_bid,
    down_ask, btc_price). Returns None if any critical field is NaN."""
    try:
        elapsed = float(row["elapsed_sec"])
        up_ask = float(row["up_ask"])
        down_ask = float(row["down_ask"])
        btc = float(row["btc_price"])
    except (TypeError, ValueError):
        return None
    if not (up_ask > 0 and down_ask > 0 and btc > 0):
        return None
    up_bid = row.get("up_bid", 0.0)
    down_bid = row.get("down_bid", 0.0)
    up_bid = float(up_bid) if pd.notna(up_bid) else 0.0
    down_bid = float(down_bid) if pd.notna(down_bid) else 0.0
    return elapsed, up_bid, up_ask, down_bid, down_ask, btc


def replay_window(csv_path, predictor, conf_threshold=0.80,
                  min_entry=MIN_ENTRY, max_entry=MAX_ENTRY):
    """Replay one CSV tick-by-tick. Returns a dict with the per-window result."""
    predictor.reset()
    winner = read_winner(csv_path)
    if winner not in ("Up", "Down"):
        return None

    raw = pd.read_csv(csv_path, comment="#")
    if len(raw) < 10:
        return None

    # --- State for entry decision ---
    next_predict_sec = 1          # next integer elapsed at which to predict
    entry_side = None
    entry_price = None
    entry_elapsed = None
    entry_prob = None
    peak_prob_up = 0.0
    peak_prob_down = 0.0           # == 1 - min(prob_up)

    # Row-ordered tick walk
    for _, r in raw.iterrows():
        tick = _clean_row(r)
        if tick is None:
            continue
        elapsed, up_bid, up_ask, down_bid, down_ask, btc = tick
        predictor.add_tick(elapsed, up_bid, up_ask, down_bid, down_ask, btc)

        # Once we have an entry, no further decisions — still ingest ticks
        # so the predictor's buffer is complete, but don't act.
        if entry_side is not None:
            continue

        # Emit a prediction at each integer second boundary as it passes.
        while next_predict_sec <= elapsed and next_predict_sec <= max_entry:
            t = next_predict_sec
            next_predict_sec += 1
            if t < min_entry:
                continue
            prob, _feats = predictor.predict(t)
            if prob is None:
                continue
            peak_prob_up = max(peak_prob_up, prob)
            peak_prob_down = max(peak_prob_down, 1.0 - prob)

            if prob >= conf_threshold:
                entry_side = "Up"
                entry_price = up_ask
                entry_elapsed = t
                entry_prob = prob
                break
            if (1.0 - prob) >= conf_threshold:
                entry_side = "Down"
                entry_price = down_ask
                entry_elapsed = t
                entry_prob = 1.0 - prob
                break

    # --- Settlement ---
    result = {
        "slug": Path(csv_path).stem,
        "winner": winner,
        "entry_side": entry_side,
        "entry_elapsed": entry_elapsed,
        "entry_price": entry_price,
        "entry_confidence": entry_prob,
        "shares": 0,
        "peak_prob_up": peak_prob_up,
        "peak_prob_down": peak_prob_down,
        "correct": None,
        "pnl": 0.0,
        "skipped": entry_side is None,
    }
    if entry_side is not None:
        correct = (entry_side == winner)
        shares = _shares_for_confidence(entry_prob)
        pnl = (shares * (1.0 - entry_price)) if correct else (-shares * entry_price)
        result["shares"] = shares
        result["correct"] = correct
        result["pnl"] = round(pnl, 4)
    return result
