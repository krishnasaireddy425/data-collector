"""
Feature engineering — SECOND-LEVEL continuous prediction (v2, expanded).

One function used by BOTH training (batch) and live prediction (online).
Given a 1-sec resampled DataFrame and the window's btc_open price, returns
a DataFrame with one feature row per input row (same length, same index).

v2: 44 features. Added wider momentum horizons, signal-to-noise ratios,
rolling correlations, longer volatility, and explicit interaction terms.
"""

import numpy as np
import pandas as pd

FEATURES = [
    # --- Core state (9) ---
    "elapsed_sec",
    "up_bid", "up_ask", "up_mid", "up_spread",
    "down_bid", "down_ask",
    "book_imbalance", "up_ask_ratio",

    # --- BTC change vs window open (2) ---
    "btc_chg", "btc_chg_pct",

    # --- BTC momentum at multiple horizons (10) ---
    "btc_mom_3s", "btc_mom_5s", "btc_mom_10s", "btc_mom_15s",
    "btc_mom_30s", "btc_mom_45s", "btc_mom_60s", "btc_mom_90s",
    "btc_mom_120s", "btc_mom_180s",

    # --- BTC pct momentum (2) ---
    "btc_mom_pct_30s", "btc_mom_pct_60s",

    # --- Up-ask momentum (3) ---
    "up_ask_mom_15s", "up_ask_mom_30s", "up_ask_mom_60s",

    # --- Volatility (4) ---
    "btc_vol_15s", "btc_vol_30s", "btc_vol_60s", "up_ask_vol_30s",

    # --- Signal-to-noise (momentum / vol) (2) ---
    "btc_sharpe_30s", "btc_sharpe_60s",

    # --- Rolling correlation BTC vs up_ask (1) ---
    "btc_corr_up_ask_60s",

    # --- Range & position (2) ---
    "up_ask_pct_of_range_60s", "up_ask_pct_of_range_120s",

    # --- Structural (6) ---
    "leader_flips_30s", "leader_flips_60s",
    "market_agrees_btc", "time_since_last_flip",
    "up_spread_chg_30s", "btc_accel_30s",

    # --- Explicit interactions (3) ---
    "btc_mom_x_elapsed",           # btc_mom_60s * (elapsed_sec / 300)
    "book_imb_x_vol",              # book_imbalance * btc_vol_30s
    "up_ask_ratio_x_chgpct",       # up_ask_ratio * btc_chg_pct
]


def _safe_div(a, b):
    return np.where(np.abs(b) > 1e-9, a / b, 0.0)


def compute_features_batch(df, btc_open):
    """Vectorised features over a 1-sec resampled, ffilled DataFrame."""
    out = pd.DataFrame(index=df.index)

    # --- Core state ---
    out["elapsed_sec"] = df["elapsed_sec"].astype(float)
    out["up_bid"] = df["up_bid"].astype(float)
    out["up_ask"] = df["up_ask"].astype(float)
    out["up_mid"] = (df["up_bid"] + df["up_ask"]) / 2.0
    out["up_spread"] = df["up_ask"] - df["up_bid"]
    out["down_bid"] = df["down_bid"].astype(float)
    out["down_ask"] = df["down_ask"].astype(float)
    out["book_imbalance"] = df["up_ask"] - df["down_ask"]

    ua_plus_da = df["up_ask"] + df["down_ask"]
    out["up_ask_ratio"] = np.where(
        ua_plus_da > 0, df["up_ask"] / ua_plus_da, 0.5
    )

    # --- BTC change vs open ---
    btc_open = float(btc_open) if btc_open else 0.0
    out["btc_chg"] = df["btc_price"] - btc_open
    out["btc_chg_pct"] = out["btc_chg"] / btc_open if btc_open > 0 else 0.0

    # --- BTC momentum (many horizons) ---
    for lb in (3, 5, 10, 15, 30, 45, 60, 90, 120, 180):
        out[f"btc_mom_{lb}s"] = df["btc_price"] - df["btc_price"].shift(lb)
    out["btc_mom_pct_30s"] = (
        out["btc_mom_30s"] / btc_open if btc_open > 0 else 0.0
    )
    out["btc_mom_pct_60s"] = (
        out["btc_mom_60s"] / btc_open if btc_open > 0 else 0.0
    )

    # --- Up-ask momentum ---
    out["up_ask_mom_15s"] = df["up_ask"] - df["up_ask"].shift(15)
    out["up_ask_mom_30s"] = df["up_ask"] - df["up_ask"].shift(30)
    out["up_ask_mom_60s"] = df["up_ask"] - df["up_ask"].shift(60)

    # --- Volatility ---
    out["btc_vol_15s"] = df["btc_price"].rolling(15, min_periods=5).std()
    out["btc_vol_30s"] = df["btc_price"].rolling(30, min_periods=5).std()
    out["btc_vol_60s"] = df["btc_price"].rolling(60, min_periods=10).std()
    out["up_ask_vol_30s"] = df["up_ask"].rolling(30, min_periods=5).std()

    # --- Sharpe-like: momentum / volatility (signal-to-noise) ---
    out["btc_sharpe_30s"] = _safe_div(out["btc_mom_30s"].values,
                                      out["btc_vol_30s"].values)
    out["btc_sharpe_60s"] = _safe_div(out["btc_mom_60s"].values,
                                      out["btc_vol_60s"].values)

    # --- Rolling correlation (market-follows-BTC strength) ---
    btc_ret = df["btc_price"].diff()
    ua_ret = df["up_ask"].diff()
    out["btc_corr_up_ask_60s"] = btc_ret.rolling(
        60, min_periods=10
    ).corr(ua_ret)

    # --- Range position ---
    for lb in (60, 120):
        rmax = df["up_ask"].rolling(lb, min_periods=5).max()
        rmin = df["up_ask"].rolling(lb, min_periods=5).min()
        denom = (rmax - rmin).replace(0, np.nan)
        out[f"up_ask_pct_of_range_{lb}s"] = (
            (df["up_ask"] - rmin) / denom
        ).fillna(0.5)

    # --- Structural / regime ---
    leader = (df["up_ask"] > df["down_ask"]).astype(int)
    flip_flags = leader.diff().abs().fillna(0)
    out["leader_flips_30s"] = flip_flags.rolling(30, min_periods=1).sum()
    out["leader_flips_60s"] = flip_flags.rolling(60, min_periods=1).sum()

    btc_up = (df["btc_price"] > btc_open).astype(int)
    out["market_agrees_btc"] = (leader == btc_up).astype(int)

    leader_groups = flip_flags.cumsum().fillna(0)
    out["time_since_last_flip"] = (
        leader_groups.groupby(leader_groups).cumcount().astype(float)
    )

    out["up_spread_chg_30s"] = out["up_spread"] - out["up_spread"].shift(30)

    mom_recent = out["btc_mom_15s"]
    mom_older = df["btc_price"].shift(15) - df["btc_price"].shift(30)
    out["btc_accel_30s"] = mom_recent - mom_older

    # --- Interactions ---
    out["btc_mom_x_elapsed"] = (
        out["btc_mom_60s"] * (out["elapsed_sec"] / 300.0)
    )
    out["book_imb_x_vol"] = out["book_imbalance"] * out["btc_vol_30s"]
    out["up_ask_ratio_x_chgpct"] = out["up_ask_ratio"] * out["btc_chg_pct"]

    # NaNs from shift()/rolling() -> 0 (model reads "no history yet").
    out = out.replace([np.inf, -np.inf], 0).fillna(0.0)
    return out[FEATURES]


def resample_window(df, max_t=300):
    """Resample a raw CSV DataFrame to 1-second grid with ffill+bfill."""
    df = df.copy()
    for col in ("up_bid", "up_ask", "down_bid", "down_ask", "btc_price",
                "elapsed_sec"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["elapsed_sec", "btc_price"])
    df = df.drop_duplicates(subset=["elapsed_sec"], keep="last")
    df = df.sort_values("elapsed_sec").reset_index(drop=True)
    if df.empty:
        return None

    new_idx = np.arange(0, max_t + 1, 1.0)
    df_r = df.set_index("elapsed_sec")
    df_1s = df_r.reindex(df_r.index.union(new_idx)).sort_index()
    df_1s = df_1s.ffill().bfill()
    df_1s = df_1s.loc[new_idx].reset_index()
    df_1s.rename(columns={"index": "elapsed_sec"}, inplace=True)
    return df_1s
