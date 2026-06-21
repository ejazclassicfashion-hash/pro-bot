"""
Multi-Timeframe (MTF) Bias Engine
Higher timeframe (4H / Daily) determines directional bias.
Lower timeframe (1H) is only used for entries IN that direction.

Real ICT methodology: top-down analysis.
Daily/Weekly → 4H structure → 1H entry
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class MTFParams:
    ema_fast: int = 21
    ema_slow: int = 50
    swing_lookback: int = 3       # smaller lookback on 4H (fewer candles available)
    require_structure_confirm: bool = True  # need HH/HL or LH/LL, not just EMA


Bias = Literal["bullish", "bearish", "neutral"]


def compute_4h_bias(df_4h: pd.DataFrame, params: MTFParams | None = None) -> pd.Series:
    """
    For each 4H candle, determine directional bias.
    Returns a Series indexed by 4H timestamps with values: 'bullish'/'bearish'/'neutral'.

    Logic:
    - EMA fast > EMA slow = bullish lean
    - Price above EMA slow = bullish lean
    - Recent swing structure (HH/HL vs LH/LL) as confirmation
    Both must agree → firm bias. Only one → neutral.
    """
    if params is None:
        params = MTFParams()

    df = df_4h.copy()
    df["ema_fast"] = df["close"].ewm(span=params.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=params.ema_slow, adjust=False).mean()

    # EMA bias
    df["ema_bias"] = "neutral"
    df.loc[df["ema_fast"] > df["ema_slow"], "ema_bias"] = "bullish"
    df.loc[df["ema_fast"] < df["ema_slow"], "ema_bias"] = "bearish"

    # Swing structure bias — rolling window
    struct_bias = []
    lb = params.swing_lookback
    for i in range(len(df)):
        if i < lb * 2 + 2:
            struct_bias.append("neutral")
            continue

        # Detect last 2 swing highs and lows in a window
        window = df.iloc[max(0, i - 40): i + 1]
        sh_idxs = []
        sl_idxs = []
        for j in range(lb, len(window) - lb):
            w_h = window["high"].iloc[j - lb: j + lb + 1]
            w_l = window["low"].iloc[j - lb: j + lb + 1]
            if window["high"].iloc[j] == w_h.max():
                sh_idxs.append((j, window["high"].iloc[j]))
            if window["low"].iloc[j] == w_l.min():
                sl_idxs.append((j, window["low"].iloc[j]))

        if len(sh_idxs) >= 2 and len(sl_idxs) >= 2:
            # Last two swing highs
            sh_last, sh_prev = sh_idxs[-1][1], sh_idxs[-2][1]
            sl_last, sl_prev = sl_idxs[-1][1], sl_idxs[-2][1]
            if sh_last > sh_prev and sl_last > sl_prev:
                struct_bias.append("bullish")   # HH + HL
            elif sh_last < sh_prev and sl_last < sl_prev:
                struct_bias.append("bearish")   # LH + LL
            else:
                struct_bias.append("neutral")
        else:
            struct_bias.append("neutral")

    df["struct_bias"] = struct_bias

    # Final bias: both must agree
    bias_series = []
    for i in range(len(df)):
        eb = df["ema_bias"].iloc[i]
        sb = df["struct_bias"].iloc[i]
        if eb == sb and eb != "neutral":
            bias_series.append(eb)
        elif not params.require_structure_confirm and eb != "neutral":
            bias_series.append(eb)
        else:
            bias_series.append("neutral")

    return pd.Series(bias_series, index=df.index, name="4h_bias")


def map_bias_to_1h(
    bias_4h: pd.Series,
    timestamps_1h: pd.DatetimeIndex,
) -> pd.Series:
    """
    For each 1H candle timestamp, find the most recently CLOSED 4H candle
    and return its bias. This avoids look-ahead bias.
    """
    result = []
    bias_times = bias_4h.index.tolist()

    for ts in timestamps_1h:
        # Find the last 4H candle that closed AT OR BEFORE this 1H candle
        valid = [t for t in bias_times if t <= ts]
        if not valid:
            result.append("neutral")
        else:
            result.append(bias_4h[valid[-1]])

    return pd.Series(result, index=timestamps_1h, name="4h_bias")


def filter_signals_by_bias(
    signals: list[dict],
    bias_1h: pd.Series,
) -> list[dict]:
    """
    Keep only signals where trade direction matches the 4H bias.
    Neutral 4H bias = skip the trade (no clear HTF context).
    """
    filtered = []
    for sig in signals:
        ts = sig["timestamp"]
        # Get bias at signal time
        if ts not in bias_1h.index:
            # Find nearest timestamp
            valid = bias_1h.index[bias_1h.index <= ts]
            if valid.empty:
                continue
            bias = bias_1h[valid[-1]]
        else:
            bias = bias_1h[ts]

        if bias == "neutral":
            continue
        if sig["direction"] == "long" and bias == "bullish":
            sig["4h_bias"] = bias
            filtered.append(sig)
        elif sig["direction"] == "short" and bias == "bearish":
            sig["4h_bias"] = bias
            filtered.append(sig)

    return filtered
