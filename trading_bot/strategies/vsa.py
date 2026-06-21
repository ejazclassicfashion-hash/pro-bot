"""
Volume Spread Analysis (VSA) Engine
Detects: No Demand, No Supply, Stopping Volume, Climax, Effort vs Result.
These are used as confirmation signals alongside SMC structure.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class VSAParams:
    volume_ma_period: int = 20         # rolling average for "low/high" volume comparison
    spread_ma_period: int = 20         # rolling average for spread comparison
    low_vol_threshold: float = 0.7     # below this ratio of avg = low volume
    high_vol_threshold: float = 1.5    # above this = high volume
    narrow_spread_threshold: float = 0.6
    wide_spread_threshold: float = 1.4
    ultra_high_vol_threshold: float = 2.5


def analyze_vsa(df: pd.DataFrame, params: VSAParams | None = None) -> pd.DataFrame:
    """
    Returns a copy of df with VSA signal columns added:
    - vsa_signal: str label of detected pattern (or None)
    - vsa_bias: 'bullish', 'bearish', or None
    - spread_ratio: candle spread vs rolling average
    - vol_ratio: candle volume vs rolling average
    - close_position: where close sits in high-low range (0=bottom, 1=top)
    """
    if params is None:
        params = VSAParams()

    df = df.copy()

    df["spread"] = df["high"] - df["low"]
    df["close_pos"] = (df["close"] - df["low"]) / df["spread"].replace(0, np.nan)
    df["vol_ma"] = df["volume"].rolling(params.volume_ma_period).mean()
    df["spread_ma"] = df["spread"].rolling(params.spread_ma_period).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"]
    df["spread_ratio"] = df["spread"] / df["spread_ma"]

    signals = []
    biases = []

    for i, row in df.iterrows():
        vr = row["vol_ratio"]
        sr = row["spread_ratio"]
        cp = row["close_pos"]

        sig = None
        bias = None

        if pd.isna(vr) or pd.isna(sr) or pd.isna(cp):
            signals.append(sig)
            biases.append(bias)
            continue

        is_bull_candle = row["close"] > row["open"]
        is_bear_candle = row["close"] < row["open"]
        low_vol = vr < params.low_vol_threshold
        high_vol = vr > params.high_vol_threshold
        ultra_vol = vr > params.ultra_high_vol_threshold
        narrow_spread = sr < params.narrow_spread_threshold
        wide_spread = sr > params.wide_spread_threshold

        # No Demand: narrow spread, low volume, close in upper half on UP candle → bearish (no real buying)
        if narrow_spread and low_vol and is_bull_candle and cp > 0.5:
            sig = "no_demand"
            bias = "bearish"

        # No Supply: narrow spread, low volume, close in lower half on DOWN candle → bullish (no real selling)
        elif narrow_spread and low_vol and is_bear_candle and cp < 0.5:
            sig = "no_supply"
            bias = "bullish"

        # Stopping Volume: very high volume, narrow-ish spread, close in upper half after down move → bullish
        elif high_vol and narrow_spread and cp > 0.6:
            sig = "stopping_volume"
            bias = "bullish"

        # Climax Up: ultra-high volume, wide spread, close near top → potential reversal (bearish)
        elif ultra_vol and wide_spread and cp > 0.7:
            sig = "climax_up"
            bias = "bearish"

        # Climax Down: ultra-high volume, wide spread, close near bottom → potential reversal (bullish)
        elif ultra_vol and wide_spread and cp < 0.3:
            sig = "climax_down"
            bias = "bullish"

        # Effort vs Result (No Result): high volume but narrow spread → hidden weakness/strength
        elif high_vol and narrow_spread and is_bear_candle:
            sig = "effort_no_result_bear"
            bias = "bullish"  # sellers couldn't push price down despite high vol

        elif high_vol and narrow_spread and is_bull_candle:
            sig = "effort_no_result_bull"
            bias = "bearish"  # buyers couldn't push price up despite high vol

        signals.append(sig)
        biases.append(bias)

    df["vsa_signal"] = signals
    df["vsa_bias"] = biases

    return df


def vsa_confirms_smc(
    df_vsa: pd.DataFrame,
    signal_idx: int,
    direction: Literal["long", "short"],
    lookback: int = 3,
) -> bool:
    """
    Check if any VSA signal in the last `lookback` candles confirms the SMC direction.
    Long needs bullish VSA, Short needs bearish VSA.
    """
    start = max(0, signal_idx - lookback)
    window = df_vsa.iloc[start : signal_idx + 1]
    needed_bias = "bullish" if direction == "long" else "bearish"
    return (window["vsa_bias"] == needed_bias).any()
