"""
Synthetic OHLCV data generator — realistic BTC-like price action with trends,
consolidations, and volume patterns that allow ICT/SMC/VSA signals to emerge.
Used when running without internet; real data comes from fetcher.py.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_ohlcv(
    n_candles: int = 1000,
    start_price: float = 42000.0,
    timeframe_minutes: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- price simulation: trending + mean-reverting regimes ---
    log_returns = []
    vol = 0.008          # hourly volatility ~0.8%
    drift = 0.0001

    regime = "trend_up"
    regime_len = 0
    regime_max = rng.integers(30, 120)
    trend_dir = 1.0

    for _ in range(n_candles):
        regime_len += 1
        if regime_len >= regime_max:
            regime_len = 0
            regime_max = rng.integers(30, 120)
            choices = ["trend_up", "trend_down", "range"]
            probs = [0.35, 0.35, 0.30]
            regime = rng.choice(choices, p=probs)
            trend_dir = 1.0 if regime == "trend_up" else -1.0

        if regime == "range":
            r = rng.normal(0, vol * 0.6)
        else:
            r = rng.normal(drift * trend_dir * 2, vol)

        log_returns.append(r)

    # close prices from log returns
    log_prices = np.cumsum([np.log(start_price)] + log_returns)
    closes = np.exp(log_prices)

    rows = []
    ts = datetime(2024, 1, 1)
    for i in range(n_candles):
        c = closes[i]
        o = closes[i - 1] if i > 0 else c * (1 + rng.normal(0, 0.002))

        spread = abs(rng.normal(0, vol)) * c
        spread = max(spread, c * 0.001)

        hi = max(o, c) + abs(rng.normal(0, spread * 0.6))
        lo = min(o, c) - abs(rng.normal(0, spread * 0.6))

        # wicks
        hi += rng.exponential(spread * 0.3)
        lo -= rng.exponential(spread * 0.3)

        # volume: higher on big moves, clustered in trends
        base_vol = rng.lognormal(10, 0.5)
        move_factor = 1 + abs(c - o) / (c * 0.005) * 2
        vol_bar = base_vol * move_factor

        rows.append({
            "timestamp": ts,
            "open": round(o, 2),
            "high": round(hi, 2),
            "low": round(lo, 2),
            "close": round(c, 2),
            "volume": round(vol_bar, 4),
        })
        ts += timedelta(minutes=timeframe_minutes)

    df = pd.DataFrame(rows).set_index("timestamp")
    return df
