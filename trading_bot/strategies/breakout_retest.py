"""
Option B — Breakout + Retest Strategy
Simpler, more mechanical, better for algorithmic backtesting.

Logic:
1. Identify key swing levels (resistance/support)
2. Detect a VALID breakout: close beyond level with volume spike
3. Wait for RETEST: price returns to the broken level
4. Entry on REJECTION candle from the retested level
5. SL beyond the retest low/high, TP at 2R minimum

Why this works better algorithmically:
- Clear, binary conditions (closed above level? yes/no)
- No discretionary judgment needed
- Works in trending AND post-breakout markets
- Volume confirmation reduces false breakouts
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BRParams:
    swing_lookback: int = 10          # candles to identify swing levels
    breakout_vol_ratio: float = 1.3   # volume must be Nx avg for valid breakout
    retest_tolerance_pct: float = 0.003  # how close price must come to level
    confirm_body_pct: float = 0.5     # rejection candle body must be >= 50% of range
    rr_ratio: float = 2.0
    sl_buffer_pct: float = 0.001
    max_retest_wait: int = 20         # max candles to wait for retest after breakout
    ema_trend_filter: bool = True
    ema_period: int = 50
    use_kill_zones: bool = True
    kill_zones: tuple = ((7, 10), (13, 16), (18, 21))


@dataclass
class BRLevel:
    idx: int
    timestamp: pd.Timestamp
    price: float
    kind: str    # "resistance" or "support"
    broken: bool = False
    broken_idx: int | None = None
    retested: bool = False


def detect_levels(df: pd.DataFrame, lookback: int = 10) -> list[BRLevel]:
    """Detect swing highs (resistance) and swing lows (support)."""
    levels = []
    n = len(df)
    for i in range(lookback, n - lookback):
        window_h = df["high"].iloc[i - lookback: i + lookback + 1]
        window_l = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window_h.max():
            levels.append(BRLevel(i, df.index[i], df["high"].iloc[i], "resistance"))
        if df["low"].iloc[i] == window_l.min():
            levels.append(BRLevel(i, df.index[i], df["low"].iloc[i], "support"))
    return levels


def run_breakout_retest(
    df: pd.DataFrame,
    params: BRParams | None = None,
) -> list[dict]:
    if params is None:
        params = BRParams()

    n = len(df)
    vol_ma = df["volume"].rolling(20).mean()
    ema = df["close"].ewm(span=params.ema_period, adjust=False).mean()
    avg_body = (df["close"] - df["open"]).abs().rolling(20).mean()

    levels = detect_levels(df, params.swing_lookback)
    signals = []
    used_levels: set[int] = set()

    for level in levels:
        if id(level) in used_levels:
            continue

        li = level.idx

        # ── Scan for breakout ───────────────────────────────────────────────
        for i in range(li + 1, min(li + 100, n - 1)):
            close = df["close"].iloc[i]
            vol = df["volume"].iloc[i]
            vol_avg = vol_ma.iloc[i] if not pd.isna(vol_ma.iloc[i]) else vol

            # Bullish breakout: close above resistance with volume
            if (level.kind == "resistance"
                    and close > level.price
                    and vol >= vol_avg * params.breakout_vol_ratio):
                level.broken = True
                level.broken_idx = i

                # ── Wait for retest ─────────────────────────────────────────
                for j in range(i + 1, min(i + params.max_retest_wait + 1, n - 1)):
                    low_j = df["low"].iloc[j]
                    close_j = df["close"].iloc[j]
                    ema_j = ema.iloc[j]
                    hour_utc = df.index[j].hour

                    # Kill zone check
                    if params.use_kill_zones:
                        if not any(s <= hour_utc < e for s, e in params.kill_zones):
                            continue

                    # EMA trend filter: only long above EMA
                    if params.ema_trend_filter and close_j < ema_j:
                        continue

                    # Price retested the broken level
                    if abs(low_j - level.price) / level.price <= params.retest_tolerance_pct:
                        # Rejection confirmation: close above level + strong body
                        body = abs(close_j - df["open"].iloc[j])
                        candle_range = df["high"].iloc[j] - low_j
                        body_pct = body / candle_range if candle_range > 0 else 0

                        if close_j > level.price and body_pct >= params.confirm_body_pct:
                            sl = level.price * (1 - params.sl_buffer_pct)
                            risk = close_j - sl
                            if risk <= 0 or risk / close_j > 0.05:
                                continue
                            tp = close_j + risk * params.rr_ratio

                            signals.append({
                                "idx": j,
                                "timestamp": df.index[j],
                                "direction": "long",
                                "entry": round(close_j, 4),
                                "sl": round(sl, 4),
                                "tp": round(tp, 4),
                                "risk_pct": round(risk / close_j * 100, 3),
                                "rr": params.rr_ratio,
                                "trigger": "breakout_retest",
                                "level": round(level.price, 4),
                            })
                            level.retested = True
                            used_levels.add(id(level))
                            break
                break  # only first breakout per level

            # Bearish breakout: close below support with volume
            elif (level.kind == "support"
                    and close < level.price
                    and vol >= vol_avg * params.breakout_vol_ratio):
                level.broken = True
                level.broken_idx = i

                for j in range(i + 1, min(i + params.max_retest_wait + 1, n - 1)):
                    high_j = df["high"].iloc[j]
                    close_j = df["close"].iloc[j]
                    ema_j = ema.iloc[j]
                    hour_utc = df.index[j].hour

                    if params.use_kill_zones:
                        if not any(s <= hour_utc < e for s, e in params.kill_zones):
                            continue

                    if params.ema_trend_filter and close_j > ema_j:
                        continue

                    if abs(high_j - level.price) / level.price <= params.retest_tolerance_pct:
                        body = abs(close_j - df["open"].iloc[j])
                        candle_range = high_j - df["low"].iloc[j]
                        body_pct = body / candle_range if candle_range > 0 else 0

                        if close_j < level.price and body_pct >= params.confirm_body_pct:
                            sl = level.price * (1 + params.sl_buffer_pct)
                            risk = sl - close_j
                            if risk <= 0 or risk / close_j > 0.05:
                                continue
                            tp = close_j - risk * params.rr_ratio

                            signals.append({
                                "idx": j,
                                "timestamp": df.index[j],
                                "direction": "short",
                                "entry": round(close_j, 4),
                                "sl": round(sl, 4),
                                "tp": round(tp, 4),
                                "risk_pct": round(risk / close_j * 100, 3),
                                "rr": params.rr_ratio,
                                "trigger": "breakout_retest",
                                "level": round(level.price, 4),
                            })
                            level.retested = True
                            used_levels.add(id(level))
                            break
                break

    signals.sort(key=lambda x: x["idx"])
    return signals
