"""
SMC / ICT Strategy Engine
Detects: Market Structure (BOS/CHoCH), Order Blocks, Fair Value Gaps,
Liquidity Sweeps, Premium/Discount zones, then generates trade signals.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────── Data structures ────────────────────────────────

@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["HH", "HL", "LH", "LL", "H", "L"]


@dataclass
class OrderBlock:
    start_idx: int
    timestamp: pd.Timestamp
    ob_high: float
    ob_low: float
    kind: Literal["bullish", "bearish"]
    mitigated: bool = False
    mitigated_idx: int | None = None


@dataclass
class FairValueGap:
    start_idx: int
    timestamp: pd.Timestamp
    gap_high: float
    gap_low: float
    kind: Literal["bullish", "bearish"]
    filled: bool = False
    filled_idx: int | None = None


@dataclass
class LiquiditySweep:
    sweep_idx: int
    timestamp: pd.Timestamp
    swept_level: float
    direction: Literal["bullish_sweep", "bearish_sweep"]  # bullish = swept lows (buy signal)


@dataclass
class StructureBreak:
    idx: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["BOS_bull", "BOS_bear", "CHoCH_bull", "CHoCH_bear"]


@dataclass
class SMCAnalysis:
    swing_highs: list[SwingPoint] = field(default_factory=list)
    swing_lows: list[SwingPoint] = field(default_factory=list)
    order_blocks: list[OrderBlock] = field(default_factory=list)
    fvgs: list[FairValueGap] = field(default_factory=list)
    sweeps: list[LiquiditySweep] = field(default_factory=list)
    structure_breaks: list[StructureBreak] = field(default_factory=list)
    signals: list[dict] = field(default_factory=list)


# ─────────────────────────── Parameters ─────────────────────────────────────

@dataclass
class SMCParams:
    swing_lookback: int = 5        # candles each side to confirm swing point
    ob_lookback: int = 3           # candles before impulse to scan for OB
    fvg_min_gap_pct: float = 0.001 # min FVG size as % of price
    sweep_threshold_pct: float = 0.002  # how far price must pierce level
    sl_ob_buffer_pct: float = 0.001     # SL = beyond OB edge + buffer
    rr_ratio: float = 2.0               # minimum risk:reward
    confluence_require_fvg: bool = False
    confluence_require_choch: bool = False
    # ── Displacement filter (the missing ICT ingredient) ──────────────────────
    require_displacement: bool = True
    displacement_body_ratio: float = 1.5   # displacement candle body must be Nx avg body
    displacement_avg_period: int = 20       # rolling avg body period for comparison
    # ── Other filters ─────────────────────────────────────────────────────────
    ema_trend_filter: bool = True
    ema_period: int = 50
    require_candle_confirm: bool = True
    min_ob_touches: int = 0
    # ── ICT Kill Zones (UTC hours) ────────────────────────────────────────────
    use_kill_zones: bool = True
    # (start_hour, end_hour) inclusive, UTC
    kill_zones: tuple = ((7, 10), (13, 16), (18, 21))


# ────────────────────────── Core detection functions ─────────────────────────

def detect_swings(df: pd.DataFrame, lookback: int = 5) -> tuple[list, list]:
    highs, lows = [], []
    n = len(df)
    for i in range(lookback, n - lookback):
        window_h = df["high"].iloc[i - lookback: i + lookback + 1]
        window_l = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window_h.max():
            highs.append(SwingPoint(i, df.index[i], df["high"].iloc[i], "H"))
        if df["low"].iloc[i] == window_l.min():
            lows.append(SwingPoint(i, df.index[i], df["low"].iloc[i], "L"))
    return highs, lows


def label_structure(swing_highs: list, swing_lows: list) -> tuple[list, list]:
    """Classify swings as HH/HL/LH/LL based on sequence."""
    labeled_h, labeled_l = [], []

    for i, sh in enumerate(swing_highs):
        if i == 0:
            sh.kind = "H"
        else:
            sh.kind = "HH" if sh.price > swing_highs[i - 1].price else "LH"
        labeled_h.append(sh)

    for i, sl in enumerate(swing_lows):
        if i == 0:
            sl.kind = "L"
        else:
            sl.kind = "HL" if sl.price > swing_lows[i - 1].price else "LL"
        labeled_l.append(sl)

    return labeled_h, labeled_l


def detect_structure_breaks(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> list[StructureBreak]:
    breaks = []
    n = len(df)

    # BOS bullish: close above previous swing high
    for sh in swing_highs:
        for i in range(sh.index + 1, n):
            if df["close"].iloc[i] > sh.price:
                kind = "BOS_bull" if sh.kind in ("HH", "H") else "CHoCH_bull"
                breaks.append(StructureBreak(i, df.index[i], sh.price, kind))
                break

    # BOS bearish: close below previous swing low
    for sl in swing_lows:
        for i in range(sl.index + 1, n):
            if df["close"].iloc[i] < sl.price:
                kind = "BOS_bear" if sl.kind in ("LL", "L") else "CHoCH_bear"
                breaks.append(StructureBreak(i, df.index[i], sl.price, kind))
                break

    breaks.sort(key=lambda x: x.idx)
    return breaks


def _displacement_is_strong(
    df: pd.DataFrame,
    ob_idx: int,
    sb_idx: int,
    body_ratio: float = 1.5,
    avg_period: int = 20,
) -> bool:
    """
    True if at least one candle in the move from OB to structure break
    has a body >= body_ratio * rolling average body size.
    This is what separates a real ICT displacement from random noise.
    """
    avg_start = max(0, ob_idx - avg_period)
    bodies = (df["close"] - df["open"]).abs().iloc[avg_start:ob_idx]
    if len(bodies) == 0 or bodies.mean() == 0:
        return True  # can't measure — don't block

    avg_body = bodies.mean()
    for i in range(ob_idx, min(ob_idx + 6, sb_idx + 1)):
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        if body >= avg_body * body_ratio:
            return True
    return False


def detect_order_blocks(
    df: pd.DataFrame,
    structure_breaks: list[StructureBreak],
    lookback: int = 3,
    require_displacement: bool = True,
    displacement_body_ratio: float = 1.5,
    displacement_avg_period: int = 20,
) -> list[OrderBlock]:
    obs = []
    for sb in structure_breaks:
        if sb.kind in ("BOS_bull", "CHoCH_bull"):
            # Only keep if the move TO this structure break was a real displacement
            if require_displacement:
                strong = _displacement_is_strong(
                    df, max(0, sb.idx - 6), sb.idx,
                    displacement_body_ratio, displacement_avg_period,
                )
                if not strong:
                    continue

            start = max(0, sb.idx - lookback - 1)
            for j in range(sb.idx - 1, start, -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    obs.append(OrderBlock(
                        start_idx=j,
                        timestamp=df.index[j],
                        ob_high=df["high"].iloc[j],
                        ob_low=df["low"].iloc[j],
                        kind="bullish",
                    ))
                    break

        elif sb.kind in ("BOS_bear", "CHoCH_bear"):
            if require_displacement:
                strong = _displacement_is_strong(
                    df, max(0, sb.idx - 6), sb.idx,
                    displacement_body_ratio, displacement_avg_period,
                )
                if not strong:
                    continue

            start = max(0, sb.idx - lookback - 1)
            for j in range(sb.idx - 1, start, -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    obs.append(OrderBlock(
                        start_idx=j,
                        timestamp=df.index[j],
                        ob_high=df["high"].iloc[j],
                        ob_low=df["low"].iloc[j],
                        kind="bearish",
                    ))
                    break
    return obs


def detect_fvgs(df: pd.DataFrame, min_gap_pct: float = 0.001) -> list[FairValueGap]:
    fvgs = []
    for i in range(1, len(df) - 1):
        c1_high = df["high"].iloc[i - 1]
        c1_low = df["low"].iloc[i - 1]
        c3_high = df["high"].iloc[i + 1]
        c3_low = df["low"].iloc[i + 1]
        mid = df["close"].iloc[i]

        # Bullish FVG: gap between c1 high and c3 low (price moved up fast)
        if c3_low > c1_high:
            gap_size = (c3_low - c1_high) / mid
            if gap_size >= min_gap_pct:
                fvgs.append(FairValueGap(
                    start_idx=i - 1,
                    timestamp=df.index[i - 1],
                    gap_high=c3_low,
                    gap_low=c1_high,
                    kind="bullish",
                ))

        # Bearish FVG: gap between c3 high and c1 low (price moved down fast)
        if c1_low > c3_high:
            gap_size = (c1_low - c3_high) / mid
            if gap_size >= min_gap_pct:
                fvgs.append(FairValueGap(
                    start_idx=i - 1,
                    timestamp=df.index[i - 1],
                    gap_high=c1_low,
                    gap_low=c3_high,
                    kind="bearish",
                ))
    return fvgs


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    threshold_pct: float = 0.002,
) -> list[LiquiditySweep]:
    sweeps = []

    # Sweep of swing highs (bearish wick above then rejection = bearish sweep)
    for sh in swing_highs:
        for i in range(sh.index + 1, min(sh.index + 20, len(df))):
            if df["high"].iloc[i] > sh.price * (1 + threshold_pct):
                if df["close"].iloc[i] < sh.price:
                    sweeps.append(LiquiditySweep(
                        sweep_idx=i,
                        timestamp=df.index[i],
                        swept_level=sh.price,
                        direction="bearish_sweep",
                    ))
                break

    # Sweep of swing lows (bullish wick below then rejection = bullish sweep)
    for sl in swing_lows:
        for i in range(sl.index + 1, min(sl.index + 20, len(df))):
            if df["low"].iloc[i] < sl.price * (1 - threshold_pct):
                if df["close"].iloc[i] > sl.price:
                    sweeps.append(LiquiditySweep(
                        sweep_idx=i,
                        timestamp=df.index[i],
                        swept_level=sl.price,
                        direction="bullish_sweep",
                    ))
                break

    sweeps.sort(key=lambda x: x.sweep_idx)
    return sweeps


def mark_mitigated(df: pd.DataFrame, obs: list[OrderBlock]) -> None:
    """Mark OBs as mitigated when price trades through them after formation."""
    for ob in obs:
        for i in range(ob.start_idx + 1, len(df)):
            if ob.kind == "bullish" and df["low"].iloc[i] <= ob.ob_low:
                ob.mitigated = True
                ob.mitigated_idx = i
                break
            elif ob.kind == "bearish" and df["high"].iloc[i] >= ob.ob_high:
                ob.mitigated = True
                ob.mitigated_idx = i
                break


def mark_fvg_filled(df: pd.DataFrame, fvgs: list[FairValueGap]) -> None:
    for fvg in fvgs:
        for i in range(fvg.start_idx + 2, len(df)):
            if fvg.kind == "bullish" and df["low"].iloc[i] <= fvg.gap_low:
                fvg.filled = True
                fvg.filled_idx = i
                break
            elif fvg.kind == "bearish" and df["high"].iloc[i] >= fvg.gap_high:
                fvg.filled = True
                fvg.filled_idx = i
                break


# ────────────────────────── Signal generation ────────────────────────────────

def generate_signals(
    df: pd.DataFrame,
    obs: list[OrderBlock],
    fvgs: list[FairValueGap],
    sweeps: list[LiquiditySweep],
    structure_breaks: list[StructureBreak],
    params: SMCParams,
    trigger_lookback: int = 40,
) -> list[dict]:
    """
    ICT/SMC entry logic (OB-first approach):
    For each un-mitigated OB, when price touches it:
    - Check if a matching displacement (sweep / CHoCH / BOS) occurred in
      the last `trigger_lookback` candles before this touch.
    - If yes → generate a signal at the OB touch candle.
    This matches real ICT methodology: displacement creates the context,
    then you WAIT for price to return to the OB.
    """
    signals = []
    n = len(df)
    triggered_obs: set[int] = set()

    # Pre-compute EMA for trend filter
    ema = df["close"].ewm(span=params.ema_period, adjust=False).mean()

    # Build quick-lookup sets of event indices with direction
    bull_event_idxs: set[int] = set()
    bear_event_idxs: set[int] = set()
    event_name: dict[int, str] = {}

    for sw in sweeps:
        if sw.direction == "bullish_sweep":
            bull_event_idxs.add(sw.sweep_idx)
            event_name[sw.sweep_idx] = "sweep"
        else:
            bear_event_idxs.add(sw.sweep_idx)
            event_name[sw.sweep_idx] = "sweep"

    for sb in structure_breaks:
        if sb.kind in ("CHoCH_bull", "BOS_bull"):
            bull_event_idxs.add(sb.idx)
            event_name[sb.idx] = sb.kind
        elif sb.kind in ("CHoCH_bear", "BOS_bear"):
            bear_event_idxs.add(sb.idx)
            event_name[sb.idx] = sb.kind

    for i in range(trigger_lookback, n - 1):
        price = df["close"].iloc[i]
        low_i = df["low"].iloc[i]
        high_i = df["high"].iloc[i]
        ema_val = ema.iloc[i]
        ts = df.index[i]

        # ── Kill Zone filter ─────────────────────────────────────────────────
        if params.use_kill_zones:
            hour_utc = ts.hour if hasattr(ts, "hour") else ts.to_pydatetime().hour
            in_kz = any(start <= hour_utc < end for start, end in params.kill_zones)
            if not in_kz:
                continue  # skip candles outside active trading sessions

        # Check if a bullish event occurred in the recent lookback window
        window_bull = [
            idx for idx in bull_event_idxs
            if i - trigger_lookback <= idx < i
        ]
        window_bear = [
            idx for idx in bear_event_idxs
            if i - trigger_lookback <= idx < i
        ]

        # ── LONG: price touches bullish OB + recent bullish event ──
        if window_bull:
            # EMA trend filter: only go long when price is above EMA (uptrend)
            if params.ema_trend_filter and price < ema_val:
                pass  # skip longs in downtrend
            else:
                latest_bull_event = max(window_bull)
                trigger_name = event_name.get(latest_bull_event, "event")

                for ob in obs:
                    already_mitigated = ob.mitigated and (ob.mitigated_idx is not None) and ob.mitigated_idx <= i
                    if (ob.kind != "bullish"
                            or id(ob) in triggered_obs
                            or already_mitigated
                            or ob.start_idx >= latest_bull_event):
                        continue
                    # Price is touching or wicking into the OB zone
                    touch = low_i <= ob.ob_high * 1.002 and price >= ob.ob_low * 0.998
                    if not touch:
                        continue

                    # Candle confirmation: close must be ABOVE OB midpoint (rejection candle)
                    if params.require_candle_confirm:
                        ob_mid = (ob.ob_high + ob.ob_low) / 2
                        if price < ob_mid:
                            continue  # closed too deep inside OB = not a clean rejection

                    if params.confluence_require_fvg:
                        has_fvg = any(
                            fvg.kind == "bullish"
                            and not fvg.filled
                            and fvg.start_idx < i
                            for fvg in fvgs
                        )
                        if not has_fvg:
                            continue

                    sl = ob.ob_low * (1 - params.sl_ob_buffer_pct)
                    risk = price - sl
                    if risk <= 0 or risk / price > 0.04:
                        continue
                    tp = price + risk * params.rr_ratio

                    signals.append({
                        "idx": i,
                        "timestamp": df.index[i],
                        "direction": "long",
                        "entry": round(price, 4),
                        "sl": round(sl, 4),
                        "tp": round(tp, 4),
                        "risk_pct": round(risk / price * 100, 3),
                        "rr": params.rr_ratio,
                        "trigger": f"{trigger_name}+OB",
                        "ob_high": ob.ob_high,
                        "ob_low": ob.ob_low,
                    })
                    triggered_obs.add(id(ob))
                    break

        # ── SHORT: price touches bearish OB + recent bearish event ──
        if window_bear:
            # EMA trend filter: only go short when price is below EMA (downtrend)
            if params.ema_trend_filter and price > ema_val:
                pass  # skip shorts in uptrend
            else:
                latest_bear_event = max(window_bear)
                trigger_name = event_name.get(latest_bear_event, "event")

                for ob in obs:
                    already_mitigated = ob.mitigated and (ob.mitigated_idx is not None) and ob.mitigated_idx <= i
                    if (ob.kind != "bearish"
                            or id(ob) in triggered_obs
                            or already_mitigated
                            or ob.start_idx >= latest_bear_event):
                        continue
                    touch = high_i >= ob.ob_low * 0.998 and price <= ob.ob_high * 1.002
                    if not touch:
                        continue

                    # Candle confirmation: close must be BELOW OB midpoint (rejection from above)
                    if params.require_candle_confirm:
                        ob_mid = (ob.ob_high + ob.ob_low) / 2
                        if price > ob_mid:
                            continue  # closed too deep inside OB = not a clean rejection

                    if params.confluence_require_fvg:
                        has_fvg = any(
                            fvg.kind == "bearish"
                            and not fvg.filled
                            and fvg.start_idx < i
                            for fvg in fvgs
                        )
                        if not has_fvg:
                            continue

                    sl = ob.ob_high * (1 + params.sl_ob_buffer_pct)
                    risk = sl - price
                    if risk <= 0 or risk / price > 0.04:
                        continue
                    tp = price - risk * params.rr_ratio

                    signals.append({
                        "idx": i,
                        "timestamp": df.index[i],
                        "direction": "short",
                        "entry": round(price, 4),
                        "sl": round(sl, 4),
                        "tp": round(tp, 4),
                        "risk_pct": round(risk / price * 100, 3),
                        "rr": params.rr_ratio,
                        "trigger": f"{trigger_name}+OB",
                        "ob_high": ob.ob_high,
                        "ob_low": ob.ob_low,
                    })
                    triggered_obs.add(id(ob))
                    break

    return signals


# ─────────────────────────── Main run function ───────────────────────────────

def run_smc_analysis(df: pd.DataFrame, params: SMCParams | None = None) -> SMCAnalysis:
    if params is None:
        params = SMCParams()

    result = SMCAnalysis()

    sh, sl = detect_swings(df, params.swing_lookback)
    sh, sl = label_structure(sh, sl)
    result.swing_highs = sh
    result.swing_lows = sl

    result.structure_breaks = detect_structure_breaks(df, sh, sl)
    result.order_blocks = detect_order_blocks(
        df, result.structure_breaks, params.ob_lookback,
        require_displacement=params.require_displacement,
        displacement_body_ratio=params.displacement_body_ratio,
        displacement_avg_period=params.displacement_avg_period,
    )
    result.fvgs = detect_fvgs(df, params.fvg_min_gap_pct)
    result.sweeps = detect_liquidity_sweeps(df, sh, sl, params.sweep_threshold_pct)

    mark_mitigated(df, result.order_blocks)
    mark_fvg_filled(df, result.fvgs)

    result.signals = generate_signals(
        df, result.order_blocks, result.fvgs, result.sweeps,
        result.structure_breaks, params
    )

    return result
