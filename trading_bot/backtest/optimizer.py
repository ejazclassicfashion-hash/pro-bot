"""
Strategy Optimizer — MTF-aware
Grid-searches parameter combinations, applies 4H bias filter on each run.
Walk-forward uses 3 splits (more data per window = more trades = reliable stats).
"""

import itertools
import pandas as pd
from dataclasses import dataclass
from .engine import run_backtest, BacktestConfig
from ..strategies.combined import run_combined, CombinedParams
from ..strategies.smc_ict import SMCParams
from ..strategies.mtf import compute_4h_bias, map_bias_to_1h, filter_signals_by_bias, MTFParams


@dataclass
class OptimizeResult:
    params: dict
    stats: dict
    rank_score: float


def _build_signals(df_1h: pd.DataFrame, p: dict, df_4h: pd.DataFrame | None = None) -> list[dict]:
    smc_params = SMCParams(
        swing_lookback=p.get("swing_lookback", 5),
        rr_ratio=p.get("rr_ratio", 2.0),
        confluence_require_fvg=p.get("confluence_require_fvg", False),
        ema_trend_filter=True,
        ema_period=p.get("ema_period", 50),
        require_candle_confirm=p.get("require_candle_confirm", True),
    )
    _, _, signals = run_combined(df_1h, CombinedParams(smc=smc_params, require_vsa_confirm=False))

    if df_4h is not None and len(signals) > 0:
        bias_4h = compute_4h_bias(df_4h, MTFParams())
        bias_1h = map_bias_to_1h(bias_4h, df_1h.index)
        signals = filter_signals_by_bias(signals, bias_1h)

    return signals


def optimize(
    df: pd.DataFrame,
    param_grid: dict | None = None,
    backtest_config: BacktestConfig | None = None,
    top_n: int = 5,
    df_4h: pd.DataFrame | None = None,
) -> list[OptimizeResult]:
    if param_grid is None:
        param_grid = {
            "swing_lookback": [5, 7, 10],
            "rr_ratio": [1.5, 2.0, 2.5],
            "ema_period": [50, 100],
            "require_candle_confirm": [True, False],
            "confluence_require_fvg": [True, False],
        }

    if backtest_config is None:
        backtest_config = BacktestConfig()

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    print(f"[Optimizer] Testing {len(combos)} parameter combinations...")

    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        try:
            signals = _build_signals(df, p, df_4h)
            if not signals:
                continue
            result = run_backtest(df, signals, backtest_config)
            stats = result.stats
            if "error" in stats or stats["total_trades"] < 5:
                continue

            dd_penalty = max(0, abs(stats["max_drawdown_pct"]) - 15) * 3
            score = (
                stats["total_return_pct"] * 0.4
                + stats["profit_factor"] * 15
                + stats["win_rate_pct"] * 0.3
                - dd_penalty
            )
            results.append(OptimizeResult(params=p, stats=stats, rank_score=round(score, 3)))
        except Exception:
            continue

    results.sort(key=lambda x: x.rank_score, reverse=True)
    print(f"[Optimizer] Done. {len(results)} valid results.")
    return results[:top_n]


def walk_forward_test(
    df: pd.DataFrame,
    best_params: dict,
    n_splits: int = 3,
    train_pct: float = 0.6,
    df_4h: pd.DataFrame | None = None,
) -> list[dict]:
    """
    3 splits instead of 5 — more candles per test window = more trades = real signal.
    60/40 train/test split — gives a larger test window.
    """
    results = []
    chunk = len(df) // n_splits

    for i in range(n_splits):
        start = i * chunk
        end = start + chunk
        if end > len(df):
            break

        split = int((end - start) * train_pct)
        test_df = df.iloc[start + split: end]

        # Slice 4H to the same time window if provided
        test_4h = None
        if df_4h is not None:
            test_start = test_df.index[0]
            test_end = test_df.index[-1]
            test_4h = df_4h[(df_4h.index >= test_start) & (df_4h.index <= test_end)]
            if len(test_4h) < 10:
                test_4h = None

        if len(test_df) < 200:
            results.append({"split": i, "note": "window too small"})
            continue

        try:
            signals = _build_signals(test_df, best_params, test_4h)
            if not signals:
                results.append({"split": i, "total_trades": 0, "total_return_pct": "N/A",
                                "win_rate_pct": "N/A", "note": "no signals in window"})
                continue
            bt = run_backtest(test_df, signals)
            results.append({"split": i, **bt.stats})
        except Exception as e:
            results.append({"split": i, "error": str(e)})

    return results
