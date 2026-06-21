"""
Multi-pair portfolio runner.
Runs the full SMC + MTF pipeline on each symbol independently,
then aggregates into a combined portfolio view.

Each pair gets equal capital allocation — default 1/N of total capital.
Signals from different pairs do NOT interfere with each other.
"""

import pandas as pd
from dataclasses import dataclass, field
from ..strategies.combined import run_combined, CombinedParams
from ..strategies.smc_ict import SMCParams, SMCAnalysis
from ..strategies.mtf import compute_4h_bias, map_bias_to_1h, filter_signals_by_bias, MTFParams
from ..strategies.breakout_retest import run_breakout_retest, BRParams
from ..backtest.engine import run_backtest, BacktestConfig, BacktestResult, Trade


@dataclass
class PairResult:
    symbol: str
    smc: SMCAnalysis
    df_vsa: pd.DataFrame
    signals_raw: int
    signals_after_mtf: int
    signals_after_kz: int
    result: BacktestResult


@dataclass
class PortfolioResult:
    pairs: list[PairResult] = field(default_factory=list)
    combined_equity: pd.Series = field(default_factory=pd.Series)
    combined_trades: list[Trade] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def run_portfolio(
    data_1h: dict[str, pd.DataFrame],
    data_4h: dict[str, pd.DataFrame],
    params: CombinedParams | None = None,
    capital_per_pair: float = 10_000.0,
    risk_per_trade_pct: float = 1.0,
    strategy: str = "smc",
    br_params: BRParams | None = None,
    br_params_per_pair: dict[str, BRParams] | None = None,  # per-symbol overrides
) -> PortfolioResult:
    """
    strategy="smc"  → SMC + ICT + 4H Bias + Kill Zones
    strategy="br"   → Breakout + Retest (simpler, more mechanical)
    """
    portfolio = PortfolioResult()
    all_equity_curves: list[pd.Series] = []

    for symbol, df_1h in data_1h.items():
        df_4h = data_4h.get(symbol)

        if strategy == "br":
            # Use per-pair params if provided, else global br_params, else default
            _br_params = (
                (br_params_per_pair or {}).get(symbol)
                or br_params
                or BRParams()
            )
            signals = run_breakout_retest(df_1h, _br_params)
            n_raw = len(signals)
            n_after_mtf = n_raw
            n_after_kz = n_raw
            # Create a dummy SMC analysis for chart compatibility
            from ..strategies.smc_ict import SMCAnalysis
            smc = SMCAnalysis()
            df_vsa = df_1h.copy()

        else:
            # Option A: SMC + MTF
            if params is None:
                params = CombinedParams(
                    smc=SMCParams(
                        swing_lookback=5, rr_ratio=2.0,
                        ema_trend_filter=True, ema_period=50,
                        require_candle_confirm=True,
                        use_kill_zones=True,
                    ),
                    require_vsa_confirm=False,
                )
            smc, df_vsa, raw_signals = run_combined(df_1h, params)
            n_raw = len(raw_signals)

            if df_4h is not None and len(df_4h) > 20:
                bias_4h = compute_4h_bias(df_4h, MTFParams())
                bias_1h = map_bias_to_1h(bias_4h, df_1h.index)
                signals = filter_signals_by_bias(raw_signals, bias_1h)
            else:
                signals = raw_signals

            n_after_mtf = len(signals)
            n_after_kz = n_after_mtf

        bt_config = BacktestConfig(
            initial_capital=capital_per_pair,
            risk_per_trade_pct=risk_per_trade_pct,
        )
        bt_result = run_backtest(df_1h, signals, bt_config)

        for t in bt_result.trades:
            t.__dict__["symbol"] = symbol

        pair_result = PairResult(
            symbol=symbol,
            smc=smc,
            df_vsa=df_vsa,
            signals_raw=n_raw,
            signals_after_mtf=n_after_mtf,
            signals_after_kz=n_after_kz,
            result=bt_result,
        )
        portfolio.pairs.append(pair_result)
        portfolio.combined_trades.extend(bt_result.trades)

        if not bt_result.equity_curve.empty:
            all_equity_curves.append(bt_result.equity_curve)

    # Combined equity = sum of all pair equity curves (reindexed to union of timestamps)
    if all_equity_curves:
        combined = pd.concat(all_equity_curves, axis=1).ffill().bfill()
        portfolio.combined_equity = combined.sum(axis=1)

    # Portfolio summary
    portfolio.summary = _portfolio_stats(portfolio, capital_per_pair)
    return portfolio


def _portfolio_stats(portfolio: PortfolioResult, capital_per_pair: float) -> dict:
    total_capital = capital_per_pair * len(portfolio.pairs)
    all_pnls = [t.pnl_usd for t in portfolio.combined_trades if t.pnl_usd is not None]
    if not all_pnls:
        return {"error": "No trades across all pairs"}

    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    total_return = sum(all_pnls) / total_capital * 100

    per_pair = {}
    for pr in portfolio.pairs:
        s = pr.result.stats
        per_pair[pr.symbol] = {
            "signals_raw": pr.signals_raw,
            "signals_final": pr.signals_after_kz,
            "trades": s.get("total_trades", 0),
            "win_rate": s.get("win_rate_pct", 0),
            "profit_factor": s.get("profit_factor", 0),
            "return_pct": s.get("total_return_pct", 0),
            "max_dd": s.get("max_drawdown_pct", 0),
        }

    import numpy as np
    avg_win = round(sum(wins) / len(wins), 2) if wins else 0
    avg_loss = round(abs(sum(losses) / len(losses)), 2) if losses else 1
    final_equity = round(total_capital + sum(all_pnls), 2)

    # Drawdown from combined equity
    eq = portfolio.combined_equity
    max_dd = 0.0
    sharpe = 0.0
    if not eq.empty:
        max_dd = round((eq / eq.cummax() - 1).min() * 100, 2)
        ret = eq.pct_change().dropna()
        sharpe = round((ret.mean() / ret.std()) * (8760 ** 0.5), 2) if ret.std() > 0 else 0

    return {
        "total_trades": len(all_pnls),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(all_pnls) * 100, 1) if all_pnls else 0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses else 999,
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "rr_actual": round(avg_win / avg_loss, 2) if avg_loss else 0,
        "final_equity": final_equity,
        "total_capital": total_capital,
        "final_value": final_equity,
        "per_pair": per_pair,
    }
