"""
Backtesting Engine
Simulates trade execution from a signal list on OHLCV data.
Tracks equity, drawdown, per-trade stats.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    risk_per_trade_pct: float = 1.0       # % of current equity risked per trade
    max_open_trades: int = 1              # max simultaneous positions
    commission_pct: float = 0.04         # 0.04% per side (Binance taker)
    slippage_pct: float = 0.01           # 0.01% slippage on entry


@dataclass
class Trade:
    signal_idx: int
    entry_idx: int
    entry_time: pd.Timestamp
    direction: str
    entry_price: float
    sl: float
    tp: float
    size_usd: float
    risk_usd: float
    trigger: str
    exit_idx: int | None = None
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_usd: float | None = None
    pnl_pct: float | None = None
    duration_bars: int | None = None


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    stats: dict = field(default_factory=dict)


def run_backtest(
    df: pd.DataFrame,
    signals: list[dict],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    if config is None:
        config = BacktestConfig()

    equity = config.initial_capital
    equity_curve = []
    timestamps = []
    open_trades: list[Trade] = []
    closed_trades: list[Trade] = []
    used_signal_idxs: set[int] = set()

    for i in range(len(df)):
        ts = df.index[i]
        bar_high = df["high"].iloc[i]
        bar_low = df["low"].iloc[i]

        # --- Check open trades for exit ---
        still_open = []
        for trade in open_trades:
            hit_sl = hit_tp = False
            if trade.direction == "long":
                if bar_low <= trade.sl:
                    hit_sl = True
                elif bar_high >= trade.tp:
                    hit_tp = True
            else:  # short
                if bar_high >= trade.sl:
                    hit_sl = True
                elif bar_low <= trade.tp:
                    hit_tp = True

            if hit_sl or hit_tp:
                exit_price = trade.sl if hit_sl else trade.tp
                exit_reason = "SL" if hit_sl else "TP"
                commission = trade.size_usd * (config.commission_pct / 100)

                if trade.direction == "long":
                    raw_pnl = (exit_price - trade.entry_price) / trade.entry_price * trade.size_usd
                else:
                    raw_pnl = (trade.entry_price - exit_price) / trade.entry_price * trade.size_usd

                pnl = raw_pnl - commission
                equity += pnl

                trade.exit_idx = i
                trade.exit_time = ts
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl_usd = round(pnl, 4)
                trade.pnl_pct = round(pnl / config.initial_capital * 100, 4)
                trade.duration_bars = i - trade.entry_idx
                closed_trades.append(trade)
            else:
                still_open.append(trade)

        open_trades = still_open

        # --- Check for new entries ---
        if len(open_trades) < config.max_open_trades:
            for sig in signals:
                if sig["idx"] != i:
                    continue
                if sig["idx"] in used_signal_idxs:
                    continue
                # Skip if a same-direction trade is already open
                if any(t.direction == sig["direction"] for t in open_trades):
                    continue

                slippage = sig["entry"] * (config.slippage_pct / 100)
                entry_price = sig["entry"] + slippage if sig["direction"] == "long" else sig["entry"] - slippage
                risk_price = abs(entry_price - sig["sl"])
                if risk_price <= 0:
                    continue

                risk_usd = equity * (config.risk_per_trade_pct / 100)
                size_usd = risk_usd / (risk_price / entry_price)
                commission = size_usd * (config.commission_pct / 100)
                equity -= commission

                trade = Trade(
                    signal_idx=sig["idx"],
                    entry_idx=i,
                    entry_time=ts,
                    direction=sig["direction"],
                    entry_price=entry_price,
                    sl=sig["sl"],
                    tp=sig["tp"],
                    size_usd=size_usd,
                    risk_usd=risk_usd,
                    trigger=sig.get("trigger", ""),
                )
                open_trades.append(trade)
                used_signal_idxs.add(sig["idx"])

        equity_curve.append(equity)
        timestamps.append(ts)

    # Force-close any remaining open trades at last price
    last_price = df["close"].iloc[-1]
    for trade in open_trades:
        commission = trade.size_usd * (config.commission_pct / 100)
        if trade.direction == "long":
            raw_pnl = (last_price - trade.entry_price) / trade.entry_price * trade.size_usd
        else:
            raw_pnl = (trade.entry_price - last_price) / trade.entry_price * trade.size_usd
        pnl = raw_pnl - commission
        equity += pnl
        trade.exit_idx = len(df) - 1
        trade.exit_time = df.index[-1]
        trade.exit_price = last_price
        trade.exit_reason = "EOD"
        trade.pnl_usd = round(pnl, 4)
        trade.pnl_pct = round(pnl / config.initial_capital * 100, 4)
        trade.duration_bars = len(df) - 1 - trade.entry_idx
        closed_trades.append(trade)

    eq_series = pd.Series(equity_curve, index=pd.DatetimeIndex(timestamps))
    peak = eq_series.cummax()
    dd_series = (eq_series - peak) / peak * 100

    stats = _compute_stats(closed_trades, config.initial_capital, eq_series)

    return BacktestResult(
        trades=closed_trades,
        equity_curve=eq_series,
        drawdown_curve=dd_series,
        stats=stats,
    )


def _compute_stats(trades: list[Trade], initial_capital: float, equity: pd.Series) -> dict:
    if not trades:
        return {"error": "No trades executed"}

    pnls = [t.pnl_usd for t in trades if t.pnl_usd is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return = (equity.iloc[-1] - initial_capital) / initial_capital * 100
    max_dd = equity.min() - equity.max() if len(equity) > 1 else 0
    max_dd_pct = (equity / equity.cummax() - 1).min() * 100

    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

    # Annualized Sharpe (assuming 1h candles = 8760 candles/year)
    returns = equity.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(8760) if returns.std() > 0 else 0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "rr_actual": round(avg_win / avg_loss, 2) if avg_loss else 0,
        "sharpe_ratio": round(sharpe, 2),
        "final_equity": round(equity.iloc[-1], 2),
    }
