"""
Stats dashboard: equity curve, drawdown, trade log, performance metrics.
All dark-themed, rendered as a single Plotly figure.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from ..backtest.engine import BacktestResult, Trade

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
BULL_COLOR = "#26a641"
BEAR_COLOR = "#f85149"
EQUITY_COLOR = "#38bdf8"
DD_COLOR = "#f59e0b"
NEUTRAL_COLOR = "#8b949e"


def build_dashboard(result: BacktestResult, title: str = "Backtest Results") -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curve", "Drawdown (%)",
            "Trade Distribution", "Win/Loss by Trade",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # ── Equity curve ─────────────────────────────────────────────────────────
    eq = result.equity_curve
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=eq.values,
            name="Equity",
            line=dict(color=EQUITY_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.05)",
        ),
        row=1, col=1,
    )

    # ── Drawdown ─────────────────────────────────────────────────────────────
    dd = result.drawdown_curve
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values,
            name="Drawdown",
            line=dict(color=DD_COLOR, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.08)",
        ),
        row=1, col=2,
    )

    # ── PnL distribution histogram ────────────────────────────────────────────
    pnls = [t.pnl_usd for t in result.trades if t.pnl_usd is not None]
    if pnls:
        fig.add_trace(
            go.Histogram(
                x=pnls,
                nbinsx=20,
                name="PnL Dist",
                marker_color=[BULL_COLOR if p > 0 else BEAR_COLOR for p in pnls],
                showlegend=False,
            ),
            row=2, col=1,
        )

    # ── Trade waterfall ───────────────────────────────────────────────────────
    if result.trades:
        colors = [BULL_COLOR if t.pnl_usd and t.pnl_usd > 0 else BEAR_COLOR for t in result.trades]
        fig.add_trace(
            go.Bar(
                x=list(range(len(result.trades))),
                y=[t.pnl_usd or 0 for t in result.trades],
                name="Trade PnL",
                marker_color=colors,
                showlegend=False,
            ),
            row=2, col=2,
        )

    # ── Stats annotation box ──────────────────────────────────────────────────
    stats = result.stats
    if "error" not in stats:
        stats_text = (
            f"<b>Performance Summary</b><br>"
            f"Total Trades: {stats['total_trades']}<br>"
            f"Win Rate: <b>{stats['win_rate_pct']}%</b><br>"
            f"Profit Factor: <b>{stats['profit_factor']}</b><br>"
            f"Total Return: <b>{stats['total_return_pct']}%</b><br>"
            f"Max Drawdown: <b>{stats['max_drawdown_pct']}%</b><br>"
            f"Sharpe Ratio: {stats['sharpe_ratio']}<br>"
            f"Avg Win: ${stats['avg_win_usd']}<br>"
            f"Avg Loss: -${stats['avg_loss_usd']}<br>"
            f"Actual RR: {stats['rr_actual']}<br>"
            f"Final Equity: <b>${stats['final_equity']:,}</b>"
        )
        fig.add_annotation(
            text=stats_text,
            x=1.02, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=11, color=TEXT_COLOR, family="JetBrains Mono, monospace"),
            align="left",
            bgcolor=PANEL_BG,
            bordercolor=GRID_COLOR,
            borderwidth=1,
            borderpad=12,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(family="JetBrains Mono, Courier New, monospace", color=TEXT_COLOR, size=10),
        height=700,
        margin=dict(l=60, r=200, t=60, b=40),
        legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_COLOR),
        showlegend=False,
    )

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(
                showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                tickfont=dict(color=TEXT_COLOR, size=9), row=r, col=c,
            )
            fig.update_yaxes(
                showgrid=True, gridcolor=GRID_COLOR, zeroline=True,
                zerolinecolor=GRID_COLOR, tickfont=dict(color=TEXT_COLOR, size=9),
                row=r, col=c,
            )

    return fig


def build_optimizer_summary(opt_results: list) -> go.Figure:
    """Table showing top N optimizer results."""
    if not opt_results:
        return go.Figure()

    rows = []
    for r in opt_results:
        row = {**r.params, **{k: v for k, v in r.stats.items() if k != "error"}}
        row["score"] = r.rank_score
        rows.append(row)

    df = pd.DataFrame(rows)

    fig = go.Figure(
        go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in df.columns],
                fill_color=PANEL_BG,
                font=dict(color=TEXT_COLOR, size=10, family="JetBrains Mono, monospace"),
                line_color=GRID_COLOR,
                align="left",
            ),
            cells=dict(
                values=[df[c].tolist() for c in df.columns],
                fill_color=DARK_BG,
                font=dict(color=TEXT_COLOR, size=9, family="JetBrains Mono, monospace"),
                line_color=GRID_COLOR,
                align="left",
            ),
        )
    )
    fig.update_layout(
        title="Optimizer Results — Top Combinations",
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
