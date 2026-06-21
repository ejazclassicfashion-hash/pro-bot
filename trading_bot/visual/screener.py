"""
Multi-pair screener dashboard.
Runs backtest on all pairs, shows which ones are profitable.
Green = live-worthy, Red = skip.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def build_screener_dashboard(
    results: list[dict],   # list of {symbol, trades, wr, pf, return_pct, max_dd, status}
    title: str = "Crypto Screener — Backtest Results",
) -> go.Figure:
    results_sorted = sorted(results, key=lambda x: x["pf"], reverse=True)

    symbols   = [r["symbol"]      for r in results_sorted]
    wrs       = [r["wr"]          for r in results_sorted]
    pfs       = [r["pf"]          for r in results_sorted]
    returns   = [r["return_pct"]  for r in results_sorted]
    trades    = [r["trades"]      for r in results_sorted]
    max_dds   = [r["max_dd"]      for r in results_sorted]
    statuses  = [r["status"]      for r in results_sorted]

    bar_colors = ["#00d084" if s == "LIVE" else "#ff4444" for s in statuses]
    pf_colors  = ["#00d084" if p >= 1.0   else "#ff4444" for p in pfs]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Profit Factor (>1.0 = live)", "Win Rate %",
                        "Total Return %", "Max Drawdown %"),
        vertical_spacing=0.18,
        horizontal_spacing=0.1,
    )

    # Profit Factor
    fig.add_trace(go.Bar(
        x=symbols, y=pfs, name="PF",
        marker_color=pf_colors,
        text=[f"{p:.2f}" for p in pfs],
        textposition="outside",
    ), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#ffcc00",
                  annotation_text="Breakeven", row=1, col=1)

    # Win Rate
    fig.add_trace(go.Bar(
        x=symbols, y=wrs, name="WR%",
        marker_color=bar_colors,
        text=[f"{w:.1f}%" for w in wrs],
        textposition="outside",
    ), row=1, col=2)
    fig.add_hline(y=38.0, line_dash="dash", line_color="#ffcc00",
                  annotation_text="Min WR", row=1, col=2)

    # Return %
    ret_colors = ["#00d084" if r >= 0 else "#ff4444" for r in returns]
    fig.add_trace(go.Bar(
        x=symbols, y=returns, name="Return%",
        marker_color=ret_colors,
        text=[f"{r:.1f}%" for r in returns],
        textposition="outside",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="#666666", row=2, col=1)

    # Max Drawdown
    fig.add_trace(go.Bar(
        x=symbols, y=max_dds, name="MaxDD%",
        marker_color="#ff8800",
        text=[f"{d:.1f}%" for d in max_dds],
        textposition="outside",
    ), row=2, col=2)

    # Summary table as annotation
    live_pairs = [r for r in results_sorted if r["status"] == "LIVE"]
    skip_pairs = [r for r in results_sorted if r["status"] == "SKIP"]
    summary = (
        f"LIVE ({len(live_pairs)}): {', '.join(r['symbol'] for r in live_pairs)}<br>"
        f"SKIP ({len(skip_pairs)}): {', '.join(r['symbol'] for r in skip_pairs)}"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#e0e0e0")),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e0e0e0", family="monospace"),
        showlegend=False,
        height=750,
        annotations=[
            dict(
                x=0.5, y=-0.08, xref="paper", yref="paper",
                text=summary,
                showarrow=False,
                font=dict(size=13, color="#00d084"),
                align="center",
            )
        ] + list(fig.layout.annotations),
    )

    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(
                gridcolor="#1e2530", showgrid=True, row=row, col=col,
                tickfont=dict(size=10),
            )
            fig.update_yaxes(gridcolor="#1e2530", showgrid=True, row=row, col=col)

    return fig
