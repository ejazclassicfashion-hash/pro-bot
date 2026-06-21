"""
Pro-terminal chart builder using Plotly.
Renders: candlesticks, volume, Order Blocks, FVGs, liquidity sweeps,
structure breaks, trade entries/exits — all on a dark TradingView-style theme.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from ..strategies.smc_ict import SMCAnalysis, OrderBlock, FairValueGap
from ..backtest.engine import Trade


DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
BULL_COLOR = "#26a641"
BEAR_COLOR = "#f85149"
OB_BULL_COLOR = "rgba(38, 166, 65, 0.15)"
OB_BEAR_COLOR = "rgba(248, 81, 73, 0.15)"
OB_BULL_BORDER = "rgba(38, 166, 65, 0.7)"
OB_BEAR_BORDER = "rgba(248, 81, 73, 0.7)"
FVG_BULL_COLOR = "rgba(56, 189, 248, 0.12)"
FVG_BEAR_COLOR = "rgba(251, 146, 60, 0.12)"
SWEEP_COLOR = "#a78bfa"
BOS_COLOR = "#60a5fa"
CHOCH_COLOR = "#f59e0b"
LONG_ENTRY_COLOR = "#26a641"
SHORT_ENTRY_COLOR = "#f85149"
TP_COLOR = "#4ade80"
SL_COLOR = "#fb7185"
EQUITY_COLOR = "#38bdf8"


def build_chart(
    df: pd.DataFrame,
    smc: SMCAnalysis,
    trades: list[Trade] | None = None,
    df_vsa: pd.DataFrame | None = None,
    last_n_candles: int = 200,
    title: str = "BTC/USDT 1H — SMC + VSA Strategy",
    bias_series: pd.Series | None = None,
) -> go.Figure:
    """
    Main chart: candlesticks + volume + all SMC annotations + trades.
    Returns a Plotly figure ready to show() or write to HTML.
    """
    # Trim to last N candles for readability
    df = df.iloc[-last_n_candles:].copy()
    start_ts = df.index[0]
    end_ts = df.index[-1]
    idx_map = {ts: i for i, ts in enumerate(df.index)}

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.65, 0.20, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=["", "Volume", "VSA Signals"],
    )

    # ── 4H Bias background shading ───────────────────────────────────────────
    if bias_series is not None:
        bias_trim = bias_series.reindex(df.index, method="ffill")
        prev_bias = None
        seg_start = None
        for ts, bias in bias_trim.items():
            if bias != prev_bias:
                if prev_bias in ("bullish", "bearish") and seg_start is not None:
                    fig.add_vrect(
                        x0=seg_start, x1=ts,
                        fillcolor="rgba(38,166,65,0.06)" if prev_bias == "bullish"
                                  else "rgba(248,81,73,0.06)",
                        layer="below", line_width=0,
                        row=1, col=1,
                    )
                seg_start = ts
                prev_bias = bias
        # Close last segment
        if prev_bias in ("bullish", "bearish") and seg_start is not None:
            fig.add_vrect(
                x0=seg_start, x1=end_ts,
                fillcolor="rgba(38,166,65,0.06)" if prev_bias == "bullish"
                          else "rgba(248,81,73,0.06)",
                layer="below", line_width=0,
                row=1, col=1,
            )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="Price",
            increasing_line_color=BULL_COLOR,
            decreasing_line_color=BEAR_COLOR,
            increasing_fillcolor=BULL_COLOR,
            decreasing_fillcolor=BEAR_COLOR,
            line_width=1,
            whiskerwidth=0.3,
        ),
        row=1, col=1,
    )

    # ── Order Blocks ─────────────────────────────────────────────────────────
    for ob in smc.order_blocks:
        if ob.timestamp < start_ts:
            continue
        right_ts = ob.mitigated_idx and df.index[min(ob.mitigated_idx, len(df) - 1)]
        right_ts = right_ts or end_ts
        color = OB_BULL_COLOR if ob.kind == "bullish" else OB_BEAR_COLOR
        border = OB_BULL_BORDER if ob.kind == "bullish" else OB_BEAR_BORDER
        label = f"{'Bull' if ob.kind == 'bullish' else 'Bear'} OB"
        opacity = 0.3 if ob.mitigated else 1.0

        fig.add_shape(
            type="rect",
            x0=ob.timestamp, x1=right_ts,
            y0=ob.ob_low, y1=ob.ob_high,
            fillcolor=color,
            line=dict(color=border, width=1, dash="solid"),
            opacity=opacity,
            row=1, col=1,
        )
        fig.add_annotation(
            x=ob.timestamp, y=ob.ob_high,
            text=label,
            font=dict(size=8, color=border.replace("0.7", "1")),
            showarrow=False, xanchor="left", yanchor="bottom",
            row=1, col=1,
        )

    # ── Fair Value Gaps ──────────────────────────────────────────────────────
    for fvg in smc.fvgs:
        if fvg.timestamp < start_ts or fvg.filled:
            continue
        color = FVG_BULL_COLOR if fvg.kind == "bullish" else FVG_BEAR_COLOR
        fig.add_shape(
            type="rect",
            x0=fvg.timestamp, x1=end_ts,
            y0=fvg.gap_low, y1=fvg.gap_high,
            fillcolor=color,
            line=dict(color=color.replace("0.12", "0.5"), width=0.5),
            row=1, col=1,
        )

    # ── Liquidity Sweeps ─────────────────────────────────────────────────────
    for sweep in smc.sweeps:
        if sweep.timestamp < start_ts:
            continue
        symbol = "triangle-up" if sweep.direction == "bullish_sweep" else "triangle-down"
        fig.add_trace(
            go.Scatter(
                x=[sweep.timestamp],
                y=[sweep.swept_level],
                mode="markers+text",
                marker=dict(symbol=symbol, size=12, color=SWEEP_COLOR),
                text=["LSW"],
                textfont=dict(size=8, color=SWEEP_COLOR),
                textposition="top center" if sweep.direction == "bullish_sweep" else "bottom center",
                name="Liq Sweep",
                showlegend=False,
            ),
            row=1, col=1,
        )

    # ── Structure Breaks (BOS / CHoCH) ───────────────────────────────────────
    for sb in smc.structure_breaks:
        if sb.timestamp < start_ts:
            continue
        color = BOS_COLOR if "BOS" in sb.kind else CHOCH_COLOR
        fig.add_shape(
            type="line",
            x0=sb.timestamp, x1=end_ts,
            y0=sb.price, y1=sb.price,
            line=dict(color=color, width=0.8, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=sb.timestamp, y=sb.price,
            text=sb.kind.replace("_", " "),
            font=dict(size=7, color=color),
            showarrow=False, xanchor="left",
            row=1, col=1,
        )

    # ── Trade Entries / Exits ────────────────────────────────────────────────
    if trades:
        for trade in trades:
            if trade.entry_time < start_ts:
                continue
            is_long = trade.direction == "long"
            entry_sym = "triangle-up" if is_long else "triangle-down"
            entry_col = LONG_ENTRY_COLOR if is_long else SHORT_ENTRY_COLOR

            fig.add_trace(
                go.Scatter(
                    x=[trade.entry_time],
                    y=[trade.entry_price],
                    mode="markers",
                    marker=dict(symbol=entry_sym, size=14, color=entry_col, line=dict(width=1.5, color="white")),
                    name=f"{'Long' if is_long else 'Short'} Entry",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{'LONG' if is_long else 'SHORT'}</b><br>"
                        f"Entry: {trade.entry_price:.2f}<br>"
                        f"SL: {trade.sl:.2f}<br>"
                        f"TP: {trade.tp:.2f}<br>"
                        f"PnL: {trade.pnl_usd:.2f} USD<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

            if trade.exit_time and trade.exit_time >= start_ts:
                outcome_col = TP_COLOR if trade.exit_reason == "TP" else SL_COLOR
                fig.add_shape(
                    type="line",
                    x0=trade.entry_time, x1=trade.exit_time,
                    y0=trade.entry_price, y1=trade.exit_price,
                    line=dict(color=outcome_col, width=1.5, dash="dash"),
                    row=1, col=1,
                )
                # SL / TP horizontal lines
                fig.add_shape(type="line", x0=trade.entry_time, x1=trade.exit_time,
                              y0=trade.sl, y1=trade.sl,
                              line=dict(color=SL_COLOR, width=0.7, dash="dot"), row=1, col=1)
                fig.add_shape(type="line", x0=trade.entry_time, x1=trade.exit_time,
                              y0=trade.tp, y1=trade.tp,
                              line=dict(color=TP_COLOR, width=0.7, dash="dot"), row=1, col=1)

    # ── Volume bars ──────────────────────────────────────────────────────────
    vol_colors = [
        BULL_COLOR if c >= o else BEAR_COLOR
        for c, o in zip(df["close"], df["open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="Volume",
               marker_color=vol_colors, opacity=0.7, showlegend=False),
        row=2, col=1,
    )
    if df_vsa is not None and "vol_ma" in df_vsa.columns:
        vol_ma_trim = df_vsa["vol_ma"].reindex(df.index)
        fig.add_trace(
            go.Scatter(x=df.index, y=vol_ma_trim, name="Vol MA",
                       line=dict(color="#f59e0b", width=1.2), showlegend=False),
            row=2, col=1,
        )

    # ── VSA signal dots ──────────────────────────────────────────────────────
    if df_vsa is not None and "vsa_signal" in df_vsa.columns:
        vsa_trim = df_vsa.reindex(df.index)
        vsa_bull = vsa_trim[vsa_trim["vsa_bias"] == "bullish"]
        vsa_bear = vsa_trim[vsa_trim["vsa_bias"] == "bearish"]

        if not vsa_bull.empty:
            fig.add_trace(
                go.Scatter(
                    x=vsa_bull.index, y=[0.5] * len(vsa_bull),
                    mode="markers+text",
                    marker=dict(symbol="circle", size=8, color=BULL_COLOR),
                    text=vsa_bull["vsa_signal"].tolist(),
                    textfont=dict(size=7, color=BULL_COLOR),
                    textposition="top center",
                    name="VSA Bull", showlegend=False,
                ),
                row=3, col=1,
            )
        if not vsa_bear.empty:
            fig.add_trace(
                go.Scatter(
                    x=vsa_bear.index, y=[-0.5] * len(vsa_bear),
                    mode="markers+text",
                    marker=dict(symbol="circle", size=8, color=BEAR_COLOR),
                    text=vsa_bear["vsa_signal"].tolist(),
                    textfont=dict(size=7, color=BEAR_COLOR),
                    textposition="bottom center",
                    name="VSA Bear", showlegend=False,
                ),
                row=3, col=1,
            )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(family="JetBrains Mono, Courier New, monospace", color=TEXT_COLOR, size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor=PANEL_BG, bordercolor=GRID_COLOR, borderwidth=1,
            font=dict(size=9, color=TEXT_COLOR),
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        height=850,
        hovermode="x unified",
    )

    for row in [1, 2, 3]:
        fig.update_xaxes(
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=0.5,
            zeroline=False, showline=True, linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=9),
            row=row, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=0.5,
            zeroline=False, showline=True, linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR, size=9),
            row=row, col=1,
        )

    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1,
                     title_font=dict(size=9, color=TEXT_COLOR))
    fig.update_yaxes(title_text="Volume", row=2, col=1,
                     title_font=dict(size=9, color=TEXT_COLOR))

    return fig
