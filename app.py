"""
Pro Bot — Web Dashboard
Run: python app.py
Open: http://localhost:8050
"""

import os, sys, json, threading, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from trading_bot.data.fetcher import fetch_binance
from trading_bot.strategies.breakout_retest import run_breakout_retest, BRParams
from trading_bot.backtest.engine import run_backtest, BacktestConfig
from trading_bot.visual.chart import build_chart
from trading_bot.strategies.smc_ict import SMCAnalysis

# ── Tuned params (same as run.py) ────────────────────────────────────────────
SCREEN_PAIRS = ["BTC","ETH","SOL","BNB","XRP","ADA","AVAX","DOGE","LINK","DOT","UNI"]

BR_PARAMS = {
    "BTC/USDT": BRParams(swing_lookback=12, breakout_vol_ratio=1.6,
                         retest_tolerance_pct=0.004, confirm_body_pct=0.55, rr_ratio=2.0,
                         ema_trend_filter=True, ema_period=50, use_kill_zones=True,
                         kill_zones=((7,10),(13,16),(18,21))),
    "ETH/USDT": BRParams(swing_lookback=15, breakout_vol_ratio=1.8,
                         retest_tolerance_pct=0.004, confirm_body_pct=0.6, rr_ratio=2.0,
                         ema_trend_filter=True, ema_period=50, use_kill_zones=True,
                         kill_zones=((7,10),(13,16),(18,21))),
    "SOL/USDT": BRParams(swing_lookback=10, breakout_vol_ratio=1.3,
                         retest_tolerance_pct=0.003, confirm_body_pct=0.5, rr_ratio=2.0,
                         ema_trend_filter=True, ema_period=50, use_kill_zones=True,
                         kill_zones=((7,10),(13,16),(18,21))),
}
DEFAULT_BR = BRParams(swing_lookback=10, breakout_vol_ratio=1.4,
                      retest_tolerance_pct=0.003, confirm_body_pct=0.5, rr_ratio=2.0,
                      ema_trend_filter=True, ema_period=50, use_kill_zones=True,
                      kill_zones=((7,10),(13,16),(18,21)))

# ── In-memory state ───────────────────────────────────────────────────────────
_screener_results = []
_live_signals     = []   # list of signal dicts
_screener_lock    = threading.Lock()
_signals_lock     = threading.Lock()

COLORS = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#21262d",
    "green":   "#00d084",
    "red":     "#ff4444",
    "yellow":  "#ffcc00",
    "text":    "#e0e0e0",
    "muted":   "#8b949e",
    "blue":    "#58a6ff",
}

# ── App setup ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Pro Bot",
)
server = app.server   # gunicorn entry point (Railway/Render)

# ── Layout helpers ────────────────────────────────────────────────────────────
def card(children, style=None):
    base = {"background": COLORS["panel"], "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px", "padding": "20px", "marginBottom": "16px"}
    if style:
        base.update(style)
    return html.Div(children, style=base)


def stat_badge(label, value, color=None):
    c = color or COLORS["text"]
    return html.Div([
        html.Div(label, style={"color": COLORS["muted"], "fontSize": "11px",
                                "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div(value, style={"color": c, "fontSize": "22px",
                                "fontWeight": "bold", "fontFamily": "monospace"}),
    ], style={"textAlign": "center", "padding": "12px 20px"})


TAB_STYLE = {"color": COLORS["muted"], "background": COLORS["panel"],
             "border": "none", "padding": "12px 28px", "fontFamily": "monospace"}
TAB_SEL   = {"color": COLORS["green"], "background": COLORS["bg"],
             "borderTop": f"2px solid {COLORS['green']}", "border": "none",
             "padding": "12px 28px", "fontFamily": "monospace"}


# ── Layout defined as function so body functions are available ────────────────
def make_layout():
    return html.Div(
    style={"background": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "monospace", "color": COLORS["text"]},
    children=[
        # Header
        html.Div([
            html.Div([
                html.Span("PRO BOT", style={"fontSize": "22px", "fontWeight": "bold",
                                             "color": COLORS["green"]}),
                html.Span("  |  Breakout+Retest  |  Paper Trading",
                          style={"color": COLORS["muted"], "fontSize": "14px"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            html.Div(id="live-clock", style={"color": COLORS["muted"], "fontSize": "13px"}),
        ], style={"background": COLORS["panel"],
                  "borderBottom": f"1px solid {COLORS['border']}",
                  "padding": "14px 28px", "display": "flex",
                  "justifyContent": "space-between", "alignItems": "center"}),

        # Tab bar
        dcc.Tabs(id="tabs", value="screener",
                 style={"background": COLORS["panel"]},
                 colors={"border": COLORS["border"], "primary": COLORS["green"],
                         "background": COLORS["panel"]},
                 children=[
            dcc.Tab(label="Screener",     value="screener",  style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Portfolio",    value="portfolio", style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Live Signals", value="live",      style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Chart",        value="chart",     style=TAB_STYLE, selected_style=TAB_SEL),
        ]),

        # All tab bodies — pre-rendered, show/hide with display
        html.Div(style={"padding": "24px 28px"}, children=[
            html.Div(id="pane-screener", children=screener_body()),
            html.Div(id="pane-portfolio", children=portfolio_body()),
            html.Div(id="pane-live",      children=live_body()),
            html.Div(id="pane-chart",     children=chart_body()),
        ]),

        # Intervals + stores
        dcc.Interval(id="clock-tick",  interval=1000,       n_intervals=0),
        dcc.Interval(id="auto-scan",   interval=3600*1000,  n_intervals=0),
        dcc.Store(id="screener-store", storage_type="memory"),
    ]
)

app.layout = make_layout


# ── Clock ─────────────────────────────────────────────────────────────────────
@app.callback(Output("live-clock", "children"), Input("clock-tick", "n_intervals"))
def update_clock(_):
    return datetime.now(timezone.utc).strftime("UTC  %Y-%m-%d  %H:%M:%S")


# ── Tab show/hide — content stays in DOM, only visibility changes ─────────────
PANES = ["screener", "portfolio", "live", "chart"]

@app.callback(
    [Output(f"pane-{p}", "style") for p in PANES],
    Input("tabs", "value"),
)
def switch_tab(active):
    return [
        {"display": "block"} if p == active else {"display": "none"}
        for p in PANES
    ]


# ══════════════════════════════════════════════════════════════════════════════
# SCREENER TAB
# ══════════════════════════════════════════════════════════════════════════════
def screener_body():
    return html.Div([
        card([
            html.Div([
                html.Div([
                    html.H5("Crypto Screener", style={"margin": 0, "color": COLORS["text"]}),
                    html.Div("Backtest all pairs — auto-select profitable ones",
                             style={"color": COLORS["muted"], "fontSize": "13px"}),
                ]),
                html.Div([
                    dcc.Dropdown(
                        id="screen-candles",
                        options=[{"label": f"{n} candles (~{n//24}d)", "value": n}
                                 for n in [1000, 2000, 3000, 5000]],
                        value=3000,
                        clearable=False,
                        style={"width": "180px", "background": COLORS["bg"],
                               "color": COLORS["text"]},
                    ),
                    html.Button("Run Screener", id="run-screener", n_clicks=0,
                                style={"background": COLORS["green"], "color": "#000",
                                       "border": "none", "borderRadius": "6px",
                                       "padding": "8px 20px", "fontWeight": "bold",
                                       "cursor": "pointer", "fontFamily": "monospace"}),
                ], style={"display": "flex", "gap": "12px", "alignItems": "center"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center"}),
        ]),

        html.Div(id="screener-status", style={"color": COLORS["yellow"],
                                               "marginBottom": "12px", "fontSize": "13px"}),
        html.Div(id="screener-charts"),
        html.Div(id="screener-table"),
    ])


@app.callback(
    Output("screener-status", "children"),
    Output("screener-charts", "children"),
    Output("screener-table", "children"),
    Input("run-screener", "n_clicks"),
    State("screen-candles", "value"),
    prevent_initial_call=True,
)
def run_screener_cb(n_clicks, candles):
    results = []
    for coin in SCREEN_PAIRS:
        symbol = f"{coin}/USDT"
        try:
            df = fetch_binance(symbol, "1h", limit=candles)
            p = BR_PARAMS.get(symbol, DEFAULT_BR)
            signals = run_breakout_retest(df, p)
            cfg = BacktestConfig(initial_capital=10_000, risk_per_trade_pct=1.0)
            res = run_backtest(df, signals, cfg)
            s = res.stats
            if "error" in s:
                results.append({"coin": coin, "trades": 0, "wr": 0,
                                 "pf": 0, "ret": 0, "dd": 0, "status": "SKIP"})
            else:
                pf  = float(s["profit_factor"])
                wr  = float(s["win_rate_pct"])
                ret = float(s["total_return_pct"])
                dd  = float(s["max_drawdown_pct"])
                tr  = int(s["total_trades"])
                ok  = pf >= 1.0 and wr >= 38.0 and tr >= 10
                results.append({"coin": coin, "trades": tr, "wr": wr,
                                 "pf": pf, "ret": ret, "dd": dd,
                                 "status": "LIVE" if ok else "SKIP"})
        except Exception as e:
            results.append({"coin": coin, "trades": 0, "wr": 0,
                             "pf": 0, "ret": 0, "dd": 0, "status": "ERROR"})

    with _screener_lock:
        _screener_results.clear()
        _screener_results.extend(results)

    live  = [r for r in results if r["status"] == "LIVE"]
    skip  = [r for r in results if r["status"] != "LIVE"]
    sorted_r = sorted(results, key=lambda x: x["pf"], reverse=True)

    # Build charts
    coins  = [r["coin"] for r in sorted_r]
    pfs    = [r["pf"]   for r in sorted_r]
    wrs    = [r["wr"]   for r in sorted_r]
    rets   = [r["ret"]  for r in sorted_r]
    stats  = [r["status"] for r in sorted_r]
    gc = [COLORS["green"] if s == "LIVE" else COLORS["red"] for s in stats]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Profit Factor", "Win Rate %", "Return %"),
                        horizontal_spacing=0.08)
    fig.add_trace(go.Bar(x=coins, y=pfs, marker_color=gc,
                         text=[f"{v:.2f}" for v in pfs], textposition="outside"), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["yellow"], row=1, col=1)
    fig.add_trace(go.Bar(x=coins, y=wrs, marker_color=gc,
                         text=[f"{v:.1f}%" for v in wrs], textposition="outside"), row=1, col=2)
    fig.add_hline(y=38.0, line_dash="dash", line_color=COLORS["yellow"], row=1, col=2)
    ret_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in rets]
    fig.add_trace(go.Bar(x=coins, y=rets, marker_color=ret_colors,
                         text=[f"{v:.1f}%" for v in rets], textposition="outside"), row=1, col=3)
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                      font=dict(color=COLORS["text"], family="monospace"),
                      showlegend=False, height=350, margin=dict(t=40, b=20))
    for c in [1,2,3]:
        fig.update_xaxes(gridcolor=COLORS["border"], row=1, col=c)
        fig.update_yaxes(gridcolor=COLORS["border"], row=1, col=c)

    chart_div = dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Table
    table_rows = []
    for r in sorted_r:
        status_color = (COLORS["green"] if r["status"] == "LIVE"
                        else COLORS["red"] if r["status"] == "SKIP" else COLORS["yellow"])
        pf_color = COLORS["green"] if r["pf"] >= 1.0 else COLORS["red"]
        ret_color = COLORS["green"] if r["ret"] >= 0 else COLORS["red"]
        table_rows.append(html.Tr([
            html.Td(r["coin"], style={"fontWeight": "bold", "color": COLORS["blue"]}),
            html.Td(r["trades"]),
            html.Td(f"{r['wr']:.1f}%", style={"color": COLORS["green"] if r["wr"] >= 38 else COLORS["red"]}),
            html.Td(f"{r['pf']:.2f}", style={"color": pf_color}),
            html.Td(f"{r['ret']:.1f}%", style={"color": ret_color}),
            html.Td(f"{r['dd']:.1f}%", style={"color": COLORS["yellow"]}),
            html.Td(html.Span(r["status"],
                              style={"background": status_color, "color": "#000",
                                     "padding": "2px 10px", "borderRadius": "4px",
                                     "fontWeight": "bold", "fontSize": "12px"})),
        ], style={"borderBottom": f"1px solid {COLORS['border']}"}))

    table_div = card([
        html.Div([
            html.Span(f"LIVE: {len(live)} pairs  ",
                      style={"color": COLORS["green"], "fontWeight": "bold"}),
            html.Span(" | ".join(r["coin"] for r in live),
                      style={"color": COLORS["green"]}),
        ], style={"marginBottom": "16px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"color": COLORS["muted"], "padding": "8px 16px",
                                   "textAlign": "left", "borderBottom": f"1px solid {COLORS['border']}"})
                for h in ["Pair", "Trades", "Win Rate", "Profit Factor", "Return", "Max DD", "Status"]
            ])),
            html.Tbody(table_rows, style={"fontSize": "14px"}),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
    ])

    status = f"Done — {len(live)} profitable pairs found from {len(results)} tested."
    return status, chart_div, table_div


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO TAB
# ══════════════════════════════════════════════════════════════════════════════
def portfolio_body():
    return html.Div([
        card([
            html.Div([
                html.H5("Portfolio Backtest", style={"margin": 0}),
                html.Div("Run backtest on selected pairs",
                         style={"color": COLORS["muted"], "fontSize": "13px"}),
            ]),
            html.Div([
                dcc.Dropdown(
                    id="port-pairs",
                    options=[{"label": c, "value": c}
                             for c in ["BTC","ETH","SOL","BNB","DOGE","LINK"]],
                    value=["BTC", "SOL"],
                    multi=True,
                    style={"width": "300px", "fontFamily": "monospace"},
                ),
                dcc.Dropdown(
                    id="port-candles",
                    options=[{"label": f"{n} candles", "value": n}
                             for n in [2000, 3000, 5000]],
                    value=3000, clearable=False,
                    style={"width": "140px"},
                ),
                html.Button("Run Backtest", id="run-portfolio", n_clicks=0,
                            style={"background": COLORS["blue"], "color": "#fff",
                                   "border": "none", "borderRadius": "6px",
                                   "padding": "8px 20px", "fontWeight": "bold",
                                   "cursor": "pointer", "fontFamily": "monospace"}),
            ], style={"display": "flex", "gap": "12px", "alignItems": "center"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                  "padding": "16px 20px"}),

        html.Div(id="portfolio-stats-row"),
        html.Div(id="portfolio-charts"),
    ])


@app.callback(
    Output("portfolio-stats-row", "children"),
    Output("portfolio-charts", "children"),
    Input("run-portfolio", "n_clicks"),
    State("port-pairs", "value"),
    State("port-candles", "value"),
    prevent_initial_call=True,
)
def run_portfolio_cb(_, pairs, candles):
    if not pairs:
        return html.Div("Select at least one pair.", style={"color": COLORS["red"]}), html.Div()

    all_trades = []
    pair_stats = []
    combined_equity = None

    for coin in pairs:
        symbol = f"{coin}/USDT"
        try:
            df = fetch_binance(symbol, "1h", limit=candles)
            p  = BR_PARAMS.get(symbol, DEFAULT_BR)
            signals = run_breakout_retest(df, p)
            cfg = BacktestConfig(initial_capital=10_000, risk_per_trade_pct=1.0)
            res = run_backtest(df, signals, cfg)
            s   = res.stats
            if "error" not in s:
                all_trades.extend(res.trades)
                pair_stats.append({"coin": coin, **s})
                eq = res.equity_curve
                combined_equity = eq if combined_equity is None else combined_equity + (eq - 10_000)
            else:
                pair_stats.append({"coin": coin, "total_trades": 0, "winning_trades": 0,
                                   "win_rate_pct": 0, "profit_factor": 0, "total_return_pct": 0})
        except Exception as e:
            pair_stats.append({"coin": coin, "total_trades": 0, "winning_trades": 0,
                               "win_rate_pct": 0, "profit_factor": 0, "total_return_pct": 0,
                               "_err": str(e)[:100]})

    real_stats = [p for p in pair_stats if p.get("total_trades", 0) > 0]
    if not real_stats:
        errs = [p.get("_err","no trades") for p in pair_stats]
        return html.Div(f"No trades found. Errors: {errs}",
                        style={"color": COLORS["red"]}), html.Div()

    total_trades = sum(int(p["total_trades"]) for p in real_stats)
    wins  = sum(int(p.get("winning_trades", 0)) for p in real_stats)
    wr    = round(wins / total_trades * 100, 1) if total_trades else 0
    total_ret = sum(float(p["total_return_pct"]) for p in real_stats)
    avg_pf = round(sum(float(p["profit_factor"]) for p in real_stats) / len(real_stats), 2)

    # Stat badges
    ret_color = COLORS["green"] if total_ret >= 0 else COLORS["red"]
    stats_row = card([
        html.Div([
            stat_badge("Total Trades", total_trades),
            stat_badge("Win Rate", f"{wr}%",
                       COLORS["green"] if wr >= 38 else COLORS["red"]),
            stat_badge("Profit Factor", avg_pf,
                       COLORS["green"] if avg_pf >= 1 else COLORS["red"]),
            stat_badge("Total Return",
                       f"{'+' if total_ret >= 0 else ''}{total_ret:.2f}%", ret_color),
        ], style={"display": "flex", "justifyContent": "space-around"}),
    ])

    # Per-pair table
    per_pair_rows = []
    for p in real_stats:
        r = float(p["total_return_pct"])
        per_pair_rows.append(html.Tr([
            html.Td(p["coin"], style={"color": COLORS["blue"], "fontWeight": "bold"}),
            html.Td(p["total_trades"]),
            html.Td(f"{p['win_rate_pct']}%",
                    style={"color": COLORS["green"] if float(p["win_rate_pct"]) >= 38 else COLORS["red"]}),
            html.Td(p["profit_factor"],
                    style={"color": COLORS["green"] if float(p["profit_factor"]) >= 1 else COLORS["red"]}),
            html.Td(f"{'+' if r >= 0 else ''}{r:.2f}%",
                    style={"color": COLORS["green"] if r >= 0 else COLORS["red"]}),
        ], style={"borderBottom": f"1px solid {COLORS['border']}"}))

    # Equity curve
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Portfolio Equity Curve", "Per-Pair Return %"))
    if combined_equity is not None:
        fig.add_trace(go.Scatter(x=combined_equity.index, y=combined_equity.values,
                                 mode="lines", line=dict(color=COLORS["blue"], width=2),
                                 name="Equity"), row=1, col=1)
        start_cap = 10_000 * len(pairs)
        fig.add_hline(y=start_cap, line_dash="dash", line_color=COLORS["muted"], row=1, col=1)

    coins_p = [p["coin"] for p in real_stats]
    rets_p  = [float(p["total_return_pct"]) for p in real_stats]
    fig.add_trace(go.Bar(x=coins_p, y=rets_p,
                         marker_color=[COLORS["green"] if r >= 0 else COLORS["red"]
                                       for r in rets_p],
                         text=[f"{r:.1f}%" for r in rets_p],
                         textposition="outside"), row=1, col=2)
    fig.add_hline(y=0, line_color=COLORS["muted"], row=1, col=2)
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                      font=dict(color=COLORS["text"], family="monospace"),
                      showlegend=False, height=350, margin=dict(t=40, b=20))
    for c in [1,2]:
        fig.update_xaxes(gridcolor=COLORS["border"], row=1, col=c)
        fig.update_yaxes(gridcolor=COLORS["border"], row=1, col=c)

    charts = html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        card([
            html.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={"color": COLORS["muted"], "padding": "8px 16px",
                                       "textAlign": "left",
                                       "borderBottom": f"1px solid {COLORS['border']}"})
                    for h in ["Pair", "Trades", "Win Rate", "PF", "Return"]
                ])),
                html.Tbody(per_pair_rows),
            ], style={"width": "100%", "borderCollapse": "collapse", "fontSize": "14px"}),
        ]),
    ])

    return stats_row, charts


# ══════════════════════════════════════════════════════════════════════════════
# LIVE SIGNALS TAB
# ══════════════════════════════════════════════════════════════════════════════
def live_body():
    return html.Div([
        card([
            html.Div([
                html.Div([
                    html.H5("Live Signal Scanner", style={"margin": 0}),
                    html.Div("Scans profitable pairs for new breakout-retest signals",
                             style={"color": COLORS["muted"], "fontSize": "13px"}),
                ]),
                html.Div([
                    html.Button("Scan Now", id="scan-now", n_clicks=0,
                                style={"background": COLORS["green"], "color": "#000",
                                       "border": "none", "borderRadius": "6px",
                                       "padding": "8px 20px", "fontWeight": "bold",
                                       "cursor": "pointer", "fontFamily": "monospace"}),
                    html.Div("Auto-scans every hour",
                             style={"color": COLORS["muted"], "fontSize": "12px"}),
                ], style={"display": "flex", "gap": "12px", "alignItems": "center"}),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
        ]),

        html.Div(id="scan-status", style={"color": COLORS["yellow"],
                                           "fontSize": "13px", "marginBottom": "12px"}),
        html.Div(id="live-signals-content"),
    ])


@app.callback(
    Output("scan-status", "children"),
    Output("live-signals-content", "children"),
    Input("scan-now", "n_clicks"),
    Input("auto-scan", "n_intervals"),
    prevent_initial_call=False,
)
def scan_signals(n_clicks, n_intervals):
    # Only show loader on first auto-scan or button click
    live_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "LINK/USDT"]
    found = []
    scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for symbol in live_pairs:
        try:
            df = fetch_binance(symbol, "1h", limit=500, use_cache=False)
            if df is None or len(df) < 50:
                continue
            p = BR_PARAMS.get(symbol, DEFAULT_BR)
            signals = run_breakout_retest(df, p)
            if not signals:
                continue
            last = signals[-1]
            last_idx = len(df) - 2
            if last["idx"] == last_idx:
                last["symbol"] = symbol
                last["scanned"] = scanned_at
                last["current_price"] = round(float(df["close"].iloc[-1]), 4)
                found.append(last)
        except:
            pass

    with _signals_lock:
        for s in found:
            _live_signals.insert(0, s)
        del _live_signals[50:]   # keep last 50

    # Build signal cards
    all_signals = []
    with _signals_lock:
        all_signals = list(_live_signals)

    if not all_signals:
        content = card([
            html.Div("No signals yet. Bot is watching 6 pairs every hour.",
                     style={"color": COLORS["muted"], "textAlign": "center",
                            "padding": "40px", "fontSize": "15px"}),
        ])
    else:
        signal_cards = []
        for sig in all_signals[:20]:
            direction = sig.get("direction", "?").upper()
            d_color = COLORS["green"] if direction == "LONG" else COLORS["red"]
            entry = sig.get("entry", 0)
            sl    = sig.get("sl", 0)
            tp    = sig.get("tp", 0)
            risk  = sig.get("risk_pct", 0)
            sym   = sig.get("symbol", "?").replace("/USDT", "")
            ts    = sig.get("scanned", "")

            signal_cards.append(
                html.Div([
                    html.Div([
                        html.Span(sym, style={"color": COLORS["blue"], "fontWeight": "bold",
                                              "fontSize": "16px"}),
                        html.Span(f"  {direction}",
                                  style={"color": d_color, "fontWeight": "bold",
                                         "fontSize": "16px"}),
                        html.Span(f"  •  {ts}",
                                  style={"color": COLORS["muted"], "fontSize": "12px"}),
                    ]),
                    html.Div([
                        html.Div([
                            html.Div("Entry", style={"color": COLORS["muted"], "fontSize": "11px"}),
                            html.Div(f"{entry:.4f}",
                                     style={"color": COLORS["text"], "fontWeight": "bold"}),
                        ]),
                        html.Div([
                            html.Div("Stop Loss", style={"color": COLORS["muted"], "fontSize": "11px"}),
                            html.Div(f"{sl:.4f}", style={"color": COLORS["red"]}),
                        ]),
                        html.Div([
                            html.Div("Take Profit", style={"color": COLORS["muted"], "fontSize": "11px"}),
                            html.Div(f"{tp:.4f}", style={"color": COLORS["green"]}),
                        ]),
                        html.Div([
                            html.Div("Risk", style={"color": COLORS["muted"], "fontSize": "11px"}),
                            html.Div(f"{risk:.2f}%", style={"color": COLORS["yellow"]}),
                        ]),
                    ], style={"display": "flex", "gap": "32px", "marginTop": "8px"}),
                ], style={
                    "background": COLORS["panel"],
                    "border": f"1px solid {d_color}44",
                    "borderLeft": f"4px solid {d_color}",
                    "borderRadius": "6px",
                    "padding": "14px 20px",
                    "marginBottom": "10px",
                })
            )
        content = html.Div(signal_cards)

    status = f"Last scan: {scanned_at}  |  {len(found)} new signal(s)  |  {len(all_signals)} total"
    return status, content


# ══════════════════════════════════════════════════════════════════════════════
# CHART TAB
# ══════════════════════════════════════════════════════════════════════════════
def chart_body():
    return html.Div([
        card([
            html.Div([
                html.H5("Candlestick Chart", style={"margin": 0}),
            ]),
            html.Div([
                dcc.Dropdown(
                    id="chart-pair",
                    options=[{"label": c, "value": f"{c}/USDT"}
                             for c in ["BTC","ETH","SOL","BNB","DOGE","LINK"]],
                    value="BTC/USDT", clearable=False,
                    style={"width": "150px"},
                ),
                dcc.Dropdown(
                    id="chart-candles",
                    options=[{"label": f"Last {n}", "value": n}
                             for n in [100, 250, 500]],
                    value=250, clearable=False,
                    style={"width": "130px"},
                ),
                html.Button("Load Chart", id="load-chart", n_clicks=0,
                            style={"background": COLORS["blue"], "color": "#fff",
                                   "border": "none", "borderRadius": "6px",
                                   "padding": "8px 20px", "fontWeight": "bold",
                                   "cursor": "pointer", "fontFamily": "monospace"}),
            ], style={"display": "flex", "gap": "12px", "alignItems": "center"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "padding": "16px 20px"}),

        html.Div(id="chart-output"),
    ])


@app.callback(
    Output("chart-output", "children"),
    Input("load-chart", "n_clicks"),
    State("chart-pair", "value"),
    State("chart-candles", "value"),
    prevent_initial_call=True,
)
def load_chart_cb(_, symbol, last_n):
    try:
        df = fetch_binance(symbol, "1h", limit=1000)
        p  = BR_PARAMS.get(symbol, DEFAULT_BR)
        signals = run_breakout_retest(df, p)

        # Build candlestick with signals
        df_plot = df.iloc[-last_n:]
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df_plot.index, open=df_plot["open"], high=df_plot["high"],
            low=df_plot["low"], close=df_plot["close"],
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
            name="Price",
        ))

        # Volume
        fig.add_trace(go.Bar(
            x=df_plot.index, y=df_plot["volume"],
            marker_color=[COLORS["green"] if c >= o else COLORS["red"]
                          for c, o in zip(df_plot["close"], df_plot["open"])],
            name="Volume", yaxis="y2", opacity=0.4,
        ))

        # EMA 50
        ema = df["close"].ewm(span=50, adjust=False).mean().iloc[-last_n:]
        fig.add_trace(go.Scatter(x=df_plot.index, y=ema, mode="lines",
                                  line=dict(color=COLORS["yellow"], width=1, dash="dot"),
                                  name="EMA 50"))

        # Plot signals
        sig_df = [s for s in signals if s["idx"] >= len(df) - last_n]
        for s in sig_df:
            ts  = df.index[s["idx"]] if s["idx"] < len(df) else None
            if ts is None:
                continue
            is_long = s["direction"] == "long"
            fig.add_trace(go.Scatter(
                x=[ts], y=[s["entry"]],
                mode="markers+text",
                marker=dict(symbol="triangle-up" if is_long else "triangle-down",
                            size=14, color=COLORS["green"] if is_long else COLORS["red"]),
                text=["L" if is_long else "S"], textposition="top center",
                textfont=dict(size=10, color=COLORS["text"]),
                showlegend=False,
            ))
            # SL/TP lines
            fig.add_hline(y=s["tp"], line_dash="dot",
                          line_color=COLORS["green"], opacity=0.4, row=1, col=1)
            fig.add_hline(y=s["sl"], line_dash="dot",
                          line_color=COLORS["red"], opacity=0.4, row=1, col=1)

        fig.update_layout(
            title=f"{symbol} — 1H  |  Breakout+Retest signals",
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["bg"],
            font=dict(color=COLORS["text"], family="monospace"),
            xaxis_rangeslider_visible=False,
            height=580,
            yaxis=dict(gridcolor=COLORS["border"], side="right"),
            yaxis2=dict(overlaying="y", side="left", showgrid=False,
                        range=[0, df_plot["volume"].max() * 4]),
            xaxis=dict(gridcolor=COLORS["border"]),
            legend=dict(bgcolor=COLORS["panel"], bordercolor=COLORS["border"]),
            margin=dict(t=50, b=20, l=20, r=60),
        )

        return dcc.Graph(figure=fig, config={"scrollZoom": True,
                                              "displayModeBar": True})

    except Exception as e:
        return html.Div(f"Error: {e}", style={"color": COLORS["red"]})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print("\n" + "=" * 55)
    print("  PRO BOT — Web Dashboard")
    print(f"  Open: http://localhost:{port}")
    print("=" * 55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
