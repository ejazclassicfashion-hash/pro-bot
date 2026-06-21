"""
Pro Bot — Multi-Pair Portfolio
Daily → 4H Bias → 1H ICT Kill Zone Entry
Pairs: BTC/USDT, ETH/USDT, SOL/USDT

Usage:
  python run.py              → synthetic demo (BTC only)
  python run.py --real       → real Binance data, all 3 pairs
  python run.py --real --optimize
  python run.py --real --pairs BTC ETH   → specific pairs
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from trading_bot.data.generator import generate_ohlcv
from trading_bot.strategies.combined import CombinedParams
from trading_bot.strategies.smc_ict import SMCParams
from trading_bot.strategies.mtf import MTFParams
from trading_bot.strategies.breakout_retest import BRParams
from trading_bot.portfolio.runner import run_portfolio
from trading_bot.backtest.optimizer import optimize, walk_forward_test
from trading_bot.visual.chart import build_chart
from trading_bot.visual.dashboard import build_dashboard, build_optimizer_summary
from trading_bot.visual.screener import build_screener_dashboard
from trading_bot.strategies.mtf import compute_4h_bias, map_bias_to_1h

CANDLES_4H_RATIO = 4

# ── Per-pair BR params (tuned from backtesting) ──────────────────────────────
DEFAULT_BR_PARAMS_PER_PAIR = {
    "BTC/USDT": BRParams(
        swing_lookback=12, breakout_vol_ratio=1.6,
        retest_tolerance_pct=0.004, confirm_body_pct=0.55, rr_ratio=2.0,
        ema_trend_filter=True, ema_period=50, use_kill_zones=True,
        kill_zones=((7, 10), (13, 16), (18, 21)),
    ),
    "ETH/USDT": BRParams(
        swing_lookback=15, breakout_vol_ratio=1.8,
        retest_tolerance_pct=0.004, confirm_body_pct=0.6, rr_ratio=2.0,
        ema_trend_filter=True, ema_period=50, use_kill_zones=True,
        kill_zones=((7, 10), (13, 16), (18, 21)),
    ),
    "SOL/USDT": BRParams(
        swing_lookback=10, breakout_vol_ratio=1.3,
        retest_tolerance_pct=0.003, confirm_body_pct=0.5, rr_ratio=2.0,
        ema_trend_filter=True, ema_period=50, use_kill_zones=True,
        kill_zones=((7, 10), (13, 16), (18, 21)),
    ),
}


def print_header(pairs: list[str], strategy: str = "br"):
    print("\n" + "=" * 65)
    print("  PRO BOT  |  ICT + Breakout-Retest  |  Paper Trading")
    print("=" * 65)
    strat = "Breakout+Retest" if strategy == "br" else "SMC+ICT+MTF"
    print(f"  {strat}  |  {' + '.join(p.split('/')[0] for p in pairs)}")
    print("=" * 65 + "\n")


def fetch_pair_data(pairs: list[str], candles_1h: int) -> tuple[dict, dict]:
    from trading_bot.data.fetcher import fetch_binance
    data_1h, data_4h = {}, {}
    candles_4h = max(500, candles_1h // CANDLES_4H_RATIO + 100)

    for symbol in pairs:
        tag = symbol.replace("/", "")
        print(f"  [{tag}] Fetching 1H ({candles_1h}) + 4H ({candles_4h}) candles...")
        data_1h[symbol] = fetch_binance(symbol, "1h", limit=candles_1h)
        data_4h[symbol] = fetch_binance(symbol, "4h", limit=candles_4h)
        print(f"         1H: {len(data_1h[symbol])} | 4H: {len(data_4h[symbol])}")

    return data_1h, data_4h


def synthetic_pair_data(pairs: list[str], candles: int) -> tuple[dict, dict]:
    seeds = {"BTC/USDT": 42, "ETH/USDT": 99, "SOL/USDT": 7}
    prices = {"BTC/USDT": 42000, "ETH/USDT": 2800, "SOL/USDT": 120}
    data_1h, data_4h = {}, {}
    for symbol in pairs:
        df_1h = generate_ohlcv(
            n_candles=candles,
            start_price=prices.get(symbol, 1000),
            seed=seeds.get(symbol, 1),
        )
        df_4h = df_1h.resample("4h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna()
        data_1h[symbol] = df_1h
        data_4h[symbol] = df_4h
    return data_1h, data_4h


def run_screener(args, screen_pairs: list[str]):
    """Backtest all pairs, build screener dashboard, auto-select profitable ones."""
    from trading_bot.data.fetcher import fetch_binance
    from trading_bot.backtest.engine import run_backtest, BacktestConfig
    from trading_bot.strategies.breakout_retest import run_breakout_retest, BRParams

    print("\n" + "=" * 65)
    print("  SCREENER — Backtesting 12 crypto pairs")
    print("=" * 65 + "\n")

    results = []
    profitable_pairs = []
    candles = args.candles if hasattr(args, "candles") else 3000

    for coin in screen_pairs:
        symbol = f"{coin}/USDT"
        try:
            print(f"  Testing {symbol}...", end=" ", flush=True)
            df = fetch_binance(symbol, "1h", limit=candles)

            # Use tuned params if available, else default
            p = DEFAULT_BR_PARAMS_PER_PAIR.get(symbol, BRParams(
                swing_lookback=10, breakout_vol_ratio=1.4,
                retest_tolerance_pct=0.003, confirm_body_pct=0.5, rr_ratio=2.0,
                ema_trend_filter=True, ema_period=50, use_kill_zones=True,
                kill_zones=((7, 10), (13, 16), (18, 21)),
            ))

            signals = run_breakout_retest(df, p)
            config = BacktestConfig(initial_capital=10_000, risk_per_trade_pct=1.0)
            result = run_backtest(df, signals, config)
            s = result.stats

            if "error" in s:
                print("no trades")
                results.append({"symbol": coin, "trades": 0, "wr": 0,
                                 "pf": 0, "return_pct": 0, "max_dd": 0, "status": "SKIP"})
                continue

            pf = float(s["profit_factor"])
            wr = float(s["win_rate_pct"])
            ret = float(s["total_return_pct"])
            dd = float(s["max_drawdown_pct"])
            trades = int(s["total_trades"])
            status = "LIVE" if (pf >= 1.0 and wr >= 38.0 and trades >= 10) else "SKIP"

            if status == "LIVE":
                profitable_pairs.append(symbol)

            results.append({"symbol": coin, "trades": trades, "wr": wr,
                             "pf": pf, "return_pct": ret, "max_dd": dd, "status": status})
            print(f"WR:{wr:.1f}% PF:{pf:.2f} Return:{ret:.1f}% → {status}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"symbol": coin, "trades": 0, "wr": 0,
                             "pf": 0, "return_pct": 0, "max_dd": 0, "status": "SKIP"})

    print(f"\n  PROFITABLE PAIRS: {profitable_pairs if profitable_pairs else 'None found'}")

    # Build screener dashboard
    os.makedirs("output", exist_ok=True)
    fig = build_screener_dashboard(results, title="Crypto Screener — Breakout+Retest Backtest")
    path = os.path.abspath("output/screener.html")
    fig.write_html(path)
    print(f"  Screener dashboard: {path}")

    if not getattr(args, "no_browser", False):
        import webbrowser
        webbrowser.open(f"file:///{path}")

    # If --live also passed, start live bot on profitable pairs
    if getattr(args, "live", False) and profitable_pairs:
        print(f"\n  Starting live bot on: {profitable_pairs}")
        _start_live(profitable_pairs)
    elif not profitable_pairs:
        print("  No pairs met criteria. Try --candles 2000 or relax filters.")


def _start_live(pairs: list[str]):
    """Start the live watcher on the given pairs."""
    from trading_bot.live.watcher import LiveWatcher
    from trading_bot.strategies.breakout_retest import BRParams

    watcher = LiveWatcher(
        pairs=pairs,
        br_params_per_pair=DEFAULT_BR_PARAMS_PER_PAIR,
        capital_per_pair=10_000.0,
        risk_pct=1.0,
        poll_seconds=3600,
    )
    watcher.run()


def print_pair_stats(portfolio):
    s = portfolio.summary
    if "error" in s:
        print(f"  ⚠  {s['error']}")
        return

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │              PORTFOLIO SUMMARY                      │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Total Trades:   {s['total_trades']:<6}  Win Rate: {s['win_rate_pct']}%          │")
    print(f"  │  Profit Factor:  {s['profit_factor']:<6}  Return:   {s['total_return_pct']}%         │")
    print(f"  │  Final Value:    ${s['final_value']:,.2f}  (Capital: ${s['total_capital']:,.0f})  │")
    print("  ├─────────────────────────────────────────────────────┤")
    print("  │  Per-Pair Breakdown:                                │")

    for sym, ps in s["per_pair"].items():
        tag = sym.split("/")[0]
        print(f"  │  {tag:<4}  Signals: {ps['signals_raw']:>3}→{ps['signals_final']:<3}  "
              f"Trades: {ps['trades']:<3}  "
              f"WR: {ps['win_rate']}%  "
              f"Return: {ps['return_pct']}%  │")

    print("  └─────────────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--candles", type=int, default=5000)
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="e.g. --pairs BTC ETH  (uses USDT pairs)")
    parser.add_argument("--strategy", choices=["smc", "br"], default="br",
                        help="smc = SMC+ICT+MTF  |  br = Breakout+Retest (default)")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--screen", action="store_true",
                        help="Backtest 12 pairs, show screener dashboard, auto-select winners")
    parser.add_argument("--live", action="store_true",
                        help="Start live paper trading on profitable pairs")
    args = parser.parse_args()

    # Screen mode: run on 12 pairs and pick winners
    SCREEN_PAIRS = [
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA",
        "AVAX", "DOGE", "LINK", "DOT", "MATIC", "UNI",
    ]

    if args.screen:
        run_screener(args, SCREEN_PAIRS)
        return

    pairs = args.pairs or ["BTC", "SOL"]   # ETH dropped (underperformer)
    pairs = [f"{p}/USDT" if "/" not in p else p for p in pairs]

    print_header(pairs, strategy=args.strategy)

    # ── [1] Data ──────────────────────────────────────────────────────────────
    print(f"[1] Loading data for {len(pairs)} pairs...")
    if args.real:
        data_1h, data_4h = fetch_pair_data(pairs, args.candles)
    else:
        data_1h, data_4h = synthetic_pair_data(pairs, args.candles)

    # ── [2] Strategy params ───────────────────────────────────────────────────
    params = CombinedParams(
        smc=SMCParams(
            swing_lookback=5,
            rr_ratio=2.0,
            confluence_require_fvg=False,
            ema_trend_filter=True,
            ema_period=50,
            require_candle_confirm=True,
            use_kill_zones=True,
            kill_zones=((7, 10), (13, 16), (18, 21)),
        ),
        require_vsa_confirm=False,
    )

    # ── [3] Run portfolio ─────────────────────────────────────────────────────
    strategy_label = "Breakout + Retest" if args.strategy == "br" else "SMC + 4H Bias + Kill Zone"
    print(f"\n[2] Running {strategy_label} on {len(pairs)} pairs...")

    # Base BR params (applies to all pairs unless overridden)
    br_base = BRParams(
        swing_lookback=10,
        breakout_vol_ratio=1.3,
        retest_tolerance_pct=0.003,
        confirm_body_pct=0.5,
        rr_ratio=2.0,
        ema_trend_filter=True,
        ema_period=50,
        use_kill_zones=True,
        kill_zones=((7, 10), (13, 16), (18, 21)),
    )

    # BTC/ETH: tighter volume filter to reduce false breakouts
    # SOL already profitable at 1.3 — keep it, but reduce swing lookback for faster signals
    portfolio = run_portfolio(
        data_1h, data_4h,
        params=params,
        capital_per_pair=10_000.0,
        strategy=args.strategy,
        br_params=br_base,
        br_params_per_pair=DEFAULT_BR_PARAMS_PER_PAIR,
    )

    for pr in portfolio.pairs:
        tag = pr.symbol.split("/")[0]
        kz_pct = (pr.signals_after_kz / pr.signals_raw * 100) if pr.signals_raw else 0
        print(f"  {tag}: {pr.signals_raw} raw → {pr.signals_after_mtf} after MTF "
              f"→ {pr.signals_after_kz} after KZ  ({100-kz_pct:.0f}% filtered)")

    print_pair_stats(portfolio)

    # ── Live mode: start watcher after showing backtest ───────────────────────
    if args.live:
        print("\n[LIVE] Starting paper trading watcher...")
        _start_live(pairs)
        return  # live loop takes over

    # ── [4] Optimizer (BTC only for speed) ───────────────────────────────────
    if args.optimize:
        print("\n[3] Optimizer (BTC only, with 4H + Kill Zones)...")
        btc_1h = data_1h.get("BTC/USDT")
        btc_4h = data_4h.get("BTC/USDT")
        if btc_1h is not None:
            opt_results = optimize(btc_1h, top_n=5, df_4h=btc_4h)
            print(f"\n  Top {len(opt_results)} configs:")
            for i, r in enumerate(opt_results, 1):
                print(f"  #{i} Score:{r.rank_score} | "
                      f"Return:{r.stats.get('total_return_pct')}% | "
                      f"WR:{r.stats.get('win_rate_pct')}% | "
                      f"PF:{r.stats.get('profit_factor')}")

            if opt_results:
                print("\n  Walk-forward (3 splits)...")
                wf = walk_forward_test(btc_1h, opt_results[0].params, df_4h=btc_4h, n_splits=3)
                for w in wf:
                    print(f"  Split {w.get('split')}: "
                          f"Trades {w.get('total_trades','N/A')} | "
                          f"Return {w.get('total_return_pct','N/A')}% | "
                          f"WR {w.get('win_rate_pct','N/A')}%")

    # ── [5] Charts ────────────────────────────────────────────────────────────
    print("\n[4] Building charts...")
    os.makedirs("output", exist_ok=True)

    # Primary chart = BTC (or first pair)
    primary = portfolio.pairs[0]
    primary_sym = primary.symbol
    df_primary = data_1h[primary_sym]
    df_4h_primary = data_4h.get(primary_sym)

    bias_series = None
    if df_4h_primary is not None:
        bias_4h = compute_4h_bias(df_4h_primary, MTFParams())
        bias_series = map_bias_to_1h(bias_4h, df_primary.index)

    chart_fig = build_chart(
        df_primary, primary.smc, primary.result.trades, primary.df_vsa,
        last_n_candles=250,
        title=f"{primary_sym} — 4H Bias + 1H ICT Kill Zone Entry",
        bias_series=bias_series,
    )

    # Dashboard = portfolio combined
    from trading_bot.backtest.engine import BacktestResult
    combined_result = BacktestResult(
        trades=portfolio.combined_trades,
        equity_curve=portfolio.combined_equity,
        drawdown_curve=(portfolio.combined_equity / portfolio.combined_equity.cummax() - 1) * 100
            if not portfolio.combined_equity.empty else portfolio.combined_equity,
        stats=portfolio.summary,
    )
    dash_fig = build_dashboard(
        combined_result,
        title=f"Portfolio Dashboard — {' + '.join(p.split('/')[0] for p in pairs)}",
    )

    chart_path = os.path.abspath("output/chart.html")
    dash_path = os.path.abspath("output/dashboard.html")
    chart_fig.write_html(chart_path)
    dash_fig.write_html(dash_path)

    print(f"  → Chart (BTC): {chart_path}")
    print(f"  → Dashboard:   {dash_path}")

    if not args.no_browser:
        import webbrowser
        webbrowser.open(f"file:///{chart_path}")
        webbrowser.open(f"file:///{dash_path}")

    print("\n" + "═" * 65)
    print("  Done. Paper trading. No real money involved.")
    print("  Pairs:    python run.py --real --pairs BTC ETH SOL")
    print("  Optimize: python run.py --real --optimize")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
