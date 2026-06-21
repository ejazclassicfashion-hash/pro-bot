"""
Generates demo HTML charts without opening browser.
Run: python generate_demo.py
Then open output/chart.html and output/dashboard.html
"""
import sys
sys.path.insert(0, '.')

from trading_bot.data.generator import generate_ohlcv
from trading_bot.strategies.combined import run_combined, CombinedParams
from trading_bot.strategies.smc_ict import SMCParams
from trading_bot.strategies.vsa import VSAParams
from trading_bot.backtest.engine import run_backtest, BacktestConfig
from trading_bot.visual.chart import build_chart
from trading_bot.visual.dashboard import build_dashboard
import os

print("Generating 1500-candle synthetic BTC dataset...")
df = generate_ohlcv(n_candles=1500, seed=42)

print("Running SMC + VSA strategy...")
params = CombinedParams(
    smc=SMCParams(swing_lookback=5, rr_ratio=2.0, confluence_require_fvg=False),
    vsa=VSAParams(),
    require_vsa_confirm=False,
)
smc, df_vsa, signals = run_combined(df, params)
print(f"  OBs: {len(smc.order_blocks)} | FVGs: {len(smc.fvgs)} | Sweeps: {len(smc.sweeps)}")
print(f"  Signals: {len(signals)}")

print("Running backtest...")
result = run_backtest(df, signals, BacktestConfig(initial_capital=10_000))
s = result.stats
if "error" not in s:
    print(f"  Trades: {s['total_trades']} | WR: {s['win_rate_pct']}% | PF: {s['profit_factor']} | Return: {s['total_return_pct']}%")

print("Building charts...")
os.makedirs("output", exist_ok=True)

chart = build_chart(df, smc, result.trades, df_vsa, last_n_candles=250)
dash = build_dashboard(result)

chart.write_html("output/chart.html")
dash.write_html("output/dashboard.html")

print("\nDone! Open these files in your browser:")
print("  output/chart.html     ← Pro trading chart")
print("  output/dashboard.html ← Backtest stats")
