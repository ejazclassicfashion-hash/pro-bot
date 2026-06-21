# Trading Bot — Setup Guide

## Step 1: Install Python (if not installed)
Download Python 3.11+ from https://python.org
During install → check "Add Python to PATH"

## Step 2: Open PowerShell in this folder
Right-click on `E:\bot` folder → "Open in Terminal"

## Step 3: Install dependencies
```
pip install -r requirements.txt
```

## Step 4: Run the bot

### Demo mode (synthetic data, no internet needed):
```
python run.py
```

### Real Binance data (internet needed, no API key):
```
python run.py --real
```

### Run optimizer (tests ~72 parameter combinations):
```
python run.py --optimize
```

### All together (real data + optimize):
```
python run.py --real --optimize
```

## What opens in your browser:
- `output/chart.html` → Pro candlestick chart with Order Blocks, FVGs, Liquidity Sweeps, BOS/CHoCH, and all trades marked
- `output/dashboard.html` → Equity curve, drawdown, trade stats, P&L distribution

## To change strategy parameters, edit run.py:
```python
params = CombinedParams(
    smc=SMCParams(
        swing_lookback=5,        # how many candles to confirm a swing point
        rr_ratio=2.0,            # minimum risk:reward ratio
        confluence_require_fvg=True,  # require FVG inside Order Block
    ),
    require_vsa_confirm=True,    # require VSA volume confirmation
)
```
