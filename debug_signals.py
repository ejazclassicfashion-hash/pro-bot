import sys
sys.path.insert(0, '.')
from trading_bot.data.generator import generate_ohlcv
from trading_bot.strategies.smc_ict import (
    detect_swings, label_structure, detect_structure_breaks,
    detect_order_blocks, detect_fvgs, detect_liquidity_sweeps,
    mark_mitigated, mark_fvg_filled, SMCParams
)

df = generate_ohlcv(n_candles=1500, seed=42)
p = SMCParams()

sh, sl = detect_swings(df, p.swing_lookback)
sh, sl = label_structure(sh, sl)
sbs = detect_structure_breaks(df, sh, sl)
obs = detect_order_blocks(df, sbs, p.ob_lookback)
fvgs = detect_fvgs(df, p.fvg_min_gap_pct)
sweeps = detect_liquidity_sweeps(df, sh, sl, p.sweep_threshold_pct)
mark_mitigated(df, obs)

mitigated = sum(1 for ob in obs if ob.mitigated)
print(f"OBs total: {len(obs)} | Mitigated: {mitigated} | Active: {len(obs)-mitigated}")

bull_idxs = set()
bear_idxs = set()
for sw in sweeps:
    if sw.direction == "bullish_sweep": bull_idxs.add(sw.sweep_idx)
    else: bear_idxs.add(sw.sweep_idx)
for sb in sbs:
    if "bull" in sb.kind: bull_idxs.add(sb.idx)
    elif "bear" in sb.kind: bear_idxs.add(sb.idx)

LOOKBACK = 40
hits_not_mitigated = 0
hits_risk_ok = 0
hits_all_pass = 0

for ob in obs:
    if ob.mitigated:
        continue
    for i in range(ob.start_idx + 1, len(df)):
        price = df["close"].iloc[i]
        low_i = df["low"].iloc[i]
        high_i = df["high"].iloc[i]

        if ob.kind == "bullish":
            touch = low_i <= ob.ob_high * 1.002 and price >= ob.ob_low * 0.998
            evs = [e for e in bull_idxs if i - LOOKBACK <= e < i and e >= ob.start_idx]
        else:
            touch = high_i >= ob.ob_low * 0.998 and price <= ob.ob_high * 1.002
            evs = [e for e in bear_idxs if i - LOOKBACK <= e < i and e >= ob.start_idx]

        if touch and evs:
            hits_not_mitigated += 1
            # risk check
            if ob.kind == "bullish":
                sl_price = ob.ob_low * (1 - p.sl_ob_buffer_pct)
                risk = price - sl_price
            else:
                sl_price = ob.ob_high * (1 + p.sl_ob_buffer_pct)
                risk = sl_price - price
            if risk > 0 and risk / price <= 0.04:
                hits_risk_ok += 1
            break

print(f"Non-mitigated OB touches WITH event: {hits_not_mitigated}")
print(f"After risk check: {hits_risk_ok}")
