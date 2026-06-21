"""
Live paper trading watcher.
Checks profitable pairs every hour for new signals.
Tracks virtual open positions and P&L.
"""

import time
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict


@dataclass
class OpenPosition:
    symbol: str
    direction: str       # "long" / "short"
    entry: float
    sl: float
    tp: float
    entry_time: str
    size_usd: float      # dollar size of position
    risk_usd: float      # dollar risk


@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    entry: float
    close_price: float
    entry_time: str
    close_time: str
    pnl_usd: float
    result: str          # "WIN" / "LOSS"


class LiveWatcher:
    STATE_FILE = "live_state.json"

    def __init__(
        self,
        pairs: list[str],
        br_params_per_pair: dict,
        capital_per_pair: float = 10_000.0,
        risk_pct: float = 1.0,
        poll_seconds: int = 3600,   # 1 hour
    ):
        self.pairs = pairs
        self.br_params_per_pair = br_params_per_pair
        self.capital_per_pair = capital_per_pair
        self.risk_pct = risk_pct
        self.poll_seconds = poll_seconds

        self.open_positions: dict[str, OpenPosition] = {}
        self.closed_trades: list[ClosedTrade] = []
        self._load_state()

    # ── State persistence ───────────────────────────────────────────────────
    def _save_state(self):
        state = {
            "open_positions": {k: asdict(v) for k, v in self.open_positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades],
        }
        with open(self.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        if not os.path.exists(self.STATE_FILE):
            return
        try:
            with open(self.STATE_FILE) as f:
                state = json.load(f)
            self.open_positions = {
                k: OpenPosition(**v) for k, v in state.get("open_positions", {}).items()
            }
            self.closed_trades = [ClosedTrade(**t) for t in state.get("closed_trades", [])]
            self._log(f"Loaded state: {len(self.open_positions)} open, {len(self.closed_trades)} closed")
        except Exception as e:
            self._log(f"State load error: {e}")

    # ── Logging ─────────────────────────────────────────────────────────────
    def _log(self, msg: str, level: str = "INFO"):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        colors = {"INFO": "\033[36m", "SIGNAL": "\033[92m",
                  "WIN": "\033[92m", "LOSS": "\033[91m", "WARN": "\033[93m"}
        c = colors.get(level, "")
        reset = "\033[0m"
        print(f"{c}[{ts}] [{level}] {msg}{reset}")

    # ── Check open positions against current price ───────────────────────────
    def _check_exits(self, symbol: str, current_high: float, current_low: float):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        now = datetime.now(timezone.utc).isoformat()

        hit_tp = hit_sl = False
        if pos.direction == "long":
            hit_tp = current_high >= pos.tp
            hit_sl = current_low <= pos.sl
        else:
            hit_tp = current_low <= pos.tp
            hit_sl = current_high >= pos.sl

        if hit_tp or hit_sl:
            close_price = pos.tp if hit_tp else pos.sl
            risk = abs(pos.entry - pos.sl)
            if hit_tp:
                pnl = pos.risk_usd * 2.0
                result = "WIN"
            else:
                pnl = -pos.risk_usd
                result = "LOSS"

            trade = ClosedTrade(
                symbol=symbol,
                direction=pos.direction,
                entry=pos.entry,
                close_price=close_price,
                entry_time=pos.entry_time,
                close_time=now,
                pnl_usd=round(pnl, 2),
                result=result,
            )
            self.closed_trades.append(trade)
            del self.open_positions[symbol]
            self._log(
                f"{symbol} {result}: {pos.direction.upper()} {pos.entry:.4f} -> {close_price:.4f}  PnL: ${pnl:+.2f}",
                level=result,
            )
            self._save_state()

    # ── Check for new entry signal ───────────────────────────────────────────
    def _check_entry(self, symbol: str, df):
        if symbol in self.open_positions:
            return  # already in a trade on this pair

        from ..strategies.breakout_retest import run_breakout_retest
        params = self.br_params_per_pair.get(symbol)
        if params is None:
            from ..strategies.breakout_retest import BRParams
            params = BRParams()

        # Generate signals on the full df, check if the LAST candle triggered
        signals = run_breakout_retest(df, params)
        if not signals:
            return

        last_sig = signals[-1]
        last_candle_idx = len(df) - 2  # -2 because we don't trade on open candle

        if last_sig["idx"] != last_candle_idx:
            return  # signal is not on the latest closed candle

        entry = last_sig["entry"]
        sl = last_sig["sl"]
        tp = last_sig["tp"]
        direction = last_sig["direction"]
        risk_per_trade = self.capital_per_pair * (self.risk_pct / 100)
        risk_price = abs(entry - sl)
        size_usd = (risk_per_trade / risk_price) * entry if risk_price > 0 else 0

        pos = OpenPosition(
            symbol=symbol,
            direction=direction,
            entry=entry,
            sl=sl,
            tp=tp,
            entry_time=datetime.now(timezone.utc).isoformat(),
            size_usd=round(size_usd, 2),
            risk_usd=round(risk_per_trade, 2),
        )
        self.open_positions[symbol] = pos
        self._save_state()

        self._log(
            f"SIGNAL  {symbol}  {direction.upper()}  Entry:{entry:.4f}  "
            f"SL:{sl:.4f}  TP:{tp:.4f}  Risk:${risk_per_trade:.0f}",
            level="SIGNAL",
        )

    # ── Print status ─────────────────────────────────────────────────────────
    def _print_status(self):
        total_pnl = sum(t.pnl_usd for t in self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t.result == "WIN")
        losses = sum(1 for t in self.closed_trades if t.result == "LOSS")
        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

        print("\n" + "=" * 60)
        print("  LIVE BOT STATUS")
        print("=" * 60)
        print(f"  Watching: {', '.join(p.split('/')[0] for p in self.pairs)}")
        print(f"  Closed trades: {len(self.closed_trades)}  WR: {wr:.1f}%  PnL: ${total_pnl:+.2f}")
        print(f"  Open positions: {len(self.open_positions)}")
        for sym, pos in self.open_positions.items():
            print(f"    {sym}: {pos.direction.upper()} @ {pos.entry:.4f}  "
                  f"SL:{pos.sl:.4f}  TP:{pos.tp:.4f}")
        print("=" * 60 + "\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    def run(self):
        from ..data.fetcher import fetch_binance

        self._log(f"Live bot started | Pairs: {self.pairs} | Poll: {self.poll_seconds}s")
        self._log("Press Ctrl+C to stop cleanly.")
        self._print_status()

        try:
            while True:
                self._log("Scanning pairs...")
                for symbol in self.pairs:
                    try:
                        df = fetch_binance(symbol, "1h", limit=500, use_cache=False)
                        if df is None or len(df) < 50:
                            self._log(f"{symbol}: insufficient data", "WARN")
                            continue

                        self._check_exits(
                            symbol,
                            current_high=df["high"].iloc[-1],
                            current_low=df["low"].iloc[-1],
                        )
                        self._check_entry(symbol, df)

                    except Exception as e:
                        err = str(e)[:300]
                        self._log(f"{symbol} error: {err}", "WARN")

                self._print_status()
                self._log(f"Next scan in {self.poll_seconds // 60} min...")
                time.sleep(self.poll_seconds)

        except KeyboardInterrupt:
            self._log("Stopped by user. Saving state...")
            self._save_state()
            self._print_status()
