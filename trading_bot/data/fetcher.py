"""
Real OHLCV data from Binance via ccxt — no API key needed for historical data.
Binance limit = 1000 candles per call, so we paginate to get as many as needed.
"""

import pandas as pd
import os
import json
import time

_exchange = None  # singleton — created once, reused

# Exchange priority. Binance is best but geo-blocked on US cloud servers
# (Railway/Render/Vercel). Set env EXCHANGE to force one, else we auto-fallback.
# All of these expose the same USDT spot pairs with identical OHLCV format.
_EXCHANGE_CHAIN = ["binance", "bybit", "okx", "kucoin", "kraken"]


def _build_exchange(name: str):
    import ccxt
    klass = getattr(ccxt, name)
    return klass({
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })


def _get_exchange():
    """Return a working exchange, falling back if one is geo-blocked."""
    global _exchange
    if _exchange is not None:
        return _exchange

    forced = os.environ.get("EXCHANGE")
    chain = [forced] if forced else _EXCHANGE_CHAIN

    for name in chain:
        try:
            ex = _build_exchange(name)
            ex.load_markets()           # this is what trips geo-blocks
            _exchange = ex
            print(f"[Data] Using exchange: {name}")
            return _exchange
        except Exception as e:
            print(f"[Data] {name} unavailable ({str(e)[:60]}), trying next...")
            continue

    # Last resort: return binance anyway (cached data may still work)
    _exchange = _build_exchange("binance")
    return _exchange


def fetch_binance(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 1000,
    cache_dir: str = "data_cache",
    force_refresh: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance with pagination — handles any limit > 1000.
    Caches locally so you don't re-download on every run.
    Delete data_cache/ folder to force a fresh download.
    """
    import ccxt

    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f"{symbol.replace('/', '_')}_{timeframe}_{limit}.json"
    )

    if os.path.exists(cache_file) and not force_refresh and use_cache:
        print(f"[Data] Loading from cache: {cache_file}")
        with open(cache_file) as f:
            raw = json.load(f)
    else:
        exchange = _get_exchange()

        # Binance hard limit = 1000 per call, so paginate backwards from now
        MAX_PER_CALL = 1000
        all_candles = []
        since = None  # start from now going backwards

        print(f"[Data] Fetching {limit} {timeframe} candles for {symbol} from Binance...")

        # Compute the earliest timestamp we need
        # timeframe string → milliseconds per candle
        tf_ms = _timeframe_to_ms(timeframe)
        now_ms = int(time.time() * 1000)
        target_since = now_ms - (limit * tf_ms)

        since = target_since
        while len(all_candles) < limit:
            batch_size = min(MAX_PER_CALL, limit - len(all_candles))
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
            if not batch:
                break
            all_candles.extend(batch)
            # Move since forward to after the last fetched candle
            since = batch[-1][0] + tf_ms
            if since >= now_ms:
                break
            if len(batch) < batch_size:
                break  # exchange returned less than asked = no more data
            time.sleep(exchange.rateLimit / 1000)

        # Deduplicate and sort by timestamp
        seen = set()
        raw = []
        for c in all_candles:
            if c[0] not in seen:
                seen.add(c[0])
                raw.append(c)
        raw.sort(key=lambda x: x[0])

        with open(cache_file, "w") as f:
            json.dump(raw, f)
        print(f"[Data] Fetched {len(raw)} candles. Cached to {cache_file}")

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df


def _timeframe_to_ms(timeframe: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    for suffix, ms in units.items():
        if timeframe.endswith(suffix):
            n = int(timeframe[:-1])
            return n * ms
    return 3_600_000  # default 1h


def fetch_news_headlines(keywords: str = "bitcoin crypto") -> list[dict]:
    """
    Scrape recent crypto news headlines for sentiment analysis.
    Uses free CryptoPanic API (no key needed for basic feed).
    """
    import requests

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": "public", "currencies": "BTC", "filter": "hot"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        results = data.get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "published": r.get("published_at", ""),
                "source": r.get("source", {}).get("title", ""),
                "url": r.get("url", ""),
            }
            for r in results[:20]
        ]
    except Exception as e:
        print(f"[News] Fetch failed: {e}")
        return []
