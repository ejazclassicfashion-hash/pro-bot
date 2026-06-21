# Deploying Pro Bot (web dashboard)

The dashboard is a stateful Dash/Flask app, so it needs a host that keeps a
process running — **not** Vercel/Netlify (those are serverless and will drop
the in-memory signal state and time out on backtests).

Use **Render** or **Railway**. Both run the `Procfile` directly.

---

## Option A — Render (recommended, has a free tier)

1. Push this folder to a GitHub repo.
2. Go to <https://render.com> → New → Blueprint → connect the repo.
3. Render reads `render.yaml` automatically. It deploys in the **Frankfurt**
   region on purpose — US regions are blocked by Binance.
4. Click Deploy. First build takes ~3-4 min. You get a public URL like
   `https://pro-bot.onrender.com`.

Free tier note: the service sleeps after 15 min idle and cold-starts on the
next visit (~30s). Fine for personal use.

---

## Option B — Railway

1. Push to GitHub.
2. <https://railway.app> → New Project → Deploy from GitHub repo.
3. Railway detects the `Procfile`. In **Settings → Region**, pick
   **EU West** or **Singapore** (NOT US) to avoid the Binance block.
4. Add a variable `EXCHANGE=binance` (optional — the app auto-falls back to
   bybit/okx/kucoin/kraken if Binance is unreachable).
5. Deploy. Railway gives you a public domain under Settings → Networking.

---

## Binance geo-block

`api.binance.com` rejects requests from US datacenters with the
`exchangeInfo` error you saw locally on some networks. Two safety nets are
built in:

1. **Pick a non-US region** (Frankfurt / EU / Singapore) — handled in the
   configs above.
2. **Auto-fallback** — `fetcher.py` tries
   `binance → bybit → okx → kucoin → kraken`. All return identical USDT
   OHLCV, so the strategy keeps working even if Binance is blocked. Force a
   specific one with the `EXCHANGE` env var.

---

## What runs where

- `app.py`  — the web dashboard (this is what gets deployed).
- `run.py`  — the CLI version (screener / live bot in the terminal). Not
  needed on the server, but you can still run it locally.

## Local run

```bash
pip install -r requirements.txt
python app.py            # http://localhost:8050
```
