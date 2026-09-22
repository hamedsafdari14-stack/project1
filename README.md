# Signal Desk — AI Crypto Trading Signal & Analysis System

A full-stack app that pulls live market data from CoinGecko, runs a technical
analysis engine (EMA/RSI/MACD/Stochastic/Bollinger/ATR + support-resistance),
blends in news sentiment, and outputs a rule-based BUY / SELL / WAIT signal
with entry, stop-loss, take-profit levels and a confidence score — shown on
a dark, TradingView-style dashboard.

**This is a decision-support tool, not financial advice.** The signal engine
is a transparent, auditable scoring system (see `backend/services/signals.py`),
not a trained predictive model. Crypto markets are volatile; always size
positions to your own risk tolerance.

## Project structure

```
crypto-signal-app/
├── backend/
│   ├── main.py                 FastAPI app + /api routes
│   ├── requirements.txt
│   ├── .env.example
│   └── services/
│       ├── coingecko.py        CoinGecko Demo API client (symbol resolve, OHLC, market data)
│       ├── indicators.py       Technical analysis (pandas-ta): EMAs, RSI, MACD, Stoch, BB, ATR, S/R
│       ├── news.py             News fetch (CryptoPanic or RSS fallback) + keyword sentiment scoring
│       └── signals.py          Multi-indicator confluence scoring → BUY/SELL/WAIT + TP/SL/RRR
└── frontend/
    └── index.html              Single-file dashboard (vanilla JS + TradingView Lightweight Charts)
```

## 1. Backend setup

Requires Python 3.10+.

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env:
#   COINGECKO_API_KEY=...     (free "Demo" key from https://www.coingecko.com/en/api/pricing)
#   CRYPTOPANIC_API_KEY=...   (optional — free token from https://cryptopanic.com/developers/api/)
#   ALLOWED_ORIGINS=http://127.0.0.1:5500,null

uvicorn main:app --reload --port 8000
```

The API is now live at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

**Note on the CoinGecko key:** the app runs on the free tier even with no key
set (public rate limits, ~10-30 calls/min), but a free Demo key raises your
limits and reliability meaningfully — recommended for anything beyond quick
testing.

### Key endpoints
- `GET /api/resolve?symbol=sol` — ticker/name → CoinGecko coin id (used for search-as-you-type)
- `GET /api/analyze?symbol=sol&days=30` — full pipeline: market data, indicators, sentiment, signal
- `GET /api/health` — liveness check

## 2. Frontend setup

No build step — it's a single static HTML file.

```bash
cd frontend
python3 -m http.server 5500
```

Open `http://127.0.0.1:5500` in your browser. If your backend runs on a
different host/port, set it before the page loads by editing the top of
`index.html`'s `<script>` block or injecting:

```html
<script>window.SIGNAL_DESK_API_BASE = "http://localhost:8000";</script>
```

(add this line right before the closing `</head>`, above the Lightweight
Charts `<script>` tag).

Then just type a symbol (`SOL`, `BTC`, `ETH`, ...) and hit **Analyze**.

## 3. How the signal is generated

`services/signals.py` scores six independent factors, each contributing a
small, documented weight to a raw confluence score:

| Factor | Signal used | Max weight |
|---|---|---|
| Trend structure | Price vs EMA20/50/200 + Golden/Death Cross | ±4 |
| RSI (14) | Oversold/overbought extremes | ±2 |
| MACD | Line vs signal, histogram sign | ±2 |
| Stochastic | %K/%D extremes and crossover | ±2 |
| Bollinger Bands | Price pressed against a band | ±1 |
| News sentiment | Keyword/vote-based article scoring | ±1 |

- Score **≥ +3** → `BUY / LONG`, **≤ -3** → `SELL / SHORT`, else `WAIT / NO ENTRY`.
- **Stop-loss** is placed at the nearest structural support/resistance, floored
  by 1.2× ATR(14), so it never sits inside normal volatility noise.
- **Take-profit levels** blend the nearest 1-2 horizontal S/R levels with
  1.618/2.618 Fibonacci extensions off the entry-to-stop risk distance.
- **Every trade signal is required to clear a minimum 1:2 risk/reward.** If
  the best available target can't clear that bar, the system automatically
  downgrades the call to `WAIT / NO ENTRY` rather than issue a sub-1:2 trade.
- **Confidence %** is a linear rescale of the raw score against the maximum
  possible score (±12), clamped to 5–95%.

All the weights above are deliberately simple and exposed in code specifically
so they can be audited, backtested, and tuned — this is a starting framework,
not a black box.

## 4. Deployment notes

- **Backend**: any ASGI host works (Render, Railway, Fly.io, a plain VM with
  `uvicorn`/`gunicorn -k uvicorn.workers.UvicornWorker`). Set `COINGECKO_API_KEY`,
  `CRYPTOPANIC_API_KEY`, and `ALLOWED_ORIGINS` (your deployed frontend's origin)
  as environment variables — don't commit `.env`.
- **Frontend**: it's static — deploy `frontend/index.html` to Netlify, Vercel,
  Cloudflare Pages, or S3/CloudFront. Set `window.SIGNAL_DESK_API_BASE` to your
  deployed backend URL as shown above.
- **CORS**: update `ALLOWED_ORIGINS` in the backend `.env` to your production
  frontend domain(s) once deployed.
- **Rate limits**: the backend has a small in-memory TTL cache per endpoint
  (20-60s) to stay within CoinGecko's free-tier limits under repeated queries
  for the same symbol. For production traffic, consider Redis-backed caching
  and a paid CoinGecko tier.

## 5. Extending it

- Swap the keyword-based sentiment scorer in `news.py` for a real NLP model
  (e.g., a hosted FinBERT endpoint) by replacing `_score_text`.
- Add more timeframes by wiring the frontend's timeframe buttons to different
  `days` values (already supported end-to-end).
- Persist historical signals to a database to track win-rate and calibrate
  the scoring weights over time.
