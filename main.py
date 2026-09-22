"""
Automated AI Crypto Trading Signal & Analysis System — FastAPI backend.

Run with:
    uvicorn main:app --reload --port 8000

See README.md at the project root for full setup instructions.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services import coingecko, news, signals
from services.coingecko import CoinGeckoError
from services.indicators import (
    build_dataframe,
    compute_indicators,
    detect_trend,
    find_support_resistance,
    latest_snapshot,
)

app = FastAPI(
    title="AI Crypto Trading Signal API",
    description="Real-time technical + sentiment analysis and rule-based trading signals.",
    version="1.0.0",
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = ["*"] if allowed_origins == "*" else [o.strip() for o in allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CandlePoint(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/resolve")
async def resolve(symbol: str = Query(..., min_length=1, max_length=20)):
    """Resolve a ticker/name to a CoinGecko coin id (used for search-as-you-type)."""
    try:
        return await coingecko.resolve_symbol(symbol)
    except CoinGeckoError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@app.get("/api/analyze")
async def analyze(
    symbol: str = Query(..., description="Ticker or name, e.g. 'sol', 'BTC', 'ethereum'"),
    days: int = Query(30, ge=2, le=365, description="Lookback window for OHLC history"),
    vs_currency: str = Query("usd"),
):
    """
    Full pipeline: resolve symbol -> fetch market data + OHLC -> compute
    indicators -> fetch news/sentiment -> generate trading signal.
    """
    try:
        coin = await coingecko.resolve_symbol(symbol)
        coin_id = coin["id"]

        market = await coingecko.get_market_snapshot(coin_id, vs_currency)
        ohlc = await coingecko.get_ohlc(coin_id, days=days, vs_currency=vs_currency)

        volumes = None
        try:
            chart = await coingecko.get_market_chart(coin_id, days=days, vs_currency=vs_currency)
            volumes = chart.get("total_volumes")
        except CoinGeckoError:
            pass  # volume is a nice-to-have; don't fail the whole request

        df = build_dataframe(ohlc, volumes)
        if len(df) < 20:
            raise HTTPException(
                status_code=422,
                detail="Not enough historical candles to compute reliable indicators. Try a longer 'days' window.",
            )

        df = compute_indicators(df)
        snap = latest_snapshot(df)
        trend = detect_trend(df)
        sr = find_support_resistance(df)
        sentiment = await news.get_news_and_sentiment(coin["name"], coin["symbol"])
        signal = signals.generate_signal(snap, trend, sr, sentiment, df)

        candles = [
            {
                "time": int(ts.timestamp()),
                "open": round(float(row["open"]), 8),
                "high": round(float(row["high"]), 8),
                "low": round(float(row["low"]), 8),
                "close": round(float(row["close"]), 8),
            }
            for ts, row in df.iterrows()
        ]

        return {
            "coin": coin,
            "market": {
                "current_price": market.get("current_price"),
                "market_cap": market.get("market_cap"),
                "market_cap_rank": market.get("market_cap_rank"),
                "total_volume": market.get("total_volume"),
                "price_change_pct_1h": market.get("price_change_percentage_1h_in_currency"),
                "price_change_pct_24h": market.get("price_change_percentage_24h_in_currency"),
                "price_change_pct_7d": market.get("price_change_percentage_7d_in_currency"),
                "ath": market.get("ath"),
                "atl": market.get("atl"),
            },
            "candles": candles,
            "indicators": snap,
            "trend": trend,
            "support_resistance": sr,
            "sentiment": sentiment,
            "signal": signal,
            "disclaimer": (
                "Automated, rule-based technical output for informational purposes only. "
                "Not financial advice. Crypto markets are highly volatile — always size "
                "positions to your own risk tolerance and do your own research."
            ),
        }

    except CoinGeckoError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
