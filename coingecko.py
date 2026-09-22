"""
CoinGecko Demo API integration.

Handles:
- symbol -> coin id resolution (/search)
- current market snapshot (/coins/markets)
- historical OHLC candles (/coins/{id}/ohlc)
- historical market_chart (fallback, for granular close-price series)

All calls go through a small in-memory TTL cache to stay well within the
free Demo API's rate limit (~30 calls/min).
"""
import os
import time
from typing import Any

import httpx

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
API_KEY = os.getenv("COINGECKO_API_KEY", "")

# ---- tiny in-memory cache -------------------------------------------------
_cache: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str, ttl_seconds: int):
    hit = _cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.time() - ts > ttl_seconds:
        return None
    return value


def _cache_set(key: str, value: Any):
    _cache[key] = (time.time(), value)


def _headers() -> dict:
    # Demo API key goes in this header (not Authorization)
    if API_KEY:
        return {"x-cg-demo-api-key": API_KEY, "accept": "application/json"}
    return {"accept": "application/json"}


class CoinGeckoError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


async def _get(path: str, params: dict | None = None, ttl: int = 30) -> Any:
    cache_key = f"{path}?{params}"
    cached = _cache_get(cache_key, ttl)
    if cached is not None:
        return cached

    url = f"{COINGECKO_BASE}{path}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params or {}, headers=_headers())

    if resp.status_code == 429:
        raise CoinGeckoError(
            "CoinGecko rate limit hit. Wait a few seconds and try again.", 429
        )
    if resp.status_code != 200:
        raise CoinGeckoError(
            f"CoinGecko request failed ({resp.status_code}): {resp.text[:200]}",
            resp.status_code,
        )

    data = resp.json()
    _cache_set(cache_key, data)
    return data


async def resolve_symbol(symbol: str) -> dict:
    """Resolve a free-text ticker/name (e.g. 'sol', 'BTC', 'ethereum') to a
    CoinGecko coin id + display metadata, preferring the highest-market-cap
    match when several coins share a ticker.
    """
    symbol = symbol.strip().lower()
    data = await _get("/search", {"query": symbol}, ttl=3600)
    coins = data.get("coins", [])
    if not coins:
        raise CoinGeckoError(f"No coin found matching '{symbol}'.", 404)

    exact = [c for c in coins if c["symbol"].lower() == symbol]
    candidates = exact if exact else coins
    # /search returns coins pre-sorted by market cap rank (None = worst)
    candidates.sort(key=lambda c: (c.get("market_cap_rank") is None, c.get("market_cap_rank") or 0))
    best = candidates[0]
    return {
        "id": best["id"],
        "symbol": best["symbol"].upper(),
        "name": best["name"],
        "thumb": best.get("large") or best.get("thumb"),
    }


async def get_market_snapshot(coin_id: str, vs_currency: str = "usd") -> dict:
    data = await _get(
        "/coins/markets",
        {
            "vs_currency": vs_currency,
            "ids": coin_id,
            "price_change_percentage": "1h,24h,7d",
        },
        ttl=20,
    )
    if not data:
        raise CoinGeckoError(f"No market data for '{coin_id}'.", 404)
    return data[0]


async def get_ohlc(coin_id: str, days: int = 30, vs_currency: str = "usd") -> list[list[float]]:
    """Returns list of [timestamp_ms, open, high, low, close]."""
    data = await _get(
        f"/coins/{coin_id}/ohlc",
        {"vs_currency": vs_currency, "days": days},
        ttl=60,
    )
    if not data or len(data) < 10:
        raise CoinGeckoError(
            f"Not enough OHLC history for '{coin_id}' over {days}d.", 422
        )
    return data


async def get_market_chart(coin_id: str, days: int = 30, vs_currency: str = "usd") -> dict:
    """Finer-grained series (prices, volumes) than /ohlc — used to backfill
    volume, which /ohlc does not provide."""
    data = await _get(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": vs_currency, "days": days},
        ttl=60,
    )
    return data
