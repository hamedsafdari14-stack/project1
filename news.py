"""
News & Sentiment Analysis Engine.

Primary source: CryptoPanic free API (if CRYPTOPANIC_API_KEY is set) — it
already tags posts with community votes we can use as a sentiment proxy.
Fallback: CoinDesk RSS feed filtered by coin name/symbol, scored with a
lightweight keyword-based sentiment heuristic (no extra ML dependency).

This is a heuristic, not a financial-grade NLP sentiment model — treat the
score as a directional hint, not ground truth.
"""
import os
import re

import feedparser
import httpx

CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
COINDESK_RSS = "https://www.coindesk.com/arc/outboundfeeds/rss/"

POSITIVE_WORDS = {
    "surge", "rally", "bullish", "soar", "gain", "gains", "breakout", "upgrade",
    "adoption", "partnership", "approval", "approved", "inflow", "inflows",
    "record high", "all-time high", "ath", "outperform", "buy", "accumulate",
    "growth", "positive", "recover", "recovery", "boost", "milestone",
}
NEGATIVE_WORDS = {
    "crash", "plunge", "bearish", "selloff", "sell-off", "dump", "hack",
    "hacked", "exploit", "lawsuit", "ban", "banned", "outflow", "outflows",
    "decline", "downgrade", "investigation", "fraud", "collapse", "liquidation",
    "liquidated", "fear", "negative", "risk-off", "delist", "delisted",
}


def _score_text(text: str) -> int:
    text_l = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text_l)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text_l)
    return pos - neg


async def _fetch_cryptopanic(symbol: str) -> list[dict] | None:
    if not CRYPTOPANIC_KEY:
        return None
    url = "https://cryptopanic.com/api/free/v1/posts/"
    params = {"auth_token": CRYPTOPANIC_KEY, "currencies": symbol.upper(), "public": "true"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
    except Exception:
        return None

    articles = []
    for post in results[:12]:
        votes = post.get("votes", {})
        vote_score = votes.get("positive", 0) - votes.get("negative", 0)
        keyword_score = _score_text(post.get("title", ""))
        combined = vote_score + keyword_score
        articles.append({
            "title": post.get("title"),
            "url": post.get("url"),
            "source": (post.get("source") or {}).get("title", "CryptoPanic"),
            "published_at": post.get("published_at"),
            "sentiment": _label(combined),
            "score": combined,
        })
    return articles


async def _fetch_rss_fallback(coin_name: str, symbol: str) -> list[dict]:
    try:
        feed = feedparser.parse(COINDESK_RSS)
    except Exception:
        return []

    name_l = coin_name.lower()
    symbol_l = symbol.lower()
    articles = []
    for entry in feed.entries[:60]:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        haystack = f"{title} {summary}".lower()
        if name_l not in haystack and f" {symbol_l} " not in f" {haystack} ":
            continue
        score = _score_text(f"{title} {summary}")
        articles.append({
            "title": title,
            "url": entry.get("link"),
            "source": "CoinDesk",
            "published_at": entry.get("published", None),
            "sentiment": _label(score),
            "score": score,
        })
        if len(articles) >= 10:
            break
    return articles


def _label(score: int) -> str:
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


async def get_news_and_sentiment(coin_name: str, symbol: str) -> dict:
    articles = await _fetch_cryptopanic(symbol)
    source_used = "CryptoPanic"
    if not articles:
        articles = await _fetch_rss_fallback(coin_name, symbol)
        source_used = "CoinDesk RSS (fallback)"

    if not articles:
        return {
            "source": source_used,
            "overall_sentiment": "Neutral",
            "sentiment_score": 0,
            "articles": [],
            "note": "No recent coin-specific articles found; treat news input as neutral for this signal.",
        }

    total = sum(a["score"] for a in articles)
    pos = sum(1 for a in articles if a["sentiment"] == "Positive")
    neg = sum(1 for a in articles if a["sentiment"] == "Negative")

    if pos > neg and total > 0:
        overall = "Positive"
    elif neg > pos and total < 0:
        overall = "Negative"
    else:
        overall = "Neutral"

    return {
        "source": source_used,
        "overall_sentiment": overall,
        "sentiment_score": total,
        "articles": articles,
    }
