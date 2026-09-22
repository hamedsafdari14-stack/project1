"""
Actionable Signal Generator.

Combines trend structure, oscillator readings, volatility (ATR), horizontal
S/R levels, and news sentiment into a single BUY / SELL / WAIT decision with
entry, stop-loss, take-profit levels, a risk/reward ratio, and a rule-based
confidence score.

IMPORTANT: This is a deterministic, rule-based scoring system (multi-factor
confluence), not a trained predictive model and not financial advice. Every
weight below is a design choice, documented inline, so it can be audited and
tuned.
"""
import pandas as pd

MIN_RRR = 2.0  # minimum acceptable reward:risk, per spec


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _score_trend(trend: dict) -> int:
    if trend["structure"] == "BULLISH":
        score = 2
    elif trend["structure"] == "BEARISH":
        score = -2
    else:
        score = 0
    if trend["golden_cross"]:
        score += 2
    if trend["death_cross"]:
        score -= 2
    return score


def _score_rsi(rsi: float | None) -> int:
    if rsi is None:
        return 0
    if rsi < 30:
        return 2       # oversold -> bullish mean-reversion bias
    if rsi < 45:
        return 1
    if rsi > 70:
        return -2      # overbought -> bearish bias
    if rsi > 55:
        return -1
    return 0


def _score_macd(macd: float | None, signal: float | None, hist: float | None) -> int:
    if macd is None or signal is None:
        return 0
    score = 1 if macd > signal else -1
    if hist is not None:
        score += 1 if hist > 0 else -1
    return score


def _score_stoch(k: float | None, d: float | None) -> int:
    if k is None or d is None:
        return 0
    score = 0
    if k < 20 and d < 20:
        score += 1
    if k > 80 and d > 80:
        score -= 1
    score += 1 if k > d else -1
    return score


def _score_bbands(close: float | None, upper: float | None, lower: float | None) -> int:
    if close is None or upper is None or lower is None:
        return 0
    if close <= lower:
        return 1   # pressed against lower band -> reversion-up bias
    if close >= upper:
        return -1  # pressed against upper band -> reversion-down bias
    return 0


def _score_sentiment(overall: str) -> int:
    return {"Positive": 1, "Negative": -1, "Neutral": 0}.get(overall, 0)


def _nearest_above(levels: list[float], price: float) -> list[float]:
    return sorted([lvl for lvl in levels if lvl > price])


def _nearest_below(levels: list[float], price: float) -> list[float]:
    return sorted([lvl for lvl in levels if lvl < price], reverse=True)


def generate_signal(
    snap: dict,
    trend: dict,
    sr: dict,
    sentiment: dict,
    df: pd.DataFrame,
) -> dict:
    price = snap["close"]
    atr = snap.get("atr14") or (price * 0.02)  # fallback ~2% synthetic ATR if unavailable

    # ---- 1. Multi-indicator confluence score -----------------------------
    components = {
        "trend_structure": _score_trend(trend),
        "rsi": _score_rsi(snap.get("rsi14")),
        "macd": _score_macd(snap.get("macd"), snap.get("macd_signal"), snap.get("macd_hist")),
        "stochastic": _score_stoch(snap.get("stoch_k"), snap.get("stoch_d")),
        "bollinger": _score_bbands(price, snap.get("bb_upper"), snap.get("bb_lower")),
        "news_sentiment": _score_sentiment(sentiment.get("overall_sentiment", "Neutral")),
    }
    raw_score = sum(components.values())
    max_possible = 4 + 2 + 2 + 2 + 1 + 1  # sum of each factor's max absolute weight

    # ---- 2. Direction ------------------------------------------------------
    if raw_score >= 3:
        direction = "BUY / LONG"
    elif raw_score <= -3:
        direction = "SELL / SHORT"
    else:
        direction = "WAIT / NO ENTRY"

    confidence = round(_clamp(50 + (raw_score / max_possible) * 50, 5, 95), 1)

    # ---- 3. Entry / SL / TP --------------------------------------------
    resistances = _nearest_above(sr["resistance"], price)
    supports = _nearest_below(sr["support"], price)

    entry = price
    entry_range = None
    tps: list[float] = []
    sl = None
    rrr = None

    if direction == "BUY / LONG":
        sl_struct = supports[0] if supports else price - 2 * atr
        sl = round(min(sl_struct, price - 1.2 * atr), 8)
        risk = price - sl

        fib_ext_1 = price + risk * 1.618
        fib_ext_2 = price + risk * 2.618
        struct_targets = resistances[:2]
        candidate_tps = sorted(set(struct_targets + [round(fib_ext_1, 8), round(fib_ext_2, 8)]))
        tps = [t for t in candidate_tps if t > price][:3]
        if not tps:
            tps = [round(price + risk * m, 8) for m in (1.5, 2.5, 4)]
        entry_range = [round(price - 0.3 * atr, 8), round(price, 8)]
        rrr = round((tps[0] - price) / risk, 2) if risk > 0 else None

    elif direction == "SELL / SHORT":
        sl_struct = resistances[0] if resistances else price + 2 * atr
        sl = round(max(sl_struct, price + 1.2 * atr), 8)
        risk = sl - price

        fib_ext_1 = price - risk * 1.618
        fib_ext_2 = price - risk * 2.618
        struct_targets = supports[:2]
        candidate_tps = sorted(set(struct_targets + [round(fib_ext_1, 8), round(fib_ext_2, 8)]), reverse=True)
        tps = [t for t in candidate_tps if t < price][:3]
        if not tps:
            tps = [round(price - risk * m, 8) for m in (1.5, 2.5, 4)]
        entry_range = [round(price, 8), round(price + 0.3 * atr, 8)]
        rrr = round((price - tps[0]) / risk, 2) if risk > 0 else None

    else:
        # WAIT: no trade, but still surface the levels that WOULD flip the bias
        sl = None
        tps = []

    # ---- 4. Enforce minimum RRR by trimming to TP1 that satisfies it, or
    # flag when even TP1 doesn't reach 1:2 -----------------------------
    meets_min_rrr = rrr is not None and rrr >= MIN_RRR
    if direction != "WAIT / NO ENTRY" and not meets_min_rrr:
        # Downgrade to WAIT if best available target can't clear 1:2 —
        # spec requires RRR >= 1:2, so we don't issue a trade signal that
        # violates it.
        direction = "WAIT / NO ENTRY"
        confidence = round(min(confidence, 45), 1)

    # ---- 5. Key drivers narrative ---------------------------------------
    drivers = _build_drivers(components, trend, snap, sentiment)

    return {
        "signal": direction,
        "confidence_pct": confidence,
        "entry_price": round(entry, 8) if direction != "WAIT / NO ENTRY" else None,
        "entry_range": entry_range if direction != "WAIT / NO ENTRY" else None,
        "take_profits": tps,
        "stop_loss": sl if direction != "WAIT / NO ENTRY" else None,
        "risk_reward_ratio": rrr if direction != "WAIT / NO ENTRY" else None,
        "min_required_rrr": MIN_RRR,
        "score_breakdown": components,
        "raw_score": raw_score,
        "key_drivers": drivers,
        "support_levels": sr["support"],
        "resistance_levels": sr["resistance"],
        "atr14": round(atr, 8),
    }


def _build_drivers(components: dict, trend: dict, snap: dict, sentiment: dict) -> list[str]:
    drivers = []

    structure = trend["structure"].capitalize()
    drivers.append(
        f"Market structure is {structure}: price is "
        f"{trend['price_vs_ema20']} EMA20, {trend['price_vs_ema50']} EMA50, "
        f"{trend['price_vs_ema200']} EMA200."
    )
    if trend["golden_cross"]:
        drivers.append("A Golden Cross (EMA50 crossing above EMA200) just triggered — bullish long-term signal.")
    if trend["death_cross"]:
        drivers.append("A Death Cross (EMA50 crossing below EMA200) just triggered — bearish long-term signal.")

    rsi = snap.get("rsi14")
    if rsi is not None:
        if rsi < 30:
            drivers.append(f"RSI(14) at {rsi:.1f} is in oversold territory, favoring a bounce.")
        elif rsi > 70:
            drivers.append(f"RSI(14) at {rsi:.1f} is in overbought territory, favoring a pullback.")
        else:
            drivers.append(f"RSI(14) is neutral at {rsi:.1f}, offering no strong extreme.")

    macd, macd_sig = snap.get("macd"), snap.get("macd_signal")
    if macd is not None and macd_sig is not None:
        rel = "above" if macd > macd_sig else "below"
        drivers.append(f"MACD line is {rel} its signal line, supporting the {'bullish' if rel == 'above' else 'bearish'} case.")

    sent = sentiment.get("overall_sentiment", "Neutral")
    if sent != "Neutral":
        drivers.append(f"News flow skews {sent.lower()}, {'reinforcing' if components['news_sentiment'] * components['trend_structure'] >= 0 else 'partially offsetting'} the technical picture.")
    else:
        drivers.append("News sentiment is currently neutral / mixed, with no strong catalyst either way.")

    return drivers[:4]
