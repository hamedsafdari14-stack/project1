"""
Technical Analysis Engine.

Takes raw OHLC(V) candles and returns a fully-populated pandas DataFrame
with EMA 20/50/200, RSI 14, MACD(12,26,9), Stochastic(14,3,3), Bollinger
Bands(20,2), and ATR(14) — plus derived structure: trend, cross events,
and horizontal support/resistance levels.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta


def build_dataframe(ohlc: list[list[float]], volumes: list[list[float]] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(ohlc, columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts").sort_index()

    if volumes:
        vol_df = pd.DataFrame(volumes, columns=["ts", "volume"])
        vol_df["ts"] = pd.to_datetime(vol_df["ts"], unit="ms")
        vol_df = vol_df.set_index("ts").sort_index()
        # align volume timestamps to nearest OHLC candle
        df = pd.merge_asof(df, vol_df, left_index=True, right_index=True, direction="nearest")
    else:
        df["volume"] = np.nan

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema20"] = ta.ema(df["close"], length=20)
    df["ema50"] = ta.ema(df["close"], length=50)
    df["ema200"] = ta.ema(df["close"], length=200)

    df["rsi14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.get("MACD_12_26_9")
        df["macd_signal"] = macd.get("MACDs_12_26_9")
        df["macd_hist"] = macd.get("MACDh_12_26_9")

    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    if stoch is not None:
        df["stoch_k"] = stoch.get("STOCHk_14_3_3")
        df["stoch_d"] = stoch.get("STOCHd_14_3_3")

    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None:
        df["bb_upper"] = bb.get("BBU_20_2.0")
        df["bb_mid"] = bb.get("BBM_20_2.0")
        df["bb_lower"] = bb.get("BBL_20_2.0")

    df["atr14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    return df


def find_support_resistance(df: pd.DataFrame, lookback: int = 60, n_levels: int = 4) -> dict:
    """Simple fractal/pivot-based S/R: local swing highs/lows over a lookback
    window, clustered into a handful of zones."""
    window = df.tail(lookback)
    highs, lows = window["high"], window["low"]

    swing_highs, swing_lows = [], []
    vals_h, vals_l = highs.values, lows.values
    for i in range(2, len(window) - 2):
        if vals_h[i] == max(vals_h[i - 2:i + 3]):
            swing_highs.append(vals_h[i])
        if vals_l[i] == min(vals_l[i - 2:i + 3]):
            swing_lows.append(vals_l[i])

    def cluster(levels: list[float], tolerance_pct: float = 0.008) -> list[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: list[list[float]] = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] <= tolerance_pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        # rank clusters by how many touches they got (stronger level)
        clusters.sort(key=len, reverse=True)
        return [float(np.mean(c)) for c in clusters[:n_levels]]

    resistance = sorted(cluster(swing_highs), reverse=True)
    support = sorted(cluster(swing_lows), reverse=True)
    return {"resistance": resistance, "support": support}


def detect_trend(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    price = last["close"]
    ema20, ema50, ema200 = last.get("ema20"), last.get("ema50"), last.get("ema200")

    golden_cross = death_cross = False
    if len(df) > 2 and pd.notna(ema50) and pd.notna(ema200):
        prev = df.iloc[-2]
        if pd.notna(prev.get("ema50")) and pd.notna(prev.get("ema200")):
            golden_cross = prev["ema50"] <= prev["ema200"] and ema50 > ema200
            death_cross = prev["ema50"] >= prev["ema200"] and ema50 < ema200

    bullish_votes = sum([
        pd.notna(ema20) and price > ema20,
        pd.notna(ema50) and price > ema50,
        pd.notna(ema200) and price > ema200,
        pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50,
    ])
    bearish_votes = sum([
        pd.notna(ema20) and price < ema20,
        pd.notna(ema50) and price < ema50,
        pd.notna(ema200) and price < ema200,
        pd.notna(ema20) and pd.notna(ema50) and ema20 < ema50,
    ])

    if bullish_votes >= 3:
        structure = "BULLISH"
    elif bearish_votes >= 3:
        structure = "BEARISH"
    else:
        structure = "RANGING"

    return {
        "structure": structure,
        "golden_cross": bool(golden_cross),
        "death_cross": bool(death_cross),
        "price_vs_ema20": "above" if pd.notna(ema20) and price > ema20 else "below",
        "price_vs_ema50": "above" if pd.notna(ema50) and price > ema50 else "below",
        "price_vs_ema200": "above" if pd.notna(ema200) and price > ema200 else "below",
    }


def latest_snapshot(df: pd.DataFrame) -> dict:
    """Flatten the last row of indicators into plain floats for the API response."""
    last = df.iloc[-1]

    def f(key):
        v = last.get(key)
        return None if v is None or pd.isna(v) else round(float(v), 6)

    return {
        "close": f("close"),
        "ema20": f("ema20"),
        "ema50": f("ema50"),
        "ema200": f("ema200"),
        "rsi14": f("rsi14"),
        "macd": f("macd"),
        "macd_signal": f("macd_signal"),
        "macd_hist": f("macd_hist"),
        "stoch_k": f("stoch_k"),
        "stoch_d": f("stoch_d"),
        "bb_upper": f("bb_upper"),
        "bb_mid": f("bb_mid"),
        "bb_lower": f("bb_lower"),
        "atr14": f("atr14"),
    }
