"""
Technical Analysis Engine.

Takes raw OHLC(V) candles and returns a fully-populated pandas DataFrame
with EMA 20/50/200, RSI 14, MACD(12,26,9), Stochastic(14,3,3), Bollinger
Bands(20,2), and ATR(14) — plus derived structure: trend, cross events,
and horizontal support/resistance levels.
"""
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


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
    n = len(df)

    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator() if n >= 20 else np.nan
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator() if n >= 50 else np.nan
    df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator() if n >= 200 else np.nan

    if n >= 14:
        df["rsi14"] = RSIIndicator(df["close"], window=14).rsi()
    else:
        df["rsi14"] = np.nan

    if n >= 26:
        macd_ind = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()
    else:
        df["macd"] = df["macd_signal"] = df["macd_hist"] = np.nan

    if n >= 14:
        stoch_ind = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
        df["stoch_k"] = stoch_ind.stoch()
        df["stoch_d"] = stoch_ind.stoch_signal()
    else:
        df["stoch_k"] = df["stoch_d"] = np.nan

    if n >= 20:
        bb_ind = BollingerBands(df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb_ind.bollinger_hband()
        df["bb_mid"] = bb_ind.bollinger_mavg()
        df["bb_lower"] = bb_ind.bollinger_lband()
    else:
        df["bb_upper"] = df["bb_mid"] = df["bb_lower"] = np.nan

    if n >= 14:
        df["atr14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    else:
        df["atr14"] = np.nan

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
