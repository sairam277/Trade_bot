"""Shared technical-indicator helpers, implemented to match Pine Script's
`ta.*` semantics exactly (not just "close enough") so ported strategies
behave the same as they did on TradingView.

Room for the advanced indicators discussed (order blocks, FVG, Fibonacci
retracement levels, volume absorption) to live here as they're added during
refinement.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's moving average, matching Pine's ta.rma(): seeded with a
    simple average over the first `length` values, then recursively
    smoothed with alpha = 1/length."""
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) < length:
        return pd.Series(out, index=series.index)
    out[length - 1] = np.nanmean(values[:length])
    for i in range(length, len(values)):
        out[i] = (out[i - 1] * (length - 1) + values[i]) / length
    return pd.Series(out, index=series.index)


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Wilder's ATR (ta.atr), matching Pine's default RMA-smoothed true range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return rma(true_range, length)


def vwma(source: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Volume-weighted moving average, matching Pine's ta.vwma(source, length):
    sma(source * volume, length) / sma(volume, length)."""
    return (source * volume).rolling(length).mean() / volume.rolling(length).mean()


# ---------------------------------------------------------------------------
# "Smart money concept" style filters (order blocks, FVG, volume absorption,
# Fibonacci). These are practical approximations built for filtering entries,
# not canonical ICT/SMC definitions — documented as such at each function.
# ---------------------------------------------------------------------------

def volume_absorption(df: pd.DataFrame, vol_lookback: int = 20, vol_mult: float = 1.5,
                       wick_ratio: float = 0.6) -> tuple[pd.Series, pd.Series]:
    """Flags bars where volume is well above average but the close still
    ends up strongly toward one side of the bar's range — read as that
    side's flow being absorbed without price following through the other
    way. Returns (bullish_absorption, bearish_absorption) boolean Series."""
    vol_avg = df["volume"].rolling(vol_lookback).mean()
    high_volume = df["volume"] > vol_avg * vol_mult
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    close_pos = (df["close"] - df["low"]) / rng  # 0 = closed at the low, 1 = closed at the high
    bullish = (high_volume & (close_pos > wick_ratio)).fillna(False)
    bearish = (high_volume & (close_pos < (1 - wick_ratio))).fillna(False)
    return bullish, bearish


def fair_value_gaps(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """3-candle imbalance (ICT-style Fair Value Gap): a bullish FVG is the
    gap between candle i-2's high and candle i's low (candle i-1 sits
    between them without overlap); bearish is the mirror image. Returns
    zone bounds as Series aligned to index i, NaN where no gap forms:
    (bull_low, bull_high, bear_low, bear_high)."""
    high, low = df["high"], df["low"]
    bull_gap = low > high.shift(2)
    bear_gap = high < low.shift(2)
    bull_low = high.shift(2).where(bull_gap)
    bull_high = low.where(bull_gap)
    bear_low = high.where(bear_gap)
    bear_high = low.shift(2).where(bear_gap)
    return bull_low, bull_high, bear_low, bear_high


def order_blocks(df: pd.DataFrame, atr_length: int = 14, move_mult: float = 1.5
                  ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bullish order block: a down candle immediately followed by a strong
    up move (next candle's body > move_mult * ATR) — the last supply
    candle before demand took over. Bearish is the mirror image. Returns
    zone bounds (the OB candle's own low/high), NaN where no OB forms:
    (bull_low, bull_high, bear_low, bear_high).

    Uses the NEXT candle's body to confirm, so a zone at index i is only
    knowable as of i+1 — callers must not treat it as available at i itself
    (see near_recent_zone's `include_current`, which defaults to excluding
    the current bar for exactly this reason)."""
    a = atr(df, atr_length)
    body_next = df["close"].shift(-1) - df["open"].shift(-1)
    is_down, is_up = df["close"] < df["open"], df["close"] > df["open"]
    bull_ob = is_down & (body_next > move_mult * a)
    bear_ob = is_up & (-body_next > move_mult * a)
    return (df["low"].where(bull_ob), df["high"].where(bull_ob),
            df["low"].where(bear_ob), df["high"].where(bear_ob))


def fibonacci_levels(df: pd.DataFrame, swing_lookback: int = 50) -> dict[str, pd.Series]:
    """Standard retracement levels (38.2%, 50%, 61.8%) between the trailing
    swing high and swing low (looking back swing_lookback bars, excluding
    the current one to avoid lookahead). Returns a dict of level-name ->
    Series of price levels."""
    swing_high = df["high"].shift(1).rolling(swing_lookback).max()
    swing_low = df["low"].shift(1).rolling(swing_lookback).min()
    diff = swing_high - swing_low
    return {f"fib_{int(pct * 1000)}": swing_high - diff * pct for pct in (0.382, 0.5, 0.618)}


def near_recent_zone(price: pd.Series, zone_low: pd.Series, zone_high: pd.Series,
                      lookback: int, tolerance_pct: float = 0.0,
                      include_current: bool = False) -> pd.Series:
    """For each bar t, True if price[t] falls inside ANY zone
    [zone_low*(1-tol), zone_high*(1+tol)] recorded in the last `lookback`
    bars. zone_low/zone_high are NaN except on bars where a zone formed.
    include_current=False (default) only looks at zones strictly before t,
    which is required for zones (like order blocks) that were only
    confirmed using data from bar t itself or later."""
    n = len(price)
    result = np.zeros(n, dtype=bool)
    price_arr, lo_arr, hi_arr = price.to_numpy(), zone_low.to_numpy(), zone_high.to_numpy()
    for t in range(n):
        start = max(0, t - lookback)
        end = t + 1 if include_current else t
        p = price_arr[t]
        for k in range(start, end):
            lo, hi = lo_arr[k], hi_arr[k]
            if np.isnan(lo) or np.isnan(hi):
                continue
            if lo * (1 - tolerance_pct) <= p <= hi * (1 + tolerance_pct):
                result[t] = True
                break
    return pd.Series(result, index=price.index)
