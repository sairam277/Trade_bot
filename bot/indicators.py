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
