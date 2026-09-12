"""NSE OHLCV data access.

Uses Yahoo Finance (via yfinance) as a free, no-API-key data source for NSE
stocks (ticker suffix ".NS"). This is a data source, not a broker — it has
no order-placement capability, which matches paper-trading-only scope.

Data is cached to disk (data/) so repeated backtests don't re-download.
"""
from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path
from .config import DATA_DIR

_CACHE_TTL_HOURS = 6


def _cache_path(symbol: str, interval: str) -> Path:
    safe = symbol.replace("/", "_")
    return DATA_DIR / f"{safe}_{interval}.pkl"


def get_ohlcv(symbol: str, period_days: int = 730, interval: str = "1d",
               use_cache: bool = True) -> pd.DataFrame:
    """Fetch OHLCV history for an NSE symbol, e.g. 'RELIANCE.NS'.

    Returns a DataFrame indexed by datetime with columns:
    open, high, low, close, volume
    """
    cache = _cache_path(symbol, interval)
    if use_cache and cache.exists():
        age_hours = (pd.Timestamp.now() - pd.Timestamp(cache.stat().st_mtime, unit="s")).total_seconds() / 3600
        if age_hours < _CACHE_TTL_HOURS:
            return pd.read_pickle(cache)

    period = f"{period_days}d" if period_days <= 730 else "max"
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} — check the ticker "
                          f"(NSE symbols need the '.NS' suffix, e.g. 'RELIANCE.NS').")

    df = df.rename(columns=str.lower)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]]
    df.to_pickle(cache)
    return df


def get_latest_price(symbol: str) -> float:
    """Real-time-ish last traded price (15-20 min delayed via Yahoo Finance)."""
    t = yf.Ticker(symbol)
    fast = t.fast_info
    return float(fast["last_price"])
