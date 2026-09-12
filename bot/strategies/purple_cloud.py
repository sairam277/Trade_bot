"""Port of the user's Pine Script strategy ("My strategy" / Purple Cloud
regime filter), originally written for TradingView Pine Script v6.

Core logic (unchanged from the original):
  - `a4` is a double volume-weighted, lag-reduced average of hl2 (two VWMAs
    at period/4 and period/2, extrapolated as 2*fast - slow, then smoothed
    again with a period-length VWMA).
  - `xl`/`xh` = close -/+ ATR(period)*alpha — a volatility band around price.
  - `b1` = Wilder-smoothed moving average of close (ta.rma equivalent).
  - buy  = a4 <= xl and close > b1
  - sell = a4 >= xh and close < b1
  - A position is opened only on the bar where the regime FLIPS (buy/sell
    just became true and wasn't the prevailing regime), matching the
    original script's `xs != xs[1]` guard — not on every bar the condition
    holds. Opposite-direction entries reverse the position, matching
    TradingView's default strategy.entry() behaviour with pyramiding=0.
  - Fixed stop-loss/take-profit distance in price points (not %), matching
    the original script's pip-distance inputs.

Ported-but-flagged, not silently changed:
  - The original script also computes a Supertrend (`ta.supertrend`) but
    never uses it in the buy/sell logic — it's dead code in the source
    (no plot, no condition references it), so it's omitted here entirely.
  - Stop-loss/take-profit are FIXED PRICE POINTS (default 25/50 rupees,
    derived from the script's 50/100 "pips" at a 0.05 tick size), not a
    percentage or ATR multiple. The same absolute stop is a huge % move on
    a low-priced stock and tiny on a high-priced one — a real scale issue
    across a multi-stock watchlist, and the top candidate for the
    refinement loop to convert to ATR-based or percentage-based sizing.
  - The strategy takes both long AND short entries. Shorting a stock
    outright (CNC/delivery) isn't available to Indian retail without
    F&O/intraday MIS margin — flagged here, not removed; which segment
    this runs in is your call.
"""
from __future__ import annotations
import math
import pandas as pd
from ..strategy_base import Strategy, Signal, Action
from ..indicators import atr, rma, vwma


class PurpleCloudStrategy(Strategy):
    name = "purple_cloud"

    def __init__(self, period: int = 40, alpha: float = 0.9,
                 stop_loss_pips: float = 50, take_profit_pips: float = 100,
                 tick_size: float = 0.05, stop_atr_mult: float | None = 1.0,
                 tp_atr_mult: float | None = 2.0, stop_atr_length: int = 14, **kw):
        super().__init__(period=period, alpha=alpha, stop_loss_pips=stop_loss_pips,
                          take_profit_pips=take_profit_pips, tick_size=tick_size,
                          stop_atr_mult=stop_atr_mult, tp_atr_mult=tp_atr_mult,
                          stop_atr_length=stop_atr_length, **kw)
        self.x1 = period
        self.alpha = alpha
        pip_size = tick_size * 10  # GetPipSize() for equities in the original script
        self.stop_points = stop_loss_pips * pip_size
        self.tp_points = take_profit_pips * pip_size
        # Refinement (accepted, see logs/refinement_log.jsonl): scale stop/target
        # by ATR(14) instead of a fixed price distance, so risk adapts to each
        # stock's own volatility. Default is now ATR-based (1.0x stop / 2.0x
        # target); pass stop_atr_mult=None to fall back to the original
        # script's fixed-point (25/50 INR) behaviour.
        self.stop_atr_mult = stop_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.stop_atr_length = stop_atr_length

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
        hl2 = (high + low) / 2

        x2 = atr(df, self.x1) * self.alpha
        df["xh"] = close + x2
        df["xl"] = close - x2
        if self.stop_atr_mult is not None or self.tp_atr_mult is not None:
            df["stop_atr"] = atr(df, self.stop_atr_length)

        len1, len2 = math.ceil(self.x1 / 4), math.ceil(self.x1 / 2)
        a1 = vwma(hl2 * volume, volume, len1) / vwma(volume, volume, len1)
        a2 = vwma(hl2 * volume, volume, len2) / vwma(volume, volume, len2)
        a3 = 2 * a1 - a2
        a4 = vwma(a3, volume, self.x1)
        b1 = rma(close, self.x1)

        buy = (a4 <= df["xl"]) & (close > b1)
        sell = (a4 >= df["xh"]) & (close < b1)

        xs_raw = pd.Series(index=df.index, dtype=float)
        xs_raw[buy], xs_raw[sell] = 1.0, -1.0
        xs = xs_raw.ffill().fillna(0.0)
        xs_prev = xs.shift(1).fillna(0.0)

        df["a4"], df["b1"] = a4, b1
        df["long_entry"] = buy & (xs != xs_prev)
        df["short_entry"] = sell & (xs != xs_prev)
        return df

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal:
        row = df.iloc[-1]
        if pd.isna(row.get("a4")) or pd.isna(row.get("b1")):
            return Signal(Action.HOLD, symbol, reason="warming up (not enough history)")

        price = float(row["close"])
        atr_val = row.get("stop_atr")
        stop_dist = (atr_val * self.stop_atr_mult if self.stop_atr_mult is not None and pd.notna(atr_val)
                     else self.stop_points)
        tp_dist = (atr_val * self.tp_atr_mult if self.tp_atr_mult is not None and pd.notna(atr_val)
                   else self.tp_points)

        if bool(row["long_entry"]):
            return Signal(Action.BUY, symbol, confidence=0.7,
                           stop_loss=price - stop_dist, take_profit=price + tp_dist,
                           reason="Purple Cloud regime flip to bullish (a4<=xl & close>b1)")
        if bool(row["short_entry"]):
            return Signal(Action.SELL, symbol, confidence=0.7,
                           stop_loss=price + stop_dist, take_profit=price - tp_dist,
                           reason="Purple Cloud regime flip to bearish (a4>=xh & close<b1)")
        return Signal(Action.HOLD, symbol, reason="no regime flip")
