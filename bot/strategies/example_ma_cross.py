"""Placeholder strategy: fast/slow moving-average crossover.

This exists only so the framework has something runnable end-to-end before
the user's Pine Script strategy is ported in. Replace/supplement with the
real strategy in this same directory once the script is shared.
"""
from __future__ import annotations
import pandas as pd
from ..strategy_base import Strategy, Signal, Action


class MACrossStrategy(Strategy):
    name = "ma_cross"

    def __init__(self, fast: int = 20, slow: int = 50, **kw):
        super().__init__(fast=fast, slow=slow, **kw)
        self.fast = fast
        self.slow = slow

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma_fast"] = df["close"].rolling(self.fast).mean()
        df["ma_slow"] = df["close"].rolling(self.slow).mean()
        return df

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.slow + 1:
            return Signal(Action.HOLD, symbol, reason="not enough history")

        prev, now = df.iloc[-2], df.iloc[-1]
        crossed_up = prev["ma_fast"] <= prev["ma_slow"] and now["ma_fast"] > now["ma_slow"]
        crossed_down = prev["ma_fast"] >= prev["ma_slow"] and now["ma_fast"] < now["ma_slow"]

        if crossed_up:
            return Signal(
                Action.BUY, symbol, confidence=0.6,
                stop_loss=now["close"] * 0.98,
                reason=f"{self.fast}-MA crossed above {self.slow}-MA",
            )
        if crossed_down:
            return Signal(Action.EXIT, symbol, confidence=0.6,
                           reason=f"{self.fast}-MA crossed below {self.slow}-MA")
        return Signal(Action.HOLD, symbol, reason="no crossover")
