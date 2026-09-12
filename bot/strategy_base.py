"""Interface every strategy plugs into.

A Strategy turns a price history into a Signal. It never places orders and
never sizes positions — that's the paper/live engine's job (keeps risk
management centralized and consistent no matter which strategy is active).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class Action(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"  # close an open position regardless of direction


@dataclass
class Signal:
    action: Action
    symbol: str
    confidence: float = 1.0          # 0-1, used for position sizing / display
    stop_loss: float | None = None
    take_profit: float | None = None
    reason: str = ""                  # human-readable rule that fired, shown in the app


class Strategy:
    """Subclass this for each strategy. One instance per symbol."""

    name: str = "unnamed_strategy"

    def __init__(self, **params):
        self.params = params

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to df. Called once before evaluate()."""
        return df

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Look at the most recent row(s) of df (already has indicators from
        prepare()) and return a Signal for 'now' (the last row)."""
        raise NotImplementedError
