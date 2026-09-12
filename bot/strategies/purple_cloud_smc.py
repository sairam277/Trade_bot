"""Purple Cloud regime-flip strategy (see purple_cloud.py) with optional
"smart money concept" entry filters layered on top: a regime flip only
becomes a trade if the enabled filters also agree. Each filter is a toggle
so the refinement loop can test them one at a time and compound only the
ones that actually earn their place (see logs/refinement_log.jsonl).

These are practical approximations of order blocks / FVG / Fibonacci /
volume absorption, not canonical ICT definitions — see bot/indicators.py
for exactly what each one checks.
"""
from __future__ import annotations
import pandas as pd
from ..strategy_base import Signal, Action
from ..indicators import (
    volume_absorption, fair_value_gaps, order_blocks, fibonacci_levels, near_recent_zone,
)
from .purple_cloud import PurpleCloudStrategy


class PurpleCloudSMC(PurpleCloudStrategy):
    name = "purple_cloud_smc"

    def __init__(self, use_volume_absorption: bool = True, use_fvg: bool = False,
                 use_order_blocks: bool = False, use_fibonacci: bool = False,
                 zone_lookback: int = 20, zone_tolerance_pct: float = 0.005,
                 fib_swing_lookback: int = 50, **kw):
        super().__init__(use_volume_absorption=use_volume_absorption, use_fvg=use_fvg,
                          use_order_blocks=use_order_blocks, use_fibonacci=use_fibonacci,
                          zone_lookback=zone_lookback, zone_tolerance_pct=zone_tolerance_pct,
                          fib_swing_lookback=fib_swing_lookback, **kw)
        self.use_volume_absorption = use_volume_absorption
        self.use_fvg = use_fvg
        self.use_order_blocks = use_order_blocks
        self.use_fibonacci = use_fibonacci
        self.zone_lookback = zone_lookback
        self.zone_tolerance_pct = zone_tolerance_pct
        self.fib_swing_lookback = fib_swing_lookback

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().prepare(df)
        close = df["close"]

        long_ok = pd.Series(True, index=df.index)
        short_ok = pd.Series(True, index=df.index)

        if self.use_volume_absorption:
            bull_abs, bear_abs = volume_absorption(df)
            # require absorption within the last few bars, not necessarily this one
            long_ok &= bull_abs.rolling(3, min_periods=1).max().astype(bool)
            short_ok &= bear_abs.rolling(3, min_periods=1).max().astype(bool)

        if self.use_fvg:
            bull_lo, bull_hi, bear_lo, bear_hi = fair_value_gaps(df)
            long_ok &= near_recent_zone(close, bull_lo, bull_hi, self.zone_lookback,
                                         self.zone_tolerance_pct, include_current=True)
            short_ok &= near_recent_zone(close, bear_lo, bear_hi, self.zone_lookback,
                                          self.zone_tolerance_pct, include_current=True)

        if self.use_order_blocks:
            bull_lo, bull_hi, bear_lo, bear_hi = order_blocks(df)
            long_ok &= near_recent_zone(close, bull_lo, bull_hi, self.zone_lookback,
                                         self.zone_tolerance_pct, include_current=False)
            short_ok &= near_recent_zone(close, bear_lo, bear_hi, self.zone_lookback,
                                          self.zone_tolerance_pct, include_current=False)

        if self.use_fibonacci:
            levels = fibonacci_levels(df, self.fib_swing_lookback)
            near_any = pd.Series(False, index=df.index)
            for level in levels.values():
                tol = level.abs() * self.zone_tolerance_pct
                near_any |= (close - level).abs() <= tol
            long_ok &= near_any
            short_ok &= near_any

        df["long_entry"] = df["long_entry"] & long_ok
        df["short_entry"] = df["short_entry"] & short_ok
        return df

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal:
        return super().evaluate(df, symbol)
