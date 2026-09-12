"""Central configuration for the trading bot.

Nothing here talks to a broker yet — paper trading only. When live trading
is added, broker API keys must come from environment variables, never from
this file.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class RiskConfig:
    """Position sizing and risk limits applied by the paper engine
    regardless of what a strategy asks for."""

    starting_capital: float = 10_000.0      # paper money, INR — matches the user's script
    max_position_pct: float = 0.20          # 20%: smallest size that lets most watchlist
                                             # stocks execute at all at 10k capital (see
                                             # logs/refinement_log.jsonl "qty_pct sweep" note).
                                             # NOT raised further on validation-window results
                                             # alone — see that log entry for why.
    max_open_positions: int = 5
    daily_loss_limit_pct: float = 0.03      # stop trading for the day at -3%
    per_trade_stop_loss_pct: float = 0.02   # hard stop if strategy has none


@dataclass
class BotConfig:
    watchlist: list[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    ])
    timeframe: str = "1d"          # yfinance interval: 1d, 1h, 15m, ...
    lookback_days: int = 730        # history window for backtests
    risk: RiskConfig = field(default_factory=RiskConfig)


DEFAULT_CONFIG = BotConfig()
