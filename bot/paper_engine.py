"""Simulated order execution, position tracking, and trade logging.

This is the ONLY thing allowed to "execute" trades right now — there is no
live broker order path in this codebase yet. Every fill here is simulated
against the historical/last-known price from data_feed.

Every closed trade is appended to logs/trades.jsonl in a shape the mobile
app can render directly: symbol, strategy, entry/exit, P&L, sentiment at
entry, and (for losses) a `post_mortem` field filled in by the refinement
pass in refine.py.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from .config import DEFAULT_CONFIG, LOGS_DIR
from .strategy_base import Action, Signal

TRADES_LOG = LOGS_DIR / "trades.jsonl"


@dataclass
class Position:
    symbol: str
    strategy: str
    qty: int
    entry_price: float
    entry_time: str
    stop_loss: float | None
    take_profit: float | None
    sentiment_score: float
    sentiment_label: str
    reason: str


@dataclass
class ClosedTrade:
    symbol: str
    strategy: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    reason_entry: str
    reason_exit: str
    sentiment_label: str
    post_mortem: str = ""  # filled in later for losing trades by refine.py


class PaperEngine:
    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.cash = config.risk.starting_capital
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[ClosedTrade] = []
        self.day_start_equity = self.cash

    # -- sizing -------------------------------------------------------
    def _position_size(self, price: float) -> int:
        budget = self.cash * self.config.risk.max_position_pct
        return max(int(budget // price), 0)

    def equity(self, mark_prices: dict[str, float]) -> float:
        val = self.cash
        for sym, pos in self.positions.items():
            val += pos.qty * mark_prices.get(sym, pos.entry_price)
        return val

    def daily_loss_breached(self, mark_prices: dict[str, float]) -> bool:
        eq = self.equity(mark_prices)
        return (self.day_start_equity - eq) / self.day_start_equity >= self.config.risk.daily_loss_limit_pct

    # -- order handling -------------------------------------------------
    def on_signal(self, signal: Signal, price: float, timestamp: datetime,
                  sentiment_score: float = 0.0, sentiment_label: str = "neutral") -> str | None:
        """Apply a strategy signal against the current price. Returns a
        short description of what happened, or None if nothing changed."""
        ts = timestamp.isoformat()

        if signal.action == Action.BUY:
            if signal.symbol in self.positions:
                return None  # already in a position
            if len(self.positions) >= self.config.risk.max_open_positions:
                return f"skipped BUY {signal.symbol}: max open positions reached"
            qty = self._position_size(price)
            if qty <= 0:
                return f"skipped BUY {signal.symbol}: position size rounds to 0"
            stop = signal.stop_loss or price * (1 - self.config.risk.per_trade_stop_loss_pct)
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol, strategy=signal.reason, qty=qty,
                entry_price=price, entry_time=ts, stop_loss=stop,
                take_profit=signal.take_profit, sentiment_score=sentiment_score,
                sentiment_label=sentiment_label, reason=signal.reason,
            )
            self.cash -= qty * price
            return f"BUY {qty} {signal.symbol} @ {price:.2f} ({signal.reason})"

        if signal.action in (Action.SELL, Action.EXIT):
            pos = self.positions.pop(signal.symbol, None)
            if not pos:
                return None
            pnl = (price - pos.entry_price) * pos.qty
            pnl_pct = (price / pos.entry_price - 1) * 100
            self.cash += pos.qty * price
            self.closed_trades.append(ClosedTrade(
                symbol=pos.symbol, strategy=pos.strategy, qty=pos.qty,
                entry_price=pos.entry_price, exit_price=price,
                entry_time=pos.entry_time, exit_time=ts, pnl=pnl, pnl_pct=pnl_pct,
                reason_entry=pos.reason, reason_exit=signal.reason,
                sentiment_label=pos.sentiment_label,
            ))
            self._log_trade(self.closed_trades[-1])
            return f"EXIT {pos.qty} {pos.symbol} @ {price:.2f} P&L {pnl:+.2f} ({pnl_pct:+.2f}%)"

        return None

    def check_stops(self, mark_prices: dict[str, float], timestamp: datetime) -> list[str]:
        """Force-exit any position that hit its stop-loss/take-profit."""
        msgs = []
        for sym in list(self.positions):
            pos = self.positions[sym]
            price = mark_prices.get(sym)
            if price is None:
                continue
            if pos.stop_loss and price <= pos.stop_loss:
                msgs.append(self.on_signal(Signal(Action.EXIT, sym, reason="stop-loss hit"), price, timestamp))
            elif pos.take_profit and price >= pos.take_profit:
                msgs.append(self.on_signal(Signal(Action.EXIT, sym, reason="take-profit hit"), price, timestamp))
        return [m for m in msgs if m]

    def _log_trade(self, trade: ClosedTrade) -> None:
        with open(TRADES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trade)) + "\n")
