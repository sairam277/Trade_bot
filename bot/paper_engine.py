"""Simulated order execution, position tracking, and trade logging.

This is the ONLY thing allowed to "execute" trades right now — there is no
live broker order path in this codebase yet. Every fill here is simulated
against the historical/last-known price from data_feed.

Supports both long and short positions (needed for strategies, like the
ported Purple Cloud strategy, that trade both directions). An opposite-side
signal reverses an open position — closes it, then opens the new side —
matching TradingView's default strategy.entry() behaviour with pyramiding=0.
Shorts are modelled notionally (no cash debited at entry, P&L applied at
close) rather than as a full margin simulation.

Every closed trade is appended to logs/trades.jsonl in a shape the mobile
app can render directly: symbol, side, strategy, entry/exit, P&L,
sentiment at entry, and (for losses) a `post_mortem` field filled in by the
refinement pass in refine.py.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from .config import DEFAULT_CONFIG, LOGS_DIR
from .strategy_base import Action, Signal

TRADES_LOG = LOGS_DIR / "trades.jsonl"


@dataclass
class Position:
    symbol: str
    side: str  # "long" or "short"
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
    side: str
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
            price = mark_prices.get(sym, pos.entry_price)
            if pos.side == "long":
                val += pos.qty * price  # cash was already debited entry cost at open
            else:
                val += (pos.entry_price - price) * pos.qty  # unrealized short P&L
        return val

    def daily_loss_breached(self, mark_prices: dict[str, float]) -> bool:
        eq = self.equity(mark_prices)
        return (self.day_start_equity - eq) / self.day_start_equity >= self.config.risk.daily_loss_limit_pct

    # -- order handling -------------------------------------------------
    def _open(self, signal: Signal, price: float, ts: str, side: str,
               sentiment_score: float, sentiment_label: str) -> str | None:
        existing = self.positions.get(signal.symbol)
        if existing:
            if existing.side == side:
                return None  # already in a position on this side, no pyramiding
            self._close(signal.symbol, price, ts, reason=f"reversed by {side} signal")

        if len(self.positions) >= self.config.risk.max_open_positions:
            return f"skipped {side.upper()} {signal.symbol}: max open positions reached"
        qty = self._position_size(price)
        if qty <= 0:
            return f"skipped {side.upper()} {signal.symbol}: position size rounds to 0"

        default_stop = (price * (1 - self.config.risk.per_trade_stop_loss_pct) if side == "long"
                         else price * (1 + self.config.risk.per_trade_stop_loss_pct))
        stop = signal.stop_loss if signal.stop_loss is not None else default_stop

        self.positions[signal.symbol] = Position(
            symbol=signal.symbol, side=side, strategy=signal.reason, qty=qty,
            entry_price=price, entry_time=ts, stop_loss=stop,
            take_profit=signal.take_profit, sentiment_score=sentiment_score,
            sentiment_label=sentiment_label, reason=signal.reason,
        )
        if side == "long":
            self.cash -= qty * price
        return f"{side.upper()} {qty} {signal.symbol} @ {price:.2f} ({signal.reason})"

    def _close(self, symbol: str, price: float, ts: str, reason: str) -> str | None:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return None
        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.qty
            self.cash += pos.qty * price
        else:
            pnl = (pos.entry_price - price) * pos.qty
            self.cash += pnl  # notional short P&L settlement

        pnl_pct = (pnl / (pos.entry_price * pos.qty)) * 100 if pos.entry_price else 0.0
        self.closed_trades.append(ClosedTrade(
            symbol=pos.symbol, side=pos.side, strategy=pos.strategy, qty=pos.qty,
            entry_price=pos.entry_price, exit_price=price,
            entry_time=pos.entry_time, exit_time=ts, pnl=pnl, pnl_pct=pnl_pct,
            reason_entry=pos.reason, reason_exit=reason,
            sentiment_label=pos.sentiment_label,
        ))
        self._log_trade(self.closed_trades[-1])
        return f"EXIT {pos.side} {pos.qty} {pos.symbol} @ {price:.2f} P&L {pnl:+.2f} ({pnl_pct:+.2f}%)"

    def on_signal(self, signal: Signal, price: float, timestamp: datetime,
                  sentiment_score: float = 0.0, sentiment_label: str = "neutral") -> str | None:
        """Apply a strategy signal against the current price. Returns a
        short description of what happened, or None if nothing changed."""
        ts = timestamp.isoformat()
        if signal.action == Action.BUY:
            return self._open(signal, price, ts, "long", sentiment_score, sentiment_label)
        if signal.action == Action.SELL:
            return self._open(signal, price, ts, "short", sentiment_score, sentiment_label)
        if signal.action == Action.EXIT:
            return self._close(signal.symbol, price, ts, reason=signal.reason)
        return None

    def check_stops(self, mark_prices: dict[str, float], timestamp: datetime) -> list[str]:
        """Force-exit any position that hit its stop-loss/take-profit."""
        msgs = []
        for sym in list(self.positions):
            pos = self.positions[sym]
            price = mark_prices.get(sym)
            if price is None:
                continue
            hit_stop = ((pos.side == "long" and pos.stop_loss is not None and price <= pos.stop_loss) or
                        (pos.side == "short" and pos.stop_loss is not None and price >= pos.stop_loss))
            hit_tp = ((pos.side == "long" and pos.take_profit is not None and price >= pos.take_profit) or
                      (pos.side == "short" and pos.take_profit is not None and price <= pos.take_profit))
            if hit_stop:
                msgs.append(self._close(sym, price, timestamp.isoformat(), "stop-loss hit"))
            elif hit_tp:
                msgs.append(self._close(sym, price, timestamp.isoformat(), "take-profit hit"))
        return [m for m in msgs if m]

    def _log_trade(self, trade: ClosedTrade) -> None:
        with open(TRADES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(trade)) + "\n")
