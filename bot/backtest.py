"""Historical backtest runner.

Walks a symbol's price history bar-by-bar through a Strategy + PaperEngine,
holding out the most recent `validation_days` as an out-of-sample check so
refinement (see refine.py) can't just curve-fit the whole history.

Usage:
    python -m bot.backtest --symbol RELIANCE.NS --strategy ma_cross
"""
from __future__ import annotations
import argparse
import importlib
import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from .config import DEFAULT_CONFIG
from .data_feed import get_ohlcv
from .paper_engine import PaperEngine
from .strategy_base import Action, Signal


@dataclass
class BacktestResult:
    trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float

    def __str__(self) -> str:
        return (f"trades={self.trades} win_rate={self.win_rate:.1%} "
                f"total_pnl={self.total_pnl:+.2f} return={self.total_return_pct:+.2f}% "
                f"max_dd={self.max_drawdown_pct:.2f}% sharpe={self.sharpe:.2f}")


def _load_strategy(name: str, **params):
    # e.g. "ma_cross" -> bot.strategies.example_ma_cross.MACrossStrategy
    mod_map = {
        "ma_cross": ("example_ma_cross", "MACrossStrategy"),
        "purple_cloud": ("purple_cloud", "PurpleCloudStrategy"),
        "purple_cloud_smc": ("purple_cloud_smc", "PurpleCloudSMC"),
    }
    if name not in mod_map:
        raise ValueError(f"Unknown strategy '{name}'. Known: {list(mod_map)}")
    mod_name, cls_name = mod_map[name]
    mod = importlib.import_module(f".strategies.{mod_name}", package="bot")
    return getattr(mod, cls_name)(**params)


def run_backtest(symbol: str, strategy_name: str, lookback_days: int = 730,
                  validation_days: int = 120, capital: float | None = None,
                  qty_pct: float | None = None, **strategy_params) -> tuple[BacktestResult, BacktestResult]:
    """Returns (in_sample_result, validation_result).

    `capital`/`qty_pct` override the default risk config (useful to match a
    strategy's own assumptions, e.g. the ported Pine script's ₹10,000
    starting capital and 2%-of-equity position sizing).
    """
    df = get_ohlcv(symbol, period_days=lookback_days)
    split = len(df) - validation_days
    train_df, val_df = df.iloc[:split], df.iloc[split - 60:]  # extra lookback for indicators

    strat = _load_strategy(strategy_name, **strategy_params)
    risk = DEFAULT_CONFIG.risk
    if capital is not None or qty_pct is not None:
        risk = replace(risk, starting_capital=capital or risk.starting_capital,
                        max_position_pct=qty_pct if qty_pct is not None else risk.max_position_pct)
    config = replace(DEFAULT_CONFIG, risk=risk)

    def _run(segment: pd.DataFrame) -> BacktestResult:
        segment = strat.prepare(segment)
        engine = PaperEngine(config=config)
        equity_curve = []
        for i in range(1, len(segment)):
            window = segment.iloc[: i + 1]
            row = segment.iloc[i]
            ts = segment.index[i]
            price = float(row["close"])
            signal = strat.evaluate(window, symbol)
            if signal.action != Action.HOLD:
                engine.on_signal(signal, price, ts)
            engine.check_stops({symbol: price}, ts)
            equity_curve.append(engine.equity({symbol: price}))

        # close anything still open at the end so P&L is realized
        if symbol in engine.positions:
            last_price = float(segment.iloc[-1]["close"])
            engine.on_signal(Signal(Action.EXIT, symbol, reason="backtest end"), last_price, segment.index[-1])

        eq = np.array(equity_curve) if equity_curve else np.array([config.risk.starting_capital])
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
        running_max = np.maximum.accumulate(eq)
        drawdown = (eq - running_max) / running_max
        wins = [t for t in engine.closed_trades if t.pnl > 0]
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)) if len(returns) > 1 else 0.0

        return BacktestResult(
            trades=len(engine.closed_trades),
            win_rate=len(wins) / len(engine.closed_trades) if engine.closed_trades else 0.0,
            total_pnl=sum(t.pnl for t in engine.closed_trades),
            total_return_pct=(eq[-1] / config.risk.starting_capital - 1) * 100 if len(eq) else 0.0,
            max_drawdown_pct=float(drawdown.min() * 100) if len(drawdown) else 0.0,
            sharpe=sharpe,
        )

    return _run(train_df), _run(val_df)


def run_portfolio_backtest(symbols: list[str], strategy_name: str, lookback_days: int = 730,
                            validation_days: int = 120, capital: float = 10_000.0,
                            qty_pct: float = 0.10, **strategy_params) -> tuple[BacktestResult, BacktestResult]:
    """Like run_backtest, but ONE PaperEngine with ONE shared cash pool trades
    all `symbols` on the same calendar — the realistic case (a single paper/
    live account trading a watchlist), where signals compete for the same
    capital instead of each symbol getting its own fresh account.

    Simplification: on days with multiple signals, symbols are evaluated in
    list order, so earlier symbols get first claim on available cash — a
    minor systematic bias worth knowing about, not eliminated here.
    """
    strat_by_symbol = {sym: _load_strategy(strategy_name, **strategy_params) for sym in symbols}
    prepared: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        prepared[sym] = strat_by_symbol[sym].prepare(get_ohlcv(sym, period_days=lookback_days))

    common_idx = None
    for df in prepared.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    common_idx = common_idx.sort_values()
    split = len(common_idx) - validation_days
    train_idx, val_idx = common_idx[:split], common_idx[split:]

    risk = replace(DEFAULT_CONFIG.risk, starting_capital=capital, max_position_pct=qty_pct)
    config = replace(DEFAULT_CONFIG, risk=risk)

    def _run(idx_slice: pd.Index) -> BacktestResult:
        engine = PaperEngine(config=config)
        equity_curve = []
        for ts in idx_slice:
            mark_prices = {}
            for sym in symbols:
                if ts not in prepared[sym].index:
                    continue
                row = prepared[sym].loc[[ts]]
                price = float(row["close"].iloc[-1])
                mark_prices[sym] = price
                signal = strat_by_symbol[sym].evaluate(row, sym)
                if signal.action != Action.HOLD:
                    engine.on_signal(signal, price, ts)
            engine.check_stops(mark_prices, ts)
            equity_curve.append(engine.equity(mark_prices))

        for sym in list(engine.positions):
            last_price = mark_prices.get(sym)
            if last_price is not None:
                engine.on_signal(Signal(Action.EXIT, sym, reason="backtest end"), last_price, idx_slice[-1])

        eq = np.array(equity_curve) if equity_curve else np.array([config.risk.starting_capital])
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
        running_max = np.maximum.accumulate(eq)
        drawdown = (eq - running_max) / running_max
        wins = [t for t in engine.closed_trades if t.pnl > 0]
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)) if len(returns) > 1 else 0.0

        return BacktestResult(
            trades=len(engine.closed_trades),
            win_rate=len(wins) / len(engine.closed_trades) if engine.closed_trades else 0.0,
            total_pnl=sum(t.pnl for t in engine.closed_trades),
            total_return_pct=(eq[-1] / config.risk.starting_capital - 1) * 100 if len(eq) else 0.0,
            max_drawdown_pct=float(drawdown.min() * 100) if len(drawdown) else 0.0,
            sharpe=sharpe,
        )

    return _run(train_idx), _run(val_idx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--strategy", default="ma_cross")
    ap.add_argument("--lookback-days", type=int, default=730)
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--qty-pct", type=float, default=None)
    args = ap.parse_args()

    in_sample, validation = run_backtest(args.symbol, args.strategy, args.lookback_days,
                                          capital=args.capital, qty_pct=args.qty_pct)
    print(f"[{args.symbol}] in-sample:  {in_sample}")
    print(f"[{args.symbol}] validation: {validation}")
