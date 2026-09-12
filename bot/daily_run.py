"""One live/forward step of the paper-trading loop — meant to run once per
trading day, just before market open, via the scheduled cloud routine.

Pure computation only: no network calls to the Artifact app's database.
The caller (a Claude session holding the Artifact tool) reads the previous
day's engine state in as a JSON file, runs this script, then reads the
JSON result back out and is responsible for:
  - fetching real news sentiment (WebSearch) for any symbol in
    `opened_symbols` and patching `new_state.positions[symbol]
    .sentiment_label` / `.sentiment_score` before persisting new_state
    (this script has no web-sentiment access, by design — keeps it
    testable offline)
  - writing `closed_trades` into the app's `trades` collection
  - recomputing and writing the app's `meta/summary` (trade count, win
    rate, total P&L, equity, equity_curve)
  - persisting `new_state` back to the app's `meta/engine_state` doc

Usage:
    python -m bot.daily_run --state state.json --out results.json
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from .config import DEFAULT_CONFIG
from .data_feed import get_ohlcv
from .backtest import _load_strategy
from .paper_engine import PaperEngine, Position
from .strategy_base import Action

STRATEGY_NAME = "purple_cloud_smc"
STRATEGY_PARAMS = dict(period=25, use_volume_absorption=True)


def _load_state(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "cash": DEFAULT_CONFIG.risk.starting_capital,
            "day_start_equity": DEFAULT_CONFIG.risk.starting_capital,
            "positions": {},
        }


def _engine_from_state(state: dict) -> PaperEngine:
    engine = PaperEngine(config=DEFAULT_CONFIG)
    engine.cash = state.get("cash", DEFAULT_CONFIG.risk.starting_capital)
    engine.day_start_equity = state.get("day_start_equity", engine.cash)
    for sym, pos in state.get("positions", {}).items():
        engine.positions[sym] = Position(**pos)
    return engine


def _state_from_engine(engine: PaperEngine) -> dict:
    return {
        "cash": engine.cash,
        "day_start_equity": engine.day_start_equity,
        "positions": {sym: asdict(pos) for sym, pos in engine.positions.items()},
    }


def run(state_path: str, out_path: str) -> dict:
    state = _load_state(state_path)
    engine = _engine_from_state(state)
    watchlist = DEFAULT_CONFIG.watchlist
    now = datetime.now(timezone.utc)

    opened_symbols, messages, mark_prices = [], [], {}

    for sym in watchlist:
        strat = _load_strategy(STRATEGY_NAME, **STRATEGY_PARAMS)
        df = get_ohlcv(sym, period_days=200, use_cache=False)
        df = strat.prepare(df)
        price = float(df["close"].iloc[-1])
        mark_prices[sym] = price

        had_position = sym in engine.positions
        signal = strat.evaluate(df, sym)
        if signal.action != Action.HOLD:
            msg = engine.on_signal(signal, price, now)
            if msg:
                messages.append(msg)
        if not had_position and sym in engine.positions:
            opened_symbols.append(sym)  # needs real sentiment attached by the caller

    messages.extend(engine.check_stops(mark_prices, now))

    result = {
        "run_time": now.isoformat(),
        "opened_symbols": opened_symbols,
        "closed_trades": [asdict(t) for t in engine.closed_trades],
        "messages": messages,
        "mark_prices": mark_prices,
        "equity": engine.equity(mark_prices),
        "new_state": _state_from_engine(engine),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = run(args.state, args.out)
    print(f"run_time={r['run_time']} opened={r['opened_symbols']} "
          f"closed={len(r['closed_trades'])} equity={r['equity']:.2f}")
    for m in r["messages"]:
        print(" -", m)
