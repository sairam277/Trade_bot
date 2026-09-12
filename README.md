# Indian Stock Market Trading Bot

Automated (paper-trading first) trading bot for NSE stocks, with a mobile
companion web app for trade approval and performance review.

## Status

- Execution mode: **paper trading only** (no live orders, no broker API keys yet)
- Data source: Yahoo Finance NSE tickers (`SYMBOL.NS`) via `yfinance` — free, no
  API key required. Live trading later will need a broker API (Zerodha Kite
  Connect / Upstox / Angel One).
- Strategy: pending — waiting on the user's Pine Script to port into
  `bot/strategies/`.

## Layout

```
bot/
  config.py          settings (watchlist, risk limits, paths)
  data_feed.py        NSE OHLCV data via yfinance
  sentiment.py         lightweight news/sentiment scoring for a symbol
  strategy_base.py     Strategy interface every strategy implements
  strategies/          plug-in strategies (one file each)
  paper_engine.py       simulated order execution + position/P&L tracking
  backtest.py           historical backtest runner
  refine.py             autonomous strategy refinement loop
app/                   mobile web app (published as a Claude Artifact)
data/                  cached OHLCV data (gitignored)
logs/                  trade logs, equity curve, refinement change-log (gitignored)
```

## Running a backtest

```
pip install -r requirements.txt
python -m bot.backtest --symbol RELIANCE.NS --strategy example_ma_cross
```

## Safety rules baked into this project

- No live order-placement code exists yet — paper engine only.
- Any change to a strategy that is running against real money requires
  explicit user approval before deployment (see `bot/refine.py` docstring).
