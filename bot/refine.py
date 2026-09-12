"""Autonomous strategy refinement loop.

Scope / safety rules (agreed with the user):
  - Runs autonomously against BACKTESTS and PAPER TRADING ONLY. It may
    rewrite files under bot/strategies/ on its own and re-run the backtest
    to compare, with no human approval needed at that stage.
  - Every change it makes gets appended to logs/refinement_log.jsonl AND
    surfaced in the mobile app's change-log feed, in plain language: what
    changed, why (which metric it targeted), and before/after numbers.
  - It must validate any change against a held-out window (see
    backtest.run_backtest's validation_days) before keeping it — a change
    that improves in-sample numbers but hurts validation numbers is
    rejected and logged as rejected, not silently dropped.
  - It NEVER modifies whatever strategy version is flagged as "live" in
    config (i.e. wired to real broker order placement). Promoting a
    refined strategy to live status is a separate, explicit user action —
    this loop only proposes and marks candidates, it does not flip that
    switch.

This module is currently a scaffold: the concrete parameter-search /
rewrite logic will be filled in once a real strategy (ported from the
user's Pine Script) exists to refine. Wiring it up now would just be
tuning the placeholder MA-crossover example, which isn't useful.
"""
from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from .config import LOGS_DIR

REFINEMENT_LOG = LOGS_DIR / "refinement_log.jsonl"


def _json_default(obj):
    """Backtest results carry numpy scalars (np.float64, np.bool, ...) in
    their before/after dicts; make those JSON-serializable instead of
    requiring every caller to cast them first."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class RefinementEntry:
    timestamp: str
    strategy: str
    change: str            # human-readable description of what changed
    target_metric: str      # e.g. "reduce max drawdown", "improve win rate"
    before: dict
    after: dict
    accepted: bool
    note: str = ""


def log_refinement(entry: RefinementEntry) -> None:
    with open(REFINEMENT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), default=_json_default) + "\n")


def new_entry(strategy: str, change: str, target_metric: str, before: dict,
              after: dict, accepted: bool, note: str = "") -> RefinementEntry:
    return RefinementEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        strategy=strategy, change=change, target_metric=target_metric,
        before=before, after=after, accepted=accepted, note=note,
    )


# TODO once the real strategy is in place:
#   def refine_once(strategy_name, symbol) -> RefinementEntry | None
#   - run_backtest() for the current version -> `before`
#   - try a small, explainable parameter/logic tweak (not a black-box search)
#   - run_backtest() for the tweaked version -> `after`
#   - accept only if validation-window metrics improve without in-sample-only gains
#   - log_refinement(...) either way (accepted=True/False) so nothing is silent
