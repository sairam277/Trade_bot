"""Rule-based post-mortem notes for losing paper trades, shown in the
mobile app. Deliberately simple/transparent heuristics on the trade's own
exit reason rather than an opaque model call — every note traces back to a
concrete, checkable fact about the trade.
"""
from __future__ import annotations


def post_mortem_for(trade: dict) -> str:
    """trade: a dict with at least pnl, reason_exit, side, pnl_pct."""
    if trade["pnl"] >= 0:
        return ""

    reason = trade.get("reason_exit", "")
    pnl_pct = trade.get("pnl_pct", 0)

    if reason == "stop-loss hit":
        return (f"Stopped out at {pnl_pct:+.2f}% before the regime flip reversed on its own. "
                "The ATR-based stop did its job limiting the loss, but this entry didn't have "
                "room to work — worth checking if a wider stop or an extra confirmation bar "
                "before entry would have avoided it.")
    if reason.startswith("reversed by"):
        return (f"Position was reversed by an opposite regime-flip signal ({pnl_pct:+.2f}%) rather "
                "than hitting its own stop or target — the trend flipped again quickly after entry. "
                "A minimum holding period, or requiring the new regime to hold for 2+ bars before "
                "reversing, could filter out this kind of whipsaw.")
    if reason in ("backtest end", "seed data end"):
        return ("Position was still open when this data window ended — not a loss from the "
                "strategy's own exit logic, just an artifact of where the backtest stops.")
    return f"Closed at {pnl_pct:+.2f}% ({reason})."
