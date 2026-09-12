"""Lightweight market-sentiment read for a symbol before a trade is proposed.

v1 is intentionally simple and transparent (word-list scoring over recent
headlines) rather than a black box — every trade card in the app shows the
headlines that drove the score. This module has a clear seam
(`score_sentiment`) to swap in a news API or an LLM-based sentiment read
later without touching callers.
"""
from __future__ import annotations
from dataclasses import dataclass, field

_POSITIVE_WORDS = {
    "beats", "surge", "rally", "upgrade", "record", "profit", "growth",
    "outperform", "bullish", "buy", "strong", "expansion", "wins", "raises",
}
_NEGATIVE_WORDS = {
    "misses", "plunge", "downgrade", "loss", "probe", "fraud", "bearish",
    "sell", "weak", "cut", "lawsuit", "scam", "default", "slump", "fine",
}


@dataclass
class SentimentReading:
    symbol: str
    score: float                 # -1 (very negative) .. +1 (very positive)
    label: str                   # "positive" / "neutral" / "negative"
    headlines: list[str] = field(default_factory=list)


def _label(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def score_headlines(headlines: list[str]) -> float:
    if not headlines:
        return 0.0
    total = 0
    for h in headlines:
        words = h.lower().split()
        total += sum(1 for w in words if w.strip(".,!") in _POSITIVE_WORDS)
        total -= sum(1 for w in words if w.strip(".,!") in _NEGATIVE_WORDS)
    return max(-1.0, min(1.0, total / max(len(headlines), 1)))


def score_sentiment(symbol: str, headlines: list[str]) -> SentimentReading:
    """headlines: recent news headlines for `symbol`, newest first.

    Callers (the live bot loop) are expected to fetch headlines via a news
    source (e.g. a WebSearch call) and pass them in here — kept separate so
    this module stays testable without network access.
    """
    score = score_headlines(headlines)
    return SentimentReading(symbol=symbol, score=score, label=_label(score), headlines=headlines[:5])
