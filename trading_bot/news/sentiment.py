"""
News sentiment analyzer.
Fetches crypto news headlines, scores them, returns a bias score [-1 to +1].
Used as a filter/confluence layer, NOT a primary signal.
Laggy by nature — do not use as entry trigger alone.
"""

import re
from dataclasses import dataclass


BULLISH_WORDS = {
    "surge", "rally", "moon", "bull", "buy", "all-time high", "ath", "breakout",
    "adoption", "approved", "etf", "institutional", "partner", "launch", "upgrade",
    "pump", "recovery", "rebound", "positive", "gain", "profit", "rise", "rising",
    "green", "boom", "accumulate", "hodl", "milestone",
}

BEARISH_WORDS = {
    "crash", "dump", "bear", "sell", "hack", "ban", "regulation", "crackdown",
    "fear", "panic", "collapse", "fraud", "scam", "lawsuit", "sec", "fine",
    "drop", "plunge", "fall", "falling", "red", "loss", "liquidation", "bankrupt",
    "delist", "warning", "risk", "volatile", "uncertainty",
}


@dataclass
class SentimentResult:
    score: float          # -1 (very bearish) to +1 (very bullish)
    label: str            # "bullish" / "bearish" / "neutral"
    n_headlines: int
    top_headlines: list[str]


def score_headline(text: str) -> float:
    text_lower = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def analyze_sentiment(headlines: list[dict]) -> SentimentResult:
    """
    headlines: list of dicts with at least 'title' key.
    Returns aggregate sentiment score.
    """
    if not headlines:
        return SentimentResult(score=0.0, label="neutral", n_headlines=0, top_headlines=[])

    scores = [score_headline(h.get("title", "")) for h in headlines]
    avg_score = sum(scores) / len(scores)

    label = "neutral"
    if avg_score > 0.1:
        label = "bullish"
    elif avg_score < -0.1:
        label = "bearish"

    # Pick top 3 most polarized headlines
    scored = sorted(zip(scores, headlines), key=lambda x: abs(x[0]), reverse=True)
    top = [h.get("title", "") for _, h in scored[:3]]

    return SentimentResult(
        score=round(avg_score, 3),
        label=label,
        n_headlines=len(headlines),
        top_headlines=top,
    )


def sentiment_allows_trade(sentiment: SentimentResult, direction: str) -> bool:
    """
    Returns True if news sentiment does NOT contradict the trade direction.
    Neutral = always allow. Strong opposition = block.
    """
    if abs(sentiment.score) < 0.15:
        return True  # neutral — don't filter
    if direction == "long" and sentiment.label == "bearish" and sentiment.score < -0.3:
        return False
    if direction == "short" and sentiment.label == "bullish" and sentiment.score > 0.3:
        return False
    return True
