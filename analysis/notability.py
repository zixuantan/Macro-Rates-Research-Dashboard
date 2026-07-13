from __future__ import annotations

from dataclasses import dataclass

from models.macro_analysis import MacroSignal


MAX_Z_SCORE = 3.0

DEFAULT_IMPORTANCE_WEIGHTS = {
    "2Y Treasury move": 1.3,
    "2s10s change": 1.2,
    "5Y breakeven": 1.3,
    "5Y5Y forward": 1.3,
    "wage growth": 1.3,
    "unemployment": 1.2,
    "payrolls": 1.1,
    "growth breadth": 1.1,
    "HY OAS": 1.2,
    "VIX": 1.0,
    "Nelson-Siegel residuals": 0.5,
}


@dataclass(frozen=True)
class NotabilityBreakdown:
    magnitude_score: float
    extremity_score: float
    importance_score: float
    breadth_bonus: float
    freshness_score: float
    total_score: float


def score_signal(
    signal: MacroSignal,
    breadth_bonus: float = 0.0,
    freshness_score: float = 0.0,
) -> NotabilityBreakdown:
    """Compute a transparent rule-based notability score."""
    magnitude = 0.0
    if signal.standardized_change is not None:
        magnitude = min(abs(signal.standardized_change), MAX_Z_SCORE) / MAX_Z_SCORE
    elif signal.change is not None:
        magnitude = min(abs(signal.change) / 100.0, 1.0)

    extremity = 0.0
    if signal.percentile is not None:
        extremity = min(abs(signal.percentile - 50.0) / 50.0, 1.0)

    importance = signal.importance_weight

    total = magnitude + extremity + importance + breadth_bonus + freshness_score
    return NotabilityBreakdown(
        magnitude_score=magnitude,
        extremity_score=extremity,
        importance_score=importance,
        breadth_bonus=breadth_bonus,
        freshness_score=freshness_score,
        total_score=total,
    )
