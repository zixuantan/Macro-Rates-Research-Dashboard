from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


SignalDirection = Literal[
    "higher",
    "lower",
    "improving",
    "weakening",
    "stable",
    "mixed",
    "unavailable",
]

SignalCategory = Literal[
    "rates",
    "curve",
    "inflation",
    "growth",
    "labor",
    "credit",
    "equities",
    "volatility",
    "dollar",
    "cross_asset",
]


@dataclass(frozen=True)
class MacroSignal:
    signal_id: str
    panel_id: str
    category: SignalCategory
    label: str
    as_of: date
    value: float | None
    unit: str
    change: float | None = None
    change_unit: str | None = None
    horizon: str | None = None
    percentile: float | None = None
    standardized_change: float | None = None
    direction: SignalDirection = "unavailable"
    importance_weight: float = 1.0
    interpretation: str = ""
    caveat: str | None = None
    group_id: str | None = None
    source_series: tuple[str, ...] = ()


@dataclass
class PanelAnalysis:
    panel_id: str
    title: str
    as_of: date
    regime: str
    headline: str
    signals: list[MacroSignal] = field(default_factory=list)
    note_fragment: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class RankedDevelopment:
    rank: int
    primary_signal: MacroSignal
    confirming_signals: list[MacroSignal]
    notability_score: float
    headline: str
    explanation: str


@dataclass
class ConflictFlag:
    conflict_id: str
    title: str
    signals: list[MacroSignal]
    explanation: str
    severity: Literal["low", "medium", "high"]


@dataclass
class ResearchSynthesis:
    as_of: date
    panel_regimes: dict[str, str]
    ranked_developments: list[RankedDevelopment]
    conflicts: list[ConflictFlag]
    confirmations: list[str]
    regime_summary: str
    draft_headline: str
    draft_note_sections: dict[str, str]
    limitations: list[str]
