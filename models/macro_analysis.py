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
    evidence_type: Literal["observed", "derived", "composite", "fitted", "classification"] = "observed"


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
class ConflictFlag:
    conflict_id: str
    title: str
    signals: list[MacroSignal]
    explanation: str
    severity: Literal["low", "medium", "high"]


@dataclass
class ResearchTension:
    tension_id: str
    title: str
    evidence: list[MacroSignal]
    explanation: str
    severity: Literal["low", "medium", "high"]
    caveat: str | None = None


@dataclass
class ResearchSynthesis:
    as_of: date
    panel_regimes: dict[str, str]
    conflicts: list[ConflictFlag]
    confirmations: list[str]
    regime_summary: str
    draft_headline: str
    draft_note_sections: dict[str, str]
    limitations: list[str]


@dataclass(frozen=True)
class HypothesisTemplate:
    hypothesis_id: str
    title: str
    description: str
    expected_rules: tuple[str, ...]
    support_signal_groups: tuple[str, ...] = ()
    challenge_signal_groups: tuple[str, ...] = ()
    support_panel_ids: tuple[str, ...] = ()
    challenge_panel_ids: tuple[str, ...] = ()


@dataclass
class EvidenceItem:
    signal: MacroSignal
    relevance_score: float
    freshness_score: float
    evidence_quality_score: float
    issue_score: float
    role: str = "contextual"
    hypothesis_effects: dict[str, str] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class HypothesisAssessment:
    hypothesis_id: str
    title: str
    supporting_evidence: list[EvidenceItem]
    challenging_evidence: list[EvidenceItem]
    neutral_evidence: list[EvidenceItem]
    support_score: float
    conclusion: str


@dataclass(frozen=True)
class IssueCategoryConfig:
    category_id: str
    display_name: str
    description: str
    primary_panels: tuple[str, ...]
    secondary_panels: tuple[str, ...]
    primary_signal_groups: tuple[str, ...]
    secondary_signal_groups: tuple[str, ...]
    default_questions: tuple[str, ...]
    note_section_order: tuple[str, ...]
    default_watch_questions: tuple[str, ...]
    required_evidence_groups: tuple[str, ...]
    optional_evidence_groups: tuple[str, ...]
    hypotheses: tuple[HypothesisTemplate, ...] = ()


@dataclass
class IssueAnalysis:
    as_of: date
    category_id: str
    category_display_name: str
    question: str
    panel_regimes: dict[str, str]
    leading_interpretation: str
    alternative_interpretations: list[str]
    evidence_items: list[EvidenceItem]
    primary_evidence: list[EvidenceItem]
    contextual_evidence: list[EvidenceItem]
    hypotheses: list[HypothesisAssessment]
    topic_conflicts: list[ConflictFlag]
    data_gaps: list[str]
    watch_points: list[str]
    note_sections: dict[str, str]
    note_section_order: tuple[str, ...]
    relevant_panels: list[str]
    relevant_signal_groups: list[str]
    evidence_coverage: dict[str, float]
    methodology: list[str]
    limitations: list[str]


@dataclass
class TradeIdea:
    enabled: bool = False
    trade_type: str = ""
    instrument: str = ""
    direction: str = ""
    entry: str = ""
    target: str = ""
    stop: str = ""
    horizon: str = ""
    conviction: str = ""
    sizing_note: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class InvalidationCondition:
    label: str
    threshold_or_condition: str
    explanation: str = ""


@dataclass(frozen=True)
class ReleaseInput:
    release_name: str
    release_date: date | None
    actual: str
    consensus: str
    previous: str
    revision: str
    user_summary: str
    event_context: str


@dataclass
class ResearchWorkspaceDraft:
    mode: str
    title: str
    sections: dict[str, str]
    trade: TradeIdea | None
    invalidation_conditions: list[InvalidationCondition]
    metadata: dict[str, object]
