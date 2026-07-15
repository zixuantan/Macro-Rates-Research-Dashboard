from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

import pandas as pd

from analysis.conflicts import detect_conflicts
from analysis.signal_groups import group_signals
from analysis.issue_note_builder import build_issue_note_sections
from models.macro_analysis import (
    ConflictFlag,
    EvidenceItem,
    HypothesisAssessment,
    IssueAnalysis,
    IssueCategoryConfig,
    MacroSignal,
)


QUALITY_BY_EVIDENCE_TYPE = {
    "observed": 1.0,
    "derived": 0.9,
    "composite": 0.75,
    "fitted": 0.65,
    "classification": 0.5,
}


def _latest_panel_date(panel_analyses) -> pd.Timestamp | None:
    dates = [pd.Timestamp(panel.as_of) for panel in panel_analyses if getattr(panel, "as_of", None) is not None]
    if not dates:
        return None
    return pd.Timestamp(max(dates))


def _freshness_score(signal: MacroSignal, as_of: pd.Timestamp | None) -> float:
    if as_of is None:
        return 0.5
    signal_date = pd.Timestamp(signal.as_of)
    age_days = max((as_of - signal_date).days, 0)
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.8
    if age_days <= 90:
        return 0.6
    if age_days <= 180:
        return 0.4
    return 0.2


def _quality_score(signal: MacroSignal) -> float:
    return QUALITY_BY_EVIDENCE_TYPE.get(signal.evidence_type, 0.75)


def _selected_panel_ids(
    category_config: IssueCategoryConfig,
    selected_panel_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    if selected_panel_ids is not None:
        return tuple(dict.fromkeys(selected_panel_ids))
    return tuple(dict.fromkeys(category_config.primary_panels + category_config.secondary_panels))


def _selected_groups(
    category_config: IssueCategoryConfig,
    selected_signal_groups: Iterable[str] | None,
) -> tuple[str, ...]:
    if selected_signal_groups is not None:
        return tuple(dict.fromkeys(selected_signal_groups))
    return tuple(dict.fromkeys(category_config.primary_signal_groups + category_config.secondary_signal_groups))


def _relevance_score(
    signal: MacroSignal,
    category_config: IssueCategoryConfig,
    selected_panel_ids: tuple[str, ...],
    selected_signal_groups: tuple[str, ...],
) -> float:
    if category_config.category_id == "custom" and not selected_panel_ids and not selected_signal_groups:
        return 0.0

    if signal.group_id in category_config.primary_signal_groups or signal.group_id in selected_signal_groups[: len(category_config.primary_signal_groups)]:
        return 1.0

    if signal.group_id in category_config.secondary_signal_groups or signal.group_id in selected_signal_groups[len(category_config.primary_signal_groups) :]:
        return 0.65

    if signal.panel_id in category_config.primary_panels or signal.panel_id in selected_panel_ids[: len(category_config.primary_panels)]:
        return 0.45

    if signal.panel_id in category_config.secondary_panels or signal.panel_id in selected_panel_ids[len(category_config.primary_panels) :]:
        return 0.25

    return 0.0


def _issue_score(
    signal: MacroSignal,
    relevance_score: float,
    as_of: pd.Timestamp | None,
) -> tuple[float, float, float]:
    freshness = _freshness_score(signal, as_of)
    quality = _quality_score(signal)
    issue_score = (0.60 * relevance_score) + (0.25 * freshness) + (0.15 * quality)
    return issue_score, freshness, quality


def _signal_group_key(signal: MacroSignal) -> str:
    return signal.group_id or signal.signal_id


def _deduplicate_items(items: list[EvidenceItem], preferred_panel_ids: Iterable[str] = ()) -> list[EvidenceItem]:
    preferred_panels = set(preferred_panel_ids)
    by_group: dict[str, list[EvidenceItem]] = {}
    for item in items:
        by_group.setdefault(_signal_group_key(item.signal), []).append(item)

    deduped: list[EvidenceItem] = []
    for group_items in by_group.values():
        ordered = sorted(
            group_items,
            key=lambda item: (
                item.issue_score,
                item.evidence_quality_score,
                1.0 if item.signal.value is not None else 0.0,
                1.0 if item.signal.panel_id in preferred_panels else 0.0,
                1.0 if item.signal.evidence_type == "observed" else 0.0,
            ),
            reverse=True,
        )
        primary = ordered[0]
        deduped.append(primary)
        for contextual in ordered[1:]:
            contextual.role = "contextual"
            contextual.rationale = contextual.rationale or f"Contextual evidence alongside {primary.signal.label}."
    return sorted(
        deduped,
        key=lambda item: (
            item.issue_score,
            item.evidence_quality_score,
        ),
        reverse=True,
    )


def _effect_for_hypothesis(
    item: EvidenceItem,
    hypothesis,
) -> str:
    signal = item.signal
    if signal.group_id in hypothesis.support_signal_groups or signal.panel_id in hypothesis.support_panel_ids:
        return "Supports"
    if signal.group_id in hypothesis.challenge_signal_groups or signal.panel_id in hypothesis.challenge_panel_ids:
        return "Challenges"
    return "Neutral"


def _assess_hypotheses(
    category_config: IssueCategoryConfig,
    evidence_items: list[EvidenceItem],
) -> list[HypothesisAssessment]:
    assessments: list[HypothesisAssessment] = []
    for hypothesis in category_config.hypotheses:
        supporting: list[EvidenceItem] = []
        challenging: list[EvidenceItem] = []
        neutral: list[EvidenceItem] = []
        support_total = 0.0
        challenge_total = 0.0
        for item in evidence_items:
            effect = _effect_for_hypothesis(item, hypothesis)
            item.hypothesis_effects[hypothesis.hypothesis_id] = effect
            if effect == "Supports":
                supporting.append(item)
                support_total += item.issue_score
            elif effect == "Challenges":
                challenging.append(item)
                challenge_total += item.issue_score
            else:
                neutral.append(item)

        net = support_total - challenge_total
        if not supporting and not challenging:
            conclusion = "Insufficient evidence"
        elif net >= 1.5:
            conclusion = "Strongly supported"
        elif net >= 0.6:
            conclusion = "Moderately supported"
        elif net <= -1.5:
            conclusion = "Weakly supported"
        else:
            conclusion = "Mixed evidence"

        assessments.append(
            HypothesisAssessment(
                hypothesis_id=hypothesis.hypothesis_id,
                title=hypothesis.title,
                supporting_evidence=supporting[:4],
                challenging_evidence=challenging[:4],
                neutral_evidence=neutral[:4],
                support_score=net,
                conclusion=conclusion,
            )
        )

    assessments.sort(key=lambda item: (item.support_score, len(item.supporting_evidence)), reverse=True)
    return assessments


def _select_topic_conflicts(
    category_config: IssueCategoryConfig,
    signals: list[MacroSignal],
) -> list[ConflictFlag]:
    conflicts = detect_conflicts(signals)
    relevant_panels = set(category_config.primary_panels + category_config.secondary_panels)
    relevant_groups = set(category_config.primary_signal_groups + category_config.secondary_signal_groups)
    selected: list[ConflictFlag] = []
    for conflict in conflicts:
        if any(signal.panel_id in relevant_panels or signal.group_id in relevant_groups for signal in conflict.signals):
            selected.append(conflict)
    return selected


def _data_gaps(
    category_config: IssueCategoryConfig,
    question: str,
    evidence_items: list[EvidenceItem],
) -> list[str]:
    gaps: list[str] = []
    seen_groups = {item.signal.group_id for item in evidence_items if item.signal.group_id}
    required_missing = [group for group in category_config.required_evidence_groups if group not in seen_groups]
    if required_missing:
        gaps.append(
            "Missing evidence groups: " + ", ".join(required_missing) + "."
        )

    lowered_question = question.lower()
    if any(term in lowered_question for term in ("supply", "term premium", "issuance", "auction", "dealer positioning", "cftc")):
        gaps.append(
            "Current dashboard evidence can assess inflation, growth and market-price explanations, but cannot directly establish a Treasury-supply or term-premium explanation."
        )

    if category_config.category_id == "custom" and not evidence_items:
        gaps.append("Custom issue mode needs manually selected evidence before it can support a conclusion.")

    return gaps


def _watch_points(
    category_config: IssueCategoryConfig,
    evidence_items: list[EvidenceItem],
) -> list[str]:
    watch_points = list(category_config.default_watch_questions)
    for item in evidence_items[:3]:
        watch_points.append(f"Watch {item.signal.label}: {item.signal.interpretation or item.signal.label}.")
    return watch_points[:5]


def _leading_and_alternatives(
    hypotheses: list[HypothesisAssessment],
    category_config: IssueCategoryConfig,
    question: str,
) -> tuple[str, list[str]]:
    if not hypotheses:
        return (
            "Manual evidence selection is required for this custom issue.",
            ["Add relevant panels and signal groups, then regenerate the draft."],
        )

    leading = hypotheses[0]
    lead_text = f"{leading.title}: {leading.conclusion}."
    alternatives = [
        f"{assessment.title}: {assessment.conclusion}."
        for assessment in hypotheses[1:3]
    ]
    if category_config.category_id == "custom" and not question.strip():
        alternatives.append("Custom issue mode uses manually selected evidence and does not infer unrestricted questions.")
    return lead_text, alternatives


def build_issue_analysis(
    category_config: IssueCategoryConfig,
    question: str,
    panel_analyses,
    selected_panel_ids: Iterable[str] | None = None,
    selected_signal_groups: Iterable[str] | None = None,
) -> IssueAnalysis:
    as_of = _latest_panel_date(panel_analyses) or pd.Timestamp(date.today())
    selected_panels = _selected_panel_ids(category_config, selected_panel_ids)
    selected_groups = _selected_groups(category_config, selected_signal_groups)

    signals: list[MacroSignal] = []
    panel_regimes: dict[str, str] = {}
    for panel in panel_analyses:
        panel_regimes[panel.panel_id] = panel.regime
        signals.extend(panel.signals)

    scored_items: list[EvidenceItem] = []
    for signal in signals:
        relevance = _relevance_score(signal, category_config, selected_panels, selected_groups)
        if relevance <= 0.0:
            continue
        issue_score, freshness, quality = _issue_score(signal, relevance, as_of)
        scored_items.append(
            EvidenceItem(
                signal=signal,
                relevance_score=relevance,
                freshness_score=freshness,
                evidence_quality_score=quality,
                issue_score=issue_score,
                role="primary" if relevance >= 0.65 else "contextual",
                rationale=f"Relevant to {category_config.display_name} via {signal.group_id or signal.panel_id}.",
            )
        )

    evidence_items = _deduplicate_items(scored_items, category_config.primary_panels)
    for item in evidence_items:
        item.hypothesis_effects = {}

    primary_evidence = [item for item in evidence_items if item.role == "primary"][:8]
    contextual_evidence = [item for item in evidence_items if item.role != "primary"][:8]

    hypotheses = _assess_hypotheses(category_config, evidence_items)
    topic_conflicts = _select_topic_conflicts(category_config, signals)
    data_gaps = _data_gaps(category_config, question, evidence_items)
    watch_points = _watch_points(category_config, evidence_items)
    leading_interpretation, alternative_interpretations = _leading_and_alternatives(hypotheses, category_config, question)

    coverage = {
        "relevant_signals": float(len(evidence_items)),
        "primary_signals": float(len(primary_evidence)),
        "contextual_signals": float(len(contextual_evidence)),
        "required_groups_covered": float(
            len(
                [
                    group
                    for group in category_config.required_evidence_groups
                    if group in {item.signal.group_id for item in evidence_items if item.signal.group_id}
                ]
            )
        ),
        "required_groups_total": float(len(category_config.required_evidence_groups)),
    }

    note_sections = build_issue_note_sections(
        category_config=category_config,
        question=question,
        leading_interpretation=leading_interpretation,
        alternative_interpretations=alternative_interpretations,
        evidence_items=evidence_items,
        hypotheses=hypotheses,
        topic_conflicts=topic_conflicts,
        data_gaps=data_gaps,
        watch_points=watch_points,
        panel_regimes=panel_regimes,
    )

    methodology = [
        "Categories control evidence selection, not the conclusion.",
        "Hypothesis scores are rule-based support labels, not probabilities.",
        "Outputs are derived from FRED panel signals and can be revised as data are updated.",
        "The issue workflow cannot establish event-specific causality without external context.",
        "Consensus-surprise data and qualitative Fed communication are not included.",
        "Market signals contain risk and liquidity premia.",
    ]

    return IssueAnalysis(
        as_of=as_of.date(),
        category_id=category_config.category_id,
        category_display_name=category_config.display_name,
        question=question,
        panel_regimes=panel_regimes,
        leading_interpretation=leading_interpretation,
        alternative_interpretations=alternative_interpretations,
        evidence_items=evidence_items,
        primary_evidence=primary_evidence,
        contextual_evidence=contextual_evidence,
        hypotheses=hypotheses,
        topic_conflicts=topic_conflicts,
        data_gaps=data_gaps,
        watch_points=watch_points,
        note_sections=note_sections,
        note_section_order=category_config.note_section_order,
        relevant_panels=list(selected_panels),
        relevant_signal_groups=list(selected_groups),
        evidence_coverage=coverage,
        methodology=methodology,
        limitations=[
            "This issue analysis is rule-based and does not establish causality or predict policy decisions.",
        ],
    )
