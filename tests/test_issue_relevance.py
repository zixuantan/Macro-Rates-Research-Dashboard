from datetime import date

import pandas as pd

from analysis.issue_analysis import _deduplicate_items, _relevance_score
from analysis.issue_categories import ISSUE_CATEGORY_REGISTRY
from models.macro_analysis import EvidenceItem, MacroSignal


def _signal(
    signal_id: str,
    panel_id: str,
    group_id: str,
    *,
    evidence_type: str = "observed",
    importance_weight: float = 1.0,
    standardized_change: float = 1.0,
) -> MacroSignal:
    return MacroSignal(
        signal_id=signal_id,
        panel_id=panel_id,
        category="growth",
        label=signal_id,
        as_of=date(2026, 1, 1),
        value=1.0,
        unit="%",
        change=1.0,
        change_unit="%",
        horizon="1M",
        standardized_change=standardized_change,
        direction="higher",
        importance_weight=importance_weight,
        interpretation=signal_id,
        group_id=group_id,
        evidence_type=evidence_type,
    )


def test_primary_groups_outweigh_secondary_groups() -> None:
    config = ISSUE_CATEGORY_REGISTRY["growth_outlook"]
    primary_signal = _signal("growth_primary", "growth", "growth_momentum")
    secondary_signal = _signal("growth_secondary", "cross_asset", "credit_risk")
    primary = _relevance_score(primary_signal, config, config.primary_panels, config.primary_signal_groups + config.secondary_signal_groups)
    secondary = _relevance_score(secondary_signal, config, config.primary_panels, config.primary_signal_groups + config.secondary_signal_groups)
    assert primary > secondary


def test_observed_evidence_scores_above_classification() -> None:
    config = ISSUE_CATEGORY_REGISTRY["cross_asset_risk"]
    observed = _signal("credit_observed", "cross_asset", "credit_risk", evidence_type="observed")
    classified = _signal("credit_classified", "cross_asset", "credit_risk", evidence_type="classification")
    observed_score = _relevance_score(observed, config, config.primary_panels, config.primary_signal_groups + config.secondary_signal_groups)
    classified_score = _relevance_score(classified, config, config.primary_panels, config.primary_signal_groups + config.secondary_signal_groups)
    assert observed_score == classified_score == 1.0

    observed_item = EvidenceItem(observed, observed_score, 1.0, 1.0, 0.9)
    classified_item = EvidenceItem(classified, classified_score, 1.0, 0.5, 0.8)
    deduped = _deduplicate_items([classified_item, observed_item])
    assert deduped[0].signal.evidence_type == "observed"


def test_deduplicates_overlapping_groups() -> None:
    item_a = EvidenceItem(_signal("a", "growth", "growth_momentum"), 1.0, 1.0, 1.0, 0.9)
    item_b = EvidenceItem(_signal("b", "growth", "growth_momentum"), 1.0, 1.0, 0.95, 0.95)
    deduped = _deduplicate_items([item_a, item_b])
    assert len(deduped) == 1
    assert deduped[0].signal.signal_id == "b"
