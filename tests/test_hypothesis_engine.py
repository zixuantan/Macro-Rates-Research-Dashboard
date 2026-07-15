from datetime import date

from analysis.issue_analysis import build_issue_analysis
from analysis.issue_categories import ISSUE_CATEGORY_REGISTRY
from models.macro_analysis import MacroSignal, PanelAnalysis


def _signal(signal_id: str, panel_id: str, group_id: str, value: float, *, evidence_type: str = "observed") -> MacroSignal:
    return MacroSignal(
        signal_id=signal_id,
        panel_id=panel_id,
        category="labor",
        label=signal_id,
        as_of=date(2026, 1, 1),
        value=value,
        unit="%",
        change=0.2,
        change_unit="%",
        horizon="1M",
        standardized_change=0.8,
        direction="higher",
        importance_weight=1.0,
        interpretation=signal_id,
        group_id=group_id,
        evidence_type=evidence_type,
    )


def test_hypothesis_support_and_challenge_rules_work() -> None:
    config = ISSUE_CATEGORY_REGISTRY["labor_policy"]
    panel = PanelAnalysis(
        panel_id="labor",
        title="Labor",
        as_of=date(2026, 1, 1),
        regime="Cooling",
        headline="Labor is cooling.",
        signals=[
            _signal("payrolls", "labor", "labor_demand", 175.0),
            _signal("unemployment", "labor", "labor_slack", 4.2),
            _signal("wages", "labor", "labor_wage_pressure", 3.8),
        ],
    )

    analysis = build_issue_analysis(config, "Is the labor market cooling or weakening?", [panel])

    assert analysis.hypotheses
    top = analysis.hypotheses[0]
    assert top.conclusion in {"Strongly supported", "Moderately supported", "Mixed evidence"}
    assert top.support_score >= -1.0


def test_issue_scores_are_not_probabilities() -> None:
    config = ISSUE_CATEGORY_REGISTRY["labor_policy"]
    panel = PanelAnalysis(
        panel_id="labor",
        title="Labor",
        as_of=date(2026, 1, 1),
        regime="Cooling",
        headline="Labor is cooling.",
        signals=[
            _signal("payrolls", "labor", "labor_demand", 175.0),
            _signal("unemployment", "labor", "labor_slack", 4.2),
        ],
    )

    analysis = build_issue_analysis(config, "Is the labor market cooling or weakening?", [panel])
    assert analysis.hypotheses[0].support_score > 1.0 or analysis.hypotheses[0].support_score < 0.0
