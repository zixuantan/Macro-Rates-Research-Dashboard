from datetime import date

from analysis.issue_analysis import build_issue_analysis
from analysis.issue_categories import ISSUE_CATEGORY_REGISTRY
from analysis.issue_note_builder import issue_note_text
from models.macro_analysis import MacroSignal, PanelAnalysis


def _signal(signal_id: str, panel_id: str, group_id: str) -> MacroSignal:
    return MacroSignal(
        signal_id=signal_id,
        panel_id=panel_id,
        category="inflation",
        label=signal_id,
        as_of=date(2026, 1, 1),
        value=2.5,
        unit="%",
        change=0.1,
        change_unit="%",
        horizon="1M",
        standardized_change=0.5,
        direction="lower",
        importance_weight=1.0,
        interpretation=signal_id,
        group_id=group_id,
    )


def test_issue_note_sections_vary_by_category() -> None:
    panel = PanelAnalysis(
        panel_id="inflation",
        title="Inflation",
        as_of=date(2026, 1, 1),
        regime="Cooling",
        headline="Inflation is cooling.",
        signals=[
            _signal("cpi", "inflation", "realized_inflation"),
            _signal("breakeven", "inflation", "medium_term_inflation_pricing"),
            _signal("wage", "labor", "labor_wage_pressure"),
        ],
    )
    analysis = build_issue_analysis(
        ISSUE_CATEGORY_REGISTRY["inflation_policy"],
        "Is disinflation continuing or stalling?",
        [panel],
    )
    note = issue_note_text(analysis.note_sections, analysis.note_section_order, "Inflation and Fed Policy")
    assert "Realized inflation" in note
    assert "Market inflation pricing" in note
    assert "Research question" in note
