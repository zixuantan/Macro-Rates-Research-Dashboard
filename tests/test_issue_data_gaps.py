from datetime import date

from analysis.issue_analysis import build_issue_analysis
from analysis.issue_categories import ISSUE_CATEGORY_REGISTRY
from models.macro_analysis import PanelAnalysis


def test_unsupported_supply_questions_produce_gap_warning() -> None:
    analysis = build_issue_analysis(
        ISSUE_CATEGORY_REGISTRY["treasury_curve"],
        "Is this move about Treasury supply or term premium?",
        [
            PanelAnalysis(
                panel_id="yield_curve",
                title="Treasury Yield Curve",
                as_of=date(2026, 1, 1),
                regime="Steady",
                headline="Curve is steady.",
                signals=[],
            )
        ],
    )

    assert any("term-premium" in gap.lower() or "term premium" in gap.lower() for gap in analysis.data_gaps)


def test_custom_mode_requires_manual_evidence_selection() -> None:
    analysis = build_issue_analysis(
        ISSUE_CATEGORY_REGISTRY["custom"],
        "Custom issue",
        [],
    )

    assert analysis.data_gaps
    assert "manual" in " ".join(analysis.data_gaps).lower()
