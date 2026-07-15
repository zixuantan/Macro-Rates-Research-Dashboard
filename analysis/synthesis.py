from __future__ import annotations

from datetime import date

from models.macro_analysis import PanelAnalysis, ResearchSynthesis


def synthesize(panels: list[PanelAnalysis], as_of: date) -> ResearchSynthesis:
    """Combine panel regimes into a mechanical macro state."""
    regimes = {panel.panel_id: panel.regime for panel in panels}
    panel_count = len(panels)

    if panel_count < 3:
        return ResearchSynthesis(
            as_of=as_of,
            panel_regimes=regimes,
            conflicts=[],
            confirmations=[],
            regime_summary="Insufficient panel coverage for a reliable cross-panel synthesis.",
            draft_headline="Insufficient coverage",
            draft_note_sections={},
            limitations=[
                "Insufficient panel coverage for a reliable cross-panel synthesis.",
            ],
        )

    inflation = regimes.get("inflation", "Unavailable").lower()
    growth = regimes.get("growth", "Unavailable").lower()
    labor = regimes.get("labor", "Unavailable").lower()
    rates = regimes.get("yield_curve", "Unavailable").lower()
    cross_asset = regimes.get("cross_asset", "Unavailable").lower()

    if "rebalancing" in labor and ("easing" in cross_asset or "risk-on" in cross_asset):
        label = "Soft landing / benign easing"
    elif "weakening" in growth or "weakening" in labor:
        label = "Growth scare"
    elif "tightening" in inflation and "firm" in inflation:
        label = "Inflationary tightening"
    elif "cooling" in growth and "rebalancing" in labor:
        label = "Disinflationary slowdown"
    elif "mixed" in cross_asset or "mixed" in growth:
        label = "Mixed / low-confidence regime"
    else:
        label = "Mixed / low-confidence regime"

    regime_summary = (
        f"Mechanical synthesis: {label}. "
        f"Inflation: {inflation or 'Unavailable'}. "
        f"Growth: {growth or 'Unavailable'}. "
        f"Labor: {labor or 'Unavailable'}. "
        f"Rates: {rates or 'Unavailable'}. "
        f"Cross-asset: {cross_asset or 'Unavailable'}."
    )

    return ResearchSynthesis(
        as_of=as_of,
        panel_regimes=regimes,
        conflicts=[],
        confirmations=[],
        regime_summary=regime_summary,
        draft_headline=label,
        draft_note_sections={},
        limitations=[
            "This synthesis combines rule-based panel signals and does not establish causality or predict policy decisions."
        ],
    )
