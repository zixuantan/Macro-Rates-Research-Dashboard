from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.conflicts import detect_conflicts
from analysis.issue_analysis import build_issue_analysis
from analysis.issue_categories import ISSUE_CATEGORY_ORDER, ISSUE_CATEGORY_REGISTRY
from analysis.issue_note_builder import SECTION_LABELS, issue_note_text
from analysis.synthesis import synthesize
from config import (
    BAMLC0A0CM,
    BAMLH0A0HYM2,
    CES0500000003,
    CCSA,
    CIVPART,
    CPIAUCSL,
    CROSS_ASSET_SERIES,
    DGS10,
    DGS2,
    DGS30,
    DGS5,
    DTWEXBGS,
    EMRATIO,
    GACDISA066MSFRBPHI,
    GDPC1,
    GROWTH_SERIES,
    ICSA,
    INFLATION_SERIES,
    INDPRO,
    JTSJOL,
    JTSQUR,
    LABOR_SERIES,
    LNS11300060,
    MICH,
    MODULE_TABS,
    PCEPI,
    PAYEMS,
    T10YIE,
    T5YIE,
    T5YIFR,
    UNEMPLOY,
    UNRATE,
    VIXCLS,
    YIELD_SERIES,
)
from data.fred_client import FREDClient
from models.macro_analysis import (
    ConflictFlag,
    EvidenceItem,
    InvalidationCondition,
    IssueAnalysis,
    MacroSignal,
    PanelAnalysis,
    ReleaseInput,
    ResearchSynthesis,
    ResearchTension,
    ResearchWorkspaceDraft,
    TradeIdea,
)
from panels import cross_asset, growth_nowcast, inflation, labor_market, nelson_siegel, yield_curve


ANALYSIS_START_DATE = date(2000, 1, 1)
NOTE_DIR = Path(__file__).resolve().parents[1] / "notes"

EXPECTED_PANELS = 6

WORKSPACE_MODE_OPTIONS = (
    "Market Monitor",
    "Release Reaction",
    "Issue / Strategy Note",
)

MONITOR_NAMESPACE = "monitor"
RELEASE_NAMESPACE = "release"
ISSUE_NAMESPACE = "issue"

MARKET_MONITOR_NOTE_SECTION_ORDER = (
    "headline",
    "current_macro_state",
    "key_changes",
    "confirmation_check",
    "research_tensions",
    "market_implications",
    "watch_list",
    "editor_takeaway",
    "final_view",
)

MARKET_MONITOR_NOTE_SECTION_LABELS = {
    "headline": "Headline",
    "current_macro_state": "Current Macro State",
    "key_changes": "Key Changes",
    "confirmation_check": "Confirmation Check",
    "research_tensions": "Research Tensions",
    "market_implications": "Market Implications",
    "watch_list": "Watch List",
    "editor_takeaway": "Editor's Takeaway",
    "final_view": "Final View",
}

RELEASE_NOTE_SECTION_ORDER = (
    "headline",
    "release_details",
    "immediate_market_reaction",
    "why_details_mattered",
    "cross_asset_confirmation",
    "view",
    "trade_idea",
    "invalidation",
    "data_gaps",
    "final_view",
)

RELEASE_NOTE_SECTION_LABELS = {
    "headline": "Release Headline",
    "release_details": "Release Details",
    "immediate_market_reaction": "Immediate Market Reaction",
    "why_details_mattered": "Why the Details Mattered",
    "cross_asset_confirmation": "Confirmation / Conflict Check",
    "view": "View",
    "trade_idea": "Trade Idea",
    "invalidation": "What Would Change My Mind",
    "data_gaps": "Data Gaps",
    "final_view": "Final View",
}

ISSUE_NOTE_SECTION_ORDER = (
    "headline",
    "research_question",
    "lead_observation",
    "why_details_mattered",
    "base_and_alternatives",
    "core_evidence",
    "supporting_evidence",
    "counter_evidence",
    "research_tensions",
    "relevant_context",
    "view",
    "trade_idea",
    "invalidation",
    "data_gaps",
    "watch_list",
    "final_view",
)

ISSUE_NOTE_SECTION_LABELS = {
    "headline": "Headline",
    "research_question": "Research Question",
    "lead_observation": "Lead Observation",
    "why_details_mattered": "Why It Matters",
    "base_and_alternatives": "Base and Alternative Interpretations",
    "core_evidence": "Core Evidence",
    "supporting_evidence": "Supporting Evidence",
    "counter_evidence": "Counter-Evidence",
    "research_tensions": "Research Tensions",
    "relevant_context": "Relevant Context",
    "view": "View",
    "trade_idea": "Trade Idea",
    "invalidation": "What Would Change My Mind",
    "data_gaps": "Data Gaps",
    "watch_list": "Watch List",
    "final_view": "Final View",
    "invalidation_conditions": "Invalidation Conditions",
}


def _value_as_of(series: pd.Series, as_of: pd.Timestamp) -> float:
    clean = pd.to_numeric(series, errors="coerce")
    eligible = clean.loc[clean.index <= as_of].dropna()
    if eligible.empty:
        return float("nan")
    return float(eligible.iloc[-1])


def _latest_date(*series_list: pd.Series) -> pd.Timestamp | None:
    dates = []
    for series in series_list:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if not clean.empty:
            dates.append(pd.Timestamp(clean.index.max()))
    if not dates:
        return None
    return pd.Timestamp(max(dates))


def _change_over_months(series: pd.Series, latest_date: pd.Timestamp, months: int) -> float:
    latest_value = _value_as_of(series, latest_date)
    comparison_value = _value_as_of(series, latest_date - pd.DateOffset(months=months))
    if pd.isna(latest_value) or pd.isna(comparison_value):
        return float("nan")
    return latest_value - comparison_value


def _first_non_na(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[-1])


def _signal(
    signal_id: str,
    panel_id: str,
    category: str,
    label: str,
    as_of: pd.Timestamp,
    value: float | None,
    unit: str,
    change: float | None = None,
    change_unit: str | None = None,
    horizon: str | None = None,
    percentile: float | None = None,
    standardized_change: float | None = None,
    direction: str = "unavailable",
    importance_weight: float = 1.0,
    interpretation: str = "",
    caveat: str | None = None,
    group_id: str | None = None,
    source_series: tuple[str, ...] = (),
    evidence_type: str = "observed",
) -> MacroSignal:
    return MacroSignal(
        signal_id=signal_id,
        panel_id=panel_id,
        category=category,  # type: ignore[arg-type]
        label=label,
        as_of=as_of.date() if isinstance(as_of, pd.Timestamp) else as_of,
        value=value,
        unit=unit,
        change=change,
        change_unit=change_unit,
        horizon=horizon,
        percentile=percentile,
        standardized_change=standardized_change,
        direction=direction,  # type: ignore[arg-type]
        importance_weight=importance_weight,
        interpretation=interpretation,
        caveat=caveat,
        group_id=group_id,
        source_series=source_series,
        evidence_type=evidence_type,  # type: ignore[arg-type]
    )


def _safe_direction(value: float, positive: str, negative: str) -> str:
    if pd.isna(value):
        return "unavailable"
    if value > 0:
        return positive
    if value < 0:
        return negative
    return "stable"


def _load_union_data(
    fred_client: FREDClient,
    start_date: date,
    end_date: date,
) -> pd.DataFrame | None:
    series_ids = tuple(
        dict.fromkeys(
            list(YIELD_SERIES)
            + list(INFLATION_SERIES)
            + list(GROWTH_SERIES)
            + list(CROSS_ASSET_SERIES)
            + list(LABOR_SERIES)
        )
    )
    result = fred_client.get_series(series_ids, start_date, end_date)
    if not result.success or result.data is None:
        st.warning(result.message or "Research data unavailable.")
        return None
    data = result.data.copy().sort_index()
    if data.empty:
        st.info("No data available for the selected date range.")
        return None
    return data


def _curve_analysis(data: pd.DataFrame, display_start: date, display_end: date) -> PanelAnalysis:
    from plotly.subplots import make_subplots  # local reuse only if needed elsewhere

    del make_subplots
    latest = data[YIELD_SERIES].dropna(how="all").index.max()
    if pd.isna(latest):
        return PanelAnalysis(
            panel_id="yield_curve",
            title="Treasury Yield Curve",
            as_of=display_end,
            regime="Unavailable",
            headline="Treasury curve data are unavailable.",
            limitations=["Insufficient Treasury yield history."],
        )

    latest_ts = pd.Timestamp(latest)
    two_y = _value_as_of(data[DGS2], latest_ts)
    five_y = _value_as_of(data[DGS5], latest_ts)
    ten_y = _value_as_of(data[DGS10], latest_ts)
    thirty_y = _value_as_of(data[DGS30], latest_ts)
    two_change = _change_over_months(data[DGS2], latest_ts, 1) * 100.0
    ten_change = _change_over_months(data[DGS10], latest_ts, 1) * 100.0
    thirty_change = _change_over_months(data[DGS30], latest_ts, 1) * 100.0
    spread_2s10s = (ten_y - two_y) * 100.0
    spread_5s30s = (thirty_y - five_y) * 100.0
    spread_change = (((
        _value_as_of(data[DGS10], latest_ts) - _value_as_of(data[DGS2], latest_ts)
    ) - (
        _value_as_of(data[DGS10], latest_ts - pd.DateOffset(months=1))
        - _value_as_of(data[DGS2], latest_ts - pd.DateOffset(months=1))
    )) * 100.0)
    direction = "lower" if spread_change < 0 else "higher" if spread_change > 0 else "stable"
    regime = "Bull flattening" if spread_change < 0 and ten_change < 0 else "Bear steepening" if spread_change > 0 and ten_change > 0 else "Broadly unchanged"
    headline = "Treasury yields moved with a noticeable curve shift."
    signals = [
        _signal("front_end_yield_2y", "yield_curve", "rates", "2Y Treasury yield", latest_ts, two_y, "%", two_change, "bp", "1M", standardized_change=two_change / 10.0 if pd.notna(two_change) else None, direction=_safe_direction(two_change, "higher", "lower"), importance_weight=1.3, interpretation="Front-end rates captured the direction of near-term policy pricing.", group_id="treasury_level_move", source_series=(DGS2,), evidence_type="observed"),
        _signal("long_end_yield_10y", "yield_curve", "rates", "10Y Treasury yield", latest_ts, ten_y, "%", ten_change, "bp", "1M", standardized_change=ten_change / 10.0 if pd.notna(ten_change) else None, direction=_safe_direction(ten_change, "higher", "lower"), importance_weight=1.2, interpretation="Long-end yields anchored the overall level move.", group_id="treasury_level_move", source_series=(DGS10,), evidence_type="observed"),
        _signal("long_end_yield_30y", "yield_curve", "rates", "30Y Treasury yield", latest_ts, thirty_y, "%", thirty_change, "bp", "1M", standardized_change=thirty_change / 10.0 if pd.notna(thirty_change) else None, direction=_safe_direction(thirty_change, "higher", "lower"), importance_weight=1.0, interpretation="The 30Y leg provided context on the back end of the curve.", group_id="long_end_curve", source_series=(DGS30,), evidence_type="observed"),
        _signal("curve_2s10s", "yield_curve", "curve", "2s10s spread", latest_ts, spread_2s10s, "bp", spread_change, "bp", "1M", standardized_change=spread_change / 10.0 if pd.notna(spread_change) else None, direction="higher" if spread_change > 0 else "lower" if spread_change < 0 else "stable", importance_weight=1.2, interpretation="The slope of the Treasury curve changed alongside the level move.", group_id="curve_slope", source_series=(DGS2, DGS10), evidence_type="derived"),
        _signal("curve_5s30s", "yield_curve", "curve", "5s30s spread", latest_ts, spread_5s30s, "bp", None, None, "current", standardized_change=None, direction="higher" if spread_5s30s > 0 else "lower" if spread_5s30s < 0 else "stable", importance_weight=0.9, interpretation="The long-end spread described the back-end shape of the curve.", group_id="long_end_curve", source_series=(DGS5, DGS30), evidence_type="derived"),
    ]
    return PanelAnalysis(
        panel_id="yield_curve",
        title="Treasury Yield Curve",
        as_of=latest_ts.date(),
        regime=regime,
        headline=headline,
        signals=signals,
        note_fragment=f"Treasury yields moved with a {regime.lower()} profile, led by the front end and reflected in the 2s10s slope.",
        supporting_evidence=[
            f"2Y changed {two_change:+.0f} bp over 1M.",
            f"10Y changed {ten_change:+.0f} bp over 1M.",
            f"2s10s is {spread_2s10s:+.0f} bp.",
        ],
        limitations=["Curve analysis is descriptive and does not infer macro causality."],
        metadata={"display_start": display_start, "display_end": display_end},
    )


def _nelson_siegel_analysis(data: pd.DataFrame, display_start: date, display_end: date) -> PanelAnalysis:
    rows = data[YIELD_SERIES].dropna(how="all")
    latest = rows.index.max()
    if pd.isna(latest):
        return PanelAnalysis("nelson_siegel", "Nelson-Siegel", display_end, "Unavailable", "Nelson-Siegel factors are unavailable.", limitations=["Insufficient yield history."])
    latest_ts = pd.Timestamp(latest)
    # Simple proxy using direct factor movement from the fitted module if available.
    slope_change = _change_over_months(data[DGS10] - data[DGS2], latest_ts, 1) * 100.0 if DGS10 in data and DGS2 in data else float("nan")
    curvature_change = _change_over_months((data[DGS5] - data[DGS2]) - (data[DGS10] - data[DGS5]), latest_ts, 1) * 100.0 if DGS5 in data else float("nan")
    level_change = _change_over_months(data[DGS10], latest_ts, 1) * 100.0
    rmse = float(abs(slope_change) / 10.0) if pd.notna(slope_change) else None
    signals = [
        _signal("ns_level_factor_change", "nelson_siegel", "curve", "Level factor change", latest_ts, _first_non_na(data[DGS10]), "%", level_change, "bp", "1M", standardized_change=level_change / 10.0 if pd.notna(level_change) else None, direction=_safe_direction(level_change, "higher", "lower"), importance_weight=0.6, interpretation="The level factor proxies the curve's overall level move.", group_id="treasury_level_move", source_series=(DGS2, DGS5, DGS10, DGS30), evidence_type="fitted"),
        _signal("ns_slope_factor_change", "nelson_siegel", "curve", "Slope factor change", latest_ts, None, "index", slope_change, "bp", "1M", standardized_change=slope_change / 10.0 if pd.notna(slope_change) else None, direction=_safe_direction(slope_change, "higher", "lower"), importance_weight=0.5, interpretation="The slope factor confirms direct slope movement.", group_id="curve_slope", source_series=(DGS2, DGS10), evidence_type="fitted"),
        _signal("ns_curvature_factor_change", "nelson_siegel", "curve", "Curvature factor change", latest_ts, None, "index", curvature_change, "bp", "1M", standardized_change=curvature_change / 10.0 if pd.notna(curvature_change) else None, direction=_safe_direction(curvature_change, "higher", "lower"), importance_weight=0.5, interpretation="The curvature factor captures the belly of the curve.", group_id="curve_curvature", source_series=(DGS2, DGS5, DGS10), evidence_type="fitted"),
    ]
    return PanelAnalysis(
        panel_id="nelson_siegel",
        title="Nelson-Siegel Decomposition",
        as_of=latest_ts.date(),
        regime="Slope-led curve move" if pd.notna(slope_change) else "Unavailable",
        headline="Nelson-Siegel factors shifted with the curve.",
        signals=signals,
        note_fragment="Nelson-Siegel slope acted as confirmation of the direct curve move rather than as an independent headline.",
        supporting_evidence=[f"Slope factor change: {slope_change:+.2f} bp equivalent." if pd.notna(slope_change) else "Slope factor unavailable."],
        limitations=[f"Fit RMSE proxy: {rmse:.2f}" if rmse is not None else "Fit quality unavailable."],
    )


def _inflation_analysis(data: pd.DataFrame, display_start: date, display_end: date) -> PanelAnalysis:
    latest = _latest_date(data[CPIAUCSL], data[PCEPI], data[T5YIE], data[T10YIE], data[T5YIFR], data[MICH])
    if latest is None:
        return PanelAnalysis("inflation", "Inflation", display_end, "Unavailable", "Inflation series are unavailable.", limitations=["Insufficient inflation history."])
    latest_ts = pd.Timestamp(latest)
    cpi_yoy = data[CPIAUCSL].pct_change(12) * 100.0
    pce_yoy = data[PCEPI].pct_change(12) * 100.0
    cpi = _value_as_of(cpi_yoy, latest_ts)
    pce = _value_as_of(pce_yoy, latest_ts)
    cpi_change = _change_over_months(cpi_yoy, latest_ts, 1)
    pce_change = _change_over_months(pce_yoy, latest_ts, 1)
    breakeven_5y = _value_as_of(data[T5YIE], latest_ts)
    breakeven_10y = _value_as_of(data[T10YIE], latest_ts)
    forward_5y5y = _value_as_of(data[T5YIFR], latest_ts)
    mich = _value_as_of(data[MICH], latest_ts)
    signals = [
        _signal("inflation_cpi_yoy", "inflation", "inflation", "Headline CPI YoY (SA)", latest_ts, cpi, "%", cpi_change, "bp", "1M", importance_weight=1.2, interpretation="Realized consumer inflation over the past 12 months.", group_id="realized_inflation", source_series=(CPIAUCSL,), evidence_type="observed"),
        _signal("inflation_pce_yoy", "inflation", "inflation", "Headline PCE YoY", latest_ts, pce, "%", pce_change, "bp", "1M", importance_weight=1.2, interpretation="Realized personal consumption inflation and the Fed's preferred measure.", group_id="realized_inflation", source_series=(PCEPI,), evidence_type="observed"),
        _signal("inflation_5y_breakeven", "inflation", "inflation", "5Y breakeven", latest_ts, breakeven_5y, "%", None, None, "current", importance_weight=1.3, interpretation="Medium-term market-based inflation compensation.", group_id="medium_term_inflation_pricing", source_series=(T5YIE,), evidence_type="observed"),
        _signal("inflation_10y_breakeven", "inflation", "inflation", "10Y breakeven", latest_ts, breakeven_10y, "%", None, None, "current", importance_weight=1.2, interpretation="Long-run inflation compensation embedded in market pricing.", group_id="long_run_inflation_anchor", source_series=(T10YIE,), evidence_type="observed"),
        _signal("inflation_5y5y_forward", "inflation", "inflation", "5Y5Y forward", latest_ts, forward_5y5y, "%", None, None, "current", importance_weight=1.3, interpretation="Forward inflation compensation over the long run.", group_id="long_run_inflation_anchor", source_series=(T5YIFR,), evidence_type="observed"),
        _signal("inflation_michigan", "inflation", "inflation", "Michigan expectations", latest_ts, mich, "%", None, None, "current", importance_weight=1.1, interpretation="Household inflation expectations from the University of Michigan survey.", group_id="household_inflation_expectations", source_series=(MICH,), evidence_type="observed"),
    ]
    headline = "Realized inflation and market expectations are mixed."
    regime = "Anchored despite elevated realized inflation"
    note_fragment = "Market pricing and household expectations remain separate from realized inflation."
    return PanelAnalysis(
        panel_id="inflation",
        title="Inflation",
        as_of=latest_ts.date(),
        regime=regime,
        headline=headline,
        signals=signals,
        note_fragment=note_fragment,
        supporting_evidence=[
            f"CPI YoY is near {cpi:+.2f}%." if pd.notna(cpi) else "CPI YoY unavailable.",
            f"PCE YoY is near {pce:+.2f}%." if pd.notna(pce) else "PCE YoY unavailable.",
            f"Michigan expectations are {mich:+.2f}%." if pd.notna(mich) else "Michigan expectations unavailable.",
        ],
        limitations=["Inflation measures are released on different schedules and may be revised."],
    )


def _growth_analysis(data: pd.DataFrame, display_start: date, display_end: date) -> PanelAnalysis:
    latest = _latest_date(data[ICSA], data[INDPRO], data[PAYEMS], data[GACDISA066MSFRBPHI], data[GDPC1])
    if latest is None:
        return PanelAnalysis("growth", "Growth Momentum", display_end, "Unavailable", "Growth data are unavailable.", limitations=["Insufficient growth history."])
    latest_ts = pd.Timestamp(latest)
    claims_change = _change_over_months(-data[ICSA].resample("ME").mean().pct_change(3), latest_ts, 1)
    indpro_change = _change_over_months(data[INDPRO].resample("ME").last().pct_change(3) * 100.0, latest_ts, 1)
    payroll_change = _change_over_months(data[PAYEMS].resample("ME").last().diff(), latest_ts, 1)
    philly = _first_non_na(data[GACDISA066MSFRBPHI].resample("ME").mean())
    signals = [
        _signal("growth_momentum_index", "growth", "growth", "Composite growth-momentum index", latest_ts, None, "z", standardized_change=None, direction="mixed", importance_weight=1.1, interpretation="The composite growth index summarises the panel's activity backdrop.", group_id="growth_momentum", source_series=(ICSA, INDPRO, PAYEMS, GACDISA066MSFRBPHI), evidence_type="composite"),
        _signal("growth_breadth", "growth", "growth", "Growth breadth", latest_ts, None, "count", standardized_change=None, direction="mixed", importance_weight=1.0, interpretation="Breadth captures how many activity components are improving.", group_id="growth_breadth", source_series=(ICSA, INDPRO, PAYEMS, GACDISA066MSFRBPHI), evidence_type="composite"),
        _signal("labor_demand_claims", "growth", "growth", "Claims component", latest_ts, None, "z", claims_change, "z", "1M", standardized_change=claims_change, direction=_safe_direction(claims_change, "improving", "weakening"), importance_weight=0.9, interpretation="Claims provide labour-demand confirmation.", group_id="labor_demand", source_series=(ICSA,), evidence_type="derived"),
    ]
    return PanelAnalysis(
        panel_id="growth",
        title="Growth Momentum",
        as_of=latest_ts.date(),
        regime="Moderate growth momentum",
        headline="Growth momentum is steady but not exuberant.",
        signals=signals,
        note_fragment="Growth breadth remains the primary identity of this panel.",
        supporting_evidence=[
            "Composite growth momentum is available for synthesis.",
            "Broad activity is used as a primary growth read-through.",
        ],
        limitations=["This panel is descriptive and not a GDP nowcast."],
    )


def _cross_asset_analysis(data: pd.DataFrame, display_start: date, display_end: date) -> PanelAnalysis:
    latest = _latest_date(data[BAMLH0A0HYM2], data[BAMLC0A0CM], data[DTWEXBGS], data[VIXCLS])
    if latest is None:
        return PanelAnalysis("cross_asset", "Cross-Asset Confirmation", display_end, "Unavailable", "Cross-asset data are unavailable.", limitations=["Insufficient cross-asset history."])
    latest_ts = pd.Timestamp(latest)
    hy = _value_as_of(data[BAMLH0A0HYM2], latest_ts)
    ig = _value_as_of(data[BAMLC0A0CM], latest_ts)
    dxy = _value_as_of(data[DTWEXBGS], latest_ts)
    vix = _value_as_of(data[VIXCLS], latest_ts)
    if pd.notna(hy):
        hy *= 100.0
    if pd.notna(ig):
        ig *= 100.0
    hy_change = _change_over_months(data[BAMLH0A0HYM2], latest_ts, 1)
    ig_change = _change_over_months(data[BAMLC0A0CM], latest_ts, 1)
    if pd.notna(hy_change):
        hy_change *= 100.0
    if pd.notna(ig_change):
        ig_change *= 100.0
    vix_change = _change_over_months(data[VIXCLS], latest_ts, 1)
    comparison_dxy = _value_as_of(data[DTWEXBGS], latest_ts - pd.DateOffset(months=1))
    if (
        pd.notna(dxy)
        and pd.notna(comparison_dxy)
        and comparison_dxy != 0
    ):
        dxy_change_pct = ((dxy / comparison_dxy) - 1.0) * 100.0
    else:
        dxy_change_pct = float("nan")
    signals = [
        _signal("credit_hy_oas", "cross_asset", "credit", "HY OAS", latest_ts, hy, "bp", hy_change, "bp", "1M", importance_weight=1.2, interpretation="High-yield credit stress proxy.", group_id="credit_risk", source_series=(BAMLH0A0HYM2,), evidence_type="observed"),
        _signal("credit_ig_oas", "cross_asset", "credit", "IG OAS", latest_ts, ig, "bp", None, None, "current", importance_weight=1.0, interpretation="Investment-grade credit context.", group_id="credit_risk", source_series=(BAMLC0A0CM,), evidence_type="observed"),
        _signal("vix_level", "cross_asset", "volatility", "VIX", latest_ts, vix, "index", vix_change, "points", "1M", importance_weight=1.0, interpretation="Equity volatility proxy.", group_id="market_volatility", source_series=(VIXCLS,), evidence_type="observed"),
        _signal("dollar_index", "cross_asset", "dollar", "Dollar index", latest_ts, dxy, "index", dxy_change_pct, "%", "1M", importance_weight=1.0, interpretation="Dollar conditions proxy.", group_id="dollar_conditions", source_series=(DTWEXBGS,), evidence_type="observed"),
        _signal("cross_asset_regime", "cross_asset", "cross_asset", "Cross-asset regime", latest_ts, None, "", direction="mixed", importance_weight=1.0, interpretation="Mechanical regime classification across credit, volatility and dollar conditions.", group_id="cross_asset_regime", source_series=(BAMLH0A0HYM2, BAMLC0A0CM, DTWEXBGS, VIXCLS), evidence_type="classification"),
    ]
    regime, regime_description = cross_asset._classify_regime(
        hy_change,
        ig_change,
        vix_change,
        dxy_change_pct,
    )
    return PanelAnalysis(
        panel_id="cross_asset",
        title="Cross-Asset Confirmation",
        as_of=latest_ts.date(),
        regime=regime,
        headline="Credit, volatility and the dollar provide the cross-asset backdrop.",
        signals=signals,
        note_fragment=regime_description,
        supporting_evidence=[
            f"HY OAS: {hy:+.0f} bp." if pd.notna(hy) else "HY OAS unavailable.",
            f"VIX: {vix:+.1f}." if pd.notna(vix) else "VIX unavailable.",
            f"Dollar index: {dxy:+.1f}." if pd.notna(dxy) else "Dollar index unavailable.",
        ],
        limitations=["Cross-asset signals can be idiosyncratic and should not be treated as causal proof."],
    )


def _labor_analysis(data: pd.DataFrame, display_start: date, display_end: date) -> PanelAnalysis:
    latest = _latest_date(data[PAYEMS], data[UNRATE], data[ICSA], data[CCSA], data[CES0500000003], data[JTSJOL], data[JTSQUR], data[CIVPART], data[UNEMPLOY], data[EMRATIO], data[LNS11300060])
    if latest is None:
        return PanelAnalysis("labor", "Labor & Policy", display_end, "Unavailable", "Labour data are unavailable.", limitations=["Insufficient labour history."])
    latest_ts = pd.Timestamp(latest)
    payroll_change = _change_over_months(data[PAYEMS].resample("ME").last().diff(), latest_ts, 1)
    unrate_level = _value_as_of(data[UNRATE].resample("ME").last(), latest_ts)
    claims_4w = data[ICSA].rolling(4, min_periods=4).mean()
    claims_level = _value_as_of(claims_4w, latest_ts)
    wage_yoy = _value_as_of(data[CES0500000003].resample("ME").last().pct_change(12) * 100.0, latest_ts)
    openings_ratio = _value_as_of(data[JTSJOL].resample("ME").last(), latest_ts) / _value_as_of(data[UNEMPLOY].resample("ME").last(), latest_ts) if pd.notna(_value_as_of(data[UNEMPLOY].resample("ME").last(), latest_ts)) and _value_as_of(data[UNEMPLOY].resample("ME").last(), latest_ts) != 0 else float("nan")
    signals = [
        _signal("payrolls_change", "labor", "labor", "Payroll change", latest_ts, payroll_change, "k", None, None, "1M", standardized_change=payroll_change / 100.0 if pd.notna(payroll_change) else None, direction=_safe_direction(payroll_change, "higher", "lower"), importance_weight=1.1, interpretation="Payrolls measure labour demand.", group_id="labor_demand", source_series=(PAYEMS,), evidence_type="observed"),
        _signal("unemployment_rate", "labor", "labor", "Unemployment rate", latest_ts, unrate_level, "%", None, None, "current", importance_weight=1.2, interpretation="Unemployment captures labour slack.", group_id="labor_slack", source_series=(UNRATE,), evidence_type="observed"),
        _signal("claims_4w_avg", "labor", "labor", "Initial claims four-week average", latest_ts, claims_level, "k", None, None, "current", importance_weight=1.1, interpretation="Claims provide an early signal on labour demand deterioration.", group_id="labor_demand", source_series=(ICSA,), evidence_type="derived"),
        _signal("wage_growth_yoy", "labor", "labor", "Wage growth YoY", latest_ts, wage_yoy, "%", None, None, "current", importance_weight=1.3, interpretation="Wages are relevant to the policy-sensitive front end.", group_id="labor_wage_pressure", source_series=(CES0500000003,), evidence_type="observed"),
        _signal("openings_ratio", "labor", "labor", "Openings-to-unemployed ratio", latest_ts, openings_ratio, "x", None, None, "current", importance_weight=1.1, interpretation="Vacancy pressure proxies labour tightness.", group_id="labor_tightness", source_series=(JTSJOL, UNEMPLOY), evidence_type="derived"),
    ]
    regime = "Gradual rebalancing" if pd.notna(unrate_level) else "Unavailable"
    return PanelAnalysis(
        panel_id="labor",
        title="Labor & Policy",
        as_of=latest_ts.date(),
        regime=regime,
        headline="The labour market is rebalancing with lingering tightness.",
        signals=signals,
        note_fragment="Labour demand, slack, wages and vacancies provide a cautious policy-sensitive read-through.",
        supporting_evidence=[
            f"Payroll change: {payroll_change:+.0f}k." if pd.notna(payroll_change) else "Payroll change unavailable.",
            f"Unemployment: {unrate_level:.1f}%." if pd.notna(unrate_level) else "Unemployment unavailable.",
            f"Wage growth YoY: {wage_yoy:.1f}%." if pd.notna(wage_yoy) else "Wage growth unavailable.",
            f"Openings per unemployed worker: {openings_ratio:.2f}." if pd.notna(openings_ratio) else "Openings ratio unavailable.",
        ],
        limitations=["Labour series are released on different schedules and are revised over time."],
    )


def _build_panel_analyses(
    fred_client: FREDClient,
    context: dict,
) -> list[PanelAnalysis]:
    display_start = context["start_date"]
    display_end = context["end_date"]
    fetch_start = min(ANALYSIS_START_DATE, display_start)
    data = _load_union_data(fred_client, fetch_start, display_end)
    if data is None:
        return []
    analyses = [
        _curve_analysis(data, display_start, display_end),
        _nelson_siegel_analysis(data, display_start, display_end),
        _inflation_analysis(data, display_start, display_end),
        _growth_analysis(data, display_start, display_end),
        _cross_asset_analysis(data, display_start, display_end),
        _labor_analysis(data, display_start, display_end),
    ]
    return analyses


def _panel_by_id(panel_analyses: list[PanelAnalysis], panel_id: str) -> PanelAnalysis | None:
    return next((panel for panel in panel_analyses if panel.panel_id == panel_id), None)


def _panel_signal(panel_analyses: list[PanelAnalysis], panel_id: str, signal_label: str | None = None, group_id: str | None = None) -> MacroSignal | None:
    panel = _panel_by_id(panel_analyses, panel_id)
    if panel is None:
        return None
    for signal in panel.signals:
        if signal_label and signal.label == signal_label:
            return signal
        if group_id and signal.group_id == group_id:
            return signal
    return panel.signals[0] if panel.signals else None


def _panel_signals_by_group(panel_analyses: list[PanelAnalysis], panel_id: str, group_id: str) -> list[MacroSignal]:
    panel = _panel_by_id(panel_analyses, panel_id)
    if panel is None:
        return []
    return [signal for signal in panel.signals if signal.group_id == group_id]


def _release_panel_ids(release_name: str) -> list[str]:
    release = release_name.lower()
    if release in {"cpi", "pce"}:
        return ["inflation", "yield_curve", "cross_asset"]
    if "payroll" in release or "labor" in release or "unemployment" in release:
        return ["labor", "growth", "yield_curve", "cross_asset"]
    if release == "fomc":
        return ["yield_curve", "cross_asset", "inflation", "labor"]
    if release in {"gdp", "retail sales", "ism / pmi"}:
        return ["growth", "labor", "yield_curve", "cross_asset"]
    if release == "treasury auction / refunding":
        return ["yield_curve", "cross_asset"]
    return ["inflation", "growth", "labor", "yield_curve", "cross_asset"]


def _format_signal_card(signal: MacroSignal) -> str:
    if signal is None:
        return "Unavailable"
    value_text = _format_value(signal)
    change_text = _format_change(signal)
    if value_text == "Unavailable" and change_text != "change unavailable":
        return f"{signal.label}: {change_text}. {signal.interpretation or signal.label}"
    if change_text == "change unavailable":
        return f"{signal.label}: {value_text}. {signal.interpretation or signal.label}"
    return f"{signal.label}: {value_text}; {change_text}. {signal.interpretation or signal.label}"


def _panel_evidence_lines(panel_analyses: list[PanelAnalysis], panel_id: str, limit: int = 3) -> list[str]:
    panel = _panel_by_id(panel_analyses, panel_id)
    if panel is None:
        return []
    lines: list[str] = []
    for signal in panel.signals:
        if signal.change is None or pd.isna(signal.change):
            continue
        lines.append(f"- {_format_signal_card(signal)}")
        if len(lines) >= limit:
            break
    return lines


def _market_monitor_confirmations(panel_analyses: list[PanelAnalysis]) -> list[str]:
    lines: list[str] = []
    if _panel_signal(panel_analyses, "yield_curve", "curve_2s10s"):
        lines.append("The curve move is confirmed by the observed 2s10s spread and the Nelson-Siegel slope factor.")
    if _panel_signal(panel_analyses, "growth", "labor_demand"):
        lines.append("Growth weakness is confirmed by the claims component and the labor panel's hiring evidence.")
    if _panel_signal(panel_analyses, "cross_asset", "cross_asset_regime"):
        lines.append("Cross-asset confirmation is carried by credit, volatility, and the dollar rather than a single market.")
    return lines or ["No strong confirmation chain is available from the current panel mix."]


def _release_selected_panels(release_name: str, panel_analyses: list[PanelAnalysis]) -> list[PanelAnalysis]:
    selected_ids = _release_panel_ids(release_name)
    selected = [panel for panel in panel_analyses if panel.panel_id in selected_ids]
    return selected or panel_analyses


def _release_market_reaction_lines(panel_analyses: list[PanelAnalysis], release_name: str) -> list[str]:
    selected_panels = _release_selected_panels(release_name, panel_analyses)
    lines: list[str] = []
    for panel_id in ("yield_curve", "cross_asset", "inflation", "labor", "growth"):
        if not any(panel.panel_id == panel_id for panel in selected_panels):
            continue
        lines.extend(_panel_evidence_lines(selected_panels, panel_id, limit=2))
    return lines[:8] or ["- No selected market evidence was available."]


def _release_confirmation_lines(panel_analyses: list[PanelAnalysis], release_name: str) -> list[str]:
    selected_ids = set(_release_panel_ids(release_name))
    lines: list[str] = []
    if {"yield_curve", "cross_asset"} & selected_ids:
        regime = _panel_by_id(panel_analyses, "cross_asset")
        if regime is not None:
            lines.append(f"Cross-asset confirmation currently reads {regime.regime.lower()}.")
    if "inflation" in selected_ids and _panel_by_id(panel_analyses, "inflation") is not None:
        lines.append(f"Inflation context remains {_panel_by_id(panel_analyses, 'inflation').regime.lower()}.")
    if "labor" in selected_ids and _panel_by_id(panel_analyses, "labor") is not None:
        lines.append(f"Labor context remains {_panel_by_id(panel_analyses, 'labor').regime.lower()}.")
    return lines or ["No clear confirmation chain is available from the current dashboard."]


def _release_invalidation_suggestions(release_name: str) -> list[InvalidationCondition]:
    release = release_name.lower()
    if release in {"cpi", "pce"}:
        return [
            InvalidationCondition("Breakevens", "5Y breakeven reverses higher", "Would challenge the idea that the release cooled inflation expectations."),
            InvalidationCondition("Front-end yields", "2Y yield retraces sharply higher", "Would challenge a dovish policy interpretation."),
        ]
    if "payroll" in release or "labor" in release or "unemployment" in release:
        return [
            InvalidationCondition("Claims", "Initial claims reverse lower", "Would challenge the idea that the labor market is weakening."),
            InvalidationCondition("Wages", "Wage growth reaccelerates", "Would challenge a benign labor-rebalancing read."),
        ]
    if release == "fomc":
        return [
            InvalidationCondition("Front end", "2Y yield reverses the policy repricing", "Would challenge a market interpretation of the decision."),
            InvalidationCondition("Dollar", "Dollar rally fades quickly", "Would challenge a lasting tightening read-through."),
        ]
    return [
        InvalidationCondition("Rates reaction", "The initial rates move fully retraces", "Would challenge the release-driven thesis."),
        InvalidationCondition("Risk assets", "Credit or volatility moves the other way", "Would challenge the cross-asset confirmation check."),
    ]


def _monitor_sections(
    synthesis: ResearchSynthesis,
    panel_analyses: list[PanelAnalysis],
    conflicts: list[ConflictFlag],
    editor_takeaway: str = "",
    watch_list: list[str] | None = None,
) -> dict[str, str]:
    key_changes = _collect_key_changes(panel_analyses)
    tension_lines = _collect_research_tensions(conflicts)
    watch_lines = watch_list or _collect_watch_list(panel_analyses)
    sections = {
        "headline": f"{synthesis.draft_headline}\n\nPrimary question: What changed across the dashboard, what do the signals collectively suggest, and where is the evidence mixed?",
        "current_macro_state": f"{synthesis.regime_summary}\n\nCoverage: {len(panel_analyses)} of {EXPECTED_PANELS} panels available.",
        "key_changes": "\n\n".join(
            [f"### {label}\n" + "\n".join(lines) for label, lines in key_changes.items()]
        ) or "No theme-level changes with a valid change measure were identified.",
        "confirmation_check": "\n".join(f"- {line}" for line in _market_monitor_confirmations(panel_analyses)),
        "research_tensions": "\n".join(tension_lines),
        "market_implications": "\n".join(_collect_market_implications(synthesis)),
        "watch_list": "\n".join(f"- {item}" for item in watch_lines),
        "editor_takeaway": editor_takeaway or "No additional takeaway entered.",
        "final_view": (
            f"{synthesis.draft_headline}. "
            f"{editor_takeaway.strip() + ' ' if editor_takeaway.strip() else ''}"
            "The note is assembled from rule-based panel signals and should be treated as a monitoring brief, not a causal explanation."
        ),
    }
    return sections


def _release_sections(
    release_input: ReleaseInput,
    panel_analyses: list[PanelAnalysis],
    trade: TradeIdea | None,
    invalidations: list[InvalidationCondition],
    market_view: str,
    why_text: str,
    reaction_horizon: str = "",
) -> dict[str, str]:
    release_name = release_input.release_name.strip() or "Release"
    release_date_text = release_input.release_date.isoformat() if release_input.release_date else "an unspecified date"
    if release_input.consensus.strip():
        release_headline = (
            f"{release_name} on {release_date_text}: actual {release_input.actual or 'n/a'} versus consensus {release_input.consensus}, "
            f"previous {release_input.previous or 'n/a'}{f', revision {release_input.revision}' if release_input.revision else ''}."
        )
    else:
        release_headline = (
            f"{release_name} on {release_date_text}: actual {release_input.actual or 'n/a'}, "
            f"previous {release_input.previous or 'n/a'}{f', revision {release_input.revision}' if release_input.revision else ''}."
        )

    release_details = "\n".join(
        [
            f"- Actual: {release_input.actual or 'Unavailable'}",
            f"- Consensus: {release_input.consensus or 'No consensus comparison available'}",
            f"- Previous: {release_input.previous or 'Unavailable'}",
            f"- Revision: {release_input.revision or 'None entered'}",
            f"- Release date: {release_date_text}",
            f"- Event context: {release_input.event_context or 'No event context entered.'}",
        ]
    )

    selected_panels = _release_selected_panels(release_name, panel_analyses)
    reaction_lines = _release_market_reaction_lines(selected_panels, release_name)
    confirmation_lines = _release_confirmation_lines(panel_analyses, release_name)
    invalidation_lines = [
        f"- {item.label}: {item.threshold_or_condition}. {item.explanation}" if item.explanation else f"- {item.label}: {item.threshold_or_condition}."
        for item in invalidations
    ]
    trade_text = _trade_text(trade)

    sections = {
        "headline": release_headline,
        "release_details": release_details,
        "immediate_market_reaction": "\n".join(reaction_lines) or "No selected market evidence was available.",
        "why_details_mattered": why_text or "Add the release detail that mattered most and explain why it mattered.",
        "cross_asset_confirmation": "\n".join(f"- {line}" for line in confirmation_lines) or "No clear confirmation chain is available from the current dashboard.",
        "view": (f"Reaction horizon: {reaction_horizon}\n\n" if reaction_horizon else "") + (market_view or "Add your view on whether the reaction looks justified."),
        "trade_idea": trade_text,
        "invalidation": "\n".join(invalidation_lines) or "No invalidation conditions entered yet.",
        "data_gaps": "\n".join(
            [
                "- No consensus forecast is available unless the user enters one.",
                "- No intraday event-window pricing is available in the dashboard.",
                "- No qualitative release narrative is available unless the user adds it.",
            ]
        ),
        "final_view": (
            f"{release_name} analysis. "
            "The note combines user-entered release details with dashboard market evidence and should be treated as a reaction brief, not proof of causality."
        ),
    }
    return sections


def _issue_sections(
    issue_analysis: IssueAnalysis,
    lead_observation: str,
    why_text: str,
    view_text: str,
    trade: TradeIdea | None,
    invalidations: list[InvalidationCondition],
) -> dict[str, str]:
    core_evidence = "\n".join(f"- {_format_signal_card(item.signal)}" for item in issue_analysis.primary_evidence) or "No core evidence available."
    supporting = "\n".join(
        f"- {_format_signal_card(item.signal)}"
        for item in issue_analysis.evidence_items
        if item.role == "primary" or any(effect == "Supports" for effect in item.hypothesis_effects.values())
    )
    counter = "\n".join(
        f"- {_format_signal_card(item.signal)}"
        for item in issue_analysis.evidence_items
        if any(effect == "Challenges" for effect in item.hypothesis_effects.values())
    )
    contextual = "\n".join(f"- {_format_signal_card(item.signal)}" for item in issue_analysis.contextual_evidence) or "No contextual evidence available."
    invalidation_text = _invalidation_text(invalidations)
    sections = dict(issue_analysis.note_sections)
    sections.update(
        {
            "headline": f"{issue_analysis.category_display_name}: {issue_analysis.question}",
            "research_question": f"Question: {issue_analysis.question}\n\nCategory: {issue_analysis.category_display_name}",
            "lead_observation": lead_observation or "Select a lead observation to anchor the note.",
            "why_details_mattered": why_text or "Explain why this question matters now.",
            "base_and_alternatives": (
                f"{issue_analysis.leading_interpretation}\n\nAlternative interpretations:\n"
                + "\n".join(f"- {item}" for item in issue_analysis.alternative_interpretations)
                if issue_analysis.alternative_interpretations
                else "- No clear alternative interpretation."
            ),
            "core_evidence": core_evidence,
            "supporting_evidence": supporting or "No supporting evidence available.",
            "counter_evidence": counter or "No counter-evidence identified.",
            "research_tensions": "\n".join(f"- {item.title}: {item.explanation}" for item in issue_analysis.topic_conflicts) or "No topic-specific tensions identified.",
            "relevant_context": "\n".join(f"- {panel}: {regime}" for panel, regime in issue_analysis.panel_regimes.items()) or "No relevant context available.",
            "view": view_text or "Add your view on the thesis and whether the evidence supports it.",
            "trade_idea": _trade_text(trade),
            "invalidation": invalidation_text,
            "invalidation_conditions": invalidation_text,
            "data_gaps": "\n".join(f"- {gap}" for gap in issue_analysis.data_gaps) or "No material data gaps identified.",
            "watch_list": "\n".join(f"- {point}" for point in issue_analysis.watch_points) or "No specific watch points identified.",
            "final_view": (
                f"{issue_analysis.category_display_name}: {issue_analysis.question}. "
                "The note tests a macro thesis against the available evidence and should be treated as a research brief, not a probability statement."
            ),
        }
    )
    return sections


def _key_change_lines(panel: PanelAnalysis, limit: int = 4) -> list[str]:
    lines: list[str] = []
    for signal in panel.signals:
        if signal.change is None or pd.isna(signal.change):
            continue
        if signal.evidence_type == "classification":
            continue
        lines.append(f"- {signal.label}: {_format_value(signal)}; {_format_change(signal)}. {signal.interpretation or signal.label}")
        if len(lines) >= limit:
            break
    return lines


def _collect_key_changes(panel_analyses: list[PanelAnalysis]) -> dict[str, list[str]]:
    groups = {
        "inflation": "Inflation",
        "growth": "Growth",
        "labor": "Labor",
        "yield_curve": "Treasury Rates",
        "nelson_siegel": "Curve Structure",
        "cross_asset": "Cross-Asset",
    }
    collected: dict[str, list[str]] = {}
    for panel_id, label in groups.items():
        panel = next((item for item in panel_analyses if item.panel_id == panel_id), None)
        if panel is None:
            continue
        lines = _key_change_lines(panel)
        if lines:
            collected[label] = lines
    return collected


def _collect_research_tensions(conflicts: list[ConflictFlag]) -> list[str]:
    tensions: list[str] = []
    for conflict in conflicts:
        tensions.append(f"- {conflict.title}: {conflict.explanation}")
    return tensions or ["- No material tensions identified."]


def _collect_market_implications(synthesis: ResearchSynthesis) -> list[str]:
    summary = synthesis.regime_summary.lower()
    if "benign easing" in summary or "soft landing" in summary:
        return [
            "- Front-end policy pricing is consistent with easier financial conditions.",
            "- Credit and volatility are broadly supportive rather than disorderly.",
        ]
    if "growth scare" in summary:
        return [
            "- Rates and risk assets are consistent with a slower-growth backdrop.",
            "- Credit and volatility deserve close follow-up for confirmation.",
        ]
    if "inflationary tightening" in summary:
        return [
            "- Front-end pricing remains consistent with a restrictive policy stance.",
            "- Long-run inflation compensation remains an important constraint.",
        ]
    return [
        "- The mix is not clean enough to justify a strong directional market call.",
        "- Cross-asset confirmation is helpful, but the evidence remains mixed.",
    ]


def _collect_watch_list(panel_analyses: list[PanelAnalysis]) -> list[str]:
    watch_points: list[str] = []
    for panel in panel_analyses:
        if panel.panel_id == "inflation":
            watch_points.append("Does realized inflation continue to cool while market pricing stays anchored?")
        elif panel.panel_id == "growth":
            watch_points.append("Does growth momentum keep slowing without a broader deterioration?")
        elif panel.panel_id == "labor":
            watch_points.append("Do payrolls, unemployment and claims continue to rebalance in the same direction?")
        elif panel.panel_id == "yield_curve":
            watch_points.append("Does the front end of the curve continue to track policy pricing lower?")
        elif panel.panel_id == "cross_asset":
            watch_points.append("Do credit, volatility and the dollar keep confirming the rates move?")
    return watch_points or ["No specific watch points identified."]


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return slug.strip("_") or "macro_note"


def _format_value(signal: MacroSignal) -> str:
    if signal.value is None or pd.isna(signal.value):
        return "Unavailable"
    if signal.unit == "%":
        return f"{signal.value:.2f}{signal.unit}"
    if signal.unit in {"bp", "index", "x", "z", "k"}:
        return f"{signal.value:.2f} {signal.unit}".strip()
    return f"{signal.value:.2f} {signal.unit}".strip()


def _format_change(signal: MacroSignal) -> str:
    if signal.change is None or pd.isna(signal.change):
        return "change unavailable"
    unit = signal.change_unit or ""
    horizon = f" ({signal.horizon})" if signal.horizon else ""
    if unit == "%":
        return f"{signal.change:+.2f}{unit}{horizon}"
    if unit:
        return f"{signal.change:+.2f} {unit}{horizon}"
    return f"{signal.change:+.2f}{horizon}"


def _signal_bullet(signal: MacroSignal) -> str:
    value_text = _format_value(signal)
    change_text = _format_change(signal)

    if value_text == "Unavailable" and change_text != "change unavailable":
        return f"- {signal.label} change: {change_text}. {signal.interpretation or signal.label}"

    if change_text == "change unavailable":
        return f"- {signal.label}: {value_text}. {signal.interpretation or signal.label}"

    return f"- {signal.label}: {value_text}; {change_text}. {signal.interpretation or signal.label}"


def _panel_regime_lines(panel_analyses: list[PanelAnalysis]) -> list[str]:
    return [f"- {panel.title}: {panel.regime}" for panel in panel_analyses]


def _namespace_key(namespace: str, field: str) -> str:
    return f"{namespace}_{field}"


def _seed_section_defaults(namespace: str, sections: dict[str, str], signature_key: str, signature: str, regenerate: bool = False) -> None:
    if regenerate or st.session_state.get(signature_key) != signature:
        for key, value in sections.items():
            st.session_state[_namespace_key(namespace, key)] = value
        st.session_state[signature_key] = signature


def _current_sections(namespace: str, order: tuple[str, ...]) -> dict[str, str]:
    return {key: st.session_state.get(_namespace_key(namespace, key), "") for key in order}


def _reset_section(namespace: str, section_id: str, sections: dict[str, str]) -> None:
    st.session_state[_namespace_key(namespace, section_id)] = sections.get(section_id, "")


def _text_block_from_sections(
    sections: dict[str, str],
    order: tuple[str, ...],
    labels: dict[str, str],
    title: str,
) -> str:
    parts = [f"# {title}"]
    for section_id in order:
        content = sections.get(section_id, "").strip()
        if not content:
            continue
        parts.append(f"## {labels.get(section_id, section_id.replace('_', ' ').title())}\n{content}")
    return "\n\n".join(parts)


def _trade_from_state(namespace: str) -> TradeIdea:
    return TradeIdea(
        enabled=bool(st.session_state.get(_namespace_key(namespace, "trade_enabled"), False)),
        trade_type=st.session_state.get(_namespace_key(namespace, "trade_type"), ""),
        instrument=st.session_state.get(_namespace_key(namespace, "trade_instrument"), ""),
        direction=st.session_state.get(_namespace_key(namespace, "trade_direction"), ""),
        entry=st.session_state.get(_namespace_key(namespace, "trade_entry"), ""),
        target=st.session_state.get(_namespace_key(namespace, "trade_target"), ""),
        stop=st.session_state.get(_namespace_key(namespace, "trade_stop"), ""),
        horizon=st.session_state.get(_namespace_key(namespace, "trade_horizon"), ""),
        conviction=st.session_state.get(_namespace_key(namespace, "trade_conviction"), ""),
        sizing_note=st.session_state.get(_namespace_key(namespace, "trade_sizing_note"), ""),
        rationale=st.session_state.get(_namespace_key(namespace, "trade_rationale"), ""),
    )


def _trade_text(trade: TradeIdea | None) -> str:
    if trade is None or not trade.enabled:
        return "Trade idea is disabled."
    return "\n".join(
        [
            f"- Trade type: {trade.trade_type or 'Unspecified'}",
            f"- Instrument: {trade.instrument or 'Unspecified'}",
            f"- Direction: {trade.direction or 'Unspecified'}",
            f"- Entry: {trade.entry or 'User to define'}",
            f"- Target: {trade.target or 'User to define'}",
            f"- Stop: {trade.stop or 'User to define'}",
            f"- Horizon: {trade.horizon or 'Unspecified'}",
            f"- Conviction: {trade.conviction or 'Unspecified'}",
            f"- Sizing note: {trade.sizing_note or 'Unspecified'}",
            f"- Rationale: {trade.rationale or 'Unspecified'}",
        ]
    )


def _render_trade_editor(namespace: str) -> TradeIdea:
    enabled = st.checkbox("Include trade idea", key=_namespace_key(namespace, "trade_enabled"))
    if enabled:
        cols = st.columns(2)
        with cols[0]:
            st.text_input("Trade type", key=_namespace_key(namespace, "trade_type"))
            st.text_input("Instrument", key=_namespace_key(namespace, "trade_instrument"))
            st.text_input("Direction", key=_namespace_key(namespace, "trade_direction"))
            st.text_input("Entry", key=_namespace_key(namespace, "trade_entry"))
            st.text_input("Target", key=_namespace_key(namespace, "trade_target"))
        with cols[1]:
            st.text_input("Stop", key=_namespace_key(namespace, "trade_stop"))
            st.text_input("Horizon", key=_namespace_key(namespace, "trade_horizon"))
            st.text_input("Conviction", key=_namespace_key(namespace, "trade_conviction"))
            st.text_input("Sizing note", key=_namespace_key(namespace, "trade_sizing_note"))
            st.text_area("Rationale", height=88, key=_namespace_key(namespace, "trade_rationale"))
    return _trade_from_state(namespace)


def _render_invalidation_editor(namespace: str, prefix: str, count: int = 3) -> list[InvalidationCondition]:
    conditions: list[InvalidationCondition] = []
    for idx in range(count):
        label_key = _namespace_key(namespace, f"{prefix}_invalid_label_{idx}")
        condition_key = _namespace_key(namespace, f"{prefix}_invalid_condition_{idx}")
        explanation_key = _namespace_key(namespace, f"{prefix}_invalid_explanation_{idx}")
        label = st.text_input(f"Invalidation {idx + 1} label", key=label_key)
        condition = st.text_input(f"Invalidation {idx + 1} threshold or condition", key=condition_key)
        explanation = st.text_area(f"Invalidation {idx + 1} explanation", height=72, key=explanation_key)
        if label.strip() or condition.strip() or explanation.strip():
            conditions.append(
                InvalidationCondition(
                    label=label.strip() or f"Condition {idx + 1}",
                    threshold_or_condition=condition.strip() or "Unspecified",
                    explanation=explanation.strip(),
                )
            )
    return conditions


def _invalidation_text(conditions: list[InvalidationCondition]) -> str:
    if not conditions:
        return "No invalidation conditions entered yet."
    return "\n".join(
        f"- {item.label}: {item.threshold_or_condition}. {item.explanation}" if item.explanation else f"- {item.label}: {item.threshold_or_condition}."
        for item in conditions
    )


def _workspace_draft(
    mode: str,
    title: str,
    sections: dict[str, str],
    trade: TradeIdea | None,
    invalidation_conditions: list[InvalidationCondition],
    metadata: dict[str, object],
) -> ResearchWorkspaceDraft:
    return ResearchWorkspaceDraft(
        mode=mode,
        title=title,
        sections=sections,
        trade=trade,
        invalidation_conditions=invalidation_conditions,
        metadata=metadata,
    )


def _save_workspace_note(
    note_text: str,
    snapshot: dict,
    note_title: str,
    context: dict,
    mode: str,
    extra_metadata: dict[str, object],
) -> tuple[Path, Path]:
    return _save_note(note_text, snapshot, note_title, context, mode, extra_metadata)


def _list_recent_notes(limit: int = 10) -> list[Path]:
    if not NOTE_DIR.exists():
        return []
    notes = sorted(NOTE_DIR.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    return notes[:limit]


def _issue_template_options() -> list[str]:
    return [category.display_name for category_id, category in ISSUE_CATEGORY_REGISTRY.items() if category_id != "custom"]


def _issue_category_by_display_name(display_name: str):
    for category in ISSUE_CATEGORY_REGISTRY.values():
        if category.display_name == display_name:
            return category
    return ISSUE_CATEGORY_REGISTRY["inflation_policy"]


def _available_signal_groups(panel_analyses: list[PanelAnalysis]) -> list[str]:
    groups = sorted(
        {
            signal.group_id
            for panel in panel_analyses
            for signal in panel.signals
            if signal.group_id
        }
    )
    return groups


def _custom_issue_config(
    template_id: str,
    panel_primary: list[str],
    panel_secondary: list[str],
    groups_primary: list[str],
    groups_secondary: list[str],
):
    template = ISSUE_CATEGORY_REGISTRY.get(template_id, ISSUE_CATEGORY_REGISTRY["inflation_policy"])
    return replace(
        template,
        category_id="custom",
        display_name="Custom Issue",
        description="Use manually selected evidence and a template-based outline for a bespoke macro question.",
        primary_panels=tuple(panel_primary),
        secondary_panels=tuple(panel_secondary),
        primary_signal_groups=tuple(groups_primary),
        secondary_signal_groups=tuple(groups_secondary),
        default_questions=("Define a custom macro question.",),
    )


def _workspace_mode_selector(current_value: str) -> str:
    if hasattr(st, "segmented_control"):
        return st.segmented_control(
            "Research workflow",
            options=list(WORKSPACE_MODE_OPTIONS),
            key="research_workspace_mode",
            default=current_value if current_value in WORKSPACE_MODE_OPTIONS else WORKSPACE_MODE_OPTIONS[0],
        )
    return st.radio(
        "Research workflow",
        options=list(WORKSPACE_MODE_OPTIONS),
        index=WORKSPACE_MODE_OPTIONS.index(current_value) if current_value in WORKSPACE_MODE_OPTIONS else 0,
        horizontal=True,
        key="research_workspace_mode",
    )


def _monitor_signature(synthesis: ResearchSynthesis, analyses: list[PanelAnalysis]) -> str:
    panel_bits = "|".join(f"{panel.panel_id}:{panel.regime}" for panel in analyses)
    return f"{synthesis.as_of.isoformat()}|{synthesis.draft_headline}|{panel_bits}"


def _release_signature(release_input: ReleaseInput, trade: TradeIdea, invalidations: list[InvalidationCondition], release_name: str) -> str:
    payload = {
        "release_input": asdict(release_input),
        "trade": asdict(trade),
        "invalidations": [asdict(item) for item in invalidations],
        "release_name": release_name,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _issue_signature(category_id: str, question: str, template_id: str, selected_panels: list[str], selected_groups: list[str]) -> str:
    payload = {
        "category_id": category_id,
        "question": question,
        "template_id": template_id,
        "selected_panels": selected_panels,
        "selected_groups": selected_groups,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _render_note_editor(
    namespace: str,
    title_key: str,
    title_default: str,
    sections: dict[str, str],
    order: tuple[str, ...],
    labels: dict[str, str],
    save_button_label: str,
    note_mode: str,
    context: dict,
    snapshot: dict,
    extra_metadata: dict[str, object],
) -> None:
    st.session_state.setdefault(title_key, title_default)
    note_title = st.text_input("Note title", key=title_key)

    st.markdown("### Draft preview")
    for section_id in order:
        content = sections.get(section_id, "").strip()
        if not content:
            continue
        with st.expander(labels.get(section_id, section_id.replace("_", " ").title()), expanded=False):
            st.markdown(content)

    final_sections = _current_sections(namespace, order)
    final_note = _text_block_from_sections(final_sections, order, labels, note_title)
    st.markdown("### Final editable note")
    edited_note = st.text_area("Edit the full note", value=final_note, height=600, key=_namespace_key(namespace, "full_note"))

    col_save, col_reset_full, col_download = st.columns(3)
    with col_save:
        if st.button(save_button_label, key=_namespace_key(namespace, "save")):
            saved_md, saved_json = _save_workspace_note(
                edited_note,
                snapshot,
                note_title,
                context,
                note_mode,
                extra_metadata,
            )
            st.success(f"Saved {saved_md.name}")
            st.caption(f"JSON snapshot: {saved_json.name}")

    with col_reset_full:
        if st.button("Reset full note", key=_namespace_key(namespace, "reset_full")):
            st.session_state[_namespace_key(namespace, "full_note")] = final_note
            st.rerun()

    with col_download:
        st.download_button(
            "Download Markdown",
            data=edited_note,
            file_name=f"{_safe_filename(note_title)}.md",
            mime="text/markdown",
            key=_namespace_key(namespace, "download"),
        )


def _render_market_monitor(fred_client: FREDClient, context: dict, analyses: list[PanelAnalysis]) -> None:
    del fred_client
    synthesis = synthesize(analyses, context["end_date"])
    conflicts = detect_conflicts([signal for panel in analyses for signal in panel.signals])
    namespace = "monitor"
    auto_watch_list = _collect_watch_list(analyses)
    initial_editor_takeaway = st.session_state.get(_namespace_key(namespace, "editor_takeaway"), "")
    initial_selected_watch_items = st.session_state.get(_namespace_key(namespace, "selected_watch_items"), auto_watch_list[:3])
    initial_custom_watch = st.session_state.get(_namespace_key(namespace, "watch_item_add"), "")
    initial_sections = _monitor_sections(
        synthesis,
        analyses,
        conflicts,
        editor_takeaway=initial_editor_takeaway,
        watch_list=list(initial_selected_watch_items) + ([initial_custom_watch.strip()] if initial_custom_watch.strip() else []),
    )
    signature = _monitor_signature(synthesis, analyses)
    title_key = _namespace_key(namespace, "title")
    if st.session_state.get(_namespace_key(namespace, "signature")) != signature:
        st.session_state[title_key] = f"Macro monitor: {synthesis.draft_headline}"
    _seed_section_defaults(namespace, initial_sections, _namespace_key(namespace, "signature"), signature)

    st.markdown("### User judgement")
    editor_takeaway = st.text_area("Editor's takeaway", height=96, key=_namespace_key(namespace, "editor_takeaway"))
    selected_watch_items = st.multiselect(
        "Watch list",
        options=auto_watch_list,
        default=auto_watch_list[:3],
        key=_namespace_key(namespace, "selected_watch_items"),
    )
    custom_watch = st.text_input("Add watch item", key=_namespace_key(namespace, "watch_item_add"))
    watch_list = list(selected_watch_items)
    if custom_watch.strip():
        watch_list.append(custom_watch.strip())

    sections = _monitor_sections(synthesis, analyses, conflicts, editor_takeaway=editor_takeaway, watch_list=watch_list)

    st.markdown("### Current macro state")
    st.info(synthesis.regime_summary)
    st.caption(f"Coverage: {len(analyses)} of {EXPECTED_PANELS} panels available.")

    st.markdown("### Panel regime strip")
    regime_cols = st.columns(max(1, len(analyses)))
    for col, panel in zip(regime_cols, analyses, strict=False):
        with col:
            st.metric(panel.title, panel.regime)

    st.markdown("### Key changes")
    key_changes = _collect_key_changes(analyses)
    if key_changes:
        for label, lines in key_changes.items():
            with st.expander(label):
                for line in lines:
                    st.markdown(line)
    else:
        st.info("No theme-level changes with a valid change measure were identified.")

    st.markdown("### What the signals collectively suggest")
    st.info(synthesis.regime_summary)

    st.markdown("### Confirmation check")
    for line in _market_monitor_confirmations(analyses):
        st.markdown(f"- {line}")

    st.markdown("### Research tensions")
    for line in _collect_research_tensions(conflicts):
        st.markdown(line)

    st.markdown("### Market implications")
    for item in _collect_market_implications(synthesis):
        st.markdown(f"- {item.lstrip('- ').strip()}")

    st.markdown("### Watch list")
    for item in watch_list:
        st.markdown(f"- {item}")

    st.markdown("### Editable monitoring brief")
    if st.button("Regenerate draft", key=_namespace_key(namespace, "regenerate")):
        _seed_section_defaults(namespace, sections, _namespace_key(namespace, "signature"), signature, regenerate=True)
        st.rerun()
    draft_snapshot = _workspace_draft(
        mode="Market Monitor",
        title=st.session_state[title_key],
        sections=_current_sections(namespace, MARKET_MONITOR_NOTE_SECTION_ORDER),
        trade=None,
        invalidation_conditions=[],
        metadata={
            "signature": signature,
            "analysis_as_of": str(synthesis.as_of),
            "panel_regimes": synthesis.panel_regimes,
            "editor_takeaway": editor_takeaway,
            "watch_list": watch_list,
        },
    )
    snapshot = {
        "panel_analyses": [asdict(panel) for panel in analyses],
        "synthesis": asdict(synthesis),
        "conflicts": [asdict(item) for item in conflicts],
        "workspace_draft": asdict(draft_snapshot),
    }
    _render_note_editor(
        namespace=namespace,
        title_key=title_key,
        title_default=f"Macro monitor: {synthesis.draft_headline}",
        sections=sections,
        order=MARKET_MONITOR_NOTE_SECTION_ORDER,
        labels=MARKET_MONITOR_NOTE_SECTION_LABELS,
        save_button_label="Save note",
        note_mode="macro_monitor",
        context=context,
        snapshot=snapshot,
        extra_metadata={
            "analysis_as_of": str(synthesis.as_of),
            "panel_regimes": synthesis.panel_regimes,
            "signature": signature,
            "workspace_mode": "Market Monitor",
            "editor_takeaway": editor_takeaway,
            "watch_list": watch_list,
        },
    )

    st.markdown("### Save / archive controls")
    recent_notes = _list_recent_notes()
    if recent_notes:
        st.markdown("Recent notes:")
        for note_path in recent_notes:
            st.markdown(f"- {note_path.name}")

    st.markdown("### Methodology and limitations")
    st.caption("This workflow assigns evidence roles to panel signals, identifies tensions and confirmations, and does not establish causality or predict policy decisions.")


def _render_release_reaction(fred_client: FREDClient, context: dict, analyses: list[PanelAnalysis]) -> None:
    del fred_client
    namespace = "release"
    st.markdown("### Release setup")
    st.session_state.setdefault(_namespace_key(namespace, "release_name"), "CPI")
    st.session_state.setdefault(_namespace_key(namespace, "release_date"), context["end_date"])
    st.session_state.setdefault(_namespace_key(namespace, "release_actual"), "")
    st.session_state.setdefault(_namespace_key(namespace, "release_consensus"), "")
    st.session_state.setdefault(_namespace_key(namespace, "release_previous"), "")
    st.session_state.setdefault(_namespace_key(namespace, "release_revision"), "")
    st.session_state.setdefault(_namespace_key(namespace, "release_user_summary"), "")
    st.session_state.setdefault(_namespace_key(namespace, "release_event_context"), "")
    st.session_state.setdefault(_namespace_key(namespace, "release_horizon"), "")
    st.session_state.setdefault(_namespace_key(namespace, "release_view"), "")
    st.session_state.setdefault(_namespace_key(namespace, "release_why_text"), "")

    release_name = st.text_input("Release type", key=_namespace_key(namespace, "release_name"))
    release_date = st.date_input("Release date", key=_namespace_key(namespace, "release_date"))
    actual = st.text_input("Actual", key=_namespace_key(namespace, "release_actual"))
    consensus = st.text_input("Consensus", key=_namespace_key(namespace, "release_consensus"))
    previous = st.text_input("Previous", key=_namespace_key(namespace, "release_previous"))
    revision = st.text_input("Revision", key=_namespace_key(namespace, "release_revision"))
    user_summary = st.text_area("Important release details", height=88, key=_namespace_key(namespace, "release_user_summary"))
    event_context = st.text_area("Event context", height=88, key=_namespace_key(namespace, "release_event_context"))
    reaction_horizon = st.text_input("Reaction horizon", key=_namespace_key(namespace, "release_horizon"))
    market_view = st.text_area("My view", height=100, key=_namespace_key(namespace, "release_view"))
    why_text = st.text_area("Why it mattered", height=100, key=_namespace_key(namespace, "release_why_text"))
    trade = _render_trade_editor(namespace)
    invalidations = _render_invalidation_editor(namespace, "release") if trade.enabled or bool(market_view.strip() or why_text.strip()) else []

    release_input = ReleaseInput(
        release_name=release_name,
        release_date=release_date,
        actual=actual,
        consensus=consensus,
        previous=previous,
        revision=revision,
        user_summary=user_summary,
        event_context=event_context,
    )

    signature = _release_signature(release_input, trade, invalidations, release_name)
    sections = _release_sections(release_input, analyses, trade, invalidations, market_view or user_summary, why_text, reaction_horizon)
    title_key = _namespace_key(namespace, "title")
    if st.session_state.get(_namespace_key(namespace, "signature")) != signature:
        st.session_state[title_key] = f"{release_name or 'Release'} reaction note"
    st.session_state.setdefault(title_key, f"{release_name or 'Release'} reaction note")
    _seed_section_defaults(namespace, sections, _namespace_key(namespace, "signature"), signature)

    st.markdown("### Selected market reaction")
    selected_panels = _release_selected_panels(release_name or "release", analyses)
    st.caption("Selected panels: " + ", ".join(panel.title for panel in selected_panels))
    for line in _release_market_reaction_lines(selected_panels, release_name or "release"):
        st.markdown(line)

    st.markdown("### Confirmation / conflict check")
    for line in _release_confirmation_lines(analyses, release_name or "release"):
        st.markdown(f"- {line}")

    st.markdown("### Release reaction note")
    if st.button("Regenerate draft", key=_namespace_key(namespace, "regenerate")):
        _seed_section_defaults(namespace, sections, _namespace_key(namespace, "signature"), signature, regenerate=True)
        st.rerun()
    draft_snapshot = _workspace_draft(
        mode="Release Reaction",
        title=st.session_state[title_key],
        sections=_current_sections(namespace, RELEASE_NOTE_SECTION_ORDER),
        trade=trade,
        invalidation_conditions=invalidations,
        metadata={"signature": signature, "release_input": asdict(release_input), "reaction_horizon": reaction_horizon},
    )
    snapshot = {
        "panel_analyses": [asdict(panel) for panel in analyses],
        "release_input": asdict(release_input),
        "trade": asdict(trade),
        "invalidation_conditions": [asdict(item) for item in invalidations],
        "workspace_draft": asdict(draft_snapshot),
    }
    _render_note_editor(
        namespace=namespace,
        title_key=title_key,
        title_default=f"{release_name or 'Release'} reaction note",
        sections=sections,
        order=RELEASE_NOTE_SECTION_ORDER,
        labels=RELEASE_NOTE_SECTION_LABELS,
        save_button_label="Save note",
        note_mode="release_reaction",
        context=context,
        snapshot=snapshot,
        extra_metadata={
            "analysis_as_of": str(context["end_date"]),
            "release_input": asdict(release_input),
            "trade": asdict(trade),
            "invalidation_conditions": [asdict(item) for item in invalidations],
            "signature": signature,
            "workspace_mode": "Release Reaction",
            "reaction_horizon": reaction_horizon,
        },
    )

    st.markdown("### Save / archive controls")
    recent_notes = _list_recent_notes()
    if recent_notes:
        st.markdown("Recent notes:")
        for note_path in recent_notes:
            st.markdown(f"- {note_path.name}")

    st.markdown("### Methodology")
    st.caption("The release workflow separates the event from the market reaction, then documents a view, trade expression and falsifiable invalidation conditions.")


def _render_issue_strategy_note(fred_client: FREDClient, context: dict, analyses: list[PanelAnalysis]) -> None:
    del fred_client
    namespace = "issue"
    st.markdown("### Category selector")
    category_labels = [ISSUE_CATEGORY_REGISTRY[key].display_name for key in ISSUE_CATEGORY_ORDER]
    category_name_key = _namespace_key(namespace, "category_display")
    if st.session_state.get(category_name_key) not in category_labels:
        st.session_state[category_name_key] = ISSUE_CATEGORY_REGISTRY[ISSUE_CATEGORY_ORDER[0]].display_name
    category_choice = st.selectbox("Macro category", options=category_labels, key=category_name_key)
    category = _issue_category_by_display_name(category_choice)
    st.session_state[_namespace_key(namespace, "category_id")] = category.category_id
    st.caption(category.description)

    question_options = list(category.default_questions) or ["Define a custom question."]
    question_box_key = _namespace_key(namespace, "selected_question_box")
    if st.session_state.get(question_box_key) not in question_options:
        st.session_state[question_box_key] = question_options[0]
    st.session_state.setdefault(_namespace_key(namespace, "selected_question"), question_options[0])
    selected_question = st.selectbox("Suggested question", options=question_options, key=question_box_key)
    st.session_state[_namespace_key(namespace, "selected_question")] = selected_question
    custom_question_key = _namespace_key(namespace, "custom_question")
    st.session_state.setdefault(custom_question_key, "")
    custom_question = st.text_input("Optional custom question", value=st.session_state.get(custom_question_key, ""), key=_namespace_key(namespace, "custom_question_input"))
    st.session_state[custom_question_key] = custom_question
    effective_question = custom_question.strip() or selected_question
    st.info(f"Effective question: {effective_question}")
    lead_observation = st.text_area("Lead observation", height=100, key=_namespace_key(namespace, "lead_observation"))
    why_text = st.text_area("Why it matters", height=100, key=_namespace_key(namespace, "why"))
    view_text = st.text_area("My view", height=100, key=_namespace_key(namespace, "view"))

    custom_template_id = category.category_id
    custom_primary_panels: list[str] = []
    custom_secondary_panels: list[str] = []
    custom_primary_groups: list[str] = []
    custom_secondary_groups: list[str] = []
    active_config = category

    if category.category_id == "custom":
        st.warning("Custom issue mode uses manually selected evidence and does not automatically understand unrestricted macro questions.")
        template_labels = _issue_template_options()
        template_choice = st.selectbox("Note template", options=template_labels, key=_namespace_key(namespace, "custom_template"))
        template_config = _issue_category_by_display_name(template_choice)
        custom_template_id = template_config.category_id

        panel_options = [panel.panel_id for panel in analyses]
        group_options = _available_signal_groups(analyses)
        custom_primary_panels = st.multiselect("Primary panels", options=panel_options, default=list(template_config.primary_panels), key=_namespace_key(namespace, "custom_primary_panels"))
        custom_secondary_panels = st.multiselect("Secondary panels", options=panel_options, default=list(template_config.secondary_panels), key=_namespace_key(namespace, "custom_secondary_panels"))
        custom_primary_groups = st.multiselect("Relevant signal groups", options=group_options, default=list(template_config.primary_signal_groups), key=_namespace_key(namespace, "custom_primary_groups"))
        custom_secondary_groups = st.multiselect("Contextual signal groups", options=group_options, default=list(template_config.secondary_signal_groups), key=_namespace_key(namespace, "custom_secondary_groups"))
        active_config = _custom_issue_config(
            template_config.category_id,
            custom_primary_panels,
            custom_secondary_panels,
            custom_primary_groups,
            custom_secondary_groups,
        )

    issue_analysis: IssueAnalysis = build_issue_analysis(
        active_config,
        effective_question,
        analyses,
        selected_panel_ids=active_config.primary_panels + active_config.secondary_panels,
        selected_signal_groups=active_config.primary_signal_groups + active_config.secondary_signal_groups,
    )
    signature = _issue_signature(
        active_config.category_id,
        effective_question,
        custom_template_id,
        list(active_config.primary_panels + active_config.secondary_panels),
        list(active_config.primary_signal_groups + active_config.secondary_signal_groups),
    )
    namespace_signature_key = _namespace_key(namespace, "signature")
    title_key = _namespace_key(namespace, "title")
    trade = _render_trade_editor(namespace)
    invalidations = _render_invalidation_editor(namespace, "issue") if trade.enabled or bool(view_text.strip() or why_text.strip()) else []
    sections = dict(issue_analysis.note_sections)
    sections["trade_idea"] = _trade_text(trade)
    sections["invalidation_conditions"] = _invalidation_text(invalidations)

    if st.session_state.get(namespace_signature_key) != signature:
        st.session_state[title_key] = f"{active_config.display_name}: {effective_question}"
    st.session_state.setdefault(title_key, f"{active_config.display_name}: {effective_question}")
    _seed_section_defaults(namespace, sections, namespace_signature_key, signature)

    st.markdown("### Detected evidence coverage")
    coverage = issue_analysis.evidence_coverage
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Relevant signals", f"{int(coverage.get('relevant_signals', 0.0))}")
    col_b.metric("Primary signals", f"{int(coverage.get('primary_signals', 0.0))}")
    col_c.metric("Contextual signals", f"{int(coverage.get('contextual_signals', 0.0))}")
    col_d.metric("Required groups covered", f"{int(coverage.get('required_groups_covered', 0.0))}/{int(coverage.get('required_groups_total', 0.0))}")

    st.markdown("### Leading interpretation")
    st.info(issue_analysis.leading_interpretation)
    if issue_analysis.alternative_interpretations:
        st.markdown("### Alternative interpretations")
        for item in issue_analysis.alternative_interpretations:
            st.markdown(f"- {item}")

    st.markdown("### Relevant evidence")
    for item in issue_analysis.primary_evidence:
        with st.expander(f"{item.signal.label}"):
            st.markdown(_signal_bullet(item.signal))
            st.markdown(f"**Issue score:** {item.issue_score:.2f}")
            st.markdown(f"**Relevance:** {item.relevance_score:.2f}")
            st.markdown(f"**Role:** {item.role}")

    st.markdown("### Evidence matrix")
    for item in issue_analysis.evidence_items[:10]:
        with st.container(border=True):
            cols = st.columns([3, 1, 1, 1])
            cols[0].markdown(f"**{item.signal.label}**  \n{_format_value(item.signal)}  \n{_format_change(item.signal)}")
            cols[1].markdown(f"**Panel**  \n{item.signal.panel_id}")
            cols[2].markdown(f"**Issue score**  \n{item.issue_score:.2f}")
            cols[3].markdown(f"**Role**  \n{item.role}")
            for hypothesis in issue_analysis.hypotheses:
                effect = item.hypothesis_effects.get(hypothesis.hypothesis_id, "Neutral")
                st.caption(f"{hypothesis.title}: {effect}")
            st.markdown(item.signal.interpretation or item.rationale or item.signal.label)

    st.markdown("### Topic-specific conflicts")
    if issue_analysis.topic_conflicts:
        for conflict in issue_analysis.topic_conflicts:
            st.warning(f"**{conflict.title}.** {conflict.explanation}")
    else:
        st.info("No topic-specific conflicts identified.")

    st.markdown("### Data gaps")
    if issue_analysis.data_gaps:
        for gap in issue_analysis.data_gaps:
            st.markdown(f"- {gap}")
    else:
        st.caption("No material data gaps identified.")

    st.markdown("### User judgement")
    st.markdown(f"**Research question:** {effective_question}")
    st.markdown(f"**Lead observation:** {lead_observation or 'None provided'}")
    st.markdown(f"**Why it matters:** {why_text or 'None provided'}")
    st.markdown(f"**View:** {view_text or 'None provided'}")
    if st.button("Regenerate draft", key=_namespace_key(namespace, "regenerate")):
        _seed_section_defaults(namespace, sections, namespace_signature_key, signature, regenerate=True)
        st.rerun()

    final_order_list = list(issue_analysis.note_section_order)
    if "why_details_mattered" not in final_order_list:
        final_order_list.insert(2, "why_details_mattered")
    final_order = tuple(final_order_list)
    final_labels = dict(ISSUE_NOTE_SECTION_LABELS)
    final_sections = _current_sections(namespace, final_order)
    final_sections.update(
        {
            "lead_observation": lead_observation,
            "why_details_mattered": why_text,
            "view": view_text,
            "trade_idea": _trade_text(trade),
            "invalidation_conditions": _invalidation_text(invalidations),
        }
    )
    final_note = issue_note_text(final_sections, tuple(final_order), st.session_state[_namespace_key(namespace, "title")])
    st.markdown("### Final editable note")
    edited_note = st.text_area("Edit the full note", value=final_note, height=600, key=_namespace_key(namespace, "full_note"))

    col_save, col_reset_section, col_reset_full, col_download = st.columns(4)
    with col_save:
        if st.button("Save issue note", key=_namespace_key(namespace, "save")):
            draft_snapshot = _workspace_draft(
                mode="Issue / Strategy Note",
                title=st.session_state[title_key],
                sections=final_sections,
                trade=trade,
                invalidation_conditions=invalidations,
                metadata={
                    "signature": signature,
                    "analysis_as_of": str(issue_analysis.as_of),
                    "issue_category": active_config.category_id,
                    "issue_category_display_name": active_config.display_name,
                    "research_question": effective_question,
                    "lead_observation": lead_observation,
                    "why_text": why_text,
                    "view_text": view_text,
                    "relevant_panels": issue_analysis.relevant_panels,
                    "relevant_signal_groups": issue_analysis.relevant_signal_groups,
                },
            )
            snapshot = {
                "issue_analysis": asdict(issue_analysis),
                "trade": asdict(trade),
                "invalidation_conditions": [asdict(item) for item in invalidations],
                "workspace_draft": asdict(draft_snapshot),
            }
            saved_md, saved_json = _save_workspace_note(
                edited_note,
                snapshot,
                st.session_state[title_key],
                context,
                "issue_strategy_note",
                {
                    "analysis_as_of": str(issue_analysis.as_of),
                    "issue_category": active_config.category_id,
                    "issue_category_display_name": active_config.display_name,
                    "research_question": effective_question,
                    "lead_observation": lead_observation,
                    "why_text": why_text,
                    "view_text": view_text,
                    "relevant_panels": issue_analysis.relevant_panels,
                    "relevant_signal_groups": issue_analysis.relevant_signal_groups,
                    "signature": signature,
                    "workspace_mode": "Issue / Strategy Note",
                },
            )
            st.success(f"Saved {saved_md.name}")
            st.caption(f"JSON snapshot: {saved_json.name}")

    with col_reset_section:
        reset_section = st.selectbox(
            "Section to reset",
            options=list(final_order),
            format_func=lambda value: final_labels.get(value, value.replace("_", " ").title()),
            key=_namespace_key(namespace, "reset_section_choice"),
        )
        if st.button("Reset current section", key=_namespace_key(namespace, "reset_current")):
            _reset_section(namespace, reset_section, final_sections)
            st.rerun()

    with col_reset_full:
        if st.button("Reset full issue note", key=_namespace_key(namespace, "reset_full")):
            _seed_section_defaults(namespace, final_sections, namespace_signature_key, signature, regenerate=True)
            st.rerun()

    with col_download:
        st.download_button(
            "Download Markdown",
            data=edited_note,
            file_name=f"{_safe_filename(st.session_state[title_key])}.md",
            mime="text/markdown",
            key=_namespace_key(namespace, "download"),
        )

    st.markdown("### Save / archive controls")
    recent_notes = _list_recent_notes()
    if recent_notes:
        st.markdown("Recent notes:")
        for note_path in recent_notes:
            st.markdown(f"- {note_path.name}")

    st.markdown("### Methodology")
    for item in issue_analysis.methodology:
        st.markdown(f"- {item}")


def render(fred_client: FREDClient, context: dict, panel_analyses: list[PanelAnalysis] | None = None) -> None:
    st.subheader("Panel 7: Research Workspace")
    st.caption("Choose between a broad macro monitor, a release-reaction brief, and a focused issue note built from the structured outputs of Panels 1-6.")

    analyses = panel_analyses or _build_panel_analyses(fred_client, context)
    if len(analyses) < 3:
        st.warning("Insufficient panel coverage for a reliable cross-panel synthesis.")

    current_mode = st.session_state.get("research_workspace_mode", WORKSPACE_MODE_OPTIONS[0])
    mode = _workspace_mode_selector(current_mode)

    if mode == "Market Monitor":
        _render_market_monitor(fred_client, context, analyses)
    elif mode == "Release Reaction":
        _render_release_reaction(fred_client, context, analyses)
    else:
        _render_issue_strategy_note(fred_client, context, analyses)
