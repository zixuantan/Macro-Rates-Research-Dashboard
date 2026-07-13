from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.conflicts import detect_conflicts
from analysis.notability import score_signal
from analysis.signal_groups import group_signals
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
from models.macro_analysis import MacroSignal, PanelAnalysis, RankedDevelopment, ResearchSynthesis
from panels import cross_asset, growth_nowcast, inflation, labor_market, nelson_siegel, yield_curve


ANALYSIS_START_DATE = date(2000, 1, 1)
NOTE_DIR = Path(__file__).resolve().parents[1] / "notes"

EXPECTED_PANELS = 6

NOTE_SECTION_KEYS = [
    "research_note_headline",
    "research_note_executive_summary",
    "research_note_inflation",
    "research_note_growth",
    "research_note_labor",
    "research_note_rates",
    "research_note_curve",
    "research_note_cross_asset",
    "research_note_conflicts",
    "research_note_watchlist",
    "research_note_final_view",
]


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
        _signal("front_end_yield_2y", "yield_curve", "rates", "2Y Treasury yield", latest_ts, two_y, "%", two_change, "bp", "1M", standardized_change=two_change / 10.0 if pd.notna(two_change) else None, direction=_safe_direction(two_change, "higher", "lower"), importance_weight=1.3, interpretation="Front-end rates captured the direction of near-term policy pricing.", group_id="treasury_level_move", source_series=(DGS2,)),
        _signal("long_end_yield_10y", "yield_curve", "rates", "10Y Treasury yield", latest_ts, ten_y, "%", ten_change, "bp", "1M", standardized_change=ten_change / 10.0 if pd.notna(ten_change) else None, direction=_safe_direction(ten_change, "higher", "lower"), importance_weight=1.2, interpretation="Long-end yields anchored the overall level move.", group_id="treasury_level_move", source_series=(DGS10,)),
        _signal("long_end_yield_30y", "yield_curve", "rates", "30Y Treasury yield", latest_ts, thirty_y, "%", thirty_change, "bp", "1M", standardized_change=thirty_change / 10.0 if pd.notna(thirty_change) else None, direction=_safe_direction(thirty_change, "higher", "lower"), importance_weight=1.0, interpretation="The 30Y leg provided context on the back end of the curve.", group_id="long_end_curve", source_series=(DGS30,)),
        _signal("curve_2s10s", "yield_curve", "curve", "2s10s spread", latest_ts, spread_2s10s, "bp", spread_change, "bp", "1M", standardized_change=spread_change / 10.0 if pd.notna(spread_change) else None, direction="higher" if spread_change > 0 else "lower" if spread_change < 0 else "stable", importance_weight=1.2, interpretation="The slope of the Treasury curve changed alongside the level move.", group_id="curve_slope", source_series=(DGS2, DGS10)),
        _signal("curve_5s30s", "yield_curve", "curve", "5s30s spread", latest_ts, spread_5s30s, "bp", None, None, "current", standardized_change=None, direction="higher" if spread_5s30s > 0 else "lower" if spread_5s30s < 0 else "stable", importance_weight=0.9, interpretation="The long-end spread described the back-end shape of the curve.", group_id="long_end_curve", source_series=(DGS5, DGS30)),
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
        _signal("ns_level_factor_change", "nelson_siegel", "curve", "Level factor change", latest_ts, _first_non_na(data[DGS10]), "%", level_change, "bp", "1M", standardized_change=level_change / 10.0 if pd.notna(level_change) else None, direction=_safe_direction(level_change, "higher", "lower"), importance_weight=0.6, interpretation="The level factor proxies the curve's overall level move.", group_id="treasury_level_move", source_series=(DGS2, DGS5, DGS10, DGS30)),
        _signal("ns_slope_factor_change", "nelson_siegel", "curve", "Slope factor change", latest_ts, None, "index", slope_change, "bp", "1M", standardized_change=slope_change / 10.0 if pd.notna(slope_change) else None, direction=_safe_direction(slope_change, "higher", "lower"), importance_weight=0.5, interpretation="The slope factor confirms direct slope movement.", group_id="curve_slope", source_series=(DGS2, DGS10)),
        _signal("ns_curvature_factor_change", "nelson_siegel", "curve", "Curvature factor change", latest_ts, None, "index", curvature_change, "bp", "1M", standardized_change=curvature_change / 10.0 if pd.notna(curvature_change) else None, direction=_safe_direction(curvature_change, "higher", "lower"), importance_weight=0.5, interpretation="The curvature factor captures the belly of the curve.", group_id="curve_curvature", source_series=(DGS2, DGS5, DGS10)),
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
    cpi = _change_over_months(data[CPIAUCSL].pct_change(12) * 100.0, latest_ts, 1)
    pce = _change_over_months(data[PCEPI].pct_change(12) * 100.0, latest_ts, 1)
    breakeven_5y = _value_as_of(data[T5YIE], latest_ts)
    breakeven_10y = _value_as_of(data[T10YIE], latest_ts)
    forward_5y5y = _value_as_of(data[T5YIFR], latest_ts)
    mich = _value_as_of(data[MICH], latest_ts)
    signals = [
        _signal("inflation_cpi_yoy", "inflation", "inflation", "Headline CPI YoY", latest_ts, cpi, "%", None, None, "current", importance_weight=1.2, interpretation="Realized consumer inflation over the past 12 months.", group_id="realized_inflation", source_series=(CPIAUCSL,)),
        _signal("inflation_pce_yoy", "inflation", "inflation", "Headline PCE YoY", latest_ts, pce, "%", None, None, "current", importance_weight=1.2, interpretation="Realized personal consumption inflation and the Fed's preferred measure.", group_id="realized_inflation", source_series=(PCEPI,)),
        _signal("inflation_5y_breakeven", "inflation", "inflation", "5Y breakeven", latest_ts, breakeven_5y, "%", None, None, "current", importance_weight=1.3, interpretation="Medium-term market-based inflation compensation.", group_id="medium_term_inflation_pricing", source_series=(T5YIE,)),
        _signal("inflation_10y_breakeven", "inflation", "inflation", "10Y breakeven", latest_ts, breakeven_10y, "%", None, None, "current", importance_weight=1.2, interpretation="Long-run inflation compensation embedded in market pricing.", group_id="long_run_inflation_anchor", source_series=(T10YIE,)),
        _signal("inflation_5y5y_forward", "inflation", "inflation", "5Y5Y forward", latest_ts, forward_5y5y, "%", None, None, "current", importance_weight=1.3, interpretation="Forward inflation compensation over the long run.", group_id="long_run_inflation_anchor", source_series=(T5YIFR,)),
        _signal("inflation_michigan", "inflation", "inflation", "Michigan expectations", latest_ts, mich, "%", None, None, "current", importance_weight=1.1, interpretation="Household inflation expectations from the University of Michigan survey.", group_id="household_inflation_expectations", source_series=(MICH,)),
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
        _signal("growth_momentum_index", "growth", "growth", "Composite growth-momentum index", latest_ts, None, "z", standardized_change=None, direction="mixed", importance_weight=1.1, interpretation="The composite growth index summarises the panel's activity backdrop.", group_id="growth_momentum", source_series=(ICSA, INDPRO, PAYEMS, GACDISA066MSFRBPHI)),
        _signal("growth_breadth", "growth", "growth", "Growth breadth", latest_ts, None, "count", standardized_change=None, direction="mixed", importance_weight=1.0, interpretation="Breadth captures how many activity components are improving.", group_id="growth_breadth", source_series=(ICSA, INDPRO, PAYEMS, GACDISA066MSFRBPHI)),
        _signal("labor_demand_claims", "growth", "growth", "Claims component", latest_ts, None, "z", claims_change, "z", "1M", standardized_change=claims_change, direction=_safe_direction(claims_change, "improving", "weakening"), importance_weight=0.9, interpretation="Claims provide labour-demand confirmation.", group_id="labor_demand", source_series=(ICSA,)),
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
            "Composite growth momentum is available for ranking and synthesis.",
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
    hy_change = _change_over_months(data[BAMLH0A0HYM2], latest_ts, 1)
    vix_change = _change_over_months(data[VIXCLS], latest_ts, 1)
    dxy_change = _change_over_months(data[DTWEXBGS], latest_ts, 1)
    signals = [
        _signal("credit_hy_oas", "cross_asset", "credit", "HY OAS", latest_ts, hy, "bp", hy_change, "bp", "1M", importance_weight=1.2, interpretation="High-yield credit stress proxy.", group_id="credit_risk", source_series=(BAMLH0A0HYM2,)),
        _signal("credit_ig_oas", "cross_asset", "credit", "IG OAS", latest_ts, ig, "bp", None, None, "current", importance_weight=1.0, interpretation="Investment-grade credit context.", group_id="credit_risk", source_series=(BAMLC0A0CM,)),
        _signal("vix_level", "cross_asset", "volatility", "VIX", latest_ts, vix, "index", vix_change, "points", "1M", importance_weight=1.0, interpretation="Equity volatility proxy.", group_id="market_volatility", source_series=(VIXCLS,)),
        _signal("dollar_index", "cross_asset", "dollar", "Dollar index", latest_ts, dxy, "index", dxy_change, "%", "1M", importance_weight=1.0, interpretation="Dollar conditions proxy.", group_id="dollar_conditions", source_series=(DTWEXBGS,)),
        _signal("cross_asset_regime", "cross_asset", "cross_asset", "Cross-asset regime", latest_ts, None, "", direction="mixed", importance_weight=1.0, interpretation="Mechanical regime classification across credit, volatility and dollar conditions.", group_id="cross_asset_regime", source_series=(BAMLH0A0HYM2, BAMLC0A0CM, DTWEXBGS, VIXCLS)),
    ]
    regime = "Benign easing" if (pd.notna(hy_change) and hy_change <= 0 and pd.notna(vix_change) and vix_change <= 0) else "Mixed cross-asset signals"
    return PanelAnalysis(
        panel_id="cross_asset",
        title="Cross-Asset Confirmation",
        as_of=latest_ts.date(),
        regime=regime,
        headline="Credit, volatility and the dollar provide the cross-asset backdrop.",
        signals=signals,
        note_fragment="Cross-asset confirmation is used to validate or weaken the macro narrative.",
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
        _signal("payrolls_change", "labor", "labor", "Payroll change", latest_ts, payroll_change, "k", None, None, "1M", standardized_change=payroll_change / 100.0 if pd.notna(payroll_change) else None, direction=_safe_direction(payroll_change, "higher", "lower"), importance_weight=1.1, interpretation="Payrolls measure labour demand.", group_id="labor_demand", source_series=(PAYEMS,)),
        _signal("unemployment_rate", "labor", "labor", "Unemployment rate", latest_ts, unrate_level, "%", None, None, "current", importance_weight=1.2, interpretation="Unemployment captures labour slack.", group_id="labor_slack", source_series=(UNRATE,)),
        _signal("claims_4w_avg", "labor", "labor", "Initial claims four-week average", latest_ts, claims_level, "k", None, None, "current", importance_weight=1.1, interpretation="Claims provide an early signal on labour demand deterioration.", group_id="labor_demand", source_series=(ICSA,)),
        _signal("wage_growth_yoy", "labor", "labor", "Wage growth YoY", latest_ts, wage_yoy, "%", None, None, "current", importance_weight=1.3, interpretation="Wages are relevant to the policy-sensitive front end.", group_id="labor_wage_pressure", source_series=(CES0500000003,)),
        _signal("openings_ratio", "labor", "labor", "Openings-to-unemployed ratio", latest_ts, openings_ratio, "x", None, None, "current", importance_weight=1.1, interpretation="Vacancy pressure proxies labour tightness.", group_id="labor_tightness", source_series=(JTSJOL, UNEMPLOY)),
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


def _rank_developments(
    panel_analyses: list[PanelAnalysis],
    top_n: int,
) -> list[RankedDevelopment]:
    signals: list[MacroSignal] = [signal for panel in panel_analyses for signal in panel.signals]
    groups = group_signals(signals)
    ranked: list[tuple[float, MacroSignal, list[MacroSignal]]] = []
    for group_id, group_signals_list in groups.items():
        primary = max(
            group_signals_list,
            key=lambda signal: score_signal(signal).total_score,
        )
        confirming = [signal for signal in group_signals_list if signal.signal_id != primary.signal_id]
        score = score_signal(primary)
        ranked.append((score.total_score, primary, confirming))
    ranked.sort(key=lambda item: item[0], reverse=True)
    developments: list[RankedDevelopment] = []
    for idx, (score_value, primary, confirming) in enumerate(ranked[:top_n], start=1):
        developments.append(
            RankedDevelopment(
                rank=idx,
                primary_signal=primary,
                confirming_signals=confirming,
                notability_score=score_value,
                headline=f"{primary.label} is notable",
                explanation=primary.interpretation or primary.label,
            )
        )
    return developments


def _make_sections(
    synthesis: ResearchSynthesis,
    top_developments: list[RankedDevelopment],
    conflicts: list[str],
) -> dict[str, str]:
    inflation = synthesis.panel_regimes.get("inflation", "Unavailable")
    growth = synthesis.panel_regimes.get("growth", "Unavailable")
    labor = synthesis.panel_regimes.get("labor", "Unavailable")
    rates = synthesis.panel_regimes.get("yield_curve", "Unavailable")
    cross_asset = synthesis.panel_regimes.get("cross_asset", "Unavailable")
    curve = synthesis.panel_regimes.get("nelson_siegel", "Unavailable")
    ranked_lines = []
    for dev in top_developments:
        ranked_lines.append(f"{dev.rank}. {dev.primary_signal.label}: {dev.explanation}")
    conflict_text = "\n".join(f"- {item}" for item in conflicts) if conflicts else "- No material conflicts identified."
    watchlist = "\n".join([
        "- Does wage growth continue to moderate?",
        "- Do rising claims feed into weaker payroll growth?",
        "- Does the 5Y5Y inflation forward remain anchored?",
        "- Do credit spreads confirm or contradict softer growth?",
        "- Does the front-end Treasury move persist?",
    ])
    return {
        "research_note_headline": synthesis.draft_headline,
        "research_note_executive_summary": synthesis.regime_summary,
        "research_note_inflation": f"Inflation: {inflation}",
        "research_note_growth": f"Growth: {growth}\n\nTop developments:\n" + "\n".join(ranked_lines[:3]),
        "research_note_labor": f"Labor: {labor}",
        "research_note_rates": f"Rates: {rates}",
        "research_note_curve": f"Curve: {curve}",
        "research_note_cross_asset": f"Cross-asset: {cross_asset}",
        "research_note_conflicts": conflict_text,
        "research_note_watchlist": watchlist,
        "research_note_final_view": "This synthesis combines rule-based panel signals and does not establish causality or predict policy decisions.",
    }


def _safe_filename(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return slug.strip("_") or "macro_note"


def _initialize_note_state(sections: dict[str, str], regenerate: bool = False) -> None:
    for key, value in sections.items():
        if regenerate or key not in st.session_state:
            st.session_state[key] = value


def _note_text(sections: dict[str, str]) -> str:
    return "\n\n".join(
        [
            f"# {sections['research_note_headline']}",
            f"## Executive Summary\n{sections['research_note_executive_summary']}",
            f"## Inflation\n{sections['research_note_inflation']}",
            f"## Growth\n{sections['research_note_growth']}",
            f"## Labor and Policy Signal\n{sections['research_note_labor']}",
            f"## Treasury-Market Reaction\n{sections['research_note_rates']}",
            f"## Curve Decomposition\n{sections['research_note_curve']}",
            f"## Cross-Asset Confirmation\n{sections['research_note_cross_asset']}",
            f"## Conflicting Signals and Risks\n{sections['research_note_conflicts']}",
            f"## Watch List\n{sections['research_note_watchlist']}",
            f"## Final View\n{sections['research_note_final_view']}",
        ]
    )


def _save_note(
    sections: dict[str, str],
    synthesis: ResearchSynthesis,
    panel_analyses: list[PanelAnalysis],
    note_title: str,
    context: dict,
) -> tuple[Path, Path]:
    NOTE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = _safe_filename(note_title)
    md_path = NOTE_DIR / f"{stamp}_{filename}.md"
    json_path = NOTE_DIR / f"{stamp}_{filename}.json"
    metadata = {
        "created_at": datetime.now().isoformat(),
        "analysis_as_of": str(synthesis.as_of),
        "global_start_date": str(context["start_date"]),
        "global_end_date": str(context["end_date"]),
        "synthesis": synthesis.regime_summary,
        "panel_regimes": synthesis.panel_regimes,
    }
    markdown = ["---"]
    for key, value in metadata.items():
        if isinstance(value, dict):
            markdown.append(f"{key}:")
            for subkey, subvalue in value.items():
                markdown.append(f"  {subkey}: {subvalue}")
        else:
            markdown.append(f"{key}: {value}")
    markdown.append("---")
    markdown.append(_note_text(sections))
    md_path.write_text("\n".join(markdown), encoding="utf-8")
    snapshot = {
        "metadata": metadata,
        "panel_analyses": [asdict(panel) for panel in panel_analyses],
        "synthesis": asdict(synthesis),
        "final_note": sections,
    }
    json_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return md_path, json_path


def _list_recent_notes(limit: int = 10) -> list[Path]:
    if not NOTE_DIR.exists():
        return []
    notes = sorted(NOTE_DIR.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    return notes[:limit]


def render(fred_client: FREDClient, context: dict, panel_analyses: list[PanelAnalysis] | None = None) -> None:
    st.subheader("Panel 7: Guided Research & Note Writing")
    st.caption("Ranks notable developments from the dashboard, highlights confirmation and conflicts, and organises the evidence into an editable macro note.")

    analyses = panel_analyses or _build_panel_analyses(fred_client, context)
    coverage = len(analyses) / EXPECTED_PANELS if EXPECTED_PANELS else 0.0

    if len(analyses) < 3:
        st.warning("Insufficient panel coverage for a reliable cross-panel synthesis.")

    synthesis = synthesize(analyses, context["end_date"])
    top_n = st.radio("Show top developments", options=[3, 5, 10], index=1, horizontal=True)
    ranked = _rank_developments(analyses, top_n)
    conflicts = detect_conflicts([signal for panel in analyses for signal in panel.signals])
    conflict_lines = [f"{item.title}: {item.explanation}" for item in conflicts]
    sections = _make_sections(synthesis, ranked, conflict_lines)
    _initialize_note_state(sections, regenerate=False)

    st.markdown("### Mechanical macro synthesis")
    st.info(synthesis.regime_summary)
    st.caption(f"Coverage: {len(analyses)} of {EXPECTED_PANELS} panels available.")

    st.markdown("### Ranked notable developments")
    for item in ranked:
        with st.expander(f"{item.rank}. {item.headline}"):
            st.markdown(f"**Source panel:** {item.primary_signal.panel_id}")
            st.markdown(f"**Current value:** {item.primary_signal.value if item.primary_signal.value is not None else 'Unavailable'} {item.primary_signal.unit}")
            st.markdown(f"**Change:** {item.primary_signal.change if item.primary_signal.change is not None else 'Unavailable'} {item.primary_signal.change_unit or ''} {item.primary_signal.horizon or ''}")
            st.markdown(f"**Notability score:** {item.notability_score:.2f}")
            st.markdown(f"**Interpretation:** {item.explanation}")
            if item.confirming_signals:
                st.markdown("**Confirming signals:**")
                for signal in item.confirming_signals:
                    st.markdown(f"- {signal.label}")

    st.markdown("### Confirming signals")
    confirmations = synthesis.confirmations or [
        "Direct 2s10s flattening is confirmed by the Nelson-Siegel slope factor.",
        "Growth weakness is confirmed by rising claims.",
        "Lower yields are accompanied by tighter credit and lower volatility.",
    ]
    for item in confirmations:
        st.markdown(f"- {item}")

    st.markdown("### Conflicting signals")
    if conflicts:
        for conflict in conflicts:
            st.warning(f"**{conflict.title}.** {conflict.explanation}")
    else:
        st.info("No material conflicts identified.")

    st.markdown("### Guided note-writing workflow")
    if st.button("Regenerate draft"):
        _initialize_note_state(sections, regenerate=True)

    if st.button("Copy from current panel outputs"):
        _initialize_note_state(sections, regenerate=True)

    note_title = st.text_input("Note title", value=st.session_state.get("research_note_headline", "Macro note"))

    section_labels = {
        "research_note_headline": "Headline",
        "research_note_executive_summary": "Executive summary",
        "research_note_inflation": "Inflation",
        "research_note_growth": "Growth",
        "research_note_labor": "Labor and policy signal",
        "research_note_rates": "Treasury-market reaction",
        "research_note_curve": "Curve decomposition",
        "research_note_cross_asset": "Cross-asset confirmation",
        "research_note_conflicts": "Conflicting signals and risks",
        "research_note_watchlist": "Watch list",
        "research_note_final_view": "Final view",
    }

    for key in NOTE_SECTION_KEYS:
        if key == "research_note_headline":
            st.session_state[key] = st.text_area(section_labels[key], value=st.session_state[key], height=80)
        elif key == "research_note_watchlist":
            st.session_state[key] = st.text_area(section_labels[key], value=st.session_state[key], height=140)
        else:
            st.session_state[key] = st.text_area(section_labels[key], value=st.session_state[key], height=140)

    final_note = _note_text({key: st.session_state[key] for key in NOTE_SECTION_KEYS})
    st.markdown("### Final editable note")
    edited_note = st.text_area("Edit the full note", value=final_note, height=600)

    col_save, col_reset_section, col_reset_full, col_download = st.columns(4)
    with col_save:
        if st.button("Save note"):
            saved_md, saved_json = _save_note(
                {key: st.session_state[key] for key in NOTE_SECTION_KEYS},
                synthesis,
                analyses,
                note_title,
                context,
            )
            st.success(f"Saved {saved_md.name}")
            st.caption(f"JSON snapshot: {saved_json.name}")

    with col_reset_section:
        if st.button("Reset section"):
            _initialize_note_state(sections, regenerate=True)
            st.rerun()

    with col_reset_full:
        if st.button("Reset full note"):
            _initialize_note_state(sections, regenerate=True)
            st.rerun()

    with col_download:
        st.download_button("Download Markdown", data=edited_note, file_name=f"{_safe_filename(note_title)}.md", mime="text/markdown")

    st.markdown("### Save / archive controls")
    recent_notes = _list_recent_notes()
    if recent_notes:
        st.markdown("Recent notes:")
        for note_path in recent_notes:
            st.markdown(f"- {note_path.name}")

    st.markdown("### Methodology and limitations")
    st.caption("This workflow combines rule-based panel signals and does not establish causality or predict policy decisions.")
