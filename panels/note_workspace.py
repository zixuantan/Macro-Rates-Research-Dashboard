from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests
import streamlit as st
from fredapi import Fred

from config import (
	FRED_API_KEY,
	FRED_CACHE_TTL_SECONDS,
	FRED_POLICY_SERIES,
	NOTE_WORKSPACE_ARCHIVE_DIR,
	NOTE_WORKSPACE_MAX_MOVES,
)
from data.fred_client import FREDClient
from data.market_events import get_scheduled_catalysts as _shared_scheduled_catalysts
from models.macro_analysis import MacroSignal, PanelAnalysis
from panels import guided_research


NOTE_NAMESPACE = "note_workspace"

COMPARISON_HORIZONS = {
	"1D": pd.DateOffset(days=1),
	"1W": pd.DateOffset(weeks=1),
	"1M": pd.DateOffset(months=1),
	"3M": pd.DateOffset(months=3),
	"6M": pd.DateOffset(months=6),
	"1Y": pd.DateOffset(years=1),
}

NOTE_TYPE_DEFAULT_HORIZON = {
	"Macro trade": "1W",
	"Event-driven trade": "1D",
	"Relative-value / thematic": "1M",
}

TRADE_IDEA_TYPES = tuple(NOTE_TYPE_DEFAULT_HORIZON)

TRADE_STANCE_OPTIONS = (
	"Not specified",
	"Long",
	"Short",
	"Steepener",
	"Flattener",
	"Wider",
	"Tighter",
	"No trade / watchlist only",
)

TRADE_HORIZON_OPTIONS = (
	"Not specified",
	"Intraday",
	"Days",
	"Weeks",
	"1–3 months",
	"3–12 months",
	"Structural / 1Y+",
)

RISK_REWARD_OPTIONS = (
	"Not specified",
	"Below 1:1",
	"1:1",
	"1.5:1",
	"2:1",
	"3:1 or better",
)

CARRY_ROLL_OPTIONS = (
	"Not assessed",
	"Positive",
	"Neutral",
	"Negative",
)

CONVICTION_OPTIONS = (
	"Not specified",
	"Watchlist only",
	"Low",
	"Medium",
	"High",
)

TRADE_CANDIDATE_IDS = ("Candidate 1", "Candidate 2", "Candidate 3")
SIGNAL_UPDATE_OPTIONS = ("Unclear", "Strengthened", "Weakened", "Unchanged", "Reversed")

CATALYST_CATEGORY_OPTIONS = (
	"Monetary policy",
	"Inflation",
	"Growth / employment",
	"Geopolitics / supply",
	"Systemic risk / liquidity",
	"Corporate action",
)

FIRST_ORDER_ASSET_OPTIONS = (
	"Short-term rates — SOFR or fed-funds futures",
	"Government bonds — Treasury futures or cash Treasuries",
	"Inflation-linked bonds — TIPS",
	"FX — G10 currency pairs",
	"Equity indices — index futures or ETFs",
	"Individual equities — shares or single-stock options",
	"Credit — CDX or corporate-bond ETFs",
	"Commodities — futures or ETFs",
	"Precious metals — futures or ETFs",
	"Volatility — options or volatility futures",
)

SECOND_ORDER_ASSET_OPTIONS = FIRST_ORDER_ASSET_OPTIONS

CONFIRMATION_DIRECTION_OPTIONS = (
	"Not specified",
	"Higher / stronger / wider",
	"Lower / weaker / tighter",
	"Steeper",
	"Flatter",
	"No material move",
)

CONFIRMATION_STATUS_OPTIONS = (
	"Not assessed",
	"Confirms",
	"Partially confirms",
	"Contradicts",
	"Not yet tested",
	"Already priced",
)

FIRST_ORDER_CONFIRMATION_IDS = (
	"yield_2y", "yield_10y", "curve_2s10s", "curve_5s30s",
	"inflation_5y_be", "inflation_10y_be", "inflation_5y5y", "dxy",
)

SECOND_ORDER_CONFIRMATION_IDS = ("sp500", "credit_ig", "credit_hy", "vix", "dxy", "nfci")

THIRD_ORDER_EXPOSURE_OPTIONS = (
	"Operational exposure — input cost or revenue driver",
	"Financing exposure — cost of debt or discount rate",
	"Asset exposure — producer, refiner, extractor or owner",
)

THIRD_ORDER_SECTOR_OPTIONS = (
	"Technology / software",
	"Semiconductors",
	"Banks / financials",
	"REITs / real estate",
	"Utilities",
	"Homebuilders",
	"Industrials / capital goods",
	"Consumer discretionary",
	"Consumer staples",
	"Energy producers",
	"Refiners",
	"Metals and mining",
	"Airlines / transportation",
	"Healthcare / biotechnology",
)

RELATIVE_STRENGTH_OPTIONS = (
	"Not assessed",
	"Bullish — buy the leader",
	"Bearish — short the laggard",
	"Relative value — long leader / short laggard",
	"No clear relative-strength signal",
)

TARGET_VEHICLE_OPTIONS = (
	"Not selected",
	"Sector ETF",
	"Industry ETF",
	"Single stock",
	"Equity option",
	"Futures contract",
	"Long-short pair",
)

LIQUIDITY_CHECK_OPTIONS = ("Not assessed", "Adequate", "Unclear", "Inadequate")

MACRO_UPDATE_GROUPS = {
	"Growth": ("cfnai",),
	"Inflation": ("inflation_core_cpi", "inflation_core_pce", "inflation_ppi"),
	"Labour": ("payrolls", "unrate", "claims", "wages", "openings_ratio"),
	"Financial conditions": ("nfci", "credit_hy", "credit_ig", "sp500", "vix", "dxy"),
}

MACRO_NARRATIVE_IMPULSES = {
	"Growth": {"cfnai": 1},
	"Inflation": {"inflation_core_cpi": 1, "inflation_core_pce": 1, "inflation_ppi": 1},
	"Labour": {"payrolls": 1, "unrate": -1, "claims": -1, "wages": 1, "openings_ratio": 1},
	"Financial conditions": {"nfci": -1, "credit_hy": -1, "credit_ig": -1, "sp500": 1, "vix": -1, "dxy": -1},
}

MARKET_REACTION_IDS = (
	"yield_2y", "yield_10y", "curve_2s10s", "curve_5s30s",
	"inflation_5y_be", "inflation_10y_be", "inflation_5y5y",
	"nfci", "credit_hy", "credit_ig", "sp500", "vix", "dxy",
)

NARRATIVE_MARKET_IDS = {
	"Growth": ("yield_2y", "yield_10y", "curve_2s10s", "sp500", "credit_hy", "vix"),
	"Inflation": ("inflation_5y_be", "inflation_10y_be", "inflation_5y5y", "yield_2y", "yield_10y", "dxy"),
	"Labour": ("yield_2y", "curve_2s10s", "sp500", "credit_hy", "vix"),
	"Financial conditions": ("nfci", "credit_hy", "credit_ig", "sp500", "vix", "dxy"),
	"Policy": ("yield_2y", "yield_10y", "curve_2s10s", "dxy"),
}

HISTORY_YEARS = 5
MIN_HISTORICAL_MOVES_BY_FREQUENCY = {
	"daily": 60,
	"weekly": 30,
	"monthly": 24,
}
EXTREME_ZSCORE_WARNING = 8.0

TRACKED_SIGNAL_SPECS = [
	{"metric_id": "yield_2y", "panel_id": "yield_curve", "signal_id": "front_end_yield_2y", "label": "2Y Treasury yield"},
	{"metric_id": "yield_10y", "panel_id": "yield_curve", "signal_id": "long_end_yield_10y", "label": "10Y Treasury yield"},
	{"metric_id": "curve_2s10s", "panel_id": "yield_curve", "signal_id": "curve_2s10s", "label": "2s10s spread"},
	{"metric_id": "curve_5s30s", "panel_id": "yield_curve", "signal_id": "curve_5s30s", "label": "5s30s spread"},
	{"metric_id": "inflation_5y_be", "panel_id": "inflation", "signal_id": "inflation_5y_breakeven", "label": "5Y breakeven"},
	{"metric_id": "inflation_10y_be", "panel_id": "inflation", "signal_id": "inflation_10y_breakeven", "label": "10Y breakeven"},
	{"metric_id": "inflation_5y5y", "panel_id": "inflation", "signal_id": "inflation_5y5y_forward", "label": "5Y5Y forward"},
	{"metric_id": "inflation_cpi", "panel_id": "inflation", "signal_id": "inflation_cpi_yoy", "label": "Headline CPI YoY (SA index)"},
	{"metric_id": "inflation_core_cpi", "panel_id": "inflation", "signal_id": "inflation_core_cpi_yoy", "label": "Core CPI YoY"},
	{"metric_id": "inflation_pce", "panel_id": "inflation", "signal_id": "inflation_pce_yoy", "label": "Headline PCE YoY"},
	{"metric_id": "inflation_core_pce", "panel_id": "inflation", "signal_id": "inflation_core_pce_yoy", "label": "Core PCE YoY"},
	{"metric_id": "inflation_ppi", "panel_id": "inflation", "signal_id": "inflation_ppi_yoy", "label": "Final-demand PPI YoY"},
	{"metric_id": "cfnai", "panel_id": "growth", "signal_id": "cfnai_3m_average", "label": "CFNAI · 3M average"},
	{"metric_id": "nfci", "panel_id": "cross_asset", "signal_id": "financial_conditions_nfci", "label": "Chicago Fed NFCI"},
	{"metric_id": "credit_hy", "panel_id": "cross_asset", "signal_id": "credit_hy_oas", "label": "HY OAS"},
	{"metric_id": "credit_ig", "panel_id": "cross_asset", "signal_id": "credit_ig_oas", "label": "IG OAS"},
	{"metric_id": "sp500", "panel_id": "cross_asset", "signal_id": "sp500_performance", "label": "S&P 500"},
	{"metric_id": "vix", "panel_id": "cross_asset", "signal_id": "vix_level", "label": "VIX"},
	{"metric_id": "dxy", "panel_id": "cross_asset", "signal_id": "dollar_index", "label": "Dollar index"},
	{"metric_id": "payrolls", "panel_id": "labor", "signal_id": "payrolls_change", "label": "Payroll change"},
	{"metric_id": "unrate", "panel_id": "labor", "signal_id": "unemployment_rate", "label": "Unemployment rate"},
	{"metric_id": "claims", "panel_id": "labor", "signal_id": "claims_4w_avg", "label": "Initial claims 4W avg"},
	{"metric_id": "wages", "panel_id": "labor", "signal_id": "wage_growth_yoy", "label": "Wage growth YoY"},
	{"metric_id": "openings_ratio", "panel_id": "labor", "signal_id": "openings_ratio", "label": "Openings/unemployed ratio"},
]

MOVE_DIRECTION_HINTS = {
	"curve_2s10s": ("steepening", "flattening"),
	"curve_5s30s": ("steepening", "flattening"),
	"ns_slope": ("steepening", "flattening"),
	"credit_hy": ("widening", "tightening"),
	"credit_ig": ("widening", "tightening"),
	"vix": ("higher", "lower"),
	"dxy": ("stronger", "softer"),
	"sp500": ("higher", "lower"),
	"nfci": ("tighter", "looser"),
	"cfnai": ("stronger", "weaker"),
	"payrolls": ("stronger", "weaker"),
	"unrate": ("higher", "lower"),
	"claims": ("higher", "lower"),
	"wages": ("higher", "lower"),
	"inflation_5y_be": ("higher", "lower"),
	"inflation_10y_be": ("higher", "lower"),
	"inflation_5y5y": ("higher", "lower"),
}

MOVE_PATTERNS = [
	{
		"name": "Risk-off pattern",
		"signals": [
			{"metric_id": "credit_hy", "expected": "up", "label": "credit spreads wider"},
			{"metric_id": "vix", "expected": "up", "label": "VIX higher"},
			{"metric_id": "sp500", "expected": "down", "label": "S&P 500 lower"},
			{"metric_id": "nfci", "expected": "up", "label": "financial conditions tighter"},
		],
	},
	{
		"name": "Risk-on / soft landing",
		"signals": [
			{"metric_id": "credit_hy", "expected": "down", "label": "HY spreads tighter"},
			{"metric_id": "credit_ig", "expected": "down", "label": "IG spreads tighter"},
			{"metric_id": "vix", "expected": "down", "label": "VIX lower"},
			{"metric_id": "sp500", "expected": "up", "label": "S&P 500 higher"},
			{"metric_id": "nfci", "expected": "down", "label": "financial conditions looser"},
		],
	},
	{
		"name": "Hawkish repricing",
		"signals": [
			{"metric_id": "yield_2y", "expected": "up", "label": "front-end yields higher"},
			{"metric_id": "curve_2s10s", "expected": "down", "label": "2s10s flatter"},
			{"metric_id": "dxy", "expected": "up", "label": "dollar stronger"},
		],
	},
	{
		"name": "Dovish repricing",
		"signals": [
			{"metric_id": "yield_2y", "expected": "down", "label": "front-end yields lower"},
			{"metric_id": "curve_2s10s", "expected": "up", "label": "curve steepening"},
			{"metric_id": "dxy", "expected": "down", "label": "dollar softer"},
			{"metric_id": "vix", "expected": "down", "label": "volatility lower"},
		],
	},
	{
		"name": "Inflation reacceleration",
		"signals": [
			{"metric_id": "inflation_core_cpi", "expected": "up", "label": "core CPI firmer"},
			{"metric_id": "inflation_core_pce", "expected": "up", "label": "core PCE firmer"},
			{"metric_id": "inflation_5y_be", "expected": "up", "label": "5Y breakevens firmer"},
			{"metric_id": "inflation_5y5y", "expected": "up", "label": "5Y5Y forward firmer"},
		],
	},
	{
		"name": "Disinflation progress",
		"signals": [
			{"metric_id": "inflation_core_cpi", "expected": "down", "label": "core CPI softer"},
			{"metric_id": "inflation_core_pce", "expected": "down", "label": "core PCE softer"},
			{"metric_id": "inflation_5y_be", "expected": "down", "label": "5Y breakevens softer"},
			{"metric_id": "inflation_5y5y", "expected": "down", "label": "5Y5Y forward softer"},
		],
	},
	{
		"name": "Growth scare",
		"signals": [
			{"metric_id": "cfnai", "expected": "down", "label": "CFNAI weaker"},
			{"metric_id": "payrolls", "expected": "down", "label": "payrolls softer"},
			{"metric_id": "credit_hy", "expected": "up", "label": "credit confirming"},
			{"metric_id": "vix", "expected": "up", "label": "volatility confirming"},
		],
	},
	{
		"name": "Labor cooling",
		"signals": [
			{"metric_id": "payrolls", "expected": "down", "label": "payrolls softer"},
			{"metric_id": "claims", "expected": "up", "label": "claims rising"},
			{"metric_id": "unrate", "expected": "up", "label": "unemployment higher"},
			{"metric_id": "openings_ratio", "expected": "down", "label": "openings ratio lower"},
			{"metric_id": "wages", "expected": "down", "label": "wage growth cooling"},
		],
	},
	{
		"name": "Labor resilience",
		"signals": [
			{"metric_id": "payrolls", "expected": "up", "label": "payrolls firm"},
			{"metric_id": "claims", "expected": "down", "label": "claims lower"},
			{"metric_id": "unrate", "expected": "down", "label": "unemployment lower"},
			{"metric_id": "openings_ratio", "expected": "up", "label": "openings ratio stronger"},
			{"metric_id": "wages", "expected": "up", "label": "wage growth firmer"},
		],
	},
	{
		"name": "Stagflation pressure",
		"signals": [
			{"metric_id": "inflation_core_cpi", "expected": "up", "label": "core CPI firming"},
			{"metric_id": "inflation_core_pce", "expected": "up", "label": "core PCE firming"},
			{"metric_id": "cfnai", "expected": "down", "label": "CFNAI weaker"},
			{"metric_id": "claims", "expected": "up", "label": "claims rising"},
			{"metric_id": "credit_hy", "expected": "up", "label": "HY spreads wider"},
		],
	},
]

CATEGORY_ORDER = [
	"Rates",
	"Curve",
	"Inflation",
	"Credit",
	"FX",
	"Equities",
	"Labour or growth",
	"Other",
]

METRIC_CATEGORY_BY_ID = {
	"yield_2y": "Rates",
	"yield_10y": "Rates",
	"ns_level": "Rates",
	"yield_curve": "Curve",
	"curve_2s10s": "Curve",
	"curve_5s30s": "Curve",
	"ns_slope": "Curve",
	"ns_curvature": "Curve",
	"inflation_5y_be": "Inflation",
	"inflation_10y_be": "Inflation",
	"inflation_5y5y": "Inflation",
	"inflation_cpi": "Inflation",
	"inflation_core_cpi": "Inflation",
	"inflation_pce": "Inflation",
	"inflation_core_pce": "Inflation",
	"inflation_ppi": "Inflation",
	"cfnai": "Labour or growth",
	"nfci": "Credit",
	"credit_hy": "Credit",
	"credit_ig": "Credit",
	"sp500": "Equities",
	"vix": "Equities",
	"dxy": "FX",
	"payrolls": "Labour or growth",
	"unrate": "Labour or growth",
	"claims": "Labour or growth",
	"wages": "Labour or growth",
	"openings_ratio": "Labour or growth",
}

ANCHOR_THRESHOLDS = {
	"yield_bp": 5.0,
	"curve_bp": 4.0,
	"breakeven_bp": 4.0,
	"credit_spread_bp": 6.0,
	"index_pct": 0.6,
	"vol_points": 1.0,
	"percent": 0.12,
	"k": 10.0,
	"x": 0.05,
}

Z_THRESHOLD = 2.0
SUSPICIOUS_Z_THRESHOLD = 10.0

TRIGGER_CATALYST_SOURCES = {
	"primary_trigger": "Internal change detection",
	"scheduled_catalysts": "FRED / BEA / Federal Reserve / U.S. Treasury calendars",
	"policy_context": "FRED / ALFRED + Federal Reserve RSS",
}


@dataclass(frozen=True)
class WorkspaceMetric:
	metric_id: str
	panel_id: str
	panel_title: str
	label: str
	value: float | None
	unit: str
	change: float | None
	change_unit: str | None
	horizon: str | None
	percentile: float | None
	standardized_change: float | None
	direction: str
	interpretation: str
	caveat: str | None
	as_of: str
	group_id: str | None


@dataclass(frozen=True)
class MoveResult:
	metric_id: str
	label: str
	panel_title: str
	category: str
	frequency: str
	selected_horizon: str
	requested_current_date: str
	requested_comparison_date: str
	effective_current_date: str | None
	effective_comparison_date: str | None
	horizon_observation_count: int
	current_value: float | None
	comparison_value: float | None
	raw_move: float | None
	raw_move_unit: str
	historical_mean_move: float | None
	historical_std_move: float | None
	z_score: float | None
	historical_sample_count: int
	freshness_status: str
	data_quality_status: str | None
	direction: str
	historical_context: str
	current_unit: str
	quality_flag: str | None = None
	percentile: float | None = None

	@property
	def value(self) -> float | None:
		return self.current_value

	@property
	def change(self) -> float | None:
		return self.raw_move

	@property
	def change_unit(self) -> str:
		return self.raw_move_unit

	@property
	def unit(self) -> str:
		return self.current_unit

	@property
	def standardized_change(self) -> float | None:
		return self.z_score

	@property
	def raw_change(self) -> float | None:
		return self.raw_move

	@property
	def zscore(self) -> float | None:
		return self.z_score

	@property
	def horizon(self) -> str:
		return self.selected_horizon

	@property
	def change_horizon(self) -> str:
		return self.selected_horizon

	@property
	def as_of(self) -> str:
		return self.requested_current_date


MoveSummary = MoveResult


@dataclass(frozen=True)
class PatternEvaluation:
	pattern_name: str
	alignment_status: str
	confidence_score: float
	aligned_signals: list[str]
	conflicting_signals: list[str]
	flat_signals: list[str]
	unavailable_signals: list[str]
	takeaway: str


@dataclass(frozen=True)
class ResearchGap:
	task: str
	reason: str
	related: str
	completed: bool = False


@dataclass(frozen=True)
class MetricScaleSpec:
	current_unit: str
	change_unit: str
	series_key: str
	value_multiplier: float = 1.0
	change_multiplier: float = 1.0


METRIC_SCALE_SPECS = {
	"yield_2y": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="yield_2y", value_multiplier=1.0, change_multiplier=100.0),
	"yield_10y": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="yield_10y", value_multiplier=1.0, change_multiplier=100.0),
	"ns_level": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="yield_10y", value_multiplier=1.0, change_multiplier=100.0),
	"curve_2s10s": MetricScaleSpec(current_unit="bp", change_unit="bp", series_key="curve_2s10s"),
	"curve_5s30s": MetricScaleSpec(current_unit="bp", change_unit="bp", series_key="curve_5s30s"),
	"ns_slope": MetricScaleSpec(current_unit="bp", change_unit="bp", series_key="ns_slope"),
	"ns_curvature": MetricScaleSpec(current_unit="bp", change_unit="bp", series_key="ns_curvature"),
	"inflation_5y_be": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_5y_be", change_multiplier=100.0),
	"inflation_10y_be": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_10y_be", change_multiplier=100.0),
	"inflation_5y5y": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_5y5y", change_multiplier=100.0),
	"inflation_cpi": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_cpi", change_multiplier=100.0),
	"inflation_core_cpi": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_core_cpi", change_multiplier=100.0),
	"inflation_pce": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_pce", change_multiplier=100.0),
	"inflation_core_pce": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_core_pce", change_multiplier=100.0),
	"inflation_ppi": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_ppi", change_multiplier=100.0),
	"cfnai": MetricScaleSpec(current_unit="index", change_unit="index", series_key="cfnai"),
	"nfci": MetricScaleSpec(current_unit="index", change_unit="index", series_key="nfci"),
	"credit_hy": MetricScaleSpec(current_unit="bp", change_unit="bp", series_key="credit_hy", value_multiplier=100.0),
	"credit_ig": MetricScaleSpec(current_unit="bp", change_unit="bp", series_key="credit_ig", value_multiplier=100.0),
	"sp500": MetricScaleSpec(current_unit="index", change_unit="index", series_key="sp500"),
	"vix": MetricScaleSpec(current_unit="index", change_unit="index", series_key="vix"),
	"dxy": MetricScaleSpec(current_unit="index", change_unit="index", series_key="dxy"),
	"payrolls": MetricScaleSpec(current_unit="k", change_unit="k", series_key="payrolls"),
	"unrate": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="unrate", change_multiplier=100.0),
	"claims": MetricScaleSpec(current_unit="k", change_unit="k", series_key="claims"),
	"wages": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="wages", change_multiplier=100.0),
	"openings_ratio": MetricScaleSpec(current_unit="x", change_unit="x", series_key="openings_ratio"),
}


def _scale_spec(metric_id: str) -> MetricScaleSpec:
	return METRIC_SCALE_SPECS.get(metric_id, MetricScaleSpec(current_unit="value", change_unit="value", series_key=metric_id))


def _canonical_series_key(metric_id: str) -> str:
	return _scale_spec(metric_id).series_key


def _validate_scale_consistency(metric_id: str, value: float | None, change: float | None, unit: str | None, change_unit: str | None) -> list[str]:
	spec = _scale_spec(metric_id)
	issues: list[str] = []
	if unit and unit != spec.current_unit and metric_id in METRIC_SCALE_SPECS:
		issues.append(f"expected current unit {spec.current_unit}, got {unit}")
	if change_unit and change_unit != spec.change_unit and metric_id in METRIC_SCALE_SPECS:
		issues.append(f"expected change unit {spec.change_unit}, got {change_unit}")
	if value is not None and not pd.isna(value):
		abs_value = abs(float(value))
		if metric_id in {"yield_2y", "yield_10y", "ns_level", "inflation_cpi", "inflation_core_cpi", "inflation_pce", "inflation_core_pce", "inflation_ppi", "inflation_5y_be", "inflation_10y_be", "inflation_5y5y", "unrate", "wages"} and abs_value > 100.0:
			issues.append(f"value {abs_value:.2f} looks mis-scaled")
		if metric_id in {"credit_hy", "credit_ig"} and abs_value < 20.0:
			issues.append(f"credit spread {abs_value:.2f} looks like percent points, not bp")
	if change is not None and not pd.isna(change):
		abs_change = abs(float(change))
		if metric_id in {"credit_hy", "credit_ig"} and abs_change < 20.0:
			issues.append(f"change {abs_change:.2f} looks like percent points, not bp")
	return issues


def _namespace_key(field: str) -> str:
	return f"{NOTE_NAMESPACE}_{field}"


def _safe_filename(name: str) -> str:
	return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.strip()).strip("_") or "note"


def _signal_by_id(panel_analyses: list[PanelAnalysis], panel_id: str, signal_id: str) -> MacroSignal | None:
	for panel in panel_analyses:
		if panel.panel_id != panel_id:
			continue
		for signal in panel.signals:
			if signal.signal_id == signal_id:
				return signal
	return None


def _tracked_metrics(panel_analyses: list[PanelAnalysis]) -> list[WorkspaceMetric]:
	metrics: list[WorkspaceMetric] = []
	for spec in TRACKED_SIGNAL_SPECS:
		signal = _signal_by_id(panel_analyses, spec["panel_id"], spec["signal_id"])
		if signal is None:
			continue
		value = signal.value if signal.value is not None else signal.change
		unit = signal.unit if signal.value is not None else (signal.change_unit or signal.unit)
		metrics.append(
			WorkspaceMetric(
				metric_id=spec["metric_id"],
				panel_id=spec["panel_id"],
				panel_title=next((panel.title for panel in panel_analyses if panel.panel_id == spec["panel_id"]), spec["panel_id"]),
				label=spec["label"],
				value=value,
				unit=unit,
				change=signal.change,
				change_unit=signal.change_unit,
				horizon=signal.horizon,
				percentile=signal.percentile,
				standardized_change=signal.standardized_change,
				direction=signal.direction,
				interpretation=signal.interpretation,
				caveat=signal.caveat,
				as_of=signal.as_of.isoformat(),
				group_id=signal.group_id,
			)
		)
	return metrics


def _metric_map(metrics: list[WorkspaceMetric]) -> dict[str, WorkspaceMetric]:
	return {metric.metric_id: metric for metric in metrics}


def _panel_history(context: dict) -> dict[str, object]:
	return context.setdefault("panel_history", {})


def _available_history_dates(history: dict[str, object]) -> list[pd.Timestamp]:
	dates: set[pd.Timestamp] = set()
	for value in history.values():
		if isinstance(value, pd.DataFrame):
			dates.update(pd.to_datetime(value.index).dropna().to_pydatetime())
		elif isinstance(value, dict):
			for item in value.values():
				if isinstance(item, pd.DataFrame):
					dates.update(pd.to_datetime(item.index).dropna().to_pydatetime())
				elif isinstance(item, pd.Series):
					dates.update(pd.to_datetime(item.index).dropna().to_pydatetime())
	return sorted({pd.Timestamp(date) for date in dates})


def _metric_frequency(metric_id: str) -> str:
	if metric_id in {"claims", "nfci"}:
		return "weekly"
	if metric_id in {"payrolls", "unrate", "wages", "inflation_cpi", "inflation_core_cpi", "inflation_pce", "inflation_core_pce", "inflation_ppi", "cfnai", "inflation_5y_be", "inflation_10y_be", "inflation_5y5y", "openings_ratio"}:
		return "monthly"
	return "daily"


def _default_horizon_for_note_type(note_type: str) -> str:
	return NOTE_TYPE_DEFAULT_HORIZON.get(note_type, "1W")


def _selected_horizon_key() -> str:
	return _namespace_key("comparison_horizon")


def _horizon_offset(horizon: str) -> pd.DateOffset:
	return COMPARISON_HORIZONS.get(horizon, COMPARISON_HORIZONS["1M"])


def _horizon_days(horizon: str) -> int:
	return {
		"1D": 1,
		"1W": 7,
		"1M": 30,
		"3M": 91,
		"6M": 182,
		"1Y": 365,
	}.get(horizon, 30)


def _requested_comparison_date(requested_current_date: pd.Timestamp, selected_horizon: str) -> pd.Timestamp:
	return pd.Timestamp(requested_current_date) - _horizon_offset(selected_horizon)


def _effective_observation_date(series: pd.Series, requested_date: pd.Timestamp) -> pd.Timestamp | None:
	clean = pd.to_numeric(series, errors="coerce").dropna()
	if clean.empty:
		return None
	if not isinstance(clean.index, pd.DatetimeIndex):
		clean.index = pd.to_datetime(clean.index)
	clean = clean.sort_index()
	eligible = clean.loc[clean.index <= requested_date]
	if eligible.empty:
		return None
	return pd.Timestamp(eligible.index.max())


def _effective_value(series: pd.Series, requested_date: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
	effective_date = _effective_observation_date(series, requested_date)
	if effective_date is None:
		return None, None
	clean = pd.to_numeric(series, errors="coerce")
	value = clean.loc[effective_date]
	if pd.isna(value):
		return None, None
	return effective_date, float(value)


def _historical_move_distribution(
	series: pd.Series,
	selected_horizon: str,
	current_date: pd.Timestamp,
	history_years: int = HISTORY_YEARS,
) -> pd.Series:
	clean = pd.to_numeric(series, errors="coerce").dropna()
	if clean.empty:
		return pd.Series(dtype="float64")
	if not isinstance(clean.index, pd.DatetimeIndex):
		clean.index = pd.to_datetime(clean.index)
	clean = clean.sort_index()
	start_date = pd.Timestamp(current_date) - pd.DateOffset(years=history_years)
	window = clean.loc[start_date:pd.Timestamp(current_date)]
	if window.empty:
		return pd.Series(dtype="float64")
	offset = _horizon_offset(selected_horizon)
	moves: list[float] = []
	for end_date in window.index.unique():
		end_value = _latest_on_or_before(clean, pd.Timestamp(end_date))
		start_value = _latest_on_or_before(clean, pd.Timestamp(end_date) - offset)
		if end_value is None or start_value is None:
			continue
		moves.append(end_value - start_value)
	return pd.Series(moves, dtype="float64")


def _metric_category(metric: WorkspaceMetric) -> str:
	return METRIC_CATEGORY_BY_ID.get(metric.metric_id, "Other")


def _threshold_key(metric: WorkspaceMetric) -> str:
	if metric.metric_id in {"yield_2y", "yield_10y", "ns_level"}:
		return "yield_bp"
	if metric.metric_id in {"curve_2s10s", "curve_5s30s", "ns_slope", "ns_curvature"}:
		return "curve_bp"
	if metric.metric_id in {"inflation_5y_be", "inflation_10y_be", "inflation_5y5y"}:
		return "breakeven_bp"
	if metric.metric_id in {"credit_hy", "credit_ig"}:
		return "credit_spread_bp"
	if metric.metric_id in {"vix"}:
		return "vol_points"
	if metric.metric_id in {"dxy", "sp500"}:
		return "index_pct"
	if metric.metric_id in {"payrolls", "claims"}:
		return "k"
	if metric.metric_id in {"unrate", "wages", "inflation_cpi", "inflation_core_cpi", "inflation_pce", "inflation_core_pce", "inflation_ppi"}:
		return "percent"
	if metric.metric_id in {"openings_ratio"}:
		return "x"
	return "index_pct"


def _anchor_threshold(metric: WorkspaceMetric) -> float:
	return ANCHOR_THRESHOLDS.get(_threshold_key(metric), 0.25)


def _metric_quality_flag(metric: WorkspaceMetric, raw_change: float | None = None, zscore: float | None = None) -> str | None:
	raw = raw_change if raw_change is not None else metric.change
	if zscore is None:
		zscore = metric.standardized_change
	if zscore is not None and not pd.isna(zscore) and abs(float(zscore)) >= SUSPICIOUS_Z_THRESHOLD:
		return f"warning: z-score {float(zscore):+.1f} looks extreme"
	if raw is not None and not pd.isna(raw):
		abs_raw = abs(float(raw))
		if metric.metric_id in {"yield_2y", "yield_10y"} and abs_raw > 200.0:
			return f"warning: {abs_raw:.1f} bp move looks extreme"
		if metric.metric_id in {"curve_2s10s", "curve_5s30s", "ns_slope", "ns_curvature", "credit_hy", "credit_ig"} and abs_raw > 300.0:
			return f"warning: {abs_raw:.1f} bp move looks extreme"
		if metric.metric_id in {"inflation_cpi", "inflation_core_cpi", "inflation_pce", "inflation_core_pce", "inflation_ppi", "inflation_5y_be", "inflation_10y_be", "inflation_5y5y", "unrate", "wages"} and abs_raw > 2000.0:
			return f"warning: {abs_raw:.1f} bp move looks extreme"
		if metric.metric_id in {"vix"} and abs_raw > 25.0:
			return f"warning: {abs_raw:.1f} point move looks extreme"
	return None


def _comparison_basis(metric: WorkspaceMetric) -> str:
	return metric.change_horizon or "panel-native"


def _format_move_sentence(move: MoveSummary) -> str:
	current_unit = move.current_unit or move.unit
	current = f"{move.current_value:.2f} {current_unit}" if move.current_value is not None and not pd.isna(move.current_value) else "Unavailable"
	move_text = f"{move.change:+.2f} {move.change_unit}"
	if move.change is not None and move.change >= 0:
		direction = "rose"
	else:
		direction = "fell"
	return f"{move.label} {direction} to {current} ({move_text})."


def _primary_market_trigger(moves: list[MoveSummary], compare_date: pd.Timestamp) -> str:
	if not moves:
		return f"No usable observed move was found versus {compare_date.date().isoformat()}."
	lead = moves[0]
	if lead.standardized_change is not None and not pd.isna(lead.standardized_change):
		context = f"This was a {lead.standardized_change:+.1f} standard deviation move"
	else:
		context = "This move was notable relative to recent history"
	return f"{lead.label} moved {lead.change:+.2f} {lead.change_unit} versus {compare_date.date().isoformat()}. {context}."


def _coerce_text(value: object) -> str:
	if value is None or pd.isna(value):
		return ""
	text = str(value).strip()
	return "" if text.lower() in {"nan", "none", "null", "not specified", "not assessed"} else text


def _coerce_numeric(value: object) -> float | None:
	text = _coerce_text(value)
	if not text:
		return None
	try:
		return float(str(text).replace(",", ""))
	except Exception:  # noqa: BLE001
		return None


def _policy_fred_value_as_of(fred: object, series_id: str, as_of_date: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
	try:
		data = fred.get_series_as_of_date(series_id, as_of_date)
	except Exception:  # noqa: BLE001
		return None, None
	if data is None or data.empty:
		return None, None
	work = data.copy()
	if "date" not in work.columns or "realtime_start" not in work.columns or "value" not in work.columns:
		return None, None
	work = work.sort_values(["date", "realtime_start"]).groupby("date", as_index=False).tail(1)
	work = work.loc[work["value"].notna()].copy()
	if work.empty:
		return None, None
	work = work.sort_values("date")
	row = work.iloc[-1]
	return float(row["value"]), pd.Timestamp(row["date"])


def _policy_snapshot_as_of(fred: object, as_of_ts: pd.Timestamp) -> dict[str, object]:
	target_low, _ = _policy_fred_value_as_of(fred, FRED_POLICY_SERIES["target_low"], as_of_ts)
	target_high, _ = _policy_fred_value_as_of(fred, FRED_POLICY_SERIES["target_high"], as_of_ts)
	effective_rate, effective_obs = _policy_fred_value_as_of(fred, FRED_POLICY_SERIES["effective_rate"], as_of_ts)
	if effective_rate is None:
		effective_rate, effective_obs = _policy_fred_value_as_of(fred, FRED_POLICY_SERIES["fallback_rate"], as_of_ts)
	sofr, sofr_obs = _policy_fred_value_as_of(fred, FRED_POLICY_SERIES["sofr"], as_of_ts)
	balance_sheet_raw, balance_obs = _policy_fred_value_as_of(fred, FRED_POLICY_SERIES["balance_sheet"], as_of_ts)
	balance_sheet = balance_sheet_raw / 1000.0 if balance_sheet_raw is not None and not pd.isna(balance_sheet_raw) else None
	target_range = None
	if target_low is not None and target_high is not None:
		target_range = (target_low, target_high)
	return {
		"target_low": target_low,
		"target_high": target_high,
		"target_range": target_range,
		"effective_rate": effective_rate,
		"effective_rate_obs": effective_obs,
		"sofr": sofr,
		"sofr_obs": sofr_obs,
		"balance_sheet": balance_sheet,
		"balance_sheet_obs": balance_obs,
	}


def _policy_rate_text(lower: float | None, upper: float | None) -> str | None:
	if lower is None and upper is None:
		return None
	if lower is not None and upper is not None:
		if abs(upper - lower) < 0.005:
			return f"{lower:.2f}%"
		return f"{lower:.2f}% - {upper:.2f}%"
	if lower is not None:
		return f"{lower:.2f}%"
	return f"{upper:.2f}%"


def _policy_balance_sheet_text(value: float | None) -> str | None:
	if value is None or pd.isna(value):
		return None
	return f"${value / 1000.0:,.2f}T"


def _policy_change_text(current: float | None, previous: float | None, *, unit: str, precision: int = 2, threshold: float | None = None) -> str:
	if current is None or previous is None or pd.isna(current) or pd.isna(previous):
		return "No comparable data for selected date"
	delta = float(current) - float(previous)
	abs_delta = abs(delta)
	if threshold is not None and abs_delta < threshold:
		return "No meaningful change"
	if unit == "bp":
		return f"{delta * 100.0:+.0f} bp"
	if unit == "bn":
		return f"{delta:+.0f} bn"
	return f"{delta:+.{precision}f}{unit}"


def _policy_event_summary(events: list[dict[str, str]]) -> tuple[str, str, str]:
	if not events:
		return ("No scheduled Fed communications.", "Unavailable", "Fed RSS feeds can be sparse; that is fine when nothing occurred.")
	ordered = sorted(events, key=lambda item: item.get("published", ""), reverse=True)
	preview = ordered[:4]
	lines = [f"{len(ordered)} Fed communications in the comparison window."]
	lines.extend(
		f"• {item.get('published', 'date unavailable')}: {item.get('title', 'Fed communication')} ({item.get('category', 'Fed')})"
		for item in preview
	)
	if len(ordered) > len(preview):
		lines.append(f"• {len(ordered) - len(preview)} more item{'s' if len(ordered) - len(preview) != 1 else ''} in the window.")
	return ("\n".join(lines), f"{len(ordered)} events", "Only factual Fed communications are shown here.")


@st.cache_data(ttl=FRED_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_policy_context(compare_date: str, as_of_date: str) -> dict[str, object]:
	compare_ts = pd.Timestamp(compare_date)
	as_of_ts = pd.Timestamp(as_of_date)
	fed_communications = _parse_fed_rss_feeds(as_of_ts)
	context = {
		"current_policy": {},
		"previous_policy": {},
		"policy_changes": [],
		"compare_date": compare_ts.date().isoformat(),
		"target_range": None,
		"effective_fed_funds_rate": None,
		"balance_sheet_direction": None,
		"fomc_note": fed_communications[0].get("title") if fed_communications else None,
		"latest_fomc_date": fed_communications[0].get("published") if fed_communications else None,
	}
	if not FRED_API_KEY:
		context["policy_takeaway"] = "No meaningful change in the Fed policy backdrop."
		return context
	fred = Fred(api_key=FRED_API_KEY)
	current = _policy_snapshot_as_of(fred, as_of_ts)
	previous = _policy_snapshot_as_of(fred, compare_ts)
	current_target = _policy_rate_text(current.get("target_low"), current.get("target_high"))
	previous_target = _policy_rate_text(previous.get("target_low"), previous.get("target_high"))
	current_rate = current.get("effective_rate")
	previous_rate = previous.get("effective_rate")
	current_sofr = current.get("sofr")
	previous_sofr = previous.get("sofr")
	current_balance = current.get("balance_sheet")
	previous_balance = previous.get("balance_sheet")
	balance_change = None
	if current_balance is not None and previous_balance is not None and not pd.isna(current_balance) and not pd.isna(previous_balance):
		balance_change = float(current_balance) - float(previous_balance)
	policy_changes = [
		{
			"label": "Effective Fed Funds",
			"text": f"{current_rate:.2f}%" if current_rate is not None else "Unavailable",
		},
		{
			"label": "SOFR",
			"text": _policy_change_text(current_sofr, previous_sofr, unit="bp", threshold=0.005),
		},
	]
	policy_takeaway = "No meaningful change in the Fed policy backdrop."
	if current_rate is not None and previous_rate is not None and abs(float(current_rate) - float(previous_rate)) >= 0.005:
		policy_takeaway = "Fed policy backdrop changed modestly versus the selected date."
	if balance_change is not None and abs(balance_change) >= 0.5:
		policy_takeaway = "Balance sheet movements were more notable than rate changes during the comparison period."
	context.update(
		{
			"current_policy": {
				"effective_fed_funds_rate": f"{current_rate:.2f}%" if current_rate is not None else None,
				"sofr": f"{current_sofr:.2f}%" if current_sofr is not None else None,
				"balance_sheet": _policy_balance_sheet_text(current_balance),
			},
			"previous_policy": {
				"effective_fed_funds_rate": f"{previous_rate:.2f}%" if previous_rate is not None else None,
				"sofr": f"{previous_sofr:.2f}%" if previous_sofr is not None else None,
				"balance_sheet": _policy_balance_sheet_text(previous_balance),
			},
			"policy_changes": policy_changes,
			"effective_fed_funds_rate": f"{current_rate:.2f}%" if current_rate is not None else None,
			"policy_takeaway": policy_takeaway,
		}
	)
	return context


def _parse_fed_rss_feed(url: str, category: str, since: pd.Timestamp, limit: int = 6) -> list[dict[str, str]]:
	try:
		response = requests.get(url, timeout=20)
		if response.status_code != 200:
			return []
		root = ET.fromstring(response.text)
	except Exception:  # noqa: BLE001
		return []

	def _tag_name(element: ET.Element) -> str:
		return element.tag.rsplit("}", 1)[-1].lower()

	items: list[dict[str, str]] = []
	for item in root.findall(".//item") or root.findall(".//{*}entry"):
		if not isinstance(item, ET.Element):
			continue
		title = ""
		link = ""
		published = ""
		summary = ""
		for child in item:
			name = _tag_name(child)
			if name in {"title"}:
				title = _coerce_text(child.text)
			elif name in {"link"}:
				link = _coerce_text(child.get("href") or child.text)
			elif name in {"pubdate", "updated", "published"}:
				published = _coerce_text(child.text)
			elif name in {"description", "summary", "content"}:
				summary = _coerce_text(child.text)
		published_dt = pd.to_datetime(published, errors="coerce")
		if pd.notna(published_dt) and published_dt < since:
			continue
		if not title and not summary:
			continue
		items.append(
			{
				"category": category,
				"title": title or summary[:120] or "Fed communication",
				"published": published_dt.date().isoformat() if pd.notna(published_dt) else "",
				"summary": summary or title,
				"link": link,
			}
		)
		if len(items) >= limit:
			break
	return items


def _parse_fed_rss_feeds(as_of_date: pd.Timestamp) -> list[dict[str, str]]:
	since = as_of_date - pd.Timedelta(days=45)
	feeds = [
		("FOMC", [
			"https://www.federalreserve.gov/feeds/press_fomc.xml",
			"https://www.federalreserve.gov/feeds/press_all.xml",
		]),
		("Minutes", [
			"https://www.federalreserve.gov/feeds/monetary.xml",
		]),
		("Speeches", [
			"https://www.federalreserve.gov/feeds/speeches.xml",
			"https://www.federalreserve.gov/feeds/speech.xml",
		]),
		("Testimony", [
			"https://www.federalreserve.gov/feeds/testimony.xml",
		]),
		("Beige Book", [
			"https://www.federalreserve.gov/feeds/beigebook.xml",
		]),
	]
	items: list[dict[str, str]] = []
	for category, urls in feeds:
		for url in urls:
			items.extend(_parse_fed_rss_feed(url, category, since, limit=3))
			if items:
				break
	return sorted(items, key=lambda item: item.get("published", ""), reverse=True)


def get_scheduled_catalysts(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
	return _shared_scheduled_catalysts(start_date, end_date)


def get_policy_context(start_date: pd.Timestamp, end_date: pd.Timestamp, as_of_date: pd.Timestamp) -> dict[str, object]:
	return _cached_policy_context(
		pd.Timestamp(start_date).date().isoformat(),
		pd.Timestamp(as_of_date).date().isoformat(),
	)


def _latest_on_or_before(series: pd.Series, target_date: pd.Timestamp) -> float | None:
	clean = pd.to_numeric(series, errors="coerce").dropna()
	if clean.empty:
		return None
	eligible = clean.loc[clean.index <= target_date]
	if eligible.empty:
		return None
	return float(eligible.iloc[-1])


def _series_change_volatility(
	series: pd.Series,
	target_date: pd.Timestamp,
	lag: pd.Timedelta,
	lookback_days: int = 365 * 2,
	scale: float = 1.0,
) -> float | None:
	clean = pd.to_numeric(series, errors="coerce").dropna()
	if clean.empty:
		return None
	if not isinstance(clean.index, pd.DatetimeIndex):
		clean.index = pd.to_datetime(clean.index)
	clean = clean.sort_index()
	start_date = target_date - pd.Timedelta(days=lookback_days)
	window = clean.loc[start_date:target_date]
	if len(window) < 4:
		return None
	changes: list[float] = []
	for end_date in window.index.unique():
		start_anchor = pd.Timestamp(end_date) - lag
		end_value = _latest_on_or_before(clean, pd.Timestamp(end_date))
		start_value = _latest_on_or_before(clean, start_anchor)
		if end_value is None or start_value is None:
			continue
		changes.append((end_value - start_value) * scale)
	if len(changes) < 2:
		return None
	vol = float(pd.Series(changes).std())
	return vol if pd.notna(vol) and vol > 0 else None


def _calculate_zscore(change: float | None, volatility: float | None) -> float | None:
	if change is None or volatility is None or pd.isna(change) or pd.isna(volatility) or volatility <= 0:
		return None
	return float(change) / float(volatility)


def _month_end_series(frame: pd.DataFrame, column: str) -> pd.Series:
	return pd.to_numeric(frame[column], errors="coerce").resample("ME").last()


def _comparison_series(metric_id: str, history: dict[str, object]) -> pd.Series | None:
	spec = _scale_spec(metric_id)
	if spec.series_key != metric_id:
		series = _comparison_series(spec.series_key, history)
		return series * spec.value_multiplier if series is not None else None

	yield_frame = history.get("yield_curve")
	inflation_frame = history.get("inflation")
	cross_asset_frame = history.get("cross_asset")
	labor_frame = history.get("labor")
	growth_frame = history.get("growth")

	if isinstance(yield_frame, pd.DataFrame):
		if metric_id == "yield_2y":
			return pd.to_numeric(yield_frame.get("DGS2"), errors="coerce")
		if metric_id == "yield_10y":
			return pd.to_numeric(yield_frame.get("DGS10"), errors="coerce")
		if metric_id == "curve_2s10s":
			return (pd.to_numeric(yield_frame.get("DGS10"), errors="coerce") - pd.to_numeric(yield_frame.get("DGS2"), errors="coerce")) * 100.0
		if metric_id == "curve_5s30s":
			return (pd.to_numeric(yield_frame.get("DGS30"), errors="coerce") - pd.to_numeric(yield_frame.get("DGS5"), errors="coerce")) * 100.0
		if metric_id == "ns_slope":
			return (pd.to_numeric(yield_frame.get("DGS10"), errors="coerce") - pd.to_numeric(yield_frame.get("DGS2"), errors="coerce")) * 100.0
		if metric_id == "ns_curvature":
			five = pd.to_numeric(yield_frame.get("DGS5"), errors="coerce")
			two = pd.to_numeric(yield_frame.get("DGS2"), errors="coerce")
			ten = pd.to_numeric(yield_frame.get("DGS10"), errors="coerce")
			return ((five - two) - (ten - five)) * 100.0

	if isinstance(inflation_frame, dict):
		raw = inflation_frame.get("raw")
		if isinstance(raw, pd.DataFrame):
			monthly = raw[[c for c in ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PPIFID", "T5YIE", "T10YIE", "T5YIFR"] if c in raw.columns]].resample("ME").last()
			if metric_id == "inflation_cpi":
				return monthly["CPIAUCSL"].pct_change(12, fill_method=None) * 100.0
			if metric_id == "inflation_core_cpi":
				return monthly["CPILFESL"].pct_change(12, fill_method=None) * 100.0
			if metric_id == "inflation_pce":
				return monthly["PCEPI"].pct_change(12, fill_method=None) * 100.0
			if metric_id == "inflation_core_pce":
				return monthly["PCEPILFE"].pct_change(12, fill_method=None) * 100.0
			if metric_id == "inflation_ppi":
				return monthly["PPIFID"].pct_change(12, fill_method=None) * 100.0
			if metric_id == "inflation_5y_be" and "T5YIE" in raw.columns:
				return pd.to_numeric(raw["T5YIE"], errors="coerce")
			if metric_id == "inflation_10y_be" and "T10YIE" in raw.columns:
				return pd.to_numeric(raw["T10YIE"], errors="coerce")
			if metric_id == "inflation_5y5y" and "T5YIFR" in raw.columns:
				return pd.to_numeric(raw["T5YIFR"], errors="coerce")

	if isinstance(cross_asset_frame, pd.DataFrame):
		if metric_id == "nfci":
			return pd.to_numeric(cross_asset_frame.get("NFCI"), errors="coerce")
		if metric_id == "credit_hy":
			return pd.to_numeric(cross_asset_frame.get("BAMLH0A0HYM2"), errors="coerce") * 100.0
		if metric_id == "credit_ig":
			return pd.to_numeric(cross_asset_frame.get("BAMLC0A0CM"), errors="coerce") * 100.0
		if metric_id == "vix":
			return pd.to_numeric(cross_asset_frame.get("VIXCLS"), errors="coerce")
		if metric_id == "sp500":
			return pd.to_numeric(cross_asset_frame.get("SP500"), errors="coerce")
		if metric_id == "dxy":
			return pd.to_numeric(cross_asset_frame.get("DTWEXBGS"), errors="coerce")

	if isinstance(labor_frame, dict):
		monthly = labor_frame.get("monthly")
		raw = labor_frame.get("raw")
		if isinstance(monthly, pd.DataFrame):
			if metric_id == "payrolls" and "PAYEMS" in monthly.columns:
				return pd.to_numeric(monthly["PAYEMS"], errors="coerce").diff()
			if metric_id == "unrate" and "UNRATE" in monthly.columns:
				return pd.to_numeric(monthly["UNRATE"], errors="coerce")
			if metric_id == "wages" and "CES0500000003" in monthly.columns:
				return pd.to_numeric(monthly["CES0500000003"], errors="coerce").pct_change(12, fill_method=None) * 100.0
			if metric_id == "openings_ratio" and {"JTSJOL", "UNEMPLOY"} <= set(monthly.columns):
				openings = pd.to_numeric(monthly["JTSJOL"], errors="coerce")
				unemployed = pd.to_numeric(monthly["UNEMPLOY"], errors="coerce")
				return openings / unemployed
		if isinstance(raw, pd.DataFrame) and metric_id == "claims":
			weekly = pd.to_numeric(raw["ICSA"], errors="coerce")
			return weekly.rolling(4).mean() / 1_000.0

	if isinstance(growth_frame, dict):
		raw = growth_frame.get("raw")
		if isinstance(raw, pd.DataFrame) and metric_id == "cfnai" and "CFNAIMA3" in raw.columns:
			return pd.to_numeric(raw["CFNAIMA3"], errors="coerce")

	return None


def _build_move_result(
	metric: WorkspaceMetric,
	selected_horizon: str,
	requested_current_date: pd.Timestamp,
	history: dict[str, object],
) -> MoveResult:
	spec = _scale_spec(metric.metric_id)
	frequency = _metric_frequency(metric.metric_id)
	series = _comparison_series(metric.metric_id, history)
	requested_current_date = pd.Timestamp(requested_current_date)
	requested_comparison_date = _requested_comparison_date(requested_current_date, selected_horizon)
	if series is None or series.dropna().empty:
		return MoveResult(
			metric_id=metric.metric_id,
			label=metric.label,
			panel_title=metric.panel_title,
			category=_metric_category(metric),
			frequency=frequency,
			selected_horizon=selected_horizon,
			requested_current_date=requested_current_date.date().isoformat(),
			requested_comparison_date=requested_comparison_date.date().isoformat(),
			effective_current_date=None,
			effective_comparison_date=None,
			horizon_observation_count=0,
			current_value=None,
			comparison_value=None,
			raw_move=None,
			raw_move_unit=spec.change_unit,
			historical_mean_move=None,
			historical_std_move=None,
			z_score=None,
			historical_sample_count=0,
			freshness_status="Unavailable",
			data_quality_status="Series unavailable",
			direction="unavailable",
			historical_context="Historical context unavailable.",
			current_unit=spec.current_unit,
			quality_flag="Series unavailable",
		)
	current_eff_date, current_value = _effective_value(series, requested_current_date)
	comparison_eff_date, comparison_value = _effective_value(series, requested_comparison_date)
	if current_eff_date is None or comparison_eff_date is None:
		return MoveResult(
			metric_id=metric.metric_id,
			label=metric.label,
			panel_title=metric.panel_title,
			category=_metric_category(metric),
			frequency=frequency,
			selected_horizon=selected_horizon,
			requested_current_date=requested_current_date.date().isoformat(),
			requested_comparison_date=requested_comparison_date.date().isoformat(),
			effective_current_date=current_eff_date.date().isoformat() if current_eff_date is not None else None,
			effective_comparison_date=comparison_eff_date.date().isoformat() if comparison_eff_date is not None else None,
			horizon_observation_count=0,
			current_value=current_value,
			comparison_value=comparison_value,
			raw_move=None,
			raw_move_unit=spec.change_unit,
			historical_mean_move=None,
			historical_std_move=None,
			z_score=None,
			historical_sample_count=0,
			freshness_status="Unavailable",
			data_quality_status="Missing observation",
			direction="unavailable",
			historical_context="Historical context unavailable.",
			current_unit=spec.current_unit,
			quality_flag="Missing observation",
		)
	raw_move = (current_value - comparison_value) * spec.change_multiplier
	raw_move_unit = spec.change_unit or metric.change_unit or metric.unit
	horizon_obs = series.loc[(pd.to_datetime(series.index) > pd.Timestamp(comparison_eff_date)) & (pd.to_datetime(series.index) <= pd.Timestamp(current_eff_date))].dropna()
	freshness_status = "Fresh" if current_eff_date != comparison_eff_date and not horizon_obs.empty else "No new release"
	historical_moves = _historical_move_distribution(series * spec.change_multiplier if spec.change_multiplier != 1.0 else series, selected_horizon, current_eff_date)
	if historical_moves.dropna().empty:
		fallback_changes = pd.to_numeric(series, errors="coerce").dropna().diff().dropna()
		if not fallback_changes.empty:
			historical_moves = fallback_changes * spec.change_multiplier
	historical_sample_count = int(historical_moves.dropna().shape[0])
	historical_mean_move = float(historical_moves.mean()) if historical_sample_count > 0 else None
	historical_std_move = float(historical_moves.std(ddof=0)) if historical_sample_count > 0 else None
	z_score = None
	data_quality_status = None
	quality_flag = None
	if freshness_status == "No new release":
		data_quality_status = "No new release"
		quality_flag = "No new release"
	elif historical_sample_count == 0 or historical_std_move is None or pd.isna(historical_std_move) or historical_std_move <= 0:
		data_quality_status = "Insufficient history"
		quality_flag = "Insufficient history"
	else:
		z_score = _calculate_zscore(raw_move, historical_std_move)
		if z_score is not None and abs(float(z_score)) > EXTREME_ZSCORE_WARNING:
			data_quality_status = f"Extreme z-score {float(z_score):+.1f}"
			quality_flag = data_quality_status
	historical_context = _context_phrase(metric, history)
	return MoveResult(
		metric_id=metric.metric_id,
		label=metric.label,
		panel_title=metric.panel_title,
		category=_metric_category(metric),
		frequency=frequency,
		selected_horizon=selected_horizon,
		requested_current_date=requested_current_date.date().isoformat(),
		requested_comparison_date=requested_comparison_date.date().isoformat(),
		effective_current_date=current_eff_date.date().isoformat(),
		effective_comparison_date=comparison_eff_date.date().isoformat(),
		horizon_observation_count=int(horizon_obs.shape[0]),
		current_value=current_value,
		comparison_value=comparison_value,
		raw_move=raw_move,
		raw_move_unit=raw_move_unit,
		historical_mean_move=historical_mean_move,
		historical_std_move=historical_std_move,
		z_score=z_score,
		historical_sample_count=historical_sample_count,
		freshness_status=freshness_status,
		data_quality_status=data_quality_status,
		direction=_direction_phrase(metric, raw_move),
		historical_context=historical_context,
		current_unit=spec.current_unit,
		quality_flag=quality_flag,
		percentile=metric.percentile,
	)


def _comparison_move(metric: WorkspaceMetric, compare_date: pd.Timestamp, history: dict[str, object]) -> MoveResult:
	spec = _scale_spec(metric.metric_id)
	frequency = _metric_frequency(metric.metric_id)
	series = _comparison_series(metric.metric_id, history)
	requested_current_date = pd.Timestamp(metric.as_of)
	requested_comparison_date = pd.Timestamp(compare_date)
	comparison_span_days = max(1, int((requested_current_date - requested_comparison_date).days))
	selected_horizon = min(
		COMPARISON_HORIZONS,
		key=lambda horizon: abs(_horizon_days(horizon) - comparison_span_days),
	)
	if series is None or series.dropna().empty:
		return MoveResult(
			metric_id=metric.metric_id,
			label=metric.label,
			panel_title=metric.panel_title,
			category=_metric_category(metric),
			frequency=frequency,
			selected_horizon=selected_horizon,
			requested_current_date=requested_current_date.date().isoformat(),
			requested_comparison_date=requested_comparison_date.date().isoformat(),
			effective_current_date=None,
			effective_comparison_date=None,
			horizon_observation_count=0,
			current_value=None,
			comparison_value=None,
			raw_move=None,
			raw_move_unit=spec.change_unit,
			historical_mean_move=None,
			historical_std_move=None,
			z_score=None,
			historical_sample_count=0,
			freshness_status="Unavailable",
			data_quality_status="Series unavailable",
			direction="unavailable",
			historical_context="Historical context unavailable.",
			current_unit=spec.current_unit,
			quality_flag="Series unavailable",
			percentile=metric.percentile,
		)
	current_eff_date, current_value = _effective_value(series, requested_current_date)
	comparison_eff_date, comparison_value = _effective_value(series, requested_comparison_date)
	if current_eff_date is None or comparison_eff_date is None:
		return MoveResult(
			metric_id=metric.metric_id,
			label=metric.label,
			panel_title=metric.panel_title,
			category=_metric_category(metric),
			frequency=frequency,
			selected_horizon=selected_horizon,
			requested_current_date=requested_current_date.date().isoformat(),
			requested_comparison_date=requested_comparison_date.date().isoformat(),
			effective_current_date=current_eff_date.date().isoformat() if current_eff_date is not None else None,
			effective_comparison_date=comparison_eff_date.date().isoformat() if comparison_eff_date is not None else None,
			horizon_observation_count=0,
			current_value=current_value,
			comparison_value=comparison_value,
			raw_move=None,
			raw_move_unit=spec.change_unit,
			historical_mean_move=None,
			historical_std_move=None,
			z_score=None,
			historical_sample_count=0,
			freshness_status="Unavailable",
			data_quality_status="Missing observation",
			direction="unavailable",
			historical_context="Historical context unavailable.",
			current_unit=spec.current_unit,
			quality_flag="Missing observation",
			percentile=metric.percentile,
		)
	raw_move = (current_value - comparison_value) * spec.change_multiplier
	raw_move_unit = spec.change_unit or metric.change_unit or metric.unit
	horizon_obs = series.loc[(pd.to_datetime(series.index) > pd.Timestamp(comparison_eff_date)) & (pd.to_datetime(series.index) <= pd.Timestamp(current_eff_date))].dropna()
	freshness_status = "Fresh" if current_eff_date != comparison_eff_date and not horizon_obs.empty else "No new release"
	historical_moves = _historical_move_distribution(series * spec.change_multiplier if spec.change_multiplier != 1.0 else series, selected_horizon, current_eff_date)
	if historical_moves.dropna().empty:
		fallback_changes = pd.to_numeric(series, errors="coerce").dropna().diff().dropna()
		if not fallback_changes.empty:
			historical_moves = fallback_changes * spec.change_multiplier
	historical_sample_count = int(historical_moves.dropna().shape[0])
	historical_mean_move = float(historical_moves.mean()) if historical_sample_count > 0 else None
	historical_std_move = float(historical_moves.std(ddof=0)) if historical_sample_count > 0 else None
	z_score = None
	data_quality_status = None
	quality_flag = None
	if freshness_status == "No new release":
		data_quality_status = "No new release"
		quality_flag = "No new release"
	elif historical_sample_count == 0 or historical_std_move is None or pd.isna(historical_std_move) or historical_std_move <= 0:
		data_quality_status = "Insufficient history"
		quality_flag = "Insufficient history"
	else:
		z_score = _calculate_zscore(raw_move, historical_std_move)
		if z_score is not None and abs(float(z_score)) > EXTREME_ZSCORE_WARNING:
			data_quality_status = f"Extreme z-score {float(z_score):+.1f}"
			quality_flag = data_quality_status
	return MoveResult(
		metric_id=metric.metric_id,
		label=metric.label,
		panel_title=metric.panel_title,
		category=_metric_category(metric),
		frequency=frequency,
		selected_horizon=selected_horizon,
		requested_current_date=requested_current_date.date().isoformat(),
		requested_comparison_date=requested_comparison_date.date().isoformat(),
		effective_current_date=current_eff_date.date().isoformat(),
		effective_comparison_date=comparison_eff_date.date().isoformat(),
		horizon_observation_count=int(horizon_obs.shape[0]),
		current_value=current_value,
		comparison_value=comparison_value,
		raw_move=raw_move,
		raw_move_unit=raw_move_unit,
		historical_mean_move=historical_mean_move,
		historical_std_move=historical_std_move,
		z_score=z_score,
		historical_sample_count=historical_sample_count,
		freshness_status=freshness_status,
		data_quality_status=data_quality_status,
		direction=_direction_phrase(metric, raw_move),
		historical_context=_context_phrase(metric, history),
		current_unit=spec.current_unit,
		quality_flag=quality_flag,
		percentile=metric.percentile,
	)


def _move_summary_rows(moves: list[MoveSummary]) -> list[dict]:
	rows: list[dict] = []
	for move in moves:
		context = move.historical_context
		if move.quality_flag:
			context = f"{context} {move.quality_flag}"
		rows.append(
			{
				"Metric": move.label,
				"Source panel": move.panel_title,
				"Current value": f"{move.current_value:.2f} {move.current_unit or move.unit}" if move.current_value is not None and not pd.isna(move.current_value) else "Unavailable",
				"Direction": move.direction,
				"Raw move": f"{move.change:+.2f} {move.change_unit}",
				"Z-score": f"{move.standardized_change:+.2f}" if move.standardized_change is not None and not pd.isna(move.standardized_change) else "Unavailable",
				"Basis": move.change_horizon or "panel-native",
				"Context": context,
			}
		)
	return rows


def _snapshot_value(value: float | None, unit: str | None) -> str:
	if value is None or pd.isna(value):
		return "Unavailable"
	return f"{float(value):.2f} {unit or ''}".strip()


def _macro_direction_score(dimension: str, moves: list[MoveResult]) -> int:
	impulses = MACRO_NARRATIVE_IMPULSES.get(dimension, {})
	scores: list[int] = []
	for move in moves:
		if move.metric_id not in impulses or move.change is None or pd.isna(move.change) or float(move.change) == 0:
			continue
		scores.append((1 if float(move.change) > 0 else -1) * impulses[move.metric_id])
	return sum(scores)


def _move_value(moves: list[MoveResult], metric_id: str, field: str) -> float | None:
	move = next((item for item in moves if item.metric_id == metric_id), None)
	if move is None:
		return None
	value = getattr(move, field, None)
	return float(value) if value is not None and not pd.isna(value) else None


def _growth_regime(value: float | None) -> tuple[str, int | None]:
	if value is None:
		return "Unclear", None
	if value >= 0.2:
		return "Above-trend growth", 3
	if value >= -0.2:
		return "Near-trend growth", 2
	if value >= -0.7:
		return "Below-trend growth", 1
	return "Contraction risk", 0


def _inflation_regime(value: float | None) -> tuple[str, int | None]:
	if value is None:
		return "Unclear", None
	if value >= 3.0:
		return "High inflation", 3
	if value >= 2.3:
		return "Above-target inflation", 2
	if value >= 1.7:
		return "Near-target inflation", 1
	return "Below-target inflation", 0


def _financial_conditions_regime(value: float | None) -> tuple[str, int | None]:
	if value is None:
		return "Unclear", None
	if value > 0.25:
		return "Tight financial conditions", 2
	if value >= -0.25:
		return "Neutral financial conditions", 1
	return "Easy financial conditions", 0


def _labour_regime(moves: list[MoveResult], field: str) -> tuple[str, int | None]:
	scores: list[int] = []
	thresholds = {
		"payrolls": lambda value: 1 if value >= 200 else 0 if value >= 75 else -1,
		"unrate": lambda value: 1 if value < 4.0 else 0 if value <= 4.5 else -1,
		"claims": lambda value: 1 if value < 225 else 0 if value <= 300 else -1,
		"wages": lambda value: 1 if value > 4.0 else 0 if value >= 3.0 else -1,
		"openings_ratio": lambda value: 1 if value > 1.2 else 0 if value >= 0.9 else -1,
	}
	for metric_id, classifier in thresholds.items():
		value = _move_value(moves, metric_id, field)
		if value is not None:
			scores.append(classifier(value))
	if not scores:
		return "Unclear", None
	average = sum(scores) / len(scores)
	if average >= 0.5:
		return "Tight labour market", 2
	if average <= -0.5:
		return "Weak labour market", 0
	return "Balanced labour market", 1


def _dimension_regime_story(dimension: str, moves: list[MoveResult]) -> tuple[str, str, str, str]:
	if dimension == "Growth":
		previous, previous_rank = _growth_regime(_move_value(moves, "cfnai", "comparison_value"))
		current, current_rank = _growth_regime(_move_value(moves, "cfnai", "current_value"))
		reinforces = "CFNAI falls further or major activity data disappoint"
		reverses = "CFNAI rebounds or major activity data surprise higher"
	elif dimension == "Inflation":
		anchor_id = "inflation_core_pce" if _move_value(moves, "inflation_core_pce", "current_value") is not None else "inflation_core_cpi"
		previous, previous_rank = _inflation_regime(_move_value(moves, anchor_id, "comparison_value"))
		current, current_rank = _inflation_regime(_move_value(moves, anchor_id, "current_value"))
		reinforces = "Core inflation and producer prices continue in the same direction"
		reverses = "Core inflation or pipeline pressures reverse direction"
	elif dimension == "Labour":
		previous, previous_rank = _labour_regime(moves, "comparison_value")
		current, current_rank = _labour_regime(moves, "current_value")
		reinforces = "Payrolls, claims and unemployment continue in the same direction"
		reverses = "Hiring or labour-demand indicators turn decisively the other way"
	elif dimension == "Financial conditions":
		previous, previous_rank = _financial_conditions_regime(_move_value(moves, "nfci", "comparison_value"))
		current, current_rank = _financial_conditions_regime(_move_value(moves, "nfci", "current_value"))
		reinforces = "NFCI and credit, equity or volatility signals move further the same way"
		reverses = "Broad risk and credit conditions move decisively the other way"
	else:
		return "Unclear", "Unclear", "Insufficient evidence", "Insufficient evidence"

	if previous_rank is None or current_rank is None:
		narrative = "Regime is unclear because comparable evidence is unavailable"
	elif current_rank > previous_rank:
		narrative = f"Moving toward {current.lower()}"
	elif current_rank < previous_rank:
		narrative = f"Moving toward {current.lower()}"
	else:
		direction = _macro_direction_score(dimension, moves)
		if direction > 0:
			verb = {
				"Growth": "Strengthening",
				"Inflation": "Firming",
				"Labour": "Firming",
				"Financial conditions": "Easing further",
			}.get(dimension, "Strengthening")
			narrative = f"{verb} within {current.lower()}"
		elif direction < 0:
			verb = {
				"Growth": "Weakening",
				"Inflation": "Disinflating",
				"Labour": "Cooling",
				"Financial conditions": "Tightening",
			}.get(dimension, "Weakening")
			narrative = f"{verb} within {current.lower()}"
		else:
			narrative = f"Remaining in {current.lower()}"
	return previous, current, narrative, f"Reinforced if: {reinforces}||Pulled back if: {reverses}"


def _key_evidence_text(moves: list[MoveResult], limit: int = 3) -> str:
	parts: list[str] = []
	for move in moves[:limit]:
		previous = _snapshot_value(move.comparison_value, move.current_unit)
		current = _snapshot_value(move.current_value, move.current_unit)
		change = f"{move.change:+.2f} {move.change_unit}" if move.change is not None and not pd.isna(move.change) else "change unavailable"
		parts.append(f"{move.label}: {previous} → {current} ({change})")
	return "; ".join(parts) or "No comparable evidence"


def _market_reaction_items_for_ids(
	moves: list[MoveResult],
	metric_ids: tuple[str, ...],
	limit: int = 6,
) -> list[dict[str, str]]:
	move_map = {move.metric_id: move for move in moves}
	relevant = [move_map[metric_id] for metric_id in metric_ids if metric_id in move_map]
	items: list[dict[str, str]] = []
	for move in relevant[:limit]:
		if move.current_value is None or pd.isna(move.current_value):
			continue
		current = _snapshot_value(move.current_value, move.current_unit).replace(" %", "%")
		previous = _snapshot_value(move.comparison_value, move.current_unit).replace(" %", "%")
		reaction = f"{move.change:+.2f} {move.change_unit}" if move.change is not None and not pd.isna(move.change) else "Unavailable"
		tone = "up" if move.change is not None and not pd.isna(move.change) and move.change > 0 else "down" if move.change is not None and not pd.isna(move.change) and move.change < 0 else "flat"
		latest_date = pd.to_datetime(getattr(move, "effective_current_date", None), errors="coerce")
		start_date = pd.to_datetime(getattr(move, "effective_comparison_date", None), errors="coerce")
		items.append(
			{
				"label": move.label,
				"current": current,
				"previous": previous,
				"reaction": reaction,
				"latest_date": f"{latest_date:%d %b %Y}" if pd.notna(latest_date) else "Latest date unavailable",
				"start_date": f"{start_date:%d %b %Y}" if pd.notna(start_date) else "Start date unavailable",
				"tone": tone,
			}
		)
	return items


def _relevant_market_reaction_items(dimension: str, moves: list[MoveResult], limit: int = 6) -> list[dict[str, str]]:
	return _market_reaction_items_for_ids(moves, NARRATIVE_MARKET_IDS.get(dimension, ()), limit=limit)


def _relevant_market_reaction_text(dimension: str, moves: list[MoveResult], limit: int = 6) -> str:
	items = _relevant_market_reaction_items(dimension, moves, limit=limit)
	return "; ".join(f"{item['label']}: {item['current']} ({item['reaction']})" for item in items) or "No relevant market reaction is available"


def _render_market_reaction_bubbles(items: list[dict[str, str]], horizon: str) -> None:
	if not items:
		st.caption("No relevant market reaction is available.")
		return
	bubbles = "".join(
		f"""
		<div class="reaction-bubble {html.escape(item['tone'])}">
			<span class="reaction-label">{html.escape(item['label'])}</span>
			<strong>{html.escape(item['reaction'])}</strong>
			<small>{html.escape(item['previous'])} ({html.escape(item['start_date'])}) → {html.escape(item['current'])} ({html.escape(item['latest_date'])})</small>
		</div>
		"""
		for item in items
	)
	st.html(
		f"""
		<style>
			.reaction-bubbles {{ display: flex; flex-wrap: wrap; gap: .48rem; margin: .2rem 0 .7rem; }}
			.reaction-bubble {{
				display: grid; grid-template-columns: minmax(7rem, auto) auto; align-items: center;
				gap: .12rem .55rem; padding: .48rem .65rem; border: 1px solid rgba(100,116,139,.22);
				border-radius: 999px; background: rgba(148,163,184,.08); line-height: 1.15;
			}}
			.reaction-bubble.up {{ border-color: rgba(37,99,235,.24); background: rgba(37,99,235,.07); }}
			.reaction-bubble.down {{ border-color: rgba(180,83,9,.22); background: rgba(245,158,11,.08); }}
			.reaction-label {{ color: #334155; font-size: .73rem; font-weight: 650; }}
			.reaction-bubble strong {{ color: #14213d; font-size: .78rem; text-align: right; white-space: nowrap; }}
			.reaction-bubble small {{ grid-column: 1 / -1; color: #64748b; font-size: .64rem; }}
		</style>
		<div class="reaction-bubbles" aria-label="Relevant market reaction over {html.escape(horizon)}">{bubbles}</div>
		"""
	)


def _upcoming_catalysts_by_dimension(upcoming_events: pd.DataFrame) -> dict[str, str]:
	buckets: dict[str, list[str]] = {dimension: [] for dimension in (*MACRO_UPDATE_GROUPS, "Policy")}
	if not isinstance(upcoming_events, pd.DataFrame) or upcoming_events.empty:
		return {dimension: "None identified" for dimension in buckets}
	for event in upcoming_events.itertuples(index=False):
		name = _coerce_text(getattr(event, "event_name", ""))
		if not name:
			continue
		text = name.lower()
		if any(term in text for term in ("gross domestic", "gdp", "industrial", "trade", "corporate profits", "productivity")):
			dimension = "Growth"
		elif any(term in text for term in ("consumer price", "producer price", "personal income", "pce")):
			dimension = "Inflation"
		elif any(term in text for term in ("employment situation", "job openings", "employment cost", "payroll")):
			dimension = "Labour"
		elif "fomc" in text or "fed" in text:
			dimension = "Policy"
		elif "auction" in text:
			dimension = "Financial conditions"
		else:
			continue
		date_value = pd.to_datetime(getattr(event, "event_date", None), errors="coerce")
		label = f"{name} ({date_value:%d %b})" if pd.notna(date_value) else name
		if label not in buckets[dimension]:
			buckets[dimension].append(label)
	return {dimension: "; ".join(events[:3]) if events else "None identified" for dimension, events in buckets.items()}


def _scenario_linked_to_catalysts(catalysts: str, condition: str) -> str:
	condition = condition.strip().rstrip(".")
	if not catalysts or catalysts == "None identified":
		return f"No scheduled catalyst identified — {condition}"
	return f"{catalysts} — {condition}"


def _dashboard_macro_update_rows(
	moves: list[MoveResult],
	policy_context: dict[str, object],
	upcoming_events: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
	move_map = {move.metric_id: move for move in moves}
	catalysts = _upcoming_catalysts_by_dimension(upcoming_events if upcoming_events is not None else pd.DataFrame())
	rows: list[dict[str, object]] = []
	for dimension, metric_ids in MACRO_UPDATE_GROUPS.items():
		available = [move_map[metric_id] for metric_id in metric_ids if metric_id in move_map]
		available = [move for move in available if move.current_value is not None and not pd.isna(move.current_value)]
		if not available:
			continue
		selected = available[:3]
		previous_regime, current_regime, narrative, scenario_text = _dimension_regime_story(dimension, available)
		reinforces, reverses = scenario_text.split("||", maxsplit=1)
		dimension_catalysts = catalysts.get(dimension, "None identified")
		rows.append(
			{
				"Macro dimension": dimension,
				"Regime path": f"{previous_regime} → {current_regime}",
				"Current narrative": narrative,
				"Key evidence": _key_evidence_text(selected),
				"Relevant market reaction": _relevant_market_reaction_text(dimension, moves),
				"_market_reaction_items": _relevant_market_reaction_items(dimension, moves),
				"Upcoming catalysts": dimension_catalysts,
				"Reinforces narrative if": _scenario_linked_to_catalysts(
					dimension_catalysts, reinforces.removeprefix("Reinforced if: ")
				),
				"Pulls narrative back if": _scenario_linked_to_catalysts(
					dimension_catalysts, reverses.removeprefix("Pulled back if: ")
				),
			}
		)

	current_policy = policy_context.get("current_policy") or {}
	previous_policy = policy_context.get("previous_policy") or {}
	policy_fields = (
		("Effective Fed Funds", "effective_fed_funds_rate"),
		("SOFR", "sofr"),
		("Balance sheet", "balance_sheet"),
	)
	if any(current_policy.get(key) or previous_policy.get(key) for _, key in policy_fields):
		current_rate = _coerce_numeric(_coerce_text(current_policy.get("effective_fed_funds_rate")).replace("%", ""))
		previous_rate = _coerce_numeric(_coerce_text(previous_policy.get("effective_fed_funds_rate")).replace("%", ""))
		if current_rate is not None and previous_rate is not None and current_rate < previous_rate:
			policy_narrative = "Moving toward an easing policy regime"
		elif current_rate is not None and previous_rate is not None and current_rate > previous_rate:
			policy_narrative = "Moving toward a tightening policy regime"
		else:
			policy_narrative = "Remaining in a policy-hold regime"
		policy_catalysts = catalysts.get("Policy", "None identified")
		rows.append(
			{
				"Macro dimension": "Policy",
				"Regime path": f"Policy rate {previous_policy.get('effective_fed_funds_rate') or 'unavailable'} → {current_policy.get('effective_fed_funds_rate') or 'unavailable'}",
				"Current narrative": policy_narrative,
				"Key evidence": _coerce_text(policy_context.get("policy_takeaway")) or "No policy update available.",
				"Relevant market reaction": _relevant_market_reaction_text("Policy", moves),
				"_market_reaction_items": _relevant_market_reaction_items("Policy", moves),
				"Upcoming catalysts": policy_catalysts,
				"Reinforces narrative if": _scenario_linked_to_catalysts(
					policy_catalysts, "Fed communication and policy-sensitive rates continue in the same direction"
				),
				"Pulls narrative back if": _scenario_linked_to_catalysts(
					policy_catalysts, "Fed communication or inflation and labour data shift the expected path the other way"
				),
			}
		)
	return rows


def _market_reaction_rows(moves: list[MoveResult]) -> list[dict[str, str]]:
	move_map = {move.metric_id: move for move in moves}
	rows: list[dict[str, str]] = []
	for metric_id in MARKET_REACTION_IDS:
		move = move_map.get(metric_id)
		if move is None or move.current_value is None or pd.isna(move.current_value):
			continue
		rows.append(
			{
				"Asset signal": move.label,
				"Prior": _snapshot_value(move.comparison_value, move.current_unit),
				"Latest": _snapshot_value(move.current_value, move.current_unit),
				"Reaction": (
					f"{move.change:+.2f} {move.change_unit}"
					if move.change is not None and not pd.isna(move.change)
					else "Unavailable"
				),
				"Observation date": move.effective_current_date or "Unavailable",
			}
		)
	return rows


def _evidence_rows_markdown(title: str, rows: list[dict[str, object]]) -> str:
	if not rows:
		return ""
	columns = [column for column in rows[0] if not column.startswith("_")]
	lines = [f"## {title}", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
	for row in rows:
		lines.append("| " + " | ".join(_note_table_cell(row.get(column)) for column in columns) + " |")
	return "\n".join(lines)


def _anchor_display_rows(metrics: list[WorkspaceMetric], history: dict[str, object], exclude_ids: set[str] | None = None) -> list[dict]:
	anchor_ids = _anchor_metric_ids(metrics, exclude_ids=exclude_ids)
	rows: list[dict] = []
	for metric in metrics:
		if metric.metric_id not in anchor_ids:
			continue
		if metric.change is None or pd.isna(metric.change):
			continue
		rows.append(
			{
				"Metric": metric.label,
				"Source panel": metric.panel_title,
				"Current value": _value_text(metric),
				"Direction": "stable",
				"Raw move": f"{float(metric.change):+.2f} {metric.change_unit or metric.unit}",
				"Z-score": f"{float(metric.standardized_change):+.2f}" if metric.standardized_change is not None and not pd.isna(metric.standardized_change) else "Unavailable",
				"Basis": metric.horizon or "panel-native",
				"Context": _context_phrase(metric, history),
			}
		)
	return rows[:4]


def _anchor_metric_ids(metrics: list[WorkspaceMetric], exclude_ids: set[str] | None = None) -> set[str]:
	anchor_priority = {
		"inflation_10y_be",
		"inflation_5y5y",
		"unrate",
		"credit_ig",
		"credit_hy",
		"yield_10y",
		"curve_2s10s",
		"vix",
		"dxy",
		"payrolls",
	}
	exclude_ids = exclude_ids or set()
	return {
		metric.metric_id
		for metric in metrics
		if metric.metric_id in anchor_priority
		and metric.metric_id not in exclude_ids
		and getattr(metric, "freshness_status", "Fresh") == "Fresh"
		and getattr(metric, "quality_flag", None) is None
		and metric.change is not None
		and not pd.isna(metric.change)
		and abs(float(metric.change)) < _anchor_threshold(metric)
		and (
			metric.standardized_change is None
			or pd.isna(metric.standardized_change)
			or abs(float(metric.standardized_change)) < Z_THRESHOLD
		)
	}


def _default_comparison_date(as_of: pd.Timestamp) -> pd.Timestamp:
	return as_of - pd.DateOffset(months=1)


def _value_text(metric: WorkspaceMetric) -> str:
	if metric.value is None or pd.isna(metric.value):
		return "Unavailable"
	if metric.unit == "%":
		return f"{metric.value:.2f}%"
	if metric.unit in {"bp", "index", "x", "z", "k"}:
		return f"{metric.value:.2f} {metric.unit}"
	return f"{metric.value:.2f} {metric.unit}".strip()


def _change_text(metric: WorkspaceMetric) -> str:
	if metric.change is None or pd.isna(metric.change):
		return "Unavailable"
	horizon = f" ({metric.horizon})" if metric.horizon else ""
	if metric.change_unit == "%":
		return f"{metric.change:+.2f}%{horizon}"
	if metric.change_unit:
		return f"{metric.change:+.2f} {metric.change_unit}{horizon}"
	return f"{metric.change:+.2f}{horizon}"


def _direction_phrase(metric: WorkspaceMetric, raw_change: float) -> str:
	positive, negative = MOVE_DIRECTION_HINTS.get(metric.metric_id, ("higher", "lower"))
	if raw_change > 0:
		return positive
	if raw_change < 0:
		return negative
	return "unchanged"


def _historical_context(metric: WorkspaceMetric) -> str:
	if metric.percentile is None or pd.isna(metric.percentile):
		return "Historical context unavailable."
	if metric.percentile >= 90:
		return "near the top of its available history"
	if metric.percentile >= 75:
		return "in the upper quartile of its available history"
	if metric.percentile <= 10:
		return "near the bottom of its available history"
	if metric.percentile <= 25:
		return "in the lower quartile of its available history"
	return "near the middle of its available history"


def _value_percentile(series: pd.Series, value: float | None) -> float | None:
	clean = pd.to_numeric(series, errors="coerce").dropna()
	if clean.empty or value is None or pd.isna(value):
		return None
	return float((clean <= float(value)).mean() * 100.0)


def _context_phrase(metric: WorkspaceMetric, history: dict[str, object]) -> str:
	spec = _scale_spec(metric.metric_id)
	series = _comparison_series(spec.series_key, history)
	if series is None or series.dropna().empty:
		return "Historical context unavailable."
	current_value = _latest_on_or_before(series, pd.Timestamp(metric.as_of))
	percentile = _value_percentile(series, current_value)
	if percentile is None:
		return "Historical context unavailable."
	if percentile >= 90:
		return "near the top of history"
	if percentile >= 75:
		return "in the upper quartile of history"
	if percentile <= 10:
		return "near the bottom of history"
	if percentile <= 25:
		return "in the lower quartile of history"
	return "roughly mid-range in history"


def _meaningful_threshold(metric: WorkspaceMetric) -> float:
	if metric.unit == "bp" or metric.change_unit == "bp":
		return 5.0
	if metric.unit == "%" or metric.change_unit == "%":
		return 0.1
	if metric.unit == "k":
		return 10.0
	if metric.unit == "index":
		return 0.5
	if metric.unit == "x":
		return 0.05
	return 0.25


def _move_score(metric: WorkspaceMetric) -> float:
	if metric.standardized_change is not None and not pd.isna(metric.standardized_change):
		return abs(float(metric.standardized_change))
	effective_change = metric.change if metric.change is not None and not pd.isna(metric.change) else (metric.value if metric.metric_id == "payrolls" and metric.value is not None and not pd.isna(metric.value) else None)
	if effective_change is not None:
		return abs(float(effective_change))
	return 0.0


def _pattern_signal_detail(spec: dict, metric: WorkspaceMetric) -> str:
	sign = "up" if metric.change and metric.change > 0 else "down" if metric.change and metric.change < 0 else "flat"
	expected = spec["expected"]
	if expected == sign:
		return f"{spec['label']} ({metric.label}): aligned with expected {expected} move"
	return f"{spec['label']} ({metric.label}): moved {sign}, expected {expected}"


def _top_moves(metrics: list[WorkspaceMetric], history: dict[str, object]) -> list[MoveSummary]:
	moves: list[MoveSummary] = []
	for metric in metrics:
		if getattr(metric, "freshness_status", "Fresh") != "Fresh":
			continue
		if getattr(metric, "quality_flag", None) is not None:
			continue
		effective_change = metric.change if metric.change is not None and not pd.isna(metric.change) else (metric.value if metric.metric_id == "payrolls" and metric.value is not None and not pd.isna(metric.value) else None)
		if effective_change is None:
			continue
		moves.append(
			MoveResult(
				metric_id=metric.metric_id,
				label=metric.label,
				panel_title=metric.panel_title,
				category=_metric_category(metric),
				frequency=_metric_frequency(metric.metric_id),
				selected_horizon=getattr(metric, "change_horizon", None) or metric.horizon,
				requested_current_date=getattr(metric, "requested_current_date", metric.as_of),
				requested_comparison_date=getattr(metric, "requested_comparison_date", metric.as_of),
				effective_current_date=getattr(metric, "effective_current_date", None),
				effective_comparison_date=getattr(metric, "effective_comparison_date", None),
				horizon_observation_count=getattr(metric, "horizon_observation_count", 0),
				current_value=metric.value,
				comparison_value=getattr(metric, "comparison_value", None),
				raw_move=float(effective_change),
				raw_move_unit=metric.change_unit or metric.unit,
				historical_mean_move=getattr(metric, "historical_mean_move", None),
				historical_std_move=getattr(metric, "historical_std_move", None),
				z_score=float(metric.standardized_change) if metric.standardized_change is not None and not pd.isna(metric.standardized_change) else None,
				historical_sample_count=getattr(metric, "historical_sample_count", 0),
				freshness_status=getattr(metric, "freshness_status", "Fresh"),
				data_quality_status=getattr(metric, "data_quality_status", None),
				direction=_direction_phrase(metric, float(effective_change)),
				historical_context=_context_phrase(metric, history),
				percentile=metric.percentile,
				current_unit=metric.unit,
				quality_flag=_metric_quality_flag(metric, float(effective_change), float(metric.standardized_change) if metric.standardized_change is not None and not pd.isna(metric.standardized_change) else None),
			)
		)
	metric_lookup = _metric_map(metrics)
	moves.sort(
		key=lambda item: (
			-_move_score(metric_lookup[item.metric_id]),
			CATEGORY_ORDER.index(_metric_category(metric_lookup[item.metric_id]))
			if _metric_category(metric_lookup[item.metric_id]) in CATEGORY_ORDER
			else len(CATEGORY_ORDER),
		)
	)
	return moves[:NOTE_WORKSPACE_MAX_MOVES]


def _pattern_row(pattern: dict, metric_map: dict[str, WorkspaceMetric]) -> PatternEvaluation | None:
	signals = pattern["signals"]
	aligned: list[str] = []
	conflicting: list[str] = []
	flat: list[str] = []
	missing: list[str] = []
	total_weight = 0.0
	matched_weight = 0.0
	conflict_weight = 0.0
	for spec in signals:
		metric = metric_map.get(spec["metric_id"])
		if metric is None or getattr(metric, "freshness_status", "Fresh") != "Fresh" or getattr(metric, "quality_flag", None) is not None or metric.change is None or pd.isna(metric.change):
			missing.append(spec["label"])
			continue
		weight = max(abs(float(metric.standardized_change)) if metric.standardized_change is not None and not pd.isna(metric.standardized_change) else abs(float(metric.change)), 0.1)
		total_weight += weight
		sign = "up" if metric.change > 0 else "down" if metric.change < 0 else "flat"
		if sign == spec["expected"]:
			aligned.append(spec["label"])
			matched_weight += weight
		elif sign == "flat":
			flat.append(spec["label"])
		else:
			conflicting.append(spec["label"])
			conflict_weight += weight
	if not aligned and not conflicting and not flat:
		return None
	available = len(aligned) + len(conflicting) + len(flat)
	if available == 0:
		return None
	match_ratio = len(aligned) / available
	conflict_ratio = len(conflicting) / available
	weight_ratio = matched_weight / total_weight if total_weight > 0 else 0.0
	confidence = max(0.0, min(1.0, 0.55 * match_ratio + 0.35 * weight_ratio - 0.2 * conflict_ratio))
	if available < 2:
		status = "Insufficient data"
	elif match_ratio >= 0.8 and conflict_ratio == 0:
		status = "Fully aligned"
	elif match_ratio >= 0.6 and conflict_ratio <= 0.2:
		status = "Mostly aligned"
	elif conflict_ratio >= 0.6 and match_ratio <= 0.2:
		status = "Mostly conflicting"
	else:
		status = "Mixed alignment"
	takeaway_bits = []
	if aligned:
		takeaway_bits.append(f"Aligned: {', '.join(aligned[:3])}")
	if conflicting:
		takeaway_bits.append(f"Conflicting: {', '.join(conflicting[:3])}")
	if flat:
		takeaway_bits.append(f"Flat: {', '.join(flat[:3])}")
	if missing:
		takeaway_bits.append(f"Unavailable: {', '.join(missing[:2])}")
	takeaway = "\n".join(takeaway_bits) if takeaway_bits else "No sub-signals available."
	return PatternEvaluation(
		pattern_name=pattern["name"],
		alignment_status=status,
		confidence_score=round(confidence, 2),
		aligned_signals=aligned,
		conflicting_signals=conflicting,
		flat_signals=flat,
		unavailable_signals=missing,
		takeaway=takeaway,
	)


def _pattern_to_row(pattern: PatternEvaluation) -> dict[str, str]:
	return {
		"Pattern": pattern.pattern_name,
		"Alignment": pattern.alignment_status,
		"Confidence": f"{pattern.confidence_score:.2f}",
		"Takeaway": pattern.takeaway,
	}


def _pattern_gap_text(pattern: PatternEvaluation) -> str:
	parts = []
	if pattern.conflicting_signals:
		parts.append("Review whether the conflicting signals or a second catalyst better explain the move.")
	if pattern.unavailable_signals:
		parts.append("Refresh the missing signals before treating this pattern as established.")
	if not parts:
		parts.append("The current evidence set is reasonably complete, but it still deserves manual verification.")
	return " ".join(parts)


def _build_research_gaps(
	primary_move: MoveSummary | None,
	scheduled_catalysts: pd.DataFrame,
	policy_context: dict[str, object],
	patterns: list[PatternEvaluation],
) -> list[ResearchGap]:
	gaps: list[ResearchGap] = []
	if primary_move is not None:
		gaps.append(
			ResearchGap(
				task="Verify the intraday sequence around the largest observed move.",
				reason="Calendar proximity and related headlines can identify candidates, but they do not establish which information caused the repricing.",
				related=primary_move.label,
			)
		)
	if scheduled_catalysts.empty:
		gaps.append(
			ResearchGap(
				task="Review the event calendar for nearby catalysts.",
				reason="No scheduled catalyst feed is currently configured, so timing remains unverified.",
				related="events since the previous snapshot",
			)
		)
	if not policy_context.get("target_range") and not policy_context.get("effective_fed_funds_rate"):
		gaps.append(
			ResearchGap(
				task="Review the latest FOMC statement or policy communication.",
				reason="Policy context is currently unavailable, so the panel cannot confirm how the move fits the Fed backdrop.",
				related="Fed and policy context",
			)
		)
	if patterns:
		leading = patterns[0]
		if leading.alignment_status in {"Mixed alignment", "Mostly conflicting"}:
			gaps.append(
				ResearchGap(
					task="Check whether the leading interpretation is being confirmed across rates, credit, FX, and equities.",
					reason=_pattern_gap_text(leading),
					related=leading.pattern_name,
				)
			)
	if primary_move is not None and primary_move.quality_flag:
		gaps.append(
			ResearchGap(
				task="Refresh the affected series and verify the observation date.",
				reason=primary_move.quality_flag,
				related=primary_move.label,
			)
		)
	return gaps[:6]


def _note_text(value: object, default: str = "No content entered.") -> str:
	if value is None or pd.isna(value):
		return default
	text = str(value).strip()
	return text or default


def _note_table_cell(value: object) -> str:
	if value is None or pd.isna(value):
		return ""
	if isinstance(value, pd.Timestamp):
		return value.date().isoformat()
	if hasattr(value, "isoformat") and not isinstance(value, str):
		try:
			return value.isoformat()
		except Exception:  # noqa: BLE001
			pass
	return str(value).strip().replace("|", "\\|").replace("\n", " ")


def _default_next_catalysts_df() -> pd.DataFrame:
	return pd.DataFrame([{"Catalyst": "", "Date": None, "What to watch": "", "Market implication": ""}])


def _default_trade_candidates_df() -> pd.DataFrame:
	return pd.DataFrame(
		[
			{
				"Candidate": candidate,
				"Expression": "",
				"Why it expresses the gap": "",
				"Catalyst": "",
				"Key risk": "",
			}
			for candidate in TRADE_CANDIDATE_IDS
		]
	)


def _trade_candidates_markdown(candidates_df: pd.DataFrame | None) -> str:
	if candidates_df is None or not isinstance(candidates_df, pd.DataFrame) or candidates_df.empty:
		return ""
	columns = ["Candidate", "Expression", "Why it expresses the gap", "Catalyst", "Key risk"]
	rows: list[list[str]] = []
	for _, row in candidates_df.reindex(columns=columns, fill_value="").iterrows():
		values = [_note_table_cell(row.get(column)) for column in columns]
		meaningful = [value for column, value in zip(columns, values) if column != "Candidate"]
		meaningful = [value for value in meaningful if value.lower() not in {"", "not specified", "not assessed"}]
		if meaningful:
			rows.append(values)
	if not rows:
		return ""
	lines = [
		"## Candidate Trade Expressions",
		"| Candidate | Expression | Why it expresses the gap | Catalyst | Key risk |",
		"| --- | --- | --- | --- | --- |",
	]
	for row in rows:
		lines.append("| " + " | ".join(row) + " |")
	return "\n".join(lines)


def _next_catalysts_markdown(catalysts_df: pd.DataFrame | None) -> str:
	if catalysts_df is None or not isinstance(catalysts_df, pd.DataFrame) or catalysts_df.empty:
		return ""
	columns = ["Catalyst", "Date", "What to watch", "Market implication"]
	rows: list[list[str]] = []
	for _, row in catalysts_df.reindex(columns=columns, fill_value="").iterrows():
		values = [_note_table_cell(row.get(column)) for column in columns]
		if any(values):
			rows.append(values)
	if not rows:
		return ""
	lines = [
		"## Next Catalysts",
		"| Catalyst | Date | What to watch | Market implication |",
		"| --- | --- | --- | --- |",
	]
	for row in rows:
		lines.append("| " + " | ".join(row) + " |")
	return "\n".join(lines)


def _policy_card_body(policy_context: dict[str, object]) -> str:
	current = policy_context.get("current_policy") or {}
	previous = policy_context.get("previous_policy") or {}
	compare_date = _coerce_text(policy_context.get("compare_date")) or "selected date"
	takeaway = _coerce_text(policy_context.get("policy_takeaway")) or "No meaningful change in the Fed policy backdrop."

	lines: list[str] = []
	lines.append("Current")
	current_fields = []
	if current.get("effective_fed_funds_rate"):
		current_fields.append(f"Effective Fed Funds: {current['effective_fed_funds_rate']}")
	if current.get("sofr"):
		current_fields.append(f"SOFR: {current['sofr']}")
	if current.get("balance_sheet"):
		current_fields.append(f"Balance sheet: {current['balance_sheet']}")
	if current_fields:
		lines.extend(f"• {field}" for field in current_fields)
	else:
		lines.append("• Current policy data unavailable.")

	lines.append("")
	lines.append(f"Data on {compare_date}")
	selected_fields = []
	if previous.get("effective_fed_funds_rate"):
		selected_fields.append(f"Effective Fed Funds: {previous['effective_fed_funds_rate']}")
	if previous.get("sofr"):
		selected_fields.append(f"SOFR: {previous['sofr']}")
	if previous.get("balance_sheet"):
		selected_fields.append(f"Balance sheet: {previous['balance_sheet']}")
	if selected_fields:
		lines.extend(f"• {field}" for field in selected_fields)
	else:
		lines.append("• Selected-date policy data unavailable.")

	lines.append("")
	lines.append("Takeaway")
	lines.append(takeaway)
	return "\n".join(lines)


def _candidate_trigger_rows(
	recent_rate_events: pd.DataFrame,
	rate_headlines: pd.DataFrame,
	policy_context: dict[str, object],
	limit: int = 5,
) -> list[dict[str, str]]:
	"""Build a short attribution queue without treating calendar proximity as causation."""
	rows: list[dict[str, str]] = []
	if isinstance(recent_rate_events, pd.DataFrame) and not recent_rate_events.empty:
		work = recent_rate_events.copy()
		work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce")
		work = work.dropna(subset=["event_date"]).sort_values("event_date", ascending=False)
		for event in work.itertuples(index=False):
			move = _coerce_text(getattr(event, "curve_move", ""))
			has_move = bool(move and move.lower() != "unavailable")
			rows.append(
				{
					"Candidate trigger": _coerce_text(getattr(event, "event_name", "")) or "Scheduled rates event",
					"Timing and evidence": (
						f"{pd.Timestamp(event.event_date):%d %b %Y} · {move}"
						if has_move
						else f"{pd.Timestamp(event.event_date):%d %b %Y} · same-day move unavailable"
					),
					"Attribution status": "Plausible" if has_move else "Unverified",
				}
			)
			if len(rows) >= max(1, limit - 1):
				break

	policy_takeaway = _coerce_text(policy_context.get("policy_takeaway"))
	if policy_takeaway and policy_takeaway != "No meaningful change in the Fed policy backdrop.":
		rows.append(
			{
				"Candidate trigger": "Change in the Fed policy backdrop",
				"Timing and evidence": policy_takeaway,
				"Attribution status": "Plausible",
			}
		)

	if isinstance(rate_headlines, pd.DataFrame) and not rate_headlines.empty and len(rows) < limit:
		latest = rate_headlines.iloc[0]
		rows.append(
			{
				"Candidate trigger": "Rates-news narrative",
				"Timing and evidence": _coerce_text(latest.get("title")) or "Related rates coverage was published in the window.",
				"Attribution status": "Unverified",
			}
		)
	return rows[:limit]


def _event_card_summary(frame: pd.DataFrame) -> tuple[str, str, str]:
	if frame.empty:
		return (
			"No scheduled catalyst feed is currently available.",
			"Unavailable",
			"This card uses FRED, BEA, Federal Reserve, and U.S. Treasury calendars.",
		)
	work = frame.copy()
	work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce")
	work = work.loc[work["event_date"].notna()].copy()
	work = work.sort_values(["event_date", "importance", "event_name"], ascending=[True, False, True])
	if work.empty:
		return (
			"No scheduled catalyst feed is currently available.",
			"Unavailable",
			"This card uses FRED, BEA, Federal Reserve, and U.S. Treasury calendars.",
		)
	first = work.iloc[0]
	last = work.iloc[-1]
	source_counts = work["source"].value_counts().to_dict()
	if len(work) == 1:
		body = [
			f"{pd.Timestamp(first['event_date']).date().isoformat()}: {first['event_name']} ({first['source']})",
		]
	else:
		body = [
			f"{len(work)} calendar events that could help explain moves between {pd.Timestamp(first['event_date']).date().isoformat()} and {pd.Timestamp(last['event_date']).date().isoformat()}.",
			f"First event: {first['event_name']} on {pd.Timestamp(first['event_date']).date().isoformat()} ({first['source']}).",
		]
		preview = work.head(3)
		preview_bits = [
			f"{pd.Timestamp(row.event_date).date().isoformat()}: {row.event_name}"
			for row in preview.itertuples(index=False)
		]
		if preview_bits:
			body.append("Preview: " + "; ".join(preview_bits))
		if len(source_counts) > 1:
			body.append("Sources: " + ", ".join(f"{source} x{count}" for source, count in source_counts.items()))
	status = f"{len(work)} catalyst{'s' if len(work) != 1 else ''}"
	footer = "These are official calendar entries that can swing market expectations, not forecasts."
	return ("\n".join(body), status, footer)


def _archive_note(title: str, markdown: str) -> Path:
	NOTE_WORKSPACE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
	slug = _safe_filename(title)
	path = NOTE_WORKSPACE_ARCHIVE_DIR / f"{stamp}_{slug}.md"
	path.write_text(markdown, encoding="utf-8")
	return path


def _load_archive_notes() -> list[Path]:
	if not NOTE_WORKSPACE_ARCHIVE_DIR.exists():
		return []
	return sorted(NOTE_WORKSPACE_ARCHIVE_DIR.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)


def _archive_browser() -> None:
	notes = _load_archive_notes()
	if not notes:
		st.caption("No archived trade ideas yet.")
		return
	options = [f"{path.stat().st_mtime_ns} - {path.stem}" for path in notes]
	choice = st.selectbox("Browse archived trade ideas", options=options, key=_namespace_key("archive_choice"))
	path = notes[options.index(choice)]
	try:
		content = path.read_text(encoding="utf-8")
	except Exception:  # noqa: BLE001
		content = "(Unable to read archived note.)"
	with st.expander("Archived trade idea", expanded=False):
		st.markdown(content)


def _build_note(
	title: str,
	note_type: str,
	trigger_text: str,
	bottom_line_text: str,
	what_moved_text: str,
	why_text: str,
	broader_context_text: str,
	trade_expression: dict[str, object],
	trade_rationale: dict[str, object],
	invalidation_text: str,
	next_catalysts_df: pd.DataFrame | None,
	market_pricing_text: str = "",
	expectation_gap_text: str = "",
	trade_candidates_df: pd.DataFrame | None = None,
	selected_candidate: str = "",
	prior_view_text: str = "",
	signal_update: str = "",
	macro_update_rows: list[dict[str, object]] | None = None,
	market_reaction_rows: list[dict[str, str]] | None = None,
) -> str:
	trade_expression = trade_expression or {}
	trade_rationale = trade_rationale or {}
	lines = [
		f"# {title}",
		f"**Idea type:** {_note_text(note_type, 'Macro trade')}",
		"",
		"## Prior Macro View",
		_note_text(prior_view_text),
	]
	macro_evidence_markdown = _evidence_rows_markdown("New Dashboard Evidence", macro_update_rows or [])
	if macro_evidence_markdown:
		lines.extend(["", macro_evidence_markdown])
	lines.extend(
		[
		"",
		"## Signal Update",
		f"**Assessment:** {_note_text(signal_update, 'Unclear')}",
		"",
		"## Updated Macro Thesis",
		_note_text(trigger_text),
		"",
		"## Macro Implications",
		_note_text(bottom_line_text),
		]
	)
	market_reaction_markdown = _evidence_rows_markdown("Observed Market Reaction", market_reaction_rows or [])
	if market_reaction_markdown:
		lines.extend(["", market_reaction_markdown])
	lines.extend(
		[
		"",
		"## Current Market Pricing",
		_note_text(market_pricing_text),
		"",
		"## Expectation Gap",
		_note_text(expectation_gap_text),
		"",
		"## First-Order Asset Impact",
		_note_text(what_moved_text),
		"",
		"## Second-Order Asset Impact",
		_note_text(why_text),
		]
	)
	candidates_markdown = _trade_candidates_markdown(trade_candidates_df)
	if candidates_markdown:
		lines.extend(["", candidates_markdown])
	effective_selected_candidate = selected_candidate if candidates_markdown else ""
	lines.extend(
		[
			"",
			"## Selected Trade Expression",
			f"**Selected candidate:** {_note_text(effective_selected_candidate, 'Not selected')}",
			"",
			_note_text(broader_context_text),
		]
	)
	trade_fields = [
		("Instrument or structure", trade_expression.get("trade_instrument")),
		("Direction", trade_expression.get("trade_direction")),
		("Time horizon", trade_expression.get("trade_horizon")),
		("Expected transmission", trade_rationale.get("trade_primary_driver")),
		("Expected catalyst", trade_rationale.get("trade_catalyst")),
		("Entry level", trade_expression.get("trade_entry")),
		("Target", trade_expression.get("trade_target")),
		("Stop or invalidation level", trade_expression.get("trade_stop")),
		("Expected risk-reward", trade_expression.get("trade_risk_reward")),
		("Carry and roll", trade_expression.get("trade_carry_roll")),
		("Sizing and conviction", trade_expression.get("trade_sizing")),
		("Main risk", trade_rationale.get("trade_main_risk")),
	]
	populated_trade_fields = [
		(label, _coerce_text(value).strip())
		for label, value in trade_fields
		if _coerce_text(value).strip()
	]
	if populated_trade_fields:
		lines.extend(["", "## Trade Construction"])
		lines.extend(f"- {label}: {value}" for label, value in populated_trade_fields)
	lines.extend(["", "## Invalidation", _note_text(invalidation_text)])
	next_catalysts_markdown = _next_catalysts_markdown(next_catalysts_df)
	if next_catalysts_markdown:
		lines.extend(["", next_catalysts_markdown])
	return "\n".join(lines)


def render(fred_client: FREDClient, context: dict, panel_analyses: list[PanelAnalysis] | None = None) -> None:
	st.subheader("Macro Trade Idea Builder")


	title_key = _namespace_key("note_title")
	note_type_key = _namespace_key("note_type")
	horizon_key = _selected_horizon_key()
	st.session_state.setdefault(title_key, "Macro trade idea")
	if st.session_state.get(note_type_key) not in TRADE_IDEA_TYPES:
		st.session_state[note_type_key] = "Macro trade"

	setup_title, setup_type, setup_horizon = st.columns([2.0, 1.2, 1.8])
	with setup_title:
		note_title = st.text_input("Idea title", key=title_key)
	def _reset_horizon_for_note_type() -> None:
		selected_note_type = st.session_state.get(note_type_key, "Macro trade")
		st.session_state[horizon_key] = _default_horizon_for_note_type(selected_note_type)
	with setup_type:
		note_type = st.selectbox(
			"Idea type",
			TRADE_IDEA_TYPES,
			key=note_type_key,
			on_change=_reset_horizon_for_note_type,
		)
	default_horizon = _default_horizon_for_note_type(note_type)
	if st.session_state.get(horizon_key) not in COMPARISON_HORIZONS:
		st.session_state[horizon_key] = default_horizon
	with setup_horizon:
		selected_horizon = st.segmented_control(
			"Comparison horizon",
			options=list(COMPARISON_HORIZONS.keys()),
			key=horizon_key,
			required=True,
			width="stretch",
		)

	analyses = panel_analyses or context.get("panel_analyses")
	if analyses is None:
		analyses = guided_research._build_panel_analyses(fred_client, context)

	as_of = context["end_date"]
	metrics = _tracked_metrics(analyses)
	history = _panel_history(context)
	st.session_state.pop(_namespace_key("comparison_date"), None)
	requested_current_date = pd.Timestamp(as_of)
	requested_compare_date = _requested_comparison_date(requested_current_date, selected_horizon)
	move_results = [_build_move_result(metric, selected_horizon, requested_current_date, history) for metric in metrics]
	moves = _top_moves(move_results, history)
	rates_evidence = context.get("rates_evidence", {})
	upcoming_rate_events = rates_evidence.get("upcoming_events", pd.DataFrame())
	policy_context = get_policy_context(pd.Timestamp(requested_compare_date), requested_current_date, requested_current_date)

	trigger_key = _namespace_key("trigger_text")
	bottom_line_key = _namespace_key("bottom_line_text")
	catalyst_category_key = _namespace_key("catalyst_category")
	market_pricing_key = _namespace_key("market_pricing_text")
	expectation_gap_key = _namespace_key("expectation_gap_text")
	what_moved_key = _namespace_key("what_moved_text")
	why_key = _namespace_key("why_text")
	first_order_asset_key = _namespace_key("first_order_confirmation_asset")
	first_order_direction_key = _namespace_key("first_order_expected_direction")
	first_order_status_key = _namespace_key("first_order_confirmation_status")
	second_order_asset_key = _namespace_key("second_order_confirmation_asset")
	second_order_direction_key = _namespace_key("second_order_expected_direction")
	second_order_status_key = _namespace_key("second_order_confirmation_status")
	third_order_exposure_key = _namespace_key("third_order_exposure")
	third_order_sector_key = _namespace_key("third_order_sector")
	third_order_mechanism_key = _namespace_key("third_order_mechanism")
	relative_strength_key = _namespace_key("relative_strength_stance")
	target_vehicle_key = _namespace_key("target_vehicle")
	liquidity_check_key = _namespace_key("target_liquidity_check")
	confounders_key = _namespace_key("target_confounders")
	broader_context_key = _namespace_key("broader_context_text")
	trade_instrument_key = _namespace_key("trade_instrument")
	trade_direction_key = _namespace_key("trade_direction")
	trade_horizon_key = _namespace_key("trade_horizon")
	trade_entry_key = _namespace_key("trade_entry")
	trade_target_key = _namespace_key("trade_target")
	trade_stop_key = _namespace_key("trade_stop")
	trade_risk_reward_key = _namespace_key("trade_risk_reward")
	trade_carry_roll_key = _namespace_key("trade_carry_roll")
	trade_sizing_key = _namespace_key("trade_sizing")
	trade_primary_driver_key = _namespace_key("trade_primary_driver")
	trade_catalyst_key = _namespace_key("trade_catalyst")
	trade_main_risk_key = _namespace_key("trade_main_risk")
	invalidation_key = _namespace_key("invalidation_text")
	next_catalysts_key = _namespace_key("next_catalysts_df")
	next_catalysts_state_key = _namespace_key("next_catalysts_state")
	trade_candidates_key = _namespace_key("trade_candidates_df")
	trade_candidates_state_key = _namespace_key("trade_candidates_state")
	selected_candidate_key = _namespace_key("selected_candidate")
	prior_view_key = _namespace_key("prior_view_text")
	signal_update_key = _namespace_key("signal_update")
	workspace_schema_key = _namespace_key("schema_version")
	if st.session_state.get(workspace_schema_key) != "trade_idea_v6":
		for stale_key in (
			first_order_asset_key,
			second_order_asset_key,
			third_order_exposure_key,
			third_order_sector_key,
		):
			st.session_state.pop(stale_key, None)
		st.session_state[workspace_schema_key] = "trade_idea_v6"

	if isinstance(upcoming_rate_events, pd.DataFrame) and not upcoming_rate_events.empty:
		default_catalysts = pd.DataFrame(
			{
				"Catalyst": upcoming_rate_events["event_name"],
				"Date": pd.to_datetime(upcoming_rate_events["event_date"], errors="coerce").dt.date,
				"What to watch": upcoming_rate_events.get("watch_area", ""),
				"Market implication": "",
			}
		).head(8).reset_index(drop=True)
	else:
		default_catalysts = _default_next_catalysts_df()
	st.session_state.setdefault(next_catalysts_state_key, default_catalysts)
	st.session_state.setdefault(trade_candidates_state_key, _default_trade_candidates_df())
	if st.session_state.get(selected_candidate_key) not in TRADE_CANDIDATE_IDS:
		st.session_state[selected_candidate_key] = TRADE_CANDIDATE_IDS[0]
	if st.session_state.get(signal_update_key) not in SIGNAL_UPDATE_OPTIONS:
		st.session_state[signal_update_key] = SIGNAL_UPDATE_OPTIONS[0]

	macro_update_rows = _dashboard_macro_update_rows(move_results, policy_context, upcoming_rate_events)

	st.markdown("### Narrative Changes")
	if macro_update_rows:
		for row in macro_update_rows:
			with st.container(border=True):
				st.markdown(f"##### {row['Macro dimension']}")
				regime_col, narrative_col, catalyst_col = st.columns([1.0, 1.15, 1.35])
				with regime_col:
					st.caption("REGIME PATH")
					st.markdown(f"**{row['Regime path']}**")
				with narrative_col:
					st.caption("CURRENT NARRATIVE")
					st.markdown(f"**{row['Current narrative']}**")
				with catalyst_col:
					st.caption("UPCOMING CATALYSTS")
					st.write(row["Upcoming catalysts"])
				st.caption("KEY EVIDENCE")
				st.write(row["Key evidence"])
				st.caption(f"RELEVANT MARKET REACTION · {selected_horizon}")
				_render_market_reaction_bubbles(row.get("_market_reaction_items", []), selected_horizon)
				reinforce_col, reverse_col = st.columns(2)
				with reinforce_col:
					st.info(f"**Reinforces the narrative if:** {row['Reinforces narrative if']}")
				with reverse_col:
					st.warning(f"**Pulls the narrative back if:** {row['Pulls narrative back if']}")
	else:
		st.info("No comparable macro observations are available for this horizon.")
	pricing_reference = "; ".join(
		f"{move.label} {move.change:+.2f} {move.change_unit}"
		for move in moves[:3]
		if move.change is not None and not pd.isna(move.change)
	) or "No usable repricing snapshot is available."
	first_order_reactions = _market_reaction_items_for_ids(move_results, FIRST_ORDER_CONFIRMATION_IDS, limit=8)
	second_order_reactions = _market_reaction_items_for_ids(move_results, SECOND_ORDER_CONFIRMATION_IDS, limit=6)
	st.markdown("###  Trade Thesis")
	st.html(
		"""
		<style>
			.trade-flow {
				display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
				gap: .65rem; margin: .35rem 0 1rem;
			}
			.trade-flow-step {
				position: relative; min-height: 4.4rem; padding: .7rem .8rem;
				border: 1px solid rgba(37,99,235,.18); border-radius: .65rem;
				background: rgba(37,99,235,.045);
			}
			.trade-flow-step span {
				display: block; margin-bottom: .3rem; color: #2563eb;
				font-size: .65rem; font-weight: 750; letter-spacing: .06em;
			}
			.trade-flow-step strong { color: #14213d; font-size: .78rem; line-height: 1.25; }
			@media (max-width: 900px) {
				.trade-flow { grid-template-columns: repeat(2, minmax(0, 1fr)); }
			}
		</style>
		<div class="trade-flow" aria-label="Trade idea development flow">
			<div class="trade-flow-step"><span>01</span><strong>Catalyst and narrative</strong></div>
			<div class="trade-flow-step"><span>02</span><strong>First-order epicenter</strong></div>
			<div class="trade-flow-step"><span>03</span><strong>Second-order confirmation</strong></div>
			<div class="trade-flow-step"><span>04</span><strong>Third-order sector</strong></div>
			<div class="trade-flow-step"><span>05</span><strong>Relative-strength target</strong></div>
			<div class="trade-flow-step"><span>06</span><strong>Construct and risk</strong></div>
		</div>
		"""
	)
	st.caption("Identify the shock's tradable epicenter, confirm its cross-asset transmission, isolate an exposed sector, then trade the leader or laggard that remains mispriced.")
	for selector_key, options in (
		(trade_direction_key, TRADE_STANCE_OPTIONS),
		(trade_horizon_key, TRADE_HORIZON_OPTIONS),
		(trade_risk_reward_key, RISK_REWARD_OPTIONS),
		(trade_carry_roll_key, CARRY_ROLL_OPTIONS),
		(trade_sizing_key, CONVICTION_OPTIONS),
		(first_order_direction_key, CONFIRMATION_DIRECTION_OPTIONS),
		(first_order_status_key, CONFIRMATION_STATUS_OPTIONS),
		(second_order_direction_key, CONFIRMATION_DIRECTION_OPTIONS),
		(second_order_status_key, CONFIRMATION_STATUS_OPTIONS),
		(relative_strength_key, RELATIVE_STRENGTH_OPTIONS),
		(target_vehicle_key, TARGET_VEHICLE_OPTIONS),
		(liquidity_check_key, LIQUIDITY_CHECK_OPTIONS),
	):
		if st.session_state.get(selector_key) not in options:
			st.session_state[selector_key] = options[0]
	with st.form("note_workspace_inputs"):
		with st.container(border=True):
			st.markdown("#### 1. Define the catalyst and narrative")
			st.caption("Identify the shock, show how it changes the prior view, and state where that view differs from current pricing.")
			prior_col, updated_col = st.columns(2)
			with prior_col:
				catalyst_category = st.selectbox(
					"Catalyst category",
					options=CATALYST_CATEGORY_OPTIONS,
					index=None,
					key=catalyst_category_key,
					placeholder="Select the source of the shock",
				)
				prior_view_text = st.text_area("Prior macro view", key=prior_view_key, height=88)
			with updated_col:
				signal_update = st.selectbox("Effect on prior view", options=SIGNAL_UPDATE_OPTIONS, key=signal_update_key)
				trigger_text = st.text_area("Narrative created by the change", key=trigger_key, height=88)
			bottom_line_text = st.text_area(
				"Macro implications",
				key=bottom_line_key,
				height=90,
				help="What the revised view implies for growth, inflation, policy and financial conditions.",
			)
			st.caption(f"Observed repricing · {pricing_reference}")
			pricing_col, gap_col = st.columns(2)
			with pricing_col:
				market_pricing_text = st.text_area(
					"What the market appears to price",
					key=market_pricing_key,
					height=110,
				)
			with gap_col:
				expectation_gap_text = st.text_area(
					"Where your view differs",
					key=expectation_gap_key,
					height=110,
				)

		with st.container(border=True):
			st.markdown("#### 2. Select the first-order asset class")
			st.caption("Choose the tradable asset class at the epicenter of the catalyst, define its expected move, and check the available market evidence.")
			asset_col, direction_col = st.columns(2)
			with asset_col:
				first_order_asset = st.selectbox(
					"First-order asset class / instrument",
					options=FIRST_ORDER_ASSET_OPTIONS,
					index=None,
					key=first_order_asset_key,
					placeholder="Select an asset",
					help="Typical starting points: policy → short-rate futures; inflation/growth → rates or inflation products; supply shocks → commodities; corporate actions → individual equities/options; systemic stress → credit, volatility or safe havens. These are guides, not fixed rules.",
				)
			with direction_col:
				first_order_direction = st.selectbox(
					"Expected move",
					options=CONFIRMATION_DIRECTION_OPTIONS,
					key=first_order_direction_key,
				)
			what_moved_text = st.text_area("Why this is the epicenter", key=what_moved_key, height=82)
			st.caption(f"AVAILABLE FIRST-ORDER MARKET EVIDENCE · {selected_horizon}")
			_render_market_reaction_bubbles(first_order_reactions, selected_horizon)
			first_order_status = st.selectbox(
				"Does the epicenter reaction support the narrative?",
				options=CONFIRMATION_STATUS_OPTIONS,
				key=first_order_status_key,
			)

		with st.container(border=True):
			st.markdown("#### 3. Select the second-order confirmation")
			st.caption("Choose a different tradable asset class that should respond through the transmission channel and use it to test the narrative.")
			asset_col, direction_col = st.columns(2)
			with asset_col:
				second_order_asset = st.selectbox(
					"Second-order asset class / instrument",
					options=SECOND_ORDER_ASSET_OPTIONS,
					index=None,
					key=second_order_asset_key,
					placeholder="Select an asset",
				)
			with direction_col:
				second_order_direction = st.selectbox(
					"Expected move",
					options=CONFIRMATION_DIRECTION_OPTIONS,
					key=second_order_direction_key,
				)
			why_text = st.text_area("Transmission from the first-order asset", key=why_key, height=82)
			st.caption(f"AVAILABLE CROSS-ASSET EVIDENCE · {selected_horizon}")
			_render_market_reaction_bubbles(second_order_reactions, selected_horizon)
			second_order_status = st.selectbox(
				"Does the cross-asset move validate the narrative?",
				options=CONFIRMATION_STATUS_OPTIONS,
				key=second_order_status_key,
			)

		with st.container(border=True):
			st.markdown("#### 4. Isolate the third-order sector")
			st.caption("Find the sector exposed to the validated asset-class move through operations, financing costs or ownership of the underlying asset.")
			exposure_col, sector_col = st.columns(2)
			with exposure_col:
				third_order_exposure = st.selectbox(
					"Exposure channel",
					options=THIRD_ORDER_EXPOSURE_OPTIONS,
					index=None,
					key=third_order_exposure_key,
					placeholder="Select how the sector is exposed",
				)
			with sector_col:
				third_order_sector = st.selectbox(
					"Exposed sector",
					options=THIRD_ORDER_SECTOR_OPTIONS,
					index=None,
					key=third_order_sector_key,
					placeholder="Select a sector",
				)
			third_order_mechanism = st.text_area(
				"Transmission into sector earnings, margins or valuation",
				key=third_order_mechanism_key,
				height=86,
			)

		with st.container(border=True):
			st.markdown("#### 5. Select the relative-strength target")
			st.caption("Within the exposed sector, use relative strength to choose the leader for a bullish view or the laggard for a bearish view, then check implementation quality.")
			strength_col, vehicle_col, liquidity_col = st.columns(3)
			with strength_col:
				relative_strength = st.selectbox("Relative-strength rule", options=RELATIVE_STRENGTH_OPTIONS, key=relative_strength_key)
			with vehicle_col:
				target_vehicle = st.selectbox("Implementation vehicle", options=TARGET_VEHICLE_OPTIONS, key=target_vehicle_key)
			with liquidity_col:
				liquidity_check = st.selectbox("Liquidity for exit", options=LIQUIDITY_CHECK_OPTIONS, key=liquidity_check_key)
			st.text_area(
				"Other factors that could distort this sector's response",
				key=confounders_key,
				height=72,
			)
			st.caption("Compare specific ETFs, securities, options or pairs that implement the sector view.")
			trade_candidates_df = st.data_editor(
				st.session_state[trade_candidates_state_key],
				key=trade_candidates_key,
				hide_index=True,
				width="stretch",
				height=190,
				disabled=["Candidate"],
			)
			st.session_state[trade_candidates_state_key] = trade_candidates_df
			selection_col, rationale_col = st.columns([0.8, 2.2])
			with selection_col:
				selected_candidate = st.selectbox("Selected target", options=TRADE_CANDIDATE_IDS, key=selected_candidate_key)
			with rationale_col:
				broader_context_text = st.text_area(
					"Why this target is the leader or laggard to trade",
					key=broader_context_key,
					height=86,
				)

		with st.container(border=True):
			st.markdown("#### 6. Construct and risk-manage the trade")
			structure_col, direction_col, horizon_col = st.columns([1.5, 1, 1])
			with structure_col:
				trade_instrument = st.text_input("Final instrument / structure", key=trade_instrument_key)
			with direction_col:
				trade_direction = st.selectbox("Direction / stance", options=TRADE_STANCE_OPTIONS, key=trade_direction_key)
			with horizon_col:
				trade_horizon = st.selectbox("Time horizon", options=TRADE_HORIZON_OPTIONS, key=trade_horizon_key)
			driver_col, catalyst_col = st.columns(2)
			with driver_col:
				trade_primary_driver = st.text_area("How the trade should make money", key=trade_primary_driver_key, height=86)
			with catalyst_col:
				trade_catalyst = st.text_area("Expected catalyst", key=trade_catalyst_key, height=86)
			risk_col, invalidation_col = st.columns(2)
			with risk_col:
				trade_main_risk = st.text_area("Main risk", key=trade_main_risk_key, height=86)
			with invalidation_col:
				invalidation_text = st.text_area("Invalidation condition", key=invalidation_key, height=86)
			with st.expander("Execution details", expanded=False):
				st.caption("Use these only when a trade has a defined implementation plan.")
				expr_cols = st.columns(2)
				with expr_cols[0]:
					trade_entry = st.text_input("Entry level", key=trade_entry_key)
					trade_target = st.text_input("Target", key=trade_target_key)
					trade_stop = st.text_input("Stop / invalidation level", key=trade_stop_key)
				with expr_cols[1]:
					trade_risk_reward = st.selectbox("Expected risk-reward", options=RISK_REWARD_OPTIONS, key=trade_risk_reward_key)
					trade_carry_roll = st.selectbox("Carry and roll", options=CARRY_ROLL_OPTIONS, key=trade_carry_roll_key)
					trade_sizing = st.selectbox("Sizing / conviction", options=CONVICTION_OPTIONS, key=trade_sizing_key)
			with st.expander("Upcoming rates catalysts", expanded=False):
				st.caption("Pre-filled from the Yield Curve panel's next-14-days calendar.")
				next_catalysts_df = st.data_editor(
					st.session_state[next_catalysts_state_key],
					key=next_catalysts_key,
					num_rows="dynamic",
					width="stretch",
					hide_index=True,
					column_config={"Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")},
				)
				st.session_state[next_catalysts_state_key] = next_catalysts_df
		submitted = st.form_submit_button("Check trade thesis", width="stretch")

	if submitted:
		candidate_frame = st.session_state.get(trade_candidates_state_key, pd.DataFrame())
		candidate_expressions = (
			candidate_frame.get("Expression", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
			if isinstance(candidate_frame, pd.DataFrame)
			else pd.Series(dtype="object")
		)
		selected_expression = ""
		if isinstance(candidate_frame, pd.DataFrame) and {"Candidate", "Expression"} <= set(candidate_frame.columns):
			selected_rows = candidate_frame.loc[candidate_frame["Candidate"].eq(selected_candidate), "Expression"]
			if not selected_rows.empty:
				selected_expression = _coerce_text(selected_rows.iloc[0])
		missing_thesis_fields = [
			label
			for label, value in (
				("prior macro view", prior_view_text),
				("updated macro thesis", trigger_text),
				("current market pricing", market_pricing_text),
				("expectation gap", expectation_gap_text),
			)
			if not value.strip()
		]
		missing_confirmation_fields = [
			label
			for label, missing in (
				("first-order confirmation asset", not first_order_asset),
				("first-order expected move", first_order_direction == "Not specified"),
				("first-order transmission rationale", not what_moved_text.strip()),
				("first-order confirmation status", first_order_status == "Not assessed"),
				("second-order confirmation asset", not second_order_asset),
				("second-order expected move", second_order_direction == "Not specified"),
				("second-order transmission rationale", not why_text.strip()),
				("second-order confirmation status", second_order_status == "Not assessed"),
			)
			if missing
		]
		missing_target_fields = [
			label
			for label, missing in (
				("third-order exposure channel", not third_order_exposure),
				("third-order sector", not third_order_sector),
				("third-order transmission mechanism", not third_order_mechanism.strip()),
				("relative-strength rule", relative_strength == "Not assessed"),
				("implementation vehicle", target_vehicle == "Not selected"),
				("liquidity check", liquidity_check == "Not assessed"),
				("target-selection rationale", not broader_context_text.strip()),
			)
			if missing
		]
		if not catalyst_category:
			st.error("Select the catalyst category before mapping its asset-class transmission.")
		elif missing_thesis_fields:
			st.error("Complete the " + ", ".join(missing_thesis_fields) + " before finalising the trade thesis.")
		elif signal_update == "Unclear":
			st.error("Assess how the new evidence affected the prior view before finalising the trade thesis.")
		elif missing_confirmation_fields:
			st.error("Complete the " + ", ".join(missing_confirmation_fields) + " before assessing the target expression.")
		elif first_order_asset == second_order_asset:
			st.error("Choose a different second-order asset class so it provides an independent cross-asset confirmation.")
		elif missing_target_fields:
			st.error("Complete the " + ", ".join(missing_target_fields) + " before constructing the trade.")
		elif int(candidate_expressions.ne("").sum()) < 2:
			st.error("Enter at least two candidate trade expressions before selecting one.")
		elif not selected_expression:
			st.error("The selected candidate must contain a trade expression.")
		elif not invalidation_text.strip():
			st.error("An invalidation condition is required before finalising the trade thesis.")
		elif first_order_status == "Contradicts":
			st.warning("The primary market contradicts the narrative. Reassess the macro interpretation before treating this as a trade.")
		elif first_order_status == "Not yet tested":
			st.info("The idea is conditional: wait for the first-order market to validate the narrative before treating the target as tradeable.")
		elif second_order_status == "Contradicts":
			st.warning("First-order evidence may support the narrative, but cross-asset confirmation conflicts. Treat the idea as lower confidence or investigate the competing regime.")
		elif second_order_status == "Not yet tested":
			st.info("First-order confirmation is present, but the second-order transmission has not yet been tested. Keep the target on the watchlist pending broader confirmation.")
		elif liquidity_check == "Inadequate":
			st.warning("The selected target does not have enough liquidity for a reliable exit. Use a more liquid ETF, future or alternative expression.")
		elif relative_strength == "No clear relative-strength signal":
			st.info("The sector exposure is plausible, but there is no clear leader or laggard yet. Keep the expression on the watchlist.")
		else:
			st.success("The narrative has sufficient market confirmation to evaluate the target trade.")
