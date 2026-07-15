from __future__ import annotations

import hashlib
import html
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from fredapi import Fred

from analysis.synthesis import synthesize
from config import (
	FRED_API_KEY,
	FRED_CACHE_TTL_SECONDS,
	FRED_POLICY_SERIES,
	NOTE_WORKSPACE_ARCHIVE_DIR,
	NOTE_WORKSPACE_MAX_MOVES,
)
from data.fred_client import FREDClient
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
	"Market Monitor": "1W",
	"Release Reaction": "1D",
	"Issue or Strategy Note": "1M",
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
	{"metric_id": "ns_level", "panel_id": "nelson_siegel", "signal_id": "ns_level_factor_change", "label": "Nelson-Siegel level"},
	{"metric_id": "ns_slope", "panel_id": "nelson_siegel", "signal_id": "ns_slope_factor_change", "label": "Nelson-Siegel slope"},
	{"metric_id": "ns_curvature", "panel_id": "nelson_siegel", "signal_id": "ns_curvature_factor_change", "label": "Nelson-Siegel curvature"},
	{"metric_id": "inflation_5y_be", "panel_id": "inflation", "signal_id": "inflation_5y_breakeven", "label": "5Y breakeven"},
	{"metric_id": "inflation_10y_be", "panel_id": "inflation", "signal_id": "inflation_10y_breakeven", "label": "10Y breakeven"},
	{"metric_id": "inflation_5y5y", "panel_id": "inflation", "signal_id": "inflation_5y5y_forward", "label": "5Y5Y forward"},
	{"metric_id": "inflation_cpi", "panel_id": "inflation", "signal_id": "inflation_cpi_yoy", "label": "Headline CPI YoY (SA index)"},
	{"metric_id": "inflation_pce", "panel_id": "inflation", "signal_id": "inflation_pce_yoy", "label": "Headline PCE YoY"},
	{"metric_id": "inflation_michigan", "panel_id": "inflation", "signal_id": "inflation_michigan", "label": "Michigan expectations"},
	{"metric_id": "growth_claims", "panel_id": "growth", "signal_id": "labor_demand_claims", "label": "Claims component"},
	{"metric_id": "credit_hy", "panel_id": "cross_asset", "signal_id": "credit_hy_oas", "label": "HY OAS"},
	{"metric_id": "credit_ig", "panel_id": "cross_asset", "signal_id": "credit_ig_oas", "label": "IG OAS"},
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
			{"metric_id": "curve_2s10s", "expected": "down", "label": "curve flattening"},
			{"metric_id": "credit_hy", "expected": "up", "label": "credit spreads wider"},
			{"metric_id": "vix", "expected": "up", "label": "VIX higher"},
		],
	},
	{
		"name": "Risk-on / soft landing",
		"signals": [
			{"metric_id": "credit_hy", "expected": "down", "label": "HY spreads tighter"},
			{"metric_id": "credit_ig", "expected": "down", "label": "IG spreads tighter"},
			{"metric_id": "vix", "expected": "down", "label": "VIX lower"},
			{"metric_id": "dxy", "expected": "down", "label": "dollar softer"},
			{"metric_id": "payrolls", "expected": "up", "label": "payrolls stronger"},
		],
	},
	{
		"name": "Hawkish repricing",
		"signals": [
			{"metric_id": "yield_2y", "expected": "up", "label": "front-end yields higher"},
			{"metric_id": "inflation_5y_be", "expected": "down", "label": "breakevens flat/down"},
			{"metric_id": "ns_slope", "expected": "down", "label": "slope flatter"},
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
			{"metric_id": "inflation_cpi", "expected": "up", "label": "headline CPI firmer"},
			{"metric_id": "inflation_pce", "expected": "up", "label": "headline PCE firmer"},
			{"metric_id": "inflation_5y_be", "expected": "up", "label": "5Y breakevens firmer"},
			{"metric_id": "inflation_10y_be", "expected": "up", "label": "10Y breakevens firmer"},
			{"metric_id": "inflation_michigan", "expected": "up", "label": "household expectations firmer"},
		],
	},
	{
		"name": "Disinflation progress",
		"signals": [
			{"metric_id": "inflation_cpi", "expected": "down", "label": "headline CPI softer"},
			{"metric_id": "inflation_pce", "expected": "down", "label": "headline PCE softer"},
			{"metric_id": "inflation_5y_be", "expected": "down", "label": "5Y breakevens softer"},
			{"metric_id": "inflation_10y_be", "expected": "down", "label": "10Y breakevens softer"},
			{"metric_id": "inflation_michigan", "expected": "down", "label": "household expectations easing"},
		],
	},
	{
		"name": "Growth scare",
		"signals": [
			{"metric_id": "growth_claims", "expected": "up", "label": "claims component weaker"},
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
			{"metric_id": "inflation_cpi", "expected": "up", "label": "headline CPI firming"},
			{"metric_id": "inflation_pce", "expected": "up", "label": "headline PCE firming"},
			{"metric_id": "payrolls", "expected": "down", "label": "payrolls softer"},
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
	"inflation_pce": "Inflation",
	"inflation_michigan": "Inflation",
	"credit_hy": "Credit",
	"credit_ig": "Credit",
	"vix": "Equities",
	"dxy": "FX",
	"payrolls": "Labour or growth",
	"unrate": "Labour or growth",
	"claims": "Labour or growth",
	"growth_claims": "Labour or growth",
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
	"scheduled_catalysts": "BLS / BEA / Federal Reserve calendars",
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
	"inflation_pce": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_pce", change_multiplier=100.0),
	"inflation_michigan": MetricScaleSpec(current_unit="%", change_unit="bp", series_key="inflation_michigan", change_multiplier=100.0),
	"credit_hy": MetricScaleSpec(current_unit="bp", change_unit="bp", series_key="credit_hy", value_multiplier=100.0),
	"credit_ig": MetricScaleSpec(current_unit="bp", change_unit="bp", series_key="credit_ig", value_multiplier=100.0),
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
		if metric_id in {"yield_2y", "yield_10y", "ns_level", "inflation_cpi", "inflation_pce", "inflation_5y_be", "inflation_10y_be", "inflation_5y5y", "inflation_michigan", "unrate", "wages"} and abs_value > 100.0:
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
	if metric_id in {"claims", "growth_claims"}:
		return "weekly"
	if metric_id in {"payrolls", "unrate", "wages", "inflation_cpi", "inflation_pce", "inflation_michigan", "inflation_5y_be", "inflation_10y_be", "inflation_5y5y", "openings_ratio"}:
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
	if metric.metric_id in {"dxy"}:
		return "index_pct"
	if metric.metric_id in {"payrolls", "claims"}:
		return "k"
	if metric.metric_id in {"unrate", "wages", "inflation_cpi", "inflation_pce", "inflation_michigan"}:
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
		if metric.metric_id in {"inflation_cpi", "inflation_pce", "inflation_5y_be", "inflation_10y_be", "inflation_5y5y", "inflation_michigan", "unrate", "wages"} and abs_raw > 2000.0:
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
		return f"No dominant market trigger was flagged versus {compare_date.date().isoformat()}."
	lead = moves[0]
	if lead.standardized_change is not None and not pd.isna(lead.standardized_change):
		context = f"This was a {lead.standardized_change:+.1f} standard deviation move"
	else:
		context = "This move was notable relative to recent history"
	return f"{lead.label} moved {lead.change:+.2f} {lead.change_unit} versus {compare_date.date().isoformat()}. {context}."


def _empty_event_frame() -> pd.DataFrame:
	return pd.DataFrame(
		columns=[
			"event_date",
			"event_time",
			"event_name",
			"event_type",
			"source",
			"importance",
			"release_id",
		]
	)


def _coerce_text(value: object) -> str:
	if value is None or pd.isna(value):
		return ""
	text = str(value).strip()
	return "" if text.lower() in {"nan", "none", "null"} else text


def _coerce_numeric(value: object) -> float | None:
	text = _coerce_text(value)
	if not text:
		return None
	try:
		return float(str(text).replace(",", ""))
	except Exception:  # noqa: BLE001
		return None


def _fetch_html_text(url: str) -> str:
	try:
		response = requests.get(url, timeout=20)
		if response.status_code != 200:
			return ""
		return response.text
	except Exception:  # noqa: BLE001
		return ""


def _read_html_tables(html_text: str) -> list[pd.DataFrame]:
	if not html_text.strip():
		return []
	try:
		return pd.read_html(StringIO(html_text))
	except Exception:  # noqa: BLE001
		return []


def _clean_page_lines(html_text: str) -> list[str]:
	text = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", html_text, flags=re.S)
	text = html.unescape(text)
	lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
	return [line for line in lines if line]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
	work = frame.copy()
	work.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in work.columns]
	return work


def _is_time_line(value: object) -> bool:
	text = _coerce_text(value).upper()
	return bool(re.fullmatch(r"\d{1,2}:\d{2}\s*(AM|PM)", text) or text == "TO BE ANNOUNCED")


def _is_day_line(value: object) -> bool:
	text = _coerce_text(value)
	return bool(re.fullmatch(r"\d{1,2}", text))


def _is_month_year_line(value: object) -> bool:
	text = _coerce_text(value)
	return bool(re.fullmatch(r"[A-Z][a-z]+ \d{4}", text) or re.fullmatch(r"[A-Z][a-z]+ \d{1,2}", text))


def _is_event_text(value: object) -> bool:
	text = _coerce_text(value)
	if not text:
		return False
	blocked = {
		"news",
		"data",
		"visual data",
		"article",
		"release schedule",
		"calendar",
		"month view",
		"list view",
		"by month",
		"by news release",
		"full schedule",
		"upcoming releases",
		"to be announced",
	}
	lowered = text.lower()
	if lowered in blocked:
		return False
	if _looks_like_date(text) or _is_time_line(text) or _is_day_line(text):
		return False
	if len(text) < 3:
		return False
	return True


def _parse_release_lines(lines: list[str], *, source_name: str, event_type: str, start_date: pd.Timestamp, end_date: pd.Timestamp, mode: str) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	current_day: str | None = None
	for idx, line in enumerate(lines):
		if mode == "bls" and _is_day_line(line):
			current_day = line
			continue
		if not _is_event_text(line):
			continue
		next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
		next_next_line = lines[idx + 2] if idx + 2 < len(lines) else ""
		if mode == "bls":
			if not (_is_month_year_line(next_line) and _is_time_line(next_next_line)):
				continue
			if current_day is None:
				continue
			event_date = _parse_date_value(f"{current_day} {next_line}")
		else:
			if not (_looks_like_date(next_line) and _is_time_line(next_next_line)):
				continue
			event_date = _parse_date_value(next_line)
		if event_date is None or event_date < start_date.normalize() or event_date > end_date.normalize():
			continue
		rows.append(
			{
				"event_date": event_date.date().isoformat(),
				"event_time": _coerce_text(next_next_line),
				"event_name": line,
				"event_type": event_type,
				"source": source_name,
				"importance": "High",
				"release_id": _safe_filename(f"{source_name}_{line}_{event_date.date().isoformat()}"),
			}
		)
	if not rows:
		return _empty_event_frame()
	frame = pd.DataFrame(rows).drop_duplicates(subset=["event_date", "event_name", "source"])
	return frame.sort_values(["event_date", "event_name"], ascending=[True, True])


def _looks_like_date(value: object) -> bool:
	return pd.notna(pd.to_datetime(value, errors="coerce"))


def _parse_date_value(value: object, default_year: int | None = None) -> pd.Timestamp | None:
	if value is None or pd.isna(value):
		return None
	candidates = [str(value).strip()]
	if default_year is not None:
		candidates.append(f"{value} {default_year}")
	for candidate in candidates:
		parsed = pd.to_datetime(candidate, errors="coerce")
		if pd.notna(parsed):
			return pd.Timestamp(parsed).normalize()
	return None


def _event_frame_from_tables(
	html_text: str,
	*,
	source_name: str,
	event_type: str,
	start_date: pd.Timestamp,
	end_date: pd.Timestamp,
	importance: str = "Medium",
) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for table in _read_html_tables(html_text):
		work = _normalize_columns(table)
		if work.empty:
			continue
		columns = list(work.columns)
		date_columns = [column for column in columns if any(token in column.lower() for token in ("date", "release", "meeting", "scheduled"))]
		text_columns = [column for column in columns if column not in date_columns]
		for _, row in work.iterrows():
			row_values = [value for value in row.tolist() if _coerce_text(value)]
			if not row_values:
				continue
			event_date = None
			for column in date_columns + columns:
				event_date = _parse_date_value(row.get(column))
				if event_date is not None:
					break
			if event_date is None:
				continue
			if event_date < start_date.normalize() or event_date > end_date.normalize():
				continue
			name_bits = [_coerce_text(row.get(column)) for column in text_columns]
			event_name = next((bit for bit in name_bits if bit and not _looks_like_date(bit)), "")
			if not event_name:
				event_name = next((bit for bit in row_values if bit and not _looks_like_date(bit)), "")
			if not event_name:
				continue
			rows.append(
				{
					"event_date": event_date.date().isoformat(),
					"event_time": "",
					"event_name": event_name,
					"event_type": event_type,
					"source": source_name,
					"importance": importance,
					"release_id": _safe_filename(f"{source_name}_{event_name}_{event_date.date().isoformat()}"),
				}
			)
	if not rows:
		return _empty_event_frame()
	frame = pd.DataFrame(rows).drop_duplicates(subset=["event_date", "event_name", "source"])
	return frame.sort_values(["event_date", "importance", "event_name"], ascending=[True, False, True])


def _fomc_calendar_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
	html_text = _fetch_html_text("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
	if not html_text:
		return _empty_event_frame()
	frame = _event_frame_from_tables(
		html_text,
		source_name="Federal Reserve",
		event_type="FOMC calendar",
		start_date=start_date,
		end_date=end_date,
		importance="High",
	)
	if not frame.empty:
		return frame
	text = re.sub(r"<script.*?</script>|<style.*?</style>|<[^>]+>", " ", html_text, flags=re.S)
	text = re.sub(r"\s+", " ", text)
	rows: list[dict[str, object]] = []
	for match in re.finditer(r"([A-Z][a-z]+ \d{1,2}(?:-\d{1,2})?, \d{4})", text):
		range_text = match.group(1)
		month_day, year = range_text.rsplit(", ", 1)
		start_day = month_day.split("-")[0]
		event_date = pd.to_datetime(f"{start_day}, {year}", errors="coerce")
		if pd.isna(event_date):
			continue
		if event_date < start_date.normalize() or event_date > end_date.normalize():
			continue
		rows.append(
			{
				"event_date": event_date.date().isoformat(),
				"event_time": "",
				"event_name": f"FOMC meeting ({match.group(1)})",
				"event_type": "FOMC calendar",
				"source": "Federal Reserve",
				"importance": "High",
				"release_id": _safe_filename(f"FOMC_{match.group(1)}"),
			}
		)
	if not rows:
		return _empty_event_frame()
	return pd.DataFrame(rows).drop_duplicates(subset=["event_date", "event_name"]).sort_values(["event_date", "event_name"])


def _bls_schedule_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
	html_text = _fetch_html_text("https://www.bls.gov/schedule/news_release/")
	frame = _event_frame_from_tables(
		html_text,
		source_name="BLS",
		event_type="Economic release",
		start_date=start_date,
		end_date=end_date,
		importance="High",
	)
	if not frame.empty:
		return frame
	return _parse_release_lines(_clean_page_lines(html_text), source_name="BLS", event_type="Economic release", start_date=start_date, end_date=end_date, mode="bls")


def _bea_schedule_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
	html_text = _fetch_html_text("https://www.bea.gov/news/schedule")
	frame = _event_frame_from_tables(
		html_text,
		source_name="BEA",
		event_type="Economic release",
		start_date=start_date,
		end_date=end_date,
		importance="High",
	)
	if not frame.empty:
		return frame
	return _parse_release_lines(_clean_page_lines(html_text), source_name="BEA", event_type="Economic release", start_date=start_date, end_date=end_date, mode="bea")


def _official_schedule_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
	frames = [
		_bls_schedule_frame(start_date, end_date),
		_bea_schedule_frame(start_date, end_date),
		_fomc_calendar_frame(start_date, end_date),
	]
	frames = [frame for frame in frames if not frame.empty]
	if not frames:
		return _empty_event_frame()
	frame = pd.concat(frames, ignore_index=True)
	frame = frame.drop_duplicates(subset=["event_date", "event_name", "source"])
	return frame.sort_values(["event_date", "importance", "event_name"], ascending=[True, False, True])


@st.cache_data(ttl=FRED_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_scheduled_catalysts(start_date: str, end_date: str) -> pd.DataFrame:
	start_ts = pd.Timestamp(start_date)
	end_ts = pd.Timestamp(end_date)
	frame = _official_schedule_frame(start_ts, end_ts)
	if frame.empty:
		return _empty_event_frame()
	return frame


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
	frame = _cached_scheduled_catalysts(pd.Timestamp(start_date).date().isoformat(), pd.Timestamp(end_date).date().isoformat())
	return frame if not frame.empty else _empty_event_frame()


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
			monthly = raw[[c for c in ["CPIAUCSL", "PCEPI", "MICH", "T5YIE", "T10YIE", "T5YIFR"] if c in raw.columns]].resample("ME").last()
			if metric_id == "inflation_cpi":
				return monthly["CPIAUCSL"].pct_change(12, fill_method=None) * 100.0
			if metric_id == "inflation_pce":
				return monthly["PCEPI"].pct_change(12, fill_method=None) * 100.0
			if metric_id == "inflation_michigan":
				return pd.to_numeric(monthly["MICH"], errors="coerce")
			if metric_id == "inflation_5y_be" and "T5YIE" in raw.columns:
				return pd.to_numeric(raw["T5YIE"], errors="coerce")
			if metric_id == "inflation_10y_be" and "T10YIE" in raw.columns:
				return pd.to_numeric(raw["T10YIE"], errors="coerce")
			if metric_id == "inflation_5y5y" and "T5YIFR" in raw.columns:
				return pd.to_numeric(raw["T5YIFR"], errors="coerce")

	if isinstance(cross_asset_frame, pd.DataFrame):
		if metric_id == "credit_hy":
			return pd.to_numeric(cross_asset_frame.get("BAMLH0A0HYM2"), errors="coerce") * 100.0
		if metric_id == "credit_ig":
			return pd.to_numeric(cross_asset_frame.get("BAMLC0A0CM"), errors="coerce") * 100.0
		if metric_id == "vix":
			return pd.to_numeric(cross_asset_frame.get("VIXCLS"), errors="coerce")
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
			return weekly.rolling(4).mean()

	if isinstance(growth_frame, dict):
		raw_features = growth_frame.get("raw_features")
		if isinstance(raw_features, pd.DataFrame) and metric_id == "growth_claims" and "claims_growth" in raw_features.columns:
			return pd.to_numeric(raw_features["claims_growth"], errors="coerce")

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
		score = _move_score(metric)
		if abs(float(effective_change)) < _meaningful_threshold(metric) and score < 0.5:
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
	moves.sort(key=lambda item: (CATEGORY_ORDER.index(_metric_category(_metric_map(metrics)[item.metric_id])) if _metric_category(_metric_map(metrics)[item.metric_id]) in CATEGORY_ORDER else len(CATEGORY_ORDER), -_move_score(_metric_map(metrics)[item.metric_id])), reverse=False)
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
				task="Check macro news and intraday headlines around the primary move.",
				reason="A dominant move was identified, but the panel does not yet have an event feed to confirm timing.",
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


def _render_html_table(rows: list[dict], headers: list[str], height_floor: int = 0, *, scrolling: bool = False) -> None:
	if not rows:
		st.info("No rows to show.")
		return
	total_lines = 0
	for row in rows:
		row_lines = 1
		for header in headers:
			cell_text = str(row.get(header, ""))
			row_lines = max(row_lines, cell_text.count("\n") + 1, len(cell_text) // 90 + 1)
		total_lines += row_lines
	df = pd.DataFrame(rows)[headers]
	df = df.applymap(lambda value: html.escape(str(value)).replace("\n", "<br>"))
	table_html = df.to_html(index=False, escape=False, border=0, classes=["note-workspace-table"])
	components.html(
		f"""
		<!doctype html>
		<html>
		<head>
			<meta charset="utf-8">
			<style>
				body {{
					margin: 0;
					font-family: sans-serif;
				}}
				table.note-workspace-table {{
					width: 100%;
					border-collapse: collapse;
					font-size: 0.92rem;
				}}
				table.note-workspace-table th, table.note-workspace-table td {{
					padding: 0.25rem 0.45rem;
					border-bottom: 1px solid rgba(49, 51, 63, 0.14);
					text-align: left;
					vertical-align: top;
					white-space: pre-wrap;
				}}
				table.note-workspace-table th {{
					background: rgba(49, 51, 63, 0.05);
				}}
			</style>
		</head>
		<body>
			{table_html}
		</body>
		</html>
		""",
		height=max(height_floor, min(48 + 24 * len(rows) + 12 * total_lines, 1200)),
		scrolling=scrolling,
	)


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


def _event_card_summary(frame: pd.DataFrame) -> tuple[str, str, str]:
	if frame.empty:
		return (
			"No scheduled catalyst feed is currently available.",
			"Unavailable",
			"This card uses public BLS, BEA, and Federal Reserve calendars.",
		)
	work = frame.copy()
	work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce")
	work = work.loc[work["event_date"].notna()].copy()
	work = work.sort_values(["event_date", "importance", "event_name"], ascending=[True, False, True])
	if work.empty:
		return (
			"No scheduled catalyst feed is currently available.",
			"Unavailable",
			"This card uses public BLS, BEA, and Federal Reserve calendars.",
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
			f"{len(work)} upcoming catalysts that could move rates, growth, or policy expectations between {pd.Timestamp(first['event_date']).date().isoformat()} and {pd.Timestamp(last['event_date']).date().isoformat()}.",
			f"Next event: {first['event_name']} on {pd.Timestamp(first['event_date']).date().isoformat()} ({first['source']}).",
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


def _summary_card_grid(cards: list[dict[str, str]]) -> str:
	def _card(card: dict[str, str]) -> str:
		title = html.escape(card.get("title", ""))
		source = html.escape(card.get("source", ""))
		status = html.escape(card.get("status", ""))
		body = html.escape(card.get("body", "")).replace("\n", "<br>")
		footer = card.get("footer", "")
		footer_html = f"<div class='nw-card-footer'>{html.escape(footer)}</div>" if footer else ""
		return f"""
			<div class="nw-card">
				<div class="nw-card-top">
					<div class="nw-card-title">{title}</div>
					<div class="nw-card-badge">{status}</div>
				</div>
				<div class="nw-card-source">{source}</div>
				<div class="nw-card-body">{body}</div>
				{footer_html}
			</div>
		"""

	card_html = "".join(_card(card) for card in cards)
	return f"""
		<!doctype html>
		<html>
		<head>
			<meta charset="utf-8">
			<style>
				body {{
					margin: 0;
					font-family: sans-serif;
				}}
				.nw-grid {{
					display: grid;
					grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
					gap: 0.6rem;
				}}
				.nw-card {{
					border: 1px solid rgba(49, 51, 63, 0.14);
					border-radius: 12px;
					background: #fff;
					padding: 0.75rem 0.85rem;
					box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
				}}
				.nw-card-top {{
					display: flex;
					justify-content: space-between;
					gap: 0.5rem;
					align-items: start;
					margin-bottom: 0.35rem;
				}}
				.nw-card-title {{
					font-weight: 700;
					font-size: 0.98rem;
				}}
				.nw-card-badge {{
					font-size: 0.72rem;
					line-height: 1;
					padding: 0.25rem 0.45rem;
					border-radius: 999px;
					background: rgba(49, 51, 63, 0.08);
					white-space: nowrap;
				}}
				.nw-card-source {{
					font-size: 0.76rem;
					color: rgba(49, 51, 63, 0.72);
					margin-bottom: 0.35rem;
				}}
				.nw-card-body {{
					font-size: 0.88rem;
					line-height: 1.45;
				}}
				.nw-card-footer {{
					margin-top: 0.45rem;
					font-size: 0.76rem;
					color: rgba(49, 51, 63, 0.72);
				}}
			</style>
		</head>
		<body>
			<div class="nw-grid">{card_html}</div>
		</body>
		</html>
	"""


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
		st.caption("No archived notes yet.")
		return
	options = [f"{path.stat().st_mtime_ns} - {path.stem}" for path in notes]
	choice = st.selectbox("Browse archived notes", options=options, key=_namespace_key("archive_choice"))
	path = notes[options.index(choice)]
	try:
		content = path.read_text(encoding="utf-8")
	except Exception:  # noqa: BLE001
		content = "(Unable to read archived note.)"
	with st.expander("Archived note content", expanded=False):
		st.markdown(content)


def _build_note(
	title: str,
	note_type: str,
	trigger_text: str,
	bottom_line_text: str,
	what_moved_text: str,
	why_text: str,
	signal_conflicts_text: str,
	broader_context_text: str,
	trade_expression: dict[str, object],
	trade_rationale: dict[str, object],
	invalidation_text: str,
	next_catalysts_df: pd.DataFrame | None,
) -> str:
	trade_expression = trade_expression or {}
	trade_rationale = trade_rationale or {}
	lines = [
		f"# {title}",
		f"**Note type:** {_note_text(note_type, 'Market Monitor')}",
		"",
		"## Trigger",
		_note_text(trigger_text),
		"",
		"## Bottom Line",
		_note_text(bottom_line_text),
		"",
		"## What Moved?",
		_note_text(what_moved_text),
		"",
		"## Why It Matters",
		_note_text(why_text),
		"",
		"## Confirmation and Conflicting Signals",
		_note_text(signal_conflicts_text),
		"",
		"## Broader Context",
		_note_text(broader_context_text),
		"",
		"## Trade Idea or Position",
		"",
		"**Trade expression**",
		f"- Instrument or structure: {_note_text(trade_expression.get('trade_instrument'))}",
		f"- Direction: {_note_text(trade_expression.get('trade_direction'))}",
		f"- Time horizon: {_note_text(trade_expression.get('trade_horizon'))}",
		f"- Entry level: {_note_text(trade_expression.get('trade_entry'))}",
		f"- Target: {_note_text(trade_expression.get('trade_target'))}",
		f"- Stop or invalidation level: {_note_text(trade_expression.get('trade_stop'))}",
		f"- Expected risk-reward: {_note_text(trade_expression.get('trade_risk_reward'))}",
		f"- Carry and roll: {_note_text(trade_expression.get('trade_carry_roll'))}",
		f"- Sizing and conviction: {_note_text(trade_expression.get('trade_sizing'))}",
		"",
		"**Trade rationale**",
		f"- Primary driver: {_note_text(trade_rationale.get('trade_primary_driver'))}",
		f"- Expected catalyst: {_note_text(trade_rationale.get('trade_catalyst'))}",
		f"- Main risk: {_note_text(trade_rationale.get('trade_main_risk'))}",
		"",
		"## What Would Change My Mind",
		_note_text(invalidation_text),
	]
	next_catalysts_markdown = _next_catalysts_markdown(next_catalysts_df)
	if next_catalysts_markdown:
		lines.extend(["", next_catalysts_markdown])
	return "\n".join(lines)


def render(fred_client: FREDClient, context: dict, panel_analyses: list[PanelAnalysis] | None = None) -> None:
	st.subheader("Panel 7: Guided Macro Note Workspace")
	st.caption("A guided note-writing workspace built from the six existing panel outputs and a local note archive.")
	st.markdown(
		"""
		<style>
			section.main h2, section.main h3, section.main h4 {
				margin-top: 0 !important;
				margin-bottom: 0 !important;
			}
			section.main p {
				margin-top: 0 !important;
				margin-bottom: 0 !important;
			}
			section.main .stCaption {
				padding-top: 0 !important;
				padding-bottom: 0 !important;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)

	analyses = panel_analyses or context.get("panel_analyses")
	if analyses is None:
		analyses = guided_research._build_panel_analyses(fred_client, context)

	as_of = context["end_date"]
	synthesis = synthesize(analyses, as_of)
	metrics = _tracked_metrics(analyses)
	history = _panel_history(context)
	note_type_key = _namespace_key("note_type")
	st.session_state.setdefault(note_type_key, "Market Monitor")
	note_type = st.session_state.get(note_type_key, "Market Monitor")
	horizon_key = _selected_horizon_key()
	default_horizon = _default_horizon_for_note_type(note_type)
	st.session_state.setdefault(horizon_key, default_horizon)
	selected_horizon = st.selectbox(
		"Comparison horizon",
		options=list(COMPARISON_HORIZONS.keys()),
		index=list(COMPARISON_HORIZONS.keys()).index(st.session_state[horizon_key]) if st.session_state.get(horizon_key) in COMPARISON_HORIZONS else list(COMPARISON_HORIZONS.keys()).index(default_horizon),
		key=horizon_key,
		help="Compare the current snapshot against the same calendar horizon in the past and use same-horizon historical moves for z-scores.",
	)
	st.session_state.pop(_namespace_key("comparison_date"), None)
	requested_current_date = pd.Timestamp(as_of)
	requested_compare_date = _requested_comparison_date(requested_current_date, selected_horizon)
	move_results = [_build_move_result(metric, selected_horizon, requested_current_date, history) for metric in metrics]
	comparison_metric_map = _metric_map(move_results)
	moves = _top_moves(move_results, history)
	primary_move = max(moves, key=lambda move: _move_score(comparison_metric_map[move.metric_id]), default=None)
	move_ids = {move.metric_id for move in moves}
	anchor_ids = _anchor_metric_ids(move_results, exclude_ids=move_ids)
	anchor_rows = _anchor_display_rows(move_results, history, exclude_ids=move_ids)
	pattern_evaluations = [row for pattern in MOVE_PATTERNS if (row := _pattern_row(pattern, comparison_metric_map)) is not None]
	scheduled_catalysts = get_scheduled_catalysts(pd.Timestamp(requested_compare_date), requested_current_date)
	policy_context = get_policy_context(pd.Timestamp(requested_compare_date), requested_current_date, requested_current_date)
	research_gaps = _build_research_gaps(primary_move, scheduled_catalysts, policy_context, pattern_evaluations)

	title_key = _namespace_key("note_title")
	trigger_key = _namespace_key("trigger_text")
	bottom_line_key = _namespace_key("bottom_line_text")
	what_moved_key = _namespace_key("what_moved_text")
	why_key = _namespace_key("why_text")
	signal_conflicts_key = _namespace_key("signal_conflicts_text")
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
	note_key = _namespace_key("final_note")
	note_editor_key = _namespace_key("final_note_editor")
	note_editor_sync_key = _namespace_key("final_note_editor_sync_pending")

	st.session_state.setdefault(title_key, "Macro note")
	st.session_state.setdefault(note_type_key, "Market Monitor")
	st.session_state.setdefault(next_catalysts_state_key, _default_next_catalysts_df())
	title_col, type_col = st.columns([2, 1])
	with title_col:
		note_title = st.text_input("Note title", key=title_key)
	with type_col:
		note_type = st.selectbox(
			"Note type",
			["Market Monitor", "Release Reaction", "Issue or Strategy Note"],
			key=note_type_key,
		)
	st.caption("Target length: approximately 300-500 words for a standard monitoring or reaction note. Longer thematic notes can run 600-1,000+ words.")

	st.markdown("### A. Trigger & Catalyst")
	st.caption("Why is this note being written now? The cards below combine public calendars, FRED/ALFRED vintages, and Federal Reserve communications.")
	trigger_cards: list[dict[str, str]] = []
	primary_trigger_text = _primary_market_trigger([primary_move], requested_compare_date) if primary_move is not None else "No dominant market trigger was flagged for the selected comparison horizon."
	trigger_cards.append(
		{
			"title": "Primary market trigger",
			"source": TRIGGER_CATALYST_SOURCES["primary_trigger"],
			"status": "Available" if primary_move is not None else "Unavailable",
			"body": primary_trigger_text,
			"footer": _comparison_basis(primary_move) if primary_move is not None else "Comparison window based on the selected horizon.",
		}
	)
	if scheduled_catalysts.empty:
		event_body, event_status, event_footer = _event_card_summary(scheduled_catalysts)
		trigger_cards.append(
			{
				"title": "Upcoming market-moving catalysts",
				"source": TRIGGER_CATALYST_SOURCES["scheduled_catalysts"],
				"status": event_status,
				"body": event_body,
				"footer": event_footer,
			}
		)
	else:
		event_body, event_status, event_footer = _event_card_summary(scheduled_catalysts)
		trigger_cards.append(
			{
				"title": "Upcoming swing catalysts",
				"source": TRIGGER_CATALYST_SOURCES["scheduled_catalysts"],
				"status": event_status,
				"body": event_body,
				"footer": event_footer,
			}
		)
	trigger_cards.append(
		{
			"title": "Fed & Policy Context",
			"source": TRIGGER_CATALYST_SOURCES["policy_context"],
			"status": "Available" if any((policy_context.get("current_policy") or {}).values()) or policy_context.get("policy_events") else "Unavailable",
			"body": _policy_card_body(policy_context),
			"footer": "Facts only. Use the takeaway to support the Why section of the note.",
		}
	)
	components.html(_summary_card_grid(trigger_cards), height=390, scrolling=False)

	st.markdown("### B. What Moved")
	st.caption(f"Comparison basis: current values vs the {selected_horizon} horizon, using the nearest valid observations.")
	_render_html_table(_move_summary_rows(moves), ["Metric", "Source panel", "Current value", "Direction", "Raw move", "Z-score", "Basis", "Context"], height_floor=220, scrolling=True)

	st.markdown("### C. What Stayed Anchored")
	st.caption("Fresh rows where the move stayed below the anchor threshold and the z-score stayed below 2.0.")
	if anchor_rows:
		_render_html_table(anchor_rows, ["Metric", "Source panel", "Current value", "Direction", "Raw move", "Z-score", "Basis", "Context"], height_floor=200, scrolling=True)
	else:
		st.caption("No rows met the stability screen for the selected horizon.")

	st.markdown("### D. Signal Interpretation")
	st.caption(f"Test competing macro patterns against the {selected_horizon} move.")
	if pattern_evaluations:
		leading_pattern = sorted(pattern_evaluations, key=lambda pattern: (pattern.confidence_score, len(pattern.aligned_signals)), reverse=True)[0]
		st.caption(f"Leading data-consistent pattern: {leading_pattern.pattern_name} ({leading_pattern.alignment_status}, confidence {leading_pattern.confidence_score:.2f}).")
		_render_html_table([_pattern_to_row(pattern) for pattern in pattern_evaluations], ["Pattern", "Alignment", "Confidence", "Takeaway"], height_floor=360, scrolling=True)
	else:
		st.caption("No patterns with meaningful movement were flagged.")
	st.markdown("### E. Research Gaps")
	st.caption("What still needs external verification before the note is complete?")
	if research_gaps:
		for idx, gap in enumerate(research_gaps):
			state_key = _namespace_key(f"research_gap_{idx}")
			st.session_state.setdefault(state_key, gap.completed)
			st.checkbox(f"{gap.task} [{gap.related}]", key=state_key, help=gap.reason)
			st.caption(gap.reason)
	else:
		st.caption("No high-value research gaps were identified.")

	st.markdown("### F. Guided Inputs")
	with st.form("note_workspace_inputs"):
		st.markdown("**1. Trigger**")
		trigger_text = st.text_area(
			"Trigger",
			key=trigger_key,
			height=96,
			placeholder="Why is this note being written now? Identify the specific development — a market move, data release, central-bank decision, or macro shift — that warrants an update. Keep this factual, not interpretive.",
		)
		st.markdown("**2. Bottom Line**")
		bottom_line_text = st.text_area(
			"Bottom Line",
			key=bottom_line_key,
			height=110,
			placeholder="What does the trigger mean, and what's your view? State the interpretation, market implication, preferred positioning, and time horizon — a reader should understand your conclusion from this section alone.",
		)
		st.markdown("**3. What Moved**")
		st.caption("Cite the specific metrics, moves, and percentiles you're referencing from the 'What Moved' table above.")
		what_moved_text = st.text_area(
			"What Moved",
			key=what_moved_key,
			height=110,
			placeholder="Quantify the relevant release or market move — which tenor, spread, factor, or asset drove it, the size and direction, and how unusual it is relative to recent history. Keep this descriptive, not interpretive.",
		)
		st.markdown("**4. Why It Matters**")
		why_text = st.text_area(
			"Why It Matters",
			key=why_key,
			height=110,
			placeholder="What's the most likely explanation, and how does it affect the macro narrative? Rank the main drivers (data surprises, policy repricing, inflation/growth repricing, positioning/technicals, Treasury supply, risk sentiment) — distinguish the primary explanation from secondary ones.",
		)
		st.markdown("**5. Confirmation and Conflicting Signals**")
		st.caption("Cite which signals from the 'Where Signals Disagree' table above confirm, partially confirm, diverge, or are inconclusive.")
		signal_conflicts_text = st.text_area(
			"Confirmation and Conflicting Signals",
			key=signal_conflicts_key,
			height=110,
			placeholder="Do other markets and indicators support the interpretation? Note confirming, partially confirming, diverging, or inconclusive signals — the dollar, equities, credit spreads, volatility, commodities, inflation expectations, or other parts of the curve.",
		)
		st.markdown("**6. Broader Context**")
		broader_context_text = st.text_area(
			"Broader Context",
			key=broader_context_key,
			height=110,
			placeholder="How does this fit the wider macro regime? Connect the trigger to only the most relevant medium-term themes (growth/inflation outlook, labor momentum, policy path, fiscal/Treasury supply, term premium, positioning, or the evolution of a prior thesis). Skip if the sections above already cover it — do not summarize every panel here.",
		)
		st.markdown("**7. Trade Idea or Position**")
		st.caption("Start with the simplest version of the idea. If you are not planning a trade, say so plainly and use the section as a watchlist note.")
		trade_instrument = st.text_input(
			"What are you considering?",
			key=trade_instrument_key,
			placeholder="e.g. 2s10s flattener, long 5Y note futures, no trade / watchlist only",
		)
		trade_direction = st.text_input(
			"Direction / stance",
			key=trade_direction_key,
			placeholder="e.g. long, short, steeper, flatter, wider, tighter, no trade",
		)
		trade_horizon = st.text_input(
			"Time horizon",
			key=trade_horizon_key,
			placeholder="e.g. days, weeks, months",
		)
		trade_primary_driver = st.text_area(
			"Why this trade?",
			key=trade_primary_driver_key,
			height=88,
			placeholder="In one or two sentences, explain the main reason you would take this trade.",
		)
		trade_main_risk = st.text_area(
			"What could go wrong?",
			key=trade_main_risk_key,
			height=88,
			placeholder="What is the key risk to the idea, or the main reason to stay out?",
		)
		with st.expander("Optional trade details", expanded=False):
			st.caption(
				"Use these only if you have a clear plan. They are helpful for more advanced users but not required for a useful note."
			)
			expr_cols = st.columns(3)
			with expr_cols[0]:
				trade_entry = st.text_input("Entry level", key=trade_entry_key)
				trade_target = st.text_input("Target", key=trade_target_key)
				trade_stop = st.text_input("Stop / invalidation level", key=trade_stop_key)
			with expr_cols[1]:
				trade_risk_reward = st.text_input("Expected risk-reward", key=trade_risk_reward_key)
				trade_carry_roll = st.text_input("Carry and roll", key=trade_carry_roll_key)
				trade_sizing = st.text_input("Sizing and conviction", key=trade_sizing_key)
			with expr_cols[2]:
				trade_catalyst = st.text_area(
					"Expected catalyst",
					key=trade_catalyst_key,
					height=88,
					placeholder="What event or data release could make the trade work?",
				)
		st.markdown("**8. What Would Change My Mind**")
		invalidation_text = st.text_area(
			"What Would Change My Mind",
			key=invalidation_key,
			height=110,
			placeholder="What specific, observable development would invalidate the view? Use a measurable condition — a data outcome, yield/spread level, policy signal, cross-market development, or defined time horizon. Avoid vague conditions.",
		)
		with st.expander("Next Catalysts (optional)", expanded=False):
			next_catalysts_df = st.data_editor(
				st.session_state[next_catalysts_state_key],
				key=next_catalysts_key,
				num_rows="dynamic",
				use_container_width=True,
				hide_index=True,
				column_config={
					"Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
				},
			)
			st.session_state[next_catalysts_state_key] = next_catalysts_df
		st.markdown("**Final Review**")
		review_items = [
			"Trigger makes clear why the note is being written now",
			"Bottom Line explains what the trigger means",
			"Facts are separated from interpretation",
			"The main move is quantified and placed in historical context",
			"Confirming and conflicting evidence are both acknowledged",
			"The trade follows logically from the analysis",
			"The invalidation condition is specific and testable",
		]
		for idx, item in enumerate(review_items):
			st.checkbox(item, key=_namespace_key(f"review_check_{idx}"))
		submitted = st.form_submit_button("Generate note")

	if submitted:
		if not invalidation_text.strip():
			st.error("Invalidation is required before generating the note.")
		else:
			st.session_state[note_key] = _build_note(
				note_title,
				note_type,
				trigger_text,
				bottom_line_text,
				what_moved_text,
				why_text,
				signal_conflicts_text,
				broader_context_text,
				{
					"trade_instrument": trade_instrument,
					"trade_direction": trade_direction,
					"trade_horizon": trade_horizon,
					"trade_entry": trade_entry,
					"trade_target": trade_target,
					"trade_stop": trade_stop,
					"trade_risk_reward": trade_risk_reward,
					"trade_carry_roll": trade_carry_roll,
					"trade_sizing": trade_sizing,
				},
				{
					"trade_primary_driver": trade_primary_driver,
					"trade_catalyst": trade_catalyst,
					"trade_main_risk": trade_main_risk,
				},
				invalidation_text,
				st.session_state.get(next_catalysts_state_key),
			)
			st.session_state[note_editor_sync_key] = True
			st.rerun()

	if note_key not in st.session_state:
		st.session_state[note_key] = _build_note(
			note_title,
			st.session_state.get(note_type_key, "Market Monitor"),
			st.session_state.get(trigger_key, ""),
			st.session_state.get(bottom_line_key, ""),
			st.session_state.get(what_moved_key, ""),
			st.session_state.get(why_key, ""),
			st.session_state.get(signal_conflicts_key, ""),
			st.session_state.get(broader_context_key, ""),
			{
				"trade_instrument": st.session_state.get(trade_instrument_key, ""),
				"trade_direction": st.session_state.get(trade_direction_key, ""),
				"trade_horizon": st.session_state.get(trade_horizon_key, ""),
				"trade_entry": st.session_state.get(trade_entry_key, ""),
				"trade_target": st.session_state.get(trade_target_key, ""),
				"trade_stop": st.session_state.get(trade_stop_key, ""),
				"trade_risk_reward": st.session_state.get(trade_risk_reward_key, ""),
				"trade_carry_roll": st.session_state.get(trade_carry_roll_key, ""),
				"trade_sizing": st.session_state.get(trade_sizing_key, ""),
			},
			{
				"trade_primary_driver": st.session_state.get(trade_primary_driver_key, ""),
				"trade_catalyst": st.session_state.get(trade_catalyst_key, ""),
				"trade_main_risk": st.session_state.get(trade_main_risk_key, ""),
			},
			st.session_state.get(invalidation_key, ""),
			st.session_state.get(next_catalysts_key),
		)
	if st.session_state.get(note_editor_sync_key) or note_editor_key not in st.session_state:
		st.session_state[note_editor_key] = st.session_state[note_key]
		st.session_state[note_editor_sync_key] = False

	st.markdown("### Final Note")
	edited_note = st.text_area("Editable markdown note", key=note_editor_key, height=620)

	col_refresh, col_save, col_reset = st.columns(3)
	with col_refresh:
		if st.button("Refresh draft from current evidence"):
			st.session_state[note_key] = _build_note(
				note_title,
				st.session_state.get(note_type_key, "Market Monitor"),
				st.session_state.get(trigger_key, ""),
				st.session_state.get(bottom_line_key, ""),
				st.session_state.get(what_moved_key, ""),
				st.session_state.get(why_key, ""),
				st.session_state.get(signal_conflicts_key, ""),
				st.session_state.get(broader_context_key, ""),
				{
					"trade_instrument": st.session_state.get(trade_instrument_key, ""),
					"trade_direction": st.session_state.get(trade_direction_key, ""),
					"trade_horizon": st.session_state.get(trade_horizon_key, ""),
					"trade_entry": st.session_state.get(trade_entry_key, ""),
					"trade_target": st.session_state.get(trade_target_key, ""),
					"trade_stop": st.session_state.get(trade_stop_key, ""),
					"trade_risk_reward": st.session_state.get(trade_risk_reward_key, ""),
					"trade_carry_roll": st.session_state.get(trade_carry_roll_key, ""),
					"trade_sizing": st.session_state.get(trade_sizing_key, ""),
				},
				{
					"trade_primary_driver": st.session_state.get(trade_primary_driver_key, ""),
					"trade_catalyst": st.session_state.get(trade_catalyst_key, ""),
					"trade_main_risk": st.session_state.get(trade_main_risk_key, ""),
				},
				st.session_state.get(invalidation_key, ""),
				st.session_state.get(next_catalysts_state_key),
			)
			st.session_state[note_editor_sync_key] = True
			st.rerun()
	with col_save:
		if st.button("Save note"):
			if not st.session_state.get(invalidation_key, "").strip():
				st.error("Cannot save without invalidation text.")
			else:
				saved_path = _archive_note(note_title, st.session_state[note_editor_key])
				st.success(f"Saved {saved_path.name}")
	with col_reset:
		if st.button("Reset final note"):
			st.session_state[note_key] = _build_note(
				note_title,
				st.session_state.get(note_type_key, "Market Monitor"),
				"",
				"",
				"",
				"",
				"",
				"",
				{
					"trade_instrument": "",
					"trade_direction": "",
					"trade_horizon": "",
					"trade_entry": "",
					"trade_target": "",
					"trade_stop": "",
					"trade_risk_reward": "",
					"trade_carry_roll": "",
					"trade_sizing": "",
				},
				{
					"trade_primary_driver": "",
					"trade_catalyst": "",
					"trade_main_risk": "",
				},
				"",
				None,
			)
			st.session_state[note_editor_sync_key] = True
			st.rerun()

	st.markdown("### Save / Archive")
	st.caption(f"Archived notes are stored in {NOTE_WORKSPACE_ARCHIVE_DIR.name}.")
	_archive_browser()

	st.markdown("### Methodology")
	st.caption(
		"This workspace reuses the current panel signals already computed elsewhere in the app. It ranks recent moves from those panel outputs, flags simple co-movement patterns, and collects your manual note fields before saving a markdown archive."
	)
