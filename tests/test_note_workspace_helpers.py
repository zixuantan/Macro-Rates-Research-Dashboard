from datetime import date
from types import SimpleNamespace
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

if "streamlit" not in sys.modules:
	class _StubContext:
		def __enter__(self):
			return self

		def __exit__(self, exc_type, exc, tb):
			return False

	streamlit_stub = SimpleNamespace(
		cache_data=lambda *dargs, **dkwargs: (lambda func: func) if dargs and callable(dargs[0]) else (lambda func: func),
		cache_resource=lambda *dargs, **dkwargs: (lambda func: func) if dargs and callable(dargs[0]) else (lambda func: func),
		caption=lambda *args, **kwargs: None,
		info=lambda *args, **kwargs: None,
		markdown=lambda *args, **kwargs: None,
		subheader=lambda *args, **kwargs: None,
		text_area=lambda *args, **kwargs: "",
		text_input=lambda *args, **kwargs: "",
		selectbox=lambda *args, **kwargs: "",
		checkbox=lambda *args, **kwargs: False,
		button=lambda *args, **kwargs: False,
		columns=lambda *args, **kwargs: (_StubContext(), _StubContext(), _StubContext()),
		expander=lambda *args, **kwargs: _StubContext(),
		form=lambda *args, **kwargs: _StubContext(),
		form_submit_button=lambda *args, **kwargs: False,
		date_input=lambda *args, **kwargs: date(2026, 1, 1),
		rerun=lambda *args, **kwargs: None,
		session_state={},
		column_config=SimpleNamespace(DateColumn=lambda *args, **kwargs: None),
	)
	components_stub = SimpleNamespace(html=lambda *args, **kwargs: None)
	streamlit_stub.components = SimpleNamespace(v1=components_stub)
	sys.modules["streamlit"] = streamlit_stub
	sys.modules["streamlit.components"] = SimpleNamespace(v1=components_stub)
	sys.modules["streamlit.components.v1"] = components_stub
if "dotenv" not in sys.modules:
	sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)
if "certifi" not in sys.modules:
	sys.modules["certifi"] = SimpleNamespace(where=lambda: "/tmp/cert.pem")
if "requests" not in sys.modules:
	sys.modules["requests"] = SimpleNamespace(get=lambda *args, **kwargs: SimpleNamespace(status_code=404, text="", json=lambda: {}))
if "fredapi" not in sys.modules:
	sys.modules["fredapi"] = SimpleNamespace(Fred=object)
sys.modules.setdefault("panels.guided_research", SimpleNamespace(_build_panel_analyses=lambda *args, **kwargs: []))

from panels import note_workspace as nw


def _metric(
	metric_id: str,
	label: str,
	*,
	value: float,
	change: float,
	unit: str,
	change_unit: str,
	standardized_change: float | None = None,
) -> nw.WorkspaceMetric:
	return nw.WorkspaceMetric(
		metric_id=metric_id,
		panel_id="test",
		panel_title="Test",
		label=label,
		value=value,
		unit=unit,
		change=change,
		change_unit=change_unit,
		horizon="1M",
		percentile=50.0,
		standardized_change=standardized_change,
		direction="higher",
		interpretation=label,
		caveat=None,
		as_of="2026-01-01",
		group_id=None,
	)


def test_comparison_move_converts_yield_changes_to_basis_points() -> None:
	history = {
		"yield_curve": pd.DataFrame(
			{"DGS2": [4.0, 4.1, 4.2, 4.5]},
			index=pd.to_datetime(["2025-12-01", "2025-12-15", "2025-12-31", "2026-01-01"]),
		)
	}
	metric = nw.WorkspaceMetric(
		metric_id="yield_2y",
		panel_id="yield_curve",
		panel_title="Yield Curve",
		label="2Y Treasury yield",
		value=4.5,
		unit="%",
		change=0.5,
		change_unit="%",
		horizon="1M",
		percentile=60.0,
		standardized_change=1.0,
		direction="higher",
		interpretation="",
		caveat=None,
		as_of="2026-01-01",
		group_id=None,
	)
	comparison = nw._comparison_move(metric, pd.Timestamp("2025-12-15"), history)
	assert comparison.change_unit == "bp"
	assert abs(comparison.change - 40.0) < 1e-6
	assert 2.0 < float(comparison.standardized_change) < 4.0
	assert abs(float(comparison.value) - 4.5) < 1e-6


def test_move_summary_rows_keep_rate_values_in_percent() -> None:
	move = nw.MoveResult(
		metric_id="yield_2y",
		label="2Y Treasury yield",
		panel_title="Yield Curve",
		category="Rates",
		frequency="daily",
		selected_horizon="1M",
		requested_current_date="2026-01-01",
		requested_comparison_date="2025-12-15",
		effective_current_date="2026-01-01",
		effective_comparison_date="2025-12-15",
		horizon_observation_count=1,
		direction="higher",
		current_value=4.26,
		comparison_value=4.07,
		raw_move=19.0,
		raw_move_unit="bp",
		historical_mean_move=15.0,
		historical_std_move=4.0,
		z_score=3.1,
		historical_sample_count=8,
		freshness_status="Fresh",
		data_quality_status=None,
		current_unit="%",
		historical_context="roughly mid-range in history",
		quality_flag=None,
		percentile=55.0,
	)
	rows = nw._move_summary_rows([move])
	assert rows[0]["Current value"] == "4.26 %"
	assert rows[0]["Raw move"] == "+19.00 bp"


def test_comparison_move_does_not_double_scale_ns_level() -> None:
	history = {
		"yield_curve": pd.DataFrame(
			{"DGS10": [4.0, 4.1, 4.2, 4.5]},
			index=pd.to_datetime(["2025-12-01", "2025-12-15", "2025-12-31", "2026-01-01"]),
		)
	}
	metric = nw.WorkspaceMetric(
		metric_id="ns_level",
		panel_id="nelson_siegel",
		panel_title="Nelson Siegel",
		label="NS level",
		value=4.5,
		unit="%",
		change=0.0,
		change_unit="bp",
		horizon="1M",
		percentile=60.0,
		standardized_change=1.0,
		direction="higher",
		interpretation="",
		caveat=None,
		as_of="2026-01-01",
		group_id=None,
	)
	comparison = nw._comparison_move(metric, pd.Timestamp("2025-12-15"), history)
	assert comparison.change_unit == "bp"
	assert abs(comparison.change - 40.0) < 1e-6
	assert 2.0 < float(comparison.standardized_change) < 4.0
	assert abs(float(comparison.value) - 4.5) < 1e-6


def test_credit_spreads_are_scaled_to_basis_points_and_round_trip_cleanly() -> None:
	history = {
		"cross_asset": pd.DataFrame(
			{"BAMLH0A0HYM2": [2.55, 2.69]},
			index=pd.to_datetime(["2025-12-15", "2026-01-01"]),
		)
	}
	metric = nw.WorkspaceMetric(
		metric_id="credit_hy",
		panel_id="cross_asset",
		panel_title="Cross Asset",
		label="HY OAS",
		value=269.0,
		unit="bp",
		change=0.0,
		change_unit="bp",
		horizon="1M",
		percentile=60.0,
		standardized_change=1.0,
		direction="wider",
		interpretation="",
		caveat=None,
		as_of="2026-01-01",
		group_id=None,
	)
	comparison = nw._comparison_move(metric, pd.Timestamp("2025-12-15"), history)
	assert comparison.change_unit == "bp"
	assert abs(comparison.change - 14.0) < 1e-6
	assert abs(float(comparison.value) - 269.0) < 1e-6
	base_value = float(comparison.value) - float(comparison.change)
	assert abs(base_value - 255.0) < 1e-6


def test_ns_level_and_ten_year_yield_share_the_same_context_series() -> None:
	history = {
		"yield_curve": pd.DataFrame(
			{"DGS10": [4.0, 4.1, 4.2, 4.5]},
			index=pd.to_datetime(["2025-12-01", "2025-12-15", "2025-12-31", "2026-01-01"]),
		)
	}
	yield_series = nw._comparison_series("yield_10y", history)
	ns_series = nw._comparison_series("ns_level", history)
	assert yield_series is not None
	assert ns_series is not None
	pd.testing.assert_series_equal(yield_series, ns_series)
	yield_metric = _metric("yield_10y", "10Y Treasury yield", value=4.5, change=0.0, unit="%", change_unit="%")
	ns_metric = _metric("ns_level", "NS level", value=4.5, change=0.0, unit="%", change_unit="%")
	assert nw._context_phrase(yield_metric, history) == nw._context_phrase(ns_metric, history)


def test_calculate_zscore_uses_matching_horizon_volatility() -> None:
	assert nw._calculate_zscore(-0.70, 0.4666666666666667) == pytest.approx(-1.5)
	assert nw._calculate_zscore(-0.70, 0.05) == pytest.approx(-14.0)


def test_scale_validator_flags_credit_and_cpi_unit_mismatches() -> None:
	cpi_issues = nw._validate_scale_consistency("inflation_cpi", 3.5, -70.0, "%", "bp")
	bad_cpi_issues = nw._validate_scale_consistency("inflation_cpi", 3.5, -0.7, "%", "%")
	credit_issues = nw._validate_scale_consistency("credit_hy", 2.69, 0.14, "bp", "bp")
	assert cpi_issues == []
	assert any("expected change unit bp" in issue for issue in bad_cpi_issues)
	assert any("looks like percent points" in issue for issue in credit_issues)


@pytest.mark.parametrize(
	"metric, history, compare_date, expected_base, expected_current_unit, expected_change_unit",
	[
		(
			nw.WorkspaceMetric(
				metric_id="yield_2y",
				panel_id="yield_curve",
				panel_title="Yield Curve",
				label="2Y Treasury yield",
				value=4.5,
				unit="%",
				change=0.0,
				change_unit="%",
				horizon="1M",
				percentile=60.0,
				standardized_change=1.0,
				direction="higher",
				interpretation="",
				caveat=None,
				as_of="2026-01-01",
				group_id=None,
			),
			{"yield_curve": pd.DataFrame({"DGS2": [4.0, 4.1, 4.5]}, index=pd.to_datetime(["2025-12-01", "2025-12-15", "2026-01-01"]))},
			pd.Timestamp("2025-12-15"),
			4.1,
			"%",
			"bp",
		),
		(
			nw.WorkspaceMetric(
				metric_id="ns_level",
				panel_id="nelson_siegel",
				panel_title="Nelson Siegel",
				label="NS level",
				value=4.5,
				unit="%",
				change=0.0,
				change_unit="%",
				horizon="1M",
				percentile=60.0,
				standardized_change=1.0,
				direction="higher",
				interpretation="",
				caveat=None,
				as_of="2026-01-01",
				group_id=None,
			),
			{"yield_curve": pd.DataFrame({"DGS10": [4.0, 4.1, 4.5]}, index=pd.to_datetime(["2025-12-01", "2025-12-15", "2026-01-01"]))},
			pd.Timestamp("2025-12-15"),
			4.1,
			"%",
			"bp",
		),
		(
			nw.WorkspaceMetric(
				metric_id="credit_hy",
				panel_id="cross_asset",
				panel_title="Cross Asset",
				label="HY OAS",
				value=269.0,
				unit="bp",
				change=0.0,
				change_unit="bp",
				horizon="1M",
				percentile=60.0,
				standardized_change=1.0,
				direction="wider",
				interpretation="",
				caveat=None,
				as_of="2026-01-01",
				group_id=None,
			),
			{"cross_asset": pd.DataFrame({"BAMLH0A0HYM2": [2.55, 2.69]}, index=pd.to_datetime(["2025-12-15", "2026-01-01"]))},
			pd.Timestamp("2025-12-15"),
			255.0,
			"bp",
			"bp",
		),
		(
			nw.WorkspaceMetric(
				metric_id="inflation_cpi",
				panel_id="inflation",
				panel_title="Inflation",
				label="Headline CPI YoY (SA)",
				value=3.5,
				unit="%",
				change=0.0,
				change_unit="bp",
				horizon="1M",
				percentile=60.0,
				standardized_change=1.0,
				direction="lower",
				interpretation="",
				caveat=None,
				as_of="2026-01-31",
				group_id=None,
			),
			{
				"inflation": {
					"raw": pd.DataFrame(
						{"CPIAUCSL": [100.0, 100.0] + [100.0] * 10 + [104.2, 103.5]},
						index=pd.date_range("2024-12-31", "2026-01-31", freq="ME"),
					)
				}
			},
			pd.Timestamp("2025-12-31"),
			4.2,
			"%",
			"bp",
		),
		(
			nw.WorkspaceMetric(
				metric_id="curve_2s10s",
				panel_id="yield_curve",
				panel_title="Yield Curve",
				label="2s10s spread",
				value=10.0,
				unit="bp",
				change=0.0,
				change_unit="bp",
				horizon="1M",
				percentile=60.0,
				standardized_change=1.0,
				direction="lower",
				interpretation="",
				caveat=None,
				as_of="2026-01-01",
				group_id=None,
			),
			{"yield_curve": pd.DataFrame({"DGS2": [4.0, 4.1, 4.5], "DGS10": [4.14, 4.25, 4.60]}, index=pd.to_datetime(["2025-12-01", "2025-12-15", "2026-01-01"]))},
			pd.Timestamp("2025-12-15"),
			15.0,
			"bp",
			"bp",
		),
	],
)
def test_current_value_minus_raw_move_round_trips_to_historical_base(
	metric: nw.WorkspaceMetric,
	history: dict[str, object],
	compare_date: pd.Timestamp,
	expected_base: float,
	expected_current_unit: str,
	expected_change_unit: str,
) -> None:
	comparison = nw._comparison_move(metric, compare_date, history)
	assert comparison.change_unit == expected_change_unit
	assert comparison.unit == metric.unit
	assert metric.unit == expected_current_unit
	normalized_raw_move = comparison.change / 100.0 if comparison.unit == "%" and comparison.change_unit == "bp" else comparison.change
	assert normalized_raw_move is not None
	assert float(comparison.value) - float(normalized_raw_move) == pytest.approx(expected_base)


def test_pattern_row_returns_structured_evaluation() -> None:
	metric_map = {
		"curve_2s10s": _metric("curve_2s10s", "2s10s", value=10.0, change=-5.0, unit="bp", change_unit="bp", standardized_change=-0.8),
		"credit_hy": _metric("credit_hy", "HY OAS", value=350.0, change=15.0, unit="bp", change_unit="bp", standardized_change=0.9),
		"vix": _metric("vix", "VIX", value=18.0, change=1.5, unit="index", change_unit="index", standardized_change=0.5),
	}
	evaluation = nw._pattern_row(nw.MOVE_PATTERNS[0], metric_map)
	assert evaluation is not None
	assert evaluation.pattern_name == "Risk-off pattern"
	assert evaluation.alignment_status in {"Mostly aligned", "Mixed alignment", "Fully aligned"}
	assert evaluation.aligned_signals


def test_research_gaps_cover_unavailable_context() -> None:
	primary_move = nw.MoveResult(
		metric_id="yield_2y",
		label="2Y Treasury yield",
		panel_title="Yield Curve",
		category="Rates",
		frequency="daily",
		selected_horizon="1M",
		requested_current_date="2026-01-01",
		requested_comparison_date="2025-12-01",
		effective_current_date="2026-01-01",
		effective_comparison_date="2025-12-01",
		horizon_observation_count=1,
		direction="higher",
		current_value=4.5,
		comparison_value=4.0,
		raw_move=50.0,
		raw_move_unit="bp",
		historical_mean_move=25.0,
		historical_std_move=2.0,
		z_score=12.0,
		historical_sample_count=8,
		freshness_status="Fresh",
		data_quality_status="Extreme z-score +12.0",
		current_unit="%",
		historical_context="near the top of history",
		percentile=90.0,
		quality_flag="warning: z-score +12.0 looks extreme",
	)
	gaps = nw._build_research_gaps(
		primary_move,
		nw.get_scheduled_catalysts(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01")),
		nw.get_policy_context(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")),
		[],
	)
	assert gaps
	assert any("policy" in gap.task.lower() for gap in gaps)


def test_empty_external_helpers_return_expected_schemas() -> None:
	catalysts = nw.get_scheduled_catalysts(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01"))
	assert list(catalysts.columns) == ["event_date", "event_time", "event_name", "event_type", "source", "importance", "release_id"]
