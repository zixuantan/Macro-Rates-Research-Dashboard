from __future__ import annotations

import math

import pandas as pd

from panels.cross_asset import _credit_figure, _equity_figure, _nfci_reading, _rebased_series


def test_nfci_positive_is_tighter_than_average() -> None:
    label, description = _nfci_reading(0.25, 0.10, "over the past month")

    assert label == "Tighter than average"
    assert "tightened by 0.10 index points" in description


def test_nfci_negative_is_looser_than_average() -> None:
    label, description = _nfci_reading(-0.40, -0.08, "over the past month")

    assert label == "Looser than average"
    assert "loosened by 0.08 index points" in description


def test_nfci_unavailable_is_explicit() -> None:
    label, _ = _nfci_reading(math.nan, math.nan, "over the past month")

    assert label == "Unavailable"


def test_credit_chart_converts_fred_percentage_points_to_basis_points() -> None:
    index = pd.to_datetime(["2026-01-01"])
    figure = _credit_figure(
        pd.Series([2.5], index=index),
        pd.Series([0.8], index=index),
    )

    assert figure.data[0].y[0] == 250.0
    assert figure.data[1].y[0] == 80.0
    assert figure.data[0].hovertemplate == "HY OAS: %{y:.0f} bp<extra></extra>"
    assert figure.data[1].hovertemplate == "IG OAS: %{y:.0f} bp<extra></extra>"


def test_equity_chart_groups_sp500_direction_with_vix() -> None:
    index = pd.to_datetime(["2026-01-01", "2026-01-02"])
    figure = _equity_figure(
        pd.Series([5000.0, 5100.0], index=index),
        pd.Series([18.0, 17.0], index=index),
        index[0],
    )

    assert figure.data[0].name == "S&P 500 · rebased"
    assert figure.data[0].y[-1] == 102.0
    assert figure.data[1].name == "VIX"


def test_rebased_series_uses_first_available_observation_after_start_date() -> None:
    index = pd.to_datetime(["2026-01-05", "2026-01-06"])
    rebased = _rebased_series(
        pd.Series([120.0, 126.0], index=index),
        pd.Timestamp("2026-01-03"),
    )

    assert rebased.iloc[0] == 100.0
    assert rebased.iloc[1] == 105.0
