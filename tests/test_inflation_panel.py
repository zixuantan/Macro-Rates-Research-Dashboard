import pandas as pd

from panels.inflation import (
    INFLATION_MEASURE_GUIDE,
    _annualized_change,
)


def test_inflation_measure_guide_covers_requested_series() -> None:
    assert [measure[0] for measure in INFLATION_MEASURE_GUIDE] == [
        "CPI",
        "Core CPI",
        "PCE",
        "Core PCE",
        "PPI",
    ]


def test_three_month_annualized_change() -> None:
    index = pd.date_range("2025-01-31", periods=4, freq="ME")
    series = pd.Series([100.0, 100.5, 101.0, 101.5], index=index)

    result = _annualized_change(series, 3)

    expected = ((101.5 / 100.0) ** 4 - 1.0) * 100.0
    assert abs(result.iloc[-1] - expected) < 1e-10
