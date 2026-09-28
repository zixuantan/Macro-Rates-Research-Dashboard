from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.yield_curve import analyze_curve, classify_curve_move, curve_snapshot
from config import DGS10, DGS2, YIELD_SERIES


def _curve_row(two_year: float, ten_year: float) -> dict[str, float]:
    row = {series_id: 4.0 for series_id in YIELD_SERIES}
    row[DGS2] = two_year
    row[DGS10] = ten_year
    return row


def test_snapshot_uses_one_complete_observation_date() -> None:
    friday = _curve_row(4.10, 4.30)
    monday_partial = _curve_row(4.20, 4.40)
    monday_partial[YIELD_SERIES[0]] = np.nan
    data = pd.DataFrame(
        [friday, monday_partial],
        index=pd.to_datetime(["2026-09-25", "2026-09-28"]),
    )

    snapshot = curve_snapshot(data, pd.Timestamp("2026-09-28"))

    assert snapshot is not None
    assert snapshot.date == pd.Timestamp("2026-09-25")
    assert snapshot.is_complete
    assert snapshot.values[DGS2] == 4.10


def test_snapshot_falls_back_to_partial_when_no_complete_curve_exists() -> None:
    row = _curve_row(4.10, 4.30)
    row[YIELD_SERIES[0]] = np.nan
    data = pd.DataFrame([row], index=pd.to_datetime(["2026-09-28"]))

    snapshot = curve_snapshot(data, pd.Timestamp("2026-09-28"))

    assert snapshot is not None
    assert not snapshot.is_complete
    assert pd.isna(snapshot.values[YIELD_SERIES[0]])


def test_analysis_uses_actual_prior_business_date_and_basis_points() -> None:
    data = pd.DataFrame(
        [_curve_row(4.00, 4.20), _curve_row(4.10, 4.25)],
        index=pd.to_datetime(["2026-09-25", "2026-09-28"]),
    )

    result = analyze_curve(
        data,
        pd.Timestamp("2026-09-28"),
        pd.Timestamp("2026-09-27"),
    )

    assert result is not None
    assert result.comparison.date == pd.Timestamp("2026-09-25")
    assert np.isclose(result.changes_bp[DGS2], 10.0)
    assert np.isclose(result.changes_bp[DGS10], 5.0)
    assert np.isclose(result.spread_changes_bp["2s10s"], -5.0)
    assert result.regime == "Bear flattening"


def test_curve_move_classification_covers_parallel_and_twist_moves() -> None:
    assert classify_curve_move(10.0, 10.5) == "Parallel bear shift"
    assert classify_curve_move(-8.0, -7.5) == "Parallel bull shift"
    assert classify_curve_move(5.0, -5.0) == "Twist flattening"
    assert classify_curve_move(-5.0, 5.0) == "Twist steepening"
