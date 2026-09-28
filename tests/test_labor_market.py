from __future__ import annotations

import pandas as pd

from panels.labor_market import _change_over_horizon


def test_change_over_horizon_uses_series_own_latest_release() -> None:
    series = pd.Series(
        [200.0, 210.0, 225.0],
        index=pd.to_datetime(["2026-01-09", "2026-02-06", "2026-03-06"]),
    )

    change = _change_over_horizon(series, pd.DateOffset(months=1))

    assert change == 15.0


def test_change_over_horizon_uses_latest_observation_before_target_date() -> None:
    series = pd.Series(
        [200.0, 205.0, 212.0],
        index=pd.to_datetime(["2026-01-30", "2026-02-27", "2026-03-06"]),
    )

    change = _change_over_horizon(series, pd.DateOffset(months=1))

    assert change == 12.0
