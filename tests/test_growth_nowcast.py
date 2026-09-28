import pandas as pd

from config import CANDH, EUANDH, NFP_SECTOR_SERIES, PANDI, PAYEMS, SOANDI
from panels.growth_nowcast import (
    _annualized_three_month_growth,
    _build_cfnai_category_snapshot,
    _monthly_payroll_changes,
)


def test_three_month_growth_is_annualized() -> None:
    index = pd.date_range("2025-01-31", periods=4, freq="ME")
    series = pd.Series([100.0, 100.5, 101.0, 101.5], index=index)

    result = _annualized_three_month_growth(series)

    expected = ((101.5 / 100.0) ** 4 - 1.0) * 100.0
    assert abs(result.iloc[-1] - expected) < 1e-10


def test_cfnai_snapshot_uses_official_categories() -> None:
    index = pd.to_datetime(["2025-01-01", "2025-02-01"])
    data = pd.DataFrame(
        {
            PANDI: [0.1, 0.2],
            EUANDH: [0.2, 0.1],
            CANDH: [-0.1, -0.2],
            SOANDI: [0.0, 0.1],
        },
        index=index,
    )

    snapshot, latest_date = _build_cfnai_category_snapshot(data)

    assert latest_date == pd.Timestamp("2025-02-28")
    assert len(snapshot) == 4
    assert abs(snapshot["Contribution"].sum() - 0.2) < 1e-10
    assert set(snapshot["Role"]) == {"Support", "Drag"}


def test_payroll_sector_changes_reconcile_with_headline() -> None:
    index = pd.to_datetime(["2025-01-01", "2025-02-01"])
    sector_changes = range(1, len(NFP_SECTOR_SERIES) + 1)
    data = {
        series_id: [100.0, 100.0 + change]
        for series_id, change in zip(NFP_SECTOR_SERIES, sector_changes)
    }
    data[PAYEMS] = [1_000.0, 1_000.0 + sum(sector_changes)]

    result = _monthly_payroll_changes(pd.DataFrame(data, index=index)).dropna()

    assert result.iloc[-1][PAYEMS] == result.iloc[-1][NFP_SECTOR_SERIES].sum()
