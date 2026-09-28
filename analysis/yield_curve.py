from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import DGS10, DGS2, DGS30, DGS5, YIELD_SERIES


@dataclass(frozen=True)
class CurveSnapshot:
    """A same-day Treasury curve observation."""

    date: pd.Timestamp
    values: pd.Series
    is_complete: bool


@dataclass(frozen=True)
class CurveAnalysis:
    """Current and comparison snapshots plus their derived measures."""

    current: CurveSnapshot
    comparison: CurveSnapshot
    changes_bp: pd.Series
    current_spreads_bp: pd.Series
    comparison_spreads_bp: pd.Series
    spread_changes_bp: pd.Series
    regime: str


def numeric_yield_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return sorted numeric yield data with all configured columns present."""
    numeric = data.copy().sort_index()
    for series_id in YIELD_SERIES:
        if series_id not in numeric.columns:
            numeric[series_id] = float("nan")
        numeric[series_id] = pd.to_numeric(numeric[series_id], errors="coerce")
    return numeric.loc[:, YIELD_SERIES].astype("float64")


def curve_snapshot(
    data: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    prefer_complete: bool = True,
) -> CurveSnapshot | None:
    """Return one internally consistent curve on or before ``as_of``."""
    eligible = numeric_yield_data(data).loc[lambda frame: frame.index <= as_of]
    if eligible.empty:
        return None

    usable = eligible.dropna(how="all")
    if usable.empty:
        return None

    complete = eligible.dropna(how="any")
    if prefer_complete and not complete.empty:
        actual_date = pd.Timestamp(complete.index[-1])
    else:
        actual_date = pd.Timestamp(usable.index[-1])

    values = eligible.loc[actual_date].copy()
    return CurveSnapshot(
        date=actual_date,
        values=values,
        is_complete=bool(values.notna().all()),
    )


def snapshot_spreads(values: pd.Series) -> pd.Series:
    """Calculate standard curve spreads from a same-day snapshot."""
    return pd.Series(
        {
            "2s10s": (values.get(DGS10) - values.get(DGS2)) * 100.0,
            "5s30s": (values.get(DGS30) - values.get(DGS5)) * 100.0,
        },
        dtype="float64",
    )


def spread_history(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate same-day 2s10s and 5s30s histories."""
    numeric = numeric_yield_data(data)
    return pd.DataFrame(
        {
            "2s10s": (numeric[DGS10] - numeric[DGS2]) * 100.0,
            "5s30s": (numeric[DGS30] - numeric[DGS5]) * 100.0,
        },
        index=numeric.index,
    )


def classify_curve_move(front_end_change: float, long_end_change: float) -> str:
    """Classify a 2s10s move from the 2Y and 10Y yield changes."""
    if pd.isna(front_end_change) or pd.isna(long_end_change):
        return "Unavailable"

    spread_change = long_end_change - front_end_change
    average_change = (front_end_change + long_end_change) / 2.0

    if abs(spread_change) < 2 and abs(average_change) < 2:
        return "Broadly unchanged"
    if abs(spread_change) < 2:
        return "Parallel bear shift" if average_change > 0 else "Parallel bull shift"

    direction = "steepening" if spread_change > 0 else "flattening"
    if front_end_change * long_end_change < 0:
        return f"Twist {direction}"
    if average_change > 0:
        return f"Bear {direction}"
    if average_change < 0:
        return f"Bull {direction}"
    return f"Curve {direction}"


def analyze_curve(
    data: pd.DataFrame,
    current_as_of: pd.Timestamp,
    comparison_as_of: pd.Timestamp,
) -> CurveAnalysis | None:
    """Build one consistent view of curve levels, changes and spreads."""
    current = curve_snapshot(data, current_as_of)
    comparison = curve_snapshot(data, comparison_as_of)
    if current is None or comparison is None:
        return None

    changes = (current.values - comparison.values) * 100.0
    current_spreads = snapshot_spreads(current.values)
    comparison_spreads = snapshot_spreads(comparison.values)
    spread_changes = current_spreads - comparison_spreads

    return CurveAnalysis(
        current=current,
        comparison=comparison,
        changes_bp=changes,
        current_spreads_bp=current_spreads,
        comparison_spreads_bp=comparison_spreads,
        spread_changes_bp=spread_changes,
        regime=classify_curve_move(changes.get(DGS2), changes.get(DGS10)),
    )


def value_as_of(series: pd.Series, as_of: pd.Timestamp) -> float:
    """Return the latest valid numeric value on or before a date."""
    eligible = pd.to_numeric(series, errors="coerce").loc[
        lambda values: values.index <= as_of
    ].dropna()
    return float(eligible.iloc[-1]) if not eligible.empty else float("nan")


def percentile_rank(series: pd.Series, value: float) -> float:
    """Calculate the historical percentile rank of a value."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or pd.isna(value):
        return float("nan")
    return float((clean <= value).mean() * 100.0)
