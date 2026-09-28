from __future__ import annotations

import html
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import streamlit as st

from config import TENOR_YEAR_MAP, YIELD_SERIES


LAMBDA_FIXED = 0.6

COMPARISON_HORIZONS = {
    "1D": 1,
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
}

TENOR_LABELS = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS5": "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y",
}

DRIVER_GUIDE = pd.DataFrame(
    [
        {
            "Driver": "Level",
            "Curve effect": "Broad, near-parallel shift",
            "Typical macro interpretation": (
                "Slower-moving inflation expectations, the expected long-run policy rate, "
                "structural growth and term-premium repricing."
            ),
        },
        {
            "Driver": "Slope",
            "Curve effect": "Front end changes relative to the long end",
            "Typical macro interpretation": (
                "Monetary-policy-cycle and near-term activity repricing; often the clearest "
                "channel for changes in the expected central-bank path."
            ),
        },
        {
            "Driver": "Curvature",
            "Curve effect": "The belly changes relative to both wings",
            "Typical macro interpretation": (
                "Intermediate-horizon policy or business-cycle uncertainty, but also possible "
                "term-premium, issuance, liquidity or sector-specific effects."
            ),
        },
        {
            "Driver": "Mixed",
            "Curve effect": "No single factor clearly dominates",
            "Typical macro interpretation": (
                "Several macro or market channels are moving together; avoid assigning one cause."
            ),
        },
    ]
)


def _render_driver_guide() -> None:
    rows = []
    for row in DRIVER_GUIDE.itertuples(index=False):
        driver = str(row.Driver)
        rows.append(
            f"""
            <tr>
                <td data-label="Driver">
                    <span class="ns-guide-badge {driver.lower()}">{html.escape(driver)}</span>
                </td>
                <td data-label="Curve effect">{html.escape(str(row[1]))}</td>
                <td data-label="Typical macro interpretation">{html.escape(str(row[2]))}</td>
            </tr>
            """
        )

    st.html(
        f"""
        <style>
            .ns-guide-wrap {{
                overflow: hidden; border: 1px solid #dfe5ee; border-radius: .75rem;
                background: #fff;
            }}
            .ns-guide-table {{
                width: 100%; border-collapse: collapse; table-layout: fixed;
                color: #334155; font-size: .79rem; line-height: 1.45;
            }}
            .ns-guide-table th {{
                padding: .62rem .75rem; border-bottom: 1px solid #dfe5ee;
                color: #64748b; background: #f6f8fb; font-size: .68rem;
                font-weight: 750; letter-spacing: .055em; text-align: left;
                text-transform: uppercase;
            }}
            .ns-guide-table th:nth-child(1) {{ width: 14%; }}
            .ns-guide-table th:nth-child(2) {{ width: 27%; }}
            .ns-guide-table th:nth-child(3) {{ width: 59%; }}
            .ns-guide-table td {{
                padding: .72rem .75rem; border-bottom: 1px solid #edf0f5;
                vertical-align: top;
            }}
            .ns-guide-table tr:last-child td {{ border-bottom: 0; }}
            .ns-guide-table tbody tr:hover {{ background: #fafbfe; }}
            .ns-guide-badge {{
                display: inline-flex; padding: .2rem .48rem; border-radius: 999px;
                font-size: .67rem; font-weight: 750; letter-spacing: .035em;
                text-transform: uppercase;
            }}
            .ns-guide-badge.level {{ color: #1d4ed8; background: #eaf0ff; }}
            .ns-guide-badge.slope {{ color: #0f766e; background: #e6f5f2; }}
            .ns-guide-badge.curvature {{ color: #b45309; background: #fff3df; }}
            .ns-guide-badge.mixed {{ color: #6d28d9; background: #f1eafe; }}
            @media (max-width: 720px) {{
                .ns-guide-table, .ns-guide-table tbody, .ns-guide-table tr,
                .ns-guide-table td {{ display: block; width: 100%; }}
                .ns-guide-table thead {{ display: none; }}
                .ns-guide-table tr {{ padding: .7rem .75rem; border-bottom: 1px solid #dfe5ee; }}
                .ns-guide-table tr:last-child {{ border-bottom: 0; }}
                .ns-guide-table td {{ padding: .22rem 0; border: 0; }}
                .ns-guide-table td:not(:first-child)::before {{
                    display: block; margin-top: .2rem; color: #64748b;
                    content: attr(data-label); font-size: .63rem; font-weight: 750;
                    letter-spacing: .045em; text-transform: uppercase;
                }}
            }}
        </style>
        <div class="ns-guide-wrap">
            <table class="ns-guide-table">
                <thead>
                    <tr>
                        <th>Driver</th>
                        <th>Curve effect</th>
                        <th>Typical macro interpretation</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """
    )

def _ns_curve(
    tau: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
) -> np.ndarray:
    """Evaluate the Nelson-Siegel yield curve."""
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        x = tau / LAMBDA_FIXED

        term1 = (
            1.0 - np.exp(-x)
        ) / x

        term1 = np.where(
            np.isfinite(term1),
            term1,
            1.0,
        )

        term2 = (
            term1 - np.exp(-x)
        )

    return (
        beta0
        + beta1 * term1
        + beta2 * term2
    )


def _fit_row(
    row: pd.Series,
) -> tuple[float, float, float] | None:
    """Fit Nelson-Siegel factors to one yield-curve observation."""
    clean_row = row.dropna()

    if len(clean_row) < 4:
        return None

    tau = np.array(
        [
            TENOR_YEAR_MAP[series_id]
            for series_id in clean_row.index
        ],
        dtype=float,
    )

    yields = clean_row.values.astype(float)

    try:
        params, _ = curve_fit(
            _ns_curve,
            tau,
            yields,
            p0=[3.0, -1.0, 1.0],
            maxfev=5000,
        )
    except Exception:  # noqa: BLE001
        return None

    return (
        float(params[0]),
        float(params[1]),
        float(params[2]),
    )


def _percentile(
    series: pd.Series,
    value: float,
    years: int | None = None,
) -> float:
    """
    Calculate a factor percentile using available observations.

    When years is provided, observations are restricted to that trailing
    window. The available history may still be shorter if the global date
    range does not include the full requested period.

    When years is None, every fitted observation available within the
    global date range is used.
    """
    clean = series.dropna()

    if clean.empty or pd.isna(value):
        return float("nan")

    if years is not None:
        cutoff = (
            clean.index.max()
            - pd.DateOffset(years=years)
        )

        clean = clean.loc[
            clean.index >= cutoff
        ]

    if clean.empty:
        return float("nan")

    return float(
        (clean <= value).mean() * 100.0
    )


def _format_date_range(
    start_date: date | pd.Timestamp,
    end_date: date | pd.Timestamp,
) -> str:
    """Format a date range for explanatory text."""
    start_timestamp = pd.Timestamp(
        start_date
    )

    end_timestamp = pd.Timestamp(
        end_date
    )

    return (
        f"{start_timestamp:%d %b %Y} to "
        f"{end_timestamp:%d %b %Y}"
    )


def _change_over_window(
    series: pd.Series,
    days: int,
) -> float:
    """
    Calculate the change in a factor over a trailing calendar window.

    The comparison uses the latest observation on or before the target
    date. NaN is returned when insufficient history is available.
    """
    clean = series.dropna()

    if clean.empty:
        return float("nan")

    latest_date = clean.index.max()
    latest_value = float(
        clean.loc[latest_date]
    )

    target_date = (
        latest_date
        - pd.Timedelta(days=days)
    )

    history_before_target = clean.loc[
        clean.index <= target_date
    ]

    if history_before_target.empty:
        return float("nan")

    reference_value = float(
        history_before_target.iloc[-1]
    )

    return (
        latest_value
        - reference_value
    )


def _historical_window_change_history(
    series: pd.Series,
    days: int,
) -> pd.Series:
    """
    Calculate a history of window changes for a factor series.

    The change at each date uses the latest observation on or before the
    target date minus the latest observation on or before the comparison
    date. This keeps standardisation aligned with the selected horizon.
    """
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return pd.Series(
            dtype="float64",
        )

    changes = []

    for current_date, current_value in clean.items():
        comparison_cutoff = (
            current_date
            - pd.Timedelta(days=days)
        )

        reference_history = clean.loc[
            clean.index <= comparison_cutoff
        ]

        if reference_history.empty:
            changes.append(np.nan)
            continue

        changes.append(
            float(
                current_value
                - reference_history.iloc[-1]
            )
        )

    return pd.Series(
        changes,
        index=clean.index,
        dtype="float64",
    )


def _window_change_volatility(
    series: pd.Series,
    days: int,
) -> float:
    """Calculate the standard deviation of historical window changes."""
    historical_changes = _historical_window_change_history(
        series,
        days,
    ).dropna()

    if historical_changes.empty:
        return float("nan")

    return float(
        historical_changes.std()
    )


def _factor_move_phrase(
    factor: str,
    change: float,
) -> str:
    """Describe the direction and sign of a factor move."""
    if factor == "unavailable":
        return (
            "The factor move is unavailable."
        )

    if pd.isna(change):
        return (
            f"The {factor} factor change is unavailable."
        )

    if factor == "level":
        if abs(change) < 0.005:
            return (
                "The level factor was broadly unchanged."
            )

        if change > 0:
            return (
                f"The level factor increased by {change:.3f}, "
                "lifting the fitted curve overall."
            )

        return (
            f"The level factor decreased by {abs(change):.3f}, "
            "lowering the fitted curve overall."
        )

    if factor == "slope":
        if abs(change) < 0.005:
            return (
                "The slope factor was broadly unchanged."
            )

        if change > 0:
            return (
                f"The slope factor increased by {change:.3f}, "
                "indicating a flatter fitted curve."
            )

        return (
            f"The slope factor decreased by {abs(change):.3f}, "
            "indicating a steeper fitted curve."
        )

    if abs(change) < 0.005:
        return (
            "The curvature factor was broadly unchanged."
        )

    if change > 0:
        return (
            f"The curvature factor increased by {change:.3f}, "
            "pointing to a more elevated belly."
        )

    return (
        f"The curvature factor decreased by {abs(change):.3f}, "
        "pointing to a more depressed belly."
    )


def _classify_primary_factor_move(
    level_change: float,
    slope_change: float,
    curvature_change: float,
    level_volatility: float,
    slope_volatility: float,
    curvature_volatility: float,
) -> dict[str, float | str]:
    """
    Classify the primary Nelson-Siegel move using standardized changes.

    The panel compares factor changes after scaling by each factor's
    historical change volatility over the same horizon. This keeps the
    classification transparent even though the raw factor scales differ.
    """
    standardized_changes = {
        "level": (
            abs(level_change) / level_volatility
            if (
                pd.notna(level_change)
                and pd.notna(level_volatility)
                and level_volatility > 0
            )
            else float("nan")
        ),
        "slope": (
            abs(slope_change) / slope_volatility
            if (
                pd.notna(slope_change)
                and pd.notna(slope_volatility)
                and slope_volatility > 0
            )
            else float("nan")
        ),
        "curvature": (
            abs(curvature_change) / curvature_volatility
            if (
                pd.notna(curvature_change)
                and pd.notna(curvature_volatility)
                and curvature_volatility > 0
            )
            else float("nan")
        ),
    }

    ranked_changes = sorted(
        standardized_changes.items(),
        key=lambda item: (
            item[1]
            if pd.notna(item[1])
            else -np.inf
        ),
        reverse=True,
    )

    if (
        not ranked_changes
        or pd.isna(ranked_changes[0][1])
        or ranked_changes[0][1] <= 0
    ):
        return {
            "classification": "Unavailable",
            "primary_factor": "unavailable",
            "secondary_factor": "unavailable",
            "primary_change": float("nan"),
            "secondary_change": float("nan"),
            "primary_score": float("nan"),
            "secondary_score": float("nan"),
        }

    primary_factor, primary_score = ranked_changes[0]

    if len(ranked_changes) > 1:
        secondary_factor, secondary_score = ranked_changes[1]
    else:
        secondary_factor = "unavailable"
        secondary_score = float("nan")

    if (
        pd.isna(secondary_score)
        or secondary_score <= 0
        or primary_score < 1.0
        or primary_score / secondary_score < 1.2
    ):
        classification = "Mixed factor move"
    else:
        classification = f"{primary_factor.title()}-driven move"

    raw_changes = {
        "level": level_change,
        "slope": slope_change,
        "curvature": curvature_change,
    }

    return {
        "classification": classification,
        "primary_factor": primary_factor,
        "secondary_factor": secondary_factor,
        "primary_change": raw_changes[
            primary_factor
        ],
        "secondary_change": raw_changes.get(
            secondary_factor,
            float("nan"),
        ),
        "primary_score": primary_score,
        "secondary_score": secondary_score,
    }


def _factor_figure(
    series: pd.Series,
    title: str,
    y_axis_title: str,
) -> go.Figure:
    """Create a single-factor history chart."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name=title,
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.45,
    )

    clean = series.dropna()

    if not clean.empty:
        current_value = float(
            clean.iloc[-1]
        )

        figure.add_hline(
            y=current_value,
            line_dash="dot",
            opacity=0.45,
            annotation_text=(
                f"Current: {current_value:.2f}"
            ),
            annotation_position="top right",
        )

    figure.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        height=300,
        margin={
            "l": 40,
            "r": 20,
            "t": 55,
            "b": 40,
        },
    )

    return figure


def _factor_contribution_figure(
    tau: np.ndarray,
    tenor_labels: list[str],
    level_change: float,
    slope_change: float,
    curvature_change: float,
    horizon: str,
) -> go.Figure:
    """Show how factor changes combine into the fitted move at each tenor."""
    x = tau / LAMBDA_FIXED
    slope_loading = np.divide(
        1.0 - np.exp(-x),
        x,
        out=np.ones_like(x),
        where=x != 0,
    )
    curvature_loading = slope_loading - np.exp(-x)
    contributions = {
        "Level": np.full_like(tau, level_change * 100.0),
        "Slope": slope_change * slope_loading * 100.0,
        "Curvature": curvature_change * curvature_loading * 100.0,
    }
    colors = {"Level": "#3157D5", "Slope": "#0F8A83", "Curvature": "#D97706"}
    figure = go.Figure()
    for factor, values in contributions.items():
        figure.add_trace(
            go.Bar(
                x=tenor_labels,
                y=values,
                name=factor,
                marker_color=colors[factor],
                hovertemplate=f"{factor}: %{{y:+.1f}} bp<extra></extra>",
            )
        )
    figure.add_hline(y=0, line_color="#94A3B8", line_width=1)
    figure.update_layout(
        title=f"Factor contribution to the {horizon} fitted move",
        xaxis_title="Tenor",
        yaxis_title="Contribution (bp)",
        template="plotly_white",
        barmode="relative",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        height=420,
    )
    return figure


def _fit_rmse_bp(
    observed: np.ndarray,
    fitted: np.ndarray,
) -> float:
    """Calculate fitted-curve RMSE in basis points."""
    if (
        observed.size == 0
        or fitted.size == 0
        or observed.size != fitted.size
    ):
        return float("nan")

    return float(
        np.sqrt(
            np.mean(
                (observed - fitted) ** 2
            )
        )
        * 100.0
    )


def _fit_quality_caption(
    rmse_bp: float,
) -> str:
    """Describe fitted-curve quality using RMSE."""
    if pd.isna(rmse_bp):
        return (
            "Fit quality could not be calculated."
        )

    if rmse_bp < 5:
        return (
            "The fitted curve closely tracks observed "
            "Treasury yields."
        )

    if rmse_bp < 10:
        return (
            "The fitted curve provides a reasonable "
            "summary of the observed curve."
        )

    if rmse_bp < 20:
        return (
            "The fit is approximate, with visible "
            "tenor-level deviations."
        )

    return (
        "The standard Nelson-Siegel specification fits "
        "the current curve poorly."
    )


def render(
    _fred_client,
    context: dict,
) -> None:
    st.subheader("Nelson-Siegel Decomposition")

    global_start_date = context[
        "start_date"
    ]

    global_end_date = context[
        "end_date"
    ]

    global_range_text = _format_date_range(
        global_start_date,
        global_end_date,
    )

    result = context.get(
        "yield_result"
    )

    if (
        result is None
        or not result.success
        or result.data is None
    ):
        st.warning(
            result.message
            if result
            else "Yield data unavailable."
        )
        return

    df = (
        result.data
        .copy()
        .sort_index()
    )

    if df.empty:
        st.info(
            "No data available for the selected date range."
        )
        return

    missing_series = [
        series_id
        for series_id in YIELD_SERIES
        if series_id not in df.columns
    ]

    if missing_series:
        st.warning(
            "The following Treasury series are missing: "
            + ", ".join(missing_series)
        )
        return

    factors = []

    for observation_date, row in df[
        YIELD_SERIES
    ].iterrows():
        fit = _fit_row(
            row
        )

        if fit is None:
            continue

        factors.append(
            (
                observation_date,
                fit[0],
                fit[1],
                fit[2],
            )
        )

    if not factors:
        st.warning(
            "Unable to fit Nelson-Siegel factors "
            "for the current selection."
        )
        return

    factors_df = (
        pd.DataFrame(
            factors,
            columns=[
                "date",
                "level",
                "slope",
                "curvature",
            ],
        )
        .set_index("date")
        .sort_index()
    )

    fitted_range_start = (
        factors_df.index.min()
    )

    fitted_range_end = (
        factors_df.index.max()
    )

    latest_date = fitted_range_end

    latest = factors_df.loc[
        latest_date
    ]

    header_left, header_right = st.columns([2, 3])
    with header_left:
        st.caption(f"As of {latest_date:%d %b %Y} · fitted to Treasury constant maturities")
    with header_right:
        comparison_horizon = st.segmented_control(
            "Comparison horizon",
            options=list(COMPARISON_HORIZONS),
            default="1M",
            required=True,
            key="nelson_siegel_comparison_horizon",
            label_visibility="collapsed",
            width="stretch",
        )
    comparison_days = COMPARISON_HORIZONS[comparison_horizon]

    # ---------------------------------------------------------
    # FACTOR PERCENTILES
    # ---------------------------------------------------------

    level_percentile_selected = _percentile(
        factors_df["level"],
        float(latest["level"]),
    )

    slope_percentile_selected = _percentile(
        factors_df["slope"],
        float(latest["slope"]),
    )

    curvature_percentile_selected = _percentile(
        factors_df["curvature"],
        float(latest["curvature"]),
    )

    level_percentile_2y = _percentile(
        factors_df["level"],
        float(latest["level"]),
        years=2,
    )

    slope_percentile_2y = _percentile(
        factors_df["slope"],
        float(latest["slope"]),
        years=2,
    )

    curvature_percentile_2y = _percentile(
        factors_df["curvature"],
        float(latest["curvature"]),
        years=2,
    )

    level_percentile_5y = _percentile(
        factors_df["level"],
        float(latest["level"]),
        years=5,
    )

    slope_percentile_5y = _percentile(
        factors_df["slope"],
        float(latest["slope"]),
        years=5,
    )

    curvature_percentile_5y = _percentile(
        factors_df["curvature"],
        float(latest["curvature"]),
        years=5,
    )

    # ---------------------------------------------------------
    # 1-WEEK AND 1-MONTH FACTOR CHANGES
    # ---------------------------------------------------------

    level_change_1w = _change_over_window(
        factors_df["level"],
        7,
    )

    level_change_1m = _change_over_window(
        factors_df["level"],
        30,
    )

    slope_change_1w = _change_over_window(
        factors_df["slope"],
        7,
    )

    slope_change_1m = _change_over_window(
        factors_df["slope"],
        30,
    )

    curvature_change_1w = _change_over_window(
        factors_df["curvature"],
        7,
    )

    curvature_change_1m = _change_over_window(
        factors_df["curvature"],
        30,
    )

    level_change = _change_over_window(
        factors_df["level"],
        comparison_days,
    )

    slope_change = _change_over_window(
        factors_df["slope"],
        comparison_days,
    )

    curvature_change = _change_over_window(
        factors_df["curvature"],
        comparison_days,
    )

    level_change_volatility = _window_change_volatility(
        factors_df["level"],
        comparison_days,
    )

    slope_change_volatility = _window_change_volatility(
        factors_df["slope"],
        comparison_days,
    )

    curvature_change_volatility = _window_change_volatility(
        factors_df["curvature"],
        comparison_days,
    )

    factor_move = _classify_primary_factor_move(
        level_change,
        slope_change,
        curvature_change,
        level_change_volatility,
        slope_change_volatility,
        curvature_change_volatility,
    )

    # ---------------------------------------------------------
    # LATEST OBSERVED AND FITTED CURVES
    # ---------------------------------------------------------

    observed = df.loc[
        latest_date,
        YIELD_SERIES,
    ].dropna()

    if observed.empty:
        st.warning(
            "The latest observed curve is unavailable."
        )
        return

    tau = np.array(
        [
            TENOR_YEAR_MAP[series_id]
            for series_id in observed.index
        ],
        dtype=float,
    )

    observed_values = (
        observed.values.astype(float)
    )

    fitted = _ns_curve(
        tau,
        float(latest["level"]),
        float(latest["slope"]),
        float(latest["curvature"]),
    )

    residuals_bp = (
        observed_values - fitted
    ) * 100.0

    rmse_bp = _fit_rmse_bp(
        observed_values,
        fitted,
    )

    # ---------------------------------------------------------
    # DETAIL TABLE DATA
    # ---------------------------------------------------------

    stats_df = pd.DataFrame(
        [
            {
                "Factor": "Level",
                "Current": round(
                    float(latest["level"]),
                    3,
                ),
                "1W chg": (
                    round(
                        level_change_1w,
                        3,
                    )
                    if pd.notna(
                        level_change_1w
                    )
                    else None
                ),
                "1M chg": (
                    round(
                        level_change_1m,
                        3,
                    )
                    if pd.notna(
                        level_change_1m
                    )
                    else None
                ),
                f"Pctile: {global_range_text}": round(
                    level_percentile_selected,
                    1,
                ),
                "Pctile: trailing 2Y": round(
                    level_percentile_2y,
                    1,
                ),
                "Pctile: trailing 5Y": round(
                    level_percentile_5y,
                    1,
                ),
            },
            {
                "Factor": "Slope",
                "Current": round(
                    float(latest["slope"]),
                    3,
                ),
                "1W chg": (
                    round(
                        slope_change_1w,
                        3,
                    )
                    if pd.notna(
                        slope_change_1w
                    )
                    else None
                ),
                "1M chg": (
                    round(
                        slope_change_1m,
                        3,
                    )
                    if pd.notna(
                        slope_change_1m
                    )
                    else None
                ),
                f"Pctile: {global_range_text}": round(
                    slope_percentile_selected,
                    1,
                ),
                "Pctile: trailing 2Y": round(
                    slope_percentile_2y,
                    1,
                ),
                "Pctile: trailing 5Y": round(
                    slope_percentile_5y,
                    1,
                ),
            },
            {
                "Factor": "Curvature",
                "Current": round(
                    float(latest["curvature"]),
                    3,
                ),
                "1W chg": (
                    round(
                        curvature_change_1w,
                        3,
                    )
                    if pd.notna(
                        curvature_change_1w
                    )
                    else None
                ),
                "1M chg": (
                    round(
                        curvature_change_1m,
                        3,
                    )
                    if pd.notna(
                        curvature_change_1m
                    )
                    else None
                ),
                f"Pctile: {global_range_text}": round(
                    curvature_percentile_selected,
                    1,
                ),
                "Pctile: trailing 2Y": round(
                    curvature_percentile_2y,
                    1,
                ),
                "Pctile: trailing 5Y": round(
                    curvature_percentile_5y,
                    1,
                ),
            },
        ]
    )

    residuals_df = pd.DataFrame(
        {
            "Series": observed.index,
            "Tenor": [
                TENOR_YEAR_MAP[
                    series_id
                ]
                for series_id in observed.index
            ],
            "Observed yield (%)": np.round(
                observed_values,
                3,
            ),
            "Fitted yield (%)": np.round(
                fitted,
                3,
            ),
            "Residual (bp)": np.round(
                residuals_bp,
                1,
            ),
        }
    )

    primary_factor = factor_move[
        "primary_factor"
    ]

    secondary_factor = factor_move[
        "secondary_factor"
    ]

    primary_factor_label = (
        primary_factor.title()
        if primary_factor != "unavailable"
        else "Unavailable"
    )

    classification_label = factor_move[
        "classification"
    ]

    if classification_label == "Unavailable":
        decomposition_summary = (
            f"The standardized {comparison_horizon} factor move could not be "
            "classified because the required volatility history is "
            "unavailable."
        )
    elif classification_label == "Mixed factor move":
        decomposition_summary = (
            "The latest move was mixed across level, slope and "
            f"curvature after standardizing {comparison_horizon} changes."
        )
    else:
        decomposition_summary = (
            f"The latest move was primarily {primary_factor}-driven "
            f"after standardizing {comparison_horizon} changes."
        )

    decomposition_details = [
        _factor_move_phrase(
            primary_factor,
            float(factor_move["primary_change"]),
        )
    ]

    if (
        secondary_factor != "unavailable"
        and secondary_factor != primary_factor
    ):
        decomposition_details.append(
            _factor_move_phrase(
                secondary_factor,
                float(factor_move["secondary_change"]),
            )
        )

    if pd.notna(rmse_bp) and rmse_bp >= 10:
        decomposition_details.append(
            f"Fit RMSE is {rmse_bp:.1f} bp, so the factor "
            "interpretation should be treated as approximate."
        )

    st.html(
        """
        <style>
            .ns-driver-heading {
                display: inline-flex !important; align-items: center; flex-wrap: nowrap;
                width: max-content; max-width: 100%; gap: .5rem; white-space: nowrap;
                margin: 1.85rem 0 .7rem;
            }
            .ns-driver-title {
                color: #14213d; font-size: 1.18rem; font-weight: 650;
                letter-spacing: -.025em; line-height: 1.25;
            }
            .ns-driver-info {
                position: relative; display: inline-flex; align-items: center;
                justify-content: center; width: 1.15rem; height: 1.15rem;
                border: 1px solid #94a3b8; border-radius: 50%; color: #64748b;
                font-size: .72rem; font-weight: 750; cursor: help;
            }
            .ns-driver-tooltip {
                position: absolute; z-index: 20; top: 1.55rem; left: 50%;
                display: block; width: min(30rem, 82vw); padding: .8rem .9rem;
                border: 1px solid #dfe5ee; border-radius: .65rem;
                background: #fff; color: #334155;
                box-shadow: 0 10px 30px rgba(30,47,78,.16);
                font-size: .76rem; font-weight: 400; line-height: 1.45;
                box-sizing: border-box; white-space: normal; overflow-wrap: anywhere;
                opacity: 0; visibility: hidden; transform: translate(-15%, -.25rem);
                transition: opacity .12s ease, transform .12s ease;
            }
            .ns-driver-tooltip strong { color: #14213d; }
            .ns-driver-tooltip div + div { margin-top: .38rem; }
            .ns-driver-info:hover .ns-driver-tooltip,
            .ns-driver-info:focus .ns-driver-tooltip {
                opacity: 1; visibility: visible; transform: translate(-15%, 0);
            }
        </style>
        <div class="ns-driver-heading">
            <span class="ns-driver-title">Dominant Curve Driver</span>
            <span class="ns-driver-info" tabindex="0" aria-label="Explain curve drivers">
                i
                <span class="ns-driver-tooltip" role="tooltip">
                    <div><strong>Level:</strong> yields moved broadly together across maturities.</div>
                    <div><strong>Slope:</strong> the front and long ends moved differently, steepening or flattening the curve.</div>
                    <div><strong>Curvature:</strong> the belly moved differently from the short and long ends.</div>
                    <div><strong>Mixed:</strong> no single factor clearly dominated the move.</div>
                </span>
            </span>
        </div>
        """
    )
    driver_label = (
        primary_factor_label
        if classification_label not in {"Mixed factor move", "Unavailable"}
        else "Mixed" if classification_label == "Mixed factor move"
        else "Unavailable"
    )
    regime_text = (
        f"**{driver_label}.** {decomposition_summary} "
        + " ".join(decomposition_details)
    )
    st.info(regime_text)

    with st.expander("Driver Interpretation", expanded=False):
        _render_driver_guide()

    # ---------------------------------------------------------
    # CURRENT FACTOR SUMMARY
    # ---------------------------------------------------------

    st.markdown("### Factor snapshot")
    st.caption(f"Current readings, {comparison_horizon} changes and historical position.")

    (
        level_column,
        slope_column,
        curvature_column,
    ) = st.columns(3)

    with level_column:
        st.metric(
            "Level · long-run anchor",
            f"{latest['level']:.2f}%",
            delta=(
                f"{level_change * 100:+.1f} bp vs {comparison_horizon}"
                if pd.notna(level_change)
                else None
            ),
            delta_color="off",
        )

    with slope_column:
        st.metric(
            "Slope · front end vs long end",
            f"{latest['slope']:.2f}%",
            delta=(
                f"{slope_change * 100:+.1f} bp vs {comparison_horizon}"
                if pd.notna(slope_change)
                else None
            ),
            delta_color="off",
        )

    with curvature_column:
        st.metric(
            "Curvature · belly vs wings",
            f"{latest['curvature']:.2f}%",
            delta=(
                f"{curvature_change * 100:+.1f} bp vs {comparison_horizon}"
                if pd.notna(curvature_change)
                else None
            ),
            delta_color="off",
        )

    # ---------------------------------------------------------
    # OBSERVED VERSUS FITTED CURVE
    # ---------------------------------------------------------

    st.markdown("### What drove the curve move")

    fig_fit = go.Figure()

    fig_fit.add_trace(
        go.Scatter(
            x=tau,
            y=observed_values,
            mode="markers+lines",
            name="Observed",
        )
    )

    fig_fit.add_trace(
        go.Scatter(
            x=tau,
            y=fitted,
            mode="lines",
            name="Fitted (NS)",
        )
    )

    fig_fit.update_layout(
        title="Model trust check: observed vs fitted",
        xaxis_title="Tenor (years)",
        yaxis_title="Yield (%)",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Curve",
        height=430,
    )

    contribution_figure = _factor_contribution_figure(
        tau,
        [TENOR_LABELS.get(series_id, series_id) for series_id in observed.index],
        level_change,
        slope_change,
        curvature_change,
        comparison_horizon,
    )

    contribution_column, fit_column = st.columns(2)
    with contribution_column:
        st.plotly_chart(contribution_figure, width="stretch")
        st.caption("Stacked contributions add up to the fitted yield change at each tenor.")
    with fit_column:
        st.plotly_chart(fig_fit, width="stretch")
        st.caption(
            "Purpose: determine whether the model's Level, Slope and Curvature output "
            "is reliable enough to interpret."
        )
        can_trust = pd.notna(rmse_bp) and rmse_bp < 10
        trust_class = "selected trust" if can_trust else ""
        caution_class = "selected caution" if not can_trust else ""
        st.html(
            f"""
            <style>
                .ns-trust-indicator {{ display: flex; gap: .45rem; margin: .15rem 0 .45rem; }}
                .ns-trust-option {{
                    display: inline-flex; align-items: center; gap: .38rem;
                    padding: .36rem .62rem; border: 1px solid #d7dde7;
                    border-radius: 999px; color: #94a3b8; background: #f8fafc;
                    font-size: .72rem; font-weight: 700;
                }}
                .ns-trust-dot {{ width: .42rem; height: .42rem; border-radius: 50%; background: #cbd5e1; }}
                .ns-trust-option.selected.trust {{ color: #0f766e; border-color: #9bd5cc; background: #e9f7f4; }}
                .ns-trust-option.selected.trust .ns-trust-dot {{ background: #0f8a83; }}
                .ns-trust-option.selected.caution {{ color: #b42318; border-color: #f0b5af; background: #fff0ee; }}
                .ns-trust-option.selected.caution .ns-trust-dot {{ background: #dc5a5a; }}
            </style>
            <div class="ns-trust-indicator" aria-label="Model trust assessment">
                <span class="ns-trust-option {trust_class}"><span class="ns-trust-dot"></span>Can trust</span>
                <span class="ns-trust-option {caution_class}"><span class="ns-trust-dot"></span>Cannot trust</span>
            </div>
            """
        )

    # ---------------------------------------------------------
    # FIT RESIDUALS
    # ---------------------------------------------------------

    residual_figure = go.Figure()

    residual_figure.add_trace(
        go.Bar(
            x=[TENOR_LABELS.get(series_id, series_id) for series_id in residuals_df["Series"]],
            y=residuals_df[
                "Residual (bp)"
            ],
            name="Observed minus fitted",
        )
    )

    residual_figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.6,
    )

    residual_figure.update_layout(
        title="Observed Yield Minus Fitted Yield",
        xaxis_title="Tenor",
        yaxis_title="Residual (basis points)",
        template="plotly_white",
        showlegend=False,
        height=350,
    )

    st.markdown("### History and diagnostics")
    history_tab, residual_tab = st.tabs(["Factor history", "Fit residuals"])
    with history_tab:
        selected_factor = st.segmented_control(
            "Factor",
            options=["Level", "Slope", "Curvature"],
            default="Level",
            required=True,
            key="nelson_siegel_history_factor",
            label_visibility="collapsed",
        )
        factor_key = selected_factor.lower()
        st.plotly_chart(
            _factor_figure(factors_df[factor_key], selected_factor, "Factor value (%)"),
            width="stretch",
        )

    with residual_tab:
        st.plotly_chart(residual_figure, width="stretch")
        st.caption(
            "Positive means the observed yield is above the fitted curve; negative means below. "
            "Residuals are model diagnostics, not standalone trade signals."
        )

    # ---------------------------------------------------------
    # DETAILED DATA
    # ---------------------------------------------------------

    with st.expander(
        "View detailed factor data",
        expanded=False,
    ):
        st.markdown(
            "#### Factor values, changes and percentiles"
        )

        st.caption(
            f"The main percentile ranks each latest factor reading "
            f"against fitted observations from {global_range_text}. "
            "The trailing 2Y and 5Y rankings use up to two and five "
            "years of observations respectively, but cannot extend "
            "beyond the global date range."
        )

        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### Tenor-level fitted values and residuals"
        )

        st.dataframe(
            residuals_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "#### Model specification"
        )

        st.write(
            {
                "Model": "Nelson-Siegel",
                "Decay parameter": LAMBDA_FIXED,
                "Global range start": global_start_date,
                "Global range end": global_end_date,
                "First fitted observation": (
                    fitted_range_start.date()
                ),
                "Latest fitted observation": (
                    fitted_range_end.date()
                ),
                "Number of fitted maturities": len(observed),
                "RMSE (bp)": (
                    round(
                        rmse_bp,
                        2,
                    )
                    if pd.notna(rmse_bp)
                    else None
                ),
            }
        )
