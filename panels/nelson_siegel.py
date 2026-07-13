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

METRIC_CAPTION_MIN_HEIGHT_PX = 112
METRIC_DELTA_SPACER_HEIGHT_PX = 30
SUMMARY_FOOTNOTE_SPACER_REM = 1.25


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


def _percentile_description(
    percentile: float,
    factor_label: str,
    start_date: date | pd.Timestamp,
    end_date: date | pd.Timestamp,
) -> str:
    """Describe a factor's percentile across the global date range."""
    date_range_text = _format_date_range(
        start_date,
        end_date,
    )

    if pd.isna(percentile):
        return (
            f"The {factor_label.lower()} percentile could not be "
            f"calculated for {date_range_text}."
        )

    if percentile >= 90:
        position = "near the top of its distribution"
    elif percentile >= 75:
        position = "in the upper quartile"
    elif percentile <= 10:
        position = "near the bottom of its distribution"
    elif percentile <= 25:
        position = "in the lower quartile"
    else:
        position = "near the middle of its distribution"

    return (
        f"Among fitted {factor_label.lower()} observations from "
        f"{date_range_text}, the current reading is {position} "
        f"({percentile:.0f}th percentile)."
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
    date. This keeps the standardisation aligned with the panel's own
    one-month change logic.
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
    """Describe the direction and sign of a factor's one-month move."""
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
    level_change_1m: float,
    slope_change_1m: float,
    curvature_change_1m: float,
    level_volatility: float,
    slope_volatility: float,
    curvature_volatility: float,
) -> dict[str, float | str]:
    """
    Classify the primary Nelson-Siegel move using standardized changes.

    The panel compares one-month factor changes after scaling by each
    factor's historical one-month change volatility. This keeps the
    classification transparent even though the raw factor scales differ.
    """
    standardized_changes = {
        "level": (
            abs(level_change_1m) / level_volatility
            if (
                pd.notna(level_change_1m)
                and pd.notna(level_volatility)
                and level_volatility > 0
            )
            else float("nan")
        ),
        "slope": (
            abs(slope_change_1m) / slope_volatility
            if (
                pd.notna(slope_change_1m)
                and pd.notna(slope_volatility)
                and slope_volatility > 0
            )
            else float("nan")
        ),
        "curvature": (
            abs(curvature_change_1m) / curvature_volatility
            if (
                pd.notna(curvature_change_1m)
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
        "level": level_change_1m,
        "slope": slope_change_1m,
        "curvature": curvature_change_1m,
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


def _factor_caption(
    factor: str,
    _value: float,
    percentile: float,
    start_date: date | pd.Timestamp,
    end_date: date | pd.Timestamp,
) -> tuple[str, str]:
    """
    Return the interpretation and percentile context for a factor.

    Sign conventions:

    short-end yield ≈ beta0 + beta1
    long-end yield  ≈ beta0

    Therefore:

    long minus short = -beta1

    A negative beta1 corresponds to an upward-sloping curve, while a
    positive beta1 corresponds to a flatter or inverted curve.

    Beta2 loads mainly on the belly. A positive beta2 elevates the belly
    relative to the wings, while a negative beta2 depresses it.
    """
    factor_labels = {
        "level": "Level",
        "slope": "Slope",
        "curvature": "Curvature",
    }

    factor_label = factor_labels[
        factor
    ]

    percentile_text = _percentile_description(
        percentile,
        factor_label,
        start_date,
        end_date,
    )

    if factor == "level":
        return (
            "Overall height of the fitted curve; mathematically, the "
            "long-run yield anchor.",
            percentile_text,
        )

    if factor == "slope":
        return (
            "Short-end position relative to the long end. More positive "
            "values imply a flatter or more inverted fitted curve.",
            percentile_text,
        )

    return (
        "Position of the fitted belly relative to the short and long "
        "ends. More positive values indicate a more elevated belly.",
        percentile_text,
    )


def _render_metric_caption(
    first_sentence: str,
    second_sentence: str | None = None,
) -> None:
    """Render an aligned metric caption with a line break."""
    first = html.escape(
        first_sentence
    )

    second = (
        html.escape(second_sentence)
        if second_sentence
        else ""
    )

    second_line = (
        f"""
        <div style="margin-top: 0.6rem;">
            {second}
        </div>
        """
        if second
        else ""
    )

    st.markdown(
        f"""
        <div style="
            min-height: {METRIC_CAPTION_MIN_HEIGHT_PX}px;
            margin-top: 0.65rem;
            color: rgba(49, 51, 63, 0.62);
            font-size: 0.875rem;
            line-height: 1.55;
        ">
            <div>{first}</div>
            {second_line}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_delta_spacer() -> None:
    """
    Reserve the space occupied by a Streamlit metric delta.

    Fit RMSE has no delta value, so the spacer aligns its caption with
    the captions below the three factor metrics.
    """
    st.markdown(
        f"""
        <div style="
            height: {METRIC_DELTA_SPACER_HEIGHT_PX}px;
        "></div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_footnote_spacer() -> None:
    """Add separation between metric captions and the summary footnote."""
    st.markdown(
        f"""
        <div style="
            height: {SUMMARY_FOOTNOTE_SPACER_REM}rem;
        "></div>
        """,
        unsafe_allow_html=True,
    )


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
    st.subheader(
        "Panel 2: Nelson-Siegel Decomposition"
    )

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

    level_change_volatility = _window_change_volatility(
        factors_df["level"],
        30,
    )

    slope_change_volatility = _window_change_volatility(
        factors_df["slope"],
        30,
    )

    curvature_change_volatility = _window_change_volatility(
        factors_df["curvature"],
        30,
    )

    factor_move = _classify_primary_factor_move(
        level_change_1m,
        slope_change_1m,
        curvature_change_1m,
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
            "The standardized one-month factor move could not be "
            "classified because the required volatility history is "
            "unavailable."
        )
    elif classification_label == "Mixed factor move":
        decomposition_summary = (
            "The latest move was mixed across level, slope and "
            "curvature after standardizing one-month changes."
        )
    else:
        decomposition_summary = (
            f"The latest move was primarily {primary_factor}-driven "
            "after standardizing one-month changes."
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

    # ---------------------------------------------------------
    # CURRENT CURVE DECOMPOSITION
    # ---------------------------------------------------------

    st.markdown(
        "### Current curve decomposition"
    )

    st.info(
        "\n".join(
            [
                f"Primary factor: {primary_factor_label}",
                f"Classification: {classification_label}",
                decomposition_summary,
                *decomposition_details,
            ]
        )
    )

    macro_note_parts = []

    if classification_label == "Unavailable":
        macro_note_parts.append(
            "The fitted Treasury curve move could not be classified "
            "because the required volatility history is unavailable."
        )
    elif classification_label == "Mixed factor move":
        macro_note_parts.append(
            "Over the past month, the fitted Treasury curve move was "
            "mixed across level, slope and curvature after standardizing "
            "one-month changes."
        )
    else:
        macro_note_parts.append(
            f"Over the past month, the fitted Treasury curve move was "
            f"primarily {primary_factor}-driven after standardizing "
            "one-month changes."
        )

    if classification_label != "Unavailable":
        macro_note_parts.extend(
            decomposition_details
        )

    if pd.notna(rmse_bp) and rmse_bp >= 10:
        macro_note_parts.append(
            f"Fit RMSE is {rmse_bp:.1f} bp, so the factor "
            "interpretation should be treated as approximate."
        )

    # ---------------------------------------------------------
    # MACRO-NOTE OUTPUT
    # ---------------------------------------------------------

    st.markdown(
        "### Macro-note output"
    )

    st.info(
        " ".join(
            macro_note_parts
        )
    )

    st.caption(
        "Nelson-Siegel factors describe the geometry of the fitted "
        "curve. They do not identify the economic cause of the move."
    )

    # ---------------------------------------------------------
    # CURRENT FACTOR SUMMARY
    # ---------------------------------------------------------

    st.markdown(
        "### Current factor summary"
    )

    st.caption(
        "Percentiles rank the latest Level, Slope and Curvature "
        f"readings against their fitted observations from "
        f"{global_range_text}."
    )

    (
        level_column,
        slope_column,
        curvature_column,
        fit_column,
    ) = st.columns(4)

    level_caption = _factor_caption(
        "level",
        float(latest["level"]),
        level_percentile_selected,
        global_start_date,
        global_end_date,
    )

    slope_caption = _factor_caption(
        "slope",
        float(latest["slope"]),
        slope_percentile_selected,
        global_start_date,
        global_end_date,
    )

    curvature_caption = _factor_caption(
        "curvature",
        float(latest["curvature"]),
        curvature_percentile_selected,
        global_start_date,
        global_end_date,
    )

    with level_column:
        st.metric(
            "Level",
            f"{latest['level']:.3f}",
            delta=(
                f"{level_change_1m:+.3f} (1M)"
                if pd.notna(
                    level_change_1m
                )
                else None
            ),
        )

        _render_metric_caption(
            level_caption[0],
            level_caption[1],
        )

    with slope_column:
        st.metric(
            "Slope",
            f"{latest['slope']:.3f}",
            delta=(
                f"{slope_change_1m:+.3f} (1M)"
                if pd.notna(
                    slope_change_1m
                )
                else None
            ),
            delta_color="inverse",
        )

        _render_metric_caption(
            slope_caption[0],
            slope_caption[1],
        )

    with curvature_column:
        st.metric(
            "Curvature",
            f"{latest['curvature']:.3f}",
            delta=(
                f"{curvature_change_1m:+.3f} (1M)"
                if pd.notna(
                    curvature_change_1m
                )
                else None
            ),
        )

        _render_metric_caption(
            curvature_caption[0],
            curvature_caption[1],
        )

    with fit_column:
        st.metric(
            "Fit RMSE",
            (
                f"{rmse_bp:.1f} bp"
                if pd.notna(rmse_bp)
                else "Unavailable"
            ),
        )

        _render_delta_spacer()

        _render_metric_caption(
            _fit_quality_caption(
                rmse_bp
            )
        )

    _render_summary_footnote_spacer()

    st.caption(
        "Factor values are estimated using a fixed "
        f"Nelson-Siegel decay parameter of {LAMBDA_FIXED:.1f}. "
        f"Latest fitted observation: {latest_date:%d %b %Y}. "
        "Slope delta color is inverted because a falling slope "
        "factor corresponds to a steepening curve."
    )

    # ---------------------------------------------------------
    # FACTOR HISTORY
    # ---------------------------------------------------------

    st.markdown(
        "### Factor history"
    )

    st.caption(
        "Each factor is shown separately because Level, Slope "
        "and Curvature have different economic interpretations "
        "and should not be compared by absolute magnitude."
    )

    (
        level_chart_column,
        slope_chart_column,
        curvature_chart_column,
    ) = st.columns(3)

    with level_chart_column:
        st.plotly_chart(
            _factor_figure(
                factors_df["level"],
                "Level",
                "Factor value",
            ),
            use_container_width=True,
        )

    with slope_chart_column:
        st.plotly_chart(
            _factor_figure(
                factors_df["slope"],
                "Slope",
                "Factor value",
            ),
            use_container_width=True,
        )

    with curvature_chart_column:
        st.plotly_chart(
            _factor_figure(
                factors_df["curvature"],
                "Curvature",
                "Factor value",
            ),
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # OBSERVED VERSUS FITTED CURVE
    # ---------------------------------------------------------

    st.markdown(
        "### Observed versus fitted curve"
    )

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
        title=(
            "Observed vs Fitted Curve "
            f"({latest_date:%Y-%m-%d})"
        ),
        xaxis_title="Tenor (years)",
        yaxis_title="Yield (%)",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Curve",
        height=430,
    )

    st.plotly_chart(
        fig_fit,
        use_container_width=True,
    )

    if pd.notna(rmse_bp):
        st.caption(
            f"Fit RMSE: {rmse_bp:.1f} bp. "
            f"{_fit_quality_caption(rmse_bp)}"
        )
    else:
        st.caption(
            _fit_quality_caption(
                rmse_bp
            )
        )

    # ---------------------------------------------------------
    # FIT RESIDUALS
    # ---------------------------------------------------------

    st.markdown(
        "### Fit residuals by tenor"
    )

    residual_figure = go.Figure()

    residual_figure.add_trace(
        go.Bar(
            x=residuals_df["Tenor"],
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
        xaxis_title="Tenor (years)",
        yaxis_title="Residual (basis points)",
        template="plotly_white",
        showlegend=False,
        height=350,
    )

    st.plotly_chart(
        residual_figure,
        use_container_width=True,
    )

    st.caption(
        "A positive residual means the observed yield is above "
        "the fitted curve. A negative residual means the observed "
        "yield is below the fitted curve."
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
