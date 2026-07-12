from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    DGS10,
    DGS2,
    DGS30,
    DGS5,
    TENOR_YEAR_MAP,
    YIELD_SERIES,
)


HISTORY_YEARS = 10

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

COMPARISON_OFFSETS = {
    "1D": pd.DateOffset(days=1),
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "1Y": pd.DateOffset(years=1),
}


def _curve_snapshot(
    data: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.Timestamp | None, pd.Series]:
    """
    Return the latest available yield curve on or before a requested date.

    The timestamp returned is the actual observation date used.
    """
    eligible = data.loc[
        data.index <= as_of,
        YIELD_SERIES,
    ].dropna(how="all")

    if eligible.empty:
        return None, pd.Series(dtype="float64")

    actual_date = eligible.index.max()

    curve = pd.to_numeric(
        eligible.loc[actual_date],
        errors="coerce",
    ).dropna()

    return actual_date, curve


def _value_as_of(
    series: pd.Series,
    as_of: pd.Timestamp,
) -> float:
    """Return the latest valid numeric value on or before a date."""
    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    eligible = numeric_series.loc[
        numeric_series.index <= as_of
    ].dropna()

    if eligible.empty:
        return float("nan")

    return float(
        eligible.iloc[-1]
    )


def _percentile_rank(
    series: pd.Series,
    value: float,
) -> float:
    """Calculate the historical percentile rank of a value."""
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty or pd.isna(value):
        return float("nan")

    return float(
        (clean <= value).mean() * 100.0
    )


def _yield_change_bp(
    series: pd.Series,
    current_date: pd.Timestamp,
    comparison_date: pd.Timestamp,
) -> float:
    """Calculate a yield change in basis points."""
    current_value = _value_as_of(
        series,
        current_date,
    )

    comparison_value = _value_as_of(
        series,
        comparison_date,
    )

    if (
        pd.isna(current_value)
        or pd.isna(comparison_value)
    ):
        return float("nan")

    return (
        current_value - comparison_value
    ) * 100.0


def _format_change(
    change: float,
    comparison_label: str | None = None,
) -> str | None:
    """Format a basis-point change for a Streamlit metric."""
    if pd.isna(change):
        return None

    if comparison_label:
        return f"{change:+.1f} bp ({comparison_label})"

    return f"{change:+.1f} bp"


def _render_html_table(
    dataframe: pd.DataFrame,
) -> None:
    """
    Render a small DataFrame as HTML.

    This avoids Streamlit's pandas-to-PyArrow serialization path, which
    can trigger a native segmentation fault in some environments.
    """
    display_dataframe = dataframe.copy()

    table_html = display_dataframe.to_html(
        index=False,
        na_rep="—",
        border=0,
        justify="left",
        escape=True,
        classes=[
            "safe-dataframe",
        ],
    )

    st.markdown(
        f"""
        <div style="
            width: 100%;
            overflow-x: auto;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        ">
            <style>
                table.safe-dataframe {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9rem;
                }}

                table.safe-dataframe thead th {{
                    text-align: left;
                    padding: 0.6rem 0.75rem;
                    border-bottom:
                        1px solid rgba(49, 51, 63, 0.28);
                    background:
                        rgba(49, 51, 63, 0.05);
                    white-space: nowrap;
                }}

                table.safe-dataframe tbody td {{
                    text-align: left;
                    padding: 0.6rem 0.75rem;
                    border-bottom:
                        1px solid rgba(49, 51, 63, 0.12);
                    white-space: nowrap;
                }}

                table.safe-dataframe tbody tr:hover {{
                    background:
                        rgba(49, 51, 63, 0.035);
                }}
            </style>

            {table_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _comparison_period_text(
    comparison_type: str,
    comparison_date: pd.Timestamp,
) -> str:
    """Return a readable description of the comparison period."""
    period_labels = {
        "1D": "over the past day",
        "1W": "over the past week",
        "1M": "over the past month",
        "3M": "over the past three months",
        "1Y": "over the past year",
    }

    if comparison_type in period_labels:
        return period_labels[
            comparison_type
        ]

    return (
        f"since "
        f"{comparison_date:%d %b %Y}"
    )


def _classify_curve_move(
    front_end_change: float,
    long_end_change: float,
) -> str:
    """
    Classify a 2s10s curve movement.

    Bull:
        Yields broadly fell.

    Bear:
        Yields broadly rose.

    Steepening:
        The 10Y minus 2Y spread increased.

    Flattening:
        The 10Y minus 2Y spread decreased.
    """
    if (
        pd.isna(front_end_change)
        or pd.isna(long_end_change)
    ):
        return "Unavailable"

    spread_change = (
        long_end_change
        - front_end_change
    )

    average_yield_change = (
        front_end_change
        + long_end_change
    ) / 2.0

    if (
        abs(spread_change) < 2
        and abs(average_yield_change) < 2
    ):
        return "Broadly unchanged"

    if abs(spread_change) < 2:
        if average_yield_change > 0:
            return "Parallel bear shift"

        if average_yield_change < 0:
            return "Parallel bull shift"

        return "Broadly unchanged"

    curve_direction = (
        "steepening"
        if spread_change > 0
        else "flattening"
    )

    if (
        front_end_change
        * long_end_change
        < 0
    ):
        return (
            f"Twist "
            f"{curve_direction}"
        )

    if average_yield_change > 0:
        return (
            f"Bear "
            f"{curve_direction}"
        )

    if average_yield_change < 0:
        return (
            f"Bull "
            f"{curve_direction}"
        )

    return (
        f"Curve "
        f"{curve_direction}"
    )


def _curve_movement_caption(
    curve_regime: str,
    comparison_type: str,
    comparison_date: pd.Timestamp,
) -> str:
    """Explain the mechanical curve-movement classification."""
    period_text = _comparison_period_text(
        comparison_type,
        comparison_date,
    )

    descriptions = {
        "Bear flattening": (
            f"Yields rose {period_text}, with shorter maturities "
            "increasing more than longer maturities."
        ),
        "Bear steepening": (
            f"Yields rose {period_text}, with longer maturities "
            "increasing more than shorter maturities."
        ),
        "Bull flattening": (
            f"Yields fell {period_text}, with longer maturities "
            "declining more than shorter maturities."
        ),
        "Bull steepening": (
            f"Yields fell {period_text}, with shorter maturities "
            "declining more than longer maturities."
        ),
        "Twist flattening": (
            f"The curve twisted flatter {period_text}: shorter-maturity "
            "yields rose while longer-maturity yields fell."
        ),
        "Twist steepening": (
            f"The curve twisted steeper {period_text}: shorter-maturity "
            "yields fell while longer-maturity yields rose."
        ),
        "Parallel bear shift": (
            "Yields rose by broadly similar amounts across the curve "
            f"{period_text}."
        ),
        "Parallel bull shift": (
            "Yields fell by broadly similar amounts across the curve "
            f"{period_text}."
        ),
        "Curve flattening": (
            f"The 2s10s spread narrowed {period_text}, while the overall "
            "direction of yields was mixed or limited."
        ),
        "Curve steepening": (
            f"The 2s10s spread widened {period_text}, while the overall "
            "direction of yields was mixed or limited."
        ),
        "Broadly unchanged": (
            "Yield levels and the 2s10s slope were broadly unchanged "
            f"{period_text}."
        ),
        "Unavailable": (
            "The curve movement could not be classified because the "
            "required yield observations are unavailable."
        ),
    }

    return descriptions.get(
        curve_regime,
        (
            f"The curve movement {period_text} is mechanically "
            f"classified as {curve_regime.lower()}."
        ),
    )


def _curve_shape(
    spread_bp: float,
) -> str:
    """Describe whether a curve spread is positive, flat or inverted."""
    if pd.isna(spread_bp):
        return "Unavailable"

    if spread_bp > 10:
        return "Positive"

    if spread_bp < -10:
        return "Inverted"

    return "Near-flat"


def _spread_movement(
    change_bp: float,
) -> str:
    """Describe whether a spread is steepening or flattening."""
    if pd.isna(change_bp):
        return "Unavailable"

    if change_bp > 2:
        return "Steepening"

    if change_bp < -2:
        return "Flattening"

    return "Broadly stable"


def _two_ten_caption(
    current_spread: float,
    spread_change: float,
) -> str:
    """Return a concise interpretation of the 2s10s spread."""
    if (
        pd.isna(current_spread)
        or pd.isna(spread_change)
    ):
        return (
            "Near-term policy outlook versus the "
            "medium-to-long-term economic outlook."
        )

    is_inverted = (
        current_spread < -10
    )

    is_flattening = (
        spread_change < -2
    )

    is_positive = (
        current_spread > 10
    )

    is_steepening = (
        spread_change > 2
    )

    if (
        is_inverted
        or is_flattening
    ):
        return (
            "May signal tighter near-term Fed policy relative "
            "to the longer-term growth and inflation outlook."
        )

    if (
        is_positive
        or is_steepening
    ):
        return (
            "May reflect expected Fed easing or stronger future "
            "growth and inflation."
        )

    return (
        "Near-term policy and the longer-term economic outlook "
        "are priced relatively close together."
    )


def _five_thirty_caption(
    current_spread: float,
    spread_change: float,
) -> str:
    """Return a concise interpretation of the 5s30s spread."""
    if (
        pd.isna(current_spread)
        or pd.isna(spread_change)
    ):
        return (
            "Long-term bond compensation relative to "
            "intermediate maturities."
        )

    is_inverted = (
        current_spread < -10
    )

    is_flattening = (
        spread_change < -2
    )

    is_positive = (
        current_spread > 10
    )

    is_steepening = (
        spread_change > 2
    )

    if (
        is_inverted
        or is_flattening
    ):
        return (
            "May reflect lower long-run growth and inflation, "
            "strong long-bond demand or elevated 5Y yields."
        )

    if (
        is_positive
        or is_steepening
    ):
        return (
            "May reflect greater compensation for inflation, "
            "fiscal, supply and long-duration risks."
        )

    return (
        "Long-end compensation is broadly stable relative "
        "to intermediate maturities."
    )


def _percentile_description(
    percentile: float,
) -> str:
    """Convert a percentile into an intuitive description."""
    if pd.isna(percentile):
        return "unavailable"

    if percentile <= 10:
        return "near the bottom decile"

    if percentile <= 25:
        return "in the lower quartile"

    if percentile >= 90:
        return "near the top decile"

    if percentile >= 75:
        return "in the upper quartile"

    return (
        "near the middle of its range"
    )


def _move_driver(
    two_year_change: float,
    ten_year_change: float,
) -> str:
    """Explain which tenor mechanically drove the 2s10s movement."""
    if (
        pd.isna(two_year_change)
        or pd.isna(ten_year_change)
    ):
        return (
            "The tenor-level driver is unavailable."
        )

    difference = abs(
        two_year_change
        - ten_year_change
    )

    if difference < 1:
        return (
            "The 2-year and 10-year yields moved "
            "by similar amounts."
        )

    if (
        two_year_change >= 0
        and ten_year_change >= 0
    ):
        if (
            two_year_change
            > ten_year_change
        ):
            return (
                "The 2-year yield rose more than the "
                "10-year yield, driving the curve flatter."
            )

        return (
            "The 10-year yield rose more than the "
            "2-year yield, driving the curve steeper."
        )

    if (
        two_year_change <= 0
        and ten_year_change <= 0
    ):
        if (
            abs(two_year_change)
            > abs(ten_year_change)
        ):
            return (
                "The 2-year yield fell more than the "
                "10-year yield, driving the curve steeper."
            )

        return (
            "The 10-year yield fell more than the "
            "2-year yield, driving the curve flatter."
        )

    if (
        two_year_change > 0
        and ten_year_change < 0
    ):
        return (
            "The 2-year yield rose while the 10-year "
            "yield fell, producing a pronounced "
            "flattening twist."
        )

    return (
        "The 2-year yield fell while the 10-year "
        "yield rose, producing a pronounced "
        "steepening twist."
    )


def _build_spread_context(
    spreads: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build current spread levels, monthly changes and percentiles.
    """
    one_year_cutoff = (
        latest_date
        - pd.DateOffset(years=1)
    )

    five_year_cutoff = (
        latest_date
        - pd.DateOffset(years=5)
    )

    one_month_date = (
        latest_date
        - pd.DateOffset(months=1)
    )

    rows: list[
        dict[str, str | float]
    ] = []

    for spread_name in [
        "2s10s",
        "5s30s",
    ]:
        spread_series = pd.to_numeric(
            spreads[spread_name],
            errors="coerce",
        ).loc[
            spreads.index <= latest_date
        ]

        current_value = _value_as_of(
            spread_series,
            latest_date,
        )

        one_month_value = _value_as_of(
            spread_series,
            one_month_date,
        )

        if (
            pd.isna(current_value)
            or pd.isna(one_month_value)
        ):
            one_month_change = float("nan")
        else:
            one_month_change = (
                current_value
                - one_month_value
            )

        one_year_history = spread_series.loc[
            spread_series.index
            >= one_year_cutoff
        ]

        five_year_history = spread_series.loc[
            spread_series.index
            >= five_year_cutoff
        ]

        one_year_percentile = _percentile_rank(
            one_year_history,
            current_value,
        )

        five_year_percentile = _percentile_rank(
            five_year_history,
            current_value,
        )

        shape = _curve_shape(
            current_value
        )

        movement = _spread_movement(
            one_month_change
        )

        rows.append(
            {
                "Spread": spread_name,
                "Current (bp)": (
                    round(
                        current_value,
                        1,
                    )
                    if pd.notna(
                        current_value
                    )
                    else np.nan
                ),
                "1M change (bp)": (
                    round(
                        one_month_change,
                        1,
                    )
                    if pd.notna(
                        one_month_change
                    )
                    else np.nan
                ),
                "1Y percentile": (
                    round(
                        one_year_percentile,
                        1,
                    )
                    if pd.notna(
                        one_year_percentile
                    )
                    else np.nan
                ),
                "5Y percentile": (
                    round(
                        five_year_percentile,
                        1,
                    )
                    if pd.notna(
                        five_year_percentile
                    )
                    else np.nan
                ),
                "Regime": (
                    f"{shape}, "
                    f"{movement.lower()}"
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Spread",
            "Current (bp)",
            "1M change (bp)",
            "1Y percentile",
            "5Y percentile",
            "Regime",
        ],
    )


def _spread_figure(
    spread: pd.Series,
    title: str,
    current_value: float,
) -> go.Figure:
    """Create a spread-history chart."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread,
            mode="lines",
            name=title,
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.6,
        annotation_text="Zero",
        annotation_position="bottom right",
    )

    if pd.notna(current_value):
        figure.add_hline(
            y=current_value,
            line_dash="dot",
            opacity=0.5,
            annotation_text=(
                f"Current: "
                f"{current_value:.0f} bp"
            ),
            annotation_position="top right",
        )

    figure.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Spread (basis points)",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )

    return figure


def _spread_direction_text(
    spread_change: float,
) -> str:
    """Return a plain-English description of a spread change."""
    if pd.isna(spread_change):
        return (
            "could not be classified"
        )

    if spread_change > 2:
        return "steepened"

    if spread_change < -2:
        return "flattened"

    return (
        "been broadly stable"
    )


def render(
    fred_client,
    context: dict,
) -> None:
    st.subheader(
        "Panel 1: Treasury Yield Curve"
    )

    display_start_date = context[
        "start_date"
    ]

    display_end_date = context[
        "end_date"
    ]

    ten_year_start = (
        pd.Timestamp(
            display_end_date
        )
        - pd.DateOffset(
            years=HISTORY_YEARS
        )
    ).date()

    history_start_date = min(
        display_start_date,
        ten_year_start,
    )

    result = fred_client.get_series(
        YIELD_SERIES,
        history_start_date,
        display_end_date,
    )

    if (
        not result.success
        or result.data is None
    ):
        st.warning(
            result.message
            or "Treasury yield data are unavailable."
        )
        return

    data = (
        result.data
        .copy()
        .sort_index()
    )

    if data.empty:
        st.info(
            "No Treasury yield data are available."
        )
        return

    missing_series = [
        series_id
        for series_id in YIELD_SERIES
        if series_id not in data.columns
    ]

    if missing_series:
        st.warning(
            "The following Treasury series are missing: "
            + ", ".join(
                missing_series
            )
        )
        return

    for series_id in YIELD_SERIES:
        data[series_id] = pd.to_numeric(
            data[series_id],
            errors="coerce",
        ).astype("float64")

    complete_curves = data[
        YIELD_SERIES
    ].dropna(
        how="any"
    )

    if complete_curves.empty:
        usable_curves = data[
            YIELD_SERIES
        ].dropna(
            how="all"
        )

        if usable_curves.empty:
            st.warning(
                "No usable Treasury curve "
                "observations were found."
            )
            return

        latest_requested_date = (
            usable_curves.index.max()
        )
    else:
        latest_requested_date = (
            complete_curves.index.max()
        )

    latest_date, current_curve = (
        _curve_snapshot(
            data,
            latest_requested_date,
        )
    )

    if (
        latest_date is None
        or current_curve.empty
    ):
        st.warning(
            "The latest Treasury curve "
            "could not be constructed."
        )
        return

    st.caption(
        "Latest complete curve observation: "
        f"{latest_date:%d %b %Y}"
    )

    comparison_type = st.radio(
        "Compare the current curve with",
        options=[
            "1D",
            "1W",
            "1M",
            "3M",
            "1Y",
            "Custom",
        ],
        index=2,
        horizontal=True,
        key="yield_curve_comparison",
    )

    if comparison_type == "Custom":
        selected_comparison_date = (
            st.date_input(
                "Custom comparison date",
                value=(
                    latest_date
                    - pd.DateOffset(
                        months=1
                    )
                ).date(),
                min_value=(
                    data.index.min().date()
                ),
                max_value=(
                    latest_date.date()
                ),
                key=(
                    "yield_curve_custom_date"
                ),
            )
        )

        requested_comparison_date = (
            pd.Timestamp(
                selected_comparison_date
            )
        )
    else:
        requested_comparison_date = (
            latest_date
            - COMPARISON_OFFSETS[
                comparison_type
            ]
        )

    comparison_date, comparison_curve = (
        _curve_snapshot(
            data,
            requested_comparison_date,
        )
    )

    if (
        comparison_date is None
        or comparison_curve.empty
    ):
        st.warning(
            "No comparison curve is available "
            "for the selected period."
        )
        return

    spreads = pd.DataFrame(
        index=data.index
    )

    spreads["2s10s"] = (
        data[DGS10]
        - data[DGS2]
    ) * 100.0

    spreads["5s30s"] = (
        data[DGS30]
        - data[DGS5]
    ) * 100.0

    current_two_year = _value_as_of(
        data[DGS2],
        latest_date,
    )

    current_ten_year = _value_as_of(
        data[DGS10],
        latest_date,
    )

    current_thirty_year = _value_as_of(
        data[DGS30],
        latest_date,
    )

    comparison_two_year = _value_as_of(
        data[DGS2],
        comparison_date,
    )

    comparison_ten_year = _value_as_of(
        data[DGS10],
        comparison_date,
    )

    comparison_thirty_year = _value_as_of(
        data[DGS30],
        comparison_date,
    )

    two_year_change = (
        current_two_year
        - comparison_two_year
    ) * 100.0

    ten_year_change = (
        current_ten_year
        - comparison_ten_year
    ) * 100.0

    thirty_year_change = (
        current_thirty_year
        - comparison_thirty_year
    ) * 100.0

    current_2s10s = _value_as_of(
        spreads["2s10s"],
        latest_date,
    )

    current_5s30s = _value_as_of(
        spreads["5s30s"],
        latest_date,
    )

    comparison_2s10s = _value_as_of(
        spreads["2s10s"],
        comparison_date,
    )

    comparison_5s30s = _value_as_of(
        spreads["5s30s"],
        comparison_date,
    )

    change_2s10s = (
        current_2s10s
        - comparison_2s10s
    )

    change_5s30s = (
        current_5s30s
        - comparison_5s30s
    )

    curve_regime = _classify_curve_move(
        two_year_change,
        ten_year_change,
    )

    curve_movement_caption = (
        _curve_movement_caption(
            curve_regime,
            comparison_type,
            comparison_date,
        )
    )

    spread_context = (
        _build_spread_context(
            spreads,
            latest_date,
        )
    )

    two_ten_row = spread_context.loc[
        spread_context["Spread"]
        == "2s10s"
    ]

    five_thirty_row = spread_context.loc[
        spread_context["Spread"]
        == "5s30s"
    ]

    one_year_2s10s_percentile = (
        float(
            two_ten_row[
                "1Y percentile"
            ].iloc[0]
        )
        if not two_ten_row.empty
        else float("nan")
    )

    one_year_5s30s_percentile = (
        float(
            five_thirty_row[
                "1Y percentile"
            ].iloc[0]
        )
        if not five_thirty_row.empty
        else float("nan")
    )

    spread_direction = (
        _spread_direction_text(
            change_2s10s
        )
    )

    interpretation = (
        f"The Treasury curve is "
        f"{_curve_shape(current_2s10s).lower()} "
        f"across 2s10s, with the spread at "
        f"{current_2s10s:+.0f} bp. "
        f"The 5s30s spread stands at "
        f"{current_5s30s:+.0f} bp. "
        f"Relative to the past year, 2s10s is "
        f"{_percentile_description(one_year_2s10s_percentile)}, "
        f"while 5s30s is "
        f"{_percentile_description(one_year_5s30s_percentile)}. "
        f"Since {comparison_date:%d %b %Y}, "
        f"2s10s has {spread_direction} "
        f"by {abs(change_2s10s):.1f} bp. "
        f"The movement is mechanically classified "
        f"as {curve_regime.lower()}. "
        f"{_move_driver(two_year_change, ten_year_change)}"
    )

    # ---------------------------------------------------------
    # TOP-OF-PAGE INTERPRETATION AND USER GUIDE
    # ---------------------------------------------------------

    st.markdown(
        "### Automated interpretation"
    )

    st.info(
        interpretation
    )

    st.caption(
        "This classification describes the observed "
        "yield movement. It does not by itself establish "
        "whether the move was caused by Federal Reserve "
        "expectations, growth, inflation, Treasury supply "
        "or term premium."
    )

    with st.expander(
        "How to use this panel",
        expanded=True,
    ):
        st.markdown(
            """
Use the panel in four steps:

1. **Check the current yield levels and curve spreads.**  
   The 2-year yield is relatively sensitive to the expected Federal
   Reserve policy path. The 10-year and 30-year yields contain more
   exposure to longer-term growth, inflation and term premium.

2. **Select a comparison horizon.**  
   Compare the current curve with one day, one week, one month,
   three months or one year ago. The selected period determines the
   yield changes and curve-movement classification shown below.

3. **Identify which tenor drove the move.**  
   The yield-change chart shows whether repricing was concentrated
   at the front end, the intermediate part of the curve or the long end.

4. **Place the move in historical context.**  
   The spread charts and detailed percentile table show whether 2s10s
   and 5s30s are unusually steep, flat or inverted relative to history.

**2s10s** is the 10-year yield minus the 2-year yield. It compares the
near-term monetary-policy outlook with the medium-to-long-term economic
outlook.

**5s30s** is the 30-year yield minus the 5-year yield. It represents the
extra yield required to hold very long-term rather than
intermediate-maturity bonds.

A **steepening** means the spread increased. A **flattening** means the
spread decreased. A **bull** move means yields broadly fell, while a
**bear** move means yields broadly rose.
            """
        )

    # ---------------------------------------------------------
    # CURRENT CURVE SUMMARY
    # ---------------------------------------------------------

    st.markdown(
        "### Current curve summary"
    )

    (
        metric_1,
        metric_2,
        metric_3,
        metric_4,
        metric_5,
        metric_6,
    ) = st.columns(6)

    with metric_1:
        st.metric(
            "2Y yield",
            f"{current_two_year:.2f}%",
            _format_change(
                two_year_change
                ,
                comparison_type,
            ),
        )

        st.caption(
            "Reflects expected Federal Reserve monetary "
            "policy in the near term."
        )

    with metric_2:
        st.metric(
            "10Y yield",
            f"{current_ten_year:.2f}%",
            _format_change(
                ten_year_change
                ,
                comparison_type,
            ),
        )

        st.caption(
            "Reflects growth, inflation and term-premium "
            "expectations."
        )

    with metric_3:
        st.metric(
            "30Y yield",
            f"{current_thirty_year:.2f}%",
            _format_change(
                thirty_year_change
                ,
                comparison_type,
            ),
        )

        st.caption(
            "Reflects long-run growth, inflation and "
            "fiscal expectations."
        )

    with metric_4:
        st.metric(
            "2s10s",
            f"{current_2s10s:+.0f} bp",
            _format_change(
                change_2s10s
                ,
                comparison_type,
            ),
        )

        st.caption(
            _two_ten_caption(
                current_2s10s,
                change_2s10s,
            )
        )

    with metric_5:
        st.metric(
            "5s30s",
            f"{current_5s30s:+.0f} bp",
            _format_change(
                change_5s30s
                ,
                comparison_type,
            ),
        )

        st.caption(
            _five_thirty_caption(
                current_5s30s,
                change_5s30s,
            )
        )

    with metric_6:
        st.metric(
            "Yield-Curve Movement",
            curve_regime,
        )

        st.caption(
            curve_movement_caption
        )

    st.caption(
        "Metric changes compare "
        f"{latest_date:%d %b %Y} with "
        f"{comparison_date:%d %b %Y}."
    )

    # ---------------------------------------------------------
    # CURRENT CURVE VS COMPARISON DATE
    # ---------------------------------------------------------

    st.markdown(
        "### Current curve versus comparison date"
    )

    current_tenors = [
        TENOR_YEAR_MAP[series_id]
        for series_id in current_curve.index
    ]

    comparison_tenors = [
        TENOR_YEAR_MAP[series_id]
        for series_id in comparison_curve.index
    ]

    curve_figure = go.Figure()

    curve_figure.add_trace(
        go.Scatter(
            x=current_tenors,
            y=current_curve.values,
            mode="lines+markers",
            name=(
                f"Current "
                f"({latest_date:%Y-%m-%d})"
            ),
        )
    )

    curve_figure.add_trace(
        go.Scatter(
            x=comparison_tenors,
            y=comparison_curve.values,
            mode="lines+markers",
            name=(
                f"Comparison "
                f"({comparison_date:%Y-%m-%d})"
            ),
        )
    )

    curve_figure.update_layout(
        title="Treasury Yield Curve",
        xaxis_title="Tenor (years)",
        yaxis_title="Yield (%)",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Curve",
    )

    st.plotly_chart(
        curve_figure,
        use_container_width=True,
    )

    # ---------------------------------------------------------
    # YIELD CHANGE BAR CHART
    # ---------------------------------------------------------

    curve_change_rows: list[
        dict[str, str | float]
    ] = []

    for series_id in YIELD_SERIES:
        current_value = _value_as_of(
            data[series_id],
            latest_date,
        )

        comparison_value = _value_as_of(
            data[series_id],
            comparison_date,
        )

        if (
            pd.isna(current_value)
            or pd.isna(comparison_value)
        ):
            continue

        curve_change_rows.append(
            {
                "Tenor": TENOR_LABELS.get(
                    series_id,
                    series_id,
                ),
                "Change": (
                    current_value
                    - comparison_value
                ) * 100.0,
            }
        )

    curve_change_data = pd.DataFrame(
        curve_change_rows,
        columns=[
            "Tenor",
            "Change",
        ],
    )

    if not curve_change_data.empty:
        change_figure = go.Figure()

        change_figure.add_trace(
            go.Bar(
                x=curve_change_data[
                    "Tenor"
                ],
                y=curve_change_data[
                    "Change"
                ],
                name="Yield change",
            )
        )

        change_figure.add_hline(
            y=0,
            line_dash="dash",
            opacity=0.6,
        )

        change_figure.update_layout(
            title=(
                "Yield Change Since "
                f"{comparison_date:%d %b %Y}"
            ),
            xaxis_title="Tenor",
            yaxis_title=(
                "Change (basis points)"
            ),
            template="plotly_white",
            showlegend=False,
        )

        st.plotly_chart(
            change_figure,
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # DETAILED YIELD-CHANGE DATA
    # ---------------------------------------------------------

    yield_change_rows: list[
        dict[str, str | float]
    ] = []

    standard_periods = {
        "1D change (bp)": pd.DateOffset(
            days=1
        ),
        "1W change (bp)": pd.DateOffset(
            weeks=1
        ),
        "1M change (bp)": pd.DateOffset(
            months=1
        ),
    }

    for series_id in YIELD_SERIES:
        current_value = _value_as_of(
            data[series_id],
            latest_date,
        )

        row: dict[
            str,
            str | float,
        ] = {
            "Tenor": str(
                TENOR_LABELS.get(
                    series_id,
                    series_id,
                )
            ),
            "Current yield (%)": (
                round(
                    current_value,
                    3,
                )
                if pd.notna(
                    current_value
                )
                else np.nan
            ),
        }

        for (
            column_name,
            offset,
        ) in standard_periods.items():
            change = _yield_change_bp(
                data[series_id],
                latest_date,
                latest_date - offset,
            )

            row[column_name] = (
                round(
                    change,
                    1,
                )
                if pd.notna(
                    change
                )
                else np.nan
            )

        yield_change_rows.append(
            row
        )

    yield_change_table = pd.DataFrame(
        yield_change_rows,
        columns=[
            "Tenor",
            "Current yield (%)",
            "1D change (bp)",
            "1W change (bp)",
            "1M change (bp)",
        ],
    )

    # ---------------------------------------------------------
    # SPREAD HISTORY
    # ---------------------------------------------------------

    display_start_timestamp = (
        pd.Timestamp(
            display_start_date
        )
    )

    display_end_timestamp = (
        pd.Timestamp(
            display_end_date
        )
    )

    displayed_spreads = spreads.loc[
        (
            spreads.index
            >= display_start_timestamp
        )
        & (
            spreads.index
            <= display_end_timestamp
        )
    ]

    st.markdown(
        "### Curve-spread history"
    )

    (
        spread_column_1,
        spread_column_2,
    ) = st.columns(2)

    with spread_column_1:
        st.plotly_chart(
            _spread_figure(
                displayed_spreads[
                    "2s10s"
                ],
                "2s10s: Policy versus economic outlook",
                current_2s10s,
            ),
            use_container_width=True,
        )

    with spread_column_2:
        st.plotly_chart(
            _spread_figure(
                displayed_spreads[
                    "5s30s"
                ],
                "5s30s: Long-duration compensation",
                current_5s30s,
            ),
            use_container_width=True,
        )

    # ---------------------------------------------------------
    # DETAILED DATA TABLES
    # ---------------------------------------------------------

    with st.expander(
        "View detailed data",
        expanded=False,
    ):
        st.markdown(
            "#### Yield changes by tenor"
        )

        _render_html_table(
            yield_change_table
        )

        st.markdown(
            "#### Historical spread context"
        )

        _render_html_table(
            spread_context
        )

        st.caption(
            "Spread percentiles use the panel's internally "
            "fetched history rather than only the date range "
            "displayed in the sidebar."
        )
