from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    GACDISA066MSFRBPHI,
    GDPC1,
    GROWTH_SERIES,
    ICSA,
    INDPRO,
    PAYEMS,
)


# The sidebar controls the displayed period.
# The panel fetches a longer history to calculate rolling z-scores.
ANALYSIS_START_DATE = date(2000, 1, 1)

# Rolling history used to standardise each indicator.
ZSCORE_WINDOW_MONTHS = 60
ZSCORE_MIN_PERIODS = 24

# Number of months used to assess recent momentum.
MOMENTUM_MONTHS = 3

FEATURE_COLUMNS = [
    "claims_growth",
    "industrial_production_growth",
    "payroll_growth",
    "philly_fed_activity",
]

FEATURE_LABELS = {
    "claims_growth": "Initial claims",
    "industrial_production_growth": "Industrial production",
    "payroll_growth": "Payroll growth",
    "philly_fed_activity": "Philadelphia Fed activity",
}

FEATURE_DESCRIPTIONS = {
    "claims_growth": (
        "Inverted growth in initial jobless claims. "
        "A higher score indicates stronger labour-market conditions."
    ),
    "industrial_production_growth": (
        "Annualised three-month growth in industrial production."
    ),
    "payroll_growth": (
        "Annualised three-month growth in nonfarm payroll employment."
    ),
    "philly_fed_activity": (
        "Level of the Philadelphia Fed activity diffusion index."
    ),
}

METRIC_CAPTION_MIN_HEIGHT_PX = 88


def _value_as_of(
    series: pd.Series,
    as_of: pd.Timestamp,
) -> float:
    """Return the latest valid numeric value on or before a date."""
    clean = pd.to_numeric(
        series,
        errors="coerce",
    )

    eligible = clean.loc[
        clean.index <= as_of
    ].dropna()

    if eligible.empty:
        return float("nan")

    return float(
        eligible.iloc[-1]
    )


def _annualized_three_month_growth(
    series: pd.Series,
) -> pd.Series:
    """
    Calculate annualised growth over the previous three months.

    Formula:
        ((current / three_months_ago) ** 4 - 1) * 100
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    growth = (
        (
            numeric
            / numeric.shift(3)
        )
        ** 4
        - 1.0
    ) * 100.0

    return growth.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _rolling_zscore(
    series: pd.Series,
    window: int = ZSCORE_WINDOW_MONTHS,
    min_periods: int = ZSCORE_MIN_PERIODS,
) -> pd.Series:
    """
    Standardise a series using a rolling mean and standard deviation.

    A rolling window prevents the score from being dominated by very old
    economic regimes while retaining enough history for context.
    """
    clean = pd.to_numeric(
        series,
        errors="coerce",
    )

    rolling_mean = clean.rolling(
        window=window,
        min_periods=min_periods,
    ).mean()

    rolling_std = clean.rolling(
        window=window,
        min_periods=min_periods,
    ).std()

    zscore = (
        clean - rolling_mean
    ) / rolling_std.replace(
        0.0,
        np.nan,
    )

    return zscore.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _percentile_rank(
    series: pd.Series,
    value: float,
) -> float:
    """Calculate the percentile rank of a value within a history."""
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if (
        clean.empty
        or pd.isna(value)
    ):
        return float("nan")

    return float(
        (clean <= value).mean()
        * 100.0
    )


def _change_over_months(
    series: pd.Series,
    latest_date: pd.Timestamp,
    months: int,
) -> float:
    """Calculate the change from an earlier monthly observation."""
    latest_value = _value_as_of(
        series,
        latest_date,
    )

    earlier_value = _value_as_of(
        series,
        latest_date
        - pd.DateOffset(
            months=months,
        ),
    )

    if (
        pd.isna(latest_value)
        or pd.isna(earlier_value)
    ):
        return float("nan")

    return (
        latest_value
        - earlier_value
    )


def _format_change(
    change: float,
    suffix: str = "",
) -> str:
    """Format a signed change."""
    if pd.isna(change):
        return "Unavailable"

    return (
        f"{change:+.2f}"
        f"{suffix}"
    )


def _format_change_or_unavailable(
    change: float,
    digits: int = 2,
) -> str:
    """Format a signed change or return an unavailable label."""
    if pd.isna(change):
        return "Unavailable"

    return (
        f"{change:+.{digits}f}"
    )


def _percentile_description(
    percentile: float,
) -> str:
    """Convert a percentile into concise historical context."""
    if pd.isna(percentile):
        return "Historical position unavailable."

    if percentile >= 90:
        position = (
            "near the top of its historical distribution"
        )
    elif percentile >= 75:
        position = (
            "in the upper quartile of its historical distribution"
        )
    elif percentile <= 10:
        position = (
            "near the bottom of its historical distribution"
        )
    elif percentile <= 25:
        position = (
            "in the lower quartile of its historical distribution"
        )
    else:
        position = (
            "near the middle of its historical distribution"
        )

    return (
        f"{percentile:.0f}th percentile; "
        f"{position}."
    )


def _render_metric_caption(
    text: str,
) -> None:
    """Render aligned explanatory text beneath a metric."""
    st.markdown(
        f"""
        <div style="
            min-height: {METRIC_CAPTION_MIN_HEIGHT_PX}px;
            margin-top: 0.55rem;
            color: rgba(49, 51, 63, 0.62);
            font-size: 0.875rem;
            line-height: 1.5;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(
    show_spinner=False,
)
def _prepare_growth_data(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    """
    Convert raw FRED observations into monthly growth indicators.

    Returns:
        raw_features:
            Economically interpretable transformed indicators.

        indicator_scores:
            Rolling z-scores where higher values consistently mean
            stronger economic activity.

        gdp_yoy:
            Quarterly real GDP growth for contextual comparison only.
    """
    numeric_data = data.copy()

    for series_id in GROWTH_SERIES:
        if series_id in numeric_data.columns:
            numeric_data[series_id] = pd.to_numeric(
                numeric_data[series_id],
                errors="coerce",
            )

    monthly = pd.DataFrame(
        {
            ICSA: (
                numeric_data[ICSA]
                .resample("ME")
                .mean()
            ),
            INDPRO: (
                numeric_data[INDPRO]
                .resample("ME")
                .last()
            ),
            PAYEMS: (
                numeric_data[PAYEMS]
                .resample("ME")
                .last()
            ),
            GACDISA066MSFRBPHI: (
                numeric_data[
                    GACDISA066MSFRBPHI
                ]
                .resample("ME")
                .mean()
            ),
        }
    )

    raw_features = pd.DataFrame(
        index=monthly.index
    )

    # Claims are inverted so higher values consistently indicate
    # stronger activity across all four indicators.
    raw_features[
        "claims_growth"
    ] = -_annualized_three_month_growth(
        monthly[ICSA]
    )

    raw_features[
        "industrial_production_growth"
    ] = _annualized_three_month_growth(
        monthly[INDPRO]
    )

    raw_features[
        "payroll_growth"
    ] = _annualized_three_month_growth(
        monthly[PAYEMS]
    )

    raw_features[
        "philly_fed_activity"
    ] = monthly[
        GACDISA066MSFRBPHI
    ]

    raw_features = raw_features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    indicator_scores = pd.DataFrame(
        index=raw_features.index
    )

    for column in FEATURE_COLUMNS:
        indicator_scores[column] = (
            _rolling_zscore(
                raw_features[column]
            )
        )

    indicator_scores[
        "growth_momentum_index"
    ] = indicator_scores[
        FEATURE_COLUMNS
    ].mean(
        axis=1,
        skipna=True,
    )

    real_gdp_quarterly = (
        numeric_data[GDPC1]
        .resample("QE-DEC")
        .last()
    )

    gdp_yoy = (
        real_gdp_quarterly
        .pct_change(
            4,
            fill_method=None,
        )
        * 100.0
    )

    gdp_yoy.name = "Real GDP YoY"

    return (
        raw_features,
        indicator_scores,
        gdp_yoy,
    )


def _growth_regime(
    composite_score: float,
) -> tuple[str, str]:
    """
    Classify the level of the composite growth-momentum score.

    This is a descriptive indicator regime, not a GDP forecast or
    recession probability.
    """
    if pd.isna(composite_score):
        return (
            "Unavailable",
            "Insufficient indicator history to classify growth conditions.",
        )

    if composite_score >= 1.0:
        return (
            "Strong growth momentum",
            (
                "Growth indicators are collectively running well above "
                "their recent historical norms."
            ),
        )

    if composite_score >= 0.35:
        return (
            "Moderate growth momentum",
            (
                "Growth indicators remain above their recent historical "
                "norms, consistent with continued expansion."
            ),
        )

    if composite_score > -0.35:
        return (
            "Neutral growth momentum",
            (
                "The indicator set is close to its recent historical norm, "
                "with no strong broad-based growth signal."
            ),
        )

    if composite_score > -1.0:
        return (
            "Moderate slowdown",
            (
                "Growth indicators are running below their recent historical "
                "norms, indicating a broad loss of momentum."
            ),
        )

    return (
        "Sharp slowdown",
        (
            "Growth indicators are collectively well below their recent "
            "historical norms."
        ),
    )


def _momentum_direction(
    change_3m: float,
) -> str:
    """Classify the recent direction of the composite score."""
    if pd.isna(change_3m):
        return "Unavailable"

    if change_3m >= 0.35:
        return "Improving"

    if change_3m <= -0.35:
        return "Deteriorating"

    return "Broadly stable"


def _indicator_direction(
    change: float,
) -> str:
    """Classify whether an indicator improved or weakened."""
    if pd.isna(change):
        return "Unavailable"

    if change > 0.10:
        return "Improving"

    if change < -0.10:
        return "Weakening"

    return "Stable"


def _build_indicator_snapshot(
    indicator_scores: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build current scores and three-month changes by indicator."""
    rows: list[
        dict[str, str | float]
    ] = []

    for column in FEATURE_COLUMNS:
        current_score = _value_as_of(
            indicator_scores[column],
            latest_date,
        )

        change_3m = _change_over_months(
            indicator_scores[column],
            latest_date,
            MOMENTUM_MONTHS,
        )

        rows.append(
            {
                "Indicator": FEATURE_LABELS[
                    column
                ],
                "Feature": column,
                "Current score": current_score,
                "3M change": change_3m,
                "Direction": _indicator_direction(
                    change_3m
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _breadth_summary(
    snapshot: pd.DataFrame,
) -> tuple[int, int, int]:
    """Count improving, weakening and stable indicators."""
    if snapshot.empty:
        return (
            0,
            0,
            0,
        )

    improving = int(
        (
            snapshot["Direction"]
            == "Improving"
        ).sum()
    )

    weakening = int(
        (
            snapshot["Direction"]
            == "Weakening"
        ).sum()
    )

    stable = int(
        (
            snapshot["Direction"]
            == "Stable"
        ).sum()
    )

    return (
        improving,
        weakening,
        stable,
    )


def _strongest_indicator(
    snapshot: pd.DataFrame,
    strongest: bool,
) -> tuple[str, float]:
    """Return the strongest or weakest current indicator score."""
    valid = snapshot.dropna(
        subset=[
            "Current score",
        ]
    )

    if valid.empty:
        return (
            "Unavailable",
            float("nan"),
        )

    if strongest:
        row = valid.loc[
            valid[
                "Current score"
            ].idxmax()
        ]
    else:
        row = valid.loc[
            valid[
                "Current score"
            ].idxmin()
        ]

    return (
        str(
            row[
                "Indicator"
            ]
        ),
        float(
            row[
                "Current score"
            ]
        ),
    )


def _largest_momentum_driver(
    snapshot: pd.DataFrame,
    improving: bool,
) -> tuple[str, float]:
    """Return the largest positive or negative three-month move."""
    valid = snapshot.dropna(
        subset=[
            "3M change",
        ]
    )

    if valid.empty:
        return (
            "Unavailable",
            float("nan"),
        )

    if improving:
        row = valid.loc[
            valid[
                "3M change"
            ].idxmax()
        ]
    else:
        row = valid.loc[
            valid[
                "3M change"
            ].idxmin()
        ]

    return (
        str(
            row[
                "Indicator"
            ]
        ),
        float(
            row[
                "3M change"
            ]
        ),
    )


def _build_macro_note(
    latest_date: pd.Timestamp,
    composite_score: float,
    regime: str,
    change_1m: float,
    change_3m: float,
    percentile: float,
    snapshot: pd.DataFrame,
) -> str:
    """Generate a growth-momentum paragraph for a macro note."""
    statements: list[str] = []

    statements.append(
        f"As of {latest_date:%d %b %Y}, the growth indicator set is "
        f"classified as {regime.lower()}, with a composite momentum "
        f"score of {composite_score:+.2f}."
    )

    if pd.notna(change_1m) and pd.notna(change_3m):
        statements.append(
            f"The composite changed {change_1m:+.2f} standard "
            f"deviations over the past month and {change_3m:+.2f} "
            "standard deviations over the past three months."
        )
    elif pd.notna(change_3m):
        statements.append(
            f"The composite changed {change_3m:+.2f} standard "
            "deviations over the past three months."
        )
    elif pd.notna(change_1m):
        statements.append(
            f"The latest monthly change was {change_1m:+.2f} "
            "standard deviations."
        )

    (
        improving_count,
        weakening_count,
        stable_count,
    ) = _breadth_summary(
        snapshot
    )

    if weakening_count > improving_count:
        statements.append(
            f"{weakening_count} of {len(FEATURE_COLUMNS)} indicators "
            "weakened over the past three months, suggesting the slowdown "
            "is relatively broad-based."
        )
    elif improving_count > weakening_count:
        statements.append(
            f"{improving_count} of {len(FEATURE_COLUMNS)} indicators "
            "improved over the past three months, indicating broadening "
            "economic resilience."
        )
    else:
        statements.append(
            "Indicator breadth is mixed, with no clear majority of "
            "components improving or weakening."
        )

    (
        largest_drag,
        largest_drag_change,
    ) = _largest_momentum_driver(
        snapshot,
        improving=False,
    )

    (
        largest_support,
        largest_support_change,
    ) = _largest_momentum_driver(
        snapshot,
        improving=True,
    )

    if (
        pd.notna(largest_drag_change)
        and largest_drag_change < -0.10
    ):
        driver_statement = (
            f"{largest_drag} recorded the largest deterioration "
            f"({largest_drag_change:+.2f} standard deviations)"
        )

        if (
            pd.notna(largest_support_change)
            and largest_support_change > 0.10
        ):
            driver_statement += (
                f", while {largest_support} provided the strongest "
                f"offset ({largest_support_change:+.2f})"
            )

        statements.append(
            driver_statement
            + "."
        )

    elif (
        pd.notna(largest_support_change)
        and largest_support_change > 0.10
    ):
        statements.append(
            f"{largest_support} recorded the strongest improvement "
            f"({largest_support_change:+.2f} standard deviations)."
        )

    if pd.notna(percentile):
        if percentile >= 75:
            context = (
                "relatively strong compared with its available history"
            )
        elif percentile <= 25:
            context = (
                "relatively weak compared with its available history"
            )
        else:
            context = (
                "near the middle of its available historical distribution"
            )

        statements.append(
            f"The current composite sits in the {percentile:.0f}th "
            f"percentile and is {context}."
        )

    return " ".join(
        statements
    )


def _composite_figure(
    composite: pd.Series,
) -> go.Figure:
    """Create a chart of the composite growth-momentum index."""
    figure = go.Figure()

    clean = composite.dropna()

    figure.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean,
            mode="lines",
            name="Growth momentum index",
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.55,
        annotation_text="Historical norm",
        annotation_position="bottom right",
    )

    figure.add_hline(
        y=1,
        line_dash="dot",
        opacity=0.3,
    )

    figure.add_hline(
        y=-1,
        line_dash="dot",
        opacity=0.3,
    )

    if not clean.empty:
        latest_value = float(
            clean.iloc[-1]
        )

        figure.add_hline(
            y=latest_value,
            line_dash="dot",
            opacity=0.5,
            annotation_text=(
                f"Current: {latest_value:+.2f}"
            ),
            annotation_position="top right",
        )

    figure.update_layout(
        title="Composite Growth Momentum Index",
        xaxis_title="Date",
        yaxis_title="Standardised score",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        height=420,
    )

    return figure


def _indicator_score_figure(
    indicator_scores: pd.DataFrame,
) -> go.Figure:
    """Create a chart of standardised component indicators."""
    figure = go.Figure()

    for column in FEATURE_COLUMNS:
        clean = indicator_scores[
            column
        ].dropna()

        figure.add_trace(
            go.Scatter(
                x=clean.index,
                y=clean,
                mode="lines",
                name=FEATURE_LABELS[
                    column
                ],
            )
        )

    figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.55,
    )

    figure.update_layout(
        title="Growth Indicator Scores",
        xaxis_title="Date",
        yaxis_title="Rolling z-score",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Indicator",
        height=440,
    )

    return figure


def _breadth_figure(
    snapshot: pd.DataFrame,
) -> go.Figure:
    """Create a chart of recent changes by indicator."""
    ordered = snapshot.sort_values(
        "3M change",
        ascending=False,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=ordered[
                "Indicator"
            ],
            y=ordered[
                "3M change"
            ],
            name="Three-month change",
            hovertemplate=(
                "%{x}<br>"
                "%{y:+.2f} standard deviations"
                "<extra></extra>"
            ),
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.6,
    )

    figure.update_layout(
        title="Three-Month Change by Growth Indicator",
        xaxis_title="Indicator",
        yaxis_title="Change in z-score",
        template="plotly_white",
        showlegend=False,
        height=380,
    )

    return figure


def _gdp_context_figure(
    composite: pd.Series,
    gdp_yoy: pd.Series,
) -> go.Figure:
    """
    Compare the growth-momentum index with released GDP.

    This is a contextual comparison, not a regression or GDP forecast.
    """
    figure = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    composite_clean = composite.dropna()
    gdp_clean = gdp_yoy.dropna()

    figure.add_trace(
        go.Scatter(
            x=composite_clean.index,
            y=composite_clean,
            mode="lines",
            name="Growth momentum index",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=gdp_clean.index,
            y=gdp_clean,
            mode="lines+markers",
            name="Real GDP YoY",
        ),
        secondary_y=True,
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.45,
        secondary_y=False,
    )

    figure.update_layout(
        title="Growth Momentum and Released Real GDP",
        xaxis_title="Date",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=440,
    )

    figure.update_yaxes(
        title_text="Growth momentum score",
        secondary_y=False,
    )

    figure.update_yaxes(
        title_text="Real GDP growth (% YoY)",
        secondary_y=True,
    )

    return figure


def render(
    fred_client,
    context: dict,
) -> None:
    st.subheader(
        "Panel 4: Growth Momentum Monitor"
    )

    st.caption(
        "The panel combines initial claims, industrial production, "
        "payroll growth and Philadelphia Fed activity into a standardised "
        "growth-momentum index. It describes the direction, breadth and "
        "drivers of activity rather than producing a point forecast for GDP."
    )

    display_start_date = context[
        "start_date"
    ]

    display_end_date = context[
        "end_date"
    ]

    fetch_start_date = min(
        ANALYSIS_START_DATE,
        display_start_date,
    )

    result = fred_client.get_series(
        GROWTH_SERIES,
        fetch_start_date,
        display_end_date,
    )

    if (
        not result.success
        or result.data is None
    ):
        st.warning(
            result.message
            or "Growth data are unavailable."
        )
        return

    data = (
        result.data
        .copy()
        .sort_index()
    )

    if data.empty:
        st.info(
            "No growth data are available."
        )
        return

    missing_series = [
        series_id
        for series_id in GROWTH_SERIES
        if series_id not in data.columns
    ]

    if missing_series:
        st.warning(
            "The following required FRED series are missing: "
            + ", ".join(
                missing_series
            )
        )
        return

    (
        raw_features,
        indicator_scores,
        gdp_yoy,
    ) = _prepare_growth_data(
        data
    )

    context.setdefault("panel_history", {})["growth"] = {
        "raw": data.copy(),
        "raw_features": raw_features.copy(),
        "indicator_scores": indicator_scores.copy(),
        "gdp_yoy": gdp_yoy.copy(),
    }

    composite = indicator_scores[
        "growth_momentum_index"
    ].dropna()

    if composite.empty:
        st.warning(
            "Insufficient indicator history to calculate the "
            "growth-momentum index."
        )
        return

    latest_date = (
        composite.index.max()
    )

    latest_score = _value_as_of(
        composite,
        latest_date,
    )

    change_1m = _change_over_months(
        composite,
        latest_date,
        1,
    )

    change_3m = _change_over_months(
        composite,
        latest_date,
        3,
    )

    change_6m = _change_over_months(
        composite,
        latest_date,
        6,
    )

    historical_percentile = (
        _percentile_rank(
            composite,
            latest_score,
        )
    )

    regime, regime_description = (
        _growth_regime(
            latest_score
        )
    )

    momentum = _momentum_direction(
        change_3m
    )

    snapshot = _build_indicator_snapshot(
        indicator_scores,
        latest_date,
    )

    (
        improving_count,
        weakening_count,
        stable_count,
    ) = _breadth_summary(
        snapshot
    )

    strongest_indicator, strongest_score = (
        _strongest_indicator(
            snapshot,
            strongest=True,
        )
    )

    weakest_indicator, weakest_score = (
        _strongest_indicator(
            snapshot,
            strongest=False,
        )
    )

    macro_note = _build_macro_note(
        latest_date,
        latest_score,
        regime,
        change_1m,
        change_3m,
        historical_percentile,
        snapshot,
    )

    display_start_timestamp = pd.Timestamp(
        display_start_date
    )

    display_end_timestamp = pd.Timestamp(
        display_end_date
    )

    displayed_scores = indicator_scores.loc[
        (
            indicator_scores.index
            >= display_start_timestamp
        )
        & (
            indicator_scores.index
            <= display_end_timestamp
        )
    ]

    displayed_composite = composite.loc[
        (
            composite.index
            >= display_start_timestamp
        )
        & (
            composite.index
            <= display_end_timestamp
        )
    ]

    displayed_gdp = gdp_yoy.loc[
        (
            gdp_yoy.index
            >= display_start_timestamp
        )
        & (
            gdp_yoy.index
            <= display_end_timestamp
        )
    ]

    if displayed_composite.empty:
        displayed_composite = (
            composite.tail(60)
        )

        displayed_scores = (
            indicator_scores.loc[
                displayed_composite.index.min():
                displayed_composite.index.max()
            ]
        )

        displayed_gdp = gdp_yoy.loc[
            displayed_composite.index.min():
            displayed_composite.index.max()
        ]

    # ---------------------------------------------------------
    # CURRENT ASSESSMENT
    # ---------------------------------------------------------

    st.markdown(
        "### Current growth assessment"
    )

    st.info(
        f"**{regime}.** "
        f"{regime_description}"
    )

    st.markdown(
        """
**Supporting evidence**
- Composite: {score}, {change_3m} over 3M
- Momentum: {momentum}, {change_1m} over 1M and {change_6m} over 6M
- Breadth: {improving} improving, {weakening} weakening, {stable} stable
- Strongest: {strongest} ({strongest_score})
- Weakest: {weakest} ({weakest_score})
        """.format(
            score=_format_change_or_unavailable(latest_score),
            change_3m=_format_change_or_unavailable(change_3m),
            momentum=momentum,
            change_1m=_format_change_or_unavailable(change_1m),
            change_6m=_format_change_or_unavailable(change_6m),
            improving=improving_count,
            weakening=weakening_count,
            stable=stable_count,
            strongest=strongest_indicator,
            strongest_score=(
                f"{_format_change_or_unavailable(strongest_score)} z-score"
            ),
            weakest=weakest_indicator,
            weakest_score=(
                f"{_format_change_or_unavailable(weakest_score)} z-score"
            ),
        )
    )

    # ---------------------------------------------------------
    # MACRO-NOTE OUTPUT
    # ---------------------------------------------------------

    st.markdown(
        "### Macro-note output"
    )

    st.code(
        macro_note,
        language=None,
        wrap_lines=True,
    )

    st.caption(
        "This paragraph is mechanically generated from indicator "
        "levels, momentum, breadth and historical context. Add release "
        "surprises and event context before using it as a final macro view."
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
            "Growth momentum index",
            f"{latest_score:+.2f}",
            (
                f"{change_3m:+.2f} over 3M"
                if pd.notna(
                    change_3m
                )
                else None
            ),
        )

        _render_metric_caption(
            "Average rolling z-score across the four growth indicators. "
            "Zero represents the indicators' recent historical norm."
        )

    with metric_2:
        st.metric(
            "Growth regime",
            regime,
        )

        _render_metric_caption(
            "Rule-based classification of the composite indicator, "
            "not a GDP forecast or recession probability."
        )

    with metric_3:
        st.metric(
            "Momentum",
            momentum,
            (
                f"{change_1m:+.2f} over 1M"
                if pd.notna(
                    change_1m
                )
                else None
            ),
        )

        _render_metric_caption(
            (
                f"Six-month change: "
                f"{_format_change(change_6m)}."
            )
        )

    with metric_4:
        st.metric(
            "Breadth",
            (
                f"{improving_count} improving / "
                f"{weakening_count} weakening"
            ),
        )

        _render_metric_caption(
            (
                f"{stable_count} indicators were broadly stable "
                "over the past three months."
            )
        )

    with metric_5:
        st.metric(
            "Strongest indicator",
            strongest_indicator,
            (
                f"{strongest_score:+.2f} z-score"
                if pd.notna(
                    strongest_score
                )
                else None
            ),
        )

        _render_metric_caption(
            "The component currently furthest above its own "
            "rolling historical norm."
        )

    with metric_6:
        st.metric(
            "Weakest indicator",
            weakest_indicator,
            (
                f"{weakest_score:+.2f} z-score"
                if pd.notna(
                    weakest_score
                )
                else None
            ),
            delta_color="inverse",
        )

        _render_metric_caption(
            "The component currently furthest below its own "
            "rolling historical norm."
        )

    st.caption(
        f"Latest complete indicator observation: "
        f"{latest_date:%d %b %Y}. "
        f"The composite is in the "
        f"{historical_percentile:.0f}th percentile of its "
        "available history."
        if pd.notna(
            historical_percentile
        )
        else (
            f"Latest complete indicator observation: "
            f"{latest_date:%d %b %Y}."
        )
    )

    # ---------------------------------------------------------
    # COMPOSITE MOMENTUM
    # ---------------------------------------------------------

    st.markdown(
        "### Composite growth momentum"
    )

    st.plotly_chart(
        _composite_figure(
            displayed_composite
        ),
        use_container_width=True,
    )

    st.caption(
        _percentile_description(
            historical_percentile
        )
    )

    # ---------------------------------------------------------
    # INDICATOR BREADTH AND DRIVERS
    # ---------------------------------------------------------

    st.markdown(
        "### Indicator breadth and recent drivers"
    )

    st.plotly_chart(
        _breadth_figure(
            snapshot
        ),
        use_container_width=True,
    )

    if weakening_count > improving_count:
        st.caption(
            f"{weakening_count} of {len(FEATURE_COLUMNS)} indicators "
            "weakened over the latest three-month period, suggesting "
            "that the loss of momentum is relatively broad-based."
        )
    elif improving_count > weakening_count:
        st.caption(
            f"{improving_count} of {len(FEATURE_COLUMNS)} indicators "
            "improved over the latest three-month period, indicating "
            "broadening growth resilience."
        )
    else:
        st.caption(
            "Recent indicator breadth is mixed, with no clear majority "
            "of components improving or weakening."
        )

    # ---------------------------------------------------------
    # COMPONENT SCORES
    # ---------------------------------------------------------

    st.markdown(
        "### Growth indicators relative to history"
    )

    st.plotly_chart(
        _indicator_score_figure(
            displayed_scores
        ),
        use_container_width=True,
    )

    st.caption(
        "Each series is standardised relative to its own rolling "
        f"{ZSCORE_WINDOW_MONTHS}-month history. Higher scores consistently "
        "indicate stronger activity because jobless-claims growth is inverted."
    )

    # ---------------------------------------------------------
    # GDP CONTEXT
    # ---------------------------------------------------------

    st.markdown(
        "### Growth momentum and released GDP"
    )

    st.plotly_chart(
        _gdp_context_figure(
            displayed_composite,
            displayed_gdp,
        ),
        use_container_width=True,
    )

    st.caption(
        "Released real GDP is shown for economic context only. The growth "
        "momentum index is not fitted to GDP and should not be interpreted "
        "as a point forecast."
    )

    # ---------------------------------------------------------
    # METHODOLOGY
    # ---------------------------------------------------------

    with st.expander(
        "View methodology and interpretation notes",
        expanded=False,
    ):
        st.markdown(
            f"""
**Indicator construction**

- **Initial claims:** weekly claims are averaged monthly and converted into
  annualised three-month growth. The sign is inverted so that higher values
  indicate stronger labour-market conditions.
- **Industrial production:** annualised three-month growth in the production
  index.
- **Payroll growth:** annualised three-month growth in nonfarm payroll
  employment.
- **Philadelphia Fed activity:** monthly level of the diffusion index.

**Composite index**

Each transformed indicator is converted into a rolling z-score using up to
{ZSCORE_WINDOW_MONTHS} months of history, with at least
{ZSCORE_MIN_PERIODS} observations required. The composite growth-momentum
index is the equal-weighted average of the available indicator scores.

- A score above zero means the indicator set is stronger than its recent norm.
- A score below zero means the indicator set is weaker than its recent norm.
- A rising score indicates improving momentum.
- A falling score indicates deteriorating momentum.

**Breadth**

Breadth counts how many indicators improved, weakened or remained broadly
stable over the latest {MOMENTUM_MONTHS}-month period. It helps distinguish a
broad economic shift from movement concentrated in a single series.

**Limitations**

This is a descriptive growth monitor rather than a formal GDP forecast or
recession model. FRED data may be revised, and the panel does not account for
historical release vintages. The equal weights and regime thresholds are
transparent analytical choices rather than statistically estimated parameters.
The indicator set also excludes several important areas, including consumption,
housing, business investment, financial conditions and trade.
            """
        )

        st.markdown(
            "#### Component descriptions"
        )

        for feature in FEATURE_COLUMNS:
            st.markdown(
                f"**{FEATURE_LABELS[feature]}:** "
                f"{FEATURE_DESCRIPTIONS[feature]}"
            )
