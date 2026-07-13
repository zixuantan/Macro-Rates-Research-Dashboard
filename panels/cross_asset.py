from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    BAMLH0A0HYM2,
    BAMLC0A0CM,
    CROSS_ASSET_SERIES,
    DTWEXBGS,
    VIXCLS,
)


ANALYSIS_START_DATE = date(2000, 1, 1)

COMPARISON_OFFSETS = {
    "1D": pd.DateOffset(days=1),
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
}

CREDIT_WIDEN_THRESHOLD_BP = 5.0
CREDIT_TIGHTEN_THRESHOLD_BP = -5.0
VIX_RISE_THRESHOLD = 1.0
VIX_FALL_THRESHOLD = -1.0
DOLLAR_RISE_THRESHOLD_PCT = 0.5
DOLLAR_FALL_THRESHOLD_PCT = -0.5

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


def _change_over_period(
    series: pd.Series,
    latest_date: pd.Timestamp,
    comparison_date: pd.Timestamp,
) -> float:
    """Return the change from a comparison date to the latest date."""
    latest_value = _value_as_of(
        series,
        latest_date,
    )

    comparison_value = _value_as_of(
        series,
        comparison_date,
    )

    if (
        pd.isna(latest_value)
        or pd.isna(comparison_value)
    ):
        return float("nan")

    return (
        latest_value
        - comparison_value
    )


def _comparison_label(
    comparison_type: str,
    comparison_date: pd.Timestamp,
) -> str:
    """Return a readable description of the comparison horizon."""
    period_labels = {
        "1D": "over the past day",
        "1W": "over the past week",
        "1M": "over the past month",
        "3M": "over the past three months",
    }

    if comparison_type in period_labels:
        return period_labels[comparison_type]

    return (
        f"since {comparison_date:%d %b %Y}"
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

    if clean.empty or pd.isna(value):
        return float("nan")

    return float(
        (clean <= value).mean() * 100.0
    )


def _percentile_description(
    percentile: float,
    window_text: str,
) -> str:
    """Convert a percentile into concise historical context."""
    if pd.isna(percentile):
        return (
            f"Historical position unavailable for {window_text}."
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
        f"{percentile:.0f}th percentile; {position} "
        f"for {window_text}."
    )


def _format_change_bp(
    change: float,
    comparison_label: str,
) -> str | None:
    """Format a basis-point change for a metric."""
    if pd.isna(change):
        return None

    return (
        f"{change:+.0f} bp ({comparison_label})"
    )


def _format_change_pct(
    change: float,
    comparison_label: str,
) -> str | None:
    """Format a percentage change for a metric."""
    if pd.isna(change):
        return None

    return (
        f"{change:+.1f}% ({comparison_label})"
    )


def _format_change(
    change: float,
    comparison_label: str,
    digits: int = 1,
    suffix: str = "",
) -> str | None:
    """Format a signed change with a configurable number of decimals."""
    if pd.isna(change):
        return None

    return (
        f"{change:+.{digits}f}{suffix} ({comparison_label})"
    )


def _directional_change_phrase(
    change: float,
    positive_phrase: str,
    negative_phrase: str,
    unit: str,
    digits: int = 0,
) -> str:
    """Return a directional phrase with explicit units."""
    if pd.isna(change):
        return "change unavailable"

    if change > 0:
        value = f"{abs(change):.{digits}f}"
        return (
            f"{positive_phrase} {value}%"
            if unit == "%"
            else f"{positive_phrase} {value} {unit}"
        )

    if change < 0:
        value = f"{abs(change):.{digits}f}"
        return (
            f"{negative_phrase} {value}%"
            if unit == "%"
            else f"{negative_phrase} {value} {unit}"
        )

    return (
        "unchanged %"
        if unit == "%"
        else f"unchanged {unit}"
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


def _rebased_series(
    series: pd.Series,
    base_date: pd.Timestamp,
) -> pd.Series:
    """Rebase a series to 100 at a given date."""
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return pd.Series(
            dtype="float64",
        )

    base_value = _value_as_of(
        clean,
        base_date,
    )

    if pd.isna(base_value) or base_value == 0:
        return pd.Series(
            dtype="float64",
        )

    return (
        clean / base_value
    ) * 100.0


def _regime_vote_score(
    hy_change_bp: float,
    ig_change_bp: float,
    vix_change: float,
    dxy_change_pct: float,
) -> tuple[int, int]:
    """Count risk-on and risk-off votes from the available signals."""
    risk_on_votes = 0
    risk_off_votes = 0

    if pd.notna(hy_change_bp):
        if hy_change_bp <= CREDIT_TIGHTEN_THRESHOLD_BP:
            risk_on_votes += 1
        elif hy_change_bp >= CREDIT_WIDEN_THRESHOLD_BP:
            risk_off_votes += 1

    if pd.notna(ig_change_bp):
        if ig_change_bp <= CREDIT_TIGHTEN_THRESHOLD_BP:
            risk_on_votes += 1
        elif ig_change_bp >= CREDIT_WIDEN_THRESHOLD_BP:
            risk_off_votes += 1

    if pd.notna(vix_change):
        if vix_change <= VIX_FALL_THRESHOLD:
            risk_on_votes += 1
        elif vix_change >= VIX_RISE_THRESHOLD:
            risk_off_votes += 1

    if pd.notna(dxy_change_pct):
        if dxy_change_pct <= DOLLAR_FALL_THRESHOLD_PCT:
            risk_on_votes += 1
        elif dxy_change_pct >= DOLLAR_RISE_THRESHOLD_PCT:
            risk_off_votes += 1

    return (
        risk_on_votes,
        risk_off_votes,
    )


def _classify_regime(
    hy_change_bp: float,
    ig_change_bp: float,
    vix_change: float,
    dxy_change_pct: float,
) -> tuple[str, str]:
    """Classify cross-asset conditions using observable market moves."""
    if all(
        pd.isna(value)
        for value in [
            hy_change_bp,
            ig_change_bp,
            vix_change,
            dxy_change_pct,
        ]
    ):
        return (
            "Unavailable",
            "Insufficient cross-asset history to classify the market tone.",
        )

    credit_wider = any(
        [
            pd.notna(hy_change_bp)
            and hy_change_bp >= CREDIT_WIDEN_THRESHOLD_BP,
            pd.notna(ig_change_bp)
            and ig_change_bp >= CREDIT_WIDEN_THRESHOLD_BP,
        ]
    )

    credit_tighter = any(
        [
            pd.notna(hy_change_bp)
            and hy_change_bp <= CREDIT_TIGHTEN_THRESHOLD_BP,
            pd.notna(ig_change_bp)
            and ig_change_bp <= CREDIT_TIGHTEN_THRESHOLD_BP,
        ]
    )

    vol_higher = pd.notna(vix_change) and vix_change >= VIX_RISE_THRESHOLD
    vol_lower = pd.notna(vix_change) and vix_change <= VIX_FALL_THRESHOLD
    dollar_stronger = (
        pd.notna(dxy_change_pct)
        and dxy_change_pct >= DOLLAR_RISE_THRESHOLD_PCT
    )
    dollar_softer = (
        pd.notna(dxy_change_pct)
        and dxy_change_pct <= DOLLAR_FALL_THRESHOLD_PCT
    )

    risk_on_votes, risk_off_votes = _regime_vote_score(
        hy_change_bp,
        ig_change_bp,
        vix_change,
        dxy_change_pct,
    )

    if credit_tighter and vol_lower and dollar_softer:
        return (
            "Benign easing",
            (
                "Credit spreads are tighter, volatility is calmer and the "
                "dollar is softer, which is consistent with easier "
                "financial conditions."
            ),
        )

    if credit_wider and vol_higher and dollar_stronger:
        return (
            "Inflationary tightening",
            (
                "Credit spreads and volatility have both moved higher while "
                "the dollar is firmer, which is consistent with tighter "
                "financial conditions."
            ),
        )

    if credit_wider and vol_higher:
        return (
            "Growth scare",
            (
                "Credit spreads widened and volatility rose, which is more "
                "consistent with a growth scare than benign easing."
            ),
        )

    if risk_on_votes >= 3 and risk_off_votes <= 1:
        return (
            "Broad risk-on",
            (
                "Credit spreads are contained, volatility is calmer and the "
                "dollar is not showing acute stress."
            ),
        )

    if risk_off_votes >= 3 and risk_on_votes <= 1:
        return (
            "Broad risk-off",
            (
                "Credit spreads are wider, volatility is firmer and the "
                "dollar is stronger, which points to defensive positioning."
            ),
        )

    return (
        "Mixed cross-asset signals",
        (
            "The available cross-asset signals are not aligned closely "
            "enough to support a single clean market regime."
        ),
    )


def _build_macro_note(
    latest_date: pd.Timestamp,
    regime: str,
    regime_description: str,
    comparison_label: str,
    hy_level: float,
    ig_level: float,
    vix_level: float,
    dxy_level: float,
    hy_change_bp: float,
    ig_change_bp: float,
    vix_change: float,
    dxy_change_pct: float,
    hy_percentile: float,
    vix_percentile: float,
) -> str:
    """Generate a concise cross-asset paragraph for a macro note."""
    statements: list[str] = []

    statements.append(
        f"As of {latest_date:%d %b %Y}, cross-asset signals are "
        f"classified as {regime.lower()}."
    )

    move_parts = []

    if pd.notna(hy_change_bp):
        move_parts.append(
            f"high-yield spreads {('widened' if hy_change_bp > 0 else 'tightened')} "
            f"by {abs(hy_change_bp):.0f} bp"
        )

    if pd.notna(ig_change_bp):
        move_parts.append(
            f"investment-grade spreads {('widened' if ig_change_bp > 0 else 'tightened')} "
            f"by {abs(ig_change_bp):.0f} bp"
        )

    if pd.notna(vix_change):
        move_parts.append(
            f"VIX {('rose' if vix_change > 0 else 'fell')} "
            f"{abs(vix_change):.1f} points"
        )

    if pd.notna(dxy_change_pct):
        move_parts.append(
            f"the dollar index {('strengthened' if dxy_change_pct > 0 else 'weakened')} "
            f"by {abs(dxy_change_pct):.1f}%"
        )

    if move_parts:
        statements.append(
            "Over the selected horizon, "
            + ", ".join(move_parts[:-1])
            + (
                f", and {move_parts[-1]}"
                if len(move_parts) > 1
                else move_parts[-1]
            )
            + "."
        )

    statements.append(
        regime_description
    )

    if pd.notna(hy_percentile):
        statements.append(
            f"High-yield spreads are in the {hy_percentile:.0f}th "
            "percentile of the available history."
        )

    if pd.notna(vix_percentile):
        statements.append(
            f"VIX is in the {vix_percentile:.0f}th percentile of the "
            "available history."
        )

    return " ".join(statements)


def _credit_volatility_figure(
    hy_series: pd.Series,
    ig_series: pd.Series,
    vix_series: pd.Series,
) -> go.Figure:
    """Create a chart showing credit spreads and volatility."""
    figure = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    figure.add_trace(
        go.Scatter(
            x=hy_series.index,
            y=hy_series,
            mode="lines",
            name="High-yield OAS",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=ig_series.index,
            y=ig_series,
            mode="lines",
            name="Investment-grade OAS",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=vix_series.index,
            y=vix_series,
            mode="lines",
            name="VIX",
        ),
        secondary_y=True,
    )

    figure.update_layout(
        title="Credit Spreads and Volatility",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=430,
    )

    figure.update_yaxes(
        title_text="Basis points",
        secondary_y=False,
    )

    figure.update_yaxes(
        title_text="VIX",
        secondary_y=True,
    )

    return figure


def _dollar_figure(
    dxy_series: pd.Series,
    base_date: pd.Timestamp,
) -> go.Figure:
    """Create a rebased chart for the dollar index."""
    rebased = _rebased_series(
        dxy_series,
        base_date,
    )

    figure = go.Figure()

    if not rebased.empty:
        figure.add_trace(
            go.Scatter(
                x=rebased.index,
                y=rebased,
                mode="lines",
                name="Dollar index (rebased to 100)",
            )
        )

        figure.add_hline(
            y=100,
            line_dash="dash",
            opacity=0.5,
        )

    figure.update_layout(
        title="Dollar Index Performance",
        xaxis_title="Date",
        yaxis_title="Rebased level",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        height=360,
    )

    return figure


def render(
    fred_client,
    context: dict,
) -> None:
    st.subheader(
        "Panel 5: Cross-Asset Confirmation"
    )

    st.caption(
        "The panel checks whether credit, volatility and the dollar are "
        "confirming or contradicting the broader macro and rates narrative."
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
        CROSS_ASSET_SERIES,
        fetch_start_date,
        display_end_date,
    )

    if (
        not result.success
        or result.data is None
    ):
        st.warning(
            result.message
            or "Cross-asset data unavailable."
        )
        return

    data = (
        result.data
        .copy()
        .sort_index()
    )

    if data.empty:
        st.info(
            "No data available for the selected date range."
        )
        return

    missing_series = [
        series_id
        for series_id in CROSS_ASSET_SERIES
        if series_id not in data.columns
    ]

    if missing_series:
        st.warning(
            "The following cross-asset series are missing: "
            + ", ".join(
                missing_series
            )
        )
        return

    for series_id in CROSS_ASSET_SERIES:
        data[series_id] = pd.to_numeric(
            data[series_id],
            errors="coerce",
        )

    usable = data[
        CROSS_ASSET_SERIES
    ].dropna(
        how="all"
    )

    if usable.empty:
        st.warning(
            "No usable cross-asset observations were found."
        )
        return

    latest_date = usable.index.max()

    comparison_type = st.radio(
        "Compare the current cross-asset backdrop with",
        options=[
            "1D",
            "1W",
            "1M",
            "3M",
            "Custom",
        ],
        index=2,
        horizontal=True,
        key="cross_asset_comparison",
    )

    if comparison_type == "Custom":
        selected_comparison_date = st.date_input(
            "Custom comparison date",
            value=(
                latest_date
                - pd.DateOffset(months=1)
            ).date(),
            min_value=data.index.min().date(),
            max_value=latest_date.date(),
            key="cross_asset_custom_date",
        )

        requested_comparison_date = pd.Timestamp(
            selected_comparison_date
        )
    else:
        requested_comparison_date = (
            latest_date
            - COMPARISON_OFFSETS[comparison_type]
        )

    comparison_date = data.index[
        data.index <= requested_comparison_date
    ].max()

    if pd.isna(comparison_date):
        st.warning(
            "No comparison observation is available for the selected horizon."
        )
        return

    comparison_label = _comparison_label(
        comparison_type,
        comparison_date,
    )

    hy_level = _value_as_of(
        data[BAMLH0A0HYM2],
        latest_date,
    )

    ig_level = _value_as_of(
        data[BAMLC0A0CM],
        latest_date,
    )

    vix_level = _value_as_of(
        data[VIXCLS],
        latest_date,
    )

    dxy_level = _value_as_of(
        data[DTWEXBGS],
        latest_date,
    )

    hy_change_bp = _change_over_period(
        data[BAMLH0A0HYM2],
        latest_date,
        comparison_date,
    )

    ig_change_bp = _change_over_period(
        data[BAMLC0A0CM],
        latest_date,
        comparison_date,
    )

    vix_change = _change_over_period(
        data[VIXCLS],
        latest_date,
        comparison_date,
    )

    comparison_dxy = _value_as_of(
        data[DTWEXBGS],
        comparison_date,
    )

    if (
        pd.notna(dxy_level)
        and pd.notna(comparison_dxy)
        and comparison_dxy != 0
    ):
        dxy_change_pct = (
            (dxy_level / comparison_dxy) - 1.0
        ) * 100.0
    else:
        dxy_change_pct = float("nan")

    hy_percentile = _percentile_rank(
        data[BAMLH0A0HYM2],
        hy_level,
    )

    ig_percentile = _percentile_rank(
        data[BAMLC0A0CM],
        ig_level,
    )

    vix_percentile = _percentile_rank(
        data[VIXCLS],
        vix_level,
    )

    dxy_percentile = _percentile_rank(
        data[DTWEXBGS],
        dxy_level,
    )

    regime, regime_description = _classify_regime(
        hy_change_bp,
        ig_change_bp,
        vix_change,
        dxy_change_pct,
    )

    st.markdown(
        "### Current cross-asset assessment"
    )

    st.info(
        f"**{regime}.** {regime_description}"
    )

    st.markdown(
        "\n".join(
            [
                "**Supporting evidence**",
                (
                    f"- HY OAS: {hy_level:.0f} bp, "
                    f"{_directional_change_phrase(hy_change_bp, 'wider by', 'tighter by', 'bp')} "
                    f"{comparison_label}"
                    if pd.notna(hy_level)
                    else "- HY OAS: Unavailable"
                ),
                (
                    f"- IG OAS: {ig_level:.0f} bp, "
                    f"{_directional_change_phrase(ig_change_bp, 'wider by', 'tighter by', 'bp')} "
                    f"{comparison_label}"
                    if pd.notna(ig_level)
                    else "- IG OAS: Unavailable"
                ),
                (
                    f"- VIX: {vix_level:.1f}, "
                    f"{_directional_change_phrase(vix_change, 'up by', 'down by', 'points', digits=1)} "
                    f"{comparison_label}"
                    if pd.notna(vix_level)
                    else "- VIX: Unavailable"
                ),
                (
                    f"- Dollar index: {dxy_level:.1f}, "
                    f"{_directional_change_phrase(dxy_change_pct, 'up by', 'down by', '%', digits=1)} "
                    f"{comparison_label}"
                    if pd.notna(dxy_level)
                    else "- Dollar index: Unavailable"
                ),
            ]
        )
    )

    macro_note = _build_macro_note(
        latest_date,
        regime,
        regime_description,
        comparison_label,
        hy_level,
        ig_level,
        vix_level,
        dxy_level,
        hy_change_bp,
        ig_change_bp,
        vix_change,
        dxy_change_pct,
        hy_percentile,
        vix_percentile,
    )

    st.markdown(
        "### Macro-note output"
    )

    st.code(
        macro_note,
        language=None,
        wrap_lines=True,
    )

    st.caption(
        "This paragraph summarises cross-asset confirmation. It does not "
        "identify the underlying event or causal driver."
    )

    (
        metric_1,
        metric_2,
        metric_3,
        metric_4,
        metric_5,
    ) = st.columns(5)

    with metric_1:
        st.metric(
            "High-yield OAS",
            (
                f"{hy_level:.0f} bp"
                if pd.notna(hy_level)
                else "Unavailable"
            ),
            _format_change_bp(
                hy_change_bp,
                comparison_label,
            ),
            delta_color="inverse",
        )

        _render_metric_caption(
            (
                "Shows the compensation investors demand for lower-quality "
                "corporate credit. "
                f"{_percentile_description(hy_percentile, 'the available history')}"
            )
        )

    with metric_2:
        st.metric(
            "Investment-grade OAS",
            (
                f"{ig_level:.0f} bp"
                if pd.notna(ig_level)
                else "Unavailable"
            ),
            _format_change_bp(
                ig_change_bp,
                comparison_label,
            ),
            delta_color="inverse",
        )

        _render_metric_caption(
            (
                "Provides a lower-risk credit read-through for financial "
                "conditions. "
                f"{_percentile_description(ig_percentile, 'the available history')}"
            )
        )

    with metric_3:
        st.metric(
            "VIX",
            (
                f"{vix_level:.1f}"
                if pd.notna(vix_level)
                else "Unavailable"
            ),
            _format_change(
                vix_change,
                comparison_label,
            ),
        )

        _render_metric_caption(
            (
                "Measures expected equity-market volatility and risk "
                "aversion. "
                f"{_percentile_description(vix_percentile, 'the available history')}"
            )
        )

    with metric_4:
        st.metric(
            "Dollar index",
            (
                f"{dxy_level:.1f}"
                if pd.notna(dxy_level)
                else "Unavailable"
            ),
            _format_change_pct(
                dxy_change_pct,
                comparison_label,
            ),
        )

        _render_metric_caption(
            (
                "Provides context on relative policy expectations and "
                "global risk demand. "
                f"{_percentile_description(dxy_percentile, 'the available history')}"
            )
        )

    with metric_5:
        st.metric(
            "Cross-asset regime",
            regime,
        )

        _render_metric_caption(
            (
                "Mechanical classification of the available signals, not a "
                "forecast or causal explanation."
            )
        )

    st.caption(
        f"Current observation date: {latest_date:%d %b %Y}. "
        f"Percentiles use the fetched history from "
        f"{pd.Timestamp(fetch_start_date):%d %b %Y} to "
        f"{pd.Timestamp(display_end_date):%d %b %Y}."
    )

    # ---------------------------------------------------------
    # CREDIT AND VOLATILITY
    # ---------------------------------------------------------

    st.markdown(
        "### Credit and volatility"
    )

    credit_vol_figure = _credit_volatility_figure(
        data[BAMLH0A0HYM2].loc[display_start_date:display_end_date],
        data[BAMLC0A0CM].loc[display_start_date:display_end_date],
        data[VIXCLS].loc[display_start_date:display_end_date],
    )

    st.plotly_chart(
        credit_vol_figure,
        use_container_width=True,
    )

    st.caption(
        "Credit spreads and volatility are shown separately because the "
        "series occupy different units and respond differently to changing "
        "financial conditions."
    )

    # ---------------------------------------------------------
    # DOLLAR PERFORMANCE
    # ---------------------------------------------------------

    st.markdown(
        "### Dollar performance"
    )

    dollar_figure = _dollar_figure(
        data[DTWEXBGS].loc[display_start_date:display_end_date],
        pd.Timestamp(display_start_date),
    )

    st.plotly_chart(
        dollar_figure,
        use_container_width=True,
    )

    st.caption(
        "The dollar index is rebased to 100 at the start of the display "
        "range so that direction, not level, is the focus."
    )

    # ---------------------------------------------------------
    # CHANGE SUMMARY
    # ---------------------------------------------------------

    st.markdown(
        "### Selected-horizon changes"
    )

    change_columns = st.columns(4)

    with change_columns[0]:
        st.metric(
            "HY OAS change",
            (
                _format_change_bp(
                    hy_change_bp,
                    comparison_label,
                )
                or "Unavailable"
            ),
        )

    with change_columns[1]:
        st.metric(
            "IG OAS change",
            (
                _format_change_bp(
                    ig_change_bp,
                    comparison_label,
                )
                or "Unavailable"
            ),
        )

    with change_columns[2]:
        st.metric(
            "VIX change",
            (
                _format_change(
                    vix_change,
                    comparison_label,
                )
                or "Unavailable"
            ),
        )

    with change_columns[3]:
        st.metric(
            "Dollar change",
            (
                _format_change_pct(
                    dxy_change_pct,
                    comparison_label,
                )
                or "Unavailable"
            ),
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
**Series used**

- **High-yield OAS:** `{BAMLH0A0HYM2}`
- **Investment-grade OAS:** `{BAMLC0A0CM}`
- **Dollar index:** `{DTWEXBGS}`
- **VIX:** `{VIXCLS}`

**Comparison horizon**

The panel compares the latest observation with the selected horizon.
The default is one month, but one day, one week, one month, three months
and a custom date are available.

**Regime rules**

- **Broad risk-on:** credit spreads are tighter, volatility is calmer and
  the dollar is not showing acute stress.
- **Broad risk-off:** credit spreads are wider, volatility is firmer and
  the dollar is stronger.
- **Benign easing:** credit spreads are tighter or stable, volatility is
  calm and the dollar is softer.
- **Growth scare:** credit spreads are wider and volatility rises.
- **Inflationary tightening:** credit and volatility both worsen while the
  dollar firms.
- **Mixed cross-asset signals:** the available series do not point in one
  clear direction.

**Sign conventions**

- **Credit spreads:** higher means more stress; lower means easier
  financial conditions.
- **VIX:** higher means more risk aversion; lower means calmer markets.
- **Dollar index:** higher means a stronger dollar, but the macro
  interpretation depends on the rest of the signal set.

**Percentile context**

Percentiles are calculated against the fetched history from
{pd.Timestamp(fetch_start_date):%d %b %Y} to {pd.Timestamp(display_end_date):%d %b %Y}.
They are meant to support the note, not dominate it.

**Limitations**

Cross-asset agreement supports a narrative but does not prove it. Credit
spreads can lag macro deterioration, the dollar can rise for policy or
safe-haven reasons, and correlations can break down when markets are
driven by idiosyncratic factors.
            """
        )
