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
    NFCI,
    NFCICREDIT,
    NFCILEVERAGE,
    NFCIRISK,
    SP500,
    VIXCLS,
)


ANALYSIS_START_DATE = date(2000, 1, 1)

COMPARISON_OFFSETS = {
    "1D": pd.DateOffset(days=1),
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}

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
    """Return the change from a comparison horizon to the latest date."""
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
        "6M": "over the past six months",
        "1Y": "over the past year",
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

    if pd.isna(base_value):
        base_value = float(clean.iloc[0])

    if base_value == 0:
        return pd.Series(
            dtype="float64",
        )

    return (
        clean / base_value
    ) * 100.0


def _nfci_reading(level: float, change: float, comparison_label: str) -> tuple[str, str]:
    """Interpret the official NFCI using the Chicago Fed's zero threshold."""
    if pd.isna(level):
        return "Unavailable", "The latest NFCI observation is unavailable."

    if level > 0:
        label = "Tighter than average"
    elif level < 0:
        label = "Looser than average"
    else:
        label = "Average financial conditions"

    if pd.isna(change):
        movement = "A comparison with the selected horizon is unavailable."
    elif change > 0:
        movement = f"Conditions tightened by {abs(change):.2f} index points {comparison_label}."
    elif change < 0:
        movement = f"Conditions loosened by {abs(change):.2f} index points {comparison_label}."
    else:
        movement = f"Conditions were unchanged {comparison_label}."

    return label, f"The NFCI is {level:+.2f}. {movement}"


def _nfci_figure(series: pd.Series) -> go.Figure:
    """Plot the official weekly Chicago Fed NFCI."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean,
            mode="lines",
            name="NFCI",
            line={"color": "#3157d5", "width": 2.5},
        )
    )
    figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="Historical mean = 0",
        annotation_position="top left",
    )
    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        height=410,
        yaxis_title="Index",
    )
    return figure


def _nfci_contribution_figure(values: dict[str, float]) -> go.Figure:
    """Show the latest official NFCI subindex contributions."""
    labels = list(values)
    contributions = [values[label] for label in labels]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=contributions,
            marker_color=["#c2410c" if value > 0 else "#0f766e" for value in contributions],
            text=[f"{value:+.2f}" for value in contributions],
            textposition="outside",
            hovertemplate="%{x}: %{y:+.3f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color="#64748b", line_width=1)
    figure.update_layout(
        title="Official NFCI subindexes",
        template="plotly_white",
        showlegend=False,
        height=410,
        yaxis_title="Contribution to financial conditions",
        margin={"l": 50, "r": 20, "t": 55, "b": 45},
    )
    return figure


def _credit_figure(
    hy_series: pd.Series,
    ig_series: pd.Series,
) -> go.Figure:
    """Create a chart showing corporate credit spreads."""
    hy_basis_points = pd.to_numeric(hy_series, errors="coerce") * 100.0
    ig_basis_points = pd.to_numeric(ig_series, errors="coerce") * 100.0
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=hy_basis_points.index,
            y=hy_basis_points,
            mode="lines",
            name="High-yield OAS",
            hovertemplate="HY OAS: %{y:.0f} bp<extra></extra>",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=ig_basis_points.index,
            y=ig_basis_points,
            mode="lines",
            name="Investment-grade OAS",
            hovertemplate="IG OAS: %{y:.0f} bp<extra></extra>",
        )
    )

    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        height=360,
        yaxis_title="Basis points",
        margin={"l": 45, "r": 15, "t": 45, "b": 40},
    )

    return figure


def _equity_figure(
    sp500_series: pd.Series,
    vix_series: pd.Series,
    base_date: pd.Timestamp,
) -> go.Figure:
    """Create a chart showing equity direction and expected volatility."""
    rebased_sp500 = _rebased_series(sp500_series, base_date)
    clean_vix = pd.to_numeric(vix_series, errors="coerce")
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=rebased_sp500.index,
            y=rebased_sp500,
            mode="lines",
            name="S&P 500 · rebased",
            line={"color": "#3157d5", "width": 2.2},
            hovertemplate="S&P 500: %{y:.1f} (start = 100)<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=clean_vix.index,
            y=clean_vix,
            mode="lines",
            name="VIX",
            line={"color": "#c2410c", "width": 1.8},
            hovertemplate="VIX: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_hline(
        y=100,
        line_dash="dash",
        opacity=0.5,
        annotation_text="Starting level",
        annotation_position="bottom left",
        secondary_y=False,
    )
    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        height=360,
        margin={"l": 45, "r": 45, "t": 45, "b": 40},
    )
    figure.update_yaxes(title_text="S&P 500 · rebased", secondary_y=False)
    figure.update_yaxes(title_text="VIX", secondary_y=True)
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
        xaxis_title="Date",
        yaxis_title="Rebased level",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        height=360,
        margin={"l": 45, "r": 15, "t": 25, "b": 40},
    )

    return figure


def render(
    fred_client,
    context: dict,
) -> None:
    st.subheader("Cross-Asset Context")

    st.caption(
        "Use the Chicago Fed NFCI for the recognized financial-conditions signal, then use credit "
        "spreads, equity performance, volatility and the dollar to understand the current market backdrop."
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

    context.setdefault("panel_history", {})["cross_asset"] = data.copy()

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

    header_left, header_right = st.columns([2, 3])
    with header_left:
        st.caption(f"Market data as of {latest_date:%d %b %Y} · series update on different schedules")
    with header_right:
        comparison_type = st.segmented_control(
            "Comparison horizon",
            options=list(COMPARISON_OFFSETS),
            default="1M",
            required=True,
            key="cross_asset_comparison",
            label_visibility="collapsed",
            width="stretch",
        )
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

    sp500_level = _value_as_of(
        data[SP500],
        latest_date,
    )

    hy_change_bp = _change_over_period(
        data[BAMLH0A0HYM2],
        latest_date,
        comparison_date,
    ) * 100.0

    ig_change_bp = _change_over_period(
        data[BAMLC0A0CM],
        latest_date,
        comparison_date,
    ) * 100.0

    if pd.notna(hy_level):
        hy_level *= 100.0
    if pd.notna(ig_level):
        ig_level *= 100.0

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

    comparison_sp500 = _value_as_of(data[SP500], comparison_date)
    if pd.notna(sp500_level) and pd.notna(comparison_sp500) and comparison_sp500 != 0:
        sp500_change_pct = ((sp500_level / comparison_sp500) - 1.0) * 100.0
    else:
        sp500_change_pct = float("nan")

    hy_percentile = _percentile_rank(
        data[BAMLH0A0HYM2] * 100.0,
        hy_level,
    )

    ig_percentile = _percentile_rank(
        data[BAMLC0A0CM] * 100.0,
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

    nfci_series = pd.to_numeric(data[NFCI], errors="coerce").dropna()
    nfci_date = nfci_series.index.max()
    nfci_level = _value_as_of(nfci_series, nfci_date)
    nfci_comparison_date = nfci_date - COMPARISON_OFFSETS[comparison_type]
    nfci_change = _change_over_period(nfci_series, nfci_date, nfci_comparison_date)
    nfci_label, nfci_description = _nfci_reading(
        nfci_level,
        nfci_change,
        comparison_label,
    )
    nfci_subindexes = {
        "Risk": _value_as_of(data[NFCIRISK], nfci_date),
        "Credit": _value_as_of(data[NFCICREDIT], nfci_date),
        "Leverage": _value_as_of(data[NFCILEVERAGE], nfci_date),
    }

    st.html(
        """
        <style>
            .cross-nfci-heading {
                display: inline-flex; align-items: center; gap: .5rem;
                margin: 1.85rem 0 .7rem;
            }
            .cross-nfci-title {
                color: #14213d; font-size: 1.18rem; font-weight: 650;
                letter-spacing: -.025em; line-height: 1.25;
            }
            .cross-nfci-info {
                position: relative; display: inline-flex; align-items: center;
                justify-content: center; width: 1.15rem; height: 1.15rem;
                border: 1px solid #94a3b8; border-radius: 50%; color: #64748b;
                font-size: .72rem; font-weight: 750; cursor: help;
            }
            .cross-nfci-tooltip {
                position: absolute; z-index: 30; top: 1.55rem; left: 50%;
                width: min(34rem, 82vw); padding: .85rem .95rem;
                border: 1px solid #dfe5ee; border-radius: .65rem;
                background: #fff; color: #334155;
                box-shadow: 0 10px 30px rgba(30,47,78,.16);
                font-size: .76rem; font-weight: 400; line-height: 1.48;
                box-sizing: border-box; white-space: normal; overflow-wrap: anywhere;
                opacity: 0; visibility: hidden; transform: translate(-12%, -.25rem);
                transition: opacity .12s ease, transform .12s ease;
            }
            .cross-nfci-info:hover .cross-nfci-tooltip,
            .cross-nfci-info:focus .cross-nfci-tooltip {
                opacity: 1; visibility: visible; transform: translate(-12%, 0);
            }
        </style>
        <div class="cross-nfci-heading">
            <span class="cross-nfci-title">Chicago Fed NFCI</span>
            <span class="cross-nfci-info" tabindex="0" aria-label="Explain the Chicago Fed NFCI">
                i
                <span class="cross-nfci-tooltip" role="tooltip">
                    The NFCI combines 105 measures from money, debt, equity and banking markets. Its official Risk, Credit and Leverage subindexes show which broad areas are tightening or loosening conditions.
                </span>
            </span>
        </div>
        """
    )
    st.info(f"**{nfci_label}.** {nfci_description}")
    st.caption(
        f"Official weekly reading as of {nfci_date:%d %b %Y}. The index is standardized to a historical "
        "mean of zero over its sample beginning in 1971: positive values are tighter than that mean and "
        "negative values are looser. Zero is not the median or a policy-neutral threshold."
    )

    nfci_column, contribution_column = st.columns([1.65, 1.0])
    with nfci_column:
        st.plotly_chart(
            _nfci_figure(data[NFCI].loc[display_start_date:display_end_date]),
            width="stretch",
        )
    with contribution_column:
        st.plotly_chart(_nfci_contribution_figure(nfci_subindexes), width="stretch")
    st.markdown("### Market diagnostics")
    st.caption("Grouped by the market channel each indicator helps diagnose.")

    credit_group, equity_group, fx_group = st.columns([1.15, 1.15, 0.8])

    with credit_group:
        with st.container(border=True):
            st.markdown("#### Credit")
            st.caption("Corporate risk appetite and funding stress")
            hy_column, ig_column = st.columns(2)
            with hy_column:
                st.metric(
                    "HY OAS",
                    f"{hy_level:.0f} bp" if pd.notna(hy_level) else "Unavailable",
                    _format_change_bp(hy_change_bp, comparison_label),
                    delta_color="inverse",
                )
                st.caption(_percentile_description(hy_percentile, "history"))
            with ig_column:
                st.metric(
                    "IG OAS",
                    f"{ig_level:.0f} bp" if pd.notna(ig_level) else "Unavailable",
                    _format_change_bp(ig_change_bp, comparison_label),
                    delta_color="inverse",
                )
                st.caption(_percentile_description(ig_percentile, "history"))

    with equity_group:
        with st.container(border=True):
            st.markdown("#### Equities")
            st.caption("Risk direction and expected uncertainty")
            sp500_column, vix_column = st.columns(2)
            with sp500_column:
                st.metric(
                    "S&P 500",
                    f"{sp500_level:,.0f}" if pd.notna(sp500_level) else "Unavailable",
                    _format_change_pct(sp500_change_pct, comparison_label),
                )
                st.caption("Higher prices generally indicate stronger risk appetite.")
            with vix_column:
                st.metric(
                    "VIX",
                    f"{vix_level:.1f}" if pd.notna(vix_level) else "Unavailable",
                    _format_change(vix_change, comparison_label),
                    delta_color="inverse",
                )
                st.caption(_percentile_description(vix_percentile, "history"))

    with fx_group:
        with st.container(border=True):
            st.markdown("#### FX")
            st.caption("Relative policy, growth and safe-haven demand")
            st.metric(
                "Broad dollar",
                f"{dxy_level:.1f}" if pd.notna(dxy_level) else "Unavailable",
                _format_change_pct(dxy_change_pct, comparison_label),
            )
            st.caption(_percentile_description(dxy_percentile, "history"))

    st.caption(
        "Credit, VIX and dollar percentiles use the fetched history from "
        f"{pd.Timestamp(fetch_start_date):%d %b %Y} to "
        f"{pd.Timestamp(display_end_date):%d %b %Y}."
    )

    interpretation_table = pd.DataFrame(
        [
            {
                "Signal group": "Credit spreads",
                "What it contributes": "Corporate risk appetite and funding stress",
                "Risk-on": "HY and IG OAS tighten",
                "Risk-off": "HY and IG OAS widen",
            },
            {
                "Signal group": "Equities",
                "What it contributes": "Risk appetite, uncertainty and demand for protection",
                "Risk-on": "S&P 500 rises and VIX falls",
                "Risk-off": "S&P 500 falls and VIX rises",
            },
            {
                "Signal group": "US dollar",
                "What it contributes": "Policy, global growth and safe-haven context",
                "Risk-on": "Dollar softens",
                "Risk-off": "Dollar strengthens",
            },
        ]
    )
    st.dataframe(
        interpretation_table,
        hide_index=True,
        width="stretch",
        height=143,
    )
    st.caption(
        "The dollar is supporting context, not a standalone risk signal: it can strengthen because "
        "of US policy divergence or relative growth as well as safe-haven demand."
    )

    st.markdown("### Market Context")
    credit_column, equity_column, dollar_column = st.columns(3)

    credit_figure = _credit_figure(
        data[BAMLH0A0HYM2].loc[display_start_date:display_end_date],
        data[BAMLC0A0CM].loc[display_start_date:display_end_date],
    )
    equity_figure = _equity_figure(
        data[SP500].loc[display_start_date:display_end_date],
        data[VIXCLS].loc[display_start_date:display_end_date],
        pd.Timestamp(display_start_date),
    )

    dollar_figure = _dollar_figure(
        data[DTWEXBGS].loc[display_start_date:display_end_date],
        pd.Timestamp(display_start_date),
    )

    with credit_column:
        with st.container(border=True):
            st.markdown("#### Credit")
            st.caption("Are corporate bond investors demanding more compensation for risk?")
            st.plotly_chart(credit_figure, width="stretch")
            st.html(
                """
                <style>
                    .credit-read-table {
                        width: 100%; table-layout: fixed; border-collapse: separate;
                        border-spacing: 0; overflow: hidden; border: 1px solid #e2e8f0;
                        border-radius: .55rem; color: #334155; font-size: .72rem;
                    }
                    .credit-read-table th {
                        padding: .48rem .42rem; background: #f8fafc; color: #475569;
                        font-size: .65rem; font-weight: 700; text-align: left;
                        text-transform: uppercase; letter-spacing: .035em;
                    }
                    .credit-read-table td {
                        padding: .55rem .42rem; border-top: 1px solid #e2e8f0;
                        vertical-align: top; line-height: 1.35; overflow-wrap: anywhere;
                    }
                    .credit-read-table th:nth-child(1),
                    .credit-read-table td:nth-child(1) { width: 26%; }
                    .credit-read-table th:nth-child(2),
                    .credit-read-table td:nth-child(2) { width: 24%; }
                    .credit-read-table th:nth-child(3),
                    .credit-read-table td:nth-child(3) { width: 50%; }
                    .credit-read-table strong { color: #14213d; }
                </style>
                <table class="credit-read-table">
                    <thead>
                        <tr><th>Market state</th><th>Spread condition</th><th>Economic implication</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><strong>Risk-on / calm</strong></td><td>Narrow / tight</td><td>High confidence and stable corporate earnings expectations</td></tr>
                        <tr><td><strong>Stress / late-cycle</strong></td><td>Widening / blowout</td><td>Rising fear, potential liquidations and recession risk</td></tr>
                    </tbody>
                </table>
                """
            )

    with equity_column:
        with st.container(border=True):
            st.markdown("#### Equities")
            st.caption("Are stock prices and expected volatility telling the same risk story?")
            st.plotly_chart(equity_figure, width="stretch")
            st.markdown(
                """
- **Stocks up + VIX down:** Risk-on
- **Stocks down + VIX up:** Risk-off / defensive
                """
            )

    with dollar_column:
        with st.container(border=True):
            st.markdown("#### FX (USD)")
            st.caption("Is the dollar adding to or easing global financial pressure?")
            st.plotly_chart(dollar_figure, width="stretch")
            st.markdown(
                """
- **Dollar down / weaker:** Usually risk-on or easier global conditions
- **Dollar up / stronger:** Often risk-off or tighter global conditions
                """
            )
            st.caption(
                "Policy divergence or stronger relative US growth can also move the dollar, so "
                "the interpretation should be confirmed with credit and equities."
            )
