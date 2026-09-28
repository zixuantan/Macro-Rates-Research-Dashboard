from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.yield_curve import (
    CurveAnalysis,
    analyze_curve,
    curve_snapshot,
    numeric_yield_data,
    percentile_rank,
    spread_history,
    value_as_of,
)
from config import DGS10, DGS2, DGS30, DGS5, YIELD_SERIES


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
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}

PERIOD_TEXT = {
    "1D": "over the past day",
    "1W": "over the past week",
    "1M": "over the past month",
    "3M": "over the past three months",
    "6M": "over the past six months",
    "1Y": "over the past year",
}

CURRENT_COLOR = "#2563EB"
COMPARISON_COLOR = "#94A3B8"
POSITIVE_COLOR = "#C2410C"
NEGATIVE_COLOR = "#0F766E"


def _format_yield(value: float) -> str:
    return f"{value:.2f}%" if pd.notna(value) else "—"


def _format_bp(value: float, *, signed: bool = True) -> str:
    if pd.isna(value):
        return "—"
    sign = "+" if signed else ""
    return f"{value:{sign}.0f} bp"


def _format_delta(value: float, horizon: str) -> str | None:
    if pd.isna(value):
        return None
    return f"{value:+.1f} bp vs {horizon}"


def _render_html_table(dataframe: pd.DataFrame) -> None:
    """Render small tables without Streamlit's Arrow serialization path."""
    table_html = dataframe.to_html(
        index=False,
        na_rep="—",
        border=0,
        justify="left",
        escape=True,
        classes=["yield-data-table"],
    )
    st.markdown(
        f"""
        <div class="yield-table-wrap">
            <style>
                .yield-table-wrap {{ width: 100%; overflow-x: auto; margin: .4rem 0 1rem; }}
                table.yield-data-table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
                table.yield-data-table th {{ background: rgba(49,51,63,.05); font-weight: 600; }}
                table.yield-data-table th, table.yield-data-table td {{
                    padding: .55rem .7rem; text-align: right; white-space: nowrap;
                    border-bottom: 1px solid rgba(49,51,63,.12);
                }}
                table.yield-data-table th:first-child,
                table.yield-data-table td:first-child {{ text-align: left; }}
            </style>
            {table_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _move_driver(two_year_change: float, ten_year_change: float) -> str:
    if pd.isna(two_year_change) or pd.isna(ten_year_change):
        return "The main tenor-level driver is unavailable."
    if abs(two_year_change - ten_year_change) < 1:
        return "The 2Y and 10Y yields moved by similar amounts."
    if two_year_change >= 0 and ten_year_change >= 0:
        leader = "2Y" if two_year_change > ten_year_change else "10Y"
        effect = "flatter" if leader == "2Y" else "steeper"
        return f"The {leader} yield rose more, pushing 2s10s {effect}."
    if two_year_change <= 0 and ten_year_change <= 0:
        leader = "2Y" if abs(two_year_change) > abs(ten_year_change) else "10Y"
        effect = "steeper" if leader == "2Y" else "flatter"
        return f"The {leader} yield fell more, pushing 2s10s {effect}."
    if two_year_change > 0:
        return "The 2Y yield rose while the 10Y yield fell, creating a flattening twist."
    return "The 2Y yield fell while the 10Y yield rose, creating a steepening twist."


def _market_read(analysis: CurveAnalysis, horizon: str) -> str:
    two_change = analysis.changes_bp.get(DGS2, np.nan)
    ten_change = analysis.changes_bp.get(DGS10, np.nan)
    spread = analysis.current_spreads_bp.get("2s10s", np.nan)
    spread_change = analysis.spread_changes_bp.get("2s10s", np.nan)
    if pd.isna(spread):
        spread_summary = "The current 2s10s level is unavailable."
    else:
        shape = "upward sloping" if spread > 10 else "inverted" if spread < -10 else "near flat"
        spread_summary = f"The 2s10s curve is {shape} at {_format_bp(spread)}."
    if pd.isna(spread_change):
        direction = "could not be compared"
    else:
        direction = "steepened" if spread_change > 2 else "flattened" if spread_change < -2 else "was broadly stable"
    return (
        f"**{analysis.regime}.** {_move_driver(two_change, ten_change)} "
        f"{spread_summary} It {direction} "
        f"{PERIOD_TEXT[horizon]}."
    )


def _curve_figure(analysis: CurveAnalysis) -> go.Figure:
    figure = go.Figure()
    tenors = [TENOR_LABELS[series_id] for series_id in YIELD_SERIES]
    figure.add_trace(
        go.Scatter(
            x=tenors,
            y=analysis.comparison.values.reindex(YIELD_SERIES),
            mode="lines+markers",
            name=f"{analysis.comparison.date:%d %b %Y}",
            line={"color": COMPARISON_COLOR, "width": 2, "dash": "dot"},
            marker={"size": 7},
            connectgaps=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=tenors,
            y=analysis.current.values.reindex(YIELD_SERIES),
            mode="lines+markers",
            name=f"{analysis.current.date:%d %b %Y}",
            line={"color": CURRENT_COLOR, "width": 3},
            marker={"size": 8},
            connectgaps=False,
        )
    )
    figure.update_layout(
        title="Curve level",
        xaxis_title=None,
        yaxis_title="Yield (%)",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 45, "r": 15, "t": 75, "b": 40},
    )
    return figure


def _change_figure(analysis: CurveAnalysis, horizon: str) -> go.Figure:
    changes = analysis.changes_bp.reindex(YIELD_SERIES)
    colors = [
        POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR if value < 0 else COMPARISON_COLOR
        for value in changes
    ]
    figure = go.Figure(
        go.Bar(
            x=[TENOR_LABELS[series_id] for series_id in YIELD_SERIES],
            y=changes,
            marker_color=colors,
            text=[f"{value:+.0f}" if pd.notna(value) else "" for value in changes],
            textposition="outside",
            hovertemplate="%{x}: %{y:+.1f} bp<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color="#64748B", line_width=1)
    figure.update_layout(
        title=f"Change versus {horizon}",
        xaxis_title=None,
        yaxis_title="Change (bp)",
        template="plotly_white",
        showlegend=False,
        margin={"l": 45, "r": 15, "t": 75, "b": 40},
    )
    return figure


def _spread_figure(spread: pd.Series, title: str, current_value: float) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=spread.index,
            y=spread,
            mode="lines",
            line={"color": CURRENT_COLOR, "width": 2},
            hovertemplate="%{x|%d %b %Y}: %{y:.1f} bp<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1)
    if pd.notna(current_value):
        figure.add_hline(
            y=current_value,
            line_dash="dot",
            line_color="#475569",
            annotation_text=f"Current {current_value:+.0f} bp",
            annotation_position="top right",
        )
    figure.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title="Spread (bp)",
        template="plotly_white",
        showlegend=False,
        hovermode="x",
        margin={"l": 45, "r": 15, "t": 60, "b": 40},
    )
    return figure


def _yield_change_table(data: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
    current = curve_snapshot(data, current_date)
    rows = []
    table_periods = {
        "1D": COMPARISON_OFFSETS["1D"],
        "1W": COMPARISON_OFFSETS["1W"],
        "1M": COMPARISON_OFFSETS["1M"],
    }
    for series_id in YIELD_SERIES:
        row: dict[str, str | float] = {
            "Tenor": TENOR_LABELS[series_id],
            "Current yield (%)": round(current.values.get(series_id), 3) if current is not None else np.nan,
        }
        for label, offset in table_periods.items():
            comparison = curve_snapshot(data, current_date - offset)
            current_value = current.values.get(series_id) if current is not None else np.nan
            comparison_value = comparison.values.get(series_id) if comparison is not None else np.nan
            change = (current_value - comparison_value) * 100.0
            row[f"{label} change (bp)"] = round(change, 1) if pd.notna(change) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _spread_context_table(spreads: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for name in ("2s10s", "5s30s"):
        series = spreads[name].loc[lambda values: values.index <= latest_date]
        current = value_as_of(series, latest_date)
        one_month = value_as_of(series, latest_date - pd.DateOffset(months=1))
        one_month_change = current - one_month
        one_year = series.loc[series.index >= latest_date - pd.DateOffset(years=1)]
        five_year = series.loc[series.index >= latest_date - pd.DateOffset(years=5)]
        rows.append(
            {
                "Spread": name,
                "Current (bp)": round(current, 1),
                "1M change (bp)": round(one_month_change, 1),
                "1Y percentile": round(percentile_rank(one_year, current), 1),
                "5Y percentile": round(percentile_rank(five_year, current), 1),
            }
        )
    return pd.DataFrame(rows)


def render(fred_client, context: dict) -> None:
    st.subheader("Treasury yield curve")

    display_start = context["start_date"]
    display_end = context["end_date"]
    history_start = min(
        display_start,
        (pd.Timestamp(display_end) - pd.DateOffset(years=HISTORY_YEARS)).date(),
    )
    result = fred_client.get_series(YIELD_SERIES, history_start, display_end)
    if not result.success or result.data is None:
        st.warning(result.message or "Treasury yield data are unavailable.")
        return

    data = numeric_yield_data(result.data)
    if data.dropna(how="all").empty:
        st.info("No Treasury yield data are available.")
        return
    context.setdefault("panel_history", {})["yield_curve"] = data.copy()

    current = curve_snapshot(data, pd.Timestamp(display_end))
    if current is None:
        st.warning("The latest Treasury curve could not be constructed.")
        return

    header_left, header_right = st.columns([2, 3])
    with header_left:
        status = "complete curve" if current.is_complete else "partial curve"
        st.caption(f"As of {current.date:%d %b %Y} · {status} · US Treasury constant maturities")
    with header_right:
        horizon = st.segmented_control(
            "Comparison horizon",
            options=list(COMPARISON_OFFSETS),
            default="1M",
            required=True,
            key="yield_curve_comparison",
            label_visibility="collapsed",
            width="stretch",
        )

    analysis = analyze_curve(
        data,
        current.date,
        current.date - COMPARISON_OFFSETS[horizon],
    )
    if analysis is None:
        st.warning("No comparison curve is available for the selected horizon.")
        return
    if not analysis.current.is_complete or not analysis.comparison.is_complete:
        unavailable = [
            TENOR_LABELS[series_id]
            for series_id in YIELD_SERIES
            if pd.isna(analysis.current.values.get(series_id))
        ]
        detail = f" Missing current tenors: {', '.join(unavailable)}." if unavailable else ""
        st.warning(f"At least one curve is partial; unavailable points are omitted.{detail}")

    metric_specs = (
        ("2Y yield", analysis.current.values.get(DGS2), analysis.changes_bp.get(DGS2), "%"),
        ("10Y yield", analysis.current.values.get(DGS10), analysis.changes_bp.get(DGS10), "%"),
        ("30Y yield", analysis.current.values.get(DGS30), analysis.changes_bp.get(DGS30), "%"),
        ("2s10s", analysis.current_spreads_bp.get("2s10s"), analysis.spread_changes_bp.get("2s10s"), "bp"),
        ("5s30s", analysis.current_spreads_bp.get("5s30s"), analysis.spread_changes_bp.get("5s30s"), "bp"),
    )
    for column, (label, value, change, unit) in zip(st.columns(5), metric_specs):
        with column:
            display_value = _format_yield(value) if unit == "%" else _format_bp(value)
            st.metric(label, display_value, _format_delta(change, horizon), delta_color="off")

    st.caption(
        f"Changes compare {analysis.current.date:%d %b %Y} with the nearest available "
        f"observation on {analysis.comparison.date:%d %b %Y}."
    )
    st.info(_market_read(analysis, horizon))

    st.markdown("### Curve and repricing")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(_curve_figure(analysis), width="stretch")
    with chart_right:
        st.plotly_chart(_change_figure(analysis, horizon), width="stretch")

    spreads = spread_history(data)
    displayed_spreads = spreads.loc[
        (spreads.index >= pd.Timestamp(display_start))
        & (spreads.index <= pd.Timestamp(display_end))
    ]
    st.markdown("### Historical slope")
    spread_left, spread_right = st.columns(2)
    with spread_left:
        st.plotly_chart(
            _spread_figure(displayed_spreads["2s10s"], "2s10s", analysis.current_spreads_bp["2s10s"]),
            width="stretch",
        )
    with spread_right:
        st.plotly_chart(
            _spread_figure(displayed_spreads["5s30s"], "5s30s", analysis.current_spreads_bp["5s30s"]),
            width="stretch",
        )

    with st.expander("Detailed data", expanded=False):
        st.markdown("#### Yield changes by tenor")
        _render_html_table(_yield_change_table(data, analysis.current.date))
        st.markdown("#### Historical spread context")
        _render_html_table(_spread_context_table(spreads, analysis.current.date))
        st.caption("Percentiles use up to ten years of internally fetched history.")

    with st.expander("Methodology and definitions", expanded=False):
        st.markdown(
            """
**2s10s** is the 10Y yield minus the 2Y yield. **5s30s** is the 30Y yield
minus the 5Y yield. A steepening means the spread increased; a flattening
means it decreased. A bull move means yields broadly fell, while a bear move
means yields broadly rose.

All headline levels come from one observation date. Comparison curves use the
latest complete observation on or before the requested horizon, which handles
weekends and market holidays without mixing dates across tenors. The movement
labels are descriptive and do not assign a macroeconomic cause.
            """
        )
