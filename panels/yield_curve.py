from __future__ import annotations

import html
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
from data.market_events import get_scheduled_catalysts
from data.rate_news import get_rate_headlines


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
    st.html(
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
        """
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
        f"**Regime: {analysis.regime}.** {_move_driver(two_change, ten_change)} "
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


def _rate_relevant_events(events: pd.DataFrame) -> pd.DataFrame:
    """Keep scheduled events with a direct and interpretable rates channel."""
    if events.empty:
        return events
    keywords = (
        "consumer price",
        "producer price",
        "employment situation",
        "job openings",
        "employment cost",
        "productivity",
        "personal income and outlays",
        "gross domestic product",
        "gdp",
        "international trade",
        "corporate profits",
        "fomc",
        "treasury auction",
        "note auction",
        "bond auction",
        "tips auction",
    )
    names = events["event_name"].fillna("").str.lower()
    types = events["event_type"].fillna("").str.lower()
    relevant = names.map(lambda name: any(keyword in name for keyword in keywords))
    relevant |= types.eq("fomc calendar")
    relevant |= types.eq("treasury auction") & ~names.str.contains("bill|frn", regex=True)
    return events.loc[relevant].copy()


def _event_watch_area(event_name: str, event_type: str) -> str:
    text = f"{event_name} {event_type}".lower()
    if "fomc" in text:
        return "Policy path; front end first, then the full curve"
    if any(term in text for term in ("consumer price", "producer price", "personal income")):
        return "Inflation expectations and the 2Y–10Y sector"
    if any(term in text for term in ("employment", "job openings", "productivity")):
        return "Policy expectations and the front end"
    if any(term in text for term in ("gdp", "gross domestic", "trade", "corporate profits")):
        return "Growth expectations and intermediate maturities"
    if "auction" in text:
        return "Demand, concession and the matching maturity sector"
    return "Rates expectations across the curve"


def _event_move_text(data: pd.DataFrame, event_date: pd.Timestamp) -> str | None:
    """Return the same-day curve move when an exact daily observation exists."""
    event_curve = curve_snapshot(data, event_date)
    if event_curve is None or event_curve.date.normalize() != event_date.normalize():
        return None
    prior_curve = curve_snapshot(data, event_date - pd.DateOffset(days=1))
    if prior_curve is None:
        return None
    two_change = (event_curve.values.get(DGS2) - prior_curve.values.get(DGS2)) * 100.0
    ten_change = (event_curve.values.get(DGS10) - prior_curve.values.get(DGS10)) * 100.0
    event_spread = (event_curve.values.get(DGS10) - event_curve.values.get(DGS2)) * 100.0
    prior_spread = (prior_curve.values.get(DGS10) - prior_curve.values.get(DGS2)) * 100.0
    spread_change = event_spread - prior_spread
    if any(pd.isna(value) for value in (two_change, ten_change, spread_change)):
        return None
    return f"2Y {two_change:+.1f} bp · 10Y {ten_change:+.1f} bp · 2s10s {spread_change:+.1f} bp"


def _event_badge(event_type: str) -> tuple[str, str]:
    lowered = event_type.lower()
    if "fomc" in lowered:
        return "FOMC", "policy"
    if "auction" in lowered:
        return "Auction", "auction"
    return "Release", "release"


def _render_event_calendar(
    events: pd.DataFrame,
    *,
    data: pd.DataFrame | None = None,
    upcoming: bool,
) -> None:
    if events.empty:
        message = (
            "No rate-relevant official events were found in the next 14 days."
            if upcoming
            else "No rate-relevant official events were found in the recent window."
        )
        st.caption(message)
        return

    work = events.copy()
    work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce")
    work = work.dropna(subset=["event_date"]).sort_values(
        ["event_date", "event_time", "event_name"],
        ascending=[upcoming, True, True],
    )
    work = work.head(10 if upcoming else 8)

    day_blocks = []
    for event_date, day_events in work.groupby("event_date", sort=False):
        event_rows = []
        for row in day_events.itertuples(index=False):
            badge, badge_class = _event_badge(str(row.event_type))
            source = html.escape(str(row.source))
            source_url = html.escape(str(row.source_url), quote=True)
            source_html = (
                f'<a href="{source_url}" target="_blank">{source}</a>'
                if source_url
                else source
            )
            time_html = (
                f'<span class="rates-event-time">{html.escape(str(row.event_time))}</span>'
                if row.event_time
                else ""
            )
            if upcoming:
                detail = (
                    '<div class="rates-event-detail"><strong>Watch:</strong> '
                    f'{html.escape(_event_watch_area(str(row.event_name), str(row.event_type)))}</div>'
                )
            else:
                move = _event_move_text(data, pd.Timestamp(event_date)) if data is not None else None
                detail = (
                    f'<div class="rates-move-chip">Curve change on the same day · {html.escape(move)}</div>'
                    if move
                    else '<div class="rates-event-detail">Daily move unavailable</div>'
                )
            event_rows.append(
                f"""
                <div class="rates-event-row">
                    <div class="rates-event-heading">
                        <span class="rates-badge {badge_class}">{badge}</span>
                        {time_html}
                        <span class="rates-event-name">{html.escape(str(row.event_name))}</span>
                    </div>
                    <div class="rates-event-source">{source_html}</div>
                    {detail}
                </div>
                """
            )

        date_value = pd.Timestamp(event_date)
        day_blocks.append(
            f"""
            <div class="rates-calendar-day">
                <div class="rates-date-tile">
                    <span>{date_value:%b}</span>
                    <strong>{date_value:%d}</strong>
                    <small>{date_value:%a}</small>
                </div>
                <div class="rates-day-events">{''.join(event_rows)}</div>
            </div>
            """
        )

    st.html(
        f"""
        <style>
            .rates-calendar {{ display: grid; gap: .65rem; margin: .35rem 0 .8rem; }}
            .rates-calendar-day {{
                display: grid; grid-template-columns: 4.5rem minmax(0, 1fr); gap: .8rem;
                padding: .75rem; border: 1px solid rgba(128,128,128,.22);
                border-radius: .7rem; background: rgba(128,128,128,.035);
            }}
            .rates-date-tile {{
                align-self: start; display: flex; flex-direction: column; align-items: center;
                overflow: hidden; border: 1px solid rgba(37,99,235,.28); border-radius: .6rem;
                background: rgba(37,99,235,.07); line-height: 1;
            }}
            .rates-date-tile span {{
                width: 100%; padding: .3rem 0; text-align: center; color: #fff;
                background: #2563eb; font-size: .68rem; font-weight: 700; letter-spacing: .08em;
                text-transform: uppercase;
            }}
            .rates-date-tile strong {{ padding-top: .38rem; font-size: 1.45rem; }}
            .rates-date-tile small {{ padding: .15rem 0 .38rem; opacity: .65; font-size: .68rem; }}
            .rates-day-events {{ min-width: 0; }}
            .rates-event-row {{ padding: .05rem 0 .7rem; }}
            .rates-event-row + .rates-event-row {{
                padding-top: .7rem; border-top: 1px solid rgba(128,128,128,.18);
            }}
            .rates-event-row:last-child {{ padding-bottom: .05rem; }}
            .rates-event-heading {{ display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; }}
            .rates-event-name {{ font-weight: 650; line-height: 1.25; }}
            .rates-event-time {{ font-size: .76rem; font-weight: 650; opacity: .72; white-space: nowrap; }}
            .rates-badge {{
                padding: .18rem .42rem; border-radius: 999px; font-size: .64rem;
                font-weight: 750; letter-spacing: .04em; text-transform: uppercase;
            }}
            .rates-badge.release {{ color: #1d4ed8; background: rgba(37,99,235,.12); }}
            .rates-badge.policy {{ color: #7e22ce; background: rgba(147,51,234,.12); }}
            .rates-badge.auction {{ color: #c2410c; background: rgba(234,88,12,.12); }}
            .rates-event-source, .rates-event-detail {{ margin-top: .28rem; font-size: .78rem; opacity: .72; }}
            .rates-event-source a {{ color: inherit; text-decoration: underline; }}
            .rates-move-chip {{
                display: inline-block; margin-top: .38rem; padding: .25rem .5rem;
                border-radius: .35rem; background: rgba(15,118,110,.10);
                color: #0f766e; font-size: .74rem; font-weight: 650;
            }}
            @media (max-width: 640px) {{
                .rates-calendar-day {{ grid-template-columns: 3.7rem minmax(0, 1fr); padding: .6rem; }}
                .rates-date-tile strong {{ font-size: 1.2rem; }}
            }}
        </style>
        <div class="rates-calendar">{''.join(day_blocks)}</div>
        """
    )


def _render_headline_feed(headlines: pd.DataFrame) -> None:
    if headlines.empty:
        st.caption("No matching rates headlines are available for this date window.")
        return

    cards = []
    for row in headlines.head(8).itertuples(index=False):
        published_at = pd.Timestamp(row.published_at)
        source_bits = [str(row.domain)]
        if str(row.source_country).strip():
            source_bits.append(str(row.source_country).strip())
        source = " · ".join(source_bits)
        cards.append(
            f"""
            <article class="rates-headline-card">
                <div class="rates-headline-meta">
                    <span class="rates-topic">{html.escape(str(row.topic))}</span>
                    <time>{published_at:%d %b · %H:%M} UTC</time>
                </div>
                <a class="rates-headline-title" href="{html.escape(str(row.url), quote=True)}"
                   target="_blank" rel="noopener noreferrer">{html.escape(str(row.title))}</a>
                <div class="rates-headline-source">{html.escape(source)}</div>
            </article>
            """
        )

    st.html(
        f"""
        <style>
            .rates-headlines {{ display: grid; gap: .55rem; margin: .35rem 0 .8rem; }}
            .rates-headline-card {{
                padding: .75rem .85rem; border: 1px solid rgba(128,128,128,.22);
                border-radius: .7rem; background: rgba(128,128,128,.035);
            }}
            .rates-headline-meta {{
                display: flex; align-items: center; justify-content: space-between;
                gap: .6rem; margin-bottom: .42rem; color: rgba(49,51,63,.62);
                font-size: .69rem;
            }}
            .rates-topic {{
                padding: .18rem .42rem; border-radius: 999px; color: #1d4ed8;
                background: rgba(37,99,235,.12); font-weight: 750;
                letter-spacing: .04em; text-transform: uppercase;
            }}
            .rates-headline-title {{
                color: inherit; font-size: .9rem; font-weight: 650;
                line-height: 1.32; text-decoration: none;
            }}
            .rates-headline-title:hover {{ color: #2563eb; text-decoration: underline; }}
            .rates-headline-source {{ margin-top: .4rem; font-size: .72rem; opacity: .65; }}
        </style>
        <div class="rates-headlines">{''.join(cards)}</div>
        """
    )


def render(fred_client, context: dict) -> None:
    st.subheader("Treasury Yield Curve")

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

    recent_start = max(
        analysis.comparison.date,
        analysis.current.date - pd.DateOffset(days=45),
    )
    upcoming_start = pd.Timestamp(display_end).normalize() + pd.DateOffset(days=1)
    upcoming_end = upcoming_start + pd.DateOffset(days=13)
    catalyst_end = max(analysis.current.date, upcoming_end)
    catalysts = _rate_relevant_events(
        get_scheduled_catalysts(recent_start, catalyst_end)
    )
    catalyst_dates = pd.to_datetime(catalysts["event_date"], errors="coerce")
    recent_events = catalysts.loc[
        (catalyst_dates >= recent_start.normalize())
        & (catalyst_dates <= analysis.current.date.normalize())
    ]
    upcoming_events = catalysts.loc[
        (catalyst_dates >= upcoming_start)
        & (catalyst_dates <= upcoming_end)
    ]
    headlines = get_rate_headlines(recent_start, analysis.current.date)

    recent_evidence = recent_events.copy()
    if not recent_evidence.empty:
        recent_evidence["watch_area"] = recent_evidence.apply(
            lambda row: _event_watch_area(str(row["event_name"]), str(row["event_type"])),
            axis=1,
        )
        recent_evidence["curve_move"] = recent_evidence["event_date"].map(
            lambda event_date: _event_move_text(data, pd.Timestamp(event_date)) or "Unavailable"
        )
    upcoming_evidence = upcoming_events.copy()
    if not upcoming_evidence.empty:
        upcoming_evidence["watch_area"] = upcoming_evidence.apply(
            lambda row: _event_watch_area(str(row["event_name"]), str(row["event_type"])),
            axis=1,
        )
    context["rates_evidence"] = {
        "recent_events": recent_evidence,
        "upcoming_events": upcoming_evidence,
        "headlines": headlines.copy(),
        "as_of": analysis.current.date,
        "recent_start": recent_start,
    }

    calendar_column, news_column = st.columns([1.08, 0.92], gap="large")
    with calendar_column:
        st.markdown("#### Rates calendar")
        recent_tab, upcoming_tab = st.tabs(["Recent events", "Next 14 days"])
        with recent_tab:
            _render_event_calendar(recent_events, data=data, upcoming=False)
        with upcoming_tab:
            st.caption("Official releases, policy events and coupon-bearing Treasury auctions.")
            _render_event_calendar(upcoming_events, upcoming=True)

    with news_column:
        st.markdown("#### Related headlines")
        st.caption("Recent rates coverage from GDELT.")
        _render_headline_feed(headlines)

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
