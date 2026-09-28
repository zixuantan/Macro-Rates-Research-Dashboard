from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    CPIAUCSL,
    CPILFESL,
    INFLATION_SERIES,
    MICH,
    PCEPI,
    PCEPILFE,
    PPIFID,
    T10YIE,
)


T5YIE = "T5YIE"
T5YIFR = "T5YIFR"

MARKET_SERIES = [
    T5YIE,
    T10YIE,
    T5YIFR,
]

REQUIRED_SERIES = [
    CPIAUCSL,
    CPILFESL,
    PCEPI,
    PCEPILFE,
    PPIFID,
    MICH,
    T5YIE,
    T10YIE,
    T5YIFR,
]

METRIC_CAPTION_MIN_HEIGHT_PX = 82


def _value_as_of(
    series: pd.Series,
    as_of: pd.Timestamp,
) -> float:
    """Return the latest valid value on or before a requested date."""
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


def _change_over_window(
    series: pd.Series,
    latest_date: pd.Timestamp,
    offset: pd.DateOffset,
) -> float:
    """Calculate the change from the latest value to an earlier date."""
    latest_value = _value_as_of(
        series,
        latest_date,
    )

    reference_value = _value_as_of(
        series,
        latest_date - offset,
    )

    if (
        pd.isna(latest_value)
        or pd.isna(reference_value)
    ):
        return float("nan")

    return (
        latest_value
        - reference_value
    )


def _percentile_rank(
    series: pd.Series,
    value: float,
) -> float:
    """Calculate the percentile rank of a value within a series."""
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty or pd.isna(value):
        return float("nan")

    return float(
        (clean <= value).mean() * 100.0
    )


def _format_change_bp(
    change_percentage_points: float,
    period: str,
) -> str | None:
    """Format a percentage-point change as basis points."""
    if pd.isna(change_percentage_points):
        return None

    change_bp = (
        change_percentage_points
        * 100.0
    )

    return (
        f"{change_bp:+.0f} bp ({period})"
    )


def _format_percentile(
    percentile: float,
) -> str:
    """Format a percentile value."""
    if pd.isna(percentile):
        return "Unavailable"

    return f"{percentile:.0f}th percentile"


def _percentile_description(
    percentile: float,
) -> str:
    """Convert a percentile into a historical-range description."""
    if pd.isna(percentile):
        return "historical position unavailable"

    if percentile >= 90:
        return "near the top of its selected-range distribution"

    if percentile >= 75:
        return "in the upper quartile of its selected-range distribution"

    if percentile <= 10:
        return "near the bottom of its selected-range distribution"

    if percentile <= 25:
        return "in the lower quartile of its selected-range distribution"

    return "near the middle of its selected-range distribution"


def _momentum_description(
    change_1m: float,
) -> str:
    """Describe one-month breakeven momentum."""
    if pd.isna(change_1m):
        return "unavailable"

    change_bp = (
        change_1m
        * 100.0
    )

    if change_bp >= 20:
        return "rising strongly"

    if change_bp >= 5:
        return "rising moderately"

    if change_bp <= -20:
        return "falling strongly"

    if change_bp <= -5:
        return "falling moderately"

    return "broadly stable"


def _inflation_regime(
    five_year_be: float,
    five_year_change_1m: float,
    five_year_forward: float,
    cpi_yoy: float,
) -> tuple[str, str]:
    """Classify the current inflation-pricing regime."""
    if (
        pd.isna(five_year_be)
        or pd.isna(five_year_forward)
    ):
        return (
            "Unavailable",
            "Insufficient market data to classify inflation pricing.",
        )

    change_bp = (
        five_year_change_1m
        * 100.0
        if pd.notna(five_year_change_1m)
        else float("nan")
    )

    if (
        five_year_be >= 3.0
        or five_year_forward >= 3.0
    ):
        return (
            "Inflation scare",
            (
                "Market-based inflation compensation is elevated, "
                "indicating concern that inflation may remain persistently high."
            ),
        )

    if (
        pd.notna(change_bp)
        and change_bp >= 20
    ):
        return (
            "Expectations repricing higher",
            (
                "Near-term inflation compensation has risen sharply, "
                "although longer-run expectations may remain anchored."
            ),
        )

    if (
        pd.notna(change_bp)
        and change_bp <= -20
    ):
        return (
            "Disinflation repricing",
            (
                "Market-based inflation compensation has fallen sharply, "
                "suggesting stronger confidence in future disinflation."
            ),
        )

    if (
        five_year_forward <= 2.5
        and five_year_be <= 2.7
    ):
        if (
            pd.notna(cpi_yoy)
            and cpi_yoy > five_year_be + 0.75
        ):
            return (
                "Anchored despite elevated realized inflation",
                (
                    "Realized inflation remains above market pricing, "
                    "but medium- and long-term expectations remain contained."
                ),
            )

        return (
            "Expectations anchored",
            (
                "Medium- and long-term inflation compensation remains "
                "contained near historically moderate levels."
            ),
        )

    return (
        "Moderately elevated expectations",
        (
            "Inflation compensation is above a fully anchored range "
            "but does not indicate a broad inflation scare."
        ),
    )


def _render_metric_caption(
    text: str,
) -> None:
    """Render aligned text beneath summary metrics."""
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


def _line_figure(
    dataframe: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
    title: str,
    y_axis_title: str = "Percent",
) -> go.Figure:
    """Create a multi-series line chart."""
    figure = go.Figure()

    for column in columns:
        if column not in dataframe.columns:
            continue

        clean = dataframe[column].dropna()

        if clean.empty:
            continue

        figure.add_trace(
            go.Scatter(
                x=clean.index,
                y=clean,
                mode="lines",
                name=labels.get(
                    column,
                    column,
                ),
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        template="plotly_white",
        hovermode="x unified",
        height=390,
        legend_title_text="Series",
    )

    return figure


def _build_macro_note(
    latest_date: pd.Timestamp,
    five_year_be: float,
    ten_year_be: float,
    five_year_forward: float,
    cpi_yoy: float,
    pce_yoy: float,
    michigan: float,
    five_year_change_1m: float,
    ten_year_change_1m: float,
    risk_premium_proxy: float,
    regime: str,
) -> str:
    """Generate a concise inflation paragraph for a macro note."""
    statements: list[str] = []

    statements.append(
        f"As of {latest_date:%d %b %Y}, the inflation backdrop is "
        f"classified as {regime.lower()}."
    )

    if (
        pd.notna(five_year_be)
        and pd.notna(ten_year_be)
        and pd.notna(five_year_forward)
    ):
        statements.append(
            f"Five-year breakevens stand at {five_year_be:.2f}%, "
            f"10-year breakevens at {ten_year_be:.2f}%, and the "
            f"5Y5Y forward rate at {five_year_forward:.2f}%."
        )

    momentum_parts = []

    if pd.notna(five_year_change_1m):
        momentum_parts.append(
            f"5Y breakevens changed "
            f"{five_year_change_1m * 100:+.0f} bp"
        )

    if pd.notna(ten_year_change_1m):
        momentum_parts.append(
            f"10Y breakevens changed "
            f"{ten_year_change_1m * 100:+.0f} bp"
        )

    if momentum_parts:
        statements.append(
            "Over the past month, "
            + " while ".join(momentum_parts)
            + "."
        )

    if (
        pd.notna(cpi_yoy)
        and pd.notna(five_year_be)
    ):
        realized_gap = (
            cpi_yoy
            - five_year_be
        )

        if realized_gap >= 0.75:
            statements.append(
                f"Realized CPI inflation remains {realized_gap:.1f} "
                "percentage points above 5Y market pricing, indicating "
                "that investors still expect meaningful disinflation."
            )
        elif realized_gap <= -0.75:
            statements.append(
                "Market pricing is running above realized CPI inflation, "
                "suggesting investors are assigning weight to renewed "
                "inflation pressure."
            )
        else:
            statements.append(
                "Realized CPI and medium-term market pricing are relatively "
                "close, implying limited disagreement about the inflation path."
            )

    if (
        pd.notna(cpi_yoy)
        and pd.notna(pce_yoy)
    ):
        statements.append(
            f"Headline CPI inflation is {cpi_yoy:.1f}% YoY, while "
            f"headline PCE inflation is {pce_yoy:.1f}% YoY."
        )

    if pd.notna(michigan):
        statements.append(
            f"Household inflation expectations (Michigan) stand at "
            f"{michigan:.1f}%."
        )

    if pd.notna(risk_premium_proxy):
        if risk_premium_proxy >= 0.25:
            statements.append(
                "The 10Y breakeven remains above household inflation expectations, "
                "suggesting markets are pricing relatively elevated long-run "
                "inflation compensation."
            )
        elif risk_premium_proxy <= -0.25:
            statements.append(
                "The 10Y breakeven remains below household inflation expectations, "
                "indicating bond markets are pricing lower long-run inflation than "
                "households currently expect."
            )
        else:
            statements.append(
                "Market- and household-based inflation expectations remain broadly aligned."
            )

    return " ".join(
        statements
    )


def _legacy_render(
    fred_client,
    context: dict,
) -> None:
    st.subheader("Inflation Monitor")

    start_date = context[
        "start_date"
    ]

    end_date = context[
        "end_date"
    ]

    result = fred_client.get_series(
        INFLATION_SERIES,
        start_date,
        end_date,
    )

    if (
        not result.success
        or result.data is None
    ):
        st.warning(
            result.message
            or "Inflation data unavailable."
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
        for series_id in REQUIRED_SERIES
        if series_id not in df.columns
    ]

    if missing_series:
        st.warning(
            "The following inflation series are missing: "
            + ", ".join(
                missing_series
            )
        )
        return

    for series_id in REQUIRED_SERIES:
        df[series_id] = pd.to_numeric(
            df[series_id],
            errors="coerce",
        )

    monthly = (
        df[
            [
                CPIAUCSL,
                PCEPI,
                MICH,
            ]
        ]
        .resample("ME")
        .last()
    )

    cpi_yoy = (
        monthly[CPIAUCSL]
        .pct_change(
            12,
            fill_method=None,
        )
        * 100.0
    )

    pce_yoy = (
        monthly[PCEPI]
        .pct_change(
            12,
            fill_method=None,
        )
        * 100.0
    )

    context.setdefault("panel_history", {})["inflation"] = {
        "raw": df.copy(),
        "monthly": monthly.copy(),
        "cpi_yoy": cpi_yoy.copy(),
        "pce_yoy": pce_yoy.copy(),
    }

    market_daily = df[
        MARKET_SERIES
    ].copy()

    usable_market_dates = (
        market_daily
        .dropna(
            how="all"
        )
    )

    if usable_market_dates.empty:
        st.warning(
            "No usable market-based inflation data are available."
        )
        return

    latest_date = (
        usable_market_dates.index.max()
    )

    latest_five_year_be = _value_as_of(
        market_daily[T5YIE],
        latest_date,
    )

    latest_ten_year_be = _value_as_of(
        market_daily[T10YIE],
        latest_date,
    )

    latest_five_year_forward = _value_as_of(
        market_daily[T5YIFR],
        latest_date,
    )

    latest_cpi_yoy = _value_as_of(
        cpi_yoy,
        latest_date,
    )

    latest_pce_yoy = _value_as_of(
        pce_yoy,
        latest_date,
    )

    latest_michigan = _value_as_of(
        monthly[MICH],
        latest_date,
    )

    five_year_change_1w = _change_over_window(
        market_daily[T5YIE],
        latest_date,
        pd.DateOffset(
            weeks=1
        ),
    )

    five_year_change_1m = _change_over_window(
        market_daily[T5YIE],
        latest_date,
        pd.DateOffset(
            months=1
        ),
    )

    five_year_change_3m = _change_over_window(
        market_daily[T5YIE],
        latest_date,
        pd.DateOffset(
            months=3
        ),
    )

    ten_year_change_1m = _change_over_window(
        market_daily[T10YIE],
        latest_date,
        pd.DateOffset(
            months=1
        ),
    )

    forward_change_1m = _change_over_window(
        market_daily[T5YIFR],
        latest_date,
        pd.DateOffset(
            months=1
        ),
    )

    five_year_percentile = _percentile_rank(
        market_daily[T5YIE],
        latest_five_year_be,
    )

    ten_year_percentile = _percentile_rank(
        market_daily[T10YIE],
        latest_ten_year_be,
    )

    forward_percentile = _percentile_rank(
        market_daily[T5YIFR],
        latest_five_year_forward,
    )

    michigan_daily = (
        monthly[MICH]
        .reindex(
            market_daily.index,
            method="ffill",
        )
    )

    risk_premium_proxy = (
        market_daily[T10YIE]
        - michigan_daily
    )

    latest_risk_premium_proxy = _value_as_of(
        risk_premium_proxy,
        latest_date,
    )

    risk_premium_change_1m = _change_over_window(
        risk_premium_proxy,
        latest_date,
        pd.DateOffset(
            months=1
        ),
    )

    realized_gap = (
        latest_cpi_yoy
        - latest_five_year_be
        if (
            pd.notna(latest_cpi_yoy)
            and pd.notna(latest_five_year_be)
        )
        else float("nan")
    )

    regime, regime_description = _inflation_regime(
        latest_five_year_be,
        five_year_change_1m,
        latest_five_year_forward,
        latest_cpi_yoy,
    )

    macro_note = _build_macro_note(
        latest_date,
        latest_five_year_be,
        latest_ten_year_be,
        latest_five_year_forward,
        latest_cpi_yoy,
        latest_pce_yoy,
        latest_michigan,
        five_year_change_1m,
        ten_year_change_1m,
        latest_risk_premium_proxy,
        regime,
    )

    # ---------------------------------------------------------
    # CURRENT ASSESSMENT
    # ---------------------------------------------------------

    st.markdown(
        "### Current inflation assessment"
    )

    st.info(
        f"**{regime}.** {regime_description}"
    )

    realized_cpi_text = (
        f"{latest_cpi_yoy:.1f}% YoY"
        if pd.notna(
            latest_cpi_yoy
        )
        else "Unavailable"
    )

    realized_pce_text = (
        f"{latest_pce_yoy:.1f}% YoY"
        if pd.notna(
            latest_pce_yoy
        )
        else "Unavailable"
    )

    five_year_be_change_text = (
        "change unavailable"
    )

    if pd.notna(five_year_change_1m):
        if five_year_change_1m > 0:
            five_year_be_change_text = (
                f"up {abs(five_year_change_1m * 100):.0f} bp over 1M"
            )
        elif five_year_change_1m < 0:
            five_year_be_change_text = (
                f"down {abs(five_year_change_1m * 100):.0f} bp over 1M"
            )
        else:
            five_year_be_change_text = (
                "unchanged over 1M"
            )

    five_year_be_text = (
        f"{latest_five_year_be:.2f}%"
        if pd.notna(
            latest_five_year_be
        )
        else "Unavailable"
    )

    five_year_forward_text = (
        f"{latest_five_year_forward:.2f}%"
        if pd.notna(
            latest_five_year_forward
        )
        else "Unavailable"
    )

    michigan_text = (
        f"{latest_michigan:.1f}%"
        if pd.notna(
            latest_michigan
        )
        else "Unavailable"
    )

    st.markdown(
        f"""
**Supporting evidence**
- Realized: CPI {realized_cpi_text} and PCE {realized_pce_text}
- Market: 5Y breakeven {five_year_be_text}, {five_year_be_change_text}
- Long-run anchor: 5Y5Y forward {five_year_forward_text}
- Households: Michigan expectations {michigan_text}
        """
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
            "5Y breakeven",
            (
                f"{latest_five_year_be:.2f}%"
                if pd.notna(
                    latest_five_year_be
                )
                else "Unavailable"
            ),
            _format_change_bp(
                five_year_change_1m,
                "1M",
            ),
        )

        _render_metric_caption(
            (
                f"{_format_percentile(five_year_percentile)}; "
                f"{_momentum_description(five_year_change_1m)} over 1M."
            )
        )

    with metric_2:
        st.metric(
            "10Y breakeven",
            (
                f"{latest_ten_year_be:.2f}%"
                if pd.notna(
                    latest_ten_year_be
                )
                else "Unavailable"
            ),
            _format_change_bp(
                ten_year_change_1m,
                "1M",
            ),
        )

        _render_metric_caption(
            (
                f"{_format_percentile(ten_year_percentile)} within "
                "the selected date range."
            )
        )

    with metric_3:
        st.metric(
            "5Y5Y forward",
            (
                f"{latest_five_year_forward:.2f}%"
                if pd.notna(
                    latest_five_year_forward
                )
                else "Unavailable"
            ),
            _format_change_bp(
                forward_change_1m,
                "1M",
            ),
        )

        _render_metric_caption(
            (
                f"{_format_percentile(forward_percentile)}; "
                "proxy for longer-run inflation compensation."
            )
        )

    with metric_4:
        st.metric(
            "Headline CPI YoY (SA)",
            (
                f"{latest_cpi_yoy:.1f}%"
                if pd.notna(
                    latest_cpi_yoy
                )
                else "Unavailable"
            ),
            (
                f"{realized_gap:+.1f} pp vs 5Y BE"
                if pd.notna(
                    realized_gap
                )
                else None
            ),
            delta_color="off",
        )

        _render_metric_caption(
            (
                "Shows realized inflation relative to market pricing."
            )
        )

    with metric_5:
        st.metric(
            "Headline PCE YoY",
            (
                f"{latest_pce_yoy:.1f}%"
                if pd.notna(
                    latest_pce_yoy
                )
                else "Unavailable"
            ),
        )

        _render_metric_caption(
            (
                "Latest realized PCE inflation, the Federal Reserve's preferred "
                "inflation measure."
            )
        )

    with metric_6:
        st.metric(
            "Michigan Expectations",
            (
                f"{latest_michigan:.1f}%"
                if pd.notna(
                    latest_michigan
                )
                else "Unavailable"
            ),
        )

        _render_metric_caption(
            (
                "Household inflation expectations from the University of "
                "Michigan survey."
            )
        )

    st.caption(
        f"Latest market observation: {latest_date:%d %b %Y}. "
        f"Percentiles rank current readings against observations from "
        f"{pd.Timestamp(start_date):%d %b %Y} to "
        f"{pd.Timestamp(end_date):%d %b %Y}."
    )

    # ---------------------------------------------------------
    # NOTE-WRITING OUTPUT
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
        "This paragraph is mechanically generated from current levels, "
        "one-month changes and cross-measure gaps. It should be combined "
        "with event context before being used as a final market view."
    )

    # ---------------------------------------------------------
    # MOMENTUM AND HISTORICAL CONTEXT
    # ---------------------------------------------------------

    st.markdown(
        "### Breakeven momentum and context"
    )

    momentum_columns = st.columns(3)

    with momentum_columns[0]:
        st.metric(
            "5Y breakeven: 1W",
            _format_change_bp(
                five_year_change_1w,
                "1W",
            )
            or "Unavailable",
        )

    with momentum_columns[1]:
        st.metric(
            "5Y breakeven: 1M",
            _format_change_bp(
                five_year_change_1m,
                "1M",
            )
            or "Unavailable",
        )

    with momentum_columns[2]:
        st.metric(
            "5Y breakeven: 3M",
            _format_change_bp(
                five_year_change_3m,
                "3M",
            )
            or "Unavailable",
        )

    st.caption(
        f"Five-year breakevens are "
        f"{_percentile_description(five_year_percentile)} and have been "
        f"{_momentum_description(five_year_change_1m)} over the past month."
    )

    # ---------------------------------------------------------
    # MARKET PRICING
    # ---------------------------------------------------------

    st.markdown(
        "### Market-based inflation pricing"
    )

    market_labels = {
        T5YIE: "5Y breakeven",
        T10YIE: "10Y breakeven",
        T5YIFR: "5Y5Y forward inflation",
    }

    st.plotly_chart(
        _line_figure(
            market_daily,
            MARKET_SERIES,
            market_labels,
            "Breakevens and Forward Inflation Compensation",
        ),
        use_container_width=True,
    )

    st.caption(
        "Breakevens reflect expected inflation plus inflation risk "
        "and liquidity premia. They should not be treated as pure forecasts."
    )

    # ---------------------------------------------------------
    # REALIZED AND SURVEY INFLATION
    # ---------------------------------------------------------

    st.markdown(
        "### Realized and survey inflation"
    )

    realized_monthly = pd.DataFrame(
        {
            "CPI YoY": cpi_yoy,
            "PCE YoY": pce_yoy,
            "Michigan expectations": monthly[MICH],
        }
    )

    realized_labels = {
        "CPI YoY": "CPI YoY",
        "PCE YoY": "PCE YoY",
        "Michigan expectations": "Michigan survey expectations",
    }

    st.plotly_chart(
        _line_figure(
            realized_monthly,
            [
                "CPI YoY",
                "PCE YoY",
                "Michigan expectations",
            ],
            realized_labels,
            "Realized Inflation and Household Expectations",
        ),
        use_container_width=True,
    )

    # ---------------------------------------------------------
    # REALIZED VERSUS MARKET PRICING
    # ---------------------------------------------------------

    st.markdown(
        "### Realized inflation versus market pricing"
    )

    comparison_daily = pd.DataFrame(
        index=market_daily.index
    )

    comparison_daily["CPI YoY"] = cpi_yoy.reindex(
        comparison_daily.index,
        method="ffill",
    )

    comparison_daily["PCE YoY"] = pce_yoy.reindex(
        comparison_daily.index,
        method="ffill",
    )

    comparison_daily["5Y breakeven"] = market_daily[
        T5YIE
    ]

    comparison_daily["10Y breakeven"] = market_daily[
        T10YIE
    ]

    comparison_labels = {
        "CPI YoY": "CPI YoY",
        "PCE YoY": "PCE YoY",
        "5Y breakeven": "5Y breakeven",
        "10Y breakeven": "10Y breakeven",
    }

    st.plotly_chart(
        _line_figure(
            comparison_daily,
            [
                "CPI YoY",
                "PCE YoY",
                "5Y breakeven",
                "10Y breakeven",
            ],
            comparison_labels,
            "Realized Inflation versus Breakeven Pricing",
        ),
        use_container_width=True,
    )

    if pd.notna(realized_gap):
        if realized_gap > 0:
            st.caption(
                f"CPI inflation is currently {realized_gap:.1f} percentage "
                "points above the 5Y breakeven, indicating that markets "
                "expect inflation to moderate from its current rate."
            )
        else:
            st.caption(
                f"The 5Y breakeven is currently {abs(realized_gap):.1f} "
                "percentage points above CPI inflation, indicating that "
                "markets price inflation above the latest realized reading."
            )

    # ---------------------------------------------------------
    # MARKET VERSUS SURVEY EXPECTATIONS
    # ---------------------------------------------------------

    st.markdown(
        "### Market versus survey expectations"
    )

    spread_figure = go.Figure()

    spread_figure.add_trace(
        go.Scatter(
            x=risk_premium_proxy.index,
            y=risk_premium_proxy,
            mode="lines",
            name="10Y breakeven − Michigan",
        )
    )

    spread_figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.6,
    )

    spread_figure.update_layout(
        title="10Y Breakeven Minus Michigan Inflation Expectations",
        xaxis_title="Date",
        yaxis_title="Percentage points",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        height=390,
    )

    st.plotly_chart(
        spread_figure,
        use_container_width=True,
    )

    st.caption(
        "A positive gap means the 10Y breakeven exceeds Michigan survey "
        "expectations. A negative gap means market pricing is below the "
        "survey measure. This spread is only a rough comparison because "
        "the measures differ in horizon, population and risk-premium content."
    )

    # ---------------------------------------------------------
    # METHOD NOTES
    # ---------------------------------------------------------

    with st.expander(
        "View methodology and interpretation notes",
        expanded=False,
    ):
        st.markdown(
            """
**How to read the measures**

- **5Y breakeven:** market inflation compensation over the next five years.
- **10Y breakeven:** market inflation compensation over the next ten years.
- **5Y5Y forward inflation:** inflation compensation over a five-year period
  beginning five years from now, often used to assess longer-run anchoring.
- **Headline CPI YoY (SA):** realized consumer inflation over the past 12 months.
- **Headline PCE YoY:** realized personal consumption inflation over the past
  12 months and the Federal Reserve's preferred inflation measure.
- **Michigan expectations:** household survey expectations for future inflation.

**Important limitations**

Breakevens contain expected inflation, inflation-risk premia and liquidity
effects. The 10Y breakeven-minus-Michigan spread is therefore not a clean
estimate of the inflation risk premium. Regime labels and the generated macro
paragraph describe the data mechanically and do not identify the causal driver.
            """
        )


INFLATION_HORIZONS = {
    "1W": pd.DateOffset(weeks=1),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
}


INFLATION_MEASURE_GUIDE = (
    (
        "CPI",
        "Consumer",
        "Prices paid directly by urban consumers; the most familiar household inflation gauge.",
    ),
    (
        "Core CPI",
        "Consumer · underlying",
        "CPI excluding food and energy, used to reveal the less volatile consumer-price trend.",
    ),
    (
        "PCE",
        "Consumer · broad",
        "A broader, chain-weighted measure of household consumption prices and the Fed's target gauge.",
    ),
    (
        "Core PCE",
        "Consumer · underlying",
        "PCE excluding food and energy; a key measure of persistent underlying inflation pressure.",
    ),
    (
        "PPI",
        "Producer",
        "Prices received by domestic producers for final-demand output; a view of pipeline pressure.",
    ),
)


def _render_inflation_measure_cards(values: dict[str, float]) -> None:
    cards = []
    for name, category, description in INFLATION_MEASURE_GUIDE:
        value = values.get(name, float("nan"))
        value_text = f"{value:.2f}%" if pd.notna(value) else "Unavailable"
        category_class = "producer" if category == "Producer" else "consumer"
        cards.append(
            f"""
            <article class="inflation-measure-card">
                <div class="inflation-measure-topline">
                    <span class="inflation-measure-name">{html.escape(name)}</span>
                    <span class="inflation-measure-type {category_class}">{html.escape(category)}</span>
                </div>
                <div class="inflation-measure-value">{value_text}</div>
                <div class="inflation-measure-period">Year-over-year</div>
                <p>{html.escape(description)}</p>
            </article>
            """
        )
    st.html(
        f"""
        <style>
            .inflation-measure-grid {{
                display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
                gap: .65rem; margin: .35rem 0 1rem;
            }}
            .inflation-measure-card {{
                min-width: 0; padding: .85rem .9rem; border: 1px solid #dfe5ee;
                border-radius: .78rem; background: rgba(255,255,255,.92);
                box-shadow: 0 6px 20px rgba(30,47,78,.045);
            }}
            .inflation-measure-topline {{
                display: flex; align-items: center; justify-content: space-between;
                gap: .45rem;
            }}
            .inflation-measure-name {{ color: #14213d; font-size: .86rem; font-weight: 750; }}
            .inflation-measure-type {{
                padding: .16rem .38rem; border-radius: 999px; font-size: .58rem;
                font-weight: 750; letter-spacing: .035em; text-transform: uppercase;
            }}
            .inflation-measure-type.consumer {{ color: #1d4ed8; background: #eaf0ff; }}
            .inflation-measure-type.producer {{ color: #b45309; background: #fff3df; }}
            .inflation-measure-value {{
                margin-top: .58rem; color: #14213d; font-size: 1.42rem;
                font-weight: 720; letter-spacing: -.025em;
            }}
            .inflation-measure-period {{ margin-top: .05rem; color: #64748b; font-size: .66rem; }}
            .inflation-measure-card p {{
                margin: .58rem 0 0; color: #64748b; font-size: .72rem; line-height: 1.42;
            }}
        </style>
        <div class="inflation-measure-grid">{''.join(cards)}</div>
        """
    )


def _annualized_change(series: pd.Series, months: int) -> pd.Series:
    """Annualize an index change over a trailing monthly window."""
    clean = pd.to_numeric(series, errors="coerce")
    return ((clean / clean.shift(months)) ** (12.0 / months) - 1.0) * 100.0


def render(
    fred_client,
    context: dict,
) -> None:
    st.subheader("Inflation Monitor")

    start_date = pd.Timestamp(context["start_date"])
    end_date = pd.Timestamp(context["end_date"])
    fetch_start = start_date - pd.DateOffset(months=15)
    result = fred_client.get_series(INFLATION_SERIES, fetch_start.date(), end_date.date())
    if not result.success or result.data is None:
        st.warning(result.message or "Inflation data unavailable.")
        return

    df = result.data.copy().sort_index()
    missing_series = [series_id for series_id in REQUIRED_SERIES if series_id not in df.columns]
    if df.empty or missing_series:
        message = "No inflation data are available." if df.empty else "Missing inflation series: " + ", ".join(missing_series)
        st.warning(message)
        return
    for series_id in REQUIRED_SERIES:
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    monthly = df[[CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, PPIFID, MICH]].resample("ME").last()
    cpi_yoy = monthly[CPIAUCSL].pct_change(12, fill_method=None) * 100.0
    core_cpi_yoy = monthly[CPILFESL].pct_change(12, fill_method=None) * 100.0
    pce_yoy = monthly[PCEPI].pct_change(12, fill_method=None) * 100.0
    core_pce_yoy = monthly[PCEPILFE].pct_change(12, fill_method=None) * 100.0
    ppi_yoy = monthly[PPIFID].pct_change(12, fill_method=None) * 100.0
    core_cpi_3m = _annualized_change(monthly[CPILFESL], 3)
    core_pce_3m = _annualized_change(monthly[PCEPILFE], 3)
    market_daily = df[MARKET_SERIES].copy()
    usable_market = market_daily.dropna(how="all")
    if usable_market.empty:
        st.warning("No usable market-based inflation data are available.")
        return
    latest_date = usable_market.index.max()

    context.setdefault("panel_history", {})["inflation"] = {
        "raw": df.copy(),
        "monthly": monthly.copy(),
        "cpi_yoy": cpi_yoy.copy(),
        "pce_yoy": pce_yoy.copy(),
        "core_cpi_yoy": core_cpi_yoy.copy(),
        "core_pce_yoy": core_pce_yoy.copy(),
        "ppi_yoy": ppi_yoy.copy(),
        "core_cpi_3m": core_cpi_3m.copy(),
        "core_pce_3m": core_pce_3m.copy(),
    }

    header_left, header_right = st.columns([2, 3])
    with header_left:
        st.caption(f"Market data as of {latest_date:%d %b %Y} · realized data update monthly")
    with header_right:
        horizon = st.segmented_control(
            "Comparison horizon",
            options=list(INFLATION_HORIZONS),
            default="1M",
            required=True,
            key="inflation_comparison_horizon",
            label_visibility="collapsed",
            width="stretch",
        )

    latest_five_year_be = _value_as_of(market_daily[T5YIE], latest_date)
    latest_ten_year_be = _value_as_of(market_daily[T10YIE], latest_date)
    latest_forward = _value_as_of(market_daily[T5YIFR], latest_date)
    latest_core_cpi_yoy = _value_as_of(core_cpi_yoy, latest_date)
    latest_core_pce_yoy = _value_as_of(core_pce_yoy, latest_date)
    latest_cpi_yoy = _value_as_of(cpi_yoy, latest_date)
    latest_pce_yoy = _value_as_of(pce_yoy, latest_date)
    latest_ppi_yoy = _value_as_of(ppi_yoy, latest_date)
    horizon_offset = INFLATION_HORIZONS[horizon]
    five_year_change = _change_over_window(market_daily[T5YIE], latest_date, horizon_offset)
    ten_year_change = _change_over_window(market_daily[T10YIE], latest_date, horizon_offset)
    forward_change = _change_over_window(market_daily[T5YIFR], latest_date, horizon_offset)

    display_start = start_date.normalize()
    st.markdown("### Inflation measures")
    _render_inflation_measure_cards(
        {
            "CPI": latest_cpi_yoy,
            "Core CPI": latest_core_cpi_yoy,
            "PCE": latest_pce_yoy,
            "Core PCE": latest_core_pce_yoy,
            "PPI": latest_ppi_yoy,
        }
    )
    inflation_measure_history = pd.DataFrame(
        {
            "CPI": cpi_yoy,
            "Core CPI": core_cpi_yoy,
            "PCE": pce_yoy,
            "Core PCE": core_pce_yoy,
            "PPI": ppi_yoy,
        }
    ).loc[lambda frame: frame.index >= display_start]
    st.plotly_chart(
        _line_figure(
            inflation_measure_history,
            list(inflation_measure_history.columns),
            {column: column for column in inflation_measure_history.columns},
            "Consumer and producer inflation",
        ),
        width="stretch",
    )
    st.caption(
        "All five measures are shown as year-over-year changes for a like-for-like comparison. "
        "PPI is typically more volatile and measures producer selling prices, not prices paid by consumers."
    )

    st.markdown("### Market-implied inflation")

    metric_specs = (
        (
            "5Y breakeven",
            latest_five_year_be,
            _format_change_bp(five_year_change, horizon),
            "Isolates near-term / cyclical shocks",
            "The market's expected average annual CPI inflation rate over the *next 5 years*.",
        ),
        (
            "10Y breakeven",
            latest_ten_year_be,
            _format_change_bp(ten_year_change, horizon),
            "Blends near- and long-term expectations",
            "The market's expected average annual CPI inflation rate over the *next 10 years*.",
        ),
        (
            "5Y5Y forward",
            latest_forward,
            _format_change_bp(forward_change, horizon),
            "Isolates structural inflation",
            "The market's expected average annual CPI inflation rate over a *5-year period starting 5 years from today*.",
        ),
    )
    for column, (label, value, delta, purpose, description) in zip(st.columns(3), metric_specs):
        with column:
            with st.container(border=True):
                st.metric(
                    label,
                    f"{value:.2f}%" if pd.notna(value) else "Unavailable",
                    delta,
                    delta_color="off",
                )
                st.markdown(f"**{purpose}**")
                st.caption(description)

    core_momentum = pd.DataFrame(
        {
            "Core CPI · 3M annualized": core_cpi_3m,
            "Core CPI · YoY": core_cpi_yoy,
            "Core PCE · 3M annualized": core_pce_3m,
            "Core PCE · YoY": core_pce_yoy,
        }
    ).loc[lambda frame: frame.index >= display_start]
    expectations = market_daily.copy()
    expectations = expectations.loc[expectations.index >= display_start]

    st.markdown("### Inflation Momentum and Inflation Expectations")
    momentum_column, pricing_column = st.columns(2)
    with momentum_column:
        momentum_figure = _line_figure(
            core_momentum,
            list(core_momentum.columns),
            {column: column for column in core_momentum.columns},
            "Core Inflation Momentum",
        )
        momentum_styles = {
            "Core CPI · 3M annualized": {"color": "#2563eb", "dash": "solid", "width": 2.5},
            "Core CPI · YoY": {"color": "#2563eb", "dash": "dash", "width": 1.6},
            "Core PCE · 3M annualized": {"color": "#d97706", "dash": "solid", "width": 2.5},
            "Core PCE · YoY": {"color": "#d97706", "dash": "dash", "width": 1.6},
        }
        for trace in momentum_figure.data:
            trace.update(
                line=momentum_styles[trace.name],
                opacity=0.65 if trace.name.endswith("YoY") else 1.0,
            )
        st.caption("Showing whether current underlying inflation is accelerating or decelerating")
        st.plotly_chart(
            momentum_figure,
            width="stretch",
        )
    with pricing_column:
        st.caption(
            "Showing whether bond markets are pricing greater or lower future inflation risk; "
            "breakevens also include risk and liquidity premia."
        )
        st.plotly_chart(
            _line_figure(
                expectations,
                [T5YIE, T10YIE, T5YIFR],
                {
                    T5YIE: "5Y breakeven",
                    T10YIE: "10Y breakeven",
                    T5YIFR: "5Y5Y forward",
                },
                "Market-Implied Inflation Compensation",
            ),
            width="stretch",
        )

    st.markdown("#### Reading momentum against market pricing")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Momentum": "Rising",
                    "Market pricing": "Rising",
                    "Possible interpretation": "Inflation pressure is broadening and markets are responding",
                },
                {
                    "Momentum": "Rising",
                    "Market pricing": "Stable",
                    "Possible interpretation": "Current inflation is stronger, but markets view it as temporary",
                },
                {
                    "Momentum": "Falling",
                    "Market pricing": "Rising",
                    "Possible interpretation": "Markets may be concerned about future inflation despite improving current data",
                },
                {
                    "Momentum": "Falling",
                    "Market pricing": "Falling",
                    "Possible interpretation": "Disinflation is occurring and markets are pricing it forward",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
