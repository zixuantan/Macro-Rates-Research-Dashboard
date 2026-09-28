from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    CANDH,
    CFNAI,
    CFNAIMA3,
    CFNAI_SERIES,
    EUANDH,
    GACDISA066MSFRBPHI,
    GDPC1,
    GROWTH_SERIES,
    ICSA,
    INDPRO,
    NFP_SECTOR_SERIES,
    PANDI,
    PAYEMS,
    SOANDI,
)


NFP_SECTOR_LABELS = {
    "USMINE": "Mining & logging",
    "USCONS": "Construction",
    "MANEMP": "Manufacturing",
    "USTPU": "Trade, transport & utilities",
    "USINFO": "Information",
    "USFIRE": "Financial activities",
    "USPBS": "Professional & business services",
    "USEHS": "Education & health services",
    "USLAH": "Leisure & hospitality",
    "USSERV": "Other services",
    "USGOVT": "Government",
}

CFNAI_CATEGORY_LABELS = {
    PANDI: "Production & income",
    EUANDH: "Employment, unemployment & hours",
    CANDH: "Personal consumption & housing",
    SOANDI: "Sales, orders & inventories",
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


@st.cache_data(
    show_spinner=False,
)
def _prepare_growth_data(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Convert raw FRED observations into monthly growth indicators.

    Returns:
        raw_features:
            Economically interpretable transformed indicators.

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
        gdp_yoy,
    )


def _cfnai_figure(cfnai_ma3: pd.Series) -> go.Figure:
    """Create a chart of the recognized CFNAI three-month average."""
    figure = go.Figure()
    clean = cfnai_ma3.dropna()

    figure.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean,
            mode="lines",
            name="CFNAI · 3M average",
            line={"color": "#3157d5", "width": 2.5},
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.55,
        annotation_text="Trend growth",
        annotation_position="bottom right",
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
        xaxis_title="Date",
        yaxis_title="CFNAI index",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        height=420,
    )

    return figure


def _build_cfnai_category_snapshot(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build the latest official CFNAI category contributions and monthly changes."""
    categories = (
        data[list(CFNAI_CATEGORY_LABELS)]
        .apply(pd.to_numeric, errors="coerce")
        .resample("ME")
        .last()
        .dropna(how="any")
    )
    if categories.empty:
        return pd.DataFrame(), pd.NaT

    latest_date = categories.index.max()
    latest = categories.loc[latest_date]
    prior = categories.shift(1).loc[latest_date]
    rows = []
    for series_id, label in CFNAI_CATEGORY_LABELS.items():
        contribution = float(latest[series_id])
        change = float(latest[series_id] - prior[series_id]) if pd.notna(prior[series_id]) else float("nan")
        rows.append(
            {
                "Category": label,
                "Contribution": contribution,
                "1M change": change,
                "Role": "Support" if contribution > 0 else "Drag" if contribution < 0 else "Neutral",
            }
        )
    return pd.DataFrame(rows), latest_date


def _cfnai_category_figure(snapshot: pd.DataFrame) -> go.Figure:
    """Show the Chicago Fed's four official CFNAI category contributions."""
    ordered = snapshot.sort_values("Contribution", ascending=False)

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=ordered["Category"],
            y=ordered["Contribution"],
            name="Contribution",
            marker_color=[
                "#138a72" if value >= 0 else "#d4534c"
                for value in ordered["Contribution"]
            ],
            text=[f"{value:+.2f}" for value in ordered["Contribution"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>"
                "%{y:+.2f} contribution"
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
        xaxis_title="CFNAI category",
        yaxis_title="Contribution to monthly CFNAI",
        template="plotly_white",
        showlegend=False,
        height=380,
    )

    return figure


def _monthly_payroll_changes(data: pd.DataFrame) -> pd.DataFrame:
    """Return monthly headline and major-industry payroll changes in thousands."""
    series_ids = [PAYEMS, *NFP_SECTOR_SERIES]
    monthly_levels = data[series_ids].apply(pd.to_numeric, errors="coerce").resample("ME").last()
    return monthly_levels.diff()


def _payroll_decomposition_figure(changes: pd.Series, period_label: str) -> go.Figure:
    """Show positive and negative payroll contributions by major industry."""
    labelled = changes.rename(index=NFP_SECTOR_LABELS).dropna().sort_values()
    colors = ["#d4534c" if value < 0 else "#138a72" for value in labelled]

    figure = go.Figure(
        go.Bar(
            x=labelled.values,
            y=labelled.index,
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.0f}k" for value in labelled],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:+.0f}k jobs<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_dash="dash", opacity=0.55)
    figure.update_layout(
        title=f"Payroll Change by Industry · {period_label}",
        xaxis_title="Jobs added or lost (thousands)",
        yaxis_title=None,
        template="plotly_white",
        showlegend=False,
        height=460,
        margin={"l": 10, "r": 55, "t": 55, "b": 45},
    )
    return figure


def _gdp_context_figure(
    cfnai_ma3: pd.Series,
    gdp_yoy: pd.Series,
) -> go.Figure:
    """Compare the CFNAI three-month average with released GDP."""
    figure = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    cfnai_clean = cfnai_ma3.dropna()
    gdp_clean = gdp_yoy.dropna()

    figure.add_trace(
        go.Scatter(
            x=cfnai_clean.index,
            y=cfnai_clean,
            mode="lines",
            name="CFNAI · 3M average",
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
        title="CFNAI and Released Real GDP",
        xaxis_title="Date",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=440,
    )

    figure.update_yaxes(
        title_text="CFNAI index",
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
    st.subheader("Growth Momentum Monitor")

    st.markdown("#### Indicators watched")
    watched_indicators = (
        (
            "Initial claims",
            "LABOUR STRESS",
            "Tracks new unemployment claims; fewer claims indicate stronger labour conditions.",
        ),
        (
            "Industrial production",
            "REAL OUTPUT",
            "Tracks output across manufacturing, mining and utilities.",
        ),
        (
            "Non-Farm Payrolls Growth",
            "HIRING",
            "Tracks the pace of job creation across the economy.",
        ),
        (
            "Philadelphia Fed activity",
            "BUSINESS ACTIVITY",
            "Tracks the direction of regional manufacturing conditions.",
        ),
    )
    for column, (label, category, description) in zip(st.columns(4), watched_indicators):
        with column:
            with st.container(border=True):
                st.caption(category)
                st.markdown(f"**{label}**")
                st.caption(description)

    display_start_date = context[
        "start_date"
    ]

    display_end_date = context[
        "end_date"
    ]

    fetch_start_date = (
        pd.Timestamp(display_start_date) - pd.DateOffset(months=15)
    ).date()

    panel_series = [*GROWTH_SERIES, *NFP_SECTOR_SERIES, *CFNAI_SERIES]
    result = fred_client.get_series(
        panel_series,
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
        for series_id in panel_series
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
        gdp_yoy,
    ) = _prepare_growth_data(
        data
    )

    context.setdefault("panel_history", {})["growth"] = {
        "raw": data.copy(),
        "raw_features": raw_features.copy(),
        "gdp_yoy": gdp_yoy.copy(),
    }

    cfnai_ma3 = pd.to_numeric(data[CFNAIMA3], errors="coerce").dropna()
    cfnai_category_snapshot, cfnai_category_date = _build_cfnai_category_snapshot(data)

    snapshot_as_of = pd.Timestamp(display_end_date)
    claims_4w_average = pd.to_numeric(data[ICSA], errors="coerce").dropna().rolling(4).mean() / 1_000.0
    industrial_production_3m = raw_features["industrial_production_growth"]
    monthly_payrolls = pd.to_numeric(data[PAYEMS], errors="coerce").resample("ME").last()
    payroll_gain_3m_average = monthly_payrolls.diff().rolling(3).mean()
    philly_fed_level = raw_features["philly_fed_activity"]

    latest_claims = _value_as_of(claims_4w_average, snapshot_as_of)
    claims_change = _change_over_months(claims_4w_average, snapshot_as_of, 3)
    latest_industrial_production = _value_as_of(industrial_production_3m, snapshot_as_of)
    industrial_production_change = _change_over_months(industrial_production_3m, snapshot_as_of, 3)
    latest_payroll_gain = _value_as_of(payroll_gain_3m_average, snapshot_as_of)
    payroll_gain_change = _change_over_months(payroll_gain_3m_average, snapshot_as_of, 3)
    latest_philly_fed = _value_as_of(philly_fed_level, snapshot_as_of)
    philly_fed_change = _change_over_months(philly_fed_level, snapshot_as_of, 3)
    monthly_payroll_changes = _monthly_payroll_changes(data)

    display_start_timestamp = pd.Timestamp(
        display_start_date
    )

    display_end_timestamp = pd.Timestamp(
        display_end_date
    )

    displayed_cfnai = cfnai_ma3.loc[
        (
            cfnai_ma3.index
            >= display_start_timestamp
        )
        & (
            cfnai_ma3.index
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

    if displayed_cfnai.empty:
        displayed_cfnai = (
            cfnai_ma3.tail(60)
        )

        displayed_gdp = gdp_yoy.loc[
            displayed_cfnai.index.min():
            displayed_cfnai.index.max()
        ]

    st.markdown("### Latest growth indicators")
    snapshot_metrics = (
        (
            "Initial claims · 4W avg",
            f"{latest_claims:.0f}k" if pd.notna(latest_claims) else "Unavailable",
            f"{claims_change:+.0f}k vs 3M ago" if pd.notna(claims_change) else None,
            "Lower claims generally indicate a firmer labour market.",
            "inverse",
        ),
        (
            "Industrial production · 3M ann.",
            f"{latest_industrial_production:.2f}%" if pd.notna(latest_industrial_production) else "Unavailable",
            f"{industrial_production_change:+.2f} pp vs 3M ago" if pd.notna(industrial_production_change) else None,
            "Higher means production is expanding faster.",
            "normal",
        ),
        (
            "NFP Growth · 3M avg",
            f"{latest_payroll_gain:.0f}k/mo" if pd.notna(latest_payroll_gain) else "Unavailable",
            f"{payroll_gain_change:+.0f}k vs prior 3M" if pd.notna(payroll_gain_change) else None,
            "Higher means a faster monthly pace of job creation.",
            "normal",
        ),
        (
            "Philadelphia Fed activity",
            f"{latest_philly_fed:+.1f}" if pd.notna(latest_philly_fed) else "Unavailable",
            f"{philly_fed_change:+.1f} vs 3M ago" if pd.notna(philly_fed_change) else None,
            "Above zero means more firms report expansion than contraction.",
            "normal",
        ),
    )
    for column, (label, value, delta, description, delta_color) in zip(st.columns(4), snapshot_metrics):
        with column:
            with st.container(border=True):
                st.metric(label, value, delta, delta_color=delta_color)
                st.caption(description)

    st.html(
        """
        <style>
            .growth-nfp-heading {
                margin: .45rem 0 .65rem !important;
                color: #14213d;
                font-size: 1.18rem !important;
                font-weight: 650;
                letter-spacing: -.025em;
                line-height: 1.3;
            }
        </style>
        <h3 class="growth-nfp-heading">NFP industry breakdown</h3>
        """
    )
    complete_monthly_payroll_changes = monthly_payroll_changes.dropna(
        subset=[PAYEMS, *NFP_SECTOR_SERIES]
    )
    if complete_monthly_payroll_changes.empty:
        st.info("Industry payroll decomposition is unavailable for the selected period.")
    else:
        available_months = complete_monthly_payroll_changes.loc[
            (complete_monthly_payroll_changes.index >= display_start_timestamp)
            & (complete_monthly_payroll_changes.index <= display_end_timestamp)
        ].index.tolist()
        if not available_months:
            available_months = complete_monthly_payroll_changes.tail(60).index.tolist()
        month_options = list(reversed(available_months))

        quick_view_column, month_column = st.columns([3, 2])
        with quick_view_column:
            payroll_view = st.segmented_control(
                "View",
                options=["Latest month", "3M average", "Selected month"],
                default="Latest month",
                key="growth_payroll_decomposition_period",
                width="stretch",
            )
        with month_column:
            selected_month = st.selectbox(
                "Month",
                options=month_options,
                format_func=lambda value: value.strftime("%b %Y"),
                key="growth_payroll_selected_month",
                disabled=payroll_view != "Selected month",
                width="stretch",
            )

        if payroll_view == "3M average":
            complete_payroll_changes = monthly_payroll_changes.rolling(3).mean().dropna(
                subset=[PAYEMS, *NFP_SECTOR_SERIES]
            )
            payroll_period_end = complete_payroll_changes.index.max()
            payroll_row = complete_payroll_changes.loc[payroll_period_end]
            period_label = f"3M average through {payroll_period_end:%b %Y}"
        elif payroll_view == "Selected month":
            payroll_period_end = pd.Timestamp(selected_month)
            payroll_row = complete_monthly_payroll_changes.loc[payroll_period_end]
            period_label = payroll_period_end.strftime("%b %Y")
        else:
            payroll_period_end = complete_monthly_payroll_changes.index.max()
            payroll_row = complete_monthly_payroll_changes.loc[payroll_period_end]
            period_label = payroll_period_end.strftime("%b %Y")

        headline_payroll_change = float(payroll_row[PAYEMS])
        sector_payroll_changes = payroll_row[NFP_SECTOR_SERIES]
        sector_sum = float(sector_payroll_changes.sum())
        st.caption(
            f"Headline NFP: {headline_payroll_change:+.0f}k · "
            f"Major-industry sum: {sector_sum:+.0f}k. "
            "Green sectors added jobs; red sectors lost jobs."
        )
        st.plotly_chart(
            _payroll_decomposition_figure(sector_payroll_changes, period_label),
            width="stretch",
        )

    st.markdown("### Growth Drivers - CFNAI (Chicago Fed National Activity Index)")
    if cfnai_category_snapshot.empty:
        st.info("CFNAI category contributions are unavailable for the selected period.")
    else:
        drivers_column, table_column = st.columns(2)
        with drivers_column:
            st.plotly_chart(
                _cfnai_category_figure(cfnai_category_snapshot),
                width="stretch",
            )
        with table_column:
            st.markdown("#### What drives the index")
            st.dataframe(
                cfnai_category_snapshot,
                hide_index=True,
                width="stretch",
                height=330,
                column_config={
                    "Contribution": st.column_config.NumberColumn(format="%+.2f"),
                    "1M change": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
        category_sum = float(cfnai_category_snapshot["Contribution"].sum())
        monthly_cfnai = _value_as_of(pd.to_numeric(data[CFNAI], errors="coerce"), cfnai_category_date)
        st.caption(
            f"Official Chicago Fed category contributions for {cfnai_category_date:%b %Y}. "
            f"They sum to {category_sum:+.2f}, versus the published monthly CFNAI of {monthly_cfnai:+.2f}. "
            "Positive contributions support above-trend activity; negative contributions are drags."
        )

    st.html(
        """
        <style>
            .growth-composite-heading {
                display: inline-flex; align-items: center; gap: .5rem;
                margin: 1.85rem 0 .7rem;
            }
            .growth-composite-title {
                color: #14213d; font-size: 1.18rem; font-weight: 650;
                letter-spacing: -.025em; line-height: 1.25;
            }
            .growth-composite-info {
                position: relative; display: inline-flex; align-items: center;
                justify-content: center; width: 1.15rem; height: 1.15rem;
                border: 1px solid #94a3b8; border-radius: 50%; color: #64748b;
                font-size: .72rem; font-weight: 750; cursor: help;
            }
            .growth-composite-tooltip {
                position: absolute; z-index: 30; top: 1.55rem; left: 50%;
                width: min(36rem, 82vw); padding: .85rem .95rem;
                border: 1px solid #dfe5ee; border-radius: .65rem;
                background: #fff; color: #334155;
                box-shadow: 0 10px 30px rgba(30,47,78,.16);
                font-size: .76rem; font-weight: 400; line-height: 1.48;
                box-sizing: border-box; white-space: normal; overflow-wrap: anywhere;
                opacity: 0; visibility: hidden; transform: translate(-12%, -.25rem);
                transition: opacity .12s ease, transform .12s ease;
            }
            .growth-composite-tooltip strong { color: #14213d; }
            .growth-composite-tooltip div + div { margin-top: .42rem; }
            .growth-composite-info:hover .growth-composite-tooltip,
            .growth-composite-info:focus .growth-composite-tooltip {
                opacity: 1; visibility: visible; transform: translate(-12%, 0);
            }
        </style>
        <div class="growth-composite-heading">
            <span class="growth-composite-title">Chicago Fed National Activity Index · 3M Average</span>
            <span class="growth-composite-info" tabindex="0" aria-label="Explain the Chicago Fed National Activity Index">
                i
                <span class="growth-composite-tooltip" role="tooltip">
                    <div><strong>What it is:</strong> a Federal Reserve Bank of Chicago index built from 85 indicators of national economic activity.</div>
                    <div><strong>How it is built:</strong> the first principal component captures the common movement across production and income; employment, unemployment and hours; consumption and housing; and sales, orders and inventories.</div>
                    <div><strong>How to read it:</strong> zero represents trend growth, positive values indicate above-trend activity and negative values indicate below-trend activity.</div>
                    <div><strong>Why 3M:</strong> the three-month moving average reduces volatility in the monthly index.</div>
                </span>
            </span>
        </div>
        """
    )
    st.plotly_chart(
        _cfnai_figure(displayed_cfnai),
        width="stretch",
    )

    st.markdown("### GDP Cross-Check")
    st.caption(
        "Compare CFNAI with released real GDP growth to see whether the monthly activity signal is "
        "consistent with the broader economy. Alignment adds confidence; divergence calls for caution, "
        "but does not automatically invalidate CFNAI."
    )
    st.plotly_chart(
        _gdp_context_figure(displayed_cfnai, displayed_gdp),
        width="stretch",
    )
