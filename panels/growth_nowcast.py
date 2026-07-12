from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    GACDISA066MSFRBPHI,
    GDPC1,
    GROWTH_SERIES,
    ICSA,
    INDPRO,
    PAYEMS,
)


# Global controls determine the chart display range.
# The model independently fetches a longer history for training.
MODEL_START_DATE = date(2000, 1, 1)

# Five years of quarterly data before the first historical prediction.
MIN_TRAINING_QUARTERS = 20

FEATURE_COLUMNS = [
    "claims_growth",
    "industrial_production_growth",
    "payroll_growth",
    "philly_fed_activity",
]


def _quarter_label(timestamp: pd.Timestamp) -> str:
    """Convert a timestamp into a readable quarter label."""
    return f"Q{timestamp.quarter} {timestamp.year}"


def _annualized_three_month_growth(
    series: pd.Series,
) -> pd.Series:
    """
    Calculate annualized growth over the previous three months.

    Formula:
        ((current / three_months_ago) ** 4 - 1) * 100
    """
    growth = ((series / series.shift(3)) ** 4 - 1.0) * 100.0

    return growth.replace(
        [np.inf, -np.inf],
        np.nan,
    )


@st.cache_data(show_spinner=False)
def _prepare_model_inputs(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convert the raw FRED data into quarterly model features
    and a quarterly GDP YoY target.
    """
    monthly = pd.DataFrame(
        {
            # Weekly initial claims averaged within each month.
            ICSA: data[ICSA].resample("ME").mean(),

            # Monthly index and employment levels.
            INDPRO: data[INDPRO].resample("ME").last(),
            PAYEMS: data[PAYEMS].resample("ME").last(),

            # Monthly Philadelphia Fed diffusion index.
            GACDISA066MSFRBPHI: (
                data[GACDISA066MSFRBPHI]
                .resample("ME")
                .mean()
            ),
        }
    )

    features_monthly = pd.DataFrame(
        index=monthly.index
    )

    # Higher claims normally indicate weaker growth,
    # so the growth rate is inverted.
    features_monthly["claims_growth"] = (
        -_annualized_three_month_growth(
            monthly[ICSA]
        )
    )

    features_monthly[
        "industrial_production_growth"
    ] = _annualized_three_month_growth(
        monthly[INDPRO]
    )

    features_monthly[
        "payroll_growth"
    ] = _annualized_three_month_growth(
        monthly[PAYEMS]
    )

    # The Philadelphia Fed series is a diffusion index.
    # Use its level rather than percentage change because
    # the index can be zero or negative.
    features_monthly[
        "philly_fed_activity"
    ] = monthly[GACDISA066MSFRBPHI]

    features_monthly = features_monthly.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Aggregate monthly indicators into quarterly averages.
    features_quarterly = (
        features_monthly
        .resample("QE-DEC")
        .mean()
    )

    # Convert quarterly real GDP into YoY percentage growth.
    real_gdp_quarterly = (
        data[GDPC1]
        .resample("QE-DEC")
        .last()
    )

    gdp_yoy = (
        real_gdp_quarterly.pct_change(4) * 100.0
    )

    gdp_yoy.name = "gdp_yoy"

    return features_quarterly, gdp_yoy


def _build_model():
    """Create the standardised linear regression model."""
    return make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )


@st.cache_data(show_spinner=False)
def _generate_historical_predictions(
    model_data: pd.DataFrame,
) -> pd.Series:
    """
    Generate expanding-window historical predictions.

    Each quarter is predicted using only data that would have
    been available before that quarter.
    """
    predictions = pd.Series(
        index=model_data.index,
        dtype=float,
        name="historical_nowcast",
    )

    for position in range(
        MIN_TRAINING_QUARTERS,
        len(model_data),
    ):
        training_data = model_data.iloc[:position]
        prediction_data = model_data.iloc[[position]]

        model = _build_model()

        model.fit(
            training_data[FEATURE_COLUMNS],
            training_data["gdp_yoy"],
        )

        prediction = model.predict(
            prediction_data[FEATURE_COLUMNS]
        )[0]

        predictions.iloc[position] = prediction

    return predictions


def render(
    fred_client,
    context: dict,
) -> None:
    st.subheader("Panel 4: Growth Nowcast")

    st.caption(
        "Methodology: linear regression of real GDP YoY on "
        "initial claims, industrial production, payroll growth "
        "and the Philadelphia Fed activity index. The global "
        "date range controls the chart display, while the model "
        "uses data from 2000 onward for training."
    )

    display_start_date = context["start_date"]
    display_end_date = context["end_date"]

    # Fetch a long model-training history regardless of the
    # date range selected in the global sidebar controls.
    fetch_start_date = min(
        MODEL_START_DATE,
        display_start_date,
    )

    result = fred_client.get_series(
        GROWTH_SERIES,
        fetch_start_date,
        display_end_date,
    )

    if not result.success or result.data is None:
        st.warning(
            result.message
            or "Growth data are unavailable."
        )
        return

    data = result.data.copy()

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
            + ", ".join(missing_series)
        )
        return

    features_quarterly, gdp_yoy = (
        _prepare_model_inputs(data)
    )

    model_data = (
        features_quarterly
        .join(
            gdp_yoy,
            how="inner",
        )
        .dropna(
            subset=FEATURE_COLUMNS + ["gdp_yoy"]
        )
    )

    minimum_observations = (
        MIN_TRAINING_QUARTERS + 1
    )

    if len(model_data) < minimum_observations:
        st.warning(
            "Not enough observations to train the growth "
            f"model. Found {len(model_data)} usable quarters; "
            f"at least {minimum_observations} are required."
        )
        return

    historical_predictions = (
        _generate_historical_predictions(
            model_data
        )
    )

    evaluation_data = pd.DataFrame(
        {
            "actual": model_data["gdp_yoy"],
            "prediction": historical_predictions,
        }
    ).dropna()

    if evaluation_data.empty:
        st.warning(
            "The model could not generate historical "
            "predictions."
        )
        return

    # Train the latest model using all quarters for which
    # an actual GDP observation is available.
    final_model = _build_model()

    final_model.fit(
        model_data[FEATURE_COLUMNS],
        model_data["gdp_yoy"],
    )

    available_features = (
        features_quarterly
        .dropna(subset=FEATURE_COLUMNS)
    )

    if available_features.empty:
        st.warning(
            "No complete indicator observations are available."
        )
        return

    latest_feature_date = (
        available_features.index[-1]
    )

    latest_feature_row = (
        available_features.iloc[[-1]]
    )

    latest_nowcast = float(
        final_model.predict(
            latest_feature_row[FEATURE_COLUMNS]
        )[0]
    )

    if len(available_features) >= 2:
        previous_feature_row = (
            available_features.iloc[[-2]]
        )

        previous_nowcast = float(
            final_model.predict(
                previous_feature_row[FEATURE_COLUMNS]
            )[0]
        )

        nowcast_change = (
            latest_nowcast - previous_nowcast
        )
    else:
        previous_nowcast = np.nan
        nowcast_change = np.nan

    if pd.notna(previous_nowcast):
        direction = (
            "Accelerating"
            if latest_nowcast >= previous_nowcast
            else "Decelerating"
        )
    else:
        direction = "Unavailable"

    mae = mean_absolute_error(
        evaluation_data["actual"],
        evaluation_data["prediction"],
    )

    rmse = np.sqrt(
        mean_squared_error(
            evaluation_data["actual"],
            evaluation_data["prediction"],
        )
    )

    chart_data = pd.DataFrame(
        {
            "Actual GDP YoY": gdp_yoy,
            "Historical nowcast": historical_predictions,
        }
    )

    display_start_timestamp = pd.Timestamp(
        display_start_date
    )

    display_end_timestamp = pd.Timestamp(
        display_end_date
    )

    # The sidebar dates only filter the displayed chart.
    chart_data = chart_data.loc[
        (chart_data.index >= display_start_timestamp)
        & (chart_data.index <= display_end_timestamp)
    ]

    # Fall back to the most recent 12 quarters if the selected
    # period has no completed GDP observations.
    if chart_data.dropna(how="all").empty:
        chart_data = pd.DataFrame(
            {
                "Actual GDP YoY": gdp_yoy,
                "Historical nowcast": (
                    historical_predictions
                ),
            }
        ).tail(12)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Actual GDP YoY"],
            mode="lines+markers",
            name="Actual GDP YoY",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["Historical nowcast"],
            mode="lines+markers",
            name="Historical out-of-sample nowcast",
        )
    )

    # Add the latest model estimate separately because current
    # GDP may not have been released yet.
    figure.add_trace(
        go.Scatter(
            x=[latest_feature_date],
            y=[latest_nowcast],
            mode="markers",
            marker={"size": 12, "symbol": "diamond"},
            name="Latest model estimate",
        )
    )

    figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.5,
    )

    figure.update_layout(
        title="GDP Growth Nowcast vs Actual",
        xaxis_title="Quarter",
        yaxis_title="Real GDP growth (% YoY)",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        label=(
            "Latest estimate "
            f"({_quarter_label(latest_feature_date)})"
        ),
        value=f"{latest_nowcast:.2f}%",
        delta=(
            f"{nowcast_change:+.2f} pp vs prior quarter"
            if pd.notna(nowcast_change)
            else None
        ),
    )

    metric_2.metric(
        label="Growth direction",
        value=direction,
    )

    metric_3.metric(
        label="Backtest MAE",
        value=f"{mae:.2f} pp",
    )

    metric_4.metric(
        label="Backtest RMSE",
        value=f"{rmse:.2f} pp",
    )

    released_gdp = gdp_yoy.dropna()

    if not released_gdp.empty:
        latest_gdp_date = released_gdp.index[-1]
        latest_gdp_value = float(
            released_gdp.iloc[-1]
        )

        st.caption(
            "Latest released real GDP observation: "
            f"{_quarter_label(latest_gdp_date)}, "
            f"{latest_gdp_value:.2f}% YoY. "
            "Latest indicator estimate: "
            f"{_quarter_label(latest_feature_date)}."
        )

    if latest_feature_date.date() > display_end_date:
        st.info(
            f"The {_quarter_label(latest_feature_date)} "
            "estimate is based on partial-quarter data "
            f"available through "
            f"{display_end_date:%d %b %Y}."
        )
