from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    CES0500000003,
    CCSA,
    CIVPART,
    EMRATIO,
    ICSA,
    JTSJOL,
    JTSQUR,
    LABOR_SERIES,
    LNS11300060,
    PAYEMS,
    UNEMPLOY,
    UNRATE,
)


ANALYSIS_START_DATE = date(2000, 1, 1)

PAYROLL_TREND_MONTHS = 6
CLAIMS_MOVING_AVERAGE_WEEKS = 4
WAGE_SHORT_TERM_MONTHS = 3
WAGE_YEAR_OVER_YEAR_MONTHS = 12

UNRATE_STABLE_THRESHOLD_PP = 0.1
UNRATE_MODERATE_RISE_PP = 0.2
UNRATE_MATERIAL_RISE_PP = 0.4

CLAIMS_MILD_RISE_K = 10.0
CLAIMS_MATERIAL_RISE_K = 25.0

PAYROLL_TREND_MARGIN_K = 25.0
WAGE_MOM_FIRM_PCT = 0.3
WAGE_COOLING_THRESHOLD_PCT = 0.2

OPENINGS_RATIO_TIGHT_THRESHOLD = 1.2
OPENINGS_RATIO_NORMAL_THRESHOLD = 1.0

QUITS_FIRM_THRESHOLD_PP = 0.0
QUITS_COOLING_THRESHOLD_PP = -0.1

METRIC_CAPTION_MIN_HEIGHT_PX = 88


def _latest_valid_date(series: pd.Series) -> pd.Timestamp | None:
    """Return the latest non-null observation date for a series."""
    clean = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return None

    return pd.Timestamp(clean.index.max())


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
    """Return the latest value minus an earlier value."""
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
    labels = {
        "1M": "over the past month",
        "3M": "over the past three months",
        "6M": "over the past six months",
    }

    if comparison_type in labels:
        return labels[comparison_type]

    return f"since {comparison_date:%d %b %Y}"


def _format_thousands(
    value: float,
    digits: int = 0,
) -> str:
    """Format a value in thousands."""
    if pd.isna(value):
        return "Unavailable"

    return f"{value:+.{digits}f}k"


def _format_percent(
    value: float,
    digits: int = 1,
) -> str:
    """Format a percentage value."""
    if pd.isna(value):
        return "Unavailable"

    return f"{value:.{digits}f}%"


def _format_ratio(
    value: float,
    digits: int = 2,
) -> str:
    """Format a ratio."""
    if pd.isna(value):
        return "Unavailable"

    return f"{value:.{digits}f}x"


def _format_delta(
    value: float,
    suffix: str = "",
    digits: int = 1,
) -> str | None:
    """Format a signed metric delta."""
    if pd.isna(value):
        return None

    return f"{value:+.{digits}f}{suffix}"


def _render_metric_caption(text: str) -> None:
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


def _monthly_series(
    data: pd.DataFrame,
    series_id: str,
) -> pd.Series:
    """Return a month-end series for a given FRED series."""
    return (
        pd.to_numeric(
            data[series_id],
            errors="coerce",
        )
        .resample("ME")
        .last()
        .astype("float64")
    )


def _weekly_series(
    data: pd.DataFrame,
    series_id: str,
) -> pd.Series:
    """Return a cleaned weekly series."""
    return pd.to_numeric(
        data[series_id],
        errors="coerce",
    ).astype("float64")


def _latest_available_date(*series_list: pd.Series) -> pd.Timestamp | None:
    """Return the latest non-null observation date across several series."""
    dates = [
        series.dropna().index.max()
        for series in series_list
        if not series.dropna().empty
    ]

    if not dates:
        return None

    return pd.Timestamp(max(dates))


def _moving_average(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """Return a rolling moving average."""
    return series.rolling(
        window=window,
        min_periods=window,
    ).mean()


def _weekly_average(
    series: pd.Series,
    latest_date: pd.Timestamp,
    weeks: int = CLAIMS_MOVING_AVERAGE_WEEKS,
) -> float:
    """Return the latest rolling weekly average."""
    average = _moving_average(
        series,
        weeks,
    )

    return _value_as_of(
        average,
        latest_date,
    )


def _change_direction(
    change: float,
    positive_text: str,
    negative_text: str,
    digits: int = 1,
    suffix: str = "",
) -> str:
    """Return a concise directional description for a change."""
    if pd.isna(change):
        return "change unavailable"

    if change > 0:
        return f"{positive_text} {abs(change):.{digits}f}{suffix}"

    if change < 0:
        return f"{negative_text} {abs(change):.{digits}f}{suffix}"

    return f"unchanged {suffix}".strip()


def _trend_flag(value: float, positive_label: str, negative_label: str) -> str:
    """Return a simple qualitative trend flag."""
    if pd.isna(value):
        return "Unavailable"

    if value > 0:
        return positive_label

    if value < 0:
        return negative_label

    return "Stable"


def _supporting_evidence_lines(
    payroll_change_k: float,
    payroll_trailing_avg_k: float,
    unrate_level: float,
    unrate_change_3m_pp: float,
    civpart_level: float,
    civpart_change_3m_pp: float,
    wage_mom_pct: float,
    wage_3m_ann_pct: float,
    wage_yoy_pct: float,
    claims_4w_avg_k: float,
    claims_change_1m_k: float,
    continuing_claims_k: float,
    continuing_claims_change_1m_k: float,
    openings_ratio: float,
    openings_ratio_change_3m: float,
    quits_change_3m_pp: float,
) -> list[str]:
    """Build concise evidence bullets for the labour market panel."""
    claims_direction = (
        "rising"
        if (
            pd.notna(claims_change_1m_k)
            and claims_change_1m_k > 0
        )
        else (
            "falling"
            if (
                pd.notna(claims_change_1m_k)
                and claims_change_1m_k < 0
            )
            else "stable"
        )
    )

    quits_direction = _trend_flag(
        quits_change_3m_pp,
        "rising",
        "easing",
    )

    return [
        (
            f"- Payrolls: {_format_thousands(payroll_change_k)} versus "
            f"a six-month average of {_format_thousands(payroll_trailing_avg_k)}"
        ),
        (
            f"- Unemployment: {_format_percent(unrate_level)}, "
            f"{_change_direction(unrate_change_3m_pp, 'up', 'down', digits=1, suffix=' pp over three months')}"
            f"; participation: {_format_percent(civpart_level)}, "
            f"{_change_direction(civpart_change_3m_pp, 'up', 'down', digits=1, suffix=' pp over three months')}"
        ),
        (
            f"- Wages: {_format_percent(wage_mom_pct)} MoM, "
            f"{_format_percent(wage_3m_ann_pct)} annualised over three months, "
            f"{_format_percent(wage_yoy_pct)} YoY"
        ),
        (
            f"- Claims: {_format_thousands(claims_4w_avg_k)} four-week average, "
            f"continuing claims {_format_thousands(continuing_claims_k)}, "
            f"{_change_direction(claims_change_1m_k, 'up', 'down', suffix=' vs one month ago')}"
            f"; continuing claims {_change_direction(continuing_claims_change_1m_k, 'up', 'down', suffix=' over one month')}"
        ),
        (
            f"- Labour tightness: {_format_ratio(openings_ratio)}, "
            f"{_change_direction(openings_ratio_change_3m, 'up', 'down', digits=2, suffix=' versus three months ago')}; "
            f"quits are {quits_direction}"
        ),
    ]


def _classify_labor_regime(
    payroll_change_k: float,
    payroll_trailing_avg_k: float,
    unrate_change_3m_pp: float,
    claims_change_1m_k: float,
    continuing_claims_change_1m_k: float,
    wage_mom_pct: float,
    wage_3m_ann_pct: float,
    wage_yoy_pct: float,
    openings_ratio: float,
    openings_ratio_change_3m: float,
    quits_change_3m_pp: float,
) -> tuple[str, str]:
    """Classify the labour market using transparent, descriptive rules."""
    if all(
        pd.isna(value)
        for value in [
            payroll_change_k,
            payroll_trailing_avg_k,
            unrate_change_3m_pp,
            claims_change_1m_k,
            continuing_claims_change_1m_k,
            wage_mom_pct,
            wage_3m_ann_pct,
            wage_yoy_pct,
            openings_ratio,
            openings_ratio_change_3m,
            quits_change_3m_pp,
        ]
    ):
        return (
            "Unavailable",
            "Insufficient labour-market history is available to classify the regime.",
        )

    payroll_above_trend = (
        pd.notna(payroll_change_k)
        and pd.notna(payroll_trailing_avg_k)
        and payroll_change_k >= payroll_trailing_avg_k + PAYROLL_TREND_MARGIN_K
    )
    payroll_positive = pd.notna(payroll_change_k) and payroll_change_k > 0
    payroll_weak = pd.notna(payroll_change_k) and payroll_change_k <= 0

    unemployment_stable_or_lower = (
        pd.notna(unrate_change_3m_pp)
        and unrate_change_3m_pp <= UNRATE_STABLE_THRESHOLD_PP
    )
    unemployment_modestly_higher = (
        pd.notna(unrate_change_3m_pp)
        and unrate_change_3m_pp > UNRATE_STABLE_THRESHOLD_PP
        and unrate_change_3m_pp <= UNRATE_MODERATE_RISE_PP
    )
    unemployment_materially_higher = (
        pd.notna(unrate_change_3m_pp)
        and unrate_change_3m_pp >= UNRATE_MATERIAL_RISE_PP
    )

    claims_stable_or_lower = (
        pd.notna(claims_change_1m_k)
        and claims_change_1m_k <= 0
        and pd.notna(continuing_claims_change_1m_k)
        and continuing_claims_change_1m_k <= 0
    )
    claims_mildly_higher = (
        pd.notna(claims_change_1m_k)
        and claims_change_1m_k > 0
        and claims_change_1m_k <= CLAIMS_MILD_RISE_K
    )
    claims_materially_higher = (
        pd.notna(claims_change_1m_k)
        and claims_change_1m_k >= CLAIMS_MATERIAL_RISE_K
    ) or (
        pd.notna(continuing_claims_change_1m_k)
        and continuing_claims_change_1m_k >= CLAIMS_MATERIAL_RISE_K
    )

    wage_firm = (
        pd.notna(wage_mom_pct)
        and pd.notna(wage_3m_ann_pct)
        and pd.notna(wage_yoy_pct)
        and wage_mom_pct >= WAGE_MOM_FIRM_PCT
        and wage_3m_ann_pct >= wage_yoy_pct - 0.1
    )
    wage_cooling = (
        pd.notna(wage_mom_pct)
        and wage_mom_pct <= WAGE_COOLING_THRESHOLD_PCT
    ) or (
        pd.notna(wage_3m_ann_pct)
        and pd.notna(wage_yoy_pct)
        and wage_3m_ann_pct < wage_yoy_pct
    )

    ratio_elevated = (
        pd.notna(openings_ratio)
        and openings_ratio >= OPENINGS_RATIO_TIGHT_THRESHOLD
    )
    ratio_normalising = (
        pd.notna(openings_ratio_change_3m)
        and openings_ratio_change_3m < 0
    ) or (
        pd.notna(openings_ratio)
        and openings_ratio <= OPENINGS_RATIO_NORMAL_THRESHOLD
    )
    quits_falling = (
        pd.notna(quits_change_3m_pp)
        and quits_change_3m_pp <= QUITS_COOLING_THRESHOLD_PP
    )
    quits_firm = (
        pd.notna(quits_change_3m_pp)
        and quits_change_3m_pp >= QUITS_FIRM_THRESHOLD_PP
    )

    tightening_votes = sum(
        [
            payroll_above_trend,
            unemployment_stable_or_lower,
            claims_stable_or_lower,
            wage_firm,
            ratio_elevated,
            quits_firm,
        ]
    )

    weakening_votes = sum(
        [
            payroll_weak,
            unemployment_materially_higher,
            claims_materially_higher,
            wage_cooling,
            ratio_normalising,
            quits_falling,
        ]
    )

    if (
        tightening_votes >= 5
        and weakening_votes <= 1
    ):
        return (
            "Tight and reaccelerating",
            (
                "Payroll growth is running above recent trend, unemployment is stable or lower, "
                "claims are not worsening, wage pressure is firm, and vacancies remain elevated. "
                "This is a descriptive labour-market reading, not a formal recession signal or a Fed forecast."
            ),
        )

    if (
        payroll_positive
        and not unemployment_materially_higher
        and (claims_mildly_higher or claims_stable_or_lower)
        and wage_cooling
        and ratio_normalising
        and quits_falling
    ):
        return (
            "Tight but cooling",
            (
                "Payrolls are still growing, unemployment is not materially rising, claims are only "
                "mildly higher, wage growth is moderating, and vacancies and quits are easing. "
                "This is a descriptive labour-market reading, not a formal recession signal or a Fed forecast."
            ),
        )

    if (
        payroll_positive
        and not payroll_above_trend
        and unemployment_modestly_higher
        and (claims_mildly_higher or claims_stable_or_lower)
        and wage_cooling
        and ratio_normalising
    ):
        return (
            "Gradual rebalancing",
            (
                "Payroll growth remains positive but below recent trend, unemployment is rising modestly, "
                "claims are moving higher, wage growth is cooling, and vacancies are normalising. "
                "This is a descriptive labour-market reading, not a formal recession signal or a Fed forecast."
            ),
        )

    if (
        (payroll_weak or not payroll_positive)
        and unemployment_materially_higher
        and claims_materially_higher
        and wage_cooling
        and ratio_normalising
        and weakening_votes >= 4
    ):
        return (
            "Broad labour-market weakening",
            (
                "Payroll growth is weak or negative, unemployment is materially higher, claims are rising, "
                "wage growth is softening, and vacancies are easing. "
                "This is a descriptive labour-market reading, not a formal recession signal or a Fed forecast."
            ),
        )

    if tightening_votes >= weakening_votes + 2:
        return (
            "Tight but cooling",
            (
                "The labour market still looks relatively tight, but the balance of indicators is moving "
                "toward softer hiring, slower wages and less vacancy pressure. "
                "This is a descriptive labour-market reading, not a formal recession signal or a Fed forecast."
            ),
        )

    if weakening_votes >= tightening_votes + 2:
        return (
            "Gradual rebalancing",
            (
                "The labour market appears to be rebalancing as hiring cools, claims edge up and vacancies "
                "ease. That combination may reduce wage pressure, but it does not by itself imply a recession or a Fed decision."
            ),
        )

    return (
        "Mixed labour signals",
        (
            "The available labour-market indicators do not point cleanly in one direction. "
            "This is a descriptive labour-market reading, not a formal recession signal or a Fed forecast."
        ),
    )


def _build_macro_note(
    latest_monthly_date: pd.Timestamp,
    regime: str,
    payroll_change_k: float,
    payroll_trailing_avg_k: float,
    unrate_level: float,
    unrate_change_3m_pp: float,
    civpart_level: float,
    civpart_change_3m_pp: float,
    claims_4w_avg_k: float,
    claims_change_1m_k: float,
    continuing_claims_k: float,
    continuing_claims_change_1m_k: float,
    wage_mom_pct: float,
    wage_3m_ann_pct: float,
    wage_yoy_pct: float,
    openings_ratio: float,
    openings_ratio_change_3m: float,
    quits_rate_level: float,
    quits_change_3m_pp: float,
) -> str:
    """Build the generated labour-market note."""
    payroll_clause = "remained positive"
    if pd.notna(payroll_change_k) and pd.notna(payroll_trailing_avg_k):
        if payroll_change_k >= payroll_trailing_avg_k:
            payroll_clause = (
                f"remained positive and ran above its six-month average"
            )
        else:
            payroll_clause = (
                f"remained positive but ran below its six-month average"
            )
    elif pd.notna(payroll_change_k):
        payroll_clause = (
            f"registered a monthly change of {_format_thousands(payroll_change_k)}"
        )

    unemployment_clause = "was broadly stable"
    if pd.notna(unrate_change_3m_pp):
        if unrate_change_3m_pp >= UNRATE_MODERATE_RISE_PP:
            unemployment_clause = (
                f"rose modestly over the past three months"
            )
        elif unrate_change_3m_pp <= -UNRATE_STABLE_THRESHOLD_PP:
            unemployment_clause = (
                f"edged lower over the past three months"
            )

    participation_clause = "Participation was broadly stable."
    if pd.notna(civpart_change_3m_pp):
        if civpart_change_3m_pp > 0:
            participation_clause = (
                "Participation also rose, suggesting labour supply contributed to the unemployment increase."
            )
        elif civpart_change_3m_pp < 0:
            participation_clause = (
                "Participation slipped, which can help explain part of the labour-market slack."
            )

    claims_clause = "Claims were broadly stable"
    if pd.notna(claims_change_1m_k) or pd.notna(continuing_claims_change_1m_k):
        if (
            pd.notna(claims_change_1m_k)
            and claims_change_1m_k > 0
        ) or (
            pd.notna(continuing_claims_change_1m_k)
            and continuing_claims_change_1m_k > 0
        ):
            claims_clause = "Initial and continuing claims moved higher"
        elif (
            pd.notna(claims_change_1m_k)
            and claims_change_1m_k < 0
        ) or (
            pd.notna(continuing_claims_change_1m_k)
            and continuing_claims_change_1m_k < 0
        ):
            claims_clause = "Initial and continuing claims eased"

    wage_clause = "wage growth was broadly steady"
    if pd.notna(wage_mom_pct) and pd.notna(wage_3m_ann_pct) and pd.notna(wage_yoy_pct):
        if wage_3m_ann_pct < wage_yoy_pct:
            wage_clause = "wage growth continued to moderate"
        elif wage_3m_ann_pct > wage_yoy_pct:
            wage_clause = "wage growth remained firm"

    tightness_clause = "vacancies and quits were broadly stable"
    if pd.notna(openings_ratio_change_3m) and pd.notna(quits_change_3m_pp):
        if openings_ratio_change_3m < 0 and quits_change_3m_pp <= 0:
            tightness_clause = "vacancies and quits eased, pointing to less labour-market tightness"
        elif openings_ratio_change_3m > 0 and quits_change_3m_pp >= 0:
            tightness_clause = "vacancies and quits remained elevated"

    policy_clause = (
        "is consistent with a less restrictive policy signal, although it does not imply any specific Fed decision"
    )
    if regime == "Tight and reaccelerating":
        policy_clause = (
            "could keep the policy-sensitive front end attentive to sticky labour pressure, although it still does not imply any specific Fed decision"
        )
    elif regime == "Broad labour-market weakening":
        policy_clause = (
            "is consistent with a softer policy signal for the front end, although it still does not imply any specific Fed decision"
        )

    return (
        f"As of {latest_monthly_date:%b %Y}, the labour market is classified as {regime.lower()}. "
        f"Payroll growth {payroll_clause}, while unemployment {unemployment_clause}. "
        f"{participation_clause} "
        f"{claims_clause.lower()}, wage growth {wage_clause}, and {tightness_clause}. "
        f"Overall, the mix {policy_clause}."
    )


def _payroll_figure(
    payroll_change: pd.Series,
    payroll_ma_3m: pd.Series,
    payroll_ma_6m: pd.Series,
    display_start_date: date,
    display_end_date: date,
) -> go.Figure:
    """Create the payroll momentum chart."""
    window = payroll_change.loc[display_start_date:display_end_date]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=window.index,
            y=window,
            name="Monthly change",
            marker_color="rgba(55, 83, 109, 0.55)",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=payroll_ma_3m.loc[display_start_date:display_end_date].index,
            y=payroll_ma_3m.loc[display_start_date:display_end_date],
            mode="lines",
            name="3M average",
            line={"width": 2.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=payroll_ma_6m.loc[display_start_date:display_end_date].index,
            y=payroll_ma_6m.loc[display_start_date:display_end_date],
            mode="lines",
            name="6M average",
            line={"width": 2.5, "dash": "dash"},
        )
    )

    figure.update_layout(
        title="Payroll momentum",
        xaxis_title="Date",
        yaxis_title="Thousands of jobs",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=420,
    )
    return figure


def _unemployment_figure(
    unrate: pd.Series,
    civpart: pd.Series,
    emratio: pd.Series | None,
    prime_age_participation: pd.Series | None,
    display_start_date: date,
    display_end_date: date,
) -> go.Figure:
    """Create the unemployment and participation chart."""
    figure = make_subplots(
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ]
        ]
    )

    window_unrate = unrate.loc[display_start_date:display_end_date]
    window_civpart = civpart.loc[display_start_date:display_end_date]

    figure.add_trace(
        go.Scatter(
            x=window_unrate.index,
            y=window_unrate,
            mode="lines",
            name="Unemployment rate",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=window_civpart.index,
            y=window_civpart,
            mode="lines",
            name="Participation rate",
            line={"dash": "dash"},
        ),
        secondary_y=True,
    )

    if emratio is not None and not emratio.dropna().empty:
        figure.add_trace(
            go.Scatter(
                x=emratio.loc[display_start_date:display_end_date].index,
                y=emratio.loc[display_start_date:display_end_date],
                mode="lines",
                name="Employment-population ratio",
                line={"dash": "dot"},
            ),
            secondary_y=True,
        )

    if prime_age_participation is not None and not prime_age_participation.dropna().empty:
        figure.add_trace(
            go.Scatter(
                x=prime_age_participation.loc[display_start_date:display_end_date].index,
                y=prime_age_participation.loc[display_start_date:display_end_date],
                mode="lines",
                name="Prime-age participation",
                line={"dash": "dot"},
            ),
            secondary_y=True,
        )

    figure.update_layout(
        title="Unemployment and labour supply",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=430,
    )
    figure.update_yaxes(title_text="Unemployment rate (%)", secondary_y=False)
    figure.update_yaxes(title_text="Participation / employment ratios (%)", secondary_y=True)
    return figure


def _claims_figure(
    initial_claims: pd.Series,
    claims_4w_avg: pd.Series,
    continuing_claims: pd.Series,
    display_start_date: date,
    display_end_date: date,
) -> go.Figure:
    """Create the claims chart."""
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
            x=initial_claims.loc[display_start_date:display_end_date].index,
            y=initial_claims.loc[display_start_date:display_end_date],
            mode="lines",
            name="Initial claims",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=claims_4w_avg.loc[display_start_date:display_end_date].index,
            y=claims_4w_avg.loc[display_start_date:display_end_date],
            mode="lines",
            name="4W average",
            line={"width": 2.5},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=continuing_claims.loc[display_start_date:display_end_date].index,
            y=continuing_claims.loc[display_start_date:display_end_date],
            mode="lines",
            name="Continuing claims",
            line={"dash": "dash"},
        ),
        secondary_y=True,
    )

    figure.update_layout(
        title="Claims and labour-market deterioration",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=430,
    )
    figure.update_yaxes(title_text="Thousands", secondary_y=False)
    figure.update_yaxes(title_text="Thousands", secondary_y=True)
    return figure


def _wage_figure(
    wage_yoy: pd.Series,
    wage_3m_ann: pd.Series,
    display_start_date: date,
    display_end_date: date,
) -> go.Figure:
    """Create the wage pressure chart."""
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=wage_yoy.loc[display_start_date:display_end_date].index,
            y=wage_yoy.loc[display_start_date:display_end_date],
            mode="lines",
            name="YoY growth",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=wage_3m_ann.loc[display_start_date:display_end_date].index,
            y=wage_3m_ann.loc[display_start_date:display_end_date],
            mode="lines",
            name="3M annualised",
            line={"dash": "dash"},
        )
    )

    figure.update_layout(
        title="Wage pressure",
        xaxis_title="Date",
        yaxis_title="Percent",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=390,
    )
    return figure


def _tightness_figure(
    openings: pd.Series,
    quits_rate: pd.Series,
    openings_ratio: pd.Series,
    latest_ratio_date: pd.Timestamp,
    display_start_date: date,
    display_end_date: date,
) -> go.Figure:
    """Create the vacancies and quits chart plus the openings ratio."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        specs=[
            [
                {
                    "secondary_y": True,
                }
            ],
            [
                {},
            ],
        ],
    )

    figure.add_trace(
        go.Scatter(
            x=openings.loc[display_start_date:display_end_date].index,
            y=openings.loc[display_start_date:display_end_date],
            mode="lines",
            name="Job openings",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=quits_rate.loc[display_start_date:display_end_date].index,
            y=quits_rate.loc[display_start_date:display_end_date],
            mode="lines",
            name="Quits rate",
            line={"dash": "dash"},
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=openings_ratio.loc[display_start_date:display_end_date].index,
            y=openings_ratio.loc[display_start_date:display_end_date],
            mode="lines",
            name="Openings per unemployed worker",
        ),
        row=2,
        col=1,
    )

    if pd.notna(_value_as_of(openings_ratio, latest_ratio_date)):
        figure.add_hline(
            y=1.0,
            row=2,
            col=1,
            line_dash="dot",
            opacity=0.5,
        )
        figure.add_annotation(
            x=latest_ratio_date,
            y=_value_as_of(openings_ratio, latest_ratio_date),
            text=(
                f"Latest: {_format_ratio(_value_as_of(openings_ratio, latest_ratio_date))}"
            ),
            showarrow=True,
            arrowhead=2,
            row=2,
            col=1,
        )

    figure.update_layout(
        title="Vacancies, quits and labour tightness",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=560,
    )
    figure.update_yaxes(title_text="Job openings", row=1, col=1, secondary_y=False)
    figure.update_yaxes(title_text="Quits rate (%)", row=1, col=1, secondary_y=True)
    figure.update_yaxes(title_text="Openings per unemployed worker", row=2, col=1)
    figure.update_xaxes(title_text="Date", row=2, col=1)
    return figure


def _participation_context_figure(
    unrate: pd.Series,
    civpart: pd.Series,
    emratio: pd.Series | None,
    prime_age_participation: pd.Series | None,
    display_start_date: date,
    display_end_date: date,
) -> go.Figure:
    """Create the participation and unemployment context chart."""
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
            x=unrate.loc[display_start_date:display_end_date].index,
            y=unrate.loc[display_start_date:display_end_date],
            mode="lines",
            name="Unemployment rate",
        ),
        secondary_y=False,
    )

    figure.add_trace(
        go.Scatter(
            x=civpart.loc[display_start_date:display_end_date].index,
            y=civpart.loc[display_start_date:display_end_date],
            mode="lines",
            name="Participation rate",
            line={"dash": "dash"},
        ),
        secondary_y=True,
    )

    if emratio is not None and not emratio.dropna().empty:
        figure.add_trace(
            go.Scatter(
                x=emratio.loc[display_start_date:display_end_date].index,
                y=emratio.loc[display_start_date:display_end_date],
                mode="lines",
                name="Employment-population ratio",
                line={"dash": "dot"},
            ),
            secondary_y=True,
        )

    if prime_age_participation is not None and not prime_age_participation.dropna().empty:
        figure.add_trace(
            go.Scatter(
                x=prime_age_participation.loc[display_start_date:display_end_date].index,
                y=prime_age_participation.loc[display_start_date:display_end_date],
                mode="lines",
                name="Prime-age participation",
                line={"dash": "dot"},
            ),
            secondary_y=True,
        )

    figure.update_layout(
        title="Participation and unemployment context",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Series",
        height=430,
    )
    figure.update_yaxes(title_text="Unemployment rate (%)", secondary_y=False)
    figure.update_yaxes(title_text="Participation / employment ratios (%)", secondary_y=True)
    return figure


def render(
    fred_client,
    context: dict,
) -> None:
    """Render the labour-market and policy-signal panel."""
    st.subheader("Panel 6: Labor Market & Policy Signal")

    st.caption(
        "This panel examines labour demand, labour supply, wage pressure, claims, vacancies and quits to provide a cautious policy-sensitive read-through."
    )

    display_start_date = context["start_date"]
    display_end_date = context["end_date"]
    fetch_start_date = min(ANALYSIS_START_DATE, display_start_date)

    result = fred_client.get_series(
        LABOR_SERIES,
        fetch_start_date,
        display_end_date,
    )

    if (
        not result.success
        or result.data is None
    ):
        st.warning(
            result.message
            or "Labour-market data unavailable."
        )
        return

    data = (
        result.data
        .copy()
        .sort_index()
    )

    if data.empty:
        st.info("No labour-market data available for the selected date range.")
        return

    missing_series = [
        series_id
        for series_id in LABOR_SERIES
        if series_id not in data.columns
    ]

    if missing_series:
        st.warning(
            "The following labour series are missing: "
            + ", ".join(missing_series)
        )
        return

    for series_id in LABOR_SERIES:
        data[series_id] = pd.to_numeric(
            data[series_id],
            errors="coerce",
        ).astype("float64")

    monthly = pd.DataFrame(
        {
            PAYEMS: _monthly_series(data, PAYEMS),
            UNRATE: _monthly_series(data, UNRATE),
            CCSA: _monthly_series(data, CCSA),
            CES0500000003: _monthly_series(data, CES0500000003),
            JTSJOL: _monthly_series(data, JTSJOL),
            JTSQUR: _monthly_series(data, JTSQUR),
            CIVPART: _monthly_series(data, CIVPART),
            UNEMPLOY: _monthly_series(data, UNEMPLOY),
            EMRATIO: _monthly_series(data, EMRATIO),
            LNS11300060: _monthly_series(data, LNS11300060),
        }
    )

    monthly = monthly.sort_index()

    if monthly.dropna(how="all").empty:
        st.warning("No usable monthly labour observations were found.")
        return

    weekly_claims = _weekly_series(data, ICSA)
    weekly_continuing = _weekly_series(data, CCSA)

    if weekly_claims.dropna().empty:
        st.warning("No usable claims observations were found.")
        return

    payems = monthly[PAYEMS].dropna()
    unrate = monthly[UNRATE].dropna()
    wages = monthly[CES0500000003].dropna()
    openings = monthly[JTSJOL].dropna()
    quits_rate = monthly[JTSQUR].dropna()
    civpart = monthly[CIVPART].dropna()
    unemployed = monthly[UNEMPLOY].dropna()
    emratio = monthly[EMRATIO].dropna()
    prime_age_participation = monthly[LNS11300060].dropna()

    latest_monthly_date = _latest_available_date(
        payems,
        unrate,
        wages,
        openings,
        quits_rate,
        civpart,
        unemployed,
        emratio,
        prime_age_participation,
    )

    latest_weekly_claims_date = _latest_available_date(
        weekly_claims,
        weekly_continuing,
    )

    if latest_monthly_date is None or latest_weekly_claims_date is None:
        st.warning("Insufficient labour-market history to render the panel.")
        return

    payems_change = payems.diff()
    payems_trailing_3m_avg = _moving_average(payems_change, 3)
    payems_trailing_6m_avg = _moving_average(payems_change, PAYROLL_TREND_MONTHS)

    latest_payroll_change = _value_as_of(payems_change, latest_monthly_date)
    payroll_avg_6m = _value_as_of(payems_trailing_6m_avg, latest_monthly_date)

    unrate_change_3m = _change_over_period(
        unrate,
        latest_monthly_date,
        latest_monthly_date - pd.DateOffset(months=3),
    )
    unrate_level = _value_as_of(unrate, latest_monthly_date)

    civpart_level = _value_as_of(civpart, latest_monthly_date)
    civpart_change_3m = _change_over_period(
        civpart,
        latest_monthly_date,
        latest_monthly_date - pd.DateOffset(months=3),
    )

    wage_mom = wages.pct_change() * 100.0
    wage_3m_ann = (
        (wages / wages.shift(WAGE_SHORT_TERM_MONTHS)) ** (12 / WAGE_SHORT_TERM_MONTHS) - 1.0
    ) * 100.0
    wage_yoy = wages.pct_change(WAGE_YEAR_OVER_YEAR_MONTHS) * 100.0

    wage_mom_level = _value_as_of(wage_mom, latest_monthly_date)
    wage_3m_ann_level = _value_as_of(wage_3m_ann, latest_monthly_date)
    wage_yoy_level = _value_as_of(wage_yoy, latest_monthly_date)

    claims_4w_avg = _moving_average(weekly_claims, CLAIMS_MOVING_AVERAGE_WEEKS)
    claims_4w_avg_level = _value_as_of(claims_4w_avg, latest_weekly_claims_date)
    claims_4w_avg_1m_ago = _value_as_of(
        claims_4w_avg,
        latest_weekly_claims_date - pd.DateOffset(months=1),
    )
    claims_4w_avg_change_1m = (
        claims_4w_avg_level - claims_4w_avg_1m_ago
        if pd.notna(claims_4w_avg_level) and pd.notna(claims_4w_avg_1m_ago)
        else float("nan")
    )

    continuing_claims_level = _value_as_of(weekly_continuing, latest_weekly_claims_date)
    continuing_claims_1m_ago = _value_as_of(
        weekly_continuing,
        latest_weekly_claims_date - pd.DateOffset(months=1),
    )
    continuing_claims_change_1m = (
        continuing_claims_level - continuing_claims_1m_ago
        if pd.notna(continuing_claims_level) and pd.notna(continuing_claims_1m_ago)
        else float("nan")
    )

    openings_level = _value_as_of(openings, latest_monthly_date)
    unemployed_level = _value_as_of(unemployed, latest_monthly_date)
    openings_ratio = (
        openings_level / unemployed_level
        if pd.notna(openings_level) and pd.notna(unemployed_level) and unemployed_level != 0
        else float("nan")
    )
    openings_ratio_series = openings / unemployed
    openings_ratio_change_3m = _change_over_period(
        openings_ratio_series,
        latest_monthly_date,
        latest_monthly_date - pd.DateOffset(months=3),
    )

    quits_level = _value_as_of(quits_rate, latest_monthly_date)
    quits_change_3m = _change_over_period(
        quits_rate,
        latest_monthly_date,
        latest_monthly_date - pd.DateOffset(months=3),
    )

    emratio_series = monthly[EMRATIO].dropna()
    prime_age_series = monthly[LNS11300060].dropna()

    regime, regime_description = _classify_labor_regime(
        latest_payroll_change,
        payroll_avg_6m,
        unrate_change_3m,
        claims_4w_avg_change_1m,
        continuing_claims_change_1m,
        wage_mom_level,
        wage_3m_ann_level,
        wage_yoy_level,
        openings_ratio,
        openings_ratio_change_3m,
        quits_change_3m,
    )

    macro_note = _build_macro_note(
        latest_monthly_date,
        regime,
        latest_payroll_change,
        payroll_avg_6m,
        unrate_level,
        unrate_change_3m,
        civpart_level,
        civpart_change_3m,
        claims_4w_avg_level,
        claims_4w_avg_change_1m,
        continuing_claims_level,
        continuing_claims_change_1m,
        wage_mom_level,
        wage_3m_ann_level,
        wage_yoy_level,
        openings_ratio,
        openings_ratio_change_3m,
        quits_level,
        quits_change_3m,
    )

    st.markdown("### Current labour assessment")
    st.info(
        f"**{regime}.** {regime_description}"
    )

    st.markdown(
        "\n".join(
            [
                "**Supporting evidence**",
                (
                    f"- Payrolls: {_format_thousands(latest_payroll_change)} versus "
                    f"a six-month average of {_format_thousands(payroll_avg_6m)}"
                ),
                (
                    f"- Unemployment: {_format_percent(unrate_level)}, "
                    f"{_change_direction(unrate_change_3m, 'up', 'down', digits=1, suffix=' pp over three months')}"
                    f"; participation: {_format_percent(civpart_level)}, "
                    f"{_change_direction(civpart_change_3m, 'up', 'down', digits=1, suffix=' pp over three months')}"
                ),
                (
                    f"- Wages: {_format_percent(wage_mom_level)} MoM, "
                    f"{_format_percent(wage_3m_ann_level)} annualised over three months, "
                    f"{_format_percent(wage_yoy_level)} YoY"
                ),
                (
                    f"- Claims: {_format_thousands(claims_4w_avg_level)} four-week average, "
                    f"continuing claims {_format_thousands(continuing_claims_level)}, "
                    f"{_change_direction(claims_4w_avg_change_1m, 'rising', 'falling', digits=0, suffix=' vs one month ago')}"
                ),
                (
                    f"- Labour tightness: {_format_ratio(openings_ratio)}, "
                    f"{_change_direction(openings_ratio_change_3m, 'higher', 'lower', digits=2, suffix=' versus three months ago')}; "
                    f"quits are {_trend_flag(quits_change_3m, 'rising', 'easing')}"
                ),
            ]
        )
    )

    st.markdown("### Macro-note output")
    st.info(macro_note)
    st.caption(
        "This paragraph is mechanically generated from labour levels, momentum and trend comparisons. It does not include consensus forecasts or event-specific context."
    )

    monthly_latest_caption = (
        f"Monthly labour series latest observation: {latest_monthly_date:%d %b %Y}."
    )
    claims_latest_caption = (
        f"Weekly claims latest observation: {latest_weekly_claims_date:%d %b %Y}."
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
            "Payroll change",
            _format_thousands(latest_payroll_change),
            (
                _format_delta(
                    (
                        latest_payroll_change - payroll_avg_6m
                        if pd.notna(latest_payroll_change) and pd.notna(payroll_avg_6m)
                        else float("nan")
                    ),
                    suffix="k vs 6M avg",
                    digits=0,
                )
            ),
        )
        _render_metric_caption(
            "Latest month-over-month change in nonfarm payroll employment, measured in thousands. It helps show whether hiring is running above or below its recent pace."
        )

    with metric_2:
        st.metric(
            "Unemployment rate",
            _format_percent(unrate_level),
            _format_delta(unrate_change_3m, suffix=" pp vs 3M ago", digits=1),
            delta_color="inverse",
        )
        _render_metric_caption(
            "The unemployment rate captures labour slack. A rising rate can reflect softer demand, a growing labour force, or both, so it is best read with participation."
        )

    with metric_3:
        st.metric(
            "Wage growth",
            _format_percent(wage_yoy_level),
            None,
        )
        _render_metric_caption(
            f"Latest MoM: {_format_percent(wage_mom_level)}, 3M annualised: {_format_percent(wage_3m_ann_level)}. Wage growth is relevant to the services-inflation and policy outlook, but it is not itself a direct measure of inflation."
        )

    with metric_4:
        st.metric(
            "Initial claims",
            _format_thousands(claims_4w_avg_level),
            _format_delta(claims_4w_avg_change_1m, suffix="k vs 1M ago", digits=0),
            delta_color="inverse",
        )
        _render_metric_caption(
            "Initial claims are noisy week to week, so the four-week average is used to highlight changes in labour-demand deterioration more clearly."
        )

    with metric_5:
        st.metric(
            "Openings per unemployed worker",
            _format_ratio(openings_ratio),
            _format_delta(openings_ratio_change_3m, suffix=" vs 3M ago", digits=2),
        )
        _render_metric_caption(
            "This ratio uses JOLTS openings divided by the unemployed population. Higher values indicate a tighter labour market and more vacancy pressure."
        )

    with metric_6:
        st.metric(
            "Labour regime",
            regime,
        )
        _render_metric_caption(
            "Mechanical classification of labour-market breadth and momentum. It is descriptive, not a recession signal or a Fed forecast."
        )

    st.caption(
        f"{monthly_latest_caption} {claims_latest_caption}"
    )

    payroll_3m_avg = _moving_average(payems_change, 3)
    payroll_figure = _payroll_figure(
        payems_change,
        payroll_3m_avg,
        payems_trailing_6m_avg,
        display_start_date,
        display_end_date,
    )

    unemployment_figure = _unemployment_figure(
        unrate,
        civpart,
        None,
        None,
        display_start_date,
        display_end_date,
    )

    claims_figure = _claims_figure(
        weekly_claims,
        claims_4w_avg,
        weekly_continuing,
        display_start_date,
        display_end_date,
    )

    wage_figure = _wage_figure(
        wage_yoy,
        wage_3m_ann,
        display_start_date,
        display_end_date,
    )

    tightness_figure = _tightness_figure(
        openings,
        quits_rate,
        openings_ratio_series,
        latest_monthly_date,
        display_start_date,
        display_end_date,
    )

    participation_context_figure = _participation_context_figure(
        unrate,
        civpart,
        emratio_series if not emratio_series.empty else None,
        prime_age_series if not prime_age_series.empty else None,
        display_start_date,
        display_end_date,
    )

    st.markdown("### Payrolls and unemployment")
    st.plotly_chart(payroll_figure, use_container_width=True)
    st.plotly_chart(unemployment_figure, use_container_width=True)
    st.caption(
        "Payroll momentum shows hiring pace, while unemployment captures labour slack. The two can diverge because they come from different surveys and measure different concepts."
    )
    st.caption(
        f"Payrolls / unemployment latest monthly observation: {latest_monthly_date:%d %b %Y}."
    )

    st.markdown("### Claims and labour-market deterioration")
    st.plotly_chart(claims_figure, use_container_width=True)
    st.caption(
        "Rising initial claims can provide an early signal of weakening labour demand. Rising continuing claims suggest unemployed workers are taking longer to find jobs. Weekly data are noisy and should be read through moving averages."
    )
    st.caption(
        f"Claims latest observation: {latest_weekly_claims_date:%d %b %Y}."
    )

    st.markdown("### Wage pressure")
    st.plotly_chart(wage_figure, use_container_width=True)
    st.caption(
        f"Wage latest observation: {latest_monthly_date:%d %b %Y}."
    )

    st.markdown("### Vacancies, quits and labour tightness")
    st.plotly_chart(tightness_figure, use_container_width=True)
    st.caption(
        "Higher openings per unemployed worker indicates a tighter labour market. Falling quits can indicate reduced worker confidence or weaker labour demand. JOLTS is lagged and revised, so it should not be treated as a real-time release indicator."
    )
    st.caption(
        f"JOLTS latest observation: {latest_monthly_date:%d %b %Y}."
    )

    st.markdown("### Participation and unemployment context")
    st.plotly_chart(participation_context_figure, use_container_width=True)
    st.caption(
        "This section checks whether unemployment changes coincide with changes in labour supply. It avoids simplistic claims about whether unemployment is 'real' or 'not real'."
    )

    with st.expander("View methodology and limitations", expanded=False):
        st.markdown(
            f"""
**Series used**

- **Payrolls:** `{PAYEMS}`
- **Unemployment rate:** `{UNRATE}`
- **Initial claims:** `{ICSA}`
- **Continuing claims:** `{CCSA}`
- **Wages:** `{CES0500000003}`
- **Job openings:** `{JTSJOL}`
- **Quits rate:** `{JTSQUR}`
- **Participation:** `{CIVPART}`
- **Unemployed:** `{UNEMPLOY}`
- **Employment-population ratio:** `{EMRATIO}`
- **Prime-age participation:** `{LNS11300060}`

**Payroll change**

Payroll momentum is measured as the month-over-month change in nonfarm payroll employment, expressed in thousands.

**Claims**

Initial claims are shown weekly, along with a four-week moving average. Continuing claims are shown at their weekly frequency. Weekly claims are noisy, so moving averages help separate signal from noise.

**Wages**

Average hourly earnings are converted into month-over-month growth, three-month annualised growth and year-over-year growth.

**Vacancies and tightness**

The openings-to-unemployed ratio is calculated as `{JTSJOL}` divided by `{UNEMPLOY}`. Units are kept compatible before division.

**Participation**

Participation is read alongside unemployment to judge whether changes in unemployment may reflect labour-force entry, labour-force withdrawal, or weaker employment conditions.

**Trend comparisons**

Trend labels are based on recent changes versus trailing averages or prior observations. They are descriptive comparisons, not consensus surprises.

**Regime rules**

The labour regime is a transparent mechanical classification that combines payrolls, unemployment, claims, wages, vacancies and quits. It is not a formal recession signal and it does not predict Fed decisions.

**Limitations**

Payrolls and JOLTS are revised. Claims are noisy. JOLTS is delayed. Household and establishment surveys can diverge. Wage growth is not a direct inflation measure, and the policy interpretation should remain cautious.
            """
        )
