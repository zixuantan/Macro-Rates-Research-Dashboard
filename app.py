from __future__ import annotations
import os
import sys

os.environ["PANDAS_STRING_STORAGE"] = "python"
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
import pandas as pd
pd.options.mode.string_storage = "python"
import numpy as np
if not hasattr(np, 'long'):
    np.long = int

from datetime import date, timedelta

import streamlit as st

from config import DEFAULT_DATE_RANGE_YEARS, MODULE_TABS, YIELD_SERIES
from data.fred_client import FREDClient
from panels import cross_asset, growth_nowcast, guided_research, inflation, labor_market, nelson_siegel, note_workspace, yield_curve


def _default_start_date() -> date:
	return date.today() - timedelta(days=365 * DEFAULT_DATE_RANGE_YEARS)


def _inject_metric_wrap_styles() -> None:
	st.markdown(
		"""
		<style>
			div[data-testid="stMetric"] {
				overflow: visible;
				align-items: flex-start;
			}

			div[data-testid="stMetricLabel"] {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
				line-height: 1.2;
				font-weight: 700;
			}

			div[data-testid="stMetricLabel"] div[data-testid="stMarkdownContainer"] {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
			}

			div[data-testid="stMetricLabel"] div[data-testid="stMarkdownContainer"] p {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
				margin: 0 !important;
			}

			div[data-testid="stMetricValue"] {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
				line-height: 1.1;
				font-size: 1.05rem;
				font-weight: 600;
			}

			div[data-testid="stMetricValue"] div[data-testid="stMarkdownContainer"] {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
				line-height: 1.1;
				font-size: 1.05rem;
				font-weight: 600;
			}

			div[data-testid="stMetricValue"] div[data-testid="stMarkdownContainer"] p {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
				margin: 0 !important;
				line-height: 1.1;
				font-size: 1.05rem;
				font-weight: 600;
			}

			div[data-testid="stMetricDelta"] {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
			}

			div[data-testid="stMetricDelta"] div[data-testid="stMarkdownContainer"] {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
			}

			div[data-testid="stMetricDelta"] div[data-testid="stMarkdownContainer"] p {
				white-space: normal !important;
				overflow: visible !important;
				text-overflow: unset !important;
				word-break: break-word !important;
				margin: 0 !important;
			}
		</style>
		""",
		unsafe_allow_html=True,
	)


def main() -> None:
	st.set_page_config(page_title="Macro/Rates Quant Dashboard", layout="wide")
	st.title("Macro/Rates Research Dashboard")
	_inject_metric_wrap_styles()

	client = FREDClient()

	st.sidebar.header("Global Controls")
	start_date, end_date = st.sidebar.date_input(
		"Date range",
		value=(_default_start_date(), date.today()),
		min_value=date(1970, 1, 1),
		max_value=date.today(),
	)

	if start_date >= end_date:
		st.error("Start date must be before end date.")
		st.stop()

	yield_result = client.get_series(YIELD_SERIES, start_date, end_date)
	refreshed_at = yield_result.refreshed_at

	if refreshed_at is not None:
		st.sidebar.caption(f"Data last refreshed: {refreshed_at.strftime('%Y-%m-%d %H:%M UTC')}")
	else:
		st.sidebar.caption("Data last refreshed: unavailable")

	context = {
		"start_date": start_date,
		"end_date": end_date,
		"yield_result": yield_result,
	}
	panel_analyses = guided_research._build_panel_analyses(client, context)
	context["panel_analyses"] = panel_analyses

	tabs = st.tabs(MODULE_TABS)
	with tabs[0]:
		yield_curve.render(client, context)
	with tabs[1]:
		nelson_siegel.render(client, context)
	with tabs[2]:
		inflation.render(client, context)
	with tabs[3]:
		growth_nowcast.render(client, context)
	with tabs[4]:
		cross_asset.render(client, context)
	with tabs[5]:
		labor_market.render(client, context)
	with tabs[6]:
		note_workspace.render(client, context, panel_analyses=panel_analyses)


if __name__ == "__main__":
	main()
