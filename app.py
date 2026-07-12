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
from panels import cross_asset, growth_nowcast, inflation, nelson_siegel, yield_curve


def _default_start_date() -> date:
	return date.today() - timedelta(days=365 * DEFAULT_DATE_RANGE_YEARS)


def main() -> None:
	st.set_page_config(page_title="Macro/Rates Quant Dashboard", layout="wide")
	st.title("Macro/Rates Quant Dashboard")
	st.caption("Module 1: Macro + fixed income analytics")

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


if __name__ == "__main__":
	main()
