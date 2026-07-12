from __future__ import annotations

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

from config import BAMLH0A0HYM2, BAMLC0A0CM, CROSS_ASSET_SERIES, DTWEXBGS, VIXCLS


def render(fred_client, context: dict) -> None:
	st.subheader("Panel 5: Cross-Asset Context")
	start_date = context["start_date"]
	end_date = context["end_date"]

	result = fred_client.get_series(CROSS_ASSET_SERIES, start_date, end_date)
	if not result.success or result.data is None:
		st.warning(result.message or "Cross-asset data unavailable.")
		return

	df = result.data.copy()
	if df.empty:
		st.info("No data available for selected date range.")
		return

	fig = make_subplots(
		rows=4,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.04,
		subplot_titles=(
			"Dollar Index (DXY)",
			"High Yield OAS",
			"Investment Grade OAS",
			"VIX",
		),
	)

	fig.add_trace(go.Scatter(x=df.index, y=df[DTWEXBGS], mode="lines", name="DXY"), row=1, col=1)
	fig.add_trace(go.Scatter(x=df.index, y=df[BAMLH0A0HYM2], mode="lines", name="HY OAS"), row=2, col=1)
	fig.add_trace(go.Scatter(x=df.index, y=df[BAMLC0A0CM], mode="lines", name="IG OAS"), row=3, col=1)
	fig.add_trace(go.Scatter(x=df.index, y=df[VIXCLS], mode="lines", name="VIX"), row=4, col=1)

	fig.update_layout(height=900, title="Cross-Asset Context", template="plotly_white", showlegend=False)
	fig.update_yaxes(title_text="Index", row=1, col=1)
	fig.update_yaxes(title_text="bps", row=2, col=1)
	fig.update_yaxes(title_text="bps", row=3, col=1)
	fig.update_yaxes(title_text="Index", row=4, col=1)
	fig.update_xaxes(title_text="Date", row=4, col=1)

	st.plotly_chart(fig, use_container_width=True)
