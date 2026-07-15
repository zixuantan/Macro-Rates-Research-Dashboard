# Macro/Rates Quant Dashboard

Live app: https://zixuantan-macro-rates-research-dashboard-app-brugc0.streamlit.app/

Streamlit dashboard for macro and fixed-income analysis with a modular panel architecture.

## What’s Included

- Treasury Yield Curve
- Nelson-Siegel
- Inflation
- Growth Nowcast
- Cross-Asset
- Labor & Policy
- Guided Macro Note Workspace

## Project Layout

```text
macro_dashboard/
	app.py
	config.py
	data/
		fred_client.py
	analysis/
		synthesis.py
		issue_note_builder.py
	panels/
		yield_curve.py
		nelson_siegel.py
		inflation.py
		growth_nowcast.py
		cross_asset.py
		labor_market.py
		guided_research.py
		note_workspace.py
	tests/
	requirements.txt
	.env.example
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example`.
4. Add your FRED API key:

```bash
FRED_API_KEY=your_actual_key
```

Optional environment settings:

- `FRED_API_KEY` enables ALFRED vintages, policy-series lookups, and the release-surprise proxy.
- `TRADING_ECONOMICS_*` settings are not required for the current version of the note workspace because public government and Federal Reserve sources are used instead.

## Run Locally

From the `macro_dashboard/` directory:

```bash
streamlit run app.py
```

## Streamlit Community Cloud

To deploy on Streamlit Community Cloud:

1. Push the `macro_dashboard/` project to GitHub.
2. Connect the GitHub repository in Streamlit Community Cloud.
3. Set the app entrypoint to `app.py`.
4. Add any required secrets, especially `FRED_API_KEY`.

If Streamlit Cloud says the app is not connected to GitHub, double-check that:

- You are deploying the same repository you pushed.
- The selected branch is the one that contains the latest commits.
- The app is pointing at the correct `app.py` file.

## Notes

- All FRED calls go through `data/fred_client.py` with cache and retry handling.
- The note workspace uses public BLS, BEA, and Federal Reserve pages for upcoming catalyst timing.
- Fed and policy context is assembled from FRED/ALFRED series plus Federal Reserve RSS feeds.
- The note workspace compares the current snapshot against a selected horizon and keeps same-horizon historical moves for z-scores.
- Panel modules expose `render(fred_client, context)` and are wired in `app.py` via tabs.
- The architecture is modular, so additional tabs can be added without refactoring the existing panel code.
