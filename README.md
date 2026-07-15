# Macro/Rates Quant Dashboard 

Streamlit dashboard for macro and fixed income analysis with a modular panel architecture.

## Structure

```
macro_dashboard/
	app.py
	config.py
	data/
		fred_client.py
	panels/
		yield_curve.py
		nelson_siegel.py
		inflation.py
		growth_nowcast.py
		cross_asset.py
		labor_market.py
		guided_research.py
	requirements.txt
	.env.example
```

## Setup

1. Create a virtual environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Get a FRED API key:
    - Go to the Federal Reserve Economic Data website and create an account.
    - Generate an API key from your account settings.
4. Create a `.env` file (copy from `.env.example`) and set:

```bash
FRED_API_KEY=your_actual_key
```

Optional:
- `FRED_API_KEY` enables ALFRED vintages, policy-series lookups, and the release-surprise proxy.

## Run

```bash
streamlit run app.py
```

## Notes

- All FRED calls go through `data/fred_client.py` with cache + retry handling.
- The note workspace uses public BLS, BEA, and Federal Reserve pages for catalyst timing, plus FRED/ALFRED vintage data for the release proxy.
- Panel modules expose `render(fred_client, context)` and are wired in `app.py` via tabs.
- The current tab set is:
  - Treasury Yield Curve
  - Nelson-Siegel
  - Inflation
  - Growth Nowcast
  - Cross-Asset
  - Labor & Policy
  - Guided Research
- The panel architecture is modular, so additional tabs can be added without refactoring the existing panel code.
