# Macro/Rates Quant Dashboard (Module 1)

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

## Run

```bash
streamlit run app.py
```

## Notes

- All FRED calls go through `data/fred_client.py` with cache + retry handling.
- Panel modules expose `render(fred_client, context)` and are wired in `app.py` via tabs.
- Future Module 2 (Context) and Module 3 (Notes) can be added as new tabs/modules without refactoring existing panel code.
