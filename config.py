from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()

# Central cache/retry policy for all modules.
FRED_CACHE_TTL_SECONDS = 60 * 60 * 4
FRED_MAX_RETRIES = 1
FRED_RETRY_SLEEP_SECONDS = 1.0

FRED_POLICY_SERIES = {
	"target_low": "DFEDTARL",
	"target_high": "DFEDTARU",
	"effective_rate": "EFFR",
	"sofr": "SOFR",
	"fallback_rate": "FEDFUNDS",
	"balance_sheet": "WALCL",
}

DEFAULT_DATE_RANGE_YEARS = 2

# Module/tabs can be extended in future modules without refactoring app wiring.
MODULE_TABS = [
	"Yield Curve",
	"Nelson-Siegel",
	"Inflation",
	"Growth Nowcast",
	"Cross-Asset",
	"Labor & Policy",
	"Guided Macro Note Workspace",
]

NOTE_WORKSPACE_MAX_MOVES = 6
NOTE_WORKSPACE_ARCHIVE_DIR = BASE_DIR / "notes"

# Panel 1 / Panel 2 shared yield series.
DGS1MO = "DGS1MO"
DGS3MO = "DGS3MO"
DGS6MO = "DGS6MO"
DGS1 = "DGS1"
DGS2 = "DGS2"
DGS5 = "DGS5"
DGS10 = "DGS10"
DGS30 = "DGS30"

YIELD_SERIES = [DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS30]

TENOR_YEAR_MAP = {
	DGS1MO: 1 / 12,
	DGS3MO: 3 / 12,
	DGS6MO: 6 / 12,
	DGS1: 1,
	DGS2: 2,
	DGS5: 5,
	DGS10: 10,
	DGS30: 30,
}

# Panel 3 inflation series.
T5YIE = "T5YIE"
T10YIE = "T10YIE"
T5YIFR = "T5YIFR"
CPIAUCSL = "CPIAUCSL"
PCEPI = "PCEPI"
MICH = "MICH"

INFLATION_SERIES = [T5YIE, T10YIE, T5YIFR, CPIAUCSL, PCEPI, MICH]

# Panel 4 growth nowcast series.
ICSA = "ICSA"
INDPRO = "INDPRO"
PAYEMS = "PAYEMS"
GACDISA066MSFRBPHI = "GACDFSA066MSFRBPHI"
GDPC1 = "GDPC1"

GROWTH_SERIES = [ICSA, INDPRO, PAYEMS, GACDISA066MSFRBPHI, GDPC1]

# Panel 5 cross-asset context series.
DTWEXBGS = "DTWEXBGS"
BAMLH0A0HYM2 = "BAMLH0A0HYM2"
BAMLC0A0CM = "BAMLC0A0CM"
VIXCLS = "VIXCLS"

CROSS_ASSET_SERIES = [DTWEXBGS, BAMLH0A0HYM2, BAMLC0A0CM, VIXCLS]

# Panel 6 labour market series.
UNRATE = "UNRATE"
CCSA = "CCSA"
CES0500000003 = "CES0500000003"
JTSJOL = "JTSJOL"
JTSQUR = "JTSQUR"
CIVPART = "CIVPART"
UNEMPLOY = "UNEMPLOY"
EMRATIO = "EMRATIO"
LNS11300060 = "LNS11300060"

LABOR_SERIES = [
	PAYEMS,
	UNRATE,
	ICSA,
	CCSA,
	CES0500000003,
	JTSJOL,
	JTSQUR,
	CIVPART,
	UNEMPLOY,
	EMRATIO,
	LNS11300060,
]
