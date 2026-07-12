from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable
import time

import pandas as pd
import os

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
from fredapi import Fred
import streamlit as st

from config import (
	FRED_API_KEY,
	FRED_CACHE_TTL_SECONDS,
	FRED_MAX_RETRIES,
	FRED_RETRY_SLEEP_SECONDS,
)


@dataclass
class FREDResult:
	data: pd.DataFrame | None
	success: bool
	message: str | None = None
	refreshed_at: pd.Timestamp | None = None


@st.cache_data(ttl=FRED_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_fetch_series(
	api_key: str,
	series_ids: tuple[str, ...],
	start_date: str,
	end_date: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:
	fred = Fred(api_key=api_key)
	frames: list[pd.Series] = []
	for series_id in series_ids:
		series = fred.get_series(
			series_id,
			observation_start=start_date,
			observation_end=end_date,
		)
		series = pd.to_numeric(series, errors="coerce")
		series.name = series_id
		frames.append(series)

	if not frames:
		empty = pd.DataFrame()
		empty.index.name = "date"
		return empty, pd.Timestamp.utcnow()

	df = pd.concat(frames, axis=1).sort_index()
	df.index = pd.to_datetime(df.index)
	df.index.name = "date"
	return df, pd.Timestamp.utcnow()


class FREDClient:
	def __init__(self, api_key: str | None = None) -> None:
		self.api_key = (api_key or FRED_API_KEY).strip()

	def is_configured(self) -> bool:
		return bool(self.api_key)

	def get_series(
		self,
		series_ids: Iterable[str],
		start_date: date,
		end_date: date,
	) -> FREDResult:
		if not self.is_configured():
			return FREDResult(
				data=None,
				success=False,
				message="FRED_API_KEY is missing. Add it to your .env file.",
			)

		series_tuple = tuple(sorted(set(series_ids)))
		if not series_tuple:
			return FREDResult(
				data=pd.DataFrame(),
				success=True,
				refreshed_at=pd.Timestamp.utcnow(),
			)

		last_error: Exception | None = None
		for attempt in range(FRED_MAX_RETRIES + 1):
			try:
				data, refreshed_at = _cached_fetch_series(
					self.api_key,
					series_tuple,
					start_date.isoformat(),
					end_date.isoformat(),
				)
				return FREDResult(
					data=data,
					success=True,
					refreshed_at=refreshed_at,
				)
			except Exception as exc:  # noqa: BLE001
				last_error = exc
				if attempt < FRED_MAX_RETRIES:
					time.sleep(FRED_RETRY_SLEEP_SECONDS)
					continue
				break

		return FREDResult(
			data=None,
			success=False,
			message=f"Data unavailable from FRED for this panel ({last_error}).",
		)
