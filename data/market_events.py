from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re

import pandas as pd
import requests
import streamlit as st

from config import FRED_API_KEY, FRED_CACHE_TTL_SECONDS


EVENT_COLUMNS = [
    "event_date",
    "event_time",
    "event_name",
    "event_type",
    "source",
    "source_url",
    "importance",
    "release_id",
]

SOURCE_URLS = {
    "BEA": "https://www.bea.gov/news/schedule",
    "Federal Reserve": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "FRED": "https://fred.stlouisfed.org/releases/calendar",
    "U.S. Treasury": "https://treasurydirect.gov/auctions/upcoming/",
}

BEA_RELEASES_URL = "https://apps.bea.gov/API/signup/release_dates.json"
FRED_RELEASES_URL = "https://api.stlouisfed.org/fred/releases"
FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"

TREASURY_AUCTIONS_API = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v1/accounting/od/auctions_query"
)


def empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _release_id(source: str, name: str, event_date: pd.Timestamp) -> str:
    raw = f"{source}_{name}_{event_date.date().isoformat()}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")


def _fetch_html(url: str) -> str:
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception:  # noqa: BLE001
        return ""


def _parse_date(value: object, default_year: int | None = None) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    candidates = [str(value).strip()]
    if default_year is not None:
        candidates.append(f"{value} {default_year}")
    for candidate in candidates:
        parsed = pd.to_datetime(candidate, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    return None


def _event_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return empty_event_frame()
    frame = pd.DataFrame(rows).reindex(columns=EVENT_COLUMNS)
    return frame.drop_duplicates(subset=["event_date", "event_name", "source"]).sort_values(
        ["event_date", "importance", "event_name"],
        ascending=[True, False, True],
    )


def _fred_release_events(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Use the Fed's release calendar when agency sites block automated access."""
    if not FRED_API_KEY:
        return empty_event_frame()
    relevant_names = {
        "Consumer Price Index",
        "Employment Cost Index",
        "Employment Situation",
        "Job Openings and Labor Turnover Survey",
        "Producer Price Index",
        "Productivity and Costs",
        "U.S. Import and Export Price Indexes",
    }
    ten_am_releases = {"Job Openings and Labor Turnover Survey"}
    try:
        response = requests.get(
            FRED_RELEASES_URL,
            params={
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "limit": 1000,
            },
            timeout=20,
        )
        response.raise_for_status()
        releases = {
            int(record["id"]): _text(record.get("name"))
            for record in response.json().get("releases", [])
            if _text(record.get("name")) in relevant_names
        }
    except Exception:  # noqa: BLE001
        return empty_event_frame()

    def _release_dates(item: tuple[int, str]) -> tuple[str, list[dict[str, object]]]:
        release_id, release_name = item
        try:
            dates_response = requests.get(
                FRED_RELEASE_DATES_URL,
                params={
                    "release_id": release_id,
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "include_release_dates_with_no_data": "true",
                    "limit": 10000,
                    "sort_order": "desc",
                },
                timeout=20,
            )
            dates_response.raise_for_status()
            return release_name, dates_response.json().get("release_dates", [])
        except Exception:  # noqa: BLE001
            return release_name, []

    with ThreadPoolExecutor(max_workers=max(1, len(releases))) as executor:
        release_dates = list(executor.map(_release_dates, releases.items()))

    rows = []
    for name, records in release_dates:
        for record in records:
            event_date = _parse_date(record.get("date"))
            if event_date is None or not start_date.normalize() <= event_date <= end_date.normalize():
                continue
            event_time = "10:00 AM ET" if name in ten_am_releases else "8:30 AM ET"
            rows.append(
                {
                    "event_date": event_date.date().isoformat(),
                    "event_time": event_time,
                    "event_name": name,
                    "event_type": "Economic release",
                    "source": "FRED",
                    "source_url": SOURCE_URLS["FRED"],
                    "importance": "High",
                    "release_id": _release_id("FRED", name, event_date),
                }
            )
    return _event_frame(rows)


def _bea_release_events(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        response = requests.get(BEA_RELEASES_URL, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception:  # noqa: BLE001
        return empty_event_frame()

    rows = []
    for name, details in payload.items():
        if name == "file_last_updated" or not isinstance(details, dict):
            continue
        for date_text in details.get("release_dates", []):
            event_timestamp = pd.to_datetime(date_text, errors="coerce", utc=True)
            if pd.isna(event_timestamp):
                continue
            event_date = pd.Timestamp(event_timestamp).tz_convert("America/New_York")
            normalized = event_date.tz_localize(None).normalize()
            if not start_date.normalize() <= normalized <= end_date.normalize():
                continue
            rows.append(
                {
                    "event_date": normalized.date().isoformat(),
                    "event_time": f"{event_date:%-I:%M %p} ET",
                    "event_name": name,
                    "event_type": "Economic release",
                    "source": "BEA",
                    "source_url": SOURCE_URLS["BEA"],
                    "importance": "High",
                    "release_id": _release_id("BEA", name, normalized),
                }
            )
    return _event_frame(rows)


def _fomc_events(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    page = _fetch_html(SOURCE_URLS["Federal Reserve"])
    plain_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page))
    rows = []
    month_pattern = (
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)"
    )
    for year in range(start_date.year, end_date.year + 1):
        heading = f"{year} FOMC Meetings"
        block_start = plain_text.find(heading)
        if block_start < 0:
            continue
        remainder = plain_text[block_start + len(heading):]
        next_heading = re.search(r"\b\d{4} FOMC Meetings\b", remainder)
        year_block = remainder[: next_heading.start()] if next_heading else remainder
        meeting_pattern = re.compile(
            rf"(?<!Released ){month_pattern}\s+(\d{{1,2}})(?:-(\d{{1,2}}))?\*?\s+"
            rf"(?=Statement:|{month_pattern}|\* Meeting)"
        )
        for match in meeting_pattern.finditer(year_block):
            month = match.group(1)
            start_day = match.group(2)
            end_day = match.group(3) or start_day
            event_date = _parse_date(f"{month} {end_day}, {year}")
            if event_date is None or not start_date.normalize() <= event_date <= end_date.normalize():
                continue
            range_text = start_day if start_day == end_day else f"{start_day}-{end_day}"
            name = f"FOMC meeting ({month} {range_text}, {year})"
            rows.append(
                {
                    "event_date": event_date.date().isoformat(),
                    "event_time": "2:00 PM ET",
                    "event_name": name,
                    "event_type": "FOMC calendar",
                    "source": "Federal Reserve",
                    "source_url": SOURCE_URLS["Federal Reserve"],
                    "importance": "High",
                    "release_id": _release_id("Federal Reserve", name, event_date),
                }
            )
    return _event_frame(rows)


def _treasury_auction_events(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        response = requests.get(
            TREASURY_AUCTIONS_API,
            params={
                "fields": "security_type,security_term,auction_date,reopening",
                "filter": (
                    f"auction_date:gte:{start_date.date().isoformat()},"
                    f"auction_date:lte:{end_date.date().isoformat()}"
                ),
                "sort": "auction_date,security_type,security_term",
                "page[size]": 100,
            },
            timeout=20,
        )
        response.raise_for_status()
        records = response.json().get("data", [])
    except Exception:  # noqa: BLE001
        return empty_event_frame()

    rows = []
    for record in records:
        event_date = _parse_date(record.get("auction_date"))
        if event_date is None:
            continue
        security_type = _text(record.get("security_type"))
        security_term = _text(record.get("security_term"))
        reopening = _text(record.get("reopening"))
        suffix = " reopening" if reopening.lower() == "yes" else ""
        name = f"{security_term} {security_type} auction{suffix}".strip()
        rows.append(
            {
                "event_date": event_date.date().isoformat(),
                "event_time": "1:00 PM ET",
                "event_name": name,
                "event_type": "Treasury auction",
                "source": "U.S. Treasury",
                "source_url": SOURCE_URLS["U.S. Treasury"],
                "importance": "Medium",
                "release_id": _release_id("U.S. Treasury", name, event_date),
            }
        )
    return _event_frame(rows)


@st.cache_data(ttl=FRED_CACHE_TTL_SECONDS, show_spinner=False)
def _cached_events(start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    fetchers = (
        lambda: _fred_release_events(start, end),
        lambda: _bea_release_events(start, end),
        lambda: _fomc_events(start, end),
        lambda: _treasury_auction_events(start, end),
    )
    with ThreadPoolExecutor(max_workers=len(fetchers)) as executor:
        frames = list(executor.map(lambda fetch: fetch(), fetchers))
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return empty_event_frame()
    combined = pd.concat(available, ignore_index=True).reindex(columns=EVENT_COLUMNS)
    return combined.drop_duplicates(subset=["event_date", "event_name", "source"]).sort_values(
        ["event_date", "importance", "event_name"],
        ascending=[True, False, True],
    )


def get_scheduled_catalysts(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Return official economic, policy and Treasury-auction events."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        return empty_event_frame()
    return _cached_events(start.date().isoformat(), end.date().isoformat()).copy()
