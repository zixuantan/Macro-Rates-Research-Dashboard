from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

from config import FRED_CACHE_TTL_SECONDS


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADLINE_COLUMNS = [
    "published_at",
    "title",
    "url",
    "domain",
    "source_country",
    "language",
    "topic",
]

RATE_NEWS_QUERY = (
    '("Treasury yield" OR "Treasury yields" OR "bond yields" OR FOMC '
    'OR "Federal Reserve rates") sourcelang:english'
)


def empty_headline_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=HEADLINE_COLUMNS)


def _safe_http_url(value: object) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    return url if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def classify_rate_topic(title: object) -> str:
    text = str(title or "").lower()
    if any(term in text for term in ("cpi", "pce", "ppi", "inflation", "prices")):
        return "Inflation"
    if any(term in text for term in ("payroll", "jobs", "employment", "labor", "labour", "jolts")):
        return "Labor"
    if any(term in text for term in ("fomc", "federal reserve", "fed ", "powell", "rate cut", "rate hike")):
        return "Fed"
    if any(term in text for term in ("gdp", "growth", "recession", "economy")):
        return "Growth"
    if any(term in text for term in ("auction", "treasury sale", "note sale", "bond sale")):
        return "Auction"
    return "Rates"


def _parse_gdelt_articles(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return empty_headline_frame()
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        return empty_headline_frame()

    rows = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "").strip()
        url = _safe_http_url(article.get("url"))
        published_at = pd.to_datetime(article.get("seendate"), errors="coerce", utc=True)
        if not title or not url or pd.isna(published_at):
            continue
        rows.append(
            {
                "published_at": published_at.tz_convert(None),
                "title": title,
                "url": url,
                "domain": str(article.get("domain") or urlparse(url).netloc).removeprefix("www."),
                "source_country": str(article.get("sourcecountry") or "").strip(),
                "language": str(article.get("language") or "").strip(),
                "topic": classify_rate_topic(title),
            }
        )

    if not rows:
        return empty_headline_frame()
    frame = pd.DataFrame(rows).reindex(columns=HEADLINE_COLUMNS)
    frame["title_key"] = frame["title"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    frame = frame.drop_duplicates(subset=["url"]).drop_duplicates(subset=["title_key"])
    return (
        frame.drop(columns="title_key")
        .sort_values("published_at", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_data(ttl=FRED_CACHE_TTL_SECONDS, show_spinner=False)
def get_rate_headlines(
    start_date: object,
    end_date: object,
    max_records: int = 50,
) -> pd.DataFrame:
    """Fetch recent English-language rates headlines from the GDELT DOC API."""
    start = pd.Timestamp(start_date).tz_localize(None).normalize()
    end = pd.Timestamp(end_date).tz_localize(None).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    available_start = now - pd.Timedelta(days=89)

    # The DOC API exposes a rolling three-month window. Historical dashboard
    # dates outside it should fail quietly rather than showing unrelated news.
    if end < available_start or start > now:
        return empty_headline_frame()
    request_start = max(start, available_start)
    request_end = min(end, now)
    if request_start > request_end:
        return empty_headline_frame()

    try:
        response = requests.get(
            GDELT_DOC_API,
            params={
                "query": RATE_NEWS_QUERY,
                "mode": "ArtList",
                "format": "json",
                "sort": "DateDesc",
                "maxrecords": min(max(int(max_records), 1), 250),
                "startdatetime": request_start.strftime("%Y%m%d%H%M%S"),
                "enddatetime": request_end.strftime("%Y%m%d%H%M%S"),
            },
            headers={"User-Agent": "RatesMonitor/1.0"},
            timeout=20,
        )
        response.raise_for_status()
        return _parse_gdelt_articles(response.json())
    except (requests.RequestException, ValueError, TypeError):
        return empty_headline_frame()
