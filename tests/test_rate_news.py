import pandas as pd

from data.rate_news import _parse_gdelt_articles, classify_rate_topic


def test_classify_rate_topic_prefers_specific_macro_theme() -> None:
    assert classify_rate_topic("Treasury yields jump after hot CPI report") == "Inflation"
    assert classify_rate_topic("Fed signals patience on rate cuts") == "Fed"


def test_parse_gdelt_articles_deduplicates_and_rejects_unsafe_urls() -> None:
    payload = {
        "articles": [
            {
                "title": "Treasury yields rise after jobs report",
                "url": "https://example.com/rates",
                "domain": "example.com",
                "seendate": "20260928T120000Z",
                "sourcecountry": "United States",
                "language": "English",
            },
            {
                "title": "Treasury yields rise after jobs report",
                "url": "https://mirror.example.com/rates",
                "seendate": "20260928T120100Z",
            },
            {
                "title": "Unsafe item",
                "url": "javascript:alert(1)",
                "seendate": "20260928T120200Z",
            },
        ]
    }

    result = _parse_gdelt_articles(payload)

    assert len(result) == 1
    assert result.iloc[0]["topic"] == "Labor"
    assert pd.api.types.is_datetime64_any_dtype(result["published_at"])
