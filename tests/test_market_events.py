from __future__ import annotations

import pandas as pd

from data import market_events


def test_fomc_parser_uses_the_decision_day(monkeypatch) -> None:
    page = """
    <h3>2026 FOMC Meetings</h3>
    <div>September 15-16* Statement:</div>
    <div>October 27-28 December 8-9* * Meeting associated with projections.</div>
    <h3>2025 FOMC Meetings</h3>
    """
    monkeypatch.setattr(market_events, "_fetch_html", lambda _url: page)

    events = market_events._fomc_events(
        pd.Timestamp("2026-09-01"),
        pd.Timestamp("2026-10-31"),
    )

    assert events["event_date"].tolist() == ["2026-09-16", "2026-10-28"]


def test_invalid_event_window_returns_empty_frame() -> None:
    events = market_events.get_scheduled_catalysts(
        pd.Timestamp("2026-10-01"),
        pd.Timestamp("2026-09-01"),
    )

    assert events.empty
    assert list(events.columns) == market_events.EVENT_COLUMNS
