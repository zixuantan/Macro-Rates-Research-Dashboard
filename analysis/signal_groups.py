from __future__ import annotations

from collections import defaultdict

from models.macro_analysis import MacroSignal


def group_signals(signals: list[MacroSignal]) -> dict[str, list[MacroSignal]]:
    """Group related signals to avoid double-counting."""
    groups: dict[str, list[MacroSignal]] = defaultdict(list)
    for signal in signals:
        group_id = signal.group_id or signal.signal_id
        groups[group_id].append(signal)
    return dict(groups)
