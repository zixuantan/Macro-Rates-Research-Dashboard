from __future__ import annotations

from models.macro_analysis import ConflictFlag, MacroSignal


def detect_conflicts(signals: list[MacroSignal]) -> list[ConflictFlag]:
    """Return a small set of rule-based conflicts."""
    by_id = {signal.signal_id: signal for signal in signals}
    conflicts: list[ConflictFlag] = []

    cpi = by_id.get("inflation_cpi_yoy")
    pce = by_id.get("inflation_pce_yoy")
    breakeven_5y = by_id.get("inflation_5y_breakeven")
    front_end = by_id.get("front_end_yield_change")
    growth = by_id.get("growth_regime")
    labor = by_id.get("labor_regime")
    rates = by_id.get("yield_curve_regime")
    credit = by_id.get("credit_regime")
    vix = by_id.get("vix_level")

    if cpi and breakeven_5y and cpi.value is not None and breakeven_5y.value is not None and cpi.value > breakeven_5y.value + 0.5:
        conflicts.append(
            ConflictFlag(
                conflict_id="realized_vs_market_inflation",
                title="Realized versus market inflation",
                signals=[cpi, breakeven_5y],
                explanation=(
                    "Realized inflation remains elevated while market pricing implies future disinflation."
                ),
                severity="medium",
            )
        )

    if growth and labor and growth.direction == "weakening" and labor.direction == "stable":
        conflicts.append(
            ConflictFlag(
                conflict_id="growth_vs_labor",
                title="Growth versus labor",
                signals=[growth, labor],
                explanation=(
                    "Broad activity is softening, but labour-market deterioration remains incomplete."
                ),
                severity="medium",
            )
        )

    if rates and credit and vix and rates.direction in {"lower", "mixed"} and credit.direction == "higher" and vix.direction == "higher":
        conflicts.append(
            ConflictFlag(
                conflict_id="rates_vs_risk",
                title="Rates versus cross-asset risk",
                signals=[rates, credit, vix],
                explanation=(
                    "The rates move is accompanied by weaker risk sentiment, consistent with a growth scare rather than benign easing."
                ),
                severity="high",
            )
        )

    if front_end and breakeven_5y and front_end.direction == "lower" and breakeven_5y.direction == "higher":
        conflicts.append(
            ConflictFlag(
                conflict_id="inflation_vs_rates",
                title="Inflation repricing versus front-end yields",
                signals=[front_end, breakeven_5y],
                explanation=(
                    "Breakevens are rising while front-end nominal yields are falling, which is a meaningful divergence."
                ),
                severity="medium",
            )
        )

    return conflicts
