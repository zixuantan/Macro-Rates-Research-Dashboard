from __future__ import annotations

from collections import defaultdict

from models.macro_analysis import EvidenceItem, HypothesisAssessment, IssueCategoryConfig


SECTION_LABELS = {
    "research_question": "Research question",
    "conclusion": "Conclusion",
    "realized_inflation": "Realized inflation",
    "market_inflation_pricing": "Market inflation pricing",
    "wage_and_labor_pressure": "Wage and labor pressure",
    "policy_implications": "Policy implications",
    "counter_evidence": "Counter-evidence",
    "rates_implications": "Rates implications",
    "risks_and_watch_list": "Risks and watch list",
    "activity_momentum": "Activity momentum",
    "breadth_and_drivers": "Breadth and drivers",
    "labor_confirmation": "Labor confirmation",
    "cross_asset_confirmation": "Cross-asset confirmation",
    "observed_treasury_move": "Observed Treasury move",
    "tenor_attribution": "Tenor attribution",
    "curve_decomposition": "Curve decomposition",
    "inflation_context": "Inflation context",
    "growth_and_labor_context": "Growth and labor context",
    "most_consistent_interpretation": "Most consistent interpretation",
    "alternative_explanations": "Alternative explanations",
    "data_gaps": "Data gaps",
    "rates_view_and_watch_list": "Rates view and watch list",
    "base_case": "Base case",
    "evidence_for_benign_slowing": "Evidence for benign slowing",
    "evidence_for_growth_risk": "Evidence for growth risk",
    "inflation_constraint": "Inflation constraint",
    "labor_conditions": "Labor conditions",
    "cross_asset_verdict": "Cross-asset verdict",
    "invalidation_conditions": "Invalidation conditions",
    "hiring_momentum": "Hiring momentum",
    "unemployment_and_claims": "Unemployment and claims",
    "wage_pressure": "Wage pressure",
    "vacancies_and_quits": "Vacancies and quits",
    "broader_growth_context": "Broader growth context",
    "market_confirmation": "Market confirmation",
    "equities_and_volatility": "Equities and volatility",
    "dollar": "Dollar",
    "rates_context": "Rates context",
    "macro_confirmation": "Macro confirmation",
    "conflicting_signals": "Conflicting signals",
    "credit": "Credit",
    "supporting_evidence": "Supporting evidence",
    "implications": "Implications",
    "invalidation_conditions": "Invalidation conditions",
}


SECTION_GROUPS = {
    "realized_inflation": ("realized_inflation",),
    "market_inflation_pricing": ("medium_term_inflation_pricing", "long_run_inflation_anchor"),
    "wage_and_labor_pressure": ("labor_wage_pressure", "labor_tightness"),
    "policy_implications": ("front_end_rates", "curve_slope"),
    "counter_evidence": (),
    "rates_implications": ("front_end_rates", "curve_slope", "long_end_curve"),
    "activity_momentum": ("growth_momentum",),
    "breadth_and_drivers": ("growth_breadth", "growth_momentum"),
    "labor_confirmation": ("labor_demand", "labor_slack"),
    "cross_asset_confirmation": ("credit_risk", "equity_risk", "market_volatility", "dollar_conditions"),
    "observed_treasury_move": ("treasury_level_move", "front_end_rates"),
    "tenor_attribution": ("treasury_level_move", "front_end_rates", "long_end_curve"),
    "curve_decomposition": ("curve_slope", "curve_curvature"),
    "inflation_context": ("medium_term_inflation_pricing", "long_run_inflation_anchor", "realized_inflation"),
    "growth_and_labor_context": ("growth_momentum", "labor_demand", "labor_slack"),
    "most_consistent_interpretation": (),
    "alternative_explanations": (),
    "data_gaps": (),
    "rates_view_and_watch_list": ("front_end_rates", "curve_slope", "long_end_curve"),
    "base_case": (),
    "evidence_for_benign_slowing": ("growth_momentum", "labor_slack", "credit_risk"),
    "evidence_for_growth_risk": ("growth_breadth", "labor_demand", "credit_risk", "market_volatility"),
    "inflation_constraint": ("realized_inflation", "medium_term_inflation_pricing", "household_inflation_expectations"),
    "labor_conditions": ("labor_demand", "labor_slack", "labor_wage_pressure"),
    "cross_asset_verdict": ("credit_risk", "equity_risk", "market_volatility", "dollar_conditions"),
    "invalidation_conditions": (),
    "hiring_momentum": ("labor_demand",),
    "unemployment_and_claims": ("labor_slack", "labor_demand"),
    "wage_pressure": ("labor_wage_pressure",),
    "vacancies_and_quits": ("labor_tightness", "labor_supply"),
    "broader_growth_context": ("growth_momentum", "growth_breadth"),
    "market_confirmation": ("credit_risk", "equity_risk", "market_volatility"),
    "equities_and_volatility": ("equity_risk", "market_volatility"),
    "dollar": ("dollar_conditions",),
    "rates_context": ("front_end_rates", "curve_slope", "treasury_level_move"),
    "macro_confirmation": ("growth_momentum", "labor_demand", "realized_inflation"),
    "conflicting_signals": (),
    "credit": ("credit_risk",),
    "supporting_evidence": (),
    "implications": (),
    "invalidation_conditions": (),
}


def _signal_value(signal) -> str:
    if signal.value is None or signal.value != signal.value:
        return "Unavailable"
    if signal.unit == "%":
        return f"{signal.value:.2f}{signal.unit}"
    if signal.unit in {"bp", "index", "x", "z", "k"}:
        return f"{signal.value:.2f} {signal.unit}".strip()
    return f"{signal.value:.2f} {signal.unit}".strip()


def _signal_change(signal) -> str:
    if signal.change is None or signal.change != signal.change:
        return "change unavailable"
    unit = signal.change_unit or ""
    horizon = f" ({signal.horizon})" if signal.horizon else ""
    if unit == "%":
        return f"{signal.change:+.2f}{unit}{horizon}"
    if unit:
        return f"{signal.change:+.2f} {unit}{horizon}"
    return f"{signal.change:+.2f}{horizon}"


def _top_items_for_groups(
    evidence_items: list[EvidenceItem],
    groups: tuple[str, ...],
    limit: int = 3,
) -> list[EvidenceItem]:
    if not groups:
        return evidence_items[:limit]
    selected = [item for item in evidence_items if item.signal.group_id in groups]
    return selected[:limit]


def _format_item(item: EvidenceItem) -> str:
    signal = item.signal
    return (
        f"- {signal.label}: {_signal_value(signal)}; {_signal_change(signal)}. "
        f"{signal.interpretation or item.rationale or signal.label}"
    )


def _evidence_block(
    heading: str,
    items: list[EvidenceItem],
    fallback: str = "",
) -> str:
    if not items:
        return fallback
    lines = [f"{heading}:"]
    lines.extend(_format_item(item) for item in items[:3])
    return "\n".join(lines)


def _hypothesis_block(hypotheses: list[HypothesisAssessment]) -> str:
    if not hypotheses:
        return "Manual evidence selection is required for this custom issue."
    lines = []
    for item in hypotheses[:3]:
        lines.append(f"- {item.title}: {item.conclusion} ({item.support_score:+.2f}).")
    return "\n".join(lines)


def _invalidation_block(
    hypotheses: list[HypothesisAssessment],
    data_gaps: list[str],
) -> str:
    if not hypotheses:
        return "Manual evidence selection is required to define invalidation conditions."
    leading = hypotheses[0]
    support = ", ".join(item.signal.label for item in leading.supporting_evidence[:3]) or "the supporting evidence set"
    challenge = ", ".join(item.signal.label for item in leading.challenging_evidence[:3]) or "the challenging evidence set"
    lines = [
        f"- The leading reading would weaken if {challenge} started to dominate {support}.",
    ]
    if data_gaps:
        lines.append(f"- Missing evidence remains: {data_gaps[0]}")
    return "\n".join(lines)


def build_issue_note_sections(
    category_config: IssueCategoryConfig,
    question: str,
    leading_interpretation: str,
    alternative_interpretations: list[str],
    evidence_items: list[EvidenceItem],
    hypotheses: list[HypothesisAssessment],
    topic_conflicts,
    data_gaps: list[str],
    watch_points: list[str],
    panel_regimes: dict[str, str],
) -> dict[str, str]:
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence_items:
        grouped[item.signal.group_id or item.signal.signal_id].append(item)

    sections: dict[str, str] = {}
    for section_id in category_config.note_section_order:
        groups = SECTION_GROUPS.get(section_id, ())
        selected_items: list[EvidenceItem] = []
        for group in groups:
            selected_items.extend(grouped.get(group, []))
        selected_items = sorted(selected_items, key=lambda item: item.issue_score, reverse=True)

        if section_id == "research_question":
            sections[section_id] = f"Question: {question}\n\nCategory: {category_config.description}"
        elif section_id == "conclusion":
            alt_text = "\n".join(f"- {item}" for item in alternative_interpretations) if alternative_interpretations else "- No clear alternative interpretation."
            sections[section_id] = f"{leading_interpretation}\n\nAlternative interpretations:\n{alt_text}"
        elif section_id == "counter_evidence":
            challenged = [item for item in evidence_items if any(effect == "Challenges" for effect in item.hypothesis_effects.values())]
            sections[section_id] = _evidence_block("Counter-evidence", challenged)
        elif section_id == "supporting_evidence":
            supported = [item for item in evidence_items if any(effect == "Supports" for effect in item.hypothesis_effects.values())]
            sections[section_id] = _evidence_block("Supporting evidence", supported)
        elif section_id == "implications":
            sections[section_id] = f"{leading_interpretation}\n\n{_hypothesis_block(hypotheses)}"
        elif section_id == "invalidation_conditions":
            sections[section_id] = _invalidation_block(hypotheses, data_gaps)
        elif section_id == "data_gaps":
            sections[section_id] = "\n".join(f"- {gap}" for gap in data_gaps) if data_gaps else ""
        elif section_id == "risks_and_watch_list":
            sections[section_id] = "\n".join(f"- {point}" for point in watch_points) if watch_points else "No specific watch points identified."
        elif section_id == "base_case":
            sections[section_id] = leading_interpretation
        elif section_id == "most_consistent_interpretation":
            sections[section_id] = f"{leading_interpretation}\n\n{_hypothesis_block(hypotheses)}"
        elif section_id == "alternative_explanations":
            sections[section_id] = "\n".join(f"- {item}" for item in alternative_interpretations) if alternative_interpretations else "No clear alternative explanations identified."
        elif section_id == "conflicting_signals":
            if topic_conflicts:
                sections[section_id] = "\n".join(f"- {conflict.title}: {conflict.explanation}" for conflict in topic_conflicts[:4])
            else:
                sections[section_id] = ""
        elif section_id == "macro_confirmation":
            confirmation_lines = [f"- {panel}: {regime}" for panel, regime in panel_regimes.items()]
            sections[section_id] = "\n".join(confirmation_lines) if confirmation_lines else ""
        else:
            sections[section_id] = _evidence_block(
                SECTION_LABELS.get(section_id, section_id.replace("_", " ").title()),
                selected_items,
            )

    return sections


def issue_note_text(
    sections: dict[str, str],
    section_order: tuple[str, ...],
    title: str,
) -> str:
    parts = [f"# {title}"]
    for section_id in section_order:
        content = sections.get(section_id, "").strip()
        if not content:
            continue
        heading = SECTION_LABELS.get(section_id, section_id.replace("_", " ").title())
        parts.append(f"## {heading}\n{content}")
    return "\n\n".join(parts)
