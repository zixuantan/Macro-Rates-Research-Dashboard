from analysis.issue_categories import ISSUE_CATEGORY_ORDER, ISSUE_CATEGORY_REGISTRY


def test_issue_categories_cover_expected_modes() -> None:
    assert ISSUE_CATEGORY_ORDER == (
        "inflation_policy",
        "growth_outlook",
        "labor_policy",
        "treasury_curve",
        "soft_landing_growth_scare",
        "cross_asset_risk",
        "custom",
    )

    for key in ISSUE_CATEGORY_ORDER:
        assert key in ISSUE_CATEGORY_REGISTRY
        config = ISSUE_CATEGORY_REGISTRY[key]
        assert config.category_id == key
        assert config.display_name
        assert config.description
        assert config.note_section_order
