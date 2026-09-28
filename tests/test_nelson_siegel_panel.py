import numpy as np

from panels.nelson_siegel import _factor_contribution_figure, _ns_curve


def test_factor_contributions_reconstruct_fitted_curve_change() -> None:
    tau = np.array([0.25, 2.0, 10.0, 30.0])
    changes = (0.12, -0.08, 0.05)

    figure = _factor_contribution_figure(
        tau,
        ["3M", "2Y", "10Y", "30Y"],
        *changes,
        "1M",
    )

    chart_total_bp = sum(np.asarray(trace.y) for trace in figure.data)
    direct_total_bp = (
        _ns_curve(tau, *changes) - _ns_curve(tau, 0.0, 0.0, 0.0)
    ) * 100.0

    np.testing.assert_allclose(chart_total_bp, direct_total_bp)
