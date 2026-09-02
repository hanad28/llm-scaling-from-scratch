import numpy as np
import pytest

from scaling_lm.scaling_fit import fit_power_law

PARAMETER_COUNTS = np.array([1e6, 5e6, 1.2e7, 2.5e7, 1e8])


def test_recovers_exact_power_law():
    true_alpha, true_coefficient = 0.076, 12.0
    losses = true_coefficient * PARAMETER_COUNTS**-true_alpha
    fit = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=200)
    assert fit.alpha == pytest.approx(true_alpha, abs=1e-9)
    assert fit.coefficient == pytest.approx(true_coefficient, rel=1e-6)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.alpha_standard_error == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(fit.predict(PARAMETER_COUNTS), losses)


def test_noise_widens_interval_and_interval_contains_truth():
    generator = np.random.default_rng(0)
    true_alpha = 0.076
    losses = 10.0 * PARAMETER_COUNTS**-true_alpha * np.exp(generator.normal(0, 0.01, 5))
    fit = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=2000)
    assert fit.alpha_standard_error > 0
    assert fit.alpha_ci_low < fit.alpha < fit.alpha_ci_high
    assert fit.alpha_ci_low < true_alpha < fit.alpha_ci_high
    assert fit.kaplan_within_ci
    assert fit.alpha_bootstrap_ci_low <= fit.alpha <= fit.alpha_bootstrap_ci_high


def test_kaplan_flag_false_when_exponent_clearly_different():
    losses = 10.0 * PARAMETER_COUNTS**-0.3
    fit = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=200)
    assert not fit.kaplan_within_ci


def test_rejects_too_few_points_and_non_positive_values():
    with pytest.raises(ValueError):
        fit_power_law(np.array([1e6, 1e7]), np.array([3.0, 2.5]))
    with pytest.raises(ValueError):
        fit_power_law(np.array([1e6, 1e7, 1e8]), np.array([3.0, -2.5, 2.0]))


def test_to_dict_is_json_friendly():
    losses = 10.0 * PARAMETER_COUNTS**-0.08
    payload = fit_power_law(PARAMETER_COUNTS, losses, bootstrap_resamples=100).to_dict()
    assert payload["n_points"] == 5
    assert isinstance(payload["kaplan_within_ci"], bool)
