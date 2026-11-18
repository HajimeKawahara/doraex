"""Independent integration and score checks for the bounded A marginal."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from doraex.inference.normalization_marginalization import (
    conditional_inverse_normalization_moments,
    conditional_normalization_moments,
    draw_conditional_normalization,
    marginal_normalization_log_likelihood,
)


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64():
        yield


def scipy_reference(observed, prediction, variance, lower=1.0, upper=1.2):
    """Integrate directly in A, using breakpoints to resolve narrow peaks."""

    observed, prediction, variance = np.broadcast_arrays(
        np.asarray(observed, dtype=float),
        np.asarray(prediction, dtype=float),
        np.asarray(variance, dtype=float),
    )
    precision = np.sum(prediction**2 / variance)
    rhs = np.sum(prediction * observed / variance)
    inverse_center = np.clip(
        rhs / precision if precision else 1.0 / ((lower + upper) / 2.0),
        1.0 / upper,
        1.0 / lower,
    )
    center = 1.0 / inverse_center
    residual_center = observed - prediction / center
    log_peak = -0.5 * np.sum(
        residual_center**2 / variance + np.log(2.0 * np.pi * variance)
    )
    slope = abs(np.sum(prediction * residual_center / variance))
    scale = min(
        upper - lower,
        center**2 / max(np.sqrt(precision) + slope, np.finfo(float).tiny),
    )
    points = np.unique(
        np.clip(
            center + scale * np.asarray([-128, -32, -8, -2, 0, 2, 8, 32, 128]),
            lower,
            upper,
        )
    )

    def relative_likelihood(a):
        residual = observed - prediction / a
        # Difference of squares avoids subtracting large log likelihoods.
        difference = np.sum(
            (residual - residual_center) * (residual + residual_center) / variance
        )
        return np.exp(-0.5 * difference), residual

    def integrate(moment):
        def integrand(a):
            density, residual = relative_likelihood(a)
            return density * moment(a, residual)

        return quad(
            integrand,
            lower,
            upper,
            points=points,
            epsabs=1e-12 * scale,
            epsrel=2e-12,
            limit=200,
        )[0]

    normalizer = integrate(lambda a, residual: 1.0)
    mean = integrate(lambda a, residual: a) / normalizer
    conditional_variance = integrate(lambda a, residual: (a - mean) ** 2) / normalizer
    feature = np.linspace(-0.5, 0.5, observed.size).reshape(observed.shape)
    # Integration by parts gives the amplitude score without cancellation of
    # very large positive and negative conditional scores for narrow peaks.
    amplitude_score = 1.0 + (
        lower * relative_likelihood(lower)[0] - upper * relative_likelihood(upper)[0]
    ) / normalizer
    scores = np.asarray(
        [
            amplitude_score,
            integrate(lambda a, r: np.sum(r * feature / (a * variance))) / normalizer,
            integrate(lambda a, r: np.sum(r**2 / variance - 1.0)) / normalizer,
        ]
    )
    return log_peak + np.log(normalizer / (upper - lower)), mean, conditional_variance, scores


def synthetic_case(a_true, noise, count, heterogeneous=False):
    phase = np.linspace(0.0, 5.0, count)
    prediction = 1.0 + 0.15 * np.sin(phase)
    variance = noise**2 * (1.0 + 0.4 * np.cos(phase)) if heterogeneous else noise**2
    observed = prediction / a_true + np.sqrt(variance) * 0.2 * np.sin(2.7 * phase)
    return observed, prediction, variance


@pytest.mark.parametrize(
    "a_true,noise,count,heterogeneous",
    [
        (1.1, 0.03, 14336, False),
        (1.0, 0.03, 256, True),
        (1.2, 0.03, 256, True),
        (0.85, 0.03, 128, True),
        (1.35, 0.03, 128, True),
        (1.101234, 1e-5, 128, False),
        (1.1, 3.0, 7, True),
        (-1.0, 0.2, 32, False),
    ],
)
def test_log_likelihood_moments_and_scores_against_scipy(
    a_true, noise, count, heterogeneous
):
    observed, prediction, variance = synthetic_case(a_true, noise, count, heterogeneous)
    expected = scipy_reference(observed, prediction, variance)
    feature = jnp.linspace(-0.5, 0.5, count)

    def objective(parameters):
        return marginal_normalization_log_likelihood(
            observed,
            prediction * jnp.exp(parameters[0]) + feature * parameters[1],
            variance * jnp.exp(2.0 * parameters[2]),
        )

    actual_log_likelihood, actual_scores = jax.jit(jax.value_and_grad(objective))(
        jnp.zeros(3)
    )
    actual_moments = conditional_normalization_moments(observed, prediction, variance)
    assert actual_log_likelihood.dtype == jnp.float64
    np.testing.assert_allclose(actual_log_likelihood, expected[0], rtol=1e-12, atol=2e-8)
    np.testing.assert_allclose(actual_moments[0], expected[1], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(actual_moments[1], expected[2], rtol=2e-7, atol=1e-20)
    np.testing.assert_allclose(actual_scores, expected[3], rtol=2e-7, atol=2e-7)
    higher_order = marginal_normalization_log_likelihood(
        observed, prediction, variance, quadrature_order=128
    )
    np.testing.assert_allclose(actual_log_likelihood, higher_order, rtol=0.0, atol=2e-10)


@pytest.mark.parametrize("a_true", [1.0, 1.2])
def test_gradient_at_exact_prior_boundary(a_true):
    prediction = jnp.ones(32)
    observed = prediction / a_true
    expected = scipy_reference(observed, prediction, 0.01**2)
    value, gradient = jax.jit(
        jax.value_and_grad(
            lambda scale: marginal_normalization_log_likelihood(
                observed, prediction * jnp.exp(scale), 0.01**2
            )
        )
    )(jnp.asarray(0.0))
    np.testing.assert_allclose(value, expected[0], atol=1e-10)
    np.testing.assert_allclose(gradient, expected[3][0], rtol=1e-10)


def test_uninformative_prediction_retains_uniform_a_prior_and_finite_gradient():
    observed = jnp.linspace(0.8, 1.1, 9)
    prediction = jnp.zeros_like(observed)
    variance = 0.03**2
    lower, upper = 0.7, 1.8
    expected_log_likelihood = -0.5 * jnp.sum(
        observed**2 / variance + jnp.log(2.0 * jnp.pi * variance)
    )
    value, gradient = jax.jit(
        jax.value_and_grad(
            lambda x: marginal_normalization_log_likelihood(
                observed, x, variance, lower=lower, upper=upper
            )
        )
    )(prediction)
    moments = conditional_normalization_moments(
        observed, prediction, variance, lower=lower, upper=upper
    )
    np.testing.assert_allclose(value, expected_log_likelihood, atol=1e-10)
    np.testing.assert_allclose(moments, [(lower + upper) / 2.0, (upper - lower) ** 2 / 12])
    expected_inverse_a = np.log(upper / lower) / (upper - lower)
    np.testing.assert_allclose(gradient, observed * expected_inverse_a / variance, rtol=1e-12)


def test_vmap_handles_independent_chips_and_scalar_or_vector_noise():
    observed, prediction, variance = synthetic_case(1.1, 0.04, 30)
    batched = jax.jit(jax.vmap(marginal_normalization_log_likelihood))(
        jnp.stack([observed, observed]),
        jnp.stack([prediction, prediction * 1.01]),
        jnp.full((2, observed.size), variance),
    )
    independent = [
        marginal_normalization_log_likelihood(observed, prediction * factor, variance)
        for factor in (1.0, 1.01)
    ]
    np.testing.assert_allclose(batched, independent, rtol=1e-13)


def test_requires_explicit_x64_and_valid_static_configuration():
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="X64"):
            marginal_normalization_log_likelihood([1.0], [1.1], 0.01)
    for lower, upper in [(0.0, 1.0), (1.2, 1.0), (1.0, np.inf)]:
        with pytest.raises(ValueError, match="bounds"):
            marginal_normalization_log_likelihood([1.0], [1.1], 0.01, lower, upper)
    for order in (0, 1, True, 3.5):
        with pytest.raises(ValueError, match="quadrature_order"):
            marginal_normalization_log_likelihood([1.0], [1.1], 0.01, quadrature_order=order)


@pytest.mark.parametrize(
    "a_true,noise,count",
    [(1.1, 0.03, 128), (1.1, 1.0, 8), (1.0, 0.03, 128), (1.2, 0.03, 128)],
)
def test_continuous_conditional_draws_match_moments(a_true, noise, count):
    observed, prediction, variance = synthetic_case(a_true, noise, count, True)
    expected_mean, expected_variance = map(
        float, conditional_normalization_moments(observed, prediction, variance)
    )
    rng = np.random.default_rng(1909)
    draws = np.asarray(
        [draw_conditional_normalization(observed, prediction, variance, rng) for _ in range(3000)]
    )
    assert np.all((1.0 <= draws) & (draws <= 1.2))
    assert np.unique(draws).size == draws.size
    mean_standard_error = np.sqrt(expected_variance / draws.size)
    assert abs(np.mean(draws) - expected_mean) < 5.0 * mean_standard_error
    squared_deviations = (draws - expected_mean) ** 2
    variance_standard_error = np.std(squared_deviations, ddof=1) / np.sqrt(draws.size)
    assert abs(np.mean(squared_deviations) - expected_variance) < 5.0 * variance_standard_error


@pytest.mark.parametrize("a_true", [0.85, 1.35])
def test_conditional_draws_resolve_far_gaussian_tails(a_true):
    # The likelihood peak is over one thousand standard deviations out of
    # bounds, where subtraction of ordinary Gaussian CDFs loses all precision.
    observed, prediction, variance = synthetic_case(a_true, 0.001, 128)
    expected_mean, expected_variance = map(
        float, conditional_normalization_moments(observed, prediction, variance)
    )
    rng = np.random.default_rng(441)
    draws = np.asarray(
        [draw_conditional_normalization(observed, prediction, variance, rng) for _ in range(3000)]
    )
    assert np.all((1.0 < draws) & (draws < 1.2))
    assert np.unique(draws).size > 2990
    assert abs(np.mean(draws) - expected_mean) < 5.0 * np.sqrt(expected_variance / draws.size)
    squared_deviations = (draws - expected_mean) ** 2
    standard_error = np.std(squared_deviations, ddof=1) / np.sqrt(draws.size)
    assert abs(np.mean(squared_deviations) - expected_variance) < 5.0 * standard_error


def test_zero_prediction_draw_uses_original_uniform_prior_and_cpu_rng():
    rng, reference_rng = np.random.default_rng(93), np.random.default_rng(93)
    with jax.enable_x64(False):
        draws = [
            draw_conditional_normalization([0.9], [0.0], 0.01, rng, lower=0.7, upper=1.9)
            for _ in range(10)
        ]
    np.testing.assert_array_equal(draws, reference_rng.uniform(0.7, 1.9, size=10))


def test_conditional_draw_validates_inputs_and_has_explicit_trial_limit(monkeypatch):
    rng = np.random.default_rng(5)
    for observed, prediction, variance in [
        ([np.nan], [1.0], 0.01),
        ([1.0], [np.inf], 0.01),
        ([1.0], [1.0], -0.01),
        ([1.0], [1.0], 0.0),
        ([1.0], [1.0], np.inf),
        ([], [], 0.01),
    ]:
        with pytest.raises(ValueError):
            draw_conditional_normalization(observed, prediction, variance, rng)
    with pytest.raises(TypeError, match="Generator"):
        draw_conditional_normalization([1.0], [1.1], 0.01, 93)
    for max_trials in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_trials"):
            draw_conditional_normalization([1.0], [1.1], 0.01, rng, max_trials=max_trials)

    # Very wide bounds can make the rejection probability arbitrarily high;
    # a finite explicit trial limit must fail, without silently altering draws.
    monkeypatch.setattr(
        "doraex.inference.normalization_marginalization.truncnorm.rvs", lambda *a, **k: 1.0
    )
    with pytest.raises(RuntimeError, match="max_trials"):
        draw_conditional_normalization(
            [1.0], [1.0], 0.01, rng, lower=1e-6, upper=1e6, max_trials=2
        )
    monkeypatch.setattr(
        "doraex.inference.normalization_marginalization.truncnorm.rvs", lambda *a, **k: np.nan
    )
    with pytest.raises(FloatingPointError, match="bounds"):
        draw_conditional_normalization([1.0], [1.0], 0.01, rng)


def test_inverse_moments_retain_the_original_uniform_a_prior():
    lower, upper = .7, 1.8
    mean, variance = conditional_inverse_normalization_moments(
        [1.0], [0.0], .01, lower=lower, upper=upper
    )
    expected_mean = np.log(upper / lower) / (upper - lower)
    expected_second = (1 / lower - 1 / upper) / (upper - lower)
    np.testing.assert_allclose(mean, expected_mean, rtol=1e-13)
    np.testing.assert_allclose(variance, expected_second - expected_mean**2, rtol=1e-12)


def test_inverse_variance_resolves_a_very_narrow_interior_gaussian():
    prediction = jnp.ones(128)
    observed = prediction / 1.1
    noise_variance = 1e-14
    mean_s, variance_s = conditional_inverse_normalization_moments(
        observed, prediction, noise_variance
    )
    precision_s = prediction.size / noise_variance
    np.testing.assert_allclose(mean_s, 1 / 1.1, rtol=1e-13)
    # At this width the s**-2 prior changes Gaussian variance below machine
    # precision. Computing it after a reciprocal round trip loses ~1e-9 here.
    np.testing.assert_allclose(variance_s * precision_s, 1., rtol=5e-14)


@pytest.mark.parametrize("a_true", [.999, 1.0, 1.001, 1.1, 1.199, 1.2, 1.201])
def test_bounded_amplitude_metric_tracks_exact_curvature_near_prior_edges(a_true):
    prediction = jnp.ones(64)
    observed = prediction / a_true
    variance = .01**2
    mean_s, variance_s = conditional_inverse_normalization_moments(observed, prediction, variance)
    precision_s = prediction.size / variance
    fraction = jnp.clip(precision_s * variance_s, 0., 1.)
    approximate_curvature = mean_s**2 * precision_s * (1. - fraction)
    exact_curvature = jax.jit(jax.hessian(lambda log_scale: -marginal_normalization_log_likelihood(
        observed, prediction * jnp.exp(log_scale), variance
    )))(jnp.asarray(0.))
    assert np.isfinite(exact_curvature)
    assert 0. <= float(fraction) <= 1.
    assert approximate_curvature >= 0.
    if a_true == 1.1:
        # Interior amplitude scaling changes only the integrated prior volume;
        # its log likelihood is linear, so the amplitude Hessian vanishes.
        assert abs(float(exact_curvature)) < 1e-6
        assert approximate_curvature < 1e-6
    else:
        # The unbounded Schur approximation gives zero here and misses the
        # finite-prior wall. The bounded approximation captures its curvature.
        assert exact_curvature > 1e4
        assert approximate_curvature > 1e4
        np.testing.assert_allclose(approximate_curvature, exact_curvature, rtol=.01)
