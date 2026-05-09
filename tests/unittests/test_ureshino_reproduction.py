"""Regression tests against the Ureshino et al. Bayesian DI equations."""

import jax
import jax.numpy as jnp
import numpy as np

from doraex.geometry.healpix import angular_distance, angular_distance_matrix
from doraex.geometry.limb_darkening import (
    kipping_q_to_u,
    linear_limb_darkening,
    quadratic_limb_darkening,
)
from doraex.geometry.rotation import incline
from doraex.inference.map_posterior import conditional_map_posterior
from doraex.inference.marginal_likelihood import (
    dense_marginal_covariance,
    diagonal_noise_variance,
)
from doraex.operators.design_matrix import (
    full_design_matrix_from_angles,
    two_column_operator_from_angles,
)
from doraex.operators.doppler import doppler_factor
from doraex.priors.spherical_gp import (
    add_diagonal_jitter,
    squared_exponential_covariance,
)


jax.config.update("jax_enable_x64", True)


def _reference_incline(theta0, phi0, alpha):
    x0 = jnp.sin(theta0) * jnp.cos(phi0)
    y0 = jnp.sin(theta0) * jnp.sin(phi0)
    z0 = jnp.cos(theta0)
    x = jnp.cos(alpha) * x0 + jnp.sin(alpha) * z0
    y = y0
    z = -jnp.sin(alpha) * x0 + jnp.cos(alpha) * z0
    return jnp.arccos(z), jnp.arctan2(y, x)


def _reference_doppler_shift(vlos):
    beta = vlos / 299_792.458
    return (1.0 + beta) / jnp.sqrt(1.0 - beta**2)


def _reference_exposure_matrix(
    theta0, phi0, vrot, inclination, u1, u2, phase, wavelengths, line_profile
):
    phi_rot = phi0 + phase * 2.0 * jnp.pi
    theta, phi = _reference_incline(theta0, phi_rot, jnp.pi / 2.0 - inclination)
    vlos = vrot * jnp.cos(jnp.pi / 2.0 - inclination) * jnp.sin(theta0) * jnp.sin(phi_rot)
    shifted = jnp.interp(
        wavelengths[:, None] / _reference_doppler_shift(vlos),
        wavelengths,
        line_profile,
    )
    mu = jnp.sin(theta) * jnp.cos(phi)
    limb = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    visible = ((-jnp.pi / 2.0 < phi) & (phi < jnp.pi / 2.0)).astype(float)
    return visible[None, :] * mu[None, :] * limb[None, :] * shifted


def _reference_full_design_matrix(
    theta0,
    phi0,
    vrot,
    inclination,
    u1,
    u2,
    obs_times,
    period,
    wavelengths,
    line_profile,
    weights,
):
    phases = obs_times / period

    def scaled_matrix(phase, weight):
        matrix = _reference_exposure_matrix(
            theta0, phi0, vrot, inclination, u1, u2, phase, wavelengths, line_profile
        )
        return weight * matrix

    stack = jax.vmap(scaled_matrix)(phases, weights)
    return stack.reshape((len(obs_times) * len(wavelengths), -1))


def _reference_conditional_posterior(
    data, design_matrix, prior_mean, prior_covariance, sigma_d
):
    noise_variance = sigma_d**2 + 1.0e-6
    prior_precision = jnp.linalg.inv(prior_covariance)
    posterior_precision = prior_precision + (design_matrix.T / noise_variance) @ design_matrix
    posterior_covariance = jnp.linalg.inv(posterior_precision)
    residual = data.reshape(-1) - design_matrix @ prior_mean
    posterior_mean = prior_mean + posterior_covariance @ (
        design_matrix.T * (1.0 / noise_variance) @ residual
    )
    return posterior_mean, posterior_covariance


def test_geometry_and_design_matrix_match_ureshino_formulation():
    theta = jnp.array([0.45, 1.10, 1.75, 2.30])
    phi = jnp.array([-1.2, -0.2, 0.7, 1.6])
    wavelengths = jnp.linspace(655.9, 656.5, 7)
    line_profile = 1.0 - 0.35 * jnp.exp(-0.5 * ((wavelengths - 656.2) / 0.08) ** 2)
    obs_times = jnp.array([0.1, 0.7, 1.4])
    period = 2.0
    weights = jnp.array([1.0, 0.98, 1.03])
    vrot = 31.2
    inclination = jnp.deg2rad(61.0)
    u1, u2 = 0.45, 0.18

    actual = full_design_matrix_from_angles(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        obs_times / period,
        wavelengths,
        line_profile,
        weights=weights,
    )
    expected = _reference_full_design_matrix(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        obs_times,
        period,
        wavelengths,
        line_profile,
        weights,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_map_marginalization_and_conditional_posterior_match_ureshino_formulation():
    theta = jnp.array([0.55, 1.25, 1.85])
    phi = jnp.array([-0.8, 0.3, 1.4])
    wavelengths = jnp.linspace(1.0, 1.4, 5)
    line_profile = 1.0 - 0.2 * jnp.exp(-0.5 * ((wavelengths - 1.2) / 0.05) ** 2)
    phases = jnp.array([0.0, 0.35])
    weights = jnp.array([1.0, 1.02])
    design_matrix = full_design_matrix_from_angles(
        theta,
        phi,
        18.0,
        jnp.deg2rad(50.0),
        0.3,
        0.1,
        phases,
        wavelengths,
        line_profile,
        weights=weights,
    )

    distances = angular_distance_matrix(theta, phi)
    prior_covariance = add_diagonal_jitter(
        squared_exponential_covariance(distances, amplitude=0.25, length_scale=0.9),
        jitter=1.0e-6,
    )
    prior_mean = 0.03 * jnp.ones(design_matrix.shape[1])
    true_map = jnp.array([0.02, -0.01, 0.04])
    data = design_matrix @ true_map
    sigma_d = 0.02
    noise_variance = diagonal_noise_variance(design_matrix.shape[0], sigma_d)

    actual_mean, actual_covariance = conditional_map_posterior(
        data, design_matrix, prior_mean, prior_covariance, noise_variance
    )
    expected_mean, expected_covariance = _reference_conditional_posterior(
        data, design_matrix, prior_mean, prior_covariance, sigma_d
    )

    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(
        actual_covariance, expected_covariance, rtol=1.0e-11, atol=1.0e-11
    )

    dense_covariance = dense_marginal_covariance(
        design_matrix, prior_covariance, noise_variance
    )
    expected_dense_covariance = (
        design_matrix @ prior_covariance @ design_matrix.T + jnp.diag(noise_variance)
    )
    np.testing.assert_allclose(
        dense_covariance, expected_dense_covariance, rtol=1.0e-12, atol=1.0e-12
    )


def test_two_column_operator_preserves_linear_cloud_contrast_form():
    theta = jnp.array([0.6, 1.2, 1.9])
    phi = jnp.array([-0.4, 0.5, 1.1])
    wavelengths = jnp.linspace(2.0, 2.4, 6)
    clear = jnp.ones_like(wavelengths) - 0.10 * jnp.exp(
        -0.5 * ((wavelengths - 2.16) / 0.05) ** 2
    )
    cloudy = jnp.ones_like(wavelengths) - 0.25 * jnp.exp(
        -0.5 * ((wavelengths - 2.18) / 0.07) ** 2
    )
    phases = jnp.array([0.1, 0.4])
    f0 = 0.35
    contrast_map = jnp.array([0.05, -0.02, 0.01])

    base, contrast_matrix = two_column_operator_from_angles(
        theta,
        phi,
        25.0,
        jnp.deg2rad(55.0),
        0.4,
        0.12,
        phases,
        wavelengths,
        clear,
        cloudy,
        f0,
    )
    model_from_linear_form = base + contrast_matrix @ contrast_map

    local_profiles = (
        1.0 - (f0 + contrast_map[:, None])
    ) * clear[None, :] + (f0 + contrast_map[:, None]) * cloudy[None, :]
    manual = jnp.zeros_like(model_from_linear_form)
    for pixel_index in range(theta.size):
        pixel_matrix = full_design_matrix_from_angles(
            theta[pixel_index : pixel_index + 1],
            phi[pixel_index : pixel_index + 1],
            25.0,
            jnp.deg2rad(55.0),
            0.4,
            0.12,
            phases,
            wavelengths,
            local_profiles[pixel_index],
        )
        manual = manual + pixel_matrix[:, 0]

    np.testing.assert_allclose(model_from_linear_form, manual, rtol=1.0e-12, atol=1.0e-12)


def test_small_public_helpers_match_expected_values():
    theta, phi = incline(jnp.array([jnp.pi / 2.0]), jnp.array([0.0]), 0.0)
    np.testing.assert_allclose(theta, jnp.array([jnp.pi / 2.0]))
    np.testing.assert_allclose(phi, jnp.array([0.0]))
    np.testing.assert_allclose(linear_limb_darkening(0.5, 0.2), 0.6)
    np.testing.assert_allclose(quadratic_limb_darkening(0.4, 0.2, 0.5), 0.75)
    u1, u2 = kipping_q_to_u(0.25, 0.5)
    np.testing.assert_allclose((u1, u2), (0.5, 0.0))
    np.testing.assert_allclose(doppler_factor(0.0), 1.0)
    np.testing.assert_allclose(angular_distance(0.0, 0.0, 0.0, 1.0), 0.0)
