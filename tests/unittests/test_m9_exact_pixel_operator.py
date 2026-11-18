"""Tests for exact Doppler integration of per-pixel local spectra."""

import jax
import jax.numpy as jnp
import numpy as np

from doraex.operators.design_matrix import (
    full_design_matrix_from_angles,
    full_design_matrix_from_times,
    integrate_pixel_profiles_from_angles,
    integrate_pixel_profiles_from_times,
    local_pixel_pressure_operator_from_angles,
    local_pixel_pressure_operator_from_times,
)
from doraex.operators.doppler import doppler_factor, shifted_pixel_profiles


jax.config.update("jax_enable_x64", True)


def test_shifted_pixel_profiles_interpolates_each_local_spectrum():
    wavelengths = jnp.asarray([100.0, 101.0, 102.0])
    rest_wavelengths = jnp.linspace(98.0, 104.0, 25)
    rest_profiles = jnp.stack(
        (
            0.2 * rest_wavelengths + 1.0,
            -0.1 * rest_wavelengths + 12.0,
            0.01 * rest_wavelengths**2,
        )
    )
    factors = doppler_factor(jnp.asarray([-1000.0, 0.0, 1000.0]))

    actual = shifted_pixel_profiles(
        wavelengths,
        rest_profiles,
        factors,
        rest_wavelengths=rest_wavelengths,
    )
    expected = np.stack(
        [
            np.interp(
                np.asarray(wavelengths / factors[index]),
                np.asarray(rest_wavelengths),
                np.asarray(rest_profiles[index]),
            )
            for index in range(rest_profiles.shape[0])
        ],
        axis=1,
    )

    assert actual.shape == (wavelengths.size, rest_profiles.shape[0])
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_uniform_pixel_profiles_match_the_existing_design_matrix():
    theta = jnp.asarray([0.55, 1.05, 1.55, 2.05])
    phi = jnp.asarray([-1.1, -0.3, 0.6, 1.5])
    phases = jnp.asarray([0.0, 0.3])
    weights = jnp.asarray([1.0, 1.04])
    wavelengths = jnp.linspace(100.0, 102.0, 5)
    rest_wavelengths = jnp.linspace(99.5, 102.5, 31)
    shared_profile = 1.0 - 0.25 * jnp.exp(
        -0.5 * ((rest_wavelengths - 101.0) / 0.18) ** 2
    )
    rest_profiles = jnp.broadcast_to(
        shared_profile,
        (theta.size, rest_wavelengths.size),
    )

    actual = integrate_pixel_profiles_from_angles(
        theta,
        phi,
        31.2,
        jnp.deg2rad(61.0),
        0.45,
        0.18,
        phases,
        wavelengths,
        rest_profiles,
        weights=weights,
        pixel_area=0.7,
        rest_wavelengths=rest_wavelengths,
    )
    matrix = full_design_matrix_from_angles(
        theta,
        phi,
        31.2,
        jnp.deg2rad(61.0),
        0.45,
        0.18,
        phases,
        wavelengths,
        shared_profile,
        weights=weights,
        pixel_area=0.7,
        rest_wavelengths=rest_wavelengths,
    )
    expected = (matrix @ jnp.ones(theta.size)).reshape(
        phases.size,
        wavelengths.size,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_time_integration_matches_a_manual_sum_over_pixel_operators():
    theta = jnp.asarray([0.6, 1.2, 1.8])
    phi = jnp.asarray([-0.9, 0.2, 1.3])
    obs_times = jnp.asarray([0.2, 0.8, 1.7])
    period = 2.4
    t0 = 0.1
    weights = jnp.asarray([0.97, 1.0, 1.03])
    wavelengths = jnp.linspace(23040.0, 23044.0, 7)
    rest_wavelengths = jnp.linspace(23035.0, 23049.0, 57)
    centers = jnp.asarray([23041.0, 23042.0, 23043.0])
    depths = jnp.asarray([0.1, 0.2, 0.3])
    rest_profiles = 1.0 - depths[:, None] * jnp.exp(
        -0.5 * ((rest_wavelengths[None, :] - centers[:, None]) / 0.35) ** 2
    )

    actual = integrate_pixel_profiles_from_times(
        theta,
        phi,
        28.0,
        jnp.deg2rad(57.0),
        0.35,
        0.12,
        obs_times,
        period,
        wavelengths,
        rest_profiles,
        weights=weights,
        t0=t0,
        pixel_area=0.8,
        rest_wavelengths=rest_wavelengths,
    )

    expected = jnp.zeros((obs_times.size * wavelengths.size,))
    for pixel_index in range(theta.size):
        matrix = full_design_matrix_from_times(
            theta[pixel_index : pixel_index + 1],
            phi[pixel_index : pixel_index + 1],
            28.0,
            jnp.deg2rad(57.0),
            0.35,
            0.12,
            obs_times,
            period,
            wavelengths,
            rest_profiles[pixel_index],
            weights=weights,
            t0=t0,
            pixel_area=0.8,
            rest_wavelengths=rest_wavelengths,
        )
        expected = expected + matrix[:, 0]
    expected = expected.reshape(obs_times.size, wavelengths.size)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_exact_pixel_operator_supports_jit_and_profile_gradients():
    theta = jnp.asarray([0.7, 1.4])
    phi = jnp.asarray([-0.4, 0.8])
    wavelengths = jnp.linspace(1.0, 1.4, 6)
    rest_profiles = jnp.asarray(
        [
            [1.00, 0.96, 0.90, 0.93, 0.98, 1.01],
            [1.02, 0.99, 0.95, 0.91, 0.96, 1.00],
        ]
    )

    def integrate(profiles):
        return integrate_pixel_profiles_from_angles(
            theta,
            phi,
            20.0,
            jnp.deg2rad(50.0),
            0.3,
            0.1,
            jnp.asarray([0.0, 0.4]),
            wavelengths,
            profiles,
        )

    spectra = jax.jit(integrate)(rest_profiles)
    gradient = jax.jit(jax.grad(lambda profiles: jnp.sum(integrate(profiles) ** 2)))(
        rest_profiles
    )

    assert spectra.shape == (2, wavelengths.size)
    assert gradient.shape == rest_profiles.shape
    assert bool(jnp.all(jnp.isfinite(spectra)))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.any(gradient != 0.0))


def test_local_pixel_pressure_operator_matches_finite_differences():
    theta = jnp.asarray([0.55, 1.15, 1.75])
    phi = jnp.asarray([-0.8, 0.1, 1.2])
    phases = jnp.asarray([0.0, 0.27, 0.61])
    weights = jnp.asarray([0.96, 1.0, 1.04])
    wavelengths = jnp.linspace(100.0, 102.0, 6)
    rest_wavelengths = jnp.linspace(99.5, 102.5, 41)
    centers = jnp.asarray([100.6, 101.0, 101.4])
    base_profiles = 1.0 - jnp.asarray([0.12, 0.18, 0.24])[:, None] * jnp.exp(
        -0.5
        * ((rest_wavelengths[None, :] - centers[:, None]) / 0.22) ** 2
    )
    pressure_derivatives = jnp.asarray([0.08, -0.05, 0.11])[:, None] * (
        rest_wavelengths[None, :] - centers[:, None]
    )

    baseline, jacobian = local_pixel_pressure_operator_from_angles(
        theta,
        phi,
        29.0,
        jnp.deg2rad(58.0),
        0.4,
        0.15,
        phases,
        wavelengths,
        base_profiles,
        pressure_derivatives,
        weights=weights,
        pixel_area=0.75,
        rest_wavelengths=rest_wavelengths,
    )

    def prediction(delta_log_pressure):
        profiles = base_profiles + (
            delta_log_pressure[:, None] * pressure_derivatives
        )
        return integrate_pixel_profiles_from_angles(
            theta,
            phi,
            29.0,
            jnp.deg2rad(58.0),
            0.4,
            0.15,
            phases,
            wavelengths,
            profiles,
            weights=weights,
            pixel_area=0.75,
            rest_wavelengths=rest_wavelengths,
        ).reshape(-1)

    origin = jnp.zeros(theta.size)
    epsilon = 1.0e-6
    finite_difference = jnp.stack(
        [
            (
                prediction(origin + epsilon * jnp.eye(theta.size)[index])
                - prediction(origin - epsilon * jnp.eye(theta.size)[index])
            )
            / (2.0 * epsilon)
            for index in range(theta.size)
        ],
        axis=1,
    )
    expected_baseline = prediction(origin)

    assert baseline.shape == (phases.size * wavelengths.size,)
    assert jacobian.shape == (phases.size * wavelengths.size, theta.size)
    np.testing.assert_allclose(baseline, expected_baseline, rtol=1.0e-12)
    np.testing.assert_allclose(
        jacobian,
        finite_difference,
        rtol=2.0e-7,
        atol=2.0e-10,
    )


def test_local_pixel_pressure_time_operator_matches_angles():
    theta = jnp.asarray([0.7, 1.3])
    phi = jnp.asarray([-0.5, 0.9])
    obs_times = jnp.asarray([0.2, 0.8])
    period = 1.6
    t0 = 0.1
    wavelengths = jnp.linspace(1.0, 1.5, 5)
    profiles = jnp.asarray(
        [[1.0, 0.9, 0.8, 0.9, 1.0], [0.95, 0.88, 0.84, 0.92, 1.02]]
    )
    derivatives = jnp.asarray(
        [[0.1, 0.05, 0.0, -0.05, -0.1], [-0.03, 0.0, 0.04, 0.0, -0.03]]
    )
    arguments = (
        theta,
        phi,
        24.0,
        jnp.deg2rad(53.0),
        0.32,
        0.09,
    )

    actual = local_pixel_pressure_operator_from_times(
        *arguments,
        obs_times,
        period,
        wavelengths,
        profiles,
        derivatives,
        t0=t0,
    )
    expected = local_pixel_pressure_operator_from_angles(
        *arguments,
        (obs_times - t0) / period,
        wavelengths,
        profiles,
        derivatives,
    )

    np.testing.assert_allclose(actual[0], expected[0], rtol=1.0e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1.0e-12)
