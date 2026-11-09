"""Tests for the M9 exact pixel-RT pressure-map model."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers

from doraex.workflows import on_the_fly_pressure_retrieval as retrieval
from doraex.workflows.on_the_fly_pressure_retrieval import (
    on_the_fly_pressure_model,
    pressure_map_from_whitened_latent,
    whitened_latent_from_pressure_map,
)


def test_pressure_map_from_whitened_latent_is_zero_mean_and_differentiable():
    latent = jnp.asarray([0.4, -0.7])
    eigenvalues = jnp.asarray([2.0, 0.5])
    eigenvectors = jnp.asarray(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.0, -2.0],
        ]
    ) / jnp.sqrt(jnp.asarray([2.0, 6.0]))
    sigma = 0.3
    jitter = 5.0e-7

    pressure_map = pressure_map_from_whitened_latent(
        latent,
        sigma,
        eigenvalues,
        eigenvectors,
        jitter,
    )
    jacobian = jax.jacrev(pressure_map_from_whitened_latent)(
        latent,
        sigma,
        eigenvalues,
        eigenvectors,
        jitter,
    )

    expected_jacobian = eigenvectors * jnp.sqrt(
        sigma**2 * eigenvalues + jitter
    )[None, :]
    np.testing.assert_allclose(np.asarray(jnp.sum(pressure_map)), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        np.asarray(jacobian),
        np.asarray(expected_jacobian),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    recovered_latent = whitened_latent_from_pressure_map(
        pressure_map,
        sigma,
        eigenvalues,
        eigenvectors,
        jitter,
    )
    np.testing.assert_allclose(recovered_latent, latent, rtol=1.0e-6)


def test_exact_response_selects_local_normalization_mode(monkeypatch):
    calls = []

    class StubColumnModel:
        def __init__(self, observed_wavelengths, **kwargs):
            del kwargs
            self.profile_size = len(observed_wavelengths)

        def _evaluate(self, mode, *args, rt_batch_size, **kwargs):
            del kwargs
            pressures = jnp.asarray(args[-1])
            calls.append((mode, rt_batch_size))
            value = 1.0 if mode == "per_pixel_mean" else 2.0
            return jnp.full((pressures.size, self.profile_size), value)

        def cloudy_map_at_log_vmrs(self, *args, rt_batch_size, **kwargs):
            return self._evaluate(
                "per_pixel_mean",
                *args,
                rt_batch_size=rt_batch_size,
                **kwargs,
            )

        def cloudy_raw_map_at_log_vmrs(self, *args, rt_batch_size, **kwargs):
            return self._evaluate(
                "none",
                *args,
                rt_batch_size=rt_batch_size,
                **kwargs,
            )

    monkeypatch.setattr(retrieval, "Luhman16BPowerLawColumnModel", StubColumnModel)
    chip = SimpleNamespace(chip_index=0, wavelengths=np.linspace(2.0, 2.2, 3))
    base_args = {
        "database_dir": Path("."),
        "opacity_cache_dir": Path("."),
        "nx": 8,
        "opacity_mode": "diffgrid",
        "diffgrid_temperature_nodes": 3,
        "exact_pixel_rt": True,
        "rt_pixel_batch_size": 16,
    }
    pressures = jnp.asarray([0.8, 1.2])

    for mode, expected in (("per_pixel_mean", 1.0), ("none", 2.0)):
        args = SimpleNamespace(
            **base_args,
            exact_rt_local_normalization=mode,
        )
        response = retrieval.build_response_functions(args, [chip])[0]
        spectra = response(*([jnp.asarray(0.0)] * 7), pressures)
        np.testing.assert_allclose(spectra, expected)

    assert calls == [("per_pixel_mean", 16), ("none", 16)]

    invalid_args = SimpleNamespace(
        **{
            **base_args,
            "exact_pixel_rt": False,
            "exact_rt_local_normalization": "none",
        }
    )
    with np.testing.assert_raises_regex(ValueError, "requires --exact-pixel-rt"):
        retrieval.build_response_functions(invalid_args, [chip])


def test_exact_model_samples_explicit_map_and_uses_diagonal_likelihood(
    monkeypatch,
):
    captured_profiles = []

    def fake_integrate(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        obs_times,
        period,
        wavelengths,
        pixel_profiles,
        weights=None,
        **kwargs,
    ):
        del theta, phi, vrot, inclination, u1, u2, period, weights, kwargs
        captured_profiles.append(pixel_profiles)
        mean_profile = jnp.mean(pixel_profiles, axis=0)
        return jnp.broadcast_to(
            mean_profile,
            (len(obs_times), len(wavelengths)),
        )

    monkeypatch.setattr(
        retrieval,
        "integrate_pixel_profiles_from_times",
        fake_integrate,
    )

    n_phase = 2
    n_wavelength = 3
    eigenvalues = jnp.asarray([2.0])
    eigenvectors = jnp.asarray([[1.0], [-1.0]]) / jnp.sqrt(2.0)
    latent = jnp.asarray([0.5])
    log_p_cloud = 1.2
    sigma_log_p = 0.2
    gp_jitter = 5.0e-7
    noise_jitter = 1.0e-6
    sigma_d = 0.03
    wavelengths = jnp.linspace(2.2, 2.3, n_wavelength)

    def exact_response(*args):
        pressures = args[-1]
        return pressures[:, None] + jnp.arange(n_wavelength)[None, :] + 1.0

    def model():
        return on_the_fly_pressure_model(
            data=jnp.zeros((1, n_phase, n_wavelength)),
            wavelengths=[wavelengths],
            obs_times=jnp.arange(n_phase, dtype=jnp.float32),
            theta=jnp.asarray([0.5, 1.5]),
            phi=jnp.asarray([0.0, 2.0]),
            distance_matrix=jnp.asarray([[0.0, 1.0], [1.0, 0.0]]),
            response_functions=[exact_response],
            fixed_period=4.83,
            fixed_cosi=0.485,
            fixed_v=31.2,
            fixed_q1=0.81,
            fixed_q2=0.59,
            fix_logg=True,
            fixed_logg=4.86,
            logg_prior_mean=4.86,
            logg_prior_sigma=0.09,
            logg_bounds=(4.59, 5.13),
            t0_bounds=(1000.0, 1700.0),
            alpha_bounds=(0.05, 0.20),
            log_vmr_bounds={
                "log_vmr_co": (-6.0, -1.0),
                "log_vmr_h2o": (-6.0, -1.0),
                "log_vmr_ch4": (-6.0, -1.0),
                "log_vmr_hf": (-10.0, -5.0),
            },
            log_p_cloud_bounds=(-2.0, 2.0),
            sigma_log_p_scale=0.3,
            standardized_parameters=False,
            gaussianized_atmosphere=False,
            atmosphere_rotation=jnp.eye(8),
            parameter_centers={},
            parameter_scales={"rotated_atmosphere": False},
            fixed_ell_b=0.4,
            zero_mean_pressure_map=True,
            log_w_scale=0.1,
            zero_mean_log_w=True,
            zero_sum_log_w_basis=True,
            eigen_mode_count=None,
            eigen_sigma_scale=1.0,
            eigen_sigma_center=jnp.asarray(1.0),
            eigen_sigma_log_raw_scale=jnp.asarray(0.5),
            fixed_sigma_eigen=None,
            eigen_fixed_ell=0.4,
            fixed_nuisance_values={
                "A": jnp.asarray([1.0]),
                "log_w": jnp.zeros((1, n_phase)),
                "sigma_d": jnp.asarray([sigma_d]),
            },
            gp_jitter=gp_jitter,
            noise_jitter=noise_jitter,
            fixed_sigma_log_p=sigma_log_p,
            pressure_gp_eigenvalues=eigenvalues,
            pressure_gp_pixel_eigenvectors=eigenvectors,
            observation_indices=np.asarray([0, 2, 5]),
            exact_pixel_rt=True,
        )

    conditioned = handlers.condition(
        model,
        data={
            "T0": jnp.asarray(1219.0),
            "alpha": jnp.asarray(0.129),
            "log_vmr_co": jnp.asarray(-2.96),
            "log_vmr_h2o": jnp.asarray(-3.25),
            "log_vmr_ch4": jnp.asarray(-4.65),
            "log_vmr_hf": jnp.asarray(-7.08),
            "log_p_cloud": jnp.asarray(log_p_cloud),
            "pressure_map_latent": latent,
        },
    )
    trace = handlers.trace(handlers.seed(conditioned, jax.random.key(0))).get_trace()

    expected_delta = pressure_map_from_whitened_latent(
        latent,
        sigma_log_p,
        eigenvalues,
        eigenvectors,
        gp_jitter,
    )
    expected_map = log_p_cloud + expected_delta
    np.testing.assert_allclose(
        trace["delta_log_p"]["value"],
        expected_delta,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        trace["log_p_cloud_map"]["value"],
        expected_map,
        rtol=1.0e-6,
    )
    assert trace["pressure_map_latent"]["value"].shape == (1,)
    assert captured_profiles[0].shape == (2, n_wavelength)
    assert captured_profiles[1].shape == (2, n_wavelength)

    exact_profile = expected_map[:, None] + jnp.arange(n_wavelength)[None, :] + 1.0
    uniform_profile = (
        log_p_cloud + jnp.arange(n_wavelength, dtype=jnp.float32) + 1.0
    )
    raw_exact = jnp.mean(exact_profile, axis=0)
    normalized = jnp.tile(raw_exact / jnp.mean(uniform_profile), n_phase)
    selected = normalized[jnp.asarray([0, 2, 5])]
    likelihood = trace["obs"]["fn"].base_dist
    np.testing.assert_allclose(likelihood.loc, selected, rtol=1.0e-6)
    np.testing.assert_allclose(
        likelihood.scale,
        jnp.sqrt(sigma_d**2 + noise_jitter),
        rtol=1.0e-6,
    )
    assert trace["obs"]["value"].shape == (3,)
