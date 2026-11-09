"""Focused tests for direct M9 exact-RT primary products."""

from pathlib import Path
from types import SimpleNamespace
import sys

import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "examples" / "luhman16b_yama"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import make_m9_v1_exact_products as products  # noqa: E402


def _exact_archive():
    delta = np.asarray(
        [
            np.linspace(-0.3, 0.3, 12),
            np.roll(np.linspace(-0.3, 0.3, 12), 3),
            np.roll(np.linspace(-0.3, 0.3, 12), 7),
        ]
    )
    center = np.asarray([1.0, 1.2, 0.8])
    return {
        "exact_pixel_rt": np.asarray(True),
        "opacity_mode": np.asarray("diffgrid"),
        "zero_mean_pressure_map": np.asarray(True),
        "nside": np.asarray(1),
        "pressure_map_latent": np.empty((3, 11)),
        "delta_log_p": delta,
        "log_p_cloud": center,
        "log_p_cloud_map": center[:, None] + delta,
    }


def test_direct_pressure_map_moments_use_saved_exact_samples():
    samples = _exact_archive()
    sample_count, pixel_count = products._validate_exact_archive(samples)
    moments = products._direct_pressure_map_moments(samples, np.arange(3))

    assert sample_count == 3
    assert pixel_count == 12
    np.testing.assert_allclose(
        moments["perturbation_mean"],
        np.mean(samples["delta_log_p"], axis=0),
    )
    np.testing.assert_allclose(
        moments["perturbation_var"],
        np.var(samples["delta_log_p"], axis=0),
    )
    np.testing.assert_allclose(
        moments["log_p_cloud_mean"],
        np.mean(samples["log_p_cloud_map"], axis=0),
    )
    pressure = 10.0 ** samples["log_p_cloud_map"]
    np.testing.assert_allclose(moments["p_cloud_mean"], np.mean(pressure, axis=0))
    np.testing.assert_allclose(moments["p_cloud_std"], np.std(pressure, axis=0))


def test_exact_archive_rejects_deterministic_map_drift():
    samples = _exact_archive()
    samples["log_p_cloud_map"] = samples["log_p_cloud_map"].copy()
    samples["log_p_cloud_map"][0, 0] += 1.0e-3

    with pytest.raises(ValueError, match=r"center \+ perturbation"):
        products._validate_exact_archive(samples)


def test_exact_archive_rejects_unknown_local_normalization():
    samples = _exact_archive()
    samples["exact_rt_local_normalization"] = np.asarray("unknown")

    with pytest.raises(ValueError, match="exact_rt_local_normalization"):
        products._validate_exact_archive(samples)


def test_exact_response_rebuild_preserves_archived_local_normalization(
    monkeypatch,
):
    captured = []

    def capture(args, chip_data_list, profile_wavelengths):
        del chip_data_list, profile_wavelengths
        captured.append(args.exact_rt_local_normalization)
        return []

    monkeypatch.setattr(products, "build_response_functions", capture)
    args = SimpleNamespace(
        database_dir=Path("."),
        opacity_cache_dir=Path("."),
        nx=8,
    )
    samples = _exact_archive()
    samples.update(
        {
            "diffgrid_temperature_nodes": np.asarray(21),
            "rt_pixel_batch_size": np.asarray(16),
            "exact_rt_local_normalization": np.asarray("none"),
        }
    )

    assert products._build_exact_responses(args, [], [], samples) == []
    del samples["exact_rt_local_normalization"]
    assert products._build_exact_responses(args, [], [], samples) == []

    assert captured == ["none", "per_pixel_mean"]


def test_exact_model_uses_pixel_pressures_and_uniform_normalization(monkeypatch):
    captured = {}

    def response(*arguments):
        captured["pressures"] = np.asarray(arguments[-1])
        return jnp.asarray(
            [
                [2.0, 4.0],
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )

    def integrate(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        obs_times,
        period,
        wavelengths,
        profiles,
        *,
        weights,
        rest_wavelengths,
    ):
        captured["integration_inputs"] = (
            np.asarray(theta),
            np.asarray(phi),
            float(vrot),
            float(inclination),
            float(u1),
            float(u2),
            np.asarray(obs_times),
            float(period),
            np.asarray(wavelengths),
            np.asarray(weights),
            np.asarray(rest_wavelengths),
        )
        profiles = jnp.asarray(profiles)
        captured.setdefault("profile_calls", []).append(np.asarray(profiles))
        return jnp.sum(profiles, axis=0)[None, :]

    monkeypatch.setattr(products, "integrate_pixel_profiles_from_times", integrate)
    geometry = SimpleNamespace(
        theta=jnp.asarray([0.4, 1.2]),
        phi=jnp.asarray([0.1, 2.1]),
    )
    chip_data = SimpleNamespace(
        obs_times=np.asarray([0.0]),
        wavelengths=np.asarray([2.0, 2.1]),
        flux=np.zeros((1, 2)),
    )
    sample = {
        "T0": 1200.0,
        "alpha": 0.13,
        "log_vmr_co": -3.0,
        "log_vmr_h2o": -3.3,
        "log_vmr_ch4": -4.7,
        "log_vmr_hf": -7.1,
        "logg": 4.86,
        "log_p_cloud": 1.0,
        "v": 31.2,
        "cosi": 0.485,
        "u1": 0.3,
        "u2": 0.2,
        "P": 4.83,
        "A": np.asarray([2.0]),
        "log_w": np.zeros((1, 1)),
    }

    model = products._exact_model_for_chip(
        response,
        chip_data,
        geometry,
        np.asarray([1.9, 2.0, 2.1]),
        sample,
        np.asarray([-0.2, 0.2]),
        0,
    )

    np.testing.assert_allclose(captured["pressures"], [1.0, 0.8, 1.2])
    theta, phi, vrot, inclination, u1, u2, times, period, wavelengths, weights, rest = (
        captured["integration_inputs"]
    )
    np.testing.assert_allclose(theta, geometry.theta)
    np.testing.assert_allclose(phi, geometry.phi)
    assert vrot == pytest.approx(31.2)
    assert inclination == pytest.approx(np.arccos(0.485))
    assert u1 == pytest.approx(0.3)
    assert u2 == pytest.approx(0.2)
    np.testing.assert_allclose(times, [0.0])
    assert period == pytest.approx(4.83)
    np.testing.assert_allclose(wavelengths, [2.0, 2.1])
    np.testing.assert_allclose(weights, [1.0])
    np.testing.assert_allclose(rest, [1.9, 2.0, 2.1])
    np.testing.assert_allclose(captured["profile_calls"][0], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(captured["profile_calls"][1], [[2.0, 4.0], [2.0, 4.0]])
    np.testing.assert_allclose(model, [[1.0 / 3.0, 0.5]])
