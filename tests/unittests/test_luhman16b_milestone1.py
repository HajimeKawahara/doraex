"""Smoke tests for the Milestone 1 Luhman 16B production path."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.inference.numpyro_models import luhman16b_ureshino_model
from doraex.workflows.luhman16b_milestone1 import (
    build_luhman16b_geometry,
    compute_posterior_map_moments,
    reconstruct_spectral_timeseries,
)


jax.config.update("jax_enable_x64", True)


DATA_DIR = Path(__file__).resolve().parents[2] / "external" / "BayesianDI" / "data"


pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "fainterspectral-fits_6.pickle").exists(),
    reason="Milestone 1 external data files are not available.",
)


def test_luhman16b_chip_loader_shapes():
    chip = load_luhman16b_chip(DATA_DIR, chip_index=1)

    assert chip.flux.shape == (14, 1024)
    assert chip.wavelengths.shape == (1024,)
    assert chip.line_profile.shape == (1024,)
    assert chip.obs_times.shape == (14,)
    assert np.all(np.diff(chip.wavelengths) > 0.0)


def test_luhman16b_numpyro_model_trace_smoke():
    chip = subset_chip_data(
        load_luhman16b_chip(DATA_DIR, chip_index=1),
        wavelength_step=128,
        phase_count=2,
    )
    geometry = build_luhman16b_geometry(nside=1)

    seeded_model = handlers.seed(luhman16b_ureshino_model, jax.random.PRNGKey(0))
    trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(chip.flux),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip.obs_times),
        jnp.asarray(chip.wavelengths),
        jnp.asarray(chip.line_profile),
        period_mode="fixed",
    )

    assert trace["obs"]["value"].shape == (chip.flux.size,)
    assert trace["u1"]["value"].shape == ()
    assert trace["log_w"]["value"].shape == (chip.flux.shape[0],)


def test_luhman16b_posterior_products_smoke():
    chip = subset_chip_data(
        load_luhman16b_chip(DATA_DIR, chip_index=1),
        wavelength_step=256,
        phase_count=2,
    )
    geometry = build_luhman16b_geometry(nside=1)
    sample_count = 2
    samples = {
        "cosi": np.array([0.5, 0.55]),
        "v": np.array([30.0, 31.0]),
        "q1": np.array([0.5, 0.55]),
        "q2": np.array([0.5, 0.45]),
        "u1": np.array([0.70710678, 0.6670832]),
        "u2": np.array([0.0, 0.07416198]),
        "log_w": np.zeros((sample_count, chip.flux.shape[0])),
        "sigma_d": np.array([0.04, 0.041]),
        "mu_a": np.array([0.008, 0.009]),
        "sigma_a": np.array([0.002, 0.0025]),
        "ell": np.array([0.4, 0.45]),
        "P": np.array([5.0, 5.0]),
    }

    mean, variance = compute_posterior_map_moments(chip, geometry, samples)
    model, median_sample = reconstruct_spectral_timeseries(chip, geometry, samples, mean)

    assert mean.shape == geometry.theta.shape
    assert variance.shape == geometry.theta.shape
    assert model.shape == chip.flux.shape
    assert np.isfinite(np.asarray(mean)).all()
    assert np.isfinite(np.asarray(variance)).all()
    assert np.isfinite(np.asarray(model)).all()
    assert float(median_sample["sigma_d"]) > 0.0
