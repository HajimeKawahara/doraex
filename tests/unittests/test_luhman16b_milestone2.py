"""Smoke tests for Milestone 2-1 fixed two-column retrieval."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.inference.numpyro_models import fixed_two_column_doppler_model
from doraex.spectra.exojax_forward import synthetic_two_column_profiles
from doraex.workflows.luhman16b_milestone1 import build_luhman16b_geometry
from doraex.workflows.luhman16b_milestone2 import run_fixed_two_column_mcmc


jax.config.update("jax_enable_x64", True)


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "fainterspectral-fits_6.pickle").exists(),
    reason="Milestone input data files are not available.",
)


def _small_inputs():
    chip = subset_chip_data(
        load_luhman16b_chip(DATA_DIR, chip_index=1),
        wavelength_step=256,
        phase_count=2,
    )
    geometry = build_luhman16b_geometry(nside=1)
    clear_profile, cloudy_profile = synthetic_two_column_profiles(chip.wavelengths)
    return chip, geometry, clear_profile, cloudy_profile


def test_fixed_two_column_model_trace_smoke():
    chip, geometry, clear_profile, cloudy_profile = _small_inputs()
    seeded_model = handlers.seed(fixed_two_column_doppler_model, jax.random.PRNGKey(0))
    trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(chip.flux),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip.obs_times),
        jnp.asarray(chip.wavelengths),
        jnp.asarray(clear_profile),
        jnp.asarray(cloudy_profile),
        period_mode="fixed",
    )

    assert trace["obs"]["value"].shape == (chip.flux.size,)
    assert trace["f_cloud"]["value"].shape == ()
    assert trace["surface_scale"]["value"].shape == ()
    assert trace["log_w"]["value"].shape == (chip.flux.shape[0],)


def test_fixed_two_column_mcmc_smoke():
    chip, geometry, clear_profile, cloudy_profile = _small_inputs()
    mcmc = run_fixed_two_column_mcmc(
        chip,
        geometry,
        clear_profile,
        cloudy_profile,
        num_warmup=2,
        num_samples=2,
        period_mode="fixed",
        dense_mass=False,
        max_tree_depth=3,
        progress_bar=False,
    )
    samples = mcmc.get_samples()

    assert samples["f_cloud"].shape == (2,)
    assert samples["surface_scale"].shape == (2,)
    assert samples["sigma_b"].shape == (2,)
    assert np.isfinite(np.asarray(samples["f_cloud"])).all()
    assert np.isfinite(np.asarray(samples["surface_scale"])).all()
