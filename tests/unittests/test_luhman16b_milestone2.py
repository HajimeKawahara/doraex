"""Smoke tests for Milestone 2-1 fixed two-column retrieval."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.inference.numpyro_models import (
    fixed_two_column_doppler_model,
    free_cloud_two_column_doppler_model,
    free_t0_cloud_two_column_doppler_model,
)
from doraex.spectra.exojax_forward import (
    synthetic_cloud_profile_grid,
    synthetic_t0_cloud_profile_grid,
    synthetic_two_column_profiles,
)
from doraex.workflows.luhman16b_milestone1 import build_luhman16b_geometry
from doraex.workflows.luhman16b_milestone2 import (
    run_fixed_two_column_mcmc,
    run_free_cloud_two_column_mcmc,
    run_free_t0_cloud_two_column_mcmc,
)


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
        sigma_b_scale=0.1,
        fixed_ell_b=0.4,
        fix_geometry=True,
    )

    assert trace["obs"]["value"].shape == (chip.flux.size,)
    assert trace["f_cloud"]["value"].shape == ()
    assert trace["surface_scale"]["value"].shape == ()
    assert trace["ell_b"]["value"].shape == ()
    assert trace["cosi"]["value"].shape == ()
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
        sigma_b_scale=0.1,
        fixed_ell_b=0.4,
        fix_geometry=True,
        progress_bar=False,
    )
    samples = mcmc.get_samples()

    assert samples["f_cloud"].shape == (2,)
    assert samples["surface_scale"].shape == (2,)
    assert samples["sigma_b"].shape == (2,)
    assert np.isfinite(np.asarray(samples["f_cloud"])).all()
    assert np.isfinite(np.asarray(samples["surface_scale"])).all()


def test_free_cloud_two_column_model_trace_smoke():
    chip, geometry, _, _ = _small_inputs()
    log_p_cloud_grid = np.linspace(0.0, 2.0, 5)
    clear_profile, cloudy_profile_grid = synthetic_cloud_profile_grid(
        chip.wavelengths,
        log_p_cloud_grid,
    )
    seeded_model = handlers.seed(
        free_cloud_two_column_doppler_model,
        jax.random.PRNGKey(0),
    )
    trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(chip.flux),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip.obs_times),
        jnp.asarray(chip.wavelengths),
        jnp.asarray(clear_profile),
        jnp.asarray(log_p_cloud_grid),
        jnp.asarray(cloudy_profile_grid),
        period_mode="fixed",
    )

    assert trace["obs"]["value"].shape == (chip.flux.size,)
    assert trace["log_p_cloud"]["value"].shape == ()
    assert trace["f_cloud"]["value"].shape == ()
    assert trace["surface_scale"]["value"].shape == ()


def test_free_cloud_two_column_mcmc_smoke():
    chip, geometry, _, _ = _small_inputs()
    log_p_cloud_grid = np.linspace(0.0, 2.0, 5)
    clear_profile, cloudy_profile_grid = synthetic_cloud_profile_grid(
        chip.wavelengths,
        log_p_cloud_grid,
    )
    mcmc = run_free_cloud_two_column_mcmc(
        chip,
        geometry,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        num_warmup=2,
        num_samples=2,
        dense_mass=False,
        max_tree_depth=3,
        progress_bar=False,
    )
    samples = mcmc.get_samples()

    assert samples["log_p_cloud"].shape == (2,)
    assert samples["f_cloud"].shape == (2,)
    assert np.isfinite(np.asarray(samples["log_p_cloud"])).all()


def test_free_t0_cloud_two_column_model_trace_smoke():
    chip, geometry, _, _ = _small_inputs()
    t0_grid = np.linspace(1000.0, 1700.0, 5)
    log_p_cloud_grid = np.linspace(-2.0, 2.0, 5)
    clear_profile_grid, cloudy_profile_grid = synthetic_t0_cloud_profile_grid(
        chip.wavelengths,
        t0_grid,
        log_p_cloud_grid,
    )
    seeded_model = handlers.seed(
        free_t0_cloud_two_column_doppler_model,
        jax.random.PRNGKey(0),
    )
    trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(chip.flux),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip.obs_times),
        jnp.asarray(chip.wavelengths),
        jnp.asarray(t0_grid),
        jnp.asarray(log_p_cloud_grid),
        jnp.asarray(clear_profile_grid),
        jnp.asarray(cloudy_profile_grid),
        period_mode="fixed",
    )

    assert trace["obs"]["value"].shape == (chip.flux.size,)
    assert trace["T0"]["value"].shape == ()
    assert trace["log_p_cloud"]["value"].shape == ()
    assert trace["f_cloud"]["value"].shape == ()
    assert trace["surface_scale"]["value"].shape == ()


def test_free_t0_cloud_two_column_mcmc_smoke():
    chip, geometry, _, _ = _small_inputs()
    t0_grid = np.linspace(1000.0, 1700.0, 5)
    log_p_cloud_grid = np.linspace(-2.0, 2.0, 5)
    clear_profile_grid, cloudy_profile_grid = synthetic_t0_cloud_profile_grid(
        chip.wavelengths,
        t0_grid,
        log_p_cloud_grid,
    )
    mcmc = run_free_t0_cloud_two_column_mcmc(
        chip,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
        num_warmup=2,
        num_samples=2,
        dense_mass=False,
        max_tree_depth=3,
        progress_bar=False,
    )
    samples = mcmc.get_samples()

    assert samples["T0"].shape == (2,)
    assert samples["log_p_cloud"].shape == (2,)
    assert samples["f_cloud"].shape == (2,)
    assert np.isfinite(np.asarray(samples["T0"])).all()


def test_free_t0_cloud_two_column_free_ell_mcmc_smoke():
    chip, geometry, _, _ = _small_inputs()
    t0_grid = np.linspace(1000.0, 1700.0, 5)
    log_p_cloud_grid = np.linspace(-2.0, 2.0, 5)
    clear_profile_grid, cloudy_profile_grid = synthetic_t0_cloud_profile_grid(
        chip.wavelengths,
        t0_grid,
        log_p_cloud_grid,
    )
    mcmc = run_free_t0_cloud_two_column_mcmc(
        chip,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
        num_warmup=2,
        num_samples=2,
        dense_mass=False,
        max_tree_depth=3,
        fixed_ell_b=None,
        progress_bar=False,
    )
    samples = mcmc.get_samples()

    assert samples["ell_b"].shape == (2,)
    assert np.isfinite(np.asarray(samples["ell_b"])).all()
