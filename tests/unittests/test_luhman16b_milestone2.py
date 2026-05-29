"""Smoke tests for Milestone 2-1 fixed two-column retrieval."""

import argparse
import importlib.util
from pathlib import Path
import sys

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
    joint_free_t0_cloud_two_column_doppler_model,
)
from doraex.spectra.exojax_forward import (
    _opacity_cache_namespace,
    synthetic_cloud_profile_grid,
    synthetic_t0_cloud_profile_grid,
    synthetic_two_column_profiles,
)
from doraex.workflows.luhman16b_milestone1 import build_luhman16b_geometry
from doraex.workflows.luhman16b_milestone2 import (
    conditional_contrast_map_for_joint_free_t0_cloud_sample,
    run_fixed_two_column_mcmc,
    run_free_cloud_two_column_mcmc,
    run_free_t0_cloud_two_column_mcmc,
)


jax.config.update("jax_enable_x64", True)


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ROOT = Path(__file__).resolve().parents[2]


def _load_fixed_ell_sensitivity_script():
    script_path = (
        ROOT
        / "examples"
        / "luhman16b_yama"
        / "run_milestone2_fixed_ell_sensitivity.py"
    )
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("fixed_ell_sensitivity", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_chip_paths_script():
    script_path = ROOT / "examples" / "luhman16b_yama" / "chip_paths.py"
    spec = importlib.util.spec_from_file_location("chip_paths", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_chip_comparison_script():
    script_path = (
        ROOT
        / "examples"
        / "luhman16b_yama"
        / "summarize_milestone2_chip_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("chip_comparison", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_delta_sensitivity_script():
    script_path = (
        ROOT
        / "examples"
        / "luhman16b_yama"
        / "run_milestone3_delta_sensitivity.py"
    )
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("delta_sensitivity", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_joint_free_t0_cloud_two_column_model_trace_smoke():
    chip0 = subset_chip_data(
        load_luhman16b_chip(DATA_DIR, chip_index=0),
        wavelength_step=256,
        phase_count=2,
    )
    chip1 = subset_chip_data(
        load_luhman16b_chip(DATA_DIR, chip_index=1),
        wavelength_step=256,
        phase_count=2,
    )
    geometry = build_luhman16b_geometry(nside=1)
    t0_grid = np.linspace(1000.0, 1700.0, 5)
    log_p_cloud_grid = np.linspace(-2.0, 2.0, 5)
    clear_grids = []
    cloudy_grids = []
    for chip in (chip0, chip1):
        clear_profile_grid, cloudy_profile_grid = synthetic_t0_cloud_profile_grid(
            chip.wavelengths,
            t0_grid,
            log_p_cloud_grid,
        )
        clear_grids.append(clear_profile_grid)
        cloudy_grids.append(cloudy_profile_grid)
    seeded_model = handlers.seed(
        joint_free_t0_cloud_two_column_doppler_model,
        jax.random.PRNGKey(0),
    )
    trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(np.stack([chip0.flux, chip1.flux], axis=0)),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip0.obs_times),
        jnp.asarray(np.stack([chip0.wavelengths, chip1.wavelengths], axis=0)),
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_grids, axis=0)),
        period_mode="fixed",
        fixed_ell_b=0.3,
    )

    assert trace["obs"]["value"].shape == (chip0.flux.size + chip1.flux.size,)
    assert trace["T0"]["value"].shape == (2,)
    assert trace["log_p_cloud"]["value"].shape == (2,)
    assert trace["f_cloud"]["value"].shape == (2,)
    assert trace["surface_scale"]["value"].shape == (2,)
    assert trace["sigma_d"]["value"].shape == (2,)
    assert trace["sigma_b"]["value"].shape == ()

    shared_trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(np.stack([chip0.flux, chip1.flux], axis=0)),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip0.obs_times),
        jnp.asarray(np.stack([chip0.wavelengths, chip1.wavelengths], axis=0)),
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_grids, axis=0)),
        period_mode="fixed",
        fixed_ell_b=0.3,
        shared_atmosphere=True,
    )

    assert shared_trace["obs"]["value"].shape == (
        chip0.flux.size + chip1.flux.size,
    )
    assert shared_trace["T0"]["value"].shape == ()
    assert shared_trace["log_p_cloud"]["value"].shape == ()
    assert shared_trace["f_cloud"]["value"].shape == ()
    assert shared_trace["surface_scale"]["value"].shape == (2,)
    assert shared_trace["sigma_d"]["value"].shape == (2,)
    assert shared_trace["sigma_b"]["value"].shape == ()

    yama_trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(np.stack([chip0.flux, chip1.flux], axis=0)),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip0.obs_times),
        jnp.asarray(np.stack([chip0.wavelengths, chip1.wavelengths], axis=0)),
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_grids, axis=0)),
        period_mode="fixed",
        fixed_ell_b=0.3,
        shared_atmosphere=True,
        normalization_mode="yama",
    )

    assert yama_trace["obs"]["value"].shape == (
        chip0.flux.size + chip1.flux.size,
    )
    assert yama_trace["T0"]["value"].shape == ()
    assert yama_trace["log_p_cloud"]["value"].shape == ()
    assert yama_trace["f_cloud"]["value"].shape == ()
    assert yama_trace["A"]["value"].shape == (2,)
    assert "surface_scale" not in yama_trace
    assert yama_trace["sigma_d"]["value"].shape == (2,)

    zeta_vmr_grid = np.linspace(-0.5, 0.5, 3)
    clear_vmr_grids = [
        np.repeat(clear_grid[:, None, :], len(zeta_vmr_grid), axis=1)
        for clear_grid in clear_grids
    ]
    cloudy_vmr_grids = [
        np.repeat(cloudy_grid[:, :, None, :], len(zeta_vmr_grid), axis=2)
        for cloudy_grid in cloudy_grids
    ]
    vmr_trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(np.stack([chip0.flux, chip1.flux], axis=0)),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip0.obs_times),
        jnp.asarray(np.stack([chip0.wavelengths, chip1.wavelengths], axis=0)),
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_vmr_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_vmr_grids, axis=0)),
        zeta_vmr_grid=jnp.asarray(np.stack([zeta_vmr_grid, zeta_vmr_grid], axis=0)),
        period_mode="fixed",
        fixed_ell_b=0.3,
        shared_atmosphere=True,
        normalization_mode="yama",
    )

    assert vmr_trace["obs"]["value"].shape == (
        chip0.flux.size + chip1.flux.size,
    )
    assert vmr_trace["T0"]["value"].shape == ()
    assert vmr_trace["log_p_cloud"]["value"].shape == ()
    assert vmr_trace["zeta_vmr"]["value"].shape == ()
    assert vmr_trace["f_cloud"]["value"].shape == ()
    assert vmr_trace["A"]["value"].shape == (2,)

    alpha_grid = np.linspace(0.05, 0.20, 3)
    clear_alpha_vmr_grids = [
        np.repeat(clear_grid[:, None, None, :], len(alpha_grid), axis=1).repeat(
            len(zeta_vmr_grid),
            axis=2,
        )
        for clear_grid in clear_grids
    ]
    cloudy_alpha_vmr_grids = [
        np.repeat(cloudy_grid[:, None, :, None, :], len(alpha_grid), axis=1).repeat(
            len(zeta_vmr_grid),
            axis=3,
        )
        for cloudy_grid in cloudy_grids
    ]
    alpha_trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(np.stack([chip0.flux, chip1.flux], axis=0)),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip0.obs_times),
        jnp.asarray(np.stack([chip0.wavelengths, chip1.wavelengths], axis=0)),
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_alpha_vmr_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_alpha_vmr_grids, axis=0)),
        alpha_grid=jnp.asarray(np.stack([alpha_grid, alpha_grid], axis=0)),
        zeta_vmr_grid=jnp.asarray(np.stack([zeta_vmr_grid, zeta_vmr_grid], axis=0)),
        period_mode="fixed",
        fixed_ell_b=0.3,
        shared_atmosphere=True,
        normalization_mode="yama",
    )

    assert alpha_trace["obs"]["value"].shape == (
        chip0.flux.size + chip1.flux.size,
    )
    assert alpha_trace["T0"]["value"].shape == ()
    assert alpha_trace["alpha"]["value"].shape == ()
    assert alpha_trace["log_p_cloud"]["value"].shape == ()
    assert alpha_trace["zeta_vmr"]["value"].shape == ()
    assert alpha_trace["f_cloud"]["value"].shape == ()
    assert alpha_trace["A"]["value"].shape == (2,)
    assert yama_trace["sigma_b"]["value"].shape == ()

    double_cloud_trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(np.stack([chip0.flux, chip1.flux], axis=0)),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip0.obs_times),
        jnp.asarray(np.stack([chip0.wavelengths, chip1.wavelengths], axis=0)),
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_alpha_vmr_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_alpha_vmr_grids, axis=0)),
        alpha_grid=jnp.asarray(np.stack([alpha_grid, alpha_grid], axis=0)),
        zeta_vmr_grid=jnp.asarray(np.stack([zeta_vmr_grid, zeta_vmr_grid], axis=0)),
        period_mode="fixed",
        fixed_ell_b=0.3,
        shared_atmosphere=True,
        normalization_mode="yama",
        column_mode="double_cloud",
        fixed_cloud_delta=1.0,
        log_p_cloud_bounds=(-1.5, 1.5),
    )

    assert double_cloud_trace["obs"]["value"].shape == (
        chip0.flux.size + chip1.flux.size,
    )
    assert double_cloud_trace["T0"]["value"].shape == ()
    assert double_cloud_trace["alpha"]["value"].shape == ()
    assert double_cloud_trace["log_p_mid"]["value"].shape == ()
    assert double_cloud_trace["h_high"]["value"].shape == ()
    assert "log_p_cloud" not in double_cloud_trace
    assert "f_cloud" not in double_cloud_trace
    assert double_cloud_trace["A"]["value"].shape == (2,)

    pressure_trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(np.stack([chip0.flux, chip1.flux], axis=0)),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip0.obs_times),
        jnp.asarray(np.stack([chip0.wavelengths, chip1.wavelengths], axis=0)),
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_alpha_vmr_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_alpha_vmr_grids, axis=0)),
        alpha_grid=jnp.asarray(np.stack([alpha_grid, alpha_grid], axis=0)),
        zeta_vmr_grid=jnp.asarray(np.stack([zeta_vmr_grid, zeta_vmr_grid], axis=0)),
        period_mode="fixed",
        fixed_ell_b=0.3,
        shared_atmosphere=True,
        normalization_mode="yama",
        column_mode="pressure_perturbation",
        pressure_derivative_step=0.05,
        log_p_cloud_bounds=(-1.5, 1.5),
    )

    assert pressure_trace["obs"]["value"].shape == (
        chip0.flux.size + chip1.flux.size,
    )
    assert pressure_trace["T0"]["value"].shape == ()
    assert pressure_trace["alpha"]["value"].shape == ()
    assert pressure_trace["log_p_cloud"]["value"].shape == ()
    assert pressure_trace["zeta_vmr"]["value"].shape == ()
    assert pressure_trace["sigma_log_p"]["value"].shape == ()
    assert pressure_trace["sigma_b"]["value"].shape == ()
    assert "h_high" not in pressure_trace
    assert "f_cloud" not in pressure_trace
    assert pressure_trace["A"]["value"].shape == (2,)

    zero_mean_pressure_trace = handlers.trace(seeded_model).get_trace(
        jnp.asarray(np.stack([chip0.flux, chip1.flux], axis=0)),
        geometry.theta,
        geometry.phi,
        geometry.distance_matrix,
        jnp.asarray(chip0.obs_times),
        jnp.asarray(np.stack([chip0.wavelengths, chip1.wavelengths], axis=0)),
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_alpha_vmr_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_alpha_vmr_grids, axis=0)),
        alpha_grid=jnp.asarray(np.stack([alpha_grid, alpha_grid], axis=0)),
        zeta_vmr_grid=jnp.asarray(np.stack([zeta_vmr_grid, zeta_vmr_grid], axis=0)),
        period_mode="fixed",
        fixed_ell_b=0.3,
        shared_atmosphere=True,
        normalization_mode="yama",
        column_mode="pressure_perturbation",
        pressure_derivative_step=0.05,
        zero_mean_pressure_map=True,
        log_p_cloud_bounds=(-1.5, 1.5),
    )

    assert zero_mean_pressure_trace["obs"]["value"].shape == (
        chip0.flux.size + chip1.flux.size,
    )
    assert zero_mean_pressure_trace["sigma_log_p"]["value"].shape == ()

    sample = {
        "A": jnp.ones(2),
        "P": jnp.asarray(4.83),
        "T0": jnp.asarray(1215.0),
        "alpha": jnp.asarray(0.12),
        "cosi": jnp.asarray(0.485),
        "ell_b": jnp.asarray(0.3),
        "log_p_cloud": jnp.asarray(0.5),
        "log_w": jnp.zeros((2, chip0.flux.shape[0])),
        "q1": jnp.asarray(0.81),
        "q2": jnp.asarray(0.59),
        "sigma_b": jnp.asarray(0.2),
        "sigma_log_p": jnp.asarray(0.2),
        "sigma_d": jnp.asarray([0.05, 0.05]),
        "v": jnp.asarray(31.2),
        "zeta_vmr": jnp.asarray(0.0),
        "pressure_derivative_step": jnp.asarray(0.05),
        "zero_mean_pressure_map": jnp.asarray(True),
    }
    mean, covariance = conditional_contrast_map_for_joint_free_t0_cloud_sample(
        [chip0, chip1],
        geometry,
        jnp.asarray(np.stack([t0_grid, t0_grid], axis=0)),
        jnp.asarray(np.stack([log_p_cloud_grid, log_p_cloud_grid], axis=0)),
        jnp.asarray(np.stack(clear_alpha_vmr_grids, axis=0)),
        jnp.asarray(np.stack(cloudy_alpha_vmr_grids, axis=0)),
        sample,
        alpha_grid=jnp.asarray(np.stack([alpha_grid, alpha_grid], axis=0)),
        zeta_vmr_grid=jnp.asarray(np.stack([zeta_vmr_grid, zeta_vmr_grid], axis=0)),
    )
    assert abs(float(jnp.mean(mean))) < 1.0e-3
    assert abs(float(jnp.mean(jnp.sum(covariance, axis=0)))) < 1.0e-3


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


def test_fixed_ell_sensitivity_helpers():
    module = _load_fixed_ell_sensitivity_script()
    values = module.parse_ell_values("0.25, 0.30,0.4")

    assert values == [0.25, 0.30, 0.4]
    assert module.ell_tag(0.25) == "ell0p250"
    assert module.ell_tag(0.4) == "ell0p400"


def test_delta_sensitivity_helpers():
    module = _load_delta_sensitivity_script()
    values = module.parse_delta_values("0.03, 0.1,0.3")

    assert values == [0.03, 0.1, 0.3]
    assert module.delta_tag(0.03) == "delta0p030"
    assert module.midpoint_bounds(0.3) == (-1.85, 1.85)
    assert module.default_init_log_p_mid(0.1) == pytest.approx(1.4)


def test_chip_aware_milestone2_default_paths():
    module = _load_chip_paths_script()

    assert module.fixed_profile_path(0).name == "milestone2_fixed_profiles_chip0.npz"
    assert module.cloud_grid_path(2).name == "milestone2_cloud_grid_profiles_chip2.npz"
    assert (
        module.cloud_grid_path(3, wide=True).name
        == "milestone2_cloud_grid_profiles_wide_chip3.npz"
    )
    assert (
        module.t0_cloud_grid_path(1).name
        == "milestone2_t0_cloud_grid_profiles_chip1.npz"
    )
    assert (
        module.t0_cloud_grid_path(1, atmosphere_tag="exomol").name
        == "milestone2_t0_cloud_grid_profiles_exomol_chip1.npz"
    )
    assert (
        module.t0_vmr_cloud_grid_path(1, atmosphere_tag="exomol").name
        == "milestone2_t0_vmr_cloud_grid_profiles_exomol_chip1.npz"
    )
    assert (
        module.t0_alpha_vmr_cloud_grid_path(1, atmosphere_tag="exomol").name
        == "milestone2_t0_alpha_vmr_cloud_grid_profiles_exomol_chip1.npz"
    )
    assert (
        module.free_t0_cloud_sample_path("results/m2", 2, "fixed").name
        == "mcmc_chip2_fixed_free_t0_cloud.npz"
    )


def test_opacity_cache_namespace_depends_on_wavelength_grid():
    base = ROOT / "data" / "opacities" / "luhman16b_powerlaw"
    chip0 = load_luhman16b_chip(DATA_DIR, chip_index=0)
    chip1 = load_luhman16b_chip(DATA_DIR, chip_index=1)

    cache0 = _opacity_cache_namespace(
        base,
        chip0.wavelengths,
        nx=4500,
        pressure_top=1.0e-4,
        pressure_btm=1.0e2,
        nlayer=101,
        t_low=210.0,
        t_high=3500.0,
    )
    cache1 = _opacity_cache_namespace(
        base,
        chip1.wavelengths,
        nx=4500,
        pressure_top=1.0e-4,
        pressure_btm=1.0e2,
        nlayer=101,
        t_low=210.0,
        t_high=3500.0,
    )

    assert cache0.parent == base
    assert cache1.parent == base
    assert cache0 != cache1


def test_chip_comparison_summary_helpers(tmp_path):
    module = _load_chip_comparison_script()

    assert module.parse_chips("0, 1,3") == [0, 1, 3]
    for chip in (0, 1):
        out_dir = tmp_path / f"milestone2_3d_chip{chip}"
        out_dir.mkdir()
        np.savez(
            out_dir / f"mcmc_chip{chip}_fixed_free_t0_cloud.npz",
            T0=np.array([1200.0 + chip, 1210.0 + chip]),
            log_p_cloud=np.array([1.2, 1.3]),
            f_cloud=np.array([0.5, 0.6]),
            sigma_b=np.array([0.1, 0.2]),
            sigma_d=np.array([0.03, 0.04]),
            surface_scale=np.array([0.006, 0.007]),
            ell_b=np.array([0.3, 0.3]),
            fixed_ell_b=np.array(0.3),
            sigma_b_scale=np.array(0.1),
            nside=np.array(1),
            wavelengths=np.array([1.0, 2.0]),
            obs_times=np.array([0.0, 1.0]),
        )
        np.save(out_dir / f"residual_chip{chip}.npy", np.array([[0.0, 1.0]]))
        np.save(
            out_dir / f"cloud_fraction_mean_chip{chip}.npy",
            np.array([0.1, 0.2, 0.3]) + chip,
        )
        np.save(
            out_dir / f"contrast_mean_chip{chip}.npy",
            np.array([0.1, 0.2, 0.3]) + chip,
        )
        np.save(
            out_dir / f"delta_s_mean_chip{chip}.npy",
            np.array([0.3, 0.2, 0.1]) + chip,
        )

    args = argparse.Namespace(
        chips=[0, 1],
        results_template=str(tmp_path / "milestone2_3d_chip{chip}"),
        samples_template=None,
    )
    entries = []
    cloud_maps = {}
    contrast_maps = {}
    delta_s_maps = {}
    for chip in args.chips:
        entry, cloud_mean, contrast_mean, delta_s_mean = module._entry_for_chip(
            args, chip
        )
        entries.append(entry)
        cloud_maps[chip] = cloud_mean
        contrast_maps[chip] = contrast_mean
        delta_s_maps[chip] = delta_s_mean

    pairs = module._pairwise_map_metrics(
        entries,
        cloud_maps,
        contrast_maps,
        delta_s_maps,
    )

    assert entries[0]["sample_available"]
    assert entries[0]["products_available"]
    assert entries[0]["T0_mean"] == 1205.0
    assert entries[0]["residual_rms"] == pytest.approx(np.sqrt(0.5))
    assert entries[0]["cloud_fraction_mean_range"] == pytest.approx(0.2)
    assert pairs[0]["cloud_fraction_corr"] == pytest.approx(1.0)
    assert pairs[0]["delta_s_corr"] == pytest.approx(1.0)
    assert pairs[0]["contrast_corr"] == pytest.approx(1.0)
    assert pairs[0]["contrast_hot20_overlap"] == pytest.approx(1.0)
