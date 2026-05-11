"""Milestone 2 workflow helpers for fixed two-column Doppler retrieval."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_value

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.geometry.limb_darkening import kipping_q_to_u
from doraex.inference.map_posterior import conditional_map_posterior
from doraex.inference.numpyro_models import (
    fixed_two_column_doppler_model,
    free_cloud_two_column_doppler_model,
    _interpolate_profile_grid,
)
from doraex.operators.design_matrix import two_column_operator_from_times
from doraex.priors.spherical_gp import add_diagonal_jitter, squared_exponential_covariance
from doraex.spectra.exojax_forward import (
    load_cloud_profile_grid,
    load_two_column_profiles,
    save_cloud_profile_grid,
    save_two_column_profiles,
    synthetic_cloud_profile_grid,
    synthetic_two_column_profiles,
)
from doraex.workflows.luhman16b_milestone1 import build_luhman16b_geometry


def load_milestone2_fixed_inputs(
    data_dir,
    profiles_path=None,
    chip_index=1,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
):
    """Load Luhman 16B data, geometry, and fixed two-column spectra."""

    chip_data = load_luhman16b_chip(data_dir, chip_index=chip_index)
    if smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=smoke_wavelength_step,
            phase_count=smoke_phase_count,
        )
    geometry = build_luhman16b_geometry(nside=nside)
    if profiles_path is None:
        if not smoke_test:
            raise ValueError("profiles_path is required unless smoke_test=True")
        clear_profile, cloudy_profile = synthetic_two_column_profiles(chip_data.wavelengths)
    else:
        clear_profile, cloudy_profile = load_two_column_profiles(
            profiles_path, expected_wavelengths=chip_data.wavelengths
        )
    return chip_data, geometry, clear_profile, cloudy_profile


def load_milestone2_free_cloud_inputs(
    data_dir,
    profile_grid_path=None,
    chip_index=1,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
    smoke_log_p_cloud_grid=None,
):
    """Load Luhman 16B data, geometry, and a cloudy-profile grid."""

    chip_data = load_luhman16b_chip(data_dir, chip_index=chip_index)
    if smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=smoke_wavelength_step,
            phase_count=smoke_phase_count,
        )
    geometry = build_luhman16b_geometry(nside=nside)
    if profile_grid_path is None:
        if not smoke_test:
            raise ValueError("profile_grid_path is required unless smoke_test=True")
        if smoke_log_p_cloud_grid is None:
            smoke_log_p_cloud_grid = np.linspace(0.0, 2.0, 5)
        clear_profile, cloudy_profile_grid = synthetic_cloud_profile_grid(
            chip_data.wavelengths,
            smoke_log_p_cloud_grid,
        )
        log_p_cloud_grid = np.asarray(smoke_log_p_cloud_grid)
    else:
        clear_profile, log_p_cloud_grid, cloudy_profile_grid = load_cloud_profile_grid(
            profile_grid_path,
            expected_wavelengths=chip_data.wavelengths,
        )
    return chip_data, geometry, clear_profile, log_p_cloud_grid, cloudy_profile_grid


def make_fixed_two_column_nuts_kernel(
    model,
    n_phase,
    period_mode="sampled",
    fixed_period=5.0,
    fixed_ell_b=None,
    fix_geometry=False,
    target_accept_prob=0.9,
    dense_mass=True,
    max_tree_depth=10,
    init_log_p_cloud=None,
):
    """Create the NUTS kernel for Milestone 2."""

    init_values = {
        "log_w": jnp.zeros(n_phase),
        "f_cloud": 0.5,
        "surface_scale": 0.0077,
        "sigma_d": 0.039,
        "sigma_b": 0.05,
    }
    if not fix_geometry:
        init_values.update(
            {
                "cosi": 0.485,
                "v": 31.2,
                "q1": 0.81,
                "q2": 0.59,
            }
        )
    if fixed_ell_b is None:
        init_values["ell_b"] = 0.4
    if period_mode == "sampled":
        init_values["P"] = 4.83
    elif period_mode == "fixed":
        init_values["P"] = fixed_period
    if init_log_p_cloud is not None:
        init_values["log_p_cloud"] = init_log_p_cloud
    return NUTS(
        model,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_value(values=init_values),
    )


def run_fixed_two_column_mcmc(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    seed=0,
    period_mode="sampled",
    fixed_period=5.0,
    target_accept_prob=0.9,
    dense_mass=True,
    max_tree_depth=10,
    sigma_b_scale=0.1,
    fixed_ell_b=None,
    fix_geometry=False,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    progress_bar=True,
):
    """Run fixed-atmosphere two-column Doppler retrieval."""

    data = jnp.asarray(chip_data.flux)
    obs_times = jnp.asarray(chip_data.obs_times)
    wavelengths = jnp.asarray(chip_data.wavelengths)
    clear_profile = jnp.asarray(clear_profile)
    cloudy_profile = jnp.asarray(cloudy_profile)

    def model(data_observed):
        return fixed_two_column_doppler_model(
            data_observed,
            geometry.theta,
            geometry.phi,
            geometry.distance_matrix,
            obs_times,
            wavelengths,
            clear_profile,
            cloudy_profile,
            period_mode=period_mode,
            fixed_period=fixed_period,
            sigma_b_scale=sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=fix_geometry,
            fixed_cosi=fixed_cosi,
            fixed_v=fixed_v,
            fixed_q1=fixed_q1,
            fixed_q2=fixed_q2,
        )

    kernel = make_fixed_two_column_nuts_kernel(
        model,
        n_phase=chip_data.flux.shape[0],
        period_mode=period_mode,
        fixed_period=fixed_period,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=fix_geometry,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), data)
    return mcmc


def run_free_cloud_two_column_mcmc(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    num_warmup=500,
    num_samples=1000,
    num_chains=1,
    seed=0,
    period_mode="fixed",
    fixed_period=4.83,
    log_p_cloud_bounds=(0.0, 2.0),
    init_log_p_cloud=1.35,
    target_accept_prob=0.98,
    dense_mass=True,
    max_tree_depth=10,
    sigma_b_scale=0.1,
    fixed_ell_b=0.4,
    fix_geometry=True,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    progress_bar=True,
):
    """Run Milestone 2-2a with free cloud-top pressure."""

    data = jnp.asarray(chip_data.flux)
    obs_times = jnp.asarray(chip_data.obs_times)
    wavelengths = jnp.asarray(chip_data.wavelengths)
    clear_profile = jnp.asarray(clear_profile)
    log_p_cloud_grid = jnp.asarray(log_p_cloud_grid)
    cloudy_profile_grid = jnp.asarray(cloudy_profile_grid)

    def model(data_observed):
        return free_cloud_two_column_doppler_model(
            data_observed,
            geometry.theta,
            geometry.phi,
            geometry.distance_matrix,
            obs_times,
            wavelengths,
            clear_profile,
            log_p_cloud_grid,
            cloudy_profile_grid,
            period_mode=period_mode,
            fixed_period=fixed_period,
            log_p_cloud_bounds=log_p_cloud_bounds,
            sigma_b_scale=sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=fix_geometry,
            fixed_cosi=fixed_cosi,
            fixed_v=fixed_v,
            fixed_q1=fixed_q1,
            fixed_q2=fixed_q2,
        )

    kernel = make_fixed_two_column_nuts_kernel(
        model,
        n_phase=chip_data.flux.shape[0],
        period_mode=period_mode,
        fixed_period=fixed_period,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=fix_geometry,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        init_log_p_cloud=init_log_p_cloud,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), data)
    return mcmc


def save_fixed_two_column_samples(
    output_path,
    samples,
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    period_mode,
    sigma_b_scale=None,
    fixed_ell_b=None,
    fix_geometry=False,
):
    """Save Milestone 2-1 samples and fixed profile metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {name: np.asarray(value) for name, value in samples.items()}
    save_data.update(
        {
            "wavelengths": np.asarray(chip_data.wavelengths),
            "obs_times": np.asarray(chip_data.obs_times),
            "clear_profile": np.asarray(clear_profile),
            "cloudy_profile": np.asarray(cloudy_profile),
            "chip_index": np.asarray(chip_data.chip_index),
            "nside": np.asarray(geometry.nside),
            "period_mode": np.asarray(period_mode),
            "sigma_b_scale": np.asarray(
                np.nan if sigma_b_scale is None else sigma_b_scale
            ),
            "fixed_ell_b": np.asarray(np.nan if fixed_ell_b is None else fixed_ell_b),
            "fix_geometry": np.asarray(fix_geometry),
        }
    )
    np.savez(output_path, **save_data)


def save_free_cloud_two_column_samples(
    output_path,
    samples,
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    period_mode,
    log_p_cloud_bounds,
    sigma_b_scale=None,
    fixed_ell_b=None,
    fix_geometry=True,
):
    """Save Milestone 2-2a samples and cloudy-profile grid metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {name: np.asarray(value) for name, value in samples.items()}
    save_data.update(
        {
            "wavelengths": np.asarray(chip_data.wavelengths),
            "obs_times": np.asarray(chip_data.obs_times),
            "clear_profile": np.asarray(clear_profile),
            "log_p_cloud_grid": np.asarray(log_p_cloud_grid),
            "cloudy_profile_grid": np.asarray(cloudy_profile_grid),
            "chip_index": np.asarray(chip_data.chip_index),
            "nside": np.asarray(geometry.nside),
            "period_mode": np.asarray(period_mode),
            "log_p_cloud_bounds": np.asarray(log_p_cloud_bounds),
            "sigma_b_scale": np.asarray(
                np.nan if sigma_b_scale is None else sigma_b_scale
            ),
            "fixed_ell_b": np.asarray(np.nan if fixed_ell_b is None else fixed_ell_b),
            "fix_geometry": np.asarray(fix_geometry),
        }
    )
    np.savez(output_path, **save_data)


def save_synthetic_smoke_profiles(path, wavelengths):
    """Save synthetic two-column profiles for reproducible smoke runs."""

    clear_profile, cloudy_profile = synthetic_two_column_profiles(wavelengths)
    save_two_column_profiles(
        path,
        wavelengths,
        clear_profile,
        cloudy_profile,
        metadata={"profile_source": "synthetic_smoke"},
    )
    return clear_profile, cloudy_profile


def save_synthetic_smoke_cloud_profile_grid(path, wavelengths, log_p_cloud_grid=None):
    """Save synthetic clear/cloudy grid profiles for smoke runs."""

    if log_p_cloud_grid is None:
        log_p_cloud_grid = np.linspace(0.0, 2.0, 5)
    clear_profile, cloudy_profile_grid = synthetic_cloud_profile_grid(
        wavelengths,
        log_p_cloud_grid,
    )
    save_cloud_profile_grid(
        path,
        wavelengths,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        metadata={"profile_source": "synthetic_smoke_cloud_grid"},
    )
    return clear_profile, np.asarray(log_p_cloud_grid), cloudy_profile_grid


def _fixed_sample_at(samples, index):
    sample_names = {
        "cosi",
        "v",
        "q1",
        "q2",
        "u1",
        "u2",
        "log_w",
        "f_cloud",
        "surface_scale",
        "sigma_d",
        "sigma_b",
        "ell_b",
        "P",
        "log_p_cloud",
    }
    return {
        name: jnp.asarray(samples[name])[index]
        for name in sample_names
        if name in samples
    }


def two_column_operator_from_sample(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    sample,
):
    """Build ``(baseline, W_delta)`` for one Milestone 2-1 sample."""

    inclination = jnp.arccos(jnp.asarray(sample["cosi"]))
    u1 = jnp.asarray(sample.get("u1", kipping_q_to_u(sample["q1"], sample["q2"])[0]))
    u2 = jnp.asarray(sample.get("u2", kipping_q_to_u(sample["q1"], sample["q2"])[1]))
    weights = jnp.exp(jnp.asarray(sample["log_w"]))
    baseline, contrast_matrix = two_column_operator_from_times(
        geometry.theta,
        geometry.phi,
        jnp.asarray(sample["v"]),
        inclination,
        u1,
        u2,
        jnp.asarray(chip_data.obs_times),
        jnp.asarray(sample["P"]),
        jnp.asarray(chip_data.wavelengths),
        jnp.asarray(clear_profile),
        jnp.asarray(cloudy_profile),
        jnp.asarray(sample["f_cloud"]),
        weights=weights,
    )
    surface_scale = jnp.asarray(sample.get("surface_scale", 1.0))
    return surface_scale * baseline, surface_scale * contrast_matrix


def two_column_operator_from_free_cloud_sample(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    sample,
):
    """Build ``(baseline, W_delta)`` for one Milestone 2-2a sample."""

    cloudy_profile = _interpolate_profile_grid(
        log_p_cloud_grid,
        cloudy_profile_grid,
        sample["log_p_cloud"],
    )
    return two_column_operator_from_sample(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        sample,
    )


def conditional_contrast_map_for_sample(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    sample,
    gp_jitter=1.0e-6,
):
    """Compute conditional cloud-contrast posterior for one sample."""

    baseline, contrast_matrix = two_column_operator_from_sample(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        sample,
    )
    prior_covariance = add_diagonal_jitter(
        squared_exponential_covariance(
            geometry.distance_matrix,
            jnp.asarray(sample["sigma_b"]),
            jnp.asarray(sample["ell_b"]),
        ),
        jitter=gp_jitter,
    )
    prior_mean = jnp.zeros(contrast_matrix.shape[1])
    noise_variance = (
        jnp.asarray(sample["sigma_d"]) ** 2 * jnp.ones(contrast_matrix.shape[0])
        + 1.0e-6
    )
    return conditional_map_posterior(
        jnp.asarray(chip_data.flux).reshape(-1) - baseline,
        contrast_matrix,
        prior_mean,
        prior_covariance,
        noise_variance,
    )


def conditional_contrast_map_for_free_cloud_sample(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    sample,
    gp_jitter=1.0e-6,
):
    """Compute conditional cloud-contrast posterior for one M2-2a sample."""

    baseline, contrast_matrix = two_column_operator_from_free_cloud_sample(
        chip_data,
        geometry,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        sample,
    )
    prior_covariance = add_diagonal_jitter(
        squared_exponential_covariance(
            geometry.distance_matrix,
            jnp.asarray(sample["sigma_b"]),
            jnp.asarray(sample["ell_b"]),
        ),
        jitter=gp_jitter,
    )
    prior_mean = jnp.zeros(contrast_matrix.shape[1])
    noise_variance = (
        jnp.asarray(sample["sigma_d"]) ** 2 * jnp.ones(contrast_matrix.shape[0])
        + 1.0e-6
    )
    return conditional_map_posterior(
        jnp.asarray(chip_data.flux).reshape(-1) - baseline,
        contrast_matrix,
        prior_mean,
        prior_covariance,
        noise_variance,
    )


def compute_contrast_map_moments(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    samples,
    sample_indices=None,
):
    """Compute posterior moments for the cloud-fraction contrast map."""

    sample_count = len(np.asarray(samples["f_cloud"]))
    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    else:
        sample_indices = np.asarray(sample_indices)

    conditional_means = []
    conditional_diag_sum = jnp.zeros(geometry.theta.shape[0])
    f_cloud_values = []
    for index in sample_indices:
        sample = _fixed_sample_at(samples, int(index))
        mean, covariance = conditional_contrast_map_for_sample(
            chip_data,
            geometry,
            clear_profile,
            cloudy_profile,
            sample,
        )
        conditional_means.append(mean)
        conditional_diag_sum = conditional_diag_sum + jnp.diag(covariance)
        f_cloud_values.append(jnp.asarray(sample["f_cloud"]))

    mean_stack = jnp.stack(conditional_means, axis=0)
    contrast_mean = jnp.mean(mean_stack, axis=0)
    within = conditional_diag_sum / len(sample_indices)
    between = jnp.mean((mean_stack - contrast_mean[None, :]) ** 2, axis=0)
    contrast_variance = within + between

    f_cloud_stack = jnp.stack(f_cloud_values, axis=0)
    cloud_fraction_samples = f_cloud_stack[:, None] + mean_stack
    cloud_fraction_mean = jnp.mean(cloud_fraction_samples, axis=0)
    cloud_fraction_variance = (
        jnp.mean((cloud_fraction_samples - cloud_fraction_mean[None, :]) ** 2, axis=0)
        + within
    )
    return contrast_mean, contrast_variance, cloud_fraction_mean, cloud_fraction_variance


def compute_free_cloud_contrast_map_moments(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    samples,
    sample_indices=None,
):
    """Compute posterior cloud-map moments for Milestone 2-2a."""

    sample_count = len(np.asarray(samples["f_cloud"]))
    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    else:
        sample_indices = np.asarray(sample_indices)

    conditional_means = []
    conditional_diag_sum = jnp.zeros(geometry.theta.shape[0])
    f_cloud_values = []
    for index in sample_indices:
        sample = _fixed_sample_at(samples, int(index))
        mean, covariance = conditional_contrast_map_for_free_cloud_sample(
            chip_data,
            geometry,
            clear_profile,
            log_p_cloud_grid,
            cloudy_profile_grid,
            sample,
        )
        conditional_means.append(mean)
        conditional_diag_sum = conditional_diag_sum + jnp.diag(covariance)
        f_cloud_values.append(jnp.asarray(sample["f_cloud"]))

    mean_stack = jnp.stack(conditional_means, axis=0)
    contrast_mean = jnp.mean(mean_stack, axis=0)
    within = conditional_diag_sum / len(sample_indices)
    between = jnp.mean((mean_stack - contrast_mean[None, :]) ** 2, axis=0)
    contrast_variance = within + between

    f_cloud_stack = jnp.stack(f_cloud_values, axis=0)
    cloud_fraction_samples = f_cloud_stack[:, None] + mean_stack
    cloud_fraction_mean = jnp.mean(cloud_fraction_samples, axis=0)
    cloud_fraction_variance = (
        jnp.mean((cloud_fraction_samples - cloud_fraction_mean[None, :]) ** 2, axis=0)
        + within
    )
    return contrast_mean, contrast_variance, cloud_fraction_mean, cloud_fraction_variance


def fixed_two_column_median_sample(samples):
    """Return marginal posterior medians for Milestone 2-1 samples."""

    result = {}
    skip_names = {
        "wavelengths",
        "obs_times",
        "clear_profile",
        "cloudy_profile",
        "chip_index",
        "nside",
        "period_mode",
        "sigma_b_scale",
        "fixed_ell_b",
        "fix_geometry",
        "log_p_cloud_grid",
        "cloudy_profile_grid",
        "log_p_cloud_bounds",
    }
    for name, values in samples.items():
        array = np.asarray(values)
        if array.ndim == 0 or name in skip_names:
            continue
        result[name] = jnp.asarray(np.median(array, axis=0))
    if "u1" not in result or "u2" not in result:
        result["u1"], result["u2"] = kipping_q_to_u(result["q1"], result["q2"])
    return result


def reconstruct_fixed_two_column_timeseries(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    samples,
    contrast_map,
):
    """Reconstruct spectra from median nonlinear parameters and contrast map."""

    sample = fixed_two_column_median_sample(samples)
    baseline, contrast_matrix = two_column_operator_from_sample(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        sample,
    )
    model = baseline + contrast_matrix @ jnp.asarray(contrast_map)
    return model.reshape(chip_data.flux.shape), sample


def reconstruct_free_cloud_two_column_timeseries(
    chip_data,
    geometry,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    samples,
    contrast_map,
):
    """Reconstruct M2-2a spectra from median nonlinear parameters."""

    sample = fixed_two_column_median_sample(samples)
    baseline, contrast_matrix = two_column_operator_from_free_cloud_sample(
        chip_data,
        geometry,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        sample,
    )
    model = baseline + contrast_matrix @ jnp.asarray(contrast_map)
    return model.reshape(chip_data.flux.shape), sample
