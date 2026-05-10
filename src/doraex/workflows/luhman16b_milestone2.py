"""Milestone 2 workflow helpers for fixed two-column Doppler retrieval."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_value

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.geometry.limb_darkening import kipping_q_to_u
from doraex.inference.map_posterior import conditional_map_posterior
from doraex.inference.numpyro_models import fixed_two_column_doppler_model
from doraex.operators.design_matrix import two_column_operator_from_times
from doraex.priors.spherical_gp import add_diagonal_jitter, squared_exponential_covariance
from doraex.spectra.exojax_forward import (
    load_two_column_profiles,
    save_two_column_profiles,
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


def make_fixed_two_column_nuts_kernel(
    model,
    n_phase,
    period_mode="sampled",
    fixed_period=5.0,
    target_accept_prob=0.9,
    dense_mass=True,
    max_tree_depth=10,
):
    """Create the NUTS kernel for Milestone 2-1."""

    init_values = {
        "cosi": 0.485,
        "v": 31.2,
        "q1": 0.81,
        "q2": 0.59,
        "log_w": jnp.zeros(n_phase),
        "f_cloud": 0.5,
        "surface_scale": 0.0077,
        "sigma_d": 0.039,
        "sigma_b": 0.05,
        "ell_b": 0.4,
    }
    if period_mode == "sampled":
        init_values["P"] = 4.83
    elif period_mode == "fixed":
        init_values["P"] = fixed_period
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
        )

    kernel = make_fixed_two_column_nuts_kernel(
        model,
        n_phase=chip_data.flux.shape[0],
        period_mode=period_mode,
        fixed_period=fixed_period,
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


def save_fixed_two_column_samples(
    output_path,
    samples,
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    period_mode,
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


def compute_contrast_map_moments(
    chip_data,
    geometry,
    clear_profile,
    cloudy_profile,
    samples,
    sample_indices=None,
):
    """Compute posterior moments for the cloud-fraction contrast map."""

    sample_count = len(np.asarray(samples["v"]))
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
