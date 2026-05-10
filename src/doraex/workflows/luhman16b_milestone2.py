"""Milestone 2 workflow helpers for fixed two-column Doppler retrieval."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_value

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.inference.numpyro_models import fixed_two_column_doppler_model
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
