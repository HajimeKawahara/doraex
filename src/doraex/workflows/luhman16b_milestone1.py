"""Milestone 1 workflow helpers for the Ureshino Luhman 16B reproduction."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, init_to_value

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.geometry.healpix import angular_distance_matrix, healpix_pixel_angles
from doraex.geometry.limb_darkening import kipping_q_to_u
from doraex.inference.map_posterior import conditional_map_posterior
from doraex.inference.numpyro_models import luhman16b_ureshino_model
from doraex.operators.design_matrix import full_design_matrix_from_times
from doraex.priors.spherical_gp import add_diagonal_jitter, squared_exponential_covariance


@dataclass(frozen=True)
class Luhman16BGeometry:
    """HEALPix geometry needed by the Milestone 1 model."""

    theta: jnp.ndarray
    phi: jnp.ndarray
    distance_matrix: jnp.ndarray
    nside: int


def build_luhman16b_geometry(nside=8, order="ring"):
    """Build pixel angles and angular-distance matrix for a HEALPix grid."""

    theta, phi = healpix_pixel_angles(nside, order=order)
    return Luhman16BGeometry(
        theta=theta,
        phi=phi,
        distance_matrix=angular_distance_matrix(theta, phi),
        nside=nside,
    )


def make_nuts_kernel(
    model,
    target_accept_prob=0.9,
    dense_mass=True,
    max_tree_depth=10,
):
    """Create the NUTS kernel used for the Milestone 1 production run."""

    init_values = {
        "cosi": 0.5,
        "v": 30.0,
        "q1": 0.5,
        "q2": 0.5,
        "sigma_d": 0.01,
    }
    return NUTS(
        model,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        init_strategy=init_to_value(values=init_values),
    )


def run_luhman16b_mcmc(
    chip_data,
    geometry,
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
    """Run the Milestone 1 NumPyro MCMC.

    Args:
        chip_data: Prepared Luhman 16B chip dataset.
        geometry: HEALPix geometry returned by :func:`build_luhman16b_geometry`.
        num_warmup: Number of NUTS warmup steps.
        num_samples: Number of retained posterior samples.
        num_chains: Number of MCMC chains.
        seed: JAX PRNG seed.
        period_mode: ``"sampled"`` for the Figure 8/9 run, or ``"fixed"`` for
            the auxiliary 5-hour-period reproduction.
        fixed_period: Period used when ``period_mode="fixed"``.
        target_accept_prob: NUTS target acceptance probability.
        dense_mass: Whether NUTS should adapt a dense mass matrix.
        max_tree_depth: Maximum NUTS tree depth.
        progress_bar: Whether NumPyro should show a progress bar.

    Returns:
        A completed ``numpyro.infer.MCMC`` object.
    """

    data = jnp.asarray(chip_data.flux)
    obs_times = jnp.asarray(chip_data.obs_times)
    wavelengths = jnp.asarray(chip_data.wavelengths)
    line_profile = jnp.asarray(chip_data.line_profile)

    def model(data_observed):
        return luhman16b_ureshino_model(
            data_observed,
            geometry.theta,
            geometry.phi,
            geometry.distance_matrix,
            obs_times,
            wavelengths,
            line_profile,
            period_mode=period_mode,
            fixed_period=fixed_period,
        )

    kernel = make_nuts_kernel(
        model,
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


def save_mcmc_samples(output_path, samples, chip_data, geometry, period_mode):
    """Save MCMC samples and lightweight run metadata to an NPZ file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {name: np.asarray(value) for name, value in samples.items()}
    save_data.update(
        {
            "wavelengths": np.asarray(chip_data.wavelengths),
            "obs_times": np.asarray(chip_data.obs_times),
            "line_profile": np.asarray(chip_data.line_profile),
            "chip_index": np.asarray(chip_data.chip_index),
            "nside": np.asarray(geometry.nside),
            "period_mode": np.asarray(period_mode),
        }
    )
    np.savez(output_path, **save_data)


def load_milestone1_inputs(
    data_dir,
    chip_index=1,
    nside=8,
    smoke_test=False,
    smoke_wavelength_step=64,
    smoke_phase_count=4,
):
    """Load data and geometry, optionally reducing them for a smoke test."""

    chip_data = load_luhman16b_chip(data_dir, chip_index=chip_index)
    geometry = build_luhman16b_geometry(nside=nside)
    if smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=smoke_wavelength_step,
            phase_count=smoke_phase_count,
        )
    return chip_data, geometry


def design_matrix_from_sample(chip_data, geometry, sample):
    """Build ``W`` for one posterior sample dictionary."""

    inclination = jnp.arccos(jnp.asarray(sample["cosi"]))
    u1 = jnp.asarray(sample.get("u1", kipping_q_to_u(sample["q1"], sample["q2"])[0]))
    u2 = jnp.asarray(sample.get("u2", kipping_q_to_u(sample["q1"], sample["q2"])[1]))
    weights = jnp.exp(jnp.asarray(sample["log_w"]))
    return full_design_matrix_from_times(
        geometry.theta,
        geometry.phi,
        jnp.asarray(sample["v"]),
        inclination,
        u1,
        u2,
        jnp.asarray(chip_data.obs_times),
        jnp.asarray(sample["P"]),
        jnp.asarray(chip_data.wavelengths),
        jnp.asarray(chip_data.line_profile),
        weights=weights,
    )


def conditional_map_for_sample(chip_data, geometry, sample, gp_jitter=1.0e-6):
    """Compute conditional map posterior moments for one nonlinear sample."""

    design_matrix = design_matrix_from_sample(chip_data, geometry, sample)
    prior_covariance = add_diagonal_jitter(
        squared_exponential_covariance(
            geometry.distance_matrix,
            jnp.asarray(sample["sigma_a"]),
            jnp.asarray(sample["ell"]),
        ),
        jitter=gp_jitter,
    )
    prior_mean = jnp.asarray(sample["mu_a"]) * jnp.ones(design_matrix.shape[1])
    noise_variance = (
        jnp.asarray(sample["sigma_d"]) ** 2 * jnp.ones(design_matrix.shape[0]) + 1.0e-6
    )
    return conditional_map_posterior(
        jnp.asarray(chip_data.flux),
        design_matrix,
        prior_mean,
        prior_covariance,
        noise_variance,
    )


def _sample_at(samples, index):
    sample_names = {
        "cosi",
        "v",
        "q1",
        "q2",
        "u1",
        "u2",
        "log_w",
        "sigma_d",
        "mu_a",
        "sigma_a",
        "ell",
        "P",
    }
    return {
        name: jnp.asarray(samples[name])[index]
        for name in sample_names
        if name in samples
    }


def compute_posterior_map_moments(chip_data, geometry, samples, sample_indices=None):
    """Compute Figure 8 posterior mean and variance maps.

    Args:
        chip_data: Prepared Luhman 16B chip dataset.
        geometry: HEALPix geometry.
        samples: Mapping of posterior sample arrays, as returned by NumPyro or
            loaded from the Milestone 1 NPZ.
        sample_indices: Optional subset of sample indices. ``None`` uses all
            samples.

    Returns:
        A tuple ``(posterior_mean, posterior_variance)`` over map pixels.
    """

    sample_count = len(np.asarray(samples["v"]))
    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    else:
        sample_indices = np.asarray(sample_indices)

    conditional_means = []
    conditional_diag_sum = jnp.zeros(geometry.theta.shape[0])
    for index in sample_indices:
        mean, covariance = conditional_map_for_sample(
            chip_data, geometry, _sample_at(samples, int(index))
        )
        conditional_means.append(mean)
        conditional_diag_sum = conditional_diag_sum + jnp.diag(covariance)

    mean_stack = jnp.stack(conditional_means, axis=0)
    posterior_mean = jnp.mean(mean_stack, axis=0)
    within = conditional_diag_sum / len(sample_indices)
    between = jnp.mean((mean_stack - posterior_mean[None, :]) ** 2, axis=0)
    return posterior_mean, within + between


def median_parameter_sample(samples):
    """Return a sample-like dictionary of marginal posterior medians."""

    result = {}
    for name, values in samples.items():
        array = np.asarray(values)
        if array.ndim == 0 or name in {
            "wavelengths",
            "obs_times",
            "line_profile",
            "chip_index",
            "nside",
            "period_mode",
        }:
            continue
        result[name] = jnp.asarray(np.median(array, axis=0))
    if "u1" not in result or "u2" not in result:
        result["u1"], result["u2"] = kipping_q_to_u(result["q1"], result["q2"])
    return result


def reconstruct_spectral_timeseries(chip_data, geometry, samples, map_mean):
    """Compute the Figure 9 posterior-median spectral reconstruction."""

    sample = median_parameter_sample(samples)
    design_matrix = design_matrix_from_sample(chip_data, geometry, sample)
    model = design_matrix @ jnp.asarray(map_mean)
    return model.reshape(chip_data.flux.shape), sample
