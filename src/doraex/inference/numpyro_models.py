"""NumPyro model definitions for Doppler retrieval."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from doraex.geometry.limb_darkening import kipping_q_to_u
from doraex.inference.marginal_likelihood import diagonal_noise_variance
from doraex.operators.design_matrix import (
    full_design_matrix_from_times,
    two_column_operator_from_times,
)
from doraex.priors.spherical_gp import add_diagonal_jitter, squared_exponential_covariance


def luhman16b_ureshino_model(
    data,
    theta,
    phi,
    distance_matrix,
    obs_times,
    wavelengths,
    line_profile,
    period_mode="sampled",
    fixed_period=5.0,
    pixel_area=1.0,
    log_w_scale=0.1,
    surface_scale_location=0.0077,
    surface_scale_scale=0.3,
    gp_jitter=0.5e-6,
    noise_jitter=1.0e-6,
):
    """Ureshino et al. Luhman 16B Bayesian Doppler-imaging model.

    Args:
        data: Observed phase-resolved spectra with shape
            ``(n_phase, n_wavelength)``.
        theta: HEALPix pixel colatitudes in radians.
        phi: HEALPix pixel longitudes in radians.
        distance_matrix: Pairwise angular distances between pixels, in radians.
        obs_times: Observation timestamps in the same units as the rotation
            period.
        wavelengths: One-dimensional wavelength grid.
        line_profile: Intrinsic rest-frame local spectrum sampled on
            ``wavelengths``.
        period_mode: ``"sampled"`` samples ``P ~ Uniform(4.8, 5.4)``. ``"fixed"``
            records ``fixed_period`` as deterministic ``P``.
        fixed_period: Period used when ``period_mode="fixed"``.
        pixel_area: Optional equal-area pixel solid-angle factor passed to the
            design-matrix builder. The Ureshino reproduction path uses ``1``.
        log_w_scale: Standard deviation of the per-phase log scaling prior.
        gp_jitter: Diagonal jitter added to the map GP covariance.
        noise_jitter: Diagonal jitter added to the data noise variance.

    Returns:
        None. The function defines a NumPyro probabilistic model.
    """

    n_phase = data.shape[0]
    cosi = numpyro.sample("cosi", dist.Uniform(0.0, 1.0))
    inclination = jnp.arccos(cosi)
    vrot = numpyro.sample("v", dist.Uniform(0.0, 120.0))

    q1 = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
    q2 = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
    u1, u2 = kipping_q_to_u(q1, q2)
    numpyro.deterministic("u1", u1)
    numpyro.deterministic("u2", u2)

    log_w = numpyro.sample("log_w", dist.Normal(0.0, log_w_scale).expand([n_phase]))
    weights = jnp.exp(log_w)

    if period_mode == "sampled":
        period = numpyro.sample("P", dist.Uniform(4.8, 5.4))
    elif period_mode == "fixed":
        period = numpyro.deterministic("P", jnp.asarray(fixed_period))
    else:
        raise ValueError("period_mode must be 'sampled' or 'fixed'")

    design_matrix = full_design_matrix_from_times(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        obs_times,
        period,
        wavelengths,
        line_profile,
        weights=weights,
        pixel_area=pixel_area,
    )

    sigma_d = numpyro.sample("sigma_d", dist.LogNormal(jnp.log(0.03), 1.0))
    noise_variance = diagonal_noise_variance(
        design_matrix.shape[0], sigma_d, jitter=noise_jitter
    )

    mu_a = numpyro.sample("mu_a", dist.Uniform(0.0, 0.05))
    prior_mean = mu_a * jnp.ones(design_matrix.shape[1])
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(0.3))
    ell = numpyro.sample("ell", dist.Uniform(0.1, 1.5))
    map_covariance = add_diagonal_jitter(
        squared_exponential_covariance(distance_matrix, sigma_a, ell),
        jitter=gp_jitter,
    )
    covariance_factor = design_matrix @ jnp.linalg.cholesky(map_covariance)

    numpyro.sample(
        "obs",
        dist.LowRankMultivariateNormal(
            loc=design_matrix @ prior_mean,
            cov_factor=covariance_factor,
            cov_diag=noise_variance,
        ),
        obs=data.reshape(-1),
    )


def fixed_two_column_doppler_model(
    data,
    theta,
    phi,
    distance_matrix,
    obs_times,
    wavelengths,
    clear_profile,
    cloudy_profile,
    period_mode="sampled",
    fixed_period=5.0,
    pixel_area=1.0,
    log_w_scale=0.1,
    surface_scale_location=0.0077,
    surface_scale_scale=0.3,
    gp_jitter=0.5e-6,
    noise_jitter=1.0e-6,
):
    """Two-column Doppler retrieval with fixed atmospheric spectra.

    This is the Milestone 2-1 model. The atmospheric retrieval parameters are
    fixed outside NUTS by precomputing a clear and a cloudy local spectrum. The
    sampled linear map is the cloud-fraction contrast around a uniform mean
    cloud fraction, and it is analytically marginalized.

    Args:
        data: Observed phase-resolved spectra with shape
            ``(n_phase, n_wavelength)``.
        theta: HEALPix pixel colatitudes in radians.
        phi: HEALPix pixel longitudes in radians.
        distance_matrix: Pairwise angular distances between pixels, in radians.
        obs_times: Observation timestamps in the same units as the period.
        wavelengths: One-dimensional wavelength grid.
        clear_profile: Fixed local clear-sky spectrum sampled on
            ``wavelengths``.
        cloudy_profile: Fixed local cloudy spectrum sampled on ``wavelengths``.
        period_mode: ``"sampled"`` samples ``P ~ Uniform(4.8, 5.4)``.
            ``"fixed"`` records ``fixed_period`` as deterministic ``P``.
        fixed_period: Period used when ``period_mode="fixed"``.
        pixel_area: Optional equal-area pixel solid-angle factor.
        log_w_scale: Standard deviation of the per-phase log scaling prior.
        surface_scale_location: Median of the log-normal prior on the global
            surface-brightness scale multiplying both fixed column spectra.
        surface_scale_scale: Log-space standard deviation of the surface-scale
            prior.
        gp_jitter: Diagonal jitter added to the cloud-map GP covariance.
        noise_jitter: Diagonal jitter added to the data noise variance.

    Returns:
        None. The function defines a NumPyro probabilistic model.
    """

    n_phase = data.shape[0]
    cosi = numpyro.sample("cosi", dist.Uniform(0.0, 1.0))
    inclination = jnp.arccos(cosi)
    vrot = numpyro.sample("v", dist.Uniform(0.0, 120.0))

    q1 = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
    q2 = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
    u1, u2 = kipping_q_to_u(q1, q2)
    numpyro.deterministic("u1", u1)
    numpyro.deterministic("u2", u2)

    log_w = numpyro.sample("log_w", dist.Normal(0.0, log_w_scale).expand([n_phase]))
    weights = jnp.exp(log_w)

    if period_mode == "sampled":
        period = numpyro.sample("P", dist.Uniform(4.8, 5.4))
    elif period_mode == "fixed":
        period = numpyro.deterministic("P", jnp.asarray(fixed_period))
    else:
        raise ValueError("period_mode must be 'sampled' or 'fixed'")

    mean_cloud_fraction = numpyro.sample("f_cloud", dist.Uniform(0.0, 1.0))
    surface_scale = numpyro.sample(
        "surface_scale",
        dist.LogNormal(jnp.log(surface_scale_location), surface_scale_scale),
    )
    baseline, contrast_matrix = two_column_operator_from_times(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        obs_times,
        period,
        wavelengths,
        clear_profile,
        cloudy_profile,
        mean_cloud_fraction,
        weights=weights,
        pixel_area=pixel_area,
    )
    baseline = surface_scale * baseline
    contrast_matrix = surface_scale * contrast_matrix

    sigma_d = numpyro.sample("sigma_d", dist.LogNormal(jnp.log(0.03), 1.0))
    noise_variance = diagonal_noise_variance(
        contrast_matrix.shape[0], sigma_d, jitter=noise_jitter
    )

    sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(0.3))
    ell_b = numpyro.sample("ell_b", dist.Uniform(0.1, 1.5))
    contrast_covariance = add_diagonal_jitter(
        squared_exponential_covariance(distance_matrix, sigma_b, ell_b),
        jitter=gp_jitter,
    )
    covariance_factor = contrast_matrix @ jnp.linalg.cholesky(contrast_covariance)

    numpyro.sample(
        "obs",
        dist.LowRankMultivariateNormal(
            loc=baseline,
            cov_factor=covariance_factor,
            cov_diag=noise_variance,
        ),
        obs=data.reshape(-1),
    )
