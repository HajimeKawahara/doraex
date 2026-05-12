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


def _interpolate_profile_grid(parameter_grid, profile_grid, parameter):
    """Linearly interpolate a precomputed one-parameter profile grid."""

    parameter_grid = jnp.asarray(parameter_grid)
    profile_grid = jnp.asarray(profile_grid)
    parameter = jnp.asarray(parameter)
    index = jnp.searchsorted(parameter_grid, parameter, side="right") - 1
    index = jnp.clip(index, 0, parameter_grid.shape[0] - 2)
    left = parameter_grid[index]
    right = parameter_grid[index + 1]
    fraction = (parameter - left) / (right - left)
    return (1.0 - fraction) * profile_grid[index] + fraction * profile_grid[index + 1]


def _interpolate_profile_grid_2d(x_grid, y_grid, profile_grid, x, y):
    """Bilinearly interpolate a precomputed two-parameter profile grid."""

    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    profile_grid = jnp.asarray(profile_grid)
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    x_index = jnp.searchsorted(x_grid, x, side="right") - 1
    y_index = jnp.searchsorted(y_grid, y, side="right") - 1
    x_index = jnp.clip(x_index, 0, x_grid.shape[0] - 2)
    y_index = jnp.clip(y_index, 0, y_grid.shape[0] - 2)

    x_left = x_grid[x_index]
    x_right = x_grid[x_index + 1]
    y_left = y_grid[y_index]
    y_right = y_grid[y_index + 1]
    x_fraction = (x - x_left) / (x_right - x_left)
    y_fraction = (y - y_left) / (y_right - y_left)

    p00 = profile_grid[x_index, y_index]
    p10 = profile_grid[x_index + 1, y_index]
    p01 = profile_grid[x_index, y_index + 1]
    p11 = profile_grid[x_index + 1, y_index + 1]
    return (
        (1.0 - x_fraction) * (1.0 - y_fraction) * p00
        + x_fraction * (1.0 - y_fraction) * p10
        + (1.0 - x_fraction) * y_fraction * p01
        + x_fraction * y_fraction * p11
    )


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
    sigma_b_scale=0.3,
    fixed_ell_b=None,
    fix_geometry=False,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
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
        sigma_b_scale: Scale of the half-normal prior on cloud-fraction
            contrast-map variations.
        fixed_ell_b: Optional fixed cloud-map correlation length in radians.
            When omitted, ``ell_b`` is sampled.
        fix_geometry: If true, fix ``cosi``, ``v``, ``q1``, and ``q2`` to the
            provided Milestone-1-like values.
        fixed_cosi: Fixed cosine inclination used when ``fix_geometry`` is true.
        fixed_v: Fixed equatorial rotation velocity in km/s used when
            ``fix_geometry`` is true.
        fixed_q1: Fixed Kipping ``q1`` used when ``fix_geometry`` is true.
        fixed_q2: Fixed Kipping ``q2`` used when ``fix_geometry`` is true.
        gp_jitter: Diagonal jitter added to the cloud-map GP covariance.
        noise_jitter: Diagonal jitter added to the data noise variance.

    Returns:
        None. The function defines a NumPyro probabilistic model.
    """

    n_phase = data.shape[0]
    if fix_geometry:
        cosi = numpyro.deterministic("cosi", jnp.asarray(fixed_cosi))
        vrot = numpyro.deterministic("v", jnp.asarray(fixed_v))
        q1 = numpyro.deterministic("q1", jnp.asarray(fixed_q1))
        q2 = numpyro.deterministic("q2", jnp.asarray(fixed_q2))
    else:
        cosi = numpyro.sample("cosi", dist.Uniform(0.0, 1.0))
        vrot = numpyro.sample("v", dist.Uniform(0.0, 120.0))
        q1 = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
        q2 = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
    inclination = jnp.arccos(cosi)

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

    sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(sigma_b_scale))
    if fixed_ell_b is None:
        ell_b = numpyro.sample("ell_b", dist.Uniform(0.1, 1.5))
    else:
        ell_b = numpyro.deterministic("ell_b", jnp.asarray(fixed_ell_b))
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


def free_cloud_two_column_doppler_model(
    data,
    theta,
    phi,
    distance_matrix,
    obs_times,
    wavelengths,
    clear_profile,
    log_p_cloud_grid,
    cloudy_profile_grid,
    period_mode="fixed",
    fixed_period=4.83,
    log_p_cloud_bounds=(0.0, 2.0),
    pixel_area=1.0,
    log_w_scale=0.1,
    surface_scale_location=0.0077,
    surface_scale_scale=0.3,
    sigma_b_scale=0.1,
    fixed_ell_b=0.4,
    fix_geometry=True,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    gp_jitter=0.5e-6,
    noise_jitter=1.0e-6,
):
    """Two-column Doppler retrieval with free cloud-top pressure.

    This is the Milestone 2-2a model. The clear spectrum and a grid of cloudy
    spectra are precomputed outside NUTS. NUTS samples ``log10 Pc`` and
    linearly interpolates the cloudy local spectrum inside the JAX graph, while
    the cloud-fraction contrast map remains analytically marginalized.

    Args:
        data: Observed phase-resolved spectra with shape
            ``(n_phase, n_wavelength)``.
        theta: HEALPix pixel colatitudes in radians.
        phi: HEALPix pixel longitudes in radians.
        distance_matrix: Pairwise angular distances between pixels, in radians.
        obs_times: Observation timestamps in the same units as the period.
        wavelengths: One-dimensional wavelength grid.
        clear_profile: Fixed clear-sky spectrum sampled on ``wavelengths``.
        log_p_cloud_grid: Monotonic grid of precomputed ``log10 Pc`` values.
        cloudy_profile_grid: Cloudy spectra with shape
            ``(n_log_p_cloud, n_wavelength)``.
        period_mode: ``"sampled"`` samples ``P ~ Uniform(4.8, 5.4)``.
            ``"fixed"`` records ``fixed_period`` as deterministic ``P``.
        fixed_period: Period used when ``period_mode="fixed"``.
        log_p_cloud_bounds: Uniform prior bounds for ``log10 Pc``.
        pixel_area: Optional equal-area pixel solid-angle factor.
        log_w_scale: Standard deviation of the per-phase log scaling prior.
        surface_scale_location: Median of the log-normal prior on the global
            surface-brightness scale.
        surface_scale_scale: Log-space standard deviation of the surface-scale
            prior.
        sigma_b_scale: Scale of the half-normal prior on cloud-fraction
            contrast-map variations.
        fixed_ell_b: Optional fixed cloud-map correlation length in radians.
            When omitted, ``ell_b`` is sampled.
        fix_geometry: If true, fix ``cosi``, ``v``, ``q1``, and ``q2`` to the
            provided values.
        fixed_cosi: Fixed cosine inclination used when ``fix_geometry`` is true.
        fixed_v: Fixed equatorial rotation velocity in km/s used when
            ``fix_geometry`` is true.
        fixed_q1: Fixed Kipping ``q1`` used when ``fix_geometry`` is true.
        fixed_q2: Fixed Kipping ``q2`` used when ``fix_geometry`` is true.
        gp_jitter: Diagonal jitter added to the cloud-map GP covariance.
        noise_jitter: Diagonal jitter added to the data noise variance.

    Returns:
        None. The function defines a NumPyro probabilistic model.
    """

    log_p_cloud = numpyro.sample(
        "log_p_cloud",
        dist.Uniform(log_p_cloud_bounds[0], log_p_cloud_bounds[1]),
    )
    cloudy_profile = _interpolate_profile_grid(
        log_p_cloud_grid,
        cloudy_profile_grid,
        log_p_cloud,
    )
    return fixed_two_column_doppler_model(
        data,
        theta,
        phi,
        distance_matrix,
        obs_times,
        wavelengths,
        clear_profile,
        cloudy_profile,
        period_mode=period_mode,
        fixed_period=fixed_period,
        pixel_area=pixel_area,
        log_w_scale=log_w_scale,
        surface_scale_location=surface_scale_location,
        surface_scale_scale=surface_scale_scale,
        sigma_b_scale=sigma_b_scale,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=fix_geometry,
        fixed_cosi=fixed_cosi,
        fixed_v=fixed_v,
        fixed_q1=fixed_q1,
        fixed_q2=fixed_q2,
        gp_jitter=gp_jitter,
        noise_jitter=noise_jitter,
    )


def free_t0_cloud_two_column_doppler_model(
    data,
    theta,
    phi,
    distance_matrix,
    obs_times,
    wavelengths,
    t0_grid,
    log_p_cloud_grid,
    clear_profile_grid,
    cloudy_profile_grid,
    period_mode="fixed",
    fixed_period=4.83,
    t0_bounds=(1000.0, 1700.0),
    log_p_cloud_bounds=(-2.0, 2.0),
    pixel_area=1.0,
    log_w_scale=0.1,
    surface_scale_location=0.0077,
    surface_scale_scale=0.3,
    sigma_b_scale=0.1,
    fixed_ell_b=0.4,
    fix_geometry=True,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    gp_jitter=0.5e-6,
    noise_jitter=1.0e-6,
):
    """Two-column Doppler retrieval with free T0 and cloud-top pressure.

    This is the first grid-based simple-retrieval extension beyond Milestone
    2-2. The ExoJAX spectra are precomputed on ``T0`` and ``log10 Pc`` grids.
    NUTS samples ``T0`` and ``log10 Pc`` and interpolates the local clear and
    cloudy spectra inside the JAX graph.
    """

    t0 = numpyro.sample("T0", dist.Uniform(t0_bounds[0], t0_bounds[1]))
    log_p_cloud = numpyro.sample(
        "log_p_cloud",
        dist.Uniform(log_p_cloud_bounds[0], log_p_cloud_bounds[1]),
    )
    clear_profile = _interpolate_profile_grid(t0_grid, clear_profile_grid, t0)
    cloudy_profile = _interpolate_profile_grid_2d(
        t0_grid,
        log_p_cloud_grid,
        cloudy_profile_grid,
        t0,
        log_p_cloud,
    )
    return fixed_two_column_doppler_model(
        data,
        theta,
        phi,
        distance_matrix,
        obs_times,
        wavelengths,
        clear_profile,
        cloudy_profile,
        period_mode=period_mode,
        fixed_period=fixed_period,
        pixel_area=pixel_area,
        log_w_scale=log_w_scale,
        surface_scale_location=surface_scale_location,
        surface_scale_scale=surface_scale_scale,
        sigma_b_scale=sigma_b_scale,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=fix_geometry,
        fixed_cosi=fixed_cosi,
        fixed_v=fixed_v,
        fixed_q1=fixed_q1,
        fixed_q2=fixed_q2,
        gp_jitter=gp_jitter,
        noise_jitter=noise_jitter,
    )
