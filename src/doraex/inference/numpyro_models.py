"""NumPyro model definitions for Doppler retrieval."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from doraex.geometry.limb_darkening import kipping_q_to_u
from doraex.inference.marginal_likelihood import diagonal_noise_variance
from doraex.operators.design_matrix import (
    full_design_matrix_from_times,
    linear_profile_operator_from_times,
    two_column_operator_from_times,
)
from doraex.priors.spherical_gp import (
    add_diagonal_jitter,
    project_zero_mean_covariance,
    squared_exponential_covariance,
)


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


def _interpolate_profile_grid_3d(x_grid, y_grid, z_grid, profile_grid, x, y, z):
    """Trilinearly interpolate a precomputed three-parameter profile grid."""

    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    z_grid = jnp.asarray(z_grid)
    profile_grid = jnp.asarray(profile_grid)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)

    x_index = jnp.searchsorted(x_grid, x, side="right") - 1
    y_index = jnp.searchsorted(y_grid, y, side="right") - 1
    z_index = jnp.searchsorted(z_grid, z, side="right") - 1
    x_index = jnp.clip(x_index, 0, x_grid.shape[0] - 2)
    y_index = jnp.clip(y_index, 0, y_grid.shape[0] - 2)
    z_index = jnp.clip(z_index, 0, z_grid.shape[0] - 2)

    x_fraction = (x - x_grid[x_index]) / (x_grid[x_index + 1] - x_grid[x_index])
    y_fraction = (y - y_grid[y_index]) / (y_grid[y_index + 1] - y_grid[y_index])
    z_fraction = (z - z_grid[z_index]) / (z_grid[z_index + 1] - z_grid[z_index])
    p000 = profile_grid[x_index, y_index, z_index]
    p100 = profile_grid[x_index + 1, y_index, z_index]
    p010 = profile_grid[x_index, y_index + 1, z_index]
    p110 = profile_grid[x_index + 1, y_index + 1, z_index]
    p001 = profile_grid[x_index, y_index, z_index + 1]
    p101 = profile_grid[x_index + 1, y_index, z_index + 1]
    p011 = profile_grid[x_index, y_index + 1, z_index + 1]
    p111 = profile_grid[x_index + 1, y_index + 1, z_index + 1]
    return (
        (1.0 - x_fraction) * (1.0 - y_fraction) * (1.0 - z_fraction) * p000
        + x_fraction * (1.0 - y_fraction) * (1.0 - z_fraction) * p100
        + (1.0 - x_fraction) * y_fraction * (1.0 - z_fraction) * p010
        + x_fraction * y_fraction * (1.0 - z_fraction) * p110
        + (1.0 - x_fraction) * (1.0 - y_fraction) * z_fraction * p001
        + x_fraction * (1.0 - y_fraction) * z_fraction * p101
        + (1.0 - x_fraction) * y_fraction * z_fraction * p011
        + x_fraction * y_fraction * z_fraction * p111
    )


def _interpolate_profile_grid_4d(
    w_grid,
    x_grid,
    y_grid,
    z_grid,
    profile_grid,
    w,
    x,
    y,
    z,
):
    """Linearly interpolate a precomputed four-parameter profile grid."""

    w_grid = jnp.asarray(w_grid)
    profile_grid = jnp.asarray(profile_grid)
    w = jnp.asarray(w)
    w_index = jnp.searchsorted(w_grid, w, side="right") - 1
    w_index = jnp.clip(w_index, 0, w_grid.shape[0] - 2)
    w_fraction = (w - w_grid[w_index]) / (w_grid[w_index + 1] - w_grid[w_index])
    left = _interpolate_profile_grid_3d(
        x_grid,
        y_grid,
        z_grid,
        profile_grid[w_index],
        x,
        y,
        z,
    )
    right = _interpolate_profile_grid_3d(
        x_grid,
        y_grid,
        z_grid,
        profile_grid[w_index + 1],
        x,
        y,
        z,
    )
    return (1.0 - w_fraction) * left + w_fraction * right


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
    alpha_grid=None,
    zeta_vmr_grid=None,
    period_mode="fixed",
    fixed_period=4.83,
    t0_bounds=(1000.0, 1700.0),
    alpha_bounds=(0.05, 0.20),
    log_p_cloud_bounds=(-2.0, 2.0),
    zeta_vmr_bounds=(-0.5, 0.5),
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
    if zeta_vmr_grid is None:
        clear_profile = _interpolate_profile_grid(t0_grid, clear_profile_grid, t0)
        cloudy_profile = _interpolate_profile_grid_2d(
            t0_grid,
            log_p_cloud_grid,
            cloudy_profile_grid,
            t0,
            log_p_cloud,
        )
    else:
        zeta_vmr = numpyro.sample(
            "zeta_vmr",
            dist.Uniform(zeta_vmr_bounds[0], zeta_vmr_bounds[1]),
        )
        clear_profile = _interpolate_profile_grid_2d(
            t0_grid,
            zeta_vmr_grid,
            clear_profile_grid,
            t0,
            zeta_vmr,
        )
        cloudy_profile = _interpolate_profile_grid_3d(
            t0_grid,
            log_p_cloud_grid,
            zeta_vmr_grid,
            cloudy_profile_grid,
            t0,
            log_p_cloud,
            zeta_vmr,
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


def joint_free_t0_cloud_two_column_doppler_model(
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
    alpha_grid=None,
    zeta_vmr_grid=None,
    period_mode="fixed",
    fixed_period=4.83,
    t0_bounds=(1000.0, 1700.0),
    alpha_bounds=(0.05, 0.20),
    log_p_cloud_bounds=(-2.0, 2.0),
    zeta_vmr_bounds=(-0.5, 0.5),
    pixel_area=1.0,
    log_w_scale=0.1,
    surface_scale_location=0.0077,
    surface_scale_scale=0.3,
    sigma_b_scale=0.1,
    fixed_ell_b=0.3,
    fix_geometry=True,
    fixed_cosi=0.485,
    fixed_v=31.2,
    fixed_q1=0.81,
    fixed_q2=0.59,
    shared_atmosphere=False,
    normalization_mode="surface_scale",
    column_mode="clear_cloud",
    fixed_cloud_delta=1.0,
    pressure_derivative_step=0.025,
    zero_mean_pressure_map=False,
    gp_jitter=0.5e-6,
    noise_jitter=1.0e-6,
):
    """Joint multi-chip retrieval with a shared contrast map.

    By default the atmospheric parameters and nuisance parameters are
    chip-specific, while the cloud contrast map and its GP hyperparameters are
    shared across chips. Set ``shared_atmosphere`` to share ``T0``,
    ``log_p_cloud``, and ``f_cloud`` across chips while keeping chip-local
    calibration and noise parameters. The default normalization samples the
    legacy ``surface_scale`` amplitude; ``normalization_mode="yama"`` instead
    uses per-chip normalized spectra ``F_i / (A_i mean(F_i))`` following
    Yama et al. Set ``column_mode="double_cloud"`` to replace the clear/cloud
    endpoints with deep/high cloudy endpoints centered on ``log_p_mid`` and
    separated by ``fixed_cloud_delta`` dex. Set
    ``column_mode="pressure_perturbation"`` to use a finite-difference
    derivative of the cloudy spectrum with respect to ``log_p_cloud`` as the
    linear map basis. Set ``zero_mean_pressure_map=True`` to remove the
    monopole from that pressure-perturbation map prior.
    """

    n_chip = data.shape[0]
    n_phase = data.shape[1]
    if normalization_mode not in ("surface_scale", "yama"):
        raise ValueError("normalization_mode must be 'surface_scale' or 'yama'")
    if column_mode not in ("clear_cloud", "double_cloud", "pressure_perturbation"):
        raise ValueError(
            "column_mode must be 'clear_cloud', 'double_cloud', or "
            "'pressure_perturbation'"
        )
    pressure_name = "log_p_mid" if column_mode == "double_cloud" else "log_p_cloud"
    fraction_name = "h_high" if column_mode == "double_cloud" else "f_cloud"
    uses_fraction_map = column_mode != "pressure_perturbation"

    if fix_geometry:
        cosi = numpyro.deterministic("cosi", jnp.asarray(fixed_cosi))
        vrot = numpyro.deterministic("v", jnp.asarray(fixed_v))
        q1 = numpyro.deterministic("q1", jnp.asarray(fixed_q1))
        q2 = numpyro.deterministic("q2", jnp.asarray(fixed_q2))
    else:
        cosi = numpyro.sample("cosi", dist.Uniform(0.0, 1.0))
        vrot = numpyro.sample("v", dist.Uniform(20.0, 50.0))
        q1 = numpyro.sample("q1", dist.Uniform(0.0, 1.0))
        q2 = numpyro.sample("q2", dist.Uniform(0.0, 1.0))
    inclination = jnp.arccos(cosi)
    u1, u2 = kipping_q_to_u(q1, q2)
    numpyro.deterministic("u1", u1)
    numpyro.deterministic("u2", u2)

    if period_mode == "sampled":
        period = numpyro.sample("P", dist.Uniform(4.8, 5.4))
    elif period_mode == "fixed":
        period = numpyro.deterministic("P", jnp.asarray(fixed_period))
    else:
        raise ValueError("period_mode must be 'sampled' or 'fixed'")

    has_alpha_grid = alpha_grid is not None
    has_vmr_grid = zeta_vmr_grid is not None
    if shared_atmosphere:
        t0 = numpyro.sample("T0", dist.Uniform(t0_bounds[0], t0_bounds[1]))
        if has_alpha_grid:
            alpha = numpyro.sample(
                "alpha",
                dist.Uniform(alpha_bounds[0], alpha_bounds[1]),
            )
        else:
            alpha = None
        log_p_location = numpyro.sample(
            pressure_name,
            dist.Uniform(log_p_cloud_bounds[0], log_p_cloud_bounds[1]),
        )
        if uses_fraction_map:
            mean_cloud_fraction = numpyro.sample(
                fraction_name,
                dist.Uniform(0.0, 1.0),
            )
        else:
            mean_cloud_fraction = None
        if has_vmr_grid:
            zeta_vmr = numpyro.sample(
                "zeta_vmr",
                dist.Uniform(zeta_vmr_bounds[0], zeta_vmr_bounds[1]),
            )
        else:
            zeta_vmr = None
    else:
        t0 = numpyro.sample(
            "T0",
            dist.Uniform(t0_bounds[0], t0_bounds[1]).expand([n_chip]),
        )
        if has_alpha_grid:
            alpha = numpyro.sample(
                "alpha",
                dist.Uniform(alpha_bounds[0], alpha_bounds[1]).expand([n_chip]),
            )
        else:
            alpha = None
        log_p_location = numpyro.sample(
            pressure_name,
            dist.Uniform(log_p_cloud_bounds[0], log_p_cloud_bounds[1]).expand([n_chip]),
        )
        if uses_fraction_map:
            mean_cloud_fraction = numpyro.sample(
                fraction_name,
                dist.Uniform(0.0, 1.0).expand([n_chip]),
            )
        else:
            mean_cloud_fraction = None
        if has_vmr_grid:
            zeta_vmr = numpyro.sample(
                "zeta_vmr",
                dist.Uniform(zeta_vmr_bounds[0], zeta_vmr_bounds[1]).expand([n_chip]),
            )
        else:
            zeta_vmr = None
    if normalization_mode == "surface_scale":
        surface_scale = numpyro.sample(
            "surface_scale",
            dist.LogNormal(
                jnp.log(surface_scale_location),
                surface_scale_scale,
            ).expand([n_chip]),
        )
        normalization_factor = None
    else:
        surface_scale = None
        normalization_factor = numpyro.sample(
            "A",
            dist.Uniform(1.0, 1.2).expand([n_chip]),
        )
    log_w = numpyro.sample(
        "log_w",
        dist.Normal(0.0, log_w_scale).expand([n_chip, n_phase]),
    )
    sigma_d = numpyro.sample(
        "sigma_d",
        dist.LogNormal(jnp.log(0.03), 1.0).expand([n_chip]),
    )

    baselines = []
    contrast_matrices = []
    noise_variances = []
    for chip_index in range(n_chip):
        t0_chip = t0 if shared_atmosphere else t0[chip_index]
        alpha_chip = alpha if shared_atmosphere else (
            None if alpha is None else alpha[chip_index]
        )
        log_p_location_chip = (
            log_p_location if shared_atmosphere else log_p_location[chip_index]
        )
        if uses_fraction_map:
            mean_cloud_fraction_chip = (
                mean_cloud_fraction
                if shared_atmosphere
                else mean_cloud_fraction[chip_index]
            )
        else:
            mean_cloud_fraction_chip = None
        if has_alpha_grid:
            zeta_vmr_chip = zeta_vmr if shared_atmosphere else zeta_vmr[chip_index]
            clear_profile = _interpolate_profile_grid_3d(
                t0_grid[chip_index],
                alpha_grid[chip_index],
                zeta_vmr_grid[chip_index],
                clear_profile_grid[chip_index],
                t0_chip,
                alpha_chip,
                zeta_vmr_chip,
            )
            if column_mode == "pressure_perturbation":
                base_profile = _interpolate_profile_grid_4d(
                    t0_grid[chip_index],
                    alpha_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    alpha_chip,
                    log_p_location_chip,
                    zeta_vmr_chip,
                )
                deeper_profile = _interpolate_profile_grid_4d(
                    t0_grid[chip_index],
                    alpha_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    alpha_chip,
                    log_p_location_chip + pressure_derivative_step,
                    zeta_vmr_chip,
                )
                higher_profile = _interpolate_profile_grid_4d(
                    t0_grid[chip_index],
                    alpha_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    alpha_chip,
                    log_p_location_chip - pressure_derivative_step,
                    zeta_vmr_chip,
                )
                contrast_profile = (
                    deeper_profile - higher_profile
                ) / (2.0 * pressure_derivative_step)
            elif column_mode == "double_cloud":
                clear_profile = _interpolate_profile_grid_4d(
                    t0_grid[chip_index],
                    alpha_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    alpha_chip,
                    log_p_location_chip + 0.5 * fixed_cloud_delta,
                    zeta_vmr_chip,
                )
                cloudy_profile = _interpolate_profile_grid_4d(
                    t0_grid[chip_index],
                    alpha_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    alpha_chip,
                    log_p_location_chip - 0.5 * fixed_cloud_delta,
                    zeta_vmr_chip,
                )
                base_profile = None
                contrast_profile = None
            else:
                cloudy_profile = _interpolate_profile_grid_4d(
                    t0_grid[chip_index],
                    alpha_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    alpha_chip,
                    log_p_location_chip,
                    zeta_vmr_chip,
                )
                base_profile = None
                contrast_profile = None
        elif has_vmr_grid:
            zeta_vmr_chip = zeta_vmr if shared_atmosphere else zeta_vmr[chip_index]
            clear_profile = _interpolate_profile_grid_2d(
                t0_grid[chip_index],
                zeta_vmr_grid[chip_index],
                clear_profile_grid[chip_index],
                t0_chip,
                zeta_vmr_chip,
            )
            if column_mode == "pressure_perturbation":
                base_profile = _interpolate_profile_grid_3d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip,
                    zeta_vmr_chip,
                )
                deeper_profile = _interpolate_profile_grid_3d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip + pressure_derivative_step,
                    zeta_vmr_chip,
                )
                higher_profile = _interpolate_profile_grid_3d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip - pressure_derivative_step,
                    zeta_vmr_chip,
                )
                contrast_profile = (
                    deeper_profile - higher_profile
                ) / (2.0 * pressure_derivative_step)
            elif column_mode == "double_cloud":
                clear_profile = _interpolate_profile_grid_3d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip + 0.5 * fixed_cloud_delta,
                    zeta_vmr_chip,
                )
                cloudy_profile = _interpolate_profile_grid_3d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip - 0.5 * fixed_cloud_delta,
                    zeta_vmr_chip,
                )
                base_profile = None
                contrast_profile = None
            else:
                cloudy_profile = _interpolate_profile_grid_3d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    zeta_vmr_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip,
                    zeta_vmr_chip,
                )
                base_profile = None
                contrast_profile = None
        else:
            clear_profile = _interpolate_profile_grid(
                t0_grid[chip_index],
                clear_profile_grid[chip_index],
                t0_chip,
            )
            if column_mode == "pressure_perturbation":
                base_profile = _interpolate_profile_grid_2d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip,
                )
                deeper_profile = _interpolate_profile_grid_2d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip + pressure_derivative_step,
                )
                higher_profile = _interpolate_profile_grid_2d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip - pressure_derivative_step,
                )
                contrast_profile = (
                    deeper_profile - higher_profile
                ) / (2.0 * pressure_derivative_step)
            elif column_mode == "double_cloud":
                clear_profile = _interpolate_profile_grid_2d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip + 0.5 * fixed_cloud_delta,
                )
                cloudy_profile = _interpolate_profile_grid_2d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip - 0.5 * fixed_cloud_delta,
                )
                base_profile = None
                contrast_profile = None
            else:
                cloudy_profile = _interpolate_profile_grid_2d(
                    t0_grid[chip_index],
                    log_p_cloud_grid[chip_index],
                    cloudy_profile_grid[chip_index],
                    t0_chip,
                    log_p_location_chip,
                )
                base_profile = None
                contrast_profile = None
        if column_mode == "pressure_perturbation":
            baseline, contrast_matrix = linear_profile_operator_from_times(
                theta,
                phi,
                vrot,
                inclination,
                u1,
                u2,
                obs_times,
                period,
                wavelengths[chip_index],
                base_profile,
                contrast_profile,
                weights=jnp.exp(log_w[chip_index]),
                pixel_area=pixel_area,
            )
        else:
            baseline, contrast_matrix = two_column_operator_from_times(
                theta,
                phi,
                vrot,
                inclination,
                u1,
                u2,
                obs_times,
                period,
                wavelengths[chip_index],
                clear_profile,
                cloudy_profile,
                mean_cloud_fraction_chip,
                weights=jnp.exp(log_w[chip_index]),
                pixel_area=pixel_area,
            )
        if normalization_mode == "surface_scale":
            baseline = surface_scale[chip_index] * baseline
            contrast_matrix = surface_scale[chip_index] * contrast_matrix
        else:
            norm = normalization_factor[chip_index] * jnp.mean(baseline)
            baseline = baseline / norm
            contrast_matrix = contrast_matrix / norm
        baselines.append(baseline)
        contrast_matrices.append(contrast_matrix)
        noise_variances.append(
            diagonal_noise_variance(
                contrast_matrix.shape[0],
                sigma_d[chip_index],
                jitter=noise_jitter,
            )
        )

    baseline = jnp.concatenate(baselines, axis=0)
    contrast_matrix = jnp.concatenate(contrast_matrices, axis=0)
    noise_variance = jnp.concatenate(noise_variances, axis=0)

    if column_mode == "pressure_perturbation":
        sigma_b = numpyro.sample("sigma_log_p", dist.HalfNormal(sigma_b_scale))
        numpyro.deterministic("sigma_b", sigma_b)
    else:
        sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(sigma_b_scale))
    if fixed_ell_b is None:
        ell_b = numpyro.sample("ell_b", dist.Uniform(0.1, 1.5))
    else:
        ell_b = numpyro.deterministic("ell_b", jnp.asarray(fixed_ell_b))
    contrast_covariance = squared_exponential_covariance(
        distance_matrix,
        sigma_b,
        ell_b,
    )
    if column_mode == "pressure_perturbation" and zero_mean_pressure_map:
        contrast_covariance = project_zero_mean_covariance(contrast_covariance)
    contrast_covariance = add_diagonal_jitter(
        contrast_covariance,
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
