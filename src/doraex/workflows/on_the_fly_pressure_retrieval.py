"""Generic on-the-fly pressure-map retrieval with independent molecular VMRs."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.special as jsp_special
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.reparam import TransformReparam

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data
from doraex.geometry.healpix import angular_distance_matrix, healpix_pixel_angles
from doraex.geometry.limb_darkening import kipping_q_to_u
from doraex.inference.marginal_likelihood import diagonal_noise_variance
from doraex.operators.design_matrix import (
    linear_profile_operator_from_times,
)
from doraex.operators.doppler import doppler_padded_wavelengths
from doraex.priors.spherical_gp import (
    add_diagonal_jitter,
    squared_exponential_covariance,
    zero_mean_basis,
    zero_mean_covariance_factor,
)
from doraex.spectra.exojax_forward import (
    FixedPowerLawAtmosphere,
    Luhman16BPowerLawColumnModel,
)

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Luhman16BGeometry:
    """HEALPix geometry used by the pressure-map retrieval."""

    theta: jnp.ndarray
    phi: jnp.ndarray
    distance_matrix: jnp.ndarray
    nside: int


def build_luhman16b_geometry(nside=8, order="ring"):
    """Build the HEALPix centers and pairwise angular-distance matrix."""

    theta, phi = healpix_pixel_angles(nside, order=order)
    return Luhman16BGeometry(
        theta=theta,
        phi=phi,
        distance_matrix=angular_distance_matrix(theta, phi),
        nside=nside,
    )


def build_fixed_pressure_gp_eigendecomposition(
    distance_matrix,
    length_scale,
    *,
    theta=None,
    phi=None,
):
    """Diagonalize the fixed unit-amplitude zero-mean pressure GP.

    The returned constants permit the exact reduced covariance
    ``sigma**2 K + jitter I`` to be factored during NUTS using only scalar
    square roots.  This removes differentiation through a near-singular
    Cholesky decomposition while preserving the GP covariance.
    """

    distance = np.asarray(distance_matrix)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_scale must be finite and positive.")

    # Match the runtime kernel inputs, but perform the one-time decomposition
    # in float64 even when the NUTS target is evaluated in float32.
    if (theta is None) != (phi is None):
        raise ValueError("theta and phi must be supplied together.")
    if theta is None:
        distance64 = np.asarray(distance, dtype=np.float64)
    else:
        theta64 = np.asarray(theta, dtype=np.float64)
        phi64 = np.asarray(phi, dtype=np.float64)
        if theta64.shape != (distance.shape[0],) or phi64.shape != theta64.shape:
            raise ValueError(
                "theta and phi must match the distance-matrix dimension."
            )
        cosine_distance = (
            np.cos(theta64[:, None]) * np.cos(theta64[None, :])
            + np.sin(theta64[:, None])
            * np.sin(theta64[None, :])
            * np.cos(phi64[:, None] - phi64[None, :])
        )
        distance64 = np.arccos(
            np.clip(cosine_distance, -1.0, 1.0)
        )
        distance64 = 0.5 * (distance64 + distance64.T)
        np.fill_diagonal(distance64, 0.0)
    unit_covariance = np.exp(
        -(distance64 * distance64) / (2.0 * float(length_scale) ** 2)
    )
    basis = np.asarray(
        zero_mean_basis(distance.shape[0]),
        dtype=np.float64,
    )
    reduced_covariance = basis.T @ unit_covariance @ basis
    reduced_covariance = 0.5 * (
        reduced_covariance + reduced_covariance.T
    )
    eigenvalues, reduced_eigenvectors = np.linalg.eigh(reduced_covariance)

    maximum_eigenvalue = float(np.max(eigenvalues))
    negative_tolerance = max(1.0e-12, 1.0e-10 * maximum_eigenvalue)
    minimum_eigenvalue = float(np.min(eigenvalues))
    if minimum_eigenvalue < -negative_tolerance:
        raise ValueError(
            "The fixed pressure GP has a materially negative eigenvalue: "
            f"{minimum_eigenvalue:.6e} < {-negative_tolerance:.6e}."
        )
    negative_count = int(np.sum(eigenvalues < 0.0))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    pixel_eigenvectors = basis @ reduced_eigenvectors

    target_dtype = distance.dtype
    if not np.issubdtype(target_dtype, np.floating):
        target_dtype = np.dtype(np.float64)
    return {
        "eigenvalues": jnp.asarray(eigenvalues, dtype=target_dtype),
        "pixel_eigenvectors": jnp.asarray(
            pixel_eigenvectors,
            dtype=target_dtype,
        ),
        "minimum_raw_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": maximum_eigenvalue,
        "clipped_negative_eigenvalue_count": negative_count,
        "negative_eigenvalue_tolerance": negative_tolerance,
    }


YAMA_L16B_EXOMOL_ATMOSPHERE = FixedPowerLawAtmosphere(
    t0=1219.0,
    alpha=0.129,
    logg=4.97,
    log_vmr_co=-2.96,
    log_vmr_h2o=-3.25,
    log_vmr_ch4=-4.65,
    log_vmr_hf=-7.08,
    rv=25.66,
    log_p_cloud=1.45,
)


def _molecule_paths(database_dir):
    """Return the ExoMol database paths used by the fiducial atmosphere."""

    database = Path(os.path.expanduser(database_dir))
    return {
        "CO": database / "CO" / "12C-16O" / "Li2015",
        "H2O": database / "H2O" / "1H2-16O" / "POKAZATEL",
        "CH4": database / "CH4" / "12C-1H4" / "MM",
        "HF": database / "HF" / "1H-19F" / "Coxon-Hajig",
    }


def _cia_paths(database_dir):
    """Return the H2-H2 and H2-He CIA database paths."""

    database = Path(os.path.expanduser(database_dir))
    return {
        "H2H2": database / "H2-H2_2011.cia",
        "H2He": database / "H2-He_2011.cia",
    }

LOG_VMR_NAMES = ("log_vmr_co", "log_vmr_h2o", "log_vmr_ch4", "log_vmr_hf")
GAUSSIANIZED_ATMOSPHERE_NAMES = (
    "T0",
    "alpha",
    "log_vmr_co",
    "log_vmr_h2o",
    "log_vmr_ch4",
    "log_vmr_hf",
    "log_p_cloud",
    "sigma_log_p",
)
JOINT_ATMOSPHERE_A_SIGMA_D_SITE = "atmosphere_a_sigma_d_rotated"
A_PRIOR_BOUNDS = (1.0, 1.2)
SIGMA_D_LOGNORMAL_MEDIAN = 0.03
SIGMA_D_LOGNORMAL_SCALE = 1.0
ATMOSPHERE_NAMES = (
    "T0",
    "alpha",
    "logg",
    "log_vmr_co",
    "log_vmr_h2o",
    "log_vmr_ch4",
    "log_vmr_hf",
    "log_p_cloud",
)


def active_gaussianized_atmosphere_names(
    fixed_sigma_log_p=None,
    direct_sigma_log_p=False,
):
    """Return the physical parameters represented by the Normal vector."""

    if fixed_sigma_log_p is not None or direct_sigma_log_p:
        return GAUSSIANIZED_ATMOSPHERE_NAMES[:-1]
    return GAUSSIANIZED_ATMOSPHERE_NAMES


def sigma_log_p_parameterization(
    *,
    gaussianized_atmosphere,
    standardized_parameters,
    uses_eigen_basis,
    fixed_sigma_log_p,
    direct_sigma_log_p,
):
    """Return a serializable label for the pressure-amplitude coordinate."""

    if uses_eigen_basis:
        return "not_applicable_eigen_basis"
    if fixed_sigma_log_p is not None:
        return "fixed"
    if gaussianized_atmosphere and not direct_sigma_log_p:
        return "gaussianized_prior_cdf"
    if standardized_parameters:
        return "standardized_log_normal"
    return "direct_halfnormal"


def joint_atmosphere_a_sigma_d_names(chip_count):
    """Return the prior-normal coordinate order used by the joint rotation."""

    return (
        *(f"prior_normal({name})" for name in GAUSSIANIZED_ATMOSPHERE_NAMES),
        *(f"probit_A[{index}]" for index in range(int(chip_count))),
        *(
            f"log_sigma_d_over_0.03[{index}]"
            for index in range(int(chip_count))
        ),
    )


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_float_list(text):
    """Parse comma-separated floating point values."""

    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def parse_name_list(text):
    """Parse comma-separated sample-site names."""

    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one site name is required.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Sample-site names must be unique.")
    return values


def validate_atmosphere_rotation(values, dimension=None):
    """Return an orthogonal atmosphere rotation matrix."""

    if dimension is None:
        dimension = len(GAUSSIANIZED_ATMOSPHERE_NAMES)
    if values is None:
        return jnp.eye(dimension)
    array = np.asarray(values, dtype=float)
    if array.size != dimension * dimension:
        raise ValueError(
            "--atmosphere-rotation-matrix must contain exactly "
            f"{dimension * dimension} values, got {array.size}."
        )
    matrix = array.reshape(dimension, dimension)
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(dimension), rtol=1.0e-6, atol=1.0e-6):
        error = float(np.max(np.abs(gram - np.eye(dimension))))
        raise ValueError(
            "--atmosphere-rotation-matrix must be orthogonal; "
            f"max |Q.T Q - I| is {error:.3e}."
        )
    return jnp.asarray(matrix)


def validate_joint_atmosphere_a_sigma_d_rotation(values, chip_count):
    """Return an orthogonal joint atmosphere/A/sigma_d rotation matrix."""

    dimension = len(GAUSSIANIZED_ATMOSPHERE_NAMES) + 2 * int(chip_count)
    if values is None:
        return jnp.eye(dimension)
    array = np.asarray(values, dtype=float)
    if array.size != dimension * dimension:
        raise ValueError(
            "--joint-atmosphere-a-sigma-d-rotation-matrix must contain "
            f"exactly {dimension * dimension} values, got {array.size}."
        )
    matrix = array.reshape(dimension, dimension)
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(dimension), rtol=1.0e-6, atol=1.0e-6):
        error = float(np.max(np.abs(gram - np.eye(dimension))))
        raise ValueError(
            "--joint-atmosphere-a-sigma-d-rotation-matrix must be "
            f"orthogonal; max |Q.T Q - I| is {error:.3e}."
        )
    return jnp.asarray(matrix)


def _bounded_atmosphere_prior_arrays(
    t0_bounds,
    alpha_bounds,
    log_vmr_bounds,
    log_p_cloud_bounds,
):
    """Return lower and upper bounds in Gaussianized-atmosphere order."""

    bounded_names = GAUSSIANIZED_ATMOSPHERE_NAMES[:-1]
    bounds = {
        "T0": t0_bounds,
        "alpha": alpha_bounds,
        **log_vmr_bounds,
        "log_p_cloud": log_p_cloud_bounds,
    }
    lower = jnp.asarray([bounds[name][0] for name in bounded_names])
    upper = jnp.asarray([bounds[name][1] for name in bounded_names])
    return lower, upper


def gaussianized_atmosphere_to_physical(
    gaussianized,
    *,
    t0_bounds,
    alpha_bounds,
    log_vmr_bounds,
    log_p_cloud_bounds,
    sigma_log_p_scale,
):
    """Map standard-normal prior coordinates to the direct physical priors."""

    gaussianized = jnp.asarray(gaussianized)
    expected_shape = (len(GAUSSIANIZED_ATMOSPHERE_NAMES),)
    if gaussianized.shape != expected_shape:
        raise ValueError(
            f"gaussianized atmosphere must have shape {expected_shape}, "
            f"got {gaussianized.shape}."
        )
    lower, upper = _bounded_atmosphere_prior_arrays(
        t0_bounds,
        alpha_bounds,
        log_vmr_bounds,
        log_p_cloud_bounds,
    )
    bounded = lower + (upper - lower) * jsp_special.ndtr(gaussianized[:-1])

    # If u = Phi(g), the HalfNormal inverse CDF is
    # scale * Phi^-1((1 + u) / 2). The survival-function form below avoids
    # rounding (1 + u) / 2 to exactly one in float32 for positive g.
    sigma_log_p = -jnp.asarray(sigma_log_p_scale) * jsp_special.ndtri(
        0.5 * jsp_special.ndtr(-gaussianized[-1])
    )
    values = {
        name: bounded[index]
        for index, name in enumerate(GAUSSIANIZED_ATMOSPHERE_NAMES[:-1])
    }
    values["sigma_log_p"] = sigma_log_p
    return values


def physical_atmosphere_to_gaussianized(
    physical_values,
    *,
    t0_bounds,
    alpha_bounds,
    log_vmr_bounds,
    log_p_cloud_bounds,
    sigma_log_p_scale,
):
    """Map direct-prior physical values to standard-normal coordinates."""

    lower, upper = _bounded_atmosphere_prior_arrays(
        t0_bounds,
        alpha_bounds,
        log_vmr_bounds,
        log_p_cloud_bounds,
    )
    bounded = jnp.asarray(
        [
            physical_values[name]
            for name in GAUSSIANIZED_ATMOSPHERE_NAMES[:-1]
        ]
    )
    bounded_quantiles = (bounded - lower) / (upper - lower)
    sigma_log_p = jnp.asarray(physical_values["sigma_log_p"])
    sigma_quantile = jsp_special.erf(
        sigma_log_p / (jnp.asarray(sigma_log_p_scale) * jnp.sqrt(2.0))
    )
    quantiles = jnp.concatenate(
        [bounded_quantiles, jnp.reshape(sigma_quantile, (1,))]
    )
    return jsp_special.ndtri(quantiles)


def gaussianized_nuisance_to_physical(a_gaussianized, sigma_d_gaussianized):
    """Map standard-normal coordinates to the original A and sigma_d priors."""

    a_gaussianized = jnp.asarray(a_gaussianized)
    sigma_d_gaussianized = jnp.asarray(sigma_d_gaussianized)
    normalization_factor = A_PRIOR_BOUNDS[0] + (
        A_PRIOR_BOUNDS[1] - A_PRIOR_BOUNDS[0]
    ) * jsp_special.ndtr(a_gaussianized)
    sigma_d = SIGMA_D_LOGNORMAL_MEDIAN * jnp.exp(
        SIGMA_D_LOGNORMAL_SCALE * sigma_d_gaussianized
    )
    return normalization_factor, sigma_d


def physical_nuisance_to_gaussianized(normalization_factor, sigma_d):
    """Map A and sigma_d values to their standard-normal prior coordinates."""

    normalization_factor = jnp.asarray(normalization_factor)
    sigma_d = jnp.asarray(sigma_d)
    a_quantile = (
        normalization_factor - A_PRIOR_BOUNDS[0]
    ) / (A_PRIOR_BOUNDS[1] - A_PRIOR_BOUNDS[0])
    a_gaussianized = jsp_special.ndtri(a_quantile)
    sigma_d_gaussianized = jnp.log(
        sigma_d / SIGMA_D_LOGNORMAL_MEDIAN
    ) / SIGMA_D_LOGNORMAL_SCALE
    return a_gaussianized, sigma_d_gaussianized


def zero_sum_log_w_base(log_w):
    """Convert zero-mean phase weights to the 13-coordinate base space."""

    zero_mean = _zero_mean_log_w(log_w)
    return dist.transforms.ZeroSumTransform(1).inv(zero_mean)


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description=(
            "Run an on-the-fly atmospheric pressure-map retrieval. The shared T0, "
            "alpha, independent log VMRs, and log10(P_cloud) parameters are "
            "sampled, and d spectrum / d log10(P_cloud) is evaluated with "
            "JAX JVP inside the NumPyro model."
        )
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--chip-indices", type=parse_chips, default=parse_chips("1"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "on_the_fly_pressure_retrieval"),
    )
    parser.add_argument("--run-label", default="on_the_fly_pressure_retrieval")
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--nside", type=int, default=2)
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use all phases and wavelengths instead of the reduced smoke subset.",
    )
    parser.add_argument(
        "--mask-zero-flux",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exclude exactly zero flux samples from the likelihood and reject "
            "non-finite flux. The full arrays and static validity masks remain "
            "in the output archive."
        ),
    )
    parser.add_argument("--smoke-wavelength-step", type=int, default=128)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept-prob", type=float, default=0.95)
    parser.add_argument("--max-tree-depth", type=int, default=8)
    parser.add_argument(
        "--warmup-max-tree-depth",
        type=int,
        default=None,
        help=(
            "Optional NUTS tree-depth limit used only during warmup. "
            "--max-tree-depth remains the sampling limit. NumPyro supports "
            "the pair (warmup depth, sampling depth)."
        ),
    )
    parser.add_argument("--fixed-period", type=float, default=4.83)
    parser.add_argument("--fixed-cosi", type=float, default=0.485)
    parser.add_argument("--fixed-v", type=float, default=31.2)
    parser.add_argument("--fixed-q1", type=float, default=0.81)
    parser.add_argument("--fixed-q2", type=float, default=0.59)
    parser.add_argument("--logg-prior-mean", type=float, default=4.86)
    parser.add_argument("--logg-prior-sigma", type=float, default=0.09)
    parser.add_argument("--logg-min", type=float, default=4.59)
    parser.add_argument("--logg-max", type=float, default=5.13)
    parser.add_argument("--init-logg", type=float, default=4.86)
    parser.add_argument(
        "--fix-logg",
        action="store_true",
        help="Fix logg to --init-logg instead of sampling it.",
    )
    parser.add_argument("--t0-min", type=float, default=1000.0)
    parser.add_argument("--t0-max", type=float, default=1700.0)
    parser.add_argument("--init-t0", type=float, default=1219.0)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.20)
    parser.add_argument("--init-alpha", type=float, default=0.129)
    parser.add_argument("--log-vmr-co-min", type=float, default=-3.5)
    parser.add_argument("--log-vmr-co-max", type=float, default=-2.4)
    parser.add_argument("--init-log-vmr-co", type=float, default=-2.885)
    parser.add_argument("--log-vmr-h2o-min", type=float, default=-3.8)
    parser.add_argument("--log-vmr-h2o-max", type=float, default=-2.7)
    parser.add_argument("--init-log-vmr-h2o", type=float, default=-3.175)
    parser.add_argument("--log-vmr-ch4-min", type=float, default=-5.2)
    parser.add_argument("--log-vmr-ch4-max", type=float, default=-4.0)
    parser.add_argument("--init-log-vmr-ch4", type=float, default=-4.575)
    parser.add_argument("--log-vmr-hf-min", type=float, default=-7.6)
    parser.add_argument("--log-vmr-hf-max", type=float, default=-6.4)
    parser.add_argument("--init-log-vmr-hf", type=float, default=-7.005)
    parser.add_argument("--log-p-cloud-min", type=float, default=-1.0)
    parser.add_argument("--log-p-cloud-max", type=float, default=2.0)
    parser.add_argument("--init-log-p-cloud", type=float, default=1.35)
    parser.add_argument("--sigma-log-p-scale", type=float, default=0.1)
    parser.add_argument("--init-sigma-log-p", type=float, default=0.22)
    parser.add_argument(
        "--fixed-sigma-log-p",
        type=float,
        default=None,
        help=(
            "Fix sigma_log_p to this positive value instead of sampling it. "
            "This is intended for pressure-map amplitude diagnostics."
        ),
    )
    parser.add_argument(
        "--standardized-parameters",
        action="store_true",
        help=(
            "Sample standardized raw atmospheric coordinates and expose the "
            "physical parameters as deterministic sites."
        ),
    )
    parser.add_argument(
        "--rotated-atmosphere-parameters",
        action="store_true",
        help=(
            "Reserved for a future atmosphere-specific rotation. zeta_vmr "
            "rotations are not used with independent molecular abundances."
        ),
    )
    parser.add_argument(
        "--gaussianized-atmosphere",
        action="store_true",
        help=(
            "Preserve the direct physical priors by mapping independent standard "
            "normal coordinates through their prior CDFs. An optional fixed "
            "orthogonal rotation can align these coordinates with posterior ridges."
        ),
    )
    parser.add_argument(
        "--direct-sigma-log-p",
        action="store_true",
        help=(
            "With --gaussianized-atmosphere, sample sigma_log_p directly from "
            "its HalfNormal prior instead of including it in the prior-CDF "
            "Gaussianized atmosphere vector. The seven bounded atmospheric "
            "parameters remain Gaussianized and rotated."
        ),
    )
    parser.add_argument(
        "--atmosphere-rotation-matrix",
        type=parse_float_list,
        default=None,
        help=(
            "Optional row-major orthogonal matrix for "
            "--gaussianized-atmosphere. Its dimension is 7 when sigma_log_p "
            "is fixed or sampled with --direct-sigma-log-p, and 8 otherwise. "
            "The identity is used when omitted."
        ),
    )
    parser.add_argument(
        "--atmosphere-rotation-label",
        default="identity",
        help="Serializable provenance label for --atmosphere-rotation-matrix.",
    )
    parser.add_argument(
        "--joint-atmosphere-a-sigma-d",
        action="store_true",
        help=(
            "Sample the Gaussianized atmosphere, A, and sigma_d through one "
            "prior-preserving orthogonal coordinate system. This requires "
            "--gaussianized-atmosphere."
        ),
    )
    parser.add_argument(
        "--joint-atmosphere-a-sigma-d-rotation-matrix",
        type=parse_float_list,
        default=None,
        help=(
            "Optional row-major orthogonal matrix for the joint atmosphere, "
            "A, and sigma_d prior-normal coordinates. Its dimension is "
            "8 + 2 times the number of chips."
        ),
    )
    parser.add_argument(
        "--joint-atmosphere-a-sigma-d-rotation-label",
        default="identity",
        help="Serializable provenance label for the joint rotation matrix.",
    )
    parser.add_argument("--t0-raw-scale", type=float, default=5.0)
    parser.add_argument("--alpha-raw-scale", type=float, default=0.001)
    parser.add_argument("--logg-raw-scale", type=float, default=0.09)
    parser.add_argument("--log-vmr-raw-scale", type=float, default=0.02)
    parser.add_argument("--log-p-cloud-raw-scale", type=float, default=0.012)
    parser.add_argument("--sigma-log-p-log-raw-scale", type=float, default=0.1)
    parser.add_argument(
        "--zeta-vmr-per-t0",
        type=float,
        default=1.0e-3,
        help="Unused with independent molecular abundances.",
    )
    parser.add_argument(
        "--log-p-cloud-per-alpha",
        type=float,
        default=5.0,
        help=(
            "Ridge slope d(log_p_cloud) / d(alpha) used by rotated coordinates."
        ),
    )
    parser.add_argument("--fixed-ell-b", type=float, default=0.3)
    parser.add_argument("--zero-mean-pressure-map", action="store_true", default=True)
    parser.add_argument(
        "--no-zero-mean-pressure-map",
        action="store_false",
        dest="zero_mean_pressure_map",
    )
    parser.add_argument("--log-w-scale", type=float, default=0.1)
    parser.add_argument(
        "--zero-mean-log-w",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sample log_w_raw and expose chip-wise zero-mean log_w values. "
            "This removes the chip-common phase-weight mode from the mean model."
        ),
    )
    parser.add_argument(
        "--zero-sum-log-w-basis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Represent each chip's zero-mean phase weights with 13 independent "
            "ZeroSumNormal base coordinates rather than 14 projected coordinates."
        ),
    )
    parser.add_argument(
        "--fix-nuisance",
        action="store_true",
        help="Fix A, log_w, and sigma_d to their initial values.",
    )
    parser.add_argument(
        "--fix-a",
        action="store_true",
        help=(
            "Fix only the chip normalization factors A to their initial "
            "values. This can be combined with --fix-log-w or --fix-sigma-d."
        ),
    )
    parser.add_argument(
        "--fix-sigma-d",
        action="store_true",
        help=(
            "Fix only the chip noise amplitudes sigma_d to their initial "
            "values. This can be combined with --fix-log-w or --fix-a."
        ),
    )
    parser.add_argument(
        "--fix-log-w",
        action="store_true",
        help=(
            "Fix log_w to its chip-wise zero-mean initial value. When used "
            "alone, continue to sample A and sigma_d; it can also be combined "
            "with --fix-a or --fix-sigma-d."
        ),
    )
    parser.add_argument("--gp-jitter", type=float, default=0.5e-6)
    parser.add_argument(
        "--pressure-gp-factorization",
        choices=("cholesky", "fixed_eigen"),
        default="cholesky",
        help=(
            "Factor the pressure GP with the original per-evaluation Cholesky "
            "or a one-time eigendecomposition of the fixed-length-scale kernel."
        ),
    )
    parser.add_argument("--noise-jitter", type=float, default=1.0e-6)
    parser.add_argument(
        "--eigen-basis",
        default=None,
        help=(
            "Optional eigen-response basis NPZ. When supplied, the pressure "
            "map is replaced by the selected eigen-atmosphere maps."
        ),
    )
    parser.add_argument(
        "--eigen-mode-count",
        type=int,
        default=None,
        help="Number of eigenmodes from --eigen-basis to use.",
    )
    parser.add_argument(
        "--eigen-fixed-ell",
        type=float,
        default=None,
        help="Fixed GP length scale for eigenmode maps. Defaults to --fixed-ell-b.",
    )
    parser.add_argument(
        "--eigen-sigma-scale",
        type=float,
        default=1.0,
        help=(
            "Half-normal prior scale for dimensionless eigenmode amplitudes "
            "in direct-coordinate runs. Standardized runs use "
            "--init-sigma-eigen as the lognormal center."
        ),
    )
    parser.add_argument(
        "--init-sigma-eigen",
        type=float,
        default=1.0,
        help=(
            "Initial value for dimensionless eigenmode map amplitudes. In "
            "standardized runs this is also the lognormal prior center."
        ),
    )
    parser.add_argument(
        "--sigma-eigen-log-raw-scale",
        type=float,
        default=0.5,
        help=(
            "Log-space scale for standardized eigenmode amplitude raw "
            "coordinates."
        ),
    )
    parser.add_argument(
        "--fixed-sigma-eigen",
        type=parse_float_list,
        default=None,
        help=(
            "Comma-separated fixed dimensionless eigenmode amplitudes. When "
            "set, sigma_eigen is deterministic instead of sampled."
        ),
    )
    parser.add_argument(
        "--frozen-eigen-spectra",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use eigenspectra stored in --eigen-basis instead of recomputing "
            "eigen-response JVPs during NUTS."
        ),
    )
    parser.add_argument(
        "--init-from",
        default=str(
            ROOT
            / "results"
            / "milestone5"
            / "milestone5_on_the_fly_atmosphere_stage1_rotated_diag2_cholfix_ta095_prod_f32"
            / "legacy_figures_json"
            / "mcmc_on_the_fly_atmosphere_pressure.npz"
        ),
        help=(
            "Optional previous posterior NPZ used for A, log_w, sigma_d, "
            "atmospheric centers, log_p_cloud, and sigma_log_p median initial "
            "values."
        ),
    )
    parser.add_argument(
        "--no-init-from",
        action="store_const",
        const=None,
        dest="init_from",
        help="Disable previous-posterior initialization.",
    )
    parser.add_argument(
        "--manual-atmosphere-init",
        action="store_true",
        help=(
            "Use --init-* values for atmospheric centers while still using "
            "--init-from for nuisance values."
        ),
    )
    parser.add_argument(
        "--dense-mass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use dense NUTS mass-matrix adaptation.",
    )
    parser.add_argument(
        "--dense-atmosphere-mass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Adapt a dense mass-matrix block for the atmospheric sample sites "
            "while retaining diagonal adaptation for nuisance parameters."
        ),
    )
    parser.add_argument(
        "--dense-atmosphere-a-sigma-d-mass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use one structured NUTS block joining the atmosphere, A, and "
            "sigma_d sample sites. log_w remains outside this block."
        ),
    )
    parser.add_argument(
        "--initial-inverse-mass-diagonal",
        type=parse_float_list,
        default=None,
        help=(
            "Optional positive diagonal of the initial NumPyro inverse mass "
            "matrix, ordered by --initial-inverse-mass-sites. NumPyro calls "
            "this an inverse mass matrix but estimates "
            "it as a position-space posterior covariance."
        ),
    )
    parser.add_argument(
        "--initial-inverse-mass-sites",
        type=parse_name_list,
        default=None,
        help=(
            "Comma-separated sample-site order for an explicitly supplied "
            "diagonal metric. Every active sample site must be listed."
        ),
    )
    parser.add_argument(
        "--initial-inverse-mass-label",
        default="",
        help="Serializable provenance label for an explicitly supplied metric.",
    )
    parser.add_argument(
        "--adapt-mass-matrix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Adapt the NUTS mass matrix during warmup. Disable this to retain "
            "an explicitly supplied empirical metric while adapting only the "
            "step size."
        ),
    )
    parser.add_argument(
        "--map-init",
        action="store_true",
        help="Use numpyro-inferutils SVI MAP initialization before NUTS.",
    )
    parser.add_argument("--map-init-steps", type=int, default=1000)
    parser.add_argument("--map-init-step-size", type=float, default=1.0e-3)
    parser.add_argument(
        "--preflight-autodiff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compile and evaluate the on-the-fly spectrum/JVP response once "
            "before starting NUTS."
        ),
    )
    parser.add_argument(
        "--x64",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable JAX 64-bit mode. M5 defaults to 32-bit to reduce GPU memory.",
    )
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_chip_data(args, *, return_native_wavelengths=False):
    """Load Luhman 16B chip data for the retrieval."""

    chip_data_list = []
    native_wavelengths = []
    for chip_index in args.chip_indices:
        chip_data = load_luhman16b_chip(args.data_dir, chip_index=chip_index)
        native_wavelengths.append(np.asarray(chip_data.wavelengths))
        if not args.full_data:
            chip_data = subset_chip_data(
                chip_data,
                wavelength_step=args.smoke_wavelength_step,
                phase_count=args.smoke_phase_count,
            )
        chip_data_list.append(chip_data)
    if return_native_wavelengths:
        return chip_data_list, native_wavelengths
    return chip_data_list


def build_observation_masks(chip_data_list, *, mask_zero_flux=False):
    """Return static per-chip masks for observations used by the likelihood."""

    masks = []
    for chip_data in chip_data_list:
        flux = np.asarray(chip_data.flux)
        if not np.all(np.isfinite(flux)):
            raise ValueError(
                f"Non-finite flux values found for chip {chip_data.chip_index}."
            )
        if mask_zero_flux:
            mask = flux != 0.0
        else:
            mask = np.ones(flux.shape, dtype=bool)
        if not np.any(mask):
            raise ValueError(
                f"No valid observations remain for chip {chip_data.chip_index}."
            )
        masks.append(mask)
    return masks


def build_doppler_profile_wavelengths(wavelengths_by_chip, max_abs_velocity):
    """Build padded local-profile grids from native chip wavelength grids."""

    return [
        doppler_padded_wavelengths(
            wavelengths,
            max_abs_velocity,
        )
        for wavelengths in wavelengths_by_chip
    ]


def _response_function(spectrum_function):
    """Return a function that evaluates spectrum and pressure JVP."""

    def response(
        t0,
        alpha,
        log_vmr_co,
        log_vmr_h2o,
        log_vmr_ch4,
        log_vmr_hf,
        logg,
        log_p_cloud,
    ):
        def spectrum_at_pressure(pressure):
            return spectrum_function(
                t0,
                alpha,
                log_vmr_co,
                log_vmr_h2o,
                log_vmr_ch4,
                log_vmr_hf,
                pressure,
                logg=logg,
            )

        return jax.jvp(
            spectrum_at_pressure,
            (log_p_cloud,),
            (jnp.ones_like(log_p_cloud),),
        )

    return response


def _eigen_response_function(
    spectrum_function,
    parameter_names,
    parameter_scales,
    selected_v,
):
    """Return a function that evaluates spectrum and eigen-response JVPs."""

    parameter_names = tuple(str(name) for name in parameter_names)
    parameter_scales = jnp.asarray(parameter_scales)
    selected_v = jnp.asarray(selected_v)
    directions = parameter_scales[:, None] * selected_v

    def response(
        t0,
        alpha,
        log_vmr_co,
        log_vmr_h2o,
        log_vmr_ch4,
        log_vmr_hf,
        logg,
        log_p_cloud,
    ):
        values = {
            "T0": t0,
            "alpha": alpha,
            "logg": logg,
            "log_vmr_co": log_vmr_co,
            "log_vmr_h2o": log_vmr_h2o,
            "log_vmr_ch4": log_vmr_ch4,
            "log_vmr_hf": log_vmr_hf,
            "log_p_cloud": log_p_cloud,
        }
        xi0 = jnp.asarray([values[name] for name in parameter_names])

        def spectrum_at_xi(xi):
            local_values = dict(values)
            for index, name in enumerate(parameter_names):
                local_values[name] = xi[index]
            return spectrum_function(
                local_values["T0"],
                local_values["alpha"],
                local_values["log_vmr_co"],
                local_values["log_vmr_h2o"],
                local_values["log_vmr_ch4"],
                local_values["log_vmr_hf"],
                local_values["log_p_cloud"],
                logg=local_values["logg"],
            )

        base_profile = spectrum_at_xi(xi0)

        eigen_profiles = []
        for mode_index in range(selected_v.shape[1]):
            _, tangent = jax.jvp(
                spectrum_at_xi,
                (xi0,),
                (directions[:, mode_index],),
            )
            eigen_profiles.append(tangent)
        eigen_profiles = jnp.stack(eigen_profiles, axis=0)
        return base_profile, eigen_profiles

    return response


def _frozen_eigen_response_function(spectrum_function, eigen_profiles):
    """Return a response function with fixed precomputed eigenspectra."""

    eigen_profiles = jnp.asarray(eigen_profiles)

    def response(
        t0,
        alpha,
        log_vmr_co,
        log_vmr_h2o,
        log_vmr_ch4,
        log_vmr_hf,
        logg,
        log_p_cloud,
    ):
        base_profile = spectrum_function(
            t0,
            alpha,
            log_vmr_co,
            log_vmr_h2o,
            log_vmr_ch4,
            log_vmr_hf,
            log_p_cloud,
            logg=logg,
        )
        return base_profile, eigen_profiles

    return response


def load_eigen_basis(path, mode_count=None):
    """Load an eigen-response basis archive for E-series experiments."""

    if path is None:
        return None
    basis = np.load(path, allow_pickle=False)
    parameter_names = tuple(str(name) for name in basis["parameter_names"])
    parameter_scales = np.asarray(basis["parameter_scales"], dtype=float)
    selected_v = np.asarray(basis["selected_v"], dtype=float)
    if mode_count is not None:
        if mode_count > selected_v.shape[1]:
            raise ValueError(
                f"Requested {mode_count} eigenmodes, but basis contains "
                f"{selected_v.shape[1]} selected modes."
            )
        selected_v = selected_v[:, :mode_count]
    if selected_v.ndim != 2 or selected_v.shape[0] != len(parameter_names):
        raise ValueError(
            "Eigen basis selected_v must have shape "
            "(n_parameter, n_mode)."
        )
    if len(parameter_scales) != len(parameter_names):
        raise ValueError("Eigen basis parameter scales do not match names.")
    return {
        "path": str(path),
        "parameter_names": parameter_names,
        "parameter_scales": parameter_scales,
        "selected_v": selected_v,
        "mode_count": int(selected_v.shape[1]),
        "npz": basis,
    }


def build_response_functions(
    args,
    chip_data_list,
    eigen_basis=None,
    profile_wavelengths=None,
):
    """Build on-the-fly ExoJAX spectrum/pressure-response functions."""

    uses_separate_profile_grid = profile_wavelengths is not None
    if profile_wavelengths is None:
        profile_wavelengths = [chip.wavelengths for chip in chip_data_list]
    if len(profile_wavelengths) != len(chip_data_list):
        raise ValueError(
            "profile_wavelengths must contain one grid per observed chip."
        )

    response_functions = []
    for chip_position, chip_data in enumerate(chip_data_list):
        profile_grid = profile_wavelengths[chip_position]
        eigen_profiles = None
        if eigen_basis is not None and args.frozen_eigen_spectra:
            key = f"eigenspectra_chip{chip_data.chip_index}"
            wavelength_key = f"profile_wavelengths_chip{chip_data.chip_index}"
            if key not in eigen_basis["npz"].files:
                raise KeyError(f"Missing {key} in eigen basis {eigen_basis['path']}")
            if wavelength_key not in eigen_basis["npz"].files:
                if uses_separate_profile_grid:
                    raise ValueError(
                        f"Missing {wavelength_key} in eigen basis "
                        f"{eigen_basis['path']}; regenerate it on a padded grid."
                    )
            else:
                eigen_profile_grid = np.asarray(eigen_basis["npz"][wavelength_key])
                if not np.array_equal(eigen_profile_grid, np.asarray(profile_grid)):
                    raise ValueError(
                        f"{wavelength_key} does not match the requested padded "
                        "profile grid; regenerate the eigen basis."
                    )
            stored_eigen_profiles = np.asarray(eigen_basis["npz"][key])
            eigen_profiles = (
                stored_eigen_profiles[:, : eigen_basis["mode_count"]].T
                if stored_eigen_profiles.ndim == 2
                else stored_eigen_profiles
            )
            expected_shape = (eigen_basis["mode_count"], len(profile_grid))
            if eigen_profiles.shape != expected_shape:
                raise ValueError(
                    f"{key} must have shape {expected_shape} after selecting "
                    f"eigenmodes, but got {eigen_profiles.shape}."
                )
        sampling_kwargs = {}
        if uses_separate_profile_grid:
            sampling_kwargs["sampling_wavelengths"] = profile_grid
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
            nx=args.nx,
            **sampling_kwargs,
        )
        if eigen_basis is None:
            response_functions.append(_response_function(model.cloudy_at_log_vmrs))
        elif args.frozen_eigen_spectra:
            response_functions.append(
                _frozen_eigen_response_function(
                    model.cloudy_at_log_vmrs,
                    eigen_profiles,
                )
            )
        else:
            response_functions.append(
                _eigen_response_function(
                    model.cloudy_at_log_vmrs,
                    eigen_basis["parameter_names"],
                    eigen_basis["parameter_scales"],
                    eigen_basis["selected_v"],
                )
            )
    return response_functions


def _select_observation_rows(
    observed,
    baseline,
    design_matrix,
    noise_variance,
    observation_indices=None,
):
    """Apply one static observation-row selection to every likelihood term."""

    observed = jnp.asarray(observed).reshape(-1)
    if observation_indices is None:
        return observed, baseline, design_matrix, noise_variance
    indices = jnp.asarray(observation_indices, dtype=jnp.int32)
    return (
        jnp.take(observed, indices, axis=0),
        jnp.take(baseline, indices, axis=0),
        jnp.take(design_matrix, indices, axis=0),
        jnp.take(noise_variance, indices, axis=0),
    )


def on_the_fly_pressure_model(
    data,
    wavelengths,
    obs_times,
    theta,
    phi,
    distance_matrix,
    response_functions,
    fixed_period,
    fixed_cosi,
    fixed_v,
    fixed_q1,
    fixed_q2,
    fix_logg,
    fixed_logg,
    logg_prior_mean,
    logg_prior_sigma,
    logg_bounds,
    t0_bounds,
    alpha_bounds,
    log_vmr_bounds,
    log_p_cloud_bounds,
    sigma_log_p_scale,
    standardized_parameters,
    gaussianized_atmosphere,
    atmosphere_rotation,
    parameter_centers,
    parameter_scales,
    fixed_ell_b,
    zero_mean_pressure_map,
    log_w_scale,
    zero_mean_log_w,
    zero_sum_log_w_basis,
    eigen_mode_count,
    eigen_sigma_scale,
    eigen_sigma_center,
    eigen_sigma_log_raw_scale,
    fixed_sigma_eigen,
    eigen_fixed_ell,
    fixed_nuisance_values,
    gp_jitter,
    noise_jitter,
    fixed_sigma_log_p=None,
    direct_sigma_log_p=False,
    pressure_gp_eigenvalues=None,
    pressure_gp_pixel_eigenvectors=None,
    joint_atmosphere_a_sigma_d=False,
    joint_atmosphere_a_sigma_d_rotation=None,
    profile_wavelengths=None,
    observation_indices=None,
):
    """On-the-fly pressure-perturbation retrieval model."""

    n_chip = data.shape[0]
    n_phase = data.shape[1]
    if profile_wavelengths is None:
        profile_wavelengths = [None] * n_chip
    if len(profile_wavelengths) != n_chip:
        raise ValueError(
            "profile_wavelengths must contain one grid per observed chip."
        )
    if joint_atmosphere_a_sigma_d and not gaussianized_atmosphere:
        raise ValueError(
            "Joint atmosphere/A/sigma_d coordinates require a Gaussianized "
            "atmosphere."
        )
    if direct_sigma_log_p and not gaussianized_atmosphere:
        raise ValueError(
            "Direct sigma_log_p coordinates require a Gaussianized atmosphere."
        )
    if direct_sigma_log_p and fixed_sigma_log_p is not None:
        raise ValueError(
            "Direct sigma_log_p coordinates cannot be combined with a fixed "
            "sigma_log_p."
        )
    if direct_sigma_log_p and joint_atmosphere_a_sigma_d:
        raise ValueError(
            "Direct sigma_log_p coordinates cannot be combined with joint "
            "atmosphere/A/sigma_d coordinates."
        )
    if parameter_scales["rotated_atmosphere"]:
        raise ValueError(
            "This retrieval does not implement rotated atmosphere parameters. "
            "Use standardized or direct coordinates."
        )
    joint_a_gaussianized = None
    joint_sigma_d_gaussianized = None
    if gaussianized_atmosphere:
        if not fix_logg:
            raise ValueError(
                "Gaussianized atmosphere coordinates currently require fixed logg."
            )
        gaussianized_names = active_gaussianized_atmosphere_names(
            fixed_sigma_log_p,
            direct_sigma_log_p,
        )
        if joint_atmosphere_a_sigma_d:
            if fixed_sigma_log_p is not None:
                raise ValueError(
                    "Joint atmosphere/A/sigma_d coordinates require a sampled "
                    "sigma_log_p."
                )
            joint_dimension = len(gaussianized_names) + 2 * n_chip
            if joint_atmosphere_a_sigma_d_rotation is None:
                joint_atmosphere_a_sigma_d_rotation = jnp.eye(joint_dimension)
            elif jnp.shape(joint_atmosphere_a_sigma_d_rotation) != (
                joint_dimension,
                joint_dimension,
            ):
                raise ValueError(
                    "The joint atmosphere/A/sigma_d rotation must have shape "
                    f"({joint_dimension}, {joint_dimension})."
                )
            joint_rotated = numpyro.sample(
                JOINT_ATMOSPHERE_A_SIGMA_D_SITE,
                dist.Normal(jnp.zeros((joint_dimension,)), 1.0).to_event(1),
            )
            joint_gaussianized = numpyro.deterministic(
                "atmosphere_a_sigma_d_gaussianized",
                joint_atmosphere_a_sigma_d_rotation @ joint_rotated,
            )
            atmosphere_gaussianized = numpyro.deterministic(
                "atmosphere_gaussianized",
                joint_gaussianized[: len(gaussianized_names)],
            )
            joint_a_gaussianized = joint_gaussianized[
                len(gaussianized_names) : len(gaussianized_names) + n_chip
            ]
            joint_sigma_d_gaussianized = joint_gaussianized[
                len(gaussianized_names) + n_chip :
            ]
        else:
            atmosphere_rotated = numpyro.sample(
                "atmosphere_rotated",
                dist.Normal(
                    jnp.zeros((len(gaussianized_names),)),
                    1.0,
                ).to_event(1),
            )
            atmosphere_gaussianized = numpyro.deterministic(
                "atmosphere_gaussianized",
                atmosphere_rotation @ atmosphere_rotated,
            )
        if fixed_sigma_log_p is None and not direct_sigma_log_p:
            physical_atmosphere = gaussianized_atmosphere_to_physical(
                atmosphere_gaussianized,
                t0_bounds=t0_bounds,
                alpha_bounds=alpha_bounds,
                log_vmr_bounds=log_vmr_bounds,
                log_p_cloud_bounds=log_p_cloud_bounds,
                sigma_log_p_scale=sigma_log_p_scale,
            )
        else:
            lower, upper = _bounded_atmosphere_prior_arrays(
                t0_bounds,
                alpha_bounds,
                log_vmr_bounds,
                log_p_cloud_bounds,
            )
            bounded = lower + (upper - lower) * jsp_special.ndtr(
                atmosphere_gaussianized
            )
            physical_atmosphere = {
                name: bounded[index]
                for index, name in enumerate(gaussianized_names)
            }
        t0 = numpyro.deterministic("T0", physical_atmosphere["T0"])
        alpha = numpyro.deterministic("alpha", physical_atmosphere["alpha"])
        logg = numpyro.deterministic("logg", jnp.asarray(fixed_logg))
        log_vmrs = {
            name: numpyro.deterministic(name, physical_atmosphere[name])
            for name in LOG_VMR_NAMES
        }
        log_p_cloud = numpyro.deterministic(
            "log_p_cloud",
            physical_atmosphere["log_p_cloud"],
        )
        if eigen_mode_count is None:
            if direct_sigma_log_p:
                sigma_log_p = numpyro.sample(
                    "sigma_log_p",
                    dist.HalfNormal(sigma_log_p_scale),
                )
            else:
                sigma_log_p = numpyro.deterministic(
                    "sigma_log_p",
                    (
                        physical_atmosphere["sigma_log_p"]
                        if fixed_sigma_log_p is None
                        else jnp.asarray(fixed_sigma_log_p)
                    ),
                )
    elif standardized_parameters:
        t0_raw = numpyro.sample("T0_raw", dist.Normal(0.0, 1.0))
        alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0))
        if fix_logg:
            logg = numpyro.deterministic("logg", jnp.asarray(fixed_logg))
        else:
            logg_raw_mean = (
                logg_prior_mean - parameter_centers["logg"]
            ) / parameter_scales["logg"]
            logg_raw_lower = (
                logg_bounds[0] - parameter_centers["logg"]
            ) / parameter_scales["logg"]
            logg_raw_upper = (
                logg_bounds[1] - parameter_centers["logg"]
            ) / parameter_scales["logg"]
            logg_raw = numpyro.sample(
                "logg_raw",
                dist.TruncatedNormal(
                    logg_raw_mean,
                    logg_prior_sigma / parameter_scales["logg"],
                    low=logg_raw_lower,
                    high=logg_raw_upper,
                ),
            )
        log_vmr_raw = {
            name: numpyro.sample(f"{name}_raw", dist.Normal(0.0, 1.0))
            for name in LOG_VMR_NAMES
        }
        log_p_cloud_raw = numpyro.sample(
            "log_p_cloud_raw",
            dist.Normal(0.0, 1.0),
        )
        if eigen_mode_count is None and fixed_sigma_log_p is None:
            sigma_log_p_raw = numpyro.sample(
                "sigma_log_p_raw",
                dist.Normal(0.0, 1.0),
            )
        t0 = numpyro.deterministic(
            "T0",
            parameter_centers["T0"] + parameter_scales["T0"] * t0_raw,
        )
        alpha = numpyro.deterministic(
            "alpha",
            parameter_centers["alpha"] + parameter_scales["alpha"] * alpha_raw,
        )
        if not fix_logg:
            logg = numpyro.deterministic(
                "logg",
                parameter_centers["logg"] + parameter_scales["logg"] * logg_raw,
            )
        log_vmrs = {
            name: numpyro.deterministic(
                name,
                parameter_centers[name] + parameter_scales[name] * log_vmr_raw[name],
            )
            for name in LOG_VMR_NAMES
        }
        log_p_cloud = numpyro.deterministic(
            "log_p_cloud",
            parameter_centers["log_p_cloud"]
            + parameter_scales["log_p_cloud"] * log_p_cloud_raw,
        )
        if eigen_mode_count is None:
            sigma_log_p = numpyro.deterministic(
                "sigma_log_p",
                (
                    jnp.exp(
                        jnp.log(parameter_centers["sigma_log_p"])
                        + parameter_scales["sigma_log_p"] * sigma_log_p_raw
                    )
                    if fixed_sigma_log_p is None
                    else jnp.asarray(fixed_sigma_log_p)
                ),
            )
    else:
        if fix_logg:
            logg = numpyro.deterministic("logg", jnp.asarray(fixed_logg))
        else:
            logg = numpyro.sample(
                "logg",
                dist.TruncatedNormal(
                    logg_prior_mean,
                    logg_prior_sigma,
                    low=logg_bounds[0],
                    high=logg_bounds[1],
                ),
            )
        t0 = numpyro.sample("T0", dist.Uniform(t0_bounds[0], t0_bounds[1]))
        alpha = numpyro.sample(
            "alpha",
            dist.Uniform(alpha_bounds[0], alpha_bounds[1]),
        )
        log_vmrs = {
            name: numpyro.sample(
                name,
                dist.Uniform(log_vmr_bounds[name][0], log_vmr_bounds[name][1]),
            )
            for name in LOG_VMR_NAMES
        }
        log_p_cloud = numpyro.sample(
            "log_p_cloud",
            dist.Uniform(log_p_cloud_bounds[0], log_p_cloud_bounds[1]),
        )
        if eigen_mode_count is None:
            if fixed_sigma_log_p is None:
                sigma_log_p = numpyro.sample(
                    "sigma_log_p",
                    dist.HalfNormal(sigma_log_p_scale),
                )
            else:
                sigma_log_p = numpyro.deterministic(
                    "sigma_log_p",
                    jnp.asarray(fixed_sigma_log_p),
                )
    cosi = numpyro.deterministic("cosi", jnp.asarray(fixed_cosi))
    vrot = numpyro.deterministic("v", jnp.asarray(fixed_v))
    q1 = numpyro.deterministic("q1", jnp.asarray(fixed_q1))
    q2 = numpyro.deterministic("q2", jnp.asarray(fixed_q2))
    period = numpyro.deterministic("P", jnp.asarray(fixed_period))
    inclination = jnp.arccos(cosi)
    u1, u2 = kipping_q_to_u(q1, q2)
    numpyro.deterministic("u1", u1)
    numpyro.deterministic("u2", u2)

    fixed_nuisance_values = (
        {} if fixed_nuisance_values is None else fixed_nuisance_values
    )
    if joint_atmosphere_a_sigma_d and (
        "A" in fixed_nuisance_values or "sigma_d" in fixed_nuisance_values
    ):
        raise ValueError(
            "Joint atmosphere/A/sigma_d coordinates cannot be combined with "
            "fixed A or sigma_d values."
        )
    if joint_atmosphere_a_sigma_d:
        normalization_factor, joint_sigma_d = (
            gaussianized_nuisance_to_physical(
                joint_a_gaussianized,
                joint_sigma_d_gaussianized,
            )
        )
        normalization_factor = numpyro.deterministic(
            "A",
            normalization_factor,
        )
    elif "A" in fixed_nuisance_values:
        normalization_factor = numpyro.deterministic(
            "A",
            jnp.asarray(fixed_nuisance_values["A"]),
        )
    else:
        normalization_factor = numpyro.sample(
            "A",
            dist.Uniform(*A_PRIOR_BOUNDS).expand([n_chip]),
        )

    if "log_w" in fixed_nuisance_values:
        fixed_log_w = jnp.asarray(fixed_nuisance_values["log_w"])
        log_w = numpyro.deterministic(
            "log_w",
            fixed_log_w
            - (
                jnp.mean(fixed_log_w, axis=1, keepdims=True)
                if zero_mean_log_w
                else 0.0
            ),
        )
    else:
        if zero_sum_log_w_basis:
            with handlers.reparam(config={"log_w": TransformReparam()}):
                log_w = numpyro.sample(
                    "log_w",
                    dist.ZeroSumNormal(
                        log_w_scale,
                        event_shape=(n_phase,),
                    ).expand((n_chip,)),
                )
        elif zero_mean_log_w:
            log_w_raw = numpyro.sample(
                "log_w_raw",
                dist.Normal(0.0, log_w_scale).expand([n_chip, n_phase]),
            )
            log_w = numpyro.deterministic(
                "log_w",
                log_w_raw - jnp.mean(log_w_raw, axis=1, keepdims=True),
            )
        else:
            log_w = numpyro.sample(
                "log_w",
                dist.Normal(0.0, log_w_scale).expand([n_chip, n_phase]),
            )

    if joint_atmosphere_a_sigma_d:
        sigma_d = numpyro.deterministic("sigma_d", joint_sigma_d)
    elif "sigma_d" in fixed_nuisance_values:
        sigma_d = numpyro.deterministic(
            "sigma_d",
            jnp.asarray(fixed_nuisance_values["sigma_d"]),
        )
    else:
        sigma_d = numpyro.sample(
            "sigma_d",
            dist.LogNormal(
                jnp.log(SIGMA_D_LOGNORMAL_MEDIAN),
                SIGMA_D_LOGNORMAL_SCALE,
            ).expand([n_chip]),
        )

    baselines = []
    if eigen_mode_count is None:
        contrast_matrices = []
        contrast_matrices_by_mode = None
    else:
        contrast_matrices = None
        contrast_matrices_by_mode = [[] for _ in range(eigen_mode_count)]
    noise_variances = []
    for chip_index in range(n_chip):
        profile_operator_kwargs = {}
        if profile_wavelengths[chip_index] is not None:
            profile_operator_kwargs["rest_wavelengths"] = profile_wavelengths[
                chip_index
            ]
        response_result = response_functions[chip_index](
            t0,
            alpha,
            log_vmrs["log_vmr_co"],
            log_vmrs["log_vmr_h2o"],
            log_vmrs["log_vmr_ch4"],
            log_vmrs["log_vmr_hf"],
            logg,
            log_p_cloud,
        )
        if eigen_mode_count is None:
            base_profile, contrast_profile = response_result
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
                **profile_operator_kwargs,
            )
            norm = normalization_factor[chip_index] * jnp.mean(baseline)
            baseline = baseline / norm
            contrast_matrix = contrast_matrix / norm
        else:
            base_profile, eigen_profiles = response_result
            chip_contrast_matrices = []
            baseline = None
            norm = None
            for mode_index in range(eigen_mode_count):
                mode_baseline, mode_contrast_matrix = linear_profile_operator_from_times(
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
                    eigen_profiles[mode_index],
                    weights=jnp.exp(log_w[chip_index]),
                    **profile_operator_kwargs,
                )
                if baseline is None:
                    baseline = mode_baseline
                    norm = normalization_factor[chip_index] * jnp.mean(baseline)
                    baseline = baseline / norm
                chip_contrast_matrices.append(mode_contrast_matrix / norm)
        baselines.append(baseline)
        if eigen_mode_count is None:
            contrast_matrices.append(contrast_matrix)
        else:
            for mode_index, mode_matrix in enumerate(chip_contrast_matrices):
                contrast_matrices_by_mode[mode_index].append(mode_matrix)
        noise_variances.append(
            diagonal_noise_variance(
                baseline.shape[0],
                sigma_d[chip_index],
                jitter=noise_jitter,
            )
        )

    baseline = jnp.concatenate(baselines, axis=0)
    if eigen_mode_count is None:
        contrast_matrix = jnp.concatenate(contrast_matrices, axis=0)
    else:
        contrast_matrix = jnp.concatenate(
            [
                jnp.concatenate(mode_matrices, axis=0)
                for mode_matrices in contrast_matrices_by_mode
            ],
            axis=1,
        )
    noise_variance = jnp.concatenate(noise_variances, axis=0)
    observed, baseline, contrast_matrix, noise_variance = _select_observation_rows(
        data,
        baseline,
        contrast_matrix,
        noise_variance,
        observation_indices,
    )

    covariance_factor = None
    if eigen_mode_count is None:
        numpyro.deterministic("sigma_b", sigma_log_p)
        ell_b = numpyro.deterministic("ell_b", jnp.asarray(fixed_ell_b))
        if pressure_gp_eigenvalues is not None:
            if pressure_gp_pixel_eigenvectors is None:
                raise ValueError(
                    "pressure_gp_pixel_eigenvectors must accompany "
                    "pressure_gp_eigenvalues."
                )
            eigenvalues = jnp.asarray(pressure_gp_eigenvalues)
            pixel_eigenvectors = jnp.asarray(
                pressure_gp_pixel_eigenvectors
            )
            eigen_scales = jnp.sqrt(
                sigma_log_p**2 * eigenvalues + jnp.asarray(gp_jitter)
            )
            covariance_factor = (
                contrast_matrix @ pixel_eigenvectors
            ) * eigen_scales[None, :]
        else:
            contrast_covariance = squared_exponential_covariance(
                distance_matrix,
                sigma_log_p,
                ell_b,
            )
            if zero_mean_pressure_map:
                map_factor = zero_mean_covariance_factor(
                    contrast_covariance,
                    jitter=gp_jitter,
                )
            else:
                contrast_covariance = add_diagonal_jitter(
                    contrast_covariance,
                    jitter=gp_jitter,
                )
                map_factor = jnp.linalg.cholesky(contrast_covariance)
    else:
        if fixed_sigma_eigen is not None:
            sigma_eigen = numpyro.deterministic(
                "sigma_eigen",
                jnp.asarray(fixed_sigma_eigen),
            )
        elif standardized_parameters:
            sigma_eigen_raw = numpyro.sample(
                "sigma_eigen_raw",
                dist.Normal(0.0, 1.0).expand([eigen_mode_count]),
            )
            sigma_eigen = numpyro.deterministic(
                "sigma_eigen",
                jnp.exp(
                    jnp.log(eigen_sigma_center)
                    + eigen_sigma_log_raw_scale * sigma_eigen_raw
                ),
            )
        else:
            sigma_eigen = numpyro.sample(
                "sigma_eigen",
                dist.HalfNormal(eigen_sigma_scale).expand([eigen_mode_count]),
            )
        numpyro.deterministic("sigma_b", sigma_eigen[0])
        ell_eigen = numpyro.deterministic(
            "ell_eigen",
            jnp.full((eigen_mode_count,), eigen_fixed_ell),
        )
        mode_factors = []
        eigen_gp_jitter = jnp.maximum(jnp.asarray(gp_jitter), jnp.asarray(5.0e-6))
        for mode_index in range(eigen_mode_count):
            unit_covariance = squared_exponential_covariance(
                distance_matrix,
                jnp.asarray(1.0),
                ell_eigen[mode_index],
            )
            if zero_mean_pressure_map:
                unit_factor = zero_mean_covariance_factor(
                    unit_covariance,
                    jitter=eigen_gp_jitter,
                )
            else:
                unit_covariance = add_diagonal_jitter(
                    unit_covariance,
                    jitter=eigen_gp_jitter,
                )
                unit_factor = jnp.linalg.cholesky(unit_covariance)
            mode_factors.append(sigma_eigen[mode_index] * unit_factor)
        map_factor = jsp_linalg.block_diag(*mode_factors)
    if covariance_factor is None:
        covariance_factor = contrast_matrix @ map_factor
    numpyro.sample(
        "obs",
        dist.LowRankMultivariateNormal(
            loc=baseline,
            cov_factor=covariance_factor,
            cov_diag=noise_variance,
        ),
        obs=observed,
    )


def finite(value):
    """Return whether a JAX array is finite."""

    return bool(jnp.all(jnp.isfinite(value)))


def _median_or_default(samples, name, default):
    """Return a posterior median initial value when present."""

    if samples is None or name not in samples:
        return default
    value = np.median(np.asarray(samples[name]), axis=0)
    if np.shape(value) != np.shape(default):
        return default
    return value


def _zero_mean_log_w(log_w):
    """Return chip-wise zero-mean phase weights for initialization."""

    values = jnp.asarray(log_w)
    return values - jnp.mean(values, axis=1, keepdims=True)


def load_initial_values(args, chip_count, phase_count):
    """Build constrained initial values for the NUTS run."""

    previous = None
    if args.init_from is not None and Path(args.init_from).exists():
        previous = dict(np.load(args.init_from, allow_pickle=False))

    def log_vmr_initial(name, base_value, default):
        if previous is not None and name in previous:
            return float(_median_or_default(previous, name, default))
        if previous is not None and "zeta_vmr" in previous:
            previous_zeta = float(_median_or_default(previous, "zeta_vmr", 0.0))
            return float(base_value + previous_zeta)
        return float(default)

    values = {
        "T0": float(_median_or_default(previous, "T0", args.init_t0)),
        "alpha": float(_median_or_default(previous, "alpha", args.init_alpha)),
        "logg": float(_median_or_default(previous, "logg", args.init_logg)),
        "log_vmr_co": log_vmr_initial(
            "log_vmr_co",
            YAMA_L16B_EXOMOL_ATMOSPHERE.log_vmr_co,
            args.init_log_vmr_co,
        ),
        "log_vmr_h2o": log_vmr_initial(
            "log_vmr_h2o",
            YAMA_L16B_EXOMOL_ATMOSPHERE.log_vmr_h2o,
            args.init_log_vmr_h2o,
        ),
        "log_vmr_ch4": log_vmr_initial(
            "log_vmr_ch4",
            YAMA_L16B_EXOMOL_ATMOSPHERE.log_vmr_ch4,
            args.init_log_vmr_ch4,
        ),
        "log_vmr_hf": log_vmr_initial(
            "log_vmr_hf",
            YAMA_L16B_EXOMOL_ATMOSPHERE.log_vmr_hf,
            args.init_log_vmr_hf,
        ),
        "log_p_cloud": float(
            _median_or_default(previous, "log_p_cloud", args.init_log_p_cloud)
        ),
        "sigma_log_p": float(
            _median_or_default(previous, "sigma_log_p", args.init_sigma_log_p)
        ),
        "A": jnp.asarray(
            _median_or_default(previous, "A", np.full((chip_count,), 1.05))
        ),
        "log_w": jnp.asarray(
            _median_or_default(previous, "log_w", np.zeros((chip_count, phase_count)))
        ),
        "sigma_d": jnp.asarray(
            _median_or_default(previous, "sigma_d", np.full((chip_count,), 0.03))
        ),
    }
    return values


def build_parameter_reparameterization(args, physical_init_values):
    """Build centers and scales for standardized atmospheric coordinates."""

    centers = {
        "T0": jnp.asarray(physical_init_values["T0"]),
        "alpha": jnp.asarray(physical_init_values["alpha"]),
        "logg": jnp.asarray(physical_init_values["logg"]),
        "log_p_cloud": jnp.asarray(physical_init_values["log_p_cloud"]),
        "sigma_log_p": jnp.asarray(physical_init_values["sigma_log_p"]),
    }
    centers.update(
        {name: jnp.asarray(physical_init_values[name]) for name in LOG_VMR_NAMES}
    )
    scales = {
        "T0": jnp.asarray(args.t0_raw_scale),
        "alpha": jnp.asarray(args.alpha_raw_scale),
        "logg": jnp.asarray(args.logg_raw_scale),
        "log_p_cloud": jnp.asarray(args.log_p_cloud_raw_scale),
        "sigma_log_p": jnp.asarray(args.sigma_log_p_log_raw_scale),
        "zeta_vmr_per_t0": jnp.asarray(args.zeta_vmr_per_t0),
        "log_p_cloud_per_alpha": jnp.asarray(args.log_p_cloud_per_alpha),
        "rotated_atmosphere": bool(args.rotated_atmosphere_parameters),
    }
    scales.update({name: jnp.asarray(args.log_vmr_raw_scale) for name in LOG_VMR_NAMES})
    return centers, scales


def build_fixed_nuisance_values(args, physical_init_values):
    """Select nuisance values held fixed by the requested diagnostic."""

    fix_a = bool(getattr(args, "fix_a", False))
    fix_sigma_d = bool(getattr(args, "fix_sigma_d", False))
    partial_fix_requested = bool(args.fix_log_w or fix_a or fix_sigma_d)
    if args.fix_nuisance and partial_fix_requested:
        raise ValueError(
            "--fix-nuisance and the individual --fix-a, --fix-log-w, or "
            "--fix-sigma-d options are mutually exclusive."
        )
    if not args.fix_nuisance and not partial_fix_requested:
        return None
    fixed_values = {}
    if args.fix_nuisance or fix_a:
        fixed_values["A"] = physical_init_values["A"]
    if args.fix_nuisance or args.fix_log_w:
        fixed_log_w = jnp.asarray(physical_init_values["log_w"])
        if args.zero_mean_log_w:
            fixed_log_w = _zero_mean_log_w(fixed_log_w)
        fixed_values["log_w"] = fixed_log_w
    if args.fix_nuisance or fix_sigma_d:
        fixed_values["sigma_d"] = physical_init_values["sigma_d"]
    return fixed_values


def build_sampling_initial_values(
    args,
    physical_init_values,
    fixed_nuisance_values,
    atmosphere_rotation=None,
    joint_atmosphere_a_sigma_d_rotation=None,
):
    """Build initial values for sample sites seen by NUTS."""

    if args.rotated_atmosphere_parameters:
        raise ValueError("Rotated-atmosphere initialization is not supported.")
    direct_sigma_log_p = bool(
        getattr(args, "direct_sigma_log_p", False)
    )
    fixed_sigma_log_p = getattr(args, "fixed_sigma_log_p", None)
    joint_coordinates = bool(
        getattr(args, "joint_atmosphere_a_sigma_d", False)
    )
    if direct_sigma_log_p and not args.gaussianized_atmosphere:
        raise ValueError(
            "Direct sigma_log_p initialization requires a Gaussianized "
            "atmosphere."
        )
    if direct_sigma_log_p and fixed_sigma_log_p is not None:
        raise ValueError(
            "Direct sigma_log_p initialization cannot also fix sigma_log_p."
        )
    if direct_sigma_log_p and joint_coordinates:
        raise ValueError(
            "Direct sigma_log_p initialization cannot use joint "
            "atmosphere/A/sigma_d coordinates."
        )

    def add_nuisance_initial_values(init_values):
        fixed_names = set(
            () if fixed_nuisance_values is None else fixed_nuisance_values
        )
        if not joint_coordinates and "A" not in fixed_names:
            init_values["A"] = physical_init_values["A"]
        if "log_w" not in fixed_names:
            if args.zero_sum_log_w_basis:
                log_w_site = "log_w_base"
                log_w_value = zero_sum_log_w_base(
                    physical_init_values["log_w"]
                )
            elif args.zero_mean_log_w:
                log_w_site = "log_w_raw"
                log_w_value = _zero_mean_log_w(
                    physical_init_values["log_w"]
                )
            else:
                log_w_site = "log_w"
                log_w_value = physical_init_values["log_w"]
            init_values[log_w_site] = log_w_value
        if not joint_coordinates and "sigma_d" not in fixed_names:
            init_values["sigma_d"] = physical_init_values["sigma_d"]
        return init_values

    if args.gaussianized_atmosphere:
        direct_sigma_initial_value = None
        if direct_sigma_log_p:
            direct_sigma_initial_value = np.asarray(
                physical_init_values["sigma_log_p"],
                dtype=float,
            )
            if (
                direct_sigma_initial_value.shape != ()
                or not np.isfinite(direct_sigma_initial_value)
                or direct_sigma_initial_value <= 0.0
            ):
                raise ValueError(
                    "Direct sigma_log_p initialization must be a finite, "
                    "positive scalar."
                )
        if atmosphere_rotation is None:
            atmosphere_rotation = validate_atmosphere_rotation(
                args.atmosphere_rotation_matrix,
                dimension=len(
                    active_gaussianized_atmosphere_names(
                        fixed_sigma_log_p,
                        direct_sigma_log_p,
                    )
                ),
            )
        atmosphere_gaussianized = physical_atmosphere_to_gaussianized(
            physical_init_values,
            t0_bounds=(args.t0_min, args.t0_max),
            alpha_bounds=(args.alpha_min, args.alpha_max),
            log_vmr_bounds={
                "log_vmr_co": (args.log_vmr_co_min, args.log_vmr_co_max),
                "log_vmr_h2o": (args.log_vmr_h2o_min, args.log_vmr_h2o_max),
                "log_vmr_ch4": (args.log_vmr_ch4_min, args.log_vmr_ch4_max),
                "log_vmr_hf": (args.log_vmr_hf_min, args.log_vmr_hf_max),
            },
            log_p_cloud_bounds=(args.log_p_cloud_min, args.log_p_cloud_max),
            sigma_log_p_scale=args.sigma_log_p_scale,
        )
        if fixed_sigma_log_p is not None or direct_sigma_log_p:
            atmosphere_gaussianized = atmosphere_gaussianized[:-1]
        if not np.all(np.isfinite(np.asarray(atmosphere_gaussianized))):
            raise ValueError(
                "Gaussianized atmosphere initialization is non-finite. "
                "All bounded initial values must be strictly inside their "
                "prior bounds and sigma_log_p must be positive."
            )
        if joint_coordinates:
            if fixed_sigma_log_p is not None:
                raise ValueError(
                    "Joint atmosphere/A/sigma_d initialization requires a "
                    "sampled sigma_log_p."
                )
            if fixed_nuisance_values is not None and (
                "A" in fixed_nuisance_values
                or "sigma_d" in fixed_nuisance_values
            ):
                raise ValueError(
                    "Joint atmosphere/A/sigma_d initialization cannot fix A "
                    "or sigma_d."
                )
            a_gaussianized, sigma_d_gaussianized = (
                physical_nuisance_to_gaussianized(
                    physical_init_values["A"],
                    physical_init_values["sigma_d"],
                )
            )
            joint_gaussianized = jnp.concatenate(
                [
                    atmosphere_gaussianized,
                    jnp.ravel(a_gaussianized),
                    jnp.ravel(sigma_d_gaussianized),
                ]
            )
            if not np.all(np.isfinite(np.asarray(joint_gaussianized))):
                raise ValueError(
                    "Joint atmosphere/A/sigma_d initialization is non-finite. "
                    "A must lie strictly inside (1.0, 1.2), and sigma_d must "
                    "be finite and positive."
                )
            chip_count = int(np.size(physical_init_values["A"]))
            if joint_atmosphere_a_sigma_d_rotation is None:
                joint_atmosphere_a_sigma_d_rotation = (
                    validate_joint_atmosphere_a_sigma_d_rotation(
                        getattr(
                            args,
                            "joint_atmosphere_a_sigma_d_rotation_matrix",
                            None,
                        ),
                        chip_count,
                    )
                )
            init_values = {
                JOINT_ATMOSPHERE_A_SIGMA_D_SITE:
                joint_atmosphere_a_sigma_d_rotation.T @ joint_gaussianized
            }
            return add_nuisance_initial_values(init_values)
        init_values = {
            "atmosphere_rotated": atmosphere_rotation.T
            @ atmosphere_gaussianized
        }
        if direct_sigma_log_p:
            init_values["sigma_log_p"] = jnp.asarray(
                direct_sigma_initial_value
            )
        return add_nuisance_initial_values(init_values)

    if args.standardized_parameters:
        init_values = {
            "T0_raw": jnp.asarray(0.0),
            "alpha_raw": jnp.asarray(0.0),
            "log_p_cloud_raw": jnp.asarray(0.0),
        }
        if (
            args.eigen_basis is None
            and getattr(args, "fixed_sigma_log_p", None) is None
        ):
            init_values["sigma_log_p_raw"] = jnp.asarray(0.0)
        if not args.fix_logg:
            init_values["logg_raw"] = jnp.asarray(0.0)
        init_values.update({f"{name}_raw": jnp.asarray(0.0) for name in LOG_VMR_NAMES})
        add_nuisance_initial_values(init_values)
        if args.eigen_basis is not None and args.fixed_sigma_eigen is None:
            init_values["sigma_eigen_raw"] = jnp.zeros((args.eigen_mode_count,))
        return init_values

    fixed_names = set(
        () if fixed_nuisance_values is None else fixed_nuisance_values
    )
    init_values = {
        name: value
        for name, value in physical_init_values.items()
        if name not in fixed_names
    }
    if getattr(args, "fixed_sigma_log_p", None) is not None:
        init_values.pop("sigma_log_p", None)
    if "log_w" in init_values:
        if args.zero_sum_log_w_basis:
            init_values["log_w_base"] = zero_sum_log_w_base(
                init_values.pop("log_w")
            )
        elif args.zero_mean_log_w:
            init_values["log_w_raw"] = _zero_mean_log_w(
                init_values.pop("log_w")
            )
    if args.eigen_basis is not None and args.fixed_sigma_eigen is None:
        init_values["sigma_eigen"] = jnp.full(
            (args.eigen_mode_count,),
            args.init_sigma_eigen,
        )
    return init_values


def maybe_find_map_init(args, model, init_values):
    """Optionally refine initial values with numpyro-inferutils."""

    if not args.map_init:
        return init_values
    from numpyro_inferutils import find_map_svi

    return find_map_svi(
        model,
        step_size=args.map_init_step_size,
        num_steps=args.map_init_steps,
        rng_key=jax.random.PRNGKey(args.seed + 1000),
        p_initial=init_values,
        progress_bar=True,
    )


def build_dense_mass_specification(args):
    """Return the NumPyro mass structure and serializable metadata."""

    dense_atmosphere_a_sigma_d_mass = getattr(
        args,
        "dense_atmosphere_a_sigma_d_mass",
        False,
    )
    direct_sigma_log_p = bool(
        getattr(args, "direct_sigma_log_p", False)
    )
    selected_modes = sum(
        bool(value)
        for value in (
            args.dense_mass,
            args.dense_atmosphere_mass,
            dense_atmosphere_a_sigma_d_mass,
        )
    )
    if selected_modes > 1:
        raise ValueError(
            "--dense-mass, --dense-atmosphere-mass, and "
            "--dense-atmosphere-a-sigma-d-mass are mutually exclusive."
        )
    if args.dense_mass:
        return True, "full", ()
    if dense_atmosphere_a_sigma_d_mass:
        if not args.gaussianized_atmosphere:
            raise ValueError(
                "--dense-atmosphere-a-sigma-d-mass currently requires "
                "--gaussianized-atmosphere."
            )
        fixed_joint_site = any(
            bool(getattr(args, name, False))
            for name in ("fix_nuisance", "fix_a", "fix_sigma_d")
        )
        if fixed_joint_site:
            raise ValueError(
                "--dense-atmosphere-a-sigma-d-mass cannot be combined with "
                "--fix-nuisance, --fix-a, or --fix-sigma-d because the dense "
                "block requires both A and sigma_d sample sites."
            )
        sites = (
            (
                "atmosphere_rotated",
                "sigma_log_p",
                "A",
                "sigma_d",
            )
            if direct_sigma_log_p
            else ("atmosphere_rotated", "A", "sigma_d")
        )
        return [sites], "atmosphere_a_sigma_d_block", sites
    if not args.dense_atmosphere_mass:
        return False, "diagonal", ()
    if args.gaussianized_atmosphere:
        sites = (
            ("atmosphere_rotated", "sigma_log_p")
            if direct_sigma_log_p
            else ("atmosphere_rotated",)
        )
        return [sites], "atmosphere_block", sites
    suffix = "_raw" if args.standardized_parameters else ""
    sites = [
        f"T0{suffix}",
        f"alpha{suffix}",
        f"log_vmr_co{suffix}",
        f"log_vmr_h2o{suffix}",
        f"log_vmr_ch4{suffix}",
        f"log_vmr_hf{suffix}",
        f"log_p_cloud{suffix}",
    ]
    if not args.fix_logg:
        sites.append(f"logg{suffix}")
    if args.eigen_mode_count in (None, 0):
        if args.fixed_sigma_log_p is None:
            sites.append(f"sigma_log_p{suffix}")
    elif args.fixed_sigma_eigen is None:
        sites.append(f"sigma_eigen{suffix}")
    return [tuple(sites)], "atmosphere_block", tuple(sites)


def build_initial_inverse_mass_matrix(
    args,
    dense_mass,
    dense_mass_sites,
    init_values,
):
    """Build a validated structured diagonal metric for NumPyro."""

    values = getattr(args, "initial_inverse_mass_diagonal", None)
    if values is None:
        if getattr(args, "initial_inverse_mass_sites", None) is not None:
            raise ValueError(
                "--initial-inverse-mass-sites requires "
                "--initial-inverse-mass-diagonal."
            )
        return None, "identity", (), ()
    requested_sites = getattr(args, "initial_inverse_mass_sites", None)
    sites = tuple(
        dense_mass_sites if requested_sites is None else requested_sites
    )
    if not sites:
        raise ValueError(
            "--initial-inverse-mass-diagonal requires "
            "--initial-inverse-mass-sites or a structured mass block."
        )
    if len(set(sites)) != len(sites):
        raise ValueError("Initial inverse-mass site names must be unique.")
    if dense_mass not in (False, []) and dense_mass != [sites]:
        raise ValueError(
            "An explicit diagonal metric cannot overlap a different dense "
            "mass block."
        )
    missing_sites = [name for name in sites if name not in init_values]
    if missing_sites:
        raise ValueError(
            "Initial inverse-mass sites are absent from the NUTS initial "
            f"values: {missing_sites}."
        )
    unscaled_sites = sorted(set(init_values) - set(sites))
    if unscaled_sites:
        raise ValueError(
            "--initial-inverse-mass-sites must cover every active sample "
            f"site; missing {unscaled_sites}."
        )
    site_sizes = tuple(int(np.size(init_values[name])) for name in sites)
    diagonal = np.asarray(values, dtype=float)
    expected_size = sum(site_sizes)
    if diagonal.ndim != 1 or diagonal.size != expected_size:
        raise ValueError(
            "--initial-inverse-mass-diagonal must contain exactly "
            f"{expected_size} values for sites {sites} with sizes "
            f"{site_sizes}; got {diagonal.size}."
        )
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError(
            "--initial-inverse-mass-diagonal values must all be finite "
            "and positive."
        )
    inverse_mass_matrix = {sites: jnp.asarray(diagonal)}
    return (
        inverse_mass_matrix,
        "structured_diagonal",
        sites,
        site_sizes,
    )


def build_nuts_max_tree_depth(args):
    """Return NumPyro's warmup/sampling tree-depth specification."""

    sampling_depth = int(args.max_tree_depth)
    warmup_depth = (
        sampling_depth
        if args.warmup_max_tree_depth is None
        else int(args.warmup_max_tree_depth)
    )
    if sampling_depth < 1:
        raise ValueError("--max-tree-depth must be at least 1.")
    if warmup_depth < 1:
        raise ValueError("--warmup-max-tree-depth must be at least 1.")
    specification = (
        sampling_depth
        if warmup_depth == sampling_depth
        else (warmup_depth, sampling_depth)
    )
    return specification, warmup_depth, sampling_depth


def main():
    """Run the on-the-fly autodiff retrieval."""

    args = parse_args()
    if args.fix_nuisance and (
        args.fix_a or args.fix_log_w or args.fix_sigma_d
    ):
        raise ValueError(
            "--fix-nuisance and the individual --fix-a, --fix-log-w, or "
            "--fix-sigma-d options are mutually exclusive."
        )
    if args.direct_sigma_log_p:
        if not args.gaussianized_atmosphere:
            raise ValueError(
                "--direct-sigma-log-p requires --gaussianized-atmosphere."
            )
        if args.fixed_sigma_log_p is not None:
            raise ValueError(
                "--direct-sigma-log-p cannot be combined with "
                "--fixed-sigma-log-p."
            )
        if args.joint_atmosphere_a_sigma_d:
            raise ValueError(
                "--direct-sigma-log-p cannot be combined with "
                "--joint-atmosphere-a-sigma-d."
            )
    if args.joint_atmosphere_a_sigma_d and not args.gaussianized_atmosphere:
        raise ValueError(
            "--joint-atmosphere-a-sigma-d requires "
            "--gaussianized-atmosphere."
        )
    if (
        args.joint_atmosphere_a_sigma_d_rotation_matrix is not None
        and not args.joint_atmosphere_a_sigma_d
    ):
        raise ValueError(
            "--joint-atmosphere-a-sigma-d-rotation-matrix requires "
            "--joint-atmosphere-a-sigma-d."
        )
    if args.joint_atmosphere_a_sigma_d:
        if args.atmosphere_rotation_matrix is not None:
            raise ValueError(
                "The joint atmosphere/A/sigma_d rotation subsumes "
                "--atmosphere-rotation-matrix; do not supply both."
            )
        if args.fixed_sigma_log_p is not None:
            raise ValueError(
                "--joint-atmosphere-a-sigma-d requires a sampled sigma_log_p."
            )
        if args.fix_nuisance or args.fix_a or args.fix_sigma_d:
            raise ValueError(
                "--joint-atmosphere-a-sigma-d cannot be combined with "
                "--fix-nuisance, --fix-a, or --fix-sigma-d. --fix-log-w "
                "remains supported."
            )
        if args.dense_atmosphere_mass or args.dense_atmosphere_a_sigma_d_mass:
            raise ValueError(
                "The legacy structured dense-mass flags do not name the new "
                "joint sample site. Use diagonal adaptation for this pilot."
            )
    if args.gaussianized_atmosphere and (
        args.standardized_parameters or args.rotated_atmosphere_parameters
    ):
        raise ValueError(
            "--gaussianized-atmosphere is mutually exclusive with "
            "--standardized-parameters and --rotated-atmosphere-parameters."
        )
    if args.gaussianized_atmosphere and not args.fix_logg:
        raise ValueError("--gaussianized-atmosphere currently requires --fix-logg.")
    if args.gaussianized_atmosphere and args.eigen_basis is not None:
        raise ValueError(
            "--gaussianized-atmosphere currently supports only the pressure-map "
            "model without --eigen-basis."
        )
    if args.fixed_sigma_log_p is not None:
        if not np.isfinite(args.fixed_sigma_log_p) or args.fixed_sigma_log_p <= 0.0:
            raise ValueError("--fixed-sigma-log-p must be finite and positive.")
        if args.eigen_basis is not None:
            raise ValueError(
                "--fixed-sigma-log-p is only valid for the pressure-map model "
                "without --eigen-basis."
            )
    if not np.isfinite(args.sigma_log_p_scale) or args.sigma_log_p_scale <= 0.0:
        raise ValueError("--sigma-log-p-scale must be finite and positive.")
    if args.pressure_gp_factorization == "fixed_eigen":
        if args.eigen_basis is not None:
            raise ValueError(
                "--pressure-gp-factorization=fixed_eigen is only valid for "
                "the pressure-map model without --eigen-basis."
            )
        if not args.zero_mean_pressure_map:
            raise ValueError(
                "--pressure-gp-factorization=fixed_eigen requires "
                "--zero-mean-pressure-map."
            )
    if (
        args.atmosphere_rotation_matrix is not None
        and not args.gaussianized_atmosphere
    ):
        raise ValueError(
            "--atmosphere-rotation-matrix requires --gaussianized-atmosphere."
        )
    if args.zero_sum_log_w_basis:
        args.zero_mean_log_w = True
    atmosphere_rotation = validate_atmosphere_rotation(
        args.atmosphere_rotation_matrix
        if args.gaussianized_atmosphere else None,
        dimension=(
            len(
                active_gaussianized_atmosphere_names(
                    args.fixed_sigma_log_p,
                    args.direct_sigma_log_p,
                )
            )
            if args.gaussianized_atmosphere
            else len(GAUSSIANIZED_ATMOSPHERE_NAMES)
        ),
    )
    joint_atmosphere_a_sigma_d_rotation = (
        validate_joint_atmosphere_a_sigma_d_rotation(
            args.joint_atmosphere_a_sigma_d_rotation_matrix,
            len(args.chip_indices),
        )
        if args.joint_atmosphere_a_sigma_d
        else None
    )
    if args.rotated_atmosphere_parameters:
        args.standardized_parameters = True
    eigen_basis = load_eigen_basis(args.eigen_basis, args.eigen_mode_count)
    if eigen_basis is not None:
        args.eigen_mode_count = eigen_basis["mode_count"]
        if (
            args.fixed_sigma_eigen is not None
            and len(args.fixed_sigma_eigen) != args.eigen_mode_count
        ):
            raise ValueError(
                "--fixed-sigma-eigen length must match --eigen-mode-count "
                f"({len(args.fixed_sigma_eigen)} != {args.eigen_mode_count})."
            )
    if args.eigen_mode_count is None:
        args.eigen_mode_count = 0
    dense_mass, dense_mass_mode, dense_mass_sites = (
        build_dense_mass_specification(args)
    )
    (
        nuts_max_tree_depth,
        warmup_max_tree_depth,
        sampling_max_tree_depth,
    ) = build_nuts_max_tree_depth(args)
    eigen_fixed_ell = args.fixed_ell_b if args.eigen_fixed_ell is None else args.eigen_fixed_ell
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chip_data_list, native_wavelengths = build_chip_data(
        args,
        return_native_wavelengths=True,
    )
    geometry = build_luhman16b_geometry(nside=args.nside)
    pressure_gp_eigendecomposition = None
    pressure_gp_eigendecomposition_seconds = 0.0
    if args.pressure_gp_factorization == "fixed_eigen":
        gp_eigen_start = time.time()
        pressure_gp_eigendecomposition = (
            build_fixed_pressure_gp_eigendecomposition(
                geometry.distance_matrix,
                args.fixed_ell_b,
                theta=geometry.theta,
                phi=geometry.phi,
            )
        )
        pressure_gp_eigendecomposition_seconds = (
            time.time() - gp_eigen_start
        )
    flux_shapes = {chip.flux.shape for chip in chip_data_list}
    if len(flux_shapes) != 1:
        raise ValueError(
            "All selected chips must have the same flux shape for joint retrieval; "
            f"got {sorted(flux_shapes)}."
        )
    data = jnp.asarray(np.stack([chip.flux for chip in chip_data_list], axis=0))
    observation_masks = build_observation_masks(
        chip_data_list,
        mask_zero_flux=args.mask_zero_flux,
    )
    observation_mask = np.concatenate(
        [mask.reshape(-1) for mask in observation_masks],
        axis=0,
    )
    observation_indices = np.flatnonzero(observation_mask)
    observation_valid_count = int(observation_indices.size)
    observation_excluded_count = int(observation_mask.size - observation_valid_count)
    observation_mask_rule = (
        "finite_and_nonzero_flux" if args.mask_zero_flux else "all_observations"
    )
    wavelengths = [jnp.asarray(chip.wavelengths) for chip in chip_data_list]
    profile_wavelengths = build_doppler_profile_wavelengths(
        native_wavelengths,
        abs(args.fixed_v),
    )
    obs_times = jnp.asarray(chip_data_list[0].obs_times)

    setup_start = time.time()
    response_functions = build_response_functions(
        args,
        chip_data_list,
        eigen_basis=eigen_basis,
        profile_wavelengths=profile_wavelengths,
    )
    setup_seconds = time.time() - setup_start

    preflight_t0 = jnp.asarray(args.init_t0)
    preflight_alpha = jnp.asarray(args.init_alpha)
    preflight_log_vmr_co = jnp.asarray(args.init_log_vmr_co)
    preflight_log_vmr_h2o = jnp.asarray(args.init_log_vmr_h2o)
    preflight_log_vmr_ch4 = jnp.asarray(args.init_log_vmr_ch4)
    preflight_log_vmr_hf = jnp.asarray(args.init_log_vmr_hf)
    preflight_logg = jnp.asarray(args.init_logg)
    preflight_log_p = jnp.asarray(args.init_log_p_cloud)
    timing = {
        "setup_seconds": setup_seconds,
        "pressure_gp_eigendecomposition_seconds": (
            pressure_gp_eigendecomposition_seconds
        ),
    }
    if args.preflight_autodiff:
        for chip_position, chip_data in enumerate(chip_data_list):
            response = jax.jit(response_functions[chip_position])
            start = time.time()
            spectrum, derivative = response(
                preflight_t0,
                preflight_alpha,
                preflight_log_vmr_co,
                preflight_log_vmr_h2o,
                preflight_log_vmr_ch4,
                preflight_log_vmr_hf,
                preflight_logg,
                preflight_log_p,
            )
            spectrum.block_until_ready()
            derivative.block_until_ready()
            timing[f"chip{chip_data.chip_index}_response_compile_seconds"] = (
                time.time() - start
            )
            start = time.time()
            spectrum_second, derivative_second = response(
                preflight_t0,
                preflight_alpha,
                preflight_log_vmr_co,
                preflight_log_vmr_h2o,
                preflight_log_vmr_ch4,
                preflight_log_vmr_hf,
                preflight_logg,
                preflight_log_p,
            )
            spectrum_second.block_until_ready()
            derivative_second.block_until_ready()
            timing[f"chip{chip_data.chip_index}_response_second_seconds"] = (
                time.time() - start
            )
            timing[f"chip{chip_data.chip_index}_spectrum_all_finite"] = finite(spectrum)
            timing[f"chip{chip_data.chip_index}_derivative_all_finite"] = finite(
                derivative
            )
            timing[f"chip{chip_data.chip_index}_spectrum_rms"] = float(
                jnp.sqrt(jnp.mean(spectrum * spectrum))
            )
            timing[f"chip{chip_data.chip_index}_derivative_rms"] = float(
                jnp.sqrt(jnp.mean(derivative * derivative))
            )

    physical_init_values = load_initial_values(
        args,
        len(chip_data_list),
        len(obs_times),
    )
    if args.manual_atmosphere_init:
        physical_init_values.update(
            {
                "T0": args.init_t0,
                "alpha": args.init_alpha,
                "logg": args.init_logg,
                "log_vmr_co": args.init_log_vmr_co,
                "log_vmr_h2o": args.init_log_vmr_h2o,
                "log_vmr_ch4": args.init_log_vmr_ch4,
                "log_vmr_hf": args.init_log_vmr_hf,
                "log_p_cloud": args.init_log_p_cloud,
                "sigma_log_p": args.init_sigma_log_p,
            }
        )
    if args.fixed_sigma_log_p is not None:
        physical_init_values["sigma_log_p"] = args.fixed_sigma_log_p
    parameter_centers, parameter_scales = build_parameter_reparameterization(
        args,
        physical_init_values,
    )
    fixed_nuisance_values = build_fixed_nuisance_values(
        args,
        physical_init_values,
    )
    init_values = build_sampling_initial_values(
        args,
        physical_init_values,
        fixed_nuisance_values,
        atmosphere_rotation,
        joint_atmosphere_a_sigma_d_rotation,
    )

    def model():
        return on_the_fly_pressure_model(
            data=data,
            wavelengths=wavelengths,
            obs_times=obs_times,
            theta=geometry.theta,
            phi=geometry.phi,
            distance_matrix=geometry.distance_matrix,
            response_functions=response_functions,
            fixed_period=args.fixed_period,
            fixed_cosi=args.fixed_cosi,
            fixed_v=args.fixed_v,
            fixed_q1=args.fixed_q1,
            fixed_q2=args.fixed_q2,
            fix_logg=args.fix_logg,
            fixed_logg=args.init_logg,
            logg_prior_mean=args.logg_prior_mean,
            logg_prior_sigma=args.logg_prior_sigma,
            logg_bounds=(args.logg_min, args.logg_max),
            t0_bounds=(args.t0_min, args.t0_max),
            alpha_bounds=(args.alpha_min, args.alpha_max),
            log_vmr_bounds={
                "log_vmr_co": (args.log_vmr_co_min, args.log_vmr_co_max),
                "log_vmr_h2o": (args.log_vmr_h2o_min, args.log_vmr_h2o_max),
                "log_vmr_ch4": (args.log_vmr_ch4_min, args.log_vmr_ch4_max),
                "log_vmr_hf": (args.log_vmr_hf_min, args.log_vmr_hf_max),
            },
            log_p_cloud_bounds=(args.log_p_cloud_min, args.log_p_cloud_max),
            sigma_log_p_scale=args.sigma_log_p_scale,
            standardized_parameters=args.standardized_parameters,
            gaussianized_atmosphere=args.gaussianized_atmosphere,
            atmosphere_rotation=atmosphere_rotation,
            parameter_centers=parameter_centers,
            parameter_scales=parameter_scales,
            fixed_ell_b=args.fixed_ell_b,
            zero_mean_pressure_map=args.zero_mean_pressure_map,
            log_w_scale=args.log_w_scale,
            zero_mean_log_w=args.zero_mean_log_w,
            zero_sum_log_w_basis=args.zero_sum_log_w_basis,
            eigen_mode_count=(None if eigen_basis is None else args.eigen_mode_count),
            eigen_sigma_scale=args.eigen_sigma_scale,
            eigen_sigma_center=jnp.asarray(args.init_sigma_eigen),
            eigen_sigma_log_raw_scale=jnp.asarray(args.sigma_eigen_log_raw_scale),
            fixed_sigma_eigen=(
                None
                if args.fixed_sigma_eigen is None
                else jnp.asarray(args.fixed_sigma_eigen)
            ),
            eigen_fixed_ell=eigen_fixed_ell,
            fixed_nuisance_values=fixed_nuisance_values,
            gp_jitter=args.gp_jitter,
            noise_jitter=args.noise_jitter,
            fixed_sigma_log_p=args.fixed_sigma_log_p,
            direct_sigma_log_p=args.direct_sigma_log_p,
            pressure_gp_eigenvalues=(
                None
                if pressure_gp_eigendecomposition is None
                else pressure_gp_eigendecomposition["eigenvalues"]
            ),
            pressure_gp_pixel_eigenvectors=(
                None
                if pressure_gp_eigendecomposition is None
                else pressure_gp_eigendecomposition[
                    "pixel_eigenvectors"
                ]
            ),
            joint_atmosphere_a_sigma_d=(
                args.joint_atmosphere_a_sigma_d
            ),
            joint_atmosphere_a_sigma_d_rotation=(
                joint_atmosphere_a_sigma_d_rotation
            ),
            profile_wavelengths=profile_wavelengths,
            observation_indices=(
                observation_indices if args.mask_zero_flux else None
            ),
        )

    init_values = maybe_find_map_init(args, model, init_values)
    (
        initial_inverse_mass_matrix,
        initial_inverse_mass_matrix_mode,
        initial_inverse_mass_matrix_sites,
        initial_inverse_mass_matrix_site_sizes,
    ) = build_initial_inverse_mass_matrix(
        args,
        dense_mass,
        dense_mass_sites,
        init_values,
    )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=args.target_accept_prob,
        dense_mass=dense_mass,
        inverse_mass_matrix=initial_inverse_mass_matrix,
        adapt_mass_matrix=args.adapt_mass_matrix,
        max_tree_depth=nuts_max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    run_start = time.time()
    mcmc.run(
        jax.random.PRNGKey(args.seed),
        extra_fields=("diverging", "accept_prob", "num_steps", "potential_energy"),
    )
    run_seconds = time.time() - run_start
    if args.print_summary and args.num_samples >= 4:
        mcmc.print_summary()
    elif args.print_summary:
        print("Skipping MCMC summary because num_samples < 4.")

    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()
    final_step_size = float(
        np.asarray(mcmc.last_state.adapt_state.step_size)
    )
    maximum_tree_num_steps = 2**sampling_max_tree_depth - 1
    sampled_num_steps = np.asarray(
        extra_fields.get("num_steps", []),
        dtype=int,
    )
    tree_depth_cap_count = int(
        np.sum(sampled_num_steps == maximum_tree_num_steps)
    )
    tree_depth_cap_fraction = (
        float(tree_depth_cap_count / sampled_num_steps.size)
        if sampled_num_steps.size
        else None
    )
    output_path = out_dir / "samples.npz"
    save_data = {
        name: np.asarray(value)
        for name, value in samples.items()
    }
    save_data.update(
        {
            f"extra_{name}": np.asarray(value)
            for name, value in extra_fields.items()
        }
    )
    save_data.update(
        {
            "run_label": np.asarray(args.run_label),
            "chip_indices": np.asarray(args.chip_indices),
            "obs_times": np.asarray(obs_times),
            "nside": np.asarray(args.nside),
            "t0_bounds": np.asarray([args.t0_min, args.t0_max]),
            "alpha_bounds": np.asarray([args.alpha_min, args.alpha_max]),
            "logg_prior_mean": np.asarray(args.logg_prior_mean),
            "logg_prior_sigma": np.asarray(args.logg_prior_sigma),
            "logg_bounds": np.asarray([args.logg_min, args.logg_max]),
            "fix_logg": np.asarray(args.fix_logg),
            "fixed_logg": np.asarray(args.init_logg),
            "log_vmr_bounds": np.asarray(
                [
                    [args.log_vmr_co_min, args.log_vmr_co_max],
                    [args.log_vmr_h2o_min, args.log_vmr_h2o_max],
                    [args.log_vmr_ch4_min, args.log_vmr_ch4_max],
                    [args.log_vmr_hf_min, args.log_vmr_hf_max],
                ]
            ),
            "log_p_cloud_bounds": np.asarray(
                [args.log_p_cloud_min, args.log_p_cloud_max]
            ),
            "sigma_log_p_scale": np.asarray(args.sigma_log_p_scale),
            "fix_sigma_log_p": np.asarray(args.fixed_sigma_log_p is not None),
            "direct_sigma_log_p": np.asarray(args.direct_sigma_log_p),
            "fixed_sigma_log_p": np.asarray(
                np.nan
                if args.fixed_sigma_log_p is None
                else args.fixed_sigma_log_p
            ),
            "fixed_ell_b": np.asarray(args.fixed_ell_b),
            "gp_jitter": np.asarray(args.gp_jitter),
            "pressure_gp_factorization": np.asarray(
                args.pressure_gp_factorization
            ),
            "pressure_gp_geometry_precision": np.asarray(
                (
                    "float64_recomputed_from_angles"
                    if args.pressure_gp_factorization == "fixed_eigen"
                    else "runtime_jax_dtype"
                )
            ),
            "pressure_gp_eigenvalues": np.asarray(
                []
                if pressure_gp_eigendecomposition is None
                else pressure_gp_eigendecomposition["eigenvalues"]
            ),
            "pressure_gp_minimum_raw_eigenvalue": np.asarray(
                np.nan
                if pressure_gp_eigendecomposition is None
                else pressure_gp_eigendecomposition[
                    "minimum_raw_eigenvalue"
                ]
            ),
            "pressure_gp_maximum_eigenvalue": np.asarray(
                np.nan
                if pressure_gp_eigendecomposition is None
                else pressure_gp_eigendecomposition[
                    "maximum_eigenvalue"
                ]
            ),
            "pressure_gp_clipped_negative_eigenvalue_count": np.asarray(
                0
                if pressure_gp_eigendecomposition is None
                else pressure_gp_eigendecomposition[
                    "clipped_negative_eigenvalue_count"
                ]
            ),
            "standardized_parameters": np.asarray(args.standardized_parameters),
            "rotated_atmosphere_parameters": np.asarray(
                args.rotated_atmosphere_parameters
            ),
            "gaussianized_atmosphere": np.asarray(
                args.gaussianized_atmosphere
            ),
            "gaussianized_atmosphere_names": np.asarray(
                active_gaussianized_atmosphere_names(
                    args.fixed_sigma_log_p,
                    args.direct_sigma_log_p,
                )
            ),
            "sigma_log_p_parameterization": np.asarray(
                sigma_log_p_parameterization(
                    gaussianized_atmosphere=args.gaussianized_atmosphere,
                    standardized_parameters=args.standardized_parameters,
                    uses_eigen_basis=eigen_basis is not None,
                    fixed_sigma_log_p=args.fixed_sigma_log_p,
                    direct_sigma_log_p=args.direct_sigma_log_p,
                )
            ),
            "atmosphere_rotation_matrix": np.asarray(atmosphere_rotation),
            "atmosphere_rotation_label": np.asarray(
                args.atmosphere_rotation_label
            ),
            "joint_atmosphere_a_sigma_d": np.asarray(
                args.joint_atmosphere_a_sigma_d
            ),
            "joint_atmosphere_a_sigma_d_names": np.asarray(
                joint_atmosphere_a_sigma_d_names(len(chip_data_list))
            ),
            "joint_atmosphere_a_sigma_d_rotation_matrix": np.asarray(
                []
                if joint_atmosphere_a_sigma_d_rotation is None
                else joint_atmosphere_a_sigma_d_rotation
            ),
            "joint_atmosphere_a_sigma_d_rotation_label": np.asarray(
                args.joint_atmosphere_a_sigma_d_rotation_label
            ),
            "A_prior_bounds": np.asarray(A_PRIOR_BOUNDS),
            "sigma_d_lognormal_median": np.asarray(
                SIGMA_D_LOGNORMAL_MEDIAN
            ),
            "sigma_d_lognormal_scale": np.asarray(
                SIGMA_D_LOGNORMAL_SCALE
            ),
            "manual_atmosphere_init": np.asarray(args.manual_atmosphere_init),
            "standardized_parameter_centers": np.asarray(
                [
                    parameter_centers["T0"],
                    parameter_centers["alpha"],
                    parameter_centers["logg"],
                    parameter_centers["log_vmr_co"],
                    parameter_centers["log_vmr_h2o"],
                    parameter_centers["log_vmr_ch4"],
                    parameter_centers["log_vmr_hf"],
                    parameter_centers["log_p_cloud"],
                    parameter_centers["sigma_log_p"],
                ]
            ),
            "standardized_parameter_scales": np.asarray(
                [
                    parameter_scales["T0"],
                    parameter_scales["alpha"],
                    parameter_scales["logg"],
                    parameter_scales["log_vmr_co"],
                    parameter_scales["log_vmr_h2o"],
                    parameter_scales["log_vmr_ch4"],
                    parameter_scales["log_vmr_hf"],
                    parameter_scales["log_p_cloud"],
                    parameter_scales["sigma_log_p"],
                ]
            ),
            "atmosphere_rotation_slopes": np.asarray(
                [
                    parameter_scales["zeta_vmr_per_t0"],
                    parameter_scales["log_p_cloud_per_alpha"],
                ]
            ),
            "atmosphere_rotation_slope_names": np.asarray(
                ["zeta_vmr_per_t0", "log_p_cloud_per_alpha"]
            ),
            "standardized_parameter_names": np.asarray(
                [
                    "T0",
                    "alpha",
                    "logg",
                    "log_vmr_co",
                    "log_vmr_h2o",
                    "log_vmr_ch4",
                    "log_vmr_hf",
                    "log_p_cloud",
                    "sigma_log_p",
                ]
            ),
            "zero_mean_pressure_map": np.asarray(args.zero_mean_pressure_map),
            "zero_mean_log_w": np.asarray(args.zero_mean_log_w),
            "zero_sum_log_w_basis": np.asarray(args.zero_sum_log_w_basis),
            "eigen_basis": np.asarray("" if args.eigen_basis is None else args.eigen_basis),
            "eigen_mode_count": np.asarray(args.eigen_mode_count),
            "eigen_fixed_ell": np.asarray(eigen_fixed_ell),
            "eigen_sigma_scale": np.asarray(args.eigen_sigma_scale),
            "init_sigma_eigen": np.asarray(args.init_sigma_eigen),
            "sigma_eigen_log_raw_scale": np.asarray(args.sigma_eigen_log_raw_scale),
            "fixed_sigma_eigen": np.asarray(
                []
                if args.fixed_sigma_eigen is None
                else args.fixed_sigma_eigen,
                dtype=float,
            ),
            "frozen_eigen_spectra": np.asarray(args.frozen_eigen_spectra),
            "pressure_derivative_method": np.asarray(
                (
                    "eigen_atmosphere_frozen_response"
                    if args.frozen_eigen_spectra
                    else "eigen_atmosphere_on_the_fly_autodiff"
                )
                if eigen_basis is not None
                else "on_the_fly_autodiff"
            ),
            "full_data": np.asarray(args.full_data),
            "mask_zero_flux": np.asarray(args.mask_zero_flux),
            "observation_mask_rule": np.asarray(observation_mask_rule),
            "observation_valid_count": np.asarray(observation_valid_count),
            "observation_excluded_count": np.asarray(
                observation_excluded_count
            ),
            "observation_indices": np.asarray(
                observation_indices,
                dtype=np.int64,
            ),
            "preflight_autodiff": np.asarray(args.preflight_autodiff),
            "dense_mass": np.asarray(args.dense_mass),
            "dense_atmosphere_mass": np.asarray(
                args.dense_atmosphere_mass
            ),
            "dense_atmosphere_a_sigma_d_mass": np.asarray(
                args.dense_atmosphere_a_sigma_d_mass
            ),
            "dense_mass_mode": np.asarray(dense_mass_mode),
            "dense_mass_sites": np.asarray(dense_mass_sites),
            "adapt_mass_matrix": np.asarray(args.adapt_mass_matrix),
            "initial_inverse_mass_matrix_mode": np.asarray(
                initial_inverse_mass_matrix_mode
            ),
            "initial_inverse_mass_matrix_label": np.asarray(
                args.initial_inverse_mass_label
            ),
            "initial_inverse_mass_matrix_sites": np.asarray(
                initial_inverse_mass_matrix_sites,
                dtype=str,
            ),
            "initial_inverse_mass_matrix_site_sizes": np.asarray(
                initial_inverse_mass_matrix_site_sizes,
                dtype=int,
            ),
            "initial_inverse_mass_diagonal": np.asarray(
                (
                    ()
                    if args.initial_inverse_mass_diagonal is None
                    else args.initial_inverse_mass_diagonal
                ),
                dtype=float,
            ),
            "fix_nuisance": np.asarray(args.fix_nuisance),
            "fix_a": np.asarray(args.fix_a),
            "fix_log_w": np.asarray(args.fix_log_w),
            "fix_sigma_d": np.asarray(args.fix_sigma_d),
            "fixed_nuisance_sites": np.asarray(
                tuple(
                    ()
                    if fixed_nuisance_values is None
                    else fixed_nuisance_values
                ),
                dtype=str,
            ),
            "map_init": np.asarray(args.map_init),
            "init_from": np.asarray("" if args.init_from is None else args.init_from),
            "target_accept_prob": np.asarray(args.target_accept_prob),
            "max_tree_depth": np.asarray(sampling_max_tree_depth),
            "warmup_max_tree_depth": np.asarray(
                warmup_max_tree_depth
            ),
            "final_step_size": np.asarray(final_step_size),
            "tree_depth_cap_num_steps": np.asarray(
                maximum_tree_num_steps
            ),
            "tree_depth_cap_count": np.asarray(tree_depth_cap_count),
            "tree_depth_cap_fraction": np.asarray(
                np.nan
                if tree_depth_cap_fraction is None
                else tree_depth_cap_fraction
            ),
            "x64": np.asarray(args.x64),
        }
    )
    for chip_position, chip_data in enumerate(chip_data_list):
        save_data[f"wavelengths_chip{chip_data.chip_index}"] = np.asarray(
            chip_data.wavelengths
        )
        save_data[f"profile_wavelengths_chip{chip_data.chip_index}"] = np.asarray(
            profile_wavelengths[chip_position]
        )
        save_data[f"flux_chip{chip_data.chip_index}"] = np.asarray(chip_data.flux)
        save_data[f"observation_mask_chip{chip_data.chip_index}"] = np.asarray(
            observation_masks[chip_position],
            dtype=bool,
        )
        save_data[f"chip_position_{chip_position}"] = np.asarray(chip_data.chip_index)
    if eigen_basis is not None:
        save_data["eigen_parameter_names"] = np.asarray(eigen_basis["parameter_names"])
        save_data["eigen_parameter_scales"] = np.asarray(eigen_basis["parameter_scales"])
        save_data["eigen_selected_v"] = np.asarray(eigen_basis["selected_v"])
    np.savez(output_path, **save_data)
    log_vmr_bounds = {
        "log_vmr_co": [args.log_vmr_co_min, args.log_vmr_co_max],
        "log_vmr_h2o": [args.log_vmr_h2o_min, args.log_vmr_h2o_max],
        "log_vmr_ch4": [args.log_vmr_ch4_min, args.log_vmr_ch4_max],
        "log_vmr_hf": [args.log_vmr_hf_min, args.log_vmr_hf_max],
    }
    init_log_vmrs = {
        "log_vmr_co": args.init_log_vmr_co,
        "log_vmr_h2o": args.init_log_vmr_h2o,
        "log_vmr_ch4": args.init_log_vmr_ch4,
        "log_vmr_hf": args.init_log_vmr_hf,
    }
    diagnostics = {
        "mode": args.run_label,
        "output_path": str(output_path),
        "run_seconds": run_seconds,
        "chip_indices": args.chip_indices,
        "full_data": args.full_data,
        "mask_zero_flux": args.mask_zero_flux,
        "observation_mask_rule": observation_mask_rule,
        "observation_valid_count": observation_valid_count,
        "observation_excluded_count": observation_excluded_count,
        "observation_excluded_count_by_chip": {
            str(chip_data.chip_index): int(
                observation_masks[chip_position].size
                - np.count_nonzero(observation_masks[chip_position])
            )
            for chip_position, chip_data in enumerate(chip_data_list)
        },
        "n_chip": len(chip_data_list),
        "n_phase": int(data.shape[1]),
        "n_wavelength": int(data.shape[2]),
        "nside": args.nside,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "target_accept_prob": args.target_accept_prob,
        "max_tree_depth": sampling_max_tree_depth,
        "warmup_max_tree_depth": warmup_max_tree_depth,
        "log_p_cloud_bounds": [args.log_p_cloud_min, args.log_p_cloud_max],
        "t0_bounds": [args.t0_min, args.t0_max],
        "alpha_bounds": [args.alpha_min, args.alpha_max],
        "logg_prior_mean": args.logg_prior_mean,
        "logg_prior_sigma": args.logg_prior_sigma,
        "logg_bounds": [args.logg_min, args.logg_max],
        "log_vmr_bounds": log_vmr_bounds,
        "init_t0": args.init_t0,
        "init_alpha": args.init_alpha,
        "init_logg": args.init_logg,
        "fix_logg": args.fix_logg,
        "fixed_logg": args.init_logg,
        "init_log_vmrs": init_log_vmrs,
        "init_log_p_cloud": args.init_log_p_cloud,
        "fix_sigma_log_p": args.fixed_sigma_log_p is not None,
        "direct_sigma_log_p": args.direct_sigma_log_p,
        "fixed_sigma_log_p": args.fixed_sigma_log_p,
        "sigma_log_p_parameterization": sigma_log_p_parameterization(
            gaussianized_atmosphere=args.gaussianized_atmosphere,
            standardized_parameters=args.standardized_parameters,
            uses_eigen_basis=eigen_basis is not None,
            fixed_sigma_log_p=args.fixed_sigma_log_p,
            direct_sigma_log_p=args.direct_sigma_log_p,
        ),
        "fixed_ell_b": args.fixed_ell_b,
        "gp_jitter": args.gp_jitter,
        "pressure_gp_factorization": args.pressure_gp_factorization,
        "pressure_gp_geometry_precision": (
            "float64_recomputed_from_angles"
            if args.pressure_gp_factorization == "fixed_eigen"
            else "runtime_jax_dtype"
        ),
        "pressure_gp_minimum_raw_eigenvalue": (
            None
            if pressure_gp_eigendecomposition is None
            else pressure_gp_eigendecomposition[
                "minimum_raw_eigenvalue"
            ]
        ),
        "pressure_gp_maximum_eigenvalue": (
            None
            if pressure_gp_eigendecomposition is None
            else pressure_gp_eigendecomposition["maximum_eigenvalue"]
        ),
        "pressure_gp_clipped_negative_eigenvalue_count": (
            0
            if pressure_gp_eigendecomposition is None
            else pressure_gp_eigendecomposition[
                "clipped_negative_eigenvalue_count"
            ]
        ),
        "pressure_gp_negative_eigenvalue_tolerance": (
            None
            if pressure_gp_eigendecomposition is None
            else pressure_gp_eigendecomposition[
                "negative_eigenvalue_tolerance"
            ]
        ),
        "effective_parameter_centers": {
            name: float(value)
            for name, value in parameter_centers.items()
            if name
            in [
                "T0",
                "alpha",
                "logg",
                "log_vmr_co",
                "log_vmr_h2o",
                "log_vmr_ch4",
                "log_vmr_hf",
                "log_p_cloud",
                "sigma_log_p",
            ]
        },
        "zero_mean_pressure_map": args.zero_mean_pressure_map,
        "zero_mean_log_w": args.zero_mean_log_w,
        "zero_sum_log_w_basis": args.zero_sum_log_w_basis,
        "eigen_basis": None if args.eigen_basis is None else args.eigen_basis,
        "eigen_mode_count": args.eigen_mode_count,
        "eigen_fixed_ell": eigen_fixed_ell,
        "eigen_sigma_scale": args.eigen_sigma_scale,
        "init_sigma_eigen": args.init_sigma_eigen,
        "sigma_eigen_log_raw_scale": args.sigma_eigen_log_raw_scale,
        "fixed_sigma_eigen": (
            None if args.fixed_sigma_eigen is None else list(args.fixed_sigma_eigen)
        ),
        "frozen_eigen_spectra": args.frozen_eigen_spectra,
        "standardized_parameters": args.standardized_parameters,
        "rotated_atmosphere_parameters": args.rotated_atmosphere_parameters,
        "gaussianized_atmosphere": args.gaussianized_atmosphere,
        "gaussianized_atmosphere_names": list(
            active_gaussianized_atmosphere_names(
                args.fixed_sigma_log_p,
                args.direct_sigma_log_p,
            )
        ),
        "atmosphere_rotation_matrix": np.asarray(
            atmosphere_rotation
        ).tolist(),
        "atmosphere_rotation_label": args.atmosphere_rotation_label,
        "joint_atmosphere_a_sigma_d": args.joint_atmosphere_a_sigma_d,
        "joint_atmosphere_a_sigma_d_names": list(
            joint_atmosphere_a_sigma_d_names(len(chip_data_list))
        ),
        "joint_atmosphere_a_sigma_d_rotation_matrix": (
            None
            if joint_atmosphere_a_sigma_d_rotation is None
            else np.asarray(
                joint_atmosphere_a_sigma_d_rotation
            ).tolist()
        ),
        "joint_atmosphere_a_sigma_d_rotation_label": (
            args.joint_atmosphere_a_sigma_d_rotation_label
        ),
        "A_prior_bounds": list(A_PRIOR_BOUNDS),
        "sigma_d_lognormal_median": SIGMA_D_LOGNORMAL_MEDIAN,
        "sigma_d_lognormal_scale": SIGMA_D_LOGNORMAL_SCALE,
        "manual_atmosphere_init": args.manual_atmosphere_init,
        "standardized_parameter_centers": {
            name: float(value) for name, value in parameter_centers.items()
            if name
            in [
                "T0",
                "alpha",
                "logg",
                "log_vmr_co",
                "log_vmr_h2o",
                "log_vmr_ch4",
                "log_vmr_hf",
                "log_p_cloud",
                "sigma_log_p",
            ]
        },
        "standardized_parameter_scales": {
            name: float(value)
            for name, value in parameter_scales.items()
            if name
            in [
                "T0",
                "alpha",
                "logg",
                "log_vmr_co",
                "log_vmr_h2o",
                "log_vmr_ch4",
                "log_vmr_hf",
                "log_p_cloud",
                "sigma_log_p",
                "log_p_cloud_per_alpha",
            ]
        },
        "dense_mass": args.dense_mass,
        "dense_atmosphere_mass": args.dense_atmosphere_mass,
        "dense_atmosphere_a_sigma_d_mass": (
            args.dense_atmosphere_a_sigma_d_mass
        ),
        "dense_mass_mode": dense_mass_mode,
        "dense_mass_sites": list(dense_mass_sites),
        "adapt_mass_matrix": args.adapt_mass_matrix,
        "initial_inverse_mass_matrix_mode": (
            initial_inverse_mass_matrix_mode
        ),
        "initial_inverse_mass_matrix_label": (
            args.initial_inverse_mass_label
        ),
        "initial_inverse_mass_matrix_sites": list(
            initial_inverse_mass_matrix_sites
        ),
        "initial_inverse_mass_matrix_site_sizes": list(
            initial_inverse_mass_matrix_site_sizes
        ),
        "initial_inverse_mass_diagonal": (
            None
            if args.initial_inverse_mass_diagonal is None
            else list(args.initial_inverse_mass_diagonal)
        ),
        "fix_nuisance": args.fix_nuisance,
        "fix_a": args.fix_a,
        "fix_log_w": args.fix_log_w,
        "fix_sigma_d": args.fix_sigma_d,
        "fixed_nuisance_sites": list(
            ()
            if fixed_nuisance_values is None
            else fixed_nuisance_values
        ),
        "map_init": args.map_init,
        "x64": args.x64,
        "final_step_size": final_step_size,
        "tree_depth_cap_num_steps": maximum_tree_num_steps,
        "tree_depth_cap_count": tree_depth_cap_count,
        "tree_depth_cap_fraction": tree_depth_cap_fraction,
        "divergence_count": int(
            np.sum(np.asarray(extra_fields.get("diverging", []), dtype=bool))
        ),
        "mean_accept_prob": (
            float(np.mean(np.asarray(extra_fields["accept_prob"])))
            if "accept_prob" in extra_fields
            else None
        ),
        "max_num_steps": (
            int(np.max(np.asarray(extra_fields["num_steps"])))
            if "num_steps" in extra_fields
            else None
        ),
        **timing,
    }
    diagnostics_path = out_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
