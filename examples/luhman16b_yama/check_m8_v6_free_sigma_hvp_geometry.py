"""Check free-sigma geometry with a production-graph directional HVP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import exojax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.distributions.transforms import biject_to
from numpyro.infer.hmc_util import euclidean_kinetic_energy, velocity_verlet
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
from scipy import linalg as scipy_linalg
import scipy


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doraex.workflows import on_the_fly_pressure_retrieval as retrieval  # noqa: E402
from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v2_fixed_free_initial_point as replay,
)
from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v5_free_sigma_geometry as v5,
)


SUMMARY_NAME = "m8_v6_free_sigma_hvp_geometry_summary.json"
ARRAYS_NAME = "m8_v6_free_sigma_hvp_geometry_arrays.npz"
HVP_METHOD = "jax.jvp(jax.value_and_grad(full_potential), e_u_sigma)"
SITE_ORDER = v5.SITE_ORDER
ATMOSPHERE_SLICE = v5.ATMOSPHERE_SLICE
A_SLICE = v5.A_SLICE
SIGMA_INDEX = v5.SIGMA_INDEX
FLAT_DIMENSION = v5.FLAT_DIMENSION
DEFAULT_SIGMA_MARKERS = v5.DEFAULT_SIGMA_MARKERS
DEFAULT_FD_U_STEPS = (0.02, 0.04, 0.08)


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    """Parse a nonempty comma-separated list of finite floats."""

    return v5._parse_csv_floats(text)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-from",
        default=str(ROOT / "results/m7/p2/v6/samples.npz"),
    )
    parser.add_argument(
        "--free-samples",
        default=str(ROOT / "results/m8/v1/samples.npz"),
    )
    parser.add_argument(
        "--free-diagnostics",
        default=str(ROOT / "results/m8/v1/diagnostics.json"),
    )
    parser.add_argument(
        "--fixed-samples",
        default=str(ROOT / "results/m8/v3/fixed_seed0/samples.npz"),
    )
    parser.add_argument(
        "--fixed-diagnostics",
        default=str(ROOT / "results/m8/v3/fixed_seed0/diagnostics.json"),
    )
    parser.add_argument(
        "--initial-replay",
        default=str(
            ROOT / "results/m8/v3/fixed_seed0/initial_point_replay.json"
        ),
    )
    parser.add_argument(
        "--v5-summary",
        default=str(
            ROOT
            / "results/m8/v5/free_sigma_geometry"
            / "m8_v5_free_sigma_geometry_summary.json"
        ),
    )
    parser.add_argument(
        "--v5-arrays",
        default=str(
            ROOT
            / "results/m8/v5/free_sigma_geometry"
            / "m8_v5_free_sigma_geometry_arrays.npz"
        ),
    )
    parser.add_argument(
        "--v5-failed",
        default=str(ROOT / "results/m8/v5/free_sigma_geometry/FAILED"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/m8/v6/free_sigma_hvp_geometry"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-sigma-min", type=float, default=0.05)
    parser.add_argument("--profile-sigma-max", type=float, default=0.70)
    parser.add_argument("--profile-num", type=int, default=33)
    parser.add_argument(
        "--sigma-markers",
        type=_parse_csv_floats,
        default=DEFAULT_SIGMA_MARKERS,
    )
    parser.add_argument(
        "--fd-u-steps",
        type=_parse_csv_floats,
        default=DEFAULT_FD_U_STEPS,
    )
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--gram-chunk-size", type=int, default=4096)
    parser.add_argument("--potential-atol", type=float, default=5.0e-2)
    parser.add_argument("--no-x64", dest="x64", action="store_false")
    parser.add_argument("--x64", dest="x64", action="store_true")
    parser.set_defaults(x64=False)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def directional_value_gradient_hvp(potential_fn, flat, direction):
    """Return value, gradient, directional derivative, and one HVP column."""

    return jax.jvp(
        jax.value_and_grad(potential_fn),
        (flat,),
        (direction,),
    )


def scalar_sigma_gradient_hessian(
    potential_fn,
    flat,
    sigma_index: int = SIGMA_INDEX,
):
    """Return scalar-u gradient and Hessian through the full potential."""

    coordinate = flat[sigma_index]

    def scalar_potential(u):
        return potential_fn(flat.at[sigma_index].set(u))

    gradient_fn = jax.grad(scalar_potential)
    gradient, hessian = jax.jvp(
        gradient_fn,
        (coordinate,),
        (jnp.ones_like(coordinate),),
    )
    return gradient, hessian


def central_gradient_column(gradient_plus, gradient_minus, step: float):
    """Return one central-difference gradient column."""

    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive.")
    return (np.asarray(gradient_plus) - np.asarray(gradient_minus)) / (2.0 * step)


def vector_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return norm-relative error and cosine for two finite vectors."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or not (
        np.all(np.isfinite(reference)) and np.all(np.isfinite(candidate))
    ):
        raise ValueError("Agreement vectors must have one finite common shape.")
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    relative_error = float(
        np.linalg.norm(candidate - reference) / max(1.0, reference_norm)
    )
    cosine = float(
        np.dot(reference, candidate)
        / max(1.0e-30, reference_norm * candidate_norm)
    )
    return {
        "relative_error": relative_error,
        "cosine": cosine,
        "reference_norm": reference_norm,
        "candidate_norm": candidate_norm,
    }


def finite_difference_consistency_mask(
    shared_relative_error: np.ndarray,
    shared_cosine: np.ndarray,
    huu_relative_error: np.ndarray,
) -> np.ndarray:
    """Return marker-by-step agreement for the complete HVP criterion."""

    shared_relative_error = np.asarray(shared_relative_error, dtype=np.float64)
    shared_cosine = np.asarray(shared_cosine, dtype=np.float64)
    huu_relative_error = np.asarray(huu_relative_error, dtype=np.float64)
    if not (
        shared_relative_error.shape
        == shared_cosine.shape
        == huu_relative_error.shape
    ):
        raise ValueError("Finite-difference agreement arrays must share a shape.")
    return (
        (shared_relative_error <= 0.5)
        & (shared_cosine >= 0.9)
        & (huu_relative_error <= 0.5)
    )


def analytic_lowrank_sigma_terms(
    sigma: float,
    eigenvalues: np.ndarray,
    gp_jitter: float,
    unit_gram: np.ndarray,
    unit_rhs: np.ndarray,
    residual_dinv_residual: float,
    logdet_diagonal: float,
    observation_count: int,
) -> dict[str, float]:
    """Evaluate the LowRankMVN value and u derivatives in float64 algebra."""

    sigma = float(sigma)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    unit_gram = np.asarray(unit_gram, dtype=np.float64)
    unit_rhs = np.asarray(unit_rhs, dtype=np.float64)
    rank = eigenvalues.size
    if sigma <= 0.0 or gp_jitter <= 0.0:
        raise ValueError("sigma and gp_jitter must be positive.")
    if unit_gram.shape != (rank, rank) or unit_rhs.shape != (rank,):
        raise ValueError("Invalid analytic LowRank input shapes.")

    eigen_scales = np.sqrt(sigma**2 * eigenvalues + gp_jitter)
    capacitance = np.eye(rank) + (
        eigen_scales[:, None] * unit_gram * eigen_scales[None, :]
    )
    cholesky = np.linalg.cholesky(capacitance)
    rhs = eigen_scales * unit_rhs
    beta = scipy_linalg.cho_solve((cholesky, True), rhs, check_finite=False)
    inverse_capacitance = scipy_linalg.cho_solve(
        (cholesky, True),
        np.eye(rank),
        check_finite=False,
    )
    inverse_capacitance = 0.5 * (
        inverse_capacitance + inverse_capacitance.T
    )
    response = np.eye(rank) - inverse_capacitance
    q = sigma**2 * eigenvalues / (sigma**2 * eigenvalues + gp_jitter)

    logdet = logdet_diagonal + 2.0 * np.sum(np.log(np.diag(cholesky)))
    mahalanobis = residual_dinv_residual - float(np.dot(rhs, beta))
    negative_log_likelihood = 0.5 * (
        observation_count * math.log(2.0 * math.pi) + logdet + mahalanobis
    )

    gradient_u = float(np.dot(q, np.diag(response) - beta**2))
    q_beta = q * beta
    trace_q_response = float(np.dot(q, np.diag(response)))
    trace_q_response_q_response = float(
        np.sum(q[:, None] * q[None, :] * response**2)
    )
    hessian_uu = float(
        -2.0 * trace_q_response_q_response
        + 2.0 * trace_q_response
        + 4.0 * np.dot(q_beta, response @ q_beta)
        - 2.0 * np.dot(q, beta**2)
    )
    return {
        "negative_log_likelihood": float(negative_log_likelihood),
        "log_determinant": float(logdet),
        "mahalanobis": float(mahalanobis),
        "gradient_u_likelihood": gradient_u,
        "hessian_uu_likelihood": hessian_uu,
        "capacitance_cholesky_min_diagonal": float(np.min(np.diag(cholesky))),
    }


def build_unit_lowrank_statistics(
    reference_factor: np.ndarray,
    reference_eigen_scales: np.ndarray,
    cov_diag: np.ndarray,
    residual: np.ndarray,
    *,
    chunk_size: int,
) -> dict[str, np.ndarray | float]:
    """Build float64 sufficient statistics without duplicating the full factor."""

    reference_factor = np.asarray(reference_factor)
    reference_eigen_scales = np.asarray(
        reference_eigen_scales, dtype=np.float64
    )
    cov_diag = np.asarray(cov_diag, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if reference_factor.ndim != 2:
        raise ValueError("reference_factor must be two-dimensional.")
    rows, rank = reference_factor.shape
    if reference_eigen_scales.shape != (rank,):
        raise ValueError("reference eigen scales do not match the factor rank.")
    if cov_diag.shape != (rows,) or residual.shape != (rows,):
        raise ValueError("LowRank rows do not match cov_diag/residual.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not (
        np.all(np.isfinite(reference_factor))
        and np.all(np.isfinite(reference_eigen_scales))
        and np.all(np.isfinite(cov_diag))
        and np.all(np.isfinite(residual))
        and np.all(cov_diag > 0.0)
        and np.all(reference_eigen_scales > 0.0)
    ):
        raise ValueError("LowRank sufficient-statistic inputs are invalid.")

    unit_gram = np.zeros((rank, rank), dtype=np.float64)
    unit_rhs = np.zeros(rank, dtype=np.float64)
    residual_dinv_residual = 0.0
    for start in range(0, rows, chunk_size):
        stop = min(rows, start + chunk_size)
        factor_chunk = (
            np.asarray(reference_factor[start:stop], dtype=np.float64)
            / reference_eigen_scales[None, :]
        )
        inverse_diagonal = 1.0 / cov_diag[start:stop]
        weighted_factor = factor_chunk * inverse_diagonal[:, None]
        unit_gram += factor_chunk.T @ weighted_factor
        weighted_residual = residual[start:stop] * inverse_diagonal
        unit_rhs += factor_chunk.T @ weighted_residual
        residual_dinv_residual += float(
            np.dot(residual[start:stop], weighted_residual)
        )
    unit_gram = 0.5 * (unit_gram + unit_gram.T)
    return {
        "unit_gram": unit_gram,
        "unit_rhs": unit_rhs,
        "residual_dinv_residual": float(residual_dinv_residual),
        "logdet_diagonal": float(np.sum(np.log(cov_diag))),
    }


def _validate_configuration(args: argparse.Namespace) -> dict:
    """Validate frozen inputs, predecessor failure, and a fresh v6 output."""

    if args.repeat_count < 2:
        raise ValueError("repeat-count must be at least two.")
    if args.gram_chunk_size <= 0:
        raise ValueError("gram-chunk-size must be positive.")
    if not all(value > 0.0 for value in args.fd_u_steps):
        raise ValueError("fd-u-steps must be positive.")
    if args.x64:
        raise ValueError("The frozen production target must remain float32.")

    legacy_args = argparse.Namespace(
        init_from=args.init_from,
        free_samples=args.free_samples,
        free_diagnostics=args.free_diagnostics,
        fixed_samples=args.fixed_samples,
        fixed_diagnostics=args.fixed_diagnostics,
        initial_replay=args.initial_replay,
        out_dir=args.out_dir,
        seed=args.seed,
        profile_sigma_min=args.profile_sigma_min,
        profile_sigma_max=args.profile_sigma_max,
        profile_num=args.profile_num,
        sigma_markers=args.sigma_markers,
        curvature_sigmas=args.sigma_markers,
        cross_u_steps=args.fd_u_steps,
        potential_atol=args.potential_atol,
        x64=args.x64,
    )
    validation = v5._validate_configuration(legacy_args)

    v5_summary_path = Path(args.v5_summary)
    v5_arrays_path = Path(args.v5_arrays)
    v5_failed_path = Path(args.v5_failed)
    if not v5_failed_path.is_file():
        raise FileNotFoundError(f"Missing predecessor FAILED sentinel: {v5_failed_path}")
    if not v5_summary_path.is_file() or not v5_arrays_path.is_file():
        raise FileNotFoundError("The predecessor summary/arrays are missing.")
    predecessor = json.loads(v5_summary_path.read_text(encoding="utf-8"))
    if predecessor.get("execution_completed") is not True:
        raise ValueError("The predecessor did not complete its calculation.")
    if predecessor.get("numerical_integrity_passed") is not False:
        raise ValueError("The predecessor is not the expected derivative failure.")
    with np.load(v5_arrays_path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError("The predecessor arrays are empty.")

    output_dir = Path(args.out_dir)
    for path in (
        output_dir / SUMMARY_NAME,
        output_dir / ARRAYS_NAME,
        Path(f"{output_dir / SUMMARY_NAME}.tmp"),
        Path(f"{output_dir / ARRAYS_NAME}.tmp"),
    ):
        if path.exists():
            raise FileExistsError(f"Refusing output collision: {path}")

    validation["predecessor"] = {
        "mode": predecessor.get("mode"),
        "execution_completed": predecessor.get("execution_completed"),
        "numerical_integrity_passed": predecessor.get(
            "numerical_integrity_passed"
        ),
        "summary_sha256": v5.sha256_file(v5_summary_path),
        "arrays_sha256": v5.sha256_file(v5_arrays_path),
    }
    return validation


@dataclass
class FixedState:
    """One actual fixed-control state in explicit unconstrained order."""

    index: int
    flat: np.ndarray
    expected_constrained: np.ndarray
    sigma: float
    stored_potential: float
    robust_scale: np.ndarray


def _load_fixed_state(args: argparse.Namespace) -> FixedState:
    """Select the deterministic actual fixed-control medoid."""

    with np.load(args.fixed_samples, allow_pickle=False) as fixed_archive, np.load(
        args.free_samples, allow_pickle=False
    ) as free_archive:
        a_bounds = tuple(np.asarray(free_archive["A_prior_bounds"], dtype=float))
        fixed_coordinates = v5.unconstrained_archive_coordinates(
            fixed_archive["atmosphere_rotated"],
            fixed_archive["A"],
            None,
            a_bounds=a_bounds,
        )
        medoid = v5.robust_actual_medoid(fixed_coordinates)
        index = int(medoid["index"])
        sigma = float(fixed_archive["sigma_log_p"][index])
        flat = v5._pack_state(
            fixed_archive["atmosphere_rotated"][index],
            fixed_coordinates[index, 7:11],
            sigma,
        )
        expected_constrained = np.concatenate(
            [
                np.asarray(fixed_archive["atmosphere_rotated"][index]).reshape(7),
                np.asarray(fixed_archive["A"][index]).reshape(4),
                np.asarray([sigma]),
            ]
        )
        return FixedState(
            index=index,
            flat=flat,
            expected_constrained=expected_constrained,
            sigma=sigma,
            stored_potential=float(fixed_archive["extra_potential_energy"][index]),
            robust_scale=np.asarray(medoid["scale"], dtype=np.float64),
        )


def _run_gpu_diagnostic(args: argparse.Namespace, validation: dict) -> tuple[dict, dict]:
    """Run the production-graph HVP and independent analytic checks."""

    start_time = time.perf_counter()
    jax.config.update("jax_enable_x64", args.x64)
    rng_key = jax.random.PRNGKey(args.seed)
    print("[m8_v6_sigma_hvp] phase=build_target", flush=True)
    wrapper_args = replay._fixed_wrapper_args(Path(args.init_from), args.seed)
    (
        _fixed_model,
        free_model,
        _fixed_initial_values,
        free_initial_values,
        _physical_initial_values,
    ) = replay._build_models(wrapper_args)
    model_info = initialize_model(
        rng_key,
        free_model,
        init_strategy=init_to_value(values=free_initial_values),
        dynamic_args=False,
        forward_mode_differentiation=False,
        validate_grad=True,
    )
    potential_fn = model_info.potential_fn
    model_trace = model_info.model_trace
    transforms = {
        name: biject_to(model_trace[name]["fn"].support) for name in SITE_ORDER
    }
    del model_info, model_trace

    def unpack_flat(flat):
        return {
            "atmosphere_rotated": flat[ATMOSPHERE_SLICE],
            "A": flat[A_SLICE],
            "sigma_log_p": flat[SIGMA_INDEX],
        }

    def constrain_flat(flat):
        unconstrained = unpack_flat(flat)
        return {
            name: transforms[name](unconstrained[name]) for name in SITE_ORDER
        }

    def flat_potential(flat):
        return potential_fn(unpack_flat(flat))

    direction = jnp.zeros(FLAT_DIMENSION).at[SIGMA_INDEX].set(1.0)
    potential_jit = jax.jit(flat_potential)
    standalone_value_gradient_jit = jax.jit(jax.value_and_grad(flat_potential))
    bundle_jit = jax.jit(
        lambda flat: directional_value_gradient_hvp(
            flat_potential,
            flat,
            direction,
        )
    )
    scalar_hessian_jit = jax.jit(
        lambda flat: scalar_sigma_gradient_hessian(flat_potential, flat)
    )
    verlet_init, verlet_update = velocity_verlet(
        flat_potential,
        euclidean_kinetic_energy,
        forward_mode_differentiation=False,
    )

    def verlet_context_value_gradient(flat, step_size):
        momentum = jnp.zeros_like(flat)
        state = verlet_init(flat, momentum)
        state = verlet_update(step_size, jnp.ones_like(flat), state)
        return state.potential_energy, state.z_grad

    verlet_value_gradient_jit = jax.jit(verlet_context_value_gradient)

    fixed_state = _load_fixed_state(args)
    center_base = np.asarray(fixed_state.flat, dtype=np.float32)
    constrained = constrain_flat(jnp.asarray(center_base))
    constrained_flat = np.concatenate(
        [
            np.asarray(jax.device_get(constrained["atmosphere_rotated"])).reshape(7),
            np.asarray(jax.device_get(constrained["A"])).reshape(4),
            np.asarray([jax.device_get(constrained["sigma_log_p"])]),
        ]
    )
    roundtrip_error = float(
        np.max(np.abs(constrained_flat - fixed_state.expected_constrained))
    )

    print("[m8_v6_sigma_hvp] phase=value_profile", flush=True)
    profile_sigma = np.unique(
        np.concatenate(
            [
                np.exp(
                    np.linspace(
                        math.log(args.profile_sigma_min),
                        math.log(args.profile_sigma_max),
                        args.profile_num,
                    )
                ),
                np.asarray(args.sigma_markers, dtype=np.float64),
                np.asarray([fixed_state.sigma], dtype=np.float64),
            ]
        )
    )
    profile_sigma.sort()
    profile_total = np.empty(profile_sigma.size, dtype=np.float64)
    for index, sigma in enumerate(profile_sigma):
        candidate = center_base.copy()
        candidate[SIGMA_INDEX] = np.float32(math.log(float(sigma)))
        value = potential_jit(jnp.asarray(candidate))
        value.block_until_ready()
        profile_total[index] = float(jax.device_get(value))
    sigma_prior_profile = v5.sigma_u_prior_terms(
        profile_sigma,
        float(wrapper_args.sigma_log_p_scale),
    )
    profile_conditioned = profile_total - sigma_prior_profile["potential"]
    reference_profile_index = int(np.argmin(np.abs(profile_sigma - fixed_state.sigma)))
    stored_potential_error = float(
        profile_conditioned[reference_profile_index] - fixed_state.stored_potential
    )

    print("[m8_v6_sigma_hvp] phase=extract_lowrank", flush=True)

    def observation_arrays(flat):
        unconstrained = unpack_flat(flat)
        constrained_values = constrain_flat(flat)
        trace = handlers.trace(
            handlers.substitute(
                handlers.seed(free_model, rng_key),
                data=constrained_values,
            )
        ).get_trace()
        shared_log_prior_z = jnp.asarray(0.0)
        for name in ("atmosphere_rotated", "A"):
            value = constrained_values[name]
            shared_log_prior_z = shared_log_prior_z + jnp.sum(
                trace[name]["fn"].log_prob(value)
            ) + jnp.sum(
                transforms[name].log_abs_det_jacobian(
                    unconstrained[name],
                    value,
                )
            )
        observation = trace["obs"]
        distribution = observation["fn"]
        return (
            distribution.loc,
            distribution.cov_factor,
            distribution.cov_diag,
            observation["value"],
            -shared_log_prior_z,
        )

    observation_arrays_jit = jax.jit(observation_arrays)
    (
        loc,
        reference_factor,
        cov_diag,
        observed,
        shared_prior_potential,
    ) = observation_arrays_jit(jnp.asarray(center_base))
    reference_factor.block_until_ready()
    geometry = retrieval.build_luhman16b_geometry(nside=wrapper_args.nside)
    pressure_gp = retrieval.build_fixed_pressure_gp_eigendecomposition(
        geometry.distance_matrix,
        wrapper_args.fixed_ell_b,
        theta=geometry.theta,
        phi=geometry.phi,
    )
    eigenvalues = np.asarray(
        jax.device_get(pressure_gp["eigenvalues"]), dtype=np.float64
    )
    gp_jitter = float(wrapper_args.gp_jitter)
    reference_eigen_scales = np.sqrt(
        fixed_state.sigma**2 * eigenvalues + gp_jitter
    )
    reference_factor_host = np.asarray(jax.device_get(reference_factor))
    observation_count = int(reference_factor_host.shape[0])
    cov_diag_host = np.asarray(jax.device_get(cov_diag), dtype=np.float64)
    residual_host = np.asarray(
        jax.device_get(observed - loc), dtype=np.float64
    )
    shared_prior_host = float(jax.device_get(shared_prior_potential))
    del loc, reference_factor, cov_diag, observed, shared_prior_potential

    print("[m8_v6_sigma_hvp] phase=float64_lowrank_statistics", flush=True)
    lowrank_statistics = build_unit_lowrank_statistics(
        reference_factor_host,
        reference_eigen_scales,
        cov_diag_host,
        residual_host,
        chunk_size=args.gram_chunk_size,
    )
    del reference_factor_host, cov_diag_host, residual_host

    marker_count = len(args.sigma_markers)
    repeat_shape = (marker_count, args.repeat_count)
    repeat_value = np.empty(repeat_shape, dtype=np.float64)
    repeat_gradient = np.empty(repeat_shape + (FLAT_DIMENSION,), dtype=np.float64)
    repeat_directional_gradient = np.empty(repeat_shape, dtype=np.float64)
    repeat_hvp = np.empty(repeat_shape + (FLAT_DIMENSION,), dtype=np.float64)
    standalone_value = np.empty(marker_count, dtype=np.float64)
    standalone_gradient = np.empty((marker_count, FLAT_DIMENSION), dtype=np.float64)
    verlet_value = np.empty(marker_count, dtype=np.float64)
    verlet_gradient = np.empty((marker_count, FLAT_DIMENSION), dtype=np.float64)
    scalar_gradient_u = np.empty(marker_count, dtype=np.float64)
    scalar_hessian_uu = np.empty(marker_count, dtype=np.float64)
    analytic_nll = np.empty(marker_count, dtype=np.float64)
    analytic_total = np.empty(marker_count, dtype=np.float64)
    analytic_gradient_u = np.empty(marker_count, dtype=np.float64)
    analytic_hessian_uu = np.empty(marker_count, dtype=np.float64)
    analytic_logdet = np.empty(marker_count, dtype=np.float64)
    analytic_mahalanobis = np.empty(marker_count, dtype=np.float64)
    analytic_cholesky_min_diagonal = np.empty(marker_count, dtype=np.float64)

    fd_shape = (marker_count, len(args.fd_u_steps))
    fd_value_plus = np.empty(fd_shape, dtype=np.float64)
    fd_value_minus = np.empty(fd_shape, dtype=np.float64)
    fd_gradient_plus = np.empty(fd_shape + (FLAT_DIMENSION,), dtype=np.float64)
    fd_gradient_minus = np.empty_like(fd_gradient_plus)
    fd_column = np.empty_like(fd_gradient_plus)
    fd_hvp_relative_error = np.empty(fd_shape, dtype=np.float64)
    fd_hvp_cosine = np.empty(fd_shape, dtype=np.float64)
    fd_shared_relative_error = np.empty(fd_shape, dtype=np.float64)
    fd_shared_cosine = np.empty(fd_shape, dtype=np.float64)
    fd_huu_relative_error = np.empty(fd_shape, dtype=np.float64)
    fd_first_value_signal_ulp = np.empty(fd_shape, dtype=np.float64)
    fd_second_value_signal_ulp = np.empty(fd_shape, dtype=np.float64)
    center_value_ulp = np.empty(marker_count, dtype=np.float64)

    marker_centers = np.empty((marker_count, FLAT_DIMENSION), dtype=np.float32)
    marker_start = time.perf_counter()
    for marker_index, sigma in enumerate(args.sigma_markers):
        print(
            f"[m8_v6_sigma_hvp] phase=marker sigma={sigma:.9g}",
            flush=True,
        )
        center = center_base.copy()
        center[SIGMA_INDEX] = np.float32(math.log(float(sigma)))
        marker_centers[marker_index] = center
        for repeat_index in range(args.repeat_count):
            (value, gradient), (directional_gradient, hvp) = bundle_jit(
                jnp.asarray(center)
            )
            value.block_until_ready()
            repeat_value[marker_index, repeat_index] = float(jax.device_get(value))
            repeat_gradient[marker_index, repeat_index] = np.asarray(
                jax.device_get(gradient), dtype=np.float64
            )
            repeat_directional_gradient[marker_index, repeat_index] = float(
                jax.device_get(directional_gradient)
            )
            repeat_hvp[marker_index, repeat_index] = np.asarray(
                jax.device_get(hvp), dtype=np.float64
            )

        value, gradient = standalone_value_gradient_jit(jnp.asarray(center))
        value.block_until_ready()
        standalone_value[marker_index] = float(jax.device_get(value))
        standalone_gradient[marker_index] = np.asarray(
            jax.device_get(gradient), dtype=np.float64
        )
        value, gradient = verlet_value_gradient_jit(
            jnp.asarray(center),
            jnp.asarray(0.0),
        )
        value.block_until_ready()
        verlet_value[marker_index] = float(jax.device_get(value))
        verlet_gradient[marker_index] = np.asarray(
            jax.device_get(gradient), dtype=np.float64
        )
        scalar_gradient, scalar_hessian = scalar_hessian_jit(jnp.asarray(center))
        scalar_hessian.block_until_ready()
        scalar_gradient_u[marker_index] = float(jax.device_get(scalar_gradient))
        scalar_hessian_uu[marker_index] = float(jax.device_get(scalar_hessian))

        analytic = analytic_lowrank_sigma_terms(
            sigma,
            eigenvalues,
            gp_jitter,
            lowrank_statistics["unit_gram"],
            lowrank_statistics["unit_rhs"],
            lowrank_statistics["residual_dinv_residual"],
            lowrank_statistics["logdet_diagonal"],
            observation_count,
        )
        prior = v5.sigma_u_prior_terms(sigma, wrapper_args.sigma_log_p_scale)
        analytic_nll[marker_index] = analytic["negative_log_likelihood"]
        analytic_total[marker_index] = (
            analytic["negative_log_likelihood"]
            + shared_prior_host
            + float(prior["potential"])
        )
        analytic_gradient_u[marker_index] = (
            analytic["gradient_u_likelihood"] + float(prior["gradient_u"])
        )
        analytic_hessian_uu[marker_index] = (
            analytic["hessian_uu_likelihood"] + float(prior["curvature_uu"])
        )
        analytic_logdet[marker_index] = analytic["log_determinant"]
        analytic_mahalanobis[marker_index] = analytic["mahalanobis"]
        analytic_cholesky_min_diagonal[marker_index] = analytic[
            "capacitance_cholesky_min_diagonal"
        ]

        center_value = repeat_value[marker_index, 0]
        ulp = abs(float(np.spacing(np.float32(center_value))))
        center_value_ulp[marker_index] = ulp
        hvp_reference = repeat_hvp[marker_index, 0]
        for step_index, step in enumerate(args.fd_u_steps):
            plus = center.copy()
            minus = center.copy()
            plus[SIGMA_INDEX] += np.float32(step)
            minus[SIGMA_INDEX] -= np.float32(step)
            (value_plus, gradient_plus), _ = bundle_jit(jnp.asarray(plus))
            (value_minus, gradient_minus), _ = bundle_jit(jnp.asarray(minus))
            value_plus.block_until_ready()
            value_minus.block_until_ready()
            value_plus_host = float(jax.device_get(value_plus))
            value_minus_host = float(jax.device_get(value_minus))
            gradient_plus_host = np.asarray(
                jax.device_get(gradient_plus), dtype=np.float64
            )
            gradient_minus_host = np.asarray(
                jax.device_get(gradient_minus), dtype=np.float64
            )
            candidate_column = central_gradient_column(
                gradient_plus_host,
                gradient_minus_host,
                step,
            )
            fd_value_plus[marker_index, step_index] = value_plus_host
            fd_value_minus[marker_index, step_index] = value_minus_host
            fd_gradient_plus[marker_index, step_index] = gradient_plus_host
            fd_gradient_minus[marker_index, step_index] = gradient_minus_host
            fd_column[marker_index, step_index] = candidate_column
            full_agreement = vector_agreement(hvp_reference, candidate_column)
            shared_agreement = vector_agreement(
                hvp_reference[:SIGMA_INDEX],
                candidate_column[:SIGMA_INDEX],
            )
            fd_hvp_relative_error[marker_index, step_index] = full_agreement[
                "relative_error"
            ]
            fd_hvp_cosine[marker_index, step_index] = full_agreement["cosine"]
            fd_shared_relative_error[marker_index, step_index] = shared_agreement[
                "relative_error"
            ]
            fd_shared_cosine[marker_index, step_index] = shared_agreement["cosine"]
            fd_huu_relative_error[marker_index, step_index] = abs(
                candidate_column[SIGMA_INDEX] - hvp_reference[SIGMA_INDEX]
            ) / max(1.0, abs(hvp_reference[SIGMA_INDEX]))
            fd_first_value_signal_ulp[marker_index, step_index] = abs(
                value_plus_host - value_minus_host
            ) / ulp
            fd_second_value_signal_ulp[marker_index, step_index] = abs(
                value_plus_host - 2.0 * center_value + value_minus_host
            ) / ulp

    marker_seconds = time.perf_counter() - marker_start
    primary_hvp = repeat_hvp[:, 0]
    primary_gradient = repeat_gradient[:, 0]
    fd_steps = np.asarray(args.fd_u_steps, dtype=np.float64)
    fd_value_gradient_u = (fd_value_plus - fd_value_minus) / (
        2.0 * fd_steps[None, :]
    )
    fd_value_hessian_uu = (
        fd_value_plus
        - 2.0 * repeat_value[:, :1]
        + fd_value_minus
    ) / fd_steps[None, :] ** 2
    shared_scale = fixed_state.robust_scale[:SIGMA_INDEX]
    cross_scaled = primary_hvp[:, :SIGMA_INDEX] * shared_scale[None, :]
    cross_norm_scaled = np.linalg.norm(cross_scaled, axis=1)
    atmosphere_rotation = np.asarray(
        wrapper_args.atmosphere_rotation_matrix, dtype=np.float64
    ).reshape(7, 7)
    cross_atmosphere_gaussianized = np.einsum(
        "ij,mj->mi",
        atmosphere_rotation,
        primary_hvp[:, :7],
    )

    gaussian_coordinate = np.empty(marker_count, dtype=np.float64)
    gaussian_du_dg = np.empty(marker_count, dtype=np.float64)
    gaussian_d2u_dg2 = np.empty(marker_count, dtype=np.float64)
    gaussian_gradient = np.empty(marker_count, dtype=np.float64)
    gaussian_hessian = np.empty(marker_count, dtype=np.float64)
    gaussian_cross_norm_scaled = np.empty(marker_count, dtype=np.float64)
    for marker_index, sigma in enumerate(args.sigma_markers):
        prior = v5.sigma_u_prior_terms(sigma, wrapper_args.sigma_log_p_scale)
        transform = v5.halfnormal_cdf_coordinate_terms(
            sigma, wrapper_args.sigma_log_p_scale
        )
        likelihood_gradient = (
            primary_gradient[marker_index, SIGMA_INDEX]
            - float(prior["gradient_u"])
        )
        likelihood_hessian = (
            primary_hvp[marker_index, SIGMA_INDEX]
            - float(prior["curvature_uu"])
        )
        gaussian_coordinate[marker_index] = transform["g"]
        gaussian_du_dg[marker_index] = transform["du_dg"]
        gaussian_d2u_dg2[marker_index] = transform["d2u_dg2"]
        gaussian_gradient[marker_index] = (
            likelihood_gradient * transform["du_dg"] + transform["g"]
        )
        gaussian_hessian[marker_index] = (
            likelihood_hessian * transform["du_dg"] ** 2
            + likelihood_gradient * transform["d2u_dg2"]
            + 1.0
        )
        gaussian_cross_norm_scaled[marker_index] = (
            cross_norm_scaled[marker_index] * transform["du_dg"]
        )

    repeat_value_stable = bool(
        np.all(repeat_value == repeat_value[:, :1])
    )
    repeat_gradient_stable = bool(
        np.all(repeat_gradient == repeat_gradient[:, :1])
    )
    repeat_hvp_stable = bool(np.all(repeat_hvp == repeat_hvp[:, :1]))
    directional_identity_error = float(
        np.max(
            np.abs(
                repeat_directional_gradient
                - repeat_gradient[:, :, SIGMA_INDEX]
            )
        )
    )
    scalar_hessian_relative_error = np.abs(
        scalar_hessian_uu - primary_hvp[:, SIGMA_INDEX]
    ) / np.maximum(1.0, np.abs(primary_hvp[:, SIGMA_INDEX]))
    analytic_hessian_relative_error = np.abs(
        analytic_hessian_uu - primary_hvp[:, SIGMA_INDEX]
    ) / np.maximum(1.0, np.abs(analytic_hessian_uu))
    maximum_bundle_analytic_value_error = float(
        np.max(np.abs(repeat_value[:, 0] - analytic_total))
    )
    best_fd_index = np.argmin(fd_shared_relative_error, axis=1)
    best_fd_shared_relative_error = fd_shared_relative_error[
        np.arange(marker_count), best_fd_index
    ]
    best_fd_shared_cosine = fd_shared_cosine[
        np.arange(marker_count), best_fd_index
    ]
    best_fd_huu_relative_error = fd_huu_relative_error[
        np.arange(marker_count), best_fd_index
    ]
    hvp_fd_consistent_by_marker_step = finite_difference_consistency_mask(
        fd_shared_relative_error,
        fd_shared_cosine,
        fd_huu_relative_error,
    )
    hvp_fd_consistent_per_marker = np.any(
        hvp_fd_consistent_by_marker_step,
        axis=1,
    )

    numeric_arrays = (
        profile_total,
        profile_conditioned,
        repeat_value,
        repeat_gradient,
        repeat_directional_gradient,
        repeat_hvp,
        standalone_value,
        standalone_gradient,
        verlet_value,
        verlet_gradient,
        scalar_gradient_u,
        scalar_hessian_uu,
        analytic_total,
        analytic_gradient_u,
        analytic_hessian_uu,
        analytic_logdet,
        analytic_mahalanobis,
        analytic_cholesky_min_diagonal,
        fd_value_plus,
        fd_value_minus,
        fd_gradient_plus,
        fd_gradient_minus,
        fd_column,
        fd_hvp_relative_error,
        fd_hvp_cosine,
        fd_shared_relative_error,
        fd_shared_cosine,
        fd_huu_relative_error,
        fd_value_gradient_u,
        fd_value_hessian_uu,
        center_value_ulp,
        fd_first_value_signal_ulp,
        fd_second_value_signal_ulp,
        gaussian_gradient,
        gaussian_hessian,
        gaussian_coordinate,
        gaussian_du_dg,
        gaussian_d2u_dg2,
        gaussian_cross_norm_scaled,
        cross_atmosphere_gaussianized,
        cross_norm_scaled,
        eigenvalues,
        np.asarray(lowrank_statistics["unit_gram"]),
        np.asarray(lowrank_statistics["unit_rhs"]),
    )
    all_finite = bool(all(np.all(np.isfinite(array)) for array in numeric_arrays))
    calculation_integrity_passed = bool(
        all_finite
        and abs(stored_potential_error) <= args.potential_atol
        and roundtrip_error <= 1.0e-6
        and maximum_bundle_analytic_value_error <= 0.5
        and np.all(analytic_cholesky_min_diagonal > 0.0)
    )

    profile_minimum_index = int(np.argmin(profile_total))
    conditioned_minimum_index = int(np.argmin(profile_conditioned))
    summary = {
        "schema_version": 1,
        "mode": "m8_v6_free_sigma_production_hvp_geometry",
        "execution_completed": True,
        "calculation_integrity_passed": calculation_integrity_passed,
        "purpose": (
            "Measure the sigma Hessian column in one production computation "
            "graph and separate numerical gradient stability from geometry."
        ),
        "target": {
            "model": "frozen M8 linear marginalized free-sigma target",
            "sigma_log_p_prior": "HalfNormal(scale=0.3 dex)",
            "map_latents": "analytically marginalized",
            "exact_rt_used": False,
            "x64_target": args.x64,
            "seed": args.seed,
        },
        "method": {
            "hvp_available": True,
            "hvp_method": HVP_METHOD,
            "hvp_direction": "u_sigma_log_p only",
            "hvp_tangent_index": SIGMA_INDEX,
            "direction_count": 1,
            "hvp_fallback_allowed": False,
            "hvp_fallback_used": False,
            "full_hessian_computed": False,
            "finite_difference_role": "cross-check only; never an HVP fallback",
            "verlet_context_probe": (
                "velocity_verlet update with a dynamic step-size argument evaluated at 0.0"
            ),
            "float64_analytic_reference": (
                "host LowRank sufficient statistics and closed-form sigma score/Huu"
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpyro": numpyro.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "exojax": exojax.__version__,
            "exojax_import_path": os.path.realpath(inspect.getfile(exojax)),
            "numpy_import_path": os.path.realpath(inspect.getfile(np)),
            "scipy_import_path": os.path.realpath(inspect.getfile(scipy)),
            "scipy_linalg_import_path": os.path.realpath(
                inspect.getfile(scipy_linalg)
            ),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "sigma_markers": list(args.sigma_markers),
            "fd_u_steps": list(args.fd_u_steps),
            "repeat_count": args.repeat_count,
            "profile_sigma_min": args.profile_sigma_min,
            "profile_sigma_max": args.profile_sigma_max,
            "profile_num_actual": int(profile_sigma.size),
            "gram_chunk_size": args.gram_chunk_size,
            "potential_atol": args.potential_atol,
        },
        "state": {
            "role": "fixed-control actual medoid; conditional local probe",
            "source_index": fixed_state.index,
            "reference_sigma_log_p": fixed_state.sigma,
            "stored_fixed_potential": fixed_state.stored_potential,
            "fixed_control_tree_depth_cap_fraction": 0.36,
        },
        "integrity": {
            "all_scientific_arrays_finite": all_finite,
            "stored_fixed_potential_replay_error": stored_potential_error,
            "constrained_roundtrip_error": roundtrip_error,
            "directional_gradient_identity_error": directional_identity_error,
            "minimum_analytic_capacitance_cholesky_diagonal": float(
                np.min(analytic_cholesky_min_diagonal)
            ),
            "maximum_bundle_vs_analytic_value_error": (
                maximum_bundle_analytic_value_error
            ),
            "analytic_value_alignment_atol": 0.5,
        },
        "diagnostic_findings": {
            "repeat_value_bitwise_stable": repeat_value_stable,
            "repeat_gradient_bitwise_stable": repeat_gradient_stable,
            "repeat_hvp_bitwise_stable": repeat_hvp_stable,
            "directional_gradient_identity_consistent": bool(
                directional_identity_error <= 2.0e-3
            ),
            "scalar_hessian_consistent_5pct": bool(
                np.all(scalar_hessian_relative_error <= 0.05)
            ),
            "analytic_hessian_consistent_20pct": bool(
                np.all(analytic_hessian_relative_error <= 0.20)
            ),
            "hvp_fd_consistent_all_markers": bool(
                np.all(hvp_fd_consistent_per_marker)
            ),
            "hvp_fd_consistent_per_marker": hvp_fd_consistent_per_marker,
            "hvp_fd_consistent_by_marker_step": (
                hvp_fd_consistent_by_marker_step
            ),
            "best_fd_step_per_marker": np.asarray(args.fd_u_steps)[best_fd_index],
            "best_fd_shared_relative_error": best_fd_shared_relative_error,
            "best_fd_shared_cosine": best_fd_shared_cosine,
            "best_fd_huu_relative_error": best_fd_huu_relative_error,
            "fd_shared_relative_error_by_marker_step": fd_shared_relative_error,
            "fd_shared_cosine_by_marker_step": fd_shared_cosine,
            "fd_huu_relative_error_by_marker_step": fd_huu_relative_error,
            "fd_value_gradient_u_by_marker_step": fd_value_gradient_u,
            "fd_value_hessian_uu_by_marker_step": fd_value_hessian_uu,
            "maximum_bundle_vs_standalone_gradient_error": float(
                np.max(np.abs(primary_gradient - standalone_gradient))
            ),
            "maximum_bundle_vs_verlet_gradient_error": float(
                np.max(np.abs(primary_gradient - verlet_gradient))
            ),
            "maximum_bundle_vs_analytic_value_error": float(
                maximum_bundle_analytic_value_error
            ),
            "maximum_bundle_vs_analytic_gradient_u_error": float(
                np.max(
                    np.abs(
                        primary_gradient[:, SIGMA_INDEX] - analytic_gradient_u
                    )
                )
            ),
            "maximum_bundle_vs_analytic_hessian_uu_relative_error": float(
                np.max(analytic_hessian_relative_error)
            ),
            "minimum_fd_second_value_signal_ulp": float(
                np.min(fd_second_value_signal_ulp)
            ),
        },
        "profile": {
            "total_grid_minimum_sigma": float(profile_sigma[profile_minimum_index]),
            "conditioned_grid_minimum_sigma": float(
                profile_sigma[conditioned_minimum_index]
            ),
            "interpretation": (
                "The shared fixed-control state is held fixed; this is not a "
                "marginal or profiled posterior."
            ),
        },
        "coordinate_comparison": {
            "direct_coordinate": "u=log(sigma_log_p)",
            "same_prior_coordinate": "g=Phi^-1(F_HalfNormal(sigma_log_p))",
            "direct_cross_norm_fixed_chain_scaled": cross_norm_scaled,
            "gaussian_cross_norm_fixed_chain_scaled": gaussian_cross_norm_scaled,
            "direct_hessian_uu": primary_hvp[:, SIGMA_INDEX],
            "gaussian_hessian_gg": gaussian_hessian,
        },
        "predecessor": validation["predecessor"],
        "timing_seconds": {
            "marker_phase": marker_seconds,
            "total": time.perf_counter() - start_time,
        },
        "limitations": [
            "This is a fixed-control local diagnostic, not posterior sampling.",
            "Only the sigma Hessian column is computed; no full Hessian is built.",
            "Finite differences are diagnostic cross-checks and never replace HVP.",
            "Gradient instability is a reportable result and does not by itself mark the run FAILED.",
            "Free-chain probes should be added only after this fixed-state numerical core is understood.",
            "The frozen linear target retains the previously rejected exact-forward fidelity.",
        ],
    }

    arrays = {
        "coordinate_names": np.asarray(
            [
                *[f"atmosphere_rotated[{index}]" for index in range(7)],
                *[f"A_unconstrained[{index}]" for index in range(4)],
                "u_sigma_log_p",
            ]
        ),
        "hvp_tangent": np.eye(1, FLAT_DIMENSION, SIGMA_INDEX, dtype=np.float32)[0],
        "marker_sigma": np.asarray(args.sigma_markers, dtype=np.float64),
        "marker_center_unconstrained": marker_centers,
        "repeat_value": repeat_value,
        "repeat_gradient": repeat_gradient,
        "repeat_directional_gradient": repeat_directional_gradient,
        "hvp_column_raw": repeat_hvp,
        "standalone_value": standalone_value,
        "standalone_gradient": standalone_gradient,
        "verlet_value": verlet_value,
        "verlet_gradient": verlet_gradient,
        "scalar_gradient_u": scalar_gradient_u,
        "scalar_hessian_uu": scalar_hessian_uu,
        "analytic_negative_log_likelihood": analytic_nll,
        "analytic_total_potential": analytic_total,
        "analytic_gradient_u": analytic_gradient_u,
        "analytic_hessian_uu": analytic_hessian_uu,
        "analytic_log_determinant": analytic_logdet,
        "analytic_mahalanobis": analytic_mahalanobis,
        "analytic_cholesky_min_diagonal": analytic_cholesky_min_diagonal,
        "fd_u_steps": np.asarray(args.fd_u_steps, dtype=np.float64),
        "fd_value_plus": fd_value_plus,
        "fd_value_minus": fd_value_minus,
        "fd_gradient_plus": fd_gradient_plus,
        "fd_gradient_minus": fd_gradient_minus,
        "fd_column_raw": fd_column,
        "fd_hvp_relative_error": fd_hvp_relative_error,
        "fd_hvp_cosine": fd_hvp_cosine,
        "fd_shared_relative_error": fd_shared_relative_error,
        "fd_shared_cosine": fd_shared_cosine,
        "fd_huu_relative_error": fd_huu_relative_error,
        "hvp_fd_consistent_by_marker_step": (
            hvp_fd_consistent_by_marker_step
        ),
        "fd_value_gradient_u": fd_value_gradient_u,
        "fd_value_hessian_uu": fd_value_hessian_uu,
        "center_value_ulp": center_value_ulp,
        "fd_first_value_signal_ulp": fd_first_value_signal_ulp,
        "fd_second_value_signal_ulp": fd_second_value_signal_ulp,
        "profile_sigma": profile_sigma,
        "profile_total_potential": profile_total,
        "profile_conditioned_potential": profile_conditioned,
        "profile_sigma_prior_potential": sigma_prior_profile["potential"],
        "profile_sigma_prior_gradient_u": sigma_prior_profile["gradient_u"],
        "profile_sigma_prior_hessian_uu": sigma_prior_profile["curvature_uu"],
        "stored_fixed_potential_error": np.asarray(stored_potential_error),
        "constrained_roundtrip_error": np.asarray(roundtrip_error),
        "cross_atmosphere_gaussianized": cross_atmosphere_gaussianized,
        "cross_norm_fixed_chain_scaled": cross_norm_scaled,
        "gaussianized_sigma_coordinate": gaussian_coordinate,
        "gaussianized_du_dg": gaussian_du_dg,
        "gaussianized_d2u_dg2": gaussian_d2u_dg2,
        "gaussianized_gradient_g": gaussian_gradient,
        "gaussianized_hessian_gg": gaussian_hessian,
        "gaussianized_cross_norm_fixed_chain_scaled": gaussian_cross_norm_scaled,
        "fixed_chain_robust_scale": fixed_state.robust_scale,
        "gp_eigenvalues": eigenvalues,
        "unit_gram": np.asarray(lowrank_statistics["unit_gram"]),
        "unit_rhs": np.asarray(lowrank_statistics["unit_rhs"]),
    }
    return summary, arrays


def main() -> None:
    """Validate configuration, run the diagnostic, and atomically write outputs."""

    args = parse_args()
    validation = _validate_configuration(args)
    if args.validate_only:
        print(
            json.dumps(
                v5._json_ready(
                    {
                        "validation_passed": True,
                        "mode": "m8_v6_free_sigma_production_hvp_geometry",
                        "out_dir": str(Path(args.out_dir).resolve()),
                        "hvp_method": HVP_METHOD,
                        "hvp_fallback_allowed": False,
                        "full_hessian_computed": False,
                        "sigma_markers": args.sigma_markers,
                        "fd_u_steps": args.fd_u_steps,
                        "repeat_count": args.repeat_count,
                        "state_selection": validation["state_selection"],
                        "predecessor": validation["predecessor"],
                    }
                ),
                indent=2,
                allow_nan=False,
            )
        )
        return

    output_dir = Path(args.out_dir)
    summary_path = output_dir / SUMMARY_NAME
    arrays_path = output_dir / ARRAYS_NAME
    summary, arrays = _run_gpu_diagnostic(args, validation)
    v5._write_npz_atomic(arrays_path, arrays)
    v5._write_json_atomic(summary_path, summary)
    print(
        json.dumps(
            v5._json_ready(summary),
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
