"""Separate full-graph and LowRank precision effects for free sigma_log_p."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable

import exojax
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.distributions.continuous import (
    _batch_capacitance_tril,
    _batch_lowrank_logdet,
    _batch_lowrank_mahalanobis,
)
from numpyro.distributions.transforms import biject_to
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
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
from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v6_free_sigma_hvp_geometry as v6,
)
from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v7_free_sigma_derivative_capture as v7,
)


SUMMARY_NAME = "m8_v8_free_sigma_lowrank_precision_summary.json"
ARRAYS_NAME = "m8_v8_free_sigma_lowrank_precision_arrays.npz"
CHECKPOINT_NAME = "m8_v8_free_sigma_lowrank_precision_checkpoint.json"
EVALUATION_SIGMA = 0.27526917
U_STEP = 0.02
POTENTIAL_ATOL = 5.0e-2
ROUNDTRIP_ATOL = 1.0e-6
SCORE_ATOL = 2.0e-1
V7_EXPECTED_SHA256 = {
    "failed": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "summary": "6eafdfc283ed726f4a88b5c4fbfccf9fd39ee66b72a91d001b00fa72a9db95fb",
    "arrays": "b6a292d3b0bca7eadfc664de74626e9ea75193f0d469c90dd1d1bcb3a228bb17",
    "checkpoint": "be11f6b14fbbee306fa4aa9998a9fb50a4562b0ddb1e5b66cbc69b4e0756debe",
    "failure_hashes": "606de2a88522c753241404bb190150bf148f2e611b6a12e0695c07addd8ef9db",
    "run_log": "688de079b3a5224ccf8d07953d6bb53d83a18c0d3f8bc5be2bcfaf0e90425ab1",
    "provenance": "f30a956a71f73739fa80d48f6a7c71fee579d77d7f94353b4fd8e64e33119049",
}
POINT_LABELS = ("minus", "center", "plus")
COMPONENTS = ("total", "logdet", "mahalanobis", "sigma_prior")
FULL_COMPONENTS = ("total",)
METHODS = ("reverse", "forward")
FULL_ARMS = ("full_f32_default", "full_f32_highest")
COMPONENT_ARMS = (
    "isolated_fullfactor_f32_default",
    "isolated_fullfactor_f32_highest",
    "reduced_stats_f64_highest",
)
ARM_NAMES = (*FULL_ARMS, *COMPONENT_ARMS)
ARM_PRECISION = {
    "full_f32_default": (np.float32, None),
    "full_f32_highest": (np.float32, "highest"),
    "isolated_fullfactor_f32_default": (np.float32, None),
    "isolated_fullfactor_f32_highest": (np.float32, "highest"),
    "reduced_stats_f64_highest": (np.float64, "highest"),
}
SIGMA_INDEX = v6.SIGMA_INDEX
FLAT_DIMENSION = v6.FLAT_DIMENSION
ATMOSPHERE_SLICE = v6.ATMOSPHERE_SLICE
A_SLICE = v6.A_SLICE
SITE_ORDER = v6.SITE_ORDER


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-from", default=str(ROOT / "results/m7/p2/v6/samples.npz")
    )
    parser.add_argument(
        "--free-samples", default=str(ROOT / "results/m8/v1/samples.npz")
    )
    parser.add_argument(
        "--free-diagnostics", default=str(ROOT / "results/m8/v1/diagnostics.json")
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
        default=str(ROOT / "results/m8/v3/fixed_seed0/initial_point_replay.json"),
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
        "--v5-failed", default=str(ROOT / "results/m8/v5/free_sigma_geometry/FAILED")
    )
    parser.add_argument(
        "--v6-dir", default=str(ROOT / "results/m8/v6/free_sigma_hvp_geometry")
    )
    parser.add_argument(
        "--v7-dir",
        default=str(ROOT / "results/m8/v7/free_sigma_derivative_capture"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/m8/v8/free_sigma_lowrank_precision"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-sigma", type=float, default=EVALUATION_SIGMA)
    parser.add_argument("--u-step", type=float, default=U_STEP)
    parser.add_argument("--gram-chunk-size", type=int, default=4096)
    parser.add_argument("--potential-atol", type=float, default=POTENTIAL_ATOL)
    parser.add_argument("--no-x64", dest="x64", action="store_false")
    parser.add_argument("--x64", dest="x64", action="store_true")
    parser.set_defaults(x64=False)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def planned_call_names() -> tuple[str, ...]:
    """Return the exact first-derivative call schedule."""

    calls: list[str] = []
    for arm in FULL_ARMS:
        for component in FULL_COMPONENTS:
            for method in METHODS:
                calls.append(f"{arm}:{component}:{method}")
    for arm in COMPONENT_ARMS:
        for component in COMPONENTS:
            for method in METHODS:
                calls.append(f"{arm}:{component}:{method}")
    return tuple(calls)


def _sigma_prior(u, scale):
    """Return the HalfNormal potential including the exp-transform Jacobian."""

    sigma = jnp.exp(u)
    return 0.5 * (sigma / scale) ** 2 - u + jnp.log(scale) + 0.5 * jnp.log(jnp.pi / 2.0)


sigma_halfnormal_nll = _sigma_prior


def _sigma_scales(u, eigenvalues, gp_jitter):
    """Return the exact float-graph column scales used by the target."""

    dtype = u.dtype
    eigenvalues = jnp.asarray(eigenvalues, dtype=dtype)
    sigma = jnp.exp(u)
    return jnp.sqrt(sigma**2 * eigenvalues + jnp.asarray(gp_jitter, dtype=dtype))


def _fullfactor_component_vector(
    u,
    unit_factor,
    cov_diag,
    residual,
    eigenvalues,
    gp_jitter,
    prior_scale,
    shared_prior,
):
    """Return LowRank components from an explicit unit factor."""

    dtype = u.dtype
    unit_factor = jnp.asarray(unit_factor, dtype=dtype)
    cov_diag = jnp.asarray(cov_diag, dtype=dtype)
    residual = jnp.asarray(residual, dtype=dtype)
    eigenvalues = jnp.asarray(eigenvalues, dtype=dtype)
    scales = _sigma_scales(u, eigenvalues, gp_jitter)
    factor = unit_factor * scales[None, :]
    capacitance_tril = _batch_capacitance_tril(factor, cov_diag)
    logdet = _batch_lowrank_logdet(factor, cov_diag, capacitance_tril)
    mahalanobis = _batch_lowrank_mahalanobis(
        factor, cov_diag, residual, capacitance_tril
    )
    prior = _sigma_prior(u, jnp.asarray(prior_scale, dtype=dtype))
    constant = jnp.asarray(
        0.5 * residual.size * math.log(2.0 * math.pi) + shared_prior,
        dtype=dtype,
    )
    total = constant + 0.5 * logdet + 0.5 * mahalanobis + prior
    return jnp.stack([total, logdet, mahalanobis, prior])


def _reference_fullfactor_component_vector(
    u,
    reference_factor,
    reference_scales,
    cov_diag,
    residual,
    eigenvalues,
    gp_jitter,
    prior_scale,
    shared_prior,
):
    """Return production-order components by rescaling a reference factor."""

    dtype = u.dtype
    reference_factor = jnp.asarray(reference_factor, dtype=dtype)
    reference_scales = jnp.asarray(reference_scales, dtype=dtype)
    cov_diag = jnp.asarray(cov_diag, dtype=dtype)
    residual = jnp.asarray(residual, dtype=dtype)
    eigenvalues = jnp.asarray(eigenvalues, dtype=dtype)
    scales = _sigma_scales(u, eigenvalues, gp_jitter)
    factor = reference_factor * (scales / reference_scales)[None, :]
    capacitance_tril = _batch_capacitance_tril(factor, cov_diag)
    logdet = _batch_lowrank_logdet(factor, cov_diag, capacitance_tril)
    mahalanobis = _batch_lowrank_mahalanobis(
        factor, cov_diag, residual, capacitance_tril
    )
    prior = _sigma_prior(u, jnp.asarray(prior_scale, dtype=dtype))
    constant = jnp.asarray(
        0.5 * residual.size * math.log(2.0 * math.pi) + shared_prior,
        dtype=dtype,
    )
    total = constant + 0.5 * logdet + 0.5 * mahalanobis + prior
    return jnp.stack([total, logdet, mahalanobis, prior])


def _reduced_component_vector(
    u,
    gram,
    rhs_unit,
    residual_dinv_residual,
    logdet_diagonal,
    eigenvalues,
    gp_jitter,
    prior_scale,
    shared_prior,
    observation_count,
):
    """Return the same components from sufficient LowRank statistics."""

    dtype = u.dtype
    gram = jnp.asarray(gram, dtype=dtype)
    rhs_unit = jnp.asarray(rhs_unit, dtype=dtype)
    eigenvalues = jnp.asarray(eigenvalues, dtype=dtype)
    scales = _sigma_scales(u, eigenvalues, gp_jitter)
    capacitance = jnp.eye(eigenvalues.size, dtype=dtype) + (
        scales[:, None] * gram * scales[None, :]
    )
    cholesky = jnp.linalg.cholesky(capacitance)
    rhs = scales * rhs_unit
    solved = jax.scipy.linalg.solve_triangular(cholesky, rhs, lower=True)
    logdet = jnp.asarray(logdet_diagonal, dtype=dtype) + 2.0 * jnp.sum(
        jnp.log(jnp.diag(cholesky))
    )
    mahalanobis = jnp.asarray(residual_dinv_residual, dtype=dtype) - jnp.sum(solved**2)
    prior = _sigma_prior(u, jnp.asarray(prior_scale, dtype=dtype))
    constant = jnp.asarray(
        0.5 * observation_count * math.log(2.0 * math.pi) + shared_prior,
        dtype=dtype,
    )
    total = constant + 0.5 * logdet + 0.5 * mahalanobis + prior
    return jnp.stack([total, logdet, mahalanobis, prior])


def make_fullfactor_components(
    unit_factor,
    cov_diag,
    residual,
    *,
    eigenvalues,
    gp_jitter,
    prior_scale,
    shared_prior: float = 0.0,
) -> dict[str, Callable]:
    """Build scalar component functions for a small explicit-factor test."""

    arguments = (
        np.asarray(unit_factor),
        np.asarray(cov_diag),
        np.asarray(residual),
        np.asarray(eigenvalues),
        float(gp_jitter),
        float(prior_scale),
        float(shared_prior),
    )
    return {
        name: (
            lambda u, index=index: _fullfactor_component_vector(u, *arguments)[index]
        )
        for index, name in enumerate(COMPONENTS)
    }


def make_reduced_components(
    gram,
    rhs_unit,
    residual_dinv_residual,
    logdet_diagonal,
    eigenvalues,
    gp_jitter,
    prior_scale,
    *,
    shared_prior: float = 0.0,
    observation_count: int | None = None,
) -> dict[str, Callable]:
    """Build scalar component functions for a sufficient-statistic test."""

    if observation_count is None:
        observation_count = 0
    arguments = (
        np.asarray(gram),
        np.asarray(rhs_unit),
        float(residual_dinv_residual),
        float(logdet_diagonal),
        np.asarray(eigenvalues),
        float(gp_jitter),
        float(prior_scale),
        float(shared_prior),
        int(observation_count),
    )
    return {
        name: (lambda u, index=index: _reduced_component_vector(u, *arguments)[index])
        for index, name in enumerate(COMPONENTS)
    }


def _contextual_jit(function, *, dtype, matmul_precision):
    """JIT one callable while restoring both precision contexts afterward."""

    use_x64 = np.dtype(dtype) == np.dtype(np.float64)
    wrapped = jax.jit(function)
    if matmul_precision is not None:
        wrapped = jax.default_matmul_precision(matmul_precision)(wrapped)
    wrapped = jax.enable_x64(use_x64)(wrapped)
    return wrapped


def evaluate_scalar_reverse(
    function,
    points,
    *,
    dtype,
    matmul_precision,
) -> dict[str, np.ndarray]:
    """Evaluate one scalar with reverse-mode differentiation at each point."""

    np_dtype = np.dtype(dtype)
    compiled = _contextual_jit(
        jax.value_and_grad(function),
        dtype=np_dtype,
        matmul_precision=matmul_precision,
    )
    values: list[float] = []
    gradients: list[float] = []
    for point in np.asarray(points):
        value, gradient = compiled(np.asarray(point, dtype=np_dtype))
        gradient.block_until_ready()
        values.append(np.asarray(jax.device_get(value)).item())
        gradients.append(np.asarray(jax.device_get(gradient)).item())
    return {
        "value": np.asarray(values, dtype=np_dtype),
        "gradient": np.asarray(gradients, dtype=np_dtype),
    }


def evaluate_scalar_forward_jvp(
    function,
    points,
    *,
    dtype,
    matmul_precision,
) -> dict[str, np.ndarray]:
    """Evaluate one scalar with a pure forward JVP at each point."""

    np_dtype = np.dtype(dtype)

    def forward(coordinate):
        return jax.jvp(function, (coordinate,), (jnp.ones_like(coordinate),))

    compiled = _contextual_jit(
        forward,
        dtype=np_dtype,
        matmul_precision=matmul_precision,
    )
    values: list[float] = []
    tangents: list[float] = []
    for point in np.asarray(points):
        value, tangent = compiled(np.asarray(point, dtype=np_dtype))
        tangent.block_until_ready()
        values.append(np.asarray(jax.device_get(value)).item())
        tangents.append(np.asarray(jax.device_get(tangent)).item())
    return {
        "value": np.asarray(values, dtype=np_dtype),
        "tangent": np.asarray(tangents, dtype=np_dtype),
    }


def _finite_or_none(value) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _max_abs_or_none(values) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    return float(np.max(np.abs(values)))


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _validate_configuration(args: argparse.Namespace) -> dict:
    """Validate the frozen target, v7 evidence, and a fresh v8 output."""

    if args.x64:
        raise ValueError("The frozen full target must remain float32.")
    if args.seed != 0 or args.gram_chunk_size <= 0:
        raise ValueError("The frozen seed or Gram chunk size is invalid.")
    if not math.isclose(args.evaluation_sigma, EVALUATION_SIGMA, abs_tol=0.0):
        raise ValueError("evaluation-sigma must reproduce v7 exactly.")
    if not math.isclose(args.u_step, U_STEP, abs_tol=0.0):
        raise ValueError("u-step must reproduce v7 exactly.")
    if not math.isclose(args.potential_atol, POTENTIAL_ATOL, abs_tol=0.0):
        raise ValueError("potential-atol must reproduce the frozen replay gate.")

    legacy_args = argparse.Namespace(
        init_from=args.init_from,
        free_samples=args.free_samples,
        free_diagnostics=args.free_diagnostics,
        fixed_samples=args.fixed_samples,
        fixed_diagnostics=args.fixed_diagnostics,
        initial_replay=args.initial_replay,
        v5_summary=args.v5_summary,
        v5_arrays=args.v5_arrays,
        v5_failed=args.v5_failed,
        v6_dir=args.v6_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        evaluation_sigma=args.evaluation_sigma,
        u_step=args.u_step,
        repeat_count=5,
        gram_chunk_size=args.gram_chunk_size,
        potential_atol=args.potential_atol,
        x64=args.x64,
    )
    validation = v7._validate_configuration(legacy_args)

    v7_dir = Path(args.v7_dir)
    required = {
        "failed": v7_dir / "FAILED",
        "summary": v7_dir / v7.SUMMARY_NAME,
        "arrays": v7_dir / v7.ARRAYS_NAME,
        "checkpoint": v7_dir / v7.CHECKPOINT_NAME,
        "failure_hashes": v7_dir / "failure_outputs.sha256",
        "run_log": v7_dir / "run.log",
        "provenance": v7_dir / "provenance.sha256",
    }
    if not all(path.is_file() for path in required.values()):
        raise FileNotFoundError("The pinned v7 capture is incomplete.")
    if any(
        (v7_dir / name).exists() for name in ("RUNNING", "COMPLETE", "outputs.sha256")
    ):
        raise ValueError("The v7 status is no longer the pinned FAILED capture.")
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    with np.load(required["arrays"], allow_pickle=False) as archive:
        if not (
            summary.get("execution_completed") is True
            and summary.get("calculation_integrity_passed") is False
            and summary.get("derivative_integrity_passed") is False
            and summary.get("configuration", {}).get("fixed_control_index") == 1380
            and np.all(archive["call_returned"])
            and int(archive["checkpoint_sequence"]) == 16
        ):
            raise ValueError("The pinned v7 scientific failure has drifted.")
        expected = {
            "value": np.asarray(archive["standalone_value"], dtype=np.float64),
            "reverse": np.asarray(
                archive["standalone_gradient"][:, SIGMA_INDEX], dtype=np.float64
            ),
            "forward": np.asarray(
                archive["bundle_point_directional_gradient"], dtype=np.float64
            ),
        }

    output_dir = Path(args.out_dir)
    for name in (SUMMARY_NAME, ARRAYS_NAME, CHECKPOINT_NAME):
        path = output_dir / name
        for candidate in (path, Path(f"{path}.tmp")):
            if candidate.exists():
                raise FileExistsError(f"Refusing output collision: {candidate}")
    actual_sha256 = {name: v5.sha256_file(path) for name, path in required.items()}
    if actual_sha256 != V7_EXPECTED_SHA256:
        raise ValueError(
            "The v7 capture SHA256 values do not match the frozen predecessor."
        )
    validation["v7"] = {
        "paths": {name: str(path.resolve()) for name, path in required.items()},
        "sha256": actual_sha256,
        "expected_full_default": expected,
        "fixed_control_index": 1380,
    }
    return validation


RAW_NUMERIC_KEYS = (
    "point_u",
    "point_sigma",
    "source_potential_replay_error",
    "constrained_roundtrip_error",
    "v7_default_value_error",
    "v7_default_reverse_error",
    "v7_default_forward_error",
    "arm_component_value_reverse",
    "arm_component_value_forward",
    "arm_component_value_delta",
    "arm_component_derivative_reverse",
    "arm_component_derivative_forward",
    "arm_reverse_forward_delta",
    "arm_value_fd_gradient_reverse",
    "arm_value_fd_gradient_forward",
    "component_closure_value",
    "component_closure_residual",
    "component_derivative_closure_reverse",
    "component_derivative_closure_forward",
    "factor_input_finite",
    "cov_diag_minimum",
    "shared_prior_potential",
    "observation_count",
    "expected_component_constant",
    "reference_scales",
    "reference_scale_replay_error",
    "reference_factor_replay_error",
    "gp_jitter",
    "sigma_prior_scale",
    "eigenvalues",
    "reduced_unit_gram",
    "reduced_unit_rhs",
    "reduced_residual_dinv_residual",
    "reduced_logdet_diagonal",
)

DERIVED_NUMERIC_KEYS = (
    "arm_component_value_delta",
    "arm_reverse_forward_delta",
    "arm_value_fd_gradient_reverse",
    "arm_value_fd_gradient_forward",
    "component_closure_value",
    "component_closure_residual",
    "component_derivative_closure_reverse",
    "component_derivative_closure_forward",
)


def _initial_arrays(call_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Allocate checkpoint arrays with explicit missing-call sentinels."""

    nan = np.nan
    arrays: dict[str, np.ndarray] = {
        "arm_names": np.asarray(ARM_NAMES),
        "component_names": np.asarray(COMPONENTS),
        "method_names": np.asarray(METHODS),
        "point_labels": np.asarray(POINT_LABELS),
        "coordinate_names": np.asarray(
            [
                *[f"atmosphere_rotated[{index}]" for index in range(7)],
                *[f"A_unconstrained[{index}]" for index in range(4)],
                "u_sigma_log_p",
            ]
        ),
        "point_u": np.full(3, nan),
        "point_sigma": np.full(3, nan),
        "center_flat": np.full(FLAT_DIMENSION, nan),
        "source_sigma": np.asarray(nan),
        "fixed_control_index": np.asarray(-1, dtype=np.int64),
        "source_potential_replay_error": np.asarray(nan),
        "constrained_roundtrip_error": np.asarray(nan),
        "v7_default_value_error": np.asarray(nan),
        "v7_default_reverse_error": np.asarray(nan),
        "v7_default_forward_error": np.asarray(nan),
        "call_names": np.asarray(call_names),
        "call_returned": np.zeros(len(call_names), dtype=bool),
        "call_error_type": np.full(len(call_names), "", dtype="U128"),
        "call_error_message": np.full(len(call_names), "", dtype="U512"),
        "arm_component_value_reverse": np.full(
            (len(ARM_NAMES), 3, len(COMPONENTS)), nan
        ),
        "arm_component_value_forward": np.full(
            (len(ARM_NAMES), 3, len(COMPONENTS)), nan
        ),
        "arm_component_value_delta": np.full((len(ARM_NAMES), 3, len(COMPONENTS)), nan),
        "arm_component_derivative_reverse": np.full(
            (len(ARM_NAMES), 3, len(COMPONENTS)), nan
        ),
        "arm_component_derivative_forward": np.full(
            (len(ARM_NAMES), 3, len(COMPONENTS)), nan
        ),
        "arm_reverse_forward_delta": np.full((len(ARM_NAMES), 3, len(COMPONENTS)), nan),
        "arm_value_fd_gradient_reverse": np.full(
            (len(ARM_NAMES), len(COMPONENTS)), nan
        ),
        "arm_value_fd_gradient_forward": np.full(
            (len(ARM_NAMES), len(COMPONENTS)), nan
        ),
        "component_closure_value": np.full((len(ARM_NAMES), 3), nan),
        "component_closure_residual": np.full((len(ARM_NAMES), 3), nan),
        "component_derivative_closure_reverse": np.full((len(ARM_NAMES), 3), nan),
        "component_derivative_closure_forward": np.full((len(ARM_NAMES), 3), nan),
        "factor_input_finite": np.asarray(False),
        "factor_shape": np.asarray((-1, -1), dtype=np.int64),
        "factor_dtype": np.asarray(""),
        "factor_sha256": np.asarray(""),
        "cov_diag_minimum": np.asarray(nan),
        "shared_prior_potential": np.asarray(nan),
        "observation_count": np.asarray(-1, dtype=np.int64),
        "expected_component_constant": np.asarray(nan),
        "reference_scales": np.empty(0, dtype=np.float32),
        "reference_scale_replay_error": np.asarray(nan),
        "reference_factor_replay_error": np.asarray(nan),
        "gp_jitter": np.asarray(nan),
        "sigma_prior_scale": np.asarray(nan),
        "eigenvalues": np.empty(0, dtype=np.float32),
        "reduced_unit_gram": np.empty((0, 0), dtype=np.float64),
        "reduced_unit_rhs": np.empty(0, dtype=np.float64),
        "reduced_residual_dinv_residual": np.asarray(nan),
        "reduced_logdet_diagonal": np.asarray(nan),
        "checkpoint_sequence": np.asarray(0, dtype=np.int64),
    }
    _refresh_arrays(arrays)
    return arrays


def _refresh_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Refresh finite masks and derived first-derivative comparisons."""

    reverse = np.asarray(arrays["arm_component_derivative_reverse"])
    forward = np.asarray(arrays["arm_component_derivative_forward"])
    arrays["arm_reverse_forward_delta"] = reverse - forward
    reverse_values = np.asarray(arrays["arm_component_value_reverse"])
    forward_values = np.asarray(arrays["arm_component_value_forward"])
    arrays["arm_component_value_delta"] = reverse_values - forward_values
    point_u = np.asarray(arrays["point_u"])
    if np.all(np.isfinite(point_u)) and point_u[2] > point_u[0]:
        arrays["arm_value_fd_gradient_reverse"] = (
            reverse_values[:, 2] - reverse_values[:, 0]
        ) / (point_u[2] - point_u[0])
        arrays["arm_value_fd_gradient_forward"] = (
            forward_values[:, 2] - forward_values[:, 0]
        ) / (point_u[2] - point_u[0])
    else:
        arrays["arm_value_fd_gradient_reverse"] = np.full_like(
            arrays["arm_value_fd_gradient_reverse"], np.nan, dtype=np.float64
        )
        arrays["arm_value_fd_gradient_forward"] = np.full_like(
            arrays["arm_value_fd_gradient_forward"], np.nan, dtype=np.float64
        )
    closure = reverse_values[:, :, 0] - (
        0.5 * reverse_values[:, :, 1]
        + 0.5 * reverse_values[:, :, 2]
        + reverse_values[:, :, 3]
    )
    arrays["component_closure_value"] = closure
    expected_constant = float(arrays["expected_component_constant"])
    arrays["component_closure_residual"] = closure - expected_constant
    for method, derivative in (
        ("reverse", reverse),
        ("forward", forward),
    ):
        arrays[f"component_derivative_closure_{method}"] = derivative[:, :, 0] - (
            0.5 * derivative[:, :, 1] + 0.5 * derivative[:, :, 2] + derivative[:, :, 3]
        )
    for name in RAW_NUMERIC_KEYS:
        arrays[f"{name}_finite_mask"] = np.isfinite(np.asarray(arrays[name]))


def _checkpoint_payload(stage: str, arrays: dict[str, np.ndarray]) -> dict:
    """Build strict checkpoint metadata without serializing raw nonfinite data."""

    returned = np.asarray(arrays["call_returned"], dtype=bool)
    return {
        "schema_version": 1,
        "mode": "m8_v8_free_sigma_lowrank_precision_checkpoint",
        "stage": stage,
        "checkpoint_sequence": int(arrays["checkpoint_sequence"]),
        "call_names": arrays["call_names"],
        "call_returned": returned,
        "returned_count": int(np.count_nonzero(returned)),
        "planned_count": int(returned.size),
        "raw_nonfinite_count": {
            name: int(np.count_nonzero(~np.isfinite(np.asarray(arrays[name]))))
            for name in RAW_NUMERIC_KEYS
        },
    }


def _write_checkpoint(
    stage: str,
    arrays_path: Path,
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Atomically save every completed call and its exact finite masks."""

    _refresh_arrays(arrays)
    arrays["checkpoint_sequence"] = np.asarray(
        int(arrays["checkpoint_sequence"]) + 1, dtype=np.int64
    )
    v5._write_npz_atomic(arrays_path, arrays)
    v5._write_json_atomic(checkpoint_path, _checkpoint_payload(stage, arrays))


def _capture_call(
    name: str,
    thunk,
    assign,
    *,
    call_index: dict[str, int],
    arrays: dict[str, np.ndarray],
    arrays_path: Path,
    checkpoint_path: Path,
) -> None:
    """Capture one call, including a returned NaN, without losing the artifact."""

    index = call_index[name]
    print(f"[m8_v8_lowrank_precision] phase={name}", flush=True)
    try:
        assign(thunk())
        arrays["call_returned"][index] = True
    except Exception as error:  # Persist the exact failure before continuing.
        arrays["call_error_type"][index] = type(error).__name__
        arrays["call_error_message"][index] = str(error)[:512]
        print(
            f"[m8_v8_lowrank_precision] call failed: {name}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
    _write_checkpoint(name, arrays_path, checkpoint_path, arrays)


def _run_component_arm(
    *,
    arm: str,
    functions: dict[str, Callable],
    points: np.ndarray,
    arrays: dict[str, np.ndarray],
    call_index: dict[str, int],
    arrays_path: Path,
    checkpoint_path: Path,
) -> None:
    """Run reverse and pure-forward evaluations for every LowRank component."""

    arm_index = ARM_NAMES.index(arm)
    dtype, precision = ARM_PRECISION[arm]
    for component_index, component in enumerate(COMPONENTS):
        function = functions[component]
        for method in METHODS:
            name = f"{arm}:{component}:{method}"
            evaluator = (
                evaluate_scalar_reverse
                if method == "reverse"
                else evaluate_scalar_forward_jvp
            )

            def assign(result, ci=component_index, method_name=method):
                value_target = (
                    "arm_component_value_reverse"
                    if method_name == "reverse"
                    else "arm_component_value_forward"
                )
                derivative_target = (
                    "arm_component_derivative_reverse"
                    if method_name == "reverse"
                    else "arm_component_derivative_forward"
                )
                arrays[value_target][arm_index, :, ci] = result["value"]
                arrays[derivative_target][arm_index, :, ci] = result.get(
                    "gradient", result.get("tangent")
                )

            _capture_call(
                name,
                lambda fn=function, ev=evaluator: ev(
                    fn,
                    points,
                    dtype=dtype,
                    matmul_precision=precision,
                ),
                assign,
                call_index=call_index,
                arrays=arrays,
                arrays_path=arrays_path,
                checkpoint_path=checkpoint_path,
            )


def classify_precision_result(arrays: dict[str, np.ndarray]) -> str:
    """Classify which precision or graph boundary restores score consistency."""

    delta = np.asarray(arrays["arm_reverse_forward_delta"], dtype=np.float64)

    def arm_good(name: str) -> bool:
        values = delta[ARM_NAMES.index(name), :, 0]
        return bool(
            np.all(np.isfinite(values)) and np.max(np.abs(values)) <= SCORE_ATOL
        )

    if arm_good("full_f32_default"):
        return "full_production_score_consistent"
    if arm_good("full_f32_highest"):
        return "full_highest_precision_recovers_score"
    if arm_good("isolated_fullfactor_f32_default"):
        return "isolating_fullfactor_from_model_graph_recovers_score"
    if arm_good("isolated_fullfactor_f32_highest"):
        return "isolated_fullfactor_highest_precision_recovers_score"
    if arm_good("reduced_stats_f64_highest"):
        return "only_reduced_float64_control_is_score_consistent"
    if np.any(~np.isfinite(delta[:, :, 0])):
        return "one_or_more_score_paths_nonfinite"
    return "score_disagreement_unresolved"


def _method_contract_passed(
    arrays: dict[str, np.ndarray], *, all_calls_returned: bool
) -> bool:
    """Check capture mechanics without treating score disagreement as failure."""

    expected_constant = 0.5 * int(arrays["observation_count"]) * math.log(
        2.0 * math.pi
    ) + float(arrays["shared_prior_potential"])
    factor_hash = str(arrays["factor_sha256"])
    expected_center_u = np.float32(math.log(EVALUATION_SIGMA))
    expected_u_step = np.float32(U_STEP)
    expected_point_u = np.asarray(
        [
            expected_center_u - expected_u_step,
            expected_center_u,
            expected_center_u + expected_u_step,
        ],
        dtype=np.float32,
    ).astype(np.float64)
    expected_point_sigma = np.exp(expected_point_u)
    expected_scales = np.asarray(
        jax.device_get(
            _sigma_scales(
                jnp.asarray(expected_center_u, dtype=jnp.float32),
                jnp.asarray(arrays["eigenvalues"], dtype=jnp.float32),
                jnp.asarray(arrays["gp_jitter"], dtype=jnp.float32),
            )
        ),
        dtype=np.float32,
    )
    return bool(
        all_calls_returned
        and int(arrays["fixed_control_index"]) == 1380
        and np.all(np.isfinite(arrays["center_flat"]))
        and np.all(np.isfinite(arrays["point_u"]))
        and np.all(np.isfinite(arrays["point_sigma"]))
        and np.all(arrays["point_sigma"] > 0.0)
        and np.array_equal(arrays["point_u"], expected_point_u)
        and np.array_equal(arrays["point_sigma"], expected_point_sigma)
        and np.float32(arrays["center_flat"][SIGMA_INDEX]) == expected_center_u
        and math.isfinite(float(arrays["source_potential_replay_error"]))
        and abs(float(arrays["source_potential_replay_error"])) <= POTENTIAL_ATOL
        and math.isfinite(float(arrays["constrained_roundtrip_error"]))
        and abs(float(arrays["constrained_roundtrip_error"])) <= ROUNDTRIP_ATOL
        and bool(arrays["factor_input_finite"])
        and tuple(arrays["factor_shape"]) == (57344, 767)
        and str(arrays["factor_dtype"]) == "float32"
        and len(factor_hash) == 64
        and all(character in "0123456789abcdef" for character in factor_hash)
        and math.isfinite(float(arrays["cov_diag_minimum"]))
        and float(arrays["cov_diag_minimum"]) > 0.0
        and arrays["reference_scales"].shape == (767,)
        and np.all(np.isfinite(arrays["reference_scales"]))
        and np.all(arrays["reference_scales"] > 0.0)
        and np.array_equal(arrays["reference_scales"], expected_scales)
        and float(arrays["reference_scale_replay_error"]) == 0.0
        and float(arrays["reference_factor_replay_error"]) == 0.0
        and math.isfinite(float(arrays["gp_jitter"]))
        and float(arrays["gp_jitter"]) > 0.0
        and float(arrays["sigma_prior_scale"]) == 0.3
        and arrays["eigenvalues"].shape == (767,)
        and np.all(np.isfinite(arrays["eigenvalues"]))
        and np.all(arrays["eigenvalues"] >= 0.0)
        and arrays["reduced_unit_gram"].shape == (767, 767)
        and np.all(np.isfinite(arrays["reduced_unit_gram"]))
        and arrays["reduced_unit_rhs"].shape == (767,)
        and np.all(np.isfinite(arrays["reduced_unit_rhs"]))
        and math.isfinite(float(arrays["reduced_residual_dinv_residual"]))
        and math.isfinite(float(arrays["reduced_logdet_diagonal"]))
        and int(arrays["observation_count"]) == 57344
        and math.isfinite(float(arrays["shared_prior_potential"]))
        and math.isfinite(float(arrays["expected_component_constant"]))
        and math.isclose(
            float(arrays["expected_component_constant"]),
            expected_constant,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )
    )


def _run_gpu_diagnostic(
    args: argparse.Namespace, validation: dict
) -> tuple[dict, dict]:
    """Run the five-arm first-derivative precision diagnostic."""

    started = time.perf_counter()
    jax.config.update("jax_enable_x64", False)
    rng_key = jax.random.PRNGKey(args.seed)
    output_dir = Path(args.out_dir)
    arrays_path = output_dir / ARRAYS_NAME
    checkpoint_path = output_dir / CHECKPOINT_NAME
    call_names = planned_call_names()
    call_index = {name: index for index, name in enumerate(call_names)}
    arrays = _initial_arrays(call_names)
    _write_checkpoint("allocated", arrays_path, checkpoint_path, arrays)

    print("[m8_v8_lowrank_precision] phase=build_target", flush=True)
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
        return {name: transforms[name](unconstrained[name]) for name in SITE_ORDER}

    def flat_potential(flat):
        return potential_fn(unpack_flat(flat))

    fixed_state = v6._load_fixed_state(args)
    source_flat = np.asarray(fixed_state.flat, dtype=np.float32)
    center_flat = source_flat.copy()
    center_flat[SIGMA_INDEX] = np.float32(math.log(args.evaluation_sigma))
    point_flat = np.repeat(center_flat[None, :], 3, axis=0)
    point_flat[0, SIGMA_INDEX] -= np.float32(args.u_step)
    point_flat[2, SIGMA_INDEX] += np.float32(args.u_step)
    point_u = point_flat[:, SIGMA_INDEX].copy()
    arrays["center_flat"] = center_flat
    arrays["point_u"] = point_u.astype(np.float64)
    arrays["point_sigma"] = np.exp(point_u.astype(np.float64))
    arrays["source_sigma"] = np.asarray(fixed_state.sigma)
    arrays["fixed_control_index"] = np.asarray(fixed_state.index, dtype=np.int64)

    constrained_source = constrain_flat(jnp.asarray(source_flat))
    constrained_vector = np.concatenate(
        [
            np.asarray(jax.device_get(constrained_source["atmosphere_rotated"])),
            np.asarray(jax.device_get(constrained_source["A"])),
            np.asarray([jax.device_get(constrained_source["sigma_log_p"])]),
        ]
    )
    arrays["constrained_roundtrip_error"] = np.asarray(
        np.max(np.abs(constrained_vector - fixed_state.expected_constrained))
    )
    source_value = jax.jit(flat_potential)(jnp.asarray(source_flat))
    source_value.block_until_ready()
    source_prior = v5.sigma_u_prior_terms(
        fixed_state.sigma, wrapper_args.sigma_log_p_scale
    )["potential"]
    arrays["source_potential_replay_error"] = np.asarray(
        float(jax.device_get(source_value))
        - float(source_prior)
        - fixed_state.stored_potential
    )
    _write_checkpoint("target_replayed", arrays_path, checkpoint_path, arrays)

    for arm in FULL_ARMS:
        arm_index = ARM_NAMES.index(arm)
        dtype, precision = ARM_PRECISION[arm]

        def scalar_u_potential(u):
            return flat_potential(jnp.asarray(center_flat).at[SIGMA_INDEX].set(u))

        for method in METHODS:
            name = f"{arm}:total:{method}"
            evaluator = (
                evaluate_scalar_reverse
                if method == "reverse"
                else evaluate_scalar_forward_jvp
            )

            def assign_full(result, method_name=method):
                value_target = (
                    "arm_component_value_reverse"
                    if method_name == "reverse"
                    else "arm_component_value_forward"
                )
                arrays[value_target][arm_index, :, 0] = result["value"]
                target = (
                    "arm_component_derivative_reverse"
                    if method_name == "reverse"
                    else "arm_component_derivative_forward"
                )
                arrays[target][arm_index, :, 0] = result.get(
                    "gradient", result.get("tangent")
                )

            _capture_call(
                name,
                lambda ev=evaluator, fn=scalar_u_potential: ev(
                    fn, point_u, dtype=dtype, matmul_precision=precision
                ),
                assign_full,
                call_index=call_index,
                arrays=arrays,
                arrays_path=arrays_path,
                checkpoint_path=checkpoint_path,
            )
        jax.clear_caches()
        gc.collect()

    print("[m8_v8_lowrank_precision] phase=extract_fullfactor", flush=True)

    def observation_arrays(flat):
        unconstrained = unpack_flat(flat)
        constrained_values = constrain_flat(flat)
        trace = handlers.trace(
            handlers.substitute(
                handlers.seed(free_model, rng_key), data=constrained_values
            )
        ).get_trace()
        shared_log_prior_z = jnp.asarray(0.0)
        for name in ("atmosphere_rotated", "A"):
            value = constrained_values[name]
            shared_log_prior_z = (
                shared_log_prior_z
                + jnp.sum(trace[name]["fn"].log_prob(value))
                + jnp.sum(
                    transforms[name].log_abs_det_jacobian(unconstrained[name], value)
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

    observation_jit = jax.jit(observation_arrays)
    loc, factor, cov_diag, observed, shared_prior = observation_jit(
        jnp.asarray(center_flat)
    )
    factor.block_until_ready()
    factor_host = np.asarray(jax.device_get(factor), dtype=np.float32)
    cov_diag_host = np.asarray(jax.device_get(cov_diag), dtype=np.float32)
    residual_host = np.asarray(jax.device_get(observed - loc), dtype=np.float32)
    shared_prior_host = float(jax.device_get(shared_prior))
    del loc, factor, cov_diag, observed, shared_prior, observation_jit
    jax.clear_caches()
    gc.collect()

    geometry = retrieval.build_luhman16b_geometry(nside=wrapper_args.nside)
    pressure_gp = retrieval.build_fixed_pressure_gp_eigendecomposition(
        geometry.distance_matrix,
        wrapper_args.fixed_ell_b,
        theta=geometry.theta,
        phi=geometry.phi,
    )
    eigenvalues = np.asarray(
        jax.device_get(pressure_gp["eigenvalues"]), dtype=np.float32
    )
    gp_jitter = float(wrapper_args.gp_jitter)
    center_u_device = jnp.asarray(point_u[1], dtype=jnp.float32)
    eigenvalues_device = jnp.asarray(eigenvalues, dtype=jnp.float32)
    reference_scales_device = _sigma_scales(
        center_u_device,
        eigenvalues_device,
        jnp.asarray(gp_jitter, dtype=jnp.float32),
    )
    reference_scales_device.block_until_ready()
    reference_scales = np.asarray(
        jax.device_get(reference_scales_device), dtype=np.float32
    )
    replay_scales_device = _sigma_scales(
        center_u_device,
        eigenvalues_device,
        jnp.asarray(gp_jitter, dtype=jnp.float32),
    )
    replay_scales_device.block_until_ready()
    replay_scales = np.asarray(jax.device_get(replay_scales_device), dtype=np.float32)
    scale_replay_error = float(
        np.max(np.abs(replay_scales.astype(np.float64) - reference_scales))
    )
    replay_ratio = replay_scales / reference_scales
    if np.array_equal(replay_scales, reference_scales):
        factor_replay_error = 0.0
    else:
        column_maximum = np.max(np.abs(factor_host), axis=0).astype(np.float64)
        factor_replay_error = float(
            np.max(column_maximum * np.abs(replay_ratio.astype(np.float64) - 1.0))
        )
    arrays["factor_input_finite"] = np.asarray(
        np.all(np.isfinite(factor_host))
        and np.all(np.isfinite(cov_diag_host))
        and np.all(np.isfinite(residual_host))
        and np.all(cov_diag_host > 0.0)
    )
    arrays["factor_shape"] = np.asarray(factor_host.shape, dtype=np.int64)
    arrays["factor_dtype"] = np.asarray(str(factor_host.dtype))
    arrays["factor_sha256"] = np.asarray(_sha256_array(factor_host))
    arrays["cov_diag_minimum"] = np.asarray(np.min(cov_diag_host), dtype=np.float64)
    arrays["shared_prior_potential"] = np.asarray(shared_prior_host, dtype=np.float64)
    arrays["observation_count"] = np.asarray(factor_host.shape[0], dtype=np.int64)
    arrays["expected_component_constant"] = np.asarray(
        0.5 * factor_host.shape[0] * math.log(2.0 * math.pi) + shared_prior_host,
        dtype=np.float64,
    )
    arrays["reference_scales"] = reference_scales
    arrays["reference_scale_replay_error"] = np.asarray(scale_replay_error)
    arrays["reference_factor_replay_error"] = np.asarray(factor_replay_error)
    arrays["gp_jitter"] = np.asarray(gp_jitter, dtype=np.float64)
    arrays["sigma_prior_scale"] = np.asarray(
        wrapper_args.sigma_log_p_scale, dtype=np.float64
    )
    arrays["eigenvalues"] = eigenvalues

    statistics = v6.build_unit_lowrank_statistics(
        factor_host,
        reference_scales,
        cov_diag_host,
        residual_host,
        chunk_size=args.gram_chunk_size,
    )
    arrays["reduced_unit_gram"] = np.asarray(statistics["unit_gram"])
    arrays["reduced_unit_rhs"] = np.asarray(statistics["unit_rhs"])
    arrays["reduced_residual_dinv_residual"] = np.asarray(
        statistics["residual_dinv_residual"]
    )
    arrays["reduced_logdet_diagonal"] = np.asarray(statistics["logdet_diagonal"])
    _write_checkpoint("factor_extracted", arrays_path, checkpoint_path, arrays)

    reference_arguments = (
        factor_host,
        reference_scales,
        cov_diag_host,
        residual_host,
        eigenvalues,
        gp_jitter,
        float(wrapper_args.sigma_log_p_scale),
        shared_prior_host,
    )
    reference_functions = {
        name: (
            lambda u, index=index: _reference_fullfactor_component_vector(
                u, *reference_arguments
            )[index]
        )
        for index, name in enumerate(COMPONENTS)
    }
    for arm in (
        "isolated_fullfactor_f32_default",
        "isolated_fullfactor_f32_highest",
    ):
        _run_component_arm(
            arm=arm,
            functions=reference_functions,
            points=point_u,
            arrays=arrays,
            call_index=call_index,
            arrays_path=arrays_path,
            checkpoint_path=checkpoint_path,
        )
        jax.clear_caches()
        gc.collect()

    print("[m8_v8_lowrank_precision] phase=reduced_float64_control", flush=True)
    reduced_functions = make_reduced_components(
        statistics["unit_gram"],
        statistics["unit_rhs"],
        statistics["residual_dinv_residual"],
        statistics["logdet_diagonal"],
        eigenvalues=eigenvalues.astype(np.float64),
        gp_jitter=gp_jitter,
        prior_scale=float(wrapper_args.sigma_log_p_scale),
        shared_prior=shared_prior_host,
        observation_count=factor_host.shape[0],
    )
    _run_component_arm(
        arm="reduced_stats_f64_highest",
        functions=reduced_functions,
        points=point_u.astype(np.float64),
        arrays=arrays,
        call_index=call_index,
        arrays_path=arrays_path,
        checkpoint_path=checkpoint_path,
    )
    returned = bool(np.all(arrays["call_returned"]))
    v7_expected = validation["v7"]["expected_full_default"]
    default_index = ARM_NAMES.index("full_f32_default")
    default_value_error = _max_abs_or_none(
        arrays["arm_component_value_reverse"][default_index, :, 0]
        - v7_expected["value"]
    )
    default_reverse_error = _max_abs_or_none(
        arrays["arm_component_derivative_reverse"][default_index, :, 0]
        - v7_expected["reverse"]
    )
    default_forward_error = _max_abs_or_none(
        arrays["arm_component_derivative_forward"][default_index, :, 0]
        - v7_expected["forward"]
    )
    arrays["v7_default_value_error"] = np.asarray(
        np.nan if default_value_error is None else default_value_error
    )
    arrays["v7_default_reverse_error"] = np.asarray(
        np.nan if default_reverse_error is None else default_reverse_error
    )
    arrays["v7_default_forward_error"] = np.asarray(
        np.nan if default_forward_error is None else default_forward_error
    )
    _write_checkpoint("final", arrays_path, checkpoint_path, arrays)
    method_contract = _method_contract_passed(arrays, all_calls_returned=returned)
    classification = classify_precision_result(arrays)
    reverse_forward_max = {
        arm: _max_abs_or_none(
            arrays["arm_reverse_forward_delta"][ARM_NAMES.index(arm), :, 0]
        )
        for arm in ARM_NAMES
    }
    summary = {
        "schema_version": 1,
        "mode": "m8_v8_free_sigma_lowrank_precision",
        "execution_completed": returned,
        "method_contract_passed": method_contract,
        "artifact_integrity_passed": method_contract,
        "scientific_agreement_is_not_a_completion_gate": True,
        "classification": classification,
        "purpose": (
            "Separate full-model graph effects, explicit full-factor LowRank "
            "reductions, and a reduced float64 control without using HVPs."
        ),
        "configuration": {
            "fixed_control_index": fixed_state.index,
            "source_sigma_log_p": fixed_state.sigma,
            "evaluation_sigma_requested": args.evaluation_sigma,
            "evaluation_sigma_effective": float(arrays["point_sigma"][1]),
            "u_step_requested": args.u_step,
            "u_step_effective_minus": float(point_u[1] - point_u[0]),
            "u_step_effective_plus": float(point_u[2] - point_u[1]),
            "seed": args.seed,
            "full_target_x64": False,
            "score_atol": SCORE_ATOL,
            "potential_atol": POTENTIAL_ATOL,
        },
        "method": {
            "arms": ARM_NAMES,
            "components": COMPONENTS,
            "methods": METHODS,
            "reverse_method": "jax.value_and_grad(scalar_fn)",
            "forward_method": "jax.jvp(scalar_fn, unit_u_tangent)",
            "matmul_precision_by_arm": {
                arm: (
                    "inherited_unset"
                    if ARM_PRECISION[arm][1] is None
                    else ARM_PRECISION[arm][1]
                )
                for arm in ARM_NAMES
            },
            "component_forward_calls_are_scalar": True,
            "hvp_computed": False,
            "full_hessian_computed": False,
            "fallback_allowed": False,
            "fullfactor_f64_arm_computed": False,
            "selective_x64_scope": (
                "reduced 767x767 LowRank statistics and scalar sigma algebra only"
            ),
        },
        "integrity": {
            "source_potential_replay_error": float(
                arrays["source_potential_replay_error"]
            ),
            "constrained_roundtrip_error": float(arrays["constrained_roundtrip_error"]),
            "factor_input_finite": bool(arrays["factor_input_finite"]),
            "factor_shape": arrays["factor_shape"],
            "factor_dtype": str(arrays["factor_dtype"]),
            "factor_sha256": str(arrays["factor_sha256"]),
            "cov_diag_minimum": float(arrays["cov_diag_minimum"]),
            "reference_scale_replay_error": float(
                arrays["reference_scale_replay_error"]
            ),
            "reference_factor_replay_error": float(
                arrays["reference_factor_replay_error"]
            ),
            "gp_jitter": float(arrays["gp_jitter"]),
            "sigma_prior_scale": float(arrays["sigma_prior_scale"]),
            "reduced_statistics_source": "current_center_factor_float32",
            "observation_count": int(arrays["observation_count"]),
            "shared_prior_potential": float(arrays["shared_prior_potential"]),
            "expected_component_constant": float(arrays["expected_component_constant"]),
            "full_default_vs_v7_value_max_abs": default_value_error,
            "full_default_vs_v7_reverse_max_abs": default_reverse_error,
            "full_default_vs_v7_forward_max_abs": default_forward_error,
        },
        "findings": {
            "classification": classification,
            "reverse_forward_total_max_abs_by_arm": reverse_forward_max,
            "reverse_total_center_by_arm": {
                arm: _finite_or_none(
                    arrays["arm_component_derivative_reverse"][
                        ARM_NAMES.index(arm), 1, 0
                    ]
                )
                for arm in ARM_NAMES
            },
            "forward_total_center_by_arm": {
                arm: _finite_or_none(
                    arrays["arm_component_derivative_forward"][
                        ARM_NAMES.index(arm), 1, 0
                    ]
                )
                for arm in ARM_NAMES
            },
            "value_fd_total_by_arm": {
                arm: _finite_or_none(
                    arrays["arm_value_fd_gradient_reverse"][ARM_NAMES.index(arm), 0]
                )
                for arm in ARM_NAMES
            },
            "component_reverse_forward_center": {
                arm: {
                    component: _finite_or_none(
                        arrays["arm_reverse_forward_delta"][
                            ARM_NAMES.index(arm), 1, COMPONENTS.index(component)
                        ]
                    )
                    for component in COMPONENTS
                }
                for arm in COMPONENT_ARMS
            },
        },
        "call_status": {
            "planned_count": len(call_names),
            "returned_count": int(np.count_nonzero(arrays["call_returned"])),
            "all_returned": returned,
            "returned_nonfinite_is_a_finding": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpyro": numpyro.__version__,
            "exojax": exojax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "jax_enable_x64_after_selective_arm": bool(jax.config.x64_enabled),
            "default_matmul_precision_after_arms": (
                jax.config.jax_default_matmul_precision
            ),
            "exojax_path": str(Path(inspect.getfile(exojax)).resolve()),
            "numpyro_path": str(Path(inspect.getfile(numpyro)).resolve()),
        },
        "predecessor": validation["v7"]["sha256"],
        "timing_seconds": {"total": time.perf_counter() - started},
        "limitations": [
            "This is a one-state numerical diagnostic, not posterior sampling.",
            "The explicit-factor arms freeze the factor at the center and rescale columns.",
            "The reduced float64 arm tests feasibility, not a production mixed-precision target.",
            "A COMPLETE sentinel means the capture contract passed, not that gradients agreed.",
        ],
    }
    return summary, arrays


def _load_and_validate_checkpoint(
    arrays_path: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load one checkpoint and verify masks, schedule, and JSON agreement."""

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        *RAW_NUMERIC_KEYS,
        *[f"{name}_finite_mask" for name in RAW_NUMERIC_KEYS],
        "call_names",
        "call_returned",
        "checkpoint_sequence",
        "arm_names",
        "component_names",
        "method_names",
        "point_labels",
        "fixed_control_index",
        "factor_shape",
        "factor_dtype",
        "factor_sha256",
        "center_flat",
        "source_sigma",
        "coordinate_names",
        "call_error_type",
        "call_error_message",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Missing checkpoint arrays: {missing}")
    for name in RAW_NUMERIC_KEYS:
        if arrays[name].dtype.kind not in "biuf":
            raise ValueError(f"Raw array {name} must have a numeric dtype.")
        mask = arrays[f"{name}_finite_mask"]
        if mask.dtype.kind != "b" or mask.shape != arrays[name].shape:
            raise ValueError(f"Finite mask shape or dtype mismatch for {name}.")
        if not np.array_equal(mask, np.isfinite(arrays[name])):
            raise ValueError(f"Finite mask mismatch for {name}.")
    expected_calls = np.asarray(planned_call_names())
    if not np.array_equal(arrays["call_names"], expected_calls):
        raise ValueError("Call schedule mismatch.")
    shapes = {
        "point_u": (3,),
        "point_sigma": (3,),
        "center_flat": (FLAT_DIMENSION,),
        "source_sigma": (),
        "fixed_control_index": (),
        "source_potential_replay_error": (),
        "constrained_roundtrip_error": (),
        "v7_default_value_error": (),
        "v7_default_reverse_error": (),
        "v7_default_forward_error": (),
        "call_names": (len(expected_calls),),
        "call_returned": (len(expected_calls),),
        "call_error_type": (len(expected_calls),),
        "call_error_message": (len(expected_calls),),
        "arm_component_value_reverse": (len(ARM_NAMES), 3, len(COMPONENTS)),
        "arm_component_value_forward": (len(ARM_NAMES), 3, len(COMPONENTS)),
        "arm_component_value_delta": (len(ARM_NAMES), 3, len(COMPONENTS)),
        "arm_component_derivative_reverse": (
            len(ARM_NAMES),
            3,
            len(COMPONENTS),
        ),
        "arm_component_derivative_forward": (
            len(ARM_NAMES),
            3,
            len(COMPONENTS),
        ),
        "arm_reverse_forward_delta": (len(ARM_NAMES), 3, len(COMPONENTS)),
        "arm_value_fd_gradient_reverse": (len(ARM_NAMES), len(COMPONENTS)),
        "arm_value_fd_gradient_forward": (len(ARM_NAMES), len(COMPONENTS)),
        "component_closure_value": (len(ARM_NAMES), 3),
        "component_closure_residual": (len(ARM_NAMES), 3),
        "component_derivative_closure_reverse": (len(ARM_NAMES), 3),
        "component_derivative_closure_forward": (len(ARM_NAMES), 3),
        "factor_input_finite": (),
        "factor_shape": (2,),
        "factor_dtype": (),
        "factor_sha256": (),
        "cov_diag_minimum": (),
        "shared_prior_potential": (),
        "observation_count": (),
        "expected_component_constant": (),
        "reference_scale_replay_error": (),
        "reference_factor_replay_error": (),
        "gp_jitter": (),
        "sigma_prior_scale": (),
        "reduced_residual_dinv_residual": (),
        "reduced_logdet_diagonal": (),
        "checkpoint_sequence": (),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Unexpected shape for {name}: {arrays[name].shape}.")
    optional_extracted_shapes = {
        "reference_scales": {(0,), (767,)},
        "eigenvalues": {(0,), (767,)},
        "reduced_unit_gram": {(0, 0), (767, 767)},
        "reduced_unit_rhs": {(0,), (767,)},
    }
    for name, allowed_shapes in optional_extracted_shapes.items():
        if arrays[name].shape not in allowed_shapes:
            raise ValueError(f"Unexpected shape for {name}: {arrays[name].shape}.")
    expected_coordinates = np.asarray(
        [
            *[f"atmosphere_rotated[{index}]" for index in range(7)],
            *[f"A_unconstrained[{index}]" for index in range(4)],
            "u_sigma_log_p",
        ]
    )
    if not (
        np.array_equal(arrays["arm_names"], np.asarray(ARM_NAMES))
        and np.array_equal(arrays["component_names"], np.asarray(COMPONENTS))
        and np.array_equal(arrays["method_names"], np.asarray(METHODS))
        and np.array_equal(arrays["point_labels"], np.asarray(POINT_LABELS))
        and np.array_equal(arrays["coordinate_names"], expected_coordinates)
    ):
        raise ValueError("Artifact labels do not match the frozen method contract.")

    recomputed = {name: np.array(value, copy=True) for name, value in arrays.items()}
    _refresh_arrays(recomputed)
    for name in (
        *DERIVED_NUMERIC_KEYS,
        *[f"{key}_finite_mask" for key in RAW_NUMERIC_KEYS],
    ):
        if not np.array_equal(arrays[name], recomputed[name], equal_nan=True):
            raise ValueError(f"Derived array mismatch for {name}.")

    returned = np.asarray(arrays["call_returned"], dtype=bool)
    if np.any(returned & (arrays["call_error_type"] != "")) or np.any(
        returned & (arrays["call_error_message"] != "")
    ):
        raise ValueError("A returned call cannot also carry an exception.")
    expected_nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(arrays[name])))
        for name in RAW_NUMERIC_KEYS
    }
    allowed_stages = {
        "allocated",
        "target_replayed",
        "factor_extracted",
        "final",
        *expected_calls.tolist(),
    }
    stage = checkpoint.get("stage")
    if stage == "allocated":
        expected_sequence = 1
    elif stage == "target_replayed":
        expected_sequence = 2
    elif stage == "factor_extracted":
        expected_sequence = 7
    elif stage == "final":
        expected_sequence = len(expected_calls) + 4
    elif stage in expected_calls:
        call_position = expected_calls.tolist().index(stage)
        expected_sequence = (
            call_position + 3 if call_position < 4 else call_position + 4
        )
    else:
        expected_sequence = -1
    if not (
        checkpoint.get("schema_version") == 1
        and checkpoint.get("mode") == "m8_v8_free_sigma_lowrank_precision_checkpoint"
        and stage in allowed_stages
        and checkpoint.get("checkpoint_sequence") == int(arrays["checkpoint_sequence"])
        and int(arrays["checkpoint_sequence"]) == expected_sequence
        and checkpoint.get("call_names") == expected_calls.tolist()
        and checkpoint.get("call_returned") == returned.tolist()
        and checkpoint.get("returned_count") == int(np.count_nonzero(returned))
        and checkpoint.get("planned_count") == returned.size
        and checkpoint.get("raw_nonfinite_count") == expected_nonfinite
    ):
        raise ValueError("Checkpoint JSON and NPZ disagree.")
    return arrays, checkpoint


def validate_checkpoint_artifacts(
    checkpoint_path: Path | str,
    arrays_path: Path | str,
) -> dict:
    """Validate an incomplete or complete atomic checkpoint."""

    arrays, checkpoint = _load_and_validate_checkpoint(
        Path(arrays_path), Path(checkpoint_path)
    )
    return {
        "checkpoint_validation_passed": True,
        "stage": checkpoint.get("stage"),
        "returned_count": int(np.count_nonzero(arrays["call_returned"])),
        "planned_count": int(arrays["call_returned"].size),
    }


def validate_saved_artifacts(
    summary_path: Path | str,
    arrays_path: Path | str,
    checkpoint_path: Path | str,
) -> dict:
    """Recompute the hard artifact contract without requiring score agreement."""

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    arrays, checkpoint = _load_and_validate_checkpoint(
        Path(arrays_path), Path(checkpoint_path)
    )
    returned = bool(np.all(arrays["call_returned"]))
    expected_method = _method_contract_passed(arrays, all_calls_returned=returned)
    method = summary.get("method", {})
    configuration = summary.get("configuration", {})
    runtime = summary.get("runtime", {})
    summary_contract = bool(
        summary.get("schema_version") == 1
        and summary.get("mode") == "m8_v8_free_sigma_lowrank_precision"
        and summary.get("execution_completed") is returned
        and summary.get("method_contract_passed") is expected_method
        and summary.get("artifact_integrity_passed") is expected_method
        and summary.get("classification") == classify_precision_result(arrays)
        and summary.get("scientific_agreement_is_not_a_completion_gate") is True
        and method.get("hvp_computed") is False
        and method.get("full_hessian_computed") is False
        and method.get("fallback_allowed") is False
        and method.get("fullfactor_f64_arm_computed") is False
        and method.get("component_forward_calls_are_scalar") is True
        and method.get("reverse_method") == "jax.value_and_grad(scalar_fn)"
        and method.get("forward_method") == "jax.jvp(scalar_fn, unit_u_tangent)"
        and method.get("selective_x64_scope")
        == "reduced 767x767 LowRank statistics and scalar sigma algebra only"
        and tuple(method.get("arms", ())) == ARM_NAMES
        and tuple(method.get("components", ())) == COMPONENTS
        and tuple(method.get("methods", ())) == METHODS
        and method.get("matmul_precision_by_arm")
        == {
            arm: (
                "inherited_unset"
                if ARM_PRECISION[arm][1] is None
                else ARM_PRECISION[arm][1]
            )
            for arm in ARM_NAMES
        }
        and configuration.get("fixed_control_index") == 1380
        and configuration.get("evaluation_sigma_requested") == EVALUATION_SIGMA
        and configuration.get("u_step_requested") == U_STEP
        and configuration.get("score_atol") == SCORE_ATOL
        and configuration.get("potential_atol") == POTENTIAL_ATOL
        and configuration.get("full_target_x64") is False
        and runtime.get("jax_enable_x64_after_selective_arm") is False
        and runtime.get("default_matmul_precision_after_arms") is None
        and checkpoint.get("stage") == "final"
        and summary.get("predecessor") == V7_EXPECTED_SHA256
    )
    if not summary_contract:
        raise ValueError("Saved artifact contract validation failed.")
    return {
        "artifact_integrity_passed": expected_method,
        "classification": classify_precision_result(arrays),
        "scientific_agreement_required": False,
    }


def main() -> None:
    """Validate, run, checkpoint, and write strict final artifacts."""

    args = parse_args()
    validation = _validate_configuration(args)
    if args.validate_only:
        print(
            json.dumps(
                v5._json_ready(
                    {
                        "validation_passed": True,
                        "mode": "m8_v8_free_sigma_lowrank_precision",
                        "out_dir": str(Path(args.out_dir).resolve()),
                        "arms": ARM_NAMES,
                        "components": COMPONENTS,
                        "methods": METHODS,
                        "evaluation_sigma": args.evaluation_sigma,
                        "u_step": args.u_step,
                        "hvp_computed": False,
                        "scientific_disagreement_is_not_a_failure": True,
                        "v7": validation["v7"]["sha256"],
                    }
                ),
                indent=2,
                allow_nan=False,
            )
        )
        return

    output_dir = Path(args.out_dir)
    summary_path = output_dir / SUMMARY_NAME
    summary, arrays = _run_gpu_diagnostic(args, validation)
    v5._write_npz_atomic(output_dir / ARRAYS_NAME, arrays)
    v5._write_json_atomic(summary_path, summary)
    print(json.dumps(v5._json_ready(summary), indent=2, allow_nan=False), flush=True)
    if summary["execution_completed"] is not True:
        raise SystemExit(4)
    if summary["method_contract_passed"] is not True:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
