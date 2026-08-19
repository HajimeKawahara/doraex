"""Capture the known M8 free-sigma derivative failure without losing NaNs."""

from __future__ import annotations

import argparse
import inspect
import json
import math
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


SUMMARY_NAME = "m8_v7_free_sigma_derivative_capture_summary.json"
ARRAYS_NAME = "m8_v7_free_sigma_derivative_capture_arrays.npz"
CHECKPOINT_NAME = "m8_v7_free_sigma_derivative_capture_checkpoint.json"
EVALUATION_SIGMA = 0.27526917
U_STEP = 0.02
POTENTIAL_ATOL = 5.0e-2
VALUE_ALIGNMENT_ATOL = 5.0e-1
ROUNDTRIP_ATOL = 1.0e-6
HVP_METHOD = "jax.jvp(jax.value_and_grad(full_potential), e_u_sigma)"
POINT_LABELS = ("minus", "center", "plus")
SIGMA_INDEX = v6.SIGMA_INDEX
FLAT_DIMENSION = v6.FLAT_DIMENSION
ATMOSPHERE_SLICE = v6.ATMOSPHERE_SLICE
A_SLICE = v6.A_SLICE
SITE_ORDER = v6.SITE_ORDER


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
        "--v6-dir",
        default=str(ROOT / "results/m8/v6/free_sigma_hvp_geometry"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/m8/v7/free_sigma_derivative_capture"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-sigma", type=float, default=EVALUATION_SIGMA)
    parser.add_argument("--u-step", type=float, default=U_STEP)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--gram-chunk-size", type=int, default=4096)
    parser.add_argument("--potential-atol", type=float, default=POTENTIAL_ATOL)
    parser.add_argument("--no-x64", dest="x64", action="store_false")
    parser.add_argument("--x64", dest="x64", action="store_true")
    parser.set_defaults(x64=False)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def planned_call_names(repeat_count: int) -> tuple[str, ...]:
    """Return the exact derivative-call schedule for one capture."""

    if repeat_count < 2:
        raise ValueError("repeat_count must be at least two.")
    return tuple(
        [
            "minus_standalone",
            "center_standalone",
            "plus_standalone",
        ]
        + [f"center_bundle_{index}" for index in range(repeat_count)]
        + [
            "minus_bundle",
            "plus_bundle",
            "center_verlet",
            "center_scalar",
        ]
    )


def finite_or_none(value) -> float | None:
    """Return a finite scalar or None for strict JSON."""

    result = float(value)
    return result if math.isfinite(result) else None


def max_abs_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return max absolute error only when both complete arrays are finite."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not (
        np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    ):
        return None
    return float(np.max(np.abs(left - right)))


def vector_agreement_or_none(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int | None]:
    """Compare complete finite vectors without promoting partial agreement."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError("Agreement vectors must have one shape.")
    common_finite = np.isfinite(reference) & np.isfinite(candidate)
    result: dict[str, float | int | None] = {
        "component_count": int(reference.size),
        "common_finite_count": int(np.count_nonzero(common_finite)),
        "relative_error": None,
        "cosine": None,
    }
    if not np.all(common_finite):
        return result
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    result["relative_error"] = float(
        np.linalg.norm(candidate - reference) / max(1.0, reference_norm)
    )
    result["cosine"] = float(
        np.dot(reference, candidate)
        / max(1.0e-30, reference_norm * candidate_norm)
    )
    return result


def raw_finite_masks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return exact finite masks for every declared raw numeric array."""

    return {
        f"{name}_finite_mask": np.isfinite(np.asarray(arrays[name]))
        for name in RAW_NUMERIC_KEYS
    }


def classify_derivative_capture(arrays: dict[str, np.ndarray]) -> str:
    """Classify first-order versus higher-order numerical behavior."""

    standalone_gradient = np.asarray(arrays["standalone_gradient"])
    bundle_gradient = np.asarray(arrays["bundle_point_gradient"])
    bundle_directional = np.asarray(arrays["bundle_point_directional_gradient"])
    center_bundle_gradient = np.asarray(arrays["center_bundle_gradient"])
    center_bundle_directional = np.asarray(
        arrays["center_bundle_directional_gradient"]
    )
    verlet_gradient = np.asarray(arrays["verlet_gradient"])
    scalar_gradient = np.asarray(arrays["scalar_gradient_u"])
    bundle_hvp = np.asarray(arrays["center_bundle_hvp"])
    scalar_hessian = np.asarray(arrays["scalar_hessian_uu"])
    analytic_gradient = np.asarray(arrays["analytic_gradient_u"])
    analytic_hessian = np.asarray(arrays["analytic_hessian_uu"])

    standalone_finite = bool(np.all(np.isfinite(standalone_gradient)))
    bundle_finite = bool(
        np.all(np.isfinite(bundle_gradient))
        and np.all(np.isfinite(center_bundle_gradient))
    )
    verlet_finite = bool(np.all(np.isfinite(verlet_gradient)))
    analytic_gradient_finite = bool(np.all(np.isfinite(analytic_gradient)))
    if not standalone_finite or not verlet_finite:
        return "first_order_nonfinite_nuts_relevant"
    if not analytic_gradient_finite:
        return "analytic_first_derivative_reference_nonfinite"
    if not bundle_finite:
        return "hvp_bundle_context_first_order_nonfinite"
    if not (
        np.all(np.isfinite(bundle_directional))
        and np.all(np.isfinite(center_bundle_directional))
        and np.all(np.isfinite(scalar_gradient))
    ):
        return "directional_derivative_path_nonfinite"

    center_standalone = standalone_gradient[1]
    center_bundle = bundle_gradient[1]
    context_error = max(
        float(np.max(np.abs(center_standalone - center_bundle))),
        float(
            np.max(
                np.abs(center_bundle_gradient - center_standalone[None, :])
            )
        ),
        float(np.max(np.abs(center_standalone - verlet_gradient))),
    )
    analytic_error = abs(
        float(center_standalone[SIGMA_INDEX] - analytic_gradient[1])
    )
    if context_error > 2.0e-2 or analytic_error > 5.0e-1:
        return "first_order_context_or_reduction_disagreement"
    directional_error = max(
        float(
            np.max(
                np.abs(
                    bundle_directional - bundle_gradient[:, SIGMA_INDEX]
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    center_bundle_directional
                    - center_bundle_gradient[:, SIGMA_INDEX]
                )
            )
        ),
        abs(float(scalar_gradient - center_standalone[SIGMA_INDEX])),
    )
    if directional_error > 2.0e-2:
        return "directional_derivative_path_disagreement"

    if not np.all(np.isfinite(bundle_hvp)):
        shared_finite = np.all(np.isfinite(bundle_hvp[:, :SIGMA_INDEX]))
        huu_finite = np.all(np.isfinite(bundle_hvp[:, SIGMA_INDEX]))
        if huu_finite and not shared_finite:
            return "cross_column_higher_order_ad_nonfinite"
        return "full_hvp_higher_order_ad_nonfinite"
    if not (
        np.all(np.isfinite(scalar_hessian))
        and np.all(np.isfinite(analytic_hessian))
    ):
        return "scalar_second_order_nonfinite"
    hvp_reference = bundle_hvp[0]
    scalar_relative_error = abs(
        float(hvp_reference[SIGMA_INDEX] - scalar_hessian)
    ) / max(1.0, abs(float(hvp_reference[SIGMA_INDEX])))
    analytic_relative_error = abs(
        float(hvp_reference[SIGMA_INDEX] - analytic_hessian[1])
    ) / max(1.0, abs(float(analytic_hessian[1])))
    fd_agreement = vector_agreement_or_none(
        hvp_reference,
        arrays["standalone_fd_column"],
    )
    repeat_stable = np.array_equal(
        bundle_hvp,
        np.repeat(bundle_hvp[0:1], bundle_hvp.shape[0], axis=0),
    )
    if not (
        repeat_stable
        and scalar_relative_error <= 0.05
        and analytic_relative_error <= 0.20
        and fd_agreement["relative_error"] is not None
        and fd_agreement["relative_error"] <= 0.5
        and fd_agreement["cosine"] is not None
        and fd_agreement["cosine"] >= 0.9
    ):
        return "finite_but_second_order_disagreement"
    return "finite_local_derivative_geometry"


RAW_NUMERIC_KEYS = (
    "center_bundle_value",
    "center_bundle_gradient",
    "center_bundle_directional_gradient",
    "center_bundle_hvp",
    "standalone_value",
    "standalone_gradient",
    "bundle_point_value",
    "bundle_point_gradient",
    "bundle_point_directional_gradient",
    "bundle_point_hvp",
    "verlet_value",
    "verlet_gradient",
    "scalar_gradient_u",
    "scalar_hessian_uu",
    "analytic_negative_log_likelihood",
    "analytic_total_potential",
    "analytic_gradient_u",
    "analytic_hessian_uu",
    "analytic_log_determinant",
    "analytic_mahalanobis",
    "analytic_cholesky_min_diagonal",
    "standalone_fd_column",
    "bundle_fd_column",
    "standalone_value_fd_gradient_u",
    "standalone_value_fd_hessian_uu",
    "bundle_value_fd_gradient_u",
    "bundle_value_fd_hessian_uu",
)


def _validate_configuration(args: argparse.Namespace) -> dict:
    """Validate frozen inputs, the v6 failure, and a fresh v7 output."""

    if args.x64:
        raise ValueError("The frozen production target must remain float32.")
    if args.repeat_count < 2:
        raise ValueError("repeat-count must be at least two.")
    if args.gram_chunk_size <= 0:
        raise ValueError("gram-chunk-size must be positive.")
    if not math.isclose(args.evaluation_sigma, EVALUATION_SIGMA, abs_tol=0.0):
        raise ValueError("evaluation-sigma must reproduce the known v6 marker.")
    if not math.isclose(args.u_step, U_STEP, abs_tol=0.0):
        raise ValueError("u-step must reproduce the known v6 comparison.")
    if not math.isclose(args.potential_atol, POTENTIAL_ATOL, abs_tol=0.0):
        raise ValueError("potential-atol must reproduce the frozen replay gate.")

    legacy_args = argparse.Namespace(
        init_from=args.init_from,
        free_samples=args.free_samples,
        free_diagnostics=args.free_diagnostics,
        fixed_samples=args.fixed_samples,
        fixed_diagnostics=args.fixed_diagnostics,
        initial_replay=args.initial_replay,
        out_dir=args.out_dir,
        seed=args.seed,
        profile_sigma_min=0.05,
        profile_sigma_max=0.70,
        profile_num=3,
        sigma_markers=(args.evaluation_sigma,),
        curvature_sigmas=(args.evaluation_sigma,),
        cross_u_steps=(args.u_step,),
        potential_atol=args.potential_atol,
        x64=args.x64,
    )
    validation = v5._validate_configuration(legacy_args)

    v5_summary = Path(args.v5_summary)
    v5_arrays = Path(args.v5_arrays)
    v5_failed = Path(args.v5_failed)
    if not (v5_summary.is_file() and v5_arrays.is_file() and v5_failed.is_file()):
        raise FileNotFoundError("The pinned v5 derivative failure is incomplete.")

    v6_dir = Path(args.v6_dir)
    v6_failed = v6_dir / "FAILED"
    v6_log = v6_dir / "run.log"
    v6_provenance = v6_dir / "provenance.sha256"
    forbidden_v6 = (
        v6_dir / "COMPLETE",
        v6_dir / v6.SUMMARY_NAME,
        v6_dir / v6.ARRAYS_NAME,
        v6_dir / "outputs.sha256",
    )
    if not (
        v6_failed.is_file()
        and v6_log.is_file()
        and v6_provenance.is_file()
        and all(not path.exists() for path in forbidden_v6)
    ):
        raise ValueError("The predecessor is not the expected artifact-less v6 failure.")
    v6_log_text = v6_log.read_text(encoding="utf-8")
    required_log_fragments = (
        "phase=marker sigma=0.27526917",
        'ValueError: Agreement vectors must have one finite common shape.',
        "diagnostic failed with status 1",
    )
    if not all(fragment in v6_log_text for fragment in required_log_fragments):
        raise ValueError("The v6 failure signature has drifted.")

    output_dir = Path(args.out_dir)
    for name in (SUMMARY_NAME, ARRAYS_NAME, CHECKPOINT_NAME):
        path = output_dir / name
        for candidate in (path, Path(f"{path}.tmp")):
            if candidate.exists():
                raise FileExistsError(f"Refusing output collision: {candidate}")

    validation["v6_predecessor"] = {
        "failed_sha256": v5.sha256_file(v6_failed),
        "run_log_sha256": v5.sha256_file(v6_log),
        "provenance_sha256": v5.sha256_file(v6_provenance),
        "complete_absent": True,
        "summary_absent": True,
        "arrays_absent": True,
    }
    return validation


def _initial_arrays(
    *,
    repeat_count: int,
    call_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Allocate checkpoint arrays with explicit not-yet-executed sentinels."""

    nan = np.nan
    arrays: dict[str, np.ndarray] = {
        "coordinate_names": np.asarray(
            [
                *[f"atmosphere_rotated[{index}]" for index in range(7)],
                *[f"A_unconstrained[{index}]" for index in range(4)],
                "u_sigma_log_p",
            ]
        ),
        "point_labels": np.asarray(POINT_LABELS),
        "center_flat": np.full(FLAT_DIMENSION, nan),
        "point_flat": np.full((3, FLAT_DIMENSION), nan),
        "point_sigma": np.full(3, nan),
        "source_sigma": np.asarray(nan),
        "evaluation_sigma_requested": np.asarray(EVALUATION_SIGMA),
        "u_step_requested": np.asarray(U_STEP),
        "u_step_effective_minus": np.asarray(nan),
        "u_step_effective_plus": np.asarray(nan),
        "hvp_tangent": np.eye(1, FLAT_DIMENSION, SIGMA_INDEX)[0],
        "call_names": np.asarray(call_names),
        "call_returned": np.zeros(len(call_names), dtype=bool),
        "call_error_type": np.full(len(call_names), "", dtype="U128"),
        "call_error_message": np.full(len(call_names), "", dtype="U512"),
        "center_bundle_value": np.full(repeat_count, nan),
        "center_bundle_gradient": np.full((repeat_count, FLAT_DIMENSION), nan),
        "center_bundle_directional_gradient": np.full(repeat_count, nan),
        "center_bundle_hvp": np.full((repeat_count, FLAT_DIMENSION), nan),
        "standalone_value": np.full(3, nan),
        "standalone_gradient": np.full((3, FLAT_DIMENSION), nan),
        "bundle_point_value": np.full(3, nan),
        "bundle_point_gradient": np.full((3, FLAT_DIMENSION), nan),
        "bundle_point_directional_gradient": np.full(3, nan),
        "bundle_point_hvp": np.full((3, FLAT_DIMENSION), nan),
        "verlet_value": np.asarray(nan),
        "verlet_gradient": np.full(FLAT_DIMENSION, nan),
        "scalar_gradient_u": np.asarray(nan),
        "scalar_hessian_uu": np.asarray(nan),
        "analytic_negative_log_likelihood": np.full(3, nan),
        "analytic_total_potential": np.full(3, nan),
        "analytic_gradient_u": np.full(3, nan),
        "analytic_hessian_uu": np.full(3, nan),
        "analytic_log_determinant": np.full(3, nan),
        "analytic_mahalanobis": np.full(3, nan),
        "analytic_cholesky_min_diagonal": np.full(3, nan),
        "standalone_fd_column": np.full(FLAT_DIMENSION, nan),
        "bundle_fd_column": np.full(FLAT_DIMENSION, nan),
        "standalone_value_fd_gradient_u": np.asarray(nan),
        "standalone_value_fd_hessian_uu": np.asarray(nan),
        "bundle_value_fd_gradient_u": np.asarray(nan),
        "bundle_value_fd_hessian_uu": np.asarray(nan),
        "source_potential_replay_error": np.asarray(nan),
        "constrained_roundtrip_error": np.asarray(nan),
        "center_value_ulp": np.asarray(nan),
        "standalone_first_signal_ulp": np.asarray(nan),
        "standalone_second_signal_ulp": np.asarray(nan),
        "bundle_first_signal_ulp": np.asarray(nan),
        "bundle_second_signal_ulp": np.asarray(nan),
        "eigenvalues": np.empty(0),
        "analytic_unit_gram": np.empty((0, 0)),
        "analytic_unit_rhs": np.empty(0),
        "analytic_residual_dinv_residual": np.asarray(nan),
        "analytic_logdet_diagonal": np.asarray(nan),
        "checkpoint_sequence": np.asarray(0, dtype=np.int64),
    }
    arrays.update(raw_finite_masks(arrays))
    return arrays


def _refresh_derived_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Refresh finite masks and finite-difference arrays without raising."""

    arrays.update(raw_finite_masks(arrays))
    point_u = np.asarray(arrays["point_flat"])[:, SIGMA_INDEX]
    minus_step = float(point_u[1] - point_u[0])
    plus_step = float(point_u[2] - point_u[1])
    if math.isfinite(minus_step) and math.isfinite(plus_step):
        arrays["u_step_effective_minus"] = np.asarray(minus_step)
        arrays["u_step_effective_plus"] = np.asarray(plus_step)
        denominator = minus_step + plus_step
        if denominator > 0.0:
            arrays["standalone_fd_column"] = (
                arrays["standalone_gradient"][2]
                - arrays["standalone_gradient"][0]
            ) / denominator
            arrays["bundle_fd_column"] = (
                arrays["bundle_point_gradient"][2]
                - arrays["bundle_point_gradient"][0]
            ) / denominator
            for prefix in ("standalone", "bundle"):
                values = arrays[
                    "standalone_value" if prefix == "standalone" else "bundle_point_value"
                ]
                arrays[f"{prefix}_value_fd_gradient_u"] = np.asarray(
                    (values[2] - values[0]) / denominator
                )
                mean_step = 0.5 * denominator
                arrays[f"{prefix}_value_fd_hessian_uu"] = np.asarray(
                    (values[2] - 2.0 * values[1] + values[0]) / mean_step**2
                )
    arrays.update(raw_finite_masks(arrays))


def _checkpoint_payload(
    stage: str,
    arrays: dict[str, np.ndarray],
) -> dict:
    """Build strict-JSON checkpoint metadata without raw nonfinite values."""

    call_returned = np.asarray(arrays["call_returned"], dtype=bool)
    return {
        "schema_version": 1,
        "mode": "m8_v7_free_sigma_derivative_capture_checkpoint",
        "stage": stage,
        "checkpoint_sequence": int(arrays["checkpoint_sequence"]),
        "call_names": arrays["call_names"],
        "call_returned": call_returned,
        "returned_count": int(np.count_nonzero(call_returned)),
        "planned_count": int(call_returned.size),
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
    """Atomically persist every completed call before continuing."""

    _refresh_derived_arrays(arrays)
    arrays["checkpoint_sequence"] = np.asarray(
        int(arrays["checkpoint_sequence"]) + 1,
        dtype=np.int64,
    )
    v5._write_npz_atomic(arrays_path, arrays)
    v5._write_json_atomic(
        checkpoint_path,
        _checkpoint_payload(stage, arrays),
    )


def _capture_call(
    *,
    name: str,
    call_index: dict[str, int],
    thunk,
    assign,
    arrays_path: Path,
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Execute one planned call and checkpoint success or exception."""

    index = call_index[name]
    print(f"[m8_v7_sigma_capture] phase={name}", flush=True)
    try:
        result = thunk()
        assign(result)
        arrays["call_returned"][index] = True
    except Exception as error:  # Preserve the failure before returning nonzero.
        arrays["call_error_type"][index] = type(error).__name__
        arrays["call_error_message"][index] = str(error)[:512]
        print(
            f"[m8_v7_sigma_capture] call failed: {name}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
    _write_checkpoint(name, arrays_path, checkpoint_path, arrays)


def _host_bundle(bundle_jit, point: np.ndarray) -> tuple[float, np.ndarray, float, np.ndarray]:
    """Evaluate and synchronize one directional HVP bundle."""

    (value, gradient), (directional_gradient, hvp) = bundle_jit(jnp.asarray(point))
    hvp.block_until_ready()
    return (
        float(jax.device_get(value)),
        np.asarray(jax.device_get(gradient), dtype=np.float64),
        float(jax.device_get(directional_gradient)),
        np.asarray(jax.device_get(hvp), dtype=np.float64),
    )


def _host_value_gradient(value_gradient_jit, point: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate and synchronize one production value/gradient call."""

    value, gradient = value_gradient_jit(jnp.asarray(point))
    gradient.block_until_ready()
    return (
        float(jax.device_get(value)),
        np.asarray(jax.device_get(gradient), dtype=np.float64),
    )


def _run_gpu_diagnostic(args: argparse.Namespace, validation: dict) -> tuple[dict, dict]:
    """Run one-point derivative capture with atomic call-level checkpoints."""

    start_time = time.perf_counter()
    jax.config.update("jax_enable_x64", args.x64)
    rng_key = jax.random.PRNGKey(args.seed)
    output_dir = Path(args.out_dir)
    arrays_path = output_dir / ARRAYS_NAME
    checkpoint_path = output_dir / CHECKPOINT_NAME
    call_names = planned_call_names(args.repeat_count)
    call_index = {name: index for index, name in enumerate(call_names)}
    arrays = _initial_arrays(repeat_count=args.repeat_count, call_names=call_names)
    _write_checkpoint("allocated", arrays_path, checkpoint_path, arrays)

    print("[m8_v7_sigma_capture] phase=build_target", flush=True)
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
    standalone_jit = jax.jit(jax.value_and_grad(flat_potential))
    bundle_jit = jax.jit(
        lambda flat: v6.directional_value_gradient_hvp(
            flat_potential,
            flat,
            direction,
        )
    )
    scalar_jit = jax.jit(
        lambda flat: v6.scalar_sigma_gradient_hessian(flat_potential, flat)
    )
    verlet_init, verlet_update = velocity_verlet(
        flat_potential,
        euclidean_kinetic_energy,
        forward_mode_differentiation=False,
    )

    def verlet_context(flat, step_size):
        state = verlet_init(flat, jnp.zeros_like(flat))
        state = verlet_update(step_size, jnp.ones_like(flat), state)
        return state.potential_energy, state.z_grad

    verlet_jit = jax.jit(verlet_context)

    fixed_state = v6._load_fixed_state(args)
    source_center = np.asarray(fixed_state.flat, dtype=np.float32)
    center = source_center.copy()
    center[SIGMA_INDEX] = np.float32(math.log(args.evaluation_sigma))
    points = np.repeat(center[None, :], 3, axis=0)
    points[0, SIGMA_INDEX] -= np.float32(args.u_step)
    points[2, SIGMA_INDEX] += np.float32(args.u_step)
    arrays["center_flat"] = center
    arrays["point_flat"] = points
    arrays["point_sigma"] = np.exp(points[:, SIGMA_INDEX].astype(np.float64))
    arrays["source_sigma"] = np.asarray(fixed_state.sigma)

    constrained_source = constrain_flat(jnp.asarray(source_center))
    constrained_flat = np.concatenate(
        [
            np.asarray(jax.device_get(constrained_source["atmosphere_rotated"])),
            np.asarray(jax.device_get(constrained_source["A"])),
            np.asarray([jax.device_get(constrained_source["sigma_log_p"])]),
        ]
    )
    roundtrip_error = float(
        np.max(np.abs(constrained_flat - fixed_state.expected_constrained))
    )
    source_value = potential_jit(jnp.asarray(source_center))
    source_value.block_until_ready()
    source_prior = v5.sigma_u_prior_terms(
        fixed_state.sigma,
        wrapper_args.sigma_log_p_scale,
    )["potential"]
    source_replay_error = float(
        jax.device_get(source_value) - float(source_prior) - fixed_state.stored_potential
    )
    arrays["source_potential_replay_error"] = np.asarray(source_replay_error)
    arrays["constrained_roundtrip_error"] = np.asarray(roundtrip_error)
    _write_checkpoint("target_replayed", arrays_path, checkpoint_path, arrays)

    print("[m8_v7_sigma_capture] phase=extract_lowrank", flush=True)

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
                transforms[name].log_abs_det_jacobian(unconstrained[name], value)
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
        jnp.asarray(source_center)
    )
    factor.block_until_ready()
    geometry = retrieval.build_luhman16b_geometry(nside=wrapper_args.nside)
    pressure_gp = retrieval.build_fixed_pressure_gp_eigendecomposition(
        geometry.distance_matrix,
        wrapper_args.fixed_ell_b,
        theta=geometry.theta,
        phi=geometry.phi,
    )
    eigenvalues = np.asarray(jax.device_get(pressure_gp["eigenvalues"]), dtype=np.float64)
    gp_jitter = float(wrapper_args.gp_jitter)
    reference_scales = np.sqrt(fixed_state.sigma**2 * eigenvalues + gp_jitter)
    factor_host = np.asarray(jax.device_get(factor))
    cov_diag_host = np.asarray(jax.device_get(cov_diag), dtype=np.float64)
    residual_host = np.asarray(jax.device_get(observed - loc), dtype=np.float64)
    shared_prior_host = float(jax.device_get(shared_prior))
    observation_count = int(factor_host.shape[0])
    del loc, factor, cov_diag, observed, shared_prior

    statistics = v6.build_unit_lowrank_statistics(
        factor_host,
        reference_scales,
        cov_diag_host,
        residual_host,
        chunk_size=args.gram_chunk_size,
    )
    del factor_host, cov_diag_host, residual_host
    arrays["eigenvalues"] = eigenvalues
    arrays["analytic_unit_gram"] = np.asarray(statistics["unit_gram"])
    arrays["analytic_unit_rhs"] = np.asarray(statistics["unit_rhs"])
    arrays["analytic_residual_dinv_residual"] = np.asarray(
        statistics["residual_dinv_residual"]
    )
    arrays["analytic_logdet_diagonal"] = np.asarray(statistics["logdet_diagonal"])
    for point_index, sigma in enumerate(arrays["point_sigma"]):
        analytic = v6.analytic_lowrank_sigma_terms(
            float(sigma),
            eigenvalues,
            gp_jitter,
            statistics["unit_gram"],
            statistics["unit_rhs"],
            statistics["residual_dinv_residual"],
            statistics["logdet_diagonal"],
            observation_count,
        )
        prior = v5.sigma_u_prior_terms(sigma, wrapper_args.sigma_log_p_scale)
        arrays["analytic_negative_log_likelihood"][point_index] = analytic[
            "negative_log_likelihood"
        ]
        arrays["analytic_total_potential"][point_index] = (
            analytic["negative_log_likelihood"]
            + shared_prior_host
            + float(prior["potential"])
        )
        arrays["analytic_gradient_u"][point_index] = (
            analytic["gradient_u_likelihood"] + float(prior["gradient_u"])
        )
        arrays["analytic_hessian_uu"][point_index] = (
            analytic["hessian_uu_likelihood"] + float(prior["curvature_uu"])
        )
        arrays["analytic_log_determinant"][point_index] = analytic["log_determinant"]
        arrays["analytic_mahalanobis"][point_index] = analytic["mahalanobis"]
        arrays["analytic_cholesky_min_diagonal"][point_index] = analytic[
            "capacitance_cholesky_min_diagonal"
        ]
    _write_checkpoint("analytic_ready", arrays_path, checkpoint_path, arrays)

    for point_index, label in enumerate(POINT_LABELS):
        name = f"{label}_standalone"

        def assign_standalone(result, index=point_index):
            arrays["standalone_value"][index] = result[0]
            arrays["standalone_gradient"][index] = result[1]

        _capture_call(
            name=name,
            call_index=call_index,
            thunk=lambda point=points[point_index]: _host_value_gradient(
                standalone_jit, point
            ),
            assign=assign_standalone,
            arrays_path=arrays_path,
            checkpoint_path=checkpoint_path,
            arrays=arrays,
        )

    for repeat_index in range(args.repeat_count):
        name = f"center_bundle_{repeat_index}"

        def assign_center_bundle(result, index=repeat_index):
            arrays["center_bundle_value"][index] = result[0]
            arrays["center_bundle_gradient"][index] = result[1]
            arrays["center_bundle_directional_gradient"][index] = result[2]
            arrays["center_bundle_hvp"][index] = result[3]
            if index == 0:
                arrays["bundle_point_value"][1] = result[0]
                arrays["bundle_point_gradient"][1] = result[1]
                arrays["bundle_point_directional_gradient"][1] = result[2]
                arrays["bundle_point_hvp"][1] = result[3]

        _capture_call(
            name=name,
            call_index=call_index,
            thunk=lambda: _host_bundle(bundle_jit, center),
            assign=assign_center_bundle,
            arrays_path=arrays_path,
            checkpoint_path=checkpoint_path,
            arrays=arrays,
        )

    for point_index, label in ((0, "minus"), (2, "plus")):
        name = f"{label}_bundle"

        def assign_bundle_point(result, index=point_index):
            arrays["bundle_point_value"][index] = result[0]
            arrays["bundle_point_gradient"][index] = result[1]
            arrays["bundle_point_directional_gradient"][index] = result[2]
            arrays["bundle_point_hvp"][index] = result[3]

        _capture_call(
            name=name,
            call_index=call_index,
            thunk=lambda point=points[point_index]: _host_bundle(bundle_jit, point),
            assign=assign_bundle_point,
            arrays_path=arrays_path,
            checkpoint_path=checkpoint_path,
            arrays=arrays,
        )

    def host_verlet():
        value, gradient = verlet_jit(jnp.asarray(center), jnp.asarray(0.0))
        gradient.block_until_ready()
        return float(jax.device_get(value)), np.asarray(
            jax.device_get(gradient), dtype=np.float64
        )

    def assign_verlet(result):
        arrays["verlet_value"] = np.asarray(result[0])
        arrays["verlet_gradient"] = result[1]

    _capture_call(
        name="center_verlet",
        call_index=call_index,
        thunk=host_verlet,
        assign=assign_verlet,
        arrays_path=arrays_path,
        checkpoint_path=checkpoint_path,
        arrays=arrays,
    )

    def host_scalar():
        gradient, hessian = scalar_jit(jnp.asarray(center))
        hessian.block_until_ready()
        return float(jax.device_get(gradient)), float(jax.device_get(hessian))

    def assign_scalar(result):
        arrays["scalar_gradient_u"] = np.asarray(result[0])
        arrays["scalar_hessian_uu"] = np.asarray(result[1])

    _capture_call(
        name="center_scalar",
        call_index=call_index,
        thunk=host_scalar,
        assign=assign_scalar,
        arrays_path=arrays_path,
        checkpoint_path=checkpoint_path,
        arrays=arrays,
    )

    _refresh_derived_arrays(arrays)
    center_value = arrays["standalone_value"][1]
    value_ulp = abs(float(np.spacing(np.float32(center_value))))
    arrays["center_value_ulp"] = np.asarray(value_ulp)
    for prefix, values in (
        ("standalone", arrays["standalone_value"]),
        ("bundle", arrays["bundle_point_value"]),
    ):
        arrays[f"{prefix}_first_signal_ulp"] = np.asarray(
            abs(values[2] - values[0]) / value_ulp
        )
        arrays[f"{prefix}_second_signal_ulp"] = np.asarray(
            abs(values[2] - 2.0 * values[1] + values[0]) / value_ulp
        )
    _refresh_derived_arrays(arrays)

    all_calls_returned = bool(np.all(arrays["call_returned"]))
    analytic_finite = bool(
        all(
            np.all(np.isfinite(arrays[name]))
            for name in (
                "analytic_total_potential",
                "analytic_gradient_u",
                "analytic_hessian_uu",
                "analytic_cholesky_min_diagonal",
            )
        )
        and np.all(arrays["analytic_cholesky_min_diagonal"] > 0.0)
    )
    center_value_alignment = max_abs_or_none(
        arrays["standalone_value"][1:2],
        arrays["analytic_total_potential"][1:2],
    )
    bundle_value_alignment = max_abs_or_none(
        arrays["center_bundle_value"],
        np.full(args.repeat_count, arrays["analytic_total_potential"][1]),
    )
    calculation_integrity_passed = bool(
        all_calls_returned
        and abs(source_replay_error) <= args.potential_atol
        and abs(roundtrip_error) <= ROUNDTRIP_ATOL
        and analytic_finite
        and center_value_alignment is not None
        and center_value_alignment <= VALUE_ALIGNMENT_ATOL
        and bundle_value_alignment is not None
        and bundle_value_alignment <= VALUE_ALIGNMENT_ATOL
    )
    derivative_integrity_passed = bool(
        all(np.all(np.isfinite(arrays[name])) for name in RAW_NUMERIC_KEYS if (
            "gradient" in name or "hvp" in name or "hessian" in name
        ))
    )
    classification = classify_derivative_capture(arrays)
    nonfinite_indices = {
        name: np.argwhere(~np.isfinite(np.asarray(arrays[name]))).tolist()
        for name in RAW_NUMERIC_KEYS
        if np.any(~np.isfinite(np.asarray(arrays[name])))
    }

    summary = {
        "schema_version": 1,
        "mode": "m8_v7_free_sigma_derivative_capture",
        "execution_completed": all_calls_returned,
        "calculation_integrity_passed": calculation_integrity_passed,
        "derivative_integrity_passed": derivative_integrity_passed,
        "raw_nonfinite_preserved": True,
        "classification": classification,
        "purpose": (
            "Separate NUTS-relevant first-gradient behavior from the known "
            "nonfinite higher-order derivative path at one fixed-control state."
        ),
        "configuration": {
            "fixed_control_index": fixed_state.index,
            "source_sigma_log_p": fixed_state.sigma,
            "evaluation_sigma_requested": args.evaluation_sigma,
            "evaluation_sigma_effective": float(arrays["point_sigma"][1]),
            "u_step_requested": args.u_step,
            "u_step_effective_minus": float(arrays["u_step_effective_minus"]),
            "u_step_effective_plus": float(arrays["u_step_effective_plus"]),
            "repeat_count": args.repeat_count,
            "potential_atol": args.potential_atol,
            "value_alignment_atol": VALUE_ALIGNMENT_ATOL,
            "seed": args.seed,
            "x64": args.x64,
        },
        "method": {
            "hvp_method": HVP_METHOD,
            "hvp_direction": "u_sigma_log_p only",
            "hvp_tangent_index": SIGMA_INDEX,
            "direction_count": 1,
            "hvp_fallback_allowed": False,
            "hvp_fallback_used": False,
            "full_hessian_computed": False,
            "finite_difference_role": "independent first-gradient cross-check only",
        },
        "integrity": {
            "all_calls_returned": all_calls_returned,
            "source_potential_replay_error": source_replay_error,
            "constrained_roundtrip_error": roundtrip_error,
            "analytic_reference_finite_and_spd": analytic_finite,
            "standalone_center_value_alignment_error": center_value_alignment,
            "bundle_center_value_alignment_error": bundle_value_alignment,
            "raw_mask_consistency": True,
        },
        "findings": {
            "classification": classification,
            "center_bundle_hvp_all_finite": bool(
                np.all(np.isfinite(arrays["center_bundle_hvp"]))
            ),
            "standalone_gradient_all_finite": bool(
                np.all(np.isfinite(arrays["standalone_gradient"]))
            ),
            "bundle_primal_gradient_all_finite": bool(
                np.all(np.isfinite(arrays["bundle_point_gradient"]))
            ),
            "verlet_gradient_all_finite": bool(
                np.all(np.isfinite(arrays["verlet_gradient"]))
            ),
            "scalar_hessian_finite": bool(
                np.all(np.isfinite(arrays["scalar_hessian_uu"]))
            ),
            "nonfinite_indices": nonfinite_indices,
            "bundle_repeat_value_bitwise_stable": bool(
                np.all(arrays["center_bundle_value"] == arrays["center_bundle_value"][0])
            ),
            "bundle_repeat_gradient_bitwise_stable": bool(
                np.all(
                    arrays["center_bundle_gradient"]
                    == arrays["center_bundle_gradient"][0]
                )
            ),
            "bundle_repeat_hvp_bitwise_stable": bool(
                np.array_equal(
                    arrays["center_bundle_hvp"],
                    np.repeat(
                        arrays["center_bundle_hvp"][0:1],
                        args.repeat_count,
                        axis=0,
                    ),
                    equal_nan=True,
                )
            ),
            "standalone_vs_bundle_center_gradient_max_abs": max_abs_or_none(
                arrays["standalone_gradient"][1],
                arrays["bundle_point_gradient"][1],
            ),
            "standalone_vs_verlet_center_gradient_max_abs": max_abs_or_none(
                arrays["standalone_gradient"][1],
                arrays["verlet_gradient"],
            ),
            "standalone_sigma_gradient_vs_analytic_abs": finite_or_none(
                abs(
                    arrays["standalone_gradient"][1, SIGMA_INDEX]
                    - arrays["analytic_gradient_u"][1]
                )
            ),
            "scalar_huu_vs_analytic_abs": finite_or_none(
                abs(arrays["scalar_hessian_uu"] - arrays["analytic_hessian_uu"][1])
            ),
            "hvp_vs_standalone_fd": vector_agreement_or_none(
                arrays["center_bundle_hvp"][0],
                arrays["standalone_fd_column"],
            ),
            "hvp_vs_bundle_fd": vector_agreement_or_none(
                arrays["center_bundle_hvp"][0],
                arrays["bundle_fd_column"],
            ),
            "standalone_value_fd_gradient_u": finite_or_none(
                arrays["standalone_value_fd_gradient_u"]
            ),
            "standalone_value_fd_hessian_uu": finite_or_none(
                arrays["standalone_value_fd_hessian_uu"]
            ),
            "bundle_value_fd_gradient_u": finite_or_none(
                arrays["bundle_value_fd_gradient_u"]
            ),
            "bundle_value_fd_hessian_uu": finite_or_none(
                arrays["bundle_value_fd_hessian_uu"]
            ),
            "center_value_ulp": finite_or_none(arrays["center_value_ulp"]),
            "standalone_first_signal_ulp": finite_or_none(
                arrays["standalone_first_signal_ulp"]
            ),
            "standalone_second_signal_ulp": finite_or_none(
                arrays["standalone_second_signal_ulp"]
            ),
        },
        "call_status": {
            "names": arrays["call_names"],
            "returned": arrays["call_returned"],
            "error_type": arrays["call_error_type"],
            "error_message": arrays["call_error_message"],
        },
        "runtime": {
            "python": sys.version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpyro": numpyro.__version__,
            "exojax": exojax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "paths": {
                "jax": inspect.getfile(jax),
                "jaxlib": inspect.getfile(jaxlib),
                "numpyro": inspect.getfile(numpyro),
                "exojax": inspect.getfile(exojax),
                "numpy": inspect.getfile(np),
                "scipy": inspect.getfile(scipy),
            },
        },
        "predecessor": validation["v6_predecessor"],
        "timing_seconds": {"total": time.perf_counter() - start_time},
        "limitations": [
            "This is one fixed-control state, not posterior sampling.",
            "Nonfinite derivatives are preserved as findings, not replaced by finite differences.",
            "No full Hessian is computed.",
            "The frozen linear target retains the rejected exact-forward fidelity.",
        ],
    }
    _write_checkpoint("final", arrays_path, checkpoint_path, arrays)
    return summary, arrays


def validate_saved_artifacts(
    summary_path: Path | str,
    arrays_path: Path | str,
    checkpoint_path: Path | str,
) -> dict:
    """Validate a completed capture while allowing raw derivative NaNs."""

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        required_shapes = {
            "center_flat": (FLAT_DIMENSION,),
            "point_flat": (3, FLAT_DIMENSION),
            "point_sigma": (3,),
            "hvp_tangent": (FLAT_DIMENSION,),
            "standalone_value": (3,),
            "standalone_gradient": (3, FLAT_DIMENSION),
            "bundle_point_value": (3,),
            "bundle_point_gradient": (3, FLAT_DIMENSION),
            "bundle_point_directional_gradient": (3,),
            "bundle_point_hvp": (3, FLAT_DIMENSION),
            "verlet_value": (),
            "verlet_gradient": (FLAT_DIMENSION,),
            "scalar_gradient_u": (),
            "scalar_hessian_uu": (),
            "analytic_negative_log_likelihood": (3,),
            "analytic_total_potential": (3,),
            "analytic_gradient_u": (3,),
            "analytic_hessian_uu": (3,),
            "analytic_log_determinant": (3,),
            "analytic_mahalanobis": (3,),
            "analytic_cholesky_min_diagonal": (3,),
            "standalone_fd_column": (FLAT_DIMENSION,),
            "bundle_fd_column": (FLAT_DIMENSION,),
            "standalone_value_fd_gradient_u": (),
            "standalone_value_fd_hessian_uu": (),
            "bundle_value_fd_gradient_u": (),
            "bundle_value_fd_hessian_uu": (),
            "source_potential_replay_error": (),
            "constrained_roundtrip_error": (),
            "checkpoint_sequence": (),
        }
        repeat_count = int(summary["configuration"]["repeat_count"])
        expected_call_names = np.asarray(planned_call_names(repeat_count))
        required_shapes.update(
            {
                "center_bundle_value": (repeat_count,),
                "center_bundle_gradient": (repeat_count, FLAT_DIMENSION),
                "center_bundle_directional_gradient": (repeat_count,),
                "center_bundle_hvp": (repeat_count, FLAT_DIMENSION),
                "call_names": (expected_call_names.size,),
                "call_returned": (expected_call_names.size,),
                "call_error_type": (expected_call_names.size,),
                "call_error_message": (expected_call_names.size,),
            }
        )
        required = (
            set(required_shapes)
            | set(RAW_NUMERIC_KEYS)
            | {f"{name}_finite_mask" for name in RAW_NUMERIC_KEYS}
            | {
                "call_returned",
                "call_names",
                "call_error_type",
                "call_error_message",
                "checkpoint_sequence",
                "source_potential_replay_error",
                "constrained_roundtrip_error",
            }
        )
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Missing artifact arrays: {missing}")
        if not all(archive[name].shape == shape for name, shape in required_shapes.items()):
            raise ValueError("Artifact array shape mismatch.")
        if not np.array_equal(
            archive["hvp_tangent"],
            np.eye(1, FLAT_DIMENSION, SIGMA_INDEX)[0],
        ):
            raise ValueError("Invalid HVP tangent.")
        if not np.array_equal(archive["call_names"], expected_call_names):
            raise ValueError("Derivative call schedule mismatch.")
        for name in RAW_NUMERIC_KEYS:
            if not np.array_equal(
                archive[f"{name}_finite_mask"],
                np.isfinite(archive[name]),
            ):
                raise ValueError(f"Finite mask mismatch for {name}.")
        loaded = {name: np.asarray(archive[name]) for name in required}

    call_returned = np.asarray(loaded["call_returned"], dtype=bool)
    calls_returned = bool(np.all(call_returned))
    expected_classification = classify_derivative_capture(loaded)
    derivative_names = tuple(
        name
        for name in RAW_NUMERIC_KEYS
        if "gradient" in name or "hvp" in name or "hessian" in name
    )
    expected_derivative_integrity = bool(
        all(np.all(np.isfinite(loaded[name])) for name in derivative_names)
    )
    analytic_finite = bool(
        all(
            np.all(np.isfinite(loaded[name]))
            for name in (
                "analytic_total_potential",
                "analytic_gradient_u",
                "analytic_hessian_uu",
                "analytic_cholesky_min_diagonal",
            )
        )
        and np.all(loaded["analytic_cholesky_min_diagonal"] > 0.0)
    )
    center_alignment = max_abs_or_none(
        loaded["standalone_value"][1:2],
        loaded["analytic_total_potential"][1:2],
    )
    bundle_alignment = max_abs_or_none(
        loaded["center_bundle_value"],
        np.full(repeat_count, loaded["analytic_total_potential"][1]),
    )
    configuration = summary.get("configuration", {})
    potential_atol = float(configuration.get("potential_atol", math.nan))
    value_alignment_atol = float(
        configuration.get("value_alignment_atol", math.nan)
    )
    expected_calculation_integrity = bool(
        calls_returned
        and potential_atol == POTENTIAL_ATOL
        and value_alignment_atol == VALUE_ALIGNMENT_ATOL
        and abs(float(loaded["source_potential_replay_error"])) <= potential_atol
        and abs(float(loaded["constrained_roundtrip_error"])) <= ROUNDTRIP_ATOL
        and analytic_finite
        and center_alignment is not None
        and center_alignment <= value_alignment_atol
        and bundle_alignment is not None
        and bundle_alignment <= value_alignment_atol
    )
    expected_nonfinite_count = {
        name: int(np.count_nonzero(~np.isfinite(loaded[name])))
        for name in RAW_NUMERIC_KEYS
    }
    checkpoint_matches = bool(
        checkpoint.get("stage") == "final"
        and checkpoint.get("checkpoint_sequence")
        == int(loaded["checkpoint_sequence"])
        and checkpoint.get("call_names") == expected_call_names.tolist()
        and checkpoint.get("call_returned") == call_returned.tolist()
        and checkpoint.get("returned_count") == int(np.count_nonzero(call_returned))
        and checkpoint.get("planned_count") == call_returned.size
        and checkpoint.get("raw_nonfinite_count") == expected_nonfinite_count
    )

    method = summary.get("method", {})
    method_ok = bool(
        method.get("hvp_method") == HVP_METHOD
        and method.get("hvp_direction") == "u_sigma_log_p only"
        and method.get("hvp_tangent_index") == SIGMA_INDEX
        and method.get("direction_count") == 1
        and method.get("hvp_fallback_allowed") is False
        and method.get("hvp_fallback_used") is False
        and method.get("full_hessian_computed") is False
    )
    ok = bool(
        summary.get("execution_completed") is True
        and summary.get("execution_completed") is calls_returned
        and summary.get("calculation_integrity_passed")
        is expected_calculation_integrity
        and expected_calculation_integrity
        and summary.get("derivative_integrity_passed")
        is expected_derivative_integrity
        and summary.get("classification") == expected_classification
        and summary.get("findings", {}).get("classification")
        == expected_classification
        and summary.get("raw_nonfinite_preserved") is True
        and method_ok
        and configuration.get("evaluation_sigma_requested") == EVALUATION_SIGMA
        and configuration.get("u_step_requested") == U_STEP
        and checkpoint_matches
    )
    if not ok:
        raise ValueError("Artifact contract validation failed.")
    return {
        "artifact_validation_passed": True,
        "derivative_integrity_passed": expected_derivative_integrity,
        "classification": expected_classification,
    }


def validate_checkpoint_artifacts(
    arrays_path: Path | str,
    checkpoint_path: Path | str,
) -> dict:
    """Validate the latest atomic checkpoint after an incomplete run."""

    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        required = (
            set(RAW_NUMERIC_KEYS)
            | {f"{name}_finite_mask" for name in RAW_NUMERIC_KEYS}
            | {
                "call_names",
                "call_returned",
                "call_error_type",
                "call_error_message",
                "checkpoint_sequence",
            }
        )
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Missing checkpoint arrays: {missing}")
        for name in RAW_NUMERIC_KEYS:
            if not np.array_equal(
                archive[f"{name}_finite_mask"],
                np.isfinite(archive[name]),
            ):
                raise ValueError(f"Finite mask mismatch for {name}.")
        call_names = np.asarray(archive["call_names"])
        if call_names.ndim != 1 or call_names.size < 9:
            raise ValueError("Invalid checkpoint call schedule shape.")
        repeat_count = int(call_names.size - 7)
        expected_call_names = np.asarray(planned_call_names(repeat_count))
        if not np.array_equal(call_names, expected_call_names):
            raise ValueError("Checkpoint call schedule mismatch.")
        call_returned = np.asarray(archive["call_returned"], dtype=bool)
        if call_returned.shape != call_names.shape:
            raise ValueError("Checkpoint call completion shape mismatch.")
        returned_count = int(np.count_nonzero(archive["call_returned"]))
        planned_count = int(archive["call_returned"].size)
        checkpoint_sequence = int(archive["checkpoint_sequence"])
        expected_nonfinite_count = {
            name: int(np.count_nonzero(~np.isfinite(archive[name])))
            for name in RAW_NUMERIC_KEYS
        }
    if (
        checkpoint.get("returned_count") != returned_count
        or checkpoint.get("planned_count") != planned_count
        or checkpoint.get("checkpoint_sequence") != checkpoint_sequence
        or checkpoint.get("call_names") != call_names.tolist()
        or checkpoint.get("call_returned") != call_returned.tolist()
        or checkpoint.get("raw_nonfinite_count") != expected_nonfinite_count
    ):
        raise ValueError("Checkpoint JSON and NPZ disagree.")
    return {
        "checkpoint_validation_passed": True,
        "stage": checkpoint.get("stage"),
        "returned_count": returned_count,
        "planned_count": planned_count,
    }


def main() -> None:
    """Validate, run, checkpoint, and atomically write the final summary."""

    args = parse_args()
    validation = _validate_configuration(args)
    if args.validate_only:
        print(
            json.dumps(
                v5._json_ready(
                    {
                        "validation_passed": True,
                        "mode": "m8_v7_free_sigma_derivative_capture",
                        "out_dir": str(Path(args.out_dir).resolve()),
                        "evaluation_sigma": args.evaluation_sigma,
                        "u_step": args.u_step,
                        "repeat_count": args.repeat_count,
                        "hvp_method": HVP_METHOD,
                        "nonfinite_derivatives_are_findings": True,
                        "v6_predecessor": validation["v6_predecessor"],
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
    print(json.dumps(v5._json_ready(summary), indent=2, allow_nan=False), flush=True)
    if summary["execution_completed"] is not True:
        raise SystemExit(4)
    if summary["calculation_integrity_passed"] is not True:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
