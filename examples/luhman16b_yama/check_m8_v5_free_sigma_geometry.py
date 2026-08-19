"""Diagnose the free sigma_log_p geometry of the frozen M8 linear target."""

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
from scipy import special as scipy_special
import exojax


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
from examples.luhman16b_yama import m8_v2_run  # noqa: E402


SUMMARY_NAME = "m8_v5_free_sigma_geometry_summary.json"
ARRAYS_NAME = "m8_v5_free_sigma_geometry_arrays.npz"
SITE_ORDER = ("atmosphere_rotated", "A", "sigma_log_p")
ATMOSPHERE_SLICE = slice(0, 7)
A_SLICE = slice(7, 11)
SIGMA_INDEX = 11
FLAT_DIMENSION = 12
DEFAULT_SIGMA_MARKERS = (
    0.27526917,
    m8_v2_run.V17_INITIAL_SIGMA_LOG_P,
    0.42335291,
    0.53313493,
)
DEFAULT_CURVATURE_SIGMAS = (
    *DEFAULT_SIGMA_MARKERS,
)
DEFAULT_CROSS_U_STEPS = (0.005, 0.01, 0.02)
FREE_CHAIN_PROBES = (
    (361, 0.28090283274650574),
    (295, 0.31987935304641724),
    (477, 0.41985246539115906),
    (561, 0.5002002716064453),
    (1156, 0.5499016642570496),
)


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    """Parse a nonempty comma-separated list of finite floats."""

    values = tuple(float(item) for item in text.split(","))
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("Expected finite comma-separated floats.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse diagnostic arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-from",
        default=str(ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"),
    )
    parser.add_argument(
        "--free-samples",
        default=str(ROOT / "results" / "m8" / "v1" / "samples.npz"),
    )
    parser.add_argument(
        "--free-diagnostics",
        default=str(ROOT / "results" / "m8" / "v1" / "diagnostics.json"),
    )
    parser.add_argument(
        "--fixed-samples",
        default=str(
            ROOT / "results" / "m8" / "v3" / "fixed_seed0" / "samples.npz"
        ),
    )
    parser.add_argument(
        "--fixed-diagnostics",
        default=str(
            ROOT / "results" / "m8" / "v3" / "fixed_seed0" / "diagnostics.json"
        ),
    )
    parser.add_argument(
        "--initial-replay",
        default=str(
            ROOT
            / "results"
            / "m8"
            / "v3"
            / "fixed_seed0"
            / "initial_point_replay.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "m8" / "v5" / "free_sigma_geometry"),
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
        "--curvature-sigmas",
        type=_parse_csv_floats,
        default=DEFAULT_CURVATURE_SIGMAS,
    )
    parser.add_argument(
        "--cross-u-steps",
        type=_parse_csv_floats,
        default=DEFAULT_CROSS_U_STEPS,
    )
    parser.add_argument("--potential-atol", type=float, default=5.0e-2)
    parser.add_argument("--no-x64", dest="x64", action="store_false")
    parser.add_argument("--x64", dest="x64", action="store_true")
    parser.set_defaults(x64=False)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path | str) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    """Convert nested NumPy values to strict-JSON-compatible objects."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Nonfinite value cannot be serialized: {result!r}.")
        return result
    return value


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write strict JSON through a sibling temporary file."""

    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a compressed NPZ through a sibling temporary file."""

    temporary = Path(f"{path}.tmp")
    with temporary.open("wb") as file_obj:
        np.savez_compressed(file_obj, **arrays)
    temporary.replace(path)


def _scalar(archive, name: str):
    """Return one scalar archive value."""

    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar, got {value.shape}.")
    return value.item()


def unconstrained_archive_coordinates(
    atmosphere_rotated: np.ndarray,
    normalization_factor: np.ndarray,
    sigma_log_p: np.ndarray | None,
    *,
    a_bounds: tuple[float, float],
) -> np.ndarray:
    """Map archived physical samples to the explicit diagnostic coordinates."""

    atmosphere_rotated = np.asarray(atmosphere_rotated, dtype=np.float64)
    normalization_factor = np.asarray(normalization_factor, dtype=np.float64)
    lower, upper = (float(a_bounds[0]), float(a_bounds[1]))
    if atmosphere_rotated.ndim != 2 or atmosphere_rotated.shape[1] != 7:
        raise ValueError("atmosphere_rotated must have shape (draw, 7).")
    if normalization_factor.shape != (atmosphere_rotated.shape[0], 4):
        raise ValueError("A must have shape (draw, 4).")
    if not np.all((normalization_factor > lower) & (normalization_factor < upper)):
        raise ValueError("A samples must lie strictly inside their prior bounds.")
    a_fraction = (normalization_factor - lower) / (upper - lower)
    a_unconstrained = np.log(a_fraction) - np.log1p(-a_fraction)
    columns = [atmosphere_rotated, a_unconstrained]
    if sigma_log_p is not None:
        sigma_log_p = np.asarray(sigma_log_p, dtype=np.float64)
        if sigma_log_p.shape != (atmosphere_rotated.shape[0],):
            raise ValueError("sigma_log_p must have shape (draw,).")
        if not np.all(np.isfinite(sigma_log_p) & (sigma_log_p > 0.0)):
            raise ValueError("sigma_log_p samples must be finite and positive.")
        columns.append(np.log(sigma_log_p)[:, None])
    result = np.concatenate(columns, axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("Unconstrained archive coordinates are nonfinite.")
    return result


def robust_actual_medoid(coordinates: np.ndarray) -> dict[str, np.ndarray | int | float]:
    """Select the actual draw nearest the componentwise robust center."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0:
        raise ValueError("coordinates must be a nonempty two-dimensional array.")
    center = np.median(coordinates, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(coordinates - center), axis=0)
    standard_scale = np.std(coordinates, axis=0, ddof=1)
    scale = np.where(
        mad_scale > 1.0e-12,
        mad_scale,
        np.where(standard_scale > 1.0e-12, standard_scale, 1.0),
    )
    squared_distance = np.sum(((coordinates - center) / scale) ** 2, axis=1)
    index = int(np.argmin(squared_distance))
    return {
        "index": index,
        "center": center,
        "scale": scale,
        "squared_distance": float(squared_distance[index]),
    }


def sigma_u_prior_terms(sigma: np.ndarray | float, scale: float) -> dict[str, np.ndarray]:
    """Return HalfNormal-plus-exp-Jacobian potential terms in u=log(sigma)."""

    sigma = np.asarray(sigma, dtype=np.float64)
    scale = float(scale)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0) or scale <= 0.0:
        raise ValueError("sigma and scale must be finite and positive.")
    u = np.log(sigma)
    potential = (
        0.5 * (sigma / scale) ** 2
        - u
        + math.log(scale)
        + 0.5 * math.log(math.pi / 2.0)
    )
    return {
        "u": u,
        "potential": potential,
        "gradient_u": (sigma / scale) ** 2 - 1.0,
        "curvature_uu": 2.0 * (sigma / scale) ** 2,
    }


def halfnormal_cdf_coordinate_terms(sigma: float, scale: float) -> dict[str, float]:
    """Return g and derivatives for the same-prior Gaussian inverse-CDF map."""

    sigma = float(sigma)
    scale = float(scale)
    if not math.isfinite(sigma) or sigma <= 0.0 or scale <= 0.0:
        raise ValueError("sigma and scale must be finite and positive.")
    standardized = sigma / scale
    probability = 2.0 * scipy_special.ndtr(standardized) - 1.0
    if not 0.0 < probability < 1.0:
        raise ValueError("sigma is outside the numerically resolved CDF range.")
    gaussian = float(scipy_special.ndtri(probability))
    log_phi_gaussian = -0.5 * gaussian**2 - 0.5 * math.log(2.0 * math.pi)
    log_phi_standardized = (
        -0.5 * standardized**2 - 0.5 * math.log(2.0 * math.pi)
    )
    du_dg = math.exp(
        math.log(scale)
        - math.log(2.0)
        - math.log(sigma)
        + log_phi_gaussian
        - log_phi_standardized
    )
    d2u_dg2 = du_dg * (du_dg * (standardized**2 - 1.0) - gaussian)
    return {
        "g": gaussian,
        "du_dg": du_dg,
        "d2u_dg2": d2u_dg2,
    }


def _validate_configuration(args: argparse.Namespace) -> dict:
    """Validate artifacts and return deterministic state-selection metadata."""

    paths = {
        "initialization": Path(args.init_from),
        "free_samples": Path(args.free_samples),
        "free_diagnostics": Path(args.free_diagnostics),
        "fixed_samples": Path(args.fixed_samples),
        "fixed_diagnostics": Path(args.fixed_diagnostics),
        "initial_replay": Path(args.initial_replay),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    if not 0.0 < args.profile_sigma_min < args.profile_sigma_max:
        raise ValueError("The profile sigma bounds must be positive and ordered.")
    if args.profile_num < 3:
        raise ValueError("profile-num must be at least three.")
    if not all(value > 0.0 for value in args.sigma_markers):
        raise ValueError("All sigma markers must be positive.")
    if not all(value > 0.0 for value in args.curvature_sigmas):
        raise ValueError("All curvature sigmas must be positive.")
    if not all(value > 0.0 for value in args.cross_u_steps):
        raise ValueError("All cross-u steps must be positive.")
    if args.potential_atol <= 0.0:
        raise ValueError("potential-atol must be positive.")
    if args.x64:
        raise ValueError("The frozen M8 target is float32; --x64 is not allowed.")

    output_dir = Path(args.out_dir)
    summary = output_dir / SUMMARY_NAME
    arrays = output_dir / ARRAYS_NAME
    for path in (summary, arrays, Path(f"{summary}.tmp"), Path(f"{arrays}.tmp")):
        if path.exists():
            raise FileExistsError(f"Refusing output collision: {path}")

    required_sample_keys = {
        "atmosphere_rotated",
        "A",
        "sigma_log_p",
        "extra_potential_energy",
        "A_prior_bounds",
        "sigma_log_p_scale",
        "direct_sigma_log_p",
        "fix_sigma_log_p",
        "fixed_sigma_log_p",
        "chip_indices",
        "full_data",
        "nside",
        "fixed_ell_b",
        "pressure_gp_factorization",
        "fix_logg",
        "zero_mean_pressure_map",
        "zero_mean_log_w",
        "zero_sum_log_w_basis",
        "fix_a",
        "fix_log_w",
        "fix_sigma_d",
        "gaussianized_atmosphere",
        "atmosphere_rotation_label",
        "atmosphere_rotation_matrix",
        "gp_jitter",
        "x64",
    }
    with np.load(paths["free_samples"], allow_pickle=False) as free_archive, np.load(
        paths["fixed_samples"], allow_pickle=False
    ) as fixed_archive:
        for label, archive in (("free", free_archive), ("fixed", fixed_archive)):
            missing = sorted(required_sample_keys - set(archive.files))
            if missing:
                raise KeyError(f"{label} archive is missing keys: {missing}")
            if archive["atmosphere_rotated"].shape != (1500, 7):
                raise ValueError(f"Unexpected {label} atmosphere shape.")
            if archive["A"].shape != (1500, 4):
                raise ValueError(f"Unexpected {label} A shape.")
            for key in (
                "atmosphere_rotated",
                "A",
                "sigma_log_p",
                "extra_potential_energy",
            ):
                if not np.all(np.isfinite(np.asarray(archive[key]))):
                    raise ValueError(f"{label} archive has nonfinite {key} values.")

        common_keys = (
            "A_prior_bounds",
            "sigma_log_p_scale",
            "chip_indices",
            "full_data",
            "nside",
            "fixed_ell_b",
            "pressure_gp_factorization",
            "fix_logg",
            "zero_mean_pressure_map",
            "zero_mean_log_w",
            "zero_sum_log_w_basis",
            "fix_a",
            "fix_log_w",
            "fix_sigma_d",
            "gaussianized_atmosphere",
            "atmosphere_rotation_label",
            "atmosphere_rotation_matrix",
            "gp_jitter",
            "x64",
        )
        for key in common_keys:
            if not np.array_equal(np.asarray(free_archive[key]), np.asarray(fixed_archive[key])):
                raise ValueError(f"Fixed/free archive mismatch at {key}.")
        if bool(_scalar(free_archive, "fix_sigma_log_p")):
            raise ValueError("The free archive unexpectedly fixes sigma_log_p.")
        if not bool(_scalar(free_archive, "direct_sigma_log_p")):
            raise ValueError("The free archive does not use direct sigma_log_p.")
        if not bool(_scalar(fixed_archive, "fix_sigma_log_p")):
            raise ValueError("The fixed archive does not fix sigma_log_p.")
        if bool(_scalar(fixed_archive, "direct_sigma_log_p")):
            raise ValueError("The fixed archive unexpectedly samples direct sigma_log_p.")
        fixed_sigma = float(_scalar(fixed_archive, "fixed_sigma_log_p"))
        if fixed_sigma != m8_v2_run.V17_INITIAL_SIGMA_LOG_P:
            raise ValueError("The fixed archive sigma has drifted.")
        prior_scale = float(_scalar(free_archive, "sigma_log_p_scale"))
        if prior_scale != 0.3:
            raise ValueError("The free sigma_log_p prior scale has drifted.")

        a_bounds = tuple(np.asarray(free_archive["A_prior_bounds"], dtype=float))
        fixed_coordinates = unconstrained_archive_coordinates(
            fixed_archive["atmosphere_rotated"],
            fixed_archive["A"],
            None,
            a_bounds=a_bounds,
        )
        free_coordinates = unconstrained_archive_coordinates(
            free_archive["atmosphere_rotated"],
            free_archive["A"],
            free_archive["sigma_log_p"],
            a_bounds=a_bounds,
        )
        fixed_medoid = robust_actual_medoid(fixed_coordinates)
        free_medoid = robust_actual_medoid(free_coordinates)
        fixed_index = int(fixed_medoid["index"])
        free_index = int(free_medoid["index"])
        free_chain_probes = []
        for probe_index, expected_sigma in FREE_CHAIN_PROBES:
            actual_sigma = float(free_archive["sigma_log_p"][probe_index])
            if actual_sigma != expected_sigma:
                raise ValueError(
                    f"Free-chain probe {probe_index} sigma has drifted: "
                    f"{actual_sigma!r} != {expected_sigma!r}."
                )
            free_chain_probes.append(
                {
                    "index": probe_index,
                    "sigma_log_p": actual_sigma,
                    "stored_potential_energy": float(
                        free_archive["extra_potential_energy"][probe_index]
                    ),
                }
            )
        state_selection = {
            "fixed_control_probe": {
                "index": fixed_index,
                "sigma_log_p": float(fixed_archive["sigma_log_p"][fixed_index]),
                "stored_potential_energy": float(
                    fixed_archive["extra_potential_energy"][fixed_index]
                ),
                "robust_squared_distance": fixed_medoid["squared_distance"],
            },
            "nonstationary_free_probe": {
                "index": free_index,
                "sigma_log_p": float(free_archive["sigma_log_p"][free_index]),
                "stored_potential_energy": float(
                    free_archive["extra_potential_energy"][free_index]
                ),
                "robust_squared_distance": free_medoid["squared_distance"],
            },
            "nonstationary_free_chain_probes": free_chain_probes,
        }

    free_diagnostics = json.loads(paths["free_diagnostics"].read_text(encoding="utf-8"))
    fixed_diagnostics = json.loads(
        paths["fixed_diagnostics"].read_text(encoding="utf-8")
    )
    initial_replay = json.loads(paths["initial_replay"].read_text(encoding="utf-8"))
    if not initial_replay.get("passed", False):
        raise ValueError("The pinned initial fixed/free replay did not pass.")
    if free_diagnostics.get("mode") != (
        "m8_v1_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_prod"
    ):
        raise ValueError("Unexpected free diagnostic mode.")
    if fixed_diagnostics.get("mode") != (
        "m8_v3_fixed_sigma_log_p_v17_initial_full_dense_long_control"
    ):
        raise ValueError("Unexpected fixed diagnostic mode.")

    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "prior_scale": prior_scale,
        "a_bounds": list(a_bounds),
        "state_selection": state_selection,
        "free_chain_warning": {
            "tree_depth_cap_fraction": float(
                free_diagnostics["tree_depth_cap_fraction"]
            ),
            "divergence_count": int(free_diagnostics["divergence_count"]),
            "role": "evaluation-point source only; not converged ground truth",
        },
        "fixed_control_warning": {
            "tree_depth_cap_fraction": float(
                fixed_diagnostics["tree_depth_cap_fraction"]
            ),
            "divergence_count": int(fixed_diagnostics["divergence_count"]),
            "role": "conditional control probe; not a prespecified sampler pass",
        },
    }


def _pack_state(atmosphere: np.ndarray, a_unconstrained: np.ndarray, sigma: float):
    """Pack one state in the explicit diagnostic order."""

    result = np.concatenate(
        [
            np.asarray(atmosphere, dtype=np.float64).reshape(7),
            np.asarray(a_unconstrained, dtype=np.float64).reshape(4),
            np.asarray([math.log(float(sigma))], dtype=np.float64),
        ]
    )
    if result.shape != (FLAT_DIMENSION,) or not np.all(np.isfinite(result)):
        raise ValueError("Invalid packed diagnostic state.")
    return result


def _lowrank_profile_terms(
    u,
    reference_factor,
    reference_eigen_scales,
    eigenvalues,
    gp_jitter,
    loc,
    cov_diag,
    observed,
    shared_prior_potential,
    sigma_prior_scale,
):
    """Evaluate the exact frozen linear likelihood after changing only sigma."""

    sigma = jnp.exp(u)
    eigen_scales = jnp.sqrt(sigma**2 * eigenvalues + gp_jitter)
    cov_factor = reference_factor * (
        eigen_scales / reference_eigen_scales
    )[None, :]
    capacitance_tril = _batch_capacitance_tril(cov_factor, cov_diag)
    residual = observed - loc
    mahalanobis = _batch_lowrank_mahalanobis(
        cov_factor,
        cov_diag,
        residual,
        capacitance_tril,
    )
    log_determinant = _batch_lowrank_logdet(
        cov_factor,
        cov_diag,
        capacitance_tril,
    )
    negative_log_likelihood = 0.5 * (
        observed.size * jnp.log(2.0 * jnp.pi) + log_determinant + mahalanobis
    )
    sigma_prior_potential = (
        0.5 * (sigma / sigma_prior_scale) ** 2
        - u
        + jnp.log(sigma_prior_scale)
        + 0.5 * jnp.log(jnp.pi / 2.0)
    )
    conditioned_potential = negative_log_likelihood + shared_prior_potential
    total_potential = conditioned_potential + sigma_prior_potential
    factor_trace = jnp.sum(cov_factor * (cov_factor / cov_diag[:, None]))
    return total_potential, jnp.stack(
        [
            negative_log_likelihood,
            conditioned_potential,
            sigma_prior_potential,
            log_determinant,
            mahalanobis,
            factor_trace,
        ]
    )


def _run_gpu_diagnostic(args: argparse.Namespace, validation: dict) -> tuple[dict, dict]:
    """Build the frozen target and collect scalar profile/coupling diagnostics."""

    start_time = time.perf_counter()
    jax.config.update("jax_enable_x64", args.x64)
    rng_key = jax.random.PRNGKey(args.seed)
    wrapper_args = replay._fixed_wrapper_args(Path(args.init_from), args.seed)
    (
        _fixed_model,
        free_model,
        _fixed_initial_values,
        free_initial_values,
        physical_initial_values,
    ) = replay._build_models(wrapper_args)
    model_info = initialize_model(
        rng_key,
        free_model,
        init_strategy=init_to_value(values=free_initial_values),
        dynamic_args=False,
        forward_mode_differentiation=False,
        validate_grad=True,
    )
    initial_z = model_info.param_info.z
    initial_potential = float(jax.device_get(model_info.param_info.potential_energy))
    initial_gradient_tree = model_info.param_info.z_grad
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

    def flatten_tree(tree):
        return jnp.concatenate(
            [
                jnp.ravel(tree["atmosphere_rotated"]),
                jnp.ravel(tree["A"]),
                jnp.reshape(tree["sigma_log_p"], (1,)),
            ]
        )

    initial_flat = np.asarray(jax.device_get(flatten_tree(initial_z)), dtype=np.float64)
    initial_gradient = np.asarray(
        jax.device_get(flatten_tree(initial_gradient_tree)), dtype=np.float64
    )
    del initial_z, initial_gradient_tree

    def flat_potential(flat):
        return potential_fn(unpack_flat(flat))

    value_and_gradient = jax.jit(jax.value_and_grad(flat_potential))

    def observation_arrays(flat):
        unconstrained = unpack_flat(flat)
        constrained = constrain_flat(flat)
        trace = handlers.trace(
            handlers.substitute(
                handlers.seed(free_model, rng_key),
                data=constrained,
            )
        ).get_trace()
        shared_log_prior_z = jnp.asarray(0.0)
        for name in ("atmosphere_rotated", "A"):
            value = constrained[name]
            log_probability = jnp.sum(trace[name]["fn"].log_prob(value))
            log_jacobian = jnp.sum(
                transforms[name].log_abs_det_jacobian(
                    unconstrained[name],
                    value,
                )
            )
            shared_log_prior_z = shared_log_prior_z + log_probability + log_jacobian
        observation = trace["obs"]
        distribution = observation["fn"]
        observed = observation["value"]
        observation_log_probability = distribution.log_prob(observed)
        return (
            distribution.loc,
            distribution.cov_factor,
            distribution.cov_diag,
            observed,
            -shared_log_prior_z,
            observation_log_probability,
        )

    observation_arrays_jit = jax.jit(observation_arrays)

    geometry = retrieval.build_luhman16b_geometry(nside=wrapper_args.nside)
    pressure_gp = retrieval.build_fixed_pressure_gp_eigendecomposition(
        geometry.distance_matrix,
        wrapper_args.fixed_ell_b,
        theta=geometry.theta,
        phi=geometry.phi,
    )
    eigenvalues = jnp.asarray(pressure_gp["eigenvalues"])
    eigenvalues_host = np.asarray(jax.device_get(eigenvalues), dtype=np.float64)
    gp_jitter = float(wrapper_args.gp_jitter)
    prior_scale = float(wrapper_args.sigma_log_p_scale)

    with np.load(args.fixed_samples, allow_pickle=False) as fixed_archive, np.load(
        args.free_samples, allow_pickle=False
    ) as free_archive:
        a_bounds = tuple(np.asarray(free_archive["A_prior_bounds"], dtype=float))
        fixed_coordinates = unconstrained_archive_coordinates(
            fixed_archive["atmosphere_rotated"],
            fixed_archive["A"],
            None,
            a_bounds=a_bounds,
        )
        free_coordinates = unconstrained_archive_coordinates(
            free_archive["atmosphere_rotated"],
            free_archive["A"],
            free_archive["sigma_log_p"],
            a_bounds=a_bounds,
        )
        fixed_medoid = robust_actual_medoid(fixed_coordinates)
        free_medoid = robust_actual_medoid(free_coordinates)
        fixed_index = int(fixed_medoid["index"])
        free_index = int(free_medoid["index"])
        fixed_sigma = float(fixed_archive["sigma_log_p"][fixed_index])
        free_sigma = float(free_archive["sigma_log_p"][free_index])
        fixed_stored_potential = float(
            fixed_archive["extra_potential_energy"][fixed_index]
        )
        free_stored_potential = float(free_archive["extra_potential_energy"][free_index])
        states = [
            {
                "name": "common_initial",
                "source_index": -1,
                "flat": initial_flat,
                "reference_sigma": float(physical_initial_values["sigma_log_p"]),
                "stored_potential": initial_potential,
                "stored_target": "free",
                "expected_constrained": np.concatenate(
                    [
                        np.asarray(
                            free_initial_values["atmosphere_rotated"]
                        ).reshape(7),
                        np.asarray(free_initial_values["A"]).reshape(4),
                        np.asarray([free_initial_values["sigma_log_p"]]),
                    ]
                ),
            },
            {
                "name": "fixed_control_probe",
                "source_index": fixed_index,
                "flat": _pack_state(
                    fixed_archive["atmosphere_rotated"][fixed_index],
                    fixed_coordinates[fixed_index, 7:11],
                    fixed_sigma,
                ),
                "reference_sigma": fixed_sigma,
                "stored_potential": fixed_stored_potential,
                "stored_target": "fixed",
                "expected_constrained": np.concatenate(
                    [
                        np.asarray(
                            fixed_archive["atmosphere_rotated"][fixed_index]
                        ).reshape(7),
                        np.asarray(fixed_archive["A"][fixed_index]).reshape(4),
                        np.asarray([fixed_sigma]),
                    ]
                ),
            },
            {
                "name": "nonstationary_free_probe",
                "source_index": free_index,
                "flat": free_coordinates[free_index],
                "reference_sigma": free_sigma,
                "stored_potential": free_stored_potential,
                "stored_target": "free",
                "expected_constrained": np.concatenate(
                    [
                        np.asarray(
                            free_archive["atmosphere_rotated"][free_index]
                        ).reshape(7),
                        np.asarray(free_archive["A"][free_index]).reshape(4),
                        np.asarray([free_sigma]),
                    ]
                ),
            },
        ]
        for probe_index, expected_sigma in FREE_CHAIN_PROBES:
            probe_sigma = float(free_archive["sigma_log_p"][probe_index])
            if probe_sigma != expected_sigma:
                raise ValueError(f"Free-chain probe {probe_index} has drifted.")
            states.append(
                {
                    "name": f"nonstationary_free_probe_{probe_index}",
                    "source_index": probe_index,
                    "flat": free_coordinates[probe_index],
                    "reference_sigma": probe_sigma,
                    "stored_potential": float(
                        free_archive["extra_potential_energy"][probe_index]
                    ),
                    "stored_target": "free",
                    "expected_constrained": np.concatenate(
                        [
                            np.asarray(
                                free_archive["atmosphere_rotated"][probe_index]
                            ).reshape(7),
                            np.asarray(free_archive["A"][probe_index]).reshape(4),
                            np.asarray([probe_sigma]),
                        ]
                    ),
                }
            )
        shared_robust_scale = np.asarray(fixed_medoid["scale"], dtype=np.float64)
        fixed_medoid_center = np.asarray(fixed_medoid["center"], dtype=np.float64)
        free_medoid_center = np.asarray(free_medoid["center"], dtype=np.float64)
        free_medoid_scale = np.asarray(free_medoid["scale"], dtype=np.float64)

    profile_sigmas = np.exp(
        np.linspace(
            math.log(args.profile_sigma_min),
            math.log(args.profile_sigma_max),
            args.profile_num,
        )
    )
    profile_sigmas = np.unique(
        np.concatenate(
            [
                profile_sigmas,
                np.asarray(args.sigma_markers, dtype=float),
                np.asarray(args.curvature_sigmas, dtype=float),
                np.asarray([state["reference_sigma"] for state in states]),
            ]
        )
    )
    profile_sigmas.sort()
    profile_u = np.log(profile_sigmas)

    def profile_function(u, *values):
        return _lowrank_profile_terms(u, *values)

    def profile_vector(u, *values):
        total, auxiliary = profile_function(u, *values)
        return jnp.concatenate([jnp.reshape(total, (1,)), auxiliary])

    def profile_primal_and_derivative(u, *values):
        return jax.jvp(
            lambda coordinate: profile_vector(coordinate, *values),
            (u,),
            (jnp.ones_like(u),),
        )

    profile_primal_and_derivative_jit = jax.jit(profile_primal_and_derivative)
    profile_curvature = jax.jit(
        jax.jacfwd(
            jax.jacfwd(profile_vector, argnums=0),
            argnums=0,
        )
    )

    profile_total = np.empty((len(states), len(profile_sigmas)), dtype=np.float64)
    profile_aux = np.empty((len(states), len(profile_sigmas), 6), dtype=np.float64)
    profile_gradient_u = np.empty_like(profile_total)
    profile_component_gradient_u = np.empty(
        (len(states), len(profile_sigmas), 7), dtype=np.float64
    )
    profile_reconstruction_error = np.empty((len(states),), dtype=np.float64)
    observation_logprob_error = np.empty((len(states),), dtype=np.float64)
    stored_potential_error = np.empty((len(states),), dtype=np.float64)
    roundtrip_error = np.empty((len(states),), dtype=np.float64)
    whitened_unit_column_norm_squared = np.empty(
        (len(states), len(eigenvalues_host)), dtype=np.float64
    )
    whitened_unit_column_norm_squared_by_chip = np.empty(
        (len(states), 4, len(eigenvalues_host)), dtype=np.float64
    )
    reference_full_gradient = np.empty((len(states), FLAT_DIMENSION), dtype=np.float64)
    reference_profile_indices = np.empty((len(states),), dtype=np.int64)
    reference_curvature_total = np.empty((len(states),), dtype=np.float64)
    curvature_total = np.empty(
        (len(states), len(args.curvature_sigmas)), dtype=np.float64
    )
    curvature_likelihood = np.empty_like(curvature_total)
    curvature_components = np.empty(
        (len(states), len(args.curvature_sigmas), 7), dtype=np.float64
    )
    gaussian_coordinate = np.empty_like(curvature_total)
    gaussian_du_dg = np.empty_like(curvature_total)
    gaussian_d2u_dg2 = np.empty_like(curvature_total)
    gaussian_potential = np.empty_like(curvature_total)
    gaussian_transform_potential_error = np.empty_like(curvature_total)
    gaussian_gradient = np.empty_like(curvature_total)
    gaussian_curvature = np.empty_like(curvature_total)
    timings = []

    for state_index, state in enumerate(states):
        state_start = time.perf_counter()
        flat = jnp.asarray(state["flat"])
        constrained = constrain_flat(flat)
        constrained_sigma = float(
            jax.device_get(constrained["sigma_log_p"])
        )
        constrained_flat = np.concatenate(
            [
                np.asarray(jax.device_get(constrained["atmosphere_rotated"])).reshape(7),
                np.asarray(jax.device_get(constrained["A"])).reshape(4),
                np.asarray([constrained_sigma]),
            ]
        )
        expected_constrained = np.asarray(
            state["expected_constrained"], dtype=np.float64
        )
        roundtrip_error[state_index] = float(
            np.max(np.abs(constrained_flat - expected_constrained))
        )
        full_value, full_gradient = value_and_gradient(flat)
        full_value.block_until_ready()
        full_value_host = float(jax.device_get(full_value))
        reference_full_gradient[state_index] = np.asarray(
            jax.device_get(full_gradient), dtype=np.float64
        )
        del full_gradient
        (
            loc,
            reference_factor,
            cov_diag,
            observed,
            shared_prior_potential,
            observation_log_probability,
        ) = observation_arrays_jit(flat)
        reference_scale = jnp.sqrt(
            constrained_sigma**2 * eigenvalues + gp_jitter
        )
        unit_factor = reference_factor / reference_scale[None, :]
        unit_factor.block_until_ready()
        column_norm = jnp.sum(unit_factor**2 / cov_diag[:, None], axis=0)
        whitened_unit_column_norm_squared[state_index] = np.asarray(
            jax.device_get(column_norm), dtype=np.float64
        )
        if unit_factor.shape[0] % 4 != 0:
            raise ValueError("The four-chip covariance rows cannot be partitioned.")
        rows_per_chip = unit_factor.shape[0] // 4
        chip_column_norm = jnp.sum(
            unit_factor.reshape(4, rows_per_chip, -1) ** 2
            / cov_diag.reshape(4, rows_per_chip, 1),
            axis=1,
        )
        whitened_unit_column_norm_squared_by_chip[state_index] = np.asarray(
            jax.device_get(chip_column_norm), dtype=np.float64
        )
        profile_arguments = (
            reference_factor,
            reference_scale,
            eigenvalues,
            jnp.asarray(gp_jitter),
            loc,
            cov_diag,
            observed,
            shared_prior_potential,
            jnp.asarray(prior_scale),
        )
        for profile_index, u in enumerate(profile_u):
            primal, derivative = profile_primal_and_derivative_jit(
                jnp.asarray(u), *profile_arguments
            )
            primal.block_until_ready()
            primal_host = np.asarray(jax.device_get(primal), dtype=np.float64)
            derivative_host = np.asarray(
                jax.device_get(derivative), dtype=np.float64
            )
            profile_total[state_index, profile_index] = primal_host[0]
            profile_aux[state_index, profile_index] = primal_host[1:]
            profile_gradient_u[state_index, profile_index] = derivative_host[0]
            profile_component_gradient_u[state_index, profile_index] = (
                derivative_host
            )

        reference_profile_index = int(
            np.argmin(np.abs(profile_sigmas - constrained_sigma))
        )
        reference_profile_indices[state_index] = reference_profile_index
        if abs(profile_sigmas[reference_profile_index] - constrained_sigma) > 2.0e-7:
            raise ValueError("The exact state reference sigma is absent from the grid.")
        reference_curvature = profile_curvature(
            jnp.asarray(math.log(constrained_sigma)), *profile_arguments
        )
        reference_curvature.block_until_ready()
        reference_curvature_total[state_index] = float(
            np.asarray(jax.device_get(reference_curvature))[0]
        )
        profile_reconstruction_error[state_index] = (
            profile_total[state_index, reference_profile_index] - full_value_host
        )
        observation_logprob_error[state_index] = (
            -profile_aux[state_index, reference_profile_index, 0]
            - float(jax.device_get(observation_log_probability))
        )
        stored_comparison = full_value_host
        if state["stored_target"] == "fixed":
            stored_comparison -= float(
                sigma_u_prior_terms(constrained_sigma, prior_scale)["potential"]
            )
        stored_potential_error[state_index] = (
            stored_comparison - float(state["stored_potential"])
        )

        for curvature_index, sigma in enumerate(args.curvature_sigmas):
            u = math.log(sigma)
            component_second_derivative = profile_curvature(
                jnp.asarray(u), *profile_arguments
            )
            component_second_derivative.block_until_ready()
            component_second_derivative_host = np.asarray(
                jax.device_get(component_second_derivative), dtype=np.float64
            )
            total_hessian = float(component_second_derivative_host[0])
            prior_terms = sigma_u_prior_terms(sigma, prior_scale)
            likelihood_hessian = total_hessian - float(
                prior_terms["curvature_uu"]
            )
            nearest = int(np.argmin(np.abs(profile_sigmas - sigma)))
            if abs(profile_sigmas[nearest] - sigma) > 2.0e-12:
                raise ValueError("A curvature sigma is absent from the profile grid.")
            likelihood_gradient = (
                profile_gradient_u[state_index, nearest]
                - float(prior_terms["gradient_u"])
            )
            cdf_terms = halfnormal_cdf_coordinate_terms(sigma, prior_scale)
            gaussian_coordinate[state_index, curvature_index] = cdf_terms["g"]
            gaussian_du_dg[state_index, curvature_index] = cdf_terms["du_dg"]
            gaussian_d2u_dg2[state_index, curvature_index] = cdf_terms[
                "d2u_dg2"
            ]
            gaussian_potential[state_index, curvature_index] = (
                profile_aux[state_index, nearest, 1]
                + 0.5 * cdf_terms["g"] ** 2
                + 0.5 * math.log(2.0 * math.pi)
            )
            gaussian_transform_potential_error[state_index, curvature_index] = (
                gaussian_potential[state_index, curvature_index]
                - (
                    profile_total[state_index, nearest]
                    - math.log(cdf_terms["du_dg"])
                )
            )
            gaussian_gradient[state_index, curvature_index] = (
                likelihood_gradient * cdf_terms["du_dg"] + cdf_terms["g"]
            )
            gaussian_curvature[state_index, curvature_index] = (
                likelihood_hessian * cdf_terms["du_dg"] ** 2
                + likelihood_gradient * cdf_terms["d2u_dg2"]
                + 1.0
            )
            curvature_total[state_index, curvature_index] = total_hessian
            curvature_likelihood[state_index, curvature_index] = likelihood_hessian
            curvature_components[state_index, curvature_index] = (
                component_second_derivative_host
            )
        timings.append(time.perf_counter() - state_start)
        del (
            loc,
            cov_diag,
            observed,
            reference_factor,
            reference_scale,
            unit_factor,
            column_norm,
            chip_column_norm,
            profile_arguments,
            shared_prior_potential,
            observation_log_probability,
            full_value,
        )
        gc.collect()

    initial_value_replay, initial_gradient_replay = value_and_gradient(
        jnp.asarray(initial_flat)
    )
    initial_value_replay.block_until_ready()
    initial_value_error = float(jax.device_get(initial_value_replay)) - initial_potential
    initial_gradient_error = np.asarray(
        jax.device_get(initial_gradient_replay), dtype=np.float64
    ) - initial_gradient

    fixed_state_index = 1
    cross_columns = np.empty(
        (len(args.curvature_sigmas), len(args.cross_u_steps), FLAT_DIMENSION),
        dtype=np.float64,
    )
    cross_value_plus = np.empty(
        (len(args.curvature_sigmas), len(args.cross_u_steps)), dtype=np.float64
    )
    cross_value_minus = np.empty_like(cross_value_plus)
    cross_center_value = np.empty((len(args.curvature_sigmas),), dtype=np.float64)
    cross_center_gradient = np.empty(
        (len(args.curvature_sigmas), FLAT_DIMENSION), dtype=np.float64
    )
    cross_gradient_plus = np.empty(
        (len(args.curvature_sigmas), len(args.cross_u_steps), FLAT_DIMENSION),
        dtype=np.float64,
    )
    cross_gradient_minus = np.empty_like(cross_gradient_plus)
    for sigma_index, sigma in enumerate(args.curvature_sigmas):
        center = np.asarray(states[fixed_state_index]["flat"], dtype=np.float64).copy()
        center[SIGMA_INDEX] = math.log(sigma)
        center_value, center_gradient = value_and_gradient(jnp.asarray(center))
        center_value.block_until_ready()
        cross_center_value[sigma_index] = float(jax.device_get(center_value))
        cross_center_gradient[sigma_index] = np.asarray(
            jax.device_get(center_gradient), dtype=np.float64
        )
        for step_index, step in enumerate(args.cross_u_steps):
            plus = center.copy()
            minus = center.copy()
            plus[SIGMA_INDEX] += step
            minus[SIGMA_INDEX] -= step
            value_plus, gradient_plus = value_and_gradient(jnp.asarray(plus))
            value_minus, gradient_minus = value_and_gradient(jnp.asarray(minus))
            value_plus.block_until_ready()
            value_minus.block_until_ready()
            gradient_plus_host = np.asarray(
                jax.device_get(gradient_plus), dtype=np.float64
            )
            gradient_minus_host = np.asarray(
                jax.device_get(gradient_minus), dtype=np.float64
            )
            cross_value_plus[sigma_index, step_index] = float(
                jax.device_get(value_plus)
            )
            cross_value_minus[sigma_index, step_index] = float(
                jax.device_get(value_minus)
            )
            cross_gradient_plus[sigma_index, step_index] = gradient_plus_host
            cross_gradient_minus[sigma_index, step_index] = gradient_minus_host
            cross_columns[sigma_index, step_index] = (
                gradient_plus_host - gradient_minus_host
            ) / (2.0 * step)

    cross_shared_raw = cross_columns[..., :SIGMA_INDEX]
    cross_shared_scaled = (
        cross_shared_raw * shared_robust_scale[None, None, :]
    )
    cross_norm_raw = np.linalg.norm(cross_shared_raw, axis=-1)
    cross_norm_scaled = np.linalg.norm(cross_shared_scaled, axis=-1)
    cdf_cross_shared_scaled = np.empty(
        (len(args.curvature_sigmas), len(args.cross_u_steps)), dtype=np.float64
    )
    for sigma_index in range(len(args.curvature_sigmas)):
        cdf_cross_shared_scaled[sigma_index] = (
            cross_norm_scaled[sigma_index]
            * gaussian_du_dg[fixed_state_index, sigma_index]
        )

    atmosphere_rotation = np.asarray(
        wrapper_args.atmosphere_rotation_matrix, dtype=np.float64
    ).reshape(7, 7)
    cross_atmosphere_gaussianized = np.einsum(
        "ij,skj->ski",
        atmosphere_rotation,
        cross_shared_raw[..., :7],
    )
    cross_huu_profile_error = (
        cross_columns[..., SIGMA_INDEX]
        - curvature_total[fixed_state_index, :, None]
    )
    curvature_profile_indices = np.asarray(
        [
            int(np.argmin(np.abs(profile_sigmas - sigma)))
            for sigma in args.curvature_sigmas
        ],
        dtype=np.int64,
    )
    cross_center_value_error = (
        cross_center_value
        - profile_total[fixed_state_index, curvature_profile_indices]
    )
    cross_center_gradient_u_error = (
        cross_center_gradient[:, SIGMA_INDEX]
        - profile_gradient_u[fixed_state_index, curvature_profile_indices]
    )

    primary_step_index = int(
        np.argmin(np.abs(np.asarray(args.cross_u_steps) - 0.01))
    )
    primary_cross_step = float(args.cross_u_steps[primary_step_index])
    if abs(primary_cross_step - 0.01) > 1.0e-12:
        raise ValueError("cross-u-steps must contain the primary 0.01 step.")
    probe_cross_state_indices = np.arange(2, len(states), dtype=np.int64)
    probe_cross_columns = np.empty(
        (len(probe_cross_state_indices), FLAT_DIMENSION), dtype=np.float64
    )
    for probe_output_index, state_index in enumerate(probe_cross_state_indices):
        center = np.asarray(states[state_index]["flat"], dtype=np.float64).copy()
        plus = center.copy()
        minus = center.copy()
        plus[SIGMA_INDEX] += primary_cross_step
        minus[SIGMA_INDEX] -= primary_cross_step
        _, gradient_plus = value_and_gradient(jnp.asarray(plus))
        _, gradient_minus = value_and_gradient(jnp.asarray(minus))
        gradient_plus.block_until_ready()
        gradient_minus.block_until_ready()
        probe_cross_columns[probe_output_index] = (
            np.asarray(jax.device_get(gradient_plus), dtype=np.float64)
            - np.asarray(jax.device_get(gradient_minus), dtype=np.float64)
        ) / (2.0 * primary_cross_step)
    probe_cross_shared_scaled = (
        probe_cross_columns[:, :SIGMA_INDEX] * shared_robust_scale[None, :]
    )
    probe_cross_shared_norm_scaled = np.linalg.norm(
        probe_cross_shared_scaled, axis=1
    )
    probe_cross_atmosphere_gaussianized = np.einsum(
        "ij,pj->pi",
        atmosphere_rotation,
        probe_cross_columns[:, :7],
    )
    probe_gaussian_du_dg = np.asarray(
        [
            halfnormal_cdf_coordinate_terms(
                states[state_index]["reference_sigma"], prior_scale
            )["du_dg"]
            for state_index in probe_cross_state_indices
        ]
    )
    probe_cross_shared_norm_cdf_g_scaled = (
        probe_cross_shared_norm_scaled * probe_gaussian_du_dg
    )
    probe_cross_huu_profile_error = (
        probe_cross_columns[:, SIGMA_INDEX]
        - reference_curvature_total[probe_cross_state_indices]
    )

    profile_prior = sigma_u_prior_terms(profile_sigmas, prior_scale)
    profile_conditioned = profile_aux[:, :, 1]
    profile_nll = profile_aux[:, :, 0]
    profile_gradient_likelihood = (
        profile_gradient_u - profile_prior["gradient_u"][None, :]
    )
    reference_gradient_u_error = np.asarray(
        [
            reference_full_gradient[index, SIGMA_INDEX]
            - profile_gradient_u[index, reference_profile_indices[index]]
            for index in range(len(states))
        ]
    )
    shared_prior_reconstruction_span = np.ptp(
        profile_conditioned - profile_nll,
        axis=1,
    )
    covariance_sensitivity_squared = np.sum(
        whitened_unit_column_norm_squared * eigenvalues_host[None, :],
        axis=1,
    )
    covariance_sensitivity_squared_by_chip = np.sum(
        whitened_unit_column_norm_squared_by_chip
        * eigenvalues_host[None, None, :],
        axis=2,
    )
    conditioned_delta = profile_conditioned - profile_conditioned[:, [
        int(np.argmin(np.abs(profile_sigmas - m8_v2_run.V17_INITIAL_SIGMA_LOG_P)))
    ]]

    finite_arrays = (
        profile_total,
        profile_aux,
        profile_gradient_u,
        profile_component_gradient_u,
        curvature_total,
        curvature_likelihood,
        curvature_components,
        cross_columns,
        cross_value_plus,
        cross_value_minus,
        cross_gradient_plus,
        cross_gradient_minus,
        cross_center_value,
        cross_center_gradient,
        probe_cross_columns,
        gaussian_gradient,
        gaussian_curvature,
        gaussian_transform_potential_error,
        reference_full_gradient,
        whitened_unit_column_norm_squared,
        whitened_unit_column_norm_squared_by_chip,
    )
    all_finite = bool(all(np.all(np.isfinite(array)) for array in finite_arrays))
    maximum_profile_reconstruction_error = float(
        np.max(np.abs(profile_reconstruction_error))
    )
    maximum_stored_potential_error = float(np.max(np.abs(stored_potential_error)))
    maximum_observation_logprob_error = float(
        np.max(np.abs(observation_logprob_error))
    )
    maximum_initial_gradient_error = float(np.max(np.abs(initial_gradient_error)))
    maximum_reference_gradient_u_error = float(
        np.max(np.abs(reference_gradient_u_error))
    )
    maximum_shared_prior_span = float(
        np.max(np.abs(shared_prior_reconstruction_span))
    )
    maximum_gaussian_transform_error = float(
        np.max(np.abs(gaussian_transform_potential_error))
    )
    maximum_cross_center_value_error = float(
        np.max(np.abs(cross_center_value_error))
    )
    maximum_cross_center_gradient_u_error = float(
        np.max(np.abs(cross_center_gradient_u_error))
    )
    primary_cross_huu_relative_error = np.abs(
        cross_huu_profile_error[:, primary_step_index]
    ) / np.maximum(
        1.0,
        np.abs(curvature_total[fixed_state_index]),
    )
    probe_cross_huu_relative_error = np.abs(
        probe_cross_huu_profile_error
    ) / np.maximum(
        1.0,
        np.abs(reference_curvature_total[probe_cross_state_indices]),
    )
    primary_cross_shared = cross_shared_raw[:, primary_step_index, :]
    cross_step_relative_difference = np.linalg.norm(
        cross_shared_raw - primary_cross_shared[:, None, :],
        axis=2,
    ) / np.maximum(
        1.0,
        np.linalg.norm(primary_cross_shared, axis=1)[:, None],
    )
    cross_step_cosine = np.sum(
        cross_shared_raw * primary_cross_shared[:, None, :], axis=2
    ) / np.maximum(
        1.0e-30,
        np.linalg.norm(cross_shared_raw, axis=2)
        * np.linalg.norm(primary_cross_shared, axis=1)[:, None],
    )
    numerical_integrity_passed = bool(
        all_finite
        and maximum_profile_reconstruction_error <= args.potential_atol
        and maximum_stored_potential_error <= args.potential_atol
        and maximum_observation_logprob_error <= args.potential_atol
        and abs(initial_value_error) <= args.potential_atol
        and maximum_initial_gradient_error <= 2.0e-3
        and maximum_reference_gradient_u_error <= 2.0e-2
        and maximum_shared_prior_span <= args.potential_atol
        and maximum_gaussian_transform_error <= args.potential_atol
        and maximum_cross_center_value_error <= args.potential_atol
        and maximum_cross_center_gradient_u_error <= 2.0e-2
        and float(np.max(primary_cross_huu_relative_error)) <= 5.0e-2
        and float(np.max(probe_cross_huu_relative_error)) <= 5.0e-2
        and float(np.max(cross_step_relative_difference)) <= 2.5e-1
        and float(np.max(roundtrip_error)) <= 1.0e-6
    )

    state_names = [state["name"] for state in states]
    state_sources = [int(state["source_index"]) for state in states]
    state_flat = np.stack([np.asarray(state["flat"]) for state in states])
    arrays = {
        "state_names": np.asarray(state_names),
        "state_source_indices": np.asarray(state_sources, dtype=np.int64),
        "state_flat_unconstrained": state_flat.astype(np.float32),
        "fixed_medoid_robust_center": fixed_medoid_center,
        "fixed_medoid_robust_scale": shared_robust_scale,
        "free_medoid_robust_center": free_medoid_center,
        "free_medoid_robust_scale": free_medoid_scale,
        "coordinate_names": np.asarray(
            [
                *(f"atmosphere_rotated[{index}]" for index in range(7)),
                *(f"A_unconstrained[{index}]" for index in range(4)),
                "u_sigma_log_p",
            ]
        ),
        "profile_sigma": profile_sigmas,
        "profile_u": profile_u,
        "profile_total_potential": profile_total,
        "profile_conditioned_potential": profile_conditioned,
        "profile_conditioned_delta_from_initial_sigma": conditioned_delta,
        "profile_negative_log_likelihood": profile_nll,
        "profile_sigma_prior_potential": profile_prior["potential"],
        "profile_sigma_negative_log_prior_physical": (
            profile_prior["potential"] + profile_prior["u"]
        ),
        "profile_sigma_transform_log_jacobian": profile_prior["u"],
        "profile_log_determinant": profile_aux[:, :, 3],
        "profile_mahalanobis": profile_aux[:, :, 4],
        "profile_whitened_factor_trace": profile_aux[:, :, 5],
        "profile_gradient_u_total": profile_gradient_u,
        "profile_gradient_u_likelihood": profile_gradient_likelihood,
        "profile_component_gradient_u": profile_component_gradient_u,
        "profile_component_names": np.asarray(
            [
                "total_potential",
                "negative_log_likelihood",
                "conditioned_potential",
                "sigma_prior_potential",
                "log_determinant",
                "mahalanobis",
                "whitened_factor_trace",
            ]
        ),
        "curvature_sigma": np.asarray(args.curvature_sigmas),
        "curvature_uu_total": curvature_total,
        "curvature_uu_likelihood": curvature_likelihood,
        "curvature_uu_components": curvature_components,
        "gaussianized_sigma_coordinate": gaussian_coordinate,
        "gaussianized_du_dg": gaussian_du_dg,
        "gaussianized_d2u_dg2": gaussian_d2u_dg2,
        "gaussianized_total_potential": gaussian_potential,
        "gaussianized_transform_potential_error": (
            gaussian_transform_potential_error
        ),
        "gaussianized_gradient_g_total": gaussian_gradient,
        "gaussianized_curvature_gg_total": gaussian_curvature,
        "cross_u_steps": np.asarray(args.cross_u_steps),
        "cross_gradient_plus": cross_gradient_plus,
        "cross_gradient_minus": cross_gradient_minus,
        "cross_center_value": cross_center_value,
        "cross_center_gradient": cross_center_gradient,
        "cross_center_value_minus_cached_profile": cross_center_value_error,
        "cross_center_gradient_u_minus_cached_profile": (
            cross_center_gradient_u_error
        ),
        "cross_value_plus": cross_value_plus,
        "cross_value_minus": cross_value_minus,
        "cross_hessian_column_raw": cross_columns,
        "cross_atmosphere_gaussianized": cross_atmosphere_gaussianized,
        "cross_huu_minus_cached_profile_huu": cross_huu_profile_error,
        "cross_step_relative_difference_from_primary": (
            cross_step_relative_difference
        ),
        "cross_step_cosine_with_primary": cross_step_cosine,
        "cross_shared_norm_raw": cross_norm_raw,
        "cross_shared_norm_fixed_chain_scaled": cross_norm_scaled,
        "cross_shared_norm_cdf_g_fixed_chain_scaled": cdf_cross_shared_scaled,
        "probe_cross_state_indices": probe_cross_state_indices,
        "probe_cross_sigma": np.asarray(
            [states[index]["reference_sigma"] for index in probe_cross_state_indices]
        ),
        "probe_cross_primary_u_step": np.asarray(primary_cross_step),
        "probe_cross_hessian_column_raw": probe_cross_columns,
        "probe_cross_atmosphere_gaussianized": (
            probe_cross_atmosphere_gaussianized
        ),
        "probe_cross_shared_norm_fixed_chain_scaled": (
            probe_cross_shared_norm_scaled
        ),
        "probe_cross_shared_norm_cdf_g_fixed_chain_scaled": (
            probe_cross_shared_norm_cdf_g_scaled
        ),
        "probe_cross_huu_minus_cached_profile_huu": (
            probe_cross_huu_profile_error
        ),
        "reference_curvature_uu_total": reference_curvature_total,
        "reference_full_gradient": reference_full_gradient,
        "gp_eigenvalues": eigenvalues_host,
        "whitened_unit_column_norm_squared": whitened_unit_column_norm_squared,
        "whitened_unit_column_norm_squared_by_chip": (
            whitened_unit_column_norm_squared_by_chip
        ),
        "covariance_sensitivity_squared": covariance_sensitivity_squared,
        "covariance_sensitivity_squared_by_chip": (
            covariance_sensitivity_squared_by_chip
        ),
        "profile_sigma_times_covariance_sensitivity": (
            np.sqrt(covariance_sensitivity_squared)[:, None]
            * profile_sigmas[None, :]
        ),
        "profile_reconstruction_error": profile_reconstruction_error,
        "observation_logprob_error": observation_logprob_error,
        "stored_potential_error": stored_potential_error,
        "roundtrip_error": roundtrip_error,
        "initial_value_error": np.asarray(initial_value_error),
        "initial_gradient_error": initial_gradient_error,
        "reference_gradient_u_error": reference_gradient_u_error,
        "shared_prior_reconstruction_span": shared_prior_reconstruction_span,
    }
    summary = {
        "schema_version": 1,
        "mode": "m8_v5_free_sigma_geometry_diagnostic",
        "execution_completed": True,
        "numerical_integrity_passed": numerical_integrity_passed,
        "purpose": (
            "Separate the absolute sigma_log_p response of the frozen linear "
            "marginal likelihood from local coupling introduced by a free "
            "u=log(sigma_log_p) coordinate."
        ),
        "target": {
            "model": "frozen M8 linear marginalized target",
            "sigma_log_p_prior": "HalfNormal(scale=0.3 dex)",
            "map_latents": "analytically marginalized",
            "exact_rt_used": False,
            "x64": bool(args.x64),
            "seed": int(args.seed),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpyro": numpyro.__version__,
            "exojax": exojax.__version__,
            "exojax_import_path": inspect.getfile(exojax),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "coordinates": {
            "explicit_order": [
                "atmosphere_rotated[0:7]",
                "A_unconstrained[0:4]",
                "u_sigma_log_p=log(sigma_log_p)",
            ],
            "order_is_numpyro_mass_order": False,
            "direct_positive_transform": "sigma_log_p=exp(u_sigma_log_p)",
            "comparison_coordinate": (
                "g=Phi^-1(F_HalfNormal(sigma_log_p)); same physical prior"
            ),
            "cross_scaling": (
                "shared raw coordinates are multiplied by the fixed-control "
                "chain robust MAD scales; no full Hessian or Schur complement"
            ),
        },
        "configuration": {
            "profile_sigma_min": args.profile_sigma_min,
            "profile_sigma_max": args.profile_sigma_max,
            "profile_num_requested": args.profile_num,
            "profile_num_actual": len(profile_sigmas),
            "sigma_markers": args.sigma_markers,
            "curvature_sigmas": args.curvature_sigmas,
            "cross_u_steps": args.cross_u_steps,
            "potential_atol": args.potential_atol,
            "gp_jitter": gp_jitter,
        },
        "state_selection": validation["state_selection"],
        "free_chain_warning": validation["free_chain_warning"],
        "fixed_control_warning": validation["fixed_control_warning"],
        "integrity": {
            "all_scientific_arrays_finite": all_finite,
            "maximum_profile_vs_full_potential_error": (
                maximum_profile_reconstruction_error
            ),
            "maximum_stored_potential_replay_error": (
                maximum_stored_potential_error
            ),
            "maximum_manual_vs_distribution_logprob_error": (
                maximum_observation_logprob_error
            ),
            "initial_potential_replay_error": initial_value_error,
            "maximum_initial_gradient_replay_error": (
                maximum_initial_gradient_error
            ),
            "maximum_reference_sigma_gradient_reconstruction_error": (
                maximum_reference_gradient_u_error
            ),
            "maximum_shared_prior_profile_span": maximum_shared_prior_span,
            "maximum_gaussianized_transform_potential_error": (
                maximum_gaussian_transform_error
            ),
            "maximum_fixed_marker_full_vs_cached_value_error": (
                maximum_cross_center_value_error
            ),
            "maximum_fixed_marker_full_vs_cached_gradient_u_error": (
                maximum_cross_center_gradient_u_error
            ),
            "cross_huu_minus_cached_profile_huu_by_step": np.max(
                np.abs(cross_huu_profile_error), axis=0
            ),
            "maximum_primary_cross_huu_relative_error": float(
                np.max(primary_cross_huu_relative_error)
            ),
            "maximum_probe_cross_huu_relative_error": float(
                np.max(probe_cross_huu_relative_error)
            ),
            "maximum_cross_step_relative_difference_from_primary": float(
                np.max(cross_step_relative_difference)
            ),
            "minimum_cross_step_cosine_with_primary": float(
                np.min(cross_step_cosine)
            ),
            "maximum_constrained_roundtrip_error": float(
                np.max(roundtrip_error)
            ),
        },
        "primary_outputs": {
            "conditional_profile": (
                "conditioned potential, log determinant, Mahalanobis term, "
                "and sigma gradient at each actual shared state"
            ),
            "local_cross_column": (
                "d grad(theta,u) / d u at the fixed-control actual draw, "
                "using central differences of the production first gradient"
            ),
            "coordinate_comparison": (
                "local gradient, curvature, and scaled cross norm after the "
                "same-prior Gaussian inverse-CDF reparameterization"
            ),
        },
        "limitations": [
            "This is an exploratory local-geometry diagnostic, not posterior sampling.",
            "Conditional slices hold each shared state fixed; they are not marginalized or profiled posteriors.",
            "The free-chain actual draws are sensitivity probes from a nonstationary one-chain run, not ground truth.",
            "The fixed control still hit the depth cap in 36 percent of retained draws and is not labeled a sampler success.",
            "Only the sigma Hessian column is estimated. A full Hessian was deliberately avoided because it is unnecessary here and risks high-order RT autodiff/OOM.",
            "A fixed-sigma 0.423 sampling arm remains necessary if the local diagnostic does not settle the absolute-value effect.",
        ],
        "timing_seconds": {
            "per_state": timings,
            "total": time.perf_counter() - start_time,
        },
        "provenance_sha256": {
            **validation["sha256"],
            "diagnostic_script": sha256_file(Path(__file__)),
            "replay_helper": sha256_file(Path(replay.__file__)),
            "fixed_wrapper": sha256_file(Path(m8_v2_run.__file__)),
            "workflow": sha256_file(Path(retrieval.__file__)),
        },
    }
    return summary, arrays


def main() -> None:
    """Validate inputs or run the guarded GPU diagnostic."""

    args = parse_args()
    validation = _validate_configuration(args)
    validation_payload = {
        "mode": "m8_v5_free_sigma_geometry_validation",
        "validated": True,
        "output_dir": str(Path(args.out_dir)),
        "state_selection": validation["state_selection"],
        "profile": {
            "sigma_min": args.profile_sigma_min,
            "sigma_max": args.profile_sigma_max,
            "num": args.profile_num,
            "markers": args.sigma_markers,
            "curvature_sigmas": args.curvature_sigmas,
            "cross_u_steps": args.cross_u_steps,
        },
    }
    if args.validate_only:
        print(json.dumps(_json_ready(validation_payload), indent=2, allow_nan=False))
        return

    output_dir = Path(args.out_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(
            "The launcher must claim the fresh output directory before execution: "
            f"{output_dir}"
        )
    summary_path = output_dir / SUMMARY_NAME
    arrays_path = output_dir / ARRAYS_NAME
    summary, arrays = _run_gpu_diagnostic(args, validation)
    _write_npz_atomic(arrays_path, arrays)
    _write_json_atomic(summary_path, summary)
    print(json.dumps(_json_ready(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
