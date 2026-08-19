"""Guard and validate the M8 v10 long highest-precision free-sigma run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doraex.workflows import on_the_fly_pressure_retrieval as retrieval  # noqa: E402
from examples.luhman16b_yama import m8_v1_run  # noqa: E402


MODE = "m8_v10_free_sigma_highest_precision_long_control"
RUN_LABEL = "m8_v10_free_sigma_highest_precision_long"
DEFAULT_OUT_DIR = ROOT / "results/m8/v10/free_sigma_highest_precision_long_seed0"
DEFAULT_INIT = ROOT / "results/m7/p2/v6/samples.npz"
DEFAULT_V1_SAMPLES = ROOT / "results/m8/v1/samples.npz"
DEFAULT_V1_DIAGNOSTICS = ROOT / "results/m8/v1/diagnostics.json"
DEFAULT_V9_DIR = ROOT / "results/m8/v9/free_sigma_highest_precision_seed0"
CONTRACT_NAME = "m8_v10_free_sigma_highest_precision_long_contract.json"
SUMMARY_NAME = "m8_v10_free_sigma_highest_precision_long_control_summary.json"
EXPECTED_PRECISION = "highest"
EXPECTED_INITIAL_SIGMA = 0.32316479086875916
NUM_WARMUP = 2000
NUM_SAMPLES = 1500
NUM_CHAINS = 1
SEED = 0
TARGET_ACCEPT = 0.95
WARMUP_DEPTH = 9
SAMPLE_DEPTH = 11
CAP_NUM_STEPS = 2**SAMPLE_DEPTH - 1

EXPECTED_STATIC_SHA256 = {
    "initialization_archive": (
        "3256e8e10998c521313ab14473ac58b1e8aebb1e567a2593619477c0bfe70f89"
    ),
    "v1_samples": ("aa1bd9bb6bc9e5f146e3377aa62e3adb68c8c9a961812628144c73cd9c3a5083"),
    "v1_diagnostics": (
        "f45d32470ee996c62b0dd8683190d4fe5eaedddd37d4f95d0ea21003278cf8c4"
    ),
    "sampler_wrapper": (
        "fb17a96b1718d65a680ae6162b010cd7f0fac0a577de275d4c8df8a562312423"
    ),
    "retrieval_workflow": (
        "582f79edad35a1aa819cf6e619caeaabbac614a1ea173dca25fbbf4f8ce41230"
    ),
    "forward_source": (
        "63390bbf64f436a5ed3cfd022454b22b6e6377aebe909acb21a32a5be0859a21"
    ),
}

EXPECTED_V9_SHA256 = {
    "COMPLETE": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "GRADIENT_VALIDATED": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "initial_sigma_score_guard.json": (
        "7576255f0b1e8d9af6b03e3896cfdb62eae5d4f323fd20058c88b15abc2351ac"
    ),
    "m8_v9_free_sigma_highest_precision_control_summary.json": (
        "a1a1000166981637e4a8ac988d713c184b44664431d384dd7e94f10a179c8f81"
    ),
    "samples.npz": ("7c00bafb01dd06394193fd58c0f026aa92d47bae4cef3fdc3953156f5d1e302f"),
    "diagnostics.json": (
        "30d9eec185974be416f7e223d6757be5bdcf881ecafb671896b7510ee8a09bdb"
    ),
    "outputs.sha256": (
        "92bd64137719f84e316b7438d15c0014c183dd9e592895fa82dc2ced334cbba6"
    ),
    "provenance.sha256": (
        "6d8b79be202ecfb0f01dfe95143d8c2c61151e80bbb33d6afc8a54340ebcb73e"
    ),
}

TARGET_EQUIVALENCE_KEYS = (
    "adapt_mass_matrix",
    "chip_indices",
    "dense_mass",
    "direct_sigma_log_p",
    "fix_a",
    "fix_log_w",
    "fix_logg",
    "fix_sigma_d",
    "fixed_ell_b",
    "fixed_sigma_log_p",
    "full_data",
    "gaussianized_atmosphere",
    "init_from",
    "init_logg",
    "matmul_precision",
    "max_tree_depth",
    "nside",
    "num_chains",
    "pressure_gp_factorization",
    "seed",
    "sigma_log_p_scale",
    "target_accept_prob",
    "warmup_max_tree_depth",
    "x64",
    "zero_mean_log_w",
    "zero_mean_pressure_map",
    "zero_sum_log_w_basis",
)


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_sha256(path: Path, expected: str) -> str:
    """Return a verified file digest or raise on provenance drift."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen input: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}."
        )
    return actual


def _write_strict_json(path: Path, payload: dict) -> None:
    """Atomically write strict JSON without accepting an output collision."""

    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing output collision: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def _sampling_cli(init_from: Path, out_dir: Path) -> list[str]:
    """Return the explicit long M8 v1 sampling arguments."""

    return [
        "--init-from",
        str(Path(init_from).resolve()),
        "--out-dir",
        str(Path(out_dir).resolve()),
        "--run-label",
        RUN_LABEL,
        "--num-chains",
        str(NUM_CHAINS),
        "--seed",
        str(SEED),
        "--num-warmup",
        str(NUM_WARMUP),
        "--num-samples",
        str(NUM_SAMPLES),
        "--target-accept-prob",
        str(TARGET_ACCEPT),
        "--warmup-max-tree-depth",
        str(WARMUP_DEPTH),
        "--max-tree-depth",
        str(SAMPLE_DEPTH),
        "--dense-mass",
        "--adapt-mass-matrix",
        "--no-x64",
        "--print-summary",
    ]


def _capture_wrapper_args(arguments: list[str]):
    """Capture one M8 v1 wrapper configuration without sampling."""

    captured = {}
    original_main = retrieval.main
    original_argv = sys.argv

    def capture_args() -> None:
        captured["args"] = retrieval.parse_args()

    try:
        retrieval.main = capture_args
        sys.argv = [str(Path(m8_v1_run.__file__)), *arguments]
        m8_v1_run.main()
    finally:
        retrieval.main = original_main
        sys.argv = original_argv
    return captured["args"]


def _capture_sampling_args(init_from: Path, out_dir: Path):
    """Capture the exact v10 long wrapper configuration without sampling."""

    return _capture_wrapper_args(_sampling_cli(init_from, out_dir))


def _initial_sigma(init_from: Path) -> float:
    """Return the physical sigma initialization loaded by M8 v1."""

    with np.load(init_from, allow_pickle=False) as archive:
        sigma = float(np.median(np.asarray(archive["sigma_log_p"])))
    if sigma != EXPECTED_INITIAL_SIGMA:
        raise ValueError(
            "Initial sigma_log_p drifted: "
            f"expected {EXPECTED_INITIAL_SIGMA!r}, got {sigma!r}."
        )
    return sigma


def _sampling_contract(args) -> dict:
    """Validate and serialize the frozen long free-sigma contract."""

    expected = {
        "chip_indices": [0, 1, 2, 3],
        "full_data": True,
        "nside": 8,
        "fixed_ell_b": 0.4,
        "pressure_gp_factorization": "fixed_eigen",
        "fix_logg": True,
        "init_logg": 4.86,
        "zero_mean_pressure_map": True,
        "zero_mean_log_w": True,
        "zero_sum_log_w_basis": True,
        "fix_a": False,
        "fix_log_w": True,
        "fix_sigma_d": True,
        "gaussianized_atmosphere": True,
        "direct_sigma_log_p": True,
        "fixed_sigma_log_p": None,
        "sigma_log_p_scale": 0.3,
        "dense_mass": True,
        "adapt_mass_matrix": True,
        "preflight_autodiff": False,
        "num_chains": NUM_CHAINS,
        "seed": SEED,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "target_accept_prob": TARGET_ACCEPT,
        "warmup_max_tree_depth": WARMUP_DEPTH,
        "max_tree_depth": SAMPLE_DEPTH,
        "x64": False,
        "print_summary": True,
        "run_label": RUN_LABEL,
    }
    actual = {name: getattr(args, name) for name in expected}
    for name, expected_value in expected.items():
        actual_value = actual[name]
        if isinstance(expected_value, float):
            matches = math.isclose(
                float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = actual_value == expected_value
        if not matches:
            raise ValueError(
                f"Sampling contract drift for {name}: "
                f"expected {expected_value!r}, got {actual_value!r}."
            )

    baseline = _capture_wrapper_args([])
    differences = {
        name for name, value in vars(baseline).items() if value != vars(args)[name]
    }
    if differences != {"out_dir", "run_label"}:
        raise ValueError(
            "The v10 arguments do not isolate output metadata from M8 v1: "
            f"{sorted(differences)}."
        )
    return {
        **actual,
        "init_from": str(Path(args.init_from).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "matmul_precision": EXPECTED_PRECISION,
        "precision_scope": "dedicated Python process",
        "initial_sigma_log_p": _initial_sigma(Path(args.init_from)),
        "maximum_tree_num_steps": CAP_NUM_STEPS,
        "m8_v1_differs_only_by": ["out_dir", "run_label"],
        "sampling_argv": _sampling_cli(args.init_from, args.out_dir),
    }


def _target_view(contract: dict) -> dict:
    """Return schedule-independent fields needed to reuse the v9 guard."""

    return {name: contract.get(name) for name in TARGET_EQUIVALENCE_KEYS}


def _validate_v9_evidence(v9_dir: Path, long_contract: dict | None = None) -> dict:
    """Validate and summarize the frozen v9 guard evidence."""

    v9_dir = Path(v9_dir)
    if (v9_dir / "RUNNING").exists() or (v9_dir / "FAILED").exists():
        raise ValueError("The frozen v9 evidence has inconsistent status files.")
    files = {}
    for name, expected in EXPECTED_V9_SHA256.items():
        path = v9_dir / name
        files[name] = _checked_sha256(path, expected)

    guard = json.loads(
        (v9_dir / "initial_sigma_score_guard.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (v9_dir / "m8_v9_free_sigma_highest_precision_control_summary.json").read_text(
            encoding="utf-8"
        )
    )
    score_difference = float(guard.get("results", {}).get("score_absolute_difference"))
    score_atol = float(guard.get("thresholds", {}).get("score_atol"))
    if not (
        guard.get("execution_completed") is True
        and guard.get("passed") is True
        and guard.get("precision") == EXPECTED_PRECISION
        and guard.get("results", {}).get("all_finite") is True
        and score_difference <= score_atol
        and float(guard.get("initial_point", {}).get("physical_sigma_log_p"))
        == EXPECTED_INITIAL_SIGMA
        and summary.get("execution_completed") is True
        and summary.get("artifact_integrity_passed") is True
        and summary.get("precision") == EXPECTED_PRECISION
        and summary.get("artifact_sha256", {}).get("initial_score_guard")
        == EXPECTED_V9_SHA256["initial_sigma_score_guard.json"]
    ):
        raise ValueError("The frozen v9 guard evidence no longer matches.")
    v9_contract = guard.get("sampling_contract", {})
    if long_contract is not None and _target_view(v9_contract) != _target_view(
        long_contract
    ):
        raise ValueError("The v9 guard target differs from the long-run target.")
    return {
        "guard_reused_without_recomputation": True,
        "precision": EXPECTED_PRECISION,
        "physical_sigma_log_p": EXPECTED_INITIAL_SIGMA,
        "score_absolute_difference": score_difference,
        "score_atol": score_atol,
        "short_run_tree_depth_cap_fraction": float(
            summary["findings"]["tree_depth_cap_fraction"]
        ),
        "short_run_median_num_steps": float(summary["findings"]["num_steps"]["median"]),
        "target_equivalence_fields": _target_view(v9_contract),
        "files_sha256": files,
    }


def _static_provenance(args) -> dict:
    """Validate and return the static long-control provenance."""

    paths = {
        "initialization_archive": Path(args.init_from),
        "v1_samples": Path(args.v1_samples),
        "v1_diagnostics": Path(args.v1_diagnostics),
        "sampler_wrapper": Path(m8_v1_run.__file__),
        "retrieval_workflow": Path(retrieval.__file__),
        "forward_source": ROOT / "src/doraex/spectra/exojax_forward.py",
    }
    return {
        name: {
            "path": str(path.resolve()),
            "sha256": _checked_sha256(path, EXPECTED_STATIC_SHA256[name]),
        }
        for name, path in paths.items()
    }


def _validate_inputs(args) -> tuple[object, dict, dict, dict]:
    """Validate all static inputs without compiling or sampling the target."""

    sampling_args = _capture_sampling_args(args.init_from, args.out_dir)
    contract = _sampling_contract(sampling_args)
    evidence = _validate_v9_evidence(args.v9_dir, contract)
    provenance = _static_provenance(args)
    return sampling_args, contract, evidence, provenance


def _require_highest_precision() -> None:
    """Require the process-global precision used by the long sampler."""

    precision = jax.config.jax_default_matmul_precision
    if precision != EXPECTED_PRECISION:
        raise RuntimeError(
            "JAX_DEFAULT_MATMUL_PRECISION must be highest before JAX import; "
            f"got {precision!r}."
        )


def _expected_contract_payload(args) -> dict:
    """Return the deterministic v10 run-contract payload."""

    _, contract, evidence, provenance = _validate_inputs(args)
    return {
        "schema_version": 1,
        "mode": f"{MODE}_contract",
        "execution_completed": True,
        "precision": EXPECTED_PRECISION,
        "sampling_contract": contract,
        "v9_guard_evidence": evidence,
        "provenance_sha256": provenance,
        "limitations": [
            "The v9 one-point score guard is reused rather than recomputed.",
            "This contract does not validate the linearized forward model.",
        ],
    }


def write_contract(args) -> dict:
    """Validate precision and atomically write the v10 run contract."""

    _require_highest_precision()
    payload = _expected_contract_payload(args)
    _write_strict_json(args.contract_output, payload)
    return payload


def _scalar(archive, name):
    """Return one scalar NPZ value."""

    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"Expected scalar {name}, got {value.shape}.")
    return value.item()


def _validate_diagnostics(diagnostics: dict, *, run_label: str) -> None:
    """Validate the long M8 diagnostic configuration."""

    expected = {
        "mode": run_label,
        "chip_indices": [0, 1, 2, 3],
        "full_data": True,
        "nside": 8,
        "fixed_ell_b": 0.4,
        "gp_jitter": 0.5e-6,
        "pressure_gp_factorization": "fixed_eigen",
        "fix_logg": True,
        "fixed_logg": 4.86,
        "zero_mean_pressure_map": True,
        "zero_mean_log_w": True,
        "zero_sum_log_w_basis": True,
        "fix_a": False,
        "fix_log_w": True,
        "fix_sigma_d": True,
        "direct_sigma_log_p": True,
        "fixed_sigma_log_p": None,
        "sigma_log_p_parameterization": "direct_halfnormal",
        "gaussianized_atmosphere": True,
        "atmosphere_rotation_label": (
            "m7_v1_yama_cdf_gaussianized_bounded7_polar_n1500"
        ),
        "dense_mass": True,
        "dense_mass_mode": "full",
        "adapt_mass_matrix": True,
        "initial_inverse_mass_matrix_mode": "identity",
        "map_init": False,
        "manual_atmosphere_init": False,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "num_chains": NUM_CHAINS,
        "target_accept_prob": TARGET_ACCEPT,
        "warmup_max_tree_depth": WARMUP_DEPTH,
        "max_tree_depth": SAMPLE_DEPTH,
        "x64": False,
    }
    for name, expected_value in expected.items():
        if diagnostics.get(name) != expected_value:
            raise ValueError(
                f"Diagnostics drift for {name}: expected {expected_value!r}, "
                f"got {diagnostics.get(name)!r}."
            )


def _validate_sample_metadata(archive, *, run_label: str) -> None:
    """Validate scalar metadata stored with the long samples."""

    expected = {
        "run_label": run_label,
        "sigma_log_p_scale": 0.3,
        "direct_sigma_log_p": True,
        "fix_sigma_log_p": False,
        "sigma_log_p_parameterization": "direct_halfnormal",
        "fix_a": False,
        "fix_logg": True,
        "fixed_logg": 4.86,
        "fix_log_w": True,
        "fix_sigma_d": True,
        "fixed_ell_b": 0.4,
        "gp_jitter": 0.5e-6,
        "pressure_gp_factorization": "fixed_eigen",
        "full_data": True,
        "nside": 8,
        "zero_mean_pressure_map": True,
        "zero_mean_log_w": True,
        "zero_sum_log_w_basis": True,
        "gaussianized_atmosphere": True,
        "dense_mass": True,
        "dense_mass_mode": "full",
        "adapt_mass_matrix": True,
        "initial_inverse_mass_matrix_mode": "identity",
        "preflight_autodiff": False,
        "target_accept_prob": TARGET_ACCEPT,
        "warmup_max_tree_depth": WARMUP_DEPTH,
        "max_tree_depth": SAMPLE_DEPTH,
        "x64": False,
    }
    for name, expected_value in expected.items():
        actual_value = _scalar(archive, name)
        if actual_value != expected_value:
            raise ValueError(
                f"Samples metadata drift for {name}: "
                f"expected {expected_value!r}, got {actual_value!r}."
            )
    fixed_sigma = float(_scalar(archive, "fixed_sigma_log_p"))
    if not math.isnan(fixed_sigma):
        raise ValueError("Free-sigma samples unexpectedly record a fixed value.")
    rotation = np.asarray(archive["atmosphere_rotation_matrix"])
    expected_rotation = np.asarray(m8_v1_run.ATMOSPHERE_ROTATION, dtype=rotation.dtype)
    if not np.array_equal(rotation, expected_rotation):
        raise ValueError("Atmosphere rotation metadata drifted.")


def _validated_run_summary(
    samples_path: Path,
    diagnostics_path: Path,
    *,
    run_label: str,
) -> dict:
    """Validate one long run and return deterministic diagnostics."""

    samples_path = Path(samples_path)
    diagnostics_path = Path(diagnostics_path)
    if not samples_path.is_file() or not diagnostics_path.is_file():
        raise FileNotFoundError("Missing long-run samples or diagnostics.")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    _validate_diagnostics(diagnostics, run_label=run_label)

    scalar_draws = (
        "P",
        "T0",
        "alpha",
        "cosi",
        "ell_b",
        "log_vmr_co",
        "log_vmr_h2o",
        "log_vmr_ch4",
        "log_vmr_hf",
        "log_p_cloud",
        "logg",
        "q1",
        "q2",
        "sigma_log_p",
        "sigma_b",
        "u1",
        "u2",
        "v",
        "extra_num_steps",
        "extra_diverging",
        "extra_accept_prob",
        "extra_potential_energy",
    )
    with np.load(samples_path, allow_pickle=False) as archive:
        required_shapes = {
            **{name: (NUM_SAMPLES,) for name in scalar_draws},
            "atmosphere_rotated": (NUM_SAMPLES, 7),
            "atmosphere_gaussianized": (NUM_SAMPLES, 7),
            "A": (NUM_SAMPLES, 4),
            "log_w": (NUM_SAMPLES, 4, 14),
            "sigma_d": (NUM_SAMPLES, 4),
        }
        for name, shape in required_shapes.items():
            if name not in archive or np.asarray(archive[name]).shape != shape:
                raise ValueError(f"Missing or malformed samples array: {name}.")
            if not np.all(np.isfinite(np.asarray(archive[name], dtype=float))):
                raise ValueError(f"Nonfinite samples array: {name}.")
        _validate_sample_metadata(archive, run_label=run_label)
        sigma = np.asarray(archive["sigma_log_p"], dtype=np.float64)
        sigma_alias = np.asarray(archive["sigma_b"], dtype=np.float64)
        raw_steps = np.asarray(archive["extra_num_steps"])
        if not np.issubdtype(raw_steps.dtype, np.integer):
            raise ValueError("NUTS step counts must have an integer dtype.")
        steps = raw_steps.astype(np.int64, copy=False)
        raw_divergences = np.asarray(archive["extra_diverging"])
        if not np.issubdtype(raw_divergences.dtype, np.bool_):
            raise ValueError("NUTS divergence flags must have a boolean dtype.")
        divergences = raw_divergences.astype(bool, copy=False)
        accept = np.asarray(archive["extra_accept_prob"], dtype=np.float64)
        if not np.array_equal(sigma, sigma_alias):
            raise ValueError("sigma_b is inconsistent with sigma_log_p.")
        if not np.all(sigma > 0.0):
            raise ValueError("Free sigma_log_p contains a nonpositive draw.")
        if not np.all((steps >= 1) & (steps <= CAP_NUM_STEPS)):
            raise ValueError("NUTS step counts are outside the configured range.")
        if not np.all((accept >= 0.0) & (accept <= 1.0)):
            raise ValueError("NUTS acceptance probabilities are invalid.")
        archive_cap_num_steps = int(_scalar(archive, "tree_depth_cap_num_steps"))
        archive_cap_count = int(_scalar(archive, "tree_depth_cap_count"))
        archive_cap_fraction = float(_scalar(archive, "tree_depth_cap_fraction"))
        archive_final_step_size = float(_scalar(archive, "final_step_size"))

    cap_count = int(np.sum(steps == CAP_NUM_STEPS))
    cap_fraction = cap_count / NUM_SAMPLES
    divergence_count = int(np.sum(divergences))
    if diagnostics.get("tree_depth_cap_num_steps") != CAP_NUM_STEPS:
        raise ValueError("Diagnostics tree-depth cap value drifted.")
    if diagnostics.get("tree_depth_cap_count") != cap_count:
        raise ValueError("Diagnostics tree-depth cap count is inconsistent.")
    if not math.isclose(
        float(diagnostics.get("tree_depth_cap_fraction")),
        cap_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Diagnostics tree-depth cap fraction is inconsistent.")
    if diagnostics.get("divergence_count") != divergence_count:
        raise ValueError("Diagnostics divergence count is inconsistent.")
    if not (
        archive_cap_num_steps == CAP_NUM_STEPS
        and archive_cap_count == cap_count
        and math.isclose(archive_cap_fraction, cap_fraction, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("Samples tree-depth diagnostics are inconsistent.")
    if diagnostics.get("max_num_steps") != int(np.max(steps)):
        raise ValueError("Diagnostics maximum step count is inconsistent.")
    if not math.isclose(
        float(diagnostics.get("mean_accept_prob")),
        float(np.mean(accept)),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("Diagnostics mean acceptance is inconsistent.")
    final_step_size = float(diagnostics.get("final_step_size"))
    runtime_seconds = float(diagnostics.get("run_seconds"))
    if not (math.isfinite(final_step_size) and final_step_size > 0.0):
        raise ValueError("Diagnostics final step size is invalid.")
    if not math.isclose(
        archive_final_step_size,
        final_step_size,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("Samples final step size is inconsistent.")
    if not (math.isfinite(runtime_seconds) and runtime_seconds > 0.0):
        raise ValueError("Diagnostics runtime is invalid.")

    median_steps = float(np.median(steps))
    return {
        "divergence_count": divergence_count,
        "tree_depth_cap_count": cap_count,
        "tree_depth_cap_fraction": cap_fraction,
        "num_steps": {
            "minimum": int(np.min(steps)),
            "q25": float(np.quantile(steps, 0.25)),
            "median": median_steps,
            "q75": float(np.quantile(steps, 0.75)),
            "maximum": int(np.max(steps)),
            "mean": float(np.mean(steps)),
            "sum": int(np.sum(steps)),
        },
        "accept_prob": {
            "minimum": float(np.min(accept)),
            "q25": float(np.quantile(accept, 0.25)),
            "median": float(np.median(accept)),
            "q75": float(np.quantile(accept, 0.75)),
            "maximum": float(np.max(accept)),
            "mean": float(np.mean(accept)),
        },
        "final_step_size": final_step_size,
        "runtime_seconds": runtime_seconds,
        "sigma_log_p": {
            "minimum": float(np.min(sigma)),
            "q25": float(np.quantile(sigma, 0.25)),
            "median": float(np.median(sigma)),
            "q75": float(np.quantile(sigma, 0.75)),
            "maximum": float(np.max(sigma)),
        },
        "passes_prespecified_screening_thresholds": bool(
            divergence_count == 0 and cap_fraction < 0.05 and median_steps < 512
        ),
    }


def _expected_run_summary(args) -> dict:
    """Validate sampler artifacts and return the deterministic v10 summary."""

    _require_highest_precision()
    expected_contract = _expected_contract_payload(args)
    saved_contract = json.loads(Path(args.contract_output).read_text(encoding="utf-8"))
    if saved_contract != expected_contract:
        raise ValueError("The saved v10 contract no longer matches its inputs.")
    long_run = _validated_run_summary(
        args.samples, args.diagnostics, run_label=RUN_LABEL
    )
    baseline = _validated_run_summary(
        args.v1_samples,
        args.v1_diagnostics,
        run_label=("m8_v1_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_prod"),
    )
    return {
        "schema_version": 1,
        "mode": MODE,
        "execution_completed": True,
        "artifact_integrity_passed": True,
        "precision": EXPECTED_PRECISION,
        "sampling_contract": expected_contract["sampling_contract"],
        "v9_guard_evidence": expected_contract["v9_guard_evidence"],
        "findings": long_run,
        "v1_long_default": {
            **baseline,
            "samples_sha256": EXPECTED_STATIC_SHA256["v1_samples"],
            "diagnostics_sha256": EXPECTED_STATIC_SHA256["v1_diagnostics"],
            "matmul_precision_provenance": (
                "historical launcher inherited the unset/default policy"
            ),
        },
        "comparison": {
            "retained_step_sum_reduction_factor": (
                baseline["num_steps"]["sum"] / long_run["num_steps"]["sum"]
            ),
            "median_num_steps_reduction_factor": (
                baseline["num_steps"]["median"] / long_run["num_steps"]["median"]
            ),
            "runtime_speedup_factor": (
                baseline["runtime_seconds"] / long_run["runtime_seconds"]
            ),
            "step_size_increase_factor": (
                long_run["final_step_size"] / baseline["final_step_size"]
            ),
        },
        "artifact_sha256": {
            "contract": sha256_file(args.contract_output),
            "samples": sha256_file(args.samples),
            "diagnostics": sha256_file(args.diagnostics),
        },
        "interpretation": (
            "Sampler efficiency is a scientific finding, not a completion "
            "gate. This run changes process-global float32 matmul precision "
            "only relative to the frozen M8 v1 configuration."
        ),
        "limitations": [
            "One chain and one seed cannot establish convergence.",
            "The v9 score guard is a one-point check reused by SHA256.",
            "The historical M8 v1 artifact did not serialize matmul precision.",
            "This run does not validate the linearized forward model.",
        ],
    }


def summarize_run(args) -> dict:
    """Write a strict summary after validating long sampler artifacts."""

    payload = _expected_run_summary(args)
    _write_strict_json(args.summary_output, payload)
    return payload


def validate_saved_artifacts(args) -> dict:
    """Recompute and validate the saved summary without writing files."""

    expected = _expected_run_summary(args)
    saved = json.loads(Path(args.summary_output).read_text(encoding="utf-8"))
    passed = saved == expected
    return {
        "execution_completed": True,
        "artifact_integrity_passed": passed,
        "mode": saved.get("mode"),
    }


def parse_args():
    """Parse contract and artifact-validation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("contract", "summarize", "validate-artifacts"),
        default="contract",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--init-from", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--v1-samples", type=Path, default=DEFAULT_V1_SAMPLES)
    parser.add_argument("--v1-diagnostics", type=Path, default=DEFAULT_V1_DIAGNOSTICS)
    parser.add_argument("--v9-dir", type=Path, default=DEFAULT_V9_DIR)
    parser.add_argument("--contract-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    for name in (
        "out_dir",
        "init_from",
        "v1_samples",
        "v1_diagnostics",
        "v9_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.contract_output = (
        args.out_dir / CONTRACT_NAME
        if args.contract_output is None
        else args.contract_output.resolve()
    )
    args.summary_output = (
        args.out_dir / SUMMARY_NAME
        if args.summary_output is None
        else args.summary_output.resolve()
    )
    args.samples = (
        args.out_dir / "samples.npz" if args.samples is None else args.samples.resolve()
    )
    args.diagnostics = (
        args.out_dir / "diagnostics.json"
        if args.diagnostics is None
        else args.diagnostics.resolve()
    )
    return args


def main() -> None:
    """Run validation, write the contract, or validate completed artifacts."""

    args = parse_args()
    if args.validate_only:
        _require_highest_precision()
        _validate_inputs(args)
        for path in (args.contract_output, args.summary_output):
            if path.exists() or path.with_name(f"{path.name}.tmp").exists():
                raise FileExistsError(f"Refusing output collision: {path}")
        print(
            json.dumps(
                {
                    "validated": True,
                    "mode": MODE,
                    "planned_output": str(args.out_dir),
                    "precision": EXPECTED_PRECISION,
                },
                sort_keys=True,
            )
        )
        return
    if args.action == "contract":
        result = write_contract(args)
    elif args.action == "summarize":
        result = summarize_run(args)
    else:
        result = validate_saved_artifacts(args)
        if result["artifact_integrity_passed"] is not True:
            raise SystemExit(5)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
