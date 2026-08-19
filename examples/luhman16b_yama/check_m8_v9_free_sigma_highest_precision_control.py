"""Guard and validate the M8 v9 highest-precision free-sigma pilot."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model


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
from examples.luhman16b_yama import m8_v1_run  # noqa: E402
from examples.luhman16b_yama import m8_v2_run  # noqa: E402


MODE = "m8_v9_free_sigma_highest_precision_control"
RUN_LABEL = "m8_v9_free_sigma_highest_precision_v17_schedule"
DEFAULT_OUT_DIR = ROOT / "results/m8/v9/free_sigma_highest_precision_seed0"
DEFAULT_INIT = ROOT / "results/m7/p2/v6/samples.npz"
DEFAULT_V17_SAMPLES = ROOT / "results/m7/p2/v17/samples.npz"
DEFAULT_V17_DIAGNOSTICS = ROOT / "results/m7/p2/v17/diagnostics.json"
DEFAULT_V8_DIR = ROOT / "results/m8/v8/free_sigma_lowrank_precision"
GUARD_NAME = "initial_sigma_score_guard.json"
SUMMARY_NAME = "m8_v9_free_sigma_highest_precision_control_summary.json"
EXPECTED_PRECISION = "highest"
EXPECTED_INITIAL_SIGMA = 0.32316479086875916
POTENTIAL_ATOL = 5.0e-2
SCORE_ATOL = 2.0e-1
NUM_WARMUP = 200
NUM_SAMPLES = 20
NUM_CHAINS = 1
SEED = 0
TARGET_ACCEPT = 0.95
WARMUP_DEPTH = 9
SAMPLE_DEPTH = 11
CAP_NUM_STEPS = 2**SAMPLE_DEPTH - 1


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Return the explicit v17-length sampling arguments."""

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


def _capture_sampling_args(init_from: Path, out_dir: Path):
    """Capture the exact M8 v1 wrapper configuration without sampling."""

    captured = {}
    original_main = retrieval.main
    original_argv = sys.argv

    def capture_args() -> None:
        captured["args"] = retrieval.parse_args()

    try:
        retrieval.main = capture_args
        sys.argv = [str(Path(m8_v1_run.__file__))] + _sampling_cli(init_from, out_dir)
        m8_v1_run.main()
    finally:
        retrieval.main = original_main
        sys.argv = original_argv
    return captured["args"]


def _sampling_contract(args) -> dict:
    """Validate and serialize the frozen short free-sigma contract."""

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
    return {
        **actual,
        "init_from": str(Path(args.init_from).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "matmul_precision": EXPECTED_PRECISION,
        "precision_scope": "dedicated Python process",
        "maximum_tree_num_steps": CAP_NUM_STEPS,
        "sampling_argv": _sampling_cli(args.init_from, args.out_dir),
    }


def _validate_v8_evidence(v8_dir: Path) -> dict:
    """Validate the completed v8 result that motivates this intervention."""

    v8_dir = Path(v8_dir)
    summary_path = v8_dir / "m8_v8_free_sigma_lowrank_precision_summary.json"
    required = [
        v8_dir / "COMPLETE",
        summary_path,
        v8_dir / "m8_v8_free_sigma_lowrank_precision_arrays.npz",
        v8_dir / "m8_v8_free_sigma_lowrank_precision_checkpoint.json",
        v8_dir / "outputs.sha256",
        v8_dir / "provenance.sha256",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing v8 evidence: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    findings = summary.get("findings", {})
    by_arm = findings.get("reverse_forward_total_max_abs_by_arm", {})
    if not (
        summary.get("execution_completed") is True
        and summary.get("method_contract_passed") is True
        and summary.get("artifact_integrity_passed") is True
        and summary.get("classification") == "full_highest_precision_recovers_score"
        and float(by_arm.get("full_f32_default", math.inf)) > SCORE_ATOL
        and float(by_arm.get("full_f32_highest", math.inf)) <= SCORE_ATOL
    ):
        raise ValueError("The pinned v8 precision evidence no longer matches.")
    return {
        "classification": summary["classification"],
        "full_default_maximum_score_difference": float(by_arm["full_f32_default"]),
        "full_highest_maximum_score_difference": float(by_arm["full_f32_highest"]),
        "files": {str(path.resolve()): sha256_file(path) for path in required},
    }


def _validate_inputs(args) -> tuple[object, dict, dict]:
    """Validate static inputs without compiling the target."""

    for path in (
        args.init_from,
        args.v17_samples,
        args.v17_diagnostics,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing control input: {path}")
    sampling_args = _capture_sampling_args(args.init_from, args.out_dir)
    contract = _sampling_contract(sampling_args)
    evidence = _validate_v8_evidence(args.v8_dir)
    return sampling_args, contract, evidence


def run_initial_score_guard(args) -> dict:
    """Compare the initial reverse and pure-JVP sigma scores at highest."""

    sampling_args, contract, evidence = _validate_inputs(args)
    precision = jax.config.jax_default_matmul_precision
    if precision != EXPECTED_PRECISION:
        raise RuntimeError(
            "JAX_DEFAULT_MATMUL_PRECISION must be highest before JAX import; "
            f"got {precision!r}."
        )

    build_args = copy.copy(sampling_args)
    build_args.direct_sigma_log_p = False
    build_args.fixed_sigma_log_p = m8_v2_run.V17_INITIAL_SIGMA_LOG_P
    target_start = time.perf_counter()
    (
        fixed_model,
        free_model,
        fixed_initial_values,
        free_initial_values,
        physical_summary,
    ) = replay._build_models(build_args)
    del fixed_model, fixed_initial_values
    physical_sigma = float(physical_summary["sigma_log_p"])
    if physical_sigma != EXPECTED_INITIAL_SIGMA:
        raise ValueError(
            "Initial sigma_log_p drifted: "
            f"expected {EXPECTED_INITIAL_SIGMA!r}, got {physical_sigma!r}."
        )

    model_info = initialize_model(
        jax.random.PRNGKey(SEED),
        free_model,
        init_strategy=init_to_value(values=free_initial_values),
        dynamic_args=False,
        forward_mode_differentiation=False,
        validate_grad=True,
    )
    unconstrained = model_info.param_info.z
    reverse_value = model_info.param_info.potential_energy
    reverse_score = model_info.param_info.z_grad["sigma_log_p"]
    potential_fn = model_info.potential_fn
    site_names = tuple(unconstrained)
    if set(site_names) != {"A", "atmosphere_rotated", "sigma_log_p"}:
        raise ValueError(f"Unexpected unconstrained sites: {site_names}.")
    tangent = jax.tree.map(jnp.zeros_like, unconstrained)
    tangent = dict(tangent)
    tangent["sigma_log_p"] = jnp.ones_like(unconstrained["sigma_log_p"])
    del model_info, free_model, free_initial_values
    gc.collect()

    @jax.jit
    def forward_score(position, direction):
        return jax.jvp(potential_fn, (position,), (direction,))

    compile_start = time.perf_counter()
    forward_value, forward_derivative = forward_score(unconstrained, tangent)
    forward_value.block_until_ready()
    forward_derivative.block_until_ready()
    elapsed = time.perf_counter() - target_start
    compile_seconds = time.perf_counter() - compile_start

    reverse_value_host = float(jax.device_get(reverse_value))
    forward_value_host = float(jax.device_get(forward_value))
    reverse_score_host = float(jax.device_get(reverse_score))
    forward_score_host = float(jax.device_get(forward_derivative))
    initial_u = float(jax.device_get(unconstrained["sigma_log_p"]))
    values = np.asarray(
        [
            reverse_value_host,
            forward_value_host,
            reverse_score_host,
            forward_score_host,
            initial_u,
        ],
        dtype=np.float64,
    )
    all_finite = bool(np.all(np.isfinite(values)))
    potential_difference = abs(reverse_value_host - forward_value_host)
    score_difference = abs(reverse_score_host - forward_score_host)
    transform_matches = bool(
        np.isclose(
            np.exp(initial_u),
            physical_sigma,
            rtol=2e-7,
            atol=0.0,
        )
    )
    passed = bool(
        all_finite
        and transform_matches
        and potential_difference <= POTENTIAL_ATOL
        and score_difference <= SCORE_ATOL
        and jax.config.jax_default_matmul_precision == EXPECTED_PRECISION
    )
    payload = {
        "schema_version": 1,
        "mode": "m8_v9_initial_sigma_score_guard",
        "execution_completed": True,
        "passed": passed,
        "precision": EXPECTED_PRECISION,
        "precision_scope": (
            "process-global float32 matmul precision; not limited to LowRank"
        ),
        "methods": {
            "reverse": "NumPyro initialize_model param_info.z_grad",
            "forward": "jax.jvp(full potential, sigma-only PyTree tangent)",
            "hessian_computed": False,
            "fallback_used": False,
        },
        "initial_point": {
            "physical_sigma_log_p": physical_sigma,
            "unconstrained_log_sigma_log_p": initial_u,
            "transform_matches": transform_matches,
            "site_names": list(site_names),
            "site_shapes": {
                name: list(np.shape(jax.device_get(unconstrained[name])))
                for name in site_names
            },
        },
        "results": {
            "reverse_potential": reverse_value_host,
            "forward_primal": forward_value_host,
            "potential_absolute_difference": potential_difference,
            "reverse_sigma_score": reverse_score_host,
            "forward_sigma_score": forward_score_host,
            "score_signed_reverse_minus_forward": (
                reverse_score_host - forward_score_host
            ),
            "score_absolute_difference": score_difference,
            "all_finite": all_finite,
        },
        "thresholds": {
            "potential_atol": POTENTIAL_ATOL,
            "score_atol": SCORE_ATOL,
        },
        "sampling_contract": contract,
        "v8_evidence": evidence,
        "timing_seconds": {
            "total_target_and_guard": elapsed,
            "forward_jvp_compile_and_call": compile_seconds,
        },
        "limitations": [
            "This is a one-point sigma-direction score guard.",
            "Passing does not establish good posterior-wide NUTS geometry.",
            "This control does not validate the linearized forward model.",
        ],
    }
    _write_strict_json(args.guard_output, payload)
    if not passed:
        raise RuntimeError("Initial highest-precision sigma score guard failed.")
    return payload


def _scalar(npz, name):
    """Return one scalar NPZ value."""

    value = np.asarray(npz[name])
    if value.shape != ():
        raise ValueError(f"Expected scalar {name}, got {value.shape}.")
    return value.item()


def _expected_run_summary(args) -> dict:
    """Validate completed sampler outputs and build their summary."""

    _, contract, evidence = _validate_inputs(args)
    guard = json.loads(Path(args.guard_output).read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
    if not (
        guard.get("execution_completed") is True
        and guard.get("passed") is True
        and guard.get("precision") == EXPECTED_PRECISION
        and guard.get("sampling_contract") == contract
    ):
        raise ValueError("Initial score guard is not valid for this run.")

    expected_diagnostics = {
        "mode": RUN_LABEL,
        "chip_indices": [0, 1, 2, 3],
        "full_data": True,
        "nside": 8,
        "fixed_ell_b": 0.4,
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
        "gaussianized_atmosphere": True,
        "dense_mass": True,
        "dense_mass_mode": "full",
        "adapt_mass_matrix": True,
        "num_warmup": NUM_WARMUP,
        "num_samples": NUM_SAMPLES,
        "num_chains": NUM_CHAINS,
        "target_accept_prob": TARGET_ACCEPT,
        "warmup_max_tree_depth": WARMUP_DEPTH,
        "max_tree_depth": SAMPLE_DEPTH,
        "x64": False,
    }
    for name, expected_value in expected_diagnostics.items():
        if diagnostics.get(name) != expected_value:
            raise ValueError(
                f"Diagnostics drift for {name}: expected {expected_value!r}, "
                f"got {diagnostics.get(name)!r}."
            )

    with np.load(args.samples, allow_pickle=False) as archive:
        required_shapes = {
            "atmosphere_rotated": (NUM_SAMPLES, 7),
            "A": (NUM_SAMPLES, 4),
            "sigma_log_p": (NUM_SAMPLES,),
            "extra_num_steps": (NUM_SAMPLES,),
            "extra_diverging": (NUM_SAMPLES,),
            "extra_accept_prob": (NUM_SAMPLES,),
            "extra_potential_energy": (NUM_SAMPLES,),
        }
        for name, shape in required_shapes.items():
            if name not in archive or np.asarray(archive[name]).shape != shape:
                raise ValueError(f"Missing or malformed samples array: {name}.")
            if not np.all(np.isfinite(np.asarray(archive[name], dtype=float))):
                raise ValueError(f"Nonfinite samples array: {name}.")

        sample_metadata = {
            "run_label": RUN_LABEL,
            "sigma_log_p_scale": 0.3,
            "direct_sigma_log_p": True,
            "fix_sigma_log_p": False,
            "fix_a": False,
            "fix_logg": True,
            "fixed_logg": 4.86,
            "fix_log_w": True,
            "fix_sigma_d": True,
            "pressure_gp_factorization": "fixed_eigen",
            "full_data": True,
            "nside": 8,
            "zero_mean_pressure_map": True,
            "zero_mean_log_w": True,
            "zero_sum_log_w_basis": True,
            "gaussianized_atmosphere": True,
            "dense_mass": True,
            "adapt_mass_matrix": True,
            "target_accept_prob": TARGET_ACCEPT,
            "warmup_max_tree_depth": WARMUP_DEPTH,
            "max_tree_depth": SAMPLE_DEPTH,
            "x64": False,
        }
        for name, expected_value in sample_metadata.items():
            actual_value = _scalar(archive, name)
            if actual_value != expected_value:
                raise ValueError(
                    f"Samples metadata drift for {name}: "
                    f"expected {expected_value!r}, got {actual_value!r}."
                )
        sigma = np.asarray(archive["sigma_log_p"], dtype=np.float64)
        steps = np.asarray(archive["extra_num_steps"], dtype=np.int64)
        divergences = np.asarray(archive["extra_diverging"], dtype=bool)
        accept = np.asarray(archive["extra_accept_prob"], dtype=np.float64)
        if not np.all(sigma > 0.0):
            raise ValueError("Free sigma_log_p contains a nonpositive draw.")
        if not np.all((steps >= 1) & (steps <= CAP_NUM_STEPS)):
            raise ValueError("NUTS step counts are outside the configured range.")

    cap_count = int(np.sum(steps == CAP_NUM_STEPS))
    divergence_count = int(np.sum(divergences))
    cap_fraction = cap_count / NUM_SAMPLES
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
    if not math.isclose(
        float(diagnostics.get("mean_accept_prob")),
        float(np.mean(accept)),
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError("Diagnostics mean acceptance is inconsistent.")
    if not (
        np.isfinite(float(diagnostics.get("final_step_size")))
        and float(diagnostics["final_step_size"]) > 0.0
    ):
        raise ValueError("Diagnostics final step size is invalid.")

    baseline_diagnostics = json.loads(
        Path(args.v17_diagnostics).read_text(encoding="utf-8")
    )
    with np.load(args.v17_samples, allow_pickle=False) as baseline:
        baseline_steps = np.asarray(baseline["extra_num_steps"], dtype=np.int64)
    return {
        "schema_version": 1,
        "mode": MODE,
        "execution_completed": True,
        "artifact_integrity_passed": True,
        "precision": EXPECTED_PRECISION,
        "sampling_contract": contract,
        "initial_score_guard": {
            "passed": True,
            "physical_sigma_log_p": guard["initial_point"]["physical_sigma_log_p"],
            "score_absolute_difference": guard["results"]["score_absolute_difference"],
            "score_atol": guard["thresholds"]["score_atol"],
        },
        "findings": {
            "divergence_count": divergence_count,
            "tree_depth_cap_count": cap_count,
            "tree_depth_cap_fraction": cap_fraction,
            "num_steps": {
                "minimum": int(np.min(steps)),
                "q25": float(np.quantile(steps, 0.25)),
                "median": float(np.median(steps)),
                "q75": float(np.quantile(steps, 0.75)),
                "maximum": int(np.max(steps)),
                "mean": float(np.mean(steps)),
                "sum": int(np.sum(steps)),
            },
            "mean_accept_prob": float(np.mean(accept)),
            "final_step_size": float(diagnostics["final_step_size"]),
            "sigma_log_p": {
                "minimum": float(np.min(sigma)),
                "median": float(np.median(sigma)),
                "maximum": float(np.max(sigma)),
            },
            "prespecified_efficiency_thresholds": {
                "divergence_count": 0,
                "tree_depth_cap_fraction_less_than": 0.05,
                "median_num_steps_less_than": 512,
            },
        },
        "v17_baseline": {
            "tree_depth_cap_count": int(baseline_diagnostics["tree_depth_cap_count"]),
            "tree_depth_cap_fraction": float(
                baseline_diagnostics["tree_depth_cap_fraction"]
            ),
            "median_num_steps": float(np.median(baseline_steps)),
            "sum_num_steps": int(np.sum(baseline_steps)),
            "samples_sha256": sha256_file(args.v17_samples),
            "diagnostics_sha256": sha256_file(args.v17_diagnostics),
        },
        "v8_evidence": evidence,
        "artifact_sha256": {
            "samples": sha256_file(args.samples),
            "diagnostics": sha256_file(args.diagnostics),
            "initial_score_guard": sha256_file(args.guard_output),
        },
        "interpretation": (
            "Sampler efficiency is a scientific finding, not a completion "
            "gate. This pilot changes process-global float32 matmul precision "
            "only; it does not change the frozen linear target."
        ),
        "limitations": [
            "Twenty retained draws are a screening run, not inference.",
            "One chain and one seed cannot establish convergence.",
            "This run does not validate the linearized forward model.",
        ],
    }


def summarize_run(args) -> dict:
    """Write a strict summary after validating sampler artifacts."""

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
    """Parse guard and artifact-validation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("guard", "summarize", "validate-artifacts"),
        default="guard",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--init-from", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--v17-samples", type=Path, default=DEFAULT_V17_SAMPLES)
    parser.add_argument("--v17-diagnostics", type=Path, default=DEFAULT_V17_DIAGNOSTICS)
    parser.add_argument("--v8-dir", type=Path, default=DEFAULT_V8_DIR)
    parser.add_argument("--guard-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.init_from = args.init_from.resolve()
    args.v17_samples = args.v17_samples.resolve()
    args.v17_diagnostics = args.v17_diagnostics.resolve()
    args.v8_dir = args.v8_dir.resolve()
    args.guard_output = (
        args.out_dir / GUARD_NAME
        if args.guard_output is None
        else args.guard_output.resolve()
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
    """Run static validation, the score guard, or artifact validation."""

    args = parse_args()
    if args.validate_only:
        _validate_inputs(args)
        for path in (args.guard_output, args.summary_output):
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
    if args.action == "guard":
        result = run_initial_score_guard(args)
    elif args.action == "summarize":
        result = summarize_run(args)
    else:
        result = validate_saved_artifacts(args)
        if result["artifact_integrity_passed"] is not True:
            raise SystemExit(5)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
