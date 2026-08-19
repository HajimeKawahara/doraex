"""Tests for the M8 v10 long highest-precision free-sigma control."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doraex.workflows import on_the_fly_pressure_retrieval as retrieval  # noqa: E402
from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v10_free_sigma_highest_precision_long_control as control,
)
from examples.luhman16b_yama import m8_v1_run  # noqa: E402


LAUNCHER = ROOT / "csh/exe_m8_v10_free_sigma_highest_precision_long.csh"


@pytest.fixture(autouse=True)
def _highest_precision_environment(monkeypatch):
    """Match the dedicated launcher environment in helper-level tests."""

    previous = control.jax.config.jax_default_matmul_precision
    monkeypatch.setenv("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    control.jax.config.update("jax_default_matmul_precision", "highest")
    yield
    control.jax.config.update("jax_default_matmul_precision", previous)


def _capture_m8_v1_args():
    """Capture the unmodified long-baseline wrapper arguments."""

    captured = {}
    original_main = retrieval.main
    original_argv = sys.argv

    def capture_args():
        captured["args"] = retrieval.parse_args()

    try:
        retrieval.main = capture_args
        sys.argv = [str(Path(m8_v1_run.__file__))]
        m8_v1_run.main()
    finally:
        retrieval.main = original_main
        sys.argv = original_argv
    return captured["args"]


def _validation_args(tmp_path: Path) -> SimpleNamespace:
    """Return v10 paths rooted in an isolated temporary directory."""

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return SimpleNamespace(
        init_from=control.DEFAULT_INIT.resolve(),
        out_dir=out_dir.resolve(),
        v1_samples=control.DEFAULT_V1_SAMPLES.resolve(),
        v1_diagnostics=control.DEFAULT_V1_DIAGNOSTICS.resolve(),
        v9_dir=control.DEFAULT_V9_DIR.resolve(),
        contract_output=(out_dir / control.CONTRACT_NAME).resolve(),
        summary_output=(out_dir / control.SUMMARY_NAME).resolve(),
        samples=(out_dir / "samples.npz").resolve(),
        diagnostics=(out_dir / "diagnostics.json").resolve(),
    )


def _write_synthetic_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write finite, structurally valid long-run sample arrays."""

    steps = np.full(control.NUM_SAMPLES, 31, dtype=np.int64)
    steps[-2:] = control.CAP_NUM_STEPS
    accept = np.linspace(0.90, 0.99, control.NUM_SAMPLES)
    np.savez(
        path,
        atmosphere_rotated=np.zeros((control.NUM_SAMPLES, 7)),
        atmosphere_gaussianized=np.zeros((control.NUM_SAMPLES, 7)),
        A=np.full((control.NUM_SAMPLES, 4), 1.1),
        P=np.full(control.NUM_SAMPLES, 4.87),
        T0=np.full(control.NUM_SAMPLES, 1200.0),
        alpha=np.full(control.NUM_SAMPLES, 0.12),
        cosi=np.full(control.NUM_SAMPLES, 0.5),
        ell_b=np.full(control.NUM_SAMPLES, 0.4),
        log_vmr_co=np.full(control.NUM_SAMPLES, -3.0),
        log_vmr_h2o=np.full(control.NUM_SAMPLES, -3.3),
        log_vmr_ch4=np.full(control.NUM_SAMPLES, -4.7),
        log_vmr_hf=np.full(control.NUM_SAMPLES, -7.3),
        log_p_cloud=np.full(control.NUM_SAMPLES, 1.3),
        logg=np.full(control.NUM_SAMPLES, 4.86),
        log_w=np.zeros((control.NUM_SAMPLES, 4, 14)),
        q1=np.full(control.NUM_SAMPLES, 0.5),
        q2=np.full(control.NUM_SAMPLES, 0.5),
        sigma_d=np.full((control.NUM_SAMPLES, 4), 0.03),
        sigma_log_p=np.linspace(0.25, 0.55, control.NUM_SAMPLES),
        sigma_b=np.linspace(0.25, 0.55, control.NUM_SAMPLES),
        u1=np.full(control.NUM_SAMPLES, 0.3),
        u2=np.full(control.NUM_SAMPLES, 0.2),
        v=np.full(control.NUM_SAMPLES, 20.0),
        extra_num_steps=steps,
        extra_diverging=np.zeros(control.NUM_SAMPLES, dtype=bool),
        extra_accept_prob=accept,
        extra_potential_energy=np.linspace(-88312.0, -88300.0, control.NUM_SAMPLES),
        run_label=np.asarray(control.RUN_LABEL),
        sigma_log_p_scale=np.asarray(0.3),
        sigma_log_p_parameterization=np.asarray("direct_halfnormal"),
        direct_sigma_log_p=np.asarray(True),
        fix_sigma_log_p=np.asarray(False),
        fixed_sigma_log_p=np.asarray(np.nan),
        fix_a=np.asarray(False),
        fix_logg=np.asarray(True),
        fixed_logg=np.asarray(4.86),
        fix_log_w=np.asarray(True),
        fix_sigma_d=np.asarray(True),
        fixed_ell_b=np.asarray(0.4),
        gp_jitter=np.asarray(0.5e-6),
        pressure_gp_factorization=np.asarray("fixed_eigen"),
        full_data=np.asarray(True),
        nside=np.asarray(8),
        zero_mean_pressure_map=np.asarray(True),
        zero_mean_log_w=np.asarray(True),
        zero_sum_log_w_basis=np.asarray(True),
        gaussianized_atmosphere=np.asarray(True),
        atmosphere_rotation_matrix=np.asarray(
            m8_v1_run.ATMOSPHERE_ROTATION, dtype=np.float32
        ),
        dense_mass=np.asarray(True),
        dense_mass_mode=np.asarray("full"),
        adapt_mass_matrix=np.asarray(True),
        initial_inverse_mass_matrix_mode=np.asarray("identity"),
        preflight_autodiff=np.asarray(False),
        target_accept_prob=np.asarray(control.TARGET_ACCEPT),
        warmup_max_tree_depth=np.asarray(control.WARMUP_DEPTH),
        max_tree_depth=np.asarray(control.SAMPLE_DEPTH),
        tree_depth_cap_num_steps=np.asarray(control.CAP_NUM_STEPS),
        tree_depth_cap_count=np.asarray(2),
        tree_depth_cap_fraction=np.asarray(2 / control.NUM_SAMPLES),
        final_step_size=np.asarray(0.08),
        x64=np.asarray(False),
    )
    return steps, accept


def _write_synthetic_completed_run(args: SimpleNamespace) -> None:
    """Write a complete synthetic run matching the v10 artifact schema."""

    control.write_contract(args)
    steps, accept = _write_synthetic_samples(args.samples)
    cap_count = int(np.sum(steps == control.CAP_NUM_STEPS))
    diagnostics = {
        "mode": control.RUN_LABEL,
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
        "num_warmup": control.NUM_WARMUP,
        "num_samples": control.NUM_SAMPLES,
        "num_chains": control.NUM_CHAINS,
        "target_accept_prob": control.TARGET_ACCEPT,
        "warmup_max_tree_depth": control.WARMUP_DEPTH,
        "max_tree_depth": control.SAMPLE_DEPTH,
        "x64": False,
        "tree_depth_cap_num_steps": control.CAP_NUM_STEPS,
        "tree_depth_cap_count": cap_count,
        "tree_depth_cap_fraction": cap_count / control.NUM_SAMPLES,
        "divergence_count": 0,
        "max_num_steps": int(np.max(steps)),
        "mean_accept_prob": float(np.mean(accept)),
        "final_step_size": 0.08,
        "run_seconds": 12345.0,
    }
    args.diagnostics.write_text(
        json.dumps(diagnostics, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rewrite_npz(path: Path, name: str, value: np.ndarray) -> None:
    """Replace one array in a test NPZ without changing other fields."""

    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays[name] = np.asarray(value)
    np.savez(path, **arrays)


def test_explicit_long_configuration_matches_m8_v1_baseline(tmp_path):
    candidate = control._capture_sampling_args(control.DEFAULT_INIT, tmp_path)
    baseline = _capture_m8_v1_args()
    candidate_values = vars(candidate).copy()
    baseline_values = vars(baseline).copy()
    for name in ("out_dir", "run_label"):
        candidate_values.pop(name)
        baseline_values.pop(name)
    assert candidate_values == baseline_values

    contract = control._sampling_contract(candidate)
    assert contract["matmul_precision"] == "highest"
    assert contract["num_warmup"] == 2000
    assert contract["num_samples"] == 1500
    assert contract["num_chains"] == 1
    assert contract["seed"] == 0
    assert contract["direct_sigma_log_p"] is True
    assert contract["fixed_sigma_log_p"] is None
    assert contract["fix_a"] is False
    assert contract["fix_log_w"] is True
    assert contract["fix_sigma_d"] is True
    assert contract["dense_mass"] is True
    assert contract["adapt_mass_matrix"] is True
    assert contract["x64"] is False


def test_v9_guard_is_pinned_as_the_long_run_precondition():
    evidence = control._validate_v9_evidence(control.DEFAULT_V9_DIR)
    encoded = json.dumps(evidence, sort_keys=True)
    guard = control.DEFAULT_V9_DIR / "initial_sigma_score_guard.json"
    summary = (
        control.DEFAULT_V9_DIR
        / "m8_v9_free_sigma_highest_precision_control_summary.json"
    )
    assert control.sha256_file(guard) in encoded
    assert control.sha256_file(summary) in encoded
    assert "highest" in encoded
    assert (control.DEFAULT_V9_DIR / "COMPLETE").is_file()


def test_contract_and_summary_validator_recompute_and_reject_tampering(tmp_path):
    args = _validation_args(tmp_path)
    _write_synthetic_completed_run(args)
    contract = json.loads(args.contract_output.read_text(encoding="utf-8"))
    assert contract["execution_completed"] is True
    assert contract["precision"] == "highest"

    summary = control.summarize_run(args)
    assert summary["artifact_integrity_passed"] is True
    assert summary["sampling_contract"]["num_warmup"] == 2000
    assert summary["sampling_contract"]["num_samples"] == 1500
    assert control.validate_saved_artifacts(args)["artifact_integrity_passed"]

    saved = json.loads(args.summary_output.read_text(encoding="utf-8"))
    saved["findings"]["tree_depth_cap_count"] += 1
    args.summary_output.write_text(
        json.dumps(saved, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    assert not control.validate_saved_artifacts(args)["artifact_integrity_passed"]


def test_summary_rejects_a_tampered_v9_reuse_contract(tmp_path):
    args = _validation_args(tmp_path)
    _write_synthetic_completed_run(args)
    contract = json.loads(args.contract_output.read_text(encoding="utf-8"))
    contract["precision"] = "default"
    args.contract_output.write_text(
        json.dumps(contract, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contract|Contract"):
        control.summarize_run(args)


@pytest.mark.parametrize(
    "failure",
    ("sigma_shape", "sigma_nonfinite", "cosi_nonfinite", "log_w_shape"),
)
def test_summary_rejects_malformed_or_nonfinite_samples(tmp_path, failure):
    args = _validation_args(tmp_path)
    _write_synthetic_completed_run(args)
    if failure == "sigma_shape":
        name = "sigma_log_p"
        replacement = np.ones(control.NUM_SAMPLES - 1)
    elif failure == "sigma_nonfinite":
        name = "sigma_log_p"
        replacement = np.ones(control.NUM_SAMPLES)
        replacement[0] = np.nan
    elif failure == "cosi_nonfinite":
        name = "cosi"
        replacement = np.ones(control.NUM_SAMPLES)
        replacement[0] = np.nan
    else:
        name = "log_w"
        replacement = np.ones((control.NUM_SAMPLES, 4, 13))
    _rewrite_npz(args.samples, name, replacement)

    with pytest.raises(ValueError, match="malformed|Nonfinite"):
        control.summarize_run(args)


def test_launcher_is_native_tcsh_and_runs_only_the_long_highest_control():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env -S tcsh -f")
    assert "/bin/tcsh" not in text
    assert "setenv JAX_DEFAULT_MATMUL_PRECISION highest" in text
    assert "--num-warmup 2000" in text
    assert "--num-samples 1500" in text
    assert "--num-chains 1" in text
    assert "--seed 0" in text
    assert "--dense-mass" in text
    assert "--adapt-mass-matrix" in text
    assert "--no-x64" in text
    assert "--nohup" in text
    assert "/tmp/doraex_m7_gpu.lock" in text
    assert "Barker" not in text
    assert "NeuTra" not in text
    assert "empirical" not in text


def test_launcher_hash_pins_match_helper_and_test_sources():
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in (Path(control.__file__), Path(__file__)):
        assert f'"{control.sha256_file(path)}"' in text
