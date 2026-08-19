"""Tests for the M8 v9 highest-precision free-sigma control."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doraex.workflows import on_the_fly_pressure_retrieval as retrieval  # noqa: E402
from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v9_free_sigma_highest_precision_control as control,
)
from examples.luhman16b_yama import m7_p2_v17_run  # noqa: E402


LAUNCHER = ROOT / "csh/exe_m8_v9_free_sigma_highest_precision_v17_control.csh"


def _capture_v17_args():
    captured = {}
    original_main = retrieval.main
    original_argv = sys.argv

    def capture_args():
        captured["args"] = retrieval.parse_args()

    try:
        retrieval.main = capture_args
        sys.argv = [str(Path(m7_p2_v17_run.__file__))]
        m7_p2_v17_run.main()
    finally:
        retrieval.main = original_main
        sys.argv = original_argv
    return captured["args"]


def _validation_args(tmp_path: Path) -> SimpleNamespace:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return SimpleNamespace(
        init_from=control.DEFAULT_INIT.resolve(),
        out_dir=out_dir.resolve(),
        v17_samples=control.DEFAULT_V17_SAMPLES.resolve(),
        v17_diagnostics=control.DEFAULT_V17_DIAGNOSTICS.resolve(),
        v8_dir=control.DEFAULT_V8_DIR.resolve(),
        guard_output=(out_dir / control.GUARD_NAME).resolve(),
        summary_output=(out_dir / control.SUMMARY_NAME).resolve(),
        samples=(out_dir / "samples.npz").resolve(),
        diagnostics=(out_dir / "diagnostics.json").resolve(),
    )


def _write_synthetic_completed_run(args: SimpleNamespace) -> None:
    sampling_args = control._capture_sampling_args(args.init_from, args.out_dir)
    contract = control._sampling_contract(sampling_args)
    guard = {
        "execution_completed": True,
        "passed": True,
        "precision": "highest",
        "sampling_contract": contract,
        "initial_point": {"physical_sigma_log_p": control.EXPECTED_INITIAL_SIGMA},
        "results": {"score_absolute_difference": 0.01},
        "thresholds": {"score_atol": control.SCORE_ATOL},
    }
    args.guard_output.write_text(
        json.dumps(guard, allow_nan=False) + "\n", encoding="utf-8"
    )

    steps = np.asarray([511] * 18 + [2047] * 2, dtype=np.int64)
    accept = np.linspace(0.8, 0.99, control.NUM_SAMPLES)
    np.savez(
        args.samples,
        atmosphere_rotated=np.zeros((control.NUM_SAMPLES, 7)),
        A=np.full((control.NUM_SAMPLES, 4), 1.1),
        sigma_log_p=np.linspace(0.25, 0.35, control.NUM_SAMPLES),
        extra_num_steps=steps,
        extra_diverging=np.zeros(control.NUM_SAMPLES, dtype=bool),
        extra_accept_prob=accept,
        extra_potential_energy=np.linspace(-1.0, 1.0, control.NUM_SAMPLES),
        run_label=np.asarray(control.RUN_LABEL),
        sigma_log_p_scale=np.asarray(0.3),
        direct_sigma_log_p=np.asarray(True),
        fix_sigma_log_p=np.asarray(False),
        fix_a=np.asarray(False),
        fix_logg=np.asarray(True),
        fixed_logg=np.asarray(4.86),
        fix_log_w=np.asarray(True),
        fix_sigma_d=np.asarray(True),
        pressure_gp_factorization=np.asarray("fixed_eigen"),
        full_data=np.asarray(True),
        nside=np.asarray(8),
        zero_mean_pressure_map=np.asarray(True),
        zero_mean_log_w=np.asarray(True),
        zero_sum_log_w_basis=np.asarray(True),
        gaussianized_atmosphere=np.asarray(True),
        dense_mass=np.asarray(True),
        adapt_mass_matrix=np.asarray(True),
        target_accept_prob=np.asarray(control.TARGET_ACCEPT),
        warmup_max_tree_depth=np.asarray(control.WARMUP_DEPTH),
        max_tree_depth=np.asarray(control.SAMPLE_DEPTH),
        x64=np.asarray(False),
    )
    diagnostics = {
        "mode": control.RUN_LABEL,
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
        "num_warmup": control.NUM_WARMUP,
        "num_samples": control.NUM_SAMPLES,
        "num_chains": control.NUM_CHAINS,
        "target_accept_prob": control.TARGET_ACCEPT,
        "warmup_max_tree_depth": control.WARMUP_DEPTH,
        "max_tree_depth": control.SAMPLE_DEPTH,
        "x64": False,
        "tree_depth_cap_num_steps": control.CAP_NUM_STEPS,
        "tree_depth_cap_count": 2,
        "tree_depth_cap_fraction": 0.1,
        "divergence_count": 0,
        "mean_accept_prob": float(np.mean(accept)),
        "final_step_size": 0.002,
    }
    args.diagnostics.write_text(json.dumps(diagnostics) + "\n", encoding="utf-8")


def test_explicit_m8_configuration_matches_v17_target(tmp_path):
    m8_args = control._capture_sampling_args(control.DEFAULT_INIT, tmp_path)
    v17_args = _capture_v17_args()
    m8_values = vars(m8_args).copy()
    v17_values = vars(v17_args).copy()
    for name in ("out_dir", "run_label"):
        m8_values.pop(name)
        v17_values.pop(name)
    assert m8_values == v17_values
    contract = control._sampling_contract(m8_args)
    assert contract["matmul_precision"] == "highest"
    assert contract["num_warmup"] == 200
    assert contract["num_samples"] == 20
    assert contract["seed"] == 0


def test_v8_evidence_supports_highest_intervention():
    evidence = control._validate_v8_evidence(control.DEFAULT_V8_DIR)
    assert evidence["classification"] == "full_highest_precision_recovers_score"
    assert evidence["full_default_maximum_score_difference"] > 0.2
    assert evidence["full_highest_maximum_score_difference"] <= 0.2


def test_highest_context_reaches_dot_general_precision():
    def multiply(left, right):
        return left @ right

    with jax.default_matmul_precision("highest"):
        jaxpr = jax.make_jaxpr(multiply)(jnp.ones((2, 2)), jnp.ones((2, 2)))
    assert "HIGHEST" in str(jaxpr)


def test_strict_json_is_atomic_and_rejects_collision(tmp_path):
    output = tmp_path / "value.json"
    control._write_strict_json(output, {"finite": 1.0})
    assert json.loads(output.read_text()) == {"finite": 1.0}
    assert not (tmp_path / "value.json.tmp").exists()
    with pytest.raises(FileExistsError):
        control._write_strict_json(output, {"finite": 2.0})
    with pytest.raises(ValueError):
        control._write_strict_json(tmp_path / "nan.json", {"bad": np.nan})


def test_completed_artifacts_are_recomputed_and_tamper_fails(tmp_path):
    args = _validation_args(tmp_path)
    _write_synthetic_completed_run(args)
    summary = control.summarize_run(args)
    assert summary["artifact_integrity_passed"] is True
    assert summary["findings"]["tree_depth_cap_fraction"] == 0.1
    assert control.validate_saved_artifacts(args)["artifact_integrity_passed"]

    saved = json.loads(args.summary_output.read_text())
    saved["findings"]["tree_depth_cap_count"] = 19
    args.summary_output.write_text(json.dumps(saved) + "\n", encoding="utf-8")
    assert not control.validate_saved_artifacts(args)["artifact_integrity_passed"]


def test_launcher_is_native_tcsh_and_has_one_factor_intervention():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env -S tcsh -f")
    assert "/bin/tcsh" not in text
    assert "setenv JAX_DEFAULT_MATMUL_PRECISION highest" in text
    assert 'python -u "$sampler"' in text
    assert "--num-warmup 200" in text
    assert "--num-samples 20" in text
    assert "--nohup" in text
    assert "/tmp/doraex_m7_gpu.lock" in text
    assert "Barker" not in text
    assert "NeuTra" not in text
    assert "empirical" not in text


def test_launcher_hash_pin_matches_control_source():
    text = LAUNCHER.read_text(encoding="utf-8")
    digest = control.sha256_file(Path(control.__file__))
    assert f'"{digest}"' in text
