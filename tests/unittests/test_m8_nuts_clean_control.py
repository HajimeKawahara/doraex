"""Tests for the M8 fixed/free sigma_log_p clean causal control."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doraex.workflows import on_the_fly_pressure_retrieval as retrieval  # noqa: E402
from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v2_fixed_free_initial_point as replay_guard,
    m7_p2_v17_run,
    m8_v1_run,
    m8_v2_run,
    summarize_m8_v2_fixed_free_control as control_summary,
)


LAUNCHER = (
    ROOT
    / "csh"
    / "exe_m8_v2_fixed_sigma_log_p_v17_initial_full_dense_control.csh"
)
LONG_LAUNCHER = (
    ROOT
    / "csh"
    / "exe_m8_v3_fixed_sigma_log_p_v17_initial_full_dense_long_control.csh"
)


def _wrapper_args(monkeypatch, wrapper, name, arguments=()):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)
    monkeypatch.setattr(sys, "argv", [name, *map(str, arguments)])
    wrapper.main()
    return captured["args"]


def test_fixed_wrapper_changes_v17_only_by_the_conditional_intervention(
    monkeypatch,
):
    free_args = _wrapper_args(
        monkeypatch,
        m7_p2_v17_run,
        "m7_p2_v17_run.py",
    )
    fixed_args = _wrapper_args(monkeypatch, m8_v2_run, "m8_v2_run.py")
    differences = {
        name
        for name, value in vars(free_args).items()
        if value != vars(fixed_args)[name]
    }

    assert differences == {
        "direct_sigma_log_p",
        "fixed_sigma_log_p",
        "out_dir",
        "run_label",
    }
    assert free_args.direct_sigma_log_p
    assert free_args.fixed_sigma_log_p is None
    assert not fixed_args.direct_sigma_log_p
    assert (
        fixed_args.fixed_sigma_log_p
        == m8_v2_run.V17_INITIAL_SIGMA_LOG_P
    )
    assert fixed_args.dense_mass and free_args.dense_mass
    assert fixed_args.adapt_mass_matrix and free_args.adapt_mass_matrix
    assert fixed_args.num_warmup == free_args.num_warmup == 200
    assert fixed_args.num_samples == free_args.num_samples == 20
    assert fixed_args.seed == free_args.seed == 0
    assert fixed_args.target_accept_prob == free_args.target_accept_prob == 0.95
    assert fixed_args.warmup_max_tree_depth == free_args.warmup_max_tree_depth == 9
    assert fixed_args.max_tree_depth == free_args.max_tree_depth == 11
    assert m8_v2_run.ATMOSPHERE_ROTATION == (
        m7_p2_v17_run.ATMOSPHERE_ROTATION
    )
    assert Path(fixed_args.out_dir) == (
        ROOT / "results" / "m8" / "v2" / "fixed_seed0"
    )


def test_v17_archive_replays_exact_common_physical_and_latent_initial_point(
    monkeypatch,
):
    free_args = _wrapper_args(
        monkeypatch,
        m7_p2_v17_run,
        "m7_p2_v17_run.py",
    )
    fixed_args = _wrapper_args(monkeypatch, m8_v2_run, "m8_v2_run.py")
    free_physical = retrieval.load_initial_values(free_args, 4, 14)
    fixed_physical = retrieval.load_initial_values(fixed_args, 4, 14)

    assert free_physical["sigma_log_p"] == 0.32316479086875916
    assert free_physical["sigma_log_p"] == fixed_args.fixed_sigma_log_p
    assert float(free_physical["sigma_log_p"]).hex() == (
        "0x1.4aebb60000000p-2"
    )
    for name in free_physical:
        np.testing.assert_array_equal(
            np.asarray(free_physical[name]),
            np.asarray(fixed_physical[name]),
        )

    rotation = retrieval.validate_atmosphere_rotation(
        fixed_args.atmosphere_rotation_matrix,
        dimension=7,
    )
    free_fixed_nuisance = retrieval.build_fixed_nuisance_values(
        free_args,
        free_physical,
    )
    fixed_fixed_nuisance = retrieval.build_fixed_nuisance_values(
        fixed_args,
        fixed_physical,
    )
    free_initial = retrieval.build_sampling_initial_values(
        free_args,
        free_physical,
        free_fixed_nuisance,
        rotation,
    )
    fixed_initial = retrieval.build_sampling_initial_values(
        fixed_args,
        fixed_physical,
        fixed_fixed_nuisance,
        rotation,
    )

    assert set(free_initial) == {
        "atmosphere_rotated",
        "sigma_log_p",
        "A",
    }
    assert set(fixed_initial) == {"atmosphere_rotated", "A"}
    np.testing.assert_array_equal(
        np.asarray(free_initial["atmosphere_rotated"]),
        np.asarray(fixed_initial["atmosphere_rotated"]),
    )
    np.testing.assert_array_equal(
        np.asarray(free_initial["A"]),
        np.asarray(fixed_initial["A"]),
    )
    np.testing.assert_array_equal(
        np.asarray(free_initial["sigma_log_p"]),
        np.asarray(fixed_args.fixed_sigma_log_p, dtype=np.float32),
    )


def test_replay_guard_matches_forward_gradients_and_log_joint_terms():
    sigma = m8_v2_run.V17_INITIAL_SIGMA_LOG_P

    def model_with_sigma_control(fixed_sigma_log_p, direct_sigma_log_p):
        shared = numpyro.sample(
            "atmosphere_rotated",
            dist.Normal(jnp.zeros((2,)), 1.0).to_event(1),
        )
        normalization = numpyro.sample("A", dist.Uniform(1.0, 1.2))
        if direct_sigma_log_p:
            sigma_log_p = numpyro.sample(
                "sigma_log_p",
                dist.HalfNormal(0.3),
            )
        else:
            sigma_log_p = numpyro.deterministic(
                "sigma_log_p",
                jnp.asarray(fixed_sigma_log_p),
            )
        loc = jnp.full((3,), shared[0] / normalization)
        cov_factor = jnp.asarray(
            [[sigma_log_p], [0.5 * sigma_log_p], [0.25 * sigma_log_p]]
        )
        numpyro.sample(
            "obs",
            dist.LowRankMultivariateNormal(
                loc=loc,
                cov_factor=cov_factor,
                cov_diag=jnp.full((3,), 0.1 + 0.01 * shared[1] ** 2),
            ),
            obs=jnp.zeros((3,)),
        )

    def fixed_model():
        return model_with_sigma_control(sigma, False)

    def free_model():
        return model_with_sigma_control(None, True)

    fixed_initial = {
        "atmosphere_rotated": jnp.asarray([0.1, -0.2]),
        "A": jnp.asarray(1.1),
    }
    free_initial = {
        **fixed_initial,
        "sigma_log_p": jnp.asarray(sigma),
    }
    report = replay_guard.replay_fixed_free_initial_point(
        fixed_model,
        free_model,
        fixed_initial,
        free_initial,
        physical_sigma_log_p=sigma,
        sigma_log_p_scale=0.3,
        rng_key=jax.random.PRNGKey(0),
    )

    assert report["passed"]
    assert report["physical_sigma_log_p"]["exact_match"]
    assert report["sample_sites"]["free_adds_only_sigma_log_p"]
    assert all(item["matches"] for item in report["forward"].values())
    assert all(
        item["matches"]
        for item in report["shared_unconstrained_gradients"].values()
    )
    assert report["log_joint"]["constrained_delta_matches"]
    assert report["log_joint"]["unconstrained_delta_matches"]


def test_replay_output_collision_fails_before_target_setup(
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "initial_point_replay.json"
    output.write_text("user-owned\n", encoding="utf-8")
    monkeypatch.setattr(
        replay_guard,
        "parse_args",
        lambda: SimpleNamespace(
            init_from="unused.npz",
            output=str(output),
            seed=0,
        ),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        replay_guard.main()
    assert output.read_text(encoding="utf-8") == "user-owned\n"


def test_reused_free_arm_summary_records_work_proxy_and_warmup_limitation():
    summary = control_summary.summarize_arm(
        ROOT / "results" / "m7" / "p2" / "v17" / "samples.npz",
        ROOT / "results" / "m7" / "p2" / "v17" / "diagnostics.json",
        case="free",
        seed=0,
    )

    assert summary["divergence_count"] == 0
    assert summary["sampling_depth_cap_count"] == 18
    assert summary["sampling_depth_cap_fraction"] == 0.9
    assert summary["step_count"]["median"] == 2047.0
    assert summary["warmup_depth_cap_count"] is None
    assert "did not retain warmup" in summary["warmup_depth_cap_limitation"]
    assert summary["retained_sampling_leapfrog_step_proxy"] > 0
    assert "sigma_log_p" in summary["ess_per_retained_sampling_step_proxy"]
    assert not summary["passes_short_pilot_diagnostic_targets"]


def test_launcher_has_static_syntax_provenance_and_completion_guards():
    result = subprocess.run(
        ["tcsh", "-n", str(LAUNCHER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    source = LAUNCHER.read_text(encoding="utf-8")
    assert "/tmp/doraex_m7_gpu.lock" in source
    assert "--validate-only" in source
    assert "refusing output collision" in source
    assert "expected_workflow_sha256" in source
    assert "expected_free_samples_sha256" in source
    assert "initial_point_replay.json" in source
    assert "INITIAL_POINT_VALIDATED" in source
    assert "fixed_free_control_summary.json" in source
    assert 'mv "$output_dir/RUNNING" "$output_dir/COMPLETE"' in source
    assert "results/m8/v1" not in source


def test_long_fixed_cli_matches_the_existing_m8_v1_free_schedule(monkeypatch):
    free_args = _wrapper_args(monkeypatch, m8_v1_run, "m8_v1_run.py")
    fixed_args = _wrapper_args(
        monkeypatch,
        m8_v2_run,
        "m8_v2_run.py",
        (
            "--out-dir",
            ROOT / "results" / "m8" / "v3" / "fixed_seed0",
            "--run-label",
            "m8_v3_fixed_sigma_log_p_v17_initial_full_dense_long_control",
            "--fixed-sigma-log-p",
            m8_v2_run.V17_INITIAL_SIGMA_LOG_P,
            "--num-chains",
            1,
            "--num-warmup",
            2000,
            "--num-samples",
            1500,
            "--seed",
            0,
            "--target-accept-prob",
            0.95,
            "--warmup-max-tree-depth",
            9,
            "--max-tree-depth",
            11,
            "--dense-mass",
            "--adapt-mass-matrix",
            "--no-x64",
            "--print-summary",
        ),
    )
    differences = {
        name
        for name, value in vars(free_args).items()
        if value != vars(fixed_args)[name]
    }

    assert differences == {
        "direct_sigma_log_p",
        "fixed_sigma_log_p",
        "out_dir",
        "run_label",
    }
    assert fixed_args.fixed_sigma_log_p == m8_v2_run.V17_INITIAL_SIGMA_LOG_P
    assert not fixed_args.direct_sigma_log_p
    assert free_args.direct_sigma_log_p
    assert fixed_args.num_warmup == free_args.num_warmup == 2000
    assert fixed_args.num_samples == free_args.num_samples == 1500
    assert fixed_args.num_chains == free_args.num_chains == 1
    assert fixed_args.seed == free_args.seed == 0
    assert fixed_args.target_accept_prob == free_args.target_accept_prob == 0.95
    assert fixed_args.warmup_max_tree_depth == free_args.warmup_max_tree_depth == 9
    assert fixed_args.max_tree_depth == free_args.max_tree_depth == 11
    assert fixed_args.dense_mass and free_args.dense_mass
    assert fixed_args.adapt_mass_matrix and free_args.adapt_mass_matrix
    assert not fixed_args.x64 and not free_args.x64


def test_long_launcher_has_static_syntax_pins_and_completion_guards():
    result = subprocess.run(
        ["tcsh", "-n", str(LONG_LAUNCHER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    source = LONG_LAUNCHER.read_text(encoding="utf-8")
    assert 'set output_dir = "results/m8/v3/fixed_seed0"' in source
    assert "--fixed-sigma-log-p 0.32316479086875916" in source
    assert "--num-warmup 2000" in source
    assert "--num-samples 1500" in source
    assert "--warmup-max-tree-depth 9" in source
    assert "--max-tree-depth 11" in source
    assert "--dense-mass" in source
    assert "--adapt-mass-matrix" in source
    assert "/tmp/doraex_m7_gpu.lock" in source
    assert "--validate-only" in source
    assert "refusing output collision" in source
    assert "INITIAL_POINT_VALIDATED" in source
    assert "expected_fixed_wrapper_sha256" in source
    assert "expected_replay_script_sha256" in source
    assert "aa1bd9bb6bc9e5f146e3377aa62e3adb68c8c9a961812628144c73cd9c3a5083" in source
    assert "f45d32470ee996c62b0dd8683190d4fe5eaedddd37d4f95d0ea21003278cf8c4" in source
    assert "summarize_m8_v2_fixed_free_control.py" not in source
    assert 'mv "$output_dir/RUNNING" "$output_dir/COMPLETE"' in source
