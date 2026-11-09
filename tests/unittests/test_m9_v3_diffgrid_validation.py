"""Check likelihood-distance accounting and conservative validation gates."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/luhman16b_yama"
sys.path.insert(0, str(EXAMPLES))
import check_m9_v3_diffgrid as validation


def test_masked_comparison_matches_direct_gaussian_likelihood():
    reference = np.asarray([[2.0, 8.0], [3.0, 1.0]])
    prediction = np.asarray([[2.1, np.nan], [2.8, 1.3]])
    observed = np.asarray([[1.9, 0.0], [3.1, 1.0]])
    mask = np.asarray([[True, False], [True, True]])
    variance = 0.04
    actual = validation.comparison_metrics(prediction, reference, observed, mask, variance)
    expected_delta = -0.5 * np.sum(((observed - prediction)[mask] ** 2 - (observed - reference)[mask] ** 2) / variance)
    np.testing.assert_allclose(actual["delta_log_likelihood"], expected_delta)
    np.testing.assert_allclose(actual["gaussian_mean_kl"], 1.75)
    assert actual["observation_count"] == 3
    with pytest.raises(FloatingPointError):
        validation.comparison_metrics(prediction, reference, observed, np.ones_like(mask), variance)


def test_skipped_gradients_never_pass_the_combined_gate():
    report = {"chips": {"0": {"variants": {"nodes21": {"disk": {"gaussian_mean_kl": 0.1, "delta_log_likelihood": 0.3}}}}}}
    args = SimpleNamespace(
        nodes=(21,), chip_indices=(0,), skip_gradients=True,
        gradient_directions=("T0",), max_joint_kl=1.0,
        max_absolute_delta_log_likelihood=3.0,
        gradient_relative_tolerance=0.01, gradient_absolute_noise_tolerance=0.001,
    )
    validation.summarize_checks(report, args)
    assert report["forward_checks"]["nodes21"]["accuracy_passed"]
    assert not report["passed"]
    assert not report["forward_checks"]["nodes21"]["gradients_complete"]


def test_joint_gate_sums_chip_kl():
    derivative = {"finite_difference_checks": [{"derivative_relative_rms_error": 0.001, "derivative_difference_rms_over_noise": 0.1}]}
    variant_derivative = {**derivative, "teacher_derivative_relative_rms_error": 0.001, "teacher_derivative_difference_rms_over_noise": 0.1}
    chip = {"teacher_gradients": {"T0": derivative}, "variants": {"nodes21": {"disk": {"gaussian_mean_kl": 0.6, "delta_log_likelihood": 0.2}, "gradients": {"T0": variant_derivative}}}}
    report = {"chips": {"0": chip, "1": chip}}
    args = SimpleNamespace(
        nodes=(21,), chip_indices=(0, 1), skip_gradients=False,
        gradient_directions=("T0",), max_joint_kl=1.0,
        max_absolute_delta_log_likelihood=3.0,
        gradient_relative_tolerance=0.01, gradient_absolute_noise_tolerance=0.001,
    )
    validation.summarize_checks(report, args)
    assert report["forward_checks"]["nodes21"]["gradients_passed"]
    assert report["forward_checks"]["nodes21"]["joint_gaussian_mean_kl"] == 1.2
    assert not report["passed"]


def test_state_preserves_joint_values_and_converts_limb_darkening(tmp_path):
    state = {name: np.asarray(1.0) for name in validation.ATMOSPHERE_NAMES}
    state.update(period=np.asarray(4.83), cosi=np.asarray(0.485), v=np.asarray(31.2),
                 q1=np.asarray(0.81), q2=np.asarray(0.59), A=np.asarray([1.1]),
                 sigma_d=np.asarray([0.02]), log_w=np.zeros((1, 2)),
                 delta_log_p=np.linspace(-0.2, 0.2, 12))
    path = tmp_path / "state.npz"
    np.savez(path, **state)
    samples = {"chip_indices": np.asarray([0]), "nside": np.asarray(1), "obs_times": np.asarray([0, 1])}
    loaded = validation.load_state(path, samples)
    np.testing.assert_array_equal(loaded["delta_log_p"], state["delta_log_p"])
    np.testing.assert_allclose([loaded["u1"], loaded["u2"]], [1.062, -0.162])
    assert loaded["P"] == 4.83


def test_teacher_cache_defaults_to_separate_m9_v3_root(tmp_path):
    args = validation.parse_args(["--out-dir", str(tmp_path), "--prepare-cache-only"])
    assert args.teacher_cache_dir == validation.ROOT / "data/opacities/luhman16b_premodit_m9_v3"


def test_failed_run_keeps_provenance_and_cannot_pass(monkeypatch, tmp_path):
    def fail(args):
        args.out_dir.mkdir()
        (args.out_dir / "diffgrid_validation.json").write_text(json.dumps({
            "status": "running", "exojax_commit": "recorded_commit",
            "passed": True, "passed_by_nodes": {"21": True},
        }))
        raise RuntimeError("Teacher archive is incompatible.")

    monkeypatch.setattr(validation, "_run", fail)
    out_dir = tmp_path / "failed_run"
    with pytest.raises(RuntimeError, match="incompatible"):
        validation.main(["--out-dir", str(out_dir), "--prepare-cache-only"])
    report = json.loads((out_dir / "diffgrid_validation.json").read_text())
    assert report["status"] == "failed"
    assert report["failure_reason"] == "RuntimeError: Teacher archive is incompatible."
    assert report["exojax_commit"] == "recorded_commit"
    assert not report["passed"]
    assert report["passed_by_nodes"] == {"21": False}


def test_output_collision_does_not_rewrite_previous_report(tmp_path):
    path = tmp_path / "diffgrid_validation.json"
    previous = '{"status":"complete"}\n'
    path.write_text(previous)
    with pytest.raises(FileExistsError):
        validation.main(["--out-dir", str(tmp_path), "--prepare-cache-only"])
    assert path.read_text() == previous
