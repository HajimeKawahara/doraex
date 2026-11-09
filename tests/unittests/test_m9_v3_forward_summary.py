"""Check provenance binding, chip coverage, and joint forward gate arithmetic."""

import copy
import json
from pathlib import Path
import sys

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/luhman16b_yama"
sys.path.insert(0, str(EXAMPLES))
import summarize_m9_v3_forward as summary


def make_report(tmp_path, *, chips=(0, 1, 2, 3), gradients=True):
    state_path, sample_path = tmp_path / "state.npz", tmp_path / "samples.npz"
    state_path.write_bytes(b"coherent state")
    sample_path.write_bytes(b"source data")
    cache_root = tmp_path / "cache"
    report = {
        "mode": "m9_v3_monotone_real_space_diffgrid_validation", "status": "complete",
        "cache_root": str(cache_root), "exojax_commit": "a" * 40,
        "x64": True, "jax_x64": True, "nx": 8, "noise_jitter": 1e-6,
        "disk_normalization": "A_chip_times_global_mean_uniform_disk_spectrum",
        "local_normalization": "none", "spectral_interpolation": "existing_linear_interpolation",
        "interpolation": "fritsch_carlson_monotone_log_slopes_in_inverse_temperature",
        "teacher_profile_kernel": "real_space", "nodes": [21],
        "state_file": str(state_path), "state_sha256": summary.sha256(state_path),
        "samples": str(sample_path), "samples_sha256": summary.sha256(sample_path),
        "sample_index": 4, "chip_indices": list(chips), "chips": {},
    }
    derivative = {"finite_difference_checks": [{
        "dimensionless_step": 0.01, "derivative_relative_rms_error": 0.001,
        "derivative_difference_rms_over_noise": 0.01,
    }]}
    for chip in chips:
        records = {}
        for molecule in summary.MOLECULES:
            path = cache_root / f"chip{chip}" / f"opa{molecule}.zarr"
            path.mkdir(parents=True, exist_ok=True)
            metadata_path = path / "zarr.json"
            metadata_path.write_text(json.dumps({"attributes": {
                "opa_type": "OpaDiffgrid", "opa_state": {"teacher_profile_kernel": "real_space"},
                "array_shapes": {"inverse_temperature_grid": [21], "nu_grid": [8]},
            }}))
            records[molecule] = {"path": str(path), "teacher_profile_kernel": "real_space", "metadata_sha256": summary.sha256(metadata_path)}
        candidate = {**derivative, "teacher_derivative_relative_rms_error": 0.001, "teacher_derivative_difference_rms_over_noise": 0.01}
        report["chips"][str(chip)] = {
            "teacher_gradients": {name: copy.deepcopy(derivative) for name in summary.GRADIENT_DIRECTIONS} if gradients else {},
            "variants": {"nodes21": {
                "cache": records, "disk": {"gaussian_mean_kl": 0.1, "delta_log_likelihood": 0.2},
                "gradients": {name: copy.deepcopy(candidate) for name in summary.GRADIENT_DIRECTIONS} if gradients else {},
            }},
        }
    return report


def write_report(tmp_path, report, name="report.json"):
    path = tmp_path / name
    path.write_text(json.dumps(report))
    return path


def test_one_complete_state_can_pass_without_claiming_posterior_coverage(tmp_path):
    report = make_report(tmp_path)
    path = write_report(tmp_path, report)
    gate = summary.aggregate([path], 21)
    assert gate["passed"]
    assert gate["states"][0]["joint_gaussian_mean_kl"] == 0.4
    assert gate["states"][0]["joint_delta_log_likelihood"] == 0.8
    assert len(gate["cache_metadata_sha256"]) == 16
    assert gate["report_sha256"][str(path)] == summary.sha256(path)
    assert "listed physical states only" in gate["scope"]


def test_four_separate_chip_reports_are_aggregated_once(tmp_path):
    full = make_report(tmp_path)
    paths = []
    for chip in summary.CHIPS:
        report = {**full, "chip_indices": [chip], "chips": {str(chip): full["chips"][str(chip)]}}
        paths.append(write_report(tmp_path, report, f"chip{chip}.json"))
    assert summary.aggregate(paths, 21)["passed"]
    duplicate = write_report(tmp_path, json.loads(paths[0].read_text()), "duplicate.json")
    with pytest.raises(ValueError, match="Duplicate chip"):
        summary.aggregate([*paths, duplicate], 21)


def test_incomplete_chip_coverage_and_skipped_gradients_cannot_pass(tmp_path):
    report = make_report(tmp_path, chips=(0,), gradients=False)
    gate = summary.aggregate([write_report(tmp_path, report)], 21)
    assert not gate["passed"]
    state = gate["states"][0]
    assert state["missing_chip_indices"] == [1, 2, 3]
    assert not state["gradients_passed"]


def test_joint_kl_recomputed_despite_report_pass_boolean(tmp_path):
    report = make_report(tmp_path)
    report["passed"] = True
    for chip in report["chips"].values():
        chip["variants"]["nodes21"]["disk"]["gaussian_mean_kl"] = 0.3
    gate = summary.aggregate([write_report(tmp_path, report)], 21)
    assert not gate["passed"]
    assert gate["states"][0]["joint_gaussian_mean_kl"] == 1.2


@pytest.mark.parametrize("target", ["state", "cache"])
def test_changed_provenance_is_rejected(tmp_path, target):
    report = make_report(tmp_path)
    path = write_report(tmp_path, report)
    changed_path = Path(report["state_file"]) if target == "state" else Path(report["chips"]["0"]["variants"]["nodes21"]["cache"]["CO"]["path"]) / "zarr.json"
    changed_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        summary.aggregate([path], 21)


@pytest.mark.parametrize("status", ["failed", "running"])
def test_failed_or_running_reports_are_rejected(tmp_path, status):
    report = make_report(tmp_path)
    report["status"] = status
    with pytest.raises(ValueError, match="Only completed"):
        summary.aggregate([write_report(tmp_path, report)], 21)


def test_differing_contracts_are_rejected(tmp_path):
    first = make_report(tmp_path, chips=(0, 1))
    second = make_report(tmp_path, chips=(2, 3))
    second["x64"] = second["jax_x64"] = False
    with pytest.raises(ValueError, match="contracts differ"):
        summary.aggregate([write_report(tmp_path, first, "first.json"), write_report(tmp_path, second, "second.json")], 21)


def test_gradient_failures_are_recomputed_from_recorded_errors(tmp_path):
    report = make_report(tmp_path)
    variant = report["chips"]["0"]["variants"]["nodes21"]
    variant["gradients"]["T0"]["teacher_derivative_relative_rms_error"] = 0.5
    report["forward_checks"] = {"nodes21": {"gradients_passed": True}}
    assert not summary.aggregate([write_report(tmp_path, report)], 21)["passed"]
