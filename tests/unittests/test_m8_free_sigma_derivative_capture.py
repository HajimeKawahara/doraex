"""Unit tests for the checkpointed M8 derivative-failure capture."""

from argparse import Namespace
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v7_free_sigma_derivative_capture as capture,
)


def _classification_arrays() -> dict[str, np.ndarray]:
    arrays = {
        "standalone_gradient": np.zeros((3, capture.FLAT_DIMENSION)),
        "bundle_point_gradient": np.zeros((3, capture.FLAT_DIMENSION)),
        "bundle_point_directional_gradient": np.zeros(3),
        "center_bundle_gradient": np.zeros((2, capture.FLAT_DIMENSION)),
        "center_bundle_directional_gradient": np.zeros(2),
        "verlet_gradient": np.zeros(capture.FLAT_DIMENSION),
        "scalar_gradient_u": np.asarray(0.0),
        "center_bundle_hvp": np.zeros((2, capture.FLAT_DIMENSION)),
        "scalar_hessian_uu": np.asarray(1.0),
        "analytic_gradient_u": np.zeros(3),
        "analytic_hessian_uu": np.ones(3),
        "standalone_fd_column": np.zeros(capture.FLAT_DIMENSION),
    }
    arrays["center_bundle_hvp"][:, capture.SIGMA_INDEX] = 1.0
    arrays["standalone_fd_column"][capture.SIGMA_INDEX] = 1.0
    return arrays


def test_safe_vector_agreement_does_not_promote_finite_subset():
    """A partial finite intersection is reported but never called agreement."""

    reference = np.asarray([1.0, np.nan, 3.0])
    candidate = np.asarray([1.0, 2.0, 3.0])
    result = capture.vector_agreement_or_none(reference, candidate)
    assert result["component_count"] == 3
    assert result["common_finite_count"] == 2
    assert result["relative_error"] is None
    assert result["cosine"] is None


def test_raw_finite_masks_preserve_nan_and_infinities():
    """Raw derivative masks distinguish finite, NaN, and infinite values."""

    call_names = capture.planned_call_names(2)
    arrays = capture._initial_arrays(repeat_count=2, call_names=call_names)
    arrays["center_bundle_hvp"][0, :3] = [1.0, np.nan, np.inf]
    masks = capture.raw_finite_masks(arrays)
    np.testing.assert_array_equal(
        masks["center_bundle_hvp_finite_mask"][0, :3],
        [True, False, False],
    )


def test_classifier_separates_first_and_second_order_failures():
    """Classification prioritizes NUTS-relevant first-gradient failures."""

    arrays = _classification_arrays()
    assert (
        capture.classify_derivative_capture(arrays)
        == "finite_local_derivative_geometry"
    )

    arrays = _classification_arrays()
    arrays["center_bundle_hvp"][:, 0] = np.nan
    assert (
        capture.classify_derivative_capture(arrays)
        == "cross_column_higher_order_ad_nonfinite"
    )

    arrays = _classification_arrays()
    arrays["standalone_gradient"][0, 0] = np.nan
    assert (
        capture.classify_derivative_capture(arrays)
        == "first_order_nonfinite_nuts_relevant"
    )

    arrays = _classification_arrays()
    arrays["bundle_point_gradient"][1, 0] = np.nan
    assert (
        capture.classify_derivative_capture(arrays)
        == "hvp_bundle_context_first_order_nonfinite"
    )

    arrays = _classification_arrays()
    arrays["center_bundle_gradient"][1, 0] = np.nan
    assert (
        capture.classify_derivative_capture(arrays)
        == "hvp_bundle_context_first_order_nonfinite"
    )

    arrays = _classification_arrays()
    arrays["verlet_gradient"][0] = np.nan
    assert (
        capture.classify_derivative_capture(arrays)
        == "first_order_nonfinite_nuts_relevant"
    )

    arrays = _classification_arrays()
    arrays["verlet_gradient"][0] = 1.0
    assert (
        capture.classify_derivative_capture(arrays)
        == "first_order_context_or_reduction_disagreement"
    )

    arrays = _classification_arrays()
    arrays["scalar_gradient_u"] = np.asarray(np.nan)
    assert (
        capture.classify_derivative_capture(arrays)
        == "directional_derivative_path_nonfinite"
    )

    arrays = _classification_arrays()
    arrays["center_bundle_directional_gradient"][0] = 1.0
    assert (
        capture.classify_derivative_capture(arrays)
        == "directional_derivative_path_disagreement"
    )

    arrays = _classification_arrays()
    arrays["center_bundle_hvp"][:, capture.SIGMA_INDEX] = 1.0e30
    assert (
        capture.classify_derivative_capture(arrays)
        == "finite_but_second_order_disagreement"
    )


def test_checkpoint_roundtrip_keeps_raw_nonfinite_values(tmp_path):
    """Atomic checkpoints retain raw NaNs and their exact finite masks."""

    arrays_path = tmp_path / "capture.npz"
    checkpoint_path = tmp_path / "checkpoint.json"
    arrays = capture._initial_arrays(
        repeat_count=2,
        call_names=capture.planned_call_names(2),
    )
    arrays["center_bundle_hvp"][0, :3] = [1.0, np.nan, -np.inf]
    capture._write_checkpoint(
        "synthetic",
        arrays_path,
        checkpoint_path,
        arrays,
    )
    with np.load(arrays_path, allow_pickle=False) as archive:
        assert np.isnan(archive["center_bundle_hvp"][0, 1])
        assert np.isneginf(archive["center_bundle_hvp"][0, 2])
        np.testing.assert_array_equal(
            archive["center_bundle_hvp_finite_mask"][0, :3],
            [True, False, False],
        )
    text = checkpoint_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    result = capture.validate_checkpoint_artifacts(arrays_path, checkpoint_path)
    assert result["checkpoint_validation_passed"] is True
    assert result["returned_count"] == 0


def test_artifact_validator_accepts_observed_derivative_nan(tmp_path):
    """A captured derivative NaN is a finding, not artifact corruption."""

    summary_path = tmp_path / "summary.json"
    arrays_path = tmp_path / "arrays.npz"
    checkpoint_path = tmp_path / "checkpoint.json"
    call_names = capture.planned_call_names(2)
    arrays = capture._initial_arrays(repeat_count=2, call_names=call_names)
    arrays["call_returned"][:] = True
    arrays["center_flat"][:] = 0.0
    arrays["point_flat"][:] = 0.0
    arrays["point_sigma"][:] = capture.EVALUATION_SIGMA
    arrays["standalone_value"][:] = 0.0
    arrays["standalone_gradient"][:] = 0.0
    arrays["bundle_point_value"][:] = 0.0
    arrays["bundle_point_gradient"][:] = 0.0
    arrays["bundle_point_directional_gradient"][:] = 0.0
    arrays["bundle_point_hvp"][:] = 0.0
    arrays["center_bundle_value"][:] = 0.0
    arrays["center_bundle_gradient"][:] = 0.0
    arrays["center_bundle_directional_gradient"][:] = 0.0
    arrays["center_bundle_hvp"][:] = 0.0
    arrays["center_bundle_hvp"][0, 0] = np.nan
    arrays["verlet_value"] = np.asarray(0.0)
    arrays["verlet_gradient"][:] = 0.0
    arrays["scalar_gradient_u"] = np.asarray(0.0)
    arrays["scalar_hessian_uu"] = np.asarray(1.0)
    arrays["analytic_total_potential"][:] = 0.0
    arrays["analytic_gradient_u"][:] = 0.0
    arrays["analytic_hessian_uu"][:] = 1.0
    arrays["analytic_cholesky_min_diagonal"][:] = 1.0
    arrays["source_potential_replay_error"] = np.asarray(0.0)
    arrays["constrained_roundtrip_error"] = np.asarray(0.0)
    for name in capture.RAW_NUMERIC_KEYS:
        if name not in arrays:
            raise AssertionError(name)
    capture._write_checkpoint("final", arrays_path, checkpoint_path, arrays)
    capture.v5._write_json_atomic(
        summary_path,
        {
            "execution_completed": True,
            "calculation_integrity_passed": True,
            "derivative_integrity_passed": False,
            "raw_nonfinite_preserved": True,
            "classification": "cross_column_higher_order_ad_nonfinite",
            "configuration": {
                "repeat_count": 2,
                "evaluation_sigma_requested": capture.EVALUATION_SIGMA,
                "u_step_requested": capture.U_STEP,
                "potential_atol": 0.05,
                "value_alignment_atol": 0.5,
            },
            "findings": {
                "classification": "cross_column_higher_order_ad_nonfinite"
            },
            "method": {
                "hvp_method": capture.HVP_METHOD,
                "hvp_direction": "u_sigma_log_p only",
                "hvp_tangent_index": capture.SIGMA_INDEX,
                "direction_count": 1,
                "hvp_fallback_allowed": False,
                "hvp_fallback_used": False,
                "full_hessian_computed": False,
            },
        },
    )
    result = capture.validate_saved_artifacts(
        summary_path,
        arrays_path,
        checkpoint_path,
    )
    assert result["artifact_validation_passed"] is True
    assert result["derivative_integrity_passed"] is False

    valid_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    spoofed_summary = json.loads(json.dumps(valid_summary))
    spoofed_summary["classification"] = "finite_local_derivative_geometry"
    capture.v5._write_json_atomic(summary_path, spoofed_summary)
    with pytest.raises(ValueError, match="Artifact contract"):
        capture.validate_saved_artifacts(summary_path, arrays_path, checkpoint_path)

    spoofed_summary = json.loads(json.dumps(valid_summary))
    spoofed_summary["configuration"]["potential_atol"] = 1.0e9
    capture.v5._write_json_atomic(summary_path, spoofed_summary)
    with pytest.raises(ValueError, match="Artifact contract"):
        capture.validate_saved_artifacts(summary_path, arrays_path, checkpoint_path)

    capture.v5._write_json_atomic(summary_path, valid_summary)
    spoofed_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    spoofed_checkpoint["checkpoint_sequence"] += 1
    capture.v5._write_json_atomic(checkpoint_path, spoofed_checkpoint)
    with pytest.raises(ValueError, match="Artifact contract"):
        capture.validate_saved_artifacts(summary_path, arrays_path, checkpoint_path)


def test_real_validation_pins_v6_failure_and_fixed_state(tmp_path):
    """Validation identifies the exact artifact-less v6 failure."""

    args = Namespace(
        init_from=str(ROOT / "results/m7/p2/v6/samples.npz"),
        free_samples=str(ROOT / "results/m8/v1/samples.npz"),
        free_diagnostics=str(ROOT / "results/m8/v1/diagnostics.json"),
        fixed_samples=str(ROOT / "results/m8/v3/fixed_seed0/samples.npz"),
        fixed_diagnostics=str(ROOT / "results/m8/v3/fixed_seed0/diagnostics.json"),
        initial_replay=str(
            ROOT / "results/m8/v3/fixed_seed0/initial_point_replay.json"
        ),
        v5_summary=str(
            ROOT
            / "results/m8/v5/free_sigma_geometry"
            / "m8_v5_free_sigma_geometry_summary.json"
        ),
        v5_arrays=str(
            ROOT
            / "results/m8/v5/free_sigma_geometry"
            / "m8_v5_free_sigma_geometry_arrays.npz"
        ),
        v5_failed=str(ROOT / "results/m8/v5/free_sigma_geometry/FAILED"),
        v6_dir=str(ROOT / "results/m8/v6/free_sigma_hvp_geometry"),
        out_dir=str(tmp_path / "fresh-v7"),
        seed=0,
        evaluation_sigma=capture.EVALUATION_SIGMA,
        u_step=capture.U_STEP,
        repeat_count=5,
        gram_chunk_size=4096,
        potential_atol=0.05,
        x64=False,
    )
    validation = capture._validate_configuration(args)
    fixed_state = capture.v6._load_fixed_state(args)
    assert fixed_state.index == 1380
    assert validation["v6_predecessor"]["complete_absent"] is True


def test_launcher_is_native_tcsh_one_line_submitter():
    """The launcher can submit itself without a Bash wrapper."""

    launcher = (
        ROOT / "csh/exe_m8_v7_check_free_sigma_derivative_capture.csh"
    ).read_text(encoding="utf-8")
    source = Path(capture.__file__).read_text(encoding="utf-8")
    assert launcher.startswith("#!/usr/bin/env -S tcsh -f")
    assert '"--nohup"' in launcher
    assert "/bin/tcsh" not in launcher
    assert "2>&1" not in launcher
    assert "/tmp/doraex_m7_gpu.lock" in launcher
    assert "validate_saved_artifacts" in launcher
    assert "calculation_integrity_passed" in source
    assert "derivative_integrity_passed" not in launcher
    assert "jax.hessian" not in source
    assert capture.v5.sha256_file(Path(capture.__file__)) in launcher
