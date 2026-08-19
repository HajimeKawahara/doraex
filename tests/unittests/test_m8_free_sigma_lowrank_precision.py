"""Unit tests for the M8 selective-precision LowRank diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import json
from pathlib import Path
import sys

import jax
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v8_free_sigma_lowrank_precision as precision,
)


def _toy_lowrank_inputs() -> dict[str, np.ndarray | float]:
    """Return a deterministic, genuinely non-diagonal LowRank problem."""

    rng = np.random.default_rng(90473)
    unit_factor = rng.normal(size=(7, 3))
    residual = rng.normal(size=7)
    cov_diag = np.exp(rng.normal(scale=0.3, size=7)) + 0.2
    eigenvalues = np.asarray([0.17, 0.83, 2.4])
    inv_diag = 1.0 / cov_diag
    return {
        "unit_factor": unit_factor,
        "cov_diag": cov_diag,
        "residual": residual,
        "eigenvalues": eigenvalues,
        "gram": unit_factor.T @ (inv_diag[:, None] * unit_factor),
        "rhs_unit": unit_factor.T @ (inv_diag * residual),
        "q0": float(residual @ (inv_diag * residual)),
        "log_diag_sum": float(np.log(cov_diag).sum()),
        "gp_jitter": 3.0e-4,
        "prior_scale": 0.7,
    }


def _component(components, name: str):
    if isinstance(components, Mapping):
        return components[name]
    return getattr(components, name)


def _value_derivative(result) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the deliberately small evaluator result contract."""

    if isinstance(result, Mapping):
        value = result["value"]
        derivative = result.get("gradient", result.get("tangent"))
    else:
        value, derivative = result
    if derivative is None:
        raise AssertionError("precision evaluator omitted its derivative")
    return np.asarray(value), np.asarray(derivative)


def _toy_components():
    toy = _toy_lowrank_inputs()
    common = {
        "eigenvalues": toy["eigenvalues"],
        "gp_jitter": toy["gp_jitter"],
        "prior_scale": toy["prior_scale"],
    }
    full = precision.make_fullfactor_components(
        toy["unit_factor"],
        toy["cov_diag"],
        toy["residual"],
        **common,
    )
    reduced = precision.make_reduced_components(
        toy["gram"],
        toy["rhs_unit"],
        toy["q0"],
        toy["log_diag_sum"],
        observation_count=np.asarray(toy["residual"]).size,
        **common,
    )
    return full, reduced


def _complete_synthetic_arrays() -> dict[str, np.ndarray]:
    """Return a mechanically complete artifact with nonfinite science values."""

    arrays = precision._initial_arrays(precision.planned_call_names())
    arrays["call_returned"][:] = True
    arrays["fixed_control_index"] = np.asarray(1380)
    center_u = np.float32(np.log(precision.EVALUATION_SIGMA))
    step = np.float32(precision.U_STEP)
    arrays["center_flat"] = np.zeros(precision.FLAT_DIMENSION, dtype=np.float32)
    arrays["center_flat"][precision.SIGMA_INDEX] = center_u
    arrays["point_u"] = np.asarray(
        [center_u - step, center_u, center_u + step], dtype=np.float32
    ).astype(np.float64)
    arrays["point_sigma"] = np.exp(arrays["point_u"])
    arrays["source_sigma"] = np.asarray(0.32316479086875916)
    arrays["source_potential_replay_error"] = np.asarray(0.0)
    arrays["constrained_roundtrip_error"] = np.asarray(0.0)
    arrays["factor_input_finite"] = np.asarray(True)
    arrays["factor_shape"] = np.asarray((57344, 767))
    arrays["factor_dtype"] = np.asarray("float32")
    arrays["factor_sha256"] = np.asarray("0" * 64)
    arrays["cov_diag_minimum"] = np.asarray(0.1)
    arrays["gp_jitter"] = np.asarray(5.0e-7)
    arrays["sigma_prior_scale"] = np.asarray(0.3)
    arrays["reference_scale_replay_error"] = np.asarray(0.0)
    arrays["reference_factor_replay_error"] = np.asarray(0.0)
    arrays["eigenvalues"] = np.ones(767, dtype=np.float32)
    arrays["reference_scales"] = np.asarray(
        jax.device_get(
            precision._sigma_scales(  # noqa: SLF001
                jax.numpy.asarray(center_u, dtype=jax.numpy.float32),
                jax.numpy.asarray(arrays["eigenvalues"], dtype=jax.numpy.float32),
                jax.numpy.asarray(arrays["gp_jitter"], dtype=jax.numpy.float32),
            )
        ),
        dtype=np.float32,
    )
    arrays["reduced_unit_gram"] = np.eye(767, dtype=np.float64)
    arrays["reduced_unit_rhs"] = np.zeros(767, dtype=np.float64)
    arrays["reduced_residual_dinv_residual"] = np.asarray(1.0)
    arrays["reduced_logdet_diagonal"] = np.asarray(0.0)
    arrays["observation_count"] = np.asarray(57344)
    arrays["shared_prior_potential"] = np.asarray(3.0)
    arrays["expected_component_constant"] = np.asarray(
        0.5 * 57344 * np.log(2.0 * np.pi) + 3.0
    )
    return arrays


def _complete_synthetic_summary(arrays: dict[str, np.ndarray]) -> dict:
    """Return the strict summary corresponding to a synthetic raw artifact."""

    classification = precision.classify_precision_result(arrays)
    return {
        "schema_version": 1,
        "mode": "m8_v8_free_sigma_lowrank_precision",
        "execution_completed": True,
        "method_contract_passed": True,
        "artifact_integrity_passed": True,
        "scientific_agreement_is_not_a_completion_gate": True,
        "classification": classification,
        "method": {
            "arms": list(precision.ARM_NAMES),
            "components": list(precision.COMPONENTS),
            "methods": list(precision.METHODS),
            "reverse_method": "jax.value_and_grad(scalar_fn)",
            "forward_method": "jax.jvp(scalar_fn, unit_u_tangent)",
            "matmul_precision_by_arm": {
                arm: (
                    "inherited_unset"
                    if precision.ARM_PRECISION[arm][1] is None
                    else precision.ARM_PRECISION[arm][1]
                )
                for arm in precision.ARM_NAMES
            },
            "component_forward_calls_are_scalar": True,
            "hvp_computed": False,
            "full_hessian_computed": False,
            "fallback_allowed": False,
            "fullfactor_f64_arm_computed": False,
            "selective_x64_scope": (
                "reduced 767x767 LowRank statistics and scalar sigma algebra only"
            ),
        },
        "configuration": {
            "fixed_control_index": 1380,
            "evaluation_sigma_requested": precision.EVALUATION_SIGMA,
            "u_step_requested": precision.U_STEP,
            "score_atol": precision.SCORE_ATOL,
            "potential_atol": precision.POTENTIAL_ATOL,
            "full_target_x64": False,
        },
        "runtime": {
            "jax_enable_x64_after_selective_arm": False,
            "default_matmul_precision_after_arms": None,
        },
        "predecessor": precision.V7_EXPECTED_SHA256,
    }


def test_planned_calls_cover_each_method_component_and_precision_arm_once():
    calls = precision.planned_call_names()
    full_expected = tuple(
        f"{arm}:total:{method}"
        for arm in precision.FULL_ARMS
        for method in precision.METHODS
    )
    component_expected = tuple(
        f"{arm}:{component}:{method}"
        for arm in precision.COMPONENT_ARMS
        for component in precision.COMPONENTS
        for method in precision.METHODS
    )
    expected = full_expected + component_expected
    assert calls == expected
    assert len(calls) == 28
    assert len(calls) == len(set(calls))


def test_sigma_halfnormal_nll_includes_the_log_transform_jacobian():
    scale = 0.7
    points = np.asarray([-1.1, -0.2, 0.35])
    fn = lambda u: precision._sigma_prior(u, scale)  # noqa: SLF001, E731
    values, gradients = _value_derivative(
        precision.evaluate_scalar_reverse(
            fn,
            points,
            dtype=np.float64,
            matmul_precision="highest",
        )
    )
    sigma = np.exp(points)
    expected_values = (
        0.5 * np.square(sigma / scale)
        + np.log(scale)
        + 0.5 * np.log(np.pi / 2.0)
        - points
    )
    expected_gradients = np.square(sigma / scale) - 1.0
    np.testing.assert_allclose(values, expected_values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gradients,
        expected_gradients,
        rtol=1e-12,
        atol=1e-12,
    )


def test_toy_fullfactor_and_reduced_stats_close_in_value_and_reverse_mode():
    full, reduced = _toy_components()
    points = np.asarray([-1.25, -0.37, 0.41])

    for name in precision.COMPONENTS:
        full_result = precision.evaluate_scalar_reverse(
            _component(full, name),
            points,
            dtype=np.float64,
            matmul_precision="highest",
        )
        reduced_result = precision.evaluate_scalar_reverse(
            _component(reduced, name),
            points,
            dtype=np.float64,
            matmul_precision="highest",
        )
        full_value, full_gradient = _value_derivative(full_result)
        reduced_value, reduced_gradient = _value_derivative(reduced_result)
        np.testing.assert_allclose(full_value, reduced_value, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(
            full_gradient,
            reduced_gradient,
            rtol=2e-10,
            atol=2e-10,
        )


def test_toy_fullfactor_and_reduced_stats_close_with_pure_forward_jvp():
    full, reduced = _toy_components()
    points = np.asarray([-1.25, -0.37, 0.41])

    for name in precision.COMPONENTS:
        full_result = precision.evaluate_scalar_forward_jvp(
            _component(full, name),
            points,
            dtype=np.float64,
            matmul_precision="highest",
        )
        reduced_result = precision.evaluate_scalar_forward_jvp(
            _component(reduced, name),
            points,
            dtype=np.float64,
            matmul_precision="highest",
        )
        full_value, full_tangent = _value_derivative(full_result)
        reduced_value, reduced_tangent = _value_derivative(reduced_result)
        np.testing.assert_allclose(full_value, reduced_value, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(
            full_tangent,
            reduced_tangent,
            rtol=2e-10,
            atol=2e-10,
        )


def test_reduced_factory_does_not_round_f64_statistics_before_context_entry():
    """Selective f64 must preserve bits that are below f32 resolution."""

    gram_value = 1.0 + 2.0**-30
    eigenvalue = 0.9
    jitter = 1.0e-3
    log_diag_sum = 0.17
    u = -0.3
    components = precision.make_reduced_components(
        np.asarray([[gram_value]], dtype=np.float64),
        np.asarray([0.2], dtype=np.float64),
        np.float64(1.4),
        np.float64(log_diag_sum),
        eigenvalues=np.asarray([eigenvalue], dtype=np.float64),
        gp_jitter=jitter,
        prior_scale=0.7,
        observation_count=1,
    )
    values, _ = _value_derivative(
        precision.evaluate_scalar_reverse(
            _component(components, "logdet"),
            np.asarray([u]),
            dtype=np.float64,
            matmul_precision="highest",
        )
    )
    scale_squared = np.exp(2.0 * u) * eigenvalue + jitter
    expected = log_diag_sum + np.log1p(scale_squared * gram_value)
    np.testing.assert_allclose(values[0], expected, rtol=0.0, atol=2e-13)


@pytest.mark.parametrize(
    ("dtype", "matmul_precision"),
    ((np.float32, None), (np.float64, "highest")),
)
def test_precision_evaluators_restore_process_context(dtype, matmul_precision):
    _, reduced = _toy_components()
    fn = _component(reduced, "mahalanobis")
    points = np.asarray([-0.4, 0.1])
    before_x64 = bool(jax.config.x64_enabled)
    before_matmul = jax.config.jax_default_matmul_precision

    reverse = precision.evaluate_scalar_reverse(
        fn,
        points,
        dtype=dtype,
        matmul_precision=matmul_precision,
    )
    forward = precision.evaluate_scalar_forward_jvp(
        fn,
        points,
        dtype=dtype,
        matmul_precision=matmul_precision,
    )

    assert bool(jax.config.x64_enabled) is before_x64
    assert jax.config.jax_default_matmul_precision == before_matmul
    reverse_value, reverse_gradient = _value_derivative(reverse)
    forward_value, forward_tangent = _value_derivative(forward)
    expected_dtype = np.dtype(dtype)
    assert reverse_value.dtype == expected_dtype
    assert reverse_gradient.dtype == expected_dtype
    assert forward_value.dtype == expected_dtype
    assert forward_tangent.dtype == expected_dtype


def test_source_uses_first_derivatives_only_and_has_no_fullfactor_f64_arm():
    source = Path(precision.__file__).read_text(encoding="utf-8")
    forward_source = inspect.getsource(precision.evaluate_scalar_forward_jvp)
    assert "jax.hessian" not in source
    assert "jax.jacfwd(jax.grad" not in source
    assert "jax.jvp(jax.grad" not in source
    assert "jax.jvp(jax.value_and_grad" not in source
    assert "jax.jvp" in forward_source
    assert "jax.grad" not in forward_source
    assert "jax.value_and_grad" not in forward_source
    assert "isolated_fullfactor_f64" not in precision.COMPONENT_ARMS
    assert "reduced_stats_f64_highest" in precision.COMPONENT_ARMS


def test_classification_separates_full_graph_and_precision_recovery():
    shape = (len(precision.ARM_NAMES), 3, len(precision.COMPONENTS))
    arrays = {"arm_reverse_forward_delta": np.zeros(shape)}
    full_default = precision.ARM_NAMES.index("full_f32_default")
    full_highest = precision.ARM_NAMES.index("full_f32_highest")
    isolated_default = precision.ARM_NAMES.index("isolated_fullfactor_f32_default")

    arrays["arm_reverse_forward_delta"][full_default, :, 0] = 1.0
    assert (
        precision.classify_precision_result(arrays)
        == "full_highest_precision_recovers_score"
    )

    arrays["arm_reverse_forward_delta"][full_highest, :, 0] = 1.0
    assert (
        precision.classify_precision_result(arrays)
        == "isolating_fullfactor_from_model_graph_recovers_score"
    )

    arrays["arm_reverse_forward_delta"][isolated_default, :, 0] = 1.0
    assert (
        precision.classify_precision_result(arrays)
        == "isolated_fullfactor_highest_precision_recovers_score"
    )

    isolated_highest = precision.ARM_NAMES.index("isolated_fullfactor_f32_highest")
    arrays["arm_reverse_forward_delta"][isolated_highest, :, 0] = 1.0
    assert (
        precision.classify_precision_result(arrays)
        == "only_reduced_float64_control_is_score_consistent"
    )

    reduced_f64 = precision.ARM_NAMES.index("reduced_stats_f64_highest")
    arrays["arm_reverse_forward_delta"][reduced_f64, :, 0] = 1.0
    assert (
        precision.classify_precision_result(arrays) == "score_disagreement_unresolved"
    )

    arrays["arm_reverse_forward_delta"][full_default, 0, 0] = np.nan
    assert (
        precision.classify_precision_result(arrays)
        == "one_or_more_score_paths_nonfinite"
    )


def test_checkpoint_preserves_nonfinite_findings_without_raising(tmp_path):
    calls = precision.planned_call_names()
    arrays = precision._initial_arrays(calls)
    arrays["arm_component_derivative_forward"][0, 0, 0] = np.nan
    arrays_path = tmp_path / precision.ARRAYS_NAME
    checkpoint_path = tmp_path / precision.CHECKPOINT_NAME
    precision._write_checkpoint("allocated", arrays_path, checkpoint_path, arrays)
    result = precision.validate_checkpoint_artifacts(checkpoint_path, arrays_path)
    assert result["checkpoint_validation_passed"] is True
    with np.load(arrays_path, allow_pickle=False) as archive:
        assert not archive["arm_component_derivative_forward_finite_mask"][0, 0, 0]


def test_final_artifact_allows_complete_nonfinite_scientific_finding(tmp_path):
    arrays = _complete_synthetic_arrays()
    arrays_path = tmp_path / precision.ARRAYS_NAME
    checkpoint_path = tmp_path / precision.CHECKPOINT_NAME
    summary_path = tmp_path / precision.SUMMARY_NAME
    arrays["checkpoint_sequence"] = np.asarray(len(precision.planned_call_names()) + 3)
    precision._write_checkpoint("final", arrays_path, checkpoint_path, arrays)
    summary = _complete_synthetic_summary(arrays)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = precision.validate_saved_artifacts(
        summary_path, arrays_path, checkpoint_path
    )
    assert result["artifact_integrity_passed"] is True
    assert result["scientific_agreement_required"] is False


def test_method_contract_pins_coordinates_but_not_scientific_score_agreement():
    arrays = _complete_synthetic_arrays()
    assert np.isnan(arrays["v7_default_reverse_error"])
    assert precision._method_contract_passed(  # noqa: SLF001
        arrays, all_calls_returned=True
    )

    arrays["point_u"] = np.array(arrays["point_u"], copy=True)
    arrays["point_u"][1] += 0.01
    assert not precision._method_contract_passed(  # noqa: SLF001
        arrays, all_calls_returned=True
    )


def test_artifact_validator_recomputes_derived_arrays(tmp_path):
    arrays = _complete_synthetic_arrays()
    arrays["arm_component_derivative_reverse"][0, 0, 0] = 1.0
    arrays["arm_component_derivative_forward"][0, 0, 0] = 0.0
    arrays_path = tmp_path / precision.ARRAYS_NAME
    checkpoint_path = tmp_path / precision.CHECKPOINT_NAME
    arrays["checkpoint_sequence"] = np.asarray(len(precision.planned_call_names()) + 3)
    precision._write_checkpoint("final", arrays_path, checkpoint_path, arrays)
    with np.load(arrays_path, allow_pickle=False) as archive:
        tampered = {name: np.asarray(archive[name]) for name in archive.files}
    tampered["arm_reverse_forward_delta"] = np.array(
        tampered["arm_reverse_forward_delta"], copy=True
    )
    tampered["arm_reverse_forward_delta"][0, 0, 0] = 0.0
    precision.v5._write_npz_atomic(arrays_path, tampered)
    with pytest.raises(ValueError, match="Derived array mismatch"):
        precision.validate_checkpoint_artifacts(checkpoint_path, arrays_path)


def test_checkpoint_validator_rejects_schema_and_mode_tampering(tmp_path):
    arrays = _complete_synthetic_arrays()
    arrays_path = tmp_path / precision.ARRAYS_NAME
    checkpoint_path = tmp_path / precision.CHECKPOINT_NAME
    arrays["checkpoint_sequence"] = np.asarray(len(precision.planned_call_names()) + 3)
    precision._write_checkpoint("final", arrays_path, checkpoint_path, arrays)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["schema_version"] = 999
    checkpoint["mode"] = "unrelated"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint JSON and NPZ disagree"):
        precision.validate_checkpoint_artifacts(checkpoint_path, arrays_path)


def test_launcher_is_native_tcsh_and_exposes_validation_only():
    launcher_path = ROOT / "csh/exe_m8_v8_check_free_sigma_lowrank_precision.csh"
    launcher = launcher_path.read_text(encoding="utf-8")
    source = Path(precision.__file__).read_text(encoding="utf-8")
    assert launcher.startswith("#!/usr/bin/env -S tcsh -f")
    assert '"--nohup"' in launcher
    assert '"--validate-only"' in launcher
    assert "/bin/tcsh" not in launcher
    assert "2>&1" not in launcher
    assert "/tmp/doraex_m7_gpu.lock" in launcher
    assert "validate_saved_artifacts" in launcher
    assert precision.v5.sha256_file(Path(precision.__file__)) in launcher
    assert "jax.hessian" not in source
