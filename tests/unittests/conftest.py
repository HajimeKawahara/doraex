"""Compatibility and local-artifact policy for the unit-test matrix."""

from pathlib import Path

import jax
import pytest


ROOT = Path(__file__).resolve().parents[2]

# These checks intentionally replay the immutable workstation artifacts that
# back the M8 precision investigation.  The multi-GB result bundle is not part
# of the source distribution, so clean CI checkouts must not treat its absence
# as a unit-test failure.  Synthetic tests in the same modules still run.
_LOCAL_ARTIFACT_TESTS = {
    "tests/unittests/test_m8_free_sigma_derivative_capture.py::"
    "test_real_validation_pins_v6_failure_and_fixed_state",
    "tests/unittests/test_m8_free_sigma_geometry.py::"
    "test_real_artifact_validation_selects_actual_draws",
    "tests/unittests/test_m8_free_sigma_highest_precision_control.py::"
    "test_v8_evidence_supports_highest_intervention",
    "tests/unittests/test_m8_free_sigma_highest_precision_control.py::"
    "test_completed_artifacts_are_recomputed_and_tamper_fails",
    "tests/unittests/test_m8_free_sigma_highest_precision_long_control.py::"
    "test_explicit_long_configuration_matches_m8_v1_baseline",
    "tests/unittests/test_m8_free_sigma_highest_precision_long_control.py::"
    "test_v9_guard_is_pinned_as_the_long_run_precondition",
    "tests/unittests/test_m8_free_sigma_highest_precision_long_control.py::"
    "test_contract_and_summary_validator_recompute_and_reject_tampering",
    "tests/unittests/test_m8_free_sigma_highest_precision_long_control.py::"
    "test_summary_rejects_a_tampered_v9_reuse_contract",
    "tests/unittests/test_m8_free_sigma_highest_precision_long_control.py::"
    "test_summary_rejects_malformed_or_nonfinite_samples",
    "tests/unittests/test_m8_free_sigma_hvp_geometry.py::"
    "test_real_validation_pins_predecessor_and_fixed_medoid",
    "tests/unittests/test_m8_nuts_clean_control.py::"
    "test_v17_archive_replays_exact_common_physical_and_latent_initial_point",
    "tests/unittests/test_m8_nuts_clean_control.py::"
    "test_reused_free_arm_summary_records_work_proxy_and_warmup_limitation",
}

_LOCAL_ARTIFACT_SENTINELS = (
    ROOT / "results/m7/p2/v6/samples.npz",
    ROOT / "results/m7/p2/v17/samples.npz",
    ROOT / "results/m7/p2/v17/diagnostics.json",
    ROOT / "results/m8/v1/samples.npz",
    ROOT / "results/m8/v1/diagnostics.json",
    ROOT / "results/m8/v3/fixed_seed0/samples.npz",
    ROOT / "results/m8/v3/fixed_seed0/diagnostics.json",
    ROOT / "results/m8/v3/fixed_seed0/initial_point_replay.json",
    ROOT / "results/m8/v5/free_sigma_geometry/FAILED",
    ROOT
    / "results/m8/v5/free_sigma_geometry/m8_v5_free_sigma_geometry_summary.json",
    ROOT
    / "results/m8/v5/free_sigma_geometry/m8_v5_free_sigma_geometry_arrays.npz",
    ROOT / "results/m8/v6/free_sigma_hvp_geometry/FAILED",
    ROOT / "results/m8/v8/free_sigma_lowrank_precision/COMPLETE",
    ROOT / "results/m8/v9/free_sigma_highest_precision_seed0/COMPLETE",
)


def pytest_collection_modifyitems(items):
    """Skip workstation-only artifact replays when their bundle is absent."""

    missing = tuple(path for path in _LOCAL_ARTIFACT_SENTINELS if not path.is_file())
    if not missing:
        return
    reason = (
        "requires the uncommitted M7/M8 run-artifact bundle; "
        f"first missing path: {missing[0]}"
    )
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        base_nodeid = item.nodeid.partition("[")[0]
        if base_nodeid in _LOCAL_ARTIFACT_TESTS:
            item.add_marker(marker)


@pytest.fixture(autouse=True)
def expose_jax_enable_x64(monkeypatch):
    """Backfill the top-level X64 context on JAX 0.6.x."""

    if not hasattr(jax, "enable_x64"):
        from jax.experimental import enable_x64

        monkeypatch.setattr(jax, "enable_x64", enable_x64, raising=False)
