"""Unit tests for the production-graph M8 sigma HVP diagnostic."""

from argparse import Namespace
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v6_free_sigma_hvp_geometry as geometry,
)


def test_directional_hvp_matches_known_quadratic_column():
    """Forward-over-reverse returns one known Hessian column."""

    hessian = jnp.asarray(
        [
            [3.0, -0.4, 0.7],
            [-0.4, 2.0, -1.2],
            [0.7, -1.2, 4.0],
        ]
    )
    linear = jnp.asarray([0.2, -0.1, 0.3])
    point = jnp.asarray([0.4, -0.3, 0.8])
    direction = jnp.asarray([0.0, 0.0, 1.0])

    def potential(value):
        return 0.5 * value @ hessian @ value + linear @ value

    (value, gradient), (directional_gradient, hvp) = (
        geometry.directional_value_gradient_hvp(
            potential,
            point,
            direction,
        )
    )
    np.testing.assert_allclose(value, potential(point))
    np.testing.assert_allclose(gradient, hessian @ point + linear)
    np.testing.assert_allclose(directional_gradient, gradient[-1])
    np.testing.assert_allclose(hvp, hessian[:, -1])

    scalar_gradient, scalar_hessian = geometry.scalar_sigma_gradient_hessian(
        potential,
        point,
        sigma_index=2,
    )
    np.testing.assert_allclose(scalar_gradient, gradient[-1])
    np.testing.assert_allclose(scalar_hessian, hessian[-1, -1])


def test_central_gradient_column_converges_for_quartic():
    """Larger diagnostic steps retain the expected second-order convergence."""

    point = np.asarray([0.3, -0.2])

    def gradient(value):
        return np.asarray([value[0] ** 3 + value[1], value[0] + 2.0 * value[1]])

    expected = np.asarray([3.0 * point[0] ** 2, 1.0])
    errors = []
    for step in (0.08, 0.04, 0.02):
        plus = point.copy()
        minus = point.copy()
        plus[0] += step
        minus[0] -= step
        column = geometry.central_gradient_column(
            gradient(plus), gradient(minus), step
        )
        errors.append(np.linalg.norm(column - expected))
    assert errors[1] == pytest.approx(errors[0] / 4.0)
    assert errors[2] == pytest.approx(errors[1] / 4.0)


def test_vector_agreement_reports_direction_and_scale():
    """Agreement metrics distinguish aligned and reversed columns."""

    reference = np.asarray([3.0, 4.0])
    aligned = geometry.vector_agreement(reference, 1.1 * reference)
    reversed_result = geometry.vector_agreement(reference, -reference)
    assert aligned["cosine"] == pytest.approx(1.0)
    assert aligned["relative_error"] == pytest.approx(0.1)
    assert reversed_result["cosine"] == pytest.approx(-1.0)


def test_fd_consistency_accepts_any_step_meeting_all_criteria():
    """A marker passes when one common step satisfies every HVP check."""

    shared_relative_error = np.asarray([[0.10, 0.20], [0.60, 0.30]])
    shared_cosine = np.asarray([[0.99, 0.98], [0.95, 0.80]])
    huu_relative_error = np.asarray([[0.80, 0.20], [0.10, 0.10]])
    mask = geometry.finite_difference_consistency_mask(
        shared_relative_error,
        shared_cosine,
        huu_relative_error,
    )
    np.testing.assert_array_equal(mask, [[False, True], [False, False]])
    np.testing.assert_array_equal(np.any(mask, axis=1), [True, False])


def test_float32_value_ulp_resolves_new_fd_steps():
    """The revised steps give multiple ULPs for representative curvature."""

    value_ulp = abs(float(np.spacing(np.float32(-88310.0))))
    assert value_ulp == pytest.approx(0.0078125)
    representative_hessian = 40.0
    signal_ulp = np.asarray(
        [representative_hessian * step**2 / value_ulp for step in (0.02, 0.04, 0.08)]
    )
    np.testing.assert_allclose(signal_ulp, [2.048, 8.192, 32.768])


def test_analytic_lowrank_score_and_hessian_match_autodiff():
    """The float64 closed form matches LowRankMVN autodiff on a toy model."""

    rng = np.random.default_rng(20260818)
    observation_count = 9
    rank = 3
    unit_factor = rng.normal(size=(observation_count, rank)) * 0.15
    eigenvalues = np.asarray([1.5, 0.4, 0.08])
    gp_jitter = 5.0e-7
    cov_diag = rng.uniform(0.2, 0.8, size=observation_count)
    loc = rng.normal(size=observation_count)
    observed = rng.normal(size=observation_count)
    residual = observed - loc
    sigma = 0.32316479086875916
    reference_scales = np.sqrt(sigma**2 * eigenvalues + gp_jitter)
    reference_factor = unit_factor * reference_scales[None, :]
    statistics = geometry.build_unit_lowrank_statistics(
        reference_factor,
        reference_scales,
        cov_diag,
        residual,
        chunk_size=4,
    )
    analytic = geometry.analytic_lowrank_sigma_terms(
        sigma,
        eigenvalues,
        gp_jitter,
        statistics["unit_gram"],
        statistics["unit_rhs"],
        statistics["residual_dinv_residual"],
        statistics["logdet_diagonal"],
        observation_count,
    )

    with jax.enable_x64(True):

        def negative_log_likelihood(u):
            candidate_sigma = jnp.exp(u)
            factor = jnp.asarray(unit_factor, dtype=jnp.float64) * jnp.sqrt(
                candidate_sigma**2
                * jnp.asarray(eigenvalues, dtype=jnp.float64)
                + gp_jitter
            )[None, :]
            return -dist.LowRankMultivariateNormal(
                loc=jnp.asarray(loc, dtype=jnp.float64),
                cov_factor=factor,
                cov_diag=jnp.asarray(cov_diag, dtype=jnp.float64),
            ).log_prob(jnp.asarray(observed, dtype=jnp.float64))

        u = jnp.log(jnp.asarray(sigma, dtype=jnp.float64))
        expected_value = negative_log_likelihood(u)
        expected_gradient = jax.grad(negative_log_likelihood)(u)
        expected_hessian = jax.grad(jax.grad(negative_log_likelihood))(u)
    np.testing.assert_allclose(
        analytic["negative_log_likelihood"], expected_value, rtol=2.0e-12
    )
    np.testing.assert_allclose(
        analytic["gradient_u_likelihood"], expected_gradient, rtol=2.0e-11
    )
    np.testing.assert_allclose(
        analytic["hessian_uu_likelihood"], expected_hessian, rtol=3.0e-11
    )


def test_same_prior_cdf_hessian_formula_matches_autodiff():
    """The analytic CDF pullback includes the nonstationary gradient term."""

    sigma = 0.42335291
    scale = 0.3
    transform = geometry.v5.halfnormal_cdf_coordinate_terms(sigma, scale)
    u0 = math.log(sigma)

    def likelihood_u(u):
        return 0.7 * u**3 - 0.2 * u

    likelihood_gradient = float(jax.grad(likelihood_u)(u0))
    likelihood_hessian = float(jax.grad(jax.grad(likelihood_u))(u0))
    pulled_hessian = (
        likelihood_hessian * transform["du_dg"] ** 2
        + likelihood_gradient * transform["d2u_dg2"]
        + 1.0
    )

    def u_of_g(g):
        probability = jax.scipy.special.ndtr(g)
        return jnp.log(
            scale
            * jax.scipy.special.ndtri(0.5 * (1.0 + probability))
        )

    def gaussian_potential(g):
        return likelihood_u(u_of_g(g)) + 0.5 * g**2

    expected = jax.grad(jax.grad(gaussian_potential))(transform["g"])
    assert pulled_hessian == pytest.approx(float(expected), rel=3.0e-5)


def test_real_validation_pins_predecessor_and_fixed_medoid(tmp_path):
    """Validation requires the intended failed v5 and actual fixed draw."""

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
        out_dir=str(tmp_path / "fresh-v6"),
        seed=0,
        profile_sigma_min=0.05,
        profile_sigma_max=0.70,
        profile_num=33,
        sigma_markers=geometry.DEFAULT_SIGMA_MARKERS,
        fd_u_steps=geometry.DEFAULT_FD_U_STEPS,
        repeat_count=5,
        gram_chunk_size=4096,
        potential_atol=0.05,
        x64=False,
    )
    validation = geometry._validate_configuration(args)
    assert validation["state_selection"]["fixed_control_probe"]["index"] == 1380
    assert validation["predecessor"]["execution_completed"] is True
    assert validation["predecessor"]["numerical_integrity_passed"] is False


def test_launcher_separates_calculation_integrity_from_findings():
    """Gradient instability is an output, while missing HVP remains fatal."""

    launcher_path = ROOT / "csh/exe_m8_v6_check_free_sigma_hvp_geometry.csh"
    launcher = launcher_path.read_text(encoding="utf-8")
    source = Path(geometry.__file__).read_text(encoding="utf-8")
    assert "calculation_integrity_passed" in launcher
    assert "diagnostic_findings" in launcher
    assert "repeat_gradient_bitwise_stable" not in launcher.split(
        "artifact_gate", maxsplit=1
    )[0]
    assert 'method.get("hvp_fallback_used") is False' in launcher
    assert 'method.get("full_hessian_computed") is False' in launcher
    assert "/tmp/doraex_m7_gpu.lock" in launcher
    assert "--fd-u-steps 0.02,0.04,0.08" in launcher
    assert "jax.hessian" not in source
    assert "hvp_fallback_allowed\": False" in source
    assert geometry.v5.sha256_file(Path(geometry.__file__)) in launcher
