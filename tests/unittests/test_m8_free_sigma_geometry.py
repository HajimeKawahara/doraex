"""Unit tests for the M8 free-sigma geometry diagnostic."""

from argparse import Namespace
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy import special as scipy_special

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.luhman16b_yama import (  # noqa: E402
    check_m8_v5_free_sigma_geometry as geometry,
)


def test_sigma_u_prior_terms_include_exp_jacobian():
    """The direct positive coordinate must include d sigma / d u."""

    sigma = np.asarray([0.1, 0.3, 0.5])
    terms = geometry.sigma_u_prior_terms(sigma, 0.3)
    log_probability = np.asarray(
        dist.HalfNormal(0.3).log_prob(jnp.asarray(sigma))
    )
    np.testing.assert_allclose(
        terms["potential"],
        -(log_probability + np.log(sigma)),
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        terms["gradient_u"],
        sigma**2 / 0.3**2 - 1.0,
    )
    np.testing.assert_allclose(
        terms["curvature_uu"],
        2.0 * sigma**2 / 0.3**2,
    )


def test_halfnormal_cdf_coordinate_derivatives():
    """Analytic inverse-CDF derivatives match finite differences."""

    sigma = 0.32316479086875916
    scale = 0.3
    terms = geometry.halfnormal_cdf_coordinate_terms(sigma, scale)

    def u_of_g(gaussian):
        probability = scipy_special.ndtr(gaussian)
        return math.log(
            scale * scipy_special.ndtri(0.5 * (1.0 + probability))
        )

    step = 1.0e-4
    first = (u_of_g(terms["g"] + step) - u_of_g(terms["g"] - step)) / (
        2.0 * step
    )
    second = (
        u_of_g(terms["g"] + step)
        - 2.0 * u_of_g(terms["g"])
        + u_of_g(terms["g"] - step)
    ) / step**2
    assert terms["du_dg"] == pytest.approx(first, rel=2.0e-7)
    assert terms["d2u_dg2"] == pytest.approx(second, rel=2.0e-5)


def test_archive_coordinate_mapping_and_actual_medoid():
    """Coordinate conversion preserves rows and medoid selection returns a draw."""

    atmosphere = np.asarray(
        [
            np.full(7, -1.0),
            np.zeros(7),
            np.full(7, 0.2),
            np.full(7, 3.0),
        ]
    )
    normalization = np.asarray(
        [
            np.full(4, 1.02),
            np.full(4, 1.10),
            np.full(4, 1.11),
            np.full(4, 1.19),
        ]
    )
    sigma = np.asarray([0.1, 0.3, 0.31, 0.8])
    coordinates = geometry.unconstrained_archive_coordinates(
        atmosphere,
        normalization,
        sigma,
        a_bounds=(1.0, 1.2),
    )
    assert coordinates.shape == (4, 12)
    np.testing.assert_allclose(coordinates[1, 7:11], 0.0, atol=3.0e-15)
    np.testing.assert_allclose(coordinates[:, -1], np.log(sigma))
    result = geometry.robust_actual_medoid(coordinates)
    assert result["index"] in range(4)
    assert np.array_equal(coordinates[result["index"]], coordinates[int(result["index"])])


def test_lowrank_profile_reconstructs_numpyro_distribution():
    """Cached sigma rescaling reproduces the LowRankMVN log probability."""

    unit_factor = jnp.asarray(
        [
            [0.2, -0.1],
            [0.4, 0.3],
            [-0.2, 0.5],
            [0.1, 0.2],
        ]
    )
    eigenvalues = jnp.asarray([1.5, 0.4])
    jitter = jnp.asarray(5.0e-7)
    loc = jnp.asarray([0.1, -0.2, 0.4, 0.7])
    cov_diag = jnp.asarray([0.3, 0.5, 0.4, 0.6])
    observed = jnp.asarray([0.0, 0.3, 0.5, 0.2])
    shared_prior_potential = jnp.asarray(1.7)
    prior_scale = jnp.asarray(0.3)
    sigma = 0.32316479086875916
    u = jnp.log(jnp.asarray(sigma))
    reference_eigen_scales = jnp.sqrt(sigma**2 * eigenvalues + jitter)
    reference_factor = unit_factor * reference_eigen_scales[None, :]
    total, auxiliary = geometry._lowrank_profile_terms(
        u,
        reference_factor,
        reference_eigen_scales,
        eigenvalues,
        jitter,
        loc,
        cov_diag,
        observed,
        shared_prior_potential,
        prior_scale,
    )
    factor = unit_factor * jnp.sqrt(sigma**2 * eigenvalues + jitter)[None, :]
    expected_nll = -dist.LowRankMultivariateNormal(
        loc=loc,
        cov_factor=factor,
        cov_diag=cov_diag,
    ).log_prob(observed)
    prior = geometry.sigma_u_prior_terms(sigma, 0.3)
    np.testing.assert_allclose(auxiliary[0], expected_nll, rtol=2.0e-6)
    np.testing.assert_allclose(
        total,
        expected_nll + shared_prior_potential + prior["potential"],
        rtol=2.0e-6,
    )
    gradient = jax.grad(
        lambda coordinate: geometry._lowrank_profile_terms(
            coordinate,
            reference_factor,
            reference_eigen_scales,
            eigenvalues,
            jitter,
            loc,
            cov_diag,
            observed,
            shared_prior_potential,
            prior_scale,
        )[0]
    )(u)
    curvature = jax.grad(
        jax.grad(
            lambda coordinate: geometry._lowrank_profile_terms(
                coordinate,
                reference_factor,
                reference_eigen_scales,
                eigenvalues,
                jitter,
                loc,
                cov_diag,
                observed,
                shared_prior_potential,
                prior_scale,
            )[0]
        )
    )(u)
    def official_total(coordinate):
        candidate_sigma = jnp.exp(coordinate)
        candidate_factor = unit_factor * jnp.sqrt(
            candidate_sigma**2 * eigenvalues + jitter
        )[None, :]
        negative_log_likelihood = -dist.LowRankMultivariateNormal(
            loc=loc,
            cov_factor=candidate_factor,
            cov_diag=cov_diag,
        ).log_prob(observed)
        sigma_prior_potential = (
            0.5 * (candidate_sigma / prior_scale) ** 2
            - coordinate
            + jnp.log(prior_scale)
            + 0.5 * jnp.log(jnp.pi / 2.0)
        )
        return (
            negative_log_likelihood
            + shared_prior_potential
            + sigma_prior_potential
        )

    official_gradient = jax.grad(official_total)(u)
    official_curvature = jax.grad(jax.grad(official_total))(u)
    forward_curvature = jax.jacfwd(
        jax.jacfwd(
            lambda coordinate: geometry._lowrank_profile_terms(
                coordinate,
                reference_factor,
                reference_eigen_scales,
                eigenvalues,
                jitter,
                loc,
                cov_diag,
                observed,
                shared_prior_potential,
                prior_scale,
            )[0]
        )
    )(u)
    np.testing.assert_allclose(gradient, official_gradient, rtol=2.0e-6)
    np.testing.assert_allclose(curvature, official_curvature, rtol=2.0e-6)
    np.testing.assert_allclose(forward_curvature, official_curvature, rtol=2.0e-6)
    assert np.isfinite(float(gradient))
    assert np.isfinite(float(curvature))


def test_real_artifact_validation_selects_actual_draws(tmp_path):
    """Pinned controls select the deterministic actual medoid draws."""

    args = Namespace(
        init_from=str(ROOT / "results/m7/p2/v6/samples.npz"),
        free_samples=str(ROOT / "results/m8/v1/samples.npz"),
        free_diagnostics=str(ROOT / "results/m8/v1/diagnostics.json"),
        fixed_samples=str(ROOT / "results/m8/v3/fixed_seed0/samples.npz"),
        fixed_diagnostics=str(
            ROOT / "results/m8/v3/fixed_seed0/diagnostics.json"
        ),
        initial_replay=str(
            ROOT / "results/m8/v3/fixed_seed0/initial_point_replay.json"
        ),
        out_dir=str(tmp_path / "fresh-output"),
        profile_sigma_min=0.05,
        profile_sigma_max=0.7,
        profile_num=33,
        sigma_markers=geometry.DEFAULT_SIGMA_MARKERS,
        curvature_sigmas=geometry.DEFAULT_CURVATURE_SIGMAS,
        cross_u_steps=geometry.DEFAULT_CROSS_U_STEPS,
        potential_atol=0.05,
        x64=False,
    )
    validation = geometry._validate_configuration(args)
    assert validation["state_selection"]["fixed_control_probe"]["index"] == 1380
    assert (
        validation["state_selection"]["nonstationary_free_probe"]["index"]
        == 750
    )
    assert [
        item["index"]
        for item in validation["state_selection"][
            "nonstationary_free_chain_probes"
        ]
    ] == [361, 295, 477, 561, 1156]


def test_strict_json_rejects_nonfinite_values():
    """Scientific JSON cannot silently contain NaN or infinity."""

    with pytest.raises(ValueError, match="Nonfinite"):
        geometry._json_ready({"bad": np.nan})


def test_launcher_keeps_integrity_and_frozen_dependency_guards():
    """The launcher must fail closed before marking a diagnostic complete."""

    launcher = (
        ROOT / "csh/exe_m8_v5_check_free_sigma_geometry.csh"
    ).read_text(encoding="utf-8")
    assert "numerical_integrity_passed" in launcher
    assert "execution_completed" in launcher
    assert "marginal_likelihood.py" in launcher
    assert "src/doraex/constants.py" in launcher
    assert "expected_exojax_path" in launcher
    assert "expected_numpyro_version" in launcher
    assert "numpyro_continuous_source" in launcher
    assert "/tmp/doraex_m7_gpu.lock" in launcher
    assert (
        "--curvature-sigmas "
        "0.27526917,0.32316479086875916,0.42335291,0.53313493"
    ) in launcher
    assert "assert summary.get" not in launcher
    assert geometry.sha256_file(Path(geometry.__file__)) in launcher
