"""Tests for the M7 prior-preserving retrieval coordinates."""

from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import NUTS
from numpyro.infer.initialization import init_to_value
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doraex.workflows.on_the_fly_pressure_retrieval import (  # noqa: E402
    GAUSSIANIZED_ATMOSPHERE_NAMES,
    JOINT_ATMOSPHERE_A_SIGMA_D_SITE,
    build_dense_mass_specification,
    build_fixed_nuisance_values,
    build_fixed_pressure_gp_eigendecomposition,
    build_initial_inverse_mass_matrix,
    build_nuts_max_tree_depth,
    build_observation_masks,
    build_sampling_initial_values,
    gaussianized_atmosphere_to_physical,
    gaussianized_nuisance_to_physical,
    on_the_fly_pressure_model,
    physical_atmosphere_to_gaussianized,
    physical_nuisance_to_gaussianized,
    sigma_log_p_parameterization,
    validate_atmosphere_rotation,
    validate_joint_atmosphere_a_sigma_d_rotation,
    zero_sum_log_w_base,
    _select_observation_rows,
)
from doraex.workflows import on_the_fly_pressure_retrieval as retrieval  # noqa: E402
from doraex.priors.spherical_gp import (  # noqa: E402
    squared_exponential_covariance,
    zero_mean_covariance_factor,
)
from examples.luhman16b_yama.m7_p2_v4_run import (  # noqa: E402
    ATMOSPHERE_ROTATION,
)
from examples.luhman16b_yama.m7_p2_v5_run import (  # noqa: E402
    ATMOSPHERE_ROTATION as FIXED_SIGMA_ATMOSPHERE_ROTATION,
)
from examples.luhman16b_yama import (  # noqa: E402
    m8_v1_run,
    m7_p2_v10_run,
    m7_p2_v11_run,
    m7_p2_v12_run,
    m7_p2_v13_run,
    m7_p2_v14_run,
    m7_p2_v15_run,
    m7_p2_v16_run,
    m7_p2_v17_run,
)


T0_BOUNDS = (1000.0, 1700.0)
ALPHA_BOUNDS = (0.05, 0.20)
LOG_VMR_BOUNDS = {
    "log_vmr_co": (-6.0, -1.0),
    "log_vmr_h2o": (-6.0, -1.0),
    "log_vmr_ch4": (-6.0, -1.0),
    "log_vmr_hf": (-10.0, -5.0),
}
LOG_P_CLOUD_BOUNDS = (-2.0, 2.0)
SIGMA_LOG_P_SCALE = 0.3


def test_zero_flux_observation_masks_are_static_and_exact():
    chips = [
        SimpleNamespace(
            chip_index=0,
            flux=np.asarray([[1.0, 0.0, -0.0, -0.2]]),
        ),
        SimpleNamespace(
            chip_index=1,
            flux=np.asarray([[0.5, 0.7, 0.9, 1.1]]),
        ),
    ]

    masked = build_observation_masks(chips, mask_zero_flux=True)
    np.testing.assert_array_equal(masked[0], [[True, False, False, True]])
    np.testing.assert_array_equal(masked[1], np.ones((1, 4), dtype=bool))

    unmasked = build_observation_masks(chips, mask_zero_flux=False)
    assert all(np.all(mask) for mask in unmasked)

    chips[0].flux[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite flux"):
        build_observation_masks(chips, mask_zero_flux=True)


def test_observation_row_selection_is_shared_by_all_likelihood_terms():
    observed = jnp.arange(6.0)
    baseline = observed + 10.0
    covariance_factor = jnp.arange(12.0).reshape(6, 2)
    noise_variance = observed + 20.0
    indices = np.asarray([0, 3, 5])

    selected = _select_observation_rows(
        observed,
        baseline,
        covariance_factor,
        noise_variance,
        indices,
    )

    np.testing.assert_array_equal(np.asarray(selected[0]), [0.0, 3.0, 5.0])
    np.testing.assert_array_equal(np.asarray(selected[1]), [10.0, 13.0, 15.0])
    np.testing.assert_array_equal(
        np.asarray(selected[2]),
        np.asarray(covariance_factor)[indices],
    )
    np.testing.assert_array_equal(np.asarray(selected[3]), [20.0, 23.0, 25.0])


def _to_physical(gaussianized):
    return gaussianized_atmosphere_to_physical(
        gaussianized,
        t0_bounds=T0_BOUNDS,
        alpha_bounds=ALPHA_BOUNDS,
        log_vmr_bounds=LOG_VMR_BOUNDS,
        log_p_cloud_bounds=LOG_P_CLOUD_BOUNDS,
        sigma_log_p_scale=SIGMA_LOG_P_SCALE,
    )


def _to_gaussianized(physical):
    return physical_atmosphere_to_gaussianized(
        physical,
        t0_bounds=T0_BOUNDS,
        alpha_bounds=ALPHA_BOUNDS,
        log_vmr_bounds=LOG_VMR_BOUNDS,
        log_p_cloud_bounds=LOG_P_CLOUD_BOUNDS,
        sigma_log_p_scale=SIGMA_LOG_P_SCALE,
    )


def test_embedded_m7_p2_v4_rotation_is_orthogonal():
    matrix = validate_atmosphere_rotation(np.asarray(ATMOSPHERE_ROTATION).ravel())

    np.testing.assert_allclose(
        np.asarray(matrix.T @ matrix),
        np.eye(len(GAUSSIANIZED_ATMOSPHERE_NAMES)),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(np.linalg.det(np.asarray(matrix)), 1.0, atol=2.0e-6)


def test_embedded_m7_p2_v5_fixed_sigma_rotation_is_orthogonal():
    dimension = len(GAUSSIANIZED_ATMOSPHERE_NAMES) - 1
    matrix = validate_atmosphere_rotation(
        np.asarray(FIXED_SIGMA_ATMOSPHERE_ROTATION).ravel(),
        dimension=dimension,
    )

    np.testing.assert_allclose(
        np.asarray(matrix.T @ matrix),
        np.eye(dimension),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(np.linalg.det(np.asarray(matrix)), 1.0, atol=2.0e-6)


def test_embedded_m7_p2_v17_direct_sigma_rotation_matches_bounded_block():
    dimension = len(GAUSSIANIZED_ATMOSPHERE_NAMES) - 1
    matrix = validate_atmosphere_rotation(
        np.asarray(m7_p2_v17_run.ATMOSPHERE_ROTATION).ravel(),
        dimension=dimension,
    )

    assert (
        m7_p2_v17_run.ATMOSPHERE_ROTATION
        == FIXED_SIGMA_ATMOSPHERE_ROTATION
    )
    np.testing.assert_allclose(
        np.asarray(matrix.T @ matrix),
        np.eye(dimension),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(np.linalg.det(np.asarray(matrix)), 1.0, atol=2.0e-6)


def test_embedded_m7_p2_v12_joint_rotation_is_orthogonal():
    matrix = validate_joint_atmosphere_a_sigma_d_rotation(
        np.asarray(m7_p2_v12_run.JOINT_ROTATION).ravel(),
        chip_count=4,
    )

    assert matrix.shape == (16, 16)
    np.testing.assert_allclose(
        np.asarray(matrix.T @ matrix),
        np.eye(16),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert len(m7_p2_v12_run.JOINT_ROTATION_EIGENVALUES) == 16
    assert np.all(np.asarray(m7_p2_v12_run.JOINT_ROTATION_EIGENVALUES) > 0.0)


def test_joint_nuisance_prior_transform_round_trip_and_quantiles():
    a_gaussianized = jnp.asarray([-1.2, -0.1, 0.5, 1.4])
    sigma_d_gaussianized = jnp.asarray([-0.8, 0.0, 0.7, 1.1])

    normalization_factor, sigma_d = gaussianized_nuisance_to_physical(
        a_gaussianized,
        sigma_d_gaussianized,
    )
    recovered_a, recovered_sigma_d = physical_nuisance_to_gaussianized(
        normalization_factor,
        sigma_d,
    )

    np.testing.assert_allclose(
        np.asarray(recovered_a),
        np.asarray(a_gaussianized),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(recovered_sigma_d),
        np.asarray(sigma_d_gaussianized),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    expected_a = 1.0 + 0.2 * dist.Normal(0.0, 1.0).cdf(a_gaussianized)
    expected_sigma_d = 0.03 * jnp.exp(sigma_d_gaussianized)
    np.testing.assert_allclose(np.asarray(normalization_factor), np.asarray(expected_a))
    np.testing.assert_allclose(np.asarray(sigma_d), np.asarray(expected_sigma_d))


def test_gaussianized_physical_prior_transform_round_trip():
    gaussianized = jnp.asarray(
        [-1.2, 0.4, -0.7, 1.1, -0.3, 0.8, 1.4, 0.6]
    )

    physical = _to_physical(gaussianized)
    recovered = _to_gaussianized(physical)

    np.testing.assert_allclose(
        np.asarray(recovered),
        np.asarray(gaussianized),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    for name, bounds in {
        "T0": T0_BOUNDS,
        "alpha": ALPHA_BOUNDS,
        **LOG_VMR_BOUNDS,
        "log_p_cloud": LOG_P_CLOUD_BOUNDS,
    }.items():
        assert bounds[0] < float(physical[name]) < bounds[1]
    assert float(physical["sigma_log_p"]) > 0.0


@pytest.mark.parametrize(
    (
        "gaussianized",
        "standardized",
        "uses_eigen_basis",
        "fixed_sigma",
        "direct_sigma",
        "expected",
    ),
    [
        (False, False, True, None, False, "not_applicable_eigen_basis"),
        (True, False, False, 0.245, False, "fixed"),
        (True, False, False, None, False, "gaussianized_prior_cdf"),
        (True, False, False, None, True, "direct_halfnormal"),
        (False, True, False, None, False, "standardized_log_normal"),
        (False, False, False, None, False, "direct_halfnormal"),
    ],
)
def test_sigma_log_p_parameterization_metadata_is_unambiguous(
    gaussianized,
    standardized,
    uses_eigen_basis,
    fixed_sigma,
    direct_sigma,
    expected,
):
    assert sigma_log_p_parameterization(
        gaussianized_atmosphere=gaussianized,
        standardized_parameters=standardized,
        uses_eigen_basis=uses_eigen_basis,
        fixed_sigma_log_p=fixed_sigma,
        direct_sigma_log_p=direct_sigma,
    ) == expected


def test_gaussianized_transform_matches_original_prior_quantiles():
    quantile = jnp.asarray(0.8)
    gaussianized = jnp.full(
        (len(GAUSSIANIZED_ATMOSPHERE_NAMES),),
        dist.Normal(0.0, 1.0).icdf(quantile),
    )

    physical = _to_physical(gaussianized)

    for name, bounds in {
        "T0": T0_BOUNDS,
        "alpha": ALPHA_BOUNDS,
        **LOG_VMR_BOUNDS,
        "log_p_cloud": LOG_P_CLOUD_BOUNDS,
    }.items():
        expected = bounds[0] + float(quantile) * (bounds[1] - bounds[0])
        np.testing.assert_allclose(float(physical[name]), expected, rtol=2.0e-6)
    expected_sigma = dist.HalfNormal(SIGMA_LOG_P_SCALE).icdf(quantile)
    np.testing.assert_allclose(
        np.asarray(physical["sigma_log_p"]),
        np.asarray(expected_sigma),
        rtol=2.0e-6,
    )


def test_fixed_pressure_gp_eigendecomposition_preserves_covariance_and_gradient():
    distance = jnp.asarray(
        [
            [0.0, 0.4, 0.8, 1.2],
            [0.4, 0.0, 0.5, 0.9],
            [0.8, 0.5, 0.0, 0.6],
            [1.2, 0.9, 0.6, 0.0],
        ]
    )
    ell = 0.4
    sigma = 0.371
    jitter = 0.5e-6
    decomposition = build_fixed_pressure_gp_eigendecomposition(
        distance,
        ell,
    )
    eigenvalues = decomposition["eigenvalues"]
    pixel_eigenvectors = decomposition["pixel_eigenvectors"]
    eigen_factor = pixel_eigenvectors * jnp.sqrt(
        sigma**2 * eigenvalues + jitter
    )[None, :]
    cholesky_factor = zero_mean_covariance_factor(
        squared_exponential_covariance(distance, sigma, ell),
        jitter=jitter,
    )

    np.testing.assert_allclose(
        np.asarray(eigen_factor @ eigen_factor.T),
        np.asarray(cholesky_factor @ cholesky_factor.T),
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    def factor_variance(amplitude):
        scales = jnp.sqrt(amplitude**2 * eigenvalues + jitter)
        factor = pixel_eigenvectors * scales[None, :]
        return jnp.sum(factor * factor)

    gradient = jax.grad(factor_variance)(jnp.asarray(sigma))
    expected = 2.0 * sigma * jnp.sum(eigenvalues)
    assert np.isfinite(float(gradient))
    np.testing.assert_allclose(
        np.asarray(gradient),
        np.asarray(expected),
        rtol=2.0e-5,
    )


def test_gaussianized_atmosphere_dense_mass_is_only_an_eight_dimensional_block():
    args = SimpleNamespace(
        dense_mass=False,
        dense_atmosphere_mass=True,
        gaussianized_atmosphere=True,
        standardized_parameters=False,
        fix_logg=True,
        eigen_mode_count=0,
        fixed_sigma_log_p=None,
        fixed_sigma_eigen=None,
    )

    dense_mass, mode, sites = build_dense_mass_specification(args)

    assert dense_mass == [("atmosphere_rotated",)]
    assert mode == "atmosphere_block"
    assert sites == ("atmosphere_rotated",)


def test_gaussianized_atmosphere_a_sigma_d_mass_is_one_joint_block():
    args = SimpleNamespace(
        dense_mass=False,
        dense_atmosphere_mass=False,
        dense_atmosphere_a_sigma_d_mass=True,
        gaussianized_atmosphere=True,
        fix_nuisance=False,
    )

    dense_mass, mode, sites = build_dense_mass_specification(args)

    assert dense_mass == [("atmosphere_rotated", "A", "sigma_d")]
    assert mode == "atmosphere_a_sigma_d_block"
    assert sites == ("atmosphere_rotated", "A", "sigma_d")


@pytest.mark.parametrize(
    ("atmosphere_a_sigma_d", "expected_sites"),
    [
        (False, ("atmosphere_rotated", "sigma_log_p")),
        (
            True,
            ("atmosphere_rotated", "sigma_log_p", "A", "sigma_d"),
        ),
    ],
)
def test_direct_sigma_is_in_structured_atmosphere_dense_blocks(
    atmosphere_a_sigma_d,
    expected_sites,
):
    args = SimpleNamespace(
        dense_mass=False,
        dense_atmosphere_mass=not atmosphere_a_sigma_d,
        dense_atmosphere_a_sigma_d_mass=atmosphere_a_sigma_d,
        gaussianized_atmosphere=True,
        direct_sigma_log_p=True,
        standardized_parameters=False,
        fix_logg=True,
        fix_nuisance=False,
        fix_a=False,
        fix_sigma_d=False,
        eigen_mode_count=0,
        fixed_sigma_log_p=None,
        fixed_sigma_eigen=None,
    )

    dense_mass, _, sites = build_dense_mass_specification(args)

    assert dense_mass == [expected_sites]
    assert sites == expected_sites


@pytest.mark.parametrize(
    (
        "dense_mass",
        "dense_atmosphere_mass",
        "dense_atmosphere_a_sigma_d_mass",
    ),
    [
        (True, True, False),
        (True, False, True),
        (False, True, True),
    ],
)
def test_dense_mass_modes_are_mutually_exclusive(
    dense_mass,
    dense_atmosphere_mass,
    dense_atmosphere_a_sigma_d_mass,
):
    args = SimpleNamespace(
        dense_mass=dense_mass,
        dense_atmosphere_mass=dense_atmosphere_mass,
        dense_atmosphere_a_sigma_d_mass=(
            dense_atmosphere_a_sigma_d_mass
        ),
        gaussianized_atmosphere=True,
        fix_nuisance=False,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_dense_mass_specification(args)


@pytest.mark.parametrize("fixed_flag", ["fix_nuisance", "fix_a", "fix_sigma_d"])
def test_atmosphere_a_sigma_d_mass_rejects_fixed_nuisance(fixed_flag):
    args = SimpleNamespace(
        dense_mass=False,
        dense_atmosphere_mass=False,
        dense_atmosphere_a_sigma_d_mass=True,
        gaussianized_atmosphere=True,
        fix_nuisance=False,
        fix_a=False,
        fix_sigma_d=False,
    )
    setattr(args, fixed_flag, True)

    with pytest.raises(ValueError, match="cannot be combined"):
        build_dense_mass_specification(args)


def test_explicit_inverse_mass_diagonal_preserves_site_order():
    diagonal = tuple(float(value) for value in range(1, 17))
    args = SimpleNamespace(
        initial_inverse_mass_diagonal=diagonal,
        initial_inverse_mass_sites=(
            "atmosphere_rotated",
            "A",
            "sigma_d",
        ),
    )
    init_values = {
        "atmosphere_rotated": jnp.zeros((8,)),
        "A": jnp.ones((4,)),
        "sigma_d": jnp.full((4,), 0.03),
    }

    metric, mode, sites, sizes = build_initial_inverse_mass_matrix(
        args,
        False,
        (),
        init_values,
    )

    expected_sites = ("atmosphere_rotated", "A", "sigma_d")
    assert set(metric) == {expected_sites}
    np.testing.assert_allclose(np.asarray(metric[expected_sites]), diagonal)
    assert mode == "structured_diagonal"
    assert sites == expected_sites
    assert sizes == (8, 4, 4)


def test_numpyro_keeps_explicit_structured_metric_diagonal():
    sites = ("atmosphere_rotated", "A", "sigma_d")
    diagonal = jnp.linspace(0.01, 0.16, 16)
    init_values = {
        "atmosphere_rotated": jnp.zeros((8,)),
        "A": jnp.full((4,), 1.1),
        "sigma_d": jnp.full((4,), 0.03),
    }

    def model():
        numpyro.sample(
            "atmosphere_rotated",
            dist.Normal(0.0, 1.0).expand((8,)),
        )
        numpyro.sample(
            "A",
            dist.Uniform(1.0, 1.2).expand((4,)),
        )
        numpyro.sample(
            "sigma_d",
            dist.LogNormal(jnp.log(0.03), 1.0).expand((4,)),
        )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init_values),
        dense_mass=False,
        inverse_mass_matrix={sites: diagonal},
        adapt_mass_matrix=False,
    )
    state = kernel.init(
        jax.random.key(4),
        num_warmup=5,
        model_args=(),
        model_kwargs={},
    )

    stored_metric = state.adapt_state.inverse_mass_matrix
    assert set(stored_metric) == {sites}
    assert stored_metric[sites].shape == (16,)
    np.testing.assert_allclose(np.asarray(stored_metric[sites]), np.asarray(diagonal))


def test_numpyro_full_dense_metric_includes_direct_sigma_coordinate():
    init_values = {
        "atmosphere_rotated": jnp.zeros((7,)),
        "sigma_log_p": jnp.asarray(0.2),
        "A": jnp.full((4,), 1.1),
    }

    def model():
        numpyro.sample(
            "atmosphere_rotated",
            dist.Normal(0.0, 1.0).expand((7,)).to_event(1),
        )
        numpyro.sample("sigma_log_p", dist.HalfNormal(0.3))
        numpyro.sample(
            "A",
            dist.Uniform(1.0, 1.2).expand((4,)).to_event(1),
        )

    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init_values),
        dense_mass=True,
    )
    state = kernel.init(
        jax.random.key(9),
        num_warmup=5,
        model_args=(),
        model_kwargs={},
    )

    sites = ("A", "atmosphere_rotated", "sigma_log_p")
    assert set(state.adapt_state.inverse_mass_matrix) == {sites}
    assert state.adapt_state.inverse_mass_matrix[sites].shape == (12, 12)
    np.testing.assert_allclose(
        np.asarray(state.z["sigma_log_p"]),
        np.log(0.2),
        rtol=2.0e-6,
        atol=2.0e-6,
    )


@pytest.mark.parametrize(
    ("diagonal", "sites", "match"),
    [
        ((1.0,) * 15, ("atmosphere_rotated", "A", "sigma_d"), "16 values"),
        (
            (1.0,) * 16,
            ("atmosphere_rotated", "A"),
            "cover every active sample site",
        ),
        (
            (1.0,) * 15 + (0.0,),
            ("atmosphere_rotated", "A", "sigma_d"),
            "finite and positive",
        ),
    ],
)
def test_explicit_inverse_mass_diagonal_rejects_invalid_metric(
    diagonal,
    sites,
    match,
):
    args = SimpleNamespace(
        initial_inverse_mass_diagonal=diagonal,
        initial_inverse_mass_sites=sites,
    )
    init_values = {
        "atmosphere_rotated": jnp.zeros((8,)),
        "A": jnp.ones((4,)),
        "sigma_d": jnp.full((4,), 0.03),
    }

    with pytest.raises(ValueError, match=match):
        build_initial_inverse_mass_matrix(
            args,
            False,
            (),
            init_values,
        )


def test_nuts_tree_depth_can_be_bounded_separately_during_warmup():
    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(
        SimpleNamespace(
            warmup_max_tree_depth=9,
            max_tree_depth=11,
        )
    )

    assert specification == (9, 11)
    assert warmup_depth == 9
    assert sampling_depth == 11

    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(
        SimpleNamespace(
            warmup_max_tree_depth=None,
            max_tree_depth=11,
        )
    )
    assert specification == 11
    assert warmup_depth == sampling_depth == 11


def test_m7_p2_v10_wrapper_selects_fixed_log_w_conditional_target(
    monkeypatch,
):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)
    monkeypatch.setattr(sys, "argv", ["m7_p2_v10_run.py"])

    m7_p2_v10_run.main()

    args = captured["args"]
    assert args.chip_indices == [0, 1, 2, 3]
    assert args.full_data
    assert args.nside == 8
    assert args.fixed_ell_b == 0.4
    assert args.pressure_gp_factorization == "fixed_eigen"
    assert args.fix_logg
    assert args.fix_log_w
    assert not args.fix_nuisance
    assert args.zero_mean_log_w
    assert args.zero_sum_log_w_basis
    assert args.gaussianized_atmosphere
    assert args.dense_atmosphere_a_sigma_d_mass
    assert not args.dense_atmosphere_mass
    assert not args.dense_mass
    assert args.fixed_sigma_log_p is None
    assert args.sigma_log_p_scale == 0.3
    assert args.num_warmup == 250
    assert args.num_samples == 20
    assert args.target_accept_prob == 0.95
    assert args.max_tree_depth == 11
    assert args.warmup_max_tree_depth is None
    assert Path(args.init_from) == (
        ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"
    )
    assert Path(args.out_dir) == ROOT / "results" / "m7" / "p2" / "v10"

    dense_mass, mode, sites = build_dense_mass_specification(args)
    assert dense_mass == [("atmosphere_rotated", "A", "sigma_d")]
    assert mode == "atmosphere_a_sigma_d_block"
    assert sites == ("atmosphere_rotated", "A", "sigma_d")
    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(args)
    assert specification == 11
    assert warmup_depth == sampling_depth == 11


def test_m7_p2_v11_wrapper_uses_fixed_empirical_diagonal_metric(
    monkeypatch,
):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)
    monkeypatch.setattr(sys, "argv", ["m7_p2_v11_run.py"])

    m7_p2_v11_run.main()

    args = captured["args"]
    assert args.chip_indices == [0, 1, 2, 3]
    assert args.full_data
    assert args.nside == 8
    assert args.fixed_ell_b == 0.4
    assert args.pressure_gp_factorization == "fixed_eigen"
    assert args.fix_logg
    assert args.fix_log_w
    assert not args.fix_nuisance
    assert args.zero_mean_log_w
    assert args.zero_sum_log_w_basis
    assert args.gaussianized_atmosphere
    assert not args.dense_atmosphere_a_sigma_d_mass
    assert not args.dense_atmosphere_mass
    assert not args.dense_mass
    assert not args.adapt_mass_matrix
    assert args.initial_inverse_mass_sites == (
        "atmosphere_rotated",
        "A",
        "sigma_d",
    )
    np.testing.assert_allclose(
        np.asarray(args.initial_inverse_mass_diagonal),
        np.asarray(m7_p2_v11_run.EMPIRICAL_INVERSE_MASS_DIAGONAL),
        rtol=0.0,
        atol=0.0,
    )
    assert len(args.initial_inverse_mass_diagonal) == 16
    assert np.all(np.asarray(args.initial_inverse_mass_diagonal) > 0.0)
    assert "n1500_ddof1" in args.initial_inverse_mass_label
    assert args.fixed_sigma_log_p is None
    assert args.sigma_log_p_scale == 0.3
    assert args.num_warmup == 150
    assert args.num_samples == 20
    assert args.target_accept_prob == 0.95
    assert args.max_tree_depth == 11
    assert args.warmup_max_tree_depth is None
    assert Path(args.init_from) == (
        ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"
    )
    assert Path(args.out_dir) == ROOT / "results" / "m7" / "p2" / "v11"

    dense_mass, mode, dense_sites = build_dense_mass_specification(args)
    assert dense_mass is False
    assert mode == "diagonal"
    assert dense_sites == ()
    metric, metric_mode, metric_sites, site_sizes = (
        build_initial_inverse_mass_matrix(
            args,
            dense_mass,
            dense_sites,
            {
                "atmosphere_rotated": jnp.zeros((8,)),
                "A": jnp.ones((4,)),
                "sigma_d": jnp.full((4,), 0.03),
            },
        )
    )
    assert set(metric) == {
        ("atmosphere_rotated", "A", "sigma_d")
    }
    assert metric_mode == "structured_diagonal"
    assert metric_sites == ("atmosphere_rotated", "A", "sigma_d")
    assert site_sizes == (8, 4, 4)
    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(args)
    assert specification == 11
    assert warmup_depth == sampling_depth == 11


def test_m7_p2_v12_wrapper_uses_joint_rotation_and_fresh_diagonal_adaptation(
    monkeypatch,
):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)
    monkeypatch.setattr(sys, "argv", ["m7_p2_v12_run.py"])

    m7_p2_v12_run.main()

    args = captured["args"]
    assert args.chip_indices == [0, 1, 2, 3]
    assert args.full_data
    assert args.nside == 8
    assert args.fixed_ell_b == 0.4
    assert args.pressure_gp_factorization == "fixed_eigen"
    assert args.fix_logg
    assert args.fix_log_w
    assert not args.fix_nuisance
    assert args.zero_mean_log_w
    assert args.zero_sum_log_w_basis
    assert args.gaussianized_atmosphere
    assert args.joint_atmosphere_a_sigma_d
    assert args.atmosphere_rotation_matrix is None
    assert not args.dense_atmosphere_a_sigma_d_mass
    assert not args.dense_atmosphere_mass
    assert not args.dense_mass
    assert args.adapt_mass_matrix
    assert args.initial_inverse_mass_diagonal is None
    assert args.initial_inverse_mass_sites is None
    assert args.fixed_sigma_log_p is None
    assert args.sigma_log_p_scale == 0.3
    assert args.num_warmup == 250
    assert args.num_samples == 20
    assert args.target_accept_prob == 0.95
    assert args.max_tree_depth == 11
    assert args.warmup_max_tree_depth is None
    assert Path(args.init_from) == (
        ROOT / "results" / "m7" / "p2" / "v11" / "samples.npz"
    )
    assert Path(args.out_dir) == ROOT / "results" / "m7" / "p2" / "v12"

    rotation = validate_joint_atmosphere_a_sigma_d_rotation(
        args.joint_atmosphere_a_sigma_d_rotation_matrix,
        chip_count=4,
    )
    assert rotation.shape == (16, 16)
    dense_mass, mode, dense_sites = build_dense_mass_specification(args)
    assert dense_mass is False
    assert mode == "diagonal"
    assert dense_sites == ()
    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(args)
    assert specification == 11
    assert warmup_depth == sampling_depth == 11


@pytest.mark.parametrize(
    ("wrapper", "version", "fixed_site", "free_site"),
    [
        (m7_p2_v13_run, "v13", "A", "sigma_d"),
        (m7_p2_v14_run, "v14", "sigma_d", "A"),
    ],
)
def test_partial_nuisance_isolation_wrappers_change_only_one_v8_site(
    monkeypatch,
    wrapper,
    version,
    fixed_site,
    free_site,
):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)
    monkeypatch.setattr(sys, "argv", [f"m7_p2_{version}_run.py"])
    wrapper.main()
    args = captured["args"]

    assert args.chip_indices == [0, 1, 2, 3]
    assert args.full_data
    assert args.nside == 8
    assert args.fixed_ell_b == 0.4
    assert args.pressure_gp_factorization == "fixed_eigen"
    assert args.fix_logg
    assert args.fix_log_w
    assert not args.fix_nuisance
    assert args.fix_a is (fixed_site == "A")
    assert args.fix_sigma_d is (fixed_site == "sigma_d")
    assert args.gaussianized_atmosphere
    assert not args.joint_atmosphere_a_sigma_d
    assert args.dense_atmosphere_mass
    assert not args.dense_atmosphere_a_sigma_d_mass
    assert not args.dense_mass
    assert args.num_warmup == 200
    assert args.num_samples == 20
    assert args.target_accept_prob == 0.95
    assert args.warmup_max_tree_depth == 9
    assert args.max_tree_depth == 11
    assert Path(args.init_from) == (
        ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"
    )
    assert Path(args.out_dir) == ROOT / "results" / "m7" / "p2" / version

    dense_mass, mode, dense_sites = build_dense_mass_specification(args)
    assert dense_mass == [("atmosphere_rotated",)]
    assert mode == "atmosphere_block"
    assert dense_sites == ("atmosphere_rotated",)
    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(args)
    assert specification == (9, 11)
    assert warmup_depth == 9
    assert sampling_depth == 11

    physical = {
        "T0": 1219.0,
        "alpha": 0.129,
        "logg": 4.86,
        "log_vmr_co": -2.96,
        "log_vmr_h2o": -3.25,
        "log_vmr_ch4": -4.65,
        "log_vmr_hf": -7.08,
        "log_p_cloud": 1.45,
        "sigma_log_p": 0.2,
        "A": jnp.full((4,), 1.05),
        "log_w": jnp.zeros((4, 14)),
        "sigma_d": jnp.full((4,), 0.03),
    }
    fixed = build_fixed_nuisance_values(args, physical)
    assert set(fixed) == {fixed_site, "log_w"}
    rotation = validate_atmosphere_rotation(args.atmosphere_rotation_matrix)
    initial = build_sampling_initial_values(args, physical, fixed, rotation)
    assert set(initial) == {"atmosphere_rotated", free_site}


def test_v15_changes_only_the_v14_mass_structure(monkeypatch):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)

    monkeypatch.setattr(sys, "argv", ["m7_p2_v14_run.py"])
    m7_p2_v14_run.main()
    v14_args = vars(captured["args"]).copy()

    monkeypatch.setattr(sys, "argv", ["m7_p2_v15_run.py"])
    m7_p2_v15_run.main()
    args = captured["args"]
    v15_args = vars(args).copy()

    differences = {
        name
        for name in v14_args
        if v14_args[name] != v15_args[name]
    }
    assert differences == {
        "dense_atmosphere_mass",
        "dense_mass",
        "out_dir",
        "run_label",
    }
    assert args.fix_log_w
    assert args.fix_sigma_d
    assert not args.fix_a
    assert args.dense_mass
    assert not args.dense_atmosphere_mass
    assert args.num_warmup == 200
    assert args.num_samples == 20
    assert args.target_accept_prob == 0.95
    assert args.warmup_max_tree_depth == 9
    assert args.max_tree_depth == 11
    assert Path(args.init_from) == (
        ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"
    )
    assert Path(args.out_dir) == ROOT / "results" / "m7" / "p2" / "v15"

    dense_mass, mode, dense_sites = build_dense_mass_specification(args)
    assert dense_mass is True
    assert mode == "full"
    assert dense_sites == ()
    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(args)
    assert specification == (9, 11)
    assert warmup_depth == 9
    assert sampling_depth == 11

    physical = {
        "T0": 1219.0,
        "alpha": 0.129,
        "logg": 4.86,
        "log_vmr_co": -2.96,
        "log_vmr_h2o": -3.25,
        "log_vmr_ch4": -4.65,
        "log_vmr_hf": -7.08,
        "log_p_cloud": 1.45,
        "sigma_log_p": 0.2,
        "A": jnp.full((4,), 1.05),
        "log_w": jnp.zeros((4, 14)),
        "sigma_d": jnp.full((4,), 0.03),
    }
    fixed = build_fixed_nuisance_values(args, physical)
    assert set(fixed) == {"log_w", "sigma_d"}
    rotation = validate_atmosphere_rotation(args.atmosphere_rotation_matrix)
    initial = build_sampling_initial_values(args, physical, fixed, rotation)
    assert set(initial) == {"atmosphere_rotated", "A"}


def test_v16_changes_only_the_v15_warmup_length(monkeypatch):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)

    monkeypatch.setattr(sys, "argv", ["m7_p2_v15_run.py"])
    m7_p2_v15_run.main()
    v15_args = vars(captured["args"]).copy()

    monkeypatch.setattr(sys, "argv", ["m7_p2_v16_run.py"])
    m7_p2_v16_run.main()
    args = captured["args"]
    v16_args = vars(args).copy()

    differences = {
        name
        for name in v15_args
        if v15_args[name] != v16_args[name]
    }
    assert differences == {"num_warmup", "out_dir", "run_label"}
    assert args.num_warmup == 700
    assert args.num_samples == 20
    assert args.target_accept_prob == 0.95
    assert args.warmup_max_tree_depth == 9
    assert args.max_tree_depth == 11
    assert args.fix_log_w
    assert args.fix_sigma_d
    assert not args.fix_a
    assert args.dense_mass
    assert args.adapt_mass_matrix
    assert args.initial_inverse_mass_diagonal is None
    assert Path(args.init_from) == (
        ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"
    )
    assert Path(args.out_dir) == ROOT / "results" / "m7" / "p2" / "v16"

    dense_mass, mode, dense_sites = build_dense_mass_specification(args)
    assert dense_mass is True
    assert mode == "full"
    assert dense_sites == ()
    specification, warmup_depth, sampling_depth = build_nuts_max_tree_depth(args)
    assert specification == (9, 11)
    assert warmup_depth == 9
    assert sampling_depth == 11


def test_v17_changes_v15_only_by_the_sigma_coordinate(monkeypatch):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)

    monkeypatch.setattr(sys, "argv", ["m7_p2_v15_run.py"])
    m7_p2_v15_run.main()
    v15_args = vars(captured["args"]).copy()

    monkeypatch.setattr(sys, "argv", ["m7_p2_v17_run.py"])
    m7_p2_v17_run.main()
    args = captured["args"]
    v17_args = vars(args).copy()

    differences = {
        name
        for name in v15_args
        if v15_args[name] != v17_args[name]
    }
    assert differences == {
        "atmosphere_rotation_label",
        "atmosphere_rotation_matrix",
        "direct_sigma_log_p",
        "out_dir",
        "run_label",
    }
    assert args.direct_sigma_log_p
    assert args.fixed_sigma_log_p is None
    assert args.sigma_log_p_scale == 0.3
    assert args.fix_log_w
    assert args.fix_sigma_d
    assert not args.fix_a
    assert args.dense_mass
    assert args.num_warmup == 200
    assert args.num_samples == 20
    assert args.target_accept_prob == 0.95
    assert args.warmup_max_tree_depth == 9
    assert args.max_tree_depth == 11
    assert Path(args.init_from) == (
        ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"
    )
    assert Path(args.out_dir) == ROOT / "results" / "m7" / "p2" / "v17"

    dimension = len(GAUSSIANIZED_ATMOSPHERE_NAMES) - 1
    rotation = validate_atmosphere_rotation(
        args.atmosphere_rotation_matrix,
        dimension=dimension,
    )
    physical = {
        "T0": 1219.0,
        "alpha": 0.129,
        "logg": 4.86,
        "log_vmr_co": -2.96,
        "log_vmr_h2o": -3.25,
        "log_vmr_ch4": -4.65,
        "log_vmr_hf": -7.08,
        "log_p_cloud": 1.45,
        "sigma_log_p": 0.2,
        "A": jnp.full((4,), 1.05),
        "log_w": jnp.zeros((4, 14)),
        "sigma_d": jnp.full((4,), 0.03),
    }
    fixed = build_fixed_nuisance_values(args, physical)
    initial = build_sampling_initial_values(args, physical, fixed, rotation)
    assert set(initial) == {"atmosphere_rotated", "sigma_log_p", "A"}
    assert initial["atmosphere_rotated"].shape == (dimension,)
    assert np.shape(initial["sigma_log_p"]) == ()
    np.testing.assert_allclose(np.asarray(initial["sigma_log_p"]), 0.2)
    np.testing.assert_allclose(
        np.asarray(rotation @ initial["atmosphere_rotated"]),
        np.asarray(_to_gaussianized(physical)[:-1]),
        rtol=2.0e-5,
        atol=2.0e-5,
    )


def test_m8_v1_changes_v17_only_to_the_production_schedule(monkeypatch):
    captured = {}

    def capture_args():
        captured["args"] = retrieval.parse_args()

    monkeypatch.setattr(retrieval, "main", capture_args)

    monkeypatch.setattr(sys, "argv", ["m7_p2_v17_run.py"])
    m7_p2_v17_run.main()
    v17_args = vars(captured["args"]).copy()

    monkeypatch.setattr(sys, "argv", ["m8_v1_run.py"])
    m8_v1_run.main()
    args = captured["args"]
    m8_args = vars(args).copy()

    differences = {
        name
        for name in v17_args
        if v17_args[name] != m8_args[name]
    }
    assert differences == {
        "num_samples",
        "num_warmup",
        "out_dir",
        "run_label",
    }
    assert args.run_label == (
        "m8_v1_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_prod"
    )
    assert Path(args.out_dir) == ROOT / "results" / "m8" / "v1"
    assert Path(args.init_from) == (
        ROOT / "results" / "m7" / "p2" / "v6" / "samples.npz"
    )
    assert args.num_warmup == 2000
    assert args.num_samples == 1500
    assert args.num_chains == 1
    assert args.seed == 0
    assert args.target_accept_prob == 0.95
    assert args.warmup_max_tree_depth == 9
    assert args.max_tree_depth == 11
    assert args.adapt_mass_matrix
    assert not args.x64
    assert args.print_summary
    assert args.direct_sigma_log_p
    assert args.dense_mass
    assert args.fix_log_w
    assert args.fix_sigma_d
    assert not args.fix_a
    assert m8_v1_run.ATMOSPHERE_ROTATION == (
        m7_p2_v17_run.ATMOSPHERE_ROTATION
    )

    source = Path(m8_v1_run.__file__).read_text()
    assert "m7_p2_v17_run" not in source
    assert "from doraex.workflows.on_the_fly_pressure_retrieval" in source


@pytest.mark.parametrize(
    ("gaussianized", "fixed_sigma", "joint", "message"),
    [
        (False, None, False, "requires a Gaussianized"),
        (True, 0.2, False, "cannot also fix"),
        (True, None, True, "cannot use joint"),
    ],
)
def test_direct_sigma_initialization_rejects_incompatible_coordinates(
    gaussianized,
    fixed_sigma,
    joint,
    message,
):
    args = SimpleNamespace(
        rotated_atmosphere_parameters=False,
        gaussianized_atmosphere=gaussianized,
        direct_sigma_log_p=True,
        fixed_sigma_log_p=fixed_sigma,
        joint_atmosphere_a_sigma_d=joint,
    )

    with pytest.raises(ValueError, match=message):
        build_sampling_initial_values(args, {}, None)


def test_zero_sum_log_w_base_is_lossless_on_physical_subspace():
    log_w = jnp.arange(56, dtype=jnp.float32).reshape(4, 14) / 100.0
    log_w = log_w - jnp.mean(log_w, axis=1, keepdims=True)

    base = zero_sum_log_w_base(log_w)
    recovered = dist.transforms.ZeroSumTransform(1)(base)

    assert base.shape == (4, 13)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(log_w), atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(jnp.sum(recovered, axis=1)), 0.0, atol=2.0e-6)


def test_gaussianized_initial_values_use_rotated_and_zero_sum_sites():
    rotation = validate_atmosphere_rotation(np.asarray(ATMOSPHERE_ROTATION).ravel())
    physical = {
        "T0": 1219.0,
        "alpha": 0.129,
        "logg": 4.86,
        "log_vmr_co": -2.96,
        "log_vmr_h2o": -3.25,
        "log_vmr_ch4": -4.65,
        "log_vmr_hf": -7.08,
        "log_p_cloud": 1.45,
        "sigma_log_p": 0.2,
        "A": jnp.full((4,), 1.05),
        "log_w": jnp.zeros((4, 14)),
        "sigma_d": jnp.full((4,), 0.03),
    }
    args = SimpleNamespace(
        rotated_atmosphere_parameters=False,
        gaussianized_atmosphere=True,
        atmosphere_rotation_matrix=np.asarray(ATMOSPHERE_ROTATION).ravel(),
        zero_sum_log_w_basis=True,
        zero_mean_log_w=True,
        t0_min=T0_BOUNDS[0],
        t0_max=T0_BOUNDS[1],
        alpha_min=ALPHA_BOUNDS[0],
        alpha_max=ALPHA_BOUNDS[1],
        log_vmr_co_min=LOG_VMR_BOUNDS["log_vmr_co"][0],
        log_vmr_co_max=LOG_VMR_BOUNDS["log_vmr_co"][1],
        log_vmr_h2o_min=LOG_VMR_BOUNDS["log_vmr_h2o"][0],
        log_vmr_h2o_max=LOG_VMR_BOUNDS["log_vmr_h2o"][1],
        log_vmr_ch4_min=LOG_VMR_BOUNDS["log_vmr_ch4"][0],
        log_vmr_ch4_max=LOG_VMR_BOUNDS["log_vmr_ch4"][1],
        log_vmr_hf_min=LOG_VMR_BOUNDS["log_vmr_hf"][0],
        log_vmr_hf_max=LOG_VMR_BOUNDS["log_vmr_hf"][1],
        log_p_cloud_min=LOG_P_CLOUD_BOUNDS[0],
        log_p_cloud_max=LOG_P_CLOUD_BOUNDS[1],
        sigma_log_p_scale=SIGMA_LOG_P_SCALE,
    )

    initial = build_sampling_initial_values(args, physical, None, rotation)

    assert set(initial) == {
        "atmosphere_rotated",
        "A",
        "log_w_base",
        "sigma_d",
    }
    assert initial["atmosphere_rotated"].shape == (8,)
    assert initial["log_w_base"].shape == (4, 13)
    recovered = _to_physical(rotation @ initial["atmosphere_rotated"])
    for name in GAUSSIANIZED_ATMOSPHERE_NAMES:
        np.testing.assert_allclose(
            np.asarray(recovered[name]),
            np.asarray(physical[name]),
            rtol=2.0e-5,
            atol=2.0e-5,
        )

    fixed_nuisance = {
        "A": physical["A"],
        "log_w": physical["log_w"],
        "sigma_d": physical["sigma_d"],
    }
    conditional_initial = build_sampling_initial_values(
        args,
        physical,
        fixed_nuisance,
        rotation,
    )
    assert set(conditional_initial) == {"atmosphere_rotated"}

    fixed_log_w_initial = build_sampling_initial_values(
        args,
        physical,
        {"log_w": physical["log_w"]},
        rotation,
    )
    assert set(fixed_log_w_initial) == {
        "atmosphere_rotated",
        "A",
        "sigma_d",
    }


def test_joint_initial_values_recover_atmosphere_a_and_sigma_d():
    atmosphere_rotation = jnp.eye(len(GAUSSIANIZED_ATMOSPHERE_NAMES))
    joint_rotation = validate_joint_atmosphere_a_sigma_d_rotation(
        np.asarray(m7_p2_v12_run.JOINT_ROTATION).ravel(),
        chip_count=4,
    )
    physical = {
        "T0": 1177.3,
        "alpha": 0.1127,
        "logg": 4.86,
        "log_vmr_co": -3.155,
        "log_vmr_h2o": -3.402,
        "log_vmr_ch4": -4.732,
        "log_vmr_hf": -7.367,
        "log_p_cloud": 1.340,
        "sigma_log_p": 0.475,
        "A": jnp.asarray([1.101, 1.133, 1.148, 1.140]),
        "log_w": jnp.zeros((4, 14)),
        "sigma_d": jnp.asarray([0.049, 0.038, 0.060, 0.065]),
    }
    args = SimpleNamespace(
        rotated_atmosphere_parameters=False,
        gaussianized_atmosphere=True,
        joint_atmosphere_a_sigma_d=True,
        atmosphere_rotation_matrix=None,
        joint_atmosphere_a_sigma_d_rotation_matrix=np.asarray(
            m7_p2_v12_run.JOINT_ROTATION
        ).ravel(),
        fixed_sigma_log_p=None,
        zero_sum_log_w_basis=True,
        zero_mean_log_w=True,
        t0_min=T0_BOUNDS[0],
        t0_max=T0_BOUNDS[1],
        alpha_min=ALPHA_BOUNDS[0],
        alpha_max=ALPHA_BOUNDS[1],
        log_vmr_co_min=LOG_VMR_BOUNDS["log_vmr_co"][0],
        log_vmr_co_max=LOG_VMR_BOUNDS["log_vmr_co"][1],
        log_vmr_h2o_min=LOG_VMR_BOUNDS["log_vmr_h2o"][0],
        log_vmr_h2o_max=LOG_VMR_BOUNDS["log_vmr_h2o"][1],
        log_vmr_ch4_min=LOG_VMR_BOUNDS["log_vmr_ch4"][0],
        log_vmr_ch4_max=LOG_VMR_BOUNDS["log_vmr_ch4"][1],
        log_vmr_hf_min=LOG_VMR_BOUNDS["log_vmr_hf"][0],
        log_vmr_hf_max=LOG_VMR_BOUNDS["log_vmr_hf"][1],
        log_p_cloud_min=LOG_P_CLOUD_BOUNDS[0],
        log_p_cloud_max=LOG_P_CLOUD_BOUNDS[1],
        sigma_log_p_scale=SIGMA_LOG_P_SCALE,
    )

    initial = build_sampling_initial_values(
        args,
        physical,
        {"log_w": physical["log_w"]},
        atmosphere_rotation,
        joint_rotation,
    )

    assert set(initial) == {JOINT_ATMOSPHERE_A_SIGMA_D_SITE}
    joint_gaussianized = joint_rotation @ initial[
        JOINT_ATMOSPHERE_A_SIGMA_D_SITE
    ]
    recovered_atmosphere = _to_physical(joint_gaussianized[:8])
    recovered_a, recovered_sigma_d = gaussianized_nuisance_to_physical(
        joint_gaussianized[8:12],
        joint_gaussianized[12:16],
    )
    for name in GAUSSIANIZED_ATMOSPHERE_NAMES:
        np.testing.assert_allclose(
            np.asarray(recovered_atmosphere[name]),
            np.asarray(physical[name]),
            rtol=3.0e-5,
            atol=3.0e-5,
        )
    np.testing.assert_allclose(
        np.asarray(recovered_a),
        np.asarray(physical["A"]),
        rtol=3.0e-5,
        atol=3.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(recovered_sigma_d),
        np.asarray(physical["sigma_d"]),
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_fixed_log_w_values_are_zero_mean_and_partial():
    physical = {
        "A": jnp.asarray([1.0, 1.1]),
        "log_w": jnp.asarray(
            [
                [0.2, 0.1, -0.1],
                [1.0, 1.2, 0.8],
            ]
        ),
        "sigma_d": jnp.asarray([0.02, 0.03]),
    }
    args = SimpleNamespace(
        fix_nuisance=False,
        fix_log_w=True,
        zero_mean_log_w=True,
    )

    fixed = build_fixed_nuisance_values(args, physical)

    assert set(fixed) == {"log_w"}
    np.testing.assert_allclose(
        np.asarray(jnp.mean(fixed["log_w"], axis=1)),
        0.0,
        atol=2.0e-7,
    )

    args.fix_nuisance = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_fixed_nuisance_values(args, physical)


@pytest.mark.parametrize(
    ("fix_a", "fix_sigma_d", "expected_sites"),
    [
        (True, False, {"A", "log_w"}),
        (False, True, {"log_w", "sigma_d"}),
        (True, True, {"A", "log_w", "sigma_d"}),
    ],
)
def test_individual_nuisance_fixes_select_only_requested_sites(
    fix_a,
    fix_sigma_d,
    expected_sites,
):
    physical = {
        "A": jnp.asarray([1.05, 1.10]),
        "log_w": jnp.asarray([[0.2, -0.1], [0.4, 0.1]]),
        "sigma_d": jnp.asarray([0.02, 0.04]),
    }
    args = SimpleNamespace(
        fix_nuisance=False,
        fix_a=fix_a,
        fix_log_w=True,
        fix_sigma_d=fix_sigma_d,
        zero_mean_log_w=True,
    )

    fixed = build_fixed_nuisance_values(args, physical)

    assert set(fixed) == expected_sites
    np.testing.assert_allclose(
        np.asarray(jnp.mean(fixed["log_w"], axis=1)),
        0.0,
        atol=2.0e-7,
    )


def test_fixed_sigma_gaussianized_initial_values_have_seven_atmosphere_sites():
    dimension = len(GAUSSIANIZED_ATMOSPHERE_NAMES) - 1
    rotation = validate_atmosphere_rotation(
        np.asarray(FIXED_SIGMA_ATMOSPHERE_ROTATION).ravel(),
        dimension=dimension,
    )
    physical = {
        "T0": 1219.0,
        "alpha": 0.129,
        "logg": 4.86,
        "log_vmr_co": -2.96,
        "log_vmr_h2o": -3.25,
        "log_vmr_ch4": -4.65,
        "log_vmr_hf": -7.08,
        "log_p_cloud": 1.45,
        "sigma_log_p": 0.245,
        "A": jnp.full((4,), 1.05),
        "log_w": jnp.zeros((4, 14)),
        "sigma_d": jnp.full((4,), 0.03),
    }
    args = SimpleNamespace(
        rotated_atmosphere_parameters=False,
        gaussianized_atmosphere=True,
        atmosphere_rotation_matrix=np.asarray(
            FIXED_SIGMA_ATMOSPHERE_ROTATION
        ).ravel(),
        fixed_sigma_log_p=0.245,
        zero_sum_log_w_basis=True,
        zero_mean_log_w=True,
        t0_min=T0_BOUNDS[0],
        t0_max=T0_BOUNDS[1],
        alpha_min=ALPHA_BOUNDS[0],
        alpha_max=ALPHA_BOUNDS[1],
        log_vmr_co_min=LOG_VMR_BOUNDS["log_vmr_co"][0],
        log_vmr_co_max=LOG_VMR_BOUNDS["log_vmr_co"][1],
        log_vmr_h2o_min=LOG_VMR_BOUNDS["log_vmr_h2o"][0],
        log_vmr_h2o_max=LOG_VMR_BOUNDS["log_vmr_h2o"][1],
        log_vmr_ch4_min=LOG_VMR_BOUNDS["log_vmr_ch4"][0],
        log_vmr_ch4_max=LOG_VMR_BOUNDS["log_vmr_ch4"][1],
        log_vmr_hf_min=LOG_VMR_BOUNDS["log_vmr_hf"][0],
        log_vmr_hf_max=LOG_VMR_BOUNDS["log_vmr_hf"][1],
        log_p_cloud_min=LOG_P_CLOUD_BOUNDS[0],
        log_p_cloud_max=LOG_P_CLOUD_BOUNDS[1],
        sigma_log_p_scale=SIGMA_LOG_P_SCALE,
    )

    initial = build_sampling_initial_values(args, physical, None, rotation)

    assert initial["atmosphere_rotated"].shape == (dimension,)
    recovered_gaussianized = rotation @ initial["atmosphere_rotated"]
    expected_gaussianized = _to_gaussianized(physical)[:-1]
    np.testing.assert_allclose(
        np.asarray(recovered_gaussianized),
        np.asarray(expected_gaussianized),
        rtol=2.0e-5,
        atol=2.0e-5,
    )


@pytest.mark.parametrize(
    (
        "fixed_sigma_log_p",
        "direct_sigma_log_p",
        "rotation_values",
        "atmosphere_dimension",
        "use_fixed_pressure_gp_eigen",
        "fixed_log_w",
        "joint_coordinates",
        "fixed_extra",
    ),
    [
        (None, False, ATMOSPHERE_ROTATION, 8, False, False, False, None),
        (0.245, False, FIXED_SIGMA_ATMOSPHERE_ROTATION, 7, False, False, False, None),
        (None, False, ATMOSPHERE_ROTATION, 8, True, False, False, None),
        (None, False, ATMOSPHERE_ROTATION, 8, True, True, False, None),
        (None, False, ATMOSPHERE_ROTATION, 8, True, True, True, None),
        (None, False, ATMOSPHERE_ROTATION, 8, True, True, False, "A"),
        (None, False, ATMOSPHERE_ROTATION, 8, True, True, False, "sigma_d"),
        (
            None,
            True,
            FIXED_SIGMA_ATMOSPHERE_ROTATION,
            7,
            True,
            True,
            False,
            "sigma_d",
        ),
    ],
)
def test_model_trace_exposes_reparameterized_base_sites(
    monkeypatch,
    fixed_sigma_log_p,
    direct_sigma_log_p,
    rotation_values,
    atmosphere_dimension,
    use_fixed_pressure_gp_eigen,
    fixed_log_w,
    joint_coordinates,
    fixed_extra,
):
    def fake_profile_operator(
        theta,
        phi,
        vrot,
        inclination,
        u1,
        u2,
        obs_times,
        period,
        wavelengths,
        base_profile,
        contrast_profile,
        weights,
    ):
        del (
            phi,
            vrot,
            inclination,
            u1,
            u2,
            period,
            base_profile,
            contrast_profile,
            weights,
        )
        length = len(obs_times) * len(wavelengths)
        baseline = jnp.ones((length,))
        contrast = jnp.full((length, len(theta)), 0.01)
        return baseline, contrast

    monkeypatch.setattr(
        retrieval,
        "linear_profile_operator_from_times",
        fake_profile_operator,
    )
    rotation = validate_atmosphere_rotation(
        np.asarray(rotation_values).ravel(),
        dimension=atmosphere_dimension,
    )
    n_phase = 3
    n_wavelength = 4
    distance_matrix = jnp.asarray([[0.0, 1.0], [1.0, 0.0]])
    pressure_gp_decomposition = (
        build_fixed_pressure_gp_eigendecomposition(
            distance_matrix,
            0.4,
        )
        if use_fixed_pressure_gp_eigen
        else None
    )

    fixed_nuisance_values = {}
    if fixed_log_w:
        fixed_nuisance_values["log_w"] = jnp.asarray(
            [[0.2, -0.1, 0.05]]
        )
    if fixed_extra == "A":
        fixed_nuisance_values["A"] = jnp.asarray([1.1])
    elif fixed_extra == "sigma_d":
        fixed_nuisance_values["sigma_d"] = jnp.asarray([0.03])

    def model():
        return on_the_fly_pressure_model(
            data=jnp.zeros((1, n_phase, n_wavelength)),
            wavelengths=[jnp.linspace(2.2, 2.3, n_wavelength)],
            obs_times=jnp.arange(n_phase, dtype=jnp.float32),
            theta=jnp.asarray([0.5, 1.5]),
            phi=jnp.asarray([0.0, 2.0]),
            distance_matrix=distance_matrix,
            response_functions=[
                lambda *args: (
                    jnp.ones((n_wavelength,)),
                    jnp.ones((n_wavelength,)),
                )
            ],
            fixed_period=4.83,
            fixed_cosi=0.485,
            fixed_v=31.2,
            fixed_q1=0.81,
            fixed_q2=0.59,
            fix_logg=True,
            fixed_logg=4.86,
            logg_prior_mean=4.86,
            logg_prior_sigma=0.09,
            logg_bounds=(4.59, 5.13),
            t0_bounds=T0_BOUNDS,
            alpha_bounds=ALPHA_BOUNDS,
            log_vmr_bounds=LOG_VMR_BOUNDS,
            log_p_cloud_bounds=LOG_P_CLOUD_BOUNDS,
            sigma_log_p_scale=SIGMA_LOG_P_SCALE,
            standardized_parameters=False,
            gaussianized_atmosphere=True,
            atmosphere_rotation=rotation,
            parameter_centers={},
            parameter_scales={"rotated_atmosphere": False},
            fixed_ell_b=0.4,
            zero_mean_pressure_map=True,
            log_w_scale=0.1,
            zero_mean_log_w=True,
            zero_sum_log_w_basis=True,
            eigen_mode_count=None,
            eigen_sigma_scale=1.0,
            eigen_sigma_center=jnp.asarray(1.0),
            eigen_sigma_log_raw_scale=jnp.asarray(0.5),
            fixed_sigma_eigen=None,
            eigen_fixed_ell=0.4,
            fixed_nuisance_values=(
                fixed_nuisance_values if fixed_nuisance_values else None
            ),
            gp_jitter=0.5e-6,
            noise_jitter=1.0e-6,
            fixed_sigma_log_p=fixed_sigma_log_p,
            direct_sigma_log_p=direct_sigma_log_p,
            pressure_gp_eigenvalues=(
                None
                if pressure_gp_decomposition is None
                else pressure_gp_decomposition["eigenvalues"]
            ),
            pressure_gp_pixel_eigenvectors=(
                None
                if pressure_gp_decomposition is None
                else pressure_gp_decomposition["pixel_eigenvectors"]
            ),
            joint_atmosphere_a_sigma_d=joint_coordinates,
            joint_atmosphere_a_sigma_d_rotation=(
                jnp.eye(10) if joint_coordinates else None
            ),
        )

    trace = handlers.trace(handlers.seed(model, jax.random.key(0))).get_trace()

    if joint_coordinates:
        assert "atmosphere_rotated" not in trace
        assert trace[JOINT_ATMOSPHERE_A_SIGMA_D_SITE]["type"] == "sample"
        assert trace[JOINT_ATMOSPHERE_A_SIGMA_D_SITE]["value"].shape == (10,)
        assert trace["atmosphere_a_sigma_d_gaussianized"]["type"] == (
            "deterministic"
        )
        assert trace["atmosphere_a_sigma_d_gaussianized"]["value"].shape == (
            10,
        )
        assert trace["A"]["type"] == "deterministic"
        assert trace["sigma_d"]["type"] == "deterministic"
    else:
        assert trace["atmosphere_rotated"]["type"] == "sample"
        assert trace["atmosphere_rotated"]["value"].shape == (
            atmosphere_dimension,
        )
    assert trace["atmosphere_gaussianized"]["type"] == "deterministic"
    assert trace["atmosphere_gaussianized"]["value"].shape == (
        atmosphere_dimension,
    )
    if fixed_sigma_log_p is not None:
        np.testing.assert_allclose(
            np.asarray(trace["sigma_log_p"]["value"]),
            fixed_sigma_log_p,
        )
    elif direct_sigma_log_p:
        assert trace["sigma_log_p"]["type"] == "sample"
        assert isinstance(trace["sigma_log_p"]["fn"], dist.HalfNormal)
        np.testing.assert_allclose(
            np.asarray(trace["sigma_log_p"]["fn"].scale),
            SIGMA_LOG_P_SCALE,
        )
    else:
        assert trace["sigma_log_p"]["type"] == "deterministic"
    if fixed_log_w:
        assert "log_w_base" not in trace
        if not joint_coordinates:
            assert trace["A"]["type"] == (
                "deterministic" if fixed_extra == "A" else "sample"
            )
            assert trace["sigma_d"]["type"] == (
                "deterministic" if fixed_extra == "sigma_d" else "sample"
            )
    else:
        assert trace["log_w_base"]["type"] == "sample"
        assert trace["log_w_base"]["value"].shape == (1, n_phase - 1)
    assert trace["log_w"]["type"] == "deterministic"
    np.testing.assert_allclose(
        np.asarray(jnp.sum(trace["log_w"]["value"], axis=1)),
        0.0,
        atol=2.0e-6,
    )
    assert trace["obs"]["value"].shape == (n_phase * n_wavelength,)
