"""Small exact-RT targets checked against the existing NumPyro model."""

from types import SimpleNamespace
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density
import pytest
from scipy.integrate import quad

from doraex.workflows.exact_pressure_investigation import (
    ATMOSPHERE_NAMES,
    ExactPressureProblem,
    LOWER,
    UPPER,
)
from doraex.workflows.on_the_fly_pressure_retrieval import on_the_fly_pressure_model


@pytest.fixture
def toy_problem():
    with jax.enable_x64():
        geometry = SimpleNamespace(
            theta=jnp.asarray([0.4, 1.1, 1.8]),
            phi=jnp.asarray([0.0, 2.2, 4.5]),
        )
        eigenvectors = jnp.asarray([[1., 1.], [-1., 1.], [0., -2.]]) / jnp.sqrt(
            jnp.asarray([2., 6.])
        )
        observed_grid = np.linspace(1000., 1004., 7)
        profile_grid = jnp.linspace(998., 1006., 17)
        times = np.asarray([0.0, 0.7, 1.8])

        def response(t0, alpha, co, h2o, ch4, hf, logg, pressures):
            wavelength = profile_grid - 1002.
            pressure = pressures[:, None]
            strength = .15 + .00005 * (t0 - 1200.) + .05 * alpha
            strength += .004 * (co + h2o + ch4 + hf) + .002 * logg
            return (
                1. + .008 * wavelength + .1 * jnp.exp(.3 * pressure)
                - strength * jnp.exp(.12 * pressure - .5 * wavelength**2)
            )

        chip_data = []
        for chip_index in (3, 1):
            flux = .91 + .007 * np.cos(np.arange(21).reshape(3, 7) + chip_index)
            flux[1, chip_index] = 0.
            chip_data.append(SimpleNamespace(
                chip_index=chip_index, flux=flux, wavelengths=observed_grid,
                obs_times=times,
            ))
        fixed_state = {
            "chip_indices": np.arange(4), "P": 4.83, "cosi": .485,
            "v": 31.2, "u1": 1.062, "u2": -.162, "logg": 4.86,
            "log_w": np.asarray([[0., 0., 0.], [.02, -.03, .01],
                                 [0., 0., 0.], [.03, -.01, -.02]]),
        }
        problem = ExactPressureProblem(
            chip_data=chip_data, responses=[response, response],
            profile_wavelengths=[profile_grid, profile_grid], geometry=geometry,
            eigenvalues=jnp.asarray([1.3, .4]), eigenvectors=eigenvectors,
            fixed_state=fixed_state,
        )
        state = dict(zip(ATMOSPHERE_NAMES, [1220., .13, -3., -3.2, -4.5, -7.1, .4], strict=True))
        state.update(
            sigma_log_p=.24, sigma_d=np.asarray([.05, .04, .06, .03]),
            A=np.asarray([1.05, 1.07, 1.09, 1.12]),
            delta_log_p=np.asarray([.16, -.10, -.06]),
        )
        yield problem, state, problem.encode(state)


def _legacy_density(problem, position):
    """Return the existing target density and trace in physical sample sites."""
    fixed = problem.fixed_state
    state = problem.physical(position)
    kwargs = dict(
        data=jnp.asarray(np.stack([chip.flux for chip in problem.chip_data])),
        wavelengths=[jnp.asarray(chip.wavelengths) for chip in problem.chip_data],
        obs_times=jnp.asarray(problem.chip_data[0].obs_times),
        theta=problem.geometry.theta, phi=problem.geometry.phi,
        distance_matrix=jnp.zeros((3, 3)), response_functions=problem.responses,
        fixed_period=fixed["P"], fixed_cosi=fixed["cosi"], fixed_v=fixed["v"],
        fixed_q1=.81, fixed_q2=.59, fix_logg=True, fixed_logg=fixed["logg"],
        logg_prior_mean=4.86, logg_prior_sigma=.09, logg_bounds=(4.59, 5.13),
        t0_bounds=tuple([LOWER[0], UPPER[0]]),
        alpha_bounds=tuple([LOWER[1], UPPER[1]]),
        log_vmr_bounds={name: (LOWER[i], UPPER[i]) for i, name in enumerate(ATMOSPHERE_NAMES) if 2 <= i <= 5},
        log_p_cloud_bounds=(-2., 2.), sigma_log_p_scale=problem.sigma_log_p_scale,
        standardized_parameters=False, gaussianized_atmosphere=True,
        atmosphere_rotation=jnp.eye(7), parameter_centers={},
        parameter_scales={"rotated_atmosphere": False}, fixed_ell_b=.4,
        zero_mean_pressure_map=True, log_w_scale=.1, zero_mean_log_w=True,
        zero_sum_log_w_basis=True, eigen_mode_count=None, eigen_sigma_scale=1.,
        eigen_sigma_center=jnp.asarray(1.), eigen_sigma_log_raw_scale=jnp.asarray(.5),
        fixed_sigma_eigen=None, eigen_fixed_ell=.4,
        fixed_nuisance_values={"log_w": jnp.asarray(fixed["log_w"])[problem.source_positions]},
        gp_jitter=problem.gp_jitter, noise_jitter=problem.noise_jitter,
        direct_sigma_log_p=True, pressure_gp_eigenvalues=problem.eigenvalues,
        pressure_gp_pixel_eigenvectors=problem.eigenvectors,
        profile_wavelengths=problem.profile_wavelengths,
        observation_indices=np.flatnonzero(np.stack([chip.flux for chip in problem.chip_data]).reshape(-1) != 0.),
        exact_pixel_rt=True,
    )
    parameters = {
        "atmosphere_rotated": position[:7], "sigma_log_p": state["sigma_log_p"],
        "A": state["A"], "sigma_d": state["sigma_d"],
        "pressure_map_latent": position[problem.map_slice],
    }
    return log_density(on_the_fly_pressure_model, (), kwargs, parameters)


def test_encode_decode_preserves_coherent_map_and_selected_chip_state(toy_problem):
    problem, state, position = toy_problem
    decoded = problem.physical(position)
    for name in ATMOSPHERE_NAMES + ("sigma_log_p", "delta_log_p"):
        np.testing.assert_allclose(decoded[name], state[name], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(decoded["A"], state["A"][[3, 1]], rtol=1e-12)
    np.testing.assert_allclose(decoded["sigma_d"], state["sigma_d"][[3, 1]], rtol=1e-12)
    assert problem.dimension == 14
    assert [len(indices) for indices in problem.indices] == [20, 20]


def test_raw_prediction_and_likelihood_match_existing_exact_model(toy_problem):
    problem, _, position = toy_problem
    _, trace = _legacy_density(problem, position)
    state = problem.physical(position)
    predicted = jnp.concatenate([
        problem.raw_prediction(position, chip) / state["A"][chip]
        for chip in range(problem.nchip)
    ])
    np.testing.assert_allclose(trace["obs"]["fn"].base_dist.loc, predicted, rtol=1e-11)
    legacy_likelihood = trace["obs"]["fn"].log_prob(trace["obs"]["value"])
    np.testing.assert_allclose(
        problem.negative_log_density(position) - problem.prior_potential(position),
        -legacy_likelihood, rtol=1e-11, atol=1e-11,
    )


def test_prior_and_coordinate_jacobians_differ_only_by_constant(toy_problem):
    problem, _, initial = toy_problem
    differences = []
    for displacement in (0., .13, -.21):
        position = initial + displacement * jnp.linspace(-1., 1., problem.dimension)
        legacy_log_density, _ = _legacy_density(problem, position)
        logits = position[problem.a_slice]
        transform_log_jacobian = position[7] + jnp.sum(position[problem.noise_slice])
        transform_log_jacobian += jnp.sum(
            jnp.log(.2) - jax.nn.softplus(logits) - jax.nn.softplus(-logits)
        )
        differences.append(float(
            problem.negative_log_density(position) + legacy_log_density + transform_log_jacobian
        ))
    np.testing.assert_allclose(differences, np.full(3, differences[0]), atol=2e-10, rtol=0.)


def test_analytic_map_operator_matches_full_nonlinear_autodiff(toy_problem):
    problem, _, position = toy_problem
    for chip in range(problem.nchip):
        full = jax.jacfwd(lambda x: problem.raw_prediction(x, chip))(position)
        analytic = problem.raw_map_jacobian(position, chip)
        np.testing.assert_allclose(analytic, full[:, problem.map_slice], rtol=1e-10, atol=1e-12)
        assert np.linalg.norm(analytic) > 0.


def test_normal_equations_match_full_jacobian_fisher_and_finite_difference_gradient(toy_problem):
    problem, _, position = toy_problem
    gradient, curvature = problem.normal_equations(position)
    expected = np.asarray(jax.hessian(problem.prior_potential)(position)).copy()
    state = problem.physical(position)
    for chip in range(problem.nchip):
        jacobian = jax.jacfwd(lambda x: problem.raw_prediction(x, chip) / problem.physical(x)["A"][chip])(position)
        variance = state["sigma_d"][chip]**2 + problem.noise_jitter
        expected += np.asarray(jacobian.T @ jacobian / variance)
        expected[problem.noise_slice.start + chip, problem.noise_slice.start + chip] += float(
            2 * len(problem.indices[chip]) * (state["sigma_d"][chip]**2 / variance)**2
        )
    np.testing.assert_allclose(curvature, expected, rtol=1e-10, atol=1e-10)
    assert np.linalg.eigvalsh(curvature).min() > 0.
    step = 1e-5
    differences = []
    for direction in np.eye(problem.dimension):
        differences.append(float((problem.potential(position + step * direction) - problem.potential(position - step * direction)) / (2 * step)))
    np.testing.assert_allclose(gradient, differences, rtol=2e-7, atol=2e-7)


def test_marginalized_target_matches_independent_uniform_a_integration(toy_problem):
    problem, state, _ = toy_problem
    marginalized = ExactPressureProblem(
        chip_data=problem.chip_data, responses=problem.responses,
        profile_wavelengths=problem.profile_wavelengths, geometry=problem.geometry,
        eigenvalues=problem.eigenvalues, eigenvectors=problem.eigenvectors,
        fixed_state=problem.fixed_state, marginalized=True,
    )
    position = marginalized.encode(state)
    expected = float(marginalized.prior_potential(position))
    for chip in range(problem.nchip):
        raw = np.asarray(marginalized.raw_prediction(position, chip))
        observed = np.asarray(marginalized.observed[chip])
        variance = float(marginalized.physical(position)["sigma_d"][chip]**2 + problem.noise_jitter)

        def log_likelihood(a):
            return -.5 * np.sum((observed - raw / a)**2 / variance + np.log(2 * np.pi * variance))

        center = float(np.clip(np.dot(raw, raw) / np.dot(raw, observed), 1., 1.2))
        reference = log_likelihood(center)
        integral = quad(lambda a: np.exp(log_likelihood(a) - reference) / .2, 1., 1.2, epsabs=1e-12, epsrel=1e-12)[0]
        expected -= reference + np.log(integral)
    np.testing.assert_allclose(marginalized.potential(position), expected, rtol=1e-11, atol=1e-10)
    gradient, curvature = marginalized.normal_equations(position)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.eigvalsh(curvature).min() > 0.
    assert marginalized.dimension == problem.dimension - problem.nchip


def test_encode_rejects_nonzero_mean_seed_map(toy_problem):
    problem, state, _ = toy_problem
    with pytest.raises(ValueError, match="zero-mean"):
        problem.encode({**state, "delta_log_p": state["delta_log_p"] + .1})


def test_raw_linear_initial_map_solves_the_zero_map_conditional_posterior(toy_problem):
    problem, _, position = toy_problem
    candidate, diagnostics = problem.raw_linear_initial_position(position)
    zero = position.at[problem.map_slice].set(0.)
    state = problem.physical(zero)
    designs, residuals = [], []
    for chip in range(problem.nchip):
        standard_deviation = jnp.sqrt(state["sigma_d"][chip]**2 + problem.noise_jitter)
        prediction = lambda x: problem.raw_prediction(x, chip) / state["A"][chip]
        designs.append(jax.jacfwd(prediction)(zero)[:, problem.map_slice] / standard_deviation)
        residuals.append((problem.observed[chip] - prediction(zero)) / standard_deviation)
    design = jnp.concatenate(designs)
    residual = jnp.concatenate(residuals)
    linear_potential = lambda latent: .5 * jnp.sum((residual - design @ latent)**2) + .5 * latent @ latent
    latent = candidate[problem.map_slice]
    np.testing.assert_allclose(jax.grad(linear_potential)(latent), 0., atol=1e-11)
    assert linear_potential(latent) < linear_potential(jnp.zeros(problem.nmap))
    np.testing.assert_array_equal(candidate[:problem.map_slice.start], position[:problem.map_slice.start])
    np.testing.assert_allclose(2 * linear_potential(latent), diagnostics["linear_conditional_potential_twice"], atol=1e-11)
    assert diagnostics["solve_relative_residual"] < 1e-12
    other, _ = problem.raw_linear_initial_position(position.at[problem.map_slice].multiply(-7.))
    np.testing.assert_array_equal(other, candidate)


def test_raw_linear_marginal_arm_uses_supplied_A_only_for_the_proposal(toy_problem):
    problem, state, explicit_position = toy_problem
    marginalized = replace(problem, marginalized=True)
    marginalized_position = marginalized.encode(state)
    with pytest.raises(ValueError, match="requires physical A"):
        marginalized.raw_linear_initial_position(marginalized_position)
    normalizations = state["A"][[3, 1]]
    explicit_candidate, _ = problem.raw_linear_initial_position(explicit_position)
    marginal_candidate, diagnostics = marginalized.raw_linear_initial_position(
        marginalized_position, normalizations=normalizations,
    )
    np.testing.assert_allclose(
        marginalized.physical(marginal_candidate)["delta_log_p"],
        problem.physical(explicit_candidate)["delta_log_p"], atol=1e-12,
    )
    np.testing.assert_array_equal(diagnostics["normalizations"], normalizations)
    assert "A" not in marginalized.physical(marginal_candidate)


@pytest.mark.parametrize("normalizations", [[1.1], [1.1, np.nan], [1.1, 0.]])
def test_raw_linear_initial_map_rejects_invalid_fixed_A(toy_problem, normalizations):
    problem, _, position = toy_problem
    with pytest.raises(ValueError, match="finite positive vector"):
        problem.raw_linear_initial_position(position, normalizations=normalizations)


@pytest.mark.parametrize("a_true", [1., 1.1, 1.2])
def test_bounded_marginal_metric_is_positive_at_prior_edges_without_changing_gradient(toy_problem, a_true):
    source, state, source_position = toy_problem
    chip_data = []
    for index, chip in enumerate(source.chip_data):
        flux = np.array(chip.flux, copy=True)
        flux.reshape(-1)[np.asarray(source.indices[index])] = np.asarray(source.raw_prediction(source_position, index)) / a_true
        chip_data.append(SimpleNamespace(chip_index=chip.chip_index, flux=flux,
                                         wavelengths=chip.wavelengths, obs_times=chip.obs_times))
    problem = ExactPressureProblem(
        chip_data, source.responses, source.profile_wavelengths, source.geometry,
        source.eigenvalues, source.eigenvectors, source.fixed_state, marginalized=True,
    )
    position = problem.encode(state)
    potential_before = float(problem.potential(position))
    expected_gradient = jax.grad(problem.negative_log_density)(position)
    gradient, curvature = problem.normal_equations(position)
    assert np.linalg.eigvalsh(curvature).min() > 0.
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-10, atol=1e-10)
    assert float(problem.potential(position)) == potential_before
