"""CPU checks of exact-objective initialization refinement."""

import json

import numpy as np
import pytest

from doraex.inference.damped_gauss_newton import damped_gauss_newton


def test_linear_gaussian_solution_includes_prior_and_streamed_batches():
    design = np.asarray([[2.0, 0.5], [-0.2, 1.0], [1.0, -0.3]])
    data = np.asarray([1.4, -0.8, 0.2])
    precision = np.diag([0.3, 2.0])
    prior_mean = np.asarray([0.1, -0.4])

    def objective(position):
        residual = design @ position - data
        offset = position - prior_mean
        return 0.5 * (residual @ residual + offset @ precision @ offset)

    def normal_equations(position):
        gradient = precision @ (position - prior_mean)
        curvature = precision.copy()
        for row, datum in zip(design, data, strict=True):
            gradient += row * (row @ position - datum)
            curvature += np.outer(row, row)
        return gradient, curvature

    initial = np.asarray([3.0, 2.0])
    result = damped_gauss_newton(
        initial,
        objective_function=objective,
        normal_equations_function=normal_equations,
    )
    expected = np.linalg.solve(
        design.T @ design + precision, design.T @ data + precision @ prior_mean
    )
    np.testing.assert_allclose(result["position"], expected, rtol=0.0, atol=1.0e-6)
    np.testing.assert_array_equal(initial, [3.0, 2.0])
    assert result["status"] == "gradient_tolerance"
    assert result["objective"] < result["initial_objective"]
    json.dumps(result, allow_nan=False)


def test_curved_objective_rejects_overshoot_and_relinearizes():
    linearization_points = []

    def objective(position):
        return 0.5 * (position[0] ** 2 - 1.0) ** 2

    def normal_equations(position):
        linearization_points.append(float(position[0]))
        residual = position[0] ** 2 - 1.0
        derivative = 2.0 * position[0]
        return np.asarray([derivative * residual]), np.asarray([[derivative**2]])

    result = damped_gauss_newton(
        [0.1],
        objective_function=objective,
        normal_equations_function=normal_equations,
        max_iterations=12,
    )
    trials = [trial for iteration in result["iterations"] for trial in iteration["trials"]]
    assert not trials[0]["accepted"]
    assert any(trial["accepted"] for trial in trials)
    accepted = [trial["objective"] for trial in trials if trial["accepted"]]
    assert np.all(np.diff([result["initial_objective"], *accepted]) < 0.0)
    assert len(set(linearization_points)) > 1
    np.testing.assert_allclose(result["position"], [1.0], atol=1.0e-6)
    json.dumps(result, allow_nan=False)


def test_nonfinite_trial_is_rejected_without_polluting_json():
    def objective(position):
        return np.inf if position[0] >= 0.75 else (position[0] - 1.0) ** 2

    result = damped_gauss_newton(
        [0.0],
        objective_function=objective,
        normal_equations_function=lambda position: (
            2.0 * (position - 1.0), np.asarray([[2.0]])
        ),
        max_iterations=1,
    )
    first, second = result["iterations"][0]["trials"][:2]
    assert first["reason"] == "nonfinite_objective"
    assert first["objective"] is None
    assert second["accepted"]
    assert 0.0 < result["position"][0] < 0.75
    json.dumps(result, allow_nan=False)


def test_exact_gradient_includes_terms_outside_gauss_newton_curvature():
    # A nonquadratic prior contributes an exact gradient while the curvature
    # callback deliberately retains only the likelihood curvature.
    def objective(position):
        value = position[0]
        return 0.5 * (value - 2.0) ** 2 + 0.25 * value**4

    result = damped_gauss_newton(
        [0.2],
        objective_function=objective,
        normal_equations_function=lambda position: (
            np.asarray([position[0] - 2.0 + position[0] ** 3]), np.eye(1)
        ),
        max_iterations=30,
    )
    np.testing.assert_allclose(result["position"], [1.0], atol=2.0e-5)
    assert result["objective"] < 0.751
    for iteration in result["iterations"]:
        assert iteration["objective_after"] <= iteration["objective_before"]


def test_failed_trials_preserve_initial_point_and_respect_budget():
    result = damped_gauss_newton(
        [2.0],
        objective_function=lambda position: 1.0,
        normal_equations_function=lambda position: (np.ones(1), np.eye(1)),
        max_trials=3,
    )
    assert result["position"] == [2.0]
    assert result["status"] == "no_acceptable_step"
    assert result["accepted_steps"] == 0
    assert result["objective_evaluations"] == 4
    assert len(result["iterations"][0]["trials"]) == 3


def test_step_bound_limits_displacement():
    result = damped_gauss_newton(
        [0.0, 0.0],
        objective_function=lambda position: 0.5 * np.sum((position - 10.0) ** 2),
        normal_equations_function=lambda position: (position - 10.0, np.eye(2)),
        max_step_norm=0.25,
        max_iterations=1,
    )
    np.testing.assert_allclose(np.linalg.norm(result["position"]), 0.25)
    assert result["status"] == "max_iterations"


@pytest.mark.parametrize(
    "equations, message",
    [
        (([1.0], np.eye(2)), "shapes"),
        (([np.nan, 1.0], np.eye(2)), "finite"),
        (([1.0, 1.0], [[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
    ],
)
def test_invalid_normal_equations_fail_before_trial(equations, message):
    with pytest.raises(ValueError, match=message):
        damped_gauss_newton(
            [0.0, 0.0],
            objective_function=lambda position: 1.0,
            normal_equations_function=lambda position: equations,
        )


def test_stationary_point_does_not_propose_a_trial():
    result = damped_gauss_newton(
        [0.0],
        objective_function=lambda position: 0.5 * position @ position,
        normal_equations_function=lambda position: (position, np.eye(1)),
    )
    assert result["status"] == "gradient_tolerance"
    assert result["objective_evaluations"] == 1
    assert result["iterations"][0]["trials"] == []
