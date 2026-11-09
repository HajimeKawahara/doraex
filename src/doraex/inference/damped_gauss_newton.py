"""Finite, monotone Gauss-Newton refinement of an initialization point."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def damped_gauss_newton(
    initial_position,
    *,
    objective_function: Callable,
    normal_equations_function: Callable,
    max_iterations: int = 8,
    max_trials: int = 8,
    initial_damping: float = 1.0e-3,
    gradient_tolerance: float = 1.0e-6,
    step_tolerance: float = 1.0e-8,
    max_step_norm: float | None = None,
):
    """Refine a point using exact-objective acceptance and local GN steps.

    ``objective_function(x)`` returns the scalar exact objective to minimize.
    ``normal_equations_function(x)`` returns its gradient and a symmetric
    positive-semidefinite Gauss-Newton curvature approximation. For an
    objective ``0.5 * r.T @ r`` these are ``J.T @ r`` and ``J.T @ J``.
    The caller must include any priors, coordinate Jacobians, and other
    objective terms in the gradient and supply an appropriate positive
    curvature approximation. The callback can accumulate normal equations
    over data batches without retaining the full residual Jacobian.

    Damping is relative to the curvature diagonal. Rejected trials increase
    damping and halve the step scale; accepted steps reduce damping when the
    local approximation is accurate. Only strictly decreasing finite exact
    objectives are accepted. Non-finite trial objectives are rejected, while
    invalid current-point normal equations raise an error.

    The return value is JSON serializable, including the final ``position``
    as a list and all trial diagnostics. ``gradient_tolerance`` reports local
    stationarity only. This is an initialization utility, not a posterior
    sampler or a guarantee of global optimization.
    """

    position = np.array(initial_position, dtype=np.float64, copy=True)
    if position.ndim != 1 or not position.size or not np.all(np.isfinite(position)):
        raise ValueError("Initial position must be a nonempty finite vector.")
    for name, value, minimum in (
        ("max_iterations", max_iterations, 0),
        ("max_trials", max_trials, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name, value in (
        ("initial_damping", initial_damping),
        ("gradient_tolerance", gradient_tolerance),
        ("step_tolerance", step_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if max_step_norm is not None and (
        not np.isfinite(max_step_norm) or max_step_norm <= 0.0
    ):
        raise ValueError("max_step_norm must be finite and positive.")

    objective = float(objective_function(position.copy()))
    if not np.isfinite(objective):
        raise ValueError("Initial exact objective must be finite.")
    initial_objective = objective
    damping = float(initial_damping)
    iterations = []
    accepted_steps = 0
    objective_evaluations = 1
    status = "max_iterations"

    for iteration_index in range(max_iterations):
        gradient, curvature = normal_equations_function(position.copy())
        gradient = np.asarray(gradient, dtype=np.float64)
        curvature = np.asarray(curvature, dtype=np.float64)
        if gradient.shape != position.shape or curvature.shape != (
            position.size,
            position.size,
        ):
            raise ValueError("Normal-equation shapes must match the position.")
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(curvature)):
            raise ValueError("Current-point normal equations must be finite.")
        if not np.allclose(curvature, curvature.T, rtol=1.0e-8, atol=1.0e-12):
            raise ValueError("Gauss-Newton curvature must be symmetric.")
        curvature = 0.5 * (curvature + curvature.T)
        diagonal = np.diag(curvature).copy()
        if np.any(diagonal < 0.0):
            raise ValueError("Gauss-Newton curvature diagonal must be nonnegative.")
        diagonal = np.maximum(diagonal, max(float(np.max(diagonal)), 1.0) * 1.0e-12)
        gradient_norm = float(np.max(np.abs(gradient)))
        record = {
            "iteration": iteration_index,
            "position": position.tolist(),
            "objective_before": objective,
            "objective_after": objective,
            "gradient_inf_norm": gradient_norm,
            "scaled_gradient_inf_norm": float(
                np.max(np.abs(gradient) / np.sqrt(diagonal))
            ),
            "trials": [],
        }
        iterations.append(record)
        if gradient_norm <= gradient_tolerance:
            status = "gradient_tolerance"
            break

        accepted = False
        for trial_index in range(max_trials):
            step_scale = 0.5**trial_index
            trial = {
                "trial": trial_index,
                "damping": damping,
                "step_scale": step_scale,
                "step_norm": None,
                "objective": None,
                "predicted_reduction": None,
                "actual_reduction": None,
                "reduction_ratio": None,
                "accepted": False,
                "reason": "linear_solve_failed",
            }
            record["trials"].append(trial)
            try:
                step = np.linalg.solve(
                    curvature + np.diag(damping * diagonal), -gradient
                )
            except np.linalg.LinAlgError:
                damping = min(damping * 10.0, 1.0e16)
                continue
            step_norm = float(np.linalg.norm(step))
            if not np.isfinite(step_norm):
                trial["reason"] = "nonfinite_step"
                damping = min(damping * 10.0, 1.0e16)
                continue
            if max_step_norm is not None and step_norm > max_step_norm:
                step *= max_step_norm / step_norm
            step *= step_scale
            step_norm = float(np.linalg.norm(step))
            slope = float(gradient @ step)
            predicted_reduction = float(-slope - 0.5 * step @ curvature @ step)
            trial["step_norm"] = step_norm
            if not np.isfinite(predicted_reduction) or slope >= 0.0:
                trial["reason"] = "non_descent_step"
                damping = min(damping * 10.0, 1.0e16)
                continue
            trial["predicted_reduction"] = predicted_reduction
            candidate = position + step
            candidate_objective = float(objective_function(candidate.copy()))
            objective_evaluations += 1
            finite = bool(np.isfinite(candidate_objective))
            reduction = objective - candidate_objective if finite else None
            ratio = (
                reduction / predicted_reduction
                if finite and predicted_reduction > 0.0
                else None
            )
            trial.update(
                objective=candidate_objective if finite else None,
                actual_reduction=reduction,
                reduction_ratio=ratio,
                reason="nonfinite_objective" if not finite else "no_decrease",
            )
            if finite and reduction > 0.0:
                trial.update(accepted=True, reason="exact_objective_decreased")
                position = candidate
                objective = candidate_objective
                record["objective_after"] = objective
                accepted_steps += 1
                accepted = True
                if ratio is not None and ratio > 0.75:
                    damping = max(damping / 3.0, 1.0e-12)
                elif ratio is None or ratio < 0.25:
                    damping = min(damping * 3.0, 1.0e16)
                if step_norm <= step_tolerance * (1.0 + np.linalg.norm(position)):
                    status = "step_tolerance"
                break
            damping = min(damping * 10.0, 1.0e16)
        if not accepted:
            status = "no_acceptable_step"
            break
        if status == "step_tolerance":
            break

    return {
        "position": position.tolist(),
        "initial_objective": initial_objective,
        "objective": objective,
        "status": status,
        "accepted_steps": accepted_steps,
        "objective_evaluations": objective_evaluations,
        "normal_equation_evaluations": len(iterations),
        "final_damping": damping,
        "iterations": iterations,
    }
