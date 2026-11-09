"""Exact raw-intensity pressure target and local geometry for M9 v3 pilots.

The physical priors and disk normalization match M9 v2. The optional chip
normalization integral removes A from the sampled coordinates, preserving its
Uniform(1, 1.2) prior. No pressure Taylor approximation enters the target.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
import numpy as np

from doraex.operators.design_matrix import (
    integrate_pixel_profiles_from_times,
    local_pixel_pressure_operator_from_times,
)
from doraex.workflows.on_the_fly_pressure_retrieval import (
    A_PRIOR_BOUNDS,
    SIGMA_D_LOGNORMAL_MEDIAN,
    SIGMA_D_LOGNORMAL_SCALE,
    pressure_map_from_whitened_latent,
    whitened_latent_from_pressure_map,
)


ATMOSPHERE_NAMES = (
    "T0", "alpha", "log_vmr_co", "log_vmr_h2o", "log_vmr_ch4",
    "log_vmr_hf", "log_p_cloud",
)
LOWER = np.array([1000., .05, -6., -6., -6., -10., -2.])
UPPER = np.array([1700., .20, -1., -1., -1., -5., 2.])


@dataclass
class ExactPressureProblem:
    """A flat unconstrained target with efficient per-pixel map Jacobians.

    Coordinates are seven Gaussianized bounded atmosphere values, log GP
    amplitude, optional chip A logits, log noise, and standard-normal GP modes.
    All angles, grids, phase weights, masks, and opacities are fixed constants.
    """

    chip_data: list
    responses: list
    profile_wavelengths: list
    geometry: object
    eigenvalues: object
    eigenvectors: object
    fixed_state: dict
    marginalized: bool = False
    quadrature_order: int = 64
    gp_jitter: float = 5e-7
    noise_jitter: float = 1e-6
    sigma_log_p_scale: float = .3

    def __post_init__(self):
        self.nchip = len(self.chip_data)
        self.nmap = int(self.eigenvalues.shape[0])
        self.a_slice = slice(8, 8 + (0 if self.marginalized else self.nchip))
        self.noise_slice = slice(self.a_slice.stop, self.a_slice.stop + self.nchip)
        self.map_slice = slice(self.noise_slice.stop, self.noise_slice.stop + self.nmap)
        self.dimension = self.map_slice.stop
        self.indices = []
        self.observed = []
        source_chips = list(np.asarray(self.fixed_state["chip_indices"], dtype=int))
        self.source_positions = np.asarray(
            [source_chips.index(chip.chip_index) for chip in self.chip_data], dtype=int,
        )
        self.integration_args = []
        self.weights = []
        for chip, source_position in zip(self.chip_data, self.source_positions):
            if not np.all(np.isfinite(chip.flux)):
                raise ValueError("Non-finite observed flux is not supported.")
            indices = np.flatnonzero(np.asarray(chip.flux).reshape(-1) != 0.)
            if not indices.size:
                raise ValueError("Each chip must have valid observations.")
            self.indices.append(jnp.asarray(indices))
            self.observed.append(jnp.asarray(np.asarray(chip.flux).reshape(-1)[indices]))
            self.integration_args.append((
                self.geometry.theta, self.geometry.phi, jnp.asarray(self.fixed_state["v"]),
                jnp.arccos(jnp.asarray(self.fixed_state["cosi"])),
                jnp.asarray(self.fixed_state["u1"]), jnp.asarray(self.fixed_state["u2"]),
                jnp.asarray(chip.obs_times), jnp.asarray(self.fixed_state["P"]),
                jnp.asarray(chip.wavelengths),
            ))
            log_w = np.asarray(self.fixed_state["log_w"])[source_position]
            self.weights.append(jnp.exp(jnp.asarray(log_w - log_w.mean())))
        self.raw_functions = [jax.jit(lambda x, c=c: self.raw_prediction(x, c)) for c in range(self.nchip)]
        self.potential = jax.jit(self.negative_log_density)
        self.value_and_grad = jax.jit(jax.value_and_grad(self.negative_log_density))
        self._map_jacobians = [jax.jit(lambda x, c=c: self.raw_map_jacobian(x, c)) for c in range(self.nchip)]
        # Reuse one compiled JVP per chip for the eight shared directions.
        self._shared_jvps = [jax.jit(lambda x, d, c=c: jax.jvp(
            lambda shared: self.raw_prediction(x.at[:8].set(shared), c),
            (x[:8],), (d,),
        )[1]) for c in range(self.nchip)]

    def physical(self, position):
        """Decode coordinates without evaluating spectra."""
        position = jnp.asarray(position)
        values = jnp.asarray(LOWER) + jnp.asarray(UPPER - LOWER) * ndtr(position[:7])
        state = {name: values[i] for i, name in enumerate(ATMOSPHERE_NAMES)}
        state["sigma_log_p"] = jnp.exp(position[7])
        state["sigma_d"] = jnp.exp(position[self.noise_slice])
        if not self.marginalized:
            low, high = A_PRIOR_BOUNDS
            state["A"] = low + (high - low) * jax.nn.sigmoid(position[self.a_slice])
        state["delta_log_p"] = pressure_map_from_whitened_latent(
            position[self.map_slice], state["sigma_log_p"], self.eigenvalues,
            self.eigenvectors, self.gp_jitter,
        )
        return state

    def encode(self, state):
        """Project a coherent physical seed into the active GP basis."""
        atmosphere = np.asarray([state[name] for name in ATMOSPHERE_NAMES], dtype=float)
        fractions = (atmosphere - LOWER) / (UPPER - LOWER)
        if np.any((fractions <= 0) | (fractions >= 1)):
            raise ValueError("The atmospheric seed must lie strictly inside its prior bounds.")
        sigma = float(state["sigma_log_p"])
        if sigma <= 0:
            raise ValueError("The GP amplitude must be positive.")
        pressure_map = np.asarray(state["delta_log_p"], dtype=float)
        if pressure_map.shape != self.geometry.theta.shape or abs(pressure_map.mean()) > 1e-6:
            raise ValueError("The seed map must use the active zero-mean pixel grid.")
        latent = whitened_latent_from_pressure_map(
            jnp.asarray(pressure_map), sigma, self.eigenvalues, self.eigenvectors, self.gp_jitter,
        )
        coordinates = [ndtri(jnp.asarray(fractions)), jnp.asarray([np.log(sigma)])]
        if not self.marginalized:
            a = np.asarray(state["A"])[self.source_positions]
            frac = (a - A_PRIOR_BOUNDS[0]) / (A_PRIOR_BOUNDS[1] - A_PRIOR_BOUNDS[0])
            if np.any((frac <= 0) | (frac >= 1)):
                raise ValueError("The A seed must lie strictly inside its prior bounds.")
            coordinates.append(jnp.asarray(np.log(frac) - np.log1p(-frac)))
        coordinates += [jnp.log(jnp.asarray(state["sigma_d"])[self.source_positions]), latent]
        position = jnp.concatenate(coordinates)
        if not np.all(np.isfinite(position)):
            raise ValueError("The seed coordinates must be finite.")
        reconstructed = np.asarray(self.physical(position)["delta_log_p"])
        if np.max(np.abs(reconstructed - pressure_map)) > 2e-5:
            raise ValueError("The seed map is not represented by the active GP basis.")
        return position

    def _profiles(self, position, chip):
        state = self.physical(position)
        pressures = jnp.concatenate((jnp.atleast_1d(state["log_p_cloud"]),
                                     state["log_p_cloud"] + state["delta_log_p"]))
        return state, pressures, lambda p: self.responses[chip](
            *(state[name] for name in ATMOSPHERE_NAMES[:6]),
            jnp.asarray(self.fixed_state["logg"]), p,
        )

    def _uniform_mean(self, profiles, chip):
        uniform = integrate_pixel_profiles_from_times(
            *self.integration_args[chip], jnp.broadcast_to(profiles[0], profiles[1:].shape),
            weights=self.weights[chip], rest_wavelengths=self.profile_wavelengths[chip],
        )
        return jnp.mean(uniform)

    def raw_prediction(self, position, chip):
        """Return valid disk-spectrum entries with A removed from the denominator."""
        _, pressures, response = self._profiles(position, chip)
        profiles = response(pressures)
        disk = integrate_pixel_profiles_from_times(
            *self.integration_args[chip], profiles[1:], weights=self.weights[chip],
            rest_wavelengths=self.profile_wavelengths[chip],
        )
        return (disk / self._uniform_mean(profiles, chip)).reshape(-1)[self.indices[chip]]

    def raw_map_jacobian(self, position, chip):
        """Compute all map columns from one pressure JVP and linear integration."""
        state, pressures, response = self._profiles(position, chip)
        profiles, derivatives = jax.jvp(response, (pressures,), (jnp.ones_like(pressures),))
        _, pixel_jacobian = local_pixel_pressure_operator_from_times(
            *self.integration_args[chip], profiles[1:], derivatives[1:],
            weights=self.weights[chip], rest_wavelengths=self.profile_wavelengths[chip],
        )
        factor = self.eigenvectors * jnp.sqrt(state["sigma_log_p"]**2 * self.eigenvalues + self.gp_jitter)[None, :]
        return ((pixel_jacobian / self._uniform_mean(profiles, chip)) @ factor)[self.indices[chip]]

    def raw_linear_initial_position(self, position, normalizations=None):
        """Solve the zero-map raw linear conditional mean for initialization.

        All shared coordinates, noise scales, and the GP amplitude stay fixed.
        ``normalizations`` contains physical A values in the active chip order;
        it is required for a marginalized-A problem and otherwise defaults to
        the decoded A values. This fixed-A surrogate proposes an initial map;
        acceptance must use the exact target, including the A integral when
        applicable. No linear approximation is added to that target.

        Normal equations are accumulated one chip at a time, with the standard
        normal latent prior contributing the identity precision. Return the
        proposed position and JSON-serializable conditional-solve diagnostics.
        """
        position = jnp.asarray(position)
        if position.shape != (self.dimension,) or not np.all(np.isfinite(position)):
            raise ValueError("The raw-linear initial position must be a finite active coordinate vector.")
        zero = position.at[self.map_slice].set(0.)
        state = self.physical(zero)
        if normalizations is None:
            if self.marginalized:
                raise ValueError("The marginalized raw-linear initializer requires physical A values.")
            normalizations = state["A"]
        normalizations = np.asarray(normalizations, dtype=float)
        if normalizations.shape != (self.nchip,) or not np.all(np.isfinite(normalizations)) or np.any(normalizations <= 0.):
            raise ValueError("Physical A values must be a finite positive vector in active chip order.")
        precision = np.eye(self.nmap, dtype=np.float64)
        rhs = np.zeros(self.nmap, dtype=np.float64)
        baseline_chi_squared = 0.
        for chip, a in enumerate(normalizations):
            baseline = self.raw_functions[chip](zero) / a
            jacobian = self._map_jacobians[chip](zero) / a
            variance = state["sigma_d"][chip]**2 + self.noise_jitter
            residual = self.observed[chip] - baseline
            precision += np.asarray(jacobian.T @ jacobian / variance, dtype=np.float64)
            rhs += np.asarray(jacobian.T @ residual / variance, dtype=np.float64)
            baseline_chi_squared += float(jnp.sum(residual**2 / variance))
        if not np.all(np.isfinite(precision)) or not np.all(np.isfinite(rhs)) or not np.isfinite(baseline_chi_squared):
            raise FloatingPointError("The raw-linear conditional normal equations are non-finite.")
        precision = .5 * (precision + precision.T)
        latent = np.linalg.solve(precision, rhs)
        if not np.all(np.isfinite(latent)):
            raise FloatingPointError("The raw-linear conditional solution is non-finite.")
        candidate = zero.at[self.map_slice].set(jnp.asarray(latent, dtype=zero.dtype))
        pressure_map = np.asarray(self.physical(candidate)["delta_log_p"])
        diagnostics = {
            "linearization": "zero_map_raw_intensity_fixed_shared_parameters_and_A",
            "normalizations": normalizations.tolist(),
            "chip_indices": [int(chip.chip_index) for chip in self.chip_data],
            "map_dimension": self.nmap,
            "solve_relative_residual": float(np.linalg.norm(precision @ latent - rhs) / max(np.linalg.norm(rhs), 1e-30)),
            "baseline_chi_squared": baseline_chi_squared,
            "linear_conditional_potential_twice": float(baseline_chi_squared - 2 * rhs @ latent + latent @ precision @ latent),
            "latent": latent.tolist(),
            "delta_log_p": pressure_map.tolist(),
            "latent_norm": float(np.linalg.norm(latent)),
            "delta_log_p_rms": float(np.sqrt(np.mean(pressure_map**2))),
            "interpretation": "A fixed-A linear initialization proposal; exact objective selection is required.",
        }
        return candidate, diagnostics

    def prior_potential(self, position):
        """Negative log prior in these coordinates, including transform Jacobians."""
        value = .5 * jnp.sum(position[:7]**2) + .5 * jnp.sum(position[self.map_slice]**2)
        value += .5 * (jnp.exp(position[7]) / self.sigma_log_p_scale)**2 - position[7]
        log_noise = position[self.noise_slice]
        value += .5 * jnp.sum(((log_noise - jnp.log(SIGMA_D_LOGNORMAL_MEDIAN)) / SIGMA_D_LOGNORMAL_SCALE)**2)
        if not self.marginalized:
            logits = position[self.a_slice]
            value += jnp.sum(jax.nn.softplus(logits) + jax.nn.softplus(-logits))
        return value

    def negative_log_density(self, position):
        """Evaluate the full exact target; constants in the prior are omitted."""
        state = self.physical(position)
        value = self.prior_potential(position)
        for chip in range(self.nchip):
            raw = self.raw_functions[chip](position)
            variance = state["sigma_d"][chip]**2 + self.noise_jitter
            if self.marginalized:
                from doraex.inference.normalization_marginalization import marginal_normalization_log_likelihood
                value -= marginal_normalization_log_likelihood(
                    self.observed[chip], raw, variance, quadrature_order=self.quadrature_order,
                )
            else:
                residual = self.observed[chip] - raw / state["A"][chip]
                value += .5 * jnp.sum(residual**2 / variance + jnp.log(2 * jnp.pi * variance))
        return value

    def normalization_moments(self, position):
        """Recover conditional A moments without labelling means as posterior draws."""
        from doraex.inference.normalization_marginalization import conditional_normalization_moments
        state = self.physical(position)
        return [conditional_normalization_moments(
            self.observed[chip], self.raw_functions[chip](position),
            state["sigma_d"][chip]**2 + self.noise_jitter,
            quadrature_order=self.quadrature_order,
        ) for chip in range(self.nchip)]

    def normal_equations(self, position):
        """Return the exact gradient and a positive local Fisher approximation.

        Chip normal matrices are accumulated on device. For the marginalized
        target a bounded-amplitude rank-one correction retains curvature when
        the finite A prior constrains the local amplitude. This is a positive
        approximation, not the exact Hessian. The objective and gradient still
        use the full Uniform-A integral.
        """
        position = jnp.asarray(position)
        _, gradient = self.value_and_grad(position)
        state = self.physical(position)
        prior = np.ones(self.dimension)
        prior[7] = 2 * (float(state["sigma_log_p"]) / self.sigma_log_p_scale)**2
        if not self.marginalized:
            sigmoid = np.asarray(jax.nn.sigmoid(position[self.a_slice]))
            prior[self.a_slice] = 2 * sigmoid * (1 - sigmoid)
        prior[self.noise_slice] = 1 / SIGMA_D_LOGNORMAL_SCALE**2
        curvature = jnp.diag(jnp.asarray(prior))
        for chip in range(self.nchip):
            raw = self.raw_functions[chip](position)
            shared = jnp.stack([self._shared_jvps[chip](position, direction)
                                for direction in jnp.eye(8)], axis=1)
            map_jacobian = self._map_jacobians[chip](position)
            jacobian = jnp.zeros((raw.size, self.dimension), dtype=raw.dtype)
            jacobian = jacobian.at[:, :8].set(shared)
            jacobian = jacobian.at[:, self.map_slice].set(map_jacobian)
            variance = state["sigma_d"][chip]**2 + self.noise_jitter
            if self.marginalized:
                from doraex.inference.normalization_marginalization import conditional_inverse_normalization_moments
                scale, amplitude_variance = conditional_inverse_normalization_moments(
                    self.observed[chip], raw, variance,
                    quadrature_order=self.quadrature_order,
                )
                # Clip the fraction of removable amplitude information to
                # preserve a positive-semidefinite likelihood approximation.
                fraction = jnp.clip(amplitude_variance * jnp.dot(raw, raw) / variance, 0., 1.)
                jacobian = jacobian * scale / jnp.sqrt(variance)
                direction = raw / jnp.linalg.norm(raw)
                cross = jacobian.T @ direction
                curvature += jacobian.T @ jacobian - fraction * jnp.outer(cross, cross)
            else:
                a = state["A"][chip]
                jacobian = jacobian / a
                frac = jax.nn.sigmoid(position[self.a_slice][chip])
                derivative = -raw / a**2 * (A_PRIOR_BOUNDS[1] - A_PRIOR_BOUNDS[0]) * frac * (1 - frac)
                jacobian = jacobian.at[:, self.a_slice.start + chip].set(derivative)
                curvature += jacobian.T @ jacobian / variance
            noise_index = self.noise_slice.start + chip
            curvature = curvature.at[noise_index, noise_index].add(
                2 * raw.size * (state["sigma_d"][chip]**2 / variance)**2)
        return np.asarray(gradient, dtype=float), np.asarray(.5 * (curvature + curvature.T), dtype=float)
