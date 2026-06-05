"""Run Milestone 5 on-the-fly atmospheric pressure-map retrievals."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data  # noqa: E402
from doraex.geometry.limb_darkening import kipping_q_to_u  # noqa: E402
from doraex.inference.marginal_likelihood import diagonal_noise_variance  # noqa: E402
from doraex.operators.design_matrix import (  # noqa: E402
    linear_profile_operator_from_times,
)
from doraex.priors.spherical_gp import (  # noqa: E402
    add_diagonal_jitter,
    squared_exponential_covariance,
    zero_mean_covariance_factor,
)
from doraex.spectra.exojax_forward import (  # noqa: E402
    Luhman16BPowerLawColumnModel,
)
from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    build_luhman16b_geometry,
)
from generate_milestone2_t0_alpha_cloud_zeta_grid_profiles import (  # noqa: E402
    YAMA_L16B_EXOMOL_ATMOSPHERE,
    _cia_paths,
    _molecule_paths,
)


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description=(
            "Run a Milestone 5 on-the-fly atmospheric retrieval. The shared "
            "T0, alpha, zeta_vmr, and log10(P_cloud) parameters are sampled, "
            "and d spectrum / d log10(P_cloud) is evaluated with JAX JVP "
            "inside the NumPyro model."
        )
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--chip-indices", type=parse_chips, default=parse_chips("1"))
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "milestone5_on_the_fly_atmosphere_smoke"),
    )
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--nside", type=int, default=2)
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use all phases and wavelengths instead of the reduced smoke subset.",
    )
    parser.add_argument("--smoke-wavelength-step", type=int, default=128)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept-prob", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=int, default=5)
    parser.add_argument("--fixed-period", type=float, default=4.83)
    parser.add_argument("--fixed-cosi", type=float, default=0.485)
    parser.add_argument("--fixed-v", type=float, default=31.2)
    parser.add_argument("--fixed-q1", type=float, default=0.81)
    parser.add_argument("--fixed-q2", type=float, default=0.59)
    parser.add_argument("--t0-min", type=float, default=1000.0)
    parser.add_argument("--t0-max", type=float, default=1700.0)
    parser.add_argument("--init-t0", type=float, default=1219.0)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.20)
    parser.add_argument("--init-alpha", type=float, default=0.129)
    parser.add_argument("--zeta-vmr-min", type=float, default=-0.5)
    parser.add_argument("--zeta-vmr-max", type=float, default=0.5)
    parser.add_argument("--init-zeta-vmr", type=float, default=0.0)
    parser.add_argument("--log-p-cloud-min", type=float, default=-1.0)
    parser.add_argument("--log-p-cloud-max", type=float, default=2.0)
    parser.add_argument("--init-log-p-cloud", type=float, default=1.35)
    parser.add_argument("--sigma-log-p-scale", type=float, default=0.1)
    parser.add_argument("--init-sigma-log-p", type=float, default=0.22)
    parser.add_argument(
        "--standardized-parameters",
        action="store_true",
        help=(
            "Sample standardized raw atmospheric coordinates and expose the "
            "physical parameters as deterministic sites."
        ),
    )
    parser.add_argument(
        "--rotated-atmosphere-parameters",
        action="store_true",
        help=(
            "Sample sheared atmospheric coordinates aligned with the scanned "
            "T0-zeta_vmr and alpha-log_p_cloud posterior ridges."
        ),
    )
    parser.add_argument("--t0-raw-scale", type=float, default=100.0)
    parser.add_argument("--alpha-raw-scale", type=float, default=0.02)
    parser.add_argument("--zeta-vmr-raw-scale", type=float, default=0.2)
    parser.add_argument("--log-p-cloud-raw-scale", type=float, default=0.3)
    parser.add_argument("--sigma-log-p-log-raw-scale", type=float, default=0.5)
    parser.add_argument(
        "--zeta-vmr-per-t0",
        type=float,
        default=1.0e-3,
        help="Ridge slope d(zeta_vmr) / d(T0) used by rotated coordinates.",
    )
    parser.add_argument(
        "--log-p-cloud-per-alpha",
        type=float,
        default=5.0,
        help=(
            "Ridge slope d(log_p_cloud) / d(alpha) used by rotated coordinates."
        ),
    )
    parser.add_argument("--fixed-ell-b", type=float, default=0.3)
    parser.add_argument("--zero-mean-pressure-map", action="store_true", default=True)
    parser.add_argument(
        "--no-zero-mean-pressure-map",
        action="store_false",
        dest="zero_mean_pressure_map",
    )
    parser.add_argument("--log-w-scale", type=float, default=0.1)
    parser.add_argument(
        "--fix-nuisance",
        action="store_true",
        help="Fix A, log_w, and sigma_d to their initial values.",
    )
    parser.add_argument("--gp-jitter", type=float, default=0.5e-6)
    parser.add_argument("--noise-jitter", type=float, default=1.0e-6)
    parser.add_argument(
        "--init-from",
        default=str(
            ROOT
            / "results"
            / "milestone4_on_the_fly_autodiff_full_joint"
            / "mcmc_on_the_fly_autodiff_pressure.npz"
        ),
        help=(
            "Optional previous posterior NPZ used for A, log_w, sigma_d, "
            "log_p_cloud, and sigma_log_p median initial values."
        ),
    )
    parser.add_argument(
        "--no-init-from",
        action="store_const",
        const=None,
        dest="init_from",
        help="Disable previous-posterior initialization.",
    )
    parser.add_argument(
        "--manual-atmosphere-init",
        action="store_true",
        help=(
            "Use --init-* values for atmospheric centers while still using "
            "--init-from for nuisance values."
        ),
    )
    parser.add_argument(
        "--dense-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dense NUTS mass-matrix adaptation.",
    )
    parser.add_argument(
        "--map-init",
        action="store_true",
        help="Use numpyro-inferutils SVI MAP initialization before NUTS.",
    )
    parser.add_argument("--map-init-steps", type=int, default=1000)
    parser.add_argument("--map-init-step-size", type=float, default=1.0e-3)
    parser.add_argument(
        "--preflight-autodiff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compile and evaluate the on-the-fly spectrum/JVP response once "
            "before starting NUTS."
        ),
    )
    parser.add_argument(
        "--x64",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable JAX 64-bit mode. M5 defaults to 32-bit to reduce GPU memory.",
    )
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def build_chip_data(args):
    """Load Luhman 16B chip data for the retrieval."""

    chip_data_list = []
    for chip_index in args.chip_indices:
        chip_data = load_luhman16b_chip(args.data_dir, chip_index=chip_index)
        if not args.full_data:
            chip_data = subset_chip_data(
                chip_data,
                wavelength_step=args.smoke_wavelength_step,
                phase_count=args.smoke_phase_count,
            )
        chip_data_list.append(chip_data)
    return chip_data_list


def _response_function(spectrum_function):
    """Return a function that evaluates spectrum and pressure JVP."""

    def response(t0, alpha, zeta_vmr, log_p_cloud):
        def spectrum_at_pressure(pressure):
            return spectrum_function(t0, alpha, zeta_vmr, pressure)

        return jax.jvp(
            spectrum_at_pressure,
            (log_p_cloud,),
            (jnp.ones_like(log_p_cloud),),
        )

    return response


def build_response_functions(args, chip_data_list):
    """Build on-the-fly ExoJAX spectrum/pressure-response functions."""

    response_functions = []
    for chip_data in chip_data_list:
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=YAMA_L16B_EXOMOL_ATMOSPHERE,
            nx=args.nx,
        )
        response_functions.append(_response_function(model.cloudy_at_parameters))
    return response_functions


def on_the_fly_pressure_model(
    data,
    wavelengths,
    obs_times,
    theta,
    phi,
    distance_matrix,
    response_functions,
    fixed_period,
    fixed_cosi,
    fixed_v,
    fixed_q1,
    fixed_q2,
    t0_bounds,
    alpha_bounds,
    zeta_vmr_bounds,
    log_p_cloud_bounds,
    sigma_log_p_scale,
    standardized_parameters,
    parameter_centers,
    parameter_scales,
    fixed_ell_b,
    zero_mean_pressure_map,
    log_w_scale,
    fixed_nuisance_values,
    gp_jitter,
    noise_jitter,
):
    """On-the-fly pressure-perturbation retrieval model."""

    n_chip = data.shape[0]
    n_phase = data.shape[1]
    if parameter_scales["rotated_atmosphere"]:
        t0_zeta_raw = numpyro.sample("T0_zeta_raw", dist.Normal(0.0, 1.0))
        zeta_vmr_residual_raw = numpyro.sample(
            "zeta_vmr_residual_raw",
            dist.Normal(0.0, 1.0),
        )
        alpha_log_p_raw = numpyro.sample(
            "alpha_log_p_raw",
            dist.Normal(0.0, 1.0),
        )
        log_p_cloud_residual_raw = numpyro.sample(
            "log_p_cloud_residual_raw",
            dist.Normal(0.0, 1.0),
        )
        sigma_log_p_raw = numpyro.sample(
            "sigma_log_p_raw",
            dist.Normal(0.0, 1.0),
        )
        t0_offset = parameter_scales["T0"] * t0_zeta_raw
        alpha_offset = parameter_scales["alpha"] * alpha_log_p_raw
        t0 = numpyro.deterministic("T0", parameter_centers["T0"] + t0_offset)
        alpha = numpyro.deterministic(
            "alpha",
            parameter_centers["alpha"] + alpha_offset,
        )
        zeta_vmr = numpyro.deterministic(
            "zeta_vmr",
            parameter_centers["zeta_vmr"]
            + parameter_scales["zeta_vmr_per_t0"] * t0_offset
            + parameter_scales["zeta_vmr"] * zeta_vmr_residual_raw,
        )
        log_p_cloud = numpyro.deterministic(
            "log_p_cloud",
            parameter_centers["log_p_cloud"]
            + parameter_scales["log_p_cloud_per_alpha"] * alpha_offset
            + parameter_scales["log_p_cloud"] * log_p_cloud_residual_raw,
        )
        sigma_log_p = numpyro.deterministic(
            "sigma_log_p",
            jnp.exp(
                jnp.log(parameter_centers["sigma_log_p"])
                + parameter_scales["sigma_log_p"] * sigma_log_p_raw
            ),
        )
    elif standardized_parameters:
        t0_raw = numpyro.sample("T0_raw", dist.Normal(0.0, 1.0))
        alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0))
        zeta_vmr_raw = numpyro.sample("zeta_vmr_raw", dist.Normal(0.0, 1.0))
        log_p_cloud_raw = numpyro.sample(
            "log_p_cloud_raw",
            dist.Normal(0.0, 1.0),
        )
        sigma_log_p_raw = numpyro.sample(
            "sigma_log_p_raw",
            dist.Normal(0.0, 1.0),
        )
        t0 = numpyro.deterministic(
            "T0",
            parameter_centers["T0"] + parameter_scales["T0"] * t0_raw,
        )
        alpha = numpyro.deterministic(
            "alpha",
            parameter_centers["alpha"] + parameter_scales["alpha"] * alpha_raw,
        )
        zeta_vmr = numpyro.deterministic(
            "zeta_vmr",
            parameter_centers["zeta_vmr"]
            + parameter_scales["zeta_vmr"] * zeta_vmr_raw,
        )
        log_p_cloud = numpyro.deterministic(
            "log_p_cloud",
            parameter_centers["log_p_cloud"]
            + parameter_scales["log_p_cloud"] * log_p_cloud_raw,
        )
        sigma_log_p = numpyro.deterministic(
            "sigma_log_p",
            jnp.exp(
                jnp.log(parameter_centers["sigma_log_p"])
                + parameter_scales["sigma_log_p"] * sigma_log_p_raw
            ),
        )
    else:
        t0 = numpyro.sample("T0", dist.Uniform(t0_bounds[0], t0_bounds[1]))
        alpha = numpyro.sample(
            "alpha",
            dist.Uniform(alpha_bounds[0], alpha_bounds[1]),
        )
        zeta_vmr = numpyro.sample(
            "zeta_vmr",
            dist.Uniform(zeta_vmr_bounds[0], zeta_vmr_bounds[1]),
        )
        log_p_cloud = numpyro.sample(
            "log_p_cloud",
            dist.Uniform(log_p_cloud_bounds[0], log_p_cloud_bounds[1]),
        )
        sigma_log_p = numpyro.sample(
            "sigma_log_p",
            dist.HalfNormal(sigma_log_p_scale),
        )
    cosi = numpyro.deterministic("cosi", jnp.asarray(fixed_cosi))
    vrot = numpyro.deterministic("v", jnp.asarray(fixed_v))
    q1 = numpyro.deterministic("q1", jnp.asarray(fixed_q1))
    q2 = numpyro.deterministic("q2", jnp.asarray(fixed_q2))
    period = numpyro.deterministic("P", jnp.asarray(fixed_period))
    inclination = jnp.arccos(cosi)
    u1, u2 = kipping_q_to_u(q1, q2)
    numpyro.deterministic("u1", u1)
    numpyro.deterministic("u2", u2)

    if fixed_nuisance_values is None:
        normalization_factor = numpyro.sample(
            "A",
            dist.Uniform(1.0, 1.2).expand([n_chip]),
        )
        log_w = numpyro.sample(
            "log_w",
            dist.Normal(0.0, log_w_scale).expand([n_chip, n_phase]),
        )
        sigma_d = numpyro.sample(
            "sigma_d",
            dist.LogNormal(jnp.log(0.03), 1.0).expand([n_chip]),
        )
    else:
        normalization_factor = numpyro.deterministic(
            "A",
            jnp.asarray(fixed_nuisance_values["A"]),
        )
        log_w = numpyro.deterministic(
            "log_w",
            jnp.asarray(fixed_nuisance_values["log_w"]),
        )
        sigma_d = numpyro.deterministic(
            "sigma_d",
            jnp.asarray(fixed_nuisance_values["sigma_d"]),
        )

    baselines = []
    contrast_matrices = []
    noise_variances = []
    for chip_index in range(n_chip):
        base_profile, contrast_profile = response_functions[chip_index](
            t0,
            alpha,
            zeta_vmr,
            log_p_cloud,
        )
        baseline, contrast_matrix = linear_profile_operator_from_times(
            theta,
            phi,
            vrot,
            inclination,
            u1,
            u2,
            obs_times,
            period,
            wavelengths[chip_index],
            base_profile,
            contrast_profile,
            weights=jnp.exp(log_w[chip_index]),
        )
        norm = normalization_factor[chip_index] * jnp.mean(baseline)
        baseline = baseline / norm
        contrast_matrix = contrast_matrix / norm
        baselines.append(baseline)
        contrast_matrices.append(contrast_matrix)
        noise_variances.append(
            diagonal_noise_variance(
                contrast_matrix.shape[0],
                sigma_d[chip_index],
                jitter=noise_jitter,
            )
        )

    baseline = jnp.concatenate(baselines, axis=0)
    contrast_matrix = jnp.concatenate(contrast_matrices, axis=0)
    noise_variance = jnp.concatenate(noise_variances, axis=0)

    numpyro.deterministic("sigma_b", sigma_log_p)
    ell_b = numpyro.deterministic("ell_b", jnp.asarray(fixed_ell_b))
    contrast_covariance = squared_exponential_covariance(
        distance_matrix,
        sigma_log_p,
        ell_b,
    )
    if zero_mean_pressure_map:
        map_factor = zero_mean_covariance_factor(
            contrast_covariance,
            jitter=gp_jitter,
        )
    else:
        contrast_covariance = add_diagonal_jitter(
            contrast_covariance,
            jitter=gp_jitter,
        )
        map_factor = jnp.linalg.cholesky(contrast_covariance)
    covariance_factor = contrast_matrix @ map_factor
    numpyro.sample(
        "obs",
        dist.LowRankMultivariateNormal(
            loc=baseline,
            cov_factor=covariance_factor,
            cov_diag=noise_variance,
        ),
        obs=data.reshape(-1),
    )


def finite(value):
    """Return whether a JAX array is finite."""

    return bool(jnp.all(jnp.isfinite(value)))


def _median_or_default(samples, name, default):
    """Return a posterior median initial value when present."""

    if samples is None or name not in samples:
        return default
    value = np.median(np.asarray(samples[name]), axis=0)
    if np.shape(value) != np.shape(default):
        return default
    return value


def load_initial_values(args, chip_count, phase_count):
    """Build constrained initial values for M5 NUTS."""

    previous = None
    if args.init_from is not None and Path(args.init_from).exists():
        previous = dict(np.load(args.init_from, allow_pickle=False))
    values = {
        "T0": args.init_t0,
        "alpha": args.init_alpha,
        "zeta_vmr": args.init_zeta_vmr,
        "log_p_cloud": float(
            _median_or_default(previous, "log_p_cloud", args.init_log_p_cloud)
        ),
        "sigma_log_p": float(
            _median_or_default(previous, "sigma_log_p", args.init_sigma_log_p)
        ),
        "A": jnp.asarray(
            _median_or_default(previous, "A", np.full((chip_count,), 1.05))
        ),
        "log_w": jnp.asarray(
            _median_or_default(previous, "log_w", np.zeros((chip_count, phase_count)))
        ),
        "sigma_d": jnp.asarray(
            _median_or_default(previous, "sigma_d", np.full((chip_count,), 0.03))
        ),
    }
    return values


def build_parameter_reparameterization(args, physical_init_values):
    """Build centers and scales for standardized atmospheric coordinates."""

    centers = {
        "T0": jnp.asarray(physical_init_values["T0"]),
        "alpha": jnp.asarray(physical_init_values["alpha"]),
        "zeta_vmr": jnp.asarray(physical_init_values["zeta_vmr"]),
        "log_p_cloud": jnp.asarray(physical_init_values["log_p_cloud"]),
        "sigma_log_p": jnp.asarray(physical_init_values["sigma_log_p"]),
    }
    scales = {
        "T0": jnp.asarray(args.t0_raw_scale),
        "alpha": jnp.asarray(args.alpha_raw_scale),
        "zeta_vmr": jnp.asarray(args.zeta_vmr_raw_scale),
        "log_p_cloud": jnp.asarray(args.log_p_cloud_raw_scale),
        "sigma_log_p": jnp.asarray(args.sigma_log_p_log_raw_scale),
        "zeta_vmr_per_t0": jnp.asarray(args.zeta_vmr_per_t0),
        "log_p_cloud_per_alpha": jnp.asarray(args.log_p_cloud_per_alpha),
        "rotated_atmosphere": bool(args.rotated_atmosphere_parameters),
    }
    return centers, scales


def build_sampling_initial_values(args, physical_init_values, fixed_nuisance_values):
    """Build initial values for sample sites seen by NUTS."""

    if args.rotated_atmosphere_parameters:
        init_values = {
            "T0_zeta_raw": jnp.asarray(0.0),
            "zeta_vmr_residual_raw": jnp.asarray(0.0),
            "alpha_log_p_raw": jnp.asarray(0.0),
            "log_p_cloud_residual_raw": jnp.asarray(0.0),
            "sigma_log_p_raw": jnp.asarray(0.0),
        }
        if fixed_nuisance_values is None:
            init_values.update(
                {
                    "A": physical_init_values["A"],
                    "log_w": physical_init_values["log_w"],
                    "sigma_d": physical_init_values["sigma_d"],
                }
            )
        return init_values

    if args.standardized_parameters:
        init_values = {
            "T0_raw": jnp.asarray(0.0),
            "alpha_raw": jnp.asarray(0.0),
            "zeta_vmr_raw": jnp.asarray(0.0),
            "log_p_cloud_raw": jnp.asarray(0.0),
            "sigma_log_p_raw": jnp.asarray(0.0),
        }
        if fixed_nuisance_values is None:
            init_values.update(
                {
                    "A": physical_init_values["A"],
                    "log_w": physical_init_values["log_w"],
                    "sigma_d": physical_init_values["sigma_d"],
                }
            )
        return init_values

    if fixed_nuisance_values is None:
        return dict(physical_init_values)
    return {
        name: value
        for name, value in physical_init_values.items()
        if name not in fixed_nuisance_values
    }


def maybe_find_map_init(args, model, init_values):
    """Optionally refine initial values with numpyro-inferutils."""

    if not args.map_init:
        return init_values
    from numpyro_inferutils import find_map_svi

    return find_map_svi(
        model,
        step_size=args.map_init_step_size,
        num_steps=args.map_init_steps,
        rng_key=jax.random.PRNGKey(args.seed + 1000),
        p_initial=init_values,
        progress_bar=True,
    )


def main():
    """Run the on-the-fly autodiff retrieval."""

    args = parse_args()
    if args.rotated_atmosphere_parameters:
        args.standardized_parameters = True
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    jax.config.update("jax_enable_x64", args.x64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chip_data_list = build_chip_data(args)
    geometry = build_luhman16b_geometry(nside=args.nside)
    flux_shapes = {chip.flux.shape for chip in chip_data_list}
    if len(flux_shapes) != 1:
        raise ValueError(
            "All selected chips must have the same flux shape for joint retrieval; "
            f"got {sorted(flux_shapes)}."
        )
    data = jnp.asarray(np.stack([chip.flux for chip in chip_data_list], axis=0))
    wavelengths = [jnp.asarray(chip.wavelengths) for chip in chip_data_list]
    obs_times = jnp.asarray(chip_data_list[0].obs_times)

    setup_start = time.time()
    response_functions = build_response_functions(args, chip_data_list)
    setup_seconds = time.time() - setup_start

    preflight_t0 = jnp.asarray(args.init_t0)
    preflight_alpha = jnp.asarray(args.init_alpha)
    preflight_zeta_vmr = jnp.asarray(args.init_zeta_vmr)
    preflight_log_p = jnp.asarray(args.init_log_p_cloud)
    timing = {"setup_seconds": setup_seconds}
    if args.preflight_autodiff:
        for chip_position, chip_data in enumerate(chip_data_list):
            response = jax.jit(response_functions[chip_position])
            start = time.time()
            spectrum, derivative = response(
                preflight_t0,
                preflight_alpha,
                preflight_zeta_vmr,
                preflight_log_p,
            )
            spectrum.block_until_ready()
            derivative.block_until_ready()
            timing[f"chip{chip_data.chip_index}_response_compile_seconds"] = (
                time.time() - start
            )
            start = time.time()
            spectrum_second, derivative_second = response(
                preflight_t0,
                preflight_alpha,
                preflight_zeta_vmr,
                preflight_log_p,
            )
            spectrum_second.block_until_ready()
            derivative_second.block_until_ready()
            timing[f"chip{chip_data.chip_index}_response_second_seconds"] = (
                time.time() - start
            )
            timing[f"chip{chip_data.chip_index}_spectrum_all_finite"] = finite(spectrum)
            timing[f"chip{chip_data.chip_index}_derivative_all_finite"] = finite(
                derivative
            )
            timing[f"chip{chip_data.chip_index}_spectrum_rms"] = float(
                jnp.sqrt(jnp.mean(spectrum * spectrum))
            )
            timing[f"chip{chip_data.chip_index}_derivative_rms"] = float(
                jnp.sqrt(jnp.mean(derivative * derivative))
            )

    physical_init_values = load_initial_values(
        args,
        len(chip_data_list),
        len(obs_times),
    )
    if args.manual_atmosphere_init:
        physical_init_values.update(
            {
                "T0": args.init_t0,
                "alpha": args.init_alpha,
                "zeta_vmr": args.init_zeta_vmr,
                "log_p_cloud": args.init_log_p_cloud,
                "sigma_log_p": args.init_sigma_log_p,
            }
        )
    parameter_centers, parameter_scales = build_parameter_reparameterization(
        args,
        physical_init_values,
    )
    fixed_nuisance_values = None
    if args.fix_nuisance:
        fixed_nuisance_values = {
            "A": physical_init_values["A"],
            "log_w": physical_init_values["log_w"],
            "sigma_d": physical_init_values["sigma_d"],
        }
    init_values = build_sampling_initial_values(
        args,
        physical_init_values,
        fixed_nuisance_values,
    )

    def model():
        return on_the_fly_pressure_model(
            data=data,
            wavelengths=wavelengths,
            obs_times=obs_times,
            theta=geometry.theta,
            phi=geometry.phi,
            distance_matrix=geometry.distance_matrix,
            response_functions=response_functions,
            fixed_period=args.fixed_period,
            fixed_cosi=args.fixed_cosi,
            fixed_v=args.fixed_v,
            fixed_q1=args.fixed_q1,
            fixed_q2=args.fixed_q2,
            t0_bounds=(args.t0_min, args.t0_max),
            alpha_bounds=(args.alpha_min, args.alpha_max),
            zeta_vmr_bounds=(args.zeta_vmr_min, args.zeta_vmr_max),
            log_p_cloud_bounds=(args.log_p_cloud_min, args.log_p_cloud_max),
            sigma_log_p_scale=args.sigma_log_p_scale,
            standardized_parameters=args.standardized_parameters,
            parameter_centers=parameter_centers,
            parameter_scales=parameter_scales,
            fixed_ell_b=args.fixed_ell_b,
            zero_mean_pressure_map=args.zero_mean_pressure_map,
            log_w_scale=args.log_w_scale,
            fixed_nuisance_values=fixed_nuisance_values,
            gp_jitter=args.gp_jitter,
            noise_jitter=args.noise_jitter,
        )

    init_values = maybe_find_map_init(args, model, init_values)

    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=args.target_accept_prob,
        dense_mass=args.dense_mass,
        max_tree_depth=args.max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    run_start = time.time()
    mcmc.run(
        jax.random.PRNGKey(args.seed),
        extra_fields=("diverging", "accept_prob", "num_steps", "potential_energy"),
    )
    run_seconds = time.time() - run_start
    if args.print_summary and args.num_samples >= 4:
        mcmc.print_summary()
    elif args.print_summary:
        print("Skipping MCMC summary because num_samples < 4.")

    samples = mcmc.get_samples()
    extra_fields = mcmc.get_extra_fields()
    filename = (
        "mcmc_on_the_fly_atmosphere_pressure.npz"
        if args.full_data
        else "mcmc_on_the_fly_atmosphere_pressure_smoke.npz"
    )
    output_path = out_dir / filename
    save_data = {
        name: np.asarray(value)
        for name, value in samples.items()
    }
    save_data.update(
        {
            f"extra_{name}": np.asarray(value)
            for name, value in extra_fields.items()
        }
    )
    save_data.update(
        {
            "chip_indices": np.asarray(args.chip_indices),
            "obs_times": np.asarray(obs_times),
            "nside": np.asarray(args.nside),
            "t0_bounds": np.asarray([args.t0_min, args.t0_max]),
            "alpha_bounds": np.asarray([args.alpha_min, args.alpha_max]),
            "zeta_vmr_bounds": np.asarray([args.zeta_vmr_min, args.zeta_vmr_max]),
            "log_p_cloud_bounds": np.asarray(
                [args.log_p_cloud_min, args.log_p_cloud_max]
            ),
            "sigma_log_p_scale": np.asarray(args.sigma_log_p_scale),
            "standardized_parameters": np.asarray(args.standardized_parameters),
            "rotated_atmosphere_parameters": np.asarray(
                args.rotated_atmosphere_parameters
            ),
            "manual_atmosphere_init": np.asarray(args.manual_atmosphere_init),
            "standardized_parameter_centers": np.asarray(
                [
                    parameter_centers["T0"],
                    parameter_centers["alpha"],
                    parameter_centers["zeta_vmr"],
                    parameter_centers["log_p_cloud"],
                    parameter_centers["sigma_log_p"],
                ]
            ),
            "standardized_parameter_scales": np.asarray(
                [
                    parameter_scales["T0"],
                    parameter_scales["alpha"],
                    parameter_scales["zeta_vmr"],
                    parameter_scales["log_p_cloud"],
                    parameter_scales["sigma_log_p"],
                ]
            ),
            "atmosphere_rotation_slopes": np.asarray(
                [
                    parameter_scales["zeta_vmr_per_t0"],
                    parameter_scales["log_p_cloud_per_alpha"],
                ]
            ),
            "atmosphere_rotation_slope_names": np.asarray(
                ["zeta_vmr_per_t0", "log_p_cloud_per_alpha"]
            ),
            "standardized_parameter_names": np.asarray(
                ["T0", "alpha", "zeta_vmr", "log_p_cloud", "sigma_log_p"]
            ),
            "zero_mean_pressure_map": np.asarray(args.zero_mean_pressure_map),
            "pressure_derivative_method": np.asarray("on_the_fly_autodiff"),
            "full_data": np.asarray(args.full_data),
            "preflight_autodiff": np.asarray(args.preflight_autodiff),
            "dense_mass": np.asarray(args.dense_mass),
            "fix_nuisance": np.asarray(args.fix_nuisance),
            "map_init": np.asarray(args.map_init),
            "init_from": np.asarray("" if args.init_from is None else args.init_from),
            "x64": np.asarray(args.x64),
        }
    )
    for chip_position, chip_data in enumerate(chip_data_list):
        save_data[f"wavelengths_chip{chip_data.chip_index}"] = np.asarray(
            chip_data.wavelengths
        )
        save_data[f"flux_chip{chip_data.chip_index}"] = np.asarray(chip_data.flux)
        save_data[f"chip_position_{chip_position}"] = np.asarray(chip_data.chip_index)
    np.savez(output_path, **save_data)
    diagnostics = {
        "mode": "milestone5_on_the_fly_atmospheric_pressure_retrieval",
        "output_path": str(output_path),
        "run_seconds": run_seconds,
        "chip_indices": args.chip_indices,
        "full_data": args.full_data,
        "n_chip": len(chip_data_list),
        "n_phase": int(data.shape[1]),
        "n_wavelength": int(data.shape[2]),
        "nside": args.nside,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "log_p_cloud_bounds": [args.log_p_cloud_min, args.log_p_cloud_max],
        "t0_bounds": [args.t0_min, args.t0_max],
        "alpha_bounds": [args.alpha_min, args.alpha_max],
        "zeta_vmr_bounds": [args.zeta_vmr_min, args.zeta_vmr_max],
        "init_t0": args.init_t0,
        "init_alpha": args.init_alpha,
        "init_zeta_vmr": args.init_zeta_vmr,
        "init_log_p_cloud": args.init_log_p_cloud,
        "zero_mean_pressure_map": args.zero_mean_pressure_map,
        "standardized_parameters": args.standardized_parameters,
        "rotated_atmosphere_parameters": args.rotated_atmosphere_parameters,
        "manual_atmosphere_init": args.manual_atmosphere_init,
        "standardized_parameter_centers": {
            name: float(value) for name, value in parameter_centers.items()
            if name in ["T0", "alpha", "zeta_vmr", "log_p_cloud", "sigma_log_p"]
        },
        "standardized_parameter_scales": {
            name: float(value)
            for name, value in parameter_scales.items()
            if name
            in [
                "T0",
                "alpha",
                "zeta_vmr",
                "log_p_cloud",
                "sigma_log_p",
                "zeta_vmr_per_t0",
                "log_p_cloud_per_alpha",
            ]
        },
        "dense_mass": args.dense_mass,
        "fix_nuisance": args.fix_nuisance,
        "map_init": args.map_init,
        "x64": args.x64,
        "divergence_count": int(
            np.sum(np.asarray(extra_fields.get("diverging", []), dtype=bool))
        ),
        "mean_accept_prob": (
            float(np.mean(np.asarray(extra_fields["accept_prob"])))
            if "accept_prob" in extra_fields
            else None
        ),
        "max_num_steps": (
            int(np.max(np.asarray(extra_fields["num_steps"])))
            if "num_steps" in extra_fields
            else None
        ),
        **timing,
    }
    diagnostics_path = out_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
