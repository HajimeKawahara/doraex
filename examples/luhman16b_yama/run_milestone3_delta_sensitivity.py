"""Run Milestone 3-1 fixed cloud-delta sensitivity analyses."""

import argparse
from pathlib import Path
import sys

import jax


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    load_milestone2_joint_free_t0_alpha_vmr_cloud_inputs,
    run_joint_free_t0_cloud_two_column_mcmc,
    save_joint_free_t0_cloud_two_column_samples,
)


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_delta_values(text):
    """Parse comma-separated fixed cloud-pressure separations."""

    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one cloud delta is required.")
    return values


def delta_tag(value):
    """Return a filesystem-safe tag for a cloud-pressure separation."""

    return f"delta{value:.3f}".replace(".", "p")


def midpoint_bounds(delta, grid_bounds=(-2.0, 2.0)):
    """Return log_p_mid bounds that keep both endpoints inside the grid."""

    half_delta = 0.5 * float(delta)
    return (grid_bounds[0] + half_delta, grid_bounds[1] - half_delta)


def default_init_log_p_mid(delta):
    """Return a deep-cloud anchored initial midpoint."""

    return 1.45 - 0.5 * float(delta)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run fixed cloud-delta sensitivity checks for Milestone 3-1."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--chip-indices", type=parse_chips, default=parse_chips("0,1,2,3"))
    parser.add_argument(
        "--profile-grid-template",
        default=str(
            ROOT
            / "data"
            / "milestone2_t0_alpha_vmr_cloud_grid_profiles_exomol_chip{chip}.npz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "milestone3_1_delta_sensitivity"),
    )
    parser.add_argument(
        "--delta-values",
        type=parse_delta_values,
        default=parse_delta_values("0.03,0.1,0.3"),
        help="Comma-separated fixed cloud-pressure separations in dex.",
    )
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept-prob", type=float, default=0.99)
    parser.add_argument("--max-tree-depth", type=int, default=12)
    parser.add_argument("--period-mode", choices=("sampled", "fixed"), default="fixed")
    parser.add_argument("--fixed-period", type=float, default=4.83)
    parser.add_argument("--t0-min", type=float, default=1000.0)
    parser.add_argument("--t0-max", type=float, default=1700.0)
    parser.add_argument("--init-t0", type=float, default=1219.0)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.20)
    parser.add_argument("--init-alpha", type=float, default=0.129)
    parser.add_argument("--zeta-vmr-min", type=float, default=-0.5)
    parser.add_argument("--zeta-vmr-max", type=float, default=0.5)
    parser.add_argument("--init-zeta-vmr", type=float, default=0.0)
    parser.add_argument("--init-log-p-mid", type=float, default=None)
    parser.add_argument("--sigma-b-scale", type=float, default=0.1)
    parser.add_argument("--fix-ell-b", type=float, default=0.3)
    parser.add_argument(
        "--free-ell-b",
        action="store_true",
        help="Sample shared ell_b instead of using --fix-ell-b.",
    )
    parser.add_argument(
        "--fix-geometry-to-milestone1",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fixed-cosi", type=float, default=0.485)
    parser.add_argument("--fixed-v", type=float, default=31.2)
    parser.add_argument("--fixed-q1", type=float, default=0.81)
    parser.add_argument("--fixed-q2", type=float, default=0.59)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    """Run all fixed cloud-delta sensitivity chains."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    profile_grid_template = None if args.smoke_test else args.profile_grid_template
    (
        chip_data_list,
        geometry,
        t0_grid,
        alpha_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        clear_profile_grid,
        cloudy_profile_grid,
    ) = load_milestone2_joint_free_t0_alpha_vmr_cloud_inputs(
        args.data_dir,
        chip_indices=args.chip_indices,
        profile_grid_template=profile_grid_template,
        nside=args.nside,
        smoke_test=args.smoke_test,
        smoke_wavelength_step=args.smoke_wavelength_step,
        smoke_phase_count=args.smoke_phase_count,
    )

    num_warmup = args.num_warmup
    num_samples = args.num_samples
    max_tree_depth = args.max_tree_depth
    dense_mass = True
    if args.smoke_test:
        num_warmup = min(num_warmup, 5)
        num_samples = min(num_samples, 5)
        max_tree_depth = min(max_tree_depth, 4)
        dense_mass = False

    t0_bounds = (args.t0_min, args.t0_max)
    alpha_bounds = (args.alpha_min, args.alpha_max)
    zeta_vmr_bounds = (args.zeta_vmr_min, args.zeta_vmr_max)
    fixed_ell_b = None if args.free_ell_b else args.fix_ell_b
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for offset, fixed_cloud_delta in enumerate(args.delta_values):
        tag = delta_tag(fixed_cloud_delta)
        run_seed = args.seed + offset
        run_dir = out_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        log_p_mid_bounds = midpoint_bounds(fixed_cloud_delta)
        init_log_p_mid = (
            default_init_log_p_mid(fixed_cloud_delta)
            if args.init_log_p_mid is None
            else args.init_log_p_mid
        )
        print(
            "Running fixed cloud delta="
            f"{fixed_cloud_delta:.3f} dex with seed={run_seed}, "
            f"log_p_mid_bounds={log_p_mid_bounds}, init_log_p_mid={init_log_p_mid:.3f}"
        )
        mcmc = run_joint_free_t0_cloud_two_column_mcmc(
            chip_data_list,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            alpha_grid=alpha_grid,
            zeta_vmr_grid=zeta_vmr_grid,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=args.num_chains,
            seed=run_seed,
            period_mode=args.period_mode,
            fixed_period=args.fixed_period,
            t0_bounds=t0_bounds,
            alpha_bounds=alpha_bounds,
            log_p_cloud_bounds=log_p_mid_bounds,
            zeta_vmr_bounds=zeta_vmr_bounds,
            init_t0=args.init_t0,
            init_alpha=args.init_alpha,
            init_log_p_cloud=init_log_p_mid,
            init_zeta_vmr=args.init_zeta_vmr,
            target_accept_prob=args.target_accept_prob,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
            sigma_b_scale=args.sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=args.fix_geometry_to_milestone1,
            fixed_cosi=args.fixed_cosi,
            fixed_v=args.fixed_v,
            fixed_q1=args.fixed_q1,
            fixed_q2=args.fixed_q2,
            shared_atmosphere=True,
            normalization_mode="yama",
            column_mode="double_cloud",
            fixed_cloud_delta=fixed_cloud_delta,
            progress_bar=True,
        )
        if args.print_summary and num_samples >= 4:
            mcmc.print_summary()
        elif args.print_summary:
            print("Skipping MCMC summary because num_samples < 4.")

        suffix = "_smoke" if args.smoke_test else ""
        output_path = (
            run_dir
            / f"mcmc_joint_chips_free_t0_cloud_shared_atmosphere{suffix}.npz"
        )
        save_joint_free_t0_cloud_two_column_samples(
            output_path,
            mcmc.get_samples(),
            chip_data_list,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            period_mode=args.period_mode,
            t0_bounds=t0_bounds,
            alpha_bounds=alpha_bounds,
            log_p_cloud_bounds=log_p_mid_bounds,
            alpha_grid=alpha_grid,
            zeta_vmr_grid=zeta_vmr_grid,
            zeta_vmr_bounds=zeta_vmr_bounds,
            sigma_b_scale=args.sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=args.fix_geometry_to_milestone1,
            shared_atmosphere=True,
            normalization_mode="yama",
            column_mode="double_cloud",
            fixed_cloud_delta=fixed_cloud_delta,
        )
        print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
