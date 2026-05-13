"""Run Milestone 2-3 fixed-ell sensitivity analyses."""

import argparse
from pathlib import Path
import sys

import jax


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    load_milestone2_free_t0_cloud_inputs,
    run_free_t0_cloud_two_column_mcmc,
    save_free_t0_cloud_two_column_samples,
)
from chip_paths import t0_cloud_grid_path  # noqa: E402


def parse_ell_values(text):
    """Parse comma-separated fixed ell_b values."""

    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one ell_b value is required.")
    return values


def ell_tag(value):
    """Return a filesystem-safe tag for an ell_b value."""

    return f"ell{value:.3f}".replace(".", "p")


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run fixed-ell sensitivity checks for Milestone 2-3."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--profile-grid",
        default=None,
        help="NPZ file with T0/cloud profile grids. Defaults to a chip-aware path.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "milestone2_3c"))
    parser.add_argument(
        "--ell-values",
        type=parse_ell_values,
        default=parse_ell_values("0.25,0.30,0.35,0.40,0.50"),
        help="Comma-separated fixed ell_b values in radians.",
    )
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--num-warmup", type=int, default=1500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept-prob", type=float, default=0.98)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--period-mode", choices=("sampled", "fixed"), default="fixed")
    parser.add_argument("--fixed-period", type=float, default=4.83)
    parser.add_argument("--t0-min", type=float, default=1000.0)
    parser.add_argument("--t0-max", type=float, default=1700.0)
    parser.add_argument("--init-t0", type=float, default=1215.0)
    parser.add_argument("--log-p-cloud-min", type=float, default=-2.0)
    parser.add_argument("--log-p-cloud-max", type=float, default=2.0)
    parser.add_argument("--init-log-p-cloud", type=float, default=1.28)
    parser.add_argument("--sigma-b-scale", type=float, default=0.1)
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
    args = parser.parse_args()
    if args.profile_grid is None:
        args.profile_grid = str(t0_cloud_grid_path(args.chip_index))
    return args


def main():
    """Run all fixed-ell sensitivity chains."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    profile_grid_path = None if args.smoke_test else args.profile_grid
    (
        chip_data,
        geometry,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
    ) = load_milestone2_free_t0_cloud_inputs(
        args.data_dir,
        profile_grid_path=profile_grid_path,
        chip_index=args.chip_index,
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
    log_p_cloud_bounds = (args.log_p_cloud_min, args.log_p_cloud_max)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for offset, fixed_ell_b in enumerate(args.ell_values):
        tag = ell_tag(fixed_ell_b)
        run_seed = args.seed + offset
        print(f"Running fixed ell_b={fixed_ell_b:.3f} rad with seed={run_seed}")
        mcmc = run_free_t0_cloud_two_column_mcmc(
            chip_data,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=args.num_chains,
            seed=run_seed,
            period_mode=args.period_mode,
            fixed_period=args.fixed_period,
            t0_bounds=t0_bounds,
            log_p_cloud_bounds=log_p_cloud_bounds,
            init_t0=args.init_t0,
            init_log_p_cloud=args.init_log_p_cloud,
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
            progress_bar=True,
        )
        if args.print_summary and num_samples >= 4:
            mcmc.print_summary()
        elif args.print_summary:
            print("Skipping MCMC summary because num_samples < 4.")

        suffix = f"_{tag}_smoke" if args.smoke_test else f"_{tag}"
        output_path = (
            out_dir
            / f"mcmc_chip{args.chip_index}_{args.period_mode}_free_t0_cloud{suffix}.npz"
        )
        save_free_t0_cloud_two_column_samples(
            output_path,
            mcmc.get_samples(),
            chip_data,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
            period_mode=args.period_mode,
            t0_bounds=t0_bounds,
            log_p_cloud_bounds=log_p_cloud_bounds,
            sigma_b_scale=args.sigma_b_scale,
            fixed_ell_b=fixed_ell_b,
            fix_geometry=args.fix_geometry_to_milestone1,
        )
        print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
