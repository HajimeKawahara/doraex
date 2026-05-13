"""Run Milestone 2-2a two-column Doppler retrieval with free log10 Pc."""

import argparse
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    load_milestone2_free_cloud_inputs,
    run_free_cloud_two_column_mcmc,
    save_free_cloud_two_column_samples,
)
from chip_paths import cloud_grid_path  # noqa: E402

DEFAULT_M22A_OUT = ROOT / "results" / "milestone2_2a"
DEFAULT_M22B_OUT = ROOT / "results" / "milestone2_2b"


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Milestone 2-2 free cloud-top pressure retrieval."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--profile-grid",
        default=None,
        help="NPZ file with clear_profile and cloudy_profile_grid.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_M22A_OUT))
    parser.add_argument(
        "--m2-2b",
        action="store_true",
        help="Use Milestone 2-2b wide-cloud defaults.",
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
    parser.add_argument("--log-p-cloud-min", type=float, default=0.0)
    parser.add_argument("--log-p-cloud-max", type=float, default=2.0)
    parser.add_argument("--init-log-p-cloud", type=float, default=1.35)
    parser.add_argument("--sigma-b-scale", type=float, default=0.1)
    parser.add_argument("--fix-ell-b", type=float, default=0.4)
    parser.add_argument(
        "--free-ell-b",
        action="store_true",
        help="Sample ell_b instead of using --fix-ell-b.",
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
    args = parser.parse_args()
    if args.m2_2b:
        if args.out_dir == str(DEFAULT_M22A_OUT):
            args.out_dir = str(DEFAULT_M22B_OUT)
        if args.log_p_cloud_min == 0.0:
            args.log_p_cloud_min = -2.0
        if args.log_p_cloud_max == 2.0:
            args.log_p_cloud_max = 2.0
    if args.profile_grid is None:
        args.profile_grid = str(cloud_grid_path(args.chip_index, wide=args.m2_2b))
    return args


def main():
    """Run NUTS and save Milestone 2-2a posterior samples."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    profile_grid_path = None if args.smoke_test else args.profile_grid
    smoke_log_p_cloud_grid = None
    if args.smoke_test and args.m2_2b:
        smoke_log_p_cloud_grid = np.linspace(
            args.log_p_cloud_min,
            args.log_p_cloud_max,
            5,
        )
    chip_data, geometry, clear_profile, log_p_cloud_grid, cloudy_profile_grid = (
        load_milestone2_free_cloud_inputs(
            args.data_dir,
            profile_grid_path=profile_grid_path,
            chip_index=args.chip_index,
            nside=args.nside,
            smoke_test=args.smoke_test,
            smoke_wavelength_step=args.smoke_wavelength_step,
            smoke_phase_count=args.smoke_phase_count,
            smoke_log_p_cloud_grid=smoke_log_p_cloud_grid,
        )
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

    fixed_ell_b = None if args.free_ell_b else args.fix_ell_b
    log_p_cloud_bounds = (args.log_p_cloud_min, args.log_p_cloud_max)
    mcmc = run_free_cloud_two_column_mcmc(
        chip_data,
        geometry,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        period_mode=args.period_mode,
        fixed_period=args.fixed_period,
        log_p_cloud_bounds=log_p_cloud_bounds,
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

    milestone_suffix = "_wide" if args.m2_2b else ""
    suffix = f"{milestone_suffix}_smoke" if args.smoke_test else milestone_suffix
    output_path = (
        Path(args.out_dir)
        / f"mcmc_chip{args.chip_index}_{args.period_mode}_free_cloud{suffix}.npz"
    )
    save_free_cloud_two_column_samples(
        output_path,
        mcmc.get_samples(),
        chip_data,
        geometry,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        period_mode=args.period_mode,
        log_p_cloud_bounds=log_p_cloud_bounds,
        sigma_b_scale=args.sigma_b_scale,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=args.fix_geometry_to_milestone1,
    )
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
