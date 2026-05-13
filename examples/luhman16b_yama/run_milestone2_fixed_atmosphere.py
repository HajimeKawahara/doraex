"""Run Milestone 2-1 fixed-atmosphere two-column Doppler retrieval."""

import argparse
from pathlib import Path
import sys

import jax


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.workflows.luhman16b_milestone2 import (  # noqa: E402
    load_milestone2_fixed_inputs,
    run_fixed_two_column_mcmc,
    save_fixed_two_column_samples,
)
from chip_paths import fixed_profile_path  # noqa: E402


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run fixed-atmosphere two-column Doppler retrieval."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--profiles",
        default=None,
        help="NPZ file with fixed clear_profile and cloudy_profile.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "milestone2_1"))
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept-prob", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--period-mode", choices=("sampled", "fixed"), default="sampled")
    parser.add_argument("--fixed-period", type=float, default=5.0)
    parser.add_argument(
        "--sigma-b-scale",
        type=float,
        default=0.1,
        help="Half-normal scale for cloud-fraction contrast variations.",
    )
    parser.add_argument(
        "--fix-ell-b",
        type=float,
        default=None,
        help="Fix the cloud-map correlation length in radians instead of sampling it.",
    )
    parser.add_argument(
        "--fix-geometry-to-milestone1",
        action="store_true",
        help="Fix cosi, v, q1, and q2 to Milestone-1-like values.",
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
    if args.profiles is None:
        args.profiles = str(fixed_profile_path(args.chip_index))
    return args


def main():
    """Run NUTS and save posterior samples."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    profiles_path = None if args.smoke_test else args.profiles
    chip_data, geometry, clear_profile, cloudy_profile = load_milestone2_fixed_inputs(
        args.data_dir,
        profiles_path=profiles_path,
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

    mcmc = run_fixed_two_column_mcmc(
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        period_mode=args.period_mode,
        fixed_period=args.fixed_period,
        target_accept_prob=args.target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        sigma_b_scale=args.sigma_b_scale,
        fixed_ell_b=args.fix_ell_b,
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

    suffix = "_smoke" if args.smoke_test else ""
    output_path = (
        Path(args.out_dir)
        / f"mcmc_chip{args.chip_index}_{args.period_mode}_fixed_columns{suffix}.npz"
    )
    save_fixed_two_column_samples(
        output_path,
        mcmc.get_samples(),
        chip_data,
        geometry,
        clear_profile,
        cloudy_profile,
        period_mode=args.period_mode,
        sigma_b_scale=args.sigma_b_scale,
        fixed_ell_b=args.fix_ell_b,
        fix_geometry=args.fix_geometry_to_milestone1,
    )
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
