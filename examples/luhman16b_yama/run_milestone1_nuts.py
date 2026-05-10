"""Run the Milestone 1 Luhman 16B Ureshino-style NUTS analysis."""

import argparse
from pathlib import Path
import sys

import jax


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.workflows.luhman16b_milestone1 import (  # noqa: E402
    load_milestone1_inputs,
    run_luhman16b_mcmc,
    save_mcmc_samples,
)


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Milestone 1 Luhman 16B BayesianDI reproduction."
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "external" / "BayesianDI" / "data"),
        help="Directory containing fainterspectral-fits_6.pickle and posterior_predictive_vsini=0.npz.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "milestone1"))
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept-prob", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument(
        "--period-mode",
        choices=("sampled", "fixed"),
        default="sampled",
        help="Use sampled P for Figure 8/9 or fixed P for the auxiliary 5-hour run.",
    )
    parser.add_argument("--fixed-period", type=float, default=5.0)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a reduced wavelength/phase grid and short NUTS settings.",
    )
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--print-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print NumPyro posterior diagnostics when enough samples are available.",
    )
    return parser.parse_args()


def main():
    """Run NUTS and save posterior samples."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    chip_data, geometry = load_milestone1_inputs(
        args.data_dir,
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

    mcmc = run_luhman16b_mcmc(
        chip_data,
        geometry,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        period_mode=args.period_mode,
        fixed_period=args.fixed_period,
        target_accept_prob=args.target_accept_prob,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
        progress_bar=True,
    )
    if args.print_summary and num_samples >= 4:
        mcmc.print_summary()
    elif args.print_summary:
        print("Skipping MCMC summary because num_samples < 4.")

    suffix = "_smoke" if args.smoke_test else ""
    output_path = (
        Path(args.out_dir)
        / f"mcmc_chip{args.chip_index}_{args.period_mode}{suffix}.npz"
    )
    save_mcmc_samples(
        output_path,
        mcmc.get_samples(),
        chip_data,
        geometry,
        period_mode=args.period_mode,
    )
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
