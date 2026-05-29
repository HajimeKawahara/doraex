"""Run Milestone 2-4 joint multi-chip retrievals."""

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
    load_milestone2_joint_free_t0_cloud_inputs,
    load_milestone2_joint_free_t0_vmr_cloud_inputs,
    run_joint_free_t0_cloud_two_column_mcmc,
    save_joint_free_t0_cloud_two_column_samples,
)


def parse_chips(text):
    """Parse comma-separated chip indices."""

    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one chip index is required.")
    return values


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run joint multi-chip Milestone 2-4 retrieval."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--chip-indices", type=parse_chips, default=parse_chips("0,1,2,3"))
    parser.add_argument(
        "--profile-grid-template",
        default=str(ROOT / "data" / "milestone2_t0_cloud_grid_profiles_chip{chip}.npz"),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "milestone2_4a"))
    parser.add_argument(
        "--m2-4b",
        action="store_true",
        help="Use shared T0, log10 Pc, and f_cloud defaults for Milestone 2-4b.",
    )
    parser.add_argument(
        "--m2-4c",
        action="store_true",
        help="Use M2-4b shared atmosphere with ExoMol-consistent fixed grids.",
    )
    parser.add_argument(
        "--m2-4d",
        action="store_true",
        help="Use M2-4c grids with Yama-style per-chip mean normalization.",
    )
    parser.add_argument(
        "--m2-5a",
        action="store_true",
        help="Use shared-atmosphere T0/log10 Pc/zeta_vmr grids for Milestone 2-5a.",
    )
    parser.add_argument(
        "--m2-5b",
        action="store_true",
        help="Use shared-atmosphere T0/alpha/log10 Pc/zeta_vmr grids for M2-5b.",
    )
    parser.add_argument(
        "--m3-1",
        action="store_true",
        help="Use double-cloud endpoints with shared T0/alpha/log10 P_mid/zeta_vmr.",
    )
    parser.add_argument("--nside", type=int, default=8)
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=1500)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-accept-prob", type=float, default=0.98)
    parser.add_argument("--max-tree-depth", type=int, default=11)
    parser.add_argument("--period-mode", choices=("sampled", "fixed"), default="fixed")
    parser.add_argument("--fixed-period", type=float, default=4.83)
    parser.add_argument("--t0-min", type=float, default=1000.0)
    parser.add_argument("--t0-max", type=float, default=1700.0)
    parser.add_argument("--init-t0", type=float, default=1215.0)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.20)
    parser.add_argument("--init-alpha", type=float, default=0.129)
    parser.add_argument("--log-p-cloud-min", type=float, default=-2.0)
    parser.add_argument("--log-p-cloud-max", type=float, default=2.0)
    parser.add_argument("--init-log-p-cloud", type=float, default=1.28)
    parser.add_argument(
        "--log-p-mid-min",
        type=float,
        dest="log_p_cloud_min",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-p-mid-max",
        type=float,
        dest="log_p_cloud_max",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--init-log-p-mid",
        type=float,
        dest="init_log_p_cloud",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--fixed-cloud-delta", type=float, default=1.0)
    parser.add_argument("--zeta-vmr-min", type=float, default=-0.5)
    parser.add_argument("--zeta-vmr-max", type=float, default=0.5)
    parser.add_argument("--init-zeta-vmr", type=float, default=0.0)
    parser.add_argument("--sigma-b-scale", type=float, default=0.1)
    parser.add_argument("--fix-ell-b", type=float, default=0.3)
    parser.add_argument(
        "--free-ell-b",
        action="store_true",
        help="Sample shared ell_b instead of using --fix-ell-b.",
    )
    parser.add_argument(
        "--shared-atmosphere",
        action="store_true",
        help="Share T0, log10 Pc, and f_cloud across chips.",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=("surface_scale", "yama"),
        default="surface_scale",
        help="Use legacy surface_scale or Yama-style F_i/(A_i mean(F_i)) normalization.",
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
    if (
        args.m2_4b
        or args.m2_4c
        or args.m2_4d
        or args.m2_5a
        or args.m2_5b
        or args.m3_1
    ):
        args.shared_atmosphere = True
        default_m24a_out = str(ROOT / "results" / "milestone2_4a")
        if args.out_dir == default_m24a_out:
            if args.m3_1:
                milestone = "milestone3_1"
            elif args.m2_5b:
                milestone = "milestone2_5b"
            elif args.m2_5a:
                milestone = "milestone2_5a"
            elif args.m2_4d:
                milestone = "milestone2_4d"
            elif args.m2_4c:
                milestone = "milestone2_4c"
            else:
                milestone = "milestone2_4b"
            args.out_dir = str(ROOT / "results" / milestone)
    if args.m2_4c or args.m2_4d or args.m2_5a or args.m2_5b or args.m3_1:
        default_template = str(
            ROOT / "data" / "milestone2_t0_cloud_grid_profiles_chip{chip}.npz"
        )
        if args.profile_grid_template == default_template:
            if args.m2_5b or args.m3_1:
                grid_name = (
                    "milestone2_t0_alpha_vmr_cloud_grid_profiles_exomol_chip{chip}.npz"
                )
            elif args.m2_5a:
                grid_name = "milestone2_t0_vmr_cloud_grid_profiles_exomol_chip{chip}.npz"
            else:
                grid_name = "milestone2_t0_cloud_grid_profiles_exomol_chip{chip}.npz"
            args.profile_grid_template = str(
                ROOT / "data" / grid_name
            )
        if args.init_t0 == 1215.0:
            args.init_t0 = 1219.0
        if args.init_log_p_cloud == 1.28:
            args.init_log_p_cloud = 1.45
    if args.m2_4d or args.m2_5a or args.m2_5b or args.m3_1:
        args.normalization_mode = "yama"
    if args.m3_1:
        if args.log_p_cloud_min == -2.0:
            args.log_p_cloud_min = -1.5
        if args.log_p_cloud_max == 2.0:
            args.log_p_cloud_max = 1.5
        if args.init_log_p_cloud == 1.45:
            args.init_log_p_cloud = 1.25
    if args.m2_5b or args.m3_1:
        if args.target_accept_prob == 0.98:
            args.target_accept_prob = 0.99
        if args.max_tree_depth == 11:
            args.max_tree_depth = 12
    return args


def main():
    """Run joint multi-chip NUTS and save posterior samples."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    profile_grid_template = None if args.smoke_test else args.profile_grid_template
    if args.m2_5b or args.m3_1:
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
    elif args.m2_5a:
        (
            chip_data_list,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            zeta_vmr_grid,
            clear_profile_grid,
            cloudy_profile_grid,
        ) = load_milestone2_joint_free_t0_vmr_cloud_inputs(
            args.data_dir,
            chip_indices=args.chip_indices,
            profile_grid_template=profile_grid_template,
            nside=args.nside,
            smoke_test=args.smoke_test,
            smoke_wavelength_step=args.smoke_wavelength_step,
            smoke_phase_count=args.smoke_phase_count,
        )
        alpha_grid = None
    else:
        (
            chip_data_list,
            geometry,
            t0_grid,
            log_p_cloud_grid,
            clear_profile_grid,
            cloudy_profile_grid,
        ) = load_milestone2_joint_free_t0_cloud_inputs(
            args.data_dir,
            chip_indices=args.chip_indices,
            profile_grid_template=profile_grid_template,
            nside=args.nside,
            smoke_test=args.smoke_test,
            smoke_wavelength_step=args.smoke_wavelength_step,
            smoke_phase_count=args.smoke_phase_count,
        )
        alpha_grid = None
        zeta_vmr_grid = None

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
        seed=args.seed,
        period_mode=args.period_mode,
        fixed_period=args.fixed_period,
        t0_bounds=(args.t0_min, args.t0_max),
        alpha_bounds=(args.alpha_min, args.alpha_max),
        log_p_cloud_bounds=(args.log_p_cloud_min, args.log_p_cloud_max),
        zeta_vmr_bounds=(args.zeta_vmr_min, args.zeta_vmr_max),
        init_t0=args.init_t0,
        init_alpha=args.init_alpha,
        init_log_p_cloud=args.init_log_p_cloud,
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
        shared_atmosphere=args.shared_atmosphere,
        normalization_mode=args.normalization_mode,
        column_mode="double_cloud" if args.m3_1 else "clear_cloud",
        fixed_cloud_delta=args.fixed_cloud_delta,
    )
    if args.print_summary:
        mcmc.print_summary()

    out_dir = Path(args.out_dir)
    suffix = "_smoke" if args.smoke_test else ""
    atmosphere_suffix = "_shared_atmosphere" if args.shared_atmosphere else ""
    output_path = (
        out_dir / f"mcmc_joint_chips_free_t0_cloud{atmosphere_suffix}{suffix}.npz"
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
        t0_bounds=(args.t0_min, args.t0_max),
        log_p_cloud_bounds=(args.log_p_cloud_min, args.log_p_cloud_max),
        alpha_grid=alpha_grid,
        alpha_bounds=(args.alpha_min, args.alpha_max),
        zeta_vmr_grid=zeta_vmr_grid,
        zeta_vmr_bounds=(args.zeta_vmr_min, args.zeta_vmr_max),
        sigma_b_scale=args.sigma_b_scale,
        fixed_ell_b=fixed_ell_b,
        fix_geometry=args.fix_geometry_to_milestone1,
        shared_atmosphere=args.shared_atmosphere,
        normalization_mode=args.normalization_mode,
        column_mode="double_cloud" if args.m3_1 else "clear_cloud",
        fixed_cloud_delta=args.fixed_cloud_delta,
    )
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    main()
