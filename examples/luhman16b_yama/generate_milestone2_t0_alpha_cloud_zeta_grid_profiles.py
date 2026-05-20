"""Generate T0/alpha/cloud/zeta grids for Milestone 2-5b."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

import jax
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.data.luhman16b import load_luhman16b_chip, subset_chip_data  # noqa: E402
from doraex.spectra.exojax_forward import (  # noqa: E402
    FixedPowerLawAtmosphere,
    Luhman16BPowerLawColumnModel,
    save_t0_alpha_vmr_cloud_profile_grid,
    synthetic_t0_cloud_profile_grid,
)
from chip_paths import t0_alpha_vmr_cloud_grid_path  # noqa: E402


YAMA_L16B_EXOMOL_ATMOSPHERE = FixedPowerLawAtmosphere(
    t0=1219.0,
    alpha=0.129,
    logg=4.97,
    log_vmr_co=-2.96,
    log_vmr_h2o=-3.25,
    log_vmr_ch4=-4.65,
    log_vmr_hf=-7.08,
    rv=25.66,
    log_p_cloud=1.45,
)


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description="Generate T0/alpha/log10 Pc/zeta_vmr grids for M2-5b."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--out",
        default=None,
        help="Output NPZ path. Defaults to data/milestone2_t0_alpha_vmr_cloud_grid_profiles_exomol_chip{N}.npz.",
    )
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--t0-min", type=float, default=1000.0)
    parser.add_argument("--t0-max", type=float, default=1700.0)
    parser.add_argument("--t0-count", type=int, default=21)
    parser.add_argument("--alpha-min", type=float, default=0.05)
    parser.add_argument("--alpha-max", type=float, default=0.20)
    parser.add_argument("--alpha-count", type=int, default=9)
    parser.add_argument("--log-p-cloud-min", type=float, default=-2.0)
    parser.add_argument("--log-p-cloud-max", type=float, default=2.0)
    parser.add_argument("--log-p-cloud-count", type=int, default=33)
    parser.add_argument("--zeta-vmr-min", type=float, default=-0.5)
    parser.add_argument("--zeta-vmr-max", type=float, default=0.5)
    parser.add_argument("--zeta-vmr-count", type=int, default=11)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.out is None:
        args.out = str(
            t0_alpha_vmr_cloud_grid_path(
                args.chip_index,
                atmosphere_tag="exomol",
            )
        )
    return args


def _molecule_paths(database_dir):
    database = Path(os.path.expanduser(database_dir))
    return {
        "CO": database / "CO" / "12C-16O" / "Li2015",
        "H2O": database / "H2O" / "1H2-16O" / "POKAZATEL",
        "CH4": database / "CH4" / "12C-1H4" / "MM",
        "HF": database / "HF" / "1H-19F" / "Coxon-Hajig",
    }


def _cia_paths(database_dir):
    database = Path(os.path.expanduser(database_dir))
    return {
        "H2H2": database / "H2-H2_2011.cia",
        "H2He": database / "H2-He_2011.cia",
    }


def _smoke_grids(wavelengths, t0_grid, alpha_grid, log_p_cloud_grid, zeta_vmr_grid):
    clear_t0_grid, cloudy_t0_grid = synthetic_t0_cloud_profile_grid(
        wavelengths,
        t0_grid,
        log_p_cloud_grid,
    )
    alpha_scale = 1.0 + 0.05 * (alpha_grid - np.mean(alpha_grid)) / max(
        float(np.ptp(alpha_grid)),
        1.0e-6,
    )
    zeta_scale = 1.0 + 0.03 * zeta_vmr_grid
    clear_profile_grid = (
        clear_t0_grid[:, None, None, :]
        * alpha_scale[None, :, None, None]
        * zeta_scale[None, None, :, None]
    )
    cloudy_profile_grid = (
        cloudy_t0_grid[:, None, :, None, :]
        * alpha_scale[None, :, None, None, None]
        * zeta_scale[None, None, None, :, None]
    )
    return clear_profile_grid, cloudy_profile_grid


def main():
    """Generate and save the M2-5b profile grid."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    t0_grid = np.linspace(args.t0_min, args.t0_max, args.t0_count)
    alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_count)
    log_p_cloud_grid = np.linspace(
        args.log_p_cloud_min,
        args.log_p_cloud_max,
        args.log_p_cloud_count,
    )
    zeta_vmr_grid = np.linspace(
        args.zeta_vmr_min,
        args.zeta_vmr_max,
        args.zeta_vmr_count,
    )
    chip_data = load_luhman16b_chip(args.data_dir, chip_index=args.chip_index)
    if args.smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=args.smoke_wavelength_step,
            phase_count=args.smoke_phase_count,
        )
        clear_profile_grid, cloudy_profile_grid = _smoke_grids(
            chip_data.wavelengths,
            t0_grid,
            alpha_grid,
            log_p_cloud_grid,
            zeta_vmr_grid,
        )
        metadata = {"profile_source": "synthetic_smoke_t0_alpha_vmr_cloud_grid"}
    else:
        base_parameters = YAMA_L16B_EXOMOL_ATMOSPHERE
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=base_parameters,
            nx=args.nx,
        )
        clear_profiles = []
        cloudy_profiles = []
        for t0 in t0_grid:
            clear_for_t0 = []
            cloudy_for_t0 = []
            for alpha in alpha_grid:
                clear_for_alpha = []
                for zeta_vmr in zeta_vmr_grid:
                    model.parameters = replace(
                        base_parameters,
                        t0=float(t0),
                        alpha=float(alpha),
                        log_vmr_co=base_parameters.log_vmr_co + float(zeta_vmr),
                        log_vmr_h2o=base_parameters.log_vmr_h2o + float(zeta_vmr),
                        log_vmr_ch4=base_parameters.log_vmr_ch4 + float(zeta_vmr),
                        log_vmr_hf=base_parameters.log_vmr_hf + float(zeta_vmr),
                    )
                    clear_for_alpha.append(np.asarray(model.clear()))
                clear_for_t0.append(clear_for_alpha)

                cloudy_for_alpha = []
                for log_p_cloud in log_p_cloud_grid:
                    cloudy_for_log_p = []
                    for zeta_vmr in zeta_vmr_grid:
                        model.parameters = replace(
                            base_parameters,
                            t0=float(t0),
                            alpha=float(alpha),
                            log_p_cloud=float(log_p_cloud),
                            log_vmr_co=base_parameters.log_vmr_co + float(zeta_vmr),
                            log_vmr_h2o=base_parameters.log_vmr_h2o + float(zeta_vmr),
                            log_vmr_ch4=base_parameters.log_vmr_ch4 + float(zeta_vmr),
                            log_vmr_hf=base_parameters.log_vmr_hf + float(zeta_vmr),
                        )
                        cloudy_for_log_p.append(np.asarray(model.cloudy()))
                    cloudy_for_alpha.append(cloudy_for_log_p)
                cloudy_for_t0.append(cloudy_for_alpha)
            clear_profiles.append(clear_for_t0)
            cloudy_profiles.append(cloudy_for_t0)
        clear_profile_grid = np.asarray(clear_profiles)
        cloudy_profile_grid = np.asarray(cloudy_profiles)
        metadata = {
            "profile_source": "exojax_powerlaw_t0_alpha_vmr_cloud_grid",
            "atmosphere_preset": "yama_luhman16b_exomol",
            "reference_alpha": base_parameters.alpha,
            "logg": base_parameters.logg,
            "log_vmr_co": base_parameters.log_vmr_co,
            "log_vmr_h2o": base_parameters.log_vmr_h2o,
            "log_vmr_ch4": base_parameters.log_vmr_ch4,
            "log_vmr_hf": base_parameters.log_vmr_hf,
            "rv": base_parameters.rv,
            "reference_log_p_cloud": base_parameters.log_p_cloud,
            "cloud_width": base_parameters.cloud_width,
            "cloud_column_optical_depth": base_parameters.cloud_column_optical_depth,
        }

    save_t0_alpha_vmr_cloud_profile_grid(
        args.out,
        chip_data.wavelengths,
        t0_grid,
        alpha_grid,
        log_p_cloud_grid,
        zeta_vmr_grid,
        clear_profile_grid,
        cloudy_profile_grid,
        metadata=metadata,
    )
    print(f"T0/alpha/cloud/zeta profile grid saved to {args.out}")


if __name__ == "__main__":
    main()
