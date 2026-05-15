"""Generate T0 and cloudy-profile grids for Milestone 2-3a."""

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
    save_t0_cloud_profile_grid,
    synthetic_t0_cloud_profile_grid,
)
from chip_paths import t0_cloud_grid_path  # noqa: E402


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
        description="Generate T0/log10 Pc profile grid for Milestone 2-3a."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--out",
        default=None,
        help="Output NPZ path. Defaults to data/milestone2_t0_cloud_grid_profiles_chip{N}.npz.",
    )
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument(
        "--m2-4c",
        action="store_true",
        help="Use Yama Luhman 16B ExoMol fixed-atmosphere defaults.",
    )
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--t0-min", type=float, default=1000.0)
    parser.add_argument("--t0-max", type=float, default=1700.0)
    parser.add_argument("--t0-count", type=int, default=15)
    parser.add_argument("--log-p-cloud-min", type=float, default=-2.0)
    parser.add_argument("--log-p-cloud-max", type=float, default=2.0)
    parser.add_argument("--log-p-cloud-count", type=int, default=33)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.out is None:
        atmosphere_tag = "exomol" if args.m2_4c else None
        args.out = str(t0_cloud_grid_path(args.chip_index, atmosphere_tag=atmosphere_tag))
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


def main():
    """Generate and save a T0/cloudy-profile grid."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    t0_grid = np.linspace(args.t0_min, args.t0_max, args.t0_count)
    log_p_cloud_grid = np.linspace(
        args.log_p_cloud_min,
        args.log_p_cloud_max,
        args.log_p_cloud_count,
    )
    chip_data = load_luhman16b_chip(args.data_dir, chip_index=args.chip_index)
    if args.smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=args.smoke_wavelength_step,
            phase_count=args.smoke_phase_count,
        )
        clear_profile_grid, cloudy_profile_grid = synthetic_t0_cloud_profile_grid(
            chip_data.wavelengths,
            t0_grid,
            log_p_cloud_grid,
        )
        metadata = {"profile_source": "synthetic_smoke_t0_cloud_grid"}
    else:
        base_parameters = (
            YAMA_L16B_EXOMOL_ATMOSPHERE if args.m2_4c else FixedPowerLawAtmosphere()
        )
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
            model.parameters = replace(base_parameters, t0=float(t0))
            clear_profiles.append(np.asarray(model.clear()))
            cloudy_for_t0 = []
            for log_p_cloud in log_p_cloud_grid:
                model.parameters = replace(
                    base_parameters,
                    t0=float(t0),
                    log_p_cloud=float(log_p_cloud),
                )
                cloudy_for_t0.append(np.asarray(model.cloudy()))
            cloudy_profiles.append(cloudy_for_t0)
        clear_profile_grid = np.asarray(clear_profiles)
        cloudy_profile_grid = np.asarray(cloudy_profiles)
        metadata = {
            "profile_source": "exojax_powerlaw_t0_cloud_grid",
            "atmosphere_preset": (
                "yama_luhman16b_exomol" if args.m2_4c else "default_hitemp_h2_median"
            ),
            "alpha": base_parameters.alpha,
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

    save_t0_cloud_profile_grid(
        args.out,
        chip_data.wavelengths,
        t0_grid,
        log_p_cloud_grid,
        clear_profile_grid,
        cloudy_profile_grid,
        metadata=metadata,
    )
    print(f"T0/cloud profile grid saved to {args.out}")


if __name__ == "__main__":
    main()
