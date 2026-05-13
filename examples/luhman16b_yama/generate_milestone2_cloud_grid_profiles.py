"""Generate clear and cloudy-profile grid for Milestone 2-2a."""

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
    save_cloud_profile_grid,
    synthetic_cloud_profile_grid,
)
from chip_paths import cloud_grid_path  # noqa: E402


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description="Generate power-law cloudy profile grid for Milestone 2-2a."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--out",
        default=None,
        help="Output NPZ path. Defaults to a chip-aware Milestone 2-2 grid path.",
    )
    parser.add_argument(
        "--m2-2b",
        action="store_true",
        help="Use Milestone 2-2b wide-cloud defaults.",
    )
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument(
        "--opacity-cache-dir",
        default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"),
    )
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--log-p-cloud-min", type=float, default=0.0)
    parser.add_argument("--log-p-cloud-max", type=float, default=2.0)
    parser.add_argument("--log-p-cloud-count", type=int, default=17)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.m2_2b:
        if args.log_p_cloud_min == 0.0:
            args.log_p_cloud_min = -2.0
        if args.log_p_cloud_max == 2.0:
            args.log_p_cloud_max = 2.0
        if args.log_p_cloud_count == 17:
            args.log_p_cloud_count = 33
    if args.out is None:
        args.out = str(cloud_grid_path(args.chip_index, wide=args.m2_2b))
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
    """Generate and save a cloudy-profile grid."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
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
        clear_profile, cloudy_profile_grid = synthetic_cloud_profile_grid(
            chip_data.wavelengths,
            log_p_cloud_grid,
        )
        metadata = {"profile_source": "synthetic_smoke_cloud_grid"}
    else:
        base_parameters = FixedPowerLawAtmosphere()
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=base_parameters,
            nx=args.nx,
        )
        clear_profile = np.asarray(model.clear())
        cloudy_profiles = []
        for log_p_cloud in log_p_cloud_grid:
            model.parameters = replace(base_parameters, log_p_cloud=float(log_p_cloud))
            cloudy_profiles.append(np.asarray(model.cloudy()))
        cloudy_profile_grid = np.asarray(cloudy_profiles)
        metadata = {
            "profile_source": "exojax_powerlaw_cloud_grid",
            "t0": base_parameters.t0,
            "alpha": base_parameters.alpha,
            "logg": base_parameters.logg,
            "cloud_width": base_parameters.cloud_width,
            "cloud_column_optical_depth": base_parameters.cloud_column_optical_depth,
        }

    save_cloud_profile_grid(
        args.out,
        chip_data.wavelengths,
        clear_profile,
        log_p_cloud_grid,
        cloudy_profile_grid,
        metadata=metadata,
    )
    print(f"Cloud profile grid saved to {args.out}")


if __name__ == "__main__":
    main()
