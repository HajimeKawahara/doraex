"""Generate fixed clear/cloudy power-law profiles for Milestone 2-1."""

import argparse
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
    save_two_column_profiles,
    synthetic_two_column_profiles,
)


def parse_args():
    """Parse command-line arguments."""

    default_database = Path.home() / "data_mol" / ".database"
    parser = argparse.ArgumentParser(
        description="Generate fixed power-law clear/cloudy profiles for Milestone 2-1."
    )
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "data" / "milestone2_fixed_profiles_chip1.npz"))
    parser.add_argument("--chip-index", type=int, default=1)
    parser.add_argument("--opacity-cache-dir", default=str(ROOT / "data" / "opacities" / "luhman16b_powerlaw"))
    parser.add_argument("--database-dir", default=str(default_database))
    parser.add_argument("--nx", type=int, default=4500)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-wavelength-step", type=int, default=64)
    parser.add_argument("--smoke-phase-count", type=int, default=4)
    parser.add_argument("--x64", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


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
    """Generate and save profiles."""

    args = parse_args()
    jax.config.update("jax_enable_x64", args.x64)
    chip_data = load_luhman16b_chip(args.data_dir, chip_index=args.chip_index)
    if args.smoke_test:
        chip_data = subset_chip_data(
            chip_data,
            wavelength_step=args.smoke_wavelength_step,
            phase_count=args.smoke_phase_count,
        )
        clear_profile, cloudy_profile = synthetic_two_column_profiles(chip_data.wavelengths)
        metadata = {"profile_source": "synthetic_smoke"}
    else:
        model = Luhman16BPowerLawColumnModel(
            chip_data.wavelengths,
            molecule_paths=_molecule_paths(args.database_dir),
            cia_paths=_cia_paths(args.database_dir),
            opacity_cache_dir=args.opacity_cache_dir,
            parameters=FixedPowerLawAtmosphere(),
            nx=args.nx,
        )
        clear_profile = np.asarray(model.clear())
        cloudy_profile = np.asarray(model.cloudy())
        metadata = {
            "profile_source": "exojax_powerlaw_fixed",
            "t0": FixedPowerLawAtmosphere().t0,
            "alpha": FixedPowerLawAtmosphere().alpha,
            "logg": FixedPowerLawAtmosphere().logg,
            "log_p_cloud": FixedPowerLawAtmosphere().log_p_cloud,
        }

    save_two_column_profiles(
        args.out,
        chip_data.wavelengths,
        clear_profile,
        cloudy_profile,
        metadata=metadata,
    )
    print(f"Profiles saved to {args.out}")


if __name__ == "__main__":
    main()
