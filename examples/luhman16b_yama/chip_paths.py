"""Chip-aware default paths for Luhman 16B milestone scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def fixed_profile_path(chip_index):
    """Return the default fixed-atmosphere profile path for a chip."""

    return ROOT / "data" / f"milestone2_fixed_profiles_chip{chip_index}.npz"


def cloud_grid_path(chip_index, wide=False):
    """Return the default cloud-profile grid path for a chip."""

    suffix = "_wide" if wide else ""
    return ROOT / "data" / f"milestone2_cloud_grid_profiles{suffix}_chip{chip_index}.npz"


def t0_cloud_grid_path(chip_index, atmosphere_tag=None):
    """Return the default T0/cloud-profile grid path for a chip."""

    suffix = "" if atmosphere_tag is None else f"_{atmosphere_tag}"
    return ROOT / "data" / f"milestone2_t0_cloud_grid_profiles{suffix}_chip{chip_index}.npz"


def t0_vmr_cloud_grid_path(chip_index, atmosphere_tag=None):
    """Return the default T0/VMR/cloud-profile grid path for a chip."""

    suffix = "" if atmosphere_tag is None else f"_{atmosphere_tag}"
    return ROOT / "data" / f"milestone2_t0_vmr_cloud_grid_profiles{suffix}_chip{chip_index}.npz"


def t0_alpha_vmr_cloud_grid_path(chip_index, atmosphere_tag=None):
    """Return the default T0/alpha/VMR/cloud-profile grid path for a chip."""

    suffix = "" if atmosphere_tag is None else f"_{atmosphere_tag}"
    return (
        ROOT
        / "data"
        / f"milestone2_t0_alpha_vmr_cloud_grid_profiles{suffix}_chip{chip_index}.npz"
    )


def fixed_sample_path(out_dir, chip_index, period_mode):
    """Return the default Milestone 2-1 sample path for a chip."""

    return Path(out_dir) / f"mcmc_chip{chip_index}_{period_mode}_fixed_columns.npz"


def free_cloud_sample_path(out_dir, chip_index, period_mode, wide=False):
    """Return the default Milestone 2-2 sample path for a chip."""

    suffix = "_wide" if wide else ""
    return Path(out_dir) / f"mcmc_chip{chip_index}_{period_mode}_free_cloud{suffix}.npz"


def free_t0_cloud_sample_path(out_dir, chip_index, period_mode, free_ell=False):
    """Return the default Milestone 2-3 sample path for a chip."""

    suffix = "_free_ell" if free_ell else ""
    return Path(out_dir) / f"mcmc_chip{chip_index}_{period_mode}_free_t0_cloud{suffix}.npz"
