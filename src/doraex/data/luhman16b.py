"""Load the Luhman 16B data products used by Ureshino et al."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


URESHINO_OBS_TIMES = np.array(
    [
        0.1447,
        0.5291,
        0.9135,
        1.2987,
        1.6831,
        2.0676,
        2.4526,
        2.8374,
        3.2220,
        3.6075,
        3.9924,
        4.3760,
        4.7607,
        5.1447,
    ]
)


@dataclass(frozen=True)
class Luhman16BChipData:
    """Prepared single-chip time-series spectra for Milestone 1."""

    wavelengths: np.ndarray
    flux: np.ndarray
    line_profile: np.ndarray
    obs_times: np.ndarray
    chip_index: int


def load_luhman16b_chip(
    data_dir,
    chip_index=1,
    spectra_filename="fainterspectral-fits_6.pickle",
    template_filename="posterior_predictive_vsini=0.npz",
):
    """Load and prepare one CRIRES chip for Ureshino-style Doppler imaging.

    Args:
        data_dir: Directory containing the Crossfield/Yama/Ureshino data files.
        chip_index: Zero-based detector chip index. The Ureshino et al. Chip 2
            analysis corresponds to ``chip_index=1``.
        spectra_filename: Pickle file with observed spectra and wavelength
            solutions.
        template_filename: NPZ file with the ExoJAX posterior predictive
            rest-frame spectrum computed at ``vrot sin i = 0``.

    Returns:
        A :class:`Luhman16BChipData` instance with wavelength-sorted spectra and
        the intrinsic line profile interpolated onto the same grid.
    """

    data_path = Path(data_dir)
    template_path = data_path / template_filename
    spectra_path = data_path / spectra_filename
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template file: {template_path}")
    if not spectra_path.exists():
        raise FileNotFoundError(f"Missing spectra file: {spectra_path}")

    posterior_predictive = np.load(template_path)
    template_wavelengths = posterior_predictive["wav"]
    template_median = posterior_predictive["mu_med"]

    with spectra_path.open("rb") as file_obj:
        crires = pickle.load(file_obj, encoding="latin1")

    observed_di = crires["obs1"] / crires["chipcors"]
    chip_wavelengths = crires["wobs"][chip_index] * 1.0e4
    sort_index = np.argsort(chip_wavelengths)
    wavelengths = np.asarray(chip_wavelengths[sort_index])
    flux = np.asarray(observed_di[:, chip_index, sort_index])
    line_profile = np.interp(wavelengths, template_wavelengths, template_median)

    return Luhman16BChipData(
        wavelengths=wavelengths,
        flux=flux,
        line_profile=np.asarray(line_profile),
        obs_times=URESHINO_OBS_TIMES.copy(),
        chip_index=chip_index,
    )


def subset_chip_data(chip_data, wavelength_step=1, phase_count=None):
    """Return a reduced copy of a chip dataset for fast smoke tests.

    Args:
        chip_data: Full prepared chip dataset.
        wavelength_step: Keep every ``wavelength_step`` wavelength sample.
        phase_count: Optional number of initial phases to keep.

    Returns:
        A reduced :class:`Luhman16BChipData` instance.
    """

    if wavelength_step < 1:
        raise ValueError("wavelength_step must be >= 1")
    phase_slice = slice(None) if phase_count is None else slice(0, phase_count)
    wave_slice = slice(None, None, wavelength_step)
    return Luhman16BChipData(
        wavelengths=chip_data.wavelengths[wave_slice],
        flux=chip_data.flux[phase_slice, wave_slice],
        line_profile=chip_data.line_profile[wave_slice],
        obs_times=chip_data.obs_times[phase_slice],
        chip_index=chip_data.chip_index,
    )
