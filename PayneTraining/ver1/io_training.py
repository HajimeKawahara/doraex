"""NumPy IO for single-pixel Doraex training spectra.

Wavelength is in Angstrom, with four consecutive 4500-point chips. Spectra
are unnormalized specific intensity per wavenumber, in
erg s^-1 cm^-2 sr^-1 (cm^-1)^-1, not intensity per wavelength.
"""

from pathlib import Path

import numpy as np


PARAMETER_NAMES = (
    "T0", "alpha", "log_vmr_co", "log_vmr_h2o", "log_vmr_ch4",
    "log_vmr_hf", "log_p_cloud", "mu",
)
N_WAVELENGTH = 4 * 4500


def _validate_coordinates(parameters, wavelength):
    if parameters.ndim != 2 or parameters.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("parameters must have shape (N, 8).")
    if parameters.shape[0] == 0 or not np.all(np.isfinite(parameters)):
        raise ValueError("parameters must contain at least one finite row.")
    if wavelength.shape != (N_WAVELENGTH,):
        raise ValueError("wavelength must have shape (18000,).")
    if (not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0)
            or np.any(np.diff(wavelength) <= 0)):
        raise ValueError("wavelength must be finite, positive and increasing.")


def create(path, parameters, wavelength):
    """Create a new dataset directory and return its writable spectra memmap.

    Fill every spectra row, then call ``spectra.flush()`` before delivery.
    Existing directories are rejected to prevent accidental overwrite.
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    wavelength = np.asarray(wavelength, dtype=np.float64)
    _validate_coordinates(parameters, wavelength)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np.save(path / "parameters.npy", parameters, allow_pickle=False)
    np.save(path / "wavelength.npy", wavelength, allow_pickle=False)
    return np.lib.format.open_memmap(
        path / "spectra.npy", mode="w+", dtype=np.float64,
        shape=(parameters.shape[0], N_WAVELENGTH),
    )


def load(path, mmap_mode="r"):
    """Return (parameters, spectra, wavelength), memory-mapped by default.

    Checks array shapes and coordinates; does not scan all spectrum values.
    """
    path = Path(path)
    parameters, spectra, wavelength = (
        np.load(path / f"{name}.npy", mmap_mode=mmap_mode, allow_pickle=False)
        for name in ("parameters", "spectra", "wavelength")
    )
    _validate_coordinates(parameters, wavelength)
    if spectra.shape != (parameters.shape[0], N_WAVELENGTH):
        raise ValueError("spectra must have shape (N, 18000).")
    if any(array.dtype != np.dtype("float64")
           for array in (parameters, spectra, wavelength)):
        raise ValueError("All arrays must have dtype float64.")
    return parameters, spectra, wavelength
