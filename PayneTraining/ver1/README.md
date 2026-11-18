# Doraex training spectra

Generate 100,000 independent single-pixel atmosphere/angle samples for TransformerPayne,
using the completed M9/v3 production model (q95/seed 1). Each sample covers all
four CRIRES chips. Sampling is independent uniform within the ranges below;
abundances and cloud pressure are uniform in log10. The random seed is 0.

| Parameter column | Range |
|---|---|
| `T0` [K] | 1000–1700 |
| `alpha` | 0.05–0.20 |
| `log_vmr_co`, `log_vmr_h2o`, `log_vmr_ch4` | −6 to −1 each |
| `log_vmr_hf` | −10 to −5 |
| `log_p_cloud` [log10(bar)] | −3 to 5 |
| `mu` | 0 < mu ≤ 1 |

Cloud pressure is the **local pixel value**, not the global mean. `sigma_logP`
belongs to the spatial GP and is not a spectral input. Fixed settings match M9:
logg ≈ 4.86, 101 layers from 1e−4 to 100 bar, temperature clipped to 210–3500 K,
gray cloud width 0.3 dex and column optical depth 500, CO/H2O/CH4/HF plus
H2–H2/H2–He CIA, and the existing 81-node real-space DiffGrid opacity tables.
As in M9, H2–H2 CIA coefficients are held at their 3000 K table endpoint above
that temperature. Very deep clouds approach the cloud-free limit of this column.

## Files

Generation writes only three float64 NumPy arrays to `data/` (about 14.4 GB total):

- `parameters.npy`: `(100000, 8)`, columns in the table's order.
- `spectra.npy`: `(100000, 18000)`, unnormalized specific intensity per
  **wavenumber**, in erg s−1 cm−2 sr−1 (cm−1)−1.
- `wavelength.npy`: `(18000,)`, vacuum wavelength in Angstrom, increasing.
  Four consecutive blocks of 4500 points correspond to chips 0–3.
  Do not interpolate across chip gaps.

The internal RT grids span 22866.028–23008.110, 23028.460–23165.629,
23183.281–23315.944, and 23331.697–23459.164 Angstrom. Intensities are before
instrument convolution (R=100000), Doppler shifts, and disk normalization.
The wavelength coordinate does **not** convert intensity to per-Angstrom units.

## Read and generate

Only NumPy is needed to read the data; arrays are memory-mapped by default.
Run this example from `PayneTraining/ver1` (or copy `io_training.py` beside `data/`):

```python
from io_training import load, PARAMETER_NAMES

parameters, spectra, wavelength = load("data")
x, y = parameters[:32], spectra[:32]
```

Generation requires a CUDA-enabled float64 JAX environment and the M9-compatible
ExoJAX checkout (`7370e1842a2f97527d185d198f1333fb562009f1` was used for v1).
Provide these external inputs; datasets, caches and run reports are not in Git:

- The M9/v3 q95/seed 1 archive, including `observation_contract.npz` and
  `source/exojax_forward.py`. Its forward source must match this checkout.
- The 16 saved real-space DiffGrid opacity tables and molecular/CIA database.

Defaults follow the existing Doraex layout. To use other locations, set
`PAYNE_BASELINE` (archive directory), `PAYNE_OPACITY_CACHE` (DiffGrid cache root),
`DORAEX_DATABASE_DIR` (database root), and `EXOJAX_DIR` (ExoJAX checkout).
From the repository root:

```bash
python PayneTraining/ver1/generate.py --samples 100000 --seed 0 --out PayneTraining/ver1/data
```

Existing outputs are never overwritten. A `.partial` directory is renamed only
after successful generation. Runtime and RT validation are recorded separately
in `data.json`. `io_training.create()` provides the matching minimal writer.

For the default 100,000-row, seed-0 dataset, run `audit_integrity.py` for a CPU
array audit or `audit_rt.py` for a GPU RT replay/PreMODIT comparison. They write
`audit_integrity.json` and `audit_rt.json` separately. The latter also requires
the M9 real-space PreMODIT caches; set `PAYNE_PREMODIT_CACHE` to relocate them.
These audits target the v1 dataset in `data/`, not arbitrary sampling settings.

For a comparison that preserves the M9 model, integrate predicted intensities
with its four-point angular quadrature, then apply its fixed limb darkening and
existing instrument/disk processing. Direct use of angle-dependent intensities
in the disk integral changes the angular model; do not also apply limb darkening.
