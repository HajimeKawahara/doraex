# Luhman 16B Milestone 1

This directory contains the target-specific production entry points for reproducing the Ureshino et al. Luhman 16B Bayesian Doppler-imaging analysis with Doraex core APIs.

The Milestone 1 run uses the same precomputed intrinsic line profile as Ureshino et al., `posterior_predictive_vsini=0.npz`. ExoJAX is not run inside this milestone.

## Smoke Test

Use a reduced HEALPix grid and reduced wavelength/phase grid to verify that NUTS starts and saves samples:

```bash
python examples/luhman16b_yama/run_milestone1_nuts.py \
  --smoke-test \
  --nside 1 \
  --num-warmup 2 \
  --num-samples 2 \
  --out-dir results/milestone1_smoke
```

Build the corresponding reduced Figure 8/9 products:

```bash
python examples/luhman16b_yama/make_milestone1_products.py \
  --smoke-test \
  --nside 1 \
  --samples results/milestone1_smoke/mcmc_chip1_sampled_smoke.npz \
  --out-dir results/milestone1_smoke \
  --max-map-samples 2
```

## Production Run

Run the Figure 8/9 NUTS analysis on the full Ureshino setup:

```bash
python examples/luhman16b_yama/run_milestone1_nuts.py \
  --nside 8 \
  --chip-index 1 \
  --num-warmup 500 \
  --num-samples 1000 \
  --out-dir results/milestone1
```

After sampling finishes, reconstruct the posterior mean/uncertainty maps and spectral residuals:

```bash
python examples/luhman16b_yama/make_milestone1_products.py \
  --nside 8 \
  --chip-index 1 \
  --samples results/milestone1/mcmc_chip1_sampled.npz \
  --out-dir results/milestone1
```

The main outputs are:

- `posterior_mean_chip1.npy`
- `posterior_var_chip1.npy`
- `figure8_chip1_mean_uncertainty.png`
- `model_spectrum_chip1.npy`
- `residual_chip1.npy`
- `figure9_chip1_spectral_fit_residual.png`

## Milestone 2-1

Milestone 2-1 couples Milestone 1 to fixed clear/cloudy atmospheric columns. The NUTS run samples geometry, phase weights, mean cloud fraction, noise, and a marginalized cloud-contrast map; the atmospheric power-law parameters are fixed outside NUTS.

Create a smoke-test profile file without ExoJAX:

```bash
python examples/luhman16b_yama/generate_milestone2_fixed_profiles.py \
  --smoke-test \
  --out results/milestone2_1_smoke/fixed_profiles_smoke.npz
```

Run a reduced NUTS smoke test:

```bash
python examples/luhman16b_yama/run_milestone2_fixed_atmosphere.py \
  --smoke-test \
  --nside 1 \
  --num-warmup 2 \
  --num-samples 2 \
  --out-dir results/milestone2_1_smoke
```

For production, first generate fixed ExoJAX profiles on the full chip grid:

```bash
python examples/luhman16b_yama/generate_milestone2_fixed_profiles.py \
  --chip-index 1 \
  --out data/milestone2_fixed_profiles_chip1.npz \
  --opacity-cache-dir data/opacities/luhman16b_powerlaw \
  --database-dir ~/data_mol/.database
```

Then run the fixed-atmosphere two-column NUTS analysis:

```bash
python examples/luhman16b_yama/run_milestone2_fixed_atmosphere.py \
  --nside 8 \
  --chip-index 1 \
  --profiles data/milestone2_fixed_profiles_chip1.npz \
  --num-warmup 500 \
  --num-samples 1000 \
  --out-dir results/milestone2_1
```

If the fully free Milestone 2-1 run shows many divergences, use the stabilized
diagnostic configuration before relaxing parameters again:

```bash
python examples/luhman16b_yama/run_milestone2_fixed_atmosphere.py \
  --nside 8 \
  --chip-index 1 \
  --profiles data/milestone2_fixed_profiles_chip1.npz \
  --num-warmup 1500 \
  --num-samples 1000 \
  --target-accept-prob 0.98 \
  --period-mode fixed \
  --fixed-period 4.83 \
  --sigma-b-scale 0.1 \
  --fix-ell-b 0.4 \
  --fix-geometry-to-milestone1 \
  --out-dir results/milestone2_1_stabilized
```

The Milestone 2-1 product generator also writes
`figure8_cloud_fraction_clipped_chip1.png` and
`cloud_fraction_diagnostics_chip1.json` so cloud-fraction excursions outside
the physical interval can be checked directly.

## Milestone 2-2a

Milestone 2-2a keeps the stabilized Milestone 2-1 geometry, period, and
cloud-map correlation length, but samples the cloud-top pressure `log10 Pc`.
The cloudy local spectra are precomputed on a `log10 Pc` grid and interpolated
inside NUTS.

Generate a smoke-test cloudy grid:

```bash
python examples/luhman16b_yama/generate_milestone2_cloud_grid_profiles.py \
  --smoke-test \
  --out results/milestone2_2a_smoke/cloud_grid_profiles_smoke.npz
```

Run a reduced NUTS smoke test:

```bash
python examples/luhman16b_yama/run_milestone2_free_cloud.py \
  --smoke-test \
  --nside 1 \
  --num-warmup 2 \
  --num-samples 2 \
  --out-dir results/milestone2_2a_smoke
```

For production, first generate the full cloudy grid:

```bash
python examples/luhman16b_yama/generate_milestone2_cloud_grid_profiles.py \
  --chip-index 1 \
  --out data/milestone2_cloud_grid_profiles_chip1.npz \
  --opacity-cache-dir data/opacities/luhman16b_powerlaw \
  --database-dir ~/data_mol/.database \
  --log-p-cloud-min 0.0 \
  --log-p-cloud-max 2.0 \
  --log-p-cloud-count 17
```

Then run the stabilized free-cloud NUTS analysis:

```bash
python examples/luhman16b_yama/run_milestone2_free_cloud.py \
  --nside 8 \
  --chip-index 1 \
  --profile-grid data/milestone2_cloud_grid_profiles_chip1.npz \
  --num-warmup 1500 \
  --num-samples 1000 \
  --target-accept-prob 0.98 \
  --period-mode fixed \
  --fixed-period 4.83 \
  --sigma-b-scale 0.1 \
  --fix-ell-b 0.4 \
  --fix-geometry-to-milestone1 \
  --out-dir results/milestone2_2a
```

Build the corresponding diagnostics:

```bash
python examples/luhman16b_yama/make_milestone2_free_cloud_products.py \
  --nside 8 \
  --chip-index 1 \
  --profile-grid data/milestone2_cloud_grid_profiles_chip1.npz \
  --samples results/milestone2_2a/mcmc_chip1_fixed_free_cloud.npz \
  --out-dir results/milestone2_2a \
  --max-map-samples 1000
```

## Milestone 2-2b

Milestone 2-2b keeps the Milestone 2-2a stabilized sampler, but widens the
cloud-top pressure range to the Yama-style interval `log10 Pc in [-2, 2]`.
The atmospheric power-law parameters, cloud width, cloud column optical depth,
geometry, period, and cloud-map correlation length remain fixed.

Generate the wide cloudy grid:

```bash
python examples/luhman16b_yama/generate_milestone2_cloud_grid_profiles.py \
  --m2-2b \
  --chip-index 1 \
  --opacity-cache-dir data/opacities/luhman16b_powerlaw \
  --database-dir ~/data_mol/.database
```

Run the wide free-cloud NUTS analysis:

```bash
python examples/luhman16b_yama/run_milestone2_free_cloud.py \
  --m2-2b \
  --nside 8 \
  --chip-index 1 \
  --num-warmup 1500 \
  --num-samples 1000 \
  --target-accept-prob 0.98 \
  --period-mode fixed \
  --fixed-period 4.83 \
  --sigma-b-scale 0.1 \
  --fix-ell-b 0.4 \
  --fix-geometry-to-milestone1
```

Build maps, spectra, and wide-prior diagnostics:

```bash
python examples/luhman16b_yama/make_milestone2_free_cloud_products.py \
  --m2-2b \
  --nside 8 \
  --chip-index 1 \
  --max-map-samples 1000
```

Inspect `free_cloud_diagnostics_chip1.json` for boundary sticking in
`log10 Pc`, cloud-fraction excursions outside `[0, 1]`, clipping shifts, and
posterior correlations among `log10 Pc`, `f_cloud`, `sigma_b`, and
`surface_scale`.

## Milestone 2-3a

Milestone 2-3a keeps the Milestone 2-2b stabilized geometry, period,
cloud-map correlation length, cloud width, cloud optical depth, `alpha`, `logg`,
and molecular abundances fixed, but samples the power-law temperature parameter
`T0`. Clear spectra are precomputed on a `T0` grid and cloudy spectra are
precomputed on a `(T0, log10 Pc)` grid.

Generate the T0/cloud grid:

```bash
python examples/luhman16b_yama/generate_milestone2_t0_cloud_grid_profiles.py \
  --chip-index 1 \
  --opacity-cache-dir data/opacities/luhman16b_powerlaw \
  --database-dir ~/data_mol/.database \
  --t0-min 1000 \
  --t0-max 1700 \
  --t0-count 15 \
  --log-p-cloud-min -2.0 \
  --log-p-cloud-max 2.0 \
  --log-p-cloud-count 33
```

Run the free-T0 NUTS analysis:

```bash
python examples/luhman16b_yama/run_milestone2_free_t0_cloud.py \
  --nside 8 \
  --chip-index 1 \
  --num-warmup 1500 \
  --num-samples 1000 \
  --target-accept-prob 0.98 \
  --period-mode fixed \
  --fixed-period 4.83 \
  --sigma-b-scale 0.1 \
  --fix-ell-b 0.4 \
  --fix-geometry-to-milestone1
```

Build maps, spectra, and T0/cloud diagnostics:

```bash
python examples/luhman16b_yama/make_milestone2_free_t0_cloud_products.py \
  --nside 8 \
  --chip-index 1 \
  --max-map-samples 1000
```

Inspect `free_t0_cloud_diagnostics_chip1.json` for `T0` boundary sticking,
`T0`-`log10 Pc` correlation, cloud-fraction excursions outside `[0, 1]`, and
correlations with `f_cloud`, `sigma_b`, and `surface_scale`.

## Milestone 2-3b

Milestone 2-3b uses the same T0/cloud spectral grid as Milestone 2-3a, but
samples the cloud-map correlation length `ell_b` instead of fixing it to
`0.4 rad`. This checks whether the fixed smoothness scale is suppressing
smaller-scale cloud-fraction structure.

Run the free-ell NUTS analysis:

```bash
python examples/luhman16b_yama/run_milestone2_free_t0_cloud.py \
  --m2-3b \
  --nside 8 \
  --chip-index 1 \
  --num-samples 1000 \
  --period-mode fixed \
  --fixed-period 4.83 \
  --sigma-b-scale 0.1 \
  --fix-geometry-to-milestone1
```

The `--m2-3b` preset sets `--free-ell-b`, moves outputs to
`results/milestone2_3b`, and uses conservative NUTS defaults
`--target-accept-prob 0.99`, `--max-tree-depth 11`, and
`--num-warmup 2000`.

Build maps, spectra, and free-ell diagnostics:

```bash
python examples/luhman16b_yama/make_milestone2_free_t0_cloud_products.py \
  --m2-3b \
  --nside 8 \
  --chip-index 1 \
  --max-map-samples 1000
```

Inspect `free_t0_cloud_diagnostics_chip1.json` for `ell_b` quantiles, degree
conversion, fractions with `ell_b < 0.3`, `ell_b < 0.4`, `ell_b > 0.6`, prior
edge sticking, and correlations with `T0`, `log10 Pc`, `f_cloud`, `sigma_b`,
and `surface_scale`.

## Milestone 2-3c

Milestone 2-3c tests whether the cloud-map resolution is driven by the
`ell_b` prior choice. It runs the same free-`T0` grid retrieval as Milestone
2-3a, but repeats the analysis at fixed `ell_b` values.

Run the fixed-ell sensitivity chains:

```bash
python examples/luhman16b_yama/run_milestone2_fixed_ell_sensitivity.py \
  --nside 8 \
  --chip-index 1 \
  --ell-values 0.25,0.30,0.35,0.40,0.50 \
  --num-warmup 1500 \
  --num-samples 1000 \
  --period-mode fixed \
  --fixed-period 4.83 \
  --sigma-b-scale 0.1 \
  --fix-geometry-to-milestone1
```

Build products for each fixed-ell run:

```bash
for tag in ell0p250 ell0p300 ell0p350 ell0p400 ell0p500; do
  python examples/luhman16b_yama/make_milestone2_free_t0_cloud_products.py \
    --nside 8 \
    --chip-index 1 \
    --samples results/milestone2_3c/mcmc_chip1_fixed_free_t0_cloud_${tag}.npz \
    --out-dir results/milestone2_3c/${tag} \
    --max-map-samples 1000
done
```

Alternatively, pass explicit filenames if the shell expansion is inconvenient.
The expected sample names are
`mcmc_chip1_fixed_free_t0_cloud_ell0p250.npz`,
`mcmc_chip1_fixed_free_t0_cloud_ell0p300.npz`, and so on.

Summarize posterior parameters and residual metrics:

```bash
python examples/luhman16b_yama/summarize_milestone2_ell_sensitivity.py \
  --ell-values 0.25,0.30,0.35,0.40,0.50
```

Use the summary together with the maps to decide whether low `ell_b` values
produce sharper maps without degrading Figure 9 residuals. If low `ell_b`
significantly increases residual RMS, the smooth map is data-driven. If the
residuals are nearly unchanged, the displayed map resolution is prior-choice
sensitive.

## Chip 0-3 runs

The Luhman 16B data loader supports all four CRIRES chips with
`--chip-index 0`, `1`, `2`, or `3`. Milestone 2 profile grids are wavelength
dependent, so each chip needs its own ExoJAX grid. If `--out`,
`--profile-grid`, or `--samples` is omitted, the scripts now choose chip-aware
defaults such as `data/milestone2_t0_cloud_grid_profiles_chip2.npz` and
`mcmc_chip2_fixed_free_t0_cloud.npz`.

Generate the Milestone 2-3 T0/cloud grids for all chips:

```bash
for chip in 0 1 2 3; do
  python examples/luhman16b_yama/generate_milestone2_t0_cloud_grid_profiles.py \
    --chip-index ${chip} \
    --opacity-cache-dir data/opacities/luhman16b_powerlaw \
    --database-dir ~/data_mol/.database \
    --t0-min 1000 \
    --t0-max 1700 \
    --t0-count 15 \
    --log-p-cloud-min -2.0 \
    --log-p-cloud-max 2.0 \
    --log-p-cloud-count 33
done
```

Run the fiducial fixed-`ell_b=0.3` retrieval for all chips:

```bash
for chip in 0 1 2 3; do
  python examples/luhman16b_yama/run_milestone2_free_t0_cloud.py \
    --nside 8 \
    --chip-index ${chip} \
    --num-warmup 2000 \
    --num-samples 1500 \
    --target-accept-prob 0.98 \
    --max-tree-depth 11 \
    --period-mode fixed \
    --fixed-period 4.83 \
    --sigma-b-scale 0.1 \
    --fix-ell-b 0.3 \
    --fix-geometry-to-milestone1 \
    --out-dir results/milestone2_3d_chip${chip}
done
```

Build products with the same chip-aware defaults:

```bash
for chip in 0 1 2 3; do
  python examples/luhman16b_yama/make_milestone2_free_t0_cloud_products.py \
    --nside 8 \
    --chip-index ${chip} \
    --samples results/milestone2_3d_chip${chip}/mcmc_chip${chip}_fixed_free_t0_cloud.npz \
    --out-dir results/milestone2_3d_chip${chip} \
    --max-map-samples 1000
done
```
