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
