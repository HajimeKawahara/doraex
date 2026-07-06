#!/bin/tcsh

set run_name = "syn_m7v1_pressure_spot_fixed_eta_lat30_sigma15_lownoise"
set out_dir = "results/syn/m7v1_pressure_spot_fixed_eta_lat30_sigma15_lownoise"
set log = "logs/log_$run_name"

mkdir -p logs
mkdir -p "$out_dir"
mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /home/kawahara/.cache/jax

setenv PYTHONPATH "$cwd/src"
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /home/kawahara/.cache/jax
setenv XLA_FLAGS "--xla_gpu_autotune_level=0"

echo "[$run_name] start: `date`" | tee "$log"

python examples/luhman16b_yama/syn_m7v1_pressure_spot_fixed_eta.py \
    --samples results/m7/v1_zero_mean_log_w_run/samples.npz \
    --out-dir "$out_dir" \
    --chip-indices 0,1,2,3 \
    --nside 8 \
    --x64 \
    --baseline-p-cloud-bar 20.0 \
    --spot-p-cloud-bar 50.0 \
    --spot-lat-deg 30.0 \
    --spot-lon-deg 0.0 \
    --spot-sigma-deg 15.0 \
    --noise-scale 0.25 \
    --noise-mode posterior \
    >>& "$log"
if ( $status != 0 ) then
    echo "[$run_name] failed: `date`" | tee -a "$log"
    exit 1
endif

echo "[$run_name] done: `date`" | tee -a "$log"
