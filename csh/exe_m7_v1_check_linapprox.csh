#!/bin/tcsh

set samples = "results/m7/v1_zero_mean_log_w_run/samples.npz"
set product_dir = "results/m7/v1_zero_mean_log_w_f64_prod/baseline_f64"
set out_dir = "results/m7/v1/check_linapprox"
set log = "logs/log_m7_v1_check_linapprox"

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

echo "[m7_v1_check_linapprox] start: `date`" | tee "$log"
python examples/luhman16b_yama/make_m7_v1_linearization_check.py \
    --samples "$samples" \
    --product-dir "$product_dir" \
    --out-dir "$out_dir" \
    --chip-index 0 \
    --batch-size 16 \
    --phase-indices 0,4,8,12 \
    --x64 \
    >>& "$log"
if ( $status != 0 ) then
    echo "[m7_v1_check_linapprox] failed: `date`" | tee -a "$log"
    exit 1
endif
echo "[m7_v1_check_linapprox] done: `date`" | tee -a "$log"
