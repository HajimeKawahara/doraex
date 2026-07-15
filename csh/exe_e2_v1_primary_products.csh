#!/bin/tcsh

set samples = "results/e2/v1_run/samples.npz"
set basis = "results/e2/v1_basis/eigen_response_basis.npz"
set out_dir = "results/e2/v1_primary_products"
set log = "logs/log_e2_v1_primary_products"

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

echo "[e2_v1_primary_products] start: `date`" | tee "$log"
python examples/luhman16b_yama/make_e2_v1_primary_products.py \
    --samples "$samples" \
    --basis "$basis" \
    --out-dir "$out_dir" \
    --x64 \
    >>& "$log"
if ( $status != 0 ) then
    echo "[e2_v1_primary_products] failed: `date`" | tee -a "$log"
    exit 1
endif
echo "[e2_v1_primary_products] done: `date`" | tee -a "$log"
