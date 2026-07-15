#!/bin/tcsh

set log = "logs/log_e2_v1_basis"

mkdir -p logs
mkdir -p results/e2/v1_basis
mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /home/kawahara/.cache/jax

setenv PYTHONPATH "$cwd/src"
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /home/kawahara/.cache/jax
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv TF_GPU_ALLOCATOR cuda_malloc_async
setenv XLA_FLAGS "--xla_gpu_autotune_level=0"

echo "[e2_v1_basis] start: `date`" | tee "$log"
python examples/luhman16b_yama/generate_e2_v1_basis.py \
    >>& "$log"
if ( $status != 0 ) then
    echo "[e2_v1_basis] failed: `date`" | tee -a "$log"
    exit 1
endif
echo "[e2_v1_basis] done: `date`" | tee -a "$log"
