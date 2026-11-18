#!/usr/bin/env -S tcsh -f

# Run one explicitly selected M9 v3 investigation stage. The Python runner
# owns the shared nonblocking GPU lock and refuses output-directory collisions.
set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

if (! $?DORAEX_PYTHON) setenv DORAEX_PYTHON python
if (! $?EXOJAX_DIR) setenv EXOJAX_DIR "$repo_root:h/exojax"
setenv PYTHONPATH "$EXOJAX_DIR/src:$repo_root/src"
setenv OMP_NUM_THREADS 1
setenv OPENBLAS_NUM_THREADS 1
setenv MKL_NUM_THREADS 1
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
setenv JAX_DEFAULT_MATMUL_PRECISION highest
setenv JAX_COMPILATION_CACHE_DIR /tmp/doraex-jax-cache-m9-v3
# Validated with JAX 0.10.0; avoid embedding multi-GB opacity constants in HLO.
setenv JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS true
setenv XLA_FLAGS "--xla_gpu_autotune_level=0"

"$DORAEX_PYTHON" examples/luhman16b_yama/m9_v3_run.py $argv:q
exit $status
