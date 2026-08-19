#!/bin/tcsh

# Resolve the repository from this script so relative data/output paths do not
# depend on the caller's working directory.
set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

# Share the established lock with the M7 diagnostics so only one Luhman 16B
# retrieval can use the GPU at a time.
set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_conflict_status = 75
set gpu_lock_held = 0

if ( $#argv > 0 ) then
    if ( "$argv[1]" == "--m8-gpu-lock-held" ) then
        set gpu_lock_held = 1
    endif
endif

if ( $gpu_lock_held == 0 ) then
    flock --exclusive --nonblock \
        --conflict-exit-code "$lock_conflict_status" \
        "$gpu_lock" \
        /bin/tcsh "$script_path" --m8-gpu-lock-held
    set lock_status = $status
    if ( $lock_status == $lock_conflict_status ) then
        echo "[m8_v1] not started: another Luhman 16B run holds $gpu_lock"
        echo "[m8_v1] wait for it to finish, then retry."
        exit 2
    endif
    exit $lock_status
endif
shift argv

set init = "results/m7/p2/v6/samples.npz"
set expected_init_sha256 = "3256e8e10998c521313ab14473ac58b1e8aebb1e567a2593619477c0bfe70f89"
set output = "results/m8/v1/samples.npz"
set diagnostics = "results/m8/v1/diagnostics.json"
set log = "logs/log_m8_v1_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_prod"
set retrieval_run_pattern = 'python[0-9.]*[[:space:]].*examples/luhman16b_yama/m[78](_p[0-9]+)?_v[0-9]+_run[.]py'

mkdir -p logs
mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /tmp/doraex-jax-cache

setenv PYTHONPATH "$cwd/src"
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /tmp/doraex-jax-cache

if ( -e "$output" || -e "$diagnostics" ) then
    echo "[m8_v1] refusing to overwrite completed output in results/m8/v1"
    exit 1
endif

if ( ! -e "$init" ) then
    echo "[m8_v1] missing initialization NPZ: $init" | tee "$log"
    exit 1
endif

set actual_init_sha256 = `sha256sum "$init" | awk '{print $1}'`
if ( "$actual_init_sha256" != "$expected_init_sha256" ) then
    echo "[m8_v1] initialization NPZ SHA256 mismatch: $init" | tee "$log"
    echo "[m8_v1] expected $expected_init_sha256" | tee -a "$log"
    echo "[m8_v1] actual   $actual_init_sha256" | tee -a "$log"
    exit 1
endif

set active_retrieval_pids = (`pgrep -f "$retrieval_run_pattern"`)
if ( $#active_retrieval_pids > 0 ) then
    echo "[m8_v1] refusing concurrent retrieval; active PID(s): $active_retrieval_pids"
    pgrep -af "$retrieval_run_pattern"
    echo "[m8_v1] wait for the active run to finish, then retry."
    exit 2
endif

echo "[m8_v1] start: `date`" | tee "$log"
echo "[m8_v1] repository: $repo_root" | tee -a "$log"
echo "[m8_v1] init SHA256: $actual_init_sha256" | tee -a "$log"
echo "[m8_v1] host: `hostname`" | tee -a "$log"
python --version |& tee -a "$log"
python -c 'import jax, jaxlib, numpyro; print(f"jax={jax.__version__} jaxlib={jaxlib.__version__} numpyro={numpyro.__version__}")' |& tee -a "$log"
echo "[m8_v1] source SHA256:" | tee -a "$log"
sha256sum \
    "$script_path" \
    examples/luhman16b_yama/m8_v1_run.py \
    src/doraex/workflows/on_the_fly_pressure_retrieval.py \
    | tee -a "$log"
python examples/luhman16b_yama/m8_v1_run.py \
    --init-from "$init" \
    >>& "$log"
set run_status = $status
if ( $run_status != 0 ) then
    echo "[m8_v1] failed: `date`" | tee -a "$log"
    exit 1
endif
echo "[m8_v1] done: `date`" | tee -a "$log"
