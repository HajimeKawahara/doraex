#!/bin/tcsh

set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_conflict_status = 75
set gpu_lock_held = 0

if ( $#argv > 0 ) then
    if ( "$argv[1]" == "--m7-gpu-lock-held" ) then
        set gpu_lock_held = 1
    endif
endif

if ( $gpu_lock_held == 0 ) then
    flock --exclusive --nonblock \
        --conflict-exit-code "$lock_conflict_status" \
        "$gpu_lock" \
        /bin/tcsh "$0" --m7-gpu-lock-held
    set lock_status = $status
    if ( $lock_status == $lock_conflict_status ) then
        echo "[m7_p2_v17] not started: another m7 GPU run holds $gpu_lock"
        echo "[m7_p2_v17] wait for it to finish, then retry."
        exit 2
    endif
    exit $lock_status
endif
shift argv

set init = "results/m7/p2/v6/samples.npz"
set log = "logs/log_m7_p2_v17_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_run"
set m7_run_pattern = 'python[0-9.]*[[:space:]].*examples/luhman16b_yama/m7(_p[0-9]+)?_v[0-9]+_run[.]py'

mkdir -p logs
mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /home/kawahara/.cache/jax

setenv PYTHONPATH "$cwd/src"
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /home/kawahara/.cache/jax

if ( ! -e "$init" ) then
    echo "[m7_p2_v17] missing initialization NPZ: $init" | tee "$log"
    exit 1
endif

set active_m7_pids = (`pgrep -f "$m7_run_pattern"`)
if ( $#active_m7_pids > 0 ) then
    echo "[m7_p2_v17] refusing concurrent m7 GPU run; active PID(s): $active_m7_pids"
    pgrep -af "$m7_run_pattern"
    echo "[m7_p2_v17] wait for the active run to finish, then retry."
    exit 2
endif

echo "[m7_p2_v17] start: `date`" | tee "$log"
python examples/luhman16b_yama/m7_p2_v17_run.py \
    --init-from "$init" \
    >>& "$log"
set run_status = $status
if ( $run_status != 0 ) then
    echo "[m7_p2_v17] failed: `date`" | tee -a "$log"
    exit 1
endif
echo "[m7_p2_v17] done: `date`" | tee -a "$log"
