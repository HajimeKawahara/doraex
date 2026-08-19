#!/bin/tcsh

set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_conflict_status = 75
set gpu_lock_held = 0
set validate_only = 0

if ( $#argv > 0 ) then
    if ( "$argv[1]" == "--m8-v3-long-gpu-lock-held" ) then
        set gpu_lock_held = 1
    else if ( "$argv[1]" == "--validate-only" ) then
        set gpu_lock_held = 1
        set validate_only = 1
    else
        echo "[m8_v3_fixed_long] unsupported argument: $argv[1]"
        exit 2
    endif
endif

if ( $gpu_lock_held == 0 ) then
    flock --exclusive --nonblock \
        --conflict-exit-code "$lock_conflict_status" \
        "$gpu_lock" \
        /bin/tcsh "$script_path" --m8-v3-long-gpu-lock-held
    set lock_status = $status
    if ( $lock_status == $lock_conflict_status ) then
        echo "[m8_v3_fixed_long] not started: another Luhman 16B run holds $gpu_lock"
        echo "[m8_v3_fixed_long] wait for it to finish, then retry."
        exit 2
    endif
    exit $lock_status
endif
shift argv

set init = "results/m7/p2/v6/samples.npz"
set free_wrapper = "examples/luhman16b_yama/m7_p2_v17_run.py"
set free_samples = "results/m7/p2/v17/samples.npz"
set free_diagnostics = "results/m7/p2/v17/diagnostics.json"
set long_free_wrapper = "examples/luhman16b_yama/m8_v1_run.py"
set long_free_launcher = "csh/exe_m8_v1_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_prod.csh"
set long_free_samples = "results/m8/v1/samples.npz"
set long_free_diagnostics = "results/m8/v1/diagnostics.json"
set data_spectra = "data/fainterspectral-fits_6.pickle"
set data_template = "data/posterior_predictive_vsini=0.npz"
set workflow = "src/doraex/workflows/on_the_fly_pressure_retrieval.py"
set forward_source = "src/doraex/spectra/exojax_forward.py"
set fixed_wrapper = "examples/luhman16b_yama/m8_v2_run.py"
set replay_script = "examples/luhman16b_yama/check_m8_v2_fixed_free_initial_point.py"
set output_dir = "results/m8/v3/fixed_seed0"
set replay_output = "$output_dir/initial_point_replay.json"
set log = "$output_dir/run.log"
set provenance = "$output_dir/provenance.sha256"
set retrieval_run_pattern = 'python[0-9.]*[[:space:]].*examples/luhman16b_yama/m[78](_p[0-9]+)?_v[0-9]+_run[.]py'

set expected_init_sha256 = "3256e8e10998c521313ab14473ac58b1e8aebb1e567a2593619477c0bfe70f89"
set expected_free_wrapper_sha256 = "1d280912b78940da0450c97a8a0f12de5954264c0a307be07013499596812c61"
set expected_free_samples_sha256 = "d0dac818908d3180ff0c22cb9ff6b1f4b389351f9c458f08d24963fe636cabc9"
set expected_free_diagnostics_sha256 = "7f4f449fa961a064337b0138edf832d0557d5dfda9882833078d5587148f725d"
set expected_long_free_wrapper_sha256 = "fb17a96b1718d65a680ae6162b010cd7f0fac0a577de275d4c8df8a562312423"
set expected_long_free_launcher_sha256 = "c7d6166173dd82b5f82ab63f22312931ac5f9aa3dafb5d302dbac2a6da720203"
set expected_long_free_samples_sha256 = "aa1bd9bb6bc9e5f146e3377aa62e3adb68c8c9a961812628144c73cd9c3a5083"
set expected_long_free_diagnostics_sha256 = "f45d32470ee996c62b0dd8683190d4fe5eaedddd37d4f95d0ea21003278cf8c4"
set expected_data_spectra_sha256 = "4ce83dcfa2e5b8f4adbac34b075e02eb8175c476bc999fe6ac25bffa36aae362"
set expected_data_template_sha256 = "cb98cc5c5401de09a0f574472ea4c95fab742cefc091c371365c014c5f62284a"
set expected_workflow_sha256 = "582f79edad35a1aa819cf6e619caeaabbac614a1ea173dca25fbbf4f8ce41230"
set expected_forward_sha256 = "63390bbf64f436a5ed3cfd022454b22b6e6377aebe909acb21a32a5be0859a21"
set expected_fixed_wrapper_sha256 = "abb1b8ea4b2d0c6ca52af86035a501519503572255f484bdc472039e5b7faf3d"
set expected_replay_script_sha256 = "ed4268237691c120f759626db3f2400f3fd45bd17f2546ef3644ed423cec53ed"

if ( -e "$output_dir" ) then
    echo "[m8_v3_fixed_long] refusing output collision: $output_dir"
    exit 1
endif

foreach required ( "$init" "$free_wrapper" "$free_samples" "$free_diagnostics" "$long_free_wrapper" "$long_free_launcher" "$long_free_samples" "$long_free_diagnostics" "$data_spectra" "$data_template" "$workflow" "$forward_source" "$fixed_wrapper" "$replay_script" )
    if ( ! -f "$required" ) then
        echo "[m8_v3_fixed_long] missing required input: $required"
        exit 1
    endif
end

set actual_init_sha256 = `sha256sum "$init" | awk '{print $1}'`
set actual_free_wrapper_sha256 = `sha256sum "$free_wrapper" | awk '{print $1}'`
set actual_free_samples_sha256 = `sha256sum "$free_samples" | awk '{print $1}'`
set actual_free_diagnostics_sha256 = `sha256sum "$free_diagnostics" | awk '{print $1}'`
set actual_long_free_wrapper_sha256 = `sha256sum "$long_free_wrapper" | awk '{print $1}'`
set actual_long_free_launcher_sha256 = `sha256sum "$long_free_launcher" | awk '{print $1}'`
set actual_long_free_samples_sha256 = `sha256sum "$long_free_samples" | awk '{print $1}'`
set actual_long_free_diagnostics_sha256 = `sha256sum "$long_free_diagnostics" | awk '{print $1}'`
set actual_data_spectra_sha256 = `sha256sum "$data_spectra" | awk '{print $1}'`
set actual_data_template_sha256 = `sha256sum "$data_template" | awk '{print $1}'`
set actual_workflow_sha256 = `sha256sum "$workflow" | awk '{print $1}'`
set actual_forward_sha256 = `sha256sum "$forward_source" | awk '{print $1}'`
set actual_fixed_wrapper_sha256 = `sha256sum "$fixed_wrapper" | awk '{print $1}'`
set actual_replay_script_sha256 = `sha256sum "$replay_script" | awk '{print $1}'`

if ( "$actual_init_sha256" != "$expected_init_sha256" ) then
    echo "[m8_v3_fixed_long] initialization SHA256 mismatch: $actual_init_sha256"
    exit 1
endif
if ( "$actual_free_wrapper_sha256" != "$expected_free_wrapper_sha256" ) then
    echo "[m8_v3_fixed_long] reused free wrapper SHA256 mismatch: $actual_free_wrapper_sha256"
    exit 1
endif
if ( "$actual_free_samples_sha256" != "$expected_free_samples_sha256" ) then
    echo "[m8_v3_fixed_long] reused free samples SHA256 mismatch: $actual_free_samples_sha256"
    exit 1
endif
if ( "$actual_free_diagnostics_sha256" != "$expected_free_diagnostics_sha256" ) then
    echo "[m8_v3_fixed_long] reused free diagnostics SHA256 mismatch: $actual_free_diagnostics_sha256"
    exit 1
endif
if ( "$actual_long_free_wrapper_sha256" != "$expected_long_free_wrapper_sha256" ) then
    echo "[m8_v3_fixed_long] long free wrapper SHA256 mismatch: $actual_long_free_wrapper_sha256"
    exit 1
endif
if ( "$actual_long_free_launcher_sha256" != "$expected_long_free_launcher_sha256" ) then
    echo "[m8_v3_fixed_long] long free launcher SHA256 mismatch: $actual_long_free_launcher_sha256"
    exit 1
endif
if ( "$actual_long_free_samples_sha256" != "$expected_long_free_samples_sha256" ) then
    echo "[m8_v3_fixed_long] long free samples SHA256 mismatch: $actual_long_free_samples_sha256"
    exit 1
endif
if ( "$actual_long_free_diagnostics_sha256" != "$expected_long_free_diagnostics_sha256" ) then
    echo "[m8_v3_fixed_long] long free diagnostics SHA256 mismatch: $actual_long_free_diagnostics_sha256"
    exit 1
endif
if ( "$actual_data_spectra_sha256" != "$expected_data_spectra_sha256" ) then
    echo "[m8_v3_fixed_long] spectra input SHA256 mismatch: $actual_data_spectra_sha256"
    exit 1
endif
if ( "$actual_data_template_sha256" != "$expected_data_template_sha256" ) then
    echo "[m8_v3_fixed_long] template input SHA256 mismatch: $actual_data_template_sha256"
    exit 1
endif
if ( "$actual_workflow_sha256" != "$expected_workflow_sha256" ) then
    echo "[m8_v3_fixed_long] frozen workflow SHA256 mismatch: $actual_workflow_sha256"
    exit 1
endif
if ( "$actual_forward_sha256" != "$expected_forward_sha256" ) then
    echo "[m8_v3_fixed_long] frozen forward source SHA256 mismatch: $actual_forward_sha256"
    exit 1
endif
if ( "$actual_fixed_wrapper_sha256" != "$expected_fixed_wrapper_sha256" ) then
    echo "[m8_v3_fixed_long] fixed wrapper SHA256 mismatch: $actual_fixed_wrapper_sha256"
    exit 1
endif
if ( "$actual_replay_script_sha256" != "$expected_replay_script_sha256" ) then
    echo "[m8_v3_fixed_long] replay script SHA256 mismatch: $actual_replay_script_sha256"
    exit 1
endif

if ( $validate_only == 1 ) then
    echo "[m8_v3_fixed_long] launcher validation passed; no GPU work was started"
    exit 0
endif

set active_retrieval_pids = (`pgrep -f "$retrieval_run_pattern"`)
if ( $#active_retrieval_pids > 0 ) then
    echo "[m8_v3_fixed_long] refusing concurrent retrieval; active PID(s): $active_retrieval_pids"
    pgrep -af "$retrieval_run_pattern"
    exit 2
endif

mkdir -p "$output_dir"
touch "$output_dir/RUNNING"

mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /tmp/doraex-jax-cache

setenv PYTHONPATH "$cwd/src"
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /tmp/doraex-jax-cache

echo "[m8_v3_fixed_long] start: `date`" | tee "$log"
echo "[m8_v3_fixed_long] repository: $repo_root" | tee -a "$log"
echo "[m8_v3_fixed_long] conditional intervention: fixed target omits the free sigma_log_p dimension" | tee -a "$log"
echo "[m8_v3_fixed_long] schedule: 2000 warmup, 1500 retained; comparator: existing M8 v1 free long arm" | tee -a "$log"
echo "[m8_v3_fixed_long] git HEAD: `git rev-parse HEAD`" | tee -a "$log"
echo "[m8_v3_fixed_long] host: `hostname`" | tee -a "$log"
python --version |& tee -a "$log"
python -c 'import jax, jaxlib, numpyro; print(f"jax={jax.__version__} jaxlib={jaxlib.__version__} numpyro={numpyro.__version__}")' |& tee -a "$log"

sha256sum \
    "$script_path" \
    "$fixed_wrapper" \
    "$replay_script" \
    "$workflow" \
    "$forward_source" \
    "$init" \
    "$free_wrapper" \
    "$free_samples" \
    "$free_diagnostics" \
    "$long_free_wrapper" \
    "$long_free_launcher" \
    "$long_free_samples" \
    "$long_free_diagnostics" \
    "$data_spectra" \
    "$data_template" \
    | tee "$provenance" | tee -a "$log"

echo "[m8_v3_fixed_long] initial-point replay start: `date`" | tee -a "$log"
# This guard verifies the target and common initial point. The long schedule is
# supplied explicitly to the sampling command below.
python "$replay_script" \
    --init-from "$init" \
    --output "$replay_output" \
    --seed 0 \
    >>& "$log"
set replay_status = $status
if ( $replay_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    echo "[m8_v3_fixed_long] initial-point replay failed: `date`" | tee -a "$log"
    exit $replay_status
endif
touch "$output_dir/INITIAL_POINT_VALIDATED"
echo "[m8_v3_fixed_long] initial-point replay passed: `date`" | tee -a "$log"

python "$fixed_wrapper" \
    --init-from "$init" \
    --out-dir "$output_dir" \
    --run-label "m8_v3_fixed_sigma_log_p_v17_initial_full_dense_long_control" \
    --fixed-sigma-log-p 0.32316479086875916 \
    --num-chains 1 \
    --num-warmup 2000 \
    --num-samples 1500 \
    --seed 0 \
    --target-accept-prob 0.95 \
    --warmup-max-tree-depth 9 \
    --max-tree-depth 11 \
    --dense-mass \
    --adapt-mass-matrix \
    --no-x64 \
    --print-summary \
    >>& "$log"
set run_status = $status
if ( $run_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    echo "[m8_v3_fixed_long] sampling failed: `date`" | tee -a "$log"
    exit $run_status
endif

if ( ! -f "$output_dir/samples.npz" || ! -f "$output_dir/diagnostics.json" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    echo "[m8_v3_fixed_long] sampling returned without complete outputs" | tee -a "$log"
    exit 1
endif

mv "$output_dir/RUNNING" "$output_dir/COMPLETE"
echo "[m8_v3_fixed_long] done: `date`" | tee -a "$log"
