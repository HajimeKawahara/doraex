#!/usr/bin/env -S tcsh -f

set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

set output_dir = "results/m8/v7/free_sigma_derivative_capture"
set nohup_parent = "results/m8/v7/nohup"
set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_conflict_status = 75
set gpu_lock_held = 0
set validate_only = 0
set submit_nohup = 0

if ( $#argv > 1 ) then
    echo "[m8_v7_sigma_capture] expected no argument, --validate-only, or --nohup"
    exit 2
endif
if ( $#argv == 1 ) then
    if ( "$argv[1]" == "--m8-v7-sigma-capture-gpu-lock-held" ) then
        set gpu_lock_held = 1
    else if ( "$argv[1]" == "--validate-only" ) then
        set gpu_lock_held = 1
        set validate_only = 1
    else if ( "$argv[1]" == "--nohup" ) then
        set submit_nohup = 1
    else
        echo "[m8_v7_sigma_capture] unsupported argument: $argv[1]"
        exit 2
    endif
endif

if ( $submit_nohup == 1 ) then
    if ( -e "$output_dir" ) then
        echo "[m8_v7_sigma_capture] refusing output collision: $output_dir"
        exit 1
    endif
    mkdir -p "$nohup_parent"
    if ( $status != 0 ) then
        echo "[m8_v7_sigma_capture] failed to create nohup log parent: $nohup_parent"
        exit 1
    endif
    set outer_log = `mktemp "$nohup_parent/m8_v7_sigma_capture_nohup.XXXXXX.log"`
    set mktemp_status = $status
    if ( $mktemp_status != 0 || "$outer_log" == "" ) then
        echo "[m8_v7_sigma_capture] failed to claim a unique nohup log"
        exit 1
    endif
    set outer_log = `realpath "$outer_log"`
    setenv DORAEX_M8_V7_OUTER_LOG "$outer_log"
    /usr/bin/nohup "$script_path" < /dev/null >&! "$outer_log" &
    set submit_status = $status
    set submitted_pid = $!
    unsetenv DORAEX_M8_V7_OUTER_LOG
    if ( $submit_status != 0 ) then
        echo "[m8_v7_sigma_capture] nohup submission failed with status $submit_status"
        exit $submit_status
    endif
    echo "[m8_v7_sigma_capture] submitted PID: $submitted_pid"
    echo "[m8_v7_sigma_capture] outer log: $outer_log"
    echo "[m8_v7_sigma_capture] run log: $repo_root/$output_dir/run.log"
    exit 0
endif

if ( $gpu_lock_held == 0 ) then
    flock --exclusive --nonblock \
        --conflict-exit-code "$lock_conflict_status" \
        "$gpu_lock" \
        "$script_path" --m8-v7-sigma-capture-gpu-lock-held
    set lock_status = $status
    if ( $lock_status == $lock_conflict_status ) then
        echo "[m8_v7_sigma_capture] not started: another Luhman 16B run holds $gpu_lock"
        exit 2
    endif
    exit $lock_status
endif

set diagnostic_script = "examples/luhman16b_yama/check_m8_v7_free_sigma_derivative_capture.py"
set v6_script = "examples/luhman16b_yama/check_m8_v6_free_sigma_hvp_geometry.py"
set v6_launcher = "csh/exe_m8_v6_check_free_sigma_hvp_geometry.csh"
set v6_dir = "results/m8/v6/free_sigma_hvp_geometry"
set v6_log = "$v6_dir/run.log"
set v6_provenance = "$v6_dir/provenance.sha256"
set v6_failed = "$v6_dir/FAILED"
set v5_script = "examples/luhman16b_yama/check_m8_v5_free_sigma_geometry.py"
set v5_dir = "results/m8/v5/free_sigma_geometry"
set v5_summary = "$v5_dir/m8_v5_free_sigma_geometry_summary.json"
set v5_arrays = "$v5_dir/m8_v5_free_sigma_geometry_arrays.npz"
set v5_failed = "$v5_dir/FAILED"
set init_archive = "results/m7/p2/v6/samples.npz"
set free_wrapper = "examples/luhman16b_yama/m8_v1_run.py"
set free_launcher = "csh/exe_m8_v1_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_prod.csh"
set free_samples = "results/m8/v1/samples.npz"
set free_diagnostics = "results/m8/v1/diagnostics.json"
set fixed_wrapper = "examples/luhman16b_yama/m8_v2_run.py"
set replay_helper = "examples/luhman16b_yama/check_m8_v2_fixed_free_initial_point.py"
set fixed_launcher = "csh/exe_m8_v3_fixed_sigma_log_p_v17_initial_full_dense_long_control.csh"
set fixed_samples = "results/m8/v3/fixed_seed0/samples.npz"
set fixed_diagnostics = "results/m8/v3/fixed_seed0/diagnostics.json"
set initial_replay = "results/m8/v3/fixed_seed0/initial_point_replay.json"
set data_spectra = "data/fainterspectral-fits_6.pickle"
set data_template = "data/posterior_predictive_vsini=0.npz"
set workflow = "src/doraex/workflows/on_the_fly_pressure_retrieval.py"
set forward_source = "src/doraex/spectra/exojax_forward.py"
set chip_data_source = "src/doraex/data/luhman16b.py"
set healpix_source = "src/doraex/geometry/healpix.py"
set gp_source = "src/doraex/priors/spherical_gp.py"
set design_source = "src/doraex/operators/design_matrix.py"
set doppler_source = "src/doraex/operators/doppler.py"
set rotation_source = "src/doraex/geometry/rotation.py"
set limb_source = "src/doraex/geometry/limb_darkening.py"
set marginal_likelihood_source = "src/doraex/inference/marginal_likelihood.py"
set constants_source = "src/doraex/constants.py"
set database_dir = "/home/kawahara/data_mol/.database"
set opacity_cache_dir = "data/opacities/luhman16b_powerlaw"

set summary = "$output_dir/m8_v7_free_sigma_derivative_capture_summary.json"
set arrays = "$output_dir/m8_v7_free_sigma_derivative_capture_arrays.npz"
set checkpoint = "$output_dir/m8_v7_free_sigma_derivative_capture_checkpoint.json"
set log = "$output_dir/run.log"
set provenance = "$output_dir/provenance.sha256"
set provenance_tmp = "$provenance.tmp"
set output_hashes = "$output_dir/outputs.sha256"
set output_hashes_tmp = "$output_hashes.tmp"
set failure_hashes = "$output_dir/failure_outputs.sha256"
set failure_hashes_tmp = "$failure_hashes.tmp"

set expected_jax_version = "0.10.0"
set expected_jax_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/jax/__init__.py"
set expected_jaxlib_version = "0.10.0"
set expected_jaxlib_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/jaxlib/__init__.py"
set expected_exojax_version = "2.2.dev271+ga4c8f69b6.d20260429"
set expected_exojax_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/exojax/__init__.py"
set expected_numpyro_version = "0.21.0"
set expected_numpyro_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/__init__.py"
set numpyro_continuous_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/distributions/continuous.py"
set expected_numpy_version = "2.1.0"
set expected_numpy_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpy/__init__.py"
set expected_scipy_version = "1.17.1"
set expected_scipy_path = "/home/kawahara/miniconda3/lib/python3.12/site-packages/scipy/__init__.py"
set scipy_linalg_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/scipy/linalg/__init__.py"
set numpyro_hmc_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/infer/hmc_util.py"
set numpyro_initialization_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/infer/initialization.py"
set numpyro_util_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/infer/util.py"
set numpyro_transforms_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/numpyro/distributions/transforms.py"
set gpu_python_pattern = 'python[0-9.]*[[:space:]].*examples/luhman16b_yama/(m[78](_p[0-9]+)?_v[0-9]+_run|run_milestone5_on_the_fly_atmosphere|make_m7_v1_linearization_check|check_m8_.*(linearization|convergence|validity|geometry|hvp|derivative_capture))[.]py'

set required_files = ( \
    "$diagnostic_script" \
    "$v6_script" \
    "$v6_launcher" \
    "$v6_log" \
    "$v6_provenance" \
    "$v6_failed" \
    "$v5_script" \
    "$v5_summary" \
    "$v5_arrays" \
    "$v5_failed" \
    "$init_archive" \
    "$free_wrapper" \
    "$free_launcher" \
    "$free_samples" \
    "$free_diagnostics" \
    "$fixed_wrapper" \
    "$replay_helper" \
    "$fixed_launcher" \
    "$fixed_samples" \
    "$fixed_diagnostics" \
    "$initial_replay" \
    "$data_spectra" \
    "$data_template" \
    "$workflow" \
    "$forward_source" \
    "$chip_data_source" \
    "$healpix_source" \
    "$gp_source" \
    "$design_source" \
    "$doppler_source" \
    "$rotation_source" \
    "$limb_source" \
    "$marginal_likelihood_source" \
    "$constants_source" \
    "$expected_exojax_path" \
    "$expected_jax_path" \
    "$expected_jaxlib_path" \
    "$expected_numpyro_path" \
    "$numpyro_continuous_source" \
    "$expected_numpy_path" \
    "$expected_scipy_path" \
    "$scipy_linalg_source" \
    "$numpyro_hmc_source" \
    "$numpyro_initialization_source" \
    "$numpyro_util_source" \
    "$numpyro_transforms_source" \
)
set expected_hashes = ( \
    "bff3238f343f6610e306b8b92455cd1faec3541d6896bf02e8dea10cbbe9d5fa" \
    "b42306a436288f8d578c3a6f07d47582d19b1b756a58b6bb66f6b74b90000a1e" \
    "28d3a9d78f719a67823331f8a18e7ad1e3073d10035c92e7adb727e08b03c7fd" \
    "43be359490eaa5fc9f0387441c46203aedc1faa0474563a5242f24f1c83d1287" \
    "b794fd7d631287f3e5d8a5089736059add66ac0611bd34e2f0a9d30fe90f612d" \
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" \
    "b1e1cd60e10410d32e5921812f1f9036ede5d7d1b73141818748a42e2b32dfc5" \
    "6aa1800d331d906f4aaabb406a03b6b9957dd4159c6e2d599eba17243028d8c5" \
    "ab48316ecfd12981a33579ff7dd12d4c1d8c85f08fa638135c2289aaa9d8a2c3" \
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" \
    "3256e8e10998c521313ab14473ac58b1e8aebb1e567a2593619477c0bfe70f89" \
    "fb17a96b1718d65a680ae6162b010cd7f0fac0a577de275d4c8df8a562312423" \
    "c7d6166173dd82b5f82ab63f22312931ac5f9aa3dafb5d302dbac2a6da720203" \
    "aa1bd9bb6bc9e5f146e3377aa62e3adb68c8c9a961812628144c73cd9c3a5083" \
    "f45d32470ee996c62b0dd8683190d4fe5eaedddd37d4f95d0ea21003278cf8c4" \
    "abb1b8ea4b2d0c6ca52af86035a501519503572255f484bdc472039e5b7faf3d" \
    "ed4268237691c120f759626db3f2400f3fd45bd17f2546ef3644ed423cec53ed" \
    "8f8a8fb259819e8de761d13d7205958d93b9e3eb6d74b2f22a76a5daae8fef7c" \
    "3d9e052befbef2541954be71aed5efdb9ad7cb2bcf9d910f0520cba0f5fbc6d2" \
    "8de84b82f08222b3070551ad85222e643573ce1d055db9b8108a6f2a11de1518" \
    "4e535403123ccb92e9ed4d0cc514fab8620035fc913fafca302f22266859fde1" \
    "4ce83dcfa2e5b8f4adbac34b075e02eb8175c476bc999fe6ac25bffa36aae362" \
    "cb98cc5c5401de09a0f574472ea4c95fab742cefc091c371365c014c5f62284a" \
    "582f79edad35a1aa819cf6e619caeaabbac614a1ea173dca25fbbf4f8ce41230" \
    "63390bbf64f436a5ed3cfd022454b22b6e6377aebe909acb21a32a5be0859a21" \
    "c3fe182a9418c16d5c8a010a326869b36201c0fc048714379ab6590f3d020c44" \
    "59c4bee2b82a5b4f72fb47b82645290a8d90f438e0c505f67d93d76455a41719" \
    "8272b2722a7028bdd2a1fb56fde22b36b88ae77bafa060915baff61e47107e62" \
    "0f65ee811f4218fb66fd881cd0906ab5675e7a8f28585a8387406e60512fb45d" \
    "7f1d900c2cc14e14b7e3251d1d5f66b9f746c7c3b8cb7130079f5e2ec92b6637" \
    "fe87269df5ef51846f3fc2a51081e079efbb6fc321c575cf2facae1b49d8c066" \
    "5e42a01f1806fb2f21920de315ce1b69cf80bafa568e61fbbd0326ccef1311f3" \
    "d12f573aefeba9f0275d0bce5f872e42d717b016a7fad14216e8663f307096f3" \
    "c89ce9edaa238fece7c687e429a538c90126307afd31aca7a4e7854e38ec8fa9" \
    "f8bef220648b129cd61146b0df2a0262e1ae82b0a3be3d454b4e26a41dadb303" \
    "8317d2884580454b67ec3342bea57c3f3e83c9027702ee682e410ddb09a9cad2" \
    "2a5b37b2bc9802769f45ca43de7cdc1a8b532f0afd2bf9542131ab68f649c4a3" \
    "a76bb4e041234be9bd16a2cd968e7ed9fae472f7ab2e2486e117052a3c78400d" \
    "e781bc57cefd43bf91990c4b01b8e9c53da7001641a5c8f6cdc0d38f62f85fc8" \
    "39c42db027548f958e096e8babe3fa0e3e773d24aa39eb6363fc0e3abbec34b1" \
    "335bc6e0a9909dc7534f9569a3685a92dc8001cb8c63a6da4c239849ff02d4d0" \
    "8840662d267e8d06503125608c6333e5f50053750f1e2c273b63e5a6c0b79c1c" \
    "37eefae77a9a7df97c460c4cfbfdba34eb7b88a3a3a403e8936ff75a2bafb031" \
    "0c5bd0e8ab55ada98e029ba2f49edbc611d5fd0b60ae607e49a3cf7733b8a9b3" \
    "9e73fafb9ce8f072c431edd8297f5dadc07d7edc5a2a769b756a0f3cef0a9149" \
    "7932fa3f2bb4f68f6db9f8ad3550a45dd2a3f3cbb468bfb4e18748d897472a43" \
)

if ( -e "$output_dir" ) then
    echo "[m8_v7_sigma_capture] refusing output collision: $output_dir"
    exit 1
endif
if ( $#required_files != $#expected_hashes ) then
    echo "[m8_v7_sigma_capture] internal provenance list mismatch"
    exit 1
endif
foreach required ( $required_files )
    if ( ! -f "$required" ) then
        echo "[m8_v7_sigma_capture] missing required input: $required"
        exit 1
    endif
end
if ( -e "$v6_dir/COMPLETE" || -e "$v6_dir/m8_v6_free_sigma_hvp_geometry_summary.json" || -e "$v6_dir/m8_v6_free_sigma_hvp_geometry_arrays.npz" ) then
    echo "[m8_v7_sigma_capture] v6 predecessor is no longer the pinned artifact-less failure"
    exit 1
endif
if ( ! -d "$database_dir" || ! -d "$opacity_cache_dir" ) then
    echo "[m8_v7_sigma_capture] molecular database or opacity cache is missing"
    exit 1
endif

@ hash_index = 1
while ( $hash_index <= $#required_files )
    set actual_hash = `sha256sum "$required_files[$hash_index]" | awk '{print $1}'`
    if ( "$actual_hash" != "$expected_hashes[$hash_index]" ) then
        echo "[m8_v7_sigma_capture] SHA256 mismatch: $required_files[$hash_index]"
        echo "[m8_v7_sigma_capture] expected $expected_hashes[$hash_index]"
        echo "[m8_v7_sigma_capture] actual   $actual_hash"
        exit 1
    endif
    @ hash_index++
end

setenv PYTHONPATH "$repo_root/src"
python -c 'import inspect, os, sys, jax, jaxlib, exojax, numpyro, numpy, scipy; import scipy.linalg as scipy_linalg; import numpyro.distributions.continuous as continuous; import numpyro.distributions.transforms as transforms; import numpyro.infer.hmc_util as hmc_util; import numpyro.infer.initialization as initialization; import numpyro.infer.util as infer_util; modules=[jax,jaxlib,exojax,numpyro,continuous,numpy,scipy,scipy_linalg,hmc_util,initialization,infer_util,transforms]; actual=[os.path.realpath(inspect.getfile(module)) for module in modules]; expected=[os.path.realpath(value) for value in sys.argv[7:19]]; versions=[jax.__version__,jaxlib.__version__,exojax.__version__,numpyro.__version__,numpy.__version__,scipy.__version__]; wanted=list(sys.argv[1:7]); print(f"versions={versions} paths={actual}"); sys.exit(0 if versions == wanted and actual == expected else 4)' "$expected_jax_version" "$expected_jaxlib_version" "$expected_exojax_version" "$expected_numpyro_version" "$expected_numpy_version" "$expected_scipy_version" "$expected_jax_path" "$expected_jaxlib_path" "$expected_exojax_path" "$expected_numpyro_path" "$numpyro_continuous_source" "$expected_numpy_path" "$expected_scipy_path" "$scipy_linalg_source" "$numpyro_hmc_source" "$numpyro_initialization_source" "$numpyro_util_source" "$numpyro_transforms_source"
set package_status = $status
if ( $package_status != 0 ) then
    echo "[m8_v7_sigma_capture] frozen package validation failed"
    exit $package_status
endif

set diagnostic_args = ( \
    --init-from "$init_archive" \
    --free-samples "$free_samples" \
    --free-diagnostics "$free_diagnostics" \
    --fixed-samples "$fixed_samples" \
    --fixed-diagnostics "$fixed_diagnostics" \
    --initial-replay "$initial_replay" \
    --v5-summary "$v5_summary" \
    --v5-arrays "$v5_arrays" \
    --v5-failed "$v5_failed" \
    --v6-dir "$v6_dir" \
    --out-dir "$output_dir" \
    --seed 0 \
    --evaluation-sigma 0.27526917 \
    --u-step 0.02 \
    --repeat-count 5 \
    --gram-chunk-size 4096 \
    --potential-atol 0.05 \
    --no-x64 \
)

python "$diagnostic_script" $diagnostic_args --validate-only
set validation_status = $status
if ( $validation_status != 0 ) then
    echo "[m8_v7_sigma_capture] Python validation failed with status $validation_status"
    exit $validation_status
endif
if ( $validate_only == 1 ) then
    echo "[m8_v7_sigma_capture] launcher validation passed; no GPU work was started"
    echo "[m8_v7_sigma_capture] one-line submission: $script_path --nohup"
    echo "[m8_v7_sigma_capture] planned output: $output_dir"
    echo "[m8_v7_sigma_capture] planned run log: $log"
    exit 0
endif

set active_gpu_pids = (`pgrep -f "$gpu_python_pattern"`)
if ( $#active_gpu_pids > 0 ) then
    echo "[m8_v7_sigma_capture] refusing concurrent GPU work; active PID(s): $active_gpu_pids"
    pgrep -af "$gpu_python_pattern"
    exit 2
endif

set output_parent = "$output_dir:h"
mkdir -p "$output_parent"
if ( $status != 0 ) then
    echo "[m8_v7_sigma_capture] failed to create output parent: $output_parent"
    exit 1
endif
mkdir "$output_dir"
if ( $status != 0 ) then
    echo "[m8_v7_sigma_capture] failed to claim fresh output directory: $output_dir"
    exit 1
endif
touch "$output_dir/RUNNING"
if ( $status != 0 ) then
    echo "[m8_v7_sigma_capture] failed to create RUNNING sentinel"
    exit 1
endif

mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /tmp/doraex-jax-cache-m8-v7-sigma-capture
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /tmp/doraex-jax-cache-m8-v7-sigma-capture
setenv PYTHONUNBUFFERED 1
if ( $?XLA_FLAGS ) unsetenv XLA_FLAGS
setenv XLA_PYTHON_CLIENT_PREALLOCATE false

echo "[m8_v7_sigma_capture] start: `date`" | tee "$log"
echo "[m8_v7_sigma_capture] log: $log" | tee -a "$log"
if ( $?DORAEX_M8_V7_OUTER_LOG ) then
    echo "[m8_v7_sigma_capture] outer nohup log: $DORAEX_M8_V7_OUTER_LOG" | tee -a "$log"
endif
echo "[m8_v7_sigma_capture] repository: $repo_root" | tee -a "$log"
echo "[m8_v7_sigma_capture] git HEAD: `git rev-parse HEAD`" | tee -a "$log"
echo "[m8_v7_sigma_capture] host: `hostname`" | tee -a "$log"
echo "[m8_v7_sigma_capture] XLA autotune: production default (XLA_FLAGS unset)" | tee -a "$log"
echo "[m8_v7_sigma_capture] method: one marker; raw NaN capture; no HVP fallback" | tee -a "$log"
python --version |& tee -a "$log"
git status --short --untracked-files=normal >>& "$log"

python -c 'import inspect, os, sys, jax, jaxlib, exojax, numpyro, numpy, scipy; import scipy.linalg as scipy_linalg; import numpyro.distributions.continuous as continuous; import numpyro.distributions.transforms as transforms; import numpyro.infer.hmc_util as hmc_util; import numpyro.infer.initialization as initialization; import numpyro.infer.util as infer_util; backend=jax.default_backend(); modules=[jax,jaxlib,exojax,numpyro,continuous,numpy,scipy,scipy_linalg,hmc_util,initialization,infer_util,transforms]; actual=[os.path.realpath(inspect.getfile(module)) for module in modules]; expected=[os.path.realpath(value) for value in sys.argv[7:19]]; versions=[jax.__version__,jaxlib.__version__,exojax.__version__,numpyro.__version__,numpy.__version__,scipy.__version__]; wanted=list(sys.argv[1:7]); print(f"versions={versions} paths={actual} backend={backend} devices={jax.devices()}"); sys.exit(0 if backend == "gpu" and versions == wanted and actual == expected else 3)' "$expected_jax_version" "$expected_jaxlib_version" "$expected_exojax_version" "$expected_numpyro_version" "$expected_numpy_version" "$expected_scipy_version" "$expected_jax_path" "$expected_jaxlib_path" "$expected_exojax_path" "$expected_numpyro_path" "$numpyro_continuous_source" "$expected_numpy_path" "$expected_scipy_path" "$scipy_linalg_source" "$numpyro_hmc_source" "$numpyro_initialization_source" "$numpyro_util_source" "$numpyro_transforms_source" >>& "$log"
set backend_status = $status
tail -n 1 "$log"
if ( $backend_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v7_sigma_capture] GPU/package preflight failed; refusing fallback" | tee -a "$log"
    exit $backend_status
endif

sha256sum "$script_path" $required_files > "$provenance_tmp"
set provenance_status = $status
if ( $provenance_status != 0 || ! -s "$provenance_tmp" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v7_sigma_capture] failed to write provenance hashes" | tee -a "$log"
    if ( $provenance_status == 0 ) set provenance_status = 1
    exit $provenance_status
endif
mv "$provenance_tmp" "$provenance"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif
sha256sum -c "$provenance" >>& "$log"
set provenance_check_status = $status
if ( $provenance_check_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v7_sigma_capture] provenance self-check failed" | tee -a "$log"
    exit $provenance_check_status
endif

echo "[m8_v7_sigma_capture] diagnostic start: `date`" | tee -a "$log"
python -u "$diagnostic_script" $diagnostic_args >>& "$log"
set run_status = $status
if ( $run_status != 0 ) then
    if ( -f "$arrays" && -f "$checkpoint" ) then
        python -c 'import json, sys; from pathlib import Path; from examples.luhman16b_yama import check_m8_v7_free_sigma_derivative_capture as capture; result=capture.validate_checkpoint_artifacts(Path(sys.argv[1]),Path(sys.argv[2])); print(json.dumps(result,sort_keys=True))' "$arrays" "$checkpoint" >>& "$log"
        set checkpoint_status = $status
        if ( $checkpoint_status == 0 ) then
            if ( -f "$summary" ) then
                sha256sum "$summary" "$arrays" "$checkpoint" > "$failure_hashes_tmp"
            else
                sha256sum "$arrays" "$checkpoint" > "$failure_hashes_tmp"
            endif
            if ( $status == 0 && -s "$failure_hashes_tmp" ) then
                mv "$failure_hashes_tmp" "$failure_hashes"
                sha256sum -c "$failure_hashes" >>& "$log"
            endif
        endif
    endif
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v7_sigma_capture] diagnostic failed with status ${run_status}: `date`" | tee -a "$log"
    exit $run_status
endif

if ( ! -f "$summary" || ! -f "$arrays" || ! -f "$checkpoint" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v7_sigma_capture] diagnostic returned without complete artifacts" | tee -a "$log"
    exit 1
endif

python -c 'import json, sys; from pathlib import Path; from examples.luhman16b_yama import check_m8_v7_free_sigma_derivative_capture as capture; result=capture.validate_saved_artifacts(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])); print(json.dumps(result,sort_keys=True))' "$summary" "$arrays" "$checkpoint" >>& "$log"
set artifact_status = $status
if ( $artifact_status != 0 ) then
    sha256sum "$summary" "$arrays" "$checkpoint" > "$failure_hashes_tmp"
    if ( $status == 0 && -s "$failure_hashes_tmp" ) mv "$failure_hashes_tmp" "$failure_hashes"
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v7_sigma_capture] artifact validation failed" | tee -a "$log"
    exit $artifact_status
endif

sha256sum "$summary" "$arrays" "$checkpoint" > "$output_hashes_tmp"
set output_hash_status = $status
if ( $output_hash_status != 0 || ! -s "$output_hashes_tmp" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v7_sigma_capture] failed to write output hashes" | tee -a "$log"
    if ( $output_hash_status == 0 ) set output_hash_status = 1
    exit $output_hash_status
endif
mv "$output_hashes_tmp" "$output_hashes"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif
sha256sum -c "$output_hashes" >>& "$log"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v7_sigma_capture] output hash self-check failed" | tee -a "$log"
    exit 1
endif

mv "$output_dir/RUNNING" "$output_dir/COMPLETE"
if ( $status != 0 ) exit 1
touch "$output_dir/COMPLETE"
echo "[m8_v7_sigma_capture] done: `date`" | tee -a "$log"
