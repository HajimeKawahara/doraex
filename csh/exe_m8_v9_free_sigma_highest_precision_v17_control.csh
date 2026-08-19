#!/usr/bin/env -S tcsh -f

set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

set output_dir = "results/m8/v9/free_sigma_highest_precision_seed0"
set nohup_parent = "results/m8/v9/nohup"
set gpu_lock = "/tmp/doraex_m7_gpu.lock"
set lock_conflict_status = 75
set gpu_lock_held = 0
set validate_only = 0
set submit_nohup = 0

if ( $#argv > 1 ) then
    echo "[m8_v9_highest_free] expected no argument, --validate-only, or --nohup"
    exit 2
endif
if ( $#argv == 1 ) then
    if ( "$argv[1]" == "--m8-v9-highest-free-gpu-lock-held" ) then
        set gpu_lock_held = 1
    else if ( "$argv[1]" == "--validate-only" ) then
        set gpu_lock_held = 1
        set validate_only = 1
    else if ( "$argv[1]" == "--nohup" ) then
        set submit_nohup = 1
    else
        echo "[m8_v9_highest_free] unsupported argument: $argv[1]"
        exit 2
    endif
endif

if ( $submit_nohup == 1 ) then
    if ( -e "$output_dir" ) then
        echo "[m8_v9_highest_free] refusing output collision: $output_dir"
        exit 1
    endif
    mkdir -p "$nohup_parent"
    if ( $status != 0 ) then
        echo "[m8_v9_highest_free] failed to create nohup log parent"
        exit 1
    endif
    set outer_log = `mktemp "$nohup_parent/m8_v9_highest_free_nohup.XXXXXX.log"`
    set mktemp_status = $status
    if ( $mktemp_status != 0 || "$outer_log" == "" ) then
        echo "[m8_v9_highest_free] failed to claim a unique nohup log"
        exit 1
    endif
    set outer_log = `realpath "$outer_log"`
    setenv DORAEX_M8_V9_OUTER_LOG "$outer_log"
    /usr/bin/nohup "$script_path" < /dev/null >&! "$outer_log" &
    set submit_status = $status
    set submitted_pid = $!
    unsetenv DORAEX_M8_V9_OUTER_LOG
    if ( $submit_status != 0 ) then
        echo "[m8_v9_highest_free] nohup submission failed with status $submit_status"
        exit $submit_status
    endif
    echo "[m8_v9_highest_free] submitted PID: $submitted_pid"
    echo "[m8_v9_highest_free] outer log: $outer_log"
    echo "[m8_v9_highest_free] run log: $repo_root/$output_dir/run.log"
    exit 0
endif

if ( $gpu_lock_held == 0 ) then
    flock --exclusive --nonblock \
        --conflict-exit-code "$lock_conflict_status" \
        "$gpu_lock" \
        "$script_path" --m8-v9-highest-free-gpu-lock-held
    set lock_status = $status
    if ( $lock_status == $lock_conflict_status ) then
        echo "[m8_v9_highest_free] not started: another Luhman 16B run holds $gpu_lock"
        exit 2
    endif
    exit $lock_status
endif

set control_script = "examples/luhman16b_yama/check_m8_v9_free_sigma_highest_precision_control.py"
set control_test = "tests/unittests/test_m8_free_sigma_highest_precision_control.py"
set sampler = "examples/luhman16b_yama/m8_v1_run.py"
set sampler_launcher = "csh/exe_m8_v1_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_prod.csh"
set v17_wrapper = "examples/luhman16b_yama/m7_p2_v17_run.py"
set v17_launcher = "csh/exe_m7_p2_v17_free_a_fixed_sigma_d_direct_sigma_log_p_full_dense_run.csh"
set fixed_wrapper = "examples/luhman16b_yama/m8_v2_run.py"
set replay_helper = "examples/luhman16b_yama/check_m8_v2_fixed_free_initial_point.py"
set init_archive = "results/m7/p2/v6/samples.npz"
set v17_samples = "results/m7/p2/v17/samples.npz"
set v17_diagnostics = "results/m7/p2/v17/diagnostics.json"
set v8_dir = "results/m8/v8/free_sigma_lowrank_precision"
set v8_launcher = "csh/exe_m8_v8_check_free_sigma_lowrank_precision.csh"
set v8_script = "examples/luhman16b_yama/check_m8_v8_free_sigma_lowrank_precision.py"
set v8_test = "tests/unittests/test_m8_free_sigma_lowrank_precision.py"
set v8_complete = "$v8_dir/COMPLETE"
set v8_summary = "$v8_dir/m8_v8_free_sigma_lowrank_precision_summary.json"
set v8_arrays = "$v8_dir/m8_v8_free_sigma_lowrank_precision_arrays.npz"
set v8_checkpoint = "$v8_dir/m8_v8_free_sigma_lowrank_precision_checkpoint.json"
set v8_outputs = "$v8_dir/outputs.sha256"
set v8_provenance = "$v8_dir/provenance.sha256"
set v8_log = "$v8_dir/run.log"
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

set guard = "$output_dir/initial_sigma_score_guard.json"
set samples = "$output_dir/samples.npz"
set diagnostics = "$output_dir/diagnostics.json"
set summary = "$output_dir/m8_v9_free_sigma_highest_precision_control_summary.json"
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
set jax_config_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/jax/_src/config.py"
set jax_lax_source = "/home/kawahara/miniconda3/lib/python3.12/site-packages/jax/_src/lax/lax.py"

set required_files = ( \
    "$control_script" \
    "$control_test" \
    "$sampler" \
    "$sampler_launcher" \
    "$v17_wrapper" \
    "$v17_launcher" \
    "$fixed_wrapper" \
    "$replay_helper" \
    "$init_archive" \
    "$v17_samples" \
    "$v17_diagnostics" \
    "$v8_launcher" \
    "$v8_script" \
    "$v8_test" \
    "$v8_complete" \
    "$v8_summary" \
    "$v8_arrays" \
    "$v8_checkpoint" \
    "$v8_outputs" \
    "$v8_provenance" \
    "$v8_log" \
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
    "$jax_config_source" \
    "$jax_lax_source" \
)
set expected_hashes = ( \
    "a79c29d4d315233a7b93bd16d4bc6696f8bee73d88657971c87b049809af1f8f" \
    "9dec24b3837797de372ec7190a2dd2bc864099bcf5916481a065c366779ee2e6" \
    "fb17a96b1718d65a680ae6162b010cd7f0fac0a577de275d4c8df8a562312423" \
    "c7d6166173dd82b5f82ab63f22312931ac5f9aa3dafb5d302dbac2a6da720203" \
    "1d280912b78940da0450c97a8a0f12de5954264c0a307be07013499596812c61" \
    "8eec484f38216d41ca83294ef6183b2be2becf15cded7128778e1e063b304382" \
    "abb1b8ea4b2d0c6ca52af86035a501519503572255f484bdc472039e5b7faf3d" \
    "ed4268237691c120f759626db3f2400f3fd45bd17f2546ef3644ed423cec53ed" \
    "3256e8e10998c521313ab14473ac58b1e8aebb1e567a2593619477c0bfe70f89" \
    "d0dac818908d3180ff0c22cb9ff6b1f4b389351f9c458f08d24963fe636cabc9" \
    "7f4f449fa961a064337b0138edf832d0557d5dfda9882833078d5587148f725d" \
    "d10053ef4079185bed2dd3d1eb2673d80b4f0a04b93108c04254b8107ad1cb7a" \
    "56dd6d57fde7a38d500ca9804e4904b420bb4ef806f5d33b80c5ef8846562d39" \
    "d3f426ad894732c7ce669cded3d30df79bf7f4954338609cf80faa9a46a62bbc" \
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" \
    "7db88c6fff97d4fa1421898598d8e1d5bff99ca204cd03d70cacc303023d6efc" \
    "ce02f62cec5d208f3821143c1204ff7ee86e15efafe3a70e6ce297a79c7b8a67" \
    "e0f491fad1ee98711cbd957242afacc62968d4de8afee36500abc8112f47962d" \
    "e8cf295d25677f4e44164a8ebbb586331682288fe9c54c21afe915342cc3b8c5" \
    "6c20ec09d3e16c763fdced906233af0f0d05a9338e8ceadfd0695e466dc51bf0" \
    "20f65b1f843d4013fec4c8dd93052476b974a3c44c5a562037ab97f0512231f2" \
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
    "a9a2bccf86b35734171699cb34e2c9142add5c9bfbd5422941e1a6b192d25aad" \
    "d4da422029482f0c107a376b483b7e99f5cdde3c18a96f6201dc82b1a5fd0cc6" \
)

if ( -e "$output_dir" ) then
    echo "[m8_v9_highest_free] refusing output collision: $output_dir"
    exit 1
endif
if ( $#required_files != $#expected_hashes ) then
    echo "[m8_v9_highest_free] provenance hashes are not frozen"
    exit 1
endif
foreach required ( $required_files )
    if ( ! -f "$required" ) then
        echo "[m8_v9_highest_free] missing required input: $required"
        exit 1
    endif
end
if ( ! -d "$database_dir" || ! -d "$opacity_cache_dir" ) then
    echo "[m8_v9_highest_free] molecular database or opacity cache is missing"
    exit 1
endif
if ( $?XLA_FLAGS ) then
    echo "[m8_v9_highest_free] refusing ambient XLA_FLAGS: $XLA_FLAGS"
    exit 1
endif
if ( $?JAX_DEFAULT_MATMUL_PRECISION ) then
    echo "[m8_v9_highest_free] refusing ambient JAX_DEFAULT_MATMUL_PRECISION: $JAX_DEFAULT_MATMUL_PRECISION"
    exit 1
endif

@ hash_index = 1
while ( $hash_index <= $#required_files )
    set actual_hash = `sha256sum "$required_files[$hash_index]" | awk '{print $1}'`
    if ( "$actual_hash" != "$expected_hashes[$hash_index]" ) then
        echo "[m8_v9_highest_free] SHA256 mismatch: $required_files[$hash_index]"
        echo "[m8_v9_highest_free] expected $expected_hashes[$hash_index]"
        echo "[m8_v9_highest_free] actual   $actual_hash"
        exit 1
    endif
    @ hash_index++
end
sha256sum -c "$v8_outputs"
if ( $status != 0 ) then
    echo "[m8_v9_highest_free] v8 output manifest validation failed"
    exit 1
endif
sha256sum -c "$v8_provenance"
if ( $status != 0 ) then
    echo "[m8_v9_highest_free] v8 provenance validation failed"
    exit 1
endif

setenv PYTHONPATH "$repo_root/src"
setenv PYTHONDONTWRITEBYTECODE 1
setenv JAX_DEFAULT_MATMUL_PRECISION highest

python -c 'import inspect,os,sys,jax,jaxlib,exojax,numpyro,numpy,scipy; versions=[jax.__version__,jaxlib.__version__,exojax.__version__,numpyro.__version__,numpy.__version__,scipy.__version__]; wanted=list(sys.argv[1:7]); paths=[os.path.realpath(inspect.getfile(x)) for x in (jax,jaxlib,exojax,numpyro,numpy,scipy)]; expected=[os.path.realpath(x) for x in sys.argv[7:13]]; precision=jax.config.jax_default_matmul_precision; print(f"versions={versions} paths={paths} precision={precision}"); sys.exit(0 if versions==wanted and paths==expected and precision=="highest" else 4)' "$expected_jax_version" "$expected_jaxlib_version" "$expected_exojax_version" "$expected_numpyro_version" "$expected_numpy_version" "$expected_scipy_version" "$expected_jax_path" "$expected_jaxlib_path" "$expected_exojax_path" "$expected_numpyro_path" "$expected_numpy_path" "$expected_scipy_path"
set package_status = $status
if ( $package_status != 0 ) then
    echo "[m8_v9_highest_free] frozen package or precision validation failed"
    exit $package_status
endif

set control_args = ( \
    --init-from "$init_archive" \
    --out-dir "$output_dir" \
    --v17-samples "$v17_samples" \
    --v17-diagnostics "$v17_diagnostics" \
    --v8-dir "$v8_dir" \
)
python "$control_script" $control_args --validate-only
set validation_status = $status
if ( $validation_status != 0 ) then
    echo "[m8_v9_highest_free] Python validation failed with status $validation_status"
    exit $validation_status
endif
if ( $validate_only == 1 ) then
    echo "[m8_v9_highest_free] launcher validation passed; no GPU work was started"
    echo "[m8_v9_highest_free] one-line submission: $script_path --nohup"
    echo "[m8_v9_highest_free] planned output: $output_dir"
    echo "[m8_v9_highest_free] planned run log: $log"
    exit 0
endif

set gpu_python_pattern = 'python[0-9.]*[[:space:]].*examples/luhman16b_yama/(m[78](_p[0-9]+)?_v[0-9]+_run|m8_v1_run|check_m8_.*(linearization|convergence|validity|geometry|hvp|derivative|lowrank|highest_precision))[.]py'
set active_gpu_pids = (`pgrep -f "$gpu_python_pattern"`)
if ( $#active_gpu_pids > 0 ) then
    echo "[m8_v9_highest_free] refusing concurrent GPU work; active PID(s): $active_gpu_pids"
    pgrep -af "$gpu_python_pattern"
    exit 2
endif

set output_parent = "$output_dir:h"
mkdir -p "$output_parent"
if ( $status != 0 ) exit 1
mkdir "$output_dir"
if ( $status != 0 ) then
    echo "[m8_v9_highest_free] failed to claim fresh output directory"
    exit 1
endif
touch "$output_dir/RUNNING"
if ( $status != 0 ) exit 1

mkdir -p /tmp/numba-cache
mkdir -p /tmp/matplotlib-codex
mkdir -p /tmp/doraex-jax-cache-m8-v9-highest-free
setenv NUMBA_CACHE_DIR /tmp/numba-cache
setenv MPLCONFIGDIR /tmp/matplotlib-codex
setenv JAX_COMPILATION_CACHE_DIR /tmp/doraex-jax-cache-m8-v9-highest-free
setenv PYTHONUNBUFFERED 1
setenv XLA_PYTHON_CLIENT_PREALLOCATE false

echo "[m8_v9_highest_free] start: `date`" | tee "$log"
echo "[m8_v9_highest_free] log: $log" | tee -a "$log"
if ( $?DORAEX_M8_V9_OUTER_LOG ) then
    echo "[m8_v9_highest_free] outer nohup log: $DORAEX_M8_V9_OUTER_LOG" | tee -a "$log"
endif
echo "[m8_v9_highest_free] repository: $repo_root" | tee -a "$log"
echo "[m8_v9_highest_free] git HEAD: `git rev-parse HEAD`" | tee -a "$log"
echo "[m8_v9_highest_free] host: `hostname`" | tee -a "$log"
echo "[m8_v9_highest_free] JAX_DEFAULT_MATMUL_PRECISION: $JAX_DEFAULT_MATMUL_PRECISION" | tee -a "$log"
echo "[m8_v9_highest_free] XLA_FLAGS: unset (required)" | tee -a "$log"
echo "[m8_v9_highest_free] intervention: process-global matmul precision only" | tee -a "$log"
echo "[m8_v9_highest_free] schedule: seed=0 chains=1 warmup=200 samples=20 target_accept=0.95 depths=9,11" | tee -a "$log"
python --version |& tee -a "$log"
git status --short --untracked-files=normal >>& "$log"

python -c 'import inspect,os,sys,jax,jaxlib,exojax,numpyro,numpy,scipy; backend=jax.default_backend(); devices=jax.devices(); versions=[jax.__version__,jaxlib.__version__,exojax.__version__,numpyro.__version__,numpy.__version__,scipy.__version__]; wanted=list(sys.argv[1:7]); paths=[os.path.realpath(inspect.getfile(x)) for x in (jax,jaxlib,exojax,numpyro,numpy,scipy)]; expected=[os.path.realpath(x) for x in sys.argv[7:13]]; precision=jax.config.jax_default_matmul_precision; print(f"versions={versions} paths={paths} precision={precision} backend={backend} devices={devices}"); gpu=backend=="gpu" and devices and all(x.platform=="gpu" for x in devices); sys.exit(0 if gpu and versions==wanted and paths==expected and precision=="highest" else 3)' "$expected_jax_version" "$expected_jaxlib_version" "$expected_exojax_version" "$expected_numpyro_version" "$expected_numpy_version" "$expected_scipy_version" "$expected_jax_path" "$expected_jaxlib_path" "$expected_exojax_path" "$expected_numpyro_path" "$expected_numpy_path" "$expected_scipy_path" >>& "$log"
set backend_status = $status
tail -n 1 "$log"
if ( $backend_status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v9_highest_free] GPU/package/precision preflight failed" | tee -a "$log"
    exit $backend_status
endif

sha256sum "$script_path" $required_files > "$provenance_tmp"
set provenance_status = $status
if ( $provenance_status != 0 || ! -s "$provenance_tmp" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    if ( $provenance_status == 0 ) set provenance_status = 1
    exit $provenance_status
endif
mv "$provenance_tmp" "$provenance"
sha256sum -c "$provenance" >>& "$log"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif

echo "[m8_v9_highest_free] initial score guard start: `date`" | tee -a "$log"
python -u "$control_script" $control_args --action guard >>& "$log"
set guard_status = $status
if ( $guard_status != 0 || ! -f "$guard" ) then
    set failure_artifacts = ()
    if ( -f "$guard" ) set failure_artifacts = ( "$guard" )
    if ( $#failure_artifacts > 0 ) then
        sha256sum $failure_artifacts > "$failure_hashes_tmp"
        if ( $status == 0 && -s "$failure_hashes_tmp" ) mv "$failure_hashes_tmp" "$failure_hashes"
    endif
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v9_highest_free] initial score guard failed with status $guard_status" | tee -a "$log"
    if ( $guard_status == 0 ) set guard_status = 1
    exit $guard_status
endif
touch "$output_dir/GRADIENT_VALIDATED"

echo "[m8_v9_highest_free] sampler start: `date`" | tee -a "$log"
echo "[m8_v9_highest_free] sampler: python $sampler --init-from $repo_root/$init_archive --out-dir $repo_root/$output_dir --run-label m8_v9_free_sigma_highest_precision_v17_schedule --num-chains 1 --seed 0 --num-warmup 200 --num-samples 20 --target-accept-prob 0.95 --warmup-max-tree-depth 9 --max-tree-depth 11 --dense-mass --adapt-mass-matrix --no-x64 --print-summary" | tee -a "$log"
python -u "$sampler" \
    --init-from "$repo_root/$init_archive" \
    --out-dir "$repo_root/$output_dir" \
    --run-label m8_v9_free_sigma_highest_precision_v17_schedule \
    --num-chains 1 \
    --seed 0 \
    --num-warmup 200 \
    --num-samples 20 \
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
    set failure_artifacts = ( "$guard" )
    if ( -f "$samples" ) set failure_artifacts = ( $failure_artifacts "$samples" )
    if ( -f "$diagnostics" ) set failure_artifacts = ( $failure_artifacts "$diagnostics" )
    sha256sum $failure_artifacts > "$failure_hashes_tmp"
    if ( $status == 0 && -s "$failure_hashes_tmp" ) mv "$failure_hashes_tmp" "$failure_hashes"
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v9_highest_free] sampler failed with status ${run_status}: `date`" | tee -a "$log"
    exit $run_status
endif
if ( ! -f "$samples" || ! -f "$diagnostics" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v9_highest_free] sampler returned without required artifacts" | tee -a "$log"
    exit 1
endif

python -u "$control_script" $control_args --action summarize >>& "$log"
set summary_status = $status
if ( $summary_status != 0 || ! -f "$summary" ) then
    set failure_artifacts = ( "$guard" "$samples" "$diagnostics" )
    if ( -f "$summary" ) set failure_artifacts = ( $failure_artifacts "$summary" )
    sha256sum $failure_artifacts > "$failure_hashes_tmp"
    if ( $status == 0 && -s "$failure_hashes_tmp" ) mv "$failure_hashes_tmp" "$failure_hashes"
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v9_highest_free] output summarization failed" | tee -a "$log"
    if ( $summary_status == 0 ) set summary_status = 1
    exit $summary_status
endif
python "$control_script" $control_args --action validate-artifacts >>& "$log"
set artifact_status = $status
if ( $artifact_status != 0 ) then
    sha256sum "$guard" "$samples" "$diagnostics" "$summary" > "$failure_hashes_tmp"
    if ( $status == 0 && -s "$failure_hashes_tmp" ) mv "$failure_hashes_tmp" "$failure_hashes"
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    echo "[m8_v9_highest_free] artifact validation failed" | tee -a "$log"
    exit $artifact_status
endif

sha256sum "$guard" "$output_dir/GRADIENT_VALIDATED" "$samples" "$diagnostics" "$summary" > "$output_hashes_tmp"
set output_hash_status = $status
if ( $output_hash_status != 0 || ! -s "$output_hashes_tmp" ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    if ( $output_hash_status == 0 ) set output_hash_status = 1
    exit $output_hash_status
endif
mv "$output_hashes_tmp" "$output_hashes"
sha256sum -c "$output_hashes" >>& "$log"
if ( $status != 0 ) then
    mv "$output_dir/RUNNING" "$output_dir/FAILED"
    touch "$output_dir/FAILED"
    exit 1
endif
mv "$output_dir/RUNNING" "$output_dir/COMPLETE"
touch "$output_dir/COMPLETE"
echo "[m8_v9_highest_free] summary: $summary" | tee -a "$log"
echo "[m8_v9_highest_free] done: `date`" | tee -a "$log"
