#!/usr/bin/env -S tcsh -f

# Run two real-data production chains using the validated M8 conditional draws.
set script_path = `realpath "$0"`
set script_dir = "$script_path:h"
set repo_root = "$script_dir:h"
cd "$repo_root"

if ($#argv != 0) then
    echo "This launcher takes no arguments."
    exit 2
endif

if (! $?DORAEX_PYTHON) setenv DORAEX_PYTHON python
if (! $?EXOJAX_DIR) setenv EXOJAX_DIR "$repo_root:h/exojax"
set python = "$DORAEX_PYTHON"
set tag = `date -u +%Y%m%dT%H%M%SZ`
set run_root = "results/m9/v3/production_draw0_w2000_s1500_d9_d11_$tag"
set samples = "results/m8/v12/masked_free_sigma_d_highest_precision_long_seed0/samples.npz"
set validated_gate = "results/m9/v3/forward_gate_81.json"
set seed_dir = "results/m9/v3/seeds"
set database_dir = "$HOME/data_mol/.database"
if ($?DORAEX_M9_SAMPLES) then
    set samples = "$DORAEX_M9_SAMPLES"
endif
if ($?DORAEX_M9_FORWARD_GATE) then
    set validated_gate = "$DORAEX_M9_FORWARD_GATE"
endif
if ($?DORAEX_M9_SEED_DIR) then
    set seed_dir = "$DORAEX_M9_SEED_DIR"
endif
if ($?DORAEX_DATABASE_DIR) then
    set database_dir = "$DORAEX_DATABASE_DIR"
endif
foreach required ("$samples" "$validated_gate" "$seed_dir/m8_q50_draw0.npz" "$seed_dir/m8_q95_draw0.npz")
    if (! -f "$required") then
        echo "Missing required input: $required"
        exit 1
    endif
end
if (-e "$run_root") then
    echo "Output already exists: $run_root"
    exit 1
endif
mkdir -p "$run_root:h"
if ($status != 0) exit 1
mkdir "$run_root"
if ($status != 0) exit 1
cp "$script_path" "$run_root/launcher.csh"
if ($status != 0) exit 1
cp "$validated_gate" "$run_root/forward_gate_81.json"
if ($status != 0) exit 1
echo "Production output: $run_root"
echo "Schedule: 2000 warmup / 1500 saved draws; warmup depth 9 / sampling depth 11"

setenv PYTHONPATH "$EXOJAX_DIR/src:$repo_root/src"
setenv OMP_NUM_THREADS 1
setenv OPENBLAS_NUM_THREADS 1
setenv MKL_NUM_THREADS 1
setenv MPLCONFIGDIR /tmp/doraex-mpl-cache

set chain_seed = 0
foreach label (q50 q95)
    echo "Sampling real data: $label conditional draw, seed $chain_seed"
    # Omitting both synthetic options retains the archived real observations.
    # The runner verifies the saved forward evidence and takes the GPU lock.
    csh/exe_m9_v3_investigation.csh --stage sample --platform gpu \
        --samples "$samples" --chip-indices 0,1,2,3 \
        --state-file "$seed_dir/m8_${label}_draw0.npz" --database-dir "$database_dir" \
        --out-dir "$run_root/${label}_seed${chain_seed}" --marginalize-a \
        --initial-map-source archived --map-scale 1 \
        --nodes 81 --rt-pixel-batch-size 64 \
        --forward-validation "$run_root/forward_gate_81.json" \
        --num-warmup 2000 --num-samples 1500 \
        --warmup-max-tree-depth 9 --max-tree-depth 11 \
        --target-accept-prob 0.9 --no-adapt-mass-matrix --seed "$chain_seed" \
        >& "$run_root/${label}_seed${chain_seed}.log"
    if ($status != 0) exit 1
    @ chain_seed++
end

env JAX_PLATFORMS=cpu "$python" examples/luhman16b_yama/analyze_m9_v3_pilots.py \
    --runs "$run_root/q50_seed0" "$run_root/q95_seed1" \
    --out "$run_root/diagnostics.json" >& "$run_root/diagnostics.log"
if ($status != 0) exit 1
echo "Production completed; inspect posterior diagnostics: $run_root/diagnostics.json"
