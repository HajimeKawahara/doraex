"""Exercise production argument forwarding without running retrieval or using a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_production_launcher_preserves_schedule_and_external_inputs(tmp_path):
    tcsh = shutil.which("tcsh")
    if tcsh is None:
        pytest.skip("tcsh is required for launcher integration")
    source = Path(__file__).resolve().parents[2]
    repo = tmp_path / "repository"
    (repo / "csh").mkdir(parents=True)
    for name in ("exe_m9_v3_production.csh", "exe_m9_v3_investigation.csh"):
        shutil.copy2(source / "csh" / name, repo / "csh" / name)
    inputs = tmp_path / "external inputs"
    inputs.mkdir()
    for name in ("samples.npz", "gate.json", "m8_q50_draw0.npz", "m8_q95_draw0.npz"):
        (inputs / name).write_text("launcher-only fixture\n")
    record = tmp_path / "commands.jsonl"
    python = tmp_path / "record python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DORAEX_TEST_RECORD'], 'a') as stream:\n"
        "    stream.write(json.dumps({'argv': sys.argv[1:], 'platform': os.environ.get('JAX_PLATFORMS')}) + '\\n')\n"
    )
    python.chmod(0o755)
    environment = dict(os.environ, DORAEX_PYTHON=str(python), EXOJAX_DIR=str(tmp_path / "exojax"),
                       DORAEX_M9_SAMPLES=str(inputs / "samples.npz"), DORAEX_M9_FORWARD_GATE=str(inputs / "gate.json"),
                       DORAEX_M9_SEED_DIR=str(inputs), DORAEX_DATABASE_DIR=str(inputs / "database"),
                       DORAEX_TEST_RECORD=str(record))
    subprocess.run([tcsh, "-f", str(repo / "csh/exe_m9_v3_production.csh")], cwd=tmp_path,
                   env=environment, check=True, capture_output=True, text=True)
    commands = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(commands) == 3
    for seed, label in enumerate(("q50", "q95")):
        argv = commands[seed]["argv"]
        assert argv[0] == "examples/luhman16b_yama/m9_v3_run.py"
        for option, value in {"--num-warmup": "2000", "--num-samples": "1500",
                              "--warmup-max-tree-depth": "9", "--max-tree-depth": "11",
                              "--target-accept-prob": "0.9", "--seed": str(seed),
                              "--samples": str(inputs / "samples.npz"),
                              "--state-file": str(inputs / f"m8_{label}_draw0.npz"),
                              "--database-dir": str(inputs / "database")}.items():
            assert argv[argv.index(option) + 1] == value
        assert "--marginalize-a" in argv and "--no-adapt-mass-matrix" in argv
        assert "--observations-from" not in argv and "--synthetic-seed" not in argv
    assert commands[2]["argv"][0] == "examples/luhman16b_yama/analyze_m9_v3_pilots.py"
    assert commands[2]["platform"] == "cpu"
    run = next((repo / "results/m9/v3").iterdir())
    assert (run / "forward_gate_81.json").read_bytes() == (inputs / "gate.json").read_bytes()
    assert (run / "launcher.csh").is_file()
