"""Run E2 v1 observation-aware eigen-atmosphere retrieval."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASIS = ROOT / "results" / "e2" / "v1_basis" / "eigen_response_basis.npz"
DEFAULT_OUT_DIR = ROOT / "results" / "e2" / "v1_run"


def _has_option(name):
    """Return whether a command-line option was explicitly supplied."""

    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def _ensure_option(name, value=None):
    """Append a default command-line option when the user did not set it."""

    if _has_option(name):
        return
    sys.argv.append(name)
    if value is not None:
        sys.argv.append(str(value))


def main():
    """Run the shared sampler in E2 v1 observation-aware eigenmode mode."""

    _ensure_option("--chip-indices", "0,1,2,3")
    _ensure_option("--out-dir", DEFAULT_OUT_DIR)
    _ensure_option("--full-data")
    _ensure_option("--nside", 8)
    _ensure_option("--num-warmup", 2000)
    _ensure_option("--num-samples", 1500)
    _ensure_option("--standardized-parameters")
    _ensure_option("--fixed-ell-b", 0.4)
    _ensure_option("--fix-logg")
    _ensure_option("--init-logg", 4.86)
    _ensure_option("--zero-mean-log-w")
    _ensure_option("--eigen-basis", DEFAULT_BASIS)
    _ensure_option("--eigen-mode-count", 2)
    _ensure_option("--eigen-fixed-ell", 0.4)
    _ensure_option("--fixed-sigma-eigen", "1.0,1.0")
    _ensure_option("--frozen-eigen-spectra")
    _ensure_option("--no-preflight-autodiff")
    _ensure_option("--target-accept-prob", 0.95)
    _ensure_option("--max-tree-depth", 11)

    from m6_v1_run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
