"""Compatibility wrapper for the mean-subtracted line-stack diagnostic."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.diagnostics.mean_subtracted_line_stack import main  # noqa: E402


if __name__ == "__main__":
    main()
