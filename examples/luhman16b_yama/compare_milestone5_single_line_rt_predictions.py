"""Compatibility wrapper for the single-line pressure-response diagnostic."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.diagnostics.single_line_pressure_response import main  # noqa: E402


if __name__ == "__main__":
    main()
