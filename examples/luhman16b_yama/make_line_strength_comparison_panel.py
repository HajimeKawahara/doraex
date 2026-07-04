#!/usr/bin/env python
"""Create the line-strength four-panel comparison product."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doraex.diagnostics.line_strength_comparison_panel import main  # noqa: E402


if __name__ == "__main__":
    main()
