"""Backward-compatible entry point for the historical detection CLI.

Existing commands such as ``python detect.py IMAGE --conf 0.5`` continue to
delegate to ``scripts/detect.py`` while the package CLI is introduced.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "scripts" / "detect.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

