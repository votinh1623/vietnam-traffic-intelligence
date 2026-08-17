"""Freeze a completed pipeline run as a reasoning evaluation input set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.reasoning.freeze import (  # noqa: E402
    build_evidence_lock,
    write_evidence_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze all evidence records from one completed pipeline run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--set-id", required=True)
    parser.add_argument(
        "--split",
        choices=("calibration", "development", "evaluation"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_evidence_lock(
        run_dir=args.run_dir.resolve(),
        set_id=args.set_id,
        split=args.split,
    )
    write_evidence_lock(args.output, payload)
    print(
        f"Frozen {payload['case_count']} {args.split} cases; "
        f"lock SHA-256: {payload['lock_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
