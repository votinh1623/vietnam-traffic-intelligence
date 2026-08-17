"""Prepare, validate, and compare reasoning evaluation annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.reasoning.annotations import (  # noqa: E402
    build_adjudication_queue,
    build_annotation_template,
    build_review_index,
    queue_is_unadjudicated,
    read_jsonl,
    validate_annotation_set,
    validate_adjudication_queue,
    write_json_atomic,
    write_jsonl_atomic,
    write_review_index_atomic,
)


def _load_lock(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--lock", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--reviewers", nargs=2, default=("reviewer_a", "reviewer_b"))

    validate = subparsers.add_parser("validate")
    validate.add_argument("--lock", type=Path, required=True)
    validate.add_argument("--annotations", type=Path, required=True)
    validate.add_argument("--require-complete", action="store_true")

    compare = subparsers.add_parser("compare")
    compare.add_argument("--lock", type=Path, required=True)
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--replace-pending", action="store_true")

    validate_queue = subparsers.add_parser("validate-queue")
    validate_queue.add_argument("--lock", type=Path, required=True)
    validate_queue.add_argument("--first", type=Path, required=True)
    validate_queue.add_argument("--second", type=Path, required=True)
    validate_queue.add_argument("--queue", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = _load_lock(args.lock)
    if args.command == "prepare":
        paths = [args.output_dir / f"{name}.jsonl" for name in args.reviewers]
        paths.append(args.output_dir / "review_index.csv")
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite annotation files: "
                + ", ".join(str(path) for path in existing)
            )
        for reviewer_id in args.reviewers:
            records = build_annotation_template(lock, reviewer_id)
            path = args.output_dir / f"{reviewer_id}.jsonl"
            write_jsonl_atomic(path, records)
            print(f"Prepared {len(records)} cases: {path}")
        index_path = args.output_dir / "review_index.csv"
        write_review_index_atomic(index_path, build_review_index(lock))
        print(f"Prepared review index: {index_path}")
        return 0
    if args.command == "validate":
        reviewer_id = validate_annotation_set(
            read_jsonl(args.annotations),
            lock,
            require_complete=args.require_complete,
        )
        print(f"Valid annotation set for {reviewer_id}: {args.annotations}")
        return 0
    if args.command == "validate-queue":
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
        validate_adjudication_queue(
            queue, read_jsonl(args.first), read_jsonl(args.second), lock
        )
        print(f"Current adjudication queue: {args.queue}")
        return 0
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if not args.replace_pending:
            raise FileExistsError(
                f"refusing to overwrite adjudication queue: {args.output}"
            )
        if not queue_is_unadjudicated(existing):
            raise ValueError("refusing to replace a queue with adjudicated content")
    queue = build_adjudication_queue(
        read_jsonl(args.first), read_jsonl(args.second), lock
    )
    write_json_atomic(args.output, queue)
    print(
        f"Prepared {queue['case_count']} adjudication cases; "
        f"categorical disagreements: {queue['categorical_disagreement_cases']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
