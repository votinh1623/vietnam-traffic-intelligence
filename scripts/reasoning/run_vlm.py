"""Run or dry-run one frozen development VLM case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.reasoning.vlm_runtime import (  # noqa: E402
    load_development_case,
    run_vlm_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, request, artifact_root = load_development_case(
        args.config, args.case_id
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "case_id": request["case_id"],
                    "event_id": request["event"]["event_id"],
                    "model_id": config["vlm"]["model_id"],
                    "revision": config["vlm"]["revision"],
                    "evidence": request["evidence"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.model_dir is None or args.output is None:
        raise ValueError("--model-dir and --output are required unless --dry-run")
    prompts = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "reasoning" / "prompts_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    result = run_vlm_case(
        config=config,
        request=request,
        artifact_root=artifact_root,
        model_dir=args.model_dir,
        system_prompt=prompts["vlm"]["system"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    if result["contract_status"] != "valid":
        print(f"Invalid VLM contract saved for audit: {args.output}")
        return 2
    print(f"Validated VLM result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
