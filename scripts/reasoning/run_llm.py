"""Run or dry-run one LLM report from a validated development VLM result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.reasoning.llm_runtime import (  # noqa: E402
    prepare_llm_request,
    run_llm_case,
)
from vn_traffic.reasoning.vlm_runtime import (  # noqa: E402
    load_development_case,
    load_prompts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--vlm-result", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, vlm_request, _ = load_development_case(args.config, args.case_id)
    vlm_result, request = prepare_llm_request(args.vlm_result, vlm_request)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "case_id": vlm_request["case_id"],
                    "event_id": vlm_request["event"]["event_id"],
                    "vlm_model_id": vlm_result.get("model_id"),
                    "llm_model_id": config["llm"]["model_id"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.model_dir is None or args.output is None:
        raise ValueError("--model-dir and --output are required unless --dry-run")
    prompts = load_prompts(config, PROJECT_ROOT)
    result = run_llm_case(
        config=config,
        request=request,
        model_dir=args.model_dir,
        system_prompt=prompts["llm"]["system"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    if result["contract_status"] != "valid":
        print(f"Invalid LLM contract saved for audit: {args.output}")
        return 2
    print(f"Validated LLM result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
