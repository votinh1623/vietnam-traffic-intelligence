"""Download one pinned Hugging Face model and write a file-hash manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=("vlm", "llm"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = config[args.role]
    revision = model.get("revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("model revision must be a pinned 40-character commit SHA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model["model_id"],
        revision=revision,
        local_dir=args.output_dir,
    )

    files = []
    for path in sorted(item for item in args.output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(args.output_dir).as_posix()
        if relative.startswith(".cache/"):
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "role": args.role,
        "model_id": model["model_id"],
        "revision": revision,
        "local_directory": args.output_dir.as_posix(),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.manifest)
    print(
        f"Downloaded {payload['model_id']}@{revision}: "
        f"{payload['total_bytes']} bytes in {payload['file_count']} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
