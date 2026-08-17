"""YAML configuration loading and validation for the offline-video MVP."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    source: Path
    model: Path
    output_root: Path
    imgsz: int = 1280
    confidence: float = 0.4
    iou: float = 0.7
    device: str = "0"
    tracker: str = "bytetrack.yaml"
    codec: str = "mp4v"
    fallback_fps: float = 30.0
    max_frames: int | None = None
    config_path: Path | None = None

    def with_overrides(
        self,
        *,
        source: str | None = None,
        model: str | None = None,
        max_frames: int | None = None,
        device: str | None = None,
        imgsz: int | None = None,
    ) -> "PipelineConfig":
        updates: dict[str, Any] = {}
        if source is not None:
            updates["source"] = resolve_project_path(source)
        if model is not None:
            updates["model"] = resolve_project_path(model)
        if max_frames is not None:
            updates["max_frames"] = max_frames
        if device is not None:
            updates["device"] = device
        if imgsz is not None:
            updates["imgsz"] = imgsz
        config = replace(self, **updates)
        validate_pipeline_config(config)
        return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_tracker(value: str) -> str:
    """Resolve repository-local tracker YAML while preserving built-in names."""
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    project_candidate = (PROJECT_ROOT / path).resolve()
    return str(project_candidate) if project_candidate.is_file() else value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = resolve_project_path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError("pipeline config must be a mapping")

    perception = _mapping(raw.get("perception"), "perception")
    output = _mapping(raw.get("output"), "output")
    video = _mapping(raw.get("video"), "video")
    if not raw.get("source"):
        raise ValueError("source is required")
    if not raw.get("model"):
        raise ValueError("model is required")
    config = PipelineConfig(
        schema_version=int(raw.get("schema_version", 1)),
        source=resolve_project_path(raw["source"]),
        model=resolve_project_path(raw["model"]),
        output_root=resolve_project_path(output.get("root", "output/pipeline")),
        imgsz=int(perception.get("imgsz", 1280)),
        confidence=float(perception.get("confidence", 0.4)),
        iou=float(perception.get("iou", 0.7)),
        device=str(perception.get("device", "0")),
        tracker=resolve_tracker(str(perception.get("tracker", "bytetrack.yaml"))),
        codec=str(video.get("codec", "mp4v")),
        fallback_fps=float(video.get("fallback_fps", 30.0)),
        max_frames=(
            int(video["max_frames"])
            if video.get("max_frames") is not None
            else None
        ),
        config_path=config_path,
    )
    validate_pipeline_config(config)
    return config


def validate_pipeline_config(config: PipelineConfig) -> None:
    if config.schema_version != 1:
        raise ValueError(f"unsupported schema_version: {config.schema_version}")
    if not str(config.source):
        raise ValueError("source is required")
    if not str(config.model):
        raise ValueError("model is required")
    if config.imgsz <= 0:
        raise ValueError("perception.imgsz must be positive")
    if not 0.0 <= config.confidence <= 1.0:
        raise ValueError("perception.confidence must be between 0 and 1")
    if not 0.0 <= config.iou <= 1.0:
        raise ValueError("perception.iou must be between 0 and 1")
    if len(config.codec) != 4:
        raise ValueError("video.codec must contain exactly four characters")
    if config.fallback_fps <= 0:
        raise ValueError("video.fallback_fps must be positive")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("video.max_frames must be positive when provided")
