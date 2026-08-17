"""YAML configuration loading and validation for the offline-video MVP."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


NormalizedPoint = tuple[float, float]


@dataclass(frozen=True)
class AnalyticsConfig:
    enabled: bool = True
    roi_polygon: tuple[NormalizedPoint, ...] = (
        (0.48, 0.05),
        (0.62, 0.05),
        (0.84, 0.95),
        (0.30, 0.95),
    )
    counting_line: tuple[NormalizedPoint, NormalizedPoint] = (
        (0.32, 0.65),
        (0.78, 0.65),
    )
    line_tolerance_px: float = 8.0
    trajectory_history: int = 30
    occupancy_grid_size_px: int = 1
    dense_enter_bbox_union_occupancy: float = 0.30
    dense_exit_bbox_union_occupancy: float = 0.25
    congested_enter_bbox_union_occupancy: float = 0.50
    congested_exit_bbox_union_occupancy: float = 0.40
    dense_enter_count: int = 45
    dense_exit_count: int = 40
    congested_enter_count: int = 50
    congested_exit_count: int = 45
    congested_max_speed_px_s: float = 150.0
    congested_release_speed_px_s: float = 170.0
    transition_confirm_s: float = 2.0
    release_confirm_s: float = 4.0


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
    analytics: AnalyticsConfig = AnalyticsConfig()

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


def _normalized_points(
    value: Any, name: str, expected_length: int | None = None
) -> tuple[NormalizedPoint, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of [x, y] points")
    points: list[NormalizedPoint] = []
    for index, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly x and y")
        points.append((float(point[0]), float(point[1])))
    if expected_length is not None and len(points) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} points")
    return tuple(points)


def _load_analytics(raw: dict[str, Any]) -> AnalyticsConfig:
    defaults = AnalyticsConfig()
    analytics = _mapping(raw.get("analytics"), "analytics")
    congestion = _mapping(analytics.get("congestion"), "analytics.congestion")
    legacy_occupancy_fields = {
        "dense_enter_occupancy",
        "dense_exit_occupancy",
        "congested_enter_occupancy",
        "congested_exit_occupancy",
    }
    found_legacy = sorted(legacy_occupancy_fields.intersection(congestion))
    if found_legacy:
        raise ValueError(
            "legacy summed-occupancy fields are invalid; use bbox-union "
            f"threshold names instead: {', '.join(found_legacy)}"
        )
    roi = analytics.get("roi_polygon")
    line = analytics.get("counting_line")
    config = AnalyticsConfig(
        enabled=bool(analytics.get("enabled", defaults.enabled)),
        roi_polygon=(
            _normalized_points(roi, "analytics.roi_polygon")
            if roi is not None
            else defaults.roi_polygon
        ),
        counting_line=(
            _normalized_points(line, "analytics.counting_line", 2)  # type: ignore[arg-type]
            if line is not None
            else defaults.counting_line
        ),
        line_tolerance_px=float(
            analytics.get("line_tolerance_px", defaults.line_tolerance_px)
        ),
        trajectory_history=int(
            analytics.get("trajectory_history", defaults.trajectory_history)
        ),
        occupancy_grid_size_px=int(
            analytics.get("occupancy_grid_size_px", defaults.occupancy_grid_size_px)
        ),
        dense_enter_bbox_union_occupancy=float(
            congestion.get(
                "dense_enter_bbox_union_occupancy",
                defaults.dense_enter_bbox_union_occupancy,
            )
        ),
        dense_exit_bbox_union_occupancy=float(
            congestion.get(
                "dense_exit_bbox_union_occupancy",
                defaults.dense_exit_bbox_union_occupancy,
            )
        ),
        congested_enter_bbox_union_occupancy=float(
            congestion.get(
                "congested_enter_bbox_union_occupancy",
                defaults.congested_enter_bbox_union_occupancy,
            )
        ),
        congested_exit_bbox_union_occupancy=float(
            congestion.get(
                "congested_exit_bbox_union_occupancy",
                defaults.congested_exit_bbox_union_occupancy,
            )
        ),
        dense_enter_count=int(
            congestion.get("dense_enter_count", defaults.dense_enter_count)
        ),
        dense_exit_count=int(
            congestion.get("dense_exit_count", defaults.dense_exit_count)
        ),
        congested_enter_count=int(
            congestion.get(
                "congested_enter_count", defaults.congested_enter_count
            )
        ),
        congested_exit_count=int(
            congestion.get("congested_exit_count", defaults.congested_exit_count)
        ),
        congested_max_speed_px_s=float(
            congestion.get(
                "congested_max_speed_px_s", defaults.congested_max_speed_px_s
            )
        ),
        congested_release_speed_px_s=float(
            congestion.get(
                "congested_release_speed_px_s",
                defaults.congested_release_speed_px_s,
            )
        ),
        transition_confirm_s=float(
            congestion.get("transition_confirm_s", defaults.transition_confirm_s)
        ),
        release_confirm_s=float(
            congestion.get("release_confirm_s", defaults.release_confirm_s)
        ),
    )
    validate_analytics_config(config)
    return config


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
        analytics=_load_analytics(raw),
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
    validate_analytics_config(config.analytics)


def validate_analytics_config(config: AnalyticsConfig) -> None:
    if len(config.roi_polygon) < 3:
        raise ValueError("analytics.roi_polygon must contain at least three points")
    for name, points in (
        ("analytics.roi_polygon", config.roi_polygon),
        ("analytics.counting_line", config.counting_line),
    ):
        for x, y in points:
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError(f"{name} coordinates must be normalized to [0, 1]")
    if config.counting_line[0] == config.counting_line[1]:
        raise ValueError("analytics.counting_line endpoints must differ")
    if config.line_tolerance_px < 0:
        raise ValueError("analytics.line_tolerance_px cannot be negative")
    if config.trajectory_history < 2:
        raise ValueError("analytics.trajectory_history must be at least two")
    if config.occupancy_grid_size_px < 1:
        raise ValueError("analytics.occupancy_grid_size_px must be at least one")
    occupancy_values = (
        config.dense_exit_bbox_union_occupancy,
        config.dense_enter_bbox_union_occupancy,
        config.congested_exit_bbox_union_occupancy,
        config.congested_enter_bbox_union_occupancy,
    )
    if not all(0.0 <= value <= 1.0 for value in occupancy_values):
        raise ValueError("analytics congestion occupancy values must be in [0, 1]")
    if (
        config.dense_exit_bbox_union_occupancy
        >= config.dense_enter_bbox_union_occupancy
    ):
        raise ValueError(
            "dense_exit_bbox_union_occupancy must be below "
            "dense_enter_bbox_union_occupancy"
        )
    if (
        config.congested_exit_bbox_union_occupancy
        >= config.congested_enter_bbox_union_occupancy
    ):
        raise ValueError(
            "congested_exit_bbox_union_occupancy must be below "
            "congested_enter_bbox_union_occupancy"
        )
    if (
        config.dense_enter_bbox_union_occupancy
        >= config.congested_enter_bbox_union_occupancy
    ):
        raise ValueError(
            "dense_enter_bbox_union_occupancy must be below "
            "congested_enter_bbox_union_occupancy"
        )
    if not (
        0
        <= config.dense_exit_count
        < config.dense_enter_count
        < config.congested_enter_count
    ):
        raise ValueError("analytics congestion count thresholds are inconsistent")
    if not 0 <= config.congested_exit_count < config.congested_enter_count:
        raise ValueError("congested_exit_count must be below congested_enter_count")
    if config.congested_max_speed_px_s < 0:
        raise ValueError("congested_max_speed_px_s cannot be negative")
    if config.congested_release_speed_px_s < config.congested_max_speed_px_s:
        raise ValueError(
            "congested_release_speed_px_s must not be below entry speed"
        )
    if config.transition_confirm_s < 0 or config.release_confirm_s < 0:
        raise ValueError("analytics confirmation durations cannot be negative")
