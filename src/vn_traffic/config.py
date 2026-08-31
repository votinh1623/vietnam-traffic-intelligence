"""YAML configuration loading and validation for the offline-video MVP."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


NormalizedPoint = tuple[float, float]

ANALYTICS_MODES = ("fixed_camera", "uav_motion")

# A moving/zooming UAV camera makes a hand-drawn ground-anchored ROI point at
# the wrong region within seconds (see experiments/uav_pipeline_e2e_v1_20260818).
# uav_motion mode defaults the analytics region to the full frame instead, so
# occupancy/count reflect the whole visible scene rather than a stale patch of
# ground. This does not compensate for camera motion (no GMC); zoom changes
# still shift raw occupancy, and this remains a coarser signal than a properly
# ground-anchored ROI on a static camera.
UAV_MOTION_DEFAULT_ROI_POLYGON: tuple[NormalizedPoint, ...] = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
)
UAV_MOTION_DEFAULT_COUNTING_LINE: tuple[NormalizedPoint, NormalizedPoint] = (
    (0.0, 0.5),
    (1.0, 0.5),
)


@dataclass(frozen=True)
class AnalyticsConfig:
    enabled: bool = True
    analytics_mode: str = "fixed_camera"
    included_classes: tuple[str, ...] = (
        "bicycle",
        "bus",
        "car",
        "motor",
        "motorcycle",
        "truck",
        "van",
        "tricycle",
        "awning-tricycle",
    )
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
    # ECC-based global motion compensation (see src/vn_traffic/analytics/motion.py).
    # Only meaningful for analytics_mode="uav_motion": it re-projects the
    # roi_polygon/counting_line defined on frame 0 into every later frame's
    # pixels instead of collapsing the analytics region to the whole frame,
    # recovering location-specific ROI semantics under camera pan/zoom. It is
    # not GPS/BEV georeferencing and can lose lock (see
    # GlobalMotionCompensator.consecutive_failures) under a hard scene cut,
    # fast motion, or low-texture frames.
    gmc_enabled: bool = False
    gmc_downscale: int = 4
    # Detection-independent "stalled and dense" corroborating signal (see
    # src/vn_traffic/analytics/stillness.py). bbox_union_occupancy and ROI
    # track count both depend on the detector resolving individual boxes, so
    # detector recall collapsing under severe occlusion (a tightly packed,
    # stalled crowd) produces a false-negative blind spot exactly when
    # congestion is worst. When enabled, a cell flagged both visually dense
    # (texture) and near-motionless (optical flow) corroborates CONGESTED
    # without requiring the (possibly-diluted) detected mean speed to also be
    # low. Thresholds below are demo-calibrated on one real frame, the same
    # honesty bar the occupancy/count thresholds are held to -- not
    # calibrated across multiple scenes.
    stillness_enabled: bool = False
    stillness_downscale: int = 4
    stillness_cell_px: int = 8
    stillness_motion_threshold: float = 1.0
    stillness_texture_threshold: float = 250.0
    stillness_congested_enter_fraction: float = 0.30
    stillness_congested_exit_fraction: float = 0.20
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
    prolonged_stop_enabled: bool = False
    prolonged_stop_classes: tuple[str, ...] = (
        "bus",
        "car",
        "motorcycle",
        "truck",
    )
    prolonged_stop_max_speed_px_s: float = 5.0
    prolonged_stop_release_speed_px_s: float = 10.0
    prolonged_stop_min_duration_s: float = 5.0
    prolonged_stop_max_gap_s: float = 1.0
    # How long the perception stage must produce literally zero raw
    # detections (any class, before included_classes filtering) before
    # perception_status flips from "reliable" to "detection_silence". This
    # is independent of congestion_state/traffic_state -- see
    # AnalyticsSnapshot.perception_status.
    detection_silence_min_duration_s: float = 2.0


@dataclass(frozen=True)
class StillnessHeatmapConfig:
    """Visual-only heatmap of src/vn_traffic/analytics/stillness.py's
    "stalled and dense" signal, independent of TrafficAnalytics/
    CongestionStateMachine. Uses a frame-RELATIVE texture threshold
    (texture_percentile of that frame's own distribution), unlike
    AnalyticsConfig.stillness_* (a fixed absolute threshold meant to be
    comparable across frames for a state-machine trigger). A relative
    threshold cannot drive a cross-frame-comparable scalar decision (see
    stillness.stalled_dense_score's docstring), but is validated to
    spatially localize a real packed/stalled cluster well frame by frame --
    this exists to show that to a human operator, not to automate a
    decision.
    """

    enabled: bool = False
    downscale: int = 4
    cell_px: int = 8
    motion_threshold: float = 1.0
    texture_percentile: float = 90.0
    alpha_max: float = 0.5
    # Exponential-moving-average smoothing across frames; the raw per-frame
    # score flickers (measured mean IoU 0.686 between consecutive frames'
    # thresholded masks on the motivating clip, 0.862 with this default --
    # see StillnessHeatmapRenderer's docstring). 0 disables smoothing.
    smoothing_decay: float = 0.85


@dataclass(frozen=True)
class EvidenceConfig:
    enabled: bool = False
    keyframe_event_types: tuple[str, ...] = (
        "line_crossing",
        "congestion_transition",
    )
    clip_event_types: tuple[str, ...] = ("congestion_transition",)
    pre_event_s: float = 2.0
    post_event_s: float = 3.0
    jpeg_quality: int = 90
    clip_codec: str = "mp4v"
    # Multi-view evidence (see reports/ke-hoach-pipeline-va-mo-phong.md WP1):
    # alongside the existing full-frame keyframe, also crop and write a
    # roi_crop (from the measurement manifest's roi_polygon, when present)
    # and an event_crop (a padded crop around the event's own track bbox at
    # its frame, when the event has a track_id resolvable in tracks.csv).
    # Off by default -- existing single-keyframe behavior is unchanged
    # unless a config opts in.
    multi_view_enabled: bool = False
    # Fractional padding added around the tight track bbox before cropping
    # for event_crop, so the crop shows some surrounding context instead of
    # just the vehicle's silhouette; clamped to frame bounds.
    event_crop_padding_ratio: float = 0.25


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    source: Path
    model: Path
    output_root: Path
    imgsz: int = 1280
    confidence: float = 0.4
    iou: float = 0.7
    max_det: int = 300
    show_labels: bool = True
    show_confidence: bool = True
    line_width: int = 2
    device: str = "0"
    tracker: str = "bytetrack.yaml"
    codec: str = "mp4v"
    fallback_fps: float = 30.0
    max_frames: int | None = None
    config_path: Path | None = None
    # Optional per-video measurement geometry (src/vn_traffic/measurement_manifest.py).
    # Its absence must never fail the pipeline or synthesize region/line
    # claims -- see PipelineRunner.run()'s handling and Gate G1 in
    # reports/ke-hoach-pipeline-va-mo-phong.md.
    measurement_manifest: Path | None = None
    analytics: AnalyticsConfig = AnalyticsConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    stillness_heatmap: StillnessHeatmapConfig = StillnessHeatmapConfig()

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
    abnormal = _mapping(analytics.get("abnormal"), "analytics.abnormal")
    perception_status = _mapping(
        analytics.get("perception"), "analytics.perception"
    )
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
    mode = str(analytics.get("mode", defaults.analytics_mode))
    mode_default_roi = (
        UAV_MOTION_DEFAULT_ROI_POLYGON
        if mode == "uav_motion"
        else defaults.roi_polygon
    )
    mode_default_line = (
        UAV_MOTION_DEFAULT_COUNTING_LINE
        if mode == "uav_motion"
        else defaults.counting_line
    )
    roi = analytics.get("roi_polygon")
    line = analytics.get("counting_line")
    config = AnalyticsConfig(
        enabled=bool(analytics.get("enabled", defaults.enabled)),
        analytics_mode=mode,
        included_classes=_event_types(
            analytics.get("included_classes"),
            "analytics.included_classes",
            defaults.included_classes,
        ),
        roi_polygon=(
            _normalized_points(roi, "analytics.roi_polygon")
            if roi is not None
            else mode_default_roi
        ),
        counting_line=(
            _normalized_points(line, "analytics.counting_line", 2)  # type: ignore[arg-type]
            if line is not None
            else mode_default_line
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
        gmc_enabled=bool(analytics.get("gmc_enabled", defaults.gmc_enabled)),
        gmc_downscale=int(
            analytics.get("gmc_downscale", defaults.gmc_downscale)
        ),
        stillness_enabled=bool(
            analytics.get("stillness_enabled", defaults.stillness_enabled)
        ),
        stillness_downscale=int(
            analytics.get("stillness_downscale", defaults.stillness_downscale)
        ),
        stillness_cell_px=int(
            analytics.get("stillness_cell_px", defaults.stillness_cell_px)
        ),
        stillness_motion_threshold=float(
            analytics.get(
                "stillness_motion_threshold", defaults.stillness_motion_threshold
            )
        ),
        stillness_texture_threshold=float(
            analytics.get(
                "stillness_texture_threshold", defaults.stillness_texture_threshold
            )
        ),
        stillness_congested_enter_fraction=float(
            congestion.get(
                "stillness_congested_enter_fraction",
                defaults.stillness_congested_enter_fraction,
            )
        ),
        stillness_congested_exit_fraction=float(
            congestion.get(
                "stillness_congested_exit_fraction",
                defaults.stillness_congested_exit_fraction,
            )
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
        prolonged_stop_enabled=bool(
            abnormal.get("prolonged_stop_enabled", defaults.prolonged_stop_enabled)
        ),
        prolonged_stop_classes=_event_types(
            abnormal.get("prolonged_stop_classes"),
            "analytics.abnormal.prolonged_stop_classes",
            defaults.prolonged_stop_classes,
        ),
        prolonged_stop_max_speed_px_s=float(
            abnormal.get(
                "prolonged_stop_max_speed_px_s",
                defaults.prolonged_stop_max_speed_px_s,
            )
        ),
        prolonged_stop_release_speed_px_s=float(
            abnormal.get(
                "prolonged_stop_release_speed_px_s",
                defaults.prolonged_stop_release_speed_px_s,
            )
        ),
        prolonged_stop_min_duration_s=float(
            abnormal.get(
                "prolonged_stop_min_duration_s",
                defaults.prolonged_stop_min_duration_s,
            )
        ),
        prolonged_stop_max_gap_s=float(
            abnormal.get(
                "prolonged_stop_max_gap_s", defaults.prolonged_stop_max_gap_s
            )
        ),
        detection_silence_min_duration_s=float(
            perception_status.get(
                "detection_silence_min_duration_s",
                defaults.detection_silence_min_duration_s,
            )
        ),
    )
    validate_analytics_config(config)
    return config


def _event_types(value: Any, name: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return defaults
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    return tuple(dict.fromkeys(value))


def _load_stillness_heatmap(raw: dict[str, Any]) -> StillnessHeatmapConfig:
    defaults = StillnessHeatmapConfig()
    section = _mapping(raw.get("stillness_heatmap"), "stillness_heatmap")
    config = StillnessHeatmapConfig(
        enabled=bool(section.get("enabled", defaults.enabled)),
        downscale=int(section.get("downscale", defaults.downscale)),
        cell_px=int(section.get("cell_px", defaults.cell_px)),
        motion_threshold=float(
            section.get("motion_threshold", defaults.motion_threshold)
        ),
        texture_percentile=float(
            section.get("texture_percentile", defaults.texture_percentile)
        ),
        alpha_max=float(section.get("alpha_max", defaults.alpha_max)),
        smoothing_decay=float(
            section.get("smoothing_decay", defaults.smoothing_decay)
        ),
    )
    validate_stillness_heatmap_config(config)
    return config


def validate_stillness_heatmap_config(config: StillnessHeatmapConfig) -> None:
    if config.downscale < 1:
        raise ValueError("stillness_heatmap.downscale must be at least one")
    if config.cell_px < 1:
        raise ValueError("stillness_heatmap.cell_px must be at least one")
    if config.motion_threshold < 0:
        raise ValueError("stillness_heatmap.motion_threshold cannot be negative")
    if not 0.0 <= config.texture_percentile <= 100.0:
        raise ValueError("stillness_heatmap.texture_percentile must be in [0, 100]")
    if not 0.0 <= config.alpha_max <= 1.0:
        raise ValueError("stillness_heatmap.alpha_max must be in [0, 1]")
    if not 0.0 <= config.smoothing_decay < 1.0:
        raise ValueError("stillness_heatmap.smoothing_decay must be in [0, 1)")


def _load_evidence(raw: dict[str, Any]) -> EvidenceConfig:
    defaults = EvidenceConfig()
    evidence = _mapping(raw.get("evidence"), "evidence")
    config = EvidenceConfig(
        enabled=bool(evidence.get("enabled", defaults.enabled)),
        keyframe_event_types=_event_types(
            evidence.get("keyframe_event_types"),
            "evidence.keyframe_event_types",
            defaults.keyframe_event_types,
        ),
        clip_event_types=_event_types(
            evidence.get("clip_event_types"),
            "evidence.clip_event_types",
            defaults.clip_event_types,
        ),
        pre_event_s=float(evidence.get("pre_event_s", defaults.pre_event_s)),
        post_event_s=float(evidence.get("post_event_s", defaults.post_event_s)),
        jpeg_quality=int(evidence.get("jpeg_quality", defaults.jpeg_quality)),
        clip_codec=str(evidence.get("clip_codec", defaults.clip_codec)),
        multi_view_enabled=bool(
            evidence.get("multi_view_enabled", defaults.multi_view_enabled)
        ),
        event_crop_padding_ratio=float(
            evidence.get(
                "event_crop_padding_ratio", defaults.event_crop_padding_ratio
            )
        ),
    )
    validate_evidence_config(config)
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
        max_det=int(perception.get("max_det", 300)),
        show_labels=bool(perception.get("show_labels", True)),
        show_confidence=bool(perception.get("show_confidence", True)),
        line_width=int(perception.get("line_width", 2)),
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
        measurement_manifest=(
            resolve_project_path(raw["measurement_manifest"])
            if raw.get("measurement_manifest")
            else None
        ),
        analytics=_load_analytics(raw),
        evidence=_load_evidence(raw),
        stillness_heatmap=_load_stillness_heatmap(raw),
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
    if config.max_det <= 0:
        raise ValueError("perception.max_det must be positive")
    if config.line_width <= 0:
        raise ValueError("perception.line_width must be positive")
    if len(config.codec) != 4:
        raise ValueError("video.codec must contain exactly four characters")
    if config.fallback_fps <= 0:
        raise ValueError("video.fallback_fps must be positive")
    if config.max_frames is not None and config.max_frames <= 0:
        raise ValueError("video.max_frames must be positive when provided")
    validate_analytics_config(config.analytics)
    validate_evidence_config(config.evidence)
    validate_stillness_heatmap_config(config.stillness_heatmap)


def validate_analytics_config(config: AnalyticsConfig) -> None:
    if config.analytics_mode not in ANALYTICS_MODES:
        raise ValueError(f"analytics.mode must be one of {ANALYTICS_MODES}")
    if not config.included_classes:
        raise ValueError("analytics.included_classes cannot be empty")
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
    if config.gmc_downscale < 1:
        raise ValueError("analytics.gmc_downscale must be at least one")
    if config.gmc_enabled and config.analytics_mode != "uav_motion":
        raise ValueError("analytics.gmc_enabled requires analytics.mode: uav_motion")
    if config.stillness_enabled and config.analytics_mode == "uav_motion":
        # stillness.py's optical flow is raw (no ego-motion compensation):
        # under camera pan/zoom every pixel appears to move regardless of
        # real object motion, so "near-motionless" stops meaning anything.
        # Also, the automatic CongestionStateMachine trigger this drives is
        # a confirmed negative result even on a static camera; do not
        # additionally enable it where its one input signal is invalid.
        # Revisit once stillness has GMC-based ego-motion compensation.
        raise ValueError(
            "analytics.stillness_enabled is not valid with analytics.mode: "
            "uav_motion (raw optical flow does not account for camera "
            "motion); use analytics.mode: fixed_camera, or wait for "
            "GMC-compensated stillness support"
        )
    if config.stillness_downscale < 1:
        raise ValueError("analytics.stillness_downscale must be at least one")
    if config.stillness_cell_px < 1:
        raise ValueError("analytics.stillness_cell_px must be at least one")
    if config.stillness_motion_threshold < 0:
        raise ValueError("analytics.stillness_motion_threshold cannot be negative")
    if config.stillness_texture_threshold < 0:
        raise ValueError("analytics.stillness_texture_threshold cannot be negative")
    if not 0.0 <= config.stillness_congested_exit_fraction <= 1.0 or not (
        0.0 <= config.stillness_congested_enter_fraction <= 1.0
    ):
        raise ValueError("stillness congestion fractions must be in [0, 1]")
    if (
        config.stillness_congested_exit_fraction
        >= config.stillness_congested_enter_fraction
    ):
        raise ValueError(
            "stillness_congested_exit_fraction must be below "
            "stillness_congested_enter_fraction"
        )
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
    if not config.prolonged_stop_classes:
        raise ValueError("prolonged_stop_classes cannot be empty")
    if config.prolonged_stop_max_speed_px_s < 0:
        raise ValueError("prolonged_stop_max_speed_px_s cannot be negative")
    if (
        config.prolonged_stop_release_speed_px_s
        < config.prolonged_stop_max_speed_px_s
    ):
        raise ValueError(
            "prolonged_stop_release_speed_px_s must not be below entry speed"
        )
    if config.prolonged_stop_min_duration_s <= 0:
        raise ValueError("prolonged_stop_min_duration_s must be positive")
    if config.prolonged_stop_max_gap_s <= 0:
        raise ValueError("prolonged_stop_max_gap_s must be positive")
    if config.detection_silence_min_duration_s <= 0:
        raise ValueError("detection_silence_min_duration_s must be positive")


def validate_evidence_config(config: EvidenceConfig) -> None:
    if config.pre_event_s < 0 or config.post_event_s < 0:
        raise ValueError("evidence pre/post durations cannot be negative")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("evidence.jpeg_quality must be in [1, 100]")
    if len(config.clip_codec) != 4:
        raise ValueError("evidence.clip_codec must contain exactly four characters")
    if config.event_crop_padding_ratio < 0:
        raise ValueError("evidence.event_crop_padding_ratio cannot be negative")
