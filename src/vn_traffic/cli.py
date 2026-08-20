"""Command-line interface for the offline-video MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_pipeline_config


DEFAULT_CONFIG = "configs/pipeline/offline_video.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vn-traffic",
        description="Run detection and ByteTrack over an offline traffic video.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Pipeline YAML")
    parser.add_argument("--source", help="Override input video path")
    parser.add_argument("--model", help="Override YOLO weights path")
    parser.add_argument("--device", help="Override device, for example 0 or cpu")
    parser.add_argument("--imgsz", type=int, help="Override inference image size")
    parser.add_argument(
        "--max-frames", type=int, help="Stop after N frames for integration checks"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config without loading YOLO"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(args.config).with_overrides(
        source=args.source,
        model=args.model,
        max_frames=args.max_frames,
        device=args.device,
        imgsz=args.imgsz,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "source": str(config.source),
                    "source_exists": config.source.is_file(),
                    "model": str(config.model),
                    "model_exists": config.model.is_file(),
                    "output_root": str(config.output_root),
                },
                indent=2,
            )
        )
        return 0 if config.source.is_file() and config.model.is_file() else 2

    from .perception import UltralyticsPerception
    from .runner import PipelineRunner

    perception = UltralyticsPerception(config)
    analytics = None
    overlay = None
    evidence_exporter = None
    if config.analytics.enabled:
        from .analytics import AnalyticsOverlay, TrafficAnalytics

        analytics = TrafficAnalytics(config.analytics)
        overlay = AnalyticsOverlay(config.analytics)
    if config.evidence.enabled:
        from .evidence import EventEvidenceExporter

        evidence_exporter = EventEvidenceExporter(config.evidence)
    heatmap_renderer = None
    if config.stillness_heatmap.enabled:
        from .analytics.stillness import StillnessHeatmapRenderer

        heatmap_renderer = StillnessHeatmapRenderer(
            downscale=config.stillness_heatmap.downscale,
            cell_px=config.stillness_heatmap.cell_px,
            motion_threshold=config.stillness_heatmap.motion_threshold,
            texture_percentile=config.stillness_heatmap.texture_percentile,
            alpha_max=config.stillness_heatmap.alpha_max,
        )
    run_dir = PipelineRunner(
        config,
        perception,
        event_processor=analytics,
        overlay_renderer=overlay,
        evidence_exporter=evidence_exporter,
        heatmap_renderer=heatmap_renderer,
    ).run()
    print(f"Pipeline completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
