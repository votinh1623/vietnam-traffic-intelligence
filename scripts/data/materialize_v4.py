"""Materialize a source-grouped YOLO detection dataset without editing inputs.

Input rows come from ``audit_dataset.py``. Polygon annotations are converted
to bounding boxes. Repeated ``(source_id, frame_index)`` records are emitted
once when their converted labels agree; conflicting annotations are excluded
and reported for manual review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


CLASS_NAMES = ["bus", "car", "motorcycle", "pedestrian", "truck"]
SPLITS = {"train", "calibration", "validation", "test"}


def _number(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def convert_annotation_line(line: str, class_count: int = 5) -> tuple[str, str]:
    """Return a normalized YOLO bbox line and source format name."""
    parts = line.split()
    if not parts:
        raise ValueError("empty annotation line")
    try:
        class_id = int(parts[0])
    except ValueError as error:
        raise ValueError(f"invalid class id: {parts[0]}") from error
    if not 0 <= class_id < class_count:
        raise ValueError(f"class id outside [0, {class_count - 1}]: {class_id}")
    try:
        values = [float(value) for value in parts[1:]]
    except ValueError as error:
        raise ValueError("annotation coordinates must be numeric") from error

    if len(values) == 4:
        center_x, center_y, width, height = values
        source_format = "bbox"
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        source_format = "polygon"
    else:
        raise ValueError(f"expected bbox or polygon coordinates, got {len(values)}")

    coordinates = (center_x, center_y, width, height)
    if not all(0 <= value <= 1 for value in coordinates):
        raise ValueError(f"coordinates outside normalized range: {coordinates}")
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive box dimensions: {(width, height)}")
    converted = " ".join([str(class_id), *(_number(value) for value in coordinates)])
    return converted, source_format


def convert_label(path: Path, class_count: int = 5) -> tuple[str, Counter[str]]:
    converted: list[str] = []
    seen: set[str] = set()
    formats: Counter[str] = Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                bbox, source_format = convert_annotation_line(line, class_count)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            formats[source_format] += 1
            if bbox in seen:
                formats["exact_duplicate_removed"] += 1
                continue
            seen.add(bbox)
            converted.append(bbox)
    return "\n".join(converted) + ("\n" if converted else ""), formats


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_source_proposal(path: Path) -> dict[str, str]:
    proposal: dict[str, str] = {}
    for row in read_csv(path):
        source = row["source_id"]
        split = row["proposed_split"]
        if split not in SPLITS:
            raise ValueError(f"invalid proposed split for {source}: {split}")
        if source in proposal:
            raise ValueError(f"duplicate source in proposal: {source}")
        proposal[source] = split
    return proposal


def _record_key(row: dict[str, str]) -> tuple[str, str]:
    frame = row["frame_index"] or row["image_id"]
    return row["source_id"], frame


def select_records(
    manifest_rows: list[dict[str, str]],
    dataset_root: Path,
    source_proposal: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    exclusions: list[dict[str, object]] = []
    for row in manifest_rows:
        source = row["source_id"]
        if source == "unknown" or source not in source_proposal:
            exclusions.append(
                {
                    "reason": "unknown_or_unassigned_source",
                    "source_id": source,
                    "frame_index": row["frame_index"],
                    "image_paths": row["image_path"],
                }
            )
            continue
        grouped[_record_key(row)].append(row)

    selected: list[dict[str, object]] = []
    for (source, frame), candidates in sorted(grouped.items()):
        converted_candidates: list[tuple[dict[str, str], str, Counter[str]]] = []
        errors = []
        for row in candidates:
            label_path = dataset_root / row["label_path"]
            try:
                converted, formats = convert_label(label_path, len(CLASS_NAMES))
            except (OSError, ValueError) as error:
                errors.append(f"{row['label_path']}: {error}")
                continue
            converted_candidates.append((row, converted, formats))
        if errors:
            exclusions.append(
                {
                    "reason": "invalid_label",
                    "source_id": source,
                    "frame_index": frame,
                    "image_paths": ";".join(row["image_path"] for row in candidates),
                    "details": " | ".join(errors),
                }
            )
            continue

        unique_labels = {converted for _, converted, _ in converted_candidates}
        if len(unique_labels) > 1:
            exclusions.append(
                {
                    "reason": "duplicate_frame_label_conflict",
                    "source_id": source,
                    "frame_index": frame,
                    "image_paths": ";".join(row["image_path"] for row in candidates),
                    "details": f"{len(unique_labels)} distinct converted labels",
                }
            )
            continue

        row, converted, formats = min(
            converted_candidates, key=lambda item: item[0]["image_path"]
        )
        selected.append(
            {
                "source_id": source,
                "frame_index": frame,
                "target_split": source_proposal[source],
                "source_image_path": row["image_path"],
                "source_label_path": row["label_path"],
                "source_content_sha256": row["content_sha256"],
                "converted_label": converted,
                "bbox_count": len(converted.splitlines()),
                "source_bbox_count": formats["bbox"],
                "source_polygon_count": formats["polygon"],
                "exact_duplicate_removed": formats["exact_duplicate_removed"],
                "duplicate_candidates": len(candidates),
            }
        )
    return selected, exclusions


def _target_stem(source: str, frame: str) -> str:
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    if frame.isdigit():
        return f"{source_hash}_frame_{int(frame):06d}"
    return f"{source_hash}_{hashlib.sha256(frame.encode()).hexdigest()[:12]}"


def materialize(
    selected: Iterable[dict[str, object]],
    exclusions: list[dict[str, object]],
    dataset_root: Path,
    output_root: Path,
) -> dict[str, object]:
    output_root = output_root.resolve()
    staging = output_root.with_name(f"{output_root.name}.building")
    if output_root.exists() or staging.exists():
        raise FileExistsError(
            f"output or staging directory already exists: {output_root} / {staging}"
        )
    staging.mkdir(parents=True)

    output_manifest: list[dict[str, object]] = []
    split_counts: Counter[str] = Counter()
    conversion_counts: Counter[str] = Counter()
    try:
        for record in selected:
            split = str(record["target_split"])
            image_dir = staging / split / "images"
            label_dir = staging / split / "labels"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            source_image = dataset_root / str(record["source_image_path"])
            stem = _target_stem(str(record["source_id"]), str(record["frame_index"]))
            target_image = image_dir / f"{stem}{source_image.suffix.lower()}"
            target_label = label_dir / f"{stem}.txt"
            shutil.copy2(source_image, target_image)
            target_label.write_text(str(record["converted_label"]), encoding="utf-8")

            split_counts[split] += 1
            conversion_counts["bbox_passthrough"] += int(record["source_bbox_count"])
            conversion_counts["polygon_to_bbox"] += int(record["source_polygon_count"])
            conversion_counts["exact_duplicate_removed"] += int(
                record["exact_duplicate_removed"]
            )
            output_manifest.append(
                {
                    "image_id": record["source_content_sha256"],
                    "source_id": record["source_id"],
                    "frame_index": record["frame_index"],
                    "split": split,
                    "image_path": target_image.relative_to(staging).as_posix(),
                    "label_path": target_label.relative_to(staging).as_posix(),
                    "bbox_count": record["bbox_count"],
                    "source_image_path": record["source_image_path"],
                    "duplicate_candidates": record["duplicate_candidates"],
                }
            )

        data_yaml = "\n".join(
            [
                "# Generated by scripts/data/materialize_v4.py",
                "train: train/images",
                "val: validation/images",
                "test: test/images",
                "",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {CLASS_NAMES!r}",
                "",
            ]
        )
        (staging / "data.yaml").write_text(data_yaml, encoding="utf-8")
        write_csv(staging / "manifest.csv", output_manifest)
        if exclusions:
            normalized_exclusions = []
            fields = ("reason", "source_id", "frame_index", "image_paths", "details")
            for row in exclusions:
                normalized_exclusions.append({field: row.get(field, "") for field in fields})
            write_csv(staging / "exclusions.csv", normalized_exclusions)

        report = {
            "status": "materialized_requires_audit",
            "source_dataset": str(dataset_root),
            "image_count": len(output_manifest),
            "split_counts": dict(sorted(split_counts.items())),
            "conversion_counts": dict(sorted(conversion_counts.items())),
            "exclusion_entry_count": len(exclusions),
            "exclusion_reasons": dict(
                sorted(Counter(str(row["reason"]) for row in exclusions).items())
            ),
            "classes": CLASS_NAMES,
        }
        (staging / "materialization_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.rename(output_root)
        return report
    except Exception:
        # Preserve staging for inspection; never delete potentially useful output.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-proposal", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    manifest_rows = read_csv(args.manifest.resolve())
    proposal = load_source_proposal(args.source_proposal.resolve())
    selected, exclusions = select_records(manifest_rows, dataset_root, proposal)
    report = materialize(selected, exclusions, dataset_root, args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
