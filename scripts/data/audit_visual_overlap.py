"""Audit possible cross-source visual overlap with dependency-free dHash.

This complements source-name grouping. It is intended to catch renamed clips,
re-uploads, and repeated frames that byte hashes cannot detect.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def difference_hash(path: Path, width: int = 16, height: int = 16) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    resized = cv2.resize(image, (width + 1, height), interpolation=cv2.INTER_AREA)
    return (resized[:, 1:] > resized[:, :-1]).reshape(-1)


def hamming(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit(
    rows: list[dict[str, str]],
    dataset_root: Path,
    focus_sources: set[str],
    threshold: int,
) -> dict[str, object]:
    hashes: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for row in rows:
        source = row["source_id"]
        if source == "unknown":
            continue
        image_path = dataset_root / row["image_path"]
        hashes[source].append((row["image_path"], difference_hash(image_path)))

    comparisons = []
    all_sources = sorted(hashes)
    for focus in sorted(focus_sources):
        if focus not in hashes:
            comparisons.append({"focus_source": focus, "status": "missing"})
            continue
        for other in all_sources:
            if other == focus:
                continue
            if other in focus_sources and other < focus:
                continue
            best_distance = 10**9
            best_pair: tuple[str, str] | None = None
            matches_at_threshold = 0
            for left_path, left_hash in hashes[focus]:
                for right_path, right_hash in hashes[other]:
                    distance = hamming(left_hash, right_hash)
                    if distance < best_distance:
                        best_distance = distance
                        best_pair = (left_path, right_path)
                    if distance <= threshold:
                        matches_at_threshold += 1
            comparisons.append(
                {
                    "focus_source": focus,
                    "other_source": other,
                    "minimum_hamming": best_distance,
                    "matches_at_or_below_threshold": matches_at_threshold,
                    "best_focus_image": best_pair[0] if best_pair else "",
                    "best_other_image": best_pair[1] if best_pair else "",
                    "status": "possible_overlap" if matches_at_threshold else "no_match",
                }
            )

    possible = [row for row in comparisons if row.get("status") == "possible_overlap"]
    return {
        "method": "16x16 grayscale difference hash",
        "threshold": threshold,
        "interpretation": "candidate generator only; visual review is required",
        "focus_sources": sorted(focus_sources),
        "comparison_count": len(comparisons),
        "possible_overlap_count": len(possible),
        "possible_overlaps": possible,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--focus", nargs="+", required=True)
    parser.add_argument("--threshold", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit(
        read_manifest(args.manifest.resolve()),
        args.dataset_root.resolve(),
        set(args.focus),
        args.threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "comparisons"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
