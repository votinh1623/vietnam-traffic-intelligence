"""Compute HOTA/DetA/AssA for an existing benchmark_tracking.py run.

Reuses the prediction CSVs a completed tracking benchmark already wrote to
disk (no re-running detection/tracking) and the same VisDrone ground truth,
class map, and per-(sequence, class) split as the MOTA/IDF1 evaluation in
benchmark_tracking.py, so results are directly comparable to the existing
metrics.csv row for the same run.

HOTA was recorded as "pending TrackEval integration" everywhere in this
project's docs; this script is that integration; see hota_metrics.py for the
underlying adapter and tests/test_hota_metrics.py for its verification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from benchmark_tracking import (
    file_manifest_sha256,
    file_sha256,
    load_visdrone_ground_truth,
    resolve_path,
)
from hota_metrics import compute_hota, compute_many_hota


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Same evaluation config used for the benchmark_tracking.py run.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="predictions/ directory a completed benchmark_tracking.py run wrote.",
    )
    parser.add_argument("--output", type=Path, help="Defaults to <predictions-dir>/../hota.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    model_config = config["model"]
    dataset_root = resolve_path(dataset["root"])
    annotations_root = dataset_root / dataset["annotations_dir"]
    class_map = {int(key): value for key, value in dataset["class_map"].items()}
    expected_classes = list(model_config["expected_classes"])
    class_names = list(model_config.get("evaluation_classes", expected_classes))

    predictions_dir = resolve_path(args.predictions_dir)
    prediction_paths = sorted(predictions_dir.glob("*.csv"))
    if not prediction_paths:
        raise ValueError(f"no prediction CSVs found in {predictions_dir}")

    started = time.perf_counter()
    results_by_name: dict[str, dict] = {}
    annotation_hashes: dict[str, str] = {}
    for prediction_path in prediction_paths:
        sequence = prediction_path.stem
        annotation_path = annotations_root / f"{sequence}.txt"
        annotation_hashes[sequence] = file_sha256(annotation_path)
        ground_truth = load_visdrone_ground_truth(annotation_path, class_map)
        predictions = pd.read_csv(prediction_path)
        # Do NOT pre-filter ground_truth to predictions["frame"] here:
        # build_hota_data() already unions GT and prediction frames itself
        # (matching tracking_metrics.py's MOTA/IDF1 accumulator), so a GT
        # frame with zero predictions correctly becomes a false-negative
        # frame. Pre-filtering here discarded those frames before the union
        # ever ran, silently inflating HOTA/DetA by hiding undetected GT.
        for class_name in class_names:
            gt_class = ground_truth[ground_truth["class_name"] == class_name]
            pred_class = predictions[predictions["class_name"] == class_name]
            results_by_name[f"{sequence}:{class_name}"] = compute_hota(
                gt_class, pred_class
            )

    summary = compute_many_hota(results_by_name)
    elapsed_s = time.perf_counter() - started

    output = resolve_path(
        args.output or (predictions_dir.parent / "hota.json")
    )
    record = {
        "schema_version": 1,
        "config": {"path": str(args.config), "sha256": file_sha256(config_path)},
        "predictions": {
            "dir": str(predictions_dir),
            "manifest_sha256": file_manifest_sha256(prediction_paths, predictions_dir),
        },
        "dataset": {
            "root": dataset["root"],
            "annotation_sha256": annotation_hashes,
            "class_map": class_map,
        },
        "per_sequence_class_count": len(results_by_name),
        "overall": {key: value for key, value in summary.items()},
        "elapsed_s": elapsed_s,
        "claim_boundary": (
            "HOTA/DetA/AssA computed with TrackEval's own metric implementation "
            "via a custom in-memory adapter (not TrackEval's MOTChallenge file "
            "dataset loader, which only supports a single 'pedestrian' class). "
            "Same per-(sequence, class) split, VisDrone ground truth, and "
            "predictions as the corresponding MOTA/IDF1 run; not an official "
            "VisDrone or MOTChallenge benchmark."
        ),
    }
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary.to_string())
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
