"""Whole-video traffic-state description, batched by fixed time window.

This is a deliberately separate, lighter path from pipeline_stage.py's
per-event reasoning: that path exists to *describe an event the
deterministic stage already selected* (one VLM call per prolonged_stop/
congestion_transition, heavy JSON contract, evidence-refs audit trail) --
appropriate when the goal is an incident record with provenance. When the
goal is simply "what is the traffic condition in this video", per-event
calls are the wrong unit: (a) they miss ordinary conditions no event ever
fires on, and (b) N simultaneous prolonged_stop events in one jam produce N
near-duplicate VLM calls describing the same scene.

This path instead divides the whole run into fixed-length time windows
(independent of whether any event fired in it), summarizes each window from
already-computed analytics.csv/events.jsonl/tracks.csv (free, no model
call), and asks the VLM for one short description per window from a single
representative frame. It intentionally does not use contracts.py's
build_vlm_request/validate_vlm_assessment (that schema's incident_assessment
and per-observation confidence/evidence_refs machinery is calibration-grade
overhead this path does not need) -- see WINDOW_SCHEMA below instead.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import cv2

from .vlm_runtime import _resize_image, extract_json_object, load_vlm

if TYPE_CHECKING:
    from ..config import ReasoningConfig


_TRAFFIC_STATES = ("free_flow", "moderate", "congested", "uncertain")

WINDOW_SYSTEM_PROMPT = (
    "Bạn là trợ lý mô tả tình trạng giao thông từ ảnh chụp UAV/camera giao "
    "thông Việt Nam. Bạn được cho 1 ảnh và số liệu đo đạc đã tính sẵn (số "
    "lượng xe, độ chiếm dụng, số xe gần như đứng yên). Nhiệm vụ CHỈ là mô "
    "tả những gì thấy trong ảnh bằng tiếng Việt, không suy đoán nguyên nhân, "
    "không bịa số liệu ngoài ảnh. Luôn trả lời đúng 1 object JSON, không có "
    "văn bản nào khác ngoài JSON."
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _load_analytics_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_traffic_windows(
    run_dir: Path, *, window_seconds: float, fps: float, duration_s: float
) -> list[dict[str, Any]]:
    """Divide [0, duration_s] into fixed windows and summarize each from
    already-computed analytics/events/tracks -- no model call, no re-decode.

    A window's vehicle_counts is the peak simultaneous per-class count seen
    in it (from AnalyticsSnapshot.current_counts), not a sum -- summing
    would double-count a car present for the whole window. occupancy is the
    window's mean bbox_union_occupancy. stopped_tracks counts prolonged_stop
    events whose timestamp falls in the window (one per distinct track --
    engine.py only re-fires after the release threshold, so duplicates
    within one window are rare, not de-duplicated here for simplicity).
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    analytics_rows = _load_analytics_rows(run_dir / "analytics.csv")
    events = _load_jsonl(run_dir / "events.jsonl")

    window_count = max(1, -(-duration_s // window_seconds))  # ceil
    windows: list[dict[str, Any]] = []
    for index in range(int(window_count)):
        start_s = index * window_seconds
        end_s = min(duration_s, start_s + window_seconds)
        rows_in_window = [
            row for row in analytics_rows if start_s <= float(row["timestamp_s"]) < end_s
        ]
        peak_counts: dict[str, int] = {}
        occupancy_values: list[float] = []
        speed_values: list[float] = []
        for row in rows_in_window:
            counts = json.loads(row["current_counts_json"])
            for class_name, count in counts.items():
                peak_counts[class_name] = max(peak_counts.get(class_name, 0), count)
            occupancy_values.append(float(row["bbox_union_occupancy"]))
            if row["mean_speed_px_s"]:
                speed_values.append(float(row["mean_speed_px_s"]))
        stopped_tracks = sum(
            1
            for event in events
            if event["event_type"] == "prolonged_stop" and start_s <= event["timestamp_s"] < end_s
        )
        mean_occupancy = sum(occupancy_values) / len(occupancy_values) if occupancy_values else 0.0
        mean_speed = sum(speed_values) / len(speed_values) if speed_values else None
        # Coarse, uncalibrated heuristic (not the engine.py congestion state
        # machine) -- "low" just flags this window as worth a closer look,
        # it is not a validated threshold.
        motion_state = (
            "low" if mean_speed is not None and mean_speed < 20.0 else "normal"
        )
        windows.append(
            {
                "window_index": index,
                "window_start_s": start_s,
                "window_end_s": end_s,
                "event_ids": [
                    event["event_id"]
                    for event in events
                    if start_s <= event["timestamp_s"] < end_s
                ],
                "vehicle_counts": peak_counts,
                "stopped_tracks": stopped_tracks,
                "occupancy": round(mean_occupancy, 4),
                "mean_speed_px_s": mean_speed,
                "motion_state": motion_state,
                "representative_frame_index": round(
                    (start_s + end_s) / 2.0 * fps
                ),
            }
        )
    return windows


def extract_window_keyframe(
    source_video: Path, frame_index: int, out_path: Path, *, max_long_edge: int
) -> Path:
    """Grab one frame directly from the source video at frame_index -- not
    event-triggered, so EventEvidenceExporter's evidence.jsonl machinery
    does not apply here; this is a standalone seek-and-save."""
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video source: {source_video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"cannot decode frame {frame_index}: {source_video}")
    finally:
        capture.release()
    height, width = frame.shape[:2]
    longer_side = max(height, width)
    if longer_side > max_long_edge:
        scale = max_long_edge / longer_side
        frame = cv2.resize(
            frame, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise ValueError(f"cannot write keyframe: {out_path}")
    return out_path


def _window_prompt_text(window: dict[str, Any]) -> str:
    measured = {
        "window_start_s": window["window_start_s"],
        "window_end_s": window["window_end_s"],
        "vehicle_counts": window["vehicle_counts"],
        "stopped_tracks": window["stopped_tracks"],
        "occupancy": window["occupancy"],
        "motion_state": window["motion_state"],
    }
    output_shape = {
        "traffic_state": "free_flow | moderate | congested | uncertain",
        "observations": [
            {
                "claim_vi": "<mo ta mat do chung va muc do di chuyen, dung so lieu "
                "da cho de dien giai, khong bia them so lieu khac>",
                "evidence_refs": ["keyframe-1"],
            },
            {
                "claim_vi": "<mo ta chi tiet thi giac: loai phuong tien noi bat va "
                "mau sac, lan/huong cu the co gi khac biet, vi tri tuong doi cua "
                "cac xe trong khung hinh -- chi viet neu that su thay trong anh>",
                "evidence_refs": ["keyframe-1"],
            },
            {
                "claim_vi": "<mo ta boi canh & hanh vi giao thong: cach xe xep "
                "hang/giu khoang cach, co nguoi di bo/xe may/vach ke duong/bien "
                "bao dang chu y khong, dieu kien anh sang hoac thoi tiet neu ro "
                "-- chi viet neu that su thay trong anh, bo qua neu khong co gi "
                "dang chu y>",
                "evidence_refs": ["keyframe-1"],
            },
        ],
        "confidence": 0.5,
        "limitations": ["..."],
    }
    return (
        "Số liệu đã đo (không phải suy đoán, dùng để diễn giải ảnh):\n"
        + json.dumps(measured, ensure_ascii=False)
        + "\n\nMột ảnh đại diện của khoảng thời gian này được đính kèm.\n\n"
        "Trả về đúng 1 JSON object theo khuôn sau (claim_vi là placeholder mô "
        "tả cần viết gì, không phải văn bản mẫu để chép lại). Viết 2-4 "
        "observations bao quát các khía cạnh: mật độ/tốc độ tổng quan, chi "
        "tiết thị giác cụ thể (loại xe, màu sắc, làn/hướng, vị trí), bối cảnh "
        "và hành vi giao thông (cách xếp hàng, người đi bộ/xe máy, biển báo/"
        "vạch kẻ, ánh sáng/thời tiết nếu rõ) -- không cần đúng số lượng cố "
        "định, viết đủ để mô tả những gì thật sự thấy, mỗi claim_vi tối đa 2 "
        "câu ngắn gọn, không suy đoán nguyên nhân, không lặp lại ý đã nói ở "
        "observation khác. limitations tối đa 1 câu ngắn, hoặc để trống []. "
        "Không thêm chữ nào ngoài JSON:\n"
        + json.dumps(output_shape, ensure_ascii=False)
    )


def _validate_window_assessment(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("assessment must be a JSON object")
    if set(payload) != {"traffic_state", "observations", "confidence", "limitations"}:
        raise ValueError(f"unexpected keys: {sorted(payload)}")
    if payload["traffic_state"] not in _TRAFFIC_STATES:
        raise ValueError(f"traffic_state must be one of {_TRAFFIC_STATES}")
    observations = payload["observations"]
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty list")
    for observation in observations:
        if not isinstance(observation, dict) or set(observation) != {"claim_vi", "evidence_refs"}:
            raise ValueError("each observation needs exactly claim_vi and evidence_refs")
        if not isinstance(observation["claim_vi"], str) or not observation["claim_vi"].strip():
            raise ValueError("claim_vi must be non-empty text")
        if observation["claim_vi"].strip().startswith("<"):
            raise ValueError("claim_vi looks like the unedited placeholder")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not isinstance(payload["limitations"], list):
        raise ValueError("limitations must be a list")
    return payload


def run_window_vlm(
    *,
    window: dict[str, Any],
    image_path: Path,
    processor: Any,
    model: Any,
    system_prompt: str,
    max_long_edge: int,
    do_sample: bool,
    max_new_tokens: int,
    max_attempts: int,
) -> dict[str, Any]:
    """Minimal single-image generate + validate -- no contracts.py, no
    incident_assessment, no per-observation confidence/evidence-ref audit
    beyond checking the one ref name is self-consistent."""
    import torch
    from PIL import Image

    with Image.open(image_path) as source:
        image = _resize_image(source.convert("RGB"), max_long_edge)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _window_prompt_text(window)},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    contract_status = "invalid"
    contract_error = None
    assessment: dict[str, Any] | None = None
    raw_text = ""
    for attempt in range(max_attempts):
        # Greedy decoding is deterministic -- retrying a failed greedy
        # attempt with the same settings would just replay the identical
        # failure (see vlm_runtime.run_vlm_case's docstring on this same
        # point). Force sampling on any retry past the first attempt so a
        # retry can actually produce a different, possibly-valid output.
        attempt_do_sample = do_sample or attempt > 0
        if attempt_do_sample:
            torch.manual_seed(attempt)
        with torch.inference_mode():
            # repetition_penalty/no_repeat_ngram_size were tried here as a
            # fix for greedy decoding getting stuck repeating one sentence,
            # and measured to make things categorically worse: valid JSON
            # is legitimately repetitive at the syntax level ("claim_vi",
            # quotes, braces, the schema's own key names), and a penalty
            # strong enough to break a content repeat loop also breaks that
            # syntax -- observed output degenerated into incoherent
            # Python/JS code blocks and meta-commentary about the task
            # instead of doing it. Do not reintroduce without re-measuring
            # carefully. The actual mitigation is attempt_do_sample above
            # (retry forces real sampling, which does not get stuck the
            # same way) plus a higher max_attempts ceiling.
            output_ids = model.generate(
                **inputs, do_sample=attempt_do_sample, max_new_tokens=max_new_tokens,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_text = processor.decode(generated, skip_special_tokens=True)
        try:
            assessment = _validate_window_assessment(extract_json_object(raw_text))
            contract_status = "valid"
            contract_error = None
            break
        except ValueError as error:
            contract_status = "invalid"
            contract_error = str(error)
    torch.cuda.synchronize()
    return {
        "schema_version": 1,
        "elapsed_s": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "attempts_used": attempt + 1,
        "max_attempts": max_attempts,
        "contract_status": contract_status,
        "contract_error": contract_error,
        "raw_text": raw_text,
        "assessment": assessment,
    }


def run_traffic_window_stage(
    *, reasoning_config: "ReasoningConfig", run_dir: Path, project_root: Path,
) -> Path:
    """Describe the whole run in fixed time windows, writing
    run_dir/traffic_windows.jsonl -- see module docstring for why this is a
    separate path from pipeline_stage.run_reasoning_stage."""
    output_path = run_dir / "traffic_windows.jsonl"
    run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    fps = run_meta["video"]["fps"]
    frames_processed = run_meta.get("frames_processed", 0)
    duration_s = frames_processed / fps if fps else 0.0
    if duration_s <= 0:
        output_path.write_text("", encoding="utf-8")
        return output_path

    windows = build_traffic_windows(
        run_dir,
        window_seconds=reasoning_config.window_seconds,
        fps=fps,
        duration_s=duration_s,
    )
    print(f"[traffic_window] {len(windows)} window(s) over {duration_s:.1f}s", flush=True)

    source_video = Path(run_meta["source"])
    frames_dir = run_dir / "traffic_windows"
    for window in windows:
        keyframe_path = frames_dir / f"window-{window['window_index']:03d}.jpg"
        extract_window_keyframe(
            source_video, window["representative_frame_index"], keyframe_path,
            max_long_edge=reasoning_config.max_long_edge,
        )
        window["representative_keyframe"] = keyframe_path.relative_to(run_dir).as_posix()

    print("[traffic_window] loading VLM...", flush=True)
    load_started = time.perf_counter()
    processor, model = load_vlm(reasoning_config.vlm_model_dir)
    print(f"[traffic_window] VLM loaded in {time.perf_counter() - load_started:.1f}s", flush=True)

    prompts_path = project_root / "configs" / "reasoning" / reasoning_config.prompts
    system_prompt = WINDOW_SYSTEM_PROMPT
    if prompts_path.is_file():
        import yaml

        loaded = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))
        system_prompt = loaded.get("traffic_window", {}).get("system", WINDOW_SYSTEM_PROMPT)

    records: list[dict[str, Any]] = []
    try:
        for window in windows:
            started = time.perf_counter()
            result = run_window_vlm(
                window=window,
                image_path=run_dir / window["representative_keyframe"],
                processor=processor,
                model=model,
                system_prompt=system_prompt,
                max_long_edge=reasoning_config.max_long_edge,
                do_sample=reasoning_config.vlm.get("do_sample", False),
                max_new_tokens=reasoning_config.vlm.get("max_new_tokens", 128),
                max_attempts=reasoning_config.vlm.get("max_attempts", 1),
            )
            print(
                f"[traffic_window] {window['window_index'] + 1}/{len(windows)} "
                f"status={result['contract_status']} "
                f"attempts={result['attempts_used']}/{result['max_attempts']} "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
            records.append({**window, "vlm": result})
    finally:
        del model, processor
        import torch

        torch.cuda.empty_cache()

    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_path = run_dir / "traffic_windows_report.txt"
    report_path.write_text(_render_report(run_dir, records), encoding="utf-8")
    print(f"[traffic_window] report: {report_path}", flush=True)
    return output_path


def _render_report(run_dir: Path, records: list[dict[str, Any]]) -> str:
    """Human-readable companion to traffic_windows.jsonl -- one compact
    JSON object per line is easy to grep/parse but unreadable opened
    directly in an editor (the description is buried inside a single very
    long line, doubled by the raw model text). This renders just the part
    a person actually wants: time range, measured stats, and the VLM's
    own description, per window."""
    lines = [f"Traffic window report -- {run_dir.name}", "=" * 60, ""]
    for record in records:
        start_s, end_s = record["window_start_s"], record["window_end_s"]
        lines.append(f"Window {record['window_index'] + 1}  [{start_s:.0f}s - {end_s:.0f}s]")
        lines.append("-" * 40)
        counts = ", ".join(f"{name}={count}" for name, count in sorted(record["vehicle_counts"].items()))
        lines.append(
            f"Đo được: {counts or '(không có)'} | dừng={record['stopped_tracks']} "
            f"| occupancy={record['occupancy']:.1%} | motion={record['motion_state']}"
        )
        vlm = record["vlm"]
        if vlm["contract_status"] != "valid" or vlm["assessment"] is None:
            lines.append(f"[VLM không hợp lệ: {vlm['contract_error']}]")
            lines.append("")
            continue
        assessment = vlm["assessment"]
        lines.append(f"Trạng thái (VLM): {assessment['traffic_state']} (confidence={assessment['confidence']})")
        lines.append("Mô tả:")
        for index, observation in enumerate(assessment["observations"], start=1):
            lines.append(f"  {index}. {observation['claim_vi']}")
        if assessment["limitations"]:
            lines.append("Hạn chế:")
            for limitation in assessment["limitations"]:
                lines.append(f"  - {limitation}")
        lines.append("")
    return "\n".join(lines)
