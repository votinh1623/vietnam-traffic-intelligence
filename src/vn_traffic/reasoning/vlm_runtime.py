"""Transformers adapter for one frozen development VLM case."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from .contracts import ContractError, build_vlm_request, validate_vlm_assessment
from .freeze import file_sha256, verify_evidence_lock


_KEYFRAME_MOTION_PHRASES = (
    "đang di chuyển",
    "đang đi",
    "đang chạy",
    "đi qua",
    "đi lên",
    "đi xuống",
    "hướng lên",
    "hướng xuống",
    "hướng về",
    " is moving",
    " are moving",
)


def validate_grounding_policy(
    assessment: dict[str, Any],
    request: dict[str, Any],
    *,
    clip_frames_shown: bool,
) -> None:
    """Reject motion claims unless the model was actually shown clip frames.

    `clip_frames_shown` must reflect what was actually fed to the model,
    not whether the request references clip evidence -- `run_vlm_case`
    below only sets this True when it actually decoded and sent sampled
    clip frames (config.vlm.clip_sample_frames > 0 and evidence.clips is
    non-empty); a request with clip evidence the model never saw must
    still be treated as keyframe-only.
    """
    if clip_frames_shown:
        return
    for index, observation in enumerate(assessment["observations"]):
        claim = " " + observation["claim_vi"].casefold()
        if any(phrase in claim for phrase in _KEYFRAME_MOTION_PHRASES):
            raise ContractError(
                f"observations[{index}] claims motion from keyframe-only evidence"
            )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one top-level JSON object, allowing an optional Markdown fence."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        trailing = text[index + end :].strip()
        if trailing and trailing not in {"```", "```json"}:
            continue
        return payload
    raise ValueError("VLM output does not contain one valid JSON object")


def load_prompts(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Load the prompt file the config declares under `prompts:`.

    Both reasoning CLIs (run_vlm.py, run_llm.py) must call this instead of
    hardcoding a prompts_*.yaml filename -- a config that still points at
    an older prompt version (e.g. prompts_v1.yaml, which has the fixed
    prompt-copying bug prompts_v3.yaml fixed) should be visible in the
    config file, not buried in the CLI.
    """
    import yaml

    prompts_name = config.get("prompts")
    if not isinstance(prompts_name, str) or not prompts_name:
        raise ValueError("config must declare a `prompts:` file name")
    prompts_path = project_root / "configs" / "reasoning" / prompts_name
    if not prompts_path.is_file():
        raise FileNotFoundError(f"declared prompts file is unavailable: {prompts_path}")
    return yaml.safe_load(prompts_path.read_text(encoding="utf-8"))


def load_development_case(
    config_path: Path, case_id: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("split") != "development":
        raise ValueError("VLM development runner refuses non-development splits")
    lock_path = Path(config["input_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_evidence_lock(lock)
    if lock.get("split") != "development":
        raise ValueError("reasoning lock is not a development split")
    try:
        case = next(item for item in lock["cases"] if item["case_id"] == case_id)
    except StopIteration as error:
        raise ValueError(f"unknown development case: {case_id}") from error
    request = build_vlm_request(case)
    artifact_root = Path(config["artifact_root"])
    for collection in ("keyframes", "clips"):
        for artifact in request["evidence"][collection]:
            path = artifact_root / artifact["path"]
            if not path.is_file():
                raise FileNotFoundError(f"evidence artifact is unavailable: {path}")
            if file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"evidence artifact SHA-256 mismatch: {path}")
    return config, request, artifact_root


def _multi_view_note(request: dict[str, Any]) -> str:
    """Explain multiple images, when present, as same-instant crops -- not
    a sequence over time. Without this, a model shown 2-3 images risks
    inferring motion between them, which validate_grounding_policy would
    then correctly reject since clip_frames_shown is still False for these
    (still images of one moment carry no motion information, unlike real
    clip frames)."""
    keyframes = request["evidence"]["keyframes"]
    if len(keyframes) <= 1:
        return ""
    refs = ", ".join(keyframe["ref"] for keyframe in keyframes)
    return (
        f"\n\n{len(keyframes)} images are provided ({refs}), in this order: "
        "full frame, then (if present) a cropped region of interest, then "
        "(if present) a tight crop around the specific vehicle in this "
        "event. All images are the SAME instant in time, only the field of "
        "view differs -- they are not a sequence and show no motion. Do "
        "not describe movement, direction, or change between them."
    )


def _clip_sequence_note(frame_count: int, timestamps_s: list[float]) -> str:
    """Explain that these images are REAL frames sampled across a clip's
    time window, in order -- the opposite framing from _multi_view_note.
    Only used when run_vlm_case actually decoded frames from
    evidence.clips (clip_frames_shown=True passed to
    validate_grounding_policy), so motion claims here are grounded and
    the motion-phrase ban does not apply.
    """
    origin = timestamps_s[0] if timestamps_s else 0.0
    offsets = ", ".join(f"{t - origin:.2f}s" for t in timestamps_s)
    span_s = (timestamps_s[-1] - origin) if len(timestamps_s) > 1 else 0.0
    return (
        f"\n\n{frame_count} images are provided, in this order: real video "
        f"frames sampled across a {span_s:.2f}s window around this event, "
        f"at offsets [{offsets}] from the first frame. This IS a real "
        "sequence over time -- describing motion, stopping, or change you "
        "actually see between these frames is expected and grounded, not "
        "speculation. Do not describe anything you cannot see change or "
        "persist across the actual frames shown."
    )


# Qwen3-VL's self-attention memory over concatenated image+text tokens
# grows faster than linearly with total image count at a given
# resolution -- 3 clip frames at the source's native 1920x1080 (the same
# resolution one keyframe already runs fine at) OOM'd a 6GB card ("Tried
# to allocate 3.34 GiB" with 8.23 GiB already allocated), not 3x the cost
# of one frame. Downscaling each sampled frame keeps total token count
# for a multi-frame clip within the same rough budget one full-resolution
# keyframe already uses.
_CLIP_FRAME_MAX_SIDE = 768


def _sample_clip_frames(clip_path: Path, count: int) -> tuple[list[Any], list[float]]:
    """Evenly sample up to `count` frames across a clip video, in order,
    downscaled to _CLIP_FRAME_MAX_SIDE on the longer edge (see note above).

    Returns (images, timestamps_s) with matching order/length. Uses
    cv2.CAP_PROP_POS_FRAMES seeking rather than sequential decode since
    clips are short (evidence.py's pre/post_event_s window) and this
    keeps the sampling logic independent of clip length.
    """
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open evidence clip: {clip_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        if total <= 0:
            raise ValueError(f"evidence clip has no frames: {clip_path}")
        sample_count = max(1, min(count, total))
        indices = sorted(
            {
                round(i * (total - 1) / max(1, sample_count - 1))
                for i in range(sample_count)
            }
        )
        images: list[Any] = []
        timestamps: list[float] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"cannot decode clip frame {index}: {clip_path}")
            image = Image.fromarray(frame[:, :, ::-1])
            longer_side = max(image.width, image.height)
            if longer_side > _CLIP_FRAME_MAX_SIDE:
                scale = _CLIP_FRAME_MAX_SIDE / longer_side
                image = image.resize(
                    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                    Image.BILINEAR,
                )
            images.append(image)
            timestamps.append(index / fps)
        return images, timestamps
    finally:
        capture.release()


def _prompt_text(request: dict[str, Any], sequence_note: str) -> str:
    event = request["event"]
    # The placeholder evidence_refs example must name a ref that actually
    # exists in THIS request -- a request built from clip-only evidence
    # (see run_vlm_case's clip_frames_shown path) has no "keyframe-1" at
    # all, and a hardcoded "keyframe-1" here got copied verbatim into
    # observations[0].evidence_refs, which validate_vlm_assessment then
    # correctly rejected as "cites unknown evidence" -- measured on the
    # a probe set built offline from UIT-ADrone frame-level anomaly
    # ground truth, which is clip-only by construction.
    available_refs = [
        item["ref"]
        for item in request["evidence"]["keyframes"] + request["evidence"]["clips"]
    ]
    example_ref = available_refs[0]
    visual_context = {
        field: event[field]
        for field in (
            "schema_version",
            "event_id",
            "event_type",
            "frame_index",
            "timestamp_s",
        )
    }
    # claim_vi below is a schema placeholder, not a model answer: it starts
    # with "<" and reads as an instruction, not prose, specifically so a
    # small model cannot satisfy the request by copying it verbatim. Every
    # run before this fix reproduced the previous literal example
    # ("Quan sát thấy các phương tiện trong khung hình.") word for word --
    # see output/reasoning/adhoc/*.json and output/reasoning/dev_v1/*.json --
    # so the placeholder must not itself be valid-looking Vietnamese prose.
    # This text is shared code, not versioned per prompts_vN.yaml -- keep
    # it in sync with whatever the active system prompt actually asks for
    # (see prompts_v7.yaml's note: a stale v3-era density/vehicle-type
    # placeholder sat here unrelated to v4-v6's actual task, and was a
    # plausible source of off-task echoed text).
    output_shape = {
        "schema_version": 1,
        "case_id": request["case_id"],
        "event_id": request["event"]["event_id"],
        "observations": [
            {
                "claim_vi": "<mo ta dau hieu bat thuong THAT su nhin thay trong "
                "anh nay, hoac neu khong co thi mo ta ngan gon noi dung anh de "
                "chung minh da xem anh; KHONG duoc chep nguyen van vi du nay>",
                "confidence": 0.5,
                "evidence_refs": [example_ref],
            }
        ],
        "incident_assessment": {
            "status": "uncertain",
            "category": "none",
            "confidence": 0.5,
        },
        "limitations": ["..."],
    }
    return (
        "Event identity JSON (not visual ground truth):\n"
        + json.dumps(visual_context, ensure_ascii=False, sort_keys=True)
        + sequence_note
        + "\n\nOutput exactly one JSON object matching this shape (claim_vi "
        "below is a placeholder describing what to write, not example text "
        "to copy):\n"
        + json.dumps(output_shape, ensure_ascii=False)
    )


def load_vlm(model_dir: Path) -> tuple[Any, Any]:
    """Load (processor, model) once for reuse across many run_vlm_case
    calls -- the default per-call load in run_vlm_case exists for
    execution_policy: sequential_load_run_unload (crash isolation: a bad
    generate() call only loses that one case's process), but reloading a
    ~4GB checkpoint from disk on every case makes a many-case batch
    dramatically slower than necessary when the run is otherwise stable.
    Pass the result to run_vlm_case's model/processor params to reuse it.
    """
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not model_dir.is_dir():
        raise FileNotFoundError(f"local VLM directory is unavailable: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float16,
        device_map="cuda",
    )
    return processor, model


def run_vlm_case(
    *,
    config: dict[str, Any],
    request: dict[str, Any],
    artifact_root: Path,
    model_dir: Path,
    system_prompt: str,
    processor: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Generate once and validate its JSON. Loads the pinned local model
    unless `processor`/`model` are both given (see load_vlm) -- pass both
    or neither, never just one."""
    import torch
    from PIL import Image

    if (processor is None) != (model is None):
        raise ValueError("pass both processor and model, or neither")
    if model is None and not model_dir.is_dir():
        raise FileNotFoundError(f"local VLM directory is unavailable: {model_dir}")
    keyframes = request["evidence"]["keyframes"]
    clips = request["evidence"].get("clips", [])
    clip_sample_frames = config["vlm"].get("clip_sample_frames", 0)
    clip_frames_shown = bool(clips) and clip_sample_frames > 0
    if clip_frames_shown:
        # A real clip carries actual motion over time, unlike keyframes
        # (still images of one instant) -- when evidence.clips is present
        # and the config asks for sampled frames, show those instead of
        # the keyframe views, and tell the model so via
        # _clip_sequence_note. clip_frames_shown is passed to
        # validate_grounding_policy below to lift the motion-phrase ban
        # only for this actually-sequential evidence.
        clip_path = artifact_root / clips[0]["path"]
        images, timestamps = _sample_clip_frames(clip_path, clip_sample_frames)
        sequence_note = _clip_sequence_note(len(images), timestamps)
    else:
        if not keyframes:
            raise ValueError("keyframe-first VLM policy requires a keyframe")
        # All entries in evidence.keyframes are loaded and shown, not just
        # the first -- multi-view evidence (full frame + ROI crop + event
        # crop, see src/vn_traffic/evidence.py) puts additional still-image
        # views of the same instant here rather than in a separate
        # collection. _multi_view_note explains this ordering to the model.
        images = [
            Image.open(artifact_root / keyframe["path"]).convert("RGB")
            for keyframe in keyframes
        ]
        sequence_note = _multi_view_note(request)
    if model is None:
        processor, model = load_vlm(model_dir)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt.strip()}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} for image in images
            ] + [
                {"type": "text", "text": _prompt_text(request, sequence_note)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    do_sample = config["generation"]["do_sample"]
    base_seed = config["generation"].get("seed", 0)
    # Retrying only helps under sampling: greedy decoding (do_sample=False)
    # is deterministic, so a retry would replay the exact same failure.
    # Measured need for this: real runs against both WP1 demo videos found
    # ~60-67% of greedy-decoded outputs degenerate into diacritic-free
    # Vietnamese that paraphrases the prompt's own instructions instead of
    # describing the image (see prompts_v4.yaml/prompts_v5.yaml notes) --
    # a per-case failure mode, not something a different prompt wording
    # eliminated, so retrying with a fresh sample is the mitigation.
    max_attempts = config["generation"].get("max_attempts", 3) if do_sample else 1

    started = time.perf_counter()
    contract_status = "invalid"
    contract_error = None
    assessment: dict[str, Any] | None = None
    raw_text = ""
    for attempt in range(max_attempts):
        if do_sample:
            torch.manual_seed(base_seed + attempt)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=config["vlm"]["max_new_tokens"],
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw_text = processor.decode(generated, skip_special_tokens=True)
        try:
            assessment = extract_json_object(raw_text)
            validate_vlm_assessment(assessment, request)
            validate_grounding_policy(
                assessment, request, clip_frames_shown=clip_frames_shown
            )
            contract_status = "valid"
            contract_error = None
            break
        except (ValueError, ContractError) as error:
            contract_status = "invalid"
            contract_error = str(error)
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started
    return {
        "schema_version": 1,
        "case_id": request["case_id"],
        "model_id": config["vlm"]["model_id"],
        "revision": config["vlm"]["revision"],
        "dtype": config["vlm"]["dtype"],
        "elapsed_s": elapsed_s,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "attempts_used": attempt + 1,
        "max_attempts": max_attempts,
        "evidence_mode": "clip" if clip_frames_shown else "keyframe",
        "contract_status": contract_status,
        "contract_error": contract_error,
        "raw_text": raw_text,
        "assessment": assessment,
    }
