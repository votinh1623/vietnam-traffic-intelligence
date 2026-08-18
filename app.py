"""Streamlit dashboard for vn_traffic pipeline runs.

Run with:
    streamlit run app.py

This reads an existing (or currently in-progress) output/pipeline/runN/
directory produced by run_pipeline.py and shows the live annotated frame plus
a short status line; deeper analytics (timeline, event log, finished video)
sit collapsed below since the frame's own overlay already carries state,
track count, occupancy, and speed.

"Real-time" here means polling the run's own output files while
vn_traffic's pipeline is actively writing them: runner.py flushes
tracks.csv/analytics.csv/events.jsonl after every frame, rewrites
run.json's progress fields about once per second, and overwrites
latest_frame.jpg -- the actual live view -- every frame via a temp file plus
an atomic rename, so this dashboard never reads a half-written JPEG.
annotated.mp4 is not readable live: most containers only finalize their
index when the writer closes, so it only becomes playable once the run
completes; it is shown collapsed, purely for after-the-fact review. This
dashboard does not connect to a live camera: the project only has offline
video files processed by run_pipeline.py, no live camera source, so a
"running" run here means an offline video currently being processed, not a
live feed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_OUTPUT_ROOT = PROJECT_ROOT / "output" / "pipeline"

STATE_COLORS = {
    "NORMAL": "#12B76A",
    "DENSE": "#F79009",
    "CONGESTED": "#F04438",
}
STATUS_DOTS = {"running": "🟡", "completed": "🟢", "failed": "🔴"}


# ---- data access ----------------------------------------------------------


def list_run_dirs() -> list[Path]:
    if not PIPELINE_OUTPUT_ROOT.is_dir():
        return []
    runs = [
        path
        for path in PIPELINE_OUTPUT_ROOT.iterdir()
        if path.is_dir() and (path / "run.json").is_file()
    ]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_analytics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        # A row can be mid-flush while the pipeline is still running.
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame()


def load_events(path: Path, limit: int = 25) -> list[dict]:
    if not path.is_file():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(events))[:limit]


def event_detail(event: dict) -> str:
    event_type = event.get("event_type")
    if event_type == "congestion_transition":
        return f"{event.get('previous_state')} → {event.get('current_state')}"
    if event_type == "line_crossing":
        return f"{event.get('class_name', '')} {event.get('direction', '')}".strip()
    if event_type == "prolonged_stop":
        duration = event.get("measurements", {}).get("stopped_duration_s")
        duration_text = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "?"
        return f"{event.get('class_name', '')} dừng {duration_text}"
    return event.get("class_name", "")


# ---- page -------------------------------------------------------------

st.set_page_config(page_title="VN Traffic Intelligence", page_icon="🚦", layout="wide")
st.markdown(
    "<style>div.block-container{padding-top:2rem} "
    "[data-testid='stMetricValue']{font-size:1.1rem}</style>",
    unsafe_allow_html=True,
)

runs = list_run_dirs()
if not runs:
    st.warning(
        f"Không tìm thấy run nào trong `{PIPELINE_OUTPUT_ROOT}`. "
        "Hãy chạy `run_pipeline.py` trước."
    )
    st.stop()

with st.sidebar:
    st.header("🚦 VN Traffic")
    selected_label = st.selectbox("Run", [path.name for path in runs], index=0)
    refresh_seconds = st.slider("Chu kỳ làm mới (giây)", 1, 10, 2)
    if st.button("🔄 Làm mới ngay"):
        st.rerun()
    st.caption(
        "Không phải camera trực tiếp — đọc output của run_pipeline.py. "
        "Tự làm mới chỉ khi run đang 'running'."
    )

run_dir = PIPELINE_OUTPUT_ROOT / selected_label
metadata_path = run_dir / "run.json"
summary_path = run_dir / "summary.json"
analytics_path = run_dir / "analytics.csv"
events_path = run_dir / "events.jsonl"
video_path = run_dir / "annotated.mp4"
latest_frame_path = run_dir / "latest_frame.jpg"

status = (load_json(metadata_path) or {}).get("status", "unknown")


# ---- live view: frame + one status line -----------------------------------


@st.fragment(run_every=f"{refresh_seconds}s" if status == "running" else None)
def render_live_view() -> None:
    metadata = load_json(metadata_path) or {}
    df = load_analytics(analytics_path)
    current_status = metadata.get("status", "unknown")

    if latest_frame_path.is_file():
        # Đọc lại bytes mỗi lần fragment chạy, không cache theo đường dẫn,
        # nên luôn là frame vừa được runner.py ghi ra.
        st.image(str(latest_frame_path), width="stretch")
    else:
        st.info("Chưa có frame nào được ghi ra (run vừa mới bắt đầu).")

    state = "—"
    if not df.empty:
        last = df.iloc[-1]
        state = str(last.get("congestion_state", "NORMAL"))
    color = STATE_COLORS.get(state, "#667085")
    frames_processed = metadata.get("frames_processed", 0)
    fps = metadata.get("processing_fps")
    fps_text = f"{fps:.1f} FPS" if fps else "—"

    st.markdown(
        f"<div style='display:flex;gap:10px;align-items:center;margin-top:6px'>"
        f"<span style='background:{color};color:white;padding:4px 14px;"
        f"border-radius:999px;font-weight:700'>{state}</span>"
        f"<span>{STATUS_DOTS.get(current_status, '⚪')} {current_status}</span>"
        f"<span style='color:#667085'>· {selected_label} · frame {frames_processed} · {fps_text}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


render_live_view()

# ---- details, collapsed by default ----------------------------------------

with st.expander("Chi tiết phân tích"):
    metadata = load_json(metadata_path) or {}
    df = load_analytics(analytics_path)
    events = load_events(events_path)

    info_cols = st.columns(3)
    info_cols[0].caption(f"Model: `{Path(metadata.get('model', '?')).name}`")
    info_cols[1].caption(
        f"Analytics mode: `{(metadata.get('analytics') or {}).get('analytics_mode', '?')}`"
    )
    info_cols[2].caption(f"Nguồn: `{Path(metadata.get('source', '?')).name}`")

    if not df.empty:
        chart_cols = st.columns(2)
        chart_cols[0].caption("BBox union occupancy theo thời gian")
        chart_cols[0].line_chart(
            df.set_index("timestamp_s")[["bbox_union_occupancy"]], height=200
        )
        chart_cols[1].caption("Số track trong ROI theo thời gian")
        chart_cols[1].line_chart(
            df.set_index("timestamp_s")[["roi_track_count"]], height=200
        )

    st.caption(f"Sự kiện gần nhất ({len(events)})")
    if events:
        rows = [
            {
                "time_s": round(event.get("timestamp_s", 0.0), 2),
                "type": event.get("event_type"),
                "detail": event_detail(event),
            }
            for event in events
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if video_path.is_file():
        st.caption("Video đã annotate (chỉ xem lại sau khi run xong)")
        st.video(str(video_path))

    summary = load_json(summary_path)
    if summary:
        st.json(summary)
