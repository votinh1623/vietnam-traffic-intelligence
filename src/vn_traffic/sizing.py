"""Resolution-adaptive sizing helpers shared by the perception/detect code
paths -- a single video-resolution-aware mechanism instead of one fixed
constant tuned to whichever resolution was tested last."""

from __future__ import annotations


def adaptive_line_width(height: int, width: int) -> int:
    """Pick a box/label line width that stays legible across resolutions.

    Ultralytics' Annotator ties label text size directly to line_width
    (font scale = line_width/3, no independent font-size control) -- a
    fixed value tuned to look right on ~720-1280px video renders as
    illegibly thin/small on much higher-resolution input (e.g. 4K), since
    box/vehicle pixel size grows with resolution too. Scale it off the
    frame's short side so it stays proportionally consistent across
    resolutions; 720p is the regime it was originally tuned on.
    """
    return max(1, round(min(height, width) / 720))


def adaptive_imgsz(height: int, width: int) -> int:
    """Pick an inference size that doesn't shrink small aerial vehicles.

    A fixed imgsz=1280 letterboxes a 3840x2160 source down by ~3x before
    the detector ever sees it, shrinking already-small overhead vehicles
    below what the model can reliably detect (measured: 14 boxes at
    imgsz=1280 vs 53 at imgsz=2560 on the same 4K frame, for ~0.4GB extra
    VRAM -- well inside a 6GB budget). Scale to the source's long side,
    clamped to [1280, 2560]: unchanged for the ~720-1280px videos this was
    originally tuned on, larger only for higher-resolution sources that
    actually need it.
    """
    long_side = max(height, width)
    clamped = max(1280, min(2560, long_side))
    # Round up to a multiple of 32 (Ultralytics' stride requirement) so the
    # logged value matches what actually runs instead of being silently
    # adjusted downstream.
    return -(-clamped // 32) * 32
