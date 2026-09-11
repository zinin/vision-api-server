"""Motion-based frame selection: the ``blob`` metric and the selection rule.

Pure logic that depends on numpy and OpenCV only and never calls ffmpeg.
``video_utils`` decodes the video and feeds per-frame timestamps and metric
values into ``select_frames``.
"""
from dataclasses import dataclass
from statistics import median
from typing import Literal, Sequence

import cv2
import numpy as np

SCAN_WIDTH = 640          # width of the gray frames the metric is computed on
DIFF_THRESHOLD = 20       # brightness levels; |cur - prev| above this counts as changed
STORM_MEDIAN_BLOB = 0.02  # median blob over a segment above this = storm (rain, snow in IR)
PTS_TOLERANCE = 1e-3      # seconds; camera pts jitter tolerance in time comparisons

_KERNEL_OPEN = np.ones((3, 3), np.uint8)
_KERNEL_DILATE = np.ones((7, 7), np.uint8)

Reason = Literal["first", "grid", "motion"]


@dataclass(frozen=True)
class SelectionParams:
    """Frame selection parameters, mirrored by the query parameters of the video endpoints."""
    max_gap: float = 4.0            # grid step, seconds
    motion_threshold: float = 0.001  # blob above this is a motion candidate (fraction of the frame)
    min_interval: float = 1.0       # minimum distance between any two selected frames, seconds
    max_frames: int = 6             # cap on the number of selected frames


@dataclass(frozen=True)
class SelectedFrame:
    index: int      # frame index in the source video (0-based, ffmpeg's ``n``)
    reason: Reason


def prepare_frame(gray: np.ndarray) -> np.ndarray:
    """Blur a gray frame 3×3 once. The caller keeps the result: it is the input for two pairs."""
    return cv2.blur(gray, (3, 3))


def blob_area(prev: np.ndarray, cur: np.ndarray) -> float:
    """Area of the largest changed region between two prepared frames, as a fraction of the frame.

    Reproduces ``mask_stats`` from the research scripts, on which the default
    thresholds were tuned: absdiff, 3×3 blur, mask above ``DIFF_THRESHOLD``,
    3×3 opening, 7×7 dilation, largest 8-connected component.
    """
    diff = cv2.blur(cv2.absdiff(prev, cur), (3, 3))
    mask = (diff > DIFF_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_OPEN)
    mask = cv2.dilate(mask, _KERNEL_DILATE)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / float(mask.size)


def _round_half_up(value: float) -> int:
    return int(value + 0.5)


def median_blob(blob: Sequence[float]) -> float:
    """Median metric of a segment, ignoring blob[0] (which is always 0). 0.0 for fewer than two frames."""
    return float(median(blob[1:])) if len(blob) >= 2 else 0.0


def select_frames(
    pts: Sequence[float],
    blob: Sequence[float],
    params: SelectionParams,
) -> list[SelectedFrame]:
    """Select frames from per-frame timestamps and motion metric values.

    ``pts[i]`` is the presentation time of frame ``i``; ``blob[i]`` is the
    metric between frames ``i - 1`` and ``i`` (``blob[0]`` is 0). The rule:
    frame 0, then a time grid every ``max(max_gap, min_interval)`` seconds,
    thinned uniformly when it exceeds ``max_frames``; unless the segment is a
    storm (median blob above ``STORM_MEDIAN_BLOB``), the remaining budget is
    filled with the strongest motion peaks at least ``min_interval`` away from
    every selected frame. Deterministic; the result is sorted by index.
    """
    n = len(pts)
    if len(blob) != n:
        raise ValueError(f"pts and blob must have the same length: {n} != {len(blob)}")
    if n == 0:
        return []

    # 1-2: first frame + grid
    step = max(params.max_gap, params.min_interval)
    base = [SelectedFrame(0, "first")]
    last = pts[0]
    for i in range(1, n):
        if pts[i] - last >= step - PTS_TOLERANCE:
            base.append(SelectedFrame(i, "grid"))
            last = pts[i]

    # 3: uniform thinning keeps the first and the last grid frame; no budget is left for peaks
    cap = params.max_frames
    if len(base) > cap:
        if cap == 1:
            return [base[0]]
        positions = [_round_half_up(k * (len(base) - 1) / (cap - 1)) for k in range(cap)]
        return [base[p] for p in positions]

    # 4: storm — nearly every frame changes, the metric is blind, keep the grid only
    if median_blob(blob) > STORM_MEDIAN_BLOB:
        return base

    # 5: motion peaks, strongest first, ties by index, NMS by min_interval
    selected = list(base)
    taken = {f.index for f in selected}
    candidates = sorted(
        (i for i in range(n) if blob[i] > params.motion_threshold and i not in taken),
        key=lambda i: (-blob[i], i),
    )
    for i in candidates:
        if len(selected) >= cap:
            break
        if all(abs(pts[i] - pts[f.index]) >= params.min_interval - PTS_TOLERANCE for f in selected):
            selected.append(SelectedFrame(i, "motion"))

    return sorted(selected, key=lambda f: f.index)
