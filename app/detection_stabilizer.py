from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field

from visualization import DetectionBox

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StabilizerConfig:
    """Stabilizer configuration (from env variables via Settings)."""
    conf_factor: float = 0.4
    iou_threshold: float = 0.3
    min_vote_conf: float = 0.3
    grace_center_sec: float = 2.0
    grace_edge_sec: float = 0.5
    center_zone: float = 0.6
    max_staleness_sec: float = 5.0


@dataclass(slots=True)
class RawDetection:
    """Single detection from one frame."""
    frame_num: int
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Track:
    """A tracked object across multiple frames."""
    track_id: int
    detections: dict[int, RawDetection] = field(default_factory=dict)
    stable_class_id: int = 0
    stable_class_name: str = ""
    first_frame: int = 0
    last_frame: int = 0


@dataclass(slots=True)
class StabilizedFrame:
    """Ready-to-render detections for one frame."""
    detections: list[DetectionBox]


def compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection over Union of two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union
