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


class DetectionStabilizer:
    """Post-processor that stabilizes YOLO detections across video frames."""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self._next_track_id = 0

    def _new_track_id(self) -> int:
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    def _build_tracks(
        self,
        raw_detections: dict[int, list[RawDetection]],
        conf_threshold: float,
        detect_every: int,
        fps: float,
    ) -> tuple[list[Track], dict[int, list[RawDetection]]]:
        """Build tracks from raw detections using greedy IoU matching.

        Returns:
            (tracks, unmatched_weak): tracks list and dict of frame_num → unmatched
            weak detections for backward extension.
        """
        tracks: list[Track] = []
        unmatched_weak: dict[int, list[RawDetection]] = {}
        max_staleness_frames = int(self.config.max_staleness_sec * fps) if fps > 0 else 150

        for frame_num in sorted(raw_detections.keys()):
            detections = raw_detections[frame_num]
            if not detections:
                continue

            # Only match against active tracks (last detection within staleness window)
            active_tracks = [
                t for t in tracks
                if max(t.detections.keys()) >= frame_num - max_staleness_frames
            ]

            # Compute IoU for all (track, detection) pairs
            pairs: list[tuple[float, int, int]] = []
            for ti, track in enumerate(active_tracks):
                latest_frame = max(t for t in track.detections.keys() if t < frame_num) \
                    if any(t < frame_num for t in track.detections.keys()) else None
                if latest_frame is None:
                    continue
                track_bbox = track.detections[latest_frame].bbox
                for di, det in enumerate(detections):
                    iou = compute_iou(track_bbox, det.bbox)
                    if iou >= self.config.iou_threshold:
                        pairs.append((iou, ti, di))

            # Greedy 1:1 assignment (highest IoU first)
            pairs.sort(key=lambda x: x[0], reverse=True)
            assigned_tracks: set[int] = set()
            assigned_dets: set[int] = set()

            for iou_val, ti, di in pairs:
                if ti in assigned_tracks or di in assigned_dets:
                    continue
                active_tracks[ti].detections[frame_num] = detections[di]
                assigned_tracks.add(ti)
                assigned_dets.add(di)

            # Handle unassigned detections
            for di, det in enumerate(detections):
                if di in assigned_dets:
                    continue
                if det.confidence >= conf_threshold:
                    track = Track(
                        track_id=self._new_track_id(),
                        detections={frame_num: det},
                    )
                    tracks.append(track)
                else:
                    unmatched_weak.setdefault(frame_num, []).append(det)

        return tracks, unmatched_weak

    def _vote_class(self, track: Track) -> None:
        """Determine stable class for a track via weighted voting."""
        scores: dict[int, float] = {}
        names: dict[int, str] = {}

        for det in track.detections.values():
            if det.confidence >= self.config.min_vote_conf:
                scores[det.class_id] = scores.get(det.class_id, 0.0) + det.confidence
                names[det.class_id] = det.class_name

        if not scores:
            # Fallback: use highest-confidence detection
            best = max(track.detections.values(), key=lambda d: d.confidence)
            track.stable_class_id = best.class_id
            track.stable_class_name = best.class_name
            return

        winner_id = max(scores, key=lambda k: scores[k])
        track.stable_class_id = winner_id
        track.stable_class_name = names[winner_id]
