from __future__ import annotations

import bisect
import copy
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
    last_det_frame: int = -1  # cached latest detection frame for O(1) staleness check
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
                if t.last_det_frame >= frame_num - max_staleness_frames
            ]

            # Compute IoU for all (track, detection) pairs
            pairs: list[tuple[float, int, int]] = []
            for ti, track in enumerate(active_tracks):
                if track.last_det_frame < 0 or track.last_det_frame >= frame_num:
                    continue
                track_bbox = track.detections[track.last_det_frame].bbox
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
                active_tracks[ti].last_det_frame = frame_num
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
                        last_det_frame=frame_num,
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

    def _is_in_center(self, det: RawDetection, frame_width: int, frame_height: int) -> bool:
        """Check if bbox center is within the central zone of the frame."""
        cx = (det.x1 + det.x2) / 2
        cy = (det.y1 + det.y2) / 2
        margin_x = frame_width * (1 - self.config.center_zone) / 2
        margin_y = frame_height * (1 - self.config.center_zone) / 2
        return (margin_x <= cx <= frame_width - margin_x and
                margin_y <= cy <= frame_height - margin_y)

    def _grace_frames(self, det: RawDetection, frame_width: int, frame_height: int,
                      fps: float) -> int:
        """Compute grace period in frames based on bbox position."""
        if self._is_in_center(det, frame_width, frame_height):
            return int(self.config.grace_center_sec * fps)
        return int(self.config.grace_edge_sec * fps)

    def _fill_gaps(
        self,
        track: Track,
        frame_width: int,
        frame_height: int,
        fps: float,
        total_frames: int,
        detect_every: int,
        unmatched_weak: dict[int, list[RawDetection]],
    ) -> None:
        """Bidirectional gap filling: backward extension + grace periods."""
        if not track.detections:
            return

        sorted_frames = sorted(track.detections.keys())
        first_det_frame = sorted_frames[0]
        last_det_frame = sorted_frames[-1]

        # Backward extension: step back strictly by detect_every
        earliest_det = track.detections[first_det_frame]
        current_bbox = earliest_det.bbox
        target_frame = first_det_frame - detect_every
        while target_frame >= 0:
            if target_frame not in unmatched_weak:
                break  # no detections on this frame — stop, don't skip
            best_match = None
            best_iou = 0.0
            for det in unmatched_weak[target_frame]:
                iou = compute_iou(current_bbox, det.bbox)
                if iou >= self.config.iou_threshold and iou > best_iou:
                    best_match = det
                    best_iou = iou
            if best_match is None:
                break  # no IoU match — stop
            track.detections[target_frame] = best_match
            unmatched_weak[target_frame].remove(best_match)
            current_bbox = best_match.bbox
            target_frame -= detect_every

        # Recalculate after backward extension
        sorted_frames = sorted(track.detections.keys())
        first_det_frame = sorted_frames[0]
        last_det_frame = sorted_frames[-1]

        # Grace periods
        first_det = track.detections[first_det_frame]
        last_det = track.detections[last_det_frame]

        backward_grace = self._grace_frames(first_det, frame_width, frame_height, fps)
        forward_grace = self._grace_frames(last_det, frame_width, frame_height, fps)

        track.first_frame = max(0, first_det_frame - backward_grace)
        track.last_frame = min(total_frames - 1, last_det_frame + forward_grace)

    @staticmethod
    def _interpolate_bbox(
        det_a: RawDetection,
        det_b: RawDetection,
        frame_num: int,
    ) -> tuple[int, int, int, int]:
        """Linearly interpolate bbox between two detections."""
        if det_a.frame_num == det_b.frame_num:
            return det_a.bbox

        t = (frame_num - det_a.frame_num) / (det_b.frame_num - det_a.frame_num)
        x1 = round(det_a.x1 + t * (det_b.x1 - det_a.x1))
        y1 = round(det_a.y1 + t * (det_b.y1 - det_a.y1))
        x2 = round(det_a.x2 + t * (det_b.x2 - det_a.x2))
        y2 = round(det_a.y2 + t * (det_b.y2 - det_a.y2))
        return (x1, y1, x2, y2)

    @staticmethod
    def _get_bbox_for_frame(
        track: Track, frame_num: int, sorted_frames: list[int]
    ) -> tuple[int, int, int, int]:
        """Get interpolated bbox for a frame.

        Args:
            sorted_frames: pre-sorted list of detection frame numbers for this track.
        """
        # Exact match
        if frame_num in track.detections:
            return track.detections[frame_num].bbox

        # Before first detection — hold first bbox
        if frame_num < sorted_frames[0]:
            return track.detections[sorted_frames[0]].bbox

        # After last detection — hold last bbox
        if frame_num > sorted_frames[-1]:
            return track.detections[sorted_frames[-1]].bbox

        # Between two detections — interpolate (binary search for efficiency)
        idx = bisect.bisect_right(sorted_frames, frame_num)
        prev_frame = sorted_frames[idx - 1]
        next_frame = sorted_frames[idx]

        return DetectionStabilizer._interpolate_bbox(
            track.detections[prev_frame],
            track.detections[next_frame],
            frame_num,
        )

    @staticmethod
    def _get_track_confidence(track: Track) -> float:
        """Max confidence among detections of the winning class."""
        return max(
            (d.confidence for d in track.detections.values()
             if d.class_id == track.stable_class_id),
            default=0.0,
        )

    def stabilize(
        self,
        raw_detections: dict[int, list[RawDetection]],
        frame_width: int,
        frame_height: int,
        fps: float,
        total_frames: int,
        detect_every: int,
        conf_threshold: float,
    ) -> dict[int, StabilizedFrame]:
        """Main entry point: raw detections in, stabilized frames out.

        Returns a sparse dict: only frames with at least one active track are
        included. The caller should treat missing keys as "no detections on this frame".
        """
        # Normalize fps to avoid division issues with corrupted metadata
        if fps <= 0:
            fps = 30.0

        tracks, unmatched_weak = self._build_tracks(
            raw_detections, conf_threshold, detect_every, fps
        )

        for track in tracks:
            self._vote_class(track)
            # Each track gets its own copy to prevent order-dependent results
            weak_copy = copy.deepcopy(unmatched_weak)
            self._fill_gaps(
                track, frame_width, frame_height, fps,
                total_frames, detect_every, weak_copy,
            )

        logger.info(
            f"Stabilization: {len(tracks)} tracks from "
            f"{sum(len(d) for d in raw_detections.values())} raw detections"
        )

        # Precompute sorted frame lists per track for efficient interpolation
        track_sorted_frames = {
            track.track_id: sorted(track.detections.keys()) for track in tracks
        }

        # Precompute per-track confidence (max of winning class)
        track_confs = {
            track.track_id: self._get_track_confidence(track) for track in tracks
        }

        # Generate per-frame output (sparse — only frames with detections)
        result: dict[int, StabilizedFrame] = {}
        for frame_num in range(total_frames):
            boxes: list[DetectionBox] = []
            for track in tracks:
                if track.first_frame <= frame_num <= track.last_frame:
                    bbox = self._get_bbox_for_frame(
                        track, frame_num, track_sorted_frames[track.track_id]
                    )
                    boxes.append(DetectionBox(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                        class_id=track.stable_class_id,
                        class_name=track.stable_class_name,
                        confidence=track_confs[track.track_id],
                    ))
            if boxes:
                result[frame_num] = StabilizedFrame(detections=boxes)

        return result
