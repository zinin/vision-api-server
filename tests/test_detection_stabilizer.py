import pytest

from detection_stabilizer import (
    RawDetection,
    StabilizerConfig,
    Track,
    compute_iou,
)
from detection_stabilizer import DetectionStabilizer


class TestComputeIoU:
    def test_identical_boxes(self):
        assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert iou == pytest.approx(2500 / 17500, abs=0.001)

    def test_one_inside_other(self):
        iou = compute_iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert iou == pytest.approx(2500 / 10000, abs=0.001)

    def test_zero_area_box(self):
        assert compute_iou((50, 50, 50, 50), (0, 0, 100, 100)) == 0.0

    def test_touching_edges(self):
        assert compute_iou((0, 0, 50, 50), (50, 0, 100, 50)) == 0.0


class TestStabilizerConfig:
    def test_defaults(self):
        config = StabilizerConfig()
        assert config.conf_factor == 0.4
        assert config.iou_threshold == 0.3
        assert config.min_vote_conf == 0.3
        assert config.grace_center_sec == 2.0
        assert config.grace_edge_sec == 0.5
        assert config.center_zone == 0.6
        assert config.max_staleness_sec == 5.0


class TestRawDetection:
    def test_bbox_tuple(self):
        det = RawDetection(
            frame_num=0, x1=10, y1=20, x2=110, y2=120,
            class_id=0, class_name="person", confidence=0.9,
        )
        assert det.bbox == (10, 20, 110, 120)


def _make_raw(frame_num, x1, y1, x2, y2, class_id=0, class_name="person", conf=0.9):
    return RawDetection(
        frame_num=frame_num, x1=x1, y1=y1, x2=x2, y2=y2,
        class_id=class_id, class_name=class_name, confidence=conf,
    )


class TestBuildTracks:
    def _config(self, **overrides):
        return StabilizerConfig(**overrides)

    def test_single_object_across_frames(self):
        """One object detected on frames 0, 5, 10 — should form one track."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [_make_raw(0, 100, 100, 200, 200)],
            5: [_make_raw(5, 105, 105, 205, 205)],
            10: [_make_raw(10, 110, 110, 210, 210)],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 3

    def test_two_separate_objects(self):
        """Two non-overlapping objects — should form two tracks."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [
                _make_raw(0, 0, 0, 50, 50),
                _make_raw(0, 500, 500, 600, 600),
            ],
            5: [
                _make_raw(5, 5, 5, 55, 55),
                _make_raw(5, 505, 505, 605, 605),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 2

    def test_low_conf_does_not_create_track(self):
        """Detection below conf_threshold should not create a new track."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [_make_raw(0, 100, 100, 200, 200, conf=0.3)],
        }
        tracks, unmatched = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 0
        assert len(unmatched[0]) == 1

    def test_low_conf_extends_existing_track(self):
        """Low-conf detection matching an existing track should extend it."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [_make_raw(0, 100, 100, 200, 200, conf=0.8)],
            5: [_make_raw(5, 105, 105, 205, 205, conf=0.3)],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 2

    def test_no_match_creates_new_track(self):
        """When IoU is too low, a high-conf detection starts a new track."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [_make_raw(0, 0, 0, 50, 50, conf=0.9)],
            5: [_make_raw(5, 500, 500, 600, 600, conf=0.9)],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 2

    def test_greedy_1to1_assignment(self):
        """Each detection matches at most one track, and each track at most one detection."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [
                _make_raw(0, 100, 100, 200, 200, conf=0.9),
                _make_raw(0, 300, 300, 400, 400, conf=0.9),
            ],
            5: [
                _make_raw(5, 105, 105, 205, 205, conf=0.9),
                _make_raw(5, 305, 305, 405, 405, conf=0.9),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 2
        for t in tracks:
            assert len(t.detections) == 2


class TestClassVoting:
    def _config(self, **overrides):
        return StabilizerConfig(**overrides)

    def test_single_class(self):
        """All detections same class — voting returns that class."""
        stabilizer = DetectionStabilizer(self._config())
        track = Track(track_id=0, detections={
            0: _make_raw(0, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.8),
            5: _make_raw(5, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.7),
        })
        stabilizer._vote_class(track)
        assert track.stable_class_id == 1
        assert track.stable_class_name == "car"

    def test_weighted_voting(self):
        """Class with higher total weighted score wins."""
        stabilizer = DetectionStabilizer(self._config(min_vote_conf=0.3))
        track = Track(track_id=0, detections={
            0: _make_raw(0, 100, 100, 200, 200, class_id=0, class_name="person", conf=0.4),
            5: _make_raw(5, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.8),
            10: _make_raw(10, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.7),
        })
        stabilizer._vote_class(track)
        assert track.stable_class_id == 1
        assert track.stable_class_name == "car"

    def test_low_conf_excluded_from_voting(self):
        """Detections below min_vote_conf don't participate in voting."""
        stabilizer = DetectionStabilizer(self._config(min_vote_conf=0.5))
        track = Track(track_id=0, detections={
            0: _make_raw(0, 100, 100, 200, 200, class_id=0, class_name="person", conf=0.6),
            5: _make_raw(5, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.3),
            10: _make_raw(10, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.4),
        })
        stabilizer._vote_class(track)
        assert track.stable_class_id == 0
        assert track.stable_class_name == "person"

    def test_empty_track_after_filtering(self):
        """If all detections are below min_vote_conf, use the highest confidence one."""
        stabilizer = DetectionStabilizer(self._config(min_vote_conf=0.9))
        track = Track(track_id=0, detections={
            0: _make_raw(0, 100, 100, 200, 200, class_id=0, class_name="person", conf=0.3),
            5: _make_raw(5, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.5),
        })
        stabilizer._vote_class(track)
        assert track.stable_class_id == 1
        assert track.stable_class_name == "car"
