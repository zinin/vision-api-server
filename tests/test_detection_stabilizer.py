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


class TestDuplicateTrackSuppression:
    """Overlapping YOLO detections on the same frame should not create duplicate tracks."""

    def _config(self, **overrides):
        return StabilizerConfig(**overrides)

    def test_overlapping_detections_same_frame_create_one_track(self):
        """Two overlapping high-conf detections on the same frame → one track, not two.

        Bug: YOLO sometimes returns two slightly offset boxes for the same object.
        Both have conf >= threshold, so both create new tracks. The result is
        duplicate bounding boxes rendered on the same object.
        """
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        raw = {
            0: [
                _make_raw(0, 100, 100, 300, 300, class_id=0, class_name="dog", conf=0.82),
                _make_raw(0, 120, 110, 320, 310, class_id=0, class_name="dog", conf=0.77),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 1, f"Expected 1 track, got {len(tracks)} — duplicate track for same object"

    def test_overlapping_detections_highest_conf_wins(self):
        """When suppressing duplicates, the highest-confidence detection creates the track."""
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        raw = {
            0: [
                _make_raw(0, 100, 100, 300, 300, class_id=0, class_name="dog", conf=0.77),
                _make_raw(0, 120, 110, 320, 310, class_id=0, class_name="dog", conf=0.82),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 1
        # The track should use the higher-confidence detection
        assert tracks[0].detections[0].confidence == 0.82

    def test_non_overlapping_detections_same_frame_create_separate_tracks(self):
        """Non-overlapping detections on the same frame should still create separate tracks."""
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        raw = {
            0: [
                _make_raw(0, 0, 0, 100, 100, class_id=0, class_name="dog", conf=0.82),
                _make_raw(0, 500, 500, 700, 700, class_id=1, class_name="bench", conf=0.75),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 2

    def test_three_overlapping_detections_suppressed_to_one(self):
        """Three overlapping detections of the same object → one track."""
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        raw = {
            0: [
                _make_raw(0, 100, 100, 300, 300, conf=0.82),
                _make_raw(0, 110, 105, 310, 305, conf=0.77),
                _make_raw(0, 120, 110, 320, 310, conf=0.70),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 1

    def test_suppressed_detections_go_to_unmatched_weak(self):
        """Suppressed duplicate detections should be stored in unmatched_weak."""
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        raw = {
            0: [
                _make_raw(0, 100, 100, 300, 300, conf=0.82),
                _make_raw(0, 120, 110, 320, 310, conf=0.77),
            ],
        }
        tracks, unmatched = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 1
        assert len(unmatched.get(0, [])) == 1

    def test_duplicate_when_one_matches_existing_track(self):
        """YOLO returns two near-identical detections: one matches an existing track,
        the other must NOT create a new track.

        Bug: greedy matching assigns det_A to Track 1. det_B is unassigned.
        Input NMS only checks new candidates against each other, not against
        assigned detections, so det_B creates a spurious Track 2.
        """
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        raw = {
            0: [_make_raw(0, 300, 300, 500, 500, conf=0.9)],
            5: [
                _make_raw(5, 305, 305, 505, 505, conf=0.85),  # matches Track 1
                _make_raw(5, 310, 310, 510, 510, conf=0.80),  # YOLO duplicate
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 1, f"Expected 1 track, got {len(tracks)} — duplicate from assigned overlap"

    def test_output_nms_direct(self):
        """Direct test of _nms_boxes: overlapping boxes suppressed, non-overlapping kept."""
        from visualization import DetectionBox
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        boxes = [
            DetectionBox(x1=100, y1=100, x2=300, y2=300, class_id=0, class_name="dog", confidence=0.9),
            DetectionBox(x1=120, y1=110, x2=320, y2=310, class_id=0, class_name="dog", confidence=0.7),
            DetectionBox(x1=600, y1=600, x2=800, y2=800, class_id=1, class_name="cat", confidence=0.8),
        ]
        result = stabilizer._nms_boxes(boxes)
        assert len(result) == 2
        assert result[0].confidence == 0.9  # dog (higher conf kept)
        assert result[1].confidence == 0.8  # cat (no overlap, kept)

    def test_small_box_inside_large_suppressed_input(self):
        """Small box fully inside large box (head vs full body) — must not create 2 tracks.

        Root cause: YOLO detects full body (large box) and head/torso (small box).
        IoU is low because small_area/large_area < 0.3. But containment is ~1.0.
        NMS must use containment ratio (intersection / min_area), not just IoU.
        """
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        raw = {
            0: [
                # Full body — large box
                _make_raw(0, 100, 100, 500, 500, class_id=0, class_name="dog", conf=0.84),
                # Head only — small box fully inside the large one
                _make_raw(0, 200, 150, 350, 300, class_id=0, class_name="dog", conf=0.66),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5, fps=30.0)
        assert len(tracks) == 1, f"Expected 1 track, got {len(tracks)} — small box inside large not suppressed"

    def test_small_box_inside_large_suppressed_output(self):
        """Output NMS must also catch containment (small box inside large)."""
        from visualization import DetectionBox
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        boxes = [
            # Full body
            DetectionBox(x1=100, y1=100, x2=500, y2=500, class_id=0, class_name="dog", confidence=0.84),
            # Head — inside the full body box. IoU ≈ 0.14, but containment = 1.0
            DetectionBox(x1=200, y1=150, x2=350, y2=300, class_id=0, class_name="dog", confidence=0.66),
        ]
        result = stabilizer._nms_boxes(boxes)
        assert len(result) == 1
        assert result[0].confidence == 0.84

    def test_side_by_side_boxes_not_suppressed(self):
        """Two objects side by side with partial overlap — must NOT be suppressed."""
        from visualization import DetectionBox
        stabilizer = DetectionStabilizer(self._config(iou_threshold=0.3))
        boxes = [
            DetectionBox(x1=100, y1=100, x2=300, y2=300, class_id=0, class_name="dog", confidence=0.9),
            # Partially overlapping but mostly separate — 50px overlap on x
            DetectionBox(x1=250, y1=100, x2=450, y2=300, class_id=1, class_name="cat", confidence=0.8),
        ]
        result = stabilizer._nms_boxes(boxes)
        assert len(result) == 2  # both kept — legitimate separate objects


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


class TestGapFilling:
    def _config(self, **overrides):
        defaults = dict(grace_center_sec=2.0, grace_edge_sec=0.5, center_zone=0.6)
        defaults.update(overrides)
        return StabilizerConfig(**defaults)

    def test_forward_grace_period_center(self):
        config = self._config(grace_center_sec=1.0, grace_edge_sec=0.5)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            50: _make_raw(50, 400, 400, 600, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=200, detect_every=5, unmatched_weak={})
        assert track.last_frame == 60
        # No backward grace — first_frame stays at first detection
        assert track.first_frame == 50

    def test_forward_grace_period_edge(self):
        config = self._config(grace_center_sec=2.0, grace_edge_sec=0.5)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            50: _make_raw(50, 900, 400, 1000, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=200, detect_every=5, unmatched_weak={})
        assert track.last_frame == 55
        # No backward grace — first_frame stays at first detection
        assert track.first_frame == 50

    def test_grace_period_capped_at_total_frames(self):
        config = self._config(grace_center_sec=10.0)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            95: _make_raw(95, 400, 400, 600, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=100, detect_every=5, unmatched_weak={})
        assert track.last_frame == 99

    def test_backward_grace_not_applied(self):
        """Even with large grace settings, backward grace is not applied."""
        config = self._config(grace_center_sec=10.0)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            3: _make_raw(3, 400, 400, 600, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=200, detect_every=5, unmatched_weak={})
        # No backward grace — first_frame stays at first detection
        assert track.first_frame == 3

    def test_backward_extension_with_unmatched_weak(self):
        config = self._config(grace_center_sec=0.0, grace_edge_sec=0.0)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            15: _make_raw(15, 100, 100, 200, 200, conf=0.9),
        })
        unmatched = {
            5: [_make_raw(5, 95, 95, 195, 195, conf=0.3)],
            10: [_make_raw(10, 98, 98, 198, 198, conf=0.25)],
        }
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=100, detect_every=5, unmatched_weak=unmatched)
        assert 5 in track.detections
        assert 10 in track.detections
        assert track.first_frame == 5
        assert track.last_frame == 15

    def test_backward_extension_stops_on_no_match(self):
        config = self._config(grace_center_sec=0.0, grace_edge_sec=0.0)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            15: _make_raw(15, 100, 100, 200, 200, conf=0.9),
        })
        unmatched = {
            5: [_make_raw(5, 800, 800, 900, 900, conf=0.3)],
            10: [_make_raw(10, 98, 98, 198, 198, conf=0.25)],
        }
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=100, detect_every=5, unmatched_weak=unmatched)
        assert 10 in track.detections
        assert 5 not in track.detections

    def test_no_backward_grace_period(self):
        """Backward grace period should NOT be applied — only forward grace.

        Bug: for moving objects, backward grace holds the first detection's bbox
        for frames before the first detection, showing the box in a position the
        object hasn't reached yet.
        """
        config = self._config(grace_center_sec=2.0, grace_edge_sec=0.5)
        stabilizer = DetectionStabilizer(config)
        # Object in center of frame, first detected at frame 50
        track = Track(track_id=0, detections={
            50: _make_raw(50, 400, 400, 600, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=200, detect_every=5, unmatched_weak={})
        # Forward grace should still work
        assert track.last_frame == 70  # 50 + 2.0*10
        # Backward grace should NOT extend before first detection
        assert track.first_frame == 50

    def test_interpolation_between_detections(self):
        config = self._config(grace_center_sec=0.0, grace_edge_sec=0.0, iou_threshold=0.0)
        stabilizer = DetectionStabilizer(config)
        raw = {
            0: [_make_raw(0, 100, 100, 200, 200, conf=0.9)],
            10: [_make_raw(10, 200, 200, 300, 300, conf=0.9)],
        }
        result = stabilizer.stabilize(
            raw, frame_width=1000, frame_height=1000,
            fps=10.0, total_frames=11, detect_every=10, conf_threshold=0.5,
        )
        mid = result[5].detections[0]
        assert mid.x1 == 150
        assert mid.y1 == 150
        assert mid.x2 == 250
        assert mid.y2 == 250
