import pytest

from detection_stabilizer import (
    RawDetection,
    StabilizerConfig,
    compute_iou,
)


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
