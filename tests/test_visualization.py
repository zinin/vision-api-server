from unittest.mock import MagicMock

import pytest
import numpy as np

from visualization import Color, DetectionBox, DetectionVisualizer, encode_image_to_bytes


class TestColor:
    def test_as_tuple(self):
        c = Color(255, 128, 0)
        assert c.as_tuple() == (255, 128, 0)

    def test_is_frozen(self):
        c = Color(255, 128, 0)
        with pytest.raises(AttributeError):
            c.b = 10


class TestDetectionBox:
    def test_fields(self):
        box = DetectionBox(x1=10, y1=20, x2=100, y2=200, class_id=0, class_name="person", confidence=0.95)
        assert box.x1 == 10
        assert box.y1 == 20
        assert box.x2 == 100
        assert box.y2 == 200
        assert box.class_id == 0
        assert box.class_name == "person"
        assert box.confidence == 0.95

    def test_is_frozen(self):
        box = DetectionBox(x1=0, y1=0, x2=1, y2=1, class_id=0, class_name="a", confidence=0.5)
        with pytest.raises(AttributeError):
            box.x1 = 99


class TestAdaptiveFontScale:
    @pytest.fixture
    def visualizer(self):
        return DetectionVisualizer(class_names={0: "person"})

    def test_reference_height(self, visualizer):
        assert visualizer.calculate_adaptive_font_scale(720) == pytest.approx(0.5)

    def test_min_clamp(self, visualizer):
        # height=100 → 0.5 * 100/720 ≈ 0.069 → clamped to 0.3
        assert visualizer.calculate_adaptive_font_scale(100) == pytest.approx(0.3)

    def test_max_clamp(self, visualizer):
        # height=3000 → 0.5 * 3000/720 ≈ 2.08 → clamped to 1.5
        assert visualizer.calculate_adaptive_font_scale(3000) == pytest.approx(1.5)


class TestDrawDetection:
    def test_modifies_image(self):
        visualizer = DetectionVisualizer(class_names={0: "person"})
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = image.sum()
        det = DetectionBox(x1=10, y1=10, x2=100, y2=100, class_id=0, class_name="person", confidence=0.9)
        visualizer.draw_detection(image, det, line_width=2, show_labels=True, show_conf=True, font_scale=0.5, text_thickness=1)
        assert image.sum() > original_sum


class TestDrawYoloResults:
    def test_draws_detections_on_copy(self):
        visualizer = DetectionVisualizer(class_names={0: "person", 1: "car"})
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        xyxy = np.array([[10, 20, 100, 200], [50, 60, 150, 250]], dtype=np.float32)
        cls = np.array([0, 1], dtype=np.float32)
        conf = np.array([0.9, 0.8], dtype=np.float32)

        boxes = MagicMock()
        boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
        boxes.cls.cpu.return_value.numpy.return_value = cls
        boxes.conf.cpu.return_value.numpy.return_value = conf
        boxes.__len__ = lambda self: 2

        result = MagicMock()
        result.boxes = boxes

        annotated = visualizer.draw_yolo_results(image, [result])

        # Original image is not modified
        assert image.sum() == 0
        # Annotated copy has drawings
        assert annotated.sum() > 0

    def test_empty_results(self):
        visualizer = DetectionVisualizer(class_names={0: "person"})
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = MagicMock()
        result.boxes = None

        annotated = visualizer.draw_yolo_results(image, [result])
        # Returns a copy even with no detections
        assert annotated is not image
        assert annotated.sum() == 0


class TestEncodeImageToBytes:
    def test_encode_jpeg(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        data = encode_image_to_bytes(image, ".jpg", 90)
        assert isinstance(data, bytes)
        assert len(data) > 0
        # JPEG magic bytes
        assert data[:2] == b"\xff\xd8"
