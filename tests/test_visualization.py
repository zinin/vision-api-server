from unittest.mock import MagicMock

import cv2
import pytest
import numpy as np

from visualization import (
    Color,
    DetectionBox,
    DetectionVisualizer,
    bgr_patch_to_yuv420,
    bgr_to_yuv601,
    encode_image_to_bytes,
    yuv420_planes,
)


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


class TestYuvHelpers:
    @pytest.mark.parametrize("bgr", [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128), (255, 255, 255)])
    def test_bgr_to_yuv601_matches_opencv_i420(self, bgr):
        """OpenCV's I420 conversion is limited-range BT.601 too (1 level of rounding slack)."""
        image = np.full((2, 2, 3), bgr, np.uint8)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        expected = (int(yuv[0]), int(yuv[4]), int(yuv[5]))
        assert all(abs(a - b) <= 1 for a, b in zip(bgr_to_yuv601(bgr), expected))

    def test_patch_conversion_subsamples_chroma(self):
        patch = np.zeros((4, 6, 3), np.uint8)
        patch[:, :] = (255, 0, 0)
        y, u, v = bgr_patch_to_yuv420(patch)
        assert y.shape == (4, 6) and u.shape == (2, 3) and v.shape == (2, 3)
        assert (int(y[0, 0]), int(u[0, 0]), int(v[0, 0])) == bgr_to_yuv601((255, 0, 0))

    def test_planes_are_views_into_the_frame(self):
        frame = np.zeros(5 * 3 + 2 * 3 * 2, np.uint8)  # 5x3 frame, chroma 3x2
        y, u, v = yuv420_planes(frame, 5, 3)
        assert (y.shape, u.shape, v.shape) == ((3, 5), (2, 3), (2, 3))
        v[:] = 7
        assert frame[-6:].tolist() == [7] * 6


def _yuv_frame(width, height, seed=0):
    """A random yuv420p frame and its BGR twin (OpenCV's BT.601 conversion)."""
    rng = np.random.default_rng(seed)
    bgr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    bgr = cv2.GaussianBlur(bgr, (7, 7), 0)  # smooth, like video
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    return yuv, cv2.cvtColor(yuv.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)


def _touched_mask(visualizer, width, height, det, **kwargs):
    """Pixels the BGR renderer paints for ``det``: draw on a sentinel canvas."""
    canvas = np.full((height, width, 3), (1, 2, 3), np.uint8)
    visualizer.draw_detection(canvas, det, **kwargs)
    return (canvas != (1, 2, 3)).any(axis=2)


class TestDrawDetectionYuv420:
    W, H = 320, 240
    KW = dict(line_width=2, show_labels=True, show_conf=True, font_scale=0.5, text_thickness=1)

    def _draw(self, frame, det, width=None, height=None, **overrides):
        kwargs = {**self.KW, **overrides}
        visualizer = DetectionVisualizer(class_names={0: "person"})
        visualizer.draw_detection_yuv420(frame, width or self.W, height or self.H, det, **kwargs)

    def test_pixels_away_from_the_drawing_are_untouched(self):
        yuv, _ = _yuv_frame(self.W, self.H)
        before = yuv.copy()
        det = DetectionBox(x1=41, y1=57, x2=201, y2=181, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det)

        mask = _touched_mask(DetectionVisualizer({0: "person"}), self.W, self.H, det, **self.KW)
        near = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        y_before, u_before, v_before = yuv420_planes(before, self.W, self.H)
        y_after, u_after, v_after = yuv420_planes(yuv, self.W, self.H)
        np.testing.assert_array_equal(y_after[~near], y_before[~near])
        chroma_near = near.reshape(self.H // 2, 2, self.W // 2, 2).any(axis=(1, 3))
        np.testing.assert_array_equal(u_after[~chroma_near], u_before[~chroma_near])
        np.testing.assert_array_equal(v_after[~chroma_near], v_before[~chroma_near])

    def test_luma_matches_the_bgr_renderer(self):
        """Y is exact: every pixel the BGR renderer paints gets the luma of its colour."""
        yuv, bgr = _yuv_frame(self.W, self.H, seed=1)
        det = DetectionBox(x1=41, y1=57, x2=201, y2=181, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det)
        DetectionVisualizer({0: "person"}).draw_detection(bgr, det, **self.KW)

        reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        mask = _touched_mask(DetectionVisualizer({0: "person"}), self.W, self.H, det, **self.KW)
        ours, _, _ = yuv420_planes(yuv, self.W, self.H)
        ref, _, _ = yuv420_planes(reference, self.W, self.H)
        assert np.abs(ours.astype(int) - ref.astype(int))[mask].max() <= 1

    def test_even_aligned_outline_chroma_matches_the_bgr_renderer(self):
        """On even coordinates the half-resolution outline lands on the same
        chroma blocks OpenCV's I420 conversion of the BGR drawing colours."""
        kwargs = {**self.KW, "show_labels": False, "show_conf": False}
        yuv, bgr = _yuv_frame(self.W, self.H, seed=2)
        det = DetectionBox(x1=40, y1=60, x2=200, y2=180, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det, show_labels=False, show_conf=False)
        DetectionVisualizer({0: "person"}).draw_detection(bgr, det, **kwargs)

        reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
        mask = _touched_mask(DetectionVisualizer({0: "person"}), self.W, self.H, det, **kwargs)
        chroma_mask = mask.reshape(self.H // 2, 2, self.W // 2, 2).any(axis=(1, 3))
        _, u_ours, v_ours = yuv420_planes(yuv, self.W, self.H)
        _, u_ref, v_ref = yuv420_planes(reference, self.W, self.H)
        assert np.abs(u_ours.astype(int) - u_ref.astype(int))[chroma_mask].max() <= 1
        assert np.abs(v_ours.astype(int) - v_ref.astype(int))[chroma_mask].max() <= 1

    def test_label_text_is_white_on_the_class_colour(self):
        yuv = np.zeros(self.W * self.H * 3 // 2, np.uint8)
        det = DetectionBox(x1=40, y1=60, x2=200, y2=180, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det)
        y, _, _ = yuv420_planes(yuv, self.W, self.H)
        label_rows = y[40:60, 40:120]
        assert (label_rows > 200).any()  # white text
        assert (label_rows == bgr_to_yuv601((255, 0, 0))[0]).any()  # class-0 background

    def test_label_is_clipped_at_the_right_edge(self):
        yuv = np.zeros(self.W * self.H * 3 // 2, np.uint8)
        det = DetectionBox(x1=300, y1=60, x2=318, y2=100, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det)
        y, _, _ = yuv420_planes(yuv, self.W, self.H)
        assert (y[40:60, self.W - 1] != 0).any()

    def test_no_label_when_labels_and_confidence_are_hidden(self):
        yuv = np.zeros(self.W * self.H * 3 // 2, np.uint8)
        det = DetectionBox(x1=40, y1=60, x2=200, y2=180, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det, show_labels=False, show_conf=False)
        y, _, _ = yuv420_planes(yuv, self.W, self.H)
        assert (y[20:50, 40:120] == 0).all()  # nothing above the box
        assert (y[60, 40:200] != 0).all()  # the top edge is drawn

    def test_line_width_1(self):
        yuv = np.zeros(self.W * self.H * 3 // 2, np.uint8)
        det = DetectionBox(x1=40, y1=60, x2=200, y2=180, class_id=0, class_name="person", confidence=0.87)
        self._draw(yuv, det, line_width=1, show_labels=False, show_conf=False)
        y, u, _ = yuv420_planes(yuv, self.W, self.H)
        assert (y[120, 40] != 0) and (y[120, 42] == 0)
        assert u[60, 20] == bgr_to_yuv601((255, 0, 0))[1]

    def test_odd_frame_size(self):
        width, height = 321, 241
        yuv = np.zeros(width * height + 2 * 161 * 121, np.uint8)
        det = DetectionBox(x1=250, y1=200, x2=320, y2=240, class_id=0, class_name="person", confidence=0.5)
        self._draw(yuv, det, width=width, height=height)
        y, _, _ = yuv420_planes(yuv, width, height)
        assert (y[240, 250:320] != 0).all()
