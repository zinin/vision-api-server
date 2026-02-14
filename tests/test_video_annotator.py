import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from video_annotator import (
    VideoAnnotator,
    AnnotationParams,
)
from visualization import DetectionBox, DetectionVisualizer


# --- Helpers ---

def _make_yolo_result(boxes_data: list[tuple]):
    """Create a mock YOLO result.

    Each tuple: (x1, y1, x2, y2, class_id, confidence)
    """
    if not boxes_data:
        result = MagicMock()
        result.boxes = None
        return result

    xyxy = np.array([[b[0], b[1], b[2], b[3]] for b in boxes_data], dtype=np.float32)
    cls = np.array([b[4] for b in boxes_data], dtype=np.float32)
    conf = np.array([b[5] for b in boxes_data], dtype=np.float32)

    boxes = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
    boxes.cls.cpu.return_value.numpy.return_value = cls
    boxes.conf.cpu.return_value.numpy.return_value = conf
    boxes.__len__ = lambda self: len(cls)

    result = MagicMock()
    result.boxes = boxes
    return result


# --- Fixtures ---

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.names = {0: "person", 1: "car", 2: "dog"}
    return model


@pytest.fixture
def mock_visualizer():
    return MagicMock(spec=DetectionVisualizer)


@pytest.fixture
def annotator(mock_model, mock_visualizer):
    return VideoAnnotator(mock_model, mock_visualizer, mock_model.names)


@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def default_params():
    return AnnotationParams()


# --- _get_video_metadata ---

class TestGetVideoMetadata:
    def _ffprobe_result(self, stream_data: dict, returncode: int = 0) -> MagicMock:
        result = MagicMock()
        result.returncode = returncode
        result.stdout = json.dumps({"streams": [stream_data]})
        return result

    def test_ffprobe_success(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "900",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            fps, w, h, total = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert fps == 30.0
        assert w == 1920
        assert h == 1080
        assert total == 900

    def test_ffprobe_estimates_from_duration(self):
        stream = {
            "r_frame_rate": "25/1",
            "width": 1280,
            "height": 720,
            "nb_frames": "0",
            "duration": "10.0",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            fps, w, h, total = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert fps == 25.0
        assert total == 250

    def test_ffprobe_fractional_fps(self):
        stream = {
            "r_frame_rate": "30000/1001",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            fps, w, h, total = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert fps == pytest.approx(29.97, abs=0.01)

    def test_ffprobe_invalid_falls_back_to_cv2(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 0,
            "height": 0,
            "nb_frames": "0",
        }
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 500.0,
        }.get(prop, 0)

        with (
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
        ):
            fps, w, h, total = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert w == 1280
        assert h == 720
        assert fps == 25.0
        assert total == 500
        mock_cap.release.assert_called_once()

    def test_ffprobe_error_falls_back_to_cv2(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300.0,
        }.get(prop, 0)

        with (
            patch("video_annotator.subprocess.run", side_effect=FileNotFoundError),
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
        ):
            fps, w, h, total = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert w == 640
        assert h == 480
        mock_cap.release.assert_called_once()


# --- _extract_detections ---

class TestExtractDetections:
    def test_single_detection(self, annotator):
        result = _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        dets = annotator._extract_detections([result], None)
        assert len(dets) == 1
        assert dets[0].class_name == "person"
        assert dets[0].confidence == pytest.approx(0.9)

    def test_multiple_results(self, annotator):
        r1 = _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        r2 = _make_yolo_result([(50, 60, 150, 250, 1, 0.8)])
        dets = annotator._extract_detections([r1, r2], None)
        assert len(dets) == 2
        assert dets[0].class_name == "person"
        assert dets[1].class_name == "car"

    def test_class_filter(self, annotator):
        result = _make_yolo_result([
            (10, 20, 100, 200, 0, 0.9),
            (50, 60, 150, 250, 1, 0.8),
        ])
        dets = annotator._extract_detections([result], ["person"])
        assert len(dets) == 1
        assert dets[0].class_name == "person"

    def test_empty_boxes(self, annotator):
        result = _make_yolo_result([])
        dets = annotator._extract_detections([result], None)
        assert dets == []

    def test_unknown_class(self, annotator):
        result = _make_yolo_result([(10, 20, 100, 200, 99, 0.7)])
        dets = annotator._extract_detections([result], None)
        assert len(dets) == 1
        assert dets[0].class_name == "class_99"


# --- _draw_detections ---

class TestDrawDetections:
    def test_calls_visualizer(self, annotator, mock_visualizer, sample_frame, default_params):
        dets = [
            DetectionBox(x1=10, y1=20, x2=100, y2=200, class_id=0, class_name="person", confidence=0.9),
            DetectionBox(x1=50, y1=60, x2=150, y2=250, class_id=1, class_name="car", confidence=0.8),
        ]
        annotator._draw_detections(sample_frame, dets, default_params, font_scale=0.5)
        assert mock_visualizer.draw_detection.call_count == 2
        # Verify font_scale is passed
        for call in mock_visualizer.draw_detection.call_args_list:
            assert call.kwargs["font_scale"] == 0.5


# --- _merge_audio ---

class TestMergeAudio:
    def test_success(self, annotator, tmp_path):
        original = tmp_path / "original.mp4"
        video_only = tmp_path / "video_only.mp4"
        output = tmp_path / "output.mp4"
        original.touch()
        video_only.touch()

        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("video_annotator.subprocess.run", return_value=mock_result) as mock_run:
            with patch("video_annotator.shutil.copy2") as mock_copy:
                annotator._merge_audio(original, video_only, output)
        mock_run.assert_called_once()
        mock_copy.assert_not_called()

    def test_ffmpeg_fail_fallback(self, annotator, tmp_path):
        original = tmp_path / "original.mp4"
        video_only = tmp_path / "video_only.mp4"
        output = tmp_path / "output.mp4"
        original.touch()
        video_only.touch()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "some error"
        with patch("video_annotator.subprocess.run", return_value=mock_result):
            with patch("video_annotator.shutil.copy2") as mock_copy:
                annotator._merge_audio(original, video_only, output)
        mock_copy.assert_called_once_with(str(video_only), str(output))

    def test_timeout_fallback(self, annotator, tmp_path):
        original = tmp_path / "original.mp4"
        video_only = tmp_path / "video_only.mp4"
        output = tmp_path / "output.mp4"
        original.touch()
        video_only.touch()

        with patch(
            "video_annotator.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300),
        ):
            with patch("video_annotator.shutil.copy2") as mock_copy:
                annotator._merge_audio(original, video_only, output)
        mock_copy.assert_called_once()


# --- annotate() pipeline ---

class TestAnnotatePipeline:
    def _setup_pipeline_mocks(self, num_frames: int, width: int = 640, height: int = 480):
        """Set up common mocks for annotate pipeline tests."""
        frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]
        read_returns = [(True, f) for f in frames] + [(False, None)]

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = read_returns

        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        return mock_cap, mock_writer

    def test_full_pipeline(self, mock_model, mock_visualizer, tmp_path):
        num_frames = 6
        detect_every = 3
        mock_cap, mock_writer = self._setup_pipeline_mocks(num_frames)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        ffprobe_stream = {
            "r_frame_rate": "30/1",
            "width": 640,
            "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        merge_result = MagicMock()
        merge_result.returncode = 0

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names)

        def subprocess_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            return merge_result

        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.cv2.VideoWriter", return_value=mock_writer),
            patch("video_annotator.cv2.VideoWriter_fourcc", return_value=0),
            patch("video_annotator.subprocess.run", side_effect=subprocess_side_effect),
        ):
            stats = annotator.annotate(input_path, output_path, AnnotationParams(detect_every=detect_every))

        assert stats.total_frames == num_frames
        # Frames 0, 3 are detection frames
        assert stats.detected_frames == 2
        # Frames 1, 2, 4, 5 are tracked frames
        assert stats.tracked_frames == 4
        assert mock_writer.write.call_count == num_frames

    def test_detect_every_1(self, mock_model, mock_visualizer, tmp_path):
        """When detect_every=1, every frame gets YOLO detection, no hold frames."""
        num_frames = 4
        mock_cap, mock_writer = self._setup_pipeline_mocks(num_frames)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        ffprobe_stream = {
            "r_frame_rate": "30/1",
            "width": 640,
            "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        merge_result = MagicMock()
        merge_result.returncode = 0

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names)

        def subprocess_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            return merge_result

        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.cv2.VideoWriter", return_value=mock_writer),
            patch("video_annotator.cv2.VideoWriter_fourcc", return_value=0),
            patch("video_annotator.subprocess.run", side_effect=subprocess_side_effect),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=1)
            )

        assert stats.total_frames == num_frames
        assert stats.detected_frames == num_frames
        assert stats.tracked_frames == 0
        assert mock_model.predict.call_count == num_frames

    def test_hold_reuses_detections(self, mock_model, mock_visualizer, tmp_path):
        """Intermediate frames reuse last YOLO detections (hold mode)."""
        num_frames = 3
        mock_cap, mock_writer = self._setup_pipeline_mocks(num_frames)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        ffprobe_stream = {
            "r_frame_rate": "30/1",
            "width": 640,
            "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        merge_result = MagicMock()
        merge_result.returncode = 0

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names)

        def subprocess_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            return merge_result

        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.cv2.VideoWriter", return_value=mock_writer),
            patch("video_annotator.cv2.VideoWriter_fourcc", return_value=0),
            patch("video_annotator.subprocess.run", side_effect=subprocess_side_effect),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=3)
            )

        assert stats.total_frames == num_frames
        assert stats.detected_frames == 1
        assert mock_model.predict.call_count == 1
        assert stats.tracked_frames == 2
        assert stats.total_detections == 3
        assert mock_visualizer.draw_detection.call_count == 3

    def test_hold_clears_on_empty_detection(self, mock_model, mock_visualizer, tmp_path):
        """When detection frame returns no objects, hold frames also show nothing."""
        num_frames = 4
        mock_cap, mock_writer = self._setup_pipeline_mocks(num_frames)

        # Frame 0: 1 detection. Frame 3: 0 detections.
        mock_model.predict.side_effect = [
            [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])],
            [_make_yolo_result([])],
        ]

        ffprobe_stream = {
            "r_frame_rate": "30/1",
            "width": 640,
            "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        merge_result = MagicMock()
        merge_result.returncode = 0

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names)

        def subprocess_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            return merge_result

        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.cv2.VideoWriter", return_value=mock_writer),
            patch("video_annotator.cv2.VideoWriter_fourcc", return_value=0),
            patch("video_annotator.subprocess.run", side_effect=subprocess_side_effect),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=3)
            )

        assert stats.total_frames == num_frames
        assert stats.detected_frames == 2
        assert stats.tracked_frames == 2
        assert mock_model.predict.call_count == 2
        assert stats.total_detections == 3
        assert mock_visualizer.draw_detection.call_count == 3

    def test_cannot_open_video(self, mock_model, mock_visualizer, tmp_path):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        ffprobe_stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480, "nb_frames": "100",
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names)

        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            with pytest.raises(RuntimeError, match="Cannot open video"):
                annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())
