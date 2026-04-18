import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detection_stabilizer import StabilizerConfig
from hw_accel import HWAccelConfig, HWAccelType
from video_annotator import (
    VideoAnnotator,
    VideoMetadata,
    AnnotationParams,
    _is_nvenc_oom_error,
    _parse_fps,
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
def hw_config():
    return HWAccelConfig(accel_type=HWAccelType.CPU)


@pytest.fixture
def annotator(mock_model, mock_visualizer, hw_config):
    return VideoAnnotator(
        mock_model, mock_visualizer, mock_model.names, hw_config,
        stabilizer_config=StabilizerConfig(),
    )


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
            "codec_name": "h264",
            "bit_rate": "8000000",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == 30.0
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.total_frames == 900
        assert meta.codec_name == "h264"
        assert meta.bit_rate == 8000000

    def test_ffprobe_estimates_from_duration(self):
        stream = {
            "r_frame_rate": "25/1",
            "width": 1280,
            "height": 720,
            "nb_frames": "0",
            "duration": "10.0",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == 25.0
        assert meta.total_frames == 250

    def test_ffprobe_fractional_fps(self):
        stream = {
            "r_frame_rate": "30000/1001",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == pytest.approx(29.97, abs=0.01)

    def test_ffprobe_invalid_metadata_raises_error(self):
        """When ffprobe returns invalid metadata (e.g. 0x0), raise RuntimeError."""
        stream = {
            "r_frame_rate": "30/1",
            "width": 0,
            "height": 0,
            "nb_frames": "0",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            with pytest.raises(RuntimeError, match="ffprobe returned invalid"):
                VideoAnnotator._get_video_metadata(Path("video.mp4"))

    def test_ffprobe_error_raises_error(self):
        """When ffprobe command fails (e.g. FileNotFoundError), raise RuntimeError."""
        with patch("video_annotator.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="ffprobe failed"):
                VideoAnnotator._get_video_metadata(Path("video.mp4"))

    def test_ffprobe_nonzero_returncode_raises_error(self):
        """When ffprobe returns non-zero exit code, raise RuntimeError."""
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        with patch("video_annotator.subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="ffprobe returned non-zero"):
                VideoAnnotator._get_video_metadata(Path("video.mp4"))

    def test_missing_codec_name_returns_none(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.codec_name is None

    def test_missing_bit_rate_returns_none(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.codec_name == "h264"
        assert meta.bit_rate is None

    def test_non_numeric_bit_rate_returns_none(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
            "bit_rate": "N/A",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.bit_rate is None

    def test_zero_bit_rate_returns_none(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
            "bit_rate": "0",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.bit_rate is None

    def test_too_small_bit_rate_returns_none(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
            "bit_rate": "50000",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.bit_rate is None

    def test_too_large_bit_rate_returns_none(self):
        stream = {
            "r_frame_rate": "30/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
            "codec_name": "h264",
            "bit_rate": "999999999999",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.bit_rate is None

    def test_empty_streams_raises_error(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"streams": []})
        with patch("video_annotator.subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="no video streams"):
                VideoAnnotator._get_video_metadata(Path("video.mp4"))

    def test_invalid_frame_rate_format_defaults_to_30(self):
        stream = {
            "r_frame_rate": "invalid",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == 30.0

    def test_hevc_high_r_frame_rate_uses_avg(self):
        """HEVC streams can report r_frame_rate as timebase (90000/1).
        Should prefer avg_frame_rate."""
        stream = {
            "r_frame_rate": "90000/1",
            "avg_frame_rate": "25740000/2052571",
            "width": 2560,
            "height": 1920,
            "nb_frames": "286",
            "codec_name": "hevc",
            "bit_rate": "315539",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == pytest.approx(12.54, abs=0.01)

    def test_avg_frame_rate_preferred_over_r_frame_rate(self):
        """When both are reasonable, avg_frame_rate wins."""
        stream = {
            "r_frame_rate": "30/1",
            "avg_frame_rate": "24/1",
            "width": 1920,
            "height": 1080,
            "nb_frames": "100",
        }
        with patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)):
            meta = VideoAnnotator._get_video_metadata(Path("video.mp4"))
        assert meta.fps == 24.0


class TestParseFps:
    def test_normal_avg(self):
        assert _parse_fps("30/1", "30/1") == 30.0

    def test_avg_preferred(self):
        assert _parse_fps("24/1", "30/1") == 24.0

    def test_hevc_timebase_r_frame_rate(self):
        assert _parse_fps("25740000/2052571", "90000/1") == pytest.approx(12.54, abs=0.01)

    def test_no_avg_falls_back_to_r(self):
        assert _parse_fps(None, "25/1") == 25.0

    def test_both_none_returns_30(self):
        assert _parse_fps(None, None) == 30.0

    def test_both_invalid_returns_30(self):
        assert _parse_fps("invalid", "invalid") == 30.0

    def test_zero_den_returns_fallback(self):
        assert _parse_fps("0/0", "30/1") == 30.0

    def test_high_avg_and_high_r_uses_avg(self):
        """When both are above max, avg_frame_rate is still used (with warning)."""
        assert _parse_fps("90000/1", "90000/1") == 90000.0


# --- _extract_raw_detections ---

class TestExtractRawDetections:
    def test_single_detection(self, annotator):
        result = _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        dets = annotator._extract_raw_detections([result], frame_num=5, class_filter=None)
        assert len(dets) == 1
        assert dets[0].frame_num == 5
        assert dets[0].class_name == "person"
        assert dets[0].confidence == pytest.approx(0.9)
        assert dets[0].bbox == (10, 20, 100, 200)

    def test_class_filter(self, annotator):
        result = _make_yolo_result([
            (10, 20, 100, 200, 0, 0.9),
            (50, 60, 150, 250, 1, 0.8),
        ])
        dets = annotator._extract_raw_detections([result], frame_num=0, class_filter=["person"])
        assert len(dets) == 1
        assert dets[0].class_name == "person"

    def test_empty_boxes(self, annotator):
        result = _make_yolo_result([])
        dets = annotator._extract_raw_detections([result], frame_num=0, class_filter=None)
        assert dets == []


# --- _draw_detections ---

class TestDrawDetections:
    def test_calls_visualizer(self, annotator, mock_visualizer, sample_frame, default_params):
        dets = [
            DetectionBox(x1=10, y1=20, x2=100, y2=200, class_id=0, class_name="person", confidence=0.9),
            DetectionBox(x1=50, y1=60, x2=150, y2=250, class_id=1, class_name="car", confidence=0.8),
        ]
        annotator._draw_detections(sample_frame, dets, default_params, font_scale=0.5)
        assert mock_visualizer.draw_detection.call_count == 2
        for call in mock_visualizer.draw_detection.call_args_list:
            assert call.kwargs["font_scale"] == 0.5


# --- annotate() pipeline ---

class TestAnnotatePipeline:
    def _make_frames(self, num_frames: int, width: int = 640, height: int = 480):
        """Create a list of frames for the decoder mock to return."""
        frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]
        return frames

    def _make_decoder_mock(self, frames: list[np.ndarray]):
        """Create a single decoder mock instance with its own frame sequence."""
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = list(frames) + [None]
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)
        return mock_decoder

    def _setup_ffmpeg_mocks(self, frames: list[np.ndarray]):
        """Set up mock FFmpegDecoder and FFmpegEncoder for two-pass pipeline.

        Returns (mock_decoder_cls, mock_encoder_cls, mock_encoder_instance).
        The decoder class returns two separate instances (pass 1 and pass 2).
        """
        # Pass 1 decoder (for YOLO collection)
        decoder_pass1 = self._make_decoder_mock(frames)
        # Pass 2 decoder (for rendering)
        decoder_pass2 = self._make_decoder_mock(frames)

        mock_decoder_cls = MagicMock(side_effect=[decoder_pass1, decoder_pass2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        return mock_decoder_cls, mock_encoder_cls, mock_encoder

    def test_full_pipeline(self, mock_model, mock_visualizer, hw_config, tmp_path):
        num_frames = 6
        detect_every = 3
        frames = self._make_frames(num_frames)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        ffprobe_stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        # Disable grace periods for predictable test behavior
        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=detect_every)
            )

        assert stats.total_frames == num_frames
        # Frames 0, 3 are detection frames
        assert stats.detected_frames == 2
        # Track spans frames 0-3 (zero grace). Non-detection frames with
        # stabilized output: 1, 2 → tracked_frames = 2
        assert stats.tracked_frames == 2
        # 4 stabilized frames (0,1,2,3) × 1 detection each
        assert stats.total_detections == 4
        assert mock_encoder.write_frame.call_count == num_frames

    def test_detect_every_1(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """When detect_every=1, every frame gets YOLO detection, no hold frames."""
        num_frames = 4
        frames = self._make_frames(num_frames)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)

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

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=1)
            )

        assert stats.total_frames == num_frames
        assert stats.detected_frames == num_frames
        assert stats.tracked_frames == 0
        assert mock_model.predict.call_count == num_frames

    def test_hold_reuses_detections(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """With stabilizer, track covers detection frame range (zero grace)."""
        num_frames = 3
        frames = self._make_frames(num_frames)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)

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

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=3)
            )

        assert stats.total_frames == num_frames
        assert stats.detected_frames == 1
        assert mock_model.predict.call_count == 1
        # Only frame 0 has detection; zero grace → track covers frame 0 only
        # No non-detection frames have stabilized output
        assert stats.tracked_frames == 0
        assert stats.total_detections == 1
        assert mock_visualizer.draw_detection.call_count == 1

    def test_hold_clears_on_empty_detection(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """When detection frame returns no objects, track only covers frame 0 (zero grace)."""
        num_frames = 4
        frames = self._make_frames(num_frames)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)

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

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=3)
            )

        assert stats.total_frames == num_frames
        assert stats.detected_frames == 2
        # Track covers only frame 0 (single detection, zero grace)
        assert stats.tracked_frames == 0
        assert mock_model.predict.call_count == 2
        assert stats.total_detections == 1
        assert mock_visualizer.draw_detection.call_count == 1

    def test_decoder_failure_raises_error(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """When FFmpegDecoder.read_frame raises RuntimeError, annotate propagates it."""
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = RuntimeError("FFmpeg decoder crashed (rc=1)")
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        mock_decoder_cls = MagicMock(return_value=mock_decoder)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        ffprobe_stream = {
            "r_frame_rate": "30/1",
            "width": 640,
            "height": 480,
            "nb_frames": "100",
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            with pytest.raises(RuntimeError, match="FFmpeg decoder crashed"):
                annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

    def test_progress_callback(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """Progress callback is called during processing."""
        num_frames = 20
        frames = self._make_frames(num_frames)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)

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

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=config,
        )
        callback = MagicMock()

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            stats = annotator.annotate(input_path, output_path, AnnotationParams(), progress_callback=callback)

        assert callback.call_count > 0
        # All progress values should be <= 99
        for call in callback.call_args_list:
            assert call.args[0] <= 99


class TestAutoCodecResolve:
    """Test VIDEO_CODEC=auto resolution from input metadata."""

    def _make_frames(self, num_frames: int, width: int = 640, height: int = 480):
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]

    def _make_decoder_mock(self, frames: list[np.ndarray]):
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = list(frames) + [None]
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)
        return mock_decoder

    def _setup_ffmpeg_mocks(self, frames: list[np.ndarray]):
        decoder_pass1 = self._make_decoder_mock(frames)
        decoder_pass2 = self._make_decoder_mock(frames)

        mock_decoder_cls = MagicMock(side_effect=[decoder_pass1, decoder_pass2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)

        mock_decoder_cls = MagicMock(side_effect=[decoder_pass1, decoder_pass2])
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        return mock_decoder_cls, mock_encoder_cls, mock_encoder

    def _ffprobe_result(self, stream: dict) -> MagicMock:
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"streams": [stream]})
        return result

    def test_auto_hevc_with_bitrate(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: hevc input with bitrate -> h265 codec + bitrate in encoder."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "hevc", "bit_rate": "8000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            codec="auto", stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        # codec is the 7th positional arg (index 6), crf/bitrate are kwargs
        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.args[6] == "h265"
        assert encoder_call.kwargs.get("bitrate") == 8000000

    def test_auto_h264_with_bitrate(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: h264 input with bitrate -> h264 codec + bitrate."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "h264", "bit_rate": "5000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            codec="auto", stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.args[6] == "h264"
        assert encoder_call.kwargs.get("bitrate") == 5000000

    def test_auto_av1_with_bitrate(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: av1 input with bitrate -> av1 codec + bitrate."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "av1", "bit_rate": "4000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            codec="auto", stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.args[6] == "av1"
        assert encoder_call.kwargs.get("bitrate") == 4000000

    def test_auto_hevc_no_bitrate_uses_crf(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: hevc input without bitrate -> h265 codec + CRF 18."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "hevc",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            codec="auto", stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.args[6] == "h265"
        assert encoder_call.kwargs.get("crf") == 18
        assert encoder_call.kwargs.get("bitrate") is None

    def test_auto_unsupported_codec_fallback(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: vp9 input -> fallback to h264 + CRF 18."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "vp9", "bit_rate": "6000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            codec="auto", stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.args[6] == "h264"
        assert encoder_call.kwargs.get("crf") == 18

    def test_auto_crf_always_18_even_if_configured(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """auto mode: CRF fallback is always 18, regardless of configured crf."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "hevc",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            codec="auto", crf=23, stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.kwargs.get("crf") == 18

    def test_explicit_codec_ignores_source(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """Explicit VIDEO_CODEC=h264 ignores source codec/bitrate."""
        frames = self._make_frames(2)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)
        mock_model.predict.return_value = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": "2", "codec_name": "hevc", "bit_rate": "8000000",
        }

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            codec="h264", crf=23, stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(stream)),
        ):
            annotator.annotate(input_path, tmp_path / "out.mp4", AnnotationParams())

        encoder_call = mock_encoder_cls.call_args
        assert encoder_call.args[6] == "h264"
        assert encoder_call.kwargs.get("crf") == 23


class TestAnnotateCancellation:
    def _make_frames(self, num_frames: int, width: int = 640, height: int = 480):
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]

    def _make_decoder_mock(self, frames: list[np.ndarray]):
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = list(frames) + [None]
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)
        return mock_decoder

    def _ffprobe_result(self, num_frames: int) -> MagicMock:
        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": str(num_frames),
        }
        r = MagicMock()
        r.returncode = 0
        r.stdout = json.dumps({"streams": [stream]})
        return r

    def test_cancel_during_pass1_raises(self, mock_model, mock_visualizer, hw_config, tmp_path):
        from video_annotator import JobCancelledError

        num_frames = 20
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        decoder2 = self._make_decoder_mock(frames)
        mock_decoder_cls = MagicMock(side_effect=[decoder1, decoder2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        cancel_event = threading.Event()
        call_count = {"n": 0}

        def fake_predict(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                cancel_event.set()
            return [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        mock_model.predict.side_effect = fake_predict

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # Pass 2 must never start (second decoder never used).
        assert decoder2.read_frame.call_count == 0
        # FFmpegDecoder __exit__ was invoked for pass 1.
        assert decoder1.__exit__.called

    def test_cancel_preset_before_ffprobe_raises(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """Cancel set before annotate() — must raise before even probing metadata."""
        from video_annotator import JobCancelledError

        cancel_event = threading.Event()
        cancel_event.set()

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        mock_decoder_cls = MagicMock()
        mock_encoder_cls = MagicMock()

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run") as mock_run,
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # ffprobe must NOT have been called.
        assert mock_run.call_count == 0
        # FFmpeg pipelines must NOT have been constructed.
        assert mock_decoder_cls.call_count == 0
        assert mock_encoder_cls.call_count == 0
        # YOLO must NOT have been invoked.
        assert mock_model.predict.call_count == 0

    def test_cancel_before_pass1_first_iteration_raises(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """Cancel set after ffprobe but before first loop iter — pass1 guard catches it."""
        from video_annotator import JobCancelledError

        num_frames = 5
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        mock_decoder_cls = MagicMock(side_effect=[decoder1])

        mock_encoder_cls = MagicMock()

        cancel_event = threading.Event()

        # Fire cancel when the decoder context manager opens — this puts us
        # past the early annotate() check and past ffprobe, so the pass1
        # in-loop guard must be the one that raises.
        original_enter = decoder1.__enter__
        def enter_and_cancel(*args, **kwargs):
            cancel_event.set()
            return original_enter(*args, **kwargs)
        decoder1.__enter__ = MagicMock(side_effect=enter_and_cancel)

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # Pass 1 loop exits on the very first iteration: no predict call,
        # no decode call, pass 2 decoder/encoder never constructed.
        assert mock_model.predict.call_count == 0
        assert decoder1.read_frame.call_count == 0
        assert mock_encoder_cls.call_count == 0

    def test_cancel_during_pass2_raises(self, mock_model, mock_visualizer, hw_config, tmp_path):
        from video_annotator import JobCancelledError

        num_frames = 10
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        decoder2 = self._make_decoder_mock(frames)
        mock_decoder_cls = MagicMock(side_effect=[decoder1, decoder2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        cancel_event = threading.Event()

        # Set event after pass 1 by hooking into write_frame (pass 2 only).
        write_calls = {"n": 0}

        def fake_write(frame):
            write_calls["n"] += 1
            if write_calls["n"] >= 2:
                cancel_event.set()

        mock_encoder.write_frame.side_effect = fake_write

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # Fewer than all frames written in pass 2.
        assert mock_encoder.write_frame.call_count < num_frames

    def test_cancel_after_pass1_skips_stabilize(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """Cancel fired at end of pass 1 — DetectionStabilizer.stabilize() must not run."""
        from video_annotator import JobCancelledError

        num_frames = 5
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        mock_decoder_cls = MagicMock(side_effect=[decoder1])
        mock_encoder_cls = MagicMock()

        cancel_event = threading.Event()
        call_count = {"n": 0}

        def fake_predict(*args, **kwargs):
            call_count["n"] += 1
            result = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]
            if call_count["n"] == num_frames:
                cancel_event.set()
            return result

        mock_model.predict.side_effect = fake_predict

        # Mock DetectionStabilizer class so we can assert stabilize() wasn't called.
        mock_stabilizer_instance = MagicMock()
        mock_stabilizer_cls = MagicMock(return_value=mock_stabilizer_instance)

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.DetectionStabilizer", mock_stabilizer_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # DetectionStabilizer must NOT have been constructed or called.
        assert mock_stabilizer_cls.call_count == 0
        assert mock_stabilizer_instance.stabilize.call_count == 0
        # Pass 2 encoder must NOT have been constructed.
        assert mock_encoder_cls.call_count == 0

    def test_cancel_after_pass1_skips_pass2(self, mock_model, mock_visualizer, hw_config, tmp_path):
        """Cancel fires from last pass-1 predict — pass 2 FFmpeg must not start.

        Companion to ``test_cancel_after_pass1_skips_stabilize``: both exercise
        the same trigger (cancel from the last pass-1 predict call), hitting
        the pre-stabilize guard first, but assert different downstream effects
        (this one: no pass-2 decoder/encoder construction).
        """
        from video_annotator import JobCancelledError

        num_frames = 5
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        decoder2 = self._make_decoder_mock(frames)  # must never be used
        mock_decoder_cls = MagicMock(side_effect=[decoder1, decoder2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        cancel_event = threading.Event()
        # Fire cancel from the LAST pass-1 predict call, so pass 1 completes
        # normally and the between-passes guard is what raises.
        call_count = {"n": 0}

        def fake_predict(*args, **kwargs):
            call_count["n"] += 1
            result = [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]
            if call_count["n"] == num_frames:
                cancel_event.set()
            return result

        mock_model.predict.side_effect = fake_predict

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # Pass 2 decoder / encoder must never be constructed.
        # FFmpegDecoder was called exactly once (pass 1 only).
        assert mock_decoder_cls.call_count == 1
        assert mock_encoder_cls.call_count == 0

    def test_cancel_event_none_runs_to_completion(self, mock_model, mock_visualizer, hw_config, tmp_path):
        num_frames = 3
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        decoder2 = self._make_decoder_mock(frames)
        mock_decoder_cls = MagicMock(side_effect=[decoder1, decoder2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            # cancel_event omitted — behaviour unchanged.
            stats = annotator.annotate(
                input_path, tmp_path / "out.mp4",
                AnnotationParams(detect_every=1),
            )

        assert stats.total_frames == num_frames


class TestIsNvencOomError:
    """Classifier helper for NVENC VRAM-initialisation failure signatures."""

    def test_init_encoder_out_of_memory(self):
        msg = (
            "FFmpeg encoder pipe broken: [Errno 32] Broken pipe. stderr: "
            "[hevc_nvenc @ 0x1234] InitializeEncoder failed: out of memory (10)"
        )
        assert _is_nvenc_oom_error(msg) is True

    def test_create_input_buffer_out_of_memory(self):
        msg = (
            "FFmpeg encoder pipe broken: [Errno 32] Broken pipe. stderr: "
            "[hevc_nvenc @ 0x1234] CreateInputBuffer failed: out of memory (10)"
        )
        assert _is_nvenc_oom_error(msg) is True

    def test_cannot_allocate_memory_marker(self):
        msg = "[hevc_nvenc] encoder setup failed: Cannot allocate memory"
        assert _is_nvenc_oom_error(msg) is True

    def test_encode_api_internal_error_marker(self):
        msg = "[h264_nvenc @ 0x...] EncodeAPI Internal Error."
        assert _is_nvenc_oom_error(msg) is True

    def test_non_nvenc_message_rejected(self):
        assert _is_nvenc_oom_error("FFmpeg decoder crashed (rc=1)") is False

    def test_nvenc_without_oom_marker_rejected(self):
        assert _is_nvenc_oom_error("hevc_nvenc: some other failure") is False

    def test_empty_message(self):
        assert _is_nvenc_oom_error("") is False

    def test_case_insensitive_out_of_memory(self):
        msg = "[HEVC_NVENC] InitializeEncoder failed: Out Of Memory"
        assert _is_nvenc_oom_error(msg) is True

    def test_case_insensitive_all_caps(self):
        msg = "[HEVC_NVENC] OUT OF MEMORY (10)"
        assert _is_nvenc_oom_error(msg) is True


class TestNvencFallback:
    """Pass 2 CPU fallback when NVENC initialisation fails (VRAM pressure).

    Triggered when FFmpegEncoder.write_frame raises a RuntimeError matching
    the _is_nvenc_oom_error signature. annotate() must then re-run Pass 2
    with a CPU-only HWAccelConfig, CRF mode, and no bitrate forwarding.
    """

    def _make_frames(self, num_frames: int, width: int = 640, height: int = 480):
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]

    def _make_decoder_mock(self, frames: list[np.ndarray]):
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = list(frames) + [None]
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)
        return mock_decoder

    def _ffprobe_result(self, num_frames: int) -> MagicMock:
        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": str(num_frames),
        }
        r = MagicMock()
        r.returncode = 0
        r.stdout = json.dumps({"streams": [stream]})
        return r

    def test_nvenc_oom_triggers_cpu_fallback(self, mock_model, mock_visualizer, tmp_path):
        """First NVENC-OOM in Pass 2 is caught; Pass 2 re-runs on CPU with CRF."""
        nvidia_hw_config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        num_frames = 3
        dec_p1 = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_nvenc = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_cpu = self._make_decoder_mock(self._make_frames(num_frames))
        mock_decoder_cls = MagicMock(side_effect=[dec_p1, dec_p2_nvenc, dec_p2_cpu])

        failing_encoder = MagicMock()
        failing_encoder.__enter__ = MagicMock(return_value=failing_encoder)
        failing_encoder.__exit__ = MagicMock(return_value=False)
        failing_encoder.write_frame.side_effect = RuntimeError(
            "FFmpeg encoder pipe broken: [Errno 32] Broken pipe. stderr: "
            "[hevc_nvenc @ 0x1234] InitializeEncoder failed: "
            "out of memory (10): EncodeAPI Internal Error."
        )

        ok_encoder = MagicMock()
        ok_encoder.__enter__ = MagicMock(return_value=ok_encoder)
        ok_encoder.__exit__ = MagicMock(return_value=False)

        # Observed state of output_path at the moment the second (CPU) encoder
        # constructor is invoked — the retry must have unlinked the partial
        # output file from the failed NVENC run by then.
        output_exists_at_cpu_retry: list[bool] = []
        encoder_call_count = {"n": 0}

        def _encoder_factory(*args, **kwargs):
            encoder_call_count["n"] += 1
            if encoder_call_count["n"] == 1:
                return failing_encoder
            # Second call = CPU retry — capture output_path state right now.
            output_exists_at_cpu_retry.append(args[1].exists())
            return ok_encoder

        mock_encoder_cls = MagicMock(side_effect=_encoder_factory)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"
        # The failing NVENC run may have created a zero-byte output file.
        # Fallback must remove it before retrying.
        output_path.touch()
        assert output_path.exists()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, nvidia_hw_config,
            codec="h265", crf=18,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=1),
            )

        assert stats.total_frames == num_frames

        # Two encoder constructions: NVENC attempt then CPU retry.
        assert mock_encoder_cls.call_count == 2
        first_call, second_call = mock_encoder_cls.call_args_list

        # args[5] is hw_config, args[6] is codec.
        assert first_call.args[5].accel_type == HWAccelType.NVIDIA
        assert first_call.args[6] == "h265"
        assert second_call.args[5].accel_type == HWAccelType.CPU
        # Same codec on retry — only hw accel changes.
        assert second_call.args[6] == "h265"
        # CPU retry forces CRF, clears bitrate.
        assert second_call.kwargs.get("bitrate") is None
        assert second_call.kwargs.get("crf") == 18

        # Pass 2 decoder also rebuilt for the CPU retry.
        assert mock_decoder_cls.call_count == 3
        pass1_call, pass2_nvenc_call, pass2_cpu_call = mock_decoder_cls.call_args_list
        assert pass1_call.args[3].accel_type == HWAccelType.NVIDIA
        assert pass2_nvenc_call.args[3].accel_type == HWAccelType.NVIDIA
        assert pass2_cpu_call.args[3].accel_type == HWAccelType.CPU

        # The CPU encoder consumed all Pass 2 frames.
        assert ok_encoder.write_frame.call_count == num_frames

        # Partial output from the failed NVENC run was cleaned up before the
        # CPU retry started (D: explicit unlink contract test).
        assert output_exists_at_cpu_retry == [False]

    def test_non_nvenc_runtimeerror_propagates_without_retry(
        self, mock_model, mock_visualizer, tmp_path
    ):
        """RuntimeError without NVENC signature is re-raised; no CPU retry."""
        nvidia_hw_config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        num_frames = 3
        dec_p1 = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2 = self._make_decoder_mock(self._make_frames(num_frames))
        mock_decoder_cls = MagicMock(side_effect=[dec_p1, dec_p2])

        failing_encoder = MagicMock()
        failing_encoder.__enter__ = MagicMock(return_value=failing_encoder)
        failing_encoder.__exit__ = MagicMock(return_value=False)
        failing_encoder.write_frame.side_effect = RuntimeError(
            "FFmpeg encoder crashed (rc=1): ffmpeg binary not found"
        )
        mock_encoder_cls = MagicMock(return_value=failing_encoder)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, nvidia_hw_config,
            codec="h265", crf=18,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(RuntimeError, match="ffmpeg binary not found"):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                )

        # Encoder constructed only once — no retry attempt.
        assert mock_encoder_cls.call_count == 1

    def test_cpu_fallback_retry_failure_propagates(
        self, mock_model, mock_visualizer, tmp_path
    ):
        """If the CPU retry itself fails, the error propagates to the caller."""
        nvidia_hw_config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        num_frames = 3
        dec_p1 = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_nvenc = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_cpu = self._make_decoder_mock(self._make_frames(num_frames))
        mock_decoder_cls = MagicMock(side_effect=[dec_p1, dec_p2_nvenc, dec_p2_cpu])

        def _make_failing_encoder(msg: str) -> MagicMock:
            enc = MagicMock()
            enc.__enter__ = MagicMock(return_value=enc)
            enc.__exit__ = MagicMock(return_value=False)
            enc.write_frame.side_effect = RuntimeError(msg)
            return enc

        nvenc_fail = _make_failing_encoder(
            "FFmpeg encoder pipe broken: stderr: [hevc_nvenc] "
            "InitializeEncoder failed: out of memory (10)"
        )
        cpu_fail = _make_failing_encoder(
            "FFmpeg encoder crashed (rc=1): libx265 not available"
        )
        mock_encoder_cls = MagicMock(side_effect=[nvenc_fail, cpu_fail])

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, nvidia_hw_config,
            codec="h265", crf=18,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(RuntimeError, match="libx265 not available"):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                )

        # Two encoder attempts, no third.
        assert mock_encoder_cls.call_count == 2

    def test_auto_mode_bitrate_fallback_uses_crf_18_not_self_crf(
        self, mock_model, mock_visualizer, tmp_path
    ):
        """In VIDEO_CODEC=auto with source bitrate, the NVENC attempt uses
        -b:v (effective_crf=None). CPU retry must drop the bitrate AND pick
        CRF 18 (the auto-mode policy from _resolve_codec), NOT self.crf.
        Regression guard for the Codex P2 finding.
        """
        nvidia_hw_config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        num_frames = 3
        dec_p1 = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_nvenc = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_cpu = self._make_decoder_mock(self._make_frames(num_frames))
        mock_decoder_cls = MagicMock(side_effect=[dec_p1, dec_p2_nvenc, dec_p2_cpu])

        failing_encoder = MagicMock()
        failing_encoder.__enter__ = MagicMock(return_value=failing_encoder)
        failing_encoder.__exit__ = MagicMock(return_value=False)
        failing_encoder.write_frame.side_effect = RuntimeError(
            "FFmpeg encoder pipe broken: stderr: [hevc_nvenc] "
            "InitializeEncoder failed: out of memory (10)"
        )

        ok_encoder = MagicMock()
        ok_encoder.__enter__ = MagicMock(return_value=ok_encoder)
        ok_encoder.__exit__ = MagicMock(return_value=False)

        mock_encoder_cls = MagicMock(side_effect=[failing_encoder, ok_encoder])

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        # HEVC source with bitrate — auto-mode resolves to (h265, None, 8 Mbps).
        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": str(num_frames),
            "codec_name": "hevc", "bit_rate": "8000000",
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [stream]})

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        # Configured crf=23 on purpose — must NOT leak into the retry,
        # because _resolve_codec's auto-mode policy is CRF 18.
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, nvidia_hw_config,
            codec="auto", crf=23,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            annotator.annotate(
                input_path, tmp_path / "out.mp4",
                AnnotationParams(detect_every=1),
            )

        # NVENC attempt: bitrate mode (effective_crf=None, effective_bitrate=8M).
        first_call, second_call = mock_encoder_cls.call_args_list
        assert first_call.kwargs.get("crf") is None
        assert first_call.kwargs.get("bitrate") == 8_000_000

        # CPU retry: bitrate dropped, CRF from auto-mode policy (18), not self.crf=23.
        assert second_call.kwargs.get("bitrate") is None
        assert second_call.kwargs.get("crf") == 18

    def test_explicit_codec_fallback_preserves_self_crf(
        self, mock_model, mock_visualizer, tmp_path
    ):
        """Explicit VIDEO_CODEC=h265 with CRF=23: _resolve_codec returns
        effective_crf=23, so the CPU retry must also use crf=23.
        Complements the auto-mode case above.
        """
        nvidia_hw_config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        num_frames = 3
        dec_p1 = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_nvenc = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_cpu = self._make_decoder_mock(self._make_frames(num_frames))
        mock_decoder_cls = MagicMock(side_effect=[dec_p1, dec_p2_nvenc, dec_p2_cpu])

        failing_encoder = MagicMock()
        failing_encoder.__enter__ = MagicMock(return_value=failing_encoder)
        failing_encoder.__exit__ = MagicMock(return_value=False)
        failing_encoder.write_frame.side_effect = RuntimeError(
            "FFmpeg encoder pipe broken: stderr: [hevc_nvenc] "
            "InitializeEncoder failed: out of memory (10)"
        )

        ok_encoder = MagicMock()
        ok_encoder.__enter__ = MagicMock(return_value=ok_encoder)
        ok_encoder.__exit__ = MagicMock(return_value=False)

        mock_encoder_cls = MagicMock(side_effect=[failing_encoder, ok_encoder])

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, nvidia_hw_config,
            codec="h265", crf=23,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            annotator.annotate(
                input_path, tmp_path / "out.mp4",
                AnnotationParams(detect_every=1),
            )

        # Both attempts use crf=23; retry drops any bitrate (there wasn't one).
        first_call, second_call = mock_encoder_cls.call_args_list
        assert first_call.kwargs.get("crf") == 23
        assert second_call.kwargs.get("crf") == 23
        assert second_call.kwargs.get("bitrate") is None

    def test_cancel_between_nvenc_failure_and_cpu_retry(
        self, mock_model, mock_visualizer, tmp_path
    ):
        """If /cancel fires in the window between NVENC failure and CPU retry,
        the retry must NOT start — the job aborts with JobCancelledError."""
        from video_annotator import JobCancelledError

        nvidia_hw_config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)
        cancel_event = threading.Event()

        num_frames = 3
        dec_p1 = self._make_decoder_mock(self._make_frames(num_frames))
        dec_p2_nvenc = self._make_decoder_mock(self._make_frames(num_frames))
        # Third decoder must never be constructed — cancel should preempt
        # the CPU retry entirely.
        dec_p2_cpu_unused = self._make_decoder_mock(self._make_frames(num_frames))
        mock_decoder_cls = MagicMock(side_effect=[dec_p1, dec_p2_nvenc, dec_p2_cpu_unused])

        # NVENC write_frame fails AND sets the cancel event. This simulates a
        # user-issued /cancel arriving while the annotator was blocked on the
        # failing FFmpeg subprocess cleanup.
        failing_encoder = MagicMock()
        failing_encoder.__enter__ = MagicMock(return_value=failing_encoder)
        failing_encoder.__exit__ = MagicMock(return_value=False)

        def _fail_and_cancel(_frame):
            cancel_event.set()
            raise RuntimeError(
                "FFmpeg encoder pipe broken: stderr: [hevc_nvenc] "
                "InitializeEncoder failed: out of memory (10)"
            )

        failing_encoder.write_frame.side_effect = _fail_and_cancel

        # Second encoder should never be constructed — if it is, the test
        # fails because the retry proceeded past the cancel check.
        unused_encoder = MagicMock()
        unused_encoder.__enter__ = MagicMock(return_value=unused_encoder)
        unused_encoder.__exit__ = MagicMock(return_value=False)
        mock_encoder_cls = MagicMock(side_effect=[failing_encoder, unused_encoder])

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, nvidia_hw_config,
            codec="h265", crf=18,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # Only the NVENC encoder was constructed. CPU retry never started.
        assert mock_encoder_cls.call_count == 1
        # Third decoder (the would-be CPU retry) was never requested.
        assert dec_p2_cpu_unused.read_frame.call_count == 0
