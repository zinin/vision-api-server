import subprocess
from unittest.mock import patch, MagicMock, call
from io import BytesIO

import numpy as np
import pytest

from ffmpeg_pipe import FFmpegDecoder
from hw_accel import HWAccelConfig, HWAccelType


class TestFFmpegDecoder:
    def _make_mock_process(self, frames: list[np.ndarray]):
        """Create mock Popen that yields raw frame bytes then EOF."""
        raw_data = b"".join(f.tobytes() for f in frames)
        mock_proc = MagicMock()
        mock_proc.stdout = BytesIO(raw_data)
        # stderr must be iterable for _drain_stderr daemon thread.
        # Use a BytesIO with empty content so iteration terminates immediately.
        mock_proc.stderr = BytesIO(b"")
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        return mock_proc

    def test_reads_frames(self):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        mock_proc = self._make_mock_process(frames)
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 640, 480, config) as decoder:
                read_frames = []
                while True:
                    frame = decoder.read_frame()
                    if frame is None:
                        break
                    read_frames.append(frame)

        assert len(read_frames) == 3
        assert read_frames[0].shape == (480, 640, 3)

    def test_eof_returns_none(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 640, 480, config) as decoder:
                assert decoder.read_frame() is None

    def test_nvidia_decode_args(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegDecoder("input.mp4", 640, 480, config):
                pass
        cmd = mock_popen.call_args[0][0]
        assert "-hwaccel" in cmd
        assert "cuda" in cmd

    def test_cpu_no_hwaccel_args(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegDecoder("input.mp4", 640, 480, config):
                pass
        cmd = mock_popen.call_args[0][0]
        assert "-hwaccel" not in cmd

    def test_cleanup_on_exit(self):
        """Verify stdout is closed and process is waited on exit."""
        mock_proc = self._make_mock_process([])
        # Replace stdout with a MagicMock so we can assert close() was called.
        # Keep it behaving like empty BytesIO for read().
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_proc.stdout = mock_stdout
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 640, 480, config):
                pass

        mock_stdout.close.assert_called()
        mock_proc.wait.assert_called()

    def test_frames_are_writable(self):
        """Returned numpy arrays must be writable (for OpenCV drawing)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_proc = self._make_mock_process([frame])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 640, 480, config) as decoder:
                result = decoder.read_frame()

        assert result is not None
        assert result.flags.writeable

    def test_crash_raises_runtime_error(self):
        """If FFmpeg process crashes mid-stream, read_frame raises RuntimeError."""
        mock_proc = self._make_mock_process([])
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 640, 480, config) as decoder:
                with pytest.raises(RuntimeError, match="FFmpeg decoder crashed"):
                    decoder.read_frame()

    def test_ffmpeg_command_includes_rawvideo_output(self):
        """Verify FFmpeg command requests raw BGR24 pipe output."""
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegDecoder("input.mp4", 640, 480, config):
                pass
        cmd = mock_popen.call_args[0][0]
        assert "-f" in cmd
        assert "rawvideo" in cmd
        assert "-pix_fmt" in cmd
        assert "bgr24" in cmd
        assert "pipe:1" in cmd

    def test_amd_decode_args(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.AMD)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegDecoder("input.mp4", 640, 480, config):
                pass
        cmd = mock_popen.call_args[0][0]
        assert "-hwaccel" in cmd
        assert "vaapi" in cmd
