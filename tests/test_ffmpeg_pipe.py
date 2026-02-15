import subprocess
from unittest.mock import patch, MagicMock, call
from io import BytesIO

import numpy as np
import pytest

from ffmpeg_pipe import FFmpegDecoder, FFmpegEncoder
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


class TestFFmpegEncoder:
    def _make_mock_process(self, returncode: int = 0):
        """Create mock Popen for encoder tests."""
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        # stderr must be iterable for _drain_stderr daemon thread.
        mock_proc.stderr = BytesIO(b"")
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = returncode
        mock_proc.returncode = returncode
        return mock_proc

    def test_write_frame(self):
        """write_frame writes correct raw bytes to stdin."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
            ) as encoder:
                encoder.write_frame(frame)

        mock_proc.stdin.write.assert_called_once_with(frame.tobytes())

    def test_cpu_encode_command(self):
        """CPU encode command includes libx264 and pipe:0."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
            ):
                pass

        cmd = mock_popen.call_args[0][0]
        assert "libx264" in cmd
        assert "pipe:0" in cmd

    def test_nvidia_encode_command(self):
        """NVIDIA encode command includes h264_nvenc."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.NVIDIA)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
            ):
                pass

        cmd = mock_popen.call_args[0][0]
        assert "h264_nvenc" in cmd

    def test_audio_merge_in_command(self):
        """Command has two -i inputs, -map for audio, and aac codec."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
            ):
                pass

        cmd = mock_popen.call_args[0][0]
        # Two -i inputs: pipe:0 for video and original file for audio
        i_indices = [idx for idx, arg in enumerate(cmd) if arg == "-i"]
        assert len(i_indices) == 2
        assert cmd[i_indices[0] + 1] == "pipe:0"
        assert cmd[i_indices[1] + 1] == "input.mp4"
        # Audio mapping and codec
        assert "-map" in cmd
        assert "1:a:0?" in cmd
        assert "aac" in cmd

    def test_cleanup_on_exit(self):
        """Verify stdin is closed and process is waited on exit."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
            ):
                pass

        mock_proc.stdin.close.assert_called()
        mock_proc.wait.assert_called()

    def test_nonzero_exit_raises(self):
        """RuntimeError raised when FFmpeg exits with nonzero code."""
        mock_proc = self._make_mock_process(returncode=1)
        # wait() must also set returncode to 1 *after* being called
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg encoder failed"):
                with FFmpegEncoder(
                    original_path="input.mp4",
                    output_path="output.mp4",
                    width=640,
                    height=480,
                    fps=30.0,
                    hw_config=config,
                    codec="h264",
                    crf=18,
                ):
                    pass

    def test_amd_global_encode_args_in_command(self):
        """-vaapi_device appears before -i in the command."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.AMD)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                crf=18,
            ):
                pass

        cmd = mock_popen.call_args[0][0]
        vaapi_idx = cmd.index("-vaapi_device")
        first_i_idx = cmd.index("-i")
        assert vaapi_idx < first_i_idx, "-vaapi_device must appear before first -i"

    def test_write_frame_after_crash_raises(self):
        """write_frame raises RuntimeError if process has already crashed."""
        mock_proc = self._make_mock_process(returncode=1)
        mock_proc.poll.return_value = 1
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg encoder crashed"):
                with FFmpegEncoder(
                    original_path="input.mp4",
                    output_path="output.mp4",
                    width=640,
                    height=480,
                    fps=30.0,
                    hw_config=config,
                    codec="h264",
                    crf=18,
                ) as encoder:
                    encoder.write_frame(frame)

    def test_bitrate_mode_command(self):
        """When bitrate is passed, command uses -b:v instead of -crf."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
                bitrate=8000000,
            ):
                pass

        cmd = mock_popen.call_args[0][0]
        assert "-b:v" in cmd
        assert "8000000" in cmd
        assert "-crf" not in cmd

    def test_crf_mode_default(self):
        """When neither crf nor bitrate passed, uses crf=18 default."""
        mock_proc = self._make_mock_process()
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegEncoder(
                original_path="input.mp4",
                output_path="output.mp4",
                width=640,
                height=480,
                fps=30.0,
                hw_config=config,
                codec="h264",
            ):
                pass

        cmd = mock_popen.call_args[0][0]
        assert "-crf" in cmd
        assert "-b:v" not in cmd
