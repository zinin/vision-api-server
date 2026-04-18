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

    def test_close_with_unkillable_process_warns_and_does_not_leak_timeout(self, caplog):
        """FFmpegDecoder.close() must not leak TimeoutExpired when both
        the primary wait() and the post-kill wait() time out (D-state,
        hung NFS, GPU driver wedge). When the process survives SIGKILL
        (returncode stays None), close() must emit a WARNING so callers
        know teardown did not complete — otherwise a hung decoder
        process leaks silently."""
        import logging
        mock_proc = self._make_mock_process([])
        mock_stdout = MagicMock()
        mock_stdout.read.return_value = b""
        mock_proc.stdout = mock_stdout
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5),
        ]
        mock_proc.returncode = None  # process survives SIGKILL
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with caplog.at_level(logging.WARNING, logger="ffmpeg_pipe"):
                with FFmpegDecoder("input.mp4", 640, 480, config):
                    pass  # exit triggers close(), which hits both timeouts

        mock_proc.kill.assert_called_once()
        assert any(
            "did not exit after SIGKILL" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        ), f"expected SIGKILL-survival WARNING, got records: {caplog.records}"

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

    def test_write_frame_flushes_stdin(self):
        """write_frame must flush stdin after each frame so the Python-side
        buffer never holds residual bytes that close()'s implicit flush
        could push into a pipe ffmpeg has already closed (e.g. after
        -shortest). This is the buffered-stdin counterpart to the
        BufferedWriter's write-all-bytes contract — we keep buffered
        writes (so partial writes cannot corrupt the rawvideo stream)
        while making sure the pipe is drained on every frame."""
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
                encoder.write_frame(frame)

        assert mock_proc.stdin.flush.call_count == 2

    def test_write_frame(self):
        """write_frame writes correct raw bytes to stdin and returns True."""
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
                result = encoder.write_frame(frame)

        assert result is True
        mock_proc.stdin.write.assert_called_once_with(frame.tobytes())

    def test_write_frame_returns_false_after_eof(self):
        """After encoder finalises (e.g. -shortest), write_frame returns False
        so callers can break their loop instead of wasting CPU decoding
        frames that will never reach ffmpeg."""
        mock_proc = self._make_mock_process(returncode=0)
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
        mock_proc.wait.return_value = 0

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
                # First write triggers graceful EOF → False.
                first = encoder.write_frame(frame)
                # Subsequent writes short-circuit via _eof → False.
                second = encoder.write_frame(frame)

        assert first is False
        assert second is False

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

    def test_close_with_unkillable_process_raises(self):
        """FFmpegEncoder.close() must not leak TimeoutExpired when both the
        primary wait() and the post-kill wait() time out (D-state, hung
        NFS, GPU driver wedge). When the encoder survives SIGKILL
        (returncode stays None), close() must raise RuntimeError so
        _pass2_render cannot report success with an unfinished output
        while a stuck encoder process leaks."""
        mock_proc = self._make_mock_process()
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300),
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5),
        ]
        mock_proc.returncode = None  # process survives SIGKILL
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="did not exit after SIGKILL"):
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
                    pass  # exit triggers close(), which hits both timeouts

        mock_proc.kill.assert_called_once()

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

    def test_write_frame_graceful_eof_on_shortest(self):
        """write_frame MUST NOT raise when encoder exits cleanly (rc=0) via -shortest.

        Reproduces the real-world scenario where ffmpeg closes pipe:0 after
        audio EOF (shorter than video), Python gets BrokenPipeError, but the
        encoder process finishes normally. The output file is valid — we just
        need to stop writing further frames.
        """
        mock_proc = self._make_mock_process(returncode=0)
        # poll() returns None at the moment of the write (process still alive).
        mock_proc.poll.return_value = None
        # stdin.write raises BrokenPipeError (ffmpeg closed its stdin fd).
        mock_proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
        # wait() inside the BrokenPipe handler returns 0 — clean exit.
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0

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
                # First write triggers the graceful-eof path, must not raise.
                encoder.write_frame(frame)
                # Subsequent writes are silent no-ops (no additional stdin writes).
                encoder.write_frame(frame)
                encoder.write_frame(frame)

        # stdin.write called exactly once (the one that raised BrokenPipe).
        assert mock_proc.stdin.write.call_count == 1

    def test_write_frame_pipe_broken_with_nonzero_rc_raises(self):
        """BrokenPipe with encoder exiting rc != 0 must still raise."""
        mock_proc = self._make_mock_process(returncode=1)
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
        mock_proc.wait.return_value = 1
        mock_proc.returncode = 1

        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg encoder crashed mid-write"):
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

    def test_write_frame_pipe_broken_with_hang_kills_and_raises(self):
        """BrokenPipe + wait() timeout must kill the process and raise."""
        mock_proc = self._make_mock_process()
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
        # Three values: (1) wait() inside BrokenPipe handler times out,
        # (2) wait() after kill() returns, (3) wait() inside close() via __exit__.
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5),
            -9,
            -9,
        ]

        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg encoder hung after pipe break"):
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

        mock_proc.kill.assert_called_once()

    def test_write_frame_pipe_broken_with_unkillable_process_still_raises(self):
        """If even the post-kill wait() times out (ffmpeg stuck in D-state),
        write_frame must still surface a RuntimeError — not leak TimeoutExpired."""
        mock_proc = self._make_mock_process()
        mock_proc.poll.return_value = None
        mock_proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
        # Both wait() calls inside write_frame time out: first after BrokenPipe,
        # second after kill(). Third wait() is for close() via __exit__.
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5),
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5),
            -9,
        ]

        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="FFmpeg encoder hung after pipe break"):
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

        mock_proc.kill.assert_called_once()

    def test_write_frame_generic_oserror_with_rc_zero_is_silent(self):
        """Any OSError during stdin.write (EBADF, EINTR, etc.) — not just
        BrokenPipeError — goes through the same graceful-EOF path: check
        the encoder exit code, and if rc=0 treat it as clean finalisation."""
        mock_proc = self._make_mock_process(returncode=0)
        mock_proc.poll.return_value = None
        # errno 9 = EBADF, a plausible non-BrokenPipe OSError.
        mock_proc.stdin.write.side_effect = OSError(9, "Bad file descriptor")
        mock_proc.wait.return_value = 0

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
                result = encoder.write_frame(frame)

        assert result is False

    def test_write_frame_after_clean_exit_is_silent(self):
        """If poll() reports rc=0 before the write, treat as EOF (no raise)."""
        mock_proc = self._make_mock_process(returncode=0)
        mock_proc.poll.return_value = 0  # encoder already exited cleanly
        mock_proc.returncode = 0

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
                encoder.write_frame(frame)  # no raise
                encoder.write_frame(frame)  # no raise, silent no-op

        # stdin.write was never called because poll() short-circuited first.
        mock_proc.stdin.write.assert_not_called()

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
