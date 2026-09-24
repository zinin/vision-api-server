import logging
import os
import subprocess
import sys
from unittest.mock import patch, MagicMock, call
from io import BytesIO

import numpy as np
import pytest

import ffmpeg_pipe
from ffmpeg_pipe import FFmpegDecoder, FFmpegEncoder, _grow_pipe, frame_shape
from hw_accel import HWAccelConfig, HWAccelType


class TestFrameShape:
    def test_bgr24(self):
        assert frame_shape(640, 480, "bgr24") == (480, 640, 3)

    def test_yuv420p(self):
        assert frame_shape(640, 480, "yuv420p") == (640 * 480 * 3 // 2,)

    def test_yuv420p_odd_size_rounds_chroma_up(self):
        assert frame_shape(641, 481, "yuv420p") == (641 * 481 + 2 * 321 * 241,)

    def test_unknown_pix_fmt(self):
        with pytest.raises(ValueError, match="Unsupported pix_fmt"):
            frame_shape(640, 480, "nv12")


class _ChunkedStdout(BytesIO):
    """A pipe that hands out at most ``chunk`` bytes per readinto() call."""

    def __init__(self, data: bytes, chunk: int):
        super().__init__(data)
        self._chunk = chunk

    def readinto(self, b):
        view = memoryview(b)[:self._chunk]
        return super().readinto(view)


class TestGrowPipe:
    def test_real_pipe_grows_to_1_mib(self):
        f_getpipe = getattr(ffmpeg_pipe.fcntl, "F_GETPIPE_SZ", None)
        if f_getpipe is None:
            pytest.skip("F_GETPIPE_SZ is Linux-only")
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd, "rb") as stream:
                _grow_pipe(stream)
                assert ffmpeg_pipe.fcntl.fcntl(stream.fileno(), f_getpipe) == 1 << 20
        finally:
            os.close(write_fd)

    def test_streams_without_a_pipe_are_ignored(self):
        _grow_pipe(BytesIO(b""))
        _grow_pipe(MagicMock())


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

    def test_yuv420p_output_and_flat_frames(self):
        size = 640 * 480 * 3 // 2
        mock_proc = self._make_mock_process([])
        mock_proc.stdout = BytesIO(bytes(range(256)) * (size // 256) + bytes(size % 256))
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc) as mock_popen:
            with FFmpegDecoder("input.mp4", 640, 480, config, pix_fmt="yuv420p") as decoder:
                assert decoder.frame_shape == (size,)
                assert decoder.frame_size == size
                frame = decoder.read_frame()
                assert decoder.read_frame() is None

        cmd = mock_popen.call_args[0][0]
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
        assert frame.shape == (size,)
        assert frame[:3].tolist() == [0, 1, 2]

    def test_read_into_fills_the_buffer_in_place(self):
        frames = [np.full((4, 6, 3), i, dtype=np.uint8) for i in (7, 9)]
        mock_proc = self._make_mock_process(frames)
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        buf = np.zeros((4, 6, 3), dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                assert decoder.read_into(buf) is True
                assert (buf == 7).all()
                assert decoder.read_into(buf) is True
                assert (buf == 9).all()
                assert decoder.read_into(buf) is False

    def test_read_into_retries_short_reads(self):
        frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        mock_proc = self._make_mock_process([])
        mock_proc.stdout = _ChunkedStdout(frame.tobytes(), chunk=5)
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        buf = np.zeros_like(frame)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                assert decoder.read_into(buf) is True
        np.testing.assert_array_equal(buf, frame)

    def test_read_into_rejects_a_wrong_size_buffer(self):
        mock_proc = self._make_mock_process([])
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                with pytest.raises(ValueError, match="a frame needs 72"):
                    decoder.read_into(np.zeros(10, dtype=np.uint8))

    def test_read_into_raises_when_ffmpeg_crashed(self):
        mock_proc = self._make_mock_process([])
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                with pytest.raises(RuntimeError, match="FFmpeg decoder crashed"):
                    decoder.read_into(np.zeros((4, 6, 3), dtype=np.uint8))

    def test_abort_kills_ffmpeg_and_close_stays_quiet(self, caplog):
        mock_proc = self._make_mock_process([])
        mock_proc.returncode = -9
        config = HWAccelConfig(accel_type=HWAccelType.CPU)

        with patch("ffmpeg_pipe.subprocess.Popen", return_value=mock_proc):
            with caplog.at_level(logging.WARNING, logger="ffmpeg_pipe"):
                with FFmpegDecoder("input.mp4", 6, 4, config) as decoder:
                    decoder.abort()

        mock_proc.kill.assert_called_once()
        assert not any("exited with code" in r.message for r in caplog.records)


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
        mock_proc.stdin.write.assert_called_once()
        written = mock_proc.stdin.write.call_args.args[0]
        # A memoryview of the frame itself: no tobytes() copy of every frame
        assert isinstance(written, memoryview)
        assert written.tobytes() == frame.tobytes()

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

    def test_close_after_a_clean_early_exit_drops_the_unwritten_tail(self, caplog):
        """A real pipe to a child that exits rc=0 without reading, as ffmpeg does after -shortest.

        Frames smaller than the stdin buffer (4096 bytes) wait in it until
        flush(). The child exits while flush() is blocked on the full pipe,
        so the last frame stays buffered and close() flushes it into the
        closed pipe again: that BrokenPipeError must not escape a clean exit.
        """
        caplog.set_level(logging.DEBUG, logger="ffmpeg_pipe")
        real_popen = subprocess.Popen

        def child_that_never_reads(cmd, **kwargs):
            return real_popen([sys.executable, "-c", "import time; time.sleep(0.5)"], **kwargs)

        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        frame = np.zeros(1024, dtype=np.uint8)

        with patch("ffmpeg_pipe.subprocess.Popen", side_effect=child_that_never_reads):
            with FFmpegEncoder("input.mp4", "output.mp4", 320, 240, 10.0, config, "h264", crf=18) as encoder:
                for _ in range(10_000):  # a 1 MiB pipe is full after 1024 frames
                    if not encoder.write_frame(frame):
                        break
                else:
                    pytest.fail("write_frame never reported the early exit")

        assert "clean exit after BrokenPipe" in caplog.text  # the pipe broke inside flush()

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

    def test_yuv420p_input_format(self):
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
                pix_fmt="yuv420p",
            ) as encoder:
                encoder.write_frame(np.zeros(640 * 480 * 3 // 2, dtype=np.uint8))

        cmd = mock_popen.call_args[0][0]
        pipe_input = cmd.index("pipe:0")
        # -pix_fmt before -i pipe:0 describes the raw input
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
        assert cmd.index("-pix_fmt") < pipe_input

    def test_unknown_pix_fmt_is_rejected_before_ffmpeg_starts(self):
        config = HWAccelConfig(accel_type=HWAccelType.CPU)
        with patch("ffmpeg_pipe.subprocess.Popen") as mock_popen:
            with pytest.raises(ValueError, match="Unsupported pix_fmt"):
                FFmpegEncoder("input.mp4", "output.mp4", 640, 480, 30.0, config, "h264", pix_fmt="rgb48")
        mock_popen.assert_not_called()
