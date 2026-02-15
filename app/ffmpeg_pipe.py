import logging
import subprocess
import threading
from collections import deque
from pathlib import Path

import numpy as np

from hw_accel import HWAccelConfig

logger = logging.getLogger(__name__)


def _drain_stderr(process: subprocess.Popen, collected: deque[bytes]) -> None:
    """Daemon thread target: read stderr line-by-line to prevent pipe buffer deadlock.
    Collects bytes (Popen uses text=False). Thread-safe via deque + GIL."""
    try:
        for line in process.stderr:
            collected.append(line)
    except (ValueError, OSError):
        pass  # pipe closed


def _format_stderr(lines: deque[bytes], max_lines: int = 10) -> str:
    """Decode last N stderr lines for error messages."""
    tail = list(lines)[-max_lines:]
    return b"".join(tail).decode("utf-8", errors="replace")[:2000]


class FFmpegDecoder:
    """Decode video frames via FFmpeg subprocess pipe.

    Usage:
        with FFmpegDecoder(path, w, h, config) as decoder:
            while (frame := decoder.read_frame()) is not None:
                process(frame)
    """

    def __init__(
        self,
        input_path: str | Path,
        width: int,
        height: int,
        hw_config: HWAccelConfig,
    ):
        self._input_path = str(input_path)
        self._width = width
        self._height = height
        self._frame_size = width * height * 3  # BGR24
        self._stderr_lines: deque[bytes] = deque(maxlen=100)

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
        cmd += hw_config.decode_args
        cmd += ["-i", str(input_path), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

        logger.debug(f"FFmpegDecoder command: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
        # Start daemon thread to drain stderr and prevent deadlock
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def read_frame(self) -> np.ndarray | None:
        """Read one BGR24 frame (writable copy). Returns None on EOF."""
        raw = self._process.stdout.read(self._frame_size)
        if not raw or len(raw) < self._frame_size:
            # Check if process crashed (vs normal EOF)
            if self._process.poll() is not None and self._process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg decoder crashed (rc={self._process.returncode}): "
                    f"{_format_stderr(self._stderr_lines)}"
                )
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        ).copy()  # .copy() makes array writable for OpenCV drawing

    def close(self) -> None:
        if self._process.stdout:
            self._process.stdout.close()
        self._stderr_thread.join(timeout=5)
        if self._process.stderr:
            self._process.stderr.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        stderr_output = _format_stderr(self._stderr_lines, max_lines=50)
        if stderr_output:
            logger.debug(f"FFmpeg decoder stderr:\n{stderr_output}")
        if self._process.returncode and self._process.returncode != 0:
            logger.warning(f"FFmpeg decoder exited with code {self._process.returncode}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FFmpegEncoder:
    """Encode raw BGR24 frames via FFmpeg subprocess pipe with audio merge.

    Usage:
        with FFmpegEncoder(original, output, w, h, fps, config, codec, crf) as enc:
            for frame in frames:
                enc.write_frame(frame)
    """

    def __init__(
        self,
        original_path: str | Path,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        hw_config: HWAccelConfig,
        codec: str,
        crf: int,
    ):
        self._stderr_lines: deque[bytes] = deque(maxlen=100)

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
        cmd += hw_config.global_encode_args  # e.g. [-vaapi_device, ...] — MUST be before -i
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
            "-i", str(original_path),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-map_metadata", "1",
        ]
        cmd += hw_config.get_encode_args(codec, crf)  # codec-specific args (after inputs)
        cmd += ["-c:a", "aac", "-shortest", str(output_path)]

        logger.debug(f"FFmpegEncoder command: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
        # Start daemon thread to drain stderr and prevent deadlock
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one BGR24 frame to the encoder. Raises RuntimeError if process crashed."""
        rc = self._process.poll()
        if rc is not None:
            raise RuntimeError(
                f"FFmpeg encoder crashed (rc={rc}): "
                f"{_format_stderr(self._stderr_lines)}"
            )
        try:
            self._process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(
                f"FFmpeg encoder pipe broken: {e}. "
                f"stderr: {_format_stderr(self._stderr_lines)}"
            ) from e

    def close(self) -> None:
        """Close stdin, wait for FFmpeg to finish, check return code."""
        if self._process.stdin:
            self._process.stdin.close()
        self._stderr_thread.join(timeout=10)
        if self._process.stderr:
            self._process.stderr.close()
        try:
            self._process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        stderr_output = _format_stderr(self._stderr_lines, max_lines=50)
        if stderr_output:
            logger.debug(f"FFmpeg encoder stderr:\n{stderr_output}")
        if self._process.returncode and self._process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoder failed (rc={self._process.returncode}): "
                f"{stderr_output}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # An exception is already propagating (e.g. from write_frame crash).
            # Still clean up, but don't raise another error from close().
            try:
                self.close()
            except RuntimeError as close_err:
                logger.warning(f"Suppressed encoder close error (original exception propagating): {close_err}")
        else:
            self.close()
