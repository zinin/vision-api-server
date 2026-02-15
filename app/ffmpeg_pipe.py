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
    return b"".join(tail).decode("utf-8", errors="replace")[:500]


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
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        if self._process.returncode and self._process.returncode != 0:
            logger.warning(f"FFmpeg decoder exited with code {self._process.returncode}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
