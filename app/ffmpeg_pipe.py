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


def _rc_to_str(rc: int | None) -> str:
    """Render a subprocess return code. Negative rc means killed by signal
    |rc| (POSIX convention). Distinguishes OOM-kills and external SIGKILL
    from self-reported exit codes when these end up in error messages."""
    if rc is None:
        return "rc=?"
    if rc < 0:
        return f"killed by signal {-rc}"
    return f"rc={rc}"


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
                    f"FFmpeg decoder crashed ({_rc_to_str(self._process.returncode)}): "
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
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # SIGKILL ignored (D-state) — nothing more we can do
        stderr_output = _format_stderr(self._stderr_lines, max_lines=50)
        if stderr_output:
            logger.debug(f"FFmpeg decoder stderr:\n{stderr_output}")
        if self._process.returncode is None:
            # SIGKILL did not reap the process (D-state / hung NFS / GPU
            # driver wedge). Surface as a WARNING so operators know the
            # teardown did not finish — otherwise a stuck decoder leaks
            # silently in both pass 1 and pass 2 cleanup paths.
            logger.warning(
                "FFmpeg decoder did not exit after SIGKILL; process may be leaked"
            )
        elif self._process.returncode != 0:
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
        crf: int | None = None,
        bitrate: int | None = None,
    ):
        self._stderr_lines: deque[bytes] = deque(maxlen=100)
        # True after the encoder cleanly exits (rc=0) while we still had
        # frames to write — e.g. FFmpeg's -shortest closes pipe:0 when the
        # audio stream ends before the piped raw video. Subsequent
        # write_frame() calls become silent no-ops. Single-writer invariant:
        # callers must serialise write_frame() from one thread (the Pass 2
        # loop in VideoAnnotator is single-threaded by design).
        self._eof = False

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
        cmd += hw_config.get_encode_args(codec, crf=crf, bitrate=bitrate)
        cmd += ["-c:a", "aac", "-shortest", str(output_path)]

        logger.debug(f"FFmpegEncoder command: {' '.join(cmd)}")
        # bufsize=0 — every write goes straight to the OS pipe. This avoids a
        # spurious BrokenPipeError from close() when ffmpeg finalised early
        # (e.g. -shortest) and residual bytes in a Python-side buffer would
        # otherwise be flushed into a pipe the peer has already closed.
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
            bufsize=0,
        )
        # Start daemon thread to drain stderr and prevent deadlock
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write one BGR24 frame to the encoder.

        Returns True when the frame was written and the caller should keep
        going, False once the encoder has finalised (rc == 0) — callers
        should break their loop in that case to avoid wasting CPU on
        frames ffmpeg will never consume.

        Raises RuntimeError if the process crashed (rc != 0). A clean
        early exit (rc == 0) is treated as EOF: the frame is silently
        dropped and further calls short-circuit to False. This covers
        FFmpeg's -shortest behaviour: when the audio stream ends before
        the piped raw video, ffmpeg closes pipe:0 from its side, the
        output file is already fully written, and there's nothing left
        for Python to do.
        """
        if self._eof:
            return False
        rc = self._process.poll()
        if rc is not None:
            if rc == 0:
                self._eof = True
                logger.debug(
                    "FFmpegEncoder: clean exit before write (rc=0) — EOF reached"
                )
                return False
            raise RuntimeError(
                f"FFmpeg encoder crashed ({_rc_to_str(rc)}): "
                f"{_format_stderr(self._stderr_lines)}"
            )
        try:
            self._process.stdin.write(frame.tobytes())
        except OSError as e:  # BrokenPipeError is a subclass of OSError.
            # The pipe closed mid-write. Most often this means the
            # encoder just finalised the output (e.g. -shortest on an
            # audio stream shorter than the video pipe). Give it a
            # moment to reap, then distinguish clean exit vs crash.
            try:
                rc = self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                # SIGKILL is usually reaped within milliseconds, but in rare
                # pathologies (D-state on hung NFS, GPU driver wedge) the
                # process may linger. Swallow a second timeout so we still
                # raise the intended RuntimeError instead of leaking
                # TimeoutExpired up the stack.
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                raise RuntimeError(
                    f"FFmpeg encoder hung after pipe break: {e}. "
                    f"stderr: {_format_stderr(self._stderr_lines)}"
                ) from e
            if rc == 0:
                self._eof = True
                logger.debug(
                    "FFmpegEncoder: clean exit after BrokenPipe (rc=0) — "
                    "-shortest finalised early"
                )
                return False
            raise RuntimeError(
                f"FFmpeg encoder crashed mid-write ({_rc_to_str(rc)}): {e}. "
                f"stderr: {_format_stderr(self._stderr_lines)}"
            ) from e
        return True

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
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # SIGKILL ignored (D-state) — nothing more we can do
        stderr_output = _format_stderr(self._stderr_lines, max_lines=50)
        if stderr_output:
            logger.debug(f"FFmpeg encoder stderr:\n{stderr_output}")
        if self._process.returncode is None:
            # SIGKILL did not reap the process. Fail loudly so _pass2_render
            # cannot report success while the output is unfinished and a
            # stuck encoder leaks. __exit__ suppresses this RuntimeError
            # when another exception is already propagating.
            raise RuntimeError(
                f"FFmpeg encoder did not exit after SIGKILL; process may be leaked. "
                f"stderr: {stderr_output}"
            )
        if self._process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoder failed ({_rc_to_str(self._process.returncode)}): "
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
