import logging
import math
import subprocess
import threading
from collections import deque
from pathlib import Path

try:
    import fcntl
except ImportError:  # fcntl is Unix-only
    fcntl = None

import numpy as np

from hw_accel import HWAccelConfig

logger = logging.getLogger(__name__)

# 1 MiB is the default /proc/sys/fs/pipe-max-size, so any process may ask for it.
_PIPE_SIZE = 1 << 20


def frame_shape(width: int, height: int, pix_fmt: str) -> tuple[int, ...]:
    """numpy shape of one rawvideo frame: (h, w, 3) for bgr24, flat for yuv420p."""
    if pix_fmt == "bgr24":
        return (height, width, 3)
    if pix_fmt == "yuv420p":
        chroma = ((width + 1) // 2) * ((height + 1) // 2)
        return (width * height + 2 * chroma,)
    raise ValueError(f"Unsupported pix_fmt: {pix_fmt}")


def _grow_pipe(stream) -> None:
    """Best effort: enlarge a pipe's kernel buffer from 64 KiB to 1 MiB.

    A bigger buffer lets ffmpeg and Python hand a frame over in fewer, larger
    steps. Streams that are not pipes (test doubles, non-Linux) are skipped.
    """
    setpipe = getattr(fcntl, "F_SETPIPE_SZ", None)
    if setpipe is None:
        return
    try:
        fcntl.fcntl(stream.fileno(), setpipe, _PIPE_SIZE)
    except (OSError, ValueError, TypeError, AttributeError):
        pass


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


def rc_to_str(rc: int | None) -> str:
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

    ``pix_fmt`` is ``bgr24`` (frames shaped (h, w, 3)) or ``yuv420p`` (flat
    frames: the Y, U and V planes back to back). ``frame_shape`` and
    ``frame_size`` describe one frame.

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
        pix_fmt: str = "bgr24",
    ):
        self._input_path = str(input_path)
        self._width = width
        self._height = height
        self.frame_shape = frame_shape(width, height, pix_fmt)
        self.frame_size = math.prod(self.frame_shape)
        self._stderr_lines: deque[bytes] = deque(maxlen=100)
        self._aborted = False

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
        cmd += hw_config.decode_args
        cmd += ["-i", str(input_path), "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]

        logger.debug(f"FFmpegDecoder command: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
        )
        _grow_pipe(self._process.stdout)
        # Start daemon thread to drain stderr and prevent deadlock
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def read_into(self, buf: np.ndarray) -> bool:
        """Read the next frame straight into ``buf``. Returns False on EOF.

        ``buf`` must be a C-contiguous uint8 array of ``frame_size`` bytes.
        Filling it in place skips the bytes object and the copy that
        ``read_frame`` would make. Raises RuntimeError if ffmpeg crashed.
        """
        view = memoryview(buf).cast("B")
        if len(view) != self.frame_size:
            raise ValueError(
                f"Buffer holds {len(view)} bytes, a frame needs {self.frame_size}"
            )
        got = 0
        while got < self.frame_size:
            n = self._process.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        if got == self.frame_size:
            return True
        # Short read: normal EOF or a crash
        if self._process.poll() is not None and self._process.returncode != 0:
            raise RuntimeError(
                f"FFmpeg decoder crashed ({rc_to_str(self._process.returncode)}): "
                f"{_format_stderr(self._stderr_lines)}"
            )
        return False

    def read_frame(self) -> np.ndarray | None:
        """Read one frame into a new writable array. Returns None on EOF."""
        frame = np.empty(self.frame_shape, dtype=np.uint8)
        return frame if self.read_into(frame) else None

    def abort(self) -> None:
        """Kill ffmpeg so that a read blocked in another thread returns.

        ``close()`` cannot unblock such a read: closing the pipe waits for the
        buffered reader's lock, which the reading thread holds.
        """
        self._aborted = True
        self._process.kill()

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
        elif self._aborted:
            logger.debug(f"FFmpeg decoder aborted ({rc_to_str(self._process.returncode)})")
        elif self._process.returncode != 0:
            logger.warning(f"FFmpeg decoder exited with code {self._process.returncode}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FFmpegEncoder:
    """Encode raw frames via FFmpeg subprocess pipe with audio merge.

    ``pix_fmt`` names the layout of the frames written to the pipe:
    ``bgr24`` or ``yuv420p`` (see ``frame_shape``).

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
        pix_fmt: str = "bgr24",
    ):
        frame_shape(width, height, pix_fmt)  # rejects an unsupported pix_fmt early
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
            "-f", "rawvideo", "-pix_fmt", pix_fmt,
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
            "-i", str(original_path),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-map_metadata", "1",
        ]
        cmd += hw_config.get_encode_args(codec, crf=crf, bitrate=bitrate)
        cmd += ["-c:a", "aac", "-shortest", str(output_path)]

        logger.debug(f"FFmpegEncoder command: {' '.join(cmd)}")
        # Keep the default buffered stdin (bufsize=-1) so write() honours
        # the BufferedWriter "write all bytes" contract — raw unbuffered
        # mode can short-write under EINTR / backpressure and misalign
        # the rawvideo stream. write_frame() calls flush() after each
        # frame, which keeps the buffer empty between frames. Only a
        # flush that hits a pipe ffmpeg already closed (e.g. after
        # -shortest) leaves bytes behind; close() deals with those.
        self._process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
        )
        _grow_pipe(self._process.stdin)
        # Start daemon thread to drain stderr and prevent deadlock
        self._stderr_thread = threading.Thread(
            target=_drain_stderr, args=(self._process, self._stderr_lines), daemon=True
        )
        self._stderr_thread.start()

    def write_frame(self, frame: np.ndarray) -> bool:
        """Write one frame (laid out as ``pix_fmt``) to the encoder.

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
                f"FFmpeg encoder crashed ({rc_to_str(rc)}): "
                f"{_format_stderr(self._stderr_lines)}"
            )
        try:
            # A memoryview hands the array's own memory to write(): no copy.
            self._process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
            # Flush after every frame so the Python-side buffer holds no
            # residual bytes for close()'s implicit flush to push into a
            # pipe ffmpeg already closed (e.g. after -shortest). If this
            # flush itself hits the closed pipe, the unwritten tail stays
            # buffered; close() drops it after a clean exit.
            self._process.stdin.flush()
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
                f"FFmpeg encoder crashed mid-write ({rc_to_str(rc)}): {e}. "
                f"stderr: {_format_stderr(self._stderr_lines)}"
            ) from e
        return True

    def close(self) -> None:
        """Close stdin, wait for FFmpeg to finish, check return code."""
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                # After a clean early exit (-shortest) the flush in write_frame
                # can fail with the tail of a frame still buffered, and closing
                # flushes it into the closed pipe again. ffmpeg finished with
                # rc=0, so the tail is dropped; any other state still raises.
                if not self._eof:
                    raise
                logger.debug("FFmpegEncoder: dropped the unwritten tail of a frame after clean exit")
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
                f"FFmpeg encoder failed ({rc_to_str(self._process.returncode)}): "
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
