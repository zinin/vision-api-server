import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from ffmpeg_pipe import rc_to_str
from frame_selection import (
    PTS_TOLERANCE,
    SCAN_WIDTH,
    Reason,
    SelectedFrame,
    SelectionParams,
    blob_area,
    is_storm,
    median_blob,
    prepare_frame,
    select_frames,
)

logger = logging.getLogger(__name__)

NO_VIDEO_STREAM_MESSAGE = (
    "File contains no video stream. Only audio or metadata streams were found."
)
UNREADABLE_VIDEO_MESSAGE = (
    "File could not be read as a valid video. "
    "The file may be corrupted or not a supported video format."
)

MAX_SCAN_HEIGHT = 8 * SCAN_WIDTH  # gray scan frames taller than 1:8 are rejected before ffmpeg starts

_PTS_RE = re.compile(r"pts_time:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_SIZE_RE = re.compile(r"\bs:(\d+)x(\d+)")


@dataclass
class VideoInfo:
    """Video metadata from ffprobe. ``width`` and ``height`` are the display
    dimensions after the rotation metadata is applied, which is what ffmpeg
    outputs because it autorotates by default."""
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    rotation: int = 0


@dataclass
class ExtractedFrame:
    """Extracted frame with metadata."""
    image: np.ndarray           # BGR uint8, H×W×3
    timestamp: float
    frame_number: int
    reason: Reason


@dataclass(frozen=True)
class ShowinfoFrame:
    """One frame line of ffmpeg's ``showinfo`` filter."""
    pts: float
    width: int
    height: int


def parse_showinfo_line(line: str) -> ShowinfoFrame | None:
    """Parse one ffmpeg stderr line; None unless it is a showinfo frame line."""
    if "showinfo" not in line:
        return None
    pts = _PTS_RE.search(line)
    size = _SIZE_RE.search(line)
    if pts is None or size is None:
        return None
    return ShowinfoFrame(pts=float(pts.group(1)), width=int(size.group(1)), height=int(size.group(2)))


def parse_showinfo(stderr: str) -> list[ShowinfoFrame]:
    """All showinfo frame lines of an ffmpeg stderr dump, in order."""
    frames = []
    for line in stderr.splitlines():
        parsed = parse_showinfo_line(line)
        if parsed is not None:
            frames.append(parsed)
    return frames


@dataclass
class SelectionStats:
    """What the selection did, for the INFO log line and the corpus check."""
    total_frames: int
    median_blob: float
    storm: bool
    counts: dict[str, int]      # selected frames per reason
    pass1_seconds: float
    pass2_seconds: float = 0.0


@dataclass
class ScanResult:
    """Pass 1 output: the selection and everything needed to fetch the frames."""
    selected: list[SelectedFrame]
    pts: list[float]            # presentation time of every decoded frame
    blob: list[float]           # metric of every decoded frame; blob[0] == 0.0
    info: VideoInfo
    stats: SelectionStats


@dataclass
class ExtractionResult:
    """Both passes done: the selected frames in full resolution plus metadata."""
    frames: list[ExtractedFrame]
    info: VideoInfo
    stats: SelectionStats


class _StderrCollector:
    """Daemon thread draining an ffmpeg stderr pipe so it never blocks.
    Keeps every showinfo frame and the last non-showinfo lines for error messages."""

    def __init__(self, stream, keep_lines: int = 50):
        self.frames: list[ShowinfoFrame] = []
        self.tail: deque[str] = deque(maxlen=keep_lines)
        self._stream = stream
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for raw in self._stream:
                line = raw.decode("utf-8", errors="replace").rstrip()
                parsed = parse_showinfo_line(line)
                if parsed is not None:
                    self.frames.append(parsed)
                else:
                    self.tail.append(line)
        except (ValueError, OSError):
            pass  # pipe closed

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout)

    def tail_text(self) -> str:
        return "\n".join(self.tail)[-2000:]


class _Deadline:
    """Kills a subprocess when the wall-clock deadline passes, even if the
    main thread is blocked in a pipe read."""

    def __init__(self, process: subprocess.Popen, deadline: float):
        self.expired = False
        self._process = process
        self._timer = threading.Timer(max(0.0, deadline - time.monotonic()), self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.expired = True
        try:
            self._process.kill()
        except OSError:
            pass  # already gone

    def cancel(self) -> None:
        self._timer.cancel()


def _finish_process(process: subprocess.Popen, wait_seconds: float = 10.0) -> int | None:
    """Wait for ffmpeg to exit; escalate to SIGKILL. None means it never exited."""
    try:
        return process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            return process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg did not exit after SIGKILL; process may be leaked")
            return None


def _to_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_fps(value) -> float:
    """ffprobe frame rates are fractions like ``12500/1000``; ``0/0`` means unknown."""
    if value is None:
        return 0.0
    text = str(value)
    if "/" not in text:
        return _to_float(text)
    num, _, den = text.partition("/")
    den_f = _to_float(den)
    return _to_float(num) / den_f if den_f > 0 else 0.0


def _parse_rotation(stream: dict) -> int:
    """Rotation in degrees from the Display Matrix side data or the legacy ``rotate`` tag.
    Only the magnitude matters: the value decides whether width and height swap."""
    for side_data in stream.get("side_data_list") or []:
        if isinstance(side_data, dict) and "rotation" in side_data:
            return abs(int(round(_to_float(side_data["rotation"])))) % 360
    tag = (stream.get("tags") or {}).get("rotate")
    if tag is not None:
        return abs(int(round(_to_float(tag)))) % 360
    return 0


def parse_probe_output(data: dict) -> VideoInfo:
    """Build VideoInfo from ``ffprobe -print_format json -show_streams -show_format`` output."""
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(NO_VIDEO_STREAM_MESSAGE)
    stream = streams[0]
    width = _to_int(stream.get("width"))
    height = _to_int(stream.get("height"))
    if width <= 0 or height <= 0:
        raise ValueError(NO_VIDEO_STREAM_MESSAGE)
    rotation = _parse_rotation(stream)
    if rotation in (90, 270):
        width, height = height, width
    fmt = data.get("format") or {}
    duration = _to_float(fmt.get("duration")) or _to_float(stream.get("duration")) or 0.0
    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate")) or 0.0
    codec = stream.get("codec_name") or "unknown"
    return VideoInfo(duration=duration, width=width, height=height, fps=fps, codec=codec, rotation=rotation)


class VideoFrameExtractor:
    """Motion-based key frame extraction with ffmpeg.

    Pass 1 (``scan``) decodes the whole video into gray 640 px frames through a
    pipe, computes the ``blob`` motion metric per frame and applies
    ``select_frames``. Pass 2 (``extract_frames``) decodes again and streams out
    the selected frames in full resolution, one at a time into preallocated
    arrays: ``select`` drops the other frames after decoding, so everything up
    to the last selected frame is decoded, and the peak memory is one frame per
    selected frame.
    """

    PROBE_TIMEOUT = 30.0

    def __init__(
            self,
            ffmpeg_path: str = "ffmpeg",
            ffprobe_path: str = "ffprobe",
            timeout: float = 300.0,
    ):
        """
        Args:
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
            timeout: Wall-clock deadline for one extraction (both passes), seconds
        """
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.timeout = timeout

        self._verify_ffmpeg()

    def _verify_ffmpeg(self) -> None:
        """Verify ffmpeg and ffprobe are available."""
        try:
            subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                check=True,
                timeout=self.PROBE_TIMEOUT
            )
            subprocess.run(
                [self.ffprobe_path, "-version"],
                capture_output=True,
                check=True,
                timeout=self.PROBE_TIMEOUT
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(
                f"ffmpeg/ffprobe not found or not working. "
                f"Please install ffmpeg: apt install ffmpeg"
            ) from e

    def get_video_info(self, video_path: str) -> VideoInfo:
        """Video metadata via ffprobe JSON.

        Raises ValueError when the file is unreadable or has no video stream
        (the HTTP layer maps it to 422) and RuntimeError when ffprobe hangs.
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            "-select_streams", "v:0",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.PROBE_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"ffprobe timed out after {self.PROBE_TIMEOUT:.0f}s") from e

        if result.returncode != 0:
            logger.warning(f"ffprobe failed for video: {video_path}")
            logger.warning(f"ffprobe stderr: {result.stderr}")
            raise ValueError(UNREADABLE_VIDEO_MESSAGE)

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise ValueError(UNREADABLE_VIDEO_MESSAGE) from e

        return parse_probe_output(data)

    def scan(self, video_path: str, params: SelectionParams) -> ScanResult:
        """Pass 1 only: decode, measure motion, select. No full-resolution frames."""
        return self._scan(video_path, params, deadline=time.monotonic() + self.timeout)

    def _scan(self, video_path: str, params: SelectionParams, deadline: float) -> ScanResult:
        info = self.get_video_info(video_path)
        logger.info(
            f"Video: duration={info.duration:.2f}s, resolution={info.width}x{info.height}, "
            f"fps={info.fps:.2f}, codec={info.codec}"
        )

        started = time.monotonic()
        pts, blob = self._scan_motion(video_path, info, deadline)
        pass1_seconds = time.monotonic() - started

        if info.duration <= 0.0:
            logger.warning(f"ffprobe gave no duration; using the last frame pts {max(pts):.3f}s")
            info.duration = max(pts)
        if any(later < earlier for earlier, later in zip(pts, pts[1:])):
            logger.warning("Frame pts are not monotonic; selection uses them as reported")

        selected = select_frames(pts, blob, params)
        median_blob_value = median_blob(blob)
        stats = SelectionStats(
            total_frames=len(pts),
            median_blob=median_blob_value,
            storm=is_storm(median_blob_value),
            counts=dict(Counter(f.reason for f in selected)),
            pass1_seconds=pass1_seconds,
        )
        return ScanResult(selected=selected, pts=pts, blob=blob, info=info, stats=stats)

    def _scan_motion(
            self, video_path: str, info: VideoInfo, deadline: float
    ) -> tuple[list[float], list[float]]:
        """Stream gray 640 px frames from ffmpeg and compute the blob metric per frame.

        Memory is O(1) in the number of frames: only the previous prepared
        frame and two lists of floats are kept.
        """
        scaled_h = int(round(info.height * SCAN_WIDTH / info.width / 2)) * 2
        if scaled_h < 2:
            raise ValueError(f"Video aspect ratio {info.width}x{info.height} scales to a zero-height frame")
        if scaled_h > MAX_SCAN_HEIGHT:
            raise ValueError(
                f"Video aspect ratio {info.width}x{info.height} scales to a {scaled_h} px high frame; "
                f"the limit is {MAX_SCAN_HEIGHT} px"
            )
        frame_size = SCAN_WIDTH * scaled_h
        cmd = [
            self.ffmpeg_path, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
            "-an", "-i", video_path,
            "-map", "0:v:0",
            "-vf", f"scale={SCAN_WIDTH}:{scaled_h},showinfo",
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        logger.debug(f"Motion scan command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        collector = _StderrCollector(process.stderr)
        killer = _Deadline(process, deadline)
        blob: list[float] = []
        prev = None
        try:
            while True:
                raw = process.stdout.read(frame_size)  # BufferedReader: short only at EOF
                if len(raw) < frame_size:
                    break
                cur = prepare_frame(np.frombuffer(raw, np.uint8).reshape(scaled_h, SCAN_WIDTH))
                blob.append(blob_area(prev, cur) if prev is not None else 0.0)
                prev = cur
        finally:
            process.stdout.close()
            killer.cancel()
            returncode = _finish_process(process)
            collector.join()
            process.stderr.close()

        if killer.expired:
            raise RuntimeError(
                f"Frame extraction timed out after {self.timeout:.0f}s during the motion scan"
            )
        if not blob:
            raise RuntimeError(
                f"FFmpeg produced no frames ({rc_to_str(returncode)}): {collector.tail_text()}"
            )
        if returncode != 0:
            logger.warning(
                f"FFmpeg motion scan exited with {rc_to_str(returncode)} after {len(blob)} frames: "
                f"{collector.tail_text()}"
            )

        pts = [f.pts for f in collector.frames]
        if len(pts) != len(blob):
            logger.warning(
                f"Motion scan: {len(pts)} showinfo frames vs {len(blob)} decoded frames; "
                f"truncating to the shorter"
            )
            n = min(len(pts), len(blob))
            if n == 0:
                raise RuntimeError("FFmpeg produced frames without showinfo timestamps")
            pts, blob = pts[:n], blob[:n]
        return pts, blob

    def extract_frames(self, video_path: str, params: SelectionParams) -> ExtractionResult:
        """Both passes: select frames by motion, then fetch them in full resolution (BGR)."""
        deadline = time.monotonic() + self.timeout
        scan = self._scan(video_path, params, deadline)

        started = time.monotonic()
        frames = self._grab_frames(video_path, scan.info, scan.selected, scan.pts, deadline)
        scan.stats.pass2_seconds = time.monotonic() - started

        counts = scan.stats.counts
        logger.info(
            f"Frame selection: {len(frames)} frames (first {counts.get('first', 0)}, "
            f"grid {counts.get('grid', 0)}, motion {counts.get('motion', 0)}) of {scan.stats.total_frames}, "
            f"storm={scan.stats.storm}, median_blob={scan.stats.median_blob:.4f}, "
            f"pass1={scan.stats.pass1_seconds:.2f}s, pass2={scan.stats.pass2_seconds:.2f}s, "
            f"last_pts={max(scan.pts):.2f}s"
        )
        return ExtractionResult(frames=frames, info=scan.info, stats=scan.stats)

    def _grab_frames(
            self,
            video_path: str,
            info: VideoInfo,
            selected: list[SelectedFrame],
            pts: list[float],
            deadline: float,
    ) -> list[ExtractedFrame]:
        """Pass 2: decode again, let ``select`` pass only the chosen frames, stream them as bgr24.

        Every frame is read straight into a preallocated array, so the peak
        memory is one frame per selected frame and nothing is copied. Output
        frames are matched to the selection by pts, never by position, and every
        mismatch raises: silently returning frames with somebody else's
        timestamps is worse than failing.
        """
        if not selected:
            raise RuntimeError("No frames selected")
        expr = "+".join(f"eq(n\\,{f.index})" for f in selected)
        cmd = [
            self.ffmpeg_path, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
            "-an", "-i", video_path,
            "-map", "0:v:0",
            "-vf", f"select='{expr}',showinfo",
            "-fps_mode", "passthrough",
            "-frames:v", str(len(selected)),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        logger.debug(f"Frame grab command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        collector = _StderrCollector(process.stderr)
        killer = _Deadline(process, deadline)
        frame_size = info.width * info.height * 3
        images: list[np.ndarray] = []
        partial = 0
        try:
            while True:
                buf = np.empty((info.height, info.width, 3), np.uint8)
                n = process.stdout.readinto(buf)  # BufferedReader: short only at EOF
                if n == 0:
                    break
                if n < frame_size:
                    partial = n
                    break
                images.append(buf)
                if len(images) > len(selected):
                    raise RuntimeError(
                        f"FFmpeg frame grab returned more than the {len(selected)} selected frames"
                    )
        finally:
            process.stdout.close()
            killer.cancel()
            returncode = _finish_process(process)
            collector.join()
            process.stderr.close()

        if killer.expired:
            raise RuntimeError(
                f"Frame extraction timed out after {self.timeout:.0f}s during the frame grab"
            )

        shown = collector.frames
        tail = collector.tail_text()
        rc = rc_to_str(returncode)

        if partial:
            raise RuntimeError(
                f"FFmpeg frame grab returned {len(images) * frame_size + partial} bytes, not a "
                f"multiple of the {info.width}x{info.height} frame size ({rc}): {tail}"
            )
        count = len(images)
        if count != len(selected) or len(shown) != count:
            raise RuntimeError(
                f"FFmpeg frame grab returned {count} frames and {len(shown)} showinfo lines, "
                f"expected {len(selected)} ({rc}): {tail}"
            )
        if returncode != 0:
            logger.warning(f"FFmpeg frame grab exited with {rc} but returned all frames: {tail}")

        frames: list[ExtractedFrame] = []
        used: set[int] = set()
        for k, frame_info in enumerate(shown):
            if (frame_info.width, frame_info.height) != (info.width, info.height):
                raise RuntimeError(
                    f"FFmpeg frame size {frame_info.width}x{frame_info.height} differs from the "
                    f"expected {info.width}x{info.height}"
                )
            match = next(
                (f for f in selected
                 if f.index not in used and abs(pts[f.index] - frame_info.pts) <= PTS_TOLERANCE),
                None,
            )
            if match is None:
                raise RuntimeError(
                    f"Frame with pts {frame_info.pts:.3f} does not match any selected frame"
                )
            used.add(match.index)
            frames.append(ExtractedFrame(
                image=images[k],
                timestamp=pts[match.index],
                frame_number=match.index,
                reason=match.reason,
            ))
        frames.sort(key=lambda f: f.frame_number)
        return frames


async def extract_frames_from_video(video_data: bytes, params: SelectionParams) -> ExtractionResult:
    """Async wrapper: write the upload to a temp file and run both passes in the default executor."""

    def _extract() -> ExtractionResult:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            with tmp:
                tmp.write(video_data)
            return VideoFrameExtractor().extract_frames(tmp.name, params)
        finally:
            os.unlink(tmp.name)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _extract)