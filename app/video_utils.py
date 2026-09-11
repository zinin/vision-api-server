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
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ffmpeg_pipe import _rc_to_str
from frame_selection import (
    SCAN_WIDTH,
    STORM_MEDIAN_BLOB,
    SelectedFrame,
    SelectionParams,
    blob_area,
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

_PTS_RE = re.compile(r"pts_time:\s*(-?[0-9.]+)")
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
    image: np.ndarray
    timestamp: float
    frame_number: int


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
    ``select_frames``. Pass 2 (``extract_frames``) decodes again and fetches
    only the selected frames in full resolution.
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
                check=True
            )
            subprocess.run(
                [self.ffprobe_path, "-version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
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
        if info.width == 0 or info.height == 0:
            raise ValueError(NO_VIDEO_STREAM_MESSAGE)
        logger.info(
            f"Video: duration={info.duration:.2f}s, resolution={info.width}x{info.height}, "
            f"fps={info.fps:.2f}, codec={info.codec}"
        )

        started = time.monotonic()
        pts, blob = self._scan_motion(video_path, info, deadline)
        pass1_seconds = time.monotonic() - started

        if info.duration <= 0.0:
            logger.warning(f"ffprobe gave no duration; using the last frame pts {pts[-1]:.3f}s")
            info.duration = pts[-1]
        if any(later < earlier for earlier, later in zip(pts, pts[1:])):
            logger.warning("Frame pts are not monotonic; selection uses them as reported")

        selected = select_frames(pts, blob, params)
        median_blob_value = median_blob(blob)
        stats = SelectionStats(
            total_frames=len(pts),
            median_blob=median_blob_value,
            storm=median_blob_value > STORM_MEDIAN_BLOB,
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
        frame_size = SCAN_WIDTH * scaled_h
        cmd = [
            self.ffmpeg_path, "-hide_banner", "-nostats", "-loglevel", "info",
            "-an", "-i", video_path,
            "-vf", f"scale={SCAN_WIDTH}:{scaled_h},showinfo",
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ]
        logger.debug(f"Motion scan command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            returncode = _finish_process(process)
            killer.cancel()
            collector.join()

        if killer.expired:
            raise RuntimeError(
                f"Frame extraction timed out after {self.timeout:.0f}s during the motion scan"
            )
        if not blob:
            raise RuntimeError(
                f"FFmpeg produced no frames ({_rc_to_str(returncode)}): {collector.tail_text()}"
            )
        if returncode != 0:
            logger.warning(
                f"FFmpeg motion scan exited with {_rc_to_str(returncode)} after {len(blob)} frames: "
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

    def extract_frames(
            self,
            video_path: str,
            output_dir: Optional[str] = None,
            max_frames: int = 50
    ) -> list[ExtractedFrame]:
        """
        Extract key frames from video using smart scene detection.

        Algorithm:
        1. Always extract first frame
        2. Extract frames on scene change (if min_interval passed)
        3. Extract middle frame if only first frame was selected

        Args:
            video_path: Path to video file
            output_dir: Directory for temporary frames (uses tempdir if None)
            max_frames: Maximum number of frames to extract

        Returns:
            List of ExtractedFrame objects
        """
        video_info = self.get_video_info(video_path)
        mid_time = video_info.duration / 2

        logger.info(
            f"Video: duration={video_info.duration:.2f}s, "
            f"resolution={video_info.width}x{video_info.height}, "
            f"mid_time={mid_time:.2f}s"
        )

        if video_info.width == 0 or video_info.height == 0:
            raise ValueError(NO_VIDEO_STREAM_MESSAGE)

        # Create temp directory if not provided
        cleanup_dir = output_dir is None
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="yolo_video_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Build ffmpeg filter for smart frame selection
            # Conditions:
            # 1. eq(n,0) - first frame (always)
            # 2. gt(scene,T)*gte(t-prev_selected_t,I) - scene change + min interval
            # 3. gte(t,MID)*eq(prev_selected_n,0) - middle if only 1st selected

            select_filter = (
                f"select='"
                f"eq(n\\,0)+"
                f"(gt(scene\\,{self.scene_threshold})*gte(t-prev_selected_t\\,{self.min_interval}))+"
                f"(gte(t\\,{mid_time})*lte(prev_selected_n\\,1))"
                f"'"
            )

            output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

            cmd = [
                self.ffmpeg_path,
                "-hide_banner",
                "-loglevel", "info",
                "-i", video_path,
                "-vf", select_filter,
                "-fps_mode", "vfr",
                "-q:v", "2",
                output_pattern
            ]

            logger.debug(f"FFmpeg command: {' '.join(cmd)}")

            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout
            )

            if result.returncode != 0:
                logger.warning(f"FFmpeg stderr: {result.stderr}")
                # Try alternative method if smart select fails
                return self._extract_frames_fallback(
                    video_path, output_dir, video_info, max_frames
                )

            # Parse extracted timestamps from ffmpeg output
            timestamps = self._parse_ffmpeg_timestamps(result.stderr)

            # Load extracted frames
            frames = self._load_frames(output_dir, timestamps, max_frames)

            logger.info(f"Extracted {len(frames)} frames from video")

            return frames

        finally:
            # Cleanup temp directory if we created it
            if cleanup_dir and os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)

    def _extract_frames_fallback(
            self,
            video_path: str,
            output_dir: str,
            video_info: VideoInfo,
            max_frames: int
    ) -> list[ExtractedFrame]:
        """
        Fallback frame extraction using fixed intervals.
        Used when scene detection fails.
        """
        logger.info("Using fallback interval-based extraction")

        # Calculate interval to get reasonable number of frames
        target_frames = min(max_frames, 10)
        interval = max(1.0, video_info.duration / target_frames)

        output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "warning",
            "-i", video_path,
            "-vf", f"fps=1/{interval}",
            "-q:v", "2",
            output_pattern
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"ffmpeg fallback extraction failed for {video_path}")
            logger.error(f"ffmpeg stderr: {result.stderr}")
            raise RuntimeError(
                f"Failed to extract frames (fallback method): "
                f"ffmpeg returned exit code {result.returncode}. "
                f"stderr: {result.stderr.strip()}"
            )

        # Generate timestamps based on interval
        timestamps = {}
        for i in range(target_frames):
            timestamps[i + 1] = i * interval

        return self._load_frames(output_dir, timestamps, max_frames)

    def _parse_ffmpeg_timestamps(self, stderr: str) -> dict[int, float]:
        """
        Parse frame timestamps from ffmpeg showinfo output.

        Returns:
            Dict mapping frame index (1-based) to timestamp
        """
        timestamps = {}

        # Pattern for pts_time from showinfo filter
        pts_pattern = re.compile(r'pts_time:(\d+\.?\d*)')

        # Also try to find frame numbers
        frame_pattern = re.compile(r'n:\s*(\d+)')

        frame_idx = 0
        for line in stderr.split('\n'):
            pts_match = pts_pattern.search(line)
            if pts_match:
                frame_idx += 1
                timestamps[frame_idx] = float(pts_match.group(1))

        return timestamps

    def _load_frames(
            self,
            output_dir: str,
            timestamps: dict[int, float],
            max_frames: int
    ) -> list[ExtractedFrame]:
        """Load extracted frame images from directory."""
        frames = []

        # Find all frame files
        frame_files = sorted(Path(output_dir).glob("frame_*.jpg"))

        for idx, frame_path in enumerate(frame_files[:max_frames], start=1):
            image = cv2.imread(str(frame_path))

            if image is None:
                logger.warning(f"Failed to load frame: {frame_path}")
                continue

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            timestamp = timestamps.get(idx, idx - 1)  # fallback to index

            frames.append(ExtractedFrame(
                image=image,
                timestamp=timestamp,
                frame_number=idx
            ))

        return frames


async def extract_frames_from_video(
        video_data: bytes,
        scene_threshold: float = 0.05,
        min_interval: float = 1.0,
        max_frames: int = 50
) -> list[ExtractedFrame]:
    """
    Async wrapper for video frame extraction.

    Args:
        video_data: Video file bytes
        scene_threshold: Scene detection sensitivity
        min_interval: Minimum seconds between frames
        max_frames: Maximum frames to extract

    Returns:
        List of extracted frames
    """
    import asyncio

    def _extract():
        # Write video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_data)
            tmp_path = tmp.name

        try:
            extractor = VideoFrameExtractor(
                scene_threshold=scene_threshold,
                min_interval=min_interval
            )
            return extractor.extract_frames(tmp_path, max_frames=max_frames)
        finally:
            os.unlink(tmp_path)

    # Run in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract)