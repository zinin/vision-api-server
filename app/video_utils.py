import subprocess
import tempfile
import os
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video metadata."""
    duration: float
    width: int
    height: int
    fps: float
    codec: str


@dataclass
class ExtractedFrame:
    """Extracted frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int


class VideoFrameExtractor:
    """
    Extract key frames from video using ffmpeg with smart scene detection.
    """

    def __init__(
            self,
            scene_threshold: float = 0.05,
            min_interval: float = 1.0,
            ffmpeg_path: str = "ffmpeg",
            ffprobe_path: str = "ffprobe"
    ):
        """
        Initialize extractor.

        Args:
            scene_threshold: Scene change detection threshold (0.01-0.5, lower = more sensitive)
            min_interval: Minimum interval between frames in seconds
            ffmpeg_path: Path to ffmpeg executable
            ffprobe_path: Path to ffprobe executable
        """
        self.scene_threshold = scene_threshold
        self.min_interval = min_interval
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

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
        """
        Get video metadata using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            VideoInfo with duration, dimensions, fps, codec
        """
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name,duration",
            "-show_entries", "format=duration",
            "-of", "csv=p=0:s=,",
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed for video: {video_path}")
            logger.error(f"ffprobe stderr: {e.stderr}")
            logger.error(f"ffprobe stdout: {e.stdout}")
            raise RuntimeError(
                f"Failed to get video info: ffprobe returned exit code {e.returncode}. "
                f"stderr: {e.stderr.strip()}"
            ) from e

        output = result.stdout.strip()

        # Parse output - format varies, handle both cases
        lines = [l.strip() for l in output.split('\n') if l.strip()]

        # Try to extract values
        width = height = 0
        fps = 30.0
        duration = 0.0
        codec = "unknown"

        for line in lines:
            parts = line.split(',')
            for part in parts:
                part = part.strip()
                # Check for duration (float value)
                try:
                    val = float(part)
                    if val > 100:  # Likely width/height
                        if width == 0:
                            width = int(val)
                        elif height == 0:
                            height = int(val)
                    else:
                        duration = val
                except ValueError:
                    # Check for fps fraction
                    if '/' in part:
                        try:
                            num, den = part.split('/')
                            fps = float(num) / float(den)
                        except:
                            pass
                    elif part.isalpha():
                        codec = part

        # Alternative parsing using dedicated ffprobe calls
        if duration == 0:
            duration = self._get_duration(video_path)
        if width == 0 or height == 0:
            width, height = self._get_dimensions(video_path)

        return VideoInfo(
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            codec=codec
        )

    def _get_duration(self, video_path: str) -> float:
        """Get video duration."""
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"ffprobe _get_duration failed for {video_path}: {result.stderr}")
            return 0.0
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    def _get_dimensions(self, video_path: str) -> tuple[int, int]:
        """Get video dimensions."""
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"ffprobe _get_dimensions failed for {video_path}: {result.stderr}")
            return 0, 0
        try:
            w, h = result.stdout.strip().split('x')
            return int(w), int(h)
        except ValueError:
            return 0, 0

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