import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ffmpeg_pipe import FFmpegDecoder, FFmpegEncoder
from hw_accel import HWAccelConfig
from visualization import DetectionVisualizer, DetectionBox

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnnotationParams:
    conf: float = 0.5
    imgsz: int = 640
    max_det: int = 100
    detect_every: int = 5
    classes: list[str] | None = None
    line_width: int = 2
    show_labels: bool = True
    show_conf: bool = True


@dataclass(slots=True)
class AnnotationStats:
    """Video annotation processing statistics.

    Attributes:
        tracked_frames: Frames where detections were reused from last
            YOLO detection (hold mode). Previously: CSRT-tracked frames.
    """

    total_frames: int = 0
    detected_frames: int = 0
    tracked_frames: int = 0
    total_detections: int = 0
    processing_time_ms: int = 0


@dataclass(slots=True)
class VideoMetadata:
    """Video metadata from ffprobe."""
    fps: float
    width: int
    height: int
    total_frames: int
    codec_name: str | None = None
    bit_rate: int | None = None


# ffprobe codec_name → internal codec name
_CODEC_NAME_MAP: dict[str, str] = {
    "h264": "h264",
    "hevc": "h265",
    "av1": "av1",
}

_MIN_BITRATE = 100_000
_MAX_BITRATE = 200_000_000
_MAX_REASONABLE_FPS = 240.0


def _parse_frame_rate(raw: str | None) -> float | None:
    """Parse ffprobe frame rate string like '30/1' or '25740000/2052571'."""
    if not raw:
        return None
    try:
        num, den = map(int, raw.split("/"))
        if den == 0 or num <= 0:
            return None
        return num / den
    except (ValueError, ZeroDivisionError):
        return None


def _parse_fps(avg_frame_rate: str | None, r_frame_rate: str | None) -> float:
    """Pick best fps from ffprobe fields.

    Prefers avg_frame_rate (accurate for VFR and HEVC).
    Falls back to r_frame_rate, then 30.0.
    r_frame_rate can report timebase values (e.g. 90000/1) for HEVC streams.
    """
    avg = _parse_frame_rate(avg_frame_rate)
    r = _parse_frame_rate(r_frame_rate)

    # Prefer avg_frame_rate — it's computed from actual stream data
    if avg is not None and 0 < avg <= _MAX_REASONABLE_FPS:
        return avg
    # Fall back to r_frame_rate if reasonable
    if r is not None and 0 < r <= _MAX_REASONABLE_FPS:
        return r
    # Both unreasonable — last resort
    if avg is not None and avg > 0:
        logger.warning(f"Unusually high avg_frame_rate ({avg:.1f}), using it anyway")
        return avg
    return 30.0


class VideoAnnotator:
    """Annotate video with YOLO detections and hold mode."""

    def __init__(
        self,
        model: Any,
        visualizer: DetectionVisualizer,
        class_names: dict[int, str],
        hw_config: HWAccelConfig,
        codec: str = "h264",
        crf: int = 18,
    ):
        self.model = model
        self.visualizer = visualizer
        self.class_names = class_names
        self.hw_config = hw_config
        self.codec = codec
        self.crf = crf

    def annotate(
        self,
        input_path: Path,
        output_path: Path,
        params: AnnotationParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> AnnotationStats:
        """
        Annotate video with bounding boxes.

        This is a blocking method — run in a thread pool from async code.

        Args:
            input_path: Path to input video
            output_path: Path for final output (with audio)
            params: Annotation parameters
            progress_callback: Called with progress 0-99 during processing
        """
        metadata = self._get_video_metadata(input_path)

        # Resolve effective codec and quality from source when auto
        if self.codec == "auto":
            resolved_codec = _CODEC_NAME_MAP.get(metadata.codec_name, None)
            if resolved_codec is not None and metadata.bit_rate is not None:
                effective_codec = resolved_codec
                effective_crf = None
                effective_bitrate = metadata.bit_rate
            elif resolved_codec is not None:
                effective_codec = resolved_codec
                effective_crf = 18
                effective_bitrate = None
            else:
                effective_codec = "h264"
                effective_crf = 18
                effective_bitrate = None
            logger.info(
                f"Auto codec: source={metadata.codec_name}, resolved={effective_codec}, "
                f"bitrate={effective_bitrate}, crf={effective_crf}"
            )
        else:
            effective_codec = self.codec
            effective_crf = self.crf
            effective_bitrate = None

        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "ckpt_path", "unknown")
        model_device = getattr(self.model, "device", "unknown")
        logger.info(
            f"Starting annotation: {input_path.name}, "
            f"{metadata.width}x{metadata.height} @ {metadata.fps:.1f}fps, ~{metadata.total_frames} frames, "
            f"model={model_name}, device={model_device}, "
            f"detect_every={params.detect_every}, conf={params.conf}"
        )

        stats = AnnotationStats(total_frames=metadata.total_frames)
        current_detections: list[DetectionBox] = []
        font_scale = self.visualizer.calculate_adaptive_font_scale(metadata.height)
        frame_num = 0
        start_time = time.perf_counter()

        with FFmpegDecoder(input_path, metadata.width, metadata.height, self.hw_config) as decoder, \
             FFmpegEncoder(input_path, output_path, metadata.width, metadata.height, metadata.fps,
                           self.hw_config, effective_codec,
                           crf=effective_crf, bitrate=effective_bitrate) as encoder:

            while True:
                frame = decoder.read_frame()
                if frame is None:
                    break

                if frame_num % params.detect_every == 0:
                    results = self.model.predict(
                        source=frame,
                        conf=params.conf,
                        imgsz=params.imgsz,
                        max_det=params.max_det,
                        verbose=False,
                    )
                    current_detections = self._extract_detections(
                        results, params.classes
                    )
                    stats.detected_frames += 1
                    stats.total_detections += len(current_detections)
                    logger.debug(
                        f"Frame {frame_num}: YOLO detected "
                        f"{len(current_detections)} objects"
                    )
                else:
                    stats.tracked_frames += 1
                    stats.total_detections += len(current_detections)

                self._draw_detections(frame, current_detections, params, font_scale)
                encoder.write_frame(frame)
                frame_num += 1

                if progress_callback and metadata.total_frames > 0 and frame_num % 10 == 0:
                    progress = int((frame_num / metadata.total_frames) * 100)
                    progress_callback(min(progress, 99))

        stats.total_frames = frame_num
        stats.processing_time_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        fps_actual = frame_num / max(stats.processing_time_ms / 1000, 0.001)
        logger.info(
            f"Frame processing complete: {frame_num} frames in {stats.processing_time_ms}ms "
            f"({fps_actual:.1f} fps), detected={stats.detected_frames}, "
            f"tracked={stats.tracked_frames}, total_detections={stats.total_detections}"
        )

        return stats

    @staticmethod
    def _get_video_metadata(video_path: Path) -> VideoMetadata:
        """Get video metadata via ffprobe.

        Raises RuntimeError if ffprobe fails or returns invalid metadata.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            str(video_path),
        ]
        logger.debug(f"ffprobe command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
        except Exception as e:
            raise RuntimeError(f"ffprobe failed: {e}") from e

        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe returned non-zero exit code ({result.returncode}) "
                f"for {video_path}"
            )

        data = json.loads(result.stdout)
        streams = data.get("streams")
        if not streams:
            raise RuntimeError(f"ffprobe returned no video streams for {video_path}")
        stream = streams[0]

        fps = _parse_fps(stream.get("avg_frame_rate"), stream.get("r_frame_rate"))

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        nb_frames_raw = stream.get("nb_frames", "0")
        try:
            total_frames = int(nb_frames_raw)
        except (ValueError, TypeError):
            total_frames = 0
        if total_frames == 0:
            duration_raw = stream.get("duration", "0")
            try:
                duration = float(duration_raw)
            except (ValueError, TypeError):
                duration = 0.0
            total_frames = int(duration * fps)

        # Parse codec and bitrate for auto-matching
        codec_name = stream.get("codec_name")
        bit_rate_raw = stream.get("bit_rate")
        bit_rate = None
        if bit_rate_raw is not None:
            try:
                bit_rate = int(bit_rate_raw)
            except (ValueError, TypeError):
                bit_rate = None
        if bit_rate is not None and not (_MIN_BITRATE <= bit_rate <= _MAX_BITRATE):
            bit_rate = None

        if width > 0 and height > 0 and fps > 0:
            logger.debug(
                f"ffprobe metadata: {width}x{height} @ {fps:.2f}fps, "
                f"~{total_frames} frames, codec={codec_name or '?'}, "
                f"bitrate={bit_rate or '?'}"
            )
            return VideoMetadata(
                fps=fps, width=width, height=height,
                total_frames=total_frames,
                codec_name=codec_name, bit_rate=bit_rate,
            )

        raise RuntimeError(
            f"ffprobe returned invalid metadata: {width}x{height} @ {fps}fps "
            f"for {video_path}"
        )

    def _extract_detections(
        self, results: list, class_filter: list[str] | None
    ) -> list[DetectionBox]:
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                class_id = int(cls[i])
                class_name = self.class_names.get(
                    class_id, f"class_{class_id}"
                )
                if class_filter and class_name not in class_filter:
                    continue
                detections.append(
                    DetectionBox(
                        x1=int(xyxy[i][0]),
                        y1=int(xyxy[i][1]),
                        x2=int(xyxy[i][2]),
                        y2=int(xyxy[i][3]),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf[i]),
                    )
                )
        return detections

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[DetectionBox],
        params: AnnotationParams,
        font_scale: float,
    ) -> None:
        for det in detections:
            self.visualizer.draw_detection(
                image=frame,
                det=det,
                line_width=params.line_width,
                show_labels=params.show_labels,
                show_conf=params.show_conf,
                font_scale=font_scale,
                text_thickness=1,
            )
