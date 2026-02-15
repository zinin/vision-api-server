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
        fps, width, height, total_frames = self._get_video_metadata(input_path)

        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "ckpt_path", "unknown")
        model_device = getattr(self.model, "device", "unknown")
        logger.info(
            f"Starting annotation: {input_path.name}, "
            f"{width}x{height} @ {fps:.1f}fps, ~{total_frames} frames, "
            f"model={model_name}, device={model_device}, "
            f"detect_every={params.detect_every}, conf={params.conf}"
        )

        stats = AnnotationStats(total_frames=total_frames)
        current_detections: list[DetectionBox] = []
        font_scale = self.visualizer.calculate_adaptive_font_scale(height)
        frame_num = 0
        start_time = time.perf_counter()

        with FFmpegDecoder(input_path, width, height, self.hw_config) as decoder, \
             FFmpegEncoder(input_path, output_path, width, height, fps,
                           self.hw_config, self.codec, self.crf) as encoder:

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

                if progress_callback and total_frames > 0 and frame_num % 10 == 0:
                    progress = int((frame_num / total_frames) * 100)
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
    def _get_video_metadata(
        video_path: Path,
    ) -> tuple[float, int, int, int]:
        """Get video metadata via ffprobe. Returns (fps, width, height, total_frames).

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
        stream = data["streams"][0]
        r_rate = stream.get("r_frame_rate", "30/1")
        num, den = map(int, r_rate.split("/"))
        fps = num / den if den else 30.0
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
        if width > 0 and height > 0 and fps > 0:
            logger.debug(
                f"ffprobe metadata: {width}x{height} @ {fps:.2f}fps, "
                f"~{total_frames} frames, codec={stream.get('codec_name', '?')}"
            )
            return fps, width, height, total_frames

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
