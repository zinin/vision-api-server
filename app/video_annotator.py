import json
import logging
import subprocess
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from visualization import DetectionVisualizer, DetectionBox

logger = logging.getLogger(__name__)

CODEC_MAP = {
    "h264": "libx264",
    "h265": "libx265",
    "av1": "libsvtav1",
}


def _create_csrt_tracker() -> cv2.Tracker:
    """Create CSRT tracker, handling different OpenCV versions."""
    if hasattr(cv2, "TrackerCSRT"):
        return cv2.TrackerCSRT.create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT"):
        return cv2.legacy.TrackerCSRT.create()
    raise RuntimeError(
        "CSRT tracker not available. Install opencv-contrib-python-headless."
    )


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
    total_frames: int = 0
    detected_frames: int = 0
    tracked_frames: int = 0
    total_detections: int = 0
    processing_time_ms: int = 0


class VideoAnnotator:
    """Annotate video with YOLO detections and CSRT tracking."""

    def __init__(
        self,
        model: Any,
        visualizer: DetectionVisualizer,
        class_names: dict[int, str],
        codec: str = "h264",
        crf: int = 18,
    ):
        self.model = model
        self.visualizer = visualizer
        self.class_names = class_names
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
        # Get metadata via ffprobe (more reliable than cv2.CAP_PROP for VFR video)
        fps, width, height, total_frames = self._get_video_metadata(input_path)

        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "ckpt_path", "unknown")
        model_device = getattr(self.model, "device", "unknown")
        logger.info(
            f"Starting annotation: {input_path.name}, "
            f"{width}x{height} @ {fps:.1f}fps, ~{total_frames} frames, "
            f"model={model_name}, device={model_device}, "
            f"detect_every={params.detect_every}, conf={params.conf}"
        )

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        logger.debug(f"VideoCapture opened: {input_path}")

        writer = None
        try:

            video_only_path = output_path.parent / "video_only.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(video_only_path), fourcc, fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("Cannot create video writer")
            logger.info(f"VideoWriter created: {video_only_path}, codec=mp4v, {width}x{height} @ {fps:.1f}fps")

            stats = AnnotationStats(total_frames=total_frames)
            trackers: list[tuple[cv2.Tracker, DetectionBox]] = []
            current_detections: list[DetectionBox] = []
            font_scale = self.visualizer.calculate_adaptive_font_scale(height)
            frame_num = 0
            start_time = time.perf_counter()

            while True:
                ret, frame = cap.read()
                if not ret:
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
                    trackers = self._init_trackers(frame, current_detections)
                    stats.detected_frames += 1
                    stats.total_detections += len(current_detections)
                    logger.debug(
                        f"Frame {frame_num}: YOLO detected {len(current_detections)} objects, "
                        f"initialized {len(trackers)} trackers"
                    )
                else:
                    prev_count = len(trackers)
                    current_detections = self._update_trackers(
                        frame, trackers
                    )
                    stats.tracked_frames += 1
                    stats.total_detections += len(current_detections)
                    if len(current_detections) < prev_count:
                        logger.debug(
                            f"Frame {frame_num}: tracking {len(current_detections)}/{prev_count} "
                            f"(lost {prev_count - len(current_detections)})"
                        )

                self._draw_detections(frame, current_detections, params, font_scale)
                writer.write(frame)
                frame_num += 1

                if progress_callback and total_frames > 0 and frame_num % 10 == 0:
                    progress = int((frame_num / total_frames) * 100)
                    progress_callback(min(progress, 99))

            writer.release()
            writer = None
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

            self._merge_audio(input_path, video_only_path, output_path)

            try:
                if video_only_path.exists():
                    video_only_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to clean up intermediate file: {e}")

            return stats

        finally:
            if writer is not None:
                writer.release()
            cap.release()

    @staticmethod
    def _get_video_metadata(
        video_path: Path,
    ) -> tuple[float, int, int, int]:
        """Get video metadata via ffprobe. Returns (fps, width, height, total_frames)."""

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
            if result.returncode == 0:
                data = json.loads(result.stdout)
                stream = data["streams"][0]
                # Parse fps from r_frame_rate (e.g. "30/1")
                r_rate = stream.get("r_frame_rate", "30/1")
                num, den = map(int, r_rate.split("/"))
                fps = num / den if den else 30.0
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))
                total_frames = int(stream.get("nb_frames", 0))
                if total_frames == 0:
                    # Estimate from duration
                    duration = float(stream.get("duration", 0))
                    total_frames = int(duration * fps)
                # Validate metadata is usable
                if width > 0 and height > 0 and fps > 0:
                    logger.debug(
                        f"ffprobe metadata: {width}x{height} @ {fps:.2f}fps, "
                        f"~{total_frames} frames, codec={stream.get('codec_name', '?')}"
                    )
                    return fps, width, height, total_frames
                logger.warning(
                    f"ffprobe returned invalid metadata: {width}x{height} @ {fps}fps"
                )
        except Exception as e:
            logger.warning(f"ffprobe failed: {e}, falling back to cv2")

        # Fallback to cv2
        logger.info("Using cv2 fallback for video metadata")
        cap = cv2.VideoCapture(str(video_path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.debug(f"cv2 metadata: {width}x{height} @ {fps:.2f}fps, ~{total_frames} frames")
            return fps, width, height, total_frames
        finally:
            cap.release()

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

    def _init_trackers(
        self, frame: np.ndarray, detections: list[DetectionBox]
    ) -> list[tuple[cv2.Tracker, DetectionBox]]:
        trackers = []
        for det in detections:
            tracker = _create_csrt_tracker()
            w = det.x2 - det.x1
            h = det.y2 - det.y1
            if w <= 0 or h <= 0:
                continue
            tracker.init(frame, (det.x1, det.y1, w, h))
            trackers.append((tracker, det))
        return trackers

    def _update_trackers(
        self,
        frame: np.ndarray,
        trackers: list[tuple[cv2.Tracker, DetectionBox]],
    ) -> list[DetectionBox]:
        updated = []
        for tracker, orig_det in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                if w <= 0 or h <= 0:
                    continue
                updated.append(
                    DetectionBox(
                        x1=x,
                        y1=y,
                        x2=x + w,
                        y2=y + h,
                        class_id=orig_det.class_id,
                        class_name=orig_det.class_name,
                        confidence=orig_det.confidence,
                    )
                )
        return updated

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

    def _merge_audio(
        self, original: Path, video_only: Path, output: Path
    ) -> None:
        """Re-encode video with target codec and merge audio from original using FFmpeg."""
        encoder = CODEC_MAP.get(self.codec, "libx264")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_only),
            "-i", str(original),
            "-c:v", encoder,
            "-crf", str(self.crf),
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output),
        ]
        logger.info("Merging audio from original video")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                logger.info(f"Audio merge complete: {output}")
            else:
                logger.warning(
                    f"FFmpeg audio merge failed (rc={result.returncode}): "
                    f"{result.stderr[:500]}"
                )
                # Fallback: use video without audio
                shutil.copy2(str(video_only), str(output))
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"FFmpeg audio merge failed ({type(e).__name__}), using video only")
            shutil.copy2(str(video_only), str(output))
