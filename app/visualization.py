"""Visualization module for YOLO detection results."""

import cv2
import numpy as np
from typing import Any, Sequence
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Color:
    """BGR color representation."""

    b: int
    g: int
    r: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.b, self.g, self.r)


@dataclass(frozen=True, slots=True)
class DetectionBox:
    """Immutable detection data for rendering."""

    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float


class DetectionVisualizer:
    """High-performance detection visualizer with adaptive styling."""

    # Optimized color palette (80 colors for COCO classes)
    COLOR_PALETTE: tuple[Color, ...] = (
        Color(255, 0, 0), Color(0, 255, 0), Color(0, 0, 255),
        Color(255, 255, 0), Color(255, 0, 255), Color(0, 255, 255),
        Color(128, 0, 0), Color(0, 128, 0), Color(0, 0, 128),
        Color(128, 128, 0), Color(128, 0, 128), Color(0, 128, 128),
        Color(255, 128, 0), Color(255, 0, 128), Color(128, 255, 0),
        Color(0, 255, 128), Color(128, 0, 255), Color(0, 128, 255),
        Color(255, 128, 128), Color(128, 255, 128), Color(128, 128, 255),
        Color(192, 192, 192), Color(64, 64, 64), Color(255, 192, 128),
        Color(128, 255, 192), Color(192, 128, 255), Color(255, 255, 128),
        Color(128, 255, 255), Color(255, 128, 255), Color(64, 128, 128),
        Color(128, 64, 0), Color(0, 64, 128), Color(128, 0, 64),
        Color(64, 0, 128), Color(0, 128, 64), Color(64, 128, 0),
        Color(192, 64, 0), Color(0, 192, 64), Color(64, 0, 192),
        Color(192, 0, 64), Color(0, 64, 192), Color(64, 192, 0),
        Color(255, 192, 0), Color(0, 255, 192), Color(192, 0, 255),
        Color(255, 0, 192), Color(0, 192, 255), Color(192, 255, 0),
        Color(128, 192, 64), Color(64, 192, 128), Color(192, 64, 128),
        Color(128, 64, 192), Color(64, 128, 192), Color(192, 128, 64),
        Color(255, 64, 0), Color(0, 255, 64), Color(64, 0, 255),
        Color(255, 0, 64), Color(0, 64, 255), Color(64, 255, 0),
        Color(255, 128, 64), Color(64, 255, 128), Color(128, 64, 255),
        Color(255, 64, 128), Color(64, 128, 255), Color(128, 255, 64),
        Color(192, 192, 0), Color(0, 192, 192), Color(192, 0, 192),
        Color(96, 96, 96), Color(160, 160, 160), Color(224, 224, 224),
        Color(32, 32, 32), Color(255, 224, 192), Color(192, 255, 224),
        Color(224, 192, 255), Color(255, 255, 192), Color(192, 255, 255),
        Color(255, 192, 255), Color(160, 96, 32),
    )

    __slots__ = ("class_names", "_color_cache", "_font")

    def __init__(self, class_names: dict[int, str]):
        self.class_names = class_names
        self._color_cache: dict[int, Color] = {}
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_class_color(self, class_id: int) -> Color:
        """Get cached color for class ID."""
        if class_id not in self._color_cache:
            self._color_cache[class_id] = self.COLOR_PALETTE[
                class_id % len(self.COLOR_PALETTE)
                ]
        return self._color_cache[class_id]

    def _calculate_adaptive_font_scale(self, image_height: int) -> float:
        """Calculate font scale based on image size."""
        reference_height = 720
        scale_factor = image_height / reference_height
        return max(0.3, min(1.5, 0.5 * scale_factor))

    def draw_yolo_results(
            self,
            image: np.ndarray,
            results: Sequence[Any],
            line_width: int = 2,
            show_labels: bool = True,
            show_conf: bool = True,
            font_scale: float | None = None,
            text_thickness: int = 1
    ) -> np.ndarray:
        """
        Draw YOLO detection results on image.

        Args:
            image: Source image (BGR format)
            results: YOLO prediction results
            line_width: Bounding box line thickness
            show_labels: Display class names
            show_conf: Display confidence scores
            font_scale: Font scale (None for adaptive sizing)
            text_thickness: Text stroke thickness

        Returns:
            Annotated image copy
        """
        annotated = image.copy()

        actual_font_scale = (
            font_scale if font_scale is not None
            else self._calculate_adaptive_font_scale(image.shape[0])
        )

        # Extract all detections
        detections: list[DetectionBox] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            conf = boxes.conf.cpu().numpy()

            for i in range(len(cls)):
                class_id = int(cls[i])
                detections.append(DetectionBox(
                    x1=int(xyxy[i][0]),
                    y1=int(xyxy[i][1]),
                    x2=int(xyxy[i][2]),
                    y2=int(xyxy[i][3]),
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, f"class_{class_id}"),
                    confidence=float(conf[i])
                ))

        # Draw all detections
        for det in detections:
            self._draw_detection(
                annotated, det, line_width,
                show_labels, show_conf,
                actual_font_scale, text_thickness
            )

        return annotated

    def _draw_detection(
            self,
            image: np.ndarray,
            det: DetectionBox,
            line_width: int,
            show_labels: bool,
            show_conf: bool,
            font_scale: float,
            text_thickness: int
    ) -> None:
        """Draw a single detection box with optional label."""
        color = self._get_class_color(det.class_id)
        color_tuple = color.as_tuple()

        # Draw bounding box
        cv2.rectangle(
            image,
            (det.x1, det.y1),
            (det.x2, det.y2),
            color_tuple,
            line_width
        )

        if not (show_labels or show_conf):
            return

        # Build label
        label_parts = []
        if show_labels:
            label_parts.append(det.class_name)
        if show_conf:
            label_parts.append(f"{det.confidence:.2f}")

        label = " ".join(label_parts)
        self._draw_label_with_background(
            image, label, det.x1, det.y1,
            color_tuple, font_scale, text_thickness
        )

    def _draw_label_with_background(
            self,
            image: np.ndarray,
            label: str,
            x: int,
            y: int,
            bg_color: tuple[int, int, int],
            font_scale: float,
            thickness: int
    ) -> None:
        """Draw text label with background."""
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self._font, font_scale, thickness
        )

        padding = 4
        label_y = max(text_h + baseline + padding, y)

        # Background rectangle
        cv2.rectangle(
            image,
            (x, label_y - text_h - baseline - padding),
            (x + text_w + padding, label_y),
            bg_color,
            -1
        )

        # Text
        cv2.putText(
            image,
            label,
            (x + padding // 2, label_y - baseline - padding // 2),
            self._font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


def encode_image_to_bytes(
        image: np.ndarray,
        format: str = ".jpg",
        quality: int = 90
) -> bytes:
    """
    Encode image to bytes with configurable quality.

    Args:
        image: Image as numpy array (BGR format)
        format: Output format ('.jpg', '.png', '.webp')
        quality: JPEG/WebP quality (1-100)

    Returns:
        Encoded image bytes

    Raises:
        ValueError: If encoding fails
    """
    format_lower = format.lower()

    encode_params: list[int] = []
    if format_lower in (".jpg", ".jpeg"):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif format_lower == ".webp":
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
    elif format_lower == ".png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    success, encoded = cv2.imencode(format, image, encode_params)

    if not success:
        raise ValueError(f"Failed to encode image to {format}")

    return encoded.tobytes()