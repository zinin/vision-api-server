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


@dataclass(frozen=True, slots=True)
class _LabelBox:
    """Label background rectangle (inclusive corners) and the text origin."""

    x0: int
    y0: int
    x1: int
    y1: int
    text_x: int
    text_y: int


# Limited-range BT.601, the matrix swscale applies when the encoder turns
# untagged BGR frames into yuv420p. Rows give Y, U, V from R, G, B in 0..255.
_BT601_FROM_RGB = np.array(
    [
        [65.481, 128.553, 24.966],
        [-37.797, -74.203, 112.0],
        [112.0, -93.786, -18.214],
    ],
    dtype=np.float32,
) / 255.0
_BT601_OFFSET = np.array([16.0, 128.0, 128.0], dtype=np.float32)


def bgr_to_yuv601(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """One BGR colour as limited-range BT.601 (Y, U, V)."""
    rgb = np.array(bgr[::-1], dtype=np.float32)
    y, u, v = np.clip(np.rint(_BT601_FROM_RGB @ rgb + _BT601_OFFSET), 0, 255)
    return int(y), int(u), int(v)


def bgr_patch_to_yuv420(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y, U, V planes of a BGR patch with even sides; U and V are 2x2 means."""
    yuv = patch[..., ::-1].astype(np.float32) @ _BT601_FROM_RGB.T + _BT601_OFFSET
    h, w = patch.shape[:2]
    chroma = yuv[..., 1:].reshape(h // 2, 2, w // 2, 2, 2).mean(axis=(1, 3))

    def to_u8(plane: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(plane), 0, 255).astype(np.uint8)

    return to_u8(yuv[..., 0]), to_u8(chroma[..., 0]), to_u8(chroma[..., 1])


def yuv420_planes(
    frame: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Writable Y, U, V views into a flat yuv420p frame."""
    chroma_w, chroma_h = (width + 1) // 2, (height + 1) // 2
    y_end = width * height
    u_end = y_end + chroma_w * chroma_h
    return (
        frame[:y_end].reshape(height, width),
        frame[y_end:u_end].reshape(chroma_h, chroma_w),
        frame[u_end:u_end + chroma_w * chroma_h].reshape(chroma_h, chroma_w),
    )


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

    def calculate_adaptive_font_scale(self, image_height: int) -> float:
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
            else self.calculate_adaptive_font_scale(image.shape[0])
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
            self.draw_detection(
                annotated, det, line_width,
                show_labels, show_conf,
                actual_font_scale, text_thickness
            )

        return annotated

    def draw_detection(
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

        label = self._label_text(det, show_labels, show_conf)
        if label is None:
            return
        self._draw_label_with_background(
            image, label, det.x1, det.y1,
            color_tuple, font_scale, text_thickness
        )

    def draw_detection_yuv420(
            self,
            frame: np.ndarray,
            width: int,
            height: int,
            det: DetectionBox,
            line_width: int,
            show_labels: bool,
            show_conf: bool,
            font_scale: float,
            text_thickness: int
    ) -> None:
        """Draw a single detection onto a flat yuv420p frame, in place.

        The box goes onto Y with ``line_width`` and onto U and V at half
        resolution with ``(line_width + 1) // 2``. The label is drawn in BGR
        on a patch aligned to even coordinates, converted with BT.601 and
        pasted: luma exactly over the label rectangle, chroma over every 2x2
        block the rectangle touches.
        """
        y_plane, u_plane, v_plane = yuv420_planes(frame, width, height)
        bgr = self._get_class_color(det.class_id).as_tuple()
        luma, cb, cr = bgr_to_yuv601(bgr)

        cv2.rectangle(y_plane, (det.x1, det.y1), (det.x2, det.y2), luma, line_width)
        chroma_width = (line_width + 1) // 2
        corner1, corner2 = (det.x1 // 2, det.y1 // 2), (det.x2 // 2, det.y2 // 2)
        cv2.rectangle(u_plane, corner1, corner2, cb, chroma_width)
        cv2.rectangle(v_plane, corner1, corner2, cr, chroma_width)

        label = self._label_text(det, show_labels, show_conf)
        if label is None:
            return
        box = self._label_box(label, det.x1, det.y1, font_scale, text_thickness)

        # Patch over the label rectangle, grown to even edges for 4:2:0 chroma.
        px0, py0 = box.x0 - box.x0 % 2, box.y0 - box.y0 % 2
        px1, py1 = box.x1 + 1 + (box.x1 + 1) % 2, box.y1 + 1 + (box.y1 + 1) % 2
        patch = np.empty((py1 - py0, px1 - px0, 3), dtype=np.uint8)
        patch[:] = bgr
        cv2.putText(
            patch, label, (box.text_x - px0, box.text_y - py0), self._font,
            font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA,
        )
        patch_y, patch_u, patch_v = bgr_patch_to_yuv420(patch)

        # Luma: exactly the label rectangle, clipped to the frame.
        x0, y0 = max(box.x0, 0), max(box.y0, 0)
        x1, y1 = min(box.x1 + 1, width), min(box.y1 + 1, height)
        if x0 < x1 and y0 < y1:
            y_plane[y0:y1, x0:x1] = patch_y[y0 - py0:y1 - py0, x0 - px0:x1 - px0]

        # Chroma: the 2x2 blocks under the patch, clipped to the planes.
        cx0, cy0 = max(px0, 0) // 2, max(py0, 0) // 2
        cx1, cy1 = min(px1 // 2, u_plane.shape[1]), min(py1 // 2, u_plane.shape[0])
        if cx0 < cx1 and cy0 < cy1:
            rows = slice(cy0 - py0 // 2, cy1 - py0 // 2)
            cols = slice(cx0 - px0 // 2, cx1 - px0 // 2)
            u_plane[cy0:cy1, cx0:cx1] = patch_u[rows, cols]
            v_plane[cy0:cy1, cx0:cx1] = patch_v[rows, cols]

    @staticmethod
    def _label_text(det: DetectionBox, show_labels: bool, show_conf: bool) -> str | None:
        """The label for a detection, or None when both parts are hidden."""
        parts = []
        if show_labels:
            parts.append(det.class_name)
        if show_conf:
            parts.append(f"{det.confidence:.2f}")
        return " ".join(parts) if parts else None

    def _label_box(
            self, label: str, x: int, y: int, font_scale: float, thickness: int
    ) -> _LabelBox:
        """Where the label background and text go for a box whose top-left is (x, y)."""
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self._font, font_scale, thickness
        )
        padding = 4
        label_y = max(text_h + baseline + padding, y)
        return _LabelBox(
            x0=x,
            y0=label_y - text_h - baseline - padding,
            x1=x + text_w + padding,
            y1=label_y,
            text_x=x + padding // 2,
            text_y=label_y - baseline - padding // 2,
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
        box = self._label_box(label, x, y, font_scale, thickness)

        # Background rectangle
        cv2.rectangle(image, (box.x0, box.y0), (box.x1, box.y1), bg_color, -1)

        # Text
        cv2.putText(
            image,
            label,
            (box.text_x, box.text_y),
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