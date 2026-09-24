"""Batched YOLO inference for pass 1 of the video annotator.

``inference_size`` and ``resize_for_inference`` shrink a frame exactly the way
Ultralytics' LetterBox does (same size, cv2 INTER_LINEAR), so ``predict()`` on
the pre-resized frame sees the same input tensor as on the full frame while the
resize runs on the reader thread. ``BatchDetector`` feeds those frames to
``predict()`` in batches and maps the boxes back to the full frame.
"""
import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics.utils.checks import check_imgsz

from detection_stabilizer import RawDetection

logger = logging.getLogger(__name__)

_NVIDIA_AUTO_BATCH = 8


@dataclass(frozen=True, slots=True)
class InferenceMode:
    fp16: bool
    batch_size: int


def resolve_inference_mode(fp16: str, batch_size: str, device: str) -> InferenceMode:
    """Turn the VIDEO_FP16 and VIDEO_BATCH_SIZE values into a mode for one model.

    ``auto`` means FP16 with batch 8 on NVIDIA and FP32 with batch 1 elsewhere.
    ROCm builds of torch also call the GPU ``cuda``; ``torch.version.hip`` tells
    them apart. FP16 on CPU is refused: PyTorch has no fast half kernels there.
    """
    device_type = torch.device(device).type
    nvidia = device_type == "cuda" and torch.version.hip is None
    use_fp16 = nvidia if fp16 == "auto" else fp16 == "true"
    if use_fp16 and device_type == "cpu":
        logger.warning("VIDEO_FP16=true ignored: the model runs on CPU, staying in FP32")
        use_fp16 = False
    if batch_size == "auto":
        size = _NVIDIA_AUTO_BATCH if nvidia else 1
    else:
        size = int(batch_size)
    return InferenceMode(fp16=use_fp16, batch_size=size)


def model_stride(model: Any) -> int:
    """Largest stride of a YOLO model, floored at 32 as Ultralytics does."""
    try:
        stride = int(max(model.model.stride))
    except (AttributeError, TypeError, ValueError):
        stride = 32
    return max(stride, 32)


def inference_size(width: int, height: int, imgsz: int, stride: int) -> tuple[int, int]:
    """(w, h) that Ultralytics' LetterBox resizes a width x height frame to."""
    target_h, target_w = check_imgsz(imgsz, stride=stride, min_dim=2)
    r = min(target_h / height, target_w / width)
    return round(width * r), round(height * r)


def resize_for_inference(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A new array of ``size`` (w, h), resized the way LetterBox resizes."""
    if (frame.shape[1], frame.shape[0]) == size:
        return frame.copy()
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def extract_detections(
    result: Any,
    frame_num: int,
    class_names: dict[int, str],
    scale: tuple[float, float] = (1.0, 1.0),
) -> list[RawDetection]:
    """RawDetections of one YOLO result, boxes multiplied by ``scale`` (sx, sy)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    sx, sy = scale
    detections = []
    for i in range(len(cls)):
        class_id = int(cls[i])
        detections.append(RawDetection(
            frame_num=frame_num,
            x1=int(xyxy[i][0] * sx), y1=int(xyxy[i][1] * sy),
            x2=int(xyxy[i][2] * sx), y2=int(xyxy[i][3] * sy),
            class_id=class_id,
            class_name=class_names.get(class_id, f"class_{class_id}"),
            confidence=float(conf[i]),
        ))
    return detections


class BatchDetector:
    """Run YOLO on frames in batches and collect RawDetections per frame.

    Frames come already resized by ``resize_for_inference``; ``scale`` maps
    their boxes back to the full frame. When the GPU runs out of memory at a
    batch above 1, the batch is halved and the same frames run again.
    """

    def __init__(
        self,
        model: Any,
        class_names: dict[int, str],
        *,
        conf: float,
        imgsz: int,
        max_det: int,
        fp16: bool,
        batch_size: int,
        scale: tuple[float, float],
    ):
        self._model = model
        self._class_names = class_names
        self._conf = conf
        self._imgsz = imgsz
        self._max_det = max_det
        self._fp16 = fp16
        self._scale = scale
        self.batch_size = batch_size
        self.detections: dict[int, list[RawDetection]] = {}
        self.detected_frames = 0
        self._pending: list[tuple[int, np.ndarray]] = []

    def add(self, frame_num: int, frame: np.ndarray) -> None:
        self._pending.append((frame_num, frame))
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        items, self._pending = self._pending, []
        start = 0
        while start < len(items):
            chunk = items[start:start + self.batch_size]
            results = self._predict([frame for _, frame in chunk])
            if results is None:  # out of GPU memory: batch_size was halved, retry
                continue
            if len(results) != len(chunk):
                raise RuntimeError(
                    f"YOLO returned {len(results)} results for {len(chunk)} frames"
                )
            for (frame_num, _), result in zip(chunk, results):
                self.detected_frames += 1
                detections = extract_detections(result, frame_num, self._class_names, self._scale)
                if detections:
                    self.detections[frame_num] = detections
            start += len(chunk)

    def _predict(self, frames: list[np.ndarray]) -> list[Any] | None:
        kwargs: dict[str, Any] = {
            "source": frames, "conf": self._conf, "imgsz": self._imgsz,
            "max_det": self._max_det, "verbose": False,
        }
        if self._fp16:
            kwargs["quantize"] = 16
        try:
            return self._model.predict(**kwargs)
        except torch.cuda.OutOfMemoryError:
            if self.batch_size == 1:
                raise
        # Out of memory at batch > 1. The except block has ended, so the
        # traceback no longer pins the failed batch's tensors.
        old = self.batch_size
        self.batch_size = max(1, old // 2)
        torch.cuda.empty_cache()
        logger.warning(f"GPU out of memory at batch {old}, retrying with batch {self.batch_size}")
        return None
