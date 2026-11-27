import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np

from model_manager import ModelEntry
from models import DetectionResponse, ImageSize, Detection, BoundingBox


class InferenceExecutor:
    """Manages thread pool executor for inference operations."""

    _instance: "InferenceExecutor | None" = None
    _executor: ThreadPoolExecutor | None = None

    def __new__(cls, max_workers: int = 4) -> "InferenceExecutor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="inference_"
            )
        return cls._instance

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError("Executor not initialized")
        return self._executor

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor gracefully."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
            InferenceExecutor._instance = None


# Module-level singleton
_inference_executor: InferenceExecutor | None = None


def get_executor(max_workers: int = 4) -> InferenceExecutor:
    """Get or create the inference executor singleton."""
    global _inference_executor
    if _inference_executor is None:
        _inference_executor = InferenceExecutor(max_workers)
    return _inference_executor


def shutdown_executor(wait: bool = True) -> None:
    """Shutdown the global executor."""
    global _inference_executor
    if _inference_executor is not None:
        _inference_executor.shutdown(wait=wait)
        _inference_executor = None


async def run_inference(
    entry: ModelEntry,
    image: np.ndarray,
    conf: float,
    imgsz: int,
    max_det: int,
    max_workers: int = 4
) -> list[Any]:
    """
    Run model inference in thread pool to avoid blocking event loop.

    Calls on one model instance take turns. A YOLO instance keeps the arguments
    of a call on a predictor it reuses, and Ultralytics before 8.4.209 writes
    them there before it takes the predictor's own lock: a concurrent request
    replaces the conf, imgsz and max_det that a running one reads.
    The lock is taken in the worker thread, so it stays held while predict()
    runs even if the awaiting request is cancelled. Other model instances run
    in parallel.

    Args:
        entry: Loaded model, with the lock that serialises its calls
        image: Input image as numpy array
        conf: Confidence threshold
        imgsz: Image size for inference
        max_det: Maximum detections
        max_workers: Thread pool size

    Returns:
        YOLO prediction results
    """
    loop = asyncio.get_running_loop()
    executor = get_executor(max_workers).executor

    predict_fn = partial(
        entry.model.predict,
        source=image,
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False
    )

    def locked_predict() -> list[Any]:
        with entry.predict_lock:
            return predict_fn()

    return await loop.run_in_executor(executor, locked_predict)


def build_detection_response(
    results: list[Any],
    model_names: dict[int, str],
    processing_time_ms: int,
    image_size: tuple[int, int],
    model_path: str
) -> DetectionResponse:
    """
    Build structured detection response from YOLO results.

    Args:
        results: YOLO prediction results
        model_names: Mapping of class IDs to names
        processing_time_ms: Processing time in milliseconds
        image_size: Tuple of (width, height)
        model_path: Path to the model file

    Returns:
        Structured DetectionResponse
    """
    detections: list[Detection] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        for i in range(len(cls)):
            class_id = int(cls[i])
            detections.append(Detection(
                class_id=class_id,
                class_name=model_names.get(class_id, f"class_{class_id}"),
                confidence=round(float(conf[i]), 4),
                bbox=BoundingBox(
                    x1=round(float(xyxy[i][0]), 2),
                    y1=round(float(xyxy[i][1]), 2),
                    x2=round(float(xyxy[i][2]), 2),
                    y2=round(float(xyxy[i][3]), 2)
                )
            ))

    return DetectionResponse(
        detections=detections,
        processing_time=processing_time_ms,
        image_size=ImageSize(width=image_size[0], height=image_size[1]),
        model=model_path
    )