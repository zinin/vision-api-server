import time
import logging
import base64
from typing import Annotated
from io import BytesIO
from collections import Counter

import cv2
import torch

from fastapi import FastAPI, File, UploadFile, Query, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

from config import Settings, get_settings
from dependencies import get_model_manager
from model_manager import ModelManager
from image_utils import validate_and_decode_image
from inference_utils import run_inference, build_detection_response, shutdown_executor
from models import (
    DetectionResponse,
    VideoDetectionResponse,
    FrameDetection,
    Detection,
    BoundingBox,
    FrameExtractionResponse,
    ExtractedFrameData
)
from visualization import encode_image_to_bytes
from video_utils import extract_frames_from_video, VideoFrameExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Video file settings
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB


def validate_device_availability(device: str) -> None:
    """Validate that the specified device is available.

    Args:
        device: Device string (cpu, cuda, cuda:0, mps, etc.)

    Raises:
        RuntimeError: If device is not available.
    """
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Device '{device}' requested but CUDA not available")
    elif device.startswith("mps"):
        if not torch.backends.mps.is_available():
            raise RuntimeError(f"Device '{device}' requested but MPS not available")
    # cpu is always available


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management with proper resource cleanup."""
    settings = get_settings()

    logger.info(f"Default device for dynamic models: {settings.yolo_device}")
    logger.info(f"Models to preload: {settings.preload_model_map}")
    logger.info(f"Model cache TTL: {settings.yolo_model_ttl}s")

    try:
        # Validate all devices are available before any model loading
        devices_to_check = set(settings.preload_model_map.values())
        devices_to_check.add(settings.yolo_device)

        for device in devices_to_check:
            validate_device_availability(device)

        logger.info(f"All devices validated: {devices_to_check}")

        # Initialize ModelManager with default device for dynamic loads
        model_manager = ModelManager(
            default_device=settings.yolo_device,
            ttl_seconds=settings.yolo_model_ttl
        )

        # Preload configured models with their specific devices
        await model_manager.preload_models(settings.preload_model_map)

        # Start background cleanup task
        model_manager.start_cleanup_task(interval_seconds=60)

        app.state.model_manager = model_manager

        # Verify ffmpeg is available
        try:
            VideoFrameExtractor()
            logger.info("FFmpeg verified and ready for video processing")
        except RuntimeError as e:
            logger.warning(f"Video processing unavailable: {e}")

        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise RuntimeError(f"Initialization failed: {e}") from e

    yield

    # Cleanup
    logger.info("Shutting down gracefully...")
    try:
        shutdown_executor(wait=True)

        if hasattr(app.state, "model_manager"):
            await app.state.model_manager.shutdown()

        logger.info("Resources released successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


app = FastAPI(
    title="YOLO Detection API",
    description="REST API for image and video analysis using Ultralytics YOLO",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    settings = get_settings()
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    detail = str(exc) if settings.debug else "An internal error occurred"

    return JSONResponse(
        status_code=500,
        content={"detail": detail, "type": type(exc).__name__}
    )


# Type aliases for cleaner signatures
ConfidenceQuery = Annotated[float, Query(ge=0.0, le=1.0, description="Confidence threshold")]
ImageSizeQuery = Annotated[int, Query(ge=32, le=2016, description="Image size for inference")]
MaxDetQuery = Annotated[int, Query(ge=1, le=1000, description="Maximum detections")]
ModelQuery = Annotated[str | None, Query(description="Model name (e.g. yolo11s.pt). If not specified, uses default model.")]

# Video-specific query parameters
SceneThresholdQuery = Annotated[
    float,
    Query(ge=0.01, le=0.5, description="Scene change threshold (lower = more sensitive)")
]
MinIntervalQuery = Annotated[
    float,
    Query(ge=0.1, le=30.0, description="Minimum interval between frames (seconds)")
]
MaxFramesQuery = Annotated[
    int,
    Query(ge=1, le=200, description="Maximum frames to extract")
]


@app.get("/", tags=["Health"])
async def root(
        settings: Settings = Depends(get_settings),
        model_manager: ModelManager = Depends(get_model_manager)
):
    """Service information and status."""
    preloaded = list(model_manager._preloaded.keys())
    return {
        "service": "YOLO Detection API",
        "version": "2.2.0",
        "preloaded_models": preloaded,
        "default_device": model_manager.default_device,
        "status": "ready",
        "max_file_size_mb": round(settings.max_file_size / (1024 * 1024), 1),
        "max_video_size_mb": round(MAX_VIDEO_SIZE / (1024 * 1024), 1),
        "features": ["image_detection", "video_detection", "visualization", "frame_extraction", "multi_model"]
    }


@app.get("/health", tags=["Health"])
async def health(model_manager: ModelManager = Depends(get_model_manager)):
    """Service health check endpoint."""

    # Check ffmpeg availability
    ffmpeg_available = True
    try:
        VideoFrameExtractor()
    except RuntimeError:
        ffmpeg_available = False

    return {
        "status": "healthy",
        "models_loaded": len(model_manager._preloaded) + len(model_manager._cached),
        "preloaded_count": len(model_manager._preloaded),
        "cached_count": len(model_manager._cached),
        "default_device": model_manager.default_device,
        "video_processing": ffmpeg_available
    }


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_objects(
        file: UploadFile = File(..., description="Image for analysis"),
        conf: ConfidenceQuery = 0.5,
        imgsz: ImageSizeQuery = 640,
        max_det: MaxDetQuery = 100,
        model: ModelQuery = None,
        model_manager: ModelManager = Depends(get_model_manager),
        settings: Settings = Depends(get_settings)
):
    """
    Analyze image using YOLO object detection.

    - **file**: Image file (PNG, JPG, JPEG, WEBP, BMP)
    - **conf**: Confidence threshold (0.0 - 1.0)
    - **imgsz**: Image size for processing
    - **max_det**: Maximum number of detections to return
    - **model**: Model name (e.g. yolo11s.pt). If not specified, uses first preloaded model.
    """
    model_name = model
    if model_name is None:
        if model_manager._preloaded:
            model_name = next(iter(model_manager._preloaded.keys()))
        else:
            raise HTTPException(
                status_code=400,
                detail="No model specified and no preloaded models available"
            )
    logger.info(f"Processing image: {file.filename}, conf={conf}, model={model_name}")

    processed = await validate_and_decode_image(
        file, settings.max_file_size, settings.allowed_extensions
    )

    # Get model (may load on-demand)
    try:
        model_entry = await model_manager.get_model(model_name)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start_time = time.perf_counter()

    results = await run_inference(
        model_entry.model,
        processed.image,
        conf,
        imgsz,
        max_det,
        settings.max_executor_workers
    )

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    response = build_detection_response(
        results,
        model_entry.model.names,
        processing_time_ms,
        (processed.width, processed.height),
        model_entry.model_name
    )

    logger.info(
        f"Detection completed: {len(response.detections)} objects in {processing_time_ms}ms (model: {model_name})"
    )

    return response


@app.post("/detect/video", response_model=VideoDetectionResponse, tags=["Video Detection"])
async def detect_objects_in_video(
        file: UploadFile = File(..., description="Video file for analysis"),
        conf: ConfidenceQuery = 0.5,
        imgsz: ImageSizeQuery = 640,
        max_det: MaxDetQuery = 100,
        scene_threshold: SceneThresholdQuery = 0.05,
        min_interval: MinIntervalQuery = 1.0,
        max_frames: MaxFramesQuery = 50,
        model: ModelQuery = None,
        model_manager: ModelManager = Depends(get_model_manager),
        settings: Settings = Depends(get_settings)
):
    """
    Analyze video using YOLO object detection with smart frame extraction.

    **Frame extraction algorithm:**
    1. Always extracts the first frame
    2. Extracts frames on scene changes (respecting min_interval)
    3. Extracts middle frame if only first frame was selected

    **Parameters:**
    - **file**: Video file (MP4, AVI, MOV, MKV, WEBM, WMV, FLV)
    - **conf**: Confidence threshold (0.0 - 1.0)
    - **imgsz**: Image size for processing
    - **max_det**: Maximum detections per frame
    - **scene_threshold**: Scene change sensitivity (0.01-0.5, lower = more frames)
    - **min_interval**: Minimum seconds between extracted frames
    - **max_frames**: Maximum total frames to analyze
    - **model**: Model name (e.g. yolo11s.pt). If not specified, uses first preloaded model.
    """
    start_time = time.perf_counter()
    model_name = model
    if model_name is None:
        if model_manager._preloaded:
            model_name = next(iter(model_manager._preloaded.keys()))
        else:
            raise HTTPException(
                status_code=400,
                detail="No model specified and no preloaded models available"
            )

    # Validate file extension
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            )

    logger.info(
        f"Processing video: {file.filename}, conf={conf}, model={model_name}, "
        f"scene_threshold={scene_threshold}, min_interval={min_interval}"
    )

    # Read video data with size check
    video_data = await file.read()
    if len(video_data) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Video too large. Maximum size: {MAX_VIDEO_SIZE // (1024 * 1024)} MB"
        )

    # Extract frames
    try:
        frames = await extract_frames_from_video(
            video_data=video_data,
            scene_threshold=scene_threshold,
            min_interval=min_interval,
            max_frames=max_frames
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract frames: {str(e)}"
        )

    if not frames:
        raise HTTPException(
            status_code=400,
            detail="No frames could be extracted from video"
        )

    logger.info(f"Extracted {len(frames)} frames from video")

    # Get model (may load on-demand)
    try:
        model_entry = await model_manager.get_model(model_name)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Process each frame
    frame_results: list[FrameDetection] = []
    all_detections: list[Detection] = []
    class_counter = Counter()

    # Get video dimensions from first frame
    first_frame = frames[0]
    video_height, video_width = first_frame.image.shape[:2]

    # Estimate video duration from last frame timestamp
    video_duration = frames[-1].timestamp if frames else 0.0

    for frame in frames:
        # Run inference on frame
        results = await run_inference(
            model_entry.model,
            frame.image,
            conf,
            imgsz,
            max_det,
            settings.max_executor_workers
        )

        # Extract detections
        frame_detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls.item())
                cls_name = model_entry.model.names[cls_id]
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detection = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=confidence,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                )
                frame_detections.append(detection)
                all_detections.append(detection)
                class_counter[cls_name] += 1

        frame_results.append(FrameDetection(
            frame_number=frame.frame_number,
            timestamp=round(frame.timestamp, 3),
            detections=frame_detections,
            count=len(frame_detections)
        ))

        logger.debug(
            f"Frame {frame.frame_number} @ {frame.timestamp:.2f}s: "
            f"{len(frame_detections)} detections"
        )

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    # Build response
    response = VideoDetectionResponse(
        success=True,
        video_duration=round(video_duration, 3),
        video_resolution=(video_width, video_height),
        frames_analyzed=len(frames),
        total_detections=len(all_detections),
        unique_classes=list(class_counter.keys()),
        frames=frame_results,
        processing_time_ms=processing_time_ms,
        model=model_entry.model_name,
        class_summary=dict(class_counter)
    )

    logger.info(
        f"Video analysis completed: {len(frames)} frames, "
        f"{len(all_detections)} total detections, "
        f"{len(class_counter)} unique classes, "
        f"{processing_time_ms}ms (model: {model_name})"
    )

    return response


@app.post("/extract/frames", response_model=FrameExtractionResponse, tags=["Frame Extraction"])
async def extract_video_frames(
        file: UploadFile = File(..., description="Video file for frame extraction"),
        scene_threshold: SceneThresholdQuery = 0.05,
        min_interval: MinIntervalQuery = 1.0,
        max_frames: MaxFramesQuery = 50,
        quality: Annotated[int, Query(ge=1, le=100, description="JPEG quality")] = 85
):
    """
    Extract key frames from video without object detection.

    Returns frames as base64-encoded JPEG images.

    **Frame extraction algorithm:**
    1. Always extracts the first frame
    2. Extracts frames on scene changes (respecting min_interval)
    3. Extracts middle frame if only first frame was selected

    **Parameters:**
    - **file**: Video file (MP4, AVI, MOV, MKV, WEBM, WMV, FLV)
    - **scene_threshold**: Scene change sensitivity (0.01-0.5, lower = more frames)
    - **min_interval**: Minimum seconds between extracted frames
    - **max_frames**: Maximum total frames to extract
    - **quality**: JPEG compression quality (1-100)
    """
    start_time = time.perf_counter()

    # Validate file extension
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            )

    logger.info(
        f"Extracting frames from video: {file.filename}, "
        f"scene_threshold={scene_threshold}, min_interval={min_interval}"
    )

    # Read video data with size check
    video_data = await file.read()
    if len(video_data) > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Video too large. Maximum size: {MAX_VIDEO_SIZE // (1024 * 1024)} MB"
        )

    # Extract frames
    try:
        frames = await extract_frames_from_video(
            video_data=video_data,
            scene_threshold=scene_threshold,
            min_interval=min_interval,
            max_frames=max_frames
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract frames: {str(e)}"
        )

    if not frames:
        raise HTTPException(
            status_code=400,
            detail="No frames could be extracted from video"
        )

    logger.info(f"Extracted {len(frames)} frames from video")

    # Get video dimensions from first frame
    first_frame = frames[0]
    video_height, video_width = first_frame.image.shape[:2]

    # Estimate video duration from last frame timestamp
    video_duration = frames[-1].timestamp if frames else 0.0

    # Convert frames to base64-encoded JPEG
    frame_results: list[ExtractedFrameData] = []
    for frame in frames:
        # Convert RGB to BGR for cv2
        bgr_image = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)

        # Encode to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, jpeg_data = cv2.imencode('.jpg', bgr_image, encode_params)

        if not success:
            logger.warning(f"Failed to encode frame {frame.frame_number}")
            continue

        # Convert to base64
        image_base64 = base64.b64encode(jpeg_data.tobytes()).decode('utf-8')

        height, width = frame.image.shape[:2]
        frame_results.append(ExtractedFrameData(
            frame_number=frame.frame_number,
            timestamp=round(frame.timestamp, 3),
            image_base64=image_base64,
            width=width,
            height=height
        ))

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    response = FrameExtractionResponse(
        video_duration=round(video_duration, 3),
        video_resolution=(video_width, video_height),
        frames_extracted=len(frame_results),
        frames=frame_results,
        processing_time_ms=processing_time_ms
    )

    logger.info(
        f"Frame extraction completed: {len(frame_results)} frames, "
        f"{processing_time_ms}ms"
    )

    return response


@app.post("/detect/visualize", tags=["Detection"])
async def detect_and_visualize(
        file: UploadFile = File(..., description="Image for analysis"),
        conf: ConfidenceQuery = 0.5,
        imgsz: ImageSizeQuery = 640,
        max_det: MaxDetQuery = 100,
        line_width: Annotated[int, Query(ge=1, le=10, description="Bounding box line width")] = 2,
        show_labels: Annotated[bool, Query(description="Show class labels")] = True,
        show_conf: Annotated[bool, Query(description="Show confidence scores")] = True,
        quality: Annotated[int, Query(ge=1, le=100, description="JPEG quality")] = 90,
        model: ModelQuery = None,
        model_manager: ModelManager = Depends(get_model_manager),
        settings: Settings = Depends(get_settings)
) -> StreamingResponse:
    """
    Analyze image and return annotated visualization.

    Returns an image with drawn bounding boxes, labels, and confidence scores.

    - **model**: Model name (e.g. yolo11s.pt). If not specified, uses first preloaded model.
    """
    model_name = model
    if model_name is None:
        if model_manager._preloaded:
            model_name = next(iter(model_manager._preloaded.keys()))
        else:
            raise HTTPException(
                status_code=400,
                detail="No model specified and no preloaded models available"
            )
    logger.info(f"Processing with visualization: {file.filename}, conf={conf}, model={model_name}")

    processed = await validate_and_decode_image(
        file, settings.max_file_size, settings.allowed_extensions
    )

    # Get model (may load on-demand)
    try:
        model_entry = await model_manager.get_model(model_name)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    start_time = time.perf_counter()

    results = await run_inference(
        model_entry.model,
        processed.image,
        conf,
        imgsz,
        max_det,
        settings.max_executor_workers
    )

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
    num_detections = sum(len(r.boxes) for r in results if r.boxes is not None)

    logger.info(f"Detection completed: {num_detections} objects in {processing_time_ms}ms (model: {model_name})")

    annotated_image = model_entry.visualizer.draw_yolo_results(
        image=processed.image,
        results=results,
        line_width=line_width,
        show_labels=show_labels,
        show_conf=show_conf
    )

    image_bytes_data = encode_image_to_bytes(annotated_image, ".jpg", quality)
    image_buffer = BytesIO(image_bytes_data)

    safe_filename = "".join(
        c for c in processed.original_filename
        if c.isalnum() or c in "._-"
    )

    return StreamingResponse(
        image_buffer,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'inline; filename="detected_{safe_filename}"',
            "X-Processing-Time-Ms": str(processing_time_ms),
            "X-Detections-Count": str(num_detections),
            "Cache-Control": "no-cache"
        }
    )


@app.get("/models", tags=["Models"])
async def list_models(model_manager: ModelManager = Depends(get_model_manager)):
    """
    Get information about loaded models.

    Returns:
    - **default_model**: The default model used when no model is specified
    - **preloaded**: List of models loaded at startup (never evicted)
    - **cached**: List of on-demand loaded models with TTL info
    - **ttl_seconds**: Time-to-live for cached models
    - **device**: Device used for inference
    """
    return model_manager.get_status()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        workers=1
    )