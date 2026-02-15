from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated


class BoundingBox(BaseModel):
    """Bounding box coordinates in pixel values."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"x1": 100.5, "y1": 50.25, "x2": 300.75, "y2": 400.0}
        }
    )

    x1: Annotated[float, Field(description="Left coordinate (pixels)")]
    y1: Annotated[float, Field(description="Top coordinate (pixels)")]
    x2: Annotated[float, Field(description="Right coordinate (pixels)")]
    y2: Annotated[float, Field(description="Bottom coordinate (pixels)")]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """Single object detection result."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": {"x1": 100.0, "y1": 50.0, "x2": 300.0, "y2": 400.0}
            }
        }
    )

    class_id: Annotated[int, Field(ge=0, description="Class identifier")]
    class_name: Annotated[str, Field(min_length=1, description="Human-readable class name")]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Detection confidence score")]
    bbox: BoundingBox


class ImageSize(BaseModel):
    """Image dimensions."""

    width: Annotated[int, Field(gt=0, description="Image width in pixels")]
    height: Annotated[int, Field(gt=0, description="Image height in pixels")]


class FrameDetection(BaseModel):
    """Detection results for a single video frame."""
    frame_number: int = Field(description="Frame number (1-based)")
    timestamp: float = Field(description="Frame timestamp in seconds")
    detections: list[Detection] = Field(default_factory=list)
    count: int = Field(description="Number of detections in frame")


class DetectionResponse(BaseModel):
    """Complete detection response."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.95,
                        "bbox": {"x1": 100.0, "y1": 50.0, "x2": 300.0, "y2": 400.0}
                    }
                ],
                "processing_time": 45,
                "image_size": {"width": 1920, "height": 1080},
                "model": "yolo26s.pt"
            }
        }
    )

    detections: list[Detection] = Field(default_factory=list)
    processing_time_ms: Annotated[
        int, 
        Field(alias="processing_time", ge=0, description="Processing time in milliseconds")
    ]
    image_size: ImageSize
    model: Annotated[str, Field(description="Model file used for detection")]

    @property
    def detection_count(self) -> int:
        return len(self.detections)


class VideoDetectionResponse(BaseModel):
    """Response for video detection."""
    success: bool = True
    video_duration: float = Field(description="Video duration in seconds")
    video_resolution: tuple[int, int] = Field(description="Video resolution (width, height)")
    frames_analyzed: int = Field(description="Number of frames analyzed")
    total_detections: int = Field(description="Total detections across all frames")
    unique_classes: list[str] = Field(description="List of unique detected classes")
    frames: list[FrameDetection] = Field(description="Per-frame detection results")
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
    model: str = Field(description="Model used")

    # Aggregated statistics
    class_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each class across all frames"
    )


class VideoDetectionSettings(BaseModel):
    """Settings for video frame extraction."""
    scene_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Scene change threshold (lower = more sensitive)"
    )
    min_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Minimum interval between frames in seconds"
    )
    max_frames: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of frames to extract"
    )


class ExtractedFrameData(BaseModel):
    """Single extracted frame with base64-encoded image data."""
    frame_number: int = Field(description="Frame number (1-based)")
    timestamp: float = Field(description="Frame timestamp in seconds")
    image_base64: str = Field(description="JPEG image encoded as base64 string")
    width: int = Field(description="Frame width in pixels")
    height: int = Field(description="Frame height in pixels")


class FrameExtractionResponse(BaseModel):
    """Response for frame extraction endpoint."""
    success: bool = True
    video_duration: float = Field(description="Video duration in seconds")
    video_resolution: tuple[int, int] = Field(description="Video resolution (width, height)")
    frames_extracted: int = Field(description="Number of frames extracted")
    frames: list[ExtractedFrameData] = Field(description="Extracted frames with image data")
    processing_time_ms: int = Field(description="Processing time in milliseconds")


class JobCreatedResponse(BaseModel):
    """Response when a video annotation job is created."""
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status")
    message: str = Field(description="Human-readable message")


class JobStats(BaseModel):
    """Statistics for a completed annotation job."""
    total_frames: int = Field(description="Total frames in video")
    detected_frames: int = Field(description="Frames with YOLO detection")
    tracked_frames: int = Field(description="Frames with held detections (reused from last YOLO run)")
    total_detections: int = Field(description="Total object detections")
    processing_time_ms: int = Field(description="Total processing time in ms")


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status: queued, processing, completed, failed")
    progress: int = Field(ge=0, le=100, description="Progress percentage")
    created_at: str = Field(description="Job creation timestamp ISO format")
    completed_at: str | None = Field(default=None, description="Completion timestamp")
    download_url: str | None = Field(default=None, description="URL to download result")
    error: str | None = Field(default=None, description="Error message if failed")
    stats: JobStats | None = Field(default=None, description="Job statistics")
