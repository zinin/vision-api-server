import json

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    yolo_models: str = '{}'
    yolo_device: str = "cpu"
    yolo_model_ttl: int = 900  # 15 minutes TTL for cached models
    max_file_size: int = 10 * 1024 * 1024
    allowed_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    log_level: str = "INFO"
    inference_timeout: float = 30.0
    max_executor_workers: int = 4

    # Video annotation job settings
    video_job_ttl: int = Field(default=3600, ge=60)  # 1 hour TTL for completed jobs
    video_jobs_dir: str = "/tmp/vision_jobs"
    max_queued_jobs: int = Field(default=10, ge=1)
    default_detect_every: int = Field(default=5, ge=1, le=300)
    video_codec: str = "auto"  # auto | h264 | h265 | av1
    video_crf: int = Field(default=18, ge=0, le=63)
    video_hw_accel: str = "auto"  # auto | nvidia | amd | cpu
    vaapi_device: str = "/dev/dri/renderD128"  # VAAPI render device path

    @property
    def preload_model_map(self) -> dict[str, str]:
        """Parse YOLO_MODELS JSON into model->device mapping."""
        return json.loads(self.yolo_models)

    @field_validator("yolo_models")
    @classmethod
    def validate_yolo_models(cls, v: str) -> str:
        """Validate YOLO_MODELS is valid JSON with correct structure."""
        try:
            mapping = json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"YOLO_MODELS must be valid JSON: {e}")

        if not isinstance(mapping, dict):
            raise ValueError("YOLO_MODELS must be a JSON object")

        valid_prefixes = ("cpu", "cuda", "mps")
        for model, device in mapping.items():
            if not isinstance(device, str):
                raise ValueError(f"Device for {model} must be a string")
            if not any(device.startswith(p) for p in valid_prefixes):
                raise ValueError(f"Device '{device}' must start with one of: {valid_prefixes}")

        return v

    @field_validator("yolo_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_prefixes = ("cpu", "cuda", "mps")
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(f"Device must start with one of: {valid_prefixes}")
        return v

    @field_validator("max_file_size")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_file_size must be positive")
        if v > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError("max_file_size cannot exceed 100MB")
        return v

    @field_validator("video_codec")
    @classmethod
    def validate_video_codec(cls, v: str) -> str:
        allowed = ("auto", "h264", "h265", "av1")
        if v not in allowed:
            raise ValueError(f"video_codec must be one of: {allowed}")
        return v

    @field_validator("video_hw_accel")
    @classmethod
    def validate_video_hw_accel(cls, v: str) -> str:
        allowed = ("auto", "nvidia", "amd", "cpu")
        if v not in allowed:
            raise ValueError(f"video_hw_accel must be one of: {allowed}")
        return v

    @field_validator("yolo_model_ttl")
    @classmethod
    def validate_model_ttl(cls, v: int) -> int:
        if v < 60:
            raise ValueError("yolo_model_ttl must be at least 60 seconds")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()