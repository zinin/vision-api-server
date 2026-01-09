import json

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
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
    debug: bool = False
    inference_timeout: float = 30.0
    max_executor_workers: int = 4

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

    @field_validator("yolo_model_ttl")
    @classmethod
    def validate_model_ttl(cls, v: int) -> int:
        if v < 60:
            raise ValueError("yolo_model_ttl must be at least 60 seconds")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()