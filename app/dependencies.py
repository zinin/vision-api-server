from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from ultralytics import YOLO
    from visualization import DetectionVisualizer
    from model_manager import ModelManager
    from job_manager import JobManager


@dataclass(slots=True)
class ModelContainer:
    """Container for model and related resources (legacy, for compatibility)."""

    model: "YOLO"
    visualizer: "DetectionVisualizer"
    device: str
    model_path: str


async def get_model_container(request: Request) -> ModelContainer:
    """Get model container from app state (legacy).

    Raises:
        RuntimeError: If model container is not initialized.
    """
    container = getattr(request.app.state, "model_container", None)
    if container is None:
        raise RuntimeError("Model container not initialized")
    return container


async def get_model_manager(request: Request) -> "ModelManager":
    """Get ModelManager from app state.

    Raises:
        RuntimeError: If model manager is not initialized.
    """
    manager = getattr(request.app.state, "model_manager", None)
    if manager is None:
        raise RuntimeError("Model manager not initialized")
    return manager


async def get_job_manager(request: Request) -> "JobManager":
    """Get JobManager from app state."""
    manager = getattr(request.app.state, "job_manager", None)
    if manager is None:
        raise RuntimeError("Job manager not initialized")
    return manager