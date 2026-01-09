import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from ultralytics import YOLO

from visualization import DetectionVisualizer

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Container for a loaded YOLO model and its resources."""
    model: YOLO
    visualizer: DetectionVisualizer
    model_name: str
    device: str


@dataclass
class CachedModelEntry:
    """Model entry with TTL tracking for cache eviction."""
    entry: ModelEntry
    last_used_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last used timestamp."""
        self.last_used_at = time.time()

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired based on TTL."""
        return (time.time() - self.last_used_at) > ttl_seconds


class ModelManager:
    """
    Manages loading, caching, and unloading of YOLO models.

    - Preloaded models: loaded at startup, never evicted
    - Cached models: loaded on-demand, evicted after TTL
    """

    def __init__(self, default_device: str, ttl_seconds: int = 900):
        self.default_device = default_device
        self.ttl_seconds = ttl_seconds

        self._preloaded: dict[str, ModelEntry] = {}
        self._cached: dict[str, CachedModelEntry] = {}
        self._loading_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    @property
    def default_model(self) -> str | None:
        """Return the first preloaded model name as default."""
        if self._preloaded:
            return next(iter(self._preloaded.keys()))
        return None

    def _load_model_sync(self, model_name: str, device: str) -> ModelEntry:
        """Synchronously load a YOLO model on specified device (blocking)."""
        logger.info(f"Loading model: {model_name} on device: {device}")

        try:
            model = YOLO(model_name)
            model.to(device)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                f"Failed to load {model_name} on {device}: CUDA out of memory. "
                f"Try a smaller model or different device."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name} on {device}: {e}")

        visualizer = DetectionVisualizer(model.names)

        device_info = next(model.model.parameters()).device
        logger.info(f"Model {model_name} loaded successfully on device: {device_info}")

        return ModelEntry(
            model=model,
            visualizer=visualizer,
            model_name=model_name,
            device=device
        )

    async def _load_model_async(self, model_name: str, device: str) -> ModelEntry:
        """Load model in thread pool to avoid blocking event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._load_model_sync, model_name, device
        )

    async def preload_models(self, model_device_map: dict[str, str]) -> None:
        """Preload models at startup. These models are never evicted.

        Args:
            model_device_map: Mapping of model name to device (e.g. {"yolo11s.pt": "cpu"})
        """
        for model_name, device in model_device_map.items():
            if model_name in self._preloaded:
                logger.debug(f"Model {model_name} already preloaded, skipping")
                continue

            try:
                entry = await self._load_model_async(model_name, device)
                self._preloaded[model_name] = entry
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
                raise RuntimeError(f"Failed to preload model {model_name}: {e}") from e

        logger.info(f"Preloaded {len(self._preloaded)} models: {list(self._preloaded.keys())}")

    async def get_model(self, model_name: str | None = None) -> ModelEntry:
        """
        Get a model by name. Loads on-demand if not already loaded.

        Args:
            model_name: Model name (e.g. 'yolo11s.pt'). If None, returns default model.

        Returns:
            ModelEntry with the loaded model and visualizer.

        Raises:
            ValueError: If no model name provided and no default model available.
            RuntimeError: If model loading fails.
        """
        if model_name is None:
            # Try to use first preloaded model as default
            if self._preloaded:
                model_name = next(iter(self._preloaded.keys()))
            else:
                raise ValueError("No model specified and no default model available")

        # Check preloaded models first
        if model_name in self._preloaded:
            return self._preloaded[model_name]

        # Check cached models
        if model_name in self._cached:
            cached = self._cached[model_name]
            cached.touch()
            return cached.entry

        # Need to load the model - get or create lock for this model
        async with self._global_lock:
            if model_name not in self._loading_locks:
                self._loading_locks[model_name] = asyncio.Lock()
            lock = self._loading_locks[model_name]

        # Load with model-specific lock to prevent duplicate loads
        async with lock:
            # Double-check after acquiring lock
            if model_name in self._cached:
                cached = self._cached[model_name]
                cached.touch()
                return cached.entry

            # Load the model on default device
            try:
                entry = await self._load_model_async(model_name, self.default_device)
                self._cached[model_name] = CachedModelEntry(entry=entry)
                logger.info(f"Model {model_name} loaded on-demand on {self.default_device} and cached (TTL: {self.ttl_seconds}s)")
                return entry
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise RuntimeError(f"Failed to load model {model_name}: {e}") from e
            finally:
                # Clean up lock
                async with self._global_lock:
                    self._loading_locks.pop(model_name, None)

    async def cleanup_expired(self) -> int:
        """Remove expired models from cache. Returns count of evicted models."""
        evicted = 0
        expired_keys = [
            name for name, cached in self._cached.items()
            if cached.is_expired(self.ttl_seconds)
        ]

        for model_name in expired_keys:
            cached = self._cached.pop(model_name, None)
            if cached:
                logger.info(f"Evicting expired model from cache: {model_name}")
                # Help garbage collector
                del cached.entry.model
                del cached.entry.visualizer
                evicted += 1

        if evicted > 0 and self.default_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared after model eviction")

        return evicted

    async def _cleanup_loop(self, interval_seconds: int = 60) -> None:
        """Background task that periodically cleans up expired models."""
        logger.info(f"Starting model cleanup task (interval: {interval_seconds}s, TTL: {self.ttl_seconds}s)")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=interval_seconds
                )
                # If we get here, shutdown was signaled
                break
            except asyncio.TimeoutError:
                # Normal timeout - run cleanup
                evicted = await self.cleanup_expired()
                if evicted > 0:
                    logger.info(f"Cleanup task evicted {evicted} expired model(s)")

        logger.info("Model cleanup task stopped")

    def start_cleanup_task(self, interval_seconds: int = 60) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._shutdown_event.clear()
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(interval_seconds)
            )

    async def shutdown(self) -> None:
        """Gracefully shutdown the model manager."""
        logger.info("Shutting down ModelManager...")

        # Stop cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._cleanup_task.cancel()
                logger.warning("Cleanup task did not stop gracefully, cancelled")

        # Clear cached models
        for model_name, cached in list(self._cached.items()):
            logger.debug(f"Unloading cached model: {model_name}")
            del cached.entry.model
            del cached.entry.visualizer
        self._cached.clear()

        # Clear preloaded models
        for model_name, entry in list(self._preloaded.items()):
            logger.debug(f"Unloading preloaded model: {model_name}")
            del entry.model
            del entry.visualizer
        self._preloaded.clear()

        # Clear CUDA cache
        if self.default_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("ModelManager shutdown complete")

    def get_status(self) -> dict:
        """Get current status of loaded models."""
        now = time.time()

        preloaded_info = [
            {"name": name, "device": entry.device}
            for name, entry in self._preloaded.items()
        ]

        cached_info = [
            {
                "name": name,
                "device": cached.entry.device,
                "expires_in_seconds": int(max(0, self.ttl_seconds - (now - cached.last_used_at)))
            }
            for name, cached in self._cached.items()
        ]

        return {
            "preloaded": preloaded_info,
            "cached": cached_info,
            "default_device": self.default_device,
            "ttl_seconds": self.ttl_seconds
        }
